"""Consensus Learning Curve Analyzer.

Runs progressive difficulty scenarios to study how agent ensembles adapt.
Detects learning plateaus, breakthrough points, and generates an interactive
HTML report with performance curves, difficulty impact analysis, and
adaptive difficulty recommendations.

Usage::

    python -m src.learning_curve                     # default 5-agent ensemble
    python -m src.learning_curve --agents 8 --levels 12
    python -m src.learning_curve --output report.html --json results.json
    python -m src.learning_curve --autopilot          # auto-adjust difficulty
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine


# ---------------------------------------------------------------------------
# Difficulty Scenario Definitions
# ---------------------------------------------------------------------------

@dataclass
class DifficultyLevel:
    """Parameterizes a single difficulty tier."""
    level: int
    name: str
    num_agents: int
    byzantine_fraction: float
    confidence_mean: float
    confidence_spread: float
    threshold: float
    answer_diversity: int
    description: str


def build_difficulty_ladder(num_agents: int, levels: int = 10) -> List[DifficultyLevel]:
    """Generate a progressive difficulty ladder."""
    ladder: List[DifficultyLevel] = []
    for i in range(levels):
        t = i / max(levels - 1, 1)
        byz_frac = 0.0 + t * 0.4
        conf_mean = 0.9 - t * 0.3
        conf_spread = 0.05 + t * 0.2
        threshold = 0.6 + t * 0.3
        diversity = 1 + int(t * min(num_agents - 1, 5))
        names = [
            "Trivial", "Easy", "Gentle", "Moderate", "Challenging",
            "Hard", "Intense", "Extreme", "Brutal", "Impossible",
            "Nightmare", "Legendary", "Mythic", "Transcendent", "Godlike",
        ]
        name = names[min(i, len(names) - 1)]
        ladder.append(DifficultyLevel(
            level=i + 1,
            name=name,
            num_agents=num_agents,
            byzantine_fraction=round(byz_frac, 3),
            confidence_mean=round(conf_mean, 3),
            confidence_spread=round(conf_spread, 3),
            threshold=round(threshold, 3),
            answer_diversity=diversity,
            description=f"Byz={byz_frac:.0%}, conf~{conf_mean:.2f}+/-{conf_spread:.2f}, "
                        f"threshold={threshold:.2f}, diversity={diversity}",
        ))
    return ladder


# ---------------------------------------------------------------------------
# Trial Runner
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    level: int
    level_name: str
    trial_index: int
    committed: bool
    rounds_used: int
    aggregate_weight: float
    threshold: float
    byzantine_count: int
    agent_count: int
    slashed_count: int
    consensus_margin: float


@dataclass
class LevelSummary:
    level: int
    level_name: str
    description: str
    trials: int
    commit_rate: float
    avg_rounds: float
    avg_margin: float
    avg_slashed: float
    margin_std: float


@dataclass
class BreakthroughEvent:
    level: int
    event_type: str  # "breakthrough" | "plateau" | "collapse"
    metric: str
    value: float
    delta: float
    description: str


@dataclass
class AdaptiveRecommendation:
    category: str
    priority: str  # "high" | "medium" | "low"
    message: str


async def run_trial(
    diff: DifficultyLevel,
    trial_idx: int,
    max_rounds: int = 4,
) -> TrialResult:
    """Run a single consensus trial at the given difficulty."""
    n = diff.num_agents
    n_byz = max(0, int(n * diff.byzantine_fraction))
    answers = [f"answer_{k}" for k in range(diff.answer_diversity)]

    agents: list[MockAgent] = []
    for i in range(n):
        is_byz = i < n_byz
        conf = max(0.01, min(1.0, random.gauss(diff.confidence_mean, diff.confidence_spread)))
        ans = random.choice(answers)
        agents.append(MockAgent(
            agent_id=f"agent_{i}",
            answer=ans,
            confidence=round(conf, 3),
            byzantine=is_byz,
            accept_set={ans} if not is_byz else None,
        ))

    engine = MBFTEngine(agents, threshold=diff.threshold, max_rounds=max_rounds)
    result = await engine.run("learning_curve_task")

    if result is None:
        return TrialResult(
            level=diff.level, level_name=diff.name, trial_index=trial_idx,
            committed=False, rounds_used=max_rounds,
            aggregate_weight=0.0, threshold=diff.threshold,
            byzantine_count=n_byz, agent_count=n, slashed_count=0,
            consensus_margin=-diff.threshold,
        )

    return TrialResult(
        level=diff.level, level_name=diff.name, trial_index=trial_idx,
        committed=result.committed, rounds_used=result.round_index + 1,
        aggregate_weight=round(result.aggregate_weight, 4),
        threshold=diff.threshold,
        byzantine_count=n_byz, agent_count=n,
        slashed_count=len(result.slashed),
        consensus_margin=round(result.aggregate_weight - diff.threshold, 4),
    )


def summarize_level(diff: DifficultyLevel, trials: List[TrialResult]) -> LevelSummary:
    commits = [t for t in trials if t.committed]
    margins = [t.consensus_margin for t in trials]
    return LevelSummary(
        level=diff.level,
        level_name=diff.name,
        description=diff.description,
        trials=len(trials),
        commit_rate=len(commits) / len(trials) if trials else 0,
        avg_rounds=statistics.mean(t.rounds_used for t in trials),
        avg_margin=statistics.mean(margins) if margins else 0,
        avg_slashed=statistics.mean(t.slashed_count for t in trials),
        margin_std=statistics.stdev(margins) if len(margins) > 1 else 0,
    )


def detect_events(summaries: List[LevelSummary]) -> List[BreakthroughEvent]:
    events: List[BreakthroughEvent] = []
    for i in range(1, len(summaries)):
        prev, curr = summaries[i - 1], summaries[i]
        delta_cr = curr.commit_rate - prev.commit_rate

        if delta_cr >= 0.15:
            events.append(BreakthroughEvent(
                level=curr.level, event_type="breakthrough", metric="commit_rate",
                value=curr.commit_rate, delta=delta_cr,
                description=f"Commit rate surged +{delta_cr:.0%} at level {curr.level} ({curr.level_name})",
            ))
        elif abs(delta_cr) < 0.03 and curr.commit_rate > 0.3:
            events.append(BreakthroughEvent(
                level=curr.level, event_type="plateau", metric="commit_rate",
                value=curr.commit_rate, delta=delta_cr,
                description=f"Performance plateau at level {curr.level} ({curr.level_name}), "
                            f"commit rate stable ~{curr.commit_rate:.0%}",
            ))
        elif delta_cr <= -0.25:
            events.append(BreakthroughEvent(
                level=curr.level, event_type="collapse", metric="commit_rate",
                value=curr.commit_rate, delta=delta_cr,
                description=f"Performance collapse at level {curr.level} ({curr.level_name}), "
                            f"commit rate dropped {delta_cr:.0%}",
            ))
    return events


def generate_recommendations(
    summaries: List[LevelSummary],
    events: List[BreakthroughEvent],
) -> List[AdaptiveRecommendation]:
    recs: List[AdaptiveRecommendation] = []

    breaking = None
    for s in summaries:
        if s.commit_rate < 0.5:
            breaking = s
            break

    if breaking:
        recs.append(AdaptiveRecommendation(
            category="Difficulty Ceiling",
            priority="high",
            message=f"Ensemble breaks down at level {breaking.level} ({breaking.level_name}). "
                    f"{breaking.description}. "
                    f"Consider adding more agents or improving confidence calibration.",
        ))

    collapses = [e for e in events if e.event_type == "collapse"]
    if collapses:
        recs.append(AdaptiveRecommendation(
            category="Stability",
            priority="high",
            message=f"Detected {len(collapses)} performance collapse(s). "
                    f"The ensemble is fragile at these difficulty transitions. "
                    f"Gradual difficulty ramp or adaptive thresholds recommended.",
        ))

    plateaus = [e for e in events if e.event_type == "plateau"]
    if plateaus:
        recs.append(AdaptiveRecommendation(
            category="Adaptation",
            priority="medium",
            message=f"Detected {len(plateaus)} plateau(s) where difficulty increase had no effect. "
                    f"The ensemble may benefit from diversity injection or reputation reweighting.",
        ))

    high_slash = [s for s in summaries if s.avg_slashed > s.trials * 0.3]
    if high_slash:
        recs.append(AdaptiveRecommendation(
            category="Governance",
            priority="medium",
            message=f"High slashing rates at {len(high_slash)} levels. "
                    f"Slash factor may be too aggressive, risking over-penalization of honest agents.",
        ))

    if summaries and summaries[-1].commit_rate > 0.7:
        recs.append(AdaptiveRecommendation(
            category="Challenge",
            priority="low",
            message="Ensemble handles all difficulty levels well. "
                    "Consider adding harder scenarios or larger Byzantine fractions.",
        ))

    return recs


# ---------------------------------------------------------------------------
# Autopilot
# ---------------------------------------------------------------------------

async def run_autopilot(
    num_agents: int,
    trials_per_level: int = 20,
    max_levels: int = 20,
) -> Tuple[List[LevelSummary], List[TrialResult]]:
    """Adaptively increase difficulty until the ensemble fails."""
    all_trials: List[TrialResult] = []
    summaries: List[LevelSummary] = []
    byz_frac = 0.0
    threshold = 0.5
    diversity = 1

    for lvl in range(1, max_levels + 1):
        diff = DifficultyLevel(
            level=lvl, name=f"Auto-{lvl}", num_agents=num_agents,
            byzantine_fraction=round(byz_frac, 3),
            confidence_mean=round(max(0.3, 0.9 - lvl * 0.03), 3),
            confidence_spread=round(min(0.3, 0.05 + lvl * 0.015), 3),
            threshold=round(threshold, 3),
            answer_diversity=diversity,
            description=f"Auto: byz={byz_frac:.1%}, thr={threshold:.2f}, div={diversity}",
        )

        trials = [await run_trial(diff, t) for t in range(trials_per_level)]
        all_trials.extend(trials)
        summary = summarize_level(diff, trials)
        summaries.append(summary)

        print(f"  Level {lvl:2d} | commit={summary.commit_rate:5.0%} | "
              f"margin={summary.avg_margin:+.3f} | {diff.description}")

        if summary.commit_rate < 0.2:
            print(f"  -> Ensemble collapsed at level {lvl}. Stopping autopilot.")
            break

        if summary.commit_rate > 0.8:
            byz_frac = min(0.49, byz_frac + 0.05)
            threshold = min(0.95, threshold + 0.03)
            diversity = min(num_agents, diversity + 1)
        elif summary.commit_rate > 0.5:
            byz_frac = min(0.49, byz_frac + 0.02)
            threshold = min(0.95, threshold + 0.01)

    return summaries, all_trials


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html_report(
    summaries: List[LevelSummary],
    events: List[BreakthroughEvent],
    recommendations: List[AdaptiveRecommendation],
    all_trials: List[TrialResult],
    autopilot: bool,
) -> str:
    levels_json = json.dumps([{
        "level": s.level, "name": s.level_name, "desc": s.description,
        "commitRate": s.commit_rate, "avgRounds": s.avg_rounds,
        "avgMargin": s.avg_margin, "marginStd": s.margin_std,
        "avgSlashed": s.avg_slashed,
    } for s in summaries])

    events_json = json.dumps([{
        "level": e.level, "type": e.event_type, "metric": e.metric,
        "value": e.value, "delta": e.delta, "desc": e.description,
    } for e in events])

    recs_json = json.dumps([{
        "category": r.category, "priority": r.priority, "message": r.message,
    } for r in recommendations])

    trials_json = json.dumps([{
        "level": t.level, "committed": t.committed, "rounds": t.rounds_used,
        "margin": t.consensus_margin, "slashed": t.slashed_count,
    } for t in all_trials])

    mode_label = "(Autopilot)" if autopilot else ""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Consensus Learning Curve - mBFT</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0e17;color:#c9d1d9;min-height:100vh}}
.hdr{{background:linear-gradient(135deg,#1a1e2e,#0d1117);padding:32px 40px;border-bottom:1px solid #30363d}}
.hdr h1{{font-size:28px;color:#58a6ff}}.hdr p{{color:#8b949e;margin-top:4px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:24px 40px;max-width:1400px;margin:0 auto}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px}}
.card h2{{font-size:16px;color:#58a6ff;margin-bottom:12px}}
.full{{grid-column:1/-1}}
canvas{{width:100%;height:300px;display:block}}
.ev{{padding:8px 12px;margin:4px 0;border-radius:6px;font-size:13px}}
.ev.breakthrough{{background:#0d2818;border-left:3px solid #3fb950}}
.ev.plateau{{background:#1c1d00;border-left:3px solid #d29922}}
.ev.collapse{{background:#2d0000;border-left:3px solid #f85149}}
.rec{{padding:10px 14px;margin:6px 0;border-radius:6px;font-size:13px;background:#0d1117;border:1px solid #30363d}}
.rec .badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;margin-right:8px}}
.badge.high{{background:#f8514933;color:#f85149}}.badge.medium{{background:#d2992233;color:#d29922}}.badge.low{{background:#3fb95033;color:#3fb950}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid #21262d}}
th{{color:#8b949e;font-weight:600}}
.bar{{height:18px;border-radius:3px;transition:width .3s}}
</style></head><body>
<div class="hdr">
  <h1>Consensus Learning Curve Analyzer</h1>
  <p>mBFT ensemble performance across progressive difficulty levels {mode_label}</p>
</div>
<div class="grid">
  <div class="card full"><h2>Commit Rate &amp; Consensus Margin</h2>
    <canvas id="mainChart"></canvas></div>
  <div class="card"><h2>Difficulty Breakdown</h2>
    <div style="max-height:350px;overflow-y:auto"><table><thead>
      <tr><th>Lv</th><th>Name</th><th>Commit%</th><th>Avg Rounds</th><th>Margin</th><th>Slashed</th></tr>
    </thead><tbody id="tbl"></tbody></table></div></div>
  <div class="card"><h2>Detected Events</h2><div id="events"></div></div>
  <div class="card full"><h2>Trial Scatter</h2><canvas id="scatter"></canvas></div>
  <div class="card"><h2>Adaptive Recommendations</h2><div id="recs"></div></div>
  <div class="card"><h2>Performance Summary</h2><div id="summary"></div></div>
</div>
<script>
const L={levels_json};
const EV={events_json};
const REC={recs_json};
const TR={trials_json};

const tbl=document.getElementById('tbl');
L.forEach(l=>{{
  const c=l.commitRate>=0.7?'#3fb950':l.commitRate>=0.4?'#d29922':'#f85149';
  tbl.innerHTML+=`<tr><td>${{l.level}}</td><td>${{l.name}}</td>
    <td><div style="display:flex;align-items:center;gap:6px">
      <div class="bar" style="width:${{l.commitRate*100}}%;background:${{c}};min-width:2px">&nbsp;</div>
      <span>${{(l.commitRate*100).toFixed(0)}}%</span></div></td>
    <td>${{l.avgRounds.toFixed(1)}}</td><td>${{l.avgMargin>=0?'+':''}}${{l.avgMargin.toFixed(3)}}</td>
    <td>${{l.avgSlashed.toFixed(1)}}</td></tr>`;
}});

const evDiv=document.getElementById('events');
if(EV.length===0) evDiv.innerHTML='<p style="color:#8b949e">No significant events detected.</p>';
EV.forEach(e=>{{evDiv.innerHTML+=`<div class="ev ${{e.type}}"><strong>Lv${{e.level}}</strong> - ${{e.desc}}</div>`;}});

const recDiv=document.getElementById('recs');
if(REC.length===0) recDiv.innerHTML='<p style="color:#8b949e">No recommendations.</p>';
REC.forEach(r=>{{recDiv.innerHTML+=`<div class="rec"><span class="badge ${{r.priority}}">${{r.priority}}</span><strong>${{r.category}}:</strong> ${{r.message}}</div>`;}});

const sumDiv=document.getElementById('summary');
const peak=L.reduce((a,b)=>a.commitRate>b.commitRate?a:b,L[0]);
const worst=L.reduce((a,b)=>a.commitRate<b.commitRate?a:b,L[0]);
const avgCR=(L.reduce((s,l)=>s+l.commitRate,0)/L.length*100).toFixed(1);
sumDiv.innerHTML=`
  <p style="margin:8px 0"><strong>Levels tested:</strong> ${{L.length}}</p>
  <p style="margin:8px 0"><strong>Avg commit rate:</strong> ${{avgCR}}%</p>
  <p style="margin:8px 0"><strong>Best level:</strong> ${{peak.level}} (${{peak.name}}) - ${{(peak.commitRate*100).toFixed(0)}}%</p>
  <p style="margin:8px 0"><strong>Worst level:</strong> ${{worst.level}} (${{worst.name}}) - ${{(worst.commitRate*100).toFixed(0)}}%</p>
  <p style="margin:8px 0"><strong>Events:</strong> ${{EV.length}} (${{EV.filter(e=>e.type==='breakthrough').length}} breakthroughs, ${{EV.filter(e=>e.type==='collapse').length}} collapses, ${{EV.filter(e=>e.type==='plateau').length}} plateaus)</p>
`;

function drawMain(){{
  const cv=document.getElementById('mainChart');
  const dpr=window.devicePixelRatio||1;
  cv.width=cv.offsetWidth*dpr;cv.height=300*dpr;
  const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
  const W=cv.offsetWidth,H=300,pad=40;
  const plotW=W-2*pad,plotH=H-2*pad;

  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#21262d';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){{
    const y=pad+plotH*(1-i/4);
    ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke();
    ctx.fillStyle='#484f58';ctx.font='11px sans-serif';ctx.textAlign='right';
    ctx.fillText((i*25)+'%',pad-6,y+4);
  }}

  ctx.strokeStyle='#58a6ff';ctx.lineWidth=2;ctx.beginPath();
  L.forEach((l,i)=>{{
    const x=pad+plotW*(i/(L.length-1||1));
    const y=pad+plotH*(1-l.commitRate);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }});ctx.stroke();

  ctx.fillStyle='#3fb95020';ctx.beginPath();
  L.forEach((l,i)=>{{
    const x=pad+plotW*(i/(L.length-1||1));
    const y=pad+plotH*(1-Math.max(0,Math.min(1,(l.avgMargin+1)/2)));
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }});
  ctx.lineTo(pad+plotW,pad+plotH);ctx.lineTo(pad,pad+plotH);ctx.fill();

  EV.forEach(e=>{{
    const idx=L.findIndex(l=>l.level===e.level);if(idx<0)return;
    const x=pad+plotW*(idx/(L.length-1||1));
    const c=e.type==='breakthrough'?'#3fb950':e.type==='collapse'?'#f85149':'#d29922';
    ctx.fillStyle=c;ctx.beginPath();ctx.arc(x,pad+8,5,0,Math.PI*2);ctx.fill();
  }});

  ctx.fillStyle='#484f58';ctx.font='11px sans-serif';ctx.textAlign='center';
  L.forEach((l,i)=>{{
    if(L.length<=15||i%(Math.ceil(L.length/10))===0){{
      const x=pad+plotW*(i/(L.length-1||1));
      ctx.fillText('L'+l.level,x,H-8);
    }}
  }});

  ctx.fillStyle='#58a6ff';ctx.fillRect(W-200,10,12,3);
  ctx.fillStyle='#8b949e';ctx.font='11px sans-serif';ctx.textAlign='left';
  ctx.fillText('Commit Rate',W-182,14);
  ctx.fillStyle='#3fb95060';ctx.fillRect(W-200,22,12,8);
  ctx.fillStyle='#8b949e';ctx.fillText('Margin',W-182,30);
}}

function drawScatter(){{
  const cv=document.getElementById('scatter');
  const dpr=window.devicePixelRatio||1;
  cv.width=cv.offsetWidth*dpr;cv.height=300*dpr;
  const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
  const W=cv.offsetWidth,H=300,pad=40;
  const plotW=W-2*pad,plotH=H-2*pad;
  const maxLv=Math.max(...TR.map(t=>t.level));

  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#21262d';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){{
    const y=pad+plotH*(1-i/4);
    ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke();
  }}

  TR.forEach(t=>{{
    const x=pad+plotW*((t.level-1)/(maxLv-1||1));
    const normM=(t.margin+2)/4;
    const y=pad+plotH*(1-Math.max(0,Math.min(1,normM)));
    ctx.fillStyle=t.committed?'#3fb95060':'#f8514960';
    ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();
  }});

  ctx.fillStyle='#484f58';ctx.font='11px sans-serif';ctx.textAlign='center';
  ctx.fillText('Difficulty Level',W/2,H-4);
  ctx.save();ctx.translate(10,H/2);ctx.rotate(-Math.PI/2);
  ctx.fillText('Consensus Margin',0,0);ctx.restore();
}}

drawMain();drawScatter();
window.addEventListener('resize',()=>{{drawMain();drawScatter();}});
</script></body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Consensus Learning Curve Analyzer")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--levels", type=int, default=10, help="Difficulty levels")
    parser.add_argument("--trials", type=int, default=20, help="Trials per level")
    parser.add_argument("--output", type=str, default="learning_curve_report.html")
    parser.add_argument("--json", type=str, default=None, help="Export JSON results")
    parser.add_argument("--autopilot", action="store_true", help="Auto-adjust difficulty")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"{'='*60}")
    print("  Consensus Learning Curve Analyzer")
    print(f"  Agents: {args.agents} | Mode: {'Autopilot' if args.autopilot else f'{args.levels} levels'}")
    print(f"  Trials per level: {args.trials}")
    print(f"{'='*60}\n")

    if args.autopilot:
        summaries, all_trials = await run_autopilot(args.agents, args.trials)
    else:
        ladder = build_difficulty_ladder(args.agents, args.levels)
        all_trials: List[TrialResult] = []
        summaries: List[LevelSummary] = []

        for diff in ladder:
            trials = [await run_trial(diff, t) for t in range(args.trials)]
            all_trials.extend(trials)
            summary = summarize_level(diff, trials)
            summaries.append(summary)
            print(f"  Level {diff.level:2d} ({diff.name:12s}) | "
                  f"commit={summary.commit_rate:5.0%} | "
                  f"rounds={summary.avg_rounds:.1f} | "
                  f"margin={summary.avg_margin:+.3f} +/- {summary.margin_std:.3f}")

    events = detect_events(summaries)
    recs = generate_recommendations(summaries, events)

    print(f"\n{'-'*60}")
    print(f"  Events detected: {len(events)}")
    for e in events:
        icon = {"breakthrough": ">>", "plateau": "==", "collapse": "!!"}.get(e.event_type, "*")
        print(f"    {icon} {e.description}")

    print(f"\n  Recommendations: {len(recs)}")
    for r in recs:
        icon = {"high": "[!]", "medium": "[~]", "low": "[o]"}.get(r.priority, "*")
        print(f"    {icon} [{r.category}] {r.message}")

    html = generate_html_report(summaries, events, recs, all_trials, args.autopilot)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"\n  Report: {out.resolve()}")

    if args.json:
        data = {
            "summaries": [{"level": s.level, "name": s.level_name, "desc": s.description,
                           "commitRate": s.commit_rate, "avgRounds": s.avg_rounds,
                           "avgMargin": s.avg_margin} for s in summaries],
            "events": [{"level": e.level, "type": e.event_type, "desc": e.description}
                       for e in events],
            "recommendations": [{"category": r.category, "priority": r.priority,
                                 "message": r.message} for r in recs],
        }
        Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  JSON: {Path(args.json).resolve()}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
