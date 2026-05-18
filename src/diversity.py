"""Consensus Diversity Index — cognitive diversity measurement and groupthink detection.

Analyses mBFT consensus rounds to measure how diverse agent reasoning is,
detect groupthink / echo-chamber dynamics, and recommend composition changes
to improve swarm robustness.

Metrics computed:
- **Shannon Diversity Index** over vote-weight distributions
- **Simpson's Diversity Index** (probability two random votes differ)
- **Reasoning Overlap** — solution similarity via Jaccard on token sets
- **Confidence Spread** — coefficient of variation of proposal confidences
- **Groupthink Score** — composite 0-1 score (high = echo chamber detected)
- **Dissent Ratio** — fraction of rounds with at least one rejection
- **Leadership Monopoly** — Gini coefficient of leader selections
- **Herd Behavior Index** — how often followers match the majority direction

Usage:
    python -m src.diversity --agents 6 --rounds 20 --byzantine 1 --threshold 2.5
    python -m src.diversity --agents 8 --rounds 30 --report diversity_report.html
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import math
from collections import Counter
from typing import Any, Dict, List, Tuple

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine
from src.core.state import RoundResult
from src.stats_utils import gini as _gini_shared


# ── Diversity metrics ──────────────────────────────────────────────────────

def shannon_diversity(weights: List[float]) -> float:
    """Shannon entropy H over absolute vote weights (binned)."""
    if not weights:
        return 0.0
    bins: Counter[str] = Counter()
    for w in weights:
        if w < -0.33:
            bins["strong_reject"] += 1
        elif w < 0.0:
            bins["mild_reject"] += 1
        elif w < 0.33:
            bins["mild_accept"] += 1
        elif w < 0.66:
            bins["moderate_accept"] += 1
        else:
            bins["strong_accept"] += 1
    total = sum(bins.values())
    if total == 0:
        return 0.0
    h = 0.0
    for count in bins.values():
        p = count / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def simpson_diversity(weights: List[float]) -> float:
    """1 - Simpson's index (probability two random picks differ)."""
    if len(weights) < 2:
        return 0.0
    bins: Counter[int] = Counter()
    for w in weights:
        bins[int(round(w * 10))] += 1  # bin to nearest 0.1 as integer key
    n = len(weights)
    d = sum(c * (c - 1) for c in bins.values()) / (n * (n - 1))
    return 1.0 - d


def confidence_spread(confidences: List[float]) -> float:
    """Coefficient of variation of proposal confidences."""
    if len(confidences) < 2:
        return 0.0
    mean = sum(confidences) / len(confidences)
    if mean == 0:
        return 0.0
    var = sum((c - mean) ** 2 for c in confidences) / len(confidences)
    return math.sqrt(var) / mean


def gini_coefficient(values: List[float]) -> float:
    """Gini coefficient — 0 = perfectly equal, 1 = maximum inequality."""
    return _gini_shared(values)


def herd_behavior_index(round_votes: List[List[float]]) -> float:
    """Fraction of individual votes that match the majority direction per round."""
    if not round_votes:
        return 0.0
    herd_count = 0
    total_count = 0
    for votes in round_votes:
        if not votes:
            continue
        positives = sum(1 for v in votes if v >= 0)
        negatives = len(votes) - positives
        majority_positive = positives >= negatives
        for v in votes:
            total_count += 1
            if majority_positive and v >= 0:
                herd_count += 1
            elif not majority_positive and v < 0:
                herd_count += 1
    return herd_count / total_count if total_count > 0 else 0.0


def groupthink_score(
    shannon: float,
    simpson: float,
    conf_spread: float,
    dissent_ratio: float,
    herd_idx: float,
) -> float:
    """Composite groupthink score 0-1 (high = echo chamber)."""
    # Low diversity + low dissent + high herd = groupthink
    max_shannon = math.log2(5)  # 5 bins
    norm_shannon = min(shannon / max_shannon, 1.0) if max_shannon > 0 else 0.0
    score = (
        0.25 * (1.0 - norm_shannon)
        + 0.20 * (1.0 - simpson)
        + 0.15 * (1.0 - min(conf_spread, 1.0))
        + 0.20 * (1.0 - dissent_ratio)
        + 0.20 * herd_idx
    )
    return round(min(max(score, 0.0), 1.0), 4)


# ── Analysis engine ────────────────────────────────────────────────────────

class DiversityAnalysis:
    """Full diversity analysis result."""

    def __init__(
        self,
        shannon: float,
        simpson: float,
        conf_spread: float,
        dissent_ratio: float,
        leader_gini: float,
        herd_index: float,
        groupthink: float,
        per_round: List[Dict[str, Any]],
        recommendations: List[str],
        agent_profiles: Dict[str, Dict[str, Any]],
    ):
        self.shannon = shannon
        self.simpson = simpson
        self.conf_spread = conf_spread
        self.dissent_ratio = dissent_ratio
        self.leader_gini = leader_gini
        self.herd_index = herd_index
        self.groupthink = groupthink
        self.per_round = per_round
        self.recommendations = recommendations
        self.agent_profiles = agent_profiles

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": {
                "shannon_diversity": self.shannon,
                "simpson_diversity": self.simpson,
                "confidence_spread": self.conf_spread,
                "dissent_ratio": self.dissent_ratio,
                "leadership_gini": self.leader_gini,
                "herd_behavior_index": self.herd_index,
                "groupthink_score": self.groupthink,
            },
            "diagnosis": self._diagnosis(),
            "per_round": self.per_round,
            "agent_profiles": self.agent_profiles,
            "recommendations": self.recommendations,
        }

    def _diagnosis(self) -> str:
        if self.groupthink >= 0.7:
            return "CRITICAL: Echo chamber detected — swarm lacks cognitive diversity"
        elif self.groupthink >= 0.5:
            return "WARNING: Moderate groupthink tendencies — consider adding contrarian agents"
        elif self.groupthink >= 0.3:
            return "HEALTHY: Reasonable diversity with some convergence"
        else:
            return "EXCELLENT: High cognitive diversity — robust deliberation"


def analyze_rounds(
    results: List[RoundResult],
    reputation: Dict[str, float],
) -> DiversityAnalysis:
    """Compute diversity metrics from a sequence of mBFT rounds."""
    all_weights: List[float] = []
    all_confidences: List[float] = []
    round_vote_lists: List[List[float]] = []
    leader_counts: Counter[str] = Counter()
    dissent_rounds = 0
    per_round: List[Dict[str, Any]] = []
    agent_votes: Dict[str, List[float]] = {}
    agent_leads: Counter[str] = Counter()

    for rr in results:
        weights = [v.weight for v in rr.votes]
        all_weights.extend(weights)
        round_vote_lists.append(weights)
        leader_counts[rr.leader_id] += 1
        agent_leads[rr.leader_id] += 1

        if any(v.is_rejection for v in rr.votes):
            dissent_rounds += 1

        for v in rr.votes:
            agent_votes.setdefault(v.voter_id, []).append(v.weight)

        per_round.append({
            "round": rr.round_index,
            "leader": rr.leader_id,
            "committed": rr.committed,
            "aggregate": round(rr.aggregate_weight, 3),
            "vote_spread": round(max(weights) - min(weights), 3) if weights else 0,
            "shannon": round(shannon_diversity(weights), 3),
        })

    sh = round(shannon_diversity(all_weights), 4)
    si = round(simpson_diversity(all_weights), 4)
    cs = round(confidence_spread(all_confidences) if all_confidences else 0.0, 4)
    dr = round(dissent_rounds / len(results), 4) if results else 0.0
    lg = round(gini_coefficient(list(leader_counts.values())), 4)
    hi = round(herd_behavior_index(round_vote_lists), 4)
    gt = groupthink_score(sh, si, cs, dr, hi)

    # Agent profiles
    profiles: Dict[str, Dict[str, Any]] = {}
    for aid, votes in agent_votes.items():
        avg = sum(votes) / len(votes)
        rejections = sum(1 for v in votes if v < 0)
        profiles[aid] = {
            "avg_vote": round(avg, 3),
            "rejection_rate": round(rejections / len(votes), 3),
            "vote_std": round(
                math.sqrt(sum((v - avg) ** 2 for v in votes) / len(votes)), 3
            ) if len(votes) > 1 else 0.0,
            "times_led": agent_leads.get(aid, 0),
            "reputation": round(reputation.get(aid, 1.0), 3),
            "role": _classify_agent(avg, rejections / len(votes) if votes else 0),
        }

    # Recommendations
    recs = _generate_recommendations(gt, sh, dr, lg, hi, profiles)

    return DiversityAnalysis(
        shannon=sh, simpson=si, conf_spread=cs,
        dissent_ratio=dr, leader_gini=lg, herd_index=hi,
        groupthink=gt, per_round=per_round,
        recommendations=recs, agent_profiles=profiles,
    )


def _classify_agent(avg_vote: float, rejection_rate: float) -> str:
    if rejection_rate > 0.5:
        return "Contrarian"
    elif rejection_rate > 0.2:
        return "Skeptic"
    elif avg_vote > 0.7:
        return "Enthusiast"
    elif avg_vote > 0.4:
        return "Moderate"
    else:
        return "Cautious"


def _generate_recommendations(
    gt: float, sh: float, dr: float, lg: float, hi: float,
    profiles: Dict[str, Dict[str, Any]],
) -> List[str]:
    recs: List[str] = []
    if gt >= 0.7:
        recs.append("🚨 Add 1-2 contrarian agents with independent reasoning to break echo chamber")
    if gt >= 0.5:
        recs.append("⚠️ Introduce agents with diverse knowledge bases or reasoning strategies")
    if sh < 1.0:
        recs.append("📊 Vote distribution is narrow — consider agents with different confidence calibrations")
    if dr < 0.2:
        recs.append("🔇 Very low dissent — add skeptical agents or lower rejection threshold")
    if lg > 0.6:
        recs.append("👑 Leadership is monopolized — consider reputation decay or rotation policies")
    if hi > 0.8:
        recs.append("🐑 High herd behavior — agents may be anchoring on leader proposals instead of independent analysis")
    roles = Counter(p["role"] for p in profiles.values())
    if not roles.get("Contrarian", 0) and not roles.get("Skeptic", 0):
        recs.append("🎯 No contrarian/skeptic agents detected — swarm may lack adversarial resilience")
    if len(roles) < 3:
        recs.append("🌈 Low role diversity — aim for at least 3 distinct behavioral profiles")
    if not recs:
        recs.append("✅ Swarm diversity looks healthy — maintain current composition")
    return recs


# ── HTML report ────────────────────────────────────────────────────────────

def generate_html_report(analysis: DiversityAnalysis) -> str:
    """Generate an interactive HTML report with charts and recommendations."""
    data = analysis.to_dict()
    metrics = data["metrics"]
    profiles = data["agent_profiles"]
    per_round = data["per_round"]

    profile_rows = ""
    for aid, p in sorted(profiles.items()):
        role_color = {
            "Contrarian": "#e74c3c", "Skeptic": "#e67e22",
            "Enthusiast": "#2ecc71", "Moderate": "#3498db", "Cautious": "#9b59b6"
        }.get(p["role"], "#95a5a6")
        profile_rows += f"""<tr>
            <td>{html.escape(aid)}</td>
            <td style="color:{role_color};font-weight:bold">{p['role']}</td>
            <td>{p['avg_vote']}</td>
            <td>{p['rejection_rate']}</td>
            <td>{p['vote_std']}</td>
            <td>{p['times_led']}</td>
            <td>{p['reputation']}</td>
        </tr>"""

    round_rows = ""
    for r in per_round:
        status = "✅" if r["committed"] else "❌"
        round_rows += f"""<tr>
            <td>{r['round']}</td>
            <td>{html.escape(r['leader'])}</td>
            <td>{status}</td>
            <td>{r['aggregate']}</td>
            <td>{r['vote_spread']}</td>
            <td>{r['shannon']}</td>
        </tr>"""

    recs_html = "\n".join(f"<li>{html.escape(r)}</li>" for r in data["recommendations"])

    gt = metrics["groupthink_score"]
    gt_color = "#e74c3c" if gt >= 0.7 else "#e67e22" if gt >= 0.5 else "#2ecc71" if gt < 0.3 else "#f39c12"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>mBFT Consensus Diversity Index</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a1a;color:#e0e0e0;padding:20px}}
h1{{text-align:center;color:#00d4ff;margin:20px 0;font-size:1.8em}}
h2{{color:#00d4ff;margin:20px 0 10px;border-bottom:1px solid #1a1a3a;padding-bottom:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:20px 0}}
.card{{background:#12122a;border-radius:12px;padding:20px;text-align:center;border:1px solid #1a1a3a}}
.card .value{{font-size:2em;font-weight:bold;color:#00d4ff}}
.card .label{{font-size:.85em;color:#888;margin-top:4px}}
.diagnosis{{background:#12122a;border-radius:12px;padding:20px;margin:20px 0;text-align:center;font-size:1.2em;border-left:4px solid {gt_color}}}
table{{width:100%;border-collapse:collapse;margin:10px 0}}
th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid #1a1a3a}}
th{{background:#12122a;color:#00d4ff;font-size:.85em;text-transform:uppercase}}
tr:hover{{background:#1a1a3a}}
.recs{{background:#12122a;border-radius:12px;padding:20px;margin:20px 0}}
.recs li{{margin:8px 0;line-height:1.5}}
canvas{{background:#12122a;border-radius:12px;border:1px solid #1a1a3a;margin:10px 0}}
.gauge-container{{display:flex;justify-content:center;margin:20px 0}}
.footer{{text-align:center;color:#555;margin-top:30px;font-size:.8em}}
</style></head><body>
<h1>🧠 mBFT Consensus Diversity Index</h1>
<div class="diagnosis">{html.escape(data['diagnosis'])}</div>

<div class="grid">
  <div class="card"><div class="value" style="color:{gt_color}">{gt}</div><div class="label">Groupthink Score</div></div>
  <div class="card"><div class="value">{metrics['shannon_diversity']}</div><div class="label">Shannon Diversity</div></div>
  <div class="card"><div class="value">{metrics['simpson_diversity']}</div><div class="label">Simpson Diversity</div></div>
  <div class="card"><div class="value">{metrics['dissent_ratio']}</div><div class="label">Dissent Ratio</div></div>
  <div class="card"><div class="value">{metrics['leadership_gini']}</div><div class="label">Leadership Gini</div></div>
  <div class="card"><div class="value">{metrics['herd_behavior_index']}</div><div class="label">Herd Behavior</div></div>
</div>

<div class="gauge-container"><canvas id="gaugeCanvas" width="400" height="220"></canvas></div>

<h2>📊 Per-Round Diversity</h2>
<canvas id="roundChart" width="900" height="250"></canvas>
<table><thead><tr><th>Round</th><th>Leader</th><th>Committed</th><th>Aggregate</th><th>Vote Spread</th><th>Shannon</th></tr></thead>
<tbody>{round_rows}</tbody></table>

<h2>🤖 Agent Profiles</h2>
<canvas id="radarCanvas" width="700" height="350"></canvas>
<table><thead><tr><th>Agent</th><th>Role</th><th>Avg Vote</th><th>Reject Rate</th><th>Vote σ</th><th>Times Led</th><th>Reputation</th></tr></thead>
<tbody>{profile_rows}</tbody></table>

<h2>💡 Recommendations</h2>
<div class="recs"><ul>{recs_html}</ul></div>

<div class="footer">mBFT Consensus Diversity Index — Metacognitive BFT Analysis Tool</div>

<script>
const rounds = {json.dumps(per_round)};
const profiles = {json.dumps(profiles)};
const groupthink = {gt};

// Groupthink gauge
(function(){{
  const c=document.getElementById('gaugeCanvas'),ctx=c.getContext('2d');
  const cx=200,cy=160,r=120;
  ctx.lineWidth=20;
  // Background arc
  ctx.beginPath();ctx.arc(cx,cy,r,Math.PI,0);ctx.strokeStyle='#1a1a3a';ctx.stroke();
  // Colored arc
  const angle=Math.PI+(groupthink*Math.PI);
  const grad=ctx.createLinearGradient(cx-r,cy,cx+r,cy);
  grad.addColorStop(0,'#2ecc71');grad.addColorStop(0.5,'#f39c12');grad.addColorStop(1,'#e74c3c');
  ctx.beginPath();ctx.arc(cx,cy,r,Math.PI,angle);ctx.strokeStyle=grad;ctx.stroke();
  // Needle
  const na=Math.PI+groupthink*Math.PI;
  ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.cos(na)*100,cy+Math.sin(na)*100);
  ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
  ctx.font='bold 28px sans-serif';ctx.fillStyle='#00d4ff';ctx.textAlign='center';
  ctx.fillText(groupthink.toFixed(2),cx,cy+50);
  ctx.font='12px sans-serif';ctx.fillStyle='#888';
  ctx.fillText('GROUPTHINK GAUGE',cx,cy+70);
  ctx.fillText('Healthy',cx-r+10,cy+20);ctx.fillText('Echo Chamber',cx+r-20,cy+20);
}})();

// Round diversity chart
(function(){{
  const c=document.getElementById('roundChart'),ctx=c.getContext('2d');
  if(!rounds.length) return;
  const pad=50,w=c.width-2*pad,h=c.height-2*pad;
  const maxS=Math.max(...rounds.map(r=>r.shannon),1);
  ctx.strokeStyle='#1a1a3a';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){{
    const y=pad+h-h*(i/4);
    ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(pad+w,y);ctx.stroke();
    ctx.fillStyle='#555';ctx.font='11px sans-serif';ctx.textAlign='right';
    ctx.fillText((maxS*i/4).toFixed(1),pad-8,y+4);
  }}
  // Shannon line
  ctx.beginPath();ctx.strokeStyle='#00d4ff';ctx.lineWidth=2;
  rounds.forEach((r,i)=>{{
    const x=pad+i*w/(rounds.length-1||1),y=pad+h-h*(r.shannon/maxS);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }});ctx.stroke();
  // Vote spread line
  const maxVS=Math.max(...rounds.map(r=>r.vote_spread),1);
  ctx.beginPath();ctx.strokeStyle='#e74c3c';ctx.lineWidth=2;
  rounds.forEach((r,i)=>{{
    const x=pad+i*w/(rounds.length-1||1),y=pad+h-h*(r.vote_spread/maxVS);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }});ctx.stroke();
  ctx.font='11px sans-serif';
  ctx.fillStyle='#00d4ff';ctx.fillText('Shannon',pad+w-60,pad+15);
  ctx.fillStyle='#e74c3c';ctx.fillText('Vote Spread',pad+w-60,pad+30);
}})();

// Agent radar chart
(function(){{
  const c=document.getElementById('radarCanvas'),ctx=c.getContext('2d');
  const agents=Object.entries(profiles);if(!agents.length) return;
  const cx=350,cy=175,r=130;
  const dims=['avg_vote','rejection_rate','vote_std','reputation'];
  const labels=['Avg Vote','Reject Rate','Volatility','Reputation'];
  const n=dims.length;
  // Grid
  for(let ring=1;ring<=4;ring++){{
    ctx.beginPath();
    for(let i=0;i<=n;i++){{
      const a=-Math.PI/2+2*Math.PI*i/n;
      const rr=r*ring/4;
      const x=cx+Math.cos(a)*rr,y=cy+Math.sin(a)*rr;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }}
    ctx.strokeStyle='#1a1a3a';ctx.lineWidth=1;ctx.stroke();
  }}
  labels.forEach((l,i)=>{{
    const a=-Math.PI/2+2*Math.PI*i/n;
    ctx.fillStyle='#888';ctx.font='11px sans-serif';ctx.textAlign='center';
    ctx.fillText(l,cx+Math.cos(a)*(r+20),cy+Math.sin(a)*(r+20));
  }});
  const colors=['#00d4ff','#e74c3c','#2ecc71','#f39c12','#9b59b6','#e67e22','#1abc9c','#e84393'];
  agents.forEach(([aid,p],idx)=>{{
    ctx.beginPath();ctx.strokeStyle=colors[idx%colors.length];ctx.lineWidth=2;ctx.globalAlpha=0.7;
    dims.forEach((d,i)=>{{
      const v=Math.min(Math.abs(p[d]),1);
      const a=-Math.PI/2+2*Math.PI*i/n;
      const x=cx+Math.cos(a)*r*v,y=cy+Math.sin(a)*r*v;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }});
    ctx.closePath();ctx.stroke();ctx.globalAlpha=0.1;ctx.fillStyle=colors[idx%colors.length];ctx.fill();ctx.globalAlpha=1;
  }});
}})();
</script></body></html>"""


# ── CLI ────────────────────────────────────────────────────────────────────

async def run_simulation(
    n_agents: int,
    n_rounds: int,
    n_byzantine: int,
    threshold: float,
) -> Tuple[List[RoundResult], Dict[str, float]]:
    """Run multiple independent mBFT consensus rounds for diversity analysis."""
    import random
    agents = []
    answers = ["A", "B", "C", "D", "E"]
    for i in range(n_agents):
        ans = answers[i % len(answers)]
        conf = round(random.uniform(0.3, 0.95), 2)
        if i < n_byzantine:
            agents.append(MockAgent(f"byz-{i}", answer=ans, confidence=conf, byzantine=True))
        else:
            # Non-byzantine agents accept a range of answers for diversity
            accept = set(random.sample(answers, k=random.randint(1, 3)))
            accept.add(ans)
            agents.append(MockAgent(f"agent-{i}", answer=ans, confidence=conf, accept_set=accept))

    all_results: List[RoundResult] = []
    # Run multiple independent consensus sessions
    for _ in range(n_rounds):
        engine = MBFTEngine(agents, threshold=threshold, max_rounds=4)
        await engine.run("Diversity analysis task")
        all_results.extend(engine.history)

    # Use last engine's reputation as representative
    return all_results, engine.reputation


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Diversity Index — groupthink detection & composition analysis"
    )
    parser.add_argument("--agents", type=int, default=6, help="Number of agents (default: 6)")
    parser.add_argument("--rounds", type=int, default=15, help="Number of consensus sessions (default: 15)")
    parser.add_argument("--byzantine", type=int, default=1, help="Number of Byzantine agents (default: 1)")
    parser.add_argument("--threshold", type=float, default=2.5, help="Commit threshold (default: 2.5)")
    parser.add_argument("--report", type=str, default=None, help="Export HTML report to file")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    print(f"🧠 Running diversity analysis: {args.agents} agents, {args.rounds} sessions, "
          f"{args.byzantine} Byzantine, θ={args.threshold}")

    results, reputation = await run_simulation(
        args.agents, args.rounds, args.byzantine, args.threshold
    )

    analysis = analyze_rounds(results, reputation)
    data = analysis.to_dict()

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  CONSENSUS DIVERSITY INDEX REPORT")
        print(f"{'='*60}")
        print(f"\n  Diagnosis: {data['diagnosis']}\n")
        print(f"  Metrics:")
        for k, v in data["metrics"].items():
            print(f"    {k:30s} {v}")
        print(f"\n  Agent Profiles:")
        for aid, p in sorted(data["agent_profiles"].items()):
            print(f"    {aid:15s}  role={p['role']:12s}  avg_vote={p['avg_vote']:+.3f}  "
                  f"reject={p['rejection_rate']:.1%}  rep={p['reputation']:.3f}")
        print(f"\n  Recommendations:")
        for r in data["recommendations"]:
            print(f"    {r}")
        print()

    if args.report:
        html_content = generate_html_report(analysis)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"📄 HTML report saved to {args.report}")


if __name__ == "__main__":
    asyncio.run(main())
