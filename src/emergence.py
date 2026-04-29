"""Consensus Emergence Detector — autonomous pattern detection in mBFT histories.

Analyzes multi-run consensus histories to detect emergent coordination patterns:
- Leadership concentration (Gini coefficient over leader selections)
- Voting alignment waves (rolling pairwise voter agreement)
- Reputation convergence/divergence phases
- Spontaneous faction formation (hierarchical clustering on vote vectors)
- Consensus momentum (commit-rate trend detection)

Generates interactive HTML reports with charts and proactive recommendations.

Usage:
    python -m src.emergence [--runs N] [--agents N] [--threshold F]
                            [--auto-monitor] [--interval N] [--output FILE]
"""
from __future__ import annotations

import asyncio
import argparse
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from src.core.state import RoundResult


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class EmergenceSignal:
    """A single detected emergence pattern."""
    name: str
    severity: str  # info / warning / critical
    score: float
    description: str
    recommendation: str


@dataclass
class FactionInfo:
    """A detected voting faction."""
    members: List[str]
    cohesion: float  # 0-1 average internal agreement
    label: str


@dataclass
class EmergenceReport:
    """Full emergence analysis for a batch of consensus runs."""
    signals: List[EmergenceSignal] = field(default_factory=list)
    factions: List[FactionInfo] = field(default_factory=list)
    leadership_gini: float = 0.0
    alignment_timeline: List[float] = field(default_factory=list)
    reputation_phases: List[dict] = field(default_factory=list)
    momentum: float = 0.0
    commit_rate: float = 0.0
    total_rounds: int = 0


# ── Analysis Functions ───────────────────────────────────────────

def _gini(values: List[float]) -> float:
    """Compute Gini coefficient."""
    if not values or sum(values) == 0:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    numerator = sum((2 * i - n + 1) * v for i, v in enumerate(sorted_v))
    denominator = n * sum(sorted_v)
    return numerator / denominator if denominator else 0.0


def _pearson(x: List[float], y: List[float]) -> float:
    """Pearson correlation between two series."""
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (sx * sy)


def analyze_emergence(histories: List[List[RoundResult]]) -> EmergenceReport:
    """Run full emergence detection on a batch of consensus run histories."""
    report = EmergenceReport()
    all_rounds: List[RoundResult] = []
    for h in histories:
        all_rounds.extend(h)
    report.total_rounds = len(all_rounds)
    if not all_rounds:
        return report

    # ── Leadership Concentration ─────────────────────────────
    leader_counts = Counter(r.leader_id for r in all_rounds)
    agents = sorted(leader_counts.keys())
    counts = [leader_counts[a] for a in agents]
    report.leadership_gini = _gini(counts)

    if report.leadership_gini > 0.6:
        top_leader = max(leader_counts, key=leader_counts.get)
        pct = leader_counts[top_leader] / len(all_rounds) * 100
        report.signals.append(EmergenceSignal(
            name="Leadership Monopoly",
            severity="critical" if report.leadership_gini > 0.8 else "warning",
            score=report.leadership_gini,
            description=f"Agent '{top_leader}' leads {pct:.0f}% of rounds (Gini={report.leadership_gini:.2f})",
            recommendation="Consider reputation decay or leader rotation to prevent concentration.",
        ))
    elif report.leadership_gini < 0.2:
        report.signals.append(EmergenceSignal(
            name="Healthy Leadership Distribution",
            severity="info",
            score=report.leadership_gini,
            description=f"Leadership is well-distributed (Gini={report.leadership_gini:.2f})",
            recommendation="No action needed — good emergence of shared leadership.",
        ))

    # ── Voting Alignment Waves ───────────────────────────────
    # Per-round: average pairwise agreement among voters
    alignment_series = []
    for rr in all_rounds:
        if len(rr.votes) < 2:
            alignment_series.append(1.0)
            continue
        pairs, agree = 0, 0
        for i in range(len(rr.votes)):
            for j in range(i + 1, len(rr.votes)):
                pairs += 1
                # Agreement = same sign of vote weight
                if (rr.votes[i].weight >= 0) == (rr.votes[j].weight >= 0):
                    agree += 1
        alignment_series.append(agree / pairs if pairs else 1.0)
    report.alignment_timeline = alignment_series

    # Detect alignment waves (rolling window trend)
    window = max(3, len(alignment_series) // 5)
    if len(alignment_series) >= window * 2:
        first_half = statistics.mean(alignment_series[:window])
        last_half = statistics.mean(alignment_series[-window:])
        delta = last_half - first_half
        if delta > 0.3:
            report.signals.append(EmergenceSignal(
                name="Alignment Wave Detected",
                severity="warning",
                score=delta,
                description=f"Voter alignment increased by {delta:.2f} — agents converging toward groupthink.",
                recommendation="Introduce diversity incentives or contrarian bonuses to maintain healthy disagreement.",
            ))
        elif delta < -0.3:
            report.signals.append(EmergenceSignal(
                name="Alignment Collapse",
                severity="warning",
                score=abs(delta),
                description=f"Voter alignment dropped by {abs(delta):.2f} — consensus is fragmenting.",
                recommendation="Review agent calibration — increasing fragmentation may indicate Byzantine drift.",
            ))

    # ── Reputation Convergence/Divergence ────────────────────
    # Track reputation variance across runs
    rep_snapshots: List[Dict[str, float]] = []
    from src.core.protocol import MBFTEngine
    # We reconstruct rep from slash history
    rep = defaultdict(lambda: 1.0)
    for rr in all_rounds:
        for s in rr.slashed:
            rep[s] *= 0.5
        rep_snapshots.append(dict(rep))

    if len(rep_snapshots) >= 3:
        variances = []
        for snap in rep_snapshots:
            vals = list(snap.values())
            variances.append(statistics.variance(vals) if len(vals) > 1 else 0.0)
        # Detect phases
        phases = []
        prev_trend = None
        phase_start = 0
        for i in range(1, len(variances)):
            trend = "diverging" if variances[i] > variances[i-1] else "converging"
            if trend != prev_trend:
                if prev_trend is not None:
                    phases.append({"phase": prev_trend, "start": phase_start, "end": i - 1,
                                   "var_start": variances[phase_start], "var_end": variances[i-1]})
                phase_start = i
                prev_trend = trend
            elif i == len(variances) - 1:
                phases.append({"phase": trend, "start": phase_start, "end": i,
                               "var_start": variances[phase_start], "var_end": variances[i]})
        report.reputation_phases = phases[-5:]  # last 5 phases

        if variances[-1] > 0.3:
            report.signals.append(EmergenceSignal(
                name="Reputation Divergence",
                severity="warning",
                score=variances[-1],
                description=f"High reputation variance ({variances[-1]:.2f}) — some agents heavily penalized.",
                recommendation="Consider reputation recovery mechanisms to prevent permanent exclusion.",
            ))

    # ── Faction Detection ────────────────────────────────────
    # Build vote vectors per agent, then cluster by agreement
    agent_votes: Dict[str, List[float]] = defaultdict(list)
    for rr in all_rounds:
        for v in rr.votes:
            agent_votes[v.voter_id].append(1.0 if v.weight >= 0 else -1.0)

    if len(agent_votes) >= 3:
        # Pad to same length
        max_len = max(len(v) for v in agent_votes.values())
        for aid in agent_votes:
            while len(agent_votes[aid]) < max_len:
                agent_votes[aid].append(0.0)

        voter_ids = sorted(agent_votes.keys())
        # Simple greedy clustering: agents with pearson > 0.5 form a faction
        assigned = set()
        factions = []
        for i, a in enumerate(voter_ids):
            if a in assigned:
                continue
            faction = [a]
            assigned.add(a)
            for j in range(i + 1, len(voter_ids)):
                b = voter_ids[j]
                if b in assigned:
                    continue
                corr = _pearson(agent_votes[a], agent_votes[b])
                if corr > 0.5:
                    faction.append(b)
                    assigned.add(b)
            if len(faction) >= 2:
                # Compute cohesion
                pairs_corr = []
                for ii in range(len(faction)):
                    for jj in range(ii + 1, len(faction)):
                        pairs_corr.append(_pearson(agent_votes[faction[ii]], agent_votes[faction[jj]]))
                cohesion = statistics.mean(pairs_corr) if pairs_corr else 0.0
                factions.append(FactionInfo(members=faction, cohesion=cohesion,
                                            label=f"Faction-{len(factions)+1}"))
        report.factions = factions
        if len(factions) >= 2:
            report.signals.append(EmergenceSignal(
                name="Faction Emergence",
                severity="warning" if any(f.cohesion > 0.8 for f in factions) else "info",
                score=max(f.cohesion for f in factions),
                description=f"{len(factions)} voting factions detected (max cohesion={max(f.cohesion for f in factions):.2f})",
                recommendation="Monitor for collusion — high-cohesion factions may coordinate to manipulate consensus.",
            ))

    # ── Consensus Momentum ───────────────────────────────────
    commits = [1.0 if rr.committed else 0.0 for rr in all_rounds]
    report.commit_rate = statistics.mean(commits) if commits else 0.0
    # Linear trend via correlation with time index
    if len(commits) >= 5:
        time_idx = list(range(len(commits)))
        report.momentum = _pearson(time_idx, commits)
        if report.momentum < -0.3:
            report.signals.append(EmergenceSignal(
                name="Consensus Stalling",
                severity="critical" if report.momentum < -0.5 else "warning",
                score=abs(report.momentum),
                description=f"Commit rate trending down (r={report.momentum:.2f}) — consensus becoming harder to reach.",
                recommendation="Consider lowering threshold or increasing agent count to restore consensus flow.",
            ))
        elif report.momentum > 0.3:
            report.signals.append(EmergenceSignal(
                name="Consensus Strengthening",
                severity="info",
                score=report.momentum,
                description=f"Commit rate trending up (r={report.momentum:.2f}) — agents converging effectively.",
                recommendation="Good trend — monitor for overconfidence or suppressed dissent.",
            ))

    # Add an overall health signal
    crit_count = sum(1 for s in report.signals if s.severity == "critical")
    warn_count = sum(1 for s in report.signals if s.severity == "warning")
    if crit_count == 0 and warn_count == 0:
        report.signals.insert(0, EmergenceSignal(
            name="Healthy Emergence",
            severity="info",
            score=1.0,
            description="No concerning emergence patterns detected.",
            recommendation="Continue monitoring — emergence patterns can shift rapidly.",
        ))

    return report


# ── HTML Report Generator ────────────────────────────────────────

def generate_html_report(report: EmergenceReport) -> str:
    """Generate interactive HTML report with charts."""
    severity_colors = {"info": "#22c55e", "warning": "#f59e0b", "critical": "#ef4444"}
    severity_icons = {"info": "✅", "warning": "⚠️", "critical": "🚨"}

    signals_html = ""
    for s in report.signals:
        color = severity_colors[s.severity]
        icon = severity_icons[s.severity]
        signals_html += f"""
        <div class="signal" style="border-left: 4px solid {color}">
            <div class="signal-header">
                <span>{icon} {html.escape(s.name)}</span>
                <span class="badge" style="background:{color}">{s.severity.upper()} ({s.score:.2f})</span>
            </div>
            <p>{html.escape(s.description)}</p>
            <p class="rec">💡 {html.escape(s.recommendation)}</p>
        </div>"""

    factions_html = ""
    if report.factions:
        for f in report.factions:
            members = ", ".join(html.escape(m) for m in f.members)
            factions_html += f"""
            <div class="faction">
                <strong>{html.escape(f.label)}</strong> (cohesion: {f.cohesion:.2f})<br>
                Members: {members}
            </div>"""
    else:
        factions_html = "<p>No distinct factions detected.</p>"

    phases_html = ""
    for p in report.reputation_phases:
        arrow = "📈" if p["phase"] == "diverging" else "📉"
        phases_html += f"<div class='phase'>{arrow} Rounds {p['start']}-{p['end']}: {p['phase']} (var {p['var_start']:.3f} → {p['var_end']:.3f})</div>"

    alignment_json = json.dumps(report.alignment_timeline)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>mBFT Emergence Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
  h1 {{ color: #38bdf8; margin-bottom: 8px; }}
  h2 {{ color: #7dd3fc; margin: 24px 0 12px; }}
  .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
  .stat {{ background: #1e293b; border-radius: 8px; padding: 16px; min-width: 150px; text-align: center; }}
  .stat .val {{ font-size: 2em; font-weight: bold; color: #38bdf8; }}
  .stat .lbl {{ font-size: 0.85em; color: #94a3b8; }}
  .signal {{ background: #1e293b; border-radius: 8px; padding: 16px; margin: 8px 0; }}
  .signal-header {{ display: flex; justify-content: space-between; align-items: center; font-weight: bold; margin-bottom: 8px; }}
  .badge {{ color: #fff; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; }}
  .rec {{ color: #94a3b8; font-style: italic; margin-top: 8px; }}
  .faction {{ background: #1e293b; border-radius: 8px; padding: 12px; margin: 6px 0; }}
  .phase {{ padding: 4px 0; }}
  canvas {{ background: #1e293b; border-radius: 8px; margin: 12px 0; }}
</style></head><body>
<h1>🔮 Consensus Emergence Report</h1>
<p>Analyzed {report.total_rounds} rounds across {len(report.alignment_timeline)} consensus rounds</p>

<div class="stats">
  <div class="stat"><div class="val">{report.commit_rate:.0%}</div><div class="lbl">Commit Rate</div></div>
  <div class="stat"><div class="val">{report.leadership_gini:.2f}</div><div class="lbl">Leadership Gini</div></div>
  <div class="stat"><div class="val">{report.momentum:+.2f}</div><div class="lbl">Momentum</div></div>
  <div class="stat"><div class="val">{len(report.factions)}</div><div class="lbl">Factions</div></div>
  <div class="stat"><div class="val">{len(report.signals)}</div><div class="lbl">Signals</div></div>
</div>

<h2>🚦 Emergence Signals</h2>
{signals_html}

<h2>📊 Voting Alignment Timeline</h2>
<canvas id="alignChart" width="800" height="200"></canvas>

<h2>🏛️ Detected Factions</h2>
{factions_html}

<h2>📈 Reputation Phases</h2>
{phases_html if phases_html else "<p>Insufficient data for phase detection.</p>"}

<script>
(function() {{
  const data = {alignment_json};
  const canvas = document.getElementById('alignChart');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const pad = 40;
  const pw = w - 2 * pad, ph = h - 2 * pad;

  // Grid
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 0.5;
  for (let y = 0; y <= 1; y += 0.25) {{
    const py = pad + ph * (1 - y);
    ctx.beginPath(); ctx.moveTo(pad, py); ctx.lineTo(pad + pw, py); ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '11px monospace';
    ctx.fillText(y.toFixed(2), 4, py + 4);
  }}

  if (data.length < 2) return;
  // Line
  ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((v, i) => {{
    const x = pad + (i / (data.length - 1)) * pw;
    const y = pad + ph * (1 - v);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();

  // Rolling average
  const win = Math.max(3, Math.floor(data.length / 10));
  ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]);
  ctx.beginPath();
  let started = false;
  for (let i = win - 1; i < data.length; i++) {{
    let sum = 0;
    for (let j = i - win + 1; j <= i; j++) sum += data[j];
    const avg = sum / win;
    const x = pad + (i / (data.length - 1)) * pw;
    const y = pad + ph * (1 - avg);
    started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), started = true);
  }}
  ctx.stroke(); ctx.setLineDash([]);

  ctx.fillStyle = '#94a3b8'; ctx.font = '11px sans-serif';
  ctx.fillText('Round →', pad + pw / 2 - 20, h - 4);
}})();
</script>
</body></html>"""


# ── CLI + Simulation Runner ──────────────────────────────────────

async def run_simulation(n_runs: int, n_agents: int, threshold: float) -> EmergenceReport:
    """Run multiple consensus rounds and analyze emergence."""
    from src.agents.metacognitive import MockAgent
    from src.core.protocol import MBFTEngine

    answers = ["A", "B", "A", "A", "B", "C", "A", "B"]
    agents = [MockAgent(
        agent_id=f"agent-{i}",
        answer=answers[i % len(answers)],
        confidence=0.3 + 0.5 * (i / max(1, n_agents - 1)),
        byzantine=(i == n_agents - 1),  # last agent is Byzantine
    ) for i in range(n_agents)]

    histories = []
    for run_idx in range(n_runs):
        engine = MBFTEngine(agents=agents, threshold=threshold, max_rounds=4)
        await engine.run(f"Task-{run_idx}: Solve consensus problem #{run_idx}")
        histories.append(engine.history)

    return analyze_emergence(histories)


async def auto_monitor(n_runs: int, n_agents: int, threshold: float,
                       interval: int, output: str) -> None:
    """Continuous monitoring mode — re-run analysis periodically."""
    run_count = 0
    while True:
        run_count += 1
        print(f"\n[Monitor] Run batch #{run_count}...")
        report = await run_simulation(n_runs, n_agents, threshold)

        html_content = generate_html_report(report)
        with open(output, "w", encoding="utf-8") as f:
            f.write(html_content)

        crit = sum(1 for s in report.signals if s.severity == "critical")
        warn = sum(1 for s in report.signals if s.severity == "warning")
        print(f"[Monitor] Signals: {crit} critical, {warn} warnings, commit rate={report.commit_rate:.0%}")
        for s in report.signals:
            if s.severity in ("critical", "warning"):
                print(f"  {'🚨' if s.severity == 'critical' else '⚠️'} {s.name}: {s.description}")

        print(f"[Monitor] Report saved to {output}. Next check in {interval}s...")
        await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="mBFT Consensus Emergence Detector")
    parser.add_argument("--runs", type=int, default=20, help="Number of consensus runs to simulate")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--threshold", type=float, default=2.0, help="Consensus threshold")
    parser.add_argument("--auto-monitor", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=60, help="Monitor interval in seconds")
    parser.add_argument("--output", type=str, default="emergence_report.html", help="Output HTML file")
    args = parser.parse_args()

    if args.auto_monitor:
        print(f"🔮 Emergence Monitor — {args.agents} agents, threshold={args.threshold}, interval={args.interval}s")
        asyncio.run(auto_monitor(args.runs, args.agents, args.threshold, args.interval, args.output))
    else:
        print(f"🔮 Emergence Detector — {args.runs} runs, {args.agents} agents, threshold={args.threshold}")
        report = asyncio.run(run_simulation(args.runs, args.agents, args.threshold))

        html_content = generate_html_report(report)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"\n{'='*60}")
        print(f"Rounds analyzed: {report.total_rounds}")
        print(f"Commit rate: {report.commit_rate:.0%}")
        print(f"Leadership Gini: {report.leadership_gini:.2f}")
        print(f"Momentum: {report.momentum:+.2f}")
        print(f"Factions: {len(report.factions)}")
        print(f"\nSignals:")
        for s in report.signals:
            icon = {"info": "✅", "warning": "⚠️", "critical": "🚨"}[s.severity]
            print(f"  {icon} [{s.severity.upper()}] {s.name} ({s.score:.2f})")
            print(f"     {s.description}")
            print(f"     💡 {s.recommendation}")
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
