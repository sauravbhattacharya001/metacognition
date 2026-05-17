"""Autonomous Consensus Resilience Monitor.

Stress-tests an mBFT swarm by systematically varying the number of Byzantine
agents, confidence distributions, and threshold settings to map the protocol's
fault-tolerance boundary and produce actionable recommendations.

Usage::

    python -m src.monitor                        # default analysis
    python -m src.monitor --agents 7             # custom swarm size
    python -m src.monitor --export json           # JSON export
    python -m src.monitor --export html           # interactive HTML report
    python -m src.monitor --sweep-thresholds      # also sweep θ values
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine


@dataclass
class ScenarioResult:
    """Outcome of a single stress-test scenario."""
    total_agents: int
    byzantine_count: int
    threshold: float
    committed: bool
    rounds_used: int
    aggregate_weight: float
    solution: Optional[str]
    reputation_after: dict


@dataclass
class ResilienceReport:
    """Aggregated resilience analysis."""
    swarm_size: int
    threshold: float
    max_byzantine_tolerated: int
    fault_tolerance_ratio: float
    scenarios: List[ScenarioResult] = field(default_factory=list)
    threshold_sweep: Optional[dict] = None
    recommendations: List[str] = field(default_factory=list)


def _build_swarm(
    n: int,
    byzantine_count: int,
    honest_confidence: float = 0.80,
    byzantine_confidence: float = 0.95,
) -> list[MockAgent]:
    agents = []
    for i in range(n):
        is_byz = i >= (n - byzantine_count)
        agents.append(
            MockAgent(
                agent_id=f"a{i+1}",
                answer="correct" if not is_byz else f"byz-{i}",
                confidence=honest_confidence if not is_byz else byzantine_confidence,
                byzantine=is_byz,
            )
        )
    return agents


async def run_scenario(
    n: int, byzantine_count: int, threshold: float
) -> ScenarioResult:
    agents = _build_swarm(n, byzantine_count)
    engine = MBFTEngine(agents=agents, threshold=threshold, max_rounds=4)
    result = await engine.run("resilience-test-task")
    committed = result is not None and result.committed
    return ScenarioResult(
        total_agents=n,
        byzantine_count=byzantine_count,
        threshold=threshold,
        committed=committed,
        rounds_used=len(engine.history),
        aggregate_weight=result.aggregate_weight if result else 0.0,
        solution=result.committed_solution if result else None,
        reputation_after=engine.reputation,
    )


async def analyze_resilience(
    swarm_size: int, threshold: float, sweep_thresholds: bool = False
) -> ResilienceReport:
    scenarios: List[ScenarioResult] = []
    max_tolerated = 0

    # Sweep Byzantine count from 0 to n-1
    for byz in range(swarm_size):
        sc = await run_scenario(swarm_size, byz, threshold)
        scenarios.append(sc)
        if sc.committed and sc.solution == "correct":
            max_tolerated = byz

    report = ResilienceReport(
        swarm_size=swarm_size,
        threshold=threshold,
        max_byzantine_tolerated=max_tolerated,
        fault_tolerance_ratio=max_tolerated / swarm_size if swarm_size > 0 else 0,
        scenarios=scenarios,
    )

    # Optional threshold sweep
    if sweep_thresholds:
        sweep = {}
        for t10 in range(5, 35, 5):  # 0.5 to 3.0
            t = t10 / 10.0
            best = 0
            for byz in range(swarm_size):
                sc = await run_scenario(swarm_size, byz, t)
                if sc.committed and sc.solution == "correct":
                    best = byz
            sweep[f"{t:.1f}"] = {
                "max_byzantine": best,
                "ratio": round(best / swarm_size, 3),
            }
        report.threshold_sweep = sweep

    # Generate recommendations
    report.recommendations = _generate_recommendations(report)
    return report


def _generate_recommendations(report: ResilienceReport) -> List[str]:
    recs = []
    ratio = report.fault_tolerance_ratio

    if ratio < 0.20:
        recs.append(
            "⚠️ CRITICAL: Fault tolerance below 20%. The swarm cannot "
            "withstand even a small minority of Byzantine agents. "
            "Lower the threshold θ or increase swarm size."
        )
    elif ratio < 0.33:
        recs.append(
            "🟡 Fault tolerance is below the classic BFT ⅓ bound. "
            "Consider lowering θ to improve resilience."
        )
    else:
        recs.append(
            f"✅ Fault tolerance ratio {ratio:.1%} meets or exceeds "
            "the classic BFT ⅓ bound."
        )

    # Check if threshold sweep reveals a better setting
    if report.threshold_sweep:
        best_t = max(
            report.threshold_sweep.items(),
            key=lambda kv: kv[1]["ratio"],
        )
        if best_t[1]["ratio"] > ratio:
            recs.append(
                f"💡 Threshold θ={best_t[0]} achieves better fault tolerance "
                f"({best_t[1]['ratio']:.1%} vs current {ratio:.1%}). "
                "Consider adjusting."
            )

    # Swarm size recommendation
    if report.swarm_size < 5:
        recs.append(
            "📈 Swarm has fewer than 5 agents. Larger swarms provide "
            "more granular fault tolerance. Consider adding agents."
        )

    # Check for scenarios where Byzantine agents got correct answer committed
    false_commits = [
        s for s in report.scenarios
        if s.committed and s.solution != "correct"
    ]
    if false_commits:
        recs.append(
            f"🚨 SAFETY ALERT: {len(false_commits)} scenario(s) committed "
            "an incorrect solution! Byzantine agents may be gaming the "
            "confidence-weighted voting. Investigate threshold and reputation."
        )

    return recs


def _to_dict(report: ResilienceReport) -> dict:
    return {
        "swarm_size": report.swarm_size,
        "threshold": report.threshold,
        "max_byzantine_tolerated": report.max_byzantine_tolerated,
        "fault_tolerance_ratio": round(report.fault_tolerance_ratio, 3),
        "recommendations": report.recommendations,
        "scenarios": [
            {
                "byzantine_count": s.byzantine_count,
                "committed": s.committed,
                "solution": s.solution,
                "rounds_used": s.rounds_used,
                "aggregate_weight": round(s.aggregate_weight, 3),
            }
            for s in report.scenarios
        ],
        "threshold_sweep": report.threshold_sweep,
    }


def _render_html(report: ResilienceReport) -> str:
    data = _to_dict(report)
    scenarios_json = json.dumps(data["scenarios"])
    sweep_json = json.dumps(data.get("threshold_sweep") or {})
    recs_html = "".join(f"<li>{r}</li>" for r in report.recommendations)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>mBFT Resilience Report</title>
<style>
  :root {{ --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff; --red: #f85149; --green: #3fb950; --yellow: #d29922; --card: #161b22; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: var(--bg); color: var(--fg); padding: 2rem; }}
  h1 {{ color: var(--accent); margin-bottom: 0.5rem; }}
  .subtitle {{ color: #8b949e; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: var(--card); border-radius: 8px; padding: 1.2rem; }}
  .card h3 {{ font-size: 0.85rem; color: #8b949e; text-transform: uppercase; }}
  .card .value {{ font-size: 2rem; font-weight: 700; margin-top: 0.3rem; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }} .yellow {{ color: var(--yellow); }}
  canvas {{ background: var(--card); border-radius: 8px; margin-bottom: 2rem; }}
  .recs {{ background: var(--card); border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; }}
  .recs li {{ margin: 0.5rem 0; line-height: 1.5; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #21262d; }}
  th {{ background: #21262d; font-size: 0.85rem; text-transform: uppercase; color: #8b949e; }}
  .committed {{ color: var(--green); }} .failed {{ color: var(--red); }}
</style></head><body>
<h1>🛡️ mBFT Resilience Report</h1>
<p class="subtitle">Swarm Size: {report.swarm_size} agents | Threshold θ = {report.threshold}</p>

<div class="grid">
  <div class="card"><h3>Max Byzantine Tolerated</h3>
    <div class="value {'green' if report.fault_tolerance_ratio >= 0.33 else 'yellow' if report.fault_tolerance_ratio >= 0.2 else 'red'}">{report.max_byzantine_tolerated}</div></div>
  <div class="card"><h3>Fault Tolerance Ratio</h3>
    <div class="value">{report.fault_tolerance_ratio:.1%}</div></div>
  <div class="card"><h3>Scenarios Tested</h3>
    <div class="value">{len(report.scenarios)}</div></div>
  <div class="card"><h3>Commits Achieved</h3>
    <div class="value green">{sum(1 for s in report.scenarios if s.committed)}</div></div>
</div>

<h2 style="margin-bottom:1rem">📊 Resilience Curve</h2>
<canvas id="chart" width="800" height="300"></canvas>

<h2 style="margin-bottom:1rem">📋 Recommendations</h2>
<div class="recs"><ul>{recs_html}</ul></div>

<h2 style="margin-bottom:1rem">📄 Scenario Details</h2>
<table>
  <thead><tr><th>Byzantine</th><th>Committed</th><th>Solution</th><th>Rounds</th><th>Aggregate</th></tr></thead>
  <tbody id="tbody"></tbody>
</table>

<script>
const scenarios = {scenarios_json};
const sweep = {sweep_json};

// Scenario table
const tbody = document.getElementById('tbody');
scenarios.forEach(s => {{
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${{s.byzantine_count}}</td>
    <td class="${{s.committed ? 'committed' : 'failed'}}">${{s.committed ? '✅ Yes' : '❌ No'}}</td>
    <td>${{s.solution || '—'}}</td><td>${{s.rounds_used}}</td>
    <td>${{s.aggregate_weight.toFixed(3)}}</td>`;
  tbody.appendChild(tr);
}});

// Chart
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const pad = {{ l: 60, r: 20, t: 20, b: 40 }};
const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

ctx.fillStyle = '#8b949e'; ctx.font = '12px sans-serif';
ctx.textAlign = 'center';
ctx.fillText('Byzantine Agent Count', W / 2, H - 5);
ctx.save(); ctx.translate(15, H / 2); ctx.rotate(-Math.PI / 2);
ctx.fillText('Aggregate Weight', 0, 0); ctx.restore();

const maxW = Math.max(...scenarios.map(s => s.aggregate_weight), {report.threshold});
scenarios.forEach((s, i) => {{
  const x = pad.l + (i / Math.max(scenarios.length - 1, 1)) * pw;
  const y = pad.t + ph - (s.aggregate_weight / maxW) * ph;
  ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.fillStyle = s.committed ? '#3fb950' : '#f85149'; ctx.fill();
  ctx.fillStyle = '#8b949e'; ctx.font = '10px sans-serif';
  ctx.fillText(s.byzantine_count, x, H - pad.b + 15);
}});

// Threshold line
const ty = pad.t + ph - ({report.threshold} / maxW) * ph;
ctx.strokeStyle = '#d29922'; ctx.setLineDash([5, 5]);
ctx.beginPath(); ctx.moveTo(pad.l, ty); ctx.lineTo(W - pad.r, ty); ctx.stroke();
ctx.fillStyle = '#d29922'; ctx.textAlign = 'left';
ctx.fillText('θ = {report.threshold}', W - pad.r - 60, ty - 5);
</script></body></html>"""


def _print_report(report: ResilienceReport) -> None:
    import io, sys as _sys
    out = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    p = lambda *a, **kw: print(*a, **kw, file=out)
    p("=" * 60)
    p("  mBFT CONSENSUS RESILIENCE REPORT")
    p("=" * 60)
    p(f"  Swarm Size:              {report.swarm_size} agents")
    p(f"  Threshold:               {report.threshold}")
    p(f"  Max Byzantine Tolerated: {report.max_byzantine_tolerated}")
    p(f"  Fault Tolerance Ratio:   {report.fault_tolerance_ratio:.1%}")
    p()

    p("  SCENARIO RESULTS:")
    p(f"  {'Byz':>4}  {'Committed':>9}  {'Solution':>10}  {'Rounds':>6}  {'Sum V':>8}")
    p("  " + "-" * 48)
    for s in report.scenarios:
        marker = "YES" if s.committed else "NO"
        sol = (s.solution or "-")[:10]
        p(
            f"  {s.byzantine_count:>4}  {marker:>9}  {sol:>10}  "
            f"{s.rounds_used:>6}  {s.aggregate_weight:>8.3f}"
        )

    if report.threshold_sweep:
        p()
        p("  THRESHOLD SWEEP:")
        p(f"  {'Thr':>5}  {'Max Byz':>8}  {'Ratio':>8}")
        p("  " + "-" * 26)
        for t, info in sorted(report.threshold_sweep.items()):
            p(f"  {t:>5}  {info['max_byzantine']:>8}  {info['ratio']:>8.1%}")

    p()
    p("  RECOMMENDATIONS:")
    for r in report.recommendations:
        # Strip emoji for console
        clean = r.encode("ascii", "ignore").decode("ascii").strip()
        p(f"    {clean}")
    p("=" * 60)
    out.flush()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Resilience Monitor"
    )
    parser.add_argument(
        "--agents", "-n", type=int, default=7,
        help="Number of agents in the swarm (default: 7)",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=1.5,
        help="Consensus threshold θ (default: 1.5)",
    )
    parser.add_argument(
        "--sweep-thresholds", action="store_true",
        help="Also sweep θ values to find optimal threshold",
    )
    parser.add_argument(
        "--export", choices=["json", "html"],
        help="Export report as JSON or interactive HTML",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output file path (default: stdout for JSON, resilience_report.html for HTML)",
    )
    args = parser.parse_args()

    report = await analyze_resilience(
        swarm_size=args.agents,
        threshold=args.threshold,
        sweep_thresholds=args.sweep_thresholds,
    )

    if args.export == "json":
        data = json.dumps(_to_dict(report), indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(data)
            print(f"Report written to {args.output}")
        else:
            print(data)
    elif args.export == "html":
        html = _render_html(report)
        out = args.output or "resilience_report.html"
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Interactive report written to {out}")
    else:
        _print_report(report)


if __name__ == "__main__":
    asyncio.run(main())
