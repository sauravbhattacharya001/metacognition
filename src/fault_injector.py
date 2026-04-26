"""Autonomous Fault Injector — systematic mBFT vulnerability probing.

Runs a battery of attack scenarios against configurable mBFT swarms,
discovers failure modes, and generates an interactive HTML report with
vulnerability severity classification.

Usage::

    python -m src.fault_injector                    # default scan
    python -m src.fault_injector --agents 7         # custom swarm size
    python -m src.fault_injector --threshold 2.0    # custom threshold
    python -m src.fault_injector --output report.html
    python -m src.fault_injector --json results.json
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine


# ── Attack pattern definitions ────────────────────────────────────────

@dataclass
class AttackResult:
    """Outcome of a single attack scenario."""
    attack_name: str
    description: str
    category: str
    num_agents: int
    num_attackers: int
    threshold: float
    committed: bool
    correct: bool  # committed the CORRECT answer
    rounds_used: int
    aggregate_weight: float
    final_reputation: Dict[str, float]
    severity: str = ""  # filled post-hoc
    notes: str = ""


@dataclass
class ScanReport:
    """Full fault-injection scan results."""
    timestamp: str
    total_scenarios: int
    vulnerabilities_found: int
    critical: int
    high: int
    medium: int
    low: int
    results: List[AttackResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


CORRECT_ANSWER = "42"
WRONG_ANSWER = "99"


def _honest_agents(n: int, confidence: float = 0.8) -> List[MockAgent]:
    return [
        MockAgent(f"honest-{i}", answer=CORRECT_ANSWER, confidence=confidence)
        for i in range(n)
    ]


async def _run_scenario(
    agents: List[MockAgent],
    threshold: float,
    max_rounds: int = 4,
) -> tuple[Optional[Any], MBFTEngine]:
    engine = MBFTEngine(agents=agents, threshold=threshold, max_rounds=max_rounds)
    result = await engine.run("fault-injection-probe")
    return result, engine


# ── Attack patterns ───────────────────────────────────────────────────

async def attack_collusion(n_agents: int, threshold: float) -> AttackResult:
    """Multiple Byzantine agents collude on the same wrong answer."""
    n_byz = n_agents // 3
    honest = _honest_agents(n_agents - n_byz)
    colluders = [
        MockAgent(f"colluder-{i}", answer=WRONG_ANSWER, confidence=0.95)
        for i in range(n_byz)
    ]
    result, engine = await _run_scenario(honest + colluders, threshold)
    committed = result is not None and result.committed
    correct = committed and result.committed_solution == CORRECT_ANSWER
    return AttackResult(
        attack_name="Byzantine Collusion",
        description=f"{n_byz} colluding agents push wrong answer with high confidence",
        category="Byzantine",
        num_agents=n_agents,
        num_attackers=n_byz,
        threshold=threshold,
        committed=committed,
        correct=correct,
        rounds_used=len(engine.history),
        aggregate_weight=result.aggregate_weight if result else 0,
        final_reputation=engine.reputation,
    )


async def attack_confidence_inflation(n_agents: int, threshold: float) -> AttackResult:
    """Single attacker with artificially inflated confidence tries to dominate."""
    honest = _honest_agents(n_agents - 1, confidence=0.6)
    inflated = MockAgent("inflated-0", answer=WRONG_ANSWER, confidence=1.0)
    result, engine = await _run_scenario(honest + [inflated], threshold)
    committed = result is not None and result.committed
    correct = committed and result.committed_solution == CORRECT_ANSWER
    return AttackResult(
        attack_name="Confidence Inflation",
        description="Single attacker with max confidence proposes wrong answer",
        category="Epistemic",
        num_agents=n_agents,
        num_attackers=1,
        threshold=threshold,
        committed=committed,
        correct=correct,
        rounds_used=len(engine.history),
        aggregate_weight=result.aggregate_weight if result else 0,
        final_reputation=engine.reputation,
    )


async def attack_sybil_flood(n_agents: int, threshold: float) -> AttackResult:
    """Many low-confidence sybil identities try to overwhelm by quantity."""
    honest = _honest_agents(n_agents)
    sybils = [
        MockAgent(f"sybil-{i}", answer=WRONG_ANSWER, confidence=0.3)
        for i in range(n_agents * 2)
    ]
    result, engine = await _run_scenario(honest + sybils, threshold)
    committed = result is not None and result.committed
    correct = committed and result.committed_solution == CORRECT_ANSWER
    return AttackResult(
        attack_name="Sybil Flood",
        description=f"{n_agents*2} sybil agents flood the swarm (2x honest count)",
        category="Sybil",
        num_agents=n_agents + n_agents * 2,
        num_attackers=n_agents * 2,
        threshold=threshold,
        committed=committed,
        correct=correct,
        rounds_used=len(engine.history),
        aggregate_weight=result.aggregate_weight if result else 0,
        final_reputation=engine.reputation,
    )


async def attack_equivocation(n_agents: int, threshold: float) -> AttackResult:
    """Agents that accept everything (always vote positive regardless of answer)."""
    honest = _honest_agents(n_agents - 2, confidence=0.7)
    # Byzantine agents that always agree (accept_set includes everything)
    equivocators = [
        MockAgent(f"equivocator-{i}", answer=WRONG_ANSWER, confidence=0.9, byzantine=True)
        for i in range(2)
    ]
    result, engine = await _run_scenario(honest + equivocators, threshold)
    committed = result is not None and result.committed
    correct = committed and result.committed_solution == CORRECT_ANSWER
    return AttackResult(
        attack_name="Equivocation",
        description="2 agents always vote positively regardless of proposal quality",
        category="Byzantine",
        num_agents=n_agents,
        num_attackers=2,
        threshold=threshold,
        committed=committed,
        correct=correct,
        rounds_used=len(engine.history),
        aggregate_weight=result.aggregate_weight if result else 0,
        final_reputation=engine.reputation,
    )


async def attack_confidence_drain(n_agents: int, threshold: float) -> AttackResult:
    """All attackers reject everything, trying to prevent any consensus."""
    n_byz = n_agents // 3
    honest = _honest_agents(n_agents - n_byz, confidence=0.8)
    drainers = [
        MockAgent(f"drainer-{i}", answer=f"nonsense-{i}", confidence=0.9)
        for i in range(n_byz)
    ]
    result, engine = await _run_scenario(honest + drainers, threshold)
    committed = result is not None and result.committed
    correct = committed and result.committed_solution == CORRECT_ANSWER
    return AttackResult(
        attack_name="Consensus Denial",
        description=f"{n_byz} agents reject everything to prevent finality",
        category="Liveness",
        num_agents=n_agents,
        num_attackers=n_byz,
        threshold=threshold,
        committed=committed,
        correct=correct,
        rounds_used=len(engine.history),
        aggregate_weight=result.aggregate_weight if result else 0,
        final_reputation=engine.reputation,
    )


async def attack_slow_poison(n_agents: int, threshold: float) -> AttackResult:
    """Attacker gives correct answer but with near-zero confidence to lower aggregate."""
    honest = _honest_agents(n_agents - 2, confidence=0.8)
    poisoners = [
        MockAgent(f"poisoner-{i}", answer=CORRECT_ANSWER, confidence=0.01)
        for i in range(2)
    ]
    result, engine = await _run_scenario(honest + poisoners, threshold)
    committed = result is not None and result.committed
    correct = committed and result.committed_solution == CORRECT_ANSWER
    return AttackResult(
        attack_name="Slow Poison",
        description="2 agents agree but with near-zero confidence to dilute aggregate weight",
        category="Epistemic",
        num_agents=n_agents,
        num_attackers=2,
        threshold=threshold,
        committed=committed,
        correct=correct,
        rounds_used=len(engine.history),
        aggregate_weight=result.aggregate_weight if result else 0,
        final_reputation=engine.reputation,
    )


async def attack_minority_takeover(n_agents: int, threshold: float) -> AttackResult:
    """Minority of high-confidence attackers vs majority of low-confidence honest."""
    n_byz = max(1, n_agents // 4)
    honest = _honest_agents(n_agents - n_byz, confidence=0.4)
    attackers = [
        MockAgent(f"elite-{i}", answer=WRONG_ANSWER, confidence=1.0)
        for i in range(n_byz)
    ]
    result, engine = await _run_scenario(honest + attackers, threshold)
    committed = result is not None and result.committed
    correct = committed and result.committed_solution == CORRECT_ANSWER
    return AttackResult(
        attack_name="Minority Takeover",
        description=f"{n_byz} high-confidence attackers vs {n_agents-n_byz} low-confidence honest",
        category="Epistemic",
        num_agents=n_agents,
        num_attackers=n_byz,
        threshold=threshold,
        committed=committed,
        correct=correct,
        rounds_used=len(engine.history),
        aggregate_weight=result.aggregate_weight if result else 0,
        final_reputation=engine.reputation,
    )


async def attack_threshold_edge(n_agents: int, threshold: float) -> AttackResult:
    """Tune attacker count to land aggregate just at/below threshold boundary."""
    # Find the exact tipping point
    for n_byz in range(n_agents):
        honest = _honest_agents(n_agents - n_byz, confidence=0.8)
        byz = [
            MockAgent(f"edge-{i}", answer=WRONG_ANSWER, confidence=0.8)
            for i in range(n_byz)
        ]
        result, engine = await _run_scenario(honest + byz, threshold)
        if result is None or not result.committed:
            return AttackResult(
                attack_name="Threshold Edge",
                description=f"Consensus breaks with {n_byz}/{n_agents} disagreeing (boundary probe)",
                category="Boundary",
                num_agents=n_agents,
                num_attackers=n_byz,
                threshold=threshold,
                committed=False,
                correct=False,
                rounds_used=len(engine.history),
                aggregate_weight=result.aggregate_weight if result else 0,
                final_reputation=engine.reputation,
            )
    # If all pass, consensus is very robust
    return AttackResult(
        attack_name="Threshold Edge",
        description="Consensus survived even with all agents disagreeing",
        category="Boundary",
        num_agents=n_agents,
        num_attackers=n_agents,
        threshold=threshold,
        committed=True,
        correct=True,
        rounds_used=len(engine.history) if engine else 0,
        aggregate_weight=0,
        final_reputation={},
        notes="Fully resilient at this threshold",
    )


ALL_ATTACKS = [
    attack_collusion,
    attack_confidence_inflation,
    attack_sybil_flood,
    attack_equivocation,
    attack_confidence_drain,
    attack_slow_poison,
    attack_minority_takeover,
    attack_threshold_edge,
]


# ── Severity classification ──────────────────────────────────────────

def classify_severity(r: AttackResult) -> str:
    if r.committed and not r.correct:
        return "CRITICAL"  # wrong answer committed
    if not r.committed and r.category == "Liveness":
        return "HIGH"  # liveness failure
    if r.committed and r.correct and r.num_attackers > r.num_agents // 2:
        return "MEDIUM"  # correct but attackers were majority
    if not r.committed:
        return "MEDIUM"  # failed to reach consensus
    return "LOW"  # attack was handled


def generate_recommendations(results: List[AttackResult]) -> List[str]:
    recs: List[str] = []
    severities = {r.severity for r in results}
    categories = {r.category for r in results if r.severity in ("CRITICAL", "HIGH")}

    if "CRITICAL" in severities:
        recs.append("🚨 Critical: Wrong answers were committed. Consider raising the threshold or adding identity verification.")
    if "Sybil" in categories:
        recs.append("🛡️ Implement Sybil resistance (proof-of-work, staking, or identity attestation) before deployment.")
    if "Epistemic" in categories:
        recs.append("📊 Add confidence calibration checks — agents with consistently poor calibration should be down-weighted.")
    if "Liveness" in categories:
        recs.append("⏱️ Liveness at risk: Add timeout-based fallback or reduce threshold for degraded-mode consensus.")
    if "Boundary" in categories:
        recs.append("📐 Threshold is near the tipping point. Consider adaptive thresholds based on swarm health.")
    if any(r.rounds_used >= 3 for r in results):
        recs.append("🔄 Multiple rounds needed frequently. Consider HotStuff-style pipelining for faster convergence.")
    if not recs:
        recs.append("✅ No critical vulnerabilities found at this configuration. Consider testing with larger swarms.")
    return recs


# ── HTML report generation ────────────────────────────────────────────

def _severity_color(sev: str) -> str:
    return {"CRITICAL": "#dc3545", "HIGH": "#fd7e14", "MEDIUM": "#ffc107", "LOW": "#28a745"}.get(sev, "#6c757d")


def generate_html_report(report: ScanReport) -> str:
    rows = ""
    for r in report.results:
        sc = _severity_color(r.severity)
        outcome = "✅ Correct" if r.correct else ("❌ Wrong committed!" if r.committed else "⚠️ No consensus")
        rows += f"""<tr>
            <td><span class="badge" style="background:{sc}">{r.severity}</span></td>
            <td><strong>{html.escape(r.attack_name)}</strong><br><small>{html.escape(r.description)}</small></td>
            <td>{html.escape(r.category)}</td>
            <td>{r.num_attackers}/{r.num_agents}</td>
            <td>{outcome}</td>
            <td>{r.rounds_used}</td>
            <td>{r.aggregate_weight:.3f}</td>
        </tr>"""

    rec_items = "".join(f"<li>{html.escape(r)}</li>" for r in report.recommendations)

    # Severity chart data
    sev_counts = json.dumps([report.critical, report.high, report.medium, report.low])
    cat_data = {}
    for r in report.results:
        cat_data.setdefault(r.category, {"pass": 0, "fail": 0})
        if r.correct:
            cat_data[r.category]["pass"] += 1
        else:
            cat_data[r.category]["fail"] += 1

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>mBFT Fault Injection Report</title>
<style>
:root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9; --accent: #58a6ff; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); padding: 2rem; }}
h1 {{ color: #f0f6fc; margin-bottom: .5rem; }}
.subtitle {{ color: #8b949e; margin-bottom: 2rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
.stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem; text-align: center; }}
.stat .num {{ font-size: 2rem; font-weight: 700; }}
.stat .label {{ font-size: .85rem; color: #8b949e; margin-top: .3rem; }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; margin-bottom: 2rem; }}
th {{ background: #21262d; padding: .8rem; text-align: left; font-size: .85rem; color: #8b949e; }}
td {{ padding: .8rem; border-top: 1px solid var(--border); font-size: .9rem; }}
.badge {{ padding: 2px 8px; border-radius: 4px; color: #fff; font-weight: 600; font-size: .8rem; }}
.recs {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; }}
.recs h2 {{ color: var(--accent); margin-bottom: 1rem; }}
.recs li {{ margin-bottom: .5rem; line-height: 1.5; }}
canvas {{ background: var(--card); border-radius: 8px; border: 1px solid var(--border); }}
.charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 2rem; }}
@media (max-width: 700px) {{ .charts {{ grid-template-columns: 1fr; }} }}
.filter-bar {{ margin-bottom: 1rem; }}
.filter-bar button {{ background: var(--card); border: 1px solid var(--border); color: var(--text); padding: .4rem 1rem; border-radius: 4px; cursor: pointer; margin-right: .3rem; }}
.filter-bar button.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
</style></head><body>
<h1>🔬 mBFT Fault Injection Report</h1>
<p class="subtitle">Autonomous vulnerability scan — {html.escape(report.timestamp)}</p>

<div class="grid">
    <div class="stat"><div class="num">{report.total_scenarios}</div><div class="label">Scenarios</div></div>
    <div class="stat"><div class="num" style="color:#dc3545">{report.critical}</div><div class="label">Critical</div></div>
    <div class="stat"><div class="num" style="color:#fd7e14">{report.high}</div><div class="label">High</div></div>
    <div class="stat"><div class="num" style="color:#ffc107">{report.medium}</div><div class="label">Medium</div></div>
    <div class="stat"><div class="num" style="color:#28a745">{report.low}</div><div class="label">Low</div></div>
</div>

<div class="charts">
    <canvas id="sevChart" width="400" height="250"></canvas>
    <canvas id="catChart" width="400" height="250"></canvas>
</div>

<div class="filter-bar">
    <button class="active" onclick="filterTable('ALL')">All</button>
    <button onclick="filterTable('CRITICAL')">Critical</button>
    <button onclick="filterTable('HIGH')">High</button>
    <button onclick="filterTable('MEDIUM')">Medium</button>
    <button onclick="filterTable('LOW')">Low</button>
</div>

<table id="results">
<thead><tr><th>Severity</th><th>Attack</th><th>Category</th><th>Attackers</th><th>Outcome</th><th>Rounds</th><th>Aggregate</th></tr></thead>
<tbody>{rows}</tbody>
</table>

<div class="recs"><h2>🤖 Autonomous Recommendations</h2><ul>{rec_items}</ul></div>

<script>
// Severity doughnut
(function() {{
    const c = document.getElementById('sevChart'), ctx = c.getContext('2d');
    const data = {sev_counts}, colors = ['#dc3545','#fd7e14','#ffc107','#28a745'];
    const labels = ['Critical','High','Medium','Low'];
    const total = data.reduce((a,b) => a+b, 0) || 1;
    let angle = -Math.PI/2;
    const cx = c.width/2, cy = c.height/2, r = Math.min(cx,cy) - 40;
    data.forEach((v,i) => {{
        const slice = (v/total) * 2 * Math.PI;
        ctx.beginPath(); ctx.moveTo(cx,cy);
        ctx.arc(cx,cy,r,angle,angle+slice);
        ctx.fillStyle = colors[i]; ctx.fill();
        if (v > 0) {{
            const mid = angle + slice/2;
            ctx.fillStyle = '#fff'; ctx.font = 'bold 13px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(labels[i]+': '+v, cx + Math.cos(mid)*(r*0.65), cy + Math.sin(mid)*(r*0.65));
        }}
        angle += slice;
    }});
    ctx.beginPath(); ctx.arc(cx,cy,r*0.45,0,2*Math.PI); ctx.fillStyle = '#161b22'; ctx.fill();
    ctx.fillStyle = '#c9d1d9'; ctx.font = 'bold 18px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(total + ' tests', cx, cy + 6);
}})();

// Category bar chart
(function() {{
    const c = document.getElementById('catChart'), ctx = c.getContext('2d');
    const cats = {json.dumps(cat_data)};
    const names = Object.keys(cats);
    const barW = Math.min(60, (c.width - 80) / names.length - 10);
    const maxV = Math.max(...names.map(n => cats[n].pass + cats[n].fail), 1);
    const h = c.height - 60, baseY = h + 20;
    ctx.fillStyle = '#8b949e'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    names.forEach((n, i) => {{
        const x = 50 + i * (barW + 10);
        const pH = (cats[n].pass / maxV) * h;
        const fH = (cats[n].fail / maxV) * h;
        ctx.fillStyle = '#28a745'; ctx.fillRect(x, baseY - pH - fH, barW, pH);
        ctx.fillStyle = '#dc3545'; ctx.fillRect(x, baseY - fH, barW, fH);
        ctx.fillStyle = '#8b949e'; ctx.font = '11px sans-serif';
        ctx.fillText(n, x + barW/2, baseY + 15);
    }});
    ctx.fillStyle = '#28a745'; ctx.fillRect(c.width-120, 10, 12, 12);
    ctx.fillStyle = '#c9d1d9'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('Pass', c.width-104, 20);
    ctx.fillStyle = '#dc3545'; ctx.fillRect(c.width-120, 28, 12, 12);
    ctx.fillStyle = '#c9d1d9'; ctx.fillText('Fail', c.width-104, 38);
}})();

// Table filter
function filterTable(sev) {{
    document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('#results tbody tr').forEach(tr => {{
        const badge = tr.querySelector('.badge');
        tr.style.display = (sev === 'ALL' || badge.textContent === sev) ? '' : 'none';
    }});
}}
</script>
</body></html>"""


# ── Main entry point ──────────────────────────────────────────────────

async def run_scan(n_agents: int = 5, threshold: float = 1.5) -> ScanReport:
    """Execute all attack patterns and return a classified report."""
    results: List[AttackResult] = []
    for attack_fn in ALL_ATTACKS:
        result = await attack_fn(n_agents, threshold)
        result.severity = classify_severity(result)
        results.append(result)

    critical = sum(1 for r in results if r.severity == "CRITICAL")
    high = sum(1 for r in results if r.severity == "HIGH")
    medium = sum(1 for r in results if r.severity == "MEDIUM")
    low = sum(1 for r in results if r.severity == "LOW")

    report = ScanReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        total_scenarios=len(results),
        vulnerabilities_found=critical + high,
        critical=critical,
        high=high,
        medium=medium,
        low=low,
        results=results,
    )
    report.recommendations = generate_recommendations(results)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="mBFT Autonomous Fault Injector")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents in swarm (default: 5)")
    parser.add_argument("--threshold", type=float, default=1.5, help="Consensus threshold (default: 1.5)")
    parser.add_argument("--output", type=str, default=None, help="HTML report output path")
    parser.add_argument("--json", type=str, default=None, help="JSON results output path")
    args = parser.parse_args()

    print(f"🔬 mBFT Fault Injector — scanning {len(ALL_ATTACKS)} attack patterns")
    print(f"   agents={args.agents}  threshold={args.threshold}")
    print()

    report = asyncio.run(run_scan(args.agents, args.threshold))

    # Console summary
    for r in report.results:
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(r.severity, "⚪")
        outcome = "CORRECT" if r.correct else ("WRONG COMMITTED" if r.committed else "NO CONSENSUS")
        print(f"  {icon} [{r.severity:8s}] {r.attack_name:25s} → {outcome} (rounds={r.rounds_used}, agg={r.aggregate_weight:.3f})")

    print()
    print(f"Results: {report.critical} critical, {report.high} high, {report.medium} medium, {report.low} low")
    print()
    for rec in report.recommendations:
        print(f"  {rec}")

    if args.output:
        html_content = generate_html_report(report)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"\n📄 HTML report: {args.output}")

    if args.json:
        data = {
            "timestamp": report.timestamp,
            "total_scenarios": report.total_scenarios,
            "vulnerabilities_found": report.vulnerabilities_found,
            "severity_counts": {"critical": report.critical, "high": report.high, "medium": report.medium, "low": report.low},
            "results": [
                {
                    "attack": r.attack_name, "category": r.category, "severity": r.severity,
                    "committed": r.committed, "correct": r.correct, "rounds": r.rounds_used,
                    "aggregate": r.aggregate_weight, "attackers": r.num_attackers, "agents": r.num_agents,
                    "description": r.description, "notes": r.notes,
                }
                for r in report.results
            ],
            "recommendations": report.recommendations,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"📊 JSON results: {args.json}")


if __name__ == "__main__":
    main()
