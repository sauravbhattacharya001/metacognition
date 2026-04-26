"""Consensus Parameter Optimizer — autonomous search for optimal mBFT configs.

Sweeps across threshold, Byzantine fraction, agent count, and confidence
distributions to find configurations that maximize commit rate while
maintaining safety (no false commits under Byzantine conditions).

Usage::

    python -m src.optimizer                          # default sweep
    python -m src.optimizer --agents 3 5 7 9         # custom agent counts
    python -m src.optimizer --thresholds 0.5 1.0 1.5 2.0
    python -m src.optimizer --byzantine-fractions 0.0 0.1 0.2 0.33
    python -m src.optimizer --trials 50              # trials per config
    python -m src.optimizer --html report.html       # interactive HTML report
    python -m src.optimizer --json results.json      # JSON export
    python -m src.optimizer --auto                   # autonomous recommendation
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    committed: bool
    rounds_used: int
    aggregate_weight: float
    leader_slashed: bool
    false_commit: bool  # committed a Byzantine answer


@dataclass
class ConfigResult:
    n_agents: int
    n_byzantine: int
    threshold: float
    confidence_profile: str
    trials: int
    commit_rate: float
    avg_rounds: float
    false_commit_rate: float
    avg_aggregate: float
    safety_score: float  # 1.0 - false_commit_rate
    efficiency_score: float  # commit_rate * (1 / avg_rounds)
    overall_score: float  # safety * efficiency


@dataclass
class OptimizationReport:
    configs: List[ConfigResult] = field(default_factory=list)
    best_config: Optional[ConfigResult] = None
    recommendations: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    total_trials: int = 0


# ---------------------------------------------------------------------------
# Confidence profiles
# ---------------------------------------------------------------------------

CONFIDENCE_PROFILES = {
    "uniform_high": lambda n: [0.85 + random.random() * 0.15 for _ in range(n)],
    "uniform_mid": lambda n: [0.4 + random.random() * 0.3 for _ in range(n)],
    "mixed": lambda n: [0.3 + random.random() * 0.6 for _ in range(n)],
    "polarized": lambda n: [
        random.choice([0.2 + random.random() * 0.1, 0.8 + random.random() * 0.2])
        for _ in range(n)
    ],
}


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

async def run_trial(
    n_agents: int,
    n_byzantine: int,
    threshold: float,
    confidence_profile: str,
    max_rounds: int = 4,
) -> TrialResult:
    """Run a single mBFT trial with the given configuration."""
    confs = CONFIDENCE_PROFILES[confidence_profile](n_agents)
    correct_answer = "42"
    byzantine_answer = "999"

    agents: list[MockAgent] = []
    byzantine_ids: set[str] = set()

    # Assign Byzantine agents randomly
    indices = list(range(n_agents))
    random.shuffle(indices)
    byz_indices = set(indices[:n_byzantine])

    for i in range(n_agents):
        aid = f"agent_{i}"
        if i in byz_indices:
            agents.append(MockAgent(
                aid,
                answer=byzantine_answer,
                confidence=confs[i],
                byzantine=True,
            ))
            byzantine_ids.add(aid)
        else:
            agents.append(MockAgent(
                aid,
                answer=correct_answer,
                confidence=confs[i],
            ))

    engine = MBFTEngine(agents=agents, threshold=threshold, max_rounds=max_rounds)
    result = await engine.run("consensus-test")

    committed = result is not None and result.committed
    false_commit = (
        committed
        and result is not None
        and result.committed_solution == byzantine_answer
    )
    rounds_used = len(engine.history)
    agg = result.aggregate_weight if result else 0.0
    leader_slashed = bool(result and result.slashed)

    return TrialResult(
        committed=committed,
        rounds_used=rounds_used,
        aggregate_weight=agg,
        leader_slashed=leader_slashed,
        false_commit=false_commit,
    )


async def evaluate_config(
    n_agents: int,
    n_byzantine: int,
    threshold: float,
    confidence_profile: str,
    trials: int,
) -> ConfigResult:
    """Evaluate a parameter configuration over multiple trials."""
    results = await asyncio.gather(
        *(run_trial(n_agents, n_byzantine, threshold, confidence_profile)
          for _ in range(trials))
    )

    commits = sum(1 for r in results if r.committed)
    false_commits = sum(1 for r in results if r.false_commit)
    avg_rounds = sum(r.rounds_used for r in results) / len(results)
    avg_agg = sum(r.aggregate_weight for r in results) / len(results)

    commit_rate = commits / trials
    false_commit_rate = false_commits / trials
    safety_score = 1.0 - false_commit_rate
    efficiency_score = commit_rate * (1.0 / max(avg_rounds, 0.1))
    overall_score = safety_score * efficiency_score

    return ConfigResult(
        n_agents=n_agents,
        n_byzantine=n_byzantine,
        threshold=threshold,
        confidence_profile=confidence_profile,
        trials=trials,
        commit_rate=commit_rate,
        avg_rounds=avg_rounds,
        false_commit_rate=false_commit_rate,
        avg_aggregate=avg_agg,
        safety_score=safety_score,
        efficiency_score=efficiency_score,
        overall_score=overall_score,
    )


# ---------------------------------------------------------------------------
# Autonomous recommendation engine
# ---------------------------------------------------------------------------

def generate_recommendations(report: OptimizationReport) -> List[str]:
    """Analyze results and produce actionable recommendations."""
    recs: List[str] = []
    if not report.configs:
        return ["No configurations evaluated."]

    best = report.best_config
    if best is None:
        return ["No safe configuration found."]

    # Safety analysis
    unsafe = [c for c in report.configs if c.false_commit_rate > 0.0]
    if unsafe:
        recs.append(
            f"⚠️ {len(unsafe)} configuration(s) produced false commits. "
            f"Avoid thresholds below {min(c.threshold for c in unsafe):.2f} "
            f"with Byzantine agents present."
        )

    # Threshold analysis
    safe_configs = [c for c in report.configs if c.false_commit_rate == 0.0]
    if safe_configs:
        thresholds = sorted(set(c.threshold for c in safe_configs))
        best_t = best.threshold
        recs.append(
            f"✅ Optimal threshold: {best_t:.2f} "
            f"(commit rate {best.commit_rate:.0%}, "
            f"avg {best.avg_rounds:.1f} rounds)."
        )

    # Agent count analysis
    by_agents: Dict[int, List[ConfigResult]] = {}
    for c in safe_configs:
        by_agents.setdefault(c.n_agents, []).append(c)
    if by_agents:
        best_n = max(by_agents, key=lambda n: max(c.overall_score for c in by_agents[n]))
        recs.append(
            f"📊 Best agent count: {best_n} agents "
            f"(among safe configurations)."
        )

    # Byzantine tolerance
    if best.n_byzantine > 0:
        byz_frac = best.n_byzantine / best.n_agents
        recs.append(
            f"🛡️ Best config tolerates {best.n_byzantine}/{best.n_agents} "
            f"Byzantine agents ({byz_frac:.0%}) with "
            f"{best.commit_rate:.0%} commit rate."
        )

    # Profile analysis
    profile_scores: Dict[str, float] = {}
    for c in safe_configs:
        if c.confidence_profile not in profile_scores or c.overall_score > profile_scores[c.confidence_profile]:
            profile_scores[c.confidence_profile] = c.overall_score
    if profile_scores:
        best_prof = max(profile_scores, key=profile_scores.get)  # type: ignore[arg-type]
        recs.append(
            f"🎯 Best confidence profile: '{best_prof}' "
            f"(score {profile_scores[best_prof]:.3f})."
        )

    # Efficiency tip
    slow = [c for c in safe_configs if c.avg_rounds > 3.0]
    if slow:
        recs.append(
            f"🐢 {len(slow)} safe config(s) use >3 rounds on average. "
            f"Consider raising agent confidence or adjusting threshold."
        )

    return recs


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

def generate_html_report(report: OptimizationReport) -> str:
    """Generate an interactive HTML report with charts and tables."""
    configs_json = json.dumps([asdict(c) for c in report.configs])
    best_json = json.dumps(asdict(report.best_config)) if report.best_config else "null"
    recs_json = json.dumps(report.recommendations)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mBFT Consensus Parameter Optimizer</title>
<style>
:root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3;
  --accent: #58a6ff; --green: #3fb950; --red: #f85149; --yellow: #d29922;
  --dim: #8b949e; }}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg); color: var(--text); padding: 20px; }}
h1 {{ font-size: 1.8em; margin-bottom: 4px; }}
.subtitle {{ color: var(--dim); margin-bottom: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px; margin-bottom: 24px; }}
.stat {{ background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; }}
.stat .label {{ color: var(--dim); font-size: 0.85em; }}
.stat .value {{ font-size: 1.6em; font-weight: bold; margin-top: 4px; }}
.card {{ background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
.card h2 {{ margin-bottom: 12px; font-size: 1.2em; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ color: var(--dim); font-weight: 600; position: sticky; top: 0;
  background: var(--card); cursor: pointer; }}
th:hover {{ color: var(--accent); }}
tr:hover td {{ background: rgba(88,166,255,0.05); }}
.safe {{ color: var(--green); }} .unsafe {{ color: var(--red); }}
.rec {{ padding: 8px 12px; margin: 4px 0; background: rgba(88,166,255,0.08);
  border-left: 3px solid var(--accent); border-radius: 4px; font-size: 0.95em; }}
.best-badge {{ display: inline-block; background: var(--green); color: #000;
  font-size: 0.7em; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}
canvas {{ width: 100%; max-height: 300px; }}
.chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 768px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
.filters {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
.filters select, .filters input {{ background: var(--bg); color: var(--text);
  border: 1px solid var(--border); border-radius: 4px; padding: 4px 8px; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>🔬 mBFT Consensus Parameter Optimizer</h1>
<p class="subtitle">Autonomous parameter search • {report.total_trials} total trials • {report.elapsed_seconds:.1f}s</p>

<div class="grid">
  <div class="stat"><div class="label">Configurations</div><div class="value">{len(report.configs)}</div></div>
  <div class="stat"><div class="label">Total Trials</div><div class="value">{report.total_trials}</div></div>
  <div class="stat"><div class="label">Best Score</div>
    <div class="value">{report.best_config.overall_score:.3f if report.best_config else 'N/A'}</div></div>
  <div class="stat"><div class="label">Safe Configs</div>
    <div class="value safe">{sum(1 for c in report.configs if c.false_commit_rate == 0)}</div></div>
</div>

<div class="card">
  <h2>🎯 Recommendations</h2>
  <div id="recs"></div>
</div>

<div class="card chart-row">
  <div><h2>Commit Rate vs Threshold</h2><canvas id="chart1"></canvas></div>
  <div><h2>Safety vs Efficiency</h2><canvas id="chart2"></canvas></div>
</div>

<div class="card">
  <h2>📊 All Configurations</h2>
  <div class="filters">
    <select id="filterProfile"><option value="">All Profiles</option></select>
    <select id="filterAgents"><option value="">All Agent Counts</option></select>
    <select id="filterSafety"><option value="">All</option>
      <option value="safe">Safe Only</option><option value="unsafe">Unsafe Only</option></select>
    <input type="text" id="searchBox" placeholder="Search...">
  </div>
  <div style="max-height:400px;overflow:auto;">
    <table id="resultsTable">
      <thead><tr>
        <th data-sort="n_agents">Agents</th><th data-sort="n_byzantine">Byz</th>
        <th data-sort="threshold">Threshold</th><th data-sort="confidence_profile">Profile</th>
        <th data-sort="commit_rate">Commit%</th><th data-sort="avg_rounds">Avg Rnds</th>
        <th data-sort="false_commit_rate">False%</th><th data-sort="safety_score">Safety</th>
        <th data-sort="efficiency_score">Efficiency</th><th data-sort="overall_score">Score</th>
      </tr></thead>
      <tbody id="tableBody"></tbody>
    </table>
  </div>
</div>

<script>
const configs = {configs_json};
const best = {best_json};
const recs = {recs_json};

// Recommendations
const recsEl = document.getElementById('recs');
recs.forEach(r => {{ const d = document.createElement('div'); d.className='rec'; d.textContent=r; recsEl.appendChild(d); }});

// Filters
const profiles = [...new Set(configs.map(c=>c.confidence_profile))];
const agentCounts = [...new Set(configs.map(c=>c.n_agents))].sort((a,b)=>a-b);
const fpEl = document.getElementById('filterProfile');
profiles.forEach(p => {{ const o = document.createElement('option'); o.value=p; o.textContent=p; fpEl.appendChild(o); }});
const faEl = document.getElementById('filterAgents');
agentCounts.forEach(a => {{ const o = document.createElement('option'); o.value=a; o.textContent=a+' agents'; faEl.appendChild(o); }});

function renderTable() {{
  const prof = fpEl.value;
  const agents = faEl.value;
  const safety = document.getElementById('filterSafety').value;
  const search = document.getElementById('searchBox').value.toLowerCase();

  let filtered = configs.filter(c => {{
    if (prof && c.confidence_profile !== prof) return false;
    if (agents && c.n_agents !== +agents) return false;
    if (safety === 'safe' && c.false_commit_rate > 0) return false;
    if (safety === 'unsafe' && c.false_commit_rate === 0) return false;
    if (search && !JSON.stringify(c).toLowerCase().includes(search)) return false;
    return true;
  }});

  if (currentSort) {{
    filtered.sort((a,b) => {{
      let va = a[currentSort], vb = b[currentSort];
      if (typeof va === 'string') return sortDir * va.localeCompare(vb);
      return sortDir * (va - vb);
    }});
  }}

  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = '';
  filtered.forEach(c => {{
    const isBest = best && c.n_agents===best.n_agents && c.threshold===best.threshold
      && c.n_byzantine===best.n_byzantine && c.confidence_profile===best.confidence_profile;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${{c.n_agents}}</td><td>${{c.n_byzantine}}</td>
      <td>${{c.threshold.toFixed(2)}}</td><td>${{c.confidence_profile}}</td>
      <td>${{(c.commit_rate*100).toFixed(0)}}%</td><td>${{c.avg_rounds.toFixed(1)}}</td>
      <td class="${{c.false_commit_rate>0?'unsafe':'safe'}}">${{(c.false_commit_rate*100).toFixed(0)}}%</td>
      <td>${{c.safety_score.toFixed(2)}}</td><td>${{c.efficiency_score.toFixed(3)}}</td>
      <td>${{c.overall_score.toFixed(3)}} ${{isBest?'<span class="best-badge">BEST</span>':''}}</td>`;
    tbody.appendChild(tr);
  }});
}}

let currentSort = 'overall_score', sortDir = -1;
document.querySelectorAll('th[data-sort]').forEach(th => {{
  th.addEventListener('click', () => {{
    const s = th.dataset.sort;
    if (currentSort === s) sortDir *= -1; else {{ currentSort = s; sortDir = -1; }}
    renderTable();
  }});
}});

[fpEl, faEl, document.getElementById('filterSafety'), document.getElementById('searchBox')]
  .forEach(el => el.addEventListener('input', renderTable));

renderTable();

// Charts (simple canvas)
function drawChart(canvasId, data, xKey, yKey, colorFn) {{
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  const pad = {{ top:10, right:10, bottom:30, left:40 }};
  const pw = w-pad.left-pad.right, ph = h-pad.top-pad.bottom;

  const xs = data.map(d=>d[xKey]), ys = data.map(d=>d[yKey]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs)||1;
  const yMin = 0, yMax = Math.max(...ys, 0.01);

  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 0.5;
  for (let i=0;i<=4;i++) {{
    const y = pad.top + ph - (i/4)*ph;
    ctx.beginPath(); ctx.moveTo(pad.left,y); ctx.lineTo(w-pad.right,y); ctx.stroke();
    ctx.fillStyle='#8b949e'; ctx.font='11px sans-serif'; ctx.textAlign='right';
    ctx.fillText((yMin+(yMax-yMin)*(i/4)).toFixed(2), pad.left-4, y+4);
  }}

  data.forEach(d => {{
    const x = pad.left + ((d[xKey]-xMin)/(xMax-xMin||1))*pw;
    const y = pad.top + ph - ((d[yKey]-yMin)/(yMax-yMin||1))*ph;
    ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2);
    ctx.fillStyle = colorFn(d); ctx.fill();
  }});

  ctx.fillStyle='#8b949e'; ctx.font='11px sans-serif'; ctx.textAlign='center';
  ctx.fillText(xKey, w/2, h-4);
}}

setTimeout(() => {{
  drawChart('chart1', configs, 'threshold', 'commit_rate',
    d => d.false_commit_rate > 0 ? '#f85149' : '#3fb950');
  drawChart('chart2', configs, 'efficiency_score', 'safety_score',
    d => d.overall_score > (best?.overall_score||0)*0.8 ? '#58a6ff' : '#8b949e');
}}, 100);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

async def run_sweep(
    agent_counts: List[int],
    thresholds: List[float],
    byzantine_fractions: List[float],
    profiles: List[str],
    trials: int,
    auto: bool = False,
) -> OptimizationReport:
    """Run the full parameter sweep."""
    t0 = time.time()
    report = OptimizationReport()
    total = 0

    combos = list(itertools.product(agent_counts, thresholds, byzantine_fractions, profiles))
    print(f"Sweeping {len(combos)} configurations x {trials} trials = {len(combos)*trials} total runs...")

    for n_agents, threshold, byz_frac, profile in combos:
        n_byz = max(0, round(n_agents * byz_frac))
        if n_byz >= n_agents:
            continue  # skip all-Byzantine

        result = await evaluate_config(n_agents, n_byz, threshold, profile, trials)
        report.configs.append(result)
        total += trials

        # Progress indicator
        pct = total / (len(combos) * trials) * 100
        safe = "OK" if result.false_commit_rate == 0 else "!!"
        print(f"  [{pct:5.1f}%] {safe} n={n_agents} byz={n_byz} t={threshold:.2f} "
              f"prof={profile:15s} -> commit={result.commit_rate:.0%} "
              f"false={result.false_commit_rate:.0%} score={result.overall_score:.3f}")

    # Find best overall (safety-first: only consider zero false-commit configs)
    safe_configs = [c for c in report.configs if c.false_commit_rate == 0.0]
    if safe_configs:
        report.best_config = max(safe_configs, key=lambda c: c.overall_score)
    elif report.configs:
        report.best_config = max(report.configs, key=lambda c: c.overall_score)

    report.recommendations = generate_recommendations(report)
    report.elapsed_seconds = time.time() - t0
    report.total_trials = total

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Parameter Optimizer — autonomous search for optimal configs"
    )
    parser.add_argument(
        "--agents", type=int, nargs="+", default=[3, 5, 7, 9],
        help="Agent counts to test (default: 3 5 7 9)",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        help="Thresholds to test",
    )
    parser.add_argument(
        "--byzantine-fractions", type=float, nargs="+",
        default=[0.0, 0.1, 0.2, 0.33],
        help="Byzantine agent fractions to test",
    )
    parser.add_argument(
        "--profiles", nargs="+",
        default=list(CONFIDENCE_PROFILES.keys()),
        help="Confidence profiles to test",
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Trials per configuration (default: 30)",
    )
    parser.add_argument(
        "--html", type=str, default=None,
        help="Output interactive HTML report to file",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Output JSON results to file",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Print autonomous recommendations",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    report = asyncio.run(run_sweep(
        agent_counts=args.agents,
        thresholds=args.thresholds,
        byzantine_fractions=args.byzantine_fractions,
        profiles=args.profiles,
        trials=args.trials,
        auto=args.auto,
    ))

    # Summary
    print()
    print("=" * 60)
    print(f"SWEEP COMPLETE: {len(report.configs)} configs, "
          f"{report.total_trials} trials, {report.elapsed_seconds:.1f}s")
    print("=" * 60)

    if report.best_config:
        b = report.best_config
        print(f"\nBEST CONFIGURATION:")
        print(f"   Agents: {b.n_agents}  Byzantine: {b.n_byzantine}  "
              f"Threshold: {b.threshold:.2f}")
        print(f"   Profile: {b.confidence_profile}")
        print(f"   Commit rate: {b.commit_rate:.0%}  "
              f"False commit rate: {b.false_commit_rate:.0%}")
        print(f"   Avg rounds: {b.avg_rounds:.1f}  "
              f"Overall score: {b.overall_score:.3f}")

    if report.recommendations:
        print(f"\nRECOMMENDATIONS:")
        for rec in report.recommendations:
            print(f"   {rec}")

    # Export
    if args.html:
        html = generate_html_report(report)
        Path(args.html).write_text(html, encoding="utf-8")
        print(f"\nHTML report: {args.html}")

    if args.json:
        data = {
            "configs": [asdict(c) for c in report.configs],
            "best": asdict(report.best_config) if report.best_config else None,
            "recommendations": report.recommendations,
            "elapsed_seconds": report.elapsed_seconds,
            "total_trials": report.total_trials,
        }
        Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"JSON export: {args.json}")


if __name__ == "__main__":
    main()
