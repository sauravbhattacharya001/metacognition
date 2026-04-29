"""Trust Evolution Tracker — multi-scenario reputation analysis.

Runs a configurable battery of consensus scenarios, records per-agent
reputation after each round, detects trust anomalies (sudden drops,
stagnation, rehabilitation), and emits an interactive HTML report with
Chart.js line graphs and a summary table.

Usage::

    python -m src.trust_tracker                # default 6 scenarios
    python -m src.trust_tracker --scenarios 12 # more scenarios
    python -m src.trust_tracker --out report.html --json trust.json
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReputationSnapshot:
    scenario: int
    round_index: int
    reputations: Dict[str, float]


@dataclass
class TrustAnomaly:
    agent_id: str
    kind: str  # "sudden_drop" | "stagnation" | "rehabilitation" | "byzantine_evasion"
    scenario: int
    detail: str
    severity: str  # "low" | "medium" | "high"


@dataclass
class AgentProfile:
    agent_id: str
    final_reputation: float
    min_reputation: float
    max_reputation: float
    times_slashed: int
    times_leader: int
    anomalies: List[TrustAnomaly] = field(default_factory=list)
    trajectory: List[float] = field(default_factory=list)

    @property
    def trust_grade(self) -> str:
        r = self.final_reputation
        if r >= 0.9:
            return "A"
        if r >= 0.7:
            return "B"
        if r >= 0.5:
            return "C"
        if r >= 0.25:
            return "D"
        return "F"


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

def _random_agent_id(idx: int) -> str:
    return f"agent_{idx}"


def generate_scenario(scenario_idx: int, n_agents: int = 5) -> Tuple[List[MockAgent], float]:
    """Build a random but plausible agent swarm + threshold."""
    agents: List[MockAgent] = []
    majority_answer = random.choice(["42", "correct", "yes", "blue"])
    for i in range(n_agents):
        aid = _random_agent_id(i)
        is_byz = random.random() < 0.2
        if is_byz:
            ans = f"wrong_{i}"
            conf = round(random.uniform(0.6, 1.0), 2)
            agents.append(MockAgent(aid, answer=ans, confidence=conf, byzantine=True))
        else:
            conf = round(random.uniform(0.4, 0.95), 2)
            agents.append(MockAgent(aid, answer=majority_answer, confidence=conf))
    threshold = round(random.uniform(1.0, 2.0), 2)
    return agents, threshold


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TrustEvolutionTracker:
    def __init__(self, n_scenarios: int = 6, seed: Optional[int] = None) -> None:
        self.n_scenarios = n_scenarios
        if seed is not None:
            random.seed(seed)
        self.snapshots: List[ReputationSnapshot] = []
        self.profiles: Dict[str, AgentProfile] = {}
        self.anomalies: List[TrustAnomaly] = []
        self.scenario_results: List[dict] = []
        self._cumulative_rep: Dict[str, float] = {}

    async def run(self) -> None:
        for s in range(self.n_scenarios):
            agents, threshold = generate_scenario(s)
            engine = MBFTEngine(agents=agents, threshold=threshold, max_rounds=4)

            # Carry forward reputation from previous scenarios
            for a in agents:
                if a.id in self._cumulative_rep:
                    engine._reputation[a.id] = self._cumulative_rep[a.id]

            result = await engine.run(f"scenario_{s}")

            # Record per-round snapshots
            for rr in engine.history:
                # Reputation at this point
                snap_rep = dict(engine.reputation)  # final rep (approx)
                self.snapshots.append(ReputationSnapshot(s, rr.round_index, snap_rep))

            # Save cumulative reputation
            for aid, rep in engine.reputation.items():
                self._cumulative_rep[aid] = rep

            committed = result.committed if result else False
            self.scenario_results.append({
                "scenario": s,
                "committed": committed,
                "leader": result.leader_id if result else None,
                "rounds": len(engine.history),
                "threshold": threshold,
                "agents": [a.id for a in agents],
            })

            # Build / update profiles
            for a in agents:
                rep = engine.reputation[a.id]
                slashed_count = sum(1 for rr in engine.history if a.id in rr.slashed)
                leader_count = sum(1 for rr in engine.history if rr.leader_id == a.id)
                if a.id not in self.profiles:
                    self.profiles[a.id] = AgentProfile(
                        agent_id=a.id,
                        final_reputation=rep,
                        min_reputation=rep,
                        max_reputation=rep,
                        times_slashed=slashed_count,
                        times_leader=leader_count,
                        trajectory=[rep],
                    )
                else:
                    p = self.profiles[a.id]
                    p.final_reputation = rep
                    p.min_reputation = min(p.min_reputation, rep)
                    p.max_reputation = max(p.max_reputation, rep)
                    p.times_slashed += slashed_count
                    p.times_leader += leader_count
                    p.trajectory.append(rep)

        self._detect_anomalies()

    def _detect_anomalies(self) -> None:
        for aid, prof in self.profiles.items():
            traj = prof.trajectory
            for i in range(1, len(traj)):
                drop = traj[i - 1] - traj[i]
                if drop > 0.3:
                    a = TrustAnomaly(aid, "sudden_drop", i,
                                     f"Reputation dropped {drop:.2f} in scenario {i}", "high")
                    self.anomalies.append(a)
                    prof.anomalies.append(a)

            # Stagnation at low rep
            if len(traj) >= 3 and all(t < 0.3 for t in traj[-3:]):
                a = TrustAnomaly(aid, "stagnation", len(traj) - 1,
                                 "Stuck below 0.3 for 3+ scenarios", "medium")
                self.anomalies.append(a)
                prof.anomalies.append(a)

            # Rehabilitation: recovered from <0.3 to >0.7
            if len(traj) >= 2 and min(traj) < 0.3 and traj[-1] > 0.7:
                a = TrustAnomaly(aid, "rehabilitation", len(traj) - 1,
                                 f"Recovered from {min(traj):.2f} to {traj[-1]:.2f}", "low")
                self.anomalies.append(a)
                prof.anomalies.append(a)

    def to_json(self) -> dict:
        return {
            "scenarios": self.scenario_results,
            "profiles": {
                aid: {
                    "final_reputation": p.final_reputation,
                    "min_reputation": p.min_reputation,
                    "max_reputation": p.max_reputation,
                    "times_slashed": p.times_slashed,
                    "times_leader": p.times_leader,
                    "trust_grade": p.trust_grade,
                    "trajectory": p.trajectory,
                    "anomalies": [
                        {"kind": a.kind, "scenario": a.scenario,
                         "detail": a.detail, "severity": a.severity}
                        for a in p.anomalies
                    ],
                }
                for aid, p in sorted(self.profiles.items())
            },
            "anomalies_total": len(self.anomalies),
        }

    def to_html(self) -> str:
        data = self.to_json()
        profiles = data["profiles"]

        # Build chart datasets
        datasets_js = []
        colors = [
            "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
            "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
            "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
        ]
        for i, (aid, p) in enumerate(sorted(profiles.items())):
            c = colors[i % len(colors)]
            datasets_js.append(
                f'{{label:"{html.escape(aid)}",data:{json.dumps(p["trajectory"])},'
                f'borderColor:"{c}",backgroundColor:"{c}22",tension:0.3,fill:false}}'
            )
        datasets_str = ",".join(datasets_js)
        max_len = max((len(p["trajectory"]) for p in profiles.values()), default=1)
        labels_js = json.dumps([f"S{i}" for i in range(max_len)])

        # Profile table rows
        rows = ""
        for aid, p in sorted(profiles.items()):
            anomaly_badges = ""
            for a in p["anomalies"]:
                sev_color = {"high": "#e6194b", "medium": "#f58231", "low": "#3cb44b"}[a["severity"]]
                anomaly_badges += (
                    f'<span style="background:{sev_color};color:#fff;padding:2px 6px;'
                    f'border-radius:4px;font-size:0.75em;margin-right:4px">'
                    f'{html.escape(a["kind"])}</span>'
                )
            grade_color = {"A": "#3cb44b", "B": "#4363d8", "C": "#f58231", "D": "#e6194b", "F": "#911eb4"}
            gc = grade_color.get(p["trust_grade"], "#888")
            rows += f"""<tr>
                <td><b>{html.escape(aid)}</b></td>
                <td style="text-align:center"><span style="background:{gc};color:#fff;padding:2px 10px;border-radius:4px;font-weight:bold">{p['trust_grade']}</span></td>
                <td>{p['final_reputation']:.3f}</td>
                <td>{p['min_reputation']:.3f}</td>
                <td>{p['max_reputation']:.3f}</td>
                <td>{p['times_slashed']}</td>
                <td>{p['times_leader']}</td>
                <td>{anomaly_badges or '—'}</td>
            </tr>"""

        # Scenario summary
        scenario_rows = ""
        for s in data["scenarios"]:
            status = "✅ Committed" if s["committed"] else "❌ No consensus"
            scenario_rows += f"""<tr>
                <td>{s['scenario']}</td>
                <td>{status}</td>
                <td>{html.escape(s['leader'] or '—')}</td>
                <td>{s['rounds']}</td>
                <td>{s['threshold']}</td>
                <td>{', '.join(s['agents'])}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>mBFT Trust Evolution Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px}}
  h1{{color:#58a6ff;margin-bottom:8px}} h2{{color:#8b949e;margin:24px 0 12px}}
  .card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin:16px 0}}
  table{{width:100%;border-collapse:collapse;font-size:0.9em}}
  th{{background:#21262d;color:#8b949e;text-align:left;padding:8px 12px;border-bottom:1px solid #30363d}}
  td{{padding:8px 12px;border-bottom:1px solid #21262d}}
  tr:hover{{background:#1c2128}}
  .stat{{display:inline-block;background:#21262d;border-radius:8px;padding:12px 20px;margin:4px;text-align:center}}
  .stat .val{{font-size:1.5em;font-weight:bold;color:#58a6ff}}
  .stat .lbl{{font-size:0.8em;color:#8b949e}}
  canvas{{max-height:400px}}
</style></head><body>
<h1>🔍 mBFT Trust Evolution Report</h1>
<p style="color:#8b949e">Reputation trajectories across {len(data['scenarios'])} consensus scenarios</p>

<div style="display:flex;flex-wrap:wrap;gap:8px;margin:16px 0">
  <div class="stat"><div class="val">{len(profiles)}</div><div class="lbl">Agents Tracked</div></div>
  <div class="stat"><div class="val">{len(data['scenarios'])}</div><div class="lbl">Scenarios</div></div>
  <div class="stat"><div class="val">{sum(1 for s in data['scenarios'] if s['committed'])}</div><div class="lbl">Committed</div></div>
  <div class="stat"><div class="val">{data['anomalies_total']}</div><div class="lbl">Trust Anomalies</div></div>
</div>

<div class="card">
<h2>📈 Reputation Trajectories</h2>
<canvas id="trajChart"></canvas>
</div>

<div class="card">
<h2>🏆 Agent Trust Profiles</h2>
<table><thead><tr>
  <th>Agent</th><th>Grade</th><th>Final Rep</th><th>Min</th><th>Max</th><th>Slashed</th><th>Led</th><th>Anomalies</th>
</tr></thead><tbody>{rows}</tbody></table>
</div>

<div class="card">
<h2>📋 Scenario Summary</h2>
<table><thead><tr>
  <th>#</th><th>Outcome</th><th>Leader</th><th>Rounds</th><th>Threshold</th><th>Agents</th>
</tr></thead><tbody>{scenario_rows}</tbody></table>
</div>

<script>
new Chart(document.getElementById('trajChart'),{{
  type:'line',
  data:{{labels:{labels_js},datasets:[{datasets_str}]}},
  options:{{
    responsive:true,
    plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},
      y:{{min:0,max:1.05,ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}},title:{{display:true,text:'Reputation',color:'#8b949e'}}}}
    }}
  }}
}});
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main() -> None:
    parser = argparse.ArgumentParser(description="mBFT Trust Evolution Tracker")
    parser.add_argument("--scenarios", type=int, default=6, help="Number of scenarios (default: 6)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="trust_evolution.html", help="HTML output path")
    parser.add_argument("--json", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    tracker = TrustEvolutionTracker(n_scenarios=args.scenarios, seed=args.seed)
    await tracker.run()

    data = tracker.to_json()

    # Terminal summary
    print("=" * 60)
    print("mBFT Trust Evolution Report")
    print("=" * 60)
    print(f"Scenarios: {len(data['scenarios'])}  |  "
          f"Committed: {sum(1 for s in data['scenarios'] if s['committed'])}  |  "
          f"Anomalies: {data['anomalies_total']}")
    print()
    print(f"{'Agent':<12} {'Grade':>5} {'Final':>7} {'Min':>7} {'Max':>7} {'Slash':>5} {'Led':>4}")
    print("-" * 52)
    for aid, p in sorted(data["profiles"].items()):
        print(f"{aid:<12} {p['trust_grade']:>5} {p['final_reputation']:>7.3f} "
              f"{p['min_reputation']:>7.3f} {p['max_reputation']:>7.3f} "
              f"{p['times_slashed']:>5} {p['times_leader']:>4}")

    if tracker.anomalies:
        print(f"\nTrust Anomalies ({len(tracker.anomalies)}):")
        for a in tracker.anomalies:
            print(f"  [{a.severity.upper()}] {a.agent_id}: {a.detail}")

    # Write outputs
    Path(args.out).write_text(tracker.to_html(), encoding="utf-8")
    print(f"\nHTML report: {args.out}")

    if args.json:
        Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"JSON export: {args.json}")


if __name__ == "__main__":
    asyncio.run(_main())
