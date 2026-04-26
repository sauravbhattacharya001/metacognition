"""Consensus Memory & Grudge System.

Agents build persistent memory of past interactions — tracking betrayals,
agreements, and leadership quality. Over multiple consensus runs, agents
develop grudges (persistent distrust) or alliances (persistent trust).

Features:
- Interaction memory with decay
- Grudge & alliance detection
- Forgiveness dynamics (grudges fade unless reinforced)
- Relationship graph with stability analysis
- Interactive HTML report with relationship heatmap and timeline

Usage:
    python -m src.grudge [--agents N] [--rounds R] [--scenarios S]
                         [--grudge-threshold T] [--forgiveness-rate F]
                         [--output report.html] [--json results.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core.protocol import MBFTEngine
from src.core.state import RoundResult


# ---------------------------------------------------------------------------
# Memory & relationship models
# ---------------------------------------------------------------------------

@dataclass
class Interaction:
    """A single remembered interaction between two agents."""
    scenario: int
    round_idx: int
    agent_a: str
    agent_b: str
    kind: str          # "agreement", "betrayal", "support", "rejection"
    intensity: float   # 0-1
    timestamp: int     # global step counter


@dataclass
class Relationship:
    """Accumulated sentiment between two agents."""
    trust: float = 0.0
    grudge: float = 0.0
    interactions: int = 0
    last_betrayal: Optional[int] = None
    last_support: Optional[int] = None
    alliance_streak: int = 0
    grudge_streak: int = 0

    @property
    def sentiment(self) -> float:
        return self.trust - self.grudge

    @property
    def is_grudge(self) -> bool:
        return self.grudge > 0.5 and self.grudge > self.trust

    @property
    def is_alliance(self) -> bool:
        return self.trust > 0.5 and self.trust > self.grudge


@dataclass
class AgentMemory:
    """An agent's full memory of interactions."""
    agent_id: str
    relationships: Dict[str, Relationship] = field(default_factory=dict)
    history: List[Interaction] = field(default_factory=list)
    times_led: int = 0
    times_slashed: int = 0
    betrayals_committed: int = 0
    betrayals_received: int = 0

    def get_relationship(self, other_id: str) -> Relationship:
        if other_id not in self.relationships:
            self.relationships[other_id] = Relationship()
        return self.relationships[other_id]


# ---------------------------------------------------------------------------
# Grudge engine
# ---------------------------------------------------------------------------

class GrudgeEngine:
    """Runs multiple mBFT scenarios and tracks relationship evolution."""

    def __init__(
        self,
        n_agents: int = 6,
        n_rounds: int = 4,
        n_scenarios: int = 20,
        threshold: float = 2.0,
        grudge_threshold: float = 0.5,
        forgiveness_rate: float = 0.05,
        betrayal_weight: float = 0.3,
        support_weight: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.n_scenarios = n_scenarios
        self.threshold = threshold
        self.grudge_threshold = grudge_threshold
        self.forgiveness_rate = forgiveness_rate
        self.betrayal_weight = betrayal_weight
        self.support_weight = support_weight
        self.rng = random.Random(seed)
        self.memories: Dict[str, AgentMemory] = {}
        self.all_interactions: List[Interaction] = []
        self.snapshots: List[Dict] = []  # per-scenario relationship snapshots
        self.global_step = 0

    def _agent_ids(self) -> List[str]:
        return [f"agent_{i}" for i in range(self.n_agents)]

    async def run(self) -> Dict:
        """Execute all scenarios and return analysis."""
        from src.agents.metacognitive import MockAgent

        agent_ids = self._agent_ids()
        for aid in agent_ids:
            self.memories[aid] = AgentMemory(agent_id=aid)

        # Assign personality biases (some agents more prone to betrayal)
        personalities: Dict[str, float] = {}
        for aid in agent_ids:
            personalities[aid] = self.rng.uniform(0.1, 0.9)

        for scenario in range(self.n_scenarios):
            # Create mock agents with personality-driven behavior
            answers = [f"solution_{self.rng.randint(0, 3)}" for _ in agent_ids]
            agents = []
            for i, aid in enumerate(agent_ids):
                # More "trusting" agents accept broader answer sets
                trust_breadth = max(1, int(personalities[aid] * 4))
                accept = set(answers[:trust_breadth])
                # Low-personality agents may go Byzantine
                byz = self.rng.random() > personalities[aid] + 0.3
                agents.append(MockAgent(
                    agent_id=aid,
                    answer=answers[i],
                    confidence=self.rng.uniform(0.3, 1.0),
                    byzantine=byz,
                    accept_set=accept,
                ))

            engine = MBFTEngine(
                agents=agents,
                threshold=self.threshold,
                max_rounds=self.n_rounds,
            )

            task = f"scenario_{scenario}_task"
            result = await engine.run(task)

            # Process round results into interactions
            for rr in engine.history:
                self._process_round(scenario, rr, personalities)
                self.global_step += 1

            # Apply forgiveness decay
            self._apply_forgiveness()

            # Take snapshot
            self.snapshots.append(self._snapshot(scenario))

        return self._analyze()

    def _process_round(
        self, scenario: int, rr: RoundResult, personalities: Dict[str, float]
    ) -> None:
        leader = rr.leader_id
        self.memories[leader].times_led += 1

        if rr.slashed:
            for sid in rr.slashed:
                self.memories[sid].times_slashed += 1

        for vote in rr.votes:
            voter = vote.voter_id
            if vote.is_rejection:
                # Betrayal from leader's perspective
                kind = "betrayal"
                intensity = abs(vote.weight)
                self.memories[leader].betrayals_received += 1
                self.memories[voter].betrayals_committed += 1

                # Update relationships
                rel_leader = self.memories[leader].get_relationship(voter)
                rel_leader.grudge = min(1.0, rel_leader.grudge + intensity * self.betrayal_weight)
                rel_leader.last_betrayal = self.global_step
                rel_leader.alliance_streak = 0
                rel_leader.grudge_streak += 1
                rel_leader.interactions += 1

                rel_voter = self.memories[voter].get_relationship(leader)
                rel_voter.trust = max(0.0, rel_voter.trust - intensity * 0.1)
                rel_voter.interactions += 1
            else:
                kind = "support"
                intensity = vote.weight

                rel_leader = self.memories[leader].get_relationship(voter)
                rel_leader.trust = min(1.0, rel_leader.trust + intensity * self.support_weight)
                rel_leader.last_support = self.global_step
                rel_leader.grudge_streak = 0
                rel_leader.alliance_streak += 1
                rel_leader.interactions += 1

                rel_voter = self.memories[voter].get_relationship(leader)
                rel_voter.trust = min(1.0, rel_voter.trust + intensity * 0.05)
                rel_voter.interactions += 1

            interaction = Interaction(
                scenario=scenario,
                round_idx=rr.round_index,
                agent_a=leader,
                agent_b=voter,
                kind=kind,
                intensity=intensity,
                timestamp=self.global_step,
            )
            self.all_interactions.append(interaction)
            self.memories[leader].history.append(interaction)
            self.memories[voter].history.append(interaction)

    def _apply_forgiveness(self) -> None:
        """Grudges decay unless reinforced recently."""
        for mem in self.memories.values():
            for rel in mem.relationships.values():
                if rel.grudge > 0:
                    rel.grudge = max(0.0, rel.grudge - self.forgiveness_rate)
                # Trust also decays slightly without reinforcement
                if rel.trust > 0:
                    rel.trust = max(0.0, rel.trust - self.forgiveness_rate * 0.3)

    def _snapshot(self, scenario: int) -> Dict:
        """Capture current relationship state."""
        pairs = {}
        for aid, mem in self.memories.items():
            for oid, rel in mem.relationships.items():
                key = f"{aid}->{oid}"
                pairs[key] = {
                    "trust": round(rel.trust, 3),
                    "grudge": round(rel.grudge, 3),
                    "sentiment": round(rel.sentiment, 3),
                    "is_grudge": rel.is_grudge,
                    "is_alliance": rel.is_alliance,
                }
        return {"scenario": scenario, "relationships": pairs}

    def _analyze(self) -> Dict:
        """Produce full analysis."""
        agent_ids = self._agent_ids()

        # Relationship matrix
        matrix = {}
        grudges = []
        alliances = []
        for aid in agent_ids:
            matrix[aid] = {}
            for oid in agent_ids:
                if aid == oid:
                    matrix[aid][oid] = 0.0
                    continue
                rel = self.memories[aid].get_relationship(oid)
                matrix[aid][oid] = round(rel.sentiment, 3)
                if rel.is_grudge:
                    grudges.append({
                        "from": aid, "to": oid,
                        "grudge_level": round(rel.grudge, 3),
                        "streak": rel.grudge_streak,
                    })
                if rel.is_alliance:
                    alliances.append({
                        "from": aid, "to": oid,
                        "trust_level": round(rel.trust, 3),
                        "streak": rel.alliance_streak,
                    })

        # Agent profiles
        profiles = {}
        for aid in agent_ids:
            mem = self.memories[aid]
            total_trust = sum(r.trust for r in mem.relationships.values())
            total_grudge = sum(r.grudge for r in mem.relationships.values())
            n_allies = sum(1 for r in mem.relationships.values() if r.is_alliance)
            n_grudges = sum(1 for r in mem.relationships.values() if r.is_grudge)
            profiles[aid] = {
                "times_led": mem.times_led,
                "times_slashed": mem.times_slashed,
                "betrayals_committed": mem.betrayals_committed,
                "betrayals_received": mem.betrayals_received,
                "total_trust": round(total_trust, 3),
                "total_grudge": round(total_grudge, 3),
                "n_allies": n_allies,
                "n_grudges": n_grudges,
                "sociability": round(total_trust - total_grudge, 3),
            }

        # Stability analysis
        final_snap = self.snapshots[-1] if self.snapshots else {}
        n_grudge_pairs = len(grudges)
        n_alliance_pairs = len(alliances)
        total_pairs = self.n_agents * (self.n_agents - 1)
        stability = 1.0 - (n_grudge_pairs / max(total_pairs, 1))

        # Forgiveness events: grudges that peaked then declined
        forgiveness_events = []
        for aid in agent_ids:
            for oid in agent_ids:
                if aid == oid:
                    continue
                rel = self.memories[aid].get_relationship(oid)
                if rel.last_betrayal is not None and not rel.is_grudge and rel.interactions > 3:
                    forgiveness_events.append({
                        "forgiver": aid, "forgiven": oid,
                        "remaining_grudge": round(rel.grudge, 3),
                    })

        # Recommendations
        recommendations = []
        if n_grudge_pairs > total_pairs * 0.3:
            recommendations.append("⚠️ High grudge density — consider increasing forgiveness rate or reducing Byzantine behavior")
        if stability < 0.5:
            recommendations.append("🔴 Low network stability — consensus reliability is at risk from entrenched distrust")
        if any(p["sociability"] < -1.0 for p in profiles.values()):
            recommendations.append("👤 Some agents have deeply negative sociability — candidates for isolation or recalibration")
        most_betrayed = max(profiles.items(), key=lambda x: x[1]["betrayals_received"])
        if most_betrayed[1]["betrayals_received"] > self.n_scenarios * 0.5:
            recommendations.append(f"🎯 {most_betrayed[0]} is a frequent target — may need protective measures")
        if not recommendations:
            recommendations.append("✅ Network relationships are healthy — no intervention needed")

        return {
            "config": {
                "n_agents": self.n_agents,
                "n_scenarios": self.n_scenarios,
                "n_rounds": self.n_rounds,
                "threshold": self.threshold,
                "grudge_threshold": self.grudge_threshold,
                "forgiveness_rate": self.forgiveness_rate,
            },
            "matrix": matrix,
            "profiles": profiles,
            "grudges": grudges,
            "alliances": alliances,
            "forgiveness_events": forgiveness_events,
            "stability": round(stability, 3),
            "total_interactions": len(self.all_interactions),
            "recommendations": recommendations,
            "timeline": self.snapshots,
        }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(results: Dict) -> str:
    agents = list(results["profiles"].keys())
    matrix = results["matrix"]

    # Build heatmap data
    heatmap_data = []
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            heatmap_data.append({"x": j, "y": i, "v": matrix[a][b]})

    profiles_json = json.dumps(results["profiles"], indent=2)
    grudges_json = json.dumps(results["grudges"], indent=2)
    alliances_json = json.dumps(results["alliances"], indent=2)
    recommendations_html = "".join(f"<li>{r}</li>" for r in results["recommendations"])

    # Timeline data for chart
    timeline_data = []
    for snap in results["timeline"]:
        sc = snap["scenario"]
        g_count = sum(1 for r in snap["relationships"].values() if r["is_grudge"])
        a_count = sum(1 for r in snap["relationships"].values() if r["is_alliance"])
        avg_sent = sum(r["sentiment"] for r in snap["relationships"].values()) / max(len(snap["relationships"]), 1)
        timeline_data.append({"s": sc, "g": g_count, "a": a_count, "avg": round(avg_sent, 3)})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mBFT Consensus Memory &amp; Grudge Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e0e0e0;padding:20px}}
h1{{text-align:center;color:#7c4dff;margin-bottom:8px;font-size:1.8em}}
.subtitle{{text-align:center;color:#888;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:1200px;margin:0 auto}}
.card{{background:#14141f;border:1px solid #2a2a3a;border-radius:12px;padding:16px}}
.card h2{{color:#bb86fc;font-size:1.1em;margin-bottom:12px}}
.full{{grid-column:1/-1}}
.stat{{display:inline-block;background:#1a1a2e;border-radius:8px;padding:8px 14px;margin:4px;font-size:0.9em}}
.stat .val{{color:#7c4dff;font-weight:bold;font-size:1.3em}}
.stat .lbl{{color:#888;font-size:0.8em}}
canvas{{width:100%;border-radius:8px}}
table{{width:100%;border-collapse:collapse;font-size:0.85em}}
th,td{{padding:6px 8px;text-align:center;border:1px solid #2a2a3a}}
th{{background:#1a1a2e;color:#bb86fc}}
.grudge-tag{{background:#ff1744;color:#fff;padding:2px 6px;border-radius:4px;font-size:0.75em}}
.alliance-tag{{background:#00e676;color:#000;padding:2px 6px;border-radius:4px;font-size:0.75em}}
ul{{list-style:none;padding:0}}
ul li{{padding:6px 0;border-bottom:1px solid #1a1a2e}}
ul li:last-child{{border:none}}
.rec{{background:#1a1a2e;padding:8px 12px;border-radius:6px;margin:4px 0}}
</style>
</head>
<body>
<h1>🧠 Consensus Memory &amp; Grudge System</h1>
<p class="subtitle">mBFT relationship dynamics across {results['config']['n_scenarios']} scenarios with {results['config']['n_agents']} agents</p>

<div class="grid">
  <div class="card full" style="text-align:center">
    <div class="stat"><div class="val">{results['total_interactions']}</div><div class="lbl">Interactions</div></div>
    <div class="stat"><div class="val">{len(results['grudges'])}</div><div class="lbl">Active Grudges</div></div>
    <div class="stat"><div class="val">{len(results['alliances'])}</div><div class="lbl">Alliances</div></div>
    <div class="stat"><div class="val">{len(results['forgiveness_events'])}</div><div class="lbl">Forgiveness Events</div></div>
    <div class="stat"><div class="val">{results['stability']}</div><div class="lbl">Network Stability</div></div>
  </div>

  <div class="card full">
    <h2>🗺️ Relationship Heatmap</h2>
    <canvas id="heatmap" height="300"></canvas>
  </div>

  <div class="card full">
    <h2>📈 Relationship Timeline</h2>
    <canvas id="timeline" height="200"></canvas>
  </div>

  <div class="card">
    <h2>👤 Agent Profiles</h2>
    <table>
      <tr><th>Agent</th><th>Led</th><th>Slashed</th><th>Betrayals↑</th><th>Betrayals↓</th><th>Allies</th><th>Grudges</th><th>Score</th></tr>
      {"".join(f'<tr><td>{aid}</td><td>{p["times_led"]}</td><td>{p["times_slashed"]}</td><td>{p["betrayals_committed"]}</td><td>{p["betrayals_received"]}</td><td>{p["n_allies"]}</td><td>{p["n_grudges"]}</td><td style="color:{"#00e676" if p["sociability"]>=0 else "#ff1744"}">{p["sociability"]}</td></tr>' for aid, p in results["profiles"].items())}
    </table>
  </div>

  <div class="card">
    <h2>⚔️ Active Grudges &amp; 🤝 Alliances</h2>
    {"".join(f'<div class="rec"><span class="grudge-tag">GRUDGE</span> {g["from"]} → {g["to"]} (level: {g["grudge_level"]}, streak: {g["streak"]})</div>' for g in results["grudges"])}
    {"".join(f'<div class="rec"><span class="alliance-tag">ALLIANCE</span> {a["from"]} → {a["to"]} (trust: {a["trust_level"]}, streak: {a["streak"]})</div>' for a in results["alliances"])}
    {('<div class="rec" style="color:#888">No active grudges or alliances</div>' if not results["grudges"] and not results["alliances"] else '')}
  </div>

  <div class="card full">
    <h2>💡 Recommendations</h2>
    {"".join(f'<div class="rec">{r}</div>' for r in results["recommendations"])}
  </div>
</div>

<script>
const agents = {json.dumps(agents)};
const heatData = {json.dumps(heatmap_data)};
const timelineData = {json.dumps(timeline_data)};

// Heatmap
(function() {{
  const c = document.getElementById('heatmap');
  const ctx = c.getContext('2d');
  c.width = c.parentElement.clientWidth - 32;
  c.height = 300;
  const n = agents.length;
  const pad = 80;
  const cw = (c.width - pad) / n;
  const ch = (c.height - pad) / n;

  // Labels
  ctx.fillStyle = '#888';
  ctx.font = '11px system-ui';
  agents.forEach((a, i) => {{
    ctx.save();
    ctx.translate(pad + i * cw + cw/2, pad - 5);
    ctx.rotate(-0.5);
    ctx.fillText(a, 0, 0);
    ctx.restore();
    ctx.fillText(a, 2, pad + i * ch + ch/2 + 4);
  }});

  heatData.forEach(d => {{
    const v = d.v;
    const r = v < 0 ? Math.min(255, Math.floor(-v * 400)) : 0;
    const g = v > 0 ? Math.min(255, Math.floor(v * 400)) : 0;
    ctx.fillStyle = `rgb(${{r}}, ${{g}}, ${{Math.floor(80 + Math.abs(v) * 100)}})`;
    ctx.fillRect(pad + d.x * cw, pad + d.y * ch, cw - 1, ch - 1);

    if (d.x !== d.y) {{
      ctx.fillStyle = '#fff';
      ctx.font = '10px system-ui';
      ctx.fillText(d.v.toFixed(2), pad + d.x * cw + 4, pad + d.y * ch + ch/2 + 3);
    }}
  }});
}})();

// Timeline chart
(function() {{
  const c = document.getElementById('timeline');
  const ctx = c.getContext('2d');
  c.width = c.parentElement.clientWidth - 32;
  c.height = 200;
  const w = c.width, h = c.height;
  const pad = 40;
  const n = timelineData.length;
  if (n < 2) return;
  const dx = (w - pad * 2) / (n - 1);

  const maxG = Math.max(1, ...timelineData.map(d => d.g));
  const maxA = Math.max(1, ...timelineData.map(d => d.a));
  const maxV = Math.max(maxG, maxA);

  // Grid
  ctx.strokeStyle = '#1a1a2e';
  for (let i = 0; i <= 4; i++) {{
    const y = pad + (h - pad * 2) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
  }}

  function drawLine(data, key, color) {{
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((d, i) => {{
      const x = pad + i * dx;
      const y = h - pad - (d[key] / maxV) * (h - pad * 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }}

  drawLine(timelineData, 'g', '#ff1744');
  drawLine(timelineData, 'a', '#00e676');

  // Legend
  ctx.font = '11px system-ui';
  ctx.fillStyle = '#ff1744'; ctx.fillText('● Grudges', pad, 16);
  ctx.fillStyle = '#00e676'; ctx.fillText('● Alliances', pad + 90, 16);
  ctx.fillStyle = '#888'; ctx.fillText('Scenario →', w/2 - 30, h - 5);
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> None:
    engine = GrudgeEngine(
        n_agents=args.agents,
        n_rounds=args.rounds,
        n_scenarios=args.scenarios,
        grudge_threshold=args.grudge_threshold,
        forgiveness_rate=args.forgiveness_rate,
        seed=args.seed,
    )
    results = await engine.run()

    # Console summary
    print(f"\n{'='*60}")
    print("  CONSENSUS MEMORY & GRUDGE REPORT")
    print(f"{'='*60}")
    print(f"  Agents: {args.agents} | Scenarios: {args.scenarios} | Rounds/scenario: {args.rounds}")
    print(f"  Total interactions: {results['total_interactions']}")
    print(f"  Network stability:  {results['stability']}")
    print(f"  Active grudges:     {len(results['grudges'])}")
    print(f"  Active alliances:   {len(results['alliances'])}")
    print(f"  Forgiveness events: {len(results['forgiveness_events'])}")
    print()

    print("  AGENT PROFILES")
    print(f"  {'Agent':<12} {'Led':>4} {'Slash':>6} {'BetrayOut':>9} {'BetrayIn':>9} {'Score':>7}")
    for aid, p in results["profiles"].items():
        print(f"  {aid:<12} {p['times_led']:>4} {p['times_slashed']:>6} "
              f"{p['betrayals_committed']:>9} {p['betrayals_received']:>9} "
              f"{p['sociability']:>7.3f}")

    if results["grudges"]:
        print("\n  ACTIVE GRUDGES")
        for g in results["grudges"]:
            print(f"  [GRUDGE] {g['from']} -> {g['to']}  (level: {g['grudge_level']}, streak: {g['streak']})")

    if results["alliances"]:
        print("\n  ALLIANCES")
        for a in results["alliances"]:
            print(f"  [ALLY]   {a['from']} -> {a['to']}  (trust: {a['trust_level']}, streak: {a['streak']})")

    print("\n  RECOMMENDATIONS")
    for r in results["recommendations"]:
        # Strip emoji for Windows console compatibility
        safe = r.encode('ascii', 'replace').decode('ascii')
        print(f"  {safe}")

    # HTML report
    if args.output:
        html = generate_html_report(results)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n  HTML report: {args.output}")

    # JSON export
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  JSON export: {args.json}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Memory & Grudge System"
    )
    parser.add_argument("--agents", type=int, default=6, help="Number of agents (default: 6)")
    parser.add_argument("--rounds", type=int, default=4, help="Max rounds per scenario (default: 4)")
    parser.add_argument("--scenarios", type=int, default=20, help="Number of scenarios (default: 20)")
    parser.add_argument("--grudge-threshold", type=float, default=0.5, help="Grudge detection threshold")
    parser.add_argument("--forgiveness-rate", type=float, default=0.05, help="Grudge decay rate per scenario")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="HTML report output path")
    parser.add_argument("--json", type=str, default=None, help="JSON export path")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
