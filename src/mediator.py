"""Autonomous Consensus Mediator.

Analyzes disagreement patterns across failed mBFT rounds and proposes
mediation strategies to help the swarm reach consensus.  Supports:

- Disagreement root-cause analysis (polarization, low confidence, Byzantine)
- Faction detection (clusters of agents that tend to agree/disagree)
- Mediation strategy recommendations (threshold adjustment, agent weighting,
  topic decomposition, coalition building)
- Interactive HTML report with faction graph, vote heatmap, and strategy cards

Usage (after a failed consensus run)::

    from src.mediator import ConsensusMediator

    mediator = ConsensusMediator(engine)
    report = mediator.analyze()
    print(report.summary)
    mediator.export_html("mediation_report.html")

Or from the CLI::

    python -m src.mediator                  # default demo
    python -m src.mediator --export report.html
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.core.protocol import MBFTEngine
from src.core.state import RoundResult, Vote


# ── Data models ────────────────────────────────────────────────────────


@dataclass
class Faction:
    """A cluster of agents that voted similarly."""
    name: str
    members: List[str]
    avg_weight: float
    cohesion: float  # 0-1, how tightly they agree


@dataclass
class DisagreementPattern:
    kind: str  # polarization | low_confidence | byzantine | fragmentation
    severity: float  # 0-1
    description: str
    involved_agents: List[str]


@dataclass
class MediationStrategy:
    name: str
    description: str
    priority: int  # 1=highest
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MediationReport:
    patterns: List[DisagreementPattern]
    factions: List[Faction]
    strategies: List[MediationStrategy]
    vote_matrix: Dict[str, List[float]]
    reputation_trajectory: Dict[str, List[float]]
    rounds_analyzed: int
    consensus_reached: bool

    @property
    def summary(self) -> str:
        lines = [
            f"=== Consensus Mediation Report ({self.rounds_analyzed} rounds) ===",
            f"Consensus reached: {self.consensus_reached}",
            "",
            "--- Disagreement Patterns ---",
        ]
        for p in self.patterns:
            lines.append(f"  [{p.kind}] severity={p.severity:.2f}: {p.description}")
        lines.append("")
        lines.append("--- Factions ---")
        for f in self.factions:
            lines.append(
                f"  {f.name}: {f.members} (avg_weight={f.avg_weight:.2f}, cohesion={f.cohesion:.2f})"
            )
        lines.append("")
        lines.append("--- Recommended Strategies ---")
        for s in sorted(self.strategies, key=lambda x: x.priority):
            lines.append(f"  P{s.priority}: {s.name} — {s.description}")
            if s.parameters:
                for k, v in s.parameters.items():
                    lines.append(f"        {k}: {v}")
        return "\n".join(lines)


# ── Mediator ───────────────────────────────────────────────────────────


class ConsensusMediator:
    """Analyze an MBFTEngine's history and produce mediation guidance."""

    def __init__(self, engine: MBFTEngine) -> None:
        self.engine = engine
        self._report: Optional[MediationReport] = None

    def analyze(self) -> MediationReport:
        history = self.engine.history
        if not history:
            self._report = MediationReport(
                patterns=[], factions=[], strategies=[],
                vote_matrix={}, reputation_trajectory={},
                rounds_analyzed=0, consensus_reached=False,
            )
            return self._report

        vote_matrix = self._build_vote_matrix(history)
        rep_traj = self._build_reputation_trajectory(history)
        patterns = self._detect_patterns(history, vote_matrix)
        factions = self._detect_factions(vote_matrix)
        strategies = self._recommend_strategies(patterns, factions, history)

        self._report = MediationReport(
            patterns=patterns,
            factions=factions,
            strategies=strategies,
            vote_matrix=vote_matrix,
            reputation_trajectory=rep_traj,
            rounds_analyzed=len(history),
            consensus_reached=any(r.committed for r in history),
        )
        return self._report

    # ── Pattern detection ──────────────────────────────────────────────

    def _build_vote_matrix(
        self, history: List[RoundResult]
    ) -> Dict[str, List[float]]:
        """Agent -> list of vote weights across rounds."""
        matrix: Dict[str, List[float]] = defaultdict(list)
        for rr in history:
            voters_this_round = set()
            # Leader gets implicit +confidence
            matrix[rr.leader_id].append(1.0)
            voters_this_round.add(rr.leader_id)
            for v in rr.votes:
                matrix[v.voter_id].append(v.weight)
                voters_this_round.add(v.voter_id)
            # Agents not in this round get 0
            for aid in list(matrix.keys()):
                if aid not in voters_this_round:
                    matrix[aid].append(0.0)
        return dict(matrix)

    def _build_reputation_trajectory(
        self, history: List[RoundResult]
    ) -> Dict[str, List[float]]:
        """Reconstruct reputation changes over rounds."""
        rep = {a.id: 1.0 for a in self.engine.agents}
        traj: Dict[str, List[float]] = {a.id: [1.0] for a in self.engine.agents}
        for rr in history:
            for aid in rr.slashed:
                rep[aid] *= self.engine.slash_factor
            for aid in rep:
                traj[aid].append(rep[aid])
        return traj

    def _detect_patterns(
        self,
        history: List[RoundResult],
        vote_matrix: Dict[str, List[float]],
    ) -> List[DisagreementPattern]:
        patterns: List[DisagreementPattern] = []

        # 1. Polarization: are there agents consistently on opposite sides?
        agents = list(vote_matrix.keys())
        if len(agents) >= 2:
            pos_agents = [a for a in agents if _mean(vote_matrix[a]) > 0.3]
            neg_agents = [a for a in agents if _mean(vote_matrix[a]) < -0.1]
            if pos_agents and neg_agents:
                severity = min(1.0, len(neg_agents) / len(agents) * 2)
                patterns.append(DisagreementPattern(
                    kind="polarization",
                    severity=severity,
                    description=(
                        f"Swarm is polarized: {len(pos_agents)} supporters vs "
                        f"{len(neg_agents)} dissenters across {len(history)} rounds"
                    ),
                    involved_agents=neg_agents,
                ))

        # 2. Low confidence: leader proposals lack conviction
        leader_confs = []
        for rr in history:
            for a in self.engine.agents:
                if a.id == rr.leader_id:
                    # Approximate from aggregate minus votes
                    break
        avg_agg = _mean([rr.aggregate_weight for rr in history])
        if avg_agg < self.engine.threshold * 0.6:
            patterns.append(DisagreementPattern(
                kind="low_confidence",
                severity=min(1.0, 1.0 - avg_agg / self.engine.threshold),
                description=(
                    f"Average aggregate weight ({avg_agg:.2f}) is well below "
                    f"threshold ({self.engine.threshold:.2f})"
                ),
                involved_agents=agents,
            ))

        # 3. Byzantine suspicion: agents with reputation drops
        slashed_counts: Dict[str, int] = defaultdict(int)
        for rr in history:
            for aid in rr.slashed:
                slashed_counts[aid] += 1
        repeat_slashed = {a: c for a, c in slashed_counts.items() if c >= 2}
        if repeat_slashed:
            patterns.append(DisagreementPattern(
                kind="byzantine",
                severity=min(1.0, max(repeat_slashed.values()) / len(history)),
                description=(
                    f"Agents slashed multiple times (possible Byzantine): "
                    f"{repeat_slashed}"
                ),
                involved_agents=list(repeat_slashed.keys()),
            ))

        # 4. Fragmentation: no clear majority
        all_votes = [v.weight for rr in history for v in rr.votes]
        if all_votes:
            variance = _variance(all_votes)
            if variance > 0.3:
                patterns.append(DisagreementPattern(
                    kind="fragmentation",
                    severity=min(1.0, variance),
                    description=(
                        f"High vote variance ({variance:.2f}) indicates fragmented opinions"
                    ),
                    involved_agents=agents,
                ))

        return patterns

    # ── Faction detection ──────────────────────────────────────────────

    def _detect_factions(
        self, vote_matrix: Dict[str, List[float]]
    ) -> List[Faction]:
        """Simple correlation-based clustering."""
        agents = list(vote_matrix.keys())
        if len(agents) < 2:
            return [Faction("solo", agents, _mean(vote_matrix[agents[0]]) if agents else 0, 1.0)]

        # Compute pairwise correlation
        corr: Dict[Tuple[str, str], float] = {}
        for i, a1 in enumerate(agents):
            for a2 in agents[i + 1:]:
                corr[(a1, a2)] = _pearson(vote_matrix[a1], vote_matrix[a2])

        # Greedy clustering: merge agents with correlation > 0.5
        clusters: List[set] = [{a} for a in agents]

        def find_cluster(agent: str) -> int:
            for idx, c in enumerate(clusters):
                if agent in c:
                    return idx
            return -1

        for (a1, a2), r in sorted(corr.items(), key=lambda x: -x[1]):
            if r < 0.3:
                break
            c1, c2 = find_cluster(a1), find_cluster(a2)
            if c1 != c2 and c1 >= 0 and c2 >= 0:
                clusters[c1] |= clusters[c2]
                clusters.pop(c2)

        factions = []
        for i, cluster in enumerate(clusters):
            members = sorted(cluster)
            weights = [_mean(vote_matrix[m]) for m in members]
            cohesion = 1.0 - _variance(weights) if len(weights) > 1 else 1.0
            factions.append(Faction(
                name=f"faction_{i + 1}",
                members=members,
                avg_weight=_mean(weights),
                cohesion=max(0.0, min(1.0, cohesion)),
            ))
        return factions

    # ── Strategy recommendation ────────────────────────────────────────

    def _recommend_strategies(
        self,
        patterns: List[DisagreementPattern],
        factions: List[Faction],
        history: List[RoundResult],
    ) -> List[MediationStrategy]:
        strategies: List[MediationStrategy] = []
        pattern_kinds = {p.kind for p in patterns}

        if "polarization" in pattern_kinds:
            polar = next(p for p in patterns if p.kind == "polarization")
            strategies.append(MediationStrategy(
                name="Coalition Building",
                description=(
                    "Pair dissenting agents with supporters for pre-round "
                    "deliberation to find common ground before voting."
                ),
                priority=1,
                parameters={
                    "dissenters": polar.involved_agents,
                    "approach": "pair_deliberation",
                },
            ))

        if "low_confidence" in pattern_kinds:
            strategies.append(MediationStrategy(
                name="Threshold Adjustment",
                description=(
                    "Lower the consensus threshold temporarily to allow progress, "
                    "then ratchet back up as confidence improves."
                ),
                priority=2,
                parameters={
                    "current_threshold": self.engine.threshold,
                    "suggested_threshold": round(self.engine.threshold * 0.75, 3),
                    "ratchet_step": 0.05,
                },
            ))

        if "byzantine" in pattern_kinds:
            byz = next(p for p in patterns if p.kind == "byzantine")
            strategies.append(MediationStrategy(
                name="Quarantine & Audit",
                description=(
                    "Isolate suspected Byzantine agents and require proof "
                    "verification before re-admitting them to voting."
                ),
                priority=1,
                parameters={
                    "suspects": byz.involved_agents,
                    "action": "quarantine_then_audit",
                },
            ))

        if "fragmentation" in pattern_kinds:
            strategies.append(MediationStrategy(
                name="Task Decomposition",
                description=(
                    "Break the task into sub-questions where factions can "
                    "reach partial consensus, then compose the final answer."
                ),
                priority=2,
                parameters={
                    "num_factions": len(factions),
                    "approach": "divide_and_compose",
                },
            ))

        if len(factions) > 2:
            strategies.append(MediationStrategy(
                name="Representative Council",
                description=(
                    "Elect one representative per faction for a smaller, "
                    "more focused consensus round."
                ),
                priority=3,
                parameters={
                    "factions": [f.name for f in factions],
                    "council_size": len(factions),
                },
            ))

        # Always suggest: increase rounds if we hit the limit
        if len(history) >= self.engine.max_rounds and not any(
            r.committed for r in history
        ):
            strategies.append(MediationStrategy(
                name="Extended Deliberation",
                description=(
                    "Increase max_rounds to allow more negotiation time. "
                    "The swarm was close but ran out of rounds."
                ),
                priority=3,
                parameters={
                    "current_max": self.engine.max_rounds,
                    "suggested_max": self.engine.max_rounds * 2,
                },
            ))

        return strategies

    # ── HTML export ────────────────────────────────────────────────────

    def export_html(self, path: str) -> None:
        """Write an interactive HTML mediation report."""
        report = self._report or self.analyze()
        html = _build_html(report)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def to_json(self) -> str:
        report = self._report or self.analyze()
        return json.dumps({
            "rounds_analyzed": report.rounds_analyzed,
            "consensus_reached": report.consensus_reached,
            "patterns": [
                {"kind": p.kind, "severity": p.severity,
                 "description": p.description, "agents": p.involved_agents}
                for p in report.patterns
            ],
            "factions": [
                {"name": f.name, "members": f.members,
                 "avg_weight": f.avg_weight, "cohesion": f.cohesion}
                for f in report.factions
            ],
            "strategies": [
                {"name": s.name, "description": s.description,
                 "priority": s.priority, "parameters": s.parameters}
                for s in report.strategies
            ],
            "vote_matrix": report.vote_matrix,
            "reputation_trajectory": report.reputation_trajectory,
        }, indent=2)


# ── Utilities ──────────────────────────────────────────────────────────


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _variance(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx, my = _mean(xs[:n]), _mean(ys[:n])
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


# ── HTML template ──────────────────────────────────────────────────────


def _json_for_export(report: MediationReport) -> str:
    """Build a JSON string safe for embedding in an f-string template."""
    obj = {
        "rounds_analyzed": report.rounds_analyzed,
        "consensus_reached": report.consensus_reached,
        "patterns": [{"kind": p.kind, "severity": p.severity, "description": p.description} for p in report.patterns],
        "factions": [{"name": f.name, "members": f.members, "avg_weight": f.avg_weight} for f in report.factions],
        "strategies": [{"name": s.name, "priority": s.priority, "description": s.description} for s in report.strategies],
    }
    return json.dumps(obj)


def _build_html(report: MediationReport) -> str:
    patterns_html = ""
    for p in report.patterns:
        color = {"polarization": "#e74c3c", "low_confidence": "#f39c12",
                 "byzantine": "#9b59b6", "fragmentation": "#3498db"}.get(p.kind, "#95a5a6")
        patterns_html += f"""
        <div class="card" style="border-left:4px solid {color}">
          <h3 style="color:{color}">⚠ {p.kind.upper()}</h3>
          <div class="severity">Severity: <span style="color:{color}">{p.severity:.0%}</span></div>
          <p>{p.description}</p>
          <div class="agents">Agents: {', '.join(p.involved_agents)}</div>
        </div>"""

    factions_html = ""
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
    for i, f in enumerate(report.factions):
        c = colors[i % len(colors)]
        factions_html += f"""
        <div class="faction" style="border-color:{c}">
          <strong style="color:{c}">{f.name}</strong>
          <div>Members: {', '.join(f.members)}</div>
          <div>Avg weight: {f.avg_weight:.2f} | Cohesion: {f.cohesion:.0%}</div>
        </div>"""

    strategies_html = ""
    for s in sorted(report.strategies, key=lambda x: x.priority):
        strategies_html += f"""
        <div class="strategy">
          <span class="priority">P{s.priority}</span>
          <strong>{s.name}</strong>
          <p>{s.description}</p>
          {''.join(f'<div class="param">{k}: {v}</div>' for k, v in s.parameters.items())}
        </div>"""

    # Vote matrix for heatmap
    agents = sorted(report.vote_matrix.keys())
    max_rounds = max((len(v) for v in report.vote_matrix.values()), default=0)
    heatmap_data = json.dumps({
        "agents": agents,
        "rounds": max_rounds,
        "values": {a: report.vote_matrix.get(a, []) for a in agents},
    })

    rep_data = json.dumps(report.reputation_trajectory)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mBFT Consensus Mediation Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px}}
h1{{color:#58a6ff;margin-bottom:4px}}
h2{{color:#8b949e;margin:24px 0 12px;border-bottom:1px solid #21262d;padding-bottom:6px}}
.status{{font-size:1.1em;margin:8px 0 16px;padding:8px 16px;border-radius:8px;display:inline-block}}
.status.ok{{background:#1a3a2a;color:#3fb950}}
.status.fail{{background:#3a1a1a;color:#f85149}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin:12px 0}}
.card,.faction,.strategy{{background:#161b22;border-radius:8px;padding:16px;border:1px solid #30363d}}
.card h3{{margin-bottom:8px}}
.severity{{font-size:0.9em;margin-bottom:8px}}
.agents{{font-size:0.85em;color:#8b949e;margin-top:8px}}
.faction{{border-left:3px solid;padding-left:14px}}
.strategy{{position:relative;padding-left:50px}}
.priority{{position:absolute;left:12px;top:16px;background:#21262d;color:#58a6ff;
  border-radius:50%;width:28px;height:28px;display:flex;align-items:center;
  justify-content:center;font-weight:bold;font-size:0.8em}}
.strategy p{{margin:6px 0;color:#8b949e}}
.param{{font-size:0.8em;color:#7d8590;margin:2px 0}}
canvas{{background:#161b22;border-radius:8px;border:1px solid #30363d;margin:8px 0}}
.controls{{margin:8px 0}}
.controls button{{background:#21262d;color:#c9d1d9;border:1px solid #30363d;padding:6px 14px;
  border-radius:6px;cursor:pointer;margin-right:6px}}
.controls button:hover{{background:#30363d}}
.controls button.active{{background:#1f6feb;border-color:#1f6feb}}
</style></head><body>
<h1>🤝 Consensus Mediation Report</h1>
<div class="status {'ok' if report.consensus_reached else 'fail'}">
  {'✅ Consensus Reached' if report.consensus_reached else '❌ No Consensus'} — {report.rounds_analyzed} rounds analyzed
</div>

<h2>⚠ Disagreement Patterns</h2>
<div class="grid">{patterns_html if patterns_html else '<p style="color:#8b949e">No patterns detected.</p>'}</div>

<h2>👥 Faction Analysis</h2>
<div class="grid">{factions_html}</div>

<h2>📊 Vote Heatmap</h2>
<canvas id="heatmap" width="700" height="200"></canvas>

<h2>📉 Reputation Trajectory</h2>
<canvas id="repChart" width="700" height="200"></canvas>

<h2>💡 Mediation Strategies</h2>
<div class="grid">{strategies_html if strategies_html else '<p style="color:#8b949e">No strategies needed.</p>'}</div>

<div class="controls">
  <button onclick="exportJSON()">Export JSON</button>
</div>

<script>
const heatmapData = {heatmap_data};
const repData = {rep_data};

// Vote heatmap
(function() {{
  const c = document.getElementById('heatmap');
  const ctx = c.getContext('2d');
  const agents = heatmapData.agents;
  const rounds = heatmapData.rounds;
  if (!agents.length || !rounds) return;

  const cellW = Math.min(60, (c.width - 80) / rounds);
  const cellH = Math.min(40, (c.height - 30) / agents.length);
  c.width = 80 + cellW * rounds + 10;
  c.height = 30 + cellH * agents.length + 10;

  ctx.font = '11px monospace';
  ctx.fillStyle = '#8b949e';
  for (let r = 0; r < rounds; r++) {{
    ctx.fillText('R' + r, 80 + r * cellW + cellW/2 - 8, 18);
  }}

  agents.forEach((a, i) => {{
    ctx.fillStyle = '#8b949e';
    ctx.fillText(a, 4, 30 + i * cellH + cellH/2 + 4);
    const vals = heatmapData.values[a] || [];
    vals.forEach((v, r) => {{
      const t = (v + 1) / 2; // -1..1 -> 0..1
      const red = Math.round(255 * (1 - t));
      const green = Math.round(255 * t);
      ctx.fillStyle = `rgb(${{red}},${{green}},60)`;
      ctx.fillRect(80 + r * cellW + 1, 24 + i * cellH + 1, cellW - 2, cellH - 2);
      ctx.fillStyle = '#fff';
      ctx.fillText(v.toFixed(1), 80 + r * cellW + 4, 24 + i * cellH + cellH/2 + 4);
    }});
  }});
}})();

// Reputation chart
(function() {{
  const c = document.getElementById('repChart');
  const ctx = c.getContext('2d');
  const agents = Object.keys(repData);
  if (!agents.length) return;

  const maxLen = Math.max(...agents.map(a => repData[a].length));
  const colors = ['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff','#39d2c0'];
  const pad = {{l:50,r:20,t:20,b:30}};
  const w = c.width - pad.l - pad.r;
  const h = c.height - pad.t - pad.b;

  // Grid
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  for (let y = 0; y <= 4; y++) {{
    const py = pad.t + h - (y/4) * h;
    ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(pad.l + w, py); ctx.stroke();
    ctx.fillStyle = '#8b949e'; ctx.font = '10px monospace';
    ctx.fillText((y/4).toFixed(1), 8, py + 4);
  }}

  agents.forEach((a, i) => {{
    const vals = repData[a];
    ctx.strokeStyle = colors[i % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    vals.forEach((v, j) => {{
      const x = pad.l + (j / (maxLen - 1)) * w;
      const y = pad.t + h - v * h;
      j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.stroke();
    // Label
    const lastV = vals[vals.length - 1];
    ctx.fillStyle = colors[i % colors.length];
    ctx.font = '10px monospace';
    ctx.fillText(a, pad.l + w + 4, pad.t + h - lastV * h + 4);
  }});
}})();

const fullReport = {_json_for_export(report)};

function exportJSON() {{
  const blob = new Blob([JSON.stringify(fullReport, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'mediation_report.json';
  a.click();
}}
</script></body></html>"""


# ── CLI entry point ────────────────────────────────────────────────────


async def _demo() -> None:
    from src.agents.metacognitive import MockAgent
    from src.core.protocol import MBFTEngine

    # Build a contentious swarm that won't reach consensus easily
    agents = [
        MockAgent("alice", answer="42", confidence=0.85),
        MockAgent("bob", answer="42", confidence=0.70),
        MockAgent("carol", answer="17", confidence=0.80),  # dissenter
        MockAgent("dave", answer="17", confidence=0.75),   # dissenter
        MockAgent("eve", answer="999", confidence=0.99, byzantine=True),
    ]
    engine = MBFTEngine(agents=agents, threshold=2.0, max_rounds=4)
    result = await engine.run("What is the meaning of life?")

    mediator = ConsensusMediator(engine)
    report = mediator.analyze()
    print(report.summary)

    export_path = None
    if "--export" in sys.argv:
        idx = sys.argv.index("--export")
        if idx + 1 < len(sys.argv):
            export_path = sys.argv[idx + 1]
    if export_path is None:
        export_path = "mediation_report.html"

    mediator.export_html(export_path)
    print(f"\nHTML report exported to: {export_path}")


def main() -> None:
    import asyncio
    asyncio.run(_demo())


if __name__ == "__main__":
    main()
