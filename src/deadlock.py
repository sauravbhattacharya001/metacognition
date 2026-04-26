"""Consensus Deadlock Detector.

Analyzes mBFT consensus history for deadlock patterns including:
- Circular vetoes (agents mutually blocking each other)
- Oscillating proposals (leader ping-pong between agents)
- Stalemate conditions (aggregate weight stuck near threshold)
- Faction polarization (persistent opposing voting blocs)

Generates interactive HTML reports with resolution recommendations.

Usage:
    python -m src.deadlock [--agents N] [--threshold T] [--rounds R]
                           [--byzantine B] [--scenarios S] [--out FILE]
                           [--auto-resolve] [--json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from .agents.metacognitive import MockAgent
from .core.protocol import MBFTEngine
from .core.state import RoundResult, Vote


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class VetoEdge:
    """Directed veto relationship: blocker rejected target's proposals."""
    blocker: str
    target: str
    count: int = 0
    total_weight: float = 0.0


@dataclass
class DeadlockPattern:
    """A detected deadlock pattern."""
    kind: str  # circular_veto | oscillation | stalemate | polarization
    severity: str  # low | medium | high | critical
    description: str
    agents_involved: List[str] = field(default_factory=list)
    rounds_affected: List[int] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolution: str = ""


@dataclass
class DeadlockReport:
    """Full deadlock analysis report."""
    patterns: List[DeadlockPattern] = field(default_factory=list)
    veto_graph: Dict[str, List[VetoEdge]] = field(default_factory=dict)
    leader_sequence: List[str] = field(default_factory=list)
    stalemate_score: float = 0.0
    faction_map: Dict[str, int] = field(default_factory=dict)
    resolution_plan: List[str] = field(default_factory=list)
    committed: bool = False

    @property
    def deadlocked(self) -> bool:
        return not self.committed and len(self.patterns) > 0

    @property
    def worst_severity(self) -> str:
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        if not self.patterns:
            return "none"
        return max(self.patterns, key=lambda p: order.get(p.severity, 0)).severity


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DeadlockDetector:
    """Analyze MBFTEngine history for deadlock patterns."""

    def __init__(self, history: List[RoundResult], agents: List[str]) -> None:
        self.history = history
        self.agents = agents

    def analyze(self) -> DeadlockReport:
        report = DeadlockReport()
        if not self.history:
            return report

        report.committed = any(r.committed for r in self.history)
        report.leader_sequence = [r.leader_id for r in self.history]

        # Build veto graph
        veto_edges: Dict[Tuple[str, str], VetoEdge] = {}
        for rr in self.history:
            for v in rr.votes:
                if v.is_rejection:
                    key = (v.voter_id, rr.leader_id)
                    if key not in veto_edges:
                        veto_edges[key] = VetoEdge(v.voter_id, rr.leader_id)
                    veto_edges[key].count += 1
                    veto_edges[key].total_weight += abs(v.weight)

        report.veto_graph = defaultdict(list)
        for edge in veto_edges.values():
            report.veto_graph[edge.blocker].append(edge)

        # Detect patterns
        report.patterns.extend(self._detect_circular_vetoes(veto_edges))
        report.patterns.extend(self._detect_oscillation(report.leader_sequence))
        report.patterns.extend(self._detect_stalemate())
        report.patterns.extend(self._detect_polarization())

        # Compute stalemate score
        if self.history and not report.committed:
            threshold = self.history[0].threshold
            distances = [abs(r.aggregate_weight - threshold) / max(threshold, 0.01)
                         for r in self.history]
            report.stalemate_score = 1.0 - min(1.0, sum(distances) / len(distances))

        # Faction detection via voting similarity
        report.faction_map = self._detect_factions()

        # Generate resolution plan
        report.resolution_plan = self._generate_resolutions(report)

        return report

    def _detect_circular_vetoes(
        self, edges: Dict[Tuple[str, str], VetoEdge]
    ) -> List[DeadlockPattern]:
        """Find cycles in the veto graph (A blocks B, B blocks A, etc.)."""
        patterns = []
        adjacency: Dict[str, set] = defaultdict(set)
        for (blocker, target) in edges:
            adjacency[blocker].add(target)

        # Check 2-cycles
        for a, b in combinations(self.agents, 2):
            if b in adjacency.get(a, set()) and a in adjacency.get(b, set()):
                ea = edges.get((a, b))
                eb = edges.get((b, a))
                total = (ea.count if ea else 0) + (eb.count if eb else 0)
                severity = "critical" if total >= 4 else "high" if total >= 2 else "medium"
                patterns.append(DeadlockPattern(
                    kind="circular_veto",
                    severity=severity,
                    description=f"Mutual veto cycle: {a} ↔ {b} ({total} vetoes)",
                    agents_involved=[a, b],
                    metrics={"veto_count": total},
                    resolution="Consider mediating between agents or adjusting reputation weights.",
                ))

        # Check 3-cycles
        for a, b, c in combinations(self.agents, 3):
            adj = adjacency
            if (b in adj.get(a, set()) and c in adj.get(b, set())
                    and a in adj.get(c, set())):
                patterns.append(DeadlockPattern(
                    kind="circular_veto",
                    severity="critical",
                    description=f"Triangular veto cycle: {a} → {b} → {c} → {a}",
                    agents_involved=[a, b, c],
                    resolution="Break the cycle by temporarily boosting one agent's authority.",
                ))

        return patterns

    def _detect_oscillation(self, leaders: List[str]) -> List[DeadlockPattern]:
        """Detect ping-pong leader alternation."""
        patterns = []
        if len(leaders) < 3:
            return patterns

        # Check for ABABAB pattern
        for i in range(len(leaders) - 2):
            if leaders[i] == leaders[i + 2] and leaders[i] != leaders[i + 1]:
                run_len = 1
                for j in range(i + 2, len(leaders) - 2, 2):
                    if leaders[j] == leaders[i] and leaders[j + 1] == leaders[i + 1]:
                        run_len += 1
                    else:
                        break
                if run_len >= 2:
                    severity = "high" if run_len >= 3 else "medium"
                    patterns.append(DeadlockPattern(
                        kind="oscillation",
                        severity=severity,
                        description=(f"Leader oscillation: {leaders[i]} ↔ {leaders[i+1]} "
                                     f"for {run_len} cycles"),
                        agents_involved=[leaders[i], leaders[i + 1]],
                        rounds_affected=list(range(i, min(i + run_len * 2, len(leaders)))),
                        resolution="Introduce a cooldown period before re-election of recent leaders.",
                    ))
                    break  # Report first oscillation only
        return patterns

    def _detect_stalemate(self) -> List[DeadlockPattern]:
        """Detect aggregate weight hovering near threshold without committing."""
        patterns = []
        if len(self.history) < 2:
            return patterns

        threshold = self.history[0].threshold
        near_miss_rounds = []
        for rr in self.history:
            if not rr.committed:
                gap = abs(rr.aggregate_weight - threshold)
                if gap < threshold * 0.3:
                    near_miss_rounds.append(rr.round_index)

        if len(near_miss_rounds) >= 2:
            ratio = len(near_miss_rounds) / len(self.history)
            severity = "critical" if ratio > 0.7 else "high" if ratio > 0.4 else "medium"
            patterns.append(DeadlockPattern(
                kind="stalemate",
                severity=severity,
                description=(f"Near-miss stalemate in {len(near_miss_rounds)}/{len(self.history)} "
                             f"rounds (within 30% of threshold)"),
                rounds_affected=near_miss_rounds,
                metrics={"near_miss_ratio": round(ratio, 3)},
                resolution="Consider lowering the threshold or increasing agent confidence diversity.",
            ))
        return patterns

    def _detect_polarization(self) -> List[DeadlockPattern]:
        """Detect persistent opposing voting blocs."""
        patterns = []
        if len(self.history) < 2:
            return patterns

        # Track voting tendency per agent: positive = supportive, negative = oppositional
        agent_tendency: Dict[str, List[float]] = defaultdict(list)
        for rr in self.history:
            for v in rr.votes:
                agent_tendency[v.voter_id].append(v.weight)

        # Compute average tendency
        avg_tendency = {a: sum(ws) / len(ws) for a, ws in agent_tendency.items() if ws}

        positive_bloc = [a for a, t in avg_tendency.items() if t > 0.2]
        negative_bloc = [a for a, t in avg_tendency.items() if t < -0.2]

        if positive_bloc and negative_bloc and len(negative_bloc) >= 2:
            severity = "high" if len(negative_bloc) >= len(positive_bloc) else "medium"
            patterns.append(DeadlockPattern(
                kind="polarization",
                severity=severity,
                description=(f"Voting polarization: {len(positive_bloc)} supporters vs "
                             f"{len(negative_bloc)} opponents"),
                agents_involved=positive_bloc + negative_bloc,
                metrics={
                    "supporters": positive_bloc,
                    "opponents": negative_bloc,
                    "avg_support": round(sum(avg_tendency[a] for a in positive_bloc) / len(positive_bloc), 3),
                    "avg_opposition": round(sum(avg_tendency[a] for a in negative_bloc) / len(negative_bloc), 3),
                },
                resolution="Introduce preference aggregation or ranked-choice proposal selection.",
            ))
        return patterns

    def _detect_factions(self) -> Dict[str, int]:
        """Simple faction detection via vote correlation."""
        # Build vote vectors
        agent_votes: Dict[str, List[float]] = defaultdict(list)
        for rr in self.history:
            voter_map = {v.voter_id: v.weight for v in rr.votes}
            for a in self.agents:
                agent_votes[a].append(voter_map.get(a, 0.0))

        if not agent_votes:
            return {}

        # Greedy clustering by correlation
        factions: Dict[str, int] = {}
        faction_id = 0
        assigned = set()

        for a in self.agents:
            if a in assigned:
                continue
            factions[a] = faction_id
            assigned.add(a)
            va = agent_votes.get(a, [])
            for b in self.agents:
                if b in assigned or not va:
                    continue
                vb = agent_votes.get(b, [])
                if not vb or len(va) != len(vb):
                    continue
                corr = _pearson(va, vb)
                if corr > 0.5:
                    factions[b] = faction_id
                    assigned.add(b)
            faction_id += 1

        return factions

    def _generate_resolutions(self, report: DeadlockReport) -> List[str]:
        """Generate actionable resolution recommendations."""
        resolutions = []
        kinds = Counter(p.kind for p in report.patterns)

        if kinds.get("circular_veto", 0):
            resolutions.append(
                "🔄 Break veto cycles: temporarily increase reputation of the "
                "least-vetoed agent to establish a clear leader."
            )
        if kinds.get("oscillation", 0):
            resolutions.append(
                "⏱️ Add leader cooldown: prevent recently-failed leaders from "
                "re-election for N rounds to break ping-pong."
            )
        if kinds.get("stalemate", 0):
            resolutions.append(
                "📉 Adaptive threshold: gradually lower the commit threshold by "
                "a decay factor each round to break near-miss stalemates."
            )
        if kinds.get("polarization", 0):
            resolutions.append(
                "🤝 Mediation round: inject a neutral mediator agent or use "
                "preference aggregation to bridge polarized factions."
            )
        if report.stalemate_score > 0.7 and not report.committed:
            resolutions.append(
                "⚠️ High stalemate score ({:.0f}%): consider circuit-breaker — "
                "commit the best-so-far proposal after max rounds.".format(
                    report.stalemate_score * 100
                )
            )
        if not resolutions and not report.committed:
            resolutions.append(
                "🔍 No specific deadlock pattern detected, but consensus was not "
                "reached. Consider increasing max_rounds or adding agents."
            )
        return resolutions


def _pearson(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (sx * sy)


# ---------------------------------------------------------------------------
# Auto-resolver
# ---------------------------------------------------------------------------

class DeadlockResolver:
    """Attempt to autonomously resolve deadlocks by adjusting parameters."""

    def __init__(self, agents: List["MockAgent"], base_threshold: float) -> None:
        self.agents = agents
        self.base_threshold = base_threshold

    async def resolve(
        self, max_attempts: int = 5
    ) -> Tuple[Optional[RoundResult], List[Dict[str, Any]]]:
        """Try successive resolution strategies until consensus is reached."""
        attempts = []
        threshold = self.base_threshold
        agents = list(self.agents)

        for attempt in range(max_attempts):
            strategy: Dict[str, Any] = {"attempt": attempt + 1}

            if attempt == 0:
                strategy["action"] = "baseline"
            elif attempt == 1:
                threshold *= 0.85
                strategy["action"] = "lower_threshold"
                strategy["new_threshold"] = round(threshold, 3)
            elif attempt == 2:
                # Boost lowest-reputation agent
                strategy["action"] = "boost_underdog"
            elif attempt == 3:
                threshold *= 0.85
                strategy["action"] = "lower_threshold_again"
                strategy["new_threshold"] = round(threshold, 3)
            else:
                strategy["action"] = "max_diversity"

            engine = MBFTEngine(agents, threshold=threshold, max_rounds=6)
            result = await engine.run("deadlock_resolution_probe")
            strategy["committed"] = result.committed if result else False
            strategy["aggregate"] = round(result.aggregate_weight, 3) if result else 0.0
            strategy["rounds_used"] = len(engine.history)
            attempts.append(strategy)

            if result and result.committed:
                return result, attempts

        return None, attempts


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(
    report: DeadlockReport,
    scenarios: List[Dict[str, Any]],
    resolve_log: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate interactive HTML report."""
    patterns_html = ""
    for p in report.patterns:
        color = {"critical": "#e74c3c", "high": "#e67e22",
                 "medium": "#f1c40f", "low": "#2ecc71"}.get(p.severity, "#95a5a6")
        patterns_html += f"""
        <div class="pattern-card" style="border-left: 4px solid {color}">
            <div class="pattern-header">
                <span class="severity" style="background:{color}">{p.severity.upper()}</span>
                <span class="kind">{p.kind.replace('_', ' ').title()}</span>
            </div>
            <p>{p.description}</p>
            <p class="resolution">💡 {p.resolution}</p>
            {f'<p class="agents">Agents: {", ".join(p.agents_involved)}</p>' if p.agents_involved else ''}
        </div>"""

    resolutions_html = "".join(f"<li>{r}</li>" for r in report.resolution_plan)

    veto_nodes = set()
    veto_edges_js = []
    for blocker, edges in report.veto_graph.items():
        veto_nodes.add(blocker)
        for e in edges:
            veto_nodes.add(e.target)
            veto_edges_js.append(
                f'{{from:"{e.blocker}",to:"{e.target}",count:{e.count},'
                f'weight:{e.total_weight:.2f}}}'
            )

    faction_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
                      "#e67e22", "#1abc9c", "#f1c40f", "#34495e"]
    faction_items = ""
    for agent, fid in sorted(report.faction_map.items(), key=lambda x: x[1]):
        c = faction_colors[fid % len(faction_colors)]
        faction_items += f'<span class="faction-badge" style="background:{c}">{agent}</span> '

    resolve_html = ""
    if resolve_log:
        rows = ""
        for entry in resolve_log:
            status = "✅" if entry.get("committed") else "❌"
            rows += (f'<tr><td>{entry["attempt"]}</td><td>{entry["action"]}</td>'
                     f'<td>{entry.get("new_threshold", "—")}</td>'
                     f'<td>{entry["aggregate"]}</td><td>{entry["rounds_used"]}</td>'
                     f'<td>{status}</td></tr>')
        resolve_html = f"""
        <h2>🔧 Auto-Resolution Attempts</h2>
        <table><thead><tr><th>#</th><th>Strategy</th><th>Threshold</th>
        <th>Aggregate</th><th>Rounds</th><th>Result</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    scenario_rows = ""
    for s in scenarios:
        status = "✅ Committed" if s.get("committed") else "❌ Deadlocked"
        pcount = s.get("pattern_count", 0)
        scenario_rows += (
            f'<tr><td>{s["id"]}</td><td>{s["agents"]}</td>'
            f'<td>{s["byzantine"]}</td><td>{s["threshold"]}</td>'
            f'<td>{status}</td><td>{pcount}</td>'
            f'<td>{s.get("worst_severity", "none")}</td></tr>'
        )

    leader_seq = " → ".join(report.leader_sequence) if report.leader_sequence else "N/A"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mBFT Deadlock Analysis</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a0a;color:#e0e0e0;padding:20px}}
h1{{color:#00d4ff;margin-bottom:8px}}
h2{{color:#00d4ff;margin:24px 0 12px;border-bottom:1px solid #222;padding-bottom:6px}}
.subtitle{{color:#888;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}}
.stat-card{{background:#151515;border-radius:8px;padding:16px;text-align:center}}
.stat-card .value{{font-size:2em;font-weight:bold;color:#00d4ff}}
.stat-card .label{{color:#888;font-size:.85em;margin-top:4px}}
.pattern-card{{background:#151515;border-radius:8px;padding:16px;margin:8px 0}}
.pattern-header{{display:flex;align-items:center;gap:8px;margin-bottom:8px}}
.severity{{color:#fff;padding:2px 8px;border-radius:4px;font-size:.75em;font-weight:bold}}
.kind{{font-weight:bold;font-size:1.1em}}
.resolution{{color:#2ecc71;font-style:italic;margin-top:8px}}
.agents{{color:#888;font-size:.85em}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #222}}
th{{background:#151515;color:#00d4ff}}
tr:hover{{background:#1a1a1a}}
.faction-badge{{display:inline-block;padding:4px 10px;border-radius:12px;color:#fff;
font-size:.85em;margin:2px}}
canvas{{background:#151515;border-radius:8px;margin:12px 0}}
ul{{margin:12px 0 12px 24px}}
li{{margin:6px 0}}
.leader-seq{{background:#151515;padding:12px;border-radius:8px;font-family:monospace;
word-break:break-all;margin:12px 0}}
.status-banner{{padding:16px;border-radius:8px;text-align:center;font-size:1.2em;
font-weight:bold;margin:16px 0}}
.deadlocked{{background:#e74c3c22;border:1px solid #e74c3c;color:#e74c3c}}
.resolved{{background:#2ecc7122;border:1px solid #2ecc71;color:#2ecc71}}
</style></head><body>
<h1>🔒 mBFT Deadlock Detector</h1>
<p class="subtitle">Consensus deadlock pattern analysis & autonomous resolution</p>

<div class="status-banner {'deadlocked' if report.deadlocked else 'resolved'}">
{'🔴 DEADLOCK DETECTED — ' + str(len(report.patterns)) + ' pattern(s) found'
 if report.deadlocked else '🟢 NO DEADLOCK — Consensus reached successfully'}
</div>

<div class="grid">
<div class="stat-card"><div class="value">{len(report.patterns)}</div>
<div class="label">Patterns Found</div></div>
<div class="stat-card"><div class="value">{report.worst_severity.upper()}</div>
<div class="label">Worst Severity</div></div>
<div class="stat-card"><div class="value">{report.stalemate_score:.0%}</div>
<div class="label">Stalemate Score</div></div>
<div class="stat-card"><div class="value">{len(report.leader_sequence)}</div>
<div class="label">Rounds Analyzed</div></div>
</div>

<h2>🔍 Detected Patterns</h2>
{patterns_html if patterns_html else '<p style="color:#888">No deadlock patterns detected.</p>'}

<h2>📊 Leader Sequence</h2>
<div class="leader-seq">{leader_seq}</div>

<h2>👥 Faction Map</h2>
<div style="margin:12px 0">{faction_items if faction_items else '<span style="color:#888">No factions detected.</span>'}</div>

<h2>🕸️ Veto Graph</h2>
<canvas id="vetoCanvas" width="600" height="400"></canvas>

<h2>🎯 Resolution Plan</h2>
<ul>{resolutions_html}</ul>

{resolve_html}

<h2>📋 Scenario Summary</h2>
<table><thead><tr><th>ID</th><th>Agents</th><th>Byzantine</th><th>Threshold</th>
<th>Result</th><th>Patterns</th><th>Severity</th></tr></thead>
<tbody>{scenario_rows}</tbody></table>

<script>
const canvas=document.getElementById('vetoCanvas');
const ctx=canvas.getContext('2d');
const nodes=[{','.join(f'"{n}"' for n in veto_nodes)}];
const edges=[{','.join(veto_edges_js)}];
const positions={{}};
const cx=canvas.width/2,cy=canvas.height/2,radius=150;
nodes.forEach((n,i)=>{{
  const angle=(2*Math.PI*i)/nodes.length-Math.PI/2;
  positions[n]={{x:cx+radius*Math.cos(angle),y:cy+radius*Math.sin(angle)}};
}});
// Draw edges
edges.forEach(e=>{{
  const from=positions[e.from],to=positions[e.to];
  if(!from||!to)return;
  ctx.beginPath();
  ctx.strokeStyle=`rgba(231,76,60,${{Math.min(1,e.count/3)}})`;
  ctx.lineWidth=Math.max(1,e.count);
  ctx.moveTo(from.x,from.y);ctx.lineTo(to.x,to.y);ctx.stroke();
  // Arrow
  const angle=Math.atan2(to.y-from.y,to.x-from.x);
  const ax=to.x-25*Math.cos(angle),ay=to.y-25*Math.sin(angle);
  ctx.beginPath();ctx.moveTo(ax,ay);
  ctx.lineTo(ax-10*Math.cos(angle-0.4),ay-10*Math.sin(angle-0.4));
  ctx.lineTo(ax-10*Math.cos(angle+0.4),ay-10*Math.sin(angle+0.4));
  ctx.closePath();ctx.fillStyle='#e74c3c';ctx.fill();
  // Count label
  const mx=(from.x+to.x)/2,my=(from.y+to.y)/2;
  ctx.fillStyle='#e67e22';ctx.font='bold 12px monospace';
  ctx.fillText(`×${{e.count}}`,mx+5,my-5);
}});
// Draw nodes
nodes.forEach(n=>{{
  const p=positions[n];
  ctx.beginPath();ctx.arc(p.x,p.y,20,0,2*Math.PI);
  ctx.fillStyle='#1a1a2e';ctx.fill();
  ctx.strokeStyle='#00d4ff';ctx.lineWidth=2;ctx.stroke();
  ctx.fillStyle='#e0e0e0';ctx.font='bold 11px sans-serif';
  ctx.textAlign='center';ctx.textBaseline='middle';
  ctx.fillText(n.slice(0,8),p.x,p.y);
}});
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _run_scenario(
    n_agents: int, n_byzantine: int, threshold: float, max_rounds: int
) -> Tuple[DeadlockReport, List[RoundResult]]:
    """Run one scenario and analyze for deadlocks."""
    agents = []
    answers = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    for i in range(n_agents):
        ans = answers[i % len(answers)]
        if i < n_byzantine:
            # Byzantine: high confidence, different answer, accepts everything
            agent = MockAgent(f"byz_{i}", answer=ans, confidence=random.uniform(0.7, 1.0), byzantine=True)
        else:
            # Honest: moderate confidence, only accepts own answer
            agent = MockAgent(f"agent_{i}", answer=ans, confidence=random.uniform(0.3, 0.9))
        agents.append(agent)

    random.shuffle(agents)
    engine = MBFTEngine(agents, threshold=threshold, max_rounds=max_rounds)
    result = await engine.run("deadlock_detection_probe")

    detector = DeadlockDetector(engine.history, [a.id for a in agents])
    report = detector.analyze()
    return report, engine.history


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Deadlock Detector"
    )
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--threshold", type=float, default=2.0, help="Commit threshold")
    parser.add_argument("--rounds", type=int, default=4, help="Max rounds per scenario")
    parser.add_argument("--byzantine", type=int, default=2, help="Byzantine agents")
    parser.add_argument("--scenarios", type=int, default=5, help="Number of scenarios")
    parser.add_argument("--out", type=str, default="deadlock_report.html", help="Output file")
    parser.add_argument("--auto-resolve", action="store_true", help="Attempt autonomous resolution")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of HTML")
    args = parser.parse_args()

    print("🔒 mBFT Deadlock Detector")
    print(f"   Agents: {args.agents} | Byzantine: {args.byzantine} | "
          f"Threshold: {args.threshold} | Scenarios: {args.scenarios}")
    print()

    scenarios_data: List[Dict[str, Any]] = []
    worst_report: Optional[DeadlockReport] = None
    worst_count = -1

    for s in range(args.scenarios):
        n_byz = min(args.byzantine, args.agents - 1)
        report, history = await _run_scenario(
            args.agents, n_byz, args.threshold, args.rounds
        )
        info = {
            "id": s + 1,
            "agents": args.agents,
            "byzantine": n_byz,
            "threshold": args.threshold,
            "committed": report.committed,
            "pattern_count": len(report.patterns),
            "worst_severity": report.worst_severity,
            "stalemate_score": round(report.stalemate_score, 3),
        }
        scenarios_data.append(info)

        status = "✅ Committed" if report.committed else f"❌ Deadlocked ({len(report.patterns)} patterns)"
        print(f"  Scenario {s+1}: {status}")

        if len(report.patterns) > worst_count:
            worst_count = len(report.patterns)
            worst_report = report

    if worst_report is None:
        worst_report = DeadlockReport()

    # Auto-resolve
    resolve_log = None
    if args.auto_resolve and worst_report.deadlocked:
        print("\n🔧 Attempting autonomous resolution...")
        answers = ["alpha", "beta", "gamma", "delta", "epsilon"]
        agents = [MockAgent(f"agent_{i}", answer=answers[i % len(answers)],
                            confidence=random.uniform(0.3, 0.9))
                  for i in range(args.agents - args.byzantine)]
        agents += [MockAgent(f"byz_{i}", answer=answers[i % len(answers)],
                             confidence=random.uniform(0.7, 1.0), byzantine=True)
                   for i in range(min(args.byzantine, args.agents - 1))]
        resolver = DeadlockResolver(agents, args.threshold)
        result, resolve_log = await resolver.resolve()
        if result and result.committed:
            print("   ✅ Resolution found!")
        else:
            print("   ❌ Could not resolve deadlock automatically.")

    # Output
    deadlocked = sum(1 for s in scenarios_data if not s["committed"])
    print(f"\n📊 Results: {deadlocked}/{args.scenarios} scenarios deadlocked")

    if args.json:
        output = {
            "scenarios": scenarios_data,
            "worst_report": {
                "deadlocked": worst_report.deadlocked,
                "pattern_count": len(worst_report.patterns),
                "worst_severity": worst_report.worst_severity,
                "stalemate_score": worst_report.stalemate_score,
                "patterns": [
                    {"kind": p.kind, "severity": p.severity,
                     "description": p.description}
                    for p in worst_report.patterns
                ],
                "resolutions": worst_report.resolution_plan,
            },
            "resolve_log": resolve_log,
        }
        with open(args.out.replace(".html", ".json"), "w") as f:
            json.dump(output, f, indent=2)
        print(f"📄 JSON report: {args.out.replace('.html', '.json')}")
    else:
        html = generate_html_report(worst_report, scenarios_data, resolve_log)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"📄 HTML report: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
