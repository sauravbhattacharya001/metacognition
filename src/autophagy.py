"""Swarm Autophagy Engine — autonomous self-cleaning for mBFT swarms.

Inspired by cellular autophagy (the biological process where cells degrade
and recycle their own damaged components), this engine detects and removes
dysfunctional *internal* swarm components.  Unlike the immune system
(src/immune.py) which fights external threats, autophagy targets internal
decay: stale agents, zombie memories, circular dependencies, metabolic
waste, senescent agents, misfolded state, and organelle dysfunction.

Capabilities:

- **7 Dysfunction Detectors** — stale_agent, zombie_memory,
  circular_dependency, metabolic_waste, senescent_agent,
  protein_misfolding, organelle_dysfunction.
- **4 Autophagy Modes** — monitor → tag → degrade → recycle
  (progressive cleanup aggression).
- **Autophagy Score 0-100** — overall swarm cleanliness/health.
- **Lysosome Queue** — prioritized queue of tagged components awaiting
  processing, ordered by severity.
- **Recycling Ledger** — tracks what was removed and resources recovered.
- **Stress-Induced Escalation** — automatically upgrades mode under swarm
  stress conditions.
- **Cooldown Periods** — prevents over-aggressive cleanup.
- **Selective vs Bulk** — target specific dysfunction types or broad sweep.
- **Persistence** — JSON save/load for cross-session continuity.
- **Interactive HTML Dashboard** — score gauge, dysfunction pie chart,
  queue table, recycling timeline, mode indicator, trend sparklines.

Usage (Python API)::

    from src.autophagy import AutophagyEngine

    engine = AutophagyEngine(mode="tag", cooldown_rounds=5)

    # Feed agent activity each round
    engine.record_round(round_num=1, agent_activity={
        "agent_1": {"votes": 3, "memory_accesses": 5, "state_updates": 2},
        "agent_2": {"votes": 0, "memory_accesses": 0, "state_updates": 0},
    })

    # Run detection
    dysfunctions = engine.detect()

    # Process tagged items
    results = engine.process_queue()

    # Get report
    report = engine.get_report()
    print(report.score, report.mode, report.queue_depth)

    # Persistence
    engine.save("autophagy.json")
    engine = AutophagyEngine.load("autophagy.json")

    # HTML dashboard
    engine.export_html("autophagy_report.html")

CLI::

    python -m src.autophagy [--agents N] [--rounds R]
                            [--mode monitor|tag|degrade|recycle]
                            [--stress-level F] [--cooldown N]
                            [--output report.html] [--json results.json]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .stats_utils import clamp01


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES = ("monitor", "tag", "degrade", "recycle")
MODE_LEVELS = {m: i for i, m in enumerate(MODES)}

DYSFUNCTION_CATEGORIES = (
    "stale_agent",
    "zombie_memory",
    "circular_dependency",
    "metabolic_waste",
    "senescent_agent",
    "protein_misfolding",
    "organelle_dysfunction",
)

STRESS_ESCALATION_THRESHOLD = 0.7  # stress level above which mode auto-escalates
COOLDOWN_DEFAULT = 5  # rounds between processing cycles
STALE_THRESHOLD_ROUNDS = 3  # rounds with no activity → stale
SENESCENCE_WINDOW = 10  # rounds to evaluate performance trend
SENESCENCE_DECLINE_RATE = 0.4  # if perf drops by this fraction → senescent
MEMORY_ZOMBIE_THRESHOLD = 5  # rounds with zero accesses → zombie
WASTE_AGE_THRESHOLD = 8  # rounds for temp state to become waste
MISFOLDING_CONFLICT_THRESHOLD = 0.5  # belief conflict ratio
ORGANELLE_HEALTH_THRESHOLD = 0.3  # subsystem health below this → dysfunction


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Dysfunction:
    """A detected internal dysfunction."""
    dysfunction_id: str
    category: str
    severity: float  # 0.0-1.0
    detected_at: int  # round number
    targets: List[str]  # agent ids or component ids
    details: str = ""
    tagged: bool = False
    degraded: bool = False
    recycled: bool = False


@dataclass
class RecycleEntry:
    """Record of a recycled component."""
    dysfunction_id: str
    category: str
    targets: List[str]
    recycled_at: int
    resources_recovered: float  # abstract resource units


@dataclass
class AgentRoundData:
    """Activity data for one agent in one round."""
    votes: int = 0
    memory_accesses: int = 0
    state_updates: int = 0
    performance_score: float = 1.0
    beliefs: Optional[Dict[str, Any]] = None
    temp_state_age: int = 0


@dataclass
class SubsystemHealth:
    """Health of an internal subsystem/organelle."""
    name: str
    health: float  # 0-1
    last_updated: int = 0


@dataclass
class AutophagyReport:
    """Summary report of autophagy state."""
    score: float
    mode: str
    queue_depth: int
    total_detected: int
    total_recycled: int
    dysfunction_counts: Dict[str, int]
    stress_level: float
    cooldown_remaining: int


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AutophagyEngine:
    """Autonomous swarm self-cleaning engine."""

    def __init__(
        self,
        mode: str = "monitor",
        cooldown_rounds: int = COOLDOWN_DEFAULT,
        stress_level: float = 0.0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {MODES}")
        self.mode = mode
        self.cooldown_rounds = cooldown_rounds
        self.cooldown_remaining = 0
        self.stress_level = clamp01(stress_level)
        self.current_round = 0

        # History per agent: list of AgentRoundData per round
        self.agent_history: Dict[str, List[Tuple[int, AgentRoundData]]] = defaultdict(list)
        # Memory access tracker: memory_id → last_accessed_round
        self.memory_tracker: Dict[str, int] = {}
        # Temp state tracker: state_id → creation_round
        self.temp_state: Dict[str, int] = {}
        # Subsystem health
        self.subsystems: Dict[str, SubsystemHealth] = {}

        # Dysfunctions
        self.dysfunctions: List[Dysfunction] = []
        self.lysosome_queue: List[Dysfunction] = []
        self.recycling_ledger: List[RecycleEntry] = []

        # Score history for sparklines
        self.score_history: List[float] = []

        # Interaction graph for circular dependency detection
        self.interaction_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_round(
        self,
        round_num: int,
        agent_activity: Dict[str, Dict[str, Any]],
        memory_accesses: Optional[Dict[str, bool]] = None,
        temp_state_ids: Optional[List[str]] = None,
        subsystem_health: Optional[Dict[str, float]] = None,
        interactions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Record one round of swarm activity."""
        self.current_round = round_num

        for agent_id, activity in agent_activity.items():
            data = AgentRoundData(
                votes=activity.get("votes", 0),
                memory_accesses=activity.get("memory_accesses", 0),
                state_updates=activity.get("state_updates", 0),
                performance_score=activity.get("performance_score", 1.0),
                beliefs=activity.get("beliefs"),
                temp_state_age=activity.get("temp_state_age", 0),
            )
            self.agent_history[agent_id].append((round_num, data))

        # Track memory accesses
        if memory_accesses:
            for mem_id, accessed in memory_accesses.items():
                if accessed:
                    self.memory_tracker[mem_id] = round_num
                elif mem_id not in self.memory_tracker:
                    self.memory_tracker[mem_id] = round_num

        # Track temp state
        if temp_state_ids:
            for sid in temp_state_ids:
                if sid not in self.temp_state:
                    self.temp_state[sid] = round_num

        # Update subsystem health
        if subsystem_health:
            for name, health in subsystem_health.items():
                self.subsystems[name] = SubsystemHealth(
                    name=name, health=clamp01(health), last_updated=round_num
                )

        # Track interactions for circular dependency detection
        if interactions:
            for src, targets in interactions.items():
                for tgt in targets:
                    self.interaction_graph[src][tgt] += 1

        # Stress-induced escalation
        if self.stress_level >= STRESS_ESCALATION_THRESHOLD:
            current_level = MODE_LEVELS[self.mode]
            if current_level < len(MODES) - 1:
                self.mode = MODES[current_level + 1]

    def set_stress(self, level: float) -> None:
        """Update stress level (0-1)."""
        self.stress_level = clamp01(level)

    def detect(self) -> List[Dysfunction]:
        """Run all 7 dysfunction detectors. Returns newly detected dysfunctions.

        Deduplicates against existing unresolved dysfunctions so the same
        underlying problem (same category + targets) is not re-added every
        round.  Previously only ``_detect_circular_dependencies`` guarded
        against duplicates; the other 6 detectors would silently re-detect
        the same stale agents, zombie memories, etc. on every call —
        inflating counts, degrading the autophagy score, and flooding the
        lysosome queue.
        """
        candidates: List[Dysfunction] = []

        candidates.extend(self._detect_stale_agents())
        candidates.extend(self._detect_zombie_memory())
        candidates.extend(self._detect_circular_dependencies())
        candidates.extend(self._detect_metabolic_waste())
        candidates.extend(self._detect_senescent_agents())
        candidates.extend(self._detect_protein_misfolding())
        candidates.extend(self._detect_organelle_dysfunction())

        # Build a set of (category, frozen-targets) for unresolved existing
        # dysfunctions so we skip re-detecting the same underlying problem.
        existing_keys: set = {
            (d.category, tuple(sorted(d.targets)))
            for d in self.dysfunctions
            if not d.recycled
        }

        new_dysfunctions: List[Dysfunction] = []
        for d in candidates:
            key = (d.category, tuple(sorted(d.targets)))
            if key not in existing_keys:
                new_dysfunctions.append(d)
                existing_keys.add(key)

        self.dysfunctions.extend(new_dysfunctions)

        # Tag if mode allows
        if MODE_LEVELS[self.mode] >= MODE_LEVELS["tag"]:
            for d in new_dysfunctions:
                d.tagged = True
                self.lysosome_queue.append(d)
            # Sort queue by severity (highest first)
            self.lysosome_queue.sort(key=lambda x: x.severity, reverse=True)

        return new_dysfunctions

    def process_queue(self) -> List[RecycleEntry]:
        """Process the lysosome queue based on current mode. Returns recycled entries."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return []

        if not self.lysosome_queue:
            return []

        results: List[RecycleEntry] = []
        processed: List[Dysfunction] = []

        for item in self.lysosome_queue:
            if self.mode == "monitor":
                break  # monitor mode doesn't process
            elif self.mode == "tag":
                item.tagged = True
            elif self.mode == "degrade":
                item.degraded = True
            elif self.mode == "recycle":
                item.degraded = True
                item.recycled = True
                entry = RecycleEntry(
                    dysfunction_id=item.dysfunction_id,
                    category=item.category,
                    targets=item.targets,
                    recycled_at=self.current_round,
                    resources_recovered=item.severity * 10.0,
                )
                results.append(entry)
                self.recycling_ledger.append(entry)
                processed.append(item)

        # Remove recycled items from queue
        for p in processed:
            if p in self.lysosome_queue:
                self.lysosome_queue.remove(p)

        # Reset cooldown
        if results or (self.mode == "degrade" and self.lysosome_queue):
            self.cooldown_remaining = self.cooldown_rounds

        return results

    def get_report(self) -> AutophagyReport:
        """Generate a summary report."""
        score = self._compute_score()
        self.score_history.append(score)

        counts: Dict[str, int] = defaultdict(int)
        for d in self.dysfunctions:
            counts[d.category] += 1

        return AutophagyReport(
            score=score,
            mode=self.mode,
            queue_depth=len(self.lysosome_queue),
            total_detected=len(self.dysfunctions),
            total_recycled=len(self.recycling_ledger),
            dysfunction_counts=dict(counts),
            stress_level=self.stress_level,
            cooldown_remaining=self.cooldown_remaining,
        )

    def _compute_score(self) -> float:
        """Compute autophagy health score 0-100. Higher = cleaner swarm."""
        if not self.agent_history:
            return 100.0

        # Factors that reduce score
        penalties = 0.0
        len(self.agent_history)

        # Pending queue items
        queue_penalty = min(30.0, len(self.lysosome_queue) * 5.0)
        penalties += queue_penalty

        # Unresolved high-severity dysfunctions
        unresolved = [d for d in self.dysfunctions if not d.recycled and d.severity > 0.6]
        severity_penalty = min(40.0, len(unresolved) * 8.0)
        penalties += severity_penalty

        # Stress penalty
        stress_penalty = self.stress_level * 20.0
        penalties += stress_penalty

        # Subsystem health penalty
        if self.subsystems:
            avg_health = sum(s.health for s in self.subsystems.values()) / len(self.subsystems)
            subsystem_penalty = (1.0 - avg_health) * 10.0
            penalties += subsystem_penalty

        return max(0.0, min(100.0, 100.0 - penalties))

    # -----------------------------------------------------------------------
    # Detectors
    # -----------------------------------------------------------------------

    def _detect_stale_agents(self) -> List[Dysfunction]:
        """Detect agents with no meaningful activity for N rounds."""
        results: List[Dysfunction] = []
        for agent_id, history in self.agent_history.items():
            if len(history) < STALE_THRESHOLD_ROUNDS:
                continue
            recent = history[-STALE_THRESHOLD_ROUNDS:]
            total_activity = sum(
                d.votes + d.memory_accesses + d.state_updates for _, d in recent
            )
            if total_activity == 0:
                severity = min(1.0, len(history) / 20.0)
                results.append(Dysfunction(
                    dysfunction_id=f"stale_{agent_id}_{self.current_round}",
                    category="stale_agent",
                    severity=severity,
                    detected_at=self.current_round,
                    targets=[agent_id],
                    details=f"No activity for {STALE_THRESHOLD_ROUNDS}+ rounds",
                ))
        return results

    def _detect_zombie_memory(self) -> List[Dysfunction]:
        """Detect memory entries never accessed."""
        results: List[Dysfunction] = []
        for mem_id, last_access in self.memory_tracker.items():
            rounds_since = self.current_round - last_access
            if rounds_since >= MEMORY_ZOMBIE_THRESHOLD:
                severity = min(1.0, rounds_since / 15.0)
                results.append(Dysfunction(
                    dysfunction_id=f"zombie_mem_{mem_id}_{self.current_round}",
                    category="zombie_memory",
                    severity=severity,
                    detected_at=self.current_round,
                    targets=[mem_id],
                    details=f"Not accessed for {rounds_since} rounds",
                ))
        return results

    def _detect_circular_dependencies(self) -> List[Dysfunction]:
        """Detect agent clusters that only validate each other (echo chambers)."""
        results: List[Dysfunction] = []
        if len(self.interaction_graph) < 2:
            return results

        # Find strongly connected components with exclusive interaction
        agents = list(self.interaction_graph.keys())
        for agent in agents:
            targets = self.interaction_graph[agent]
            if not targets:
                continue
            total_interactions = sum(targets.values())
            if total_interactions == 0:
                continue

            # Check if agent interacts predominantly with a small clique
            sorted_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_targets) >= 1:
                # Top partner concentration
                top_count = sorted_targets[0][1]
                concentration = top_count / total_interactions
                # Check reciprocity with top partner
                partner = sorted_targets[0][0]
                if partner in self.interaction_graph:
                    partner_targets = self.interaction_graph[partner]
                    if agent in partner_targets:
                        partner_total = sum(partner_targets.values())
                        if partner_total > 0:
                            reciprocity = partner_targets[agent] / partner_total
                            if concentration > 0.7 and reciprocity > 0.5:
                                clique = sorted([agent, partner])
                                did = f"circular_{clique[0]}_{clique[1]}_{self.current_round}"
                                # Avoid duplicate detection
                                existing_ids = {d.dysfunction_id for d in self.dysfunctions}
                                existing_ids.update(d.dysfunction_id for d in results)
                                if did not in existing_ids:
                                    results.append(Dysfunction(
                                        dysfunction_id=did,
                                        category="circular_dependency",
                                        severity=min(1.0, concentration * reciprocity),
                                        detected_at=self.current_round,
                                        targets=clique,
                                        details=f"Echo chamber: concentration={concentration:.2f}, reciprocity={reciprocity:.2f}",
                                    ))
        return results

    def _detect_metabolic_waste(self) -> List[Dysfunction]:
        """Detect accumulated temp state that's outlived its purpose."""
        results: List[Dysfunction] = []
        for state_id, created_at in self.temp_state.items():
            age = self.current_round - created_at
            if age >= WASTE_AGE_THRESHOLD:
                severity = min(1.0, age / 20.0)
                results.append(Dysfunction(
                    dysfunction_id=f"waste_{state_id}_{self.current_round}",
                    category="metabolic_waste",
                    severity=severity,
                    detected_at=self.current_round,
                    targets=[state_id],
                    details=f"Temp state aged {age} rounds",
                ))
        return results

    def _detect_senescent_agents(self) -> List[Dysfunction]:
        """Detect agents with sustained performance decline."""
        results: List[Dysfunction] = []
        for agent_id, history in self.agent_history.items():
            if len(history) < SENESCENCE_WINDOW:
                continue
            recent = history[-SENESCENCE_WINDOW:]
            scores = [d.performance_score for _, d in recent]
            if len(scores) < 2:
                continue
            early_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
            late_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
            if early_avg > 0 and (early_avg - late_avg) / early_avg >= SENESCENCE_DECLINE_RATE:
                severity = min(1.0, (early_avg - late_avg) / early_avg)
                results.append(Dysfunction(
                    dysfunction_id=f"senescent_{agent_id}_{self.current_round}",
                    category="senescent_agent",
                    severity=severity,
                    detected_at=self.current_round,
                    targets=[agent_id],
                    details=f"Performance declined from {early_avg:.2f} to {late_avg:.2f}",
                ))
        return results

    def _detect_protein_misfolding(self) -> List[Dysfunction]:
        """Detect agents with conflicting/inconsistent beliefs."""
        results: List[Dysfunction] = []
        for agent_id, history in self.agent_history.items():
            if not history:
                continue
            _, latest = history[-1]
            if not latest.beliefs:
                continue
            # Check for contradictions: beliefs with conflicting values
            beliefs = latest.beliefs
            conflicts = 0
            total_pairs = 0
            keys = list(beliefs.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    total_pairs += 1
                    v1 = beliefs[keys[i]]
                    v2 = beliefs[keys[j]]
                    # Simple heuristic: if two beliefs are boolean opposites or
                    # numerically contradictory
                    if isinstance(v1, bool) and isinstance(v2, bool) and v1 != v2:
                        conflicts += 1
                    elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        if (v1 > 0 and v2 < 0) or (v1 < 0 and v2 > 0):
                            conflicts += 1
            if total_pairs > 0:
                conflict_ratio = conflicts / total_pairs
                if conflict_ratio >= MISFOLDING_CONFLICT_THRESHOLD:
                    results.append(Dysfunction(
                        dysfunction_id=f"misfolded_{agent_id}_{self.current_round}",
                        category="protein_misfolding",
                        severity=min(1.0, conflict_ratio),
                        detected_at=self.current_round,
                        targets=[agent_id],
                        details=f"Belief conflict ratio: {conflict_ratio:.2f} ({conflicts}/{total_pairs})",
                    ))
        return results

    def _detect_organelle_dysfunction(self) -> List[Dysfunction]:
        """Detect subsystems below functional threshold."""
        results: List[Dysfunction] = []
        for name, sub in self.subsystems.items():
            if sub.health < ORGANELLE_HEALTH_THRESHOLD:
                severity = 1.0 - sub.health  # lower health = higher severity
                results.append(Dysfunction(
                    dysfunction_id=f"organelle_{name}_{self.current_round}",
                    category="organelle_dysfunction",
                    severity=severity,
                    detected_at=self.current_round,
                    targets=[name],
                    details=f"Subsystem '{name}' health at {sub.health:.2f}",
                ))
        return results

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save engine state to JSON."""
        state = {
            "mode": self.mode,
            "cooldown_rounds": self.cooldown_rounds,
            "cooldown_remaining": self.cooldown_remaining,
            "stress_level": self.stress_level,
            "current_round": self.current_round,
            "agent_history": {
                aid: [(r, asdict(d)) for r, d in hist]
                for aid, hist in self.agent_history.items()
            },
            "memory_tracker": self.memory_tracker,
            "temp_state": self.temp_state,
            "subsystems": {n: asdict(s) for n, s in self.subsystems.items()},
            "dysfunctions": [asdict(d) for d in self.dysfunctions],
            "lysosome_queue": [asdict(d) for d in self.lysosome_queue],
            "recycling_ledger": [asdict(e) for e in self.recycling_ledger],
            "score_history": self.score_history,
            "interaction_graph": {
                k: dict(v) for k, v in self.interaction_graph.items()
            },
        }
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: str) -> "AutophagyEngine":
        """Load engine state from JSON."""
        data = json.loads(Path(path).read_text())
        engine = cls(
            mode=data["mode"],
            cooldown_rounds=data["cooldown_rounds"],
            stress_level=data["stress_level"],
        )
        engine.cooldown_remaining = data["cooldown_remaining"]
        engine.current_round = data["current_round"]

        for aid, hist in data.get("agent_history", {}).items():
            for r, d in hist:
                engine.agent_history[aid].append((r, AgentRoundData(**d)))

        engine.memory_tracker = data.get("memory_tracker", {})
        engine.temp_state = data.get("temp_state", {})

        for n, s in data.get("subsystems", {}).items():
            engine.subsystems[n] = SubsystemHealth(**s)

        engine.dysfunctions = [Dysfunction(**d) for d in data.get("dysfunctions", [])]
        engine.lysosome_queue = [Dysfunction(**d) for d in data.get("lysosome_queue", [])]
        engine.recycling_ledger = [RecycleEntry(**e) for e in data.get("recycling_ledger", [])]
        engine.score_history = data.get("score_history", [])

        for k, v in data.get("interaction_graph", {}).items():
            for k2, count in v.items():
                engine.interaction_graph[k][k2] = count

        return engine

    # -----------------------------------------------------------------------
    # HTML Dashboard
    # -----------------------------------------------------------------------

    def export_html(self, path: str) -> None:
        """Generate an interactive HTML dashboard."""
        report = self.get_report()
        dysfunction_json = json.dumps(report.dysfunction_counts)
        queue_rows = ""
        for item in self.lysosome_queue[:20]:
            status = "🔴 Tagged"
            if item.degraded:
                status = "🟡 Degraded"
            if item.recycled:
                status = "🟢 Recycled"
            queue_rows += f"""<tr>
                <td>{item.category}</td>
                <td>{', '.join(item.targets)}</td>
                <td>{item.severity:.2f}</td>
                <td>{status}</td>
                <td>{item.details}</td>
            </tr>"""

        ledger_rows = ""
        for entry in self.recycling_ledger[-20:]:
            ledger_rows += f"""<tr>
                <td>{entry.recycled_at}</td>
                <td>{entry.category}</td>
                <td>{', '.join(entry.targets)}</td>
                <td>{entry.resources_recovered:.1f}</td>
            </tr>"""

        sparkline_data = json.dumps(self.score_history[-50:])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Autophagy Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #0d1117; color: #c9d1d9; padding: 24px; }}
h1 {{ color: #58a6ff; margin-bottom: 8px; }}
.subtitle {{ color: #8b949e; margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }}
.card h3 {{ color: #58a6ff; margin-bottom: 12px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }}
.score {{ font-size: 48px; font-weight: bold; text-align: center; }}
.score.good {{ color: #3fb950; }}
.score.warn {{ color: #d29922; }}
.score.bad {{ color: #f85149; }}
.mode-badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 13px; }}
.mode-monitor {{ background: #1f6feb33; color: #58a6ff; }}
.mode-tag {{ background: #d2992233; color: #d29922; }}
.mode-degrade {{ background: #f8514933; color: #f85149; }}
.mode-recycle {{ background: #3fb95033; color: #3fb950; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; }}
th {{ color: #8b949e; font-weight: 600; }}
.sparkline {{ width: 100%; height: 40px; }}
.stat {{ text-align: center; }}
.stat-value {{ font-size: 28px; font-weight: bold; color: #58a6ff; }}
.stat-label {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
canvas {{ width: 100%; height: 150px; }}
</style>
</head>
<body>
<h1>🧬 Swarm Autophagy Dashboard</h1>
<p class="subtitle">Autonomous self-cleaning engine — Round {self.current_round}</p>

<div class="grid">
    <div class="card">
        <h3>Autophagy Score</h3>
        <div class="score {'good' if report.score >= 70 else 'warn' if report.score >= 40 else 'bad'}">{report.score:.0f}</div>
    </div>
    <div class="card">
        <h3>Mode</h3>
        <div style="text-align:center; margin-top:12px;">
            <span class="mode-badge mode-{self.mode}">{self.mode.upper()}</span>
        </div>
        <div class="stat" style="margin-top:16px;">
            <div class="stat-value">{report.stress_level:.0%}</div>
            <div class="stat-label">Stress Level</div>
        </div>
    </div>
    <div class="card">
        <h3>Statistics</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div class="stat"><div class="stat-value">{report.total_detected}</div><div class="stat-label">Detected</div></div>
            <div class="stat"><div class="stat-value">{report.total_recycled}</div><div class="stat-label">Recycled</div></div>
            <div class="stat"><div class="stat-value">{report.queue_depth}</div><div class="stat-label">Queue Depth</div></div>
            <div class="stat"><div class="stat-value">{report.cooldown_remaining}</div><div class="stat-label">Cooldown</div></div>
        </div>
    </div>
</div>

<div class="grid">
    <div class="card">
        <h3>Dysfunction Breakdown</h3>
        <canvas id="pieChart"></canvas>
    </div>
    <div class="card">
        <h3>Score Trend</h3>
        <canvas id="sparkChart"></canvas>
    </div>
</div>

<div class="card" style="margin-bottom:16px;">
    <h3>Lysosome Queue (Top 20)</h3>
    <table>
        <thead><tr><th>Category</th><th>Targets</th><th>Severity</th><th>Status</th><th>Details</th></tr></thead>
        <tbody>{queue_rows if queue_rows else '<tr><td colspan="5" style="color:#8b949e;">Queue empty — swarm is clean</td></tr>'}</tbody>
    </table>
</div>

<div class="card">
    <h3>Recycling Ledger (Recent)</h3>
    <table>
        <thead><tr><th>Round</th><th>Category</th><th>Targets</th><th>Resources</th></tr></thead>
        <tbody>{ledger_rows if ledger_rows else '<tr><td colspan="4" style="color:#8b949e;">No items recycled yet</td></tr>'}</tbody>
    </table>
</div>

<script>
const dysCounts = {dysfunction_json};
const sparkData = {sparkline_data};

// Pie chart
(function() {{
    const canvas = document.getElementById('pieChart');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = 300;
    const colors = ['#f85149','#d29922','#3fb950','#58a6ff','#bc8cff','#f778ba','#79c0ff'];
    const entries = Object.entries(dysCounts);
    const total = entries.reduce((s, e) => s + e[1], 0);
    if (total === 0) {{ ctx.fillStyle = '#8b949e'; ctx.font = '14px sans-serif'; ctx.fillText('No dysfunctions detected', 20, 30); return; }}
    let angle = 0;
    const cx = canvas.width / 4, cy = 150, r = 80;
    entries.forEach(([cat, count], i) => {{
        const slice = (count / total) * Math.PI * 2;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, r, angle, angle + slice); ctx.fillStyle = colors[i % colors.length]; ctx.fill();
        angle += slice;
    }});
    let ly = 20;
    entries.forEach(([cat, count], i) => {{
        ctx.fillStyle = colors[i % colors.length]; ctx.fillRect(cx * 2 + 20, ly, 12, 12);
        ctx.fillStyle = '#c9d1d9'; ctx.font = '11px sans-serif'; ctx.fillText(`${{cat}} (${{count}})`, cx * 2 + 40, ly + 10);
        ly += 20;
    }});
}})();

// Sparkline
(function() {{
    const canvas = document.getElementById('sparkChart');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = 300;
    if (sparkData.length < 2) {{ ctx.fillStyle = '#8b949e'; ctx.font = '14px sans-serif'; ctx.fillText('Insufficient data', 20, 30); return; }}
    const w = canvas.width, h = canvas.height, pad = 20;
    const xStep = (w - pad * 2) / (sparkData.length - 1);
    ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2; ctx.beginPath();
    sparkData.forEach((v, i) => {{
        const x = pad + i * xStep, y = h - pad - (v / 100) * (h - pad * 2);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.stroke();
}})();
</script>
</body>
</html>"""
        Path(path).write_text(html, encoding="utf-8")

    # -----------------------------------------------------------------------
    # CLI
    # -----------------------------------------------------------------------

    @classmethod
    def run_cli(cls, args: Optional[List[str]] = None) -> None:
        """Run the autophagy engine as a CLI simulation."""
        parser = argparse.ArgumentParser(
            description="Swarm Autophagy Engine — autonomous self-cleaning simulation"
        )
        parser.add_argument("--agents", type=int, default=10, help="Number of agents")
        parser.add_argument("--rounds", type=int, default=20, help="Simulation rounds")
        parser.add_argument("--mode", choices=MODES, default="recycle", help="Autophagy mode")
        parser.add_argument("--stress-level", type=float, default=0.3, help="Stress level 0-1")
        parser.add_argument("--cooldown", type=int, default=3, help="Cooldown rounds")
        parser.add_argument("--output", type=str, help="HTML report output path")
        parser.add_argument("--json", type=str, dest="json_out", help="JSON results path")
        parsed = parser.parse_args(args)

        engine = cls(
            mode=parsed.mode,
            cooldown_rounds=parsed.cooldown,
            stress_level=parsed.stress_level,
        )

        agent_ids = [f"agent_{i}" for i in range(parsed.agents)]
        memory_ids = [f"mem_{i}" for i in range(parsed.agents * 2)]
        temp_ids = [f"tmp_{i}" for i in range(parsed.agents)]
        subsystems = ["voting", "memory", "communication", "coordination"]

        print(f"🧬 Swarm Autophagy Simulation")
        print(f"   Agents: {parsed.agents} | Rounds: {parsed.rounds}")
        print(f"   Mode: {parsed.mode} | Stress: {parsed.stress_level}")
        print(f"{'─' * 60}")

        for round_num in range(1, parsed.rounds + 1):
            # Generate activity — some agents become dysfunctional
            activity: Dict[str, Dict[str, Any]] = {}
            for aid in agent_ids:
                idx = int(aid.split("_")[1])
                # Make some agents stale
                if idx % 5 == 0 and round_num > 5:
                    activity[aid] = {"votes": 0, "memory_accesses": 0, "state_updates": 0, "performance_score": 0.3}
                # Make some agents senescent (declining)
                elif idx % 7 == 0:
                    perf = max(0.1, 1.0 - (round_num * 0.05))
                    activity[aid] = {"votes": 1, "memory_accesses": 1, "state_updates": 1, "performance_score": perf}
                # Some with conflicting beliefs
                elif idx % 4 == 0 and round_num > 3:
                    activity[aid] = {
                        "votes": 2, "memory_accesses": 2, "state_updates": 1,
                        "performance_score": 0.8,
                        "beliefs": {"optimistic": True, "pessimistic": True, "growth": 5.0, "decline": -3.0, "stable": True},
                    }
                else:
                    activity[aid] = {
                        "votes": random.randint(1, 5),
                        "memory_accesses": random.randint(1, 8),
                        "state_updates": random.randint(0, 3),
                        "performance_score": random.uniform(0.7, 1.0),
                    }

            # Memory accesses — leave some as zombies
            mem_access = {}
            for mid in memory_ids:
                idx = int(mid.split("_")[1])
                mem_access[mid] = idx % 3 != 0 or round_num <= 3

            # Subsystem health — degrade one
            sub_health = {}
            for s in subsystems:
                if s == "communication" and round_num > 10:
                    sub_health[s] = max(0.1, 0.5 - round_num * 0.02)
                else:
                    sub_health[s] = random.uniform(0.6, 1.0)

            # Interactions — create echo chambers
            interactions: Dict[str, List[str]] = {}
            for aid in agent_ids:
                idx = int(aid.split("_")[1])
                if idx < 3:
                    # Agents 0,1,2 form a clique
                    others = [a for a in agent_ids[:3] if a != aid]
                    interactions[aid] = others * 3
                else:
                    interactions[aid] = [random.choice(agent_ids) for _ in range(2)]

            engine.record_round(
                round_num=round_num,
                agent_activity=activity,
                memory_accesses=mem_access,
                temp_state_ids=temp_ids if round_num == 1 else None,
                subsystem_health=sub_health,
                interactions=interactions,
            )

            # Detect & process
            new_dys = engine.detect()
            recycled = engine.process_queue()

            if new_dys or recycled:
                report = engine.get_report()
                print(f"  Round {round_num:>3} | Score: {report.score:5.1f} | "
                      f"Detected: {len(new_dys):>2} | Recycled: {len(recycled):>2} | "
                      f"Queue: {report.queue_depth:>3} | Mode: {report.mode}")

        # Final report
        report = engine.get_report()
        print(f"{'─' * 60}")
        print(f"📊 Final Report:")
        print(f"   Score: {report.score:.1f}/100")
        print(f"   Mode: {report.mode}")
        print(f"   Total detected: {report.total_detected}")
        print(f"   Total recycled: {report.total_recycled}")
        print(f"   Queue remaining: {report.queue_depth}")
        print(f"   Dysfunction breakdown:")
        for cat, count in sorted(report.dysfunction_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"     {cat}: {count}")

        if parsed.output:
            engine.export_html(parsed.output)
            print(f"\n   📄 HTML report: {parsed.output}")

        if parsed.json_out:
            result = {
                "report": asdict(report),
                "dysfunctions": [asdict(d) for d in engine.dysfunctions],
                "recycling_ledger": [asdict(e) for e in engine.recycling_ledger],
            }
            Path(parsed.json_out).write_text(json.dumps(result, indent=2))
            print(f"   📄 JSON output: {parsed.json_out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    AutophagyEngine.run_cli()
