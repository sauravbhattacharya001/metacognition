"""Swarm Dreaming Engine — autonomous offline consolidation and hypothesis generation.

Biologically-inspired by how mammalian sleep consolidates memory and
generates novel solutions.  During idle periods, agents enter *dream states*
where they replay past consensus episodes, discover latent patterns,
recombine solution fragments into novel hypotheses, and pre-plan for
anticipated future scenarios.

Capabilities:

- **Dream Cycle Controller** — manages REM/NREM-like phases: replay,
  consolidation, creative recombination, and anticipatory planning.
- **Episode Replay** — selectively replays high-signal episodes with
  noise injection to stress-test learned patterns.
- **Pattern Consolidation** — compresses episodic memory into durable
  schemas (abstract rules) that survive forgetting curves.
- **Creative Recombination** — fragments successful solutions and
  recombines them to generate novel hypothesis candidates for unseen tasks.
- **Anticipatory Planning** — extrapolates from trends to pre-generate
  response strategies for predicted future scenarios.
- **Dream Journal** — records all dream outputs for introspection and
  auditing; dreams that prove useful boost consolidation weight.
- **Lucidity Score** — measures how well dream-generated hypotheses
  connect to reality; filters out noise vs. insight.
- **Interactive HTML Dashboard** — visualizes dream cycles, consolidated
  schemas, hypothesis quality, and anticipatory plans.

Usage (Python API)::

    from src.dreaming import SwarmDreamEngine, EpisodicMemory, Episode

    mem = EpisodicMemory()
    mem.record(Episode(task="Route optimization", outcome="success",
                       solution="greedy-then-refine", agents=5, rounds=3))
    mem.record(Episode(task="Load balancing", outcome="success",
                       solution="round-robin-adaptive", agents=4, rounds=2))
    mem.record(Episode(task="Conflict resolution", outcome="failure",
                       solution=None, agents=6, rounds=5))

    engine = SwarmDreamEngine(memory=mem)
    journal = engine.dream(cycles=3)

    print(journal.schemas)        # consolidated abstract rules
    print(journal.hypotheses)     # novel recombined solutions
    print(journal.anticipations)  # pre-planned strategies

    engine.export_html("dream_report.html")

CLI::

    python -m src.dreaming                          # demo with simulated history
    python -m src.dreaming --cycles 5               # more dream cycles
    python -m src.dreaming --out report.html --json dream.json
    python -m src.dreaming --lucidity-threshold 0.6 # filter low-quality dreams
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import html as html_mod
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A single consensus episode stored in memory."""
    task: str
    outcome: str  # "success" or "failure"
    solution: Optional[str]
    agents: int
    rounds: int
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def tokens(self) -> List[str]:
        """Tokenize task into lowercase words."""
        return [w.lower().strip("?.,!;:'\"()[]{}") for w in self.task.split() if len(w) > 3]


@dataclass
class Schema:
    """A consolidated abstract rule derived from multiple episodes."""
    rule: str
    confidence: float  # 0-1
    source_count: int
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    validations: int = 0  # how many times real outcomes matched this schema


@dataclass
class Hypothesis:
    """A novel solution candidate generated via creative recombination."""
    description: str
    source_fragments: List[str]
    lucidity: float  # 0-1, how grounded in reality
    novelty: float  # 0-1, how different from known solutions
    potential_tasks: List[str] = field(default_factory=list)


@dataclass
class Anticipation:
    """A pre-planned strategy for an anticipated future scenario."""
    scenario: str
    probability: float  # estimated likelihood 0-1
    strategy: str
    rationale: str
    preparation_steps: List[str] = field(default_factory=list)


@dataclass
class DreamPhase:
    """One phase within a dream cycle."""
    phase_type: str  # "replay", "consolidation", "recombination", "anticipation"
    duration_ms: float
    inputs_processed: int
    outputs_generated: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DreamCycle:
    """A complete dream cycle with all phases."""
    cycle_id: int
    phases: List[DreamPhase] = field(default_factory=list)
    schemas_produced: List[Schema] = field(default_factory=list)
    hypotheses_produced: List[Hypothesis] = field(default_factory=list)
    anticipations_produced: List[Anticipation] = field(default_factory=list)
    total_duration_ms: float = 0.0


@dataclass
class DreamJournal:
    """Complete record of all dream outputs."""
    cycles: List[DreamCycle] = field(default_factory=list)
    schemas: List[Schema] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    anticipations: List[Anticipation] = field(default_factory=list)
    overall_lucidity: float = 0.0
    total_duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Episodic Memory Store
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """Simple episodic memory bank for consensus episodes."""

    def __init__(self) -> None:
        self._episodes: List[Episode] = []

    def record(self, episode: Episode) -> None:
        self._episodes.append(episode)

    @property
    def episodes(self) -> List[Episode]:
        return list(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def successes(self) -> List[Episode]:
        return [e for e in self._episodes if e.outcome == "success"]

    def failures(self) -> List[Episode]:
        return [e for e in self._episodes if e.outcome == "failure"]

    def recent(self, n: int = 10) -> List[Episode]:
        return sorted(self._episodes, key=lambda e: e.timestamp, reverse=True)[:n]

    def by_tag(self, tag: str) -> List[Episode]:
        return [e for e in self._episodes if tag in e.tags]

    def token_frequency(self) -> Counter:
        c: Counter = Counter()
        for e in self._episodes:
            c.update(e.tokens())
        return c

    def save(self, path: str) -> None:
        data = [asdict(e) for e in self._episodes]
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "EpisodicMemory":
        mem = cls()
        data = json.loads(Path(path).read_text())
        for d in data:
            mem.record(Episode(**d))
        return mem


# ---------------------------------------------------------------------------
# Dream Engines
# ---------------------------------------------------------------------------

class ReplayEngine:
    """Replays episodes with noise injection to stress-test patterns."""

    def __init__(self, noise_level: float = 0.2):
        self.noise_level = noise_level

    def replay(self, episodes: List[Episode], rng: random.Random) -> List[Dict[str, Any]]:
        """Replay episodes, injecting noise to find robust vs fragile patterns."""
        replays = []
        for ep in episodes:
            # Perturb agent count and rounds
            noisy_agents = max(1, ep.agents + rng.randint(-2, 2))
            noisy_rounds = max(1, ep.rounds + rng.randint(-1, 1))
            # Would the outcome change?
            stability = 1.0 if (noisy_agents >= ep.agents - 1) else 0.5
            replays.append({
                "original_task": ep.task,
                "original_outcome": ep.outcome,
                "perturbed_agents": noisy_agents,
                "perturbed_rounds": noisy_rounds,
                "stability": stability,
                "robust": stability > 0.7,
            })
        return replays


class ConsolidationEngine:
    """Compresses episodic patterns into durable schemas."""

    def consolidate(self, episodes: List[Episode]) -> List[Schema]:
        """Extract abstract rules from episode clusters."""
        schemas: List[Schema] = []

        # Rule 1: Agent count thresholds for success
        successes = [e for e in episodes if e.outcome == "success"]
        failures = [e for e in episodes if e.outcome == "failure"]

        if successes:
            avg_success_agents = statistics.mean(e.agents for e in successes)
            schemas.append(Schema(
                rule=f"Tasks succeed more often with >= {avg_success_agents:.0f} agents",
                confidence=min(1.0, len(successes) / max(len(episodes), 1)),
                source_count=len(successes),
                tags=["agent-count", "threshold"],
            ))

        # Rule 2: Round efficiency
        if successes:
            avg_rounds = statistics.mean(e.rounds for e in successes)
            schemas.append(Schema(
                rule=f"Successful consensus typically resolves in {avg_rounds:.1f} rounds",
                confidence=min(1.0, len(successes) / max(len(episodes), 1)),
                source_count=len(successes),
                tags=["efficiency", "rounds"],
            ))

        # Rule 3: Failure patterns
        if failures:
            high_round_failures = [e for e in failures if e.rounds >= 4]
            if high_round_failures:
                schemas.append(Schema(
                    rule="Prolonged negotiations (4+ rounds) correlate with failure",
                    confidence=len(high_round_failures) / max(len(failures), 1),
                    source_count=len(high_round_failures),
                    tags=["failure-pattern", "timeout"],
                ))

        # Rule 4: Common task types and their outcomes
        token_freq = Counter()
        success_tokens: Counter = Counter()
        for e in episodes:
            tokens = e.tokens()
            token_freq.update(tokens)
            if e.outcome == "success":
                success_tokens.update(tokens)

        for token, count in token_freq.most_common(5):
            if count >= 3:
                success_rate = success_tokens.get(token, 0) / count
                schemas.append(Schema(
                    rule=f"Tasks involving '{token}' have {success_rate:.0%} success rate",
                    confidence=min(1.0, count / len(episodes)),
                    source_count=count,
                    tags=["task-type", token],
                ))

        return schemas


class RecombinationEngine:
    """Fragments and recombines solutions to generate novel hypotheses."""

    def recombine(self, episodes: List[Episode], rng: random.Random) -> List[Hypothesis]:
        """Generate novel hypotheses by combining solution fragments."""
        hypotheses: List[Hypothesis] = []
        solutions = [(e.task, e.solution) for e in episodes if e.solution]

        if len(solutions) < 2:
            return hypotheses

        # Fragment solutions into parts
        fragments: List[Tuple[str, str]] = []  # (source_task, fragment)
        for task, sol in solutions:
            parts = sol.replace("-", " ").replace("_", " ").split()
            for part in parts:
                if len(part) > 2:
                    fragments.append((task, part))

        # Recombine random pairs
        num_hypotheses = min(5, len(fragments) // 2)
        for _ in range(num_hypotheses):
            if len(fragments) < 2:
                break
            f1 = rng.choice(fragments)
            f2 = rng.choice(fragments)
            if f1[0] == f2[0]:
                continue  # skip same-source combinations

            combined = f"{f1[1]}-{f2[1]}"
            # Lucidity: higher if fragments come from successful episodes
            source_tasks = [f1[0], f2[0]]
            successful_sources = sum(
                1 for t, s in solutions if t in source_tasks
            )
            lucidity = successful_sources / max(len(source_tasks), 1)
            novelty = 1.0 - (1.0 if combined in [s for _, s in solutions] else 0.0)

            hypotheses.append(Hypothesis(
                description=f"Hybrid approach: {combined}",
                source_fragments=[f"{f1[0]}:{f1[1]}", f"{f2[0]}:{f2[1]}"],
                lucidity=lucidity,
                novelty=novelty,
                potential_tasks=[f"Tasks similar to {f1[0]} or {f2[0]}"],
            ))

        return hypotheses


class AnticipationEngine:
    """Extrapolates trends to pre-generate strategies for future scenarios."""

    def anticipate(self, episodes: List[Episode]) -> List[Anticipation]:
        """Generate anticipatory plans based on observed trends."""
        anticipations: List[Anticipation] = []

        if not episodes:
            return anticipations

        # Trend 1: Growing agent requirements
        recent = sorted(episodes, key=lambda e: e.timestamp)[-10:]
        if len(recent) >= 3:
            agent_trend = [e.agents for e in recent]
            if agent_trend[-1] > agent_trend[0]:
                anticipations.append(Anticipation(
                    scenario="Increasing agent requirements for consensus",
                    probability=0.7,
                    strategy="Pre-allocate additional agent capacity",
                    rationale=f"Agent count trending from {agent_trend[0]} to {agent_trend[-1]}",
                    preparation_steps=[
                        "Reserve standby agents",
                        "Lower activation threshold for backup pool",
                        "Monitor for diminishing returns on agent count",
                    ],
                ))

        # Trend 2: Increasing failure rate
        if len(recent) >= 5:
            recent_failures = sum(1 for e in recent if e.outcome == "failure")
            failure_rate = recent_failures / len(recent)
            if failure_rate > 0.4:
                anticipations.append(Anticipation(
                    scenario="Elevated failure rate may indicate systemic issue",
                    probability=failure_rate,
                    strategy="Trigger diagnostic mode and reduce task complexity",
                    rationale=f"Recent failure rate: {failure_rate:.0%}",
                    preparation_steps=[
                        "Split complex tasks into smaller subtasks",
                        "Increase consensus threshold temporarily",
                        "Run calibration round with known-good task",
                        "Check for byzantine agent infiltration",
                    ],
                ))

        # Trend 3: Task type evolution
        early_tokens = Counter()
        late_tokens = Counter()
        midpoint = len(episodes) // 2
        for e in episodes[:midpoint]:
            early_tokens.update(e.tokens())
        for e in episodes[midpoint:]:
            late_tokens.update(e.tokens())

        emerging = set(late_tokens.keys()) - set(early_tokens.keys())
        if emerging:
            top_emerging = sorted(emerging, key=lambda t: late_tokens[t], reverse=True)[:3]
            anticipations.append(Anticipation(
                scenario=f"New task domains emerging: {', '.join(top_emerging)}",
                probability=0.6,
                strategy="Build expertise in emerging domains proactively",
                rationale="New vocabulary appearing in recent tasks",
                preparation_steps=[
                    f"Study patterns for '{t}' tasks" for t in top_emerging
                ] + ["Adjust agent specialization"],
            ))

        # Trend 4: Round inflation
        if len(recent) >= 4:
            round_trend = [e.rounds for e in recent]
            avg_early = statistics.mean(round_trend[:len(round_trend)//2])
            avg_late = statistics.mean(round_trend[len(round_trend)//2:])
            if avg_late > avg_early * 1.3:
                anticipations.append(Anticipation(
                    scenario="Consensus reaching takes increasingly more rounds",
                    probability=0.65,
                    strategy="Implement fast-track protocols for simple tasks",
                    rationale=f"Average rounds grew from {avg_early:.1f} to {avg_late:.1f}",
                    preparation_steps=[
                        "Classify tasks by expected complexity",
                        "Use single-round fast-commit for low-complexity",
                        "Reserve multi-round for genuinely complex decisions",
                    ],
                ))

        return anticipations


# ---------------------------------------------------------------------------
# Main Dream Engine
# ---------------------------------------------------------------------------

class SwarmDreamEngine:
    """Orchestrates dream cycles: replay → consolidation → recombination → anticipation."""

    def __init__(
        self,
        memory: EpisodicMemory,
        noise_level: float = 0.2,
        lucidity_threshold: float = 0.4,
        seed: Optional[int] = None,
    ):
        self.memory = memory
        self.noise_level = noise_level
        self.lucidity_threshold = lucidity_threshold
        self.rng = random.Random(seed)
        self._replay = ReplayEngine(noise_level)
        self._consolidation = ConsolidationEngine()
        self._recombination = RecombinationEngine()
        self._anticipation = AnticipationEngine()
        self._journal: Optional[DreamJournal] = None

    def dream(self, cycles: int = 3) -> DreamJournal:
        """Run dream cycles and return the dream journal."""
        journal = DreamJournal()
        all_episodes = self.memory.episodes

        if not all_episodes:
            self._journal = journal
            return journal

        # Use perf_counter for sub-millisecond elapsed-time measurement. time.time()
        # has ~15.6 ms granularity on Windows, which collapsed every fast dream
        # phase to 0.0 ms and made total_duration_ms == 0 (see test_dreaming.py::
        # test_total_duration_positive). perf_counter is the right tool for
        # measuring elapsed wall time of code paths regardless of platform.
        start = time.perf_counter()

        for cycle_id in range(cycles):
            cycle = DreamCycle(cycle_id=cycle_id)

            # Phase 1: Replay
            t0 = time.perf_counter()
            # Select subset for replay (prioritize recent + high-signal)
            replay_set = self._select_replay_set(all_episodes)
            replays = self._replay.replay(replay_set, self.rng)
            phase_replay = DreamPhase(
                phase_type="replay",
                duration_ms=(time.perf_counter() - t0) * 1000,
                inputs_processed=len(replay_set),
                outputs_generated=len(replays),
                details={"robust_count": sum(1 for r in replays if r["robust"])},
            )
            cycle.phases.append(phase_replay)

            # Phase 2: Consolidation
            t0 = time.perf_counter()
            schemas = self._consolidation.consolidate(all_episodes)
            phase_consolidation = DreamPhase(
                phase_type="consolidation",
                duration_ms=(time.perf_counter() - t0) * 1000,
                inputs_processed=len(all_episodes),
                outputs_generated=len(schemas),
            )
            cycle.phases.append(phase_consolidation)
            cycle.schemas_produced = schemas

            # Phase 3: Creative Recombination
            t0 = time.perf_counter()
            hypotheses = self._recombination.recombine(
                self.memory.successes(), self.rng
            )
            # Filter by lucidity threshold
            hypotheses = [h for h in hypotheses if h.lucidity >= self.lucidity_threshold]
            phase_recombination = DreamPhase(
                phase_type="recombination",
                duration_ms=(time.perf_counter() - t0) * 1000,
                inputs_processed=len(self.memory.successes()),
                outputs_generated=len(hypotheses),
            )
            cycle.phases.append(phase_recombination)
            cycle.hypotheses_produced = hypotheses

            # Phase 4: Anticipatory Planning
            t0 = time.perf_counter()
            anticipations = self._anticipation.anticipate(all_episodes)
            phase_anticipation = DreamPhase(
                phase_type="anticipation",
                duration_ms=(time.perf_counter() - t0) * 1000,
                inputs_processed=len(all_episodes),
                outputs_generated=len(anticipations),
            )
            cycle.phases.append(phase_anticipation)
            cycle.anticipations_produced = anticipations

            cycle.total_duration_ms = sum(p.duration_ms for p in cycle.phases)
            journal.cycles.append(cycle)
            journal.schemas.extend(schemas)
            journal.hypotheses.extend(hypotheses)
            journal.anticipations.extend(anticipations)

        # Deduplicate schemas by rule text
        seen_rules: Set[str] = set()
        unique_schemas: List[Schema] = []
        for s in journal.schemas:
            if s.rule not in seen_rules:
                seen_rules.add(s.rule)
                unique_schemas.append(s)
        journal.schemas = unique_schemas

        # Deduplicate anticipations by scenario
        seen_scenarios: Set[str] = set()
        unique_anticipations: List[Anticipation] = []
        for a in journal.anticipations:
            if a.scenario not in seen_scenarios:
                seen_scenarios.add(a.scenario)
                unique_anticipations.append(a)
        journal.anticipations = unique_anticipations

        # Compute overall lucidity
        if journal.hypotheses:
            journal.overall_lucidity = statistics.mean(
                h.lucidity for h in journal.hypotheses
            )

        journal.total_duration_ms = (time.perf_counter() - start) * 1000
        self._journal = journal
        return journal

    def _select_replay_set(self, episodes: List[Episode]) -> List[Episode]:
        """Select episodes for replay: mix of recent, failures, and random."""
        selected: List[Episode] = []
        n = min(len(episodes), 10)

        # Recent episodes (memory recency effect)
        recent = sorted(episodes, key=lambda e: e.timestamp, reverse=True)[:n // 3 + 1]
        selected.extend(recent)

        # Failures (unresolved problems get replayed)
        failures = [e for e in episodes if e.outcome == "failure"]
        selected.extend(self.rng.sample(failures, min(n // 3, len(failures))))

        # Random (creativity boost)
        remaining = [e for e in episodes if e not in selected]
        if remaining:
            selected.extend(self.rng.sample(remaining, min(n // 3, len(remaining))))

        return selected

    def validate_hypothesis(self, hypothesis: Hypothesis, real_outcome: str) -> bool:
        """Validate a dream hypothesis against a real outcome."""
        matched = real_outcome.lower() in hypothesis.description.lower()
        if matched:
            # Boost schemas that led to validated hypotheses
            for schema in (self._journal.schemas if self._journal else []):
                schema.validations += 1
        return matched

    def export_json(self, path: str) -> None:
        """Export dream journal to JSON."""
        if not self._journal:
            return
        data = {
            "cycles": len(self._journal.cycles),
            "schemas": [asdict(s) for s in self._journal.schemas],
            "hypotheses": [asdict(h) for h in self._journal.hypotheses],
            "anticipations": [asdict(a) for a in self._journal.anticipations],
            "overall_lucidity": self._journal.overall_lucidity,
            "total_duration_ms": self._journal.total_duration_ms,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def export_html(self, path: str) -> None:
        """Generate interactive HTML dashboard for dream analysis."""
        if not self._journal:
            return
        journal = self._journal

        # Build HTML
        schemas_rows = ""
        for s in journal.schemas:
            tag_badges = " ".join(
                f'<span class="badge">{html_mod.escape(t)}</span>' for t in s.tags
            )
            schemas_rows += f"""<tr>
                <td>{html_mod.escape(s.rule)}</td>
                <td>{s.confidence:.0%}</td>
                <td>{s.source_count}</td>
                <td>{s.validations}</td>
                <td>{tag_badges}</td>
            </tr>"""

        hypotheses_cards = ""
        for h in journal.hypotheses:
            sources = ", ".join(html_mod.escape(f) for f in h.source_fragments)
            tasks = ", ".join(html_mod.escape(t) for t in h.potential_tasks)
            hypotheses_cards += f"""<div class="card">
                <h4>{html_mod.escape(h.description)}</h4>
                <p><strong>Lucidity:</strong> {h.lucidity:.0%} | <strong>Novelty:</strong> {h.novelty:.0%}</p>
                <p><strong>Sources:</strong> {sources}</p>
                <p><strong>Potential tasks:</strong> {tasks}</p>
            </div>"""

        anticipation_cards = ""
        for a in journal.anticipations:
            steps = "".join(f"<li>{html_mod.escape(s)}</li>" for s in a.preparation_steps)
            anticipation_cards += f"""<div class="card">
                <h4>{html_mod.escape(a.scenario)}</h4>
                <p><strong>Probability:</strong> {a.probability:.0%}</p>
                <p><strong>Strategy:</strong> {html_mod.escape(a.strategy)}</p>
                <p><strong>Rationale:</strong> {html_mod.escape(a.rationale)}</p>
                <ul>{steps}</ul>
            </div>"""

        # Cycle timeline data for chart
        cycle_labels = [f"Cycle {c.cycle_id}" for c in journal.cycles]
        cycle_durations = [c.total_duration_ms for c in journal.cycles]
        cycle_schemas = [len(c.schemas_produced) for c in journal.cycles]
        cycle_hypotheses = [len(c.hypotheses_produced) for c in journal.cycles]

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Swarm Dream Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root {{ --bg: #0a0a1a; --card: #12122a; --accent: #7c4dff; --text: #e0e0e0; --dim: #888; }}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; }}
h1 {{ color: var(--accent); margin-bottom: 0.5rem; }}
h2 {{ color: var(--accent); margin: 2rem 0 1rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }}
h3 {{ color: #aaa; margin: 1rem 0 0.5rem; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
.stat {{ background: var(--card); border-radius: 12px; padding: 1.5rem; text-align: center; }}
.stat .value {{ font-size: 2rem; font-weight: bold; color: var(--accent); }}
.stat .label {{ color: var(--dim); font-size: 0.9rem; margin-top: 0.3rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
th, td {{ padding: 0.7rem; text-align: left; border-bottom: 1px solid #222; }}
th {{ color: var(--accent); font-weight: 600; }}
.badge {{ background: #2a2a4a; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; margin-right: 4px; }}
.card {{ background: var(--card); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }}
.card h4 {{ color: var(--accent); margin-bottom: 0.5rem; }}
.chart-container {{ background: var(--card); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; max-width: 700px; }}
ul {{ margin-left: 1.5rem; }}
li {{ margin: 0.3rem 0; }}
</style>
</head>
<body>
<h1>🌙 Swarm Dream Report</h1>
<p style="color: var(--dim);">Autonomous offline consolidation and hypothesis generation</p>

<div class="summary">
    <div class="stat"><div class="value">{len(journal.cycles)}</div><div class="label">Dream Cycles</div></div>
    <div class="stat"><div class="value">{len(journal.schemas)}</div><div class="label">Schemas Consolidated</div></div>
    <div class="stat"><div class="value">{len(journal.hypotheses)}</div><div class="label">Hypotheses Generated</div></div>
    <div class="stat"><div class="value">{len(journal.anticipations)}</div><div class="label">Anticipations</div></div>
    <div class="stat"><div class="value">{journal.overall_lucidity:.0%}</div><div class="label">Overall Lucidity</div></div>
    <div class="stat"><div class="value">{journal.total_duration_ms:.1f}ms</div><div class="label">Total Duration</div></div>
</div>

<h2>📊 Dream Cycle Overview</h2>
<div class="chart-container">
<canvas id="cycleChart" height="200"></canvas>
</div>

<h2>🧠 Consolidated Schemas</h2>
<table>
<tr><th>Rule</th><th>Confidence</th><th>Sources</th><th>Validations</th><th>Tags</th></tr>
{schemas_rows}
</table>

<h2>💡 Novel Hypotheses</h2>
{hypotheses_cards if hypotheses_cards else '<p style="color: var(--dim);">No hypotheses passed lucidity threshold.</p>'}

<h2>🔮 Anticipatory Plans</h2>
{anticipation_cards if anticipation_cards else '<p style="color: var(--dim);">Insufficient data for anticipations.</p>'}

<script>
new Chart(document.getElementById('cycleChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(cycle_labels)},
        datasets: [
            {{ label: 'Schemas', data: {json.dumps(cycle_schemas)}, backgroundColor: '#7c4dff88' }},
            {{ label: 'Hypotheses', data: {json.dumps(cycle_hypotheses)}, backgroundColor: '#ff4d9f88' }},
        ]
    }},
    options: {{
        responsive: true,
        scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#888' }} }}, x: {{ ticks: {{ color: '#888' }} }} }},
        plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }}
    }}
}});
</script>
</body>
</html>"""
        Path(path).write_text(html_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def _generate_demo_memory(rng: random.Random) -> EpisodicMemory:
    """Generate simulated episodic memory for demonstration."""
    mem = EpisodicMemory()
    tasks = [
        ("Route optimization for delivery fleet", "greedy-then-refine"),
        ("Load balancing across 12 nodes", "round-robin-adaptive"),
        ("Conflict resolution between agents A and B", None),
        ("Resource allocation for project X", "priority-weighted-split"),
        ("Anomaly detection in sensor stream", "sliding-window-zscore"),
        ("Schedule optimization for 5 teams", "constraint-propagation"),
        ("Data deduplication across shards", "hash-merge-compact"),
        ("Priority queue rebalancing", "heap-restructure"),
        ("Consensus on merge strategy", "vote-weighted-majority"),
        ("Risk assessment for deployment", None),
        ("Cache invalidation policy update", "ttl-adaptive"),
        ("Network partition recovery", "quorum-based-reconcile"),
        ("Task delegation to specialists", "skill-match-auction"),
        ("Budget allocation Q3", "proportional-performance"),
        ("Incident triage and escalation", None),
        ("Feature flag rollout decision", "gradual-canary-expand"),
        ("Database migration coordination", "blue-green-switch"),
        ("Alert fatigue reduction", "smart-dedup-cooldown"),
        ("Capacity planning for growth", "trend-extrapolate-buffer"),
        ("Security patch prioritization", "cvss-age-weighted"),
    ]

    base_time = time.time() - 86400 * 7  # 7 days ago

    for i, (task, solution) in enumerate(tasks):
        outcome = "success" if solution else "failure"
        agents = rng.randint(3, 8)
        rounds = rng.randint(1, 5) if outcome == "success" else rng.randint(3, 6)
        mem.record(Episode(
            task=task,
            outcome=outcome,
            solution=solution,
            agents=agents,
            rounds=rounds,
            timestamp=base_time + i * 3600,
            tags=[task.split()[0].lower()],
        ))

    return mem


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Swarm Dreaming Engine")
    parser.add_argument("--cycles", type=int, default=3, help="Number of dream cycles")
    parser.add_argument("--out", type=str, default=None, help="HTML output path")
    parser.add_argument("--json", type=str, default=None, help="JSON output path")
    parser.add_argument("--lucidity-threshold", type=float, default=0.4,
                        help="Minimum lucidity for hypothesis acceptance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load", type=str, default=None, help="Load memory from JSON file")
    args = parser.parse_args()

    print("🌙 Swarm Dreaming Engine")
    print("=" * 60)

    rng = random.Random(args.seed)

    if args.load:
        mem = EpisodicMemory.load(args.load)
        print(f"📂 Loaded {len(mem)} episodes from {args.load}")
    else:
        mem = _generate_demo_memory(rng)
        print(f"🎲 Generated demo memory with {len(mem)} episodes")

    engine = SwarmDreamEngine(
        memory=mem,
        lucidity_threshold=args.lucidity_threshold,
        seed=args.seed,
    )

    print(f"\n💤 Entering dream state ({args.cycles} cycles)...")
    journal = engine.dream(cycles=args.cycles)

    print(f"\n✨ Dream complete! Duration: {journal.total_duration_ms:.1f}ms")
    print(f"   Overall Lucidity: {journal.overall_lucidity:.0%}")

    print(f"\n🧠 Consolidated Schemas ({len(journal.schemas)}):")
    print("-" * 50)
    for s in journal.schemas:
        print(f"  [{s.confidence:.0%}] {s.rule}")

    print(f"\n💡 Novel Hypotheses ({len(journal.hypotheses)}):")
    print("-" * 50)
    for h in journal.hypotheses:
        print(f"  [{h.lucidity:.0%} lucid, {h.novelty:.0%} novel] {h.description}")

    print(f"\n🔮 Anticipatory Plans ({len(journal.anticipations)}):")
    print("-" * 50)
    for a in journal.anticipations:
        print(f"  [{a.probability:.0%}] {a.scenario}")
        print(f"       Strategy: {a.strategy}")

    if args.out:
        engine.export_html(args.out)
        print(f"\n📄 HTML report: {args.out}")

    if args.json:
        engine.export_json(args.json)
        print(f"📋 JSON output: {args.json}")


if __name__ == "__main__":
    asyncio.run(_main())
