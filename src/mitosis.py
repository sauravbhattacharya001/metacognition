"""Swarm Mitosis Engine — autonomous agent replication with cell-cycle phases.

Biologically inspired by eukaryotic cell division (mitosis).  Agents in a
swarm can replicate themselves when conditions are met, creating child agents
that inherit traits with small mutations.  The swarm autonomously manages its
population through cell-cycle checkpoints, growth factor signaling, contact
inhibition, and programmed cell death (apoptosis).

Capabilities:

- **Cell Cycle Phases** — agents progress through G0 (quiescent) → G1
  (growth) → S (DNA synthesis) → G2 (preparation) → M (mitosis) →
  cytokinesis, gated by checkpoint verification at each transition.
- **Growth Signal Engine** — diffusible growth factors accumulate in a
  shared pool; concentration above threshold triggers G0→G1 transition.
- **Checkpoint Verifier** — enforces 3 checkpoints (G1/S, G2/M, spindle)
  with biologically-inspired conditions (nutrient level, DNA integrity,
  growth factor concentration, synthesis completion).
- **Division Engine** — symmetric (two identical daughters) or asymmetric
  (one specialised, one stem-like) division with trait inheritance and
  gaussian mutation noise.
- **Contact Inhibition** — population-density-dependent division
  suppression; agents in crowded conditions enter G0 quiescence.
- **Apoptosis Engine** — programmed cell death triggered by telomere
  exhaustion, low DNA integrity, low fitness, or overcrowding lottery.
  Optional knowledge transfer to nearest neighbour on death.
- **Lineage Tracker** — parent→child family tree, generation depth,
  dominant/extinct lineage detection.
- **Insight Generator** — autonomous observations about population
  trends, dominant lineages, checkpoint failures, telomere crises, and
  carrying capacity pressure.
- **Health Score** — composite 0-100 metric from population sustainability,
  generation diversity, fitness distribution, checkpoint pass rate,
  telomere reserves, and growth balance.  5 tiers: Thriving / Stable /
  Stressed / Declining / Collapsing.
- **Interactive HTML Dashboard** — population timeline, generation
  distribution, lineage tree, phase breakdown, health gauge, insights.

Usage (Python API)::

    from src.mitosis import SwarmMitosisEngine

    engine = SwarmMitosisEngine(num_agents=10, carrying_capacity=50)
    report = engine.simulate(ticks=100)
    print(report.overall_health)
    print(report.insights)
    engine.export_html("mitosis_report.html")

CLI::

    python -m src.mitosis                           # demo
    python -m src.mitosis --agents 15               # custom initial pop
    python -m src.mitosis --capacity 30             # smaller carrying capacity
    python -m src.mitosis --ticks 120               # longer simulation
    python -m src.mitosis --scenario population_explosion
    python -m src.mitosis --scenario stem_cell
    python -m src.mitosis --scenario aging_crisis
    python -m src.mitosis --scenario bottleneck
    python -m src.mitosis --out report.html --json mitosis.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class CellCyclePhase(str, Enum):
    G0_QUIESCENT = "G0_QUIESCENT"
    G1_GROWTH = "G1_GROWTH"
    S_SYNTHESIS = "S_SYNTHESIS"
    G2_PREPARATION = "G2_PREPARATION"
    M_MITOSIS = "M_MITOSIS"
    CYTOKINESIS = "CYTOKINESIS"


@dataclass
class GrowthFactor:
    name: str
    potency: float = 0.5          # 0-1
    decay_rate: float = 0.05      # per tick
    concentration: float = 0.0
    source_agent: Optional[str] = None


@dataclass
class Checkpoint:
    name: str
    phase: str                    # which transition it guards
    conditions: List[str] = field(default_factory=list)
    passed: bool = False


@dataclass
class AgentCell:
    agent_id: str
    phase: CellCyclePhase = CellCyclePhase.G0_QUIESCENT
    generation: int = 0           # 0 = original founder
    parent_id: Optional[str] = None
    telomere_length: float = 100.0
    fitness: float = 70.0
    specialization: str = "general"
    division_count: int = 0
    max_divisions: int = 10       # Hayflick-like limit
    last_division_tick: Optional[int] = None
    nutrient_level: float = 50.0
    dna_integrity: float = 100.0
    checkpoints_passed: List[str] = field(default_factory=list)
    birth_tick: int = 0
    traits: Dict[str, float] = field(default_factory=lambda: {
        "growth_rate": 1.0,
        "resilience": 0.5,
        "adaptability": 0.5,
        "efficiency": 0.5,
    })
    # internal state for S-phase duration
    _s_phase_ticks: int = 0
    _s_phase_target: int = 2


@dataclass
class DivisionEvent:
    parent_id: str
    child_id: str
    tick: int
    generation: int               # child generation
    division_type: str = "symmetric"   # symmetric | asymmetric
    trait_mutations: Dict[str, float] = field(default_factory=dict)


@dataclass
class ApoptosisEvent:
    agent_id: str
    tick: int
    reason: str
    knowledge_transferred_to: Optional[str] = None


@dataclass
class MitosisSnapshot:
    tick: int
    population_size: int
    avg_fitness: float
    avg_generation: float
    division_events: int
    deaths: int
    growth_rate: float
    carrying_capacity_pressure: float   # 0-1
    health_score: float
    phase_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class MitosisReport:
    snapshots: List[MitosisSnapshot] = field(default_factory=list)
    total_divisions: int = 0
    total_deaths: int = 0
    lineage_tree: Dict[str, List[str]] = field(default_factory=dict)
    generation_distribution: Dict[int, int] = field(default_factory=dict)
    population_peak: int = 0
    overall_health: float = 0.0
    health_tier: str = "Stable"
    insights: List[str] = field(default_factory=list)
    division_events: List[DivisionEvent] = field(default_factory=list)
    apoptosis_events: List[ApoptosisEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SwarmMitosisEngine:
    """Autonomous agent replication with cell-cycle checkpoints."""

    def __init__(
        self,
        num_agents: int = 10,
        carrying_capacity: int = 50,
        max_divisions: int = 10,
        mutation_rate: float = 0.05,
        growth_factor_threshold: float = 1.5,
        telomere_loss_per_division: float = 8.0,
    ) -> None:
        self.carrying_capacity = carrying_capacity
        self.max_divisions = max_divisions
        self.mutation_rate = mutation_rate
        self.growth_factor_threshold = growth_factor_threshold
        self.telomere_loss_per_division = telomere_loss_per_division

        # State
        self._agents: Dict[str, AgentCell] = {}
        self._growth_factors: Dict[str, GrowthFactor] = {}
        self._total_growth_concentration: float = 0.0
        self._tick: int = 0
        self._snapshots: List[MitosisSnapshot] = []
        self._division_events: List[DivisionEvent] = []
        self._apoptosis_events: List[ApoptosisEvent] = []
        self._lineage: Dict[str, List[str]] = {}   # parent -> [children]
        self._next_id: int = num_agents
        self._checkpoint_attempts: int = 0
        self._checkpoint_passes: int = 0

        # Seed initial population
        for i in range(num_agents):
            aid = f"agent_{i}"
            cell = AgentCell(
                agent_id=aid,
                phase=CellCyclePhase.G0_QUIESCENT,
                generation=0,
                fitness=random.uniform(50, 90),
                nutrient_level=random.uniform(40, 70),
                dna_integrity=random.uniform(90, 100),
                max_divisions=max_divisions,
                traits={
                    "growth_rate": random.uniform(0.7, 1.3),
                    "resilience": random.uniform(0.3, 0.7),
                    "adaptability": random.uniform(0.3, 0.7),
                    "efficiency": random.uniform(0.3, 0.7),
                },
            )
            self._agents[aid] = cell
            self._lineage[aid] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tick_count(self) -> int:
        return self._tick

    @property
    def population(self) -> int:
        return len(self._agents)

    @property
    def agents(self) -> Dict[str, AgentCell]:
        return dict(self._agents)

    # ------------------------------------------------------------------
    # Growth Factors
    # ------------------------------------------------------------------

    def add_growth_factor(self, name: str, potency: float = 0.5,
                          decay_rate: float = 0.05,
                          source: Optional[str] = None) -> GrowthFactor:
        gf = GrowthFactor(name=name, potency=potency, decay_rate=decay_rate,
                          source_agent=source)
        self._growth_factors[name] = gf
        return gf

    def _decay_growth_factors(self) -> None:
        for gf in self._growth_factors.values():
            gf.concentration *= (1 - gf.decay_rate)

    def _agents_produce_growth_factors(self) -> None:
        if not self._growth_factors:
            # Create a default factor
            self.add_growth_factor("default_mitogen", potency=0.5)
        for cell in list(self._agents.values()):
            production = cell.fitness / 100.0 * cell.traits.get("growth_rate", 1.0) * 0.3
            for gf in self._growth_factors.values():
                gf.concentration += production * gf.potency
        # total concentration
        self._total_growth_concentration = sum(
            gf.concentration for gf in self._growth_factors.values()
        )

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _check_g1s(self, cell: AgentCell) -> bool:
        """G1/S checkpoint: nutrients, DNA integrity, growth factors."""
        self._checkpoint_attempts += 1
        ok = (
            cell.nutrient_level >= 40
            and cell.dna_integrity > 80
            and self._total_growth_concentration > self.growth_factor_threshold
            and cell.division_count < cell.max_divisions
        )
        if ok:
            self._checkpoint_passes += 1
            if "G1/S" not in cell.checkpoints_passed:
                cell.checkpoints_passed.append("G1/S")
        return ok

    def _check_g2m(self, cell: AgentCell) -> bool:
        """G2/M checkpoint: synthesis complete, adequate fitness."""
        self._checkpoint_attempts += 1
        ok = (
            cell._s_phase_ticks >= cell._s_phase_target
            and cell.dna_integrity > 70
            and cell.fitness > 30
        )
        if ok:
            self._checkpoint_passes += 1
            if "G2/M" not in cell.checkpoints_passed:
                cell.checkpoints_passed.append("G2/M")
        return ok

    def _check_spindle(self, cell: AgentCell) -> bool:
        """Spindle checkpoint: alignment OK (probabilistic)."""
        self._checkpoint_attempts += 1
        ok = random.random() < 0.95  # 95% pass rate
        if ok:
            self._checkpoint_passes += 1
            if "Spindle" not in cell.checkpoints_passed:
                cell.checkpoints_passed.append("Spindle")
        return ok

    # ------------------------------------------------------------------
    # Division
    # ------------------------------------------------------------------

    def _mutate_traits(self, parent_traits: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Inherit traits with gaussian mutation. Returns (child_traits, mutations)."""
        child: Dict[str, float] = {}
        mutations: Dict[str, float] = {}
        for k, v in parent_traits.items():
            noise = random.gauss(0, self.mutation_rate * max(v, 0.1))
            new_v = max(0.01, v + noise)
            child[k] = round(new_v, 4)
            mutations[k] = round(new_v - v, 4)
        return child, mutations

    def _divide(self, cell: AgentCell, division_type: str = "symmetric") -> Optional[DivisionEvent]:
        """Execute cell division. Returns event or None if blocked."""
        if len(self._agents) >= self.carrying_capacity:
            return None

        child_id = f"agent_{self._next_id}"
        self._next_id += 1
        child_gen = cell.generation + 1

        child_traits, mutations = self._mutate_traits(cell.traits)

        # Telomere shortening
        loss = random.uniform(
            self.telomere_loss_per_division * 0.8,
            self.telomere_loss_per_division * 1.2,
        )
        cell.telomere_length = max(0, cell.telomere_length - loss)
        child_telomere = cell.telomere_length  # daughter inherits shortened telomere

        if division_type == "asymmetric":
            child_spec = "stem"
            child_fitness = cell.fitness * 0.6
        else:
            child_spec = cell.specialization
            child_fitness = cell.fitness * random.uniform(0.85, 1.0)

        child_cell = AgentCell(
            agent_id=child_id,
            phase=CellCyclePhase.G0_QUIESCENT,
            generation=child_gen,
            parent_id=cell.agent_id,
            telomere_length=child_telomere,
            fitness=max(10, child_fitness),
            specialization=child_spec,
            division_count=0,
            max_divisions=cell.max_divisions,
            nutrient_level=cell.nutrient_level * 0.5,
            dna_integrity=random.uniform(90, 100),
            birth_tick=self._tick,
            traits=child_traits,
        )

        # Parent loses some resources
        cell.nutrient_level *= 0.5
        cell.division_count += 1
        cell.last_division_tick = self._tick
        cell.phase = CellCyclePhase.G0_QUIESCENT
        cell.checkpoints_passed = []
        cell._s_phase_ticks = 0

        # Register
        self._agents[child_id] = child_cell
        if cell.agent_id not in self._lineage:
            self._lineage[cell.agent_id] = []
        self._lineage[cell.agent_id].append(child_id)
        self._lineage[child_id] = []

        event = DivisionEvent(
            parent_id=cell.agent_id,
            child_id=child_id,
            tick=self._tick,
            generation=child_gen,
            division_type=division_type,
            trait_mutations=mutations,
        )
        self._division_events.append(event)
        return event

    def force_division(self, agent_id: str,
                       division_type: str = "symmetric") -> Optional[DivisionEvent]:
        """Force an agent to divide bypassing checkpoints."""
        cell = self._agents.get(agent_id)
        if cell is None:
            return None
        return self._divide(cell, division_type)

    # ------------------------------------------------------------------
    # Contact Inhibition
    # ------------------------------------------------------------------

    def _contact_inhibition_pressure(self) -> float:
        """0-1 pressure based on pop/capacity."""
        if self.carrying_capacity <= 0:
            return 1.0
        return min(1.0, len(self._agents) / self.carrying_capacity)

    # ------------------------------------------------------------------
    # Apoptosis
    # ------------------------------------------------------------------

    def _run_apoptosis(self) -> List[ApoptosisEvent]:
        events: List[ApoptosisEvent] = []
        to_remove: List[str] = []
        agents_list = list(self._agents.values())

        for cell in agents_list:
            reason = None
            if cell.telomere_length < 5:
                reason = "telomere_exhaustion"
            elif cell.dna_integrity < 30:
                reason = "dna_damage"
            elif cell.fitness < 15:
                reason = "low_fitness"

            # Overcrowding lottery
            if reason is None and len(self._agents) > self.carrying_capacity:
                if random.random() < 0.15:
                    reason = "overcrowding"

            if reason:
                to_remove.append(cell.agent_id)
                # Knowledge transfer
                transferred_to = None
                surviving = [a for a in self._agents if a != cell.agent_id and a not in to_remove]
                if surviving and cell.fitness > 20:
                    recipient_id = random.choice(surviving)
                    recipient = self._agents.get(recipient_id)
                    if recipient:
                        recipient.fitness = min(100, recipient.fitness + cell.fitness * 0.1)
                        transferred_to = recipient_id

                events.append(ApoptosisEvent(
                    agent_id=cell.agent_id,
                    tick=self._tick,
                    reason=reason,
                    knowledge_transferred_to=transferred_to,
                ))

        for aid in to_remove:
            self._agents.pop(aid, None)

        self._apoptosis_events.extend(events)
        return events

    def kill_agent(self, agent_id: str, reason: str = "manual") -> Optional[ApoptosisEvent]:
        cell = self._agents.pop(agent_id, None)
        if cell is None:
            return None
        ev = ApoptosisEvent(agent_id=agent_id, tick=self._tick, reason=reason)
        self._apoptosis_events.append(ev)
        return ev

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self) -> MitosisSnapshot:
        """Advance one cycle."""
        self._tick += 1
        tick_divisions = 0
        tick_deaths = 0
        prev_pop = len(self._agents)

        # 1. Decay & produce growth factors
        self._decay_growth_factors()
        self._agents_produce_growth_factors()

        pressure = self._contact_inhibition_pressure()

        # 2. Phase progression for each agent
        for cell in list(self._agents.values()):
            # Nutrient recovery (slow)
            cell.nutrient_level = min(100, cell.nutrient_level + random.uniform(2, 6))
            # Small DNA degradation
            cell.dna_integrity = max(0, cell.dna_integrity - random.uniform(0, 0.5))

            if cell.phase == CellCyclePhase.G0_QUIESCENT:
                # Check if growth factors trigger G1
                if self._total_growth_concentration > self.growth_factor_threshold:
                    # Contact inhibition may prevent entry
                    if random.random() > pressure * 0.7:
                        cell.phase = CellCyclePhase.G1_GROWTH

            elif cell.phase == CellCyclePhase.G1_GROWTH:
                if self._check_g1s(cell):
                    cell.phase = CellCyclePhase.S_SYNTHESIS
                    cell._s_phase_ticks = 0
                    cell._s_phase_target = random.randint(2, 3)

            elif cell.phase == CellCyclePhase.S_SYNTHESIS:
                cell._s_phase_ticks += 1
                if cell._s_phase_ticks >= cell._s_phase_target:
                    cell.phase = CellCyclePhase.G2_PREPARATION

            elif cell.phase == CellCyclePhase.G2_PREPARATION:
                if self._check_g2m(cell):
                    cell.phase = CellCyclePhase.M_MITOSIS

            elif cell.phase == CellCyclePhase.M_MITOSIS:
                if self._check_spindle(cell):
                    cell.phase = CellCyclePhase.CYTOKINESIS

            elif cell.phase == CellCyclePhase.CYTOKINESIS:
                div_type = "asymmetric" if random.random() < 0.15 else "symmetric"
                event = self._divide(cell, div_type)
                if event:
                    tick_divisions += 1

        # 3. Apoptosis
        deaths = self._run_apoptosis()
        tick_deaths = len(deaths)

        # 4. Snapshot
        cur_pop = len(self._agents)
        growth_rate = (cur_pop - prev_pop) / max(prev_pop, 1)

        phase_dist: Dict[str, int] = defaultdict(int)
        fitnesses: List[float] = []
        generations: List[float] = []
        for c in self._agents.values():
            phase_dist[c.phase.value] += 1
            fitnesses.append(c.fitness)
            generations.append(c.generation)

        snap = MitosisSnapshot(
            tick=self._tick,
            population_size=cur_pop,
            avg_fitness=statistics.mean(fitnesses) if fitnesses else 0,
            avg_generation=statistics.mean(generations) if generations else 0,
            division_events=tick_divisions,
            deaths=tick_deaths,
            growth_rate=growth_rate,
            carrying_capacity_pressure=pressure,
            health_score=self._compute_health_score(),
            phase_distribution=dict(phase_dist),
        )
        self._snapshots.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Health Score
    # ------------------------------------------------------------------

    def _compute_health_score(self) -> float:
        if not self._agents:
            return 0.0

        cells = list(self._agents.values())

        # Population sustainability (25%) - best near 50-80% of capacity
        pop_ratio = len(cells) / max(self.carrying_capacity, 1)
        if pop_ratio < 0.1:
            pop_score = pop_ratio * 10 * 100  # 0-100
        elif pop_ratio <= 0.8:
            pop_score = 100
        else:
            pop_score = max(0, 100 - (pop_ratio - 0.8) * 500)

        # Generation diversity (20%)
        gens = [c.generation for c in cells]
        unique_gens = len(set(gens))
        gen_score = min(100, unique_gens * 20)

        # Fitness distribution (20%)
        avg_fit = statistics.mean([c.fitness for c in cells])
        fit_score = min(100, avg_fit * 1.3)

        # Checkpoint pass rate (15%)
        if self._checkpoint_attempts > 0:
            cp_score = (self._checkpoint_passes / self._checkpoint_attempts) * 100
        else:
            cp_score = 50  # neutral

        # Telomere reserves (10%)
        avg_telo = statistics.mean([c.telomere_length for c in cells])
        telo_score = min(100, avg_telo)

        # Growth balance (10%) - divisions vs deaths
        recent_divs = sum(1 for e in self._division_events if e.tick > self._tick - 10)
        recent_deaths = sum(1 for e in self._apoptosis_events if e.tick > self._tick - 10)
        if recent_divs + recent_deaths > 0:
            balance = 1 - abs(recent_divs - recent_deaths) / (recent_divs + recent_deaths)
        else:
            balance = 0.5
        balance_score = balance * 100

        composite = (
            pop_score * 0.25
            + gen_score * 0.20
            + fit_score * 0.20
            + cp_score * 0.15
            + telo_score * 0.10
            + balance_score * 0.10
        )
        return round(min(100, max(0, composite)), 1)

    @staticmethod
    def _health_tier(score: float) -> str:
        if score >= 80:
            return "Thriving"
        if score >= 60:
            return "Stable"
        if score >= 40:
            return "Stressed"
        if score >= 20:
            return "Declining"
        return "Collapsing"

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    def get_lineage_tree(self) -> Dict[str, List[str]]:
        return dict(self._lineage)

    def get_agent(self, agent_id: str) -> Optional[AgentCell]:
        return self._agents.get(agent_id)

    def _count_descendants(self, agent_id: str) -> int:
        children = self._lineage.get(agent_id, [])
        total = len(children)
        for c in children:
            total += self._count_descendants(c)
        return total

    def _find_founders(self) -> List[str]:
        """Find original (generation-0) agents in lineage."""
        founders: Set[str] = set()
        for aid in self._lineage:
            # Walk up to root
            current = aid
            seen: Set[str] = set()
            while True:
                parent = None
                for p, children in self._lineage.items():
                    if current in children:
                        parent = p
                        break
                if parent is None or parent in seen:
                    founders.add(current)
                    break
                seen.add(current)
                current = parent
        return list(founders)

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------

    def _generate_insights(self) -> List[str]:
        insights: List[str] = []
        cells = list(self._agents.values())
        if not cells:
            insights.append("⚠️ Population extinct — all agents have died.")
            return insights

        pop = len(cells)
        pressure = self._contact_inhibition_pressure()

        # Population trend
        if len(self._snapshots) >= 5:
            recent = [s.population_size for s in self._snapshots[-5:]]
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                insights.append("📈 Population growing steadily over last 5 ticks.")
            elif all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                insights.append("📉 Population declining over last 5 ticks.")

        # Carrying capacity
        if pressure > 0.9:
            insights.append(f"🔴 Near carrying capacity ({pop}/{self.carrying_capacity}) — contact inhibition active.")
        elif pressure > 0.7:
            insights.append(f"🟡 Population at {pressure:.0%} of carrying capacity.")

        # Telomere crisis
        low_telo = [c for c in cells if c.telomere_length < 20]
        if len(low_telo) > len(cells) * 0.3:
            insights.append(f"🧬 Telomere crisis: {len(low_telo)}/{pop} agents have critically short telomeres.")

        # Dominant lineage
        founders = self._find_founders()
        if founders:
            desc_counts = [(f, self._count_descendants(f)) for f in founders]
            desc_counts.sort(key=lambda x: x[1], reverse=True)
            if desc_counts and desc_counts[0][1] > pop * 0.5:
                insights.append(f"👑 Dominant lineage: {desc_counts[0][0]} has {desc_counts[0][1]} descendants ({desc_counts[0][1]}/{pop}).")

        # Generation diversity
        gens = set(c.generation for c in cells)
        if len(gens) == 1:
            insights.append(f"⚠️ Low generation diversity — all agents at generation {list(gens)[0]}.")
        elif len(gens) >= 5:
            insights.append(f"🌈 High generation diversity: {len(gens)} distinct generations coexist.")

        # Checkpoint failures
        if self._checkpoint_attempts > 10:
            rate = self._checkpoint_passes / self._checkpoint_attempts
            if rate < 0.5:
                insights.append(f"🚫 High checkpoint failure rate ({rate:.0%}) — many divisions blocked.")

        # Fitness
        avg_fit = statistics.mean([c.fitness for c in cells])
        if avg_fit < 30:
            insights.append(f"⚠️ Low average fitness ({avg_fit:.1f}/100) — population quality declining.")
        elif avg_fit > 80:
            insights.append(f"💪 High average fitness ({avg_fit:.1f}/100) — strong population.")

        # Division exhaustion
        exhausted = [c for c in cells if c.division_count >= c.max_divisions]
        if len(exhausted) > len(cells) * 0.3:
            insights.append(f"🔒 {len(exhausted)}/{pop} agents have reached their Hayflick division limit.")

        return insights

    # ------------------------------------------------------------------
    # Analyze / Simulate
    # ------------------------------------------------------------------

    def simulate(self, ticks: int = 100) -> MitosisReport:
        for _ in range(ticks):
            self.tick()
        return self.analyze()

    def analyze(self) -> MitosisReport:
        cells = list(self._agents.values())
        gen_dist: Dict[int, int] = defaultdict(int)
        for c in cells:
            gen_dist[c.generation] += 1

        health = self._compute_health_score()
        return MitosisReport(
            snapshots=list(self._snapshots),
            total_divisions=len(self._division_events),
            total_deaths=len(self._apoptosis_events),
            lineage_tree=self.get_lineage_tree(),
            generation_distribution=dict(gen_dist),
            population_peak=max((s.population_size for s in self._snapshots), default=len(cells)),
            overall_health=health,
            health_tier=self._health_tier(health),
            insights=self._generate_insights(),
            division_events=list(self._division_events),
            apoptosis_events=list(self._apoptosis_events),
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self) -> Dict[str, Any]:
        report = self.analyze()
        def _ser(obj: Any) -> Any:
            if hasattr(obj, '__dict__'):
                d = {}
                for k, v in obj.__dict__.items():
                    if k.startswith('_'):
                        continue
                    d[k] = _ser(v)
                return d
            if isinstance(obj, list):
                return [_ser(i) for i in obj]
            if isinstance(obj, dict):
                return {str(k): _ser(v) for k, v in obj.items()}
            if isinstance(obj, Enum):
                return obj.value
            return obj
        return _ser(report)

    def export_html(self, path: str) -> str:
        report = self.analyze()
        h = html_mod.escape

        # Population timeline (ASCII-ish bar chart)
        max_pop = max((s.population_size for s in report.snapshots), default=1)
        pop_bars = []
        step = max(1, len(report.snapshots) // 40)
        for i in range(0, len(report.snapshots), step):
            s = report.snapshots[i]
            bar_w = int(s.population_size / max(max_pop, 1) * 200)
            color = "#4caf50" if s.health_score >= 60 else "#ff9800" if s.health_score >= 40 else "#f44336"
            pop_bars.append(
                f'<div style="display:flex;align-items:center;margin:1px 0">'
                f'<span style="width:35px;font-size:11px;color:#888">t{s.tick}</span>'
                f'<div style="height:12px;width:{bar_w}px;background:{color};border-radius:2px"></div>'
                f'<span style="font-size:11px;margin-left:4px">{s.population_size}</span></div>'
            )
        pop_html = "\n".join(pop_bars)

        # Generation distribution
        gen_bars = []
        max_gen_count = max(report.generation_distribution.values(), default=1)
        for gen in sorted(report.generation_distribution):
            cnt = report.generation_distribution[gen]
            w = int(cnt / max(max_gen_count, 1) * 150)
            gen_bars.append(
                f'<div style="display:flex;align-items:center;margin:2px 0">'
                f'<span style="width:50px;font-size:12px">Gen {gen}</span>'
                f'<div style="height:14px;width:{w}px;background:#2196f3;border-radius:2px"></div>'
                f'<span style="font-size:12px;margin-left:4px">{cnt}</span></div>'
            )
        gen_html = "\n".join(gen_bars) if gen_bars else "<p>No agents alive.</p>"

        # Phase distribution
        cells = list(self._agents.values())
        phase_counts: Dict[str, int] = defaultdict(int)
        for c in cells:
            phase_counts[c.phase.value] += 1
        phase_colors = {
            "G0_QUIESCENT": "#9e9e9e", "G1_GROWTH": "#4caf50",
            "S_SYNTHESIS": "#2196f3", "G2_PREPARATION": "#ff9800",
            "M_MITOSIS": "#e91e63", "CYTOKINESIS": "#9c27b0",
        }
        total_agents = max(len(cells), 1)
        phase_bars = []
        for ph, cnt in sorted(phase_counts.items()):
            w = int(cnt / total_agents * 200)
            col = phase_colors.get(ph, "#607d8b")
            phase_bars.append(
                f'<div style="display:flex;align-items:center;margin:2px 0">'
                f'<span style="width:120px;font-size:12px">{ph}</span>'
                f'<div style="height:14px;width:{w}px;background:{col};border-radius:2px"></div>'
                f'<span style="font-size:12px;margin-left:4px">{cnt}</span></div>'
            )
        phase_html = "\n".join(phase_bars) if phase_bars else "<p>No agents.</p>"

        # Health gauge
        tier = report.health_tier
        score = report.overall_health
        gauge_color = "#4caf50" if score >= 60 else "#ff9800" if score >= 40 else "#f44336"

        # Insights
        insights_html = "\n".join(f"<li>{h(ins)}</li>" for ins in report.insights) or "<li>No insights.</li>"

        # Lineage snippet (first 10 entries with children)
        lineage_lines = []
        shown = 0
        for parent, children in sorted(report.lineage_tree.items()):
            if children and shown < 15:
                lineage_lines.append(f"<li><b>{h(parent)}</b> → {', '.join(h(c) for c in children[:5])}"
                                     + (f" (+{len(children)-5} more)" if len(children) > 5 else "")
                                     + "</li>")
                shown += 1
        lineage_html = "\n".join(lineage_lines) if lineage_lines else "<li>No lineage recorded.</li>"

        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Swarm Mitosis Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:30px auto;padding:0 20px;background:#fafafa;color:#333}}
h1{{color:#1a237e}}h2{{color:#283593;border-bottom:2px solid #c5cae9;padding-bottom:6px}}
.card{{background:#fff;border-radius:8px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
.gauge{{display:inline-block;width:200px;height:24px;background:#e0e0e0;border-radius:12px;overflow:hidden}}
.gauge-fill{{height:100%;border-radius:12px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}}
.metric{{text-align:center;padding:10px}}.metric .val{{font-size:24px;font-weight:bold}}.metric .lbl{{font-size:12px;color:#666}}
</style></head><body>
<h1>🧬 Swarm Mitosis Report</h1>
<div class="card">
<div class="metrics">
<div class="metric"><div class="val" style="color:{gauge_color}">{score:.0f}</div><div class="lbl">Health Score ({tier})</div></div>
<div class="metric"><div class="val">{len(cells)}</div><div class="lbl">Population</div></div>
<div class="metric"><div class="val">{report.total_divisions}</div><div class="lbl">Total Divisions</div></div>
<div class="metric"><div class="val">{report.total_deaths}</div><div class="lbl">Total Deaths</div></div>
<div class="metric"><div class="val">{report.population_peak}</div><div class="lbl">Peak Population</div></div>
<div class="metric"><div class="val">{self._tick}</div><div class="lbl">Ticks</div></div>
</div>
<div style="margin-top:10px">
<span class="gauge"><span class="gauge-fill" style="width:{score}%;background:{gauge_color}"></span></span>
</div>
</div>

<h2>📊 Population Timeline</h2>
<div class="card">{pop_html}</div>

<h2>🧬 Generation Distribution</h2>
<div class="card">{gen_html}</div>

<h2>🔄 Cell Cycle Phase Distribution</h2>
<div class="card">{phase_html}</div>

<h2>🌳 Lineage Tree (top entries)</h2>
<div class="card"><ul>{lineage_html}</ul></div>

<h2>💡 Autonomous Insights</h2>
<div class="card"><ul>{insights_html}</ul></div>

<p style="text-align:center;color:#999;font-size:11px">
Generated by Swarm Mitosis Engine · metacognition framework</p>
</body></html>"""

        Path(path).write_text(html_content, encoding="utf-8")
        return html_content


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "default": dict(num_agents=10, carrying_capacity=50, max_divisions=10, ticks=80),
    "population_explosion": dict(num_agents=5, carrying_capacity=100, max_divisions=15, ticks=120,
                                  growth_factor_threshold=0.5),
    "stem_cell": dict(num_agents=8, carrying_capacity=40, max_divisions=20, ticks=100,
                       mutation_rate=0.02),
    "aging_crisis": dict(num_agents=15, carrying_capacity=50, max_divisions=4, ticks=80,
                          telomere_loss_per_division=15.0),
    "bottleneck": dict(num_agents=30, carrying_capacity=12, max_divisions=8, ticks=80),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Swarm Mitosis Engine — autonomous agent replication")
    parser.add_argument("--agents", type=int, default=None, help="Initial agent count")
    parser.add_argument("--capacity", type=int, default=None, help="Carrying capacity")
    parser.add_argument("--ticks", type=int, default=None, help="Simulation ticks")
    parser.add_argument("--max-divisions", type=int, default=None, help="Max divisions per agent")
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="default")
    parser.add_argument("--out", type=str, default=None, help="HTML report output path")
    parser.add_argument("--json", type=str, default=None, help="JSON output path")
    args = parser.parse_args(argv)

    cfg = dict(SCENARIOS[args.scenario])
    ticks = cfg.pop("ticks", 80)
    gf_thresh = cfg.pop("growth_factor_threshold", None)
    telo_loss = cfg.pop("telomere_loss_per_division", None)

    if args.agents is not None:
        cfg["num_agents"] = args.agents
    if args.capacity is not None:
        cfg["carrying_capacity"] = args.capacity
    if args.max_divisions is not None:
        cfg["max_divisions"] = args.max_divisions
    if args.mutation_rate is not None:
        cfg["mutation_rate"] = args.mutation_rate
    if args.ticks is not None:
        ticks = args.ticks

    engine_kwargs = cfg
    if gf_thresh is not None:
        engine_kwargs["growth_factor_threshold"] = gf_thresh
    if telo_loss is not None:
        engine_kwargs["telomere_loss_per_division"] = telo_loss

    engine = SwarmMitosisEngine(**engine_kwargs)
    print(f"🧬 Swarm Mitosis Engine — scenario: {args.scenario}")
    print(f"   Initial agents: {engine.population} | Capacity: {engine.carrying_capacity}")
    print(f"   Simulating {ticks} ticks...\n")

    report = engine.simulate(ticks)

    print(f"═══ Results ═══")
    print(f"  Population:    {engine.population} (peak {report.population_peak})")
    print(f"  Divisions:     {report.total_divisions}")
    print(f"  Deaths:        {report.total_deaths}")
    print(f"  Health:        {report.overall_health:.0f}/100 ({report.health_tier})")
    print(f"  Generations:   {dict(sorted(report.generation_distribution.items()))}")
    print()
    for ins in report.insights:
        print(f"  {ins}")

    if args.out:
        engine.export_html(args.out)
        print(f"\n📄 HTML report → {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(engine.export_json(), indent=2, default=str), encoding="utf-8")
        print(f"📄 JSON report → {args.json}")


if __name__ == "__main__":
    main()
