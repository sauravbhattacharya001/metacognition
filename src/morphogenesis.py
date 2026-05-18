"""Swarm Morphogenesis Engine — autonomous structural self-organization.

Biologically-inspired by embryonic development (morphogenesis), where cells
differentiate into specialized types guided by chemical gradients (morphogens),
positional information, and local signaling.  Agents in a swarm autonomously
self-organize into functional structures through:

- **Morphogen Gradient Fields** — diffusible signals that create positional
  information, enabling agents to infer their role based on local concentration.
- **Cell Fate Determination** — agents differentiate into specialized types
  (leader, relay, worker, sensor, memory, effector) based on morphogen
  exposure history and competence windows.
- **Developmental Stages** — swarm progresses through 6 stages: zygote,
  cleavage, gastrulation, organogenesis, maturation, homeostasis.
- **Induction Signaling** — differentiated agents emit secondary signals that
  recruit neighbors into complementary fates (lateral inhibition, induction).
- **Pattern Formation** — emergent spatial patterns (stripes, spots, gradients)
  from Turing-style reaction-diffusion dynamics.
- **Apoptosis & Pruning** — removal of misplaced or redundant agents to
  refine structure.
- **Regeneration** — detection of structural damage and autonomous repair
  through de-differentiation and re-patterning.
- **Developmental Health Score** — composite 0-100 measuring differentiation
  completeness, pattern regularity, stage progression, and structural integrity.
- **Interactive HTML Dashboard** — visualizes morphogen fields, fate maps,
  stage timeline, pattern snapshots, and health metrics.

Usage (Python API)::

    from src.morphogenesis import MorphogenesisEngine, CellFate

    engine = MorphogenesisEngine(grid_size=15, num_agents=20)

    # Place morphogen sources (organizers)
    engine.add_organizer(x=7, y=7, morphogen="activator", strength=2.0)
    engine.add_organizer(x=0, y=0, morphogen="inhibitor", strength=1.5)

    # Run developmental simulation
    engine.develop(steps=100)

    # Inspect results
    report = engine.analyze()
    print(report.stage)             # current developmental stage
    print(report.fate_map)          # agent -> CellFate assignments
    print(report.pattern_type)      # detected pattern (stripes/spots/gradient)
    print(report.health_score)      # 0-100 developmental health
    print(report.regeneration_events)  # any repair events

    engine.export_html("morphogenesis_report.html")

CLI::

    python -m src.morphogenesis                         # default demo
    python -m src.morphogenesis --grid 20 --agents 30   # larger swarm
    python -m src.morphogenesis --steps 200             # longer development
    python -m src.morphogenesis --damage 5              # test regeneration
    python -m src.morphogenesis --out report.html --json state.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Enums ────────────────────────────────────────────────────────────────


class CellFate(str, Enum):
    """Agent specialization types inspired by biological cell fates."""

    UNDIFFERENTIATED = "undifferentiated"
    LEADER = "leader"          # Organizer/signaling center
    RELAY = "relay"            # Signal amplifier/transmitter
    WORKER = "worker"          # Task executor
    SENSOR = "sensor"          # Environment monitor
    MEMORY = "memory"          # Knowledge store
    EFFECTOR = "effector"      # Action executor


class MorphogenType(str, Enum):
    """Types of morphogen signals."""

    ACTIVATOR = "activator"        # Promotes differentiation
    INHIBITOR = "inhibitor"        # Suppresses differentiation
    POSITIONAL = "positional"      # Encodes spatial information
    INDUCTIVE = "inductive"        # Recruits neighbors to complementary fates
    APOPTOTIC = "apoptotic"        # Signals for pruning
    REGENERATIVE = "regenerative"  # Triggers repair


class DevelopmentalStage(str, Enum):
    """Stages of swarm morphogenesis."""

    ZYGOTE = "zygote"              # Single undifferentiated mass
    CLEAVAGE = "cleavage"          # Rapid division/expansion
    GASTRULATION = "gastrulation"  # Layer formation, initial patterning
    ORGANOGENESIS = "organogenesis"  # Specialized structures emerge
    MATURATION = "maturation"      # Refinement and pruning
    HOMEOSTASIS = "homeostasis"    # Stable self-maintaining state


class PatternType(str, Enum):
    """Emergent spatial pattern categories."""

    UNIFORM = "uniform"
    GRADIENT = "gradient"
    STRIPES = "stripes"
    SPOTS = "spots"
    CLUSTERS = "clusters"
    MIXED = "mixed"


# ── Data Classes ─────────────────────────────────────────────────────────


@dataclass
class MorphogenSource:
    """A source (organizer) that emits morphogens."""

    x: float
    y: float
    morphogen: MorphogenType
    strength: float
    decay_rate: float = 0.1  # How quickly signal falls with distance
    active: bool = True


@dataclass
class AgentCell:
    """An agent undergoing morphogenesis."""

    agent_id: str
    x: float
    y: float
    fate: CellFate = CellFate.UNDIFFERENTIATED
    exposure_history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    competence_window: Tuple[int, int] = (0, 50)  # Steps during which fate can be set
    differentiation_step: Optional[int] = None
    signals_emitted: int = 0
    health: float = 1.0
    generation: int = 0


@dataclass
class InductionEvent:
    """Record of one agent inducing fate in another."""

    source_id: str
    target_id: str
    induced_fate: CellFate
    step: int
    signal_strength: float


@dataclass
class ApoptosisEvent:
    """Record of an agent being pruned."""

    agent_id: str
    reason: str
    step: int
    position: Tuple[float, float]


@dataclass
class RegenerationEvent:
    """Record of structural repair."""

    damaged_agents: List[str]
    repaired_agents: List[str]
    step: int
    repair_type: str  # "redifferentiation" or "recruitment"


@dataclass
class MorphogenesisReport:
    """Full analysis report."""

    stage: DevelopmentalStage
    step: int
    total_agents: int
    fate_map: Dict[str, str]
    fate_counts: Dict[str, int]
    pattern_type: PatternType
    pattern_regularity: float  # 0-1
    morphogen_field_summary: Dict[str, float]
    induction_events: List[Dict[str, Any]]
    apoptosis_events: List[Dict[str, Any]]
    regeneration_events: List[Dict[str, Any]]
    health_score: float  # 0-100
    health_breakdown: Dict[str, float]
    stage_history: List[Dict[str, Any]]
    insights: List[str]


# ── Engine ───────────────────────────────────────────────────────────────


class MorphogenesisEngine:
    """Autonomous swarm morphogenesis through gradient-based self-organization."""

    def __init__(
        self,
        grid_size: int = 15,
        num_agents: int = 20,
        diffusion_rate: float = 0.3,
        noise_level: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.diffusion_rate = diffusion_rate
        self.noise_level = noise_level
        self.rng = random.Random(seed)

        # State
        self.agents: Dict[str, AgentCell] = {}
        self.organizers: List[MorphogenSource] = []
        self.morphogen_field: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.step: int = 0
        self.stage: DevelopmentalStage = DevelopmentalStage.ZYGOTE
        self.stage_history: List[Dict[str, Any]] = [
            {"stage": DevelopmentalStage.ZYGOTE.value, "step": 0}
        ]
        self.induction_events: List[InductionEvent] = []
        self.apoptosis_events: List[ApoptosisEvent] = []
        self.regeneration_events: List[RegenerationEvent] = []

        # Initialize agents in random positions
        self._spawn_agents()

    def _spawn_agents(self) -> None:
        """Place initial agents randomly on the grid."""
        for i in range(self.num_agents):
            aid = f"cell-{i:03d}"
            x = self.rng.uniform(0, self.grid_size - 1)
            y = self.rng.uniform(0, self.grid_size - 1)
            # Competence windows vary per agent (biological stochasticity)
            start = self.rng.randint(5, 20)
            end = start + self.rng.randint(20, 40)
            self.agents[aid] = AgentCell(
                agent_id=aid,
                x=x,
                y=y,
                competence_window=(start, end),
            )

    def add_organizer(
        self,
        x: float,
        y: float,
        morphogen: str = "activator",
        strength: float = 2.0,
        decay_rate: float = 0.1,
    ) -> None:
        """Add a morphogen-emitting organizer (signaling center)."""
        mtype = MorphogenType(morphogen)
        self.organizers.append(
            MorphogenSource(x=x, y=y, morphogen=mtype, strength=strength, decay_rate=decay_rate)
        )
        # Invalidate cached base field so it's recomputed on next tick
        self._base_field_cache: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None

    def develop(self, steps: int = 100) -> None:
        """Run the developmental simulation for N steps."""
        for _ in range(steps):
            self._tick()

    def _tick(self) -> None:
        """Advance one developmental step."""
        self.step += 1

        # 1. Diffuse morphogens from organizers
        self._diffuse_morphogens()

        # 2. Agents sense local morphogen concentrations
        self._agents_sense()

        # 3. Determine cell fates for agents in competence window
        self._determine_fates()

        # 4. Lateral inhibition / induction signaling
        self._induction_signaling()

        # 5. Apoptosis check (remove misplaced agents)
        self._apoptosis_check()

        # 6. Update developmental stage
        self._update_stage()

    def _compute_base_field(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Precompute noise-free morphogen concentrations from all active sources.

        This is O(organizers × grid_size²) with sqrt + exp per cell, but the
        result is constant as long as organizers don't change.  Caching it
        eliminates redundant recomputation on every tick.
        """
        base: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for source in self.organizers:
            if not source.active:
                continue
            sx, sy, strength, decay = source.x, source.y, source.strength, source.decay_rate
            mtype_val = source.morphogen.value
            for gx in range(self.grid_size):
                dx = gx - sx
                for gy in range(self.grid_size):
                    dy = gy - sy
                    dist = math.sqrt(dx * dx + dy * dy)
                    base[(gx, gy)][mtype_val] += strength * math.exp(-decay * dist)
        return base

    def _diffuse_morphogens(self) -> None:
        """Compute morphogen concentrations from cached base field + per-tick noise.

        Previously recomputed O(organizers × grid²) sqrt/exp every tick.
        Now uses a lazily-built base field cache, adding only Gaussian noise
        per cell per tick — reducing per-tick cost to O(grid²) additions.
        """
        # Lazily compute/reuse the base field (invalidated when organizers change)
        if not hasattr(self, '_base_field_cache') or self._base_field_cache is None:
            self._base_field_cache = self._compute_base_field()

        base = self._base_field_cache
        noise_level = self.noise_level
        gauss = self.rng.gauss

        # Build this tick's field by copying base values + noise
        field: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for coord, mtypes in base.items():
            cell = field[coord]
            for mtype_val, base_conc in mtypes.items():
                noisy = base_conc + gauss(0, noise_level)
                if noisy > 0.0:
                    cell[mtype_val] = noisy
        self.morphogen_field = field

    def _agents_sense(self) -> None:
        """Each agent records local morphogen concentrations."""
        for agent in self.agents.values():
            gx = int(round(agent.x))
            gy = int(round(agent.y))
            gx = max(0, min(self.grid_size - 1, gx))
            gy = max(0, min(self.grid_size - 1, gy))
            local = self.morphogen_field[(gx, gy)]
            for mtype, conc in local.items():
                agent.exposure_history[mtype].append(conc)

    def _determine_fates(self) -> None:
        """Assign fates based on morphogen exposure within competence windows."""
        for agent in self.agents.values():
            if agent.fate != CellFate.UNDIFFERENTIATED:
                continue
            if not (agent.competence_window[0] <= self.step <= agent.competence_window[1]):
                continue

            # Compute cumulative exposure for each morphogen
            exposures: Dict[str, float] = {}
            for mtype, history in agent.exposure_history.items():
                exposures[mtype] = sum(history)

            activator = exposures.get(MorphogenType.ACTIVATOR.value, 0.0)
            inhibitor = exposures.get(MorphogenType.INHIBITOR.value, 0.0)
            positional = exposures.get(MorphogenType.POSITIONAL.value, 0.0)

            # Fate determination rules (threshold-based, inspired by French Flag model)
            net_signal = activator - inhibitor * 0.6

            if net_signal > 8.0:
                agent.fate = CellFate.LEADER
            elif net_signal > 5.0:
                if positional > 3.0:
                    agent.fate = CellFate.SENSOR
                else:
                    agent.fate = CellFate.RELAY
            elif net_signal > 2.5:
                agent.fate = CellFate.WORKER
            elif net_signal > 1.0:
                if positional > 2.0:
                    agent.fate = CellFate.MEMORY
                else:
                    agent.fate = CellFate.EFFECTOR
            # Below threshold: remain undifferentiated (may differentiate later)

            if agent.fate != CellFate.UNDIFFERENTIATED:
                agent.differentiation_step = self.step

    def _induction_signaling(self) -> None:
        """Differentiated agents can induce neighbors into complementary fates.

        Uses a spatial hash grid (cell size = induction range) to avoid the
        O(differentiated × undifferentiated) brute-force distance scan.
        Only agent pairs in the same or adjacent grid cells are checked,
        reducing average cost to O(D × local_density) where D is the number
        of inducting differentiated agents.
        """
        # Only active during gastrulation and organogenesis
        if self.stage not in (DevelopmentalStage.GASTRULATION, DevelopmentalStage.ORGANOGENESIS):
            return

        # Induction rules: certain fates recruit complementary neighbors
        induction_map: Dict[CellFate, CellFate] = {
            CellFate.LEADER: CellFate.RELAY,
            CellFate.SENSOR: CellFate.MEMORY,
            CellFate.RELAY: CellFate.WORKER,
        }

        INDUCTION_RANGE = 3.0
        INDUCTION_RANGE_SQ = INDUCTION_RANGE * INDUCTION_RANGE

        # Build spatial hash of undifferentiated agents (cell size = induction range)
        undiff_grid: Dict[Tuple[int, int], List[AgentCell]] = defaultdict(list)
        for a in self.agents.values():
            if a.fate == CellFate.UNDIFFERENTIATED:
                cx = int(a.x // INDUCTION_RANGE)
                cy = int(a.y // INDUCTION_RANGE)
                undiff_grid[(cx, cy)].append(a)

        if not undiff_grid:
            return

        step = self.step
        rng_random = self.rng.random

        for diff_agent in self.agents.values():
            if diff_agent.fate == CellFate.UNDIFFERENTIATED:
                continue
            target_fate = induction_map.get(diff_agent.fate)
            if target_fate is None:
                continue

            # Check only neighboring spatial cells
            cx = int(diff_agent.x // INDUCTION_RANGE)
            cy = int(diff_agent.y // INDUCTION_RANGE)
            dx_a = diff_agent.x
            dy_a = diff_agent.y

            for nx in range(cx - 1, cx + 2):
                for ny in range(cy - 1, cy + 2):
                    cell = undiff_grid.get((nx, ny))
                    if not cell:
                        continue
                    for target in cell:
                        ddx = dx_a - target.x
                        ddy = dy_a - target.y
                        dist_sq = ddx * ddx + ddy * ddy
                        if dist_sq >= INDUCTION_RANGE_SQ:
                            continue
                        dist = math.sqrt(dist_sq)
                        # Probabilistic induction (closer = more likely)
                        prob = 0.15 * (1.0 - dist / INDUCTION_RANGE)
                        if rng_random() < prob:
                            # Check competence window
                            if target.competence_window[0] <= step <= target.competence_window[1]:
                                target.fate = target_fate
                                target.differentiation_step = step
                                diff_agent.signals_emitted += 1
                                self.induction_events.append(InductionEvent(
                                    source_id=diff_agent.agent_id,
                                    target_id=target.agent_id,
                                    induced_fate=target_fate,
                                    step=step,
                                    signal_strength=prob,
                                ))

    def _apoptosis_check(self) -> None:
        """Remove agents that are misplaced or redundant (only during maturation)."""
        if self.stage != DevelopmentalStage.MATURATION:
            return

        to_remove: List[str] = []
        fate_counts = self._count_fates()

        for agent in list(self.agents.values()):
            # Rule 1: Too many of one fate type (> 40% of swarm) → prune excess
            if agent.fate != CellFate.UNDIFFERENTIATED:
                ratio = fate_counts.get(agent.fate.value, 0) / max(len(self.agents), 1)
                if ratio > 0.4 and self.rng.random() < 0.1:
                    to_remove.append(agent.agent_id)
                    self.apoptosis_events.append(ApoptosisEvent(
                        agent_id=agent.agent_id,
                        reason="excess_fate_ratio",
                        step=self.step,
                        position=(agent.x, agent.y),
                    ))
                    continue

            # Rule 2: Still undifferentiated past competence window → prune
            if (
                agent.fate == CellFate.UNDIFFERENTIATED
                and self.step > agent.competence_window[1] + 20
            ):
                if self.rng.random() < 0.3:
                    to_remove.append(agent.agent_id)
                    self.apoptosis_events.append(ApoptosisEvent(
                        agent_id=agent.agent_id,
                        reason="failed_differentiation",
                        step=self.step,
                        position=(agent.x, agent.y),
                    ))

        for aid in to_remove:
            del self.agents[aid]

    def _update_stage(self) -> None:
        """Progress through developmental stages based on swarm state."""
        prev_stage = self.stage
        diff_ratio = self._differentiation_ratio()
        len(self.agents)

        if self.stage == DevelopmentalStage.ZYGOTE and self.step >= 5:
            self.stage = DevelopmentalStage.CLEAVAGE
        elif self.stage == DevelopmentalStage.CLEAVAGE and self.step >= 15:
            self.stage = DevelopmentalStage.GASTRULATION
        elif self.stage == DevelopmentalStage.GASTRULATION and diff_ratio > 0.3:
            self.stage = DevelopmentalStage.ORGANOGENESIS
        elif self.stage == DevelopmentalStage.ORGANOGENESIS and diff_ratio > 0.7:
            self.stage = DevelopmentalStage.MATURATION
        elif self.stage == DevelopmentalStage.MATURATION and diff_ratio > 0.85:
            self.stage = DevelopmentalStage.HOMEOSTASIS

        if self.stage != prev_stage:
            self.stage_history.append({"stage": self.stage.value, "step": self.step})

    def _differentiation_ratio(self) -> float:
        """Fraction of agents that have differentiated."""
        if not self.agents:
            return 0.0
        diff = sum(1 for a in self.agents.values() if a.fate != CellFate.UNDIFFERENTIATED)
        return diff / len(self.agents)

    def _count_fates(self) -> Dict[str, int]:
        """Count agents per fate type."""
        counts: Dict[str, int] = defaultdict(int)
        for a in self.agents.values():
            counts[a.fate.value] += 1
        return dict(counts)

    # ── Damage & Regeneration ────────────────────────────────────────────

    def inflict_damage(self, n_agents: int = 3) -> List[str]:
        """Remove random differentiated agents to simulate structural damage."""
        differentiated = [a for a in self.agents.values() if a.fate != CellFate.UNDIFFERENTIATED]
        victims = self.rng.sample(differentiated, min(n_agents, len(differentiated)))
        removed = [v.agent_id for v in victims]
        for v in victims:
            del self.agents[v.agent_id]
        return removed

    def regenerate(self) -> Optional[RegenerationEvent]:
        """Attempt autonomous repair of structural damage."""
        # Detect missing fate types
        fate_counts = self._count_fates()
        all_fates = [f for f in CellFate if f != CellFate.UNDIFFERENTIATED]
        missing_fates = [f for f in all_fates if fate_counts.get(f.value, 0) == 0]

        if not missing_fates and len(self.agents) >= self.num_agents * 0.8:
            return None  # No damage detected

        # Strategy 1: Re-differentiate undifferentiated agents
        undiff = [a for a in self.agents.values() if a.fate == CellFate.UNDIFFERENTIATED]
        repaired: List[str] = []

        for fate in missing_fates:
            if undiff:
                agent = undiff.pop()
                agent.fate = fate
                agent.differentiation_step = self.step
                repaired.append(agent.agent_id)

        # Strategy 2: Recruit new agents if population too low
        recruited: List[str] = []
        while len(self.agents) < self.num_agents * 0.8:
            idx = len(self.agents)
            aid = f"cell-regen-{idx:03d}"
            x = self.rng.uniform(0, self.grid_size - 1)
            y = self.rng.uniform(0, self.grid_size - 1)
            self.agents[aid] = AgentCell(
                agent_id=aid, x=x, y=y,
                competence_window=(self.step, self.step + 30),
                generation=1,
            )
            recruited.append(aid)

        repair_type = "redifferentiation" if repaired else "recruitment"
        event = RegenerationEvent(
            damaged_agents=[],  # Already removed
            repaired_agents=repaired + recruited,
            step=self.step,
            repair_type=repair_type,
        )
        self.regeneration_events.append(event)
        return event

    # ── Pattern Detection ────────────────────────────────────────────────

    def _detect_pattern(self) -> Tuple[PatternType, float]:
        """Detect emergent spatial patterns from fate distribution.

        Uses squared distances throughout the clustering analysis to avoid
        O(n²) sqrt calls.  The clustering ratio is computed from mean squared
        distances (MSD_inter / MSD_intra) which preserves ordering — sqrt is
        only applied to the final ratio for the classification thresholds.
        """
        if not self.agents:
            return PatternType.UNIFORM, 0.0

        differentiated = [a for a in self.agents.values() if a.fate != CellFate.UNDIFFERENTIATED]
        if len(differentiated) < 4:
            return PatternType.UNIFORM, 0.0

        # Analyze spatial clustering of same-fate agents
        fate_positions: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for a in differentiated:
            fate_positions[a.fate.value].append((a.x, a.y))

        # Compute intra-cluster vs inter-cluster squared distances (avoid sqrt)
        intra_sum = 0.0
        intra_count = 0
        inter_sum = 0.0
        inter_count = 0

        fates = list(fate_positions.keys())
        for fate, positions in fate_positions.items():
            n = len(positions)
            if n < 2:
                continue
            for i in range(n):
                pi = positions[i]
                for j in range(i + 1, n):
                    pj = positions[j]
                    dx = pi[0] - pj[0]
                    dy = pi[1] - pj[1]
                    intra_sum += dx * dx + dy * dy
                    intra_count += 1

        for i in range(len(fates)):
            pos_i = fate_positions[fates[i]]
            for j in range(i + 1, len(fates)):
                pos_j = fate_positions[fates[j]]
                for p1 in pos_i:
                    p1x, p1y = p1
                    for p2 in pos_j:
                        dx = p1x - p2[0]
                        dy = p1y - p2[1]
                        inter_sum += dx * dx + dy * dy
                        inter_count += 1

        if intra_count == 0 or inter_count == 0:
            return PatternType.UNIFORM, 0.5

        # Convert mean squared distances to mean distances via sqrt for
        # threshold comparison (sqrt of mean ≈ RMS, good enough for ratios)
        avg_intra = math.sqrt(intra_sum / intra_count)
        avg_inter = math.sqrt(inter_sum / inter_count)

        # Clustering ratio: high = well-clustered
        clustering_ratio = avg_inter / max(avg_intra, 0.01)

        # Check for gradient pattern (fates ordered along an axis)
        fate_x_means: Dict[str, float] = {}
        for fate, positions in fate_positions.items():
            fate_x_means[fate] = statistics.mean([p[0] for p in positions])

        x_spread = max(fate_x_means.values()) - min(fate_x_means.values()) if fate_x_means else 0
        gradient_signal = x_spread / max(self.grid_size, 1)

        # Classify pattern
        if clustering_ratio > 2.5:
            pattern = PatternType.CLUSTERS
            regularity = min(clustering_ratio / 4.0, 1.0)
        elif gradient_signal > 0.5:
            pattern = PatternType.GRADIENT
            regularity = gradient_signal
        elif clustering_ratio > 1.5 and gradient_signal > 0.3:
            pattern = PatternType.STRIPES
            regularity = (clustering_ratio / 3.0 + gradient_signal) / 2.0
        elif len(fates) >= 3 and clustering_ratio > 1.8:
            pattern = PatternType.SPOTS
            regularity = clustering_ratio / 3.0
        elif clustering_ratio < 1.2:
            pattern = PatternType.UNIFORM
            regularity = 1.0 - (clustering_ratio - 1.0)
        else:
            pattern = PatternType.MIXED
            regularity = 0.5

        regularity = max(0.0, min(1.0, regularity))
        return pattern, regularity

    # ── Analysis ─────────────────────────────────────────────────────────

    def analyze(self) -> MorphogenesisReport:
        """Generate comprehensive development report."""
        fate_counts = self._count_fates()
        pattern_type, pattern_regularity = self._detect_pattern()

        # Morphogen field summary (average concentration per type)
        morph_summary: Dict[str, float] = defaultdict(float)
        cells_counted = 0
        for concentrations in self.morphogen_field.values():
            for mtype, conc in concentrations.items():
                morph_summary[mtype] += conc
            cells_counted += 1
        if cells_counted > 0:
            morph_summary = {k: v / cells_counted for k, v in morph_summary.items()}

        # Health score breakdown
        diff_ratio = self._differentiation_ratio()
        diversity = self._fate_diversity()
        stage_progress = list(DevelopmentalStage).index(self.stage) / 5.0

        health_breakdown = {
            "differentiation_completeness": diff_ratio * 100,
            "fate_diversity": diversity * 100,
            "pattern_regularity": pattern_regularity * 100,
            "stage_progress": stage_progress * 100,
            "structural_integrity": self._structural_integrity() * 100,
        }
        health_score = statistics.mean(health_breakdown.values())

        # Insights
        insights = self._generate_insights(fate_counts, diff_ratio, pattern_type, health_score)

        # Fate map
        fate_map = {aid: a.fate.value for aid, a in self.agents.items()}

        return MorphogenesisReport(
            stage=self.stage,
            step=self.step,
            total_agents=len(self.agents),
            fate_map=fate_map,
            fate_counts=fate_counts,
            pattern_type=pattern_type,
            pattern_regularity=pattern_regularity,
            morphogen_field_summary=dict(morph_summary),
            induction_events=[
                {"source": e.source_id, "target": e.target_id,
                 "fate": e.induced_fate.value, "step": e.step}
                for e in self.induction_events[-20:]  # Last 20
            ],
            apoptosis_events=[
                {"agent": e.agent_id, "reason": e.reason, "step": e.step}
                for e in self.apoptosis_events[-20:]
            ],
            regeneration_events=[
                {"repaired": e.repaired_agents, "step": e.step, "type": e.repair_type}
                for e in self.regeneration_events
            ],
            health_score=round(health_score, 1),
            health_breakdown={k: round(v, 1) for k, v in health_breakdown.items()},
            stage_history=self.stage_history,
            insights=insights,
        )

    def _fate_diversity(self) -> float:
        """Shannon diversity of fate distribution (normalized 0-1)."""
        counts = self._count_fates()
        total = sum(counts.values())
        if total == 0:
            return 0.0

        # Exclude undifferentiated from diversity calculation
        diff_counts = {k: v for k, v in counts.items() if k != CellFate.UNDIFFERENTIATED.value}
        if not diff_counts:
            return 0.0

        entropy = 0.0
        total_diff = sum(diff_counts.values())
        for count in diff_counts.values():
            if count > 0:
                p = count / total_diff
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(CellFate) - 1)  # Exclude undifferentiated
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _structural_integrity(self) -> float:
        """Measure structural integrity (population retention, no missing roles)."""
        pop_ratio = len(self.agents) / max(self.num_agents, 1)
        pop_score = min(pop_ratio, 1.0)

        # Check all fates represented
        fate_counts = self._count_fates()
        all_fates = [f.value for f in CellFate if f != CellFate.UNDIFFERENTIATED]
        represented = sum(1 for f in all_fates if fate_counts.get(f, 0) > 0)
        coverage = represented / len(all_fates)

        return (pop_score + coverage) / 2.0

    def _generate_insights(
        self,
        fate_counts: Dict[str, int],
        diff_ratio: float,
        pattern_type: PatternType,
        health_score: float,
    ) -> List[str]:
        """Generate human-readable developmental insights."""
        insights: List[str] = []

        if diff_ratio < 0.3:
            insights.append("⚠️ Low differentiation — most agents remain undifferentiated. "
                          "Consider adding more organizer signals or extending competence windows.")
        elif diff_ratio > 0.9:
            insights.append("✅ High differentiation achieved — nearly all agents have specialized.")

        # Check for fate imbalance
        diff_counts = {k: v for k, v in fate_counts.items()
                      if k != CellFate.UNDIFFERENTIATED.value}
        if diff_counts:
            max_fate = max(diff_counts, key=diff_counts.get)  # type: ignore
            max_ratio = diff_counts[max_fate] / sum(diff_counts.values())
            if max_ratio > 0.5:
                insights.append(f"⚠️ Fate imbalance: {max_fate} dominates ({max_ratio:.0%}). "
                              "Lateral inhibition or additional organizers may help diversify.")

        if pattern_type == PatternType.GRADIENT:
            insights.append("🎨 Gradient pattern detected — French Flag-style positional encoding active.")
        elif pattern_type == PatternType.CLUSTERS:
            insights.append("🎨 Cluster pattern — agents of same fate are spatially grouped.")
        elif pattern_type == PatternType.SPOTS:
            insights.append("🎨 Spot pattern — Turing-like reaction-diffusion may be active.")

        if self.induction_events:
            insights.append(f"🔗 {len(self.induction_events)} induction events — "
                          "cell-cell signaling is actively recruiting neighbors.")

        if self.apoptosis_events:
            insights.append(f"✂️ {len(self.apoptosis_events)} apoptosis events — "
                          "developmental pruning is refining structure.")

        if self.regeneration_events:
            insights.append(f"🔄 {len(self.regeneration_events)} regeneration events — "
                          "autonomous repair has been triggered.")

        if health_score >= 80:
            insights.append("💚 Excellent developmental health — swarm is well-organized.")
        elif health_score >= 50:
            insights.append("💛 Moderate developmental health — some structural issues remain.")
        else:
            insights.append("❤️ Poor developmental health — significant organizational deficits.")

        if self.stage == DevelopmentalStage.HOMEOSTASIS:
            insights.append("🏠 Homeostasis reached — swarm structure is stable and self-maintaining.")

        return insights

    # ── Export ────────────────────────────────────────────────────────────

    def export_json(self, path: str) -> None:
        """Export full state to JSON."""
        report = self.analyze()
        data = {
            "step": self.step,
            "stage": self.stage.value,
            "grid_size": self.grid_size,
            "total_agents": len(self.agents),
            "fate_map": report.fate_map,
            "fate_counts": report.fate_counts,
            "pattern_type": report.pattern_type.value,
            "pattern_regularity": report.pattern_regularity,
            "health_score": report.health_score,
            "health_breakdown": report.health_breakdown,
            "stage_history": report.stage_history,
            "induction_events": report.induction_events,
            "apoptosis_events": report.apoptosis_events,
            "regeneration_events": report.regeneration_events,
            "insights": report.insights,
            "agents": [
                {
                    "id": a.agent_id,
                    "x": round(a.x, 2),
                    "y": round(a.y, 2),
                    "fate": a.fate.value,
                    "differentiation_step": a.differentiation_step,
                    "generation": a.generation,
                }
                for a in self.agents.values()
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def export_html(self, path: str) -> None:
        """Generate interactive HTML dashboard."""
        report = self.analyze()
        fate_colors = {
            "undifferentiated": "#999",
            "leader": "#e74c3c",
            "relay": "#f39c12",
            "worker": "#3498db",
            "sensor": "#2ecc71",
            "memory": "#9b59b6",
            "effector": "#1abc9c",
        }

        # Build agent dots for SVG visualization
        agent_dots = ""
        for a in self.agents.values():
            color = fate_colors.get(a.fate.value, "#999")
            sx = (a.x / self.grid_size) * 400 + 50
            sy = (a.y / self.grid_size) * 400 + 50
            agent_dots += (
                f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="8" '
                f'fill="{color}" stroke="#333" stroke-width="1" '
                f'opacity="0.85"><title>{a.agent_id}: {a.fate.value}</title></circle>\n'
            )

        # Stage timeline
        stage_timeline = ""
        for entry in self.stage_history:
            stage_timeline += (
                f'<div style="display:inline-block;margin:4px 8px;padding:4px 10px;'
                f'background:#2d3436;border-radius:4px;font-size:12px;">'
                f'{html_mod.escape(entry["stage"])} (step {entry["step"]})</div>'
            )

        # Fate distribution bars
        fate_bars = ""
        total = sum(report.fate_counts.values())
        for fate, count in sorted(report.fate_counts.items()):
            pct = (count / total * 100) if total > 0 else 0
            color = fate_colors.get(fate, "#999")
            fate_bars += (
                f'<div style="margin:4px 0;">'
                f'<span style="display:inline-block;width:120px;">{html_mod.escape(fate)}</span>'
                f'<div style="display:inline-block;width:{pct*2:.0f}px;height:18px;'
                f'background:{color};border-radius:3px;margin-right:8px;"></div>'
                f'<span>{count} ({pct:.0f}%)</span></div>'
            )

        # Insights
        insights_html = ""
        for insight in report.insights:
            insights_html += f'<li style="margin:6px 0;">{html_mod.escape(insight)}</li>'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Morphogenesis Dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 12px; }}
h2 {{ color: #8b949e; margin-top: 30px; }}
.card {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px;
         padding: 20px; margin: 16px 0; }}
.metric {{ display: inline-block; text-align: center; margin: 10px 20px; }}
.metric-value {{ font-size: 28px; font-weight: bold; color: #58a6ff; }}
.metric-label {{ font-size: 12px; color: #8b949e; }}
.health-bar {{ height: 24px; border-radius: 12px; background: #21262d; overflow: hidden; }}
.health-fill {{ height: 100%; border-radius: 12px; transition: width 0.3s; }}
svg {{ background: #161b22; border-radius: 8px; border: 1px solid #21262d; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 13px; }}
.legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
</style>
</head>
<body>
<div class="container">
<h1>🧬 Swarm Morphogenesis Dashboard</h1>

<div class="card">
  <div class="metric">
    <div class="metric-value">{report.health_score:.0f}</div>
    <div class="metric-label">Health Score</div>
  </div>
  <div class="metric">
    <div class="metric-value">{report.stage.value}</div>
    <div class="metric-label">Dev Stage</div>
  </div>
  <div class="metric">
    <div class="metric-value">{report.total_agents}</div>
    <div class="metric-label">Agents</div>
  </div>
  <div class="metric">
    <div class="metric-value">{report.pattern_type.value}</div>
    <div class="metric-label">Pattern</div>
  </div>
  <div class="metric">
    <div class="metric-value">{report.step}</div>
    <div class="metric-label">Dev Steps</div>
  </div>
</div>

<h2>🗺️ Fate Map</h2>
<div class="card">
  <svg width="500" height="500" viewBox="0 0 500 500">
    <rect x="50" y="50" width="400" height="400" fill="#0d1117" stroke="#21262d"/>
    {agent_dots}
  </svg>
  <div class="legend">
    {''.join(f'<div class="legend-item"><div class="legend-dot" style="background:{c}"></div>{f}</div>' for f, c in fate_colors.items())}
  </div>
</div>

<h2>📊 Fate Distribution</h2>
<div class="card">{fate_bars}</div>

<h2>⏱️ Developmental Timeline</h2>
<div class="card">{stage_timeline}</div>

<h2>💚 Health Breakdown</h2>
<div class="card">
  {''.join(f'<div style="margin:8px 0;"><span style="display:inline-block;width:220px;">{html_mod.escape(k)}</span><div class="health-bar" style="display:inline-block;width:200px;vertical-align:middle;"><div class="health-fill" style="width:{v:.0f}%;background:{"#2ecc71" if v >= 70 else "#f39c12" if v >= 40 else "#e74c3c"}"></div></div> <span>{v:.0f}%</span></div>' for k, v in report.health_breakdown.items())}
</div>

<h2>💡 Insights</h2>
<div class="card"><ul style="padding-left:20px;">{insights_html}</ul></div>

<h2>🔗 Recent Induction Events</h2>
<div class="card">
  {''.join(f'<div style="margin:4px 0;font-size:13px;">Step {e["step"]}: {html_mod.escape(e["source"])} → {html_mod.escape(e["target"])} ({html_mod.escape(e["fate"])})</div>' for e in report.induction_events[-10:]) or '<em>No induction events yet</em>'}
</div>

<p style="text-align:center;color:#484f58;margin-top:30px;">
  Generated by Swarm Morphogenesis Engine | mBFT Metacognition Framework
</p>
</div>
</body>
</html>"""
        Path(path).write_text(html_content, encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    """Run morphogenesis simulation from command line."""
    parser = argparse.ArgumentParser(
        description="Swarm Morphogenesis Engine — autonomous structural self-organization"
    )
    parser.add_argument("--grid", type=int, default=15, help="Grid size (default: 15)")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents (default: 20)")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps (default: 100)")
    parser.add_argument("--damage", type=int, default=0, help="Inflict N damage then regenerate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Export HTML report path")
    parser.add_argument("--json", type=str, default=None, help="Export JSON state path")
    args = parser.parse_args()

    print("🧬 Swarm Morphogenesis Engine")
    print("=" * 50)
    print(f"Grid: {args.grid}×{args.grid} | Agents: {args.agents} | Steps: {args.steps}")
    print()

    engine = MorphogenesisEngine(
        grid_size=args.grid, num_agents=args.agents, seed=args.seed
    )

    # Add default organizers (two opposing gradients for French Flag patterning)
    engine.add_organizer(x=args.grid * 0.2, y=args.grid * 0.5,
                        morphogen="activator", strength=2.5, decay_rate=0.08)
    engine.add_organizer(x=args.grid * 0.8, y=args.grid * 0.5,
                        morphogen="inhibitor", strength=1.8, decay_rate=0.1)
    engine.add_organizer(x=args.grid * 0.5, y=args.grid * 0.2,
                        morphogen="positional", strength=1.5, decay_rate=0.12)

    # Run development
    print("🔬 Running developmental simulation...")
    engine.develop(steps=args.steps)

    # Optional damage + regeneration
    if args.damage > 0:
        print(f"\n💥 Inflicting damage ({args.damage} agents removed)...")
        removed = engine.inflict_damage(args.damage)
        print(f"   Removed: {removed}")
        print("🔄 Attempting regeneration...")
        event = engine.regenerate()
        if event:
            print(f"   Repaired: {event.repaired_agents} ({event.repair_type})")
        else:
            print("   No repair needed.")

    # Analyze
    report = engine.analyze()

    print(f"\n📋 Development Report")
    print(f"   Stage: {report.stage.value}")
    print(f"   Agents: {report.total_agents}")
    print(f"   Pattern: {report.pattern_type.value} (regularity: {report.pattern_regularity:.2f})")
    print(f"   Health Score: {report.health_score:.0f}/100")
    print(f"\n   Fate Distribution:")
    for fate, count in sorted(report.fate_counts.items()):
        print(f"     {fate}: {count}")
    print(f"\n   Health Breakdown:")
    for metric, val in report.health_breakdown.items():
        print(f"     {metric}: {val:.0f}%")
    print(f"\n   Insights:")
    for insight in report.insights:
        print(f"     {insight}")

    if report.induction_events:
        print(f"\n   Induction Events: {len(report.induction_events)}")
    if report.apoptosis_events:
        print(f"   Apoptosis Events: {len(report.apoptosis_events)}")

    # Export
    if args.out:
        engine.export_html(args.out)
        print(f"\n📄 HTML report: {args.out}")
    if args.json:
        engine.export_json(args.json)
        print(f"📄 JSON state: {args.json}")


if __name__ == "__main__":
    main()
