"""Swarm Chemotaxis Engine — autonomous chemical gradient navigation.

Biologically-inspired by bacterial chemotaxis (E. coli run-and-tumble,
Dictyostelium cAMP relay, neutrophil homing).  Agents sense chemical
gradients in a shared environment, adapt their receptors over time, and
navigate toward attractants and away from repellents using biased random
walks.

Capabilities:

- **Chemical Environment** — shared 2D grid with multiple chemical species
  (attractant, repellent, nutrient, toxin, signaling, trail, beacon) that
  diffuse and decay over configurable dynamics.
- **Receptor Model** — each agent has typed receptors with methylation-based
  adaptation; sensitivity decreases on sustained exposure (desensitization)
  and recovers during absence (re-sensitization).
- **Run-and-Tumble Motor** — agents alternate between straight runs and
  random reorientation tumbles; favorable gradients suppress tumble frequency
  (positive chemotaxis).
- **Collective Gradient Sensing** — agents pool gradient measurements to
  achieve super-cellular accuracy via signal averaging.
- **Source Localization** — autonomous tracking of gradient sources: triangulation
  from multi-agent observations, source strength estimation, convergence detection.
- **Chemotactic Index** — per-agent and fleet-wide efficiency metrics measuring
  directional progress toward sources relative to total displacement.
- **Adaptation Memory** — receptor methylation history enables agents to
  distinguish absolute concentrations from temporal changes (Weber's law).
- **Health Scoring** — composite 0-100 metric assessing navigation efficiency,
  receptor health, source coverage, and collective coordination.
- **Interactive HTML Dashboard** — visualizes chemical landscapes, agent
  trajectories, receptor states, and localization convergence.

Usage (Python API)::

    from src.chemotaxis import SwarmChemotaxisEngine, ChemicalType

    engine = SwarmChemotaxisEngine(grid_size=30, num_agents=8)

    # Place chemical sources
    engine.add_source(x=25, y=25, chemical=ChemicalType.ATTRACTANT, strength=5.0)
    engine.add_source(x=5, y=5, chemical=ChemicalType.REPELLENT, strength=3.0)
    engine.add_source(x=15, y=20, chemical=ChemicalType.NUTRIENT, strength=4.0)

    # Run simulation
    report = engine.simulate(steps=200)
    print(report.health.score)           # 0-100 navigation health
    print(report.localization.sources)   # detected source locations
    print(report.fleet_ci)               # fleet chemotactic index

    engine.export_html("chemotaxis_report.html")

CLI::

    python -m src.chemotaxis                         # demo with defaults
    python -m src.chemotaxis --grid 40 --agents 12   # larger environment
    python -m src.chemotaxis --steps 300             # longer simulation
    python -m src.chemotaxis --out report.html --json chemotaxis.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------

class ChemicalType(str, Enum):
    """Types of chemical species in the environment."""
    ATTRACTANT = "attractant"    # draws agents toward source
    REPELLENT = "repellent"      # pushes agents away
    NUTRIENT = "nutrient"        # beneficial resource
    TOXIN = "toxin"              # harmful substance
    SIGNALING = "signaling"      # inter-agent communication molecule
    TRAIL = "trail"              # agent-deposited breadcrumb
    BEACON = "beacon"            # strong long-range attractor


# Chemical polarity: +1 = agents move toward, -1 = agents move away
CHEMICAL_POLARITY: Dict[ChemicalType, float] = {
    ChemicalType.ATTRACTANT: 1.0,
    ChemicalType.REPELLENT: -1.0,
    ChemicalType.NUTRIENT: 1.0,
    ChemicalType.TOXIN: -1.0,
    ChemicalType.SIGNALING: 0.5,
    ChemicalType.TRAIL: 0.3,
    ChemicalType.BEACON: 1.5,
}

# Default diffusion rates (fraction spreading to neighbors per tick)
DEFAULT_DIFFUSION: Dict[ChemicalType, float] = {
    ChemicalType.ATTRACTANT: 0.05,
    ChemicalType.REPELLENT: 0.08,
    ChemicalType.NUTRIENT: 0.03,
    ChemicalType.TOXIN: 0.06,
    ChemicalType.SIGNALING: 0.10,
    ChemicalType.TRAIL: 0.02,
    ChemicalType.BEACON: 0.04,
}

# Default decay rates (fraction lost per tick)
DEFAULT_DECAY: Dict[ChemicalType, float] = {
    ChemicalType.ATTRACTANT: 0.02,
    ChemicalType.REPELLENT: 0.03,
    ChemicalType.NUTRIENT: 0.01,
    ChemicalType.TOXIN: 0.04,
    ChemicalType.SIGNALING: 0.08,
    ChemicalType.TRAIL: 0.05,
    ChemicalType.BEACON: 0.01,
}


@dataclass
class ChemicalSource:
    """A source that continuously emits a chemical."""
    x: int
    y: int
    chemical: ChemicalType
    strength: float
    source_id: str = ""
    active: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            self.source_id = f"src-{self.chemical.value}-{self.x},{self.y}"


@dataclass
class ReceptorState:
    """Receptor state for a single chemical type."""
    chemical: ChemicalType
    sensitivity: float = 1.0         # current sensitivity (0-1)
    methylation: float = 0.5         # methylation level (adaptation state)
    last_concentration: float = 0.0  # last sensed concentration
    adaptation_rate: float = 0.05    # how fast receptor adapts
    saturation_threshold: float = 10.0  # concentration for full saturation


@dataclass
class AgentMotorState:
    """Motor state controlling run-and-tumble behavior."""
    running: bool = True
    direction: float = 0.0          # radians
    speed: float = 1.0
    tumble_rate_base: float = 0.2   # base probability of tumbling per tick
    tumble_rate: float = 0.2        # current tumble probability (modulated)
    run_length: int = 0             # ticks since last tumble
    total_runs: int = 0
    total_tumbles: int = 0


@dataclass
class GradientMeasurement:
    """A single gradient measurement by an agent."""
    agent_id: str
    x: float
    y: float
    dx: float
    dy: float
    magnitude: float
    chemical: ChemicalType
    tick: int


@dataclass
class DetectedSource:
    """A source localized by collective gradient sensing."""
    estimated_x: float
    estimated_y: float
    chemical: ChemicalType
    estimated_strength: float
    confidence: float  # 0-1
    contributing_agents: int
    actual_distance: Optional[float] = None  # distance to nearest real source


@dataclass
class LocalizationReport:
    """Results of collective source localization."""
    sources: List[DetectedSource]
    convergence_score: float  # 0-1, how well agents converge
    triangulation_accuracy: float  # avg distance error


@dataclass
class AgentTrajectory:
    """Complete trajectory of a chemotactic agent."""
    agent_id: str
    positions: List[Tuple[float, float, int]]  # (x, y, tick)
    chemotactic_index: float  # directional efficiency 0-1
    total_distance: float
    net_displacement: float
    receptor_history: Dict[ChemicalType, List[float]]  # sensitivity over time
    run_tumble_ratio: float  # runs / (runs + tumbles)
    sources_reached: int


@dataclass
class FleetHealth:
    """Overall swarm chemotaxis health assessment."""
    score: float  # 0-100
    avg_chemotactic_index: float
    receptor_health: float  # avg receptor sensitivity diversity
    source_coverage: float  # fraction of sources with nearby agents
    coordination_score: float  # collective gradient sensing quality
    navigation_efficiency: float  # net displacement / total distance
    recommendations: List[str]


@dataclass
class ChemotaxisReport:
    """Complete simulation report."""
    health: FleetHealth
    localization: LocalizationReport
    trajectories: List[AgentTrajectory]
    fleet_ci: float  # fleet chemotactic index
    grid_size: int
    total_ticks: int
    num_sources: int
    config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Chemical Grid
# ---------------------------------------------------------------------------

class ChemicalGrid:
    """2D grid storing chemical concentrations with diffusion and decay."""

    def __init__(
        self,
        size: int,
        diffusion_rates: Optional[Dict[ChemicalType, float]] = None,
        decay_rates: Optional[Dict[ChemicalType, float]] = None,
    ):
        self.size = size
        self.diffusion = diffusion_rates or dict(DEFAULT_DIFFUSION)
        self.decay = decay_rates or dict(DEFAULT_DECAY)
        # grid[y][x][chem] = concentration
        self._grid: List[List[Dict[ChemicalType, float]]] = [
            [{} for _ in range(size)] for _ in range(size)
        ]

    def emit(self, x: int, y: int, chemical: ChemicalType, amount: float) -> None:
        """Add chemical at a position."""
        gx, gy = x % self.size, y % self.size
        cell = self._grid[gy][gx]
        cell[chemical] = cell.get(chemical, 0.0) + amount

    def read(self, x: int, y: int, chemical: ChemicalType) -> float:
        """Read concentration at a position."""
        gx, gy = x % self.size, y % self.size
        return self._grid[gy][gx].get(chemical, 0.0)

    def read_all(self, x: int, y: int) -> Dict[ChemicalType, float]:
        """Read all concentrations at a position."""
        gx, gy = x % self.size, y % self.size
        return dict(self._grid[gy][gx])

    def total_at(self, x: int, y: int) -> float:
        """Total concentration of all chemicals at a position."""
        gx, gy = x % self.size, y % self.size
        return sum(self._grid[gy][gx].values())

    def coverage(self) -> float:
        """Fraction of grid cells with any chemical present."""
        occupied = 0
        for row in self._grid:
            for cell in row:
                if any(v > 0.01 for v in cell.values()):
                    occupied += 1
        return occupied / (self.size * self.size) if self.size > 0 else 0.0

    def step(self, sources: List[ChemicalSource]) -> None:
        """Advance one tick: emit from sources, diffuse, decay."""
        # Emit from active sources
        for src in sources:
            if src.active:
                self.emit(src.x, src.y, src.chemical, src.strength)

        # Diffuse: spread fraction to 4-connected neighbors
        new_grid: List[List[Dict[ChemicalType, float]]] = [
            [{} for _ in range(self.size)] for _ in range(self.size)
        ]
        for y in range(self.size):
            for x in range(self.size):
                cell = self._grid[y][x]
                for chem, conc in cell.items():
                    if conc < 0.001:
                        continue
                    diff_rate = self.diffusion.get(chem, 0.05)
                    spread = conc * diff_rate
                    remain = conc - spread
                    new_grid[y][x][chem] = new_grid[y][x].get(chem, 0.0) + remain
                    per_neighbor = spread / 4.0
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = (x + dx) % self.size, (y + dy) % self.size
                        new_grid[ny][nx][chem] = new_grid[ny][nx].get(chem, 0.0) + per_neighbor

        # Decay
        for y in range(self.size):
            for x in range(self.size):
                cell = new_grid[y][x]
                to_remove = []
                for chem, conc in cell.items():
                    decay_rate = self.decay.get(chem, 0.03)
                    new_val = conc * (1.0 - decay_rate)
                    if new_val < 0.001:
                        to_remove.append(chem)
                    else:
                        cell[chem] = new_val
                for chem in to_remove:
                    del cell[chem]

        self._grid = new_grid

    def gradient_at(self, x: int, y: int, chemical: ChemicalType) -> Tuple[float, float, float]:
        """Compute gradient (dx, dy, magnitude) for a chemical at a position."""
        gx, gy = x % self.size, y % self.size
        # Central difference over neighbors
        right = self.read((gx + 1), gy, chemical)
        left = self.read((gx - 1), gy, chemical)
        up = self.read(gx, (gy - 1), chemical)
        down = self.read(gx, (gy + 1), chemical)
        dx = (right - left) / 2.0
        dy = (down - up) / 2.0
        mag = math.sqrt(dx * dx + dy * dy)
        return dx, dy, mag

    def snapshot(self) -> Dict[str, Any]:
        """Get a summary of the grid state."""
        total_conc = 0.0
        max_conc = 0.0
        chem_totals: Dict[str, float] = defaultdict(float)
        for y in range(self.size):
            for x in range(self.size):
                for chem, conc in self._grid[y][x].items():
                    total_conc += conc
                    chem_totals[chem.value] += conc
                    if conc > max_conc:
                        max_conc = conc
        return {
            "total_concentration": round(total_conc, 2),
            "max_concentration": round(max_conc, 2),
            "coverage_pct": round(self.coverage() * 100, 1),
            "chemical_totals": {k: round(v, 2) for k, v in chem_totals.items()},
        }


# ---------------------------------------------------------------------------
# Chemotactic Agent
# ---------------------------------------------------------------------------

class ChemotacticAgent:
    """An agent that navigates via run-and-tumble chemotaxis."""

    def __init__(self, agent_id: str, x: float, y: float, grid_size: int):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.grid_size = grid_size
        self.motor = AgentMotorState(
            direction=random.uniform(0, 2 * math.pi),
        )
        self.receptors: Dict[ChemicalType, ReceptorState] = {
            ct: ReceptorState(chemical=ct) for ct in ChemicalType
        }
        self.positions: List[Tuple[float, float, int]] = [(x, y, 0)]
        self.measurements: List[GradientMeasurement] = []
        self._sources_reached: Set[str] = set()
        self._receptor_history: Dict[ChemicalType, List[float]] = {
            ct: [1.0] for ct in ChemicalType
        }

    def sense(self, grid: ChemicalGrid, tick: int) -> Tuple[float, float]:
        """Sense chemical gradients and compute desired movement bias."""
        ix, iy = int(self.x) % self.grid_size, int(self.y) % self.grid_size
        total_dx, total_dy = 0.0, 0.0

        for chem in ChemicalType:
            receptor = self.receptors[chem]
            concentration = grid.read(ix, iy, chem)
            gdx, gdy, gmag = grid.gradient_at(ix, iy, chem)

            # Receptor adaptation: compare to last measurement
            delta = concentration - receptor.last_concentration
            polarity = CHEMICAL_POLARITY.get(chem, 0.0)

            # Apply sensitivity (modulated by methylation)
            effective_sensitivity = receptor.sensitivity * (0.5 + receptor.methylation)

            # Temporal gradient sensing (Weber's law)
            if receptor.last_concentration > 0.01:
                relative_change = delta / receptor.last_concentration
            else:
                relative_change = min(delta, 1.0)

            # Combine spatial and temporal gradients
            bias = polarity * effective_sensitivity
            total_dx += (gdx * bias + relative_change * polarity * 0.3)
            total_dy += (gdy * bias + relative_change * polarity * 0.3)

            # Update receptor state
            receptor.last_concentration = concentration

            # Methylation adaptation: increase when stimulus is constant, decrease on change
            if abs(delta) < 0.1:
                receptor.methylation = min(1.0, receptor.methylation + receptor.adaptation_rate * 0.5)
            else:
                receptor.methylation = max(0.0, receptor.methylation - receptor.adaptation_rate)

            # Sensitivity: desensitize on high concentration
            if concentration > receptor.saturation_threshold * 0.5:
                receptor.sensitivity = max(0.1, receptor.sensitivity - 0.02)
            else:
                receptor.sensitivity = min(1.0, receptor.sensitivity + 0.01)

            self._receptor_history[chem].append(receptor.sensitivity)

            # Record measurement
            if gmag > 0.001:
                self.measurements.append(GradientMeasurement(
                    agent_id=self.agent_id,
                    x=self.x, y=self.y,
                    dx=gdx, dy=gdy,
                    magnitude=gmag,
                    chemical=chem,
                    tick=tick,
                ))

        return total_dx, total_dy

    def move(self, grid: ChemicalGrid, tick: int) -> None:
        """Execute one step of run-and-tumble movement."""
        bias_dx, bias_dy = self.sense(grid, tick)
        bias_mag = math.sqrt(bias_dx * bias_dx + bias_dy * bias_dy)

        # Modulate tumble rate: strong favorable gradient → less tumbling
        if bias_mag > 0.01:
            # Alignment between current direction and gradient
            cos_align = (math.cos(self.motor.direction) * bias_dx +
                         math.sin(self.motor.direction) * bias_dy) / (bias_mag + 1e-9)
            # If moving toward attractant, suppress tumbling
            tumble_suppression = max(0.0, cos_align) * 0.5
            self.motor.tumble_rate = max(0.05,
                self.motor.tumble_rate_base - tumble_suppression)
        else:
            self.motor.tumble_rate = self.motor.tumble_rate_base

        # Run or tumble decision
        if random.random() < self.motor.tumble_rate:
            # Tumble: reorient
            if bias_mag > 0.01:
                # Biased tumble: prefer gradient direction
                target_angle = math.atan2(bias_dy, bias_dx)
                noise = random.gauss(0, 0.5)
                self.motor.direction = target_angle + noise
            else:
                # Random tumble
                self.motor.direction = random.uniform(0, 2 * math.pi)
            self.motor.running = False
            self.motor.total_tumbles += 1
            self.motor.run_length = 0
        else:
            # Run: continue in current direction with slight bias
            if bias_mag > 0.01:
                target_angle = math.atan2(bias_dy, bias_dx)
                angle_diff = target_angle - self.motor.direction
                # Normalize to [-pi, pi]
                angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                # Gentle steering during runs
                self.motor.direction += angle_diff * 0.1
            self.motor.running = True
            self.motor.total_runs += 1
            self.motor.run_length += 1

        # Apply movement
        dx = math.cos(self.motor.direction) * self.motor.speed
        dy = math.sin(self.motor.direction) * self.motor.speed
        self.x = (self.x + dx) % self.grid_size
        self.y = (self.y + dy) % self.grid_size
        self.positions.append((self.x, self.y, tick))

        # Deposit trail pheromone
        grid.emit(int(self.x) % self.grid_size, int(self.y) % self.grid_size,
                  ChemicalType.TRAIL, 0.1)

    def check_sources(self, sources: List[ChemicalSource], radius: float = 2.0) -> None:
        """Check if agent is near any sources."""
        for src in sources:
            if src.source_id in self._sources_reached:
                continue
            dist = math.sqrt((self.x - src.x) ** 2 + (self.y - src.y) ** 2)
            if dist <= radius:
                self._sources_reached.add(src.source_id)

    def trajectory(self) -> AgentTrajectory:
        """Build trajectory report."""
        total_dist = 0.0
        for i in range(1, len(self.positions)):
            px, py, _ = self.positions[i - 1]
            cx, cy, _ = self.positions[i]
            total_dist += math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

        if len(self.positions) >= 2:
            sx, sy, _ = self.positions[0]
            ex, ey, _ = self.positions[-1]
            net_disp = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
        else:
            net_disp = 0.0

        ci = net_disp / total_dist if total_dist > 0 else 0.0
        total_moves = self.motor.total_runs + self.motor.total_tumbles
        rt_ratio = self.motor.total_runs / total_moves if total_moves > 0 else 0.5

        return AgentTrajectory(
            agent_id=self.agent_id,
            positions=list(self.positions),
            chemotactic_index=min(1.0, ci),
            total_distance=total_dist,
            net_displacement=net_disp,
            receptor_history={k: list(v) for k, v in self._receptor_history.items()},
            run_tumble_ratio=rt_ratio,
            sources_reached=len(self._sources_reached),
        )


# ---------------------------------------------------------------------------
# Source Localizer
# ---------------------------------------------------------------------------

class SourceLocalizer:
    """Collective gradient-based source localization."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    def localize(
        self,
        measurements: List[GradientMeasurement],
        actual_sources: List[ChemicalSource],
    ) -> LocalizationReport:
        """Estimate source positions from collective gradient measurements."""
        # Group measurements by chemical type
        by_chem: Dict[ChemicalType, List[GradientMeasurement]] = defaultdict(list)
        for m in measurements:
            if m.magnitude > 0.01:
                by_chem[m.chemical].append(m)

        detected: List[DetectedSource] = []

        for chem, meas_list in by_chem.items():
            if len(meas_list) < 3:
                continue

            # Use gradient intersection: each measurement provides a ray
            # Weighted average of projected source positions
            estimates_x: List[float] = []
            estimates_y: List[float] = []
            weights: List[float] = []

            for m in meas_list[-50:]:  # use recent measurements
                # Project forward along gradient direction
                if m.magnitude < 0.01:
                    continue
                nx = m.dx / m.magnitude
                ny = m.dy / m.magnitude
                # Estimate distance from magnitude (inverse square)
                est_dist = 1.0 / (m.magnitude + 0.01)
                est_dist = min(est_dist, self.grid_size / 2)
                est_x = (m.x + nx * est_dist) % self.grid_size
                est_y = (m.y + ny * est_dist) % self.grid_size
                estimates_x.append(est_x)
                estimates_y.append(est_y)
                weights.append(m.magnitude)

            if not estimates_x:
                continue

            # Weighted centroid
            total_w = sum(weights)
            if total_w < 0.01:
                continue
            cx = sum(x * w for x, w in zip(estimates_x, weights)) / total_w
            cy = sum(y * w for y, w in zip(estimates_y, weights)) / total_w

            # Confidence from measurement spread
            if len(estimates_x) > 1:
                var_x = statistics.variance(estimates_x)
                var_y = statistics.variance(estimates_y)
                spread = math.sqrt(var_x + var_y)
                confidence = max(0.1, min(1.0, 1.0 / (1.0 + spread / self.grid_size)))
            else:
                confidence = 0.3

            # Estimate strength from avg gradient magnitude
            avg_mag = statistics.mean(weights)
            est_strength = avg_mag * 10

            # Find nearest actual source
            min_dist = None
            for src in actual_sources:
                if src.chemical == chem:
                    d = math.sqrt((cx - src.x) ** 2 + (cy - src.y) ** 2)
                    if min_dist is None or d < min_dist:
                        min_dist = d

            detected.append(DetectedSource(
                estimated_x=round(cx, 2),
                estimated_y=round(cy, 2),
                chemical=chem,
                estimated_strength=round(est_strength, 2),
                confidence=round(confidence, 3),
                contributing_agents=len(set(m.agent_id for m in meas_list)),
                actual_distance=round(min_dist, 2) if min_dist is not None else None,
            ))

        # Convergence: how tightly do agents cluster near sources
        if detected:
            convergence = statistics.mean(d.confidence for d in detected)
            accuracies = [d.actual_distance for d in detected if d.actual_distance is not None]
            avg_accuracy = statistics.mean(accuracies) if accuracies else self.grid_size
        else:
            convergence = 0.0
            avg_accuracy = self.grid_size

        return LocalizationReport(
            sources=detected,
            convergence_score=round(convergence, 3),
            triangulation_accuracy=round(avg_accuracy, 2),
        )


# ---------------------------------------------------------------------------
# Health Assessor
# ---------------------------------------------------------------------------

class HealthAssessor:
    """Assesses overall chemotactic swarm health."""

    def assess(
        self,
        trajectories: List[AgentTrajectory],
        localization: LocalizationReport,
        grid: ChemicalGrid,
        sources: List[ChemicalSource],
    ) -> FleetHealth:
        recommendations: List[str] = []

        # 1. Average chemotactic index
        if trajectories:
            ci_values = [t.chemotactic_index for t in trajectories]
            avg_ci = statistics.mean(ci_values)
        else:
            avg_ci = 0.0

        # 2. Receptor health: diversity of sensitivity across chemicals
        receptor_scores: List[float] = []
        for traj in trajectories:
            sensitivities = []
            for chem, hist in traj.receptor_history.items():
                if hist:
                    sensitivities.append(hist[-1])
            if sensitivities:
                receptor_scores.append(statistics.mean(sensitivities))
        receptor_health = statistics.mean(receptor_scores) if receptor_scores else 0.5

        # 3. Source coverage
        if sources:
            covered = 0
            for src in sources:
                for traj in trajectories:
                    if traj.sources_reached > 0:
                        # Check if any position was near this source
                        for px, py, _ in traj.positions[-20:]:
                            if math.sqrt((px - src.x) ** 2 + (py - src.y) ** 2) < 3.0:
                                covered += 1
                                break
                        else:
                            continue
                        break
            source_coverage = covered / len(sources)
        else:
            source_coverage = 1.0

        # 4. Coordination score from localization
        coordination = localization.convergence_score

        # 5. Navigation efficiency
        if trajectories:
            efficiencies = []
            for t in trajectories:
                if t.total_distance > 0:
                    efficiencies.append(t.net_displacement / t.total_distance)
                else:
                    efficiencies.append(0.0)
            nav_efficiency = statistics.mean(efficiencies)
        else:
            nav_efficiency = 0.0

        # Composite score (weighted)
        score = (
            avg_ci * 25 +
            receptor_health * 20 +
            source_coverage * 25 +
            coordination * 15 +
            nav_efficiency * 15
        )
        score = max(0.0, min(100.0, score * 100))

        # Recommendations
        if avg_ci < 0.2:
            recommendations.append("Low chemotactic index — agents are not navigating efficiently toward sources")
        if receptor_health < 0.4:
            recommendations.append("Receptor desensitization detected — consider reducing chemical concentrations")
        if source_coverage < 0.5:
            recommendations.append("Poor source coverage — many sources remain undiscovered by agents")
        if coordination < 0.3:
            recommendations.append("Weak collective sensing — increase agent count or sensing frequency")
        if nav_efficiency < 0.1:
            recommendations.append("Navigation is highly random — gradient signals may be too weak")
        if not recommendations:
            recommendations.append("Swarm chemotaxis is performing well — all metrics in healthy range")

        return FleetHealth(
            score=round(score, 1),
            avg_chemotactic_index=round(avg_ci, 3),
            receptor_health=round(receptor_health, 3),
            source_coverage=round(source_coverage, 3),
            coordination_score=round(coordination, 3),
            navigation_efficiency=round(nav_efficiency, 3),
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

def _generate_html(report: ChemotaxisReport) -> str:
    """Generate interactive HTML dashboard."""
    h = report.health
    loc = report.localization

    score_color = "#22c55e" if h.score >= 70 else "#f59e0b" if h.score >= 40 else "#ef4444"

    # Build trajectory SVG paths
    svg_size = 400
    scale = svg_size / report.grid_size
    trail_paths = []
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
              "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1"]

    for i, traj in enumerate(report.trajectories):
        color = colors[i % len(colors)]
        if len(traj.positions) > 1:
            points = " ".join(f"{p[0]*scale:.1f},{p[1]*scale:.1f}"
                              for p in traj.positions[::3])  # subsample
            trail_paths.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                f'stroke-width="1.5" opacity="0.7"/>'
            )
            # Start marker
            sx, sy = traj.positions[0][0] * scale, traj.positions[0][1] * scale
            trail_paths.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="{color}"/>')
            # End marker
            ex, ey = traj.positions[-1][0] * scale, traj.positions[-1][1] * scale
            trail_paths.append(
                f'<rect x="{ex-3:.1f}" y="{ey-3:.1f}" width="6" height="6" fill="{color}"/>'
            )

    # Source markers
    source_markers = []
    source_colors = {
        ChemicalType.ATTRACTANT: "#22c55e",
        ChemicalType.REPELLENT: "#ef4444",
        ChemicalType.NUTRIENT: "#3b82f6",
        ChemicalType.TOXIN: "#8b5cf6",
        ChemicalType.SIGNALING: "#f59e0b",
        ChemicalType.BEACON: "#ec4899",
    }
    for det in loc.sources:
        sc = source_colors.get(det.chemical, "#888")
        dx, dy = det.estimated_x * scale, det.estimated_y * scale
        source_markers.append(
            f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="8" fill="none" '
            f'stroke="{sc}" stroke-width="2" stroke-dasharray="4"/>'
        )
        source_markers.append(
            f'<text x="{dx:.1f}" y="{dy + 15:.1f}" text-anchor="middle" '
            f'font-size="10" fill="{sc}">{det.chemical.value} ({det.confidence:.0%})</text>'
        )

    trails_svg = "\n".join(trail_paths)
    sources_svg = "\n".join(source_markers)

    # Agent table rows
    agent_rows = ""
    for traj in report.trajectories:
        ci_color = "#22c55e" if traj.chemotactic_index > 0.3 else "#f59e0b" if traj.chemotactic_index > 0.15 else "#ef4444"
        agent_rows += f"""<tr>
            <td>{html_mod.escape(traj.agent_id)}</td>
            <td style="color:{ci_color}">{traj.chemotactic_index:.3f}</td>
            <td>{traj.total_distance:.1f}</td>
            <td>{traj.net_displacement:.1f}</td>
            <td>{traj.run_tumble_ratio:.2f}</td>
            <td>{traj.sources_reached}</td>
        </tr>"""

    # Detected sources table
    det_rows = ""
    for det in loc.sources:
        det_rows += f"""<tr>
            <td>{det.chemical.value}</td>
            <td>({det.estimated_x:.1f}, {det.estimated_y:.1f})</td>
            <td>{det.estimated_strength:.1f}</td>
            <td>{det.confidence:.1%}</td>
            <td>{det.contributing_agents}</td>
            <td>{det.actual_distance if det.actual_distance is not None else 'N/A'}</td>
        </tr>"""

    recs_html = "".join(f"<li>{html_mod.escape(r)}</li>" for r in h.recommendations)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Swarm Chemotaxis Report</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}}
  .header{{text-align:center;margin-bottom:32px}}
  .header h1{{font-size:28px;margin-bottom:8px}}
  .score-ring{{display:inline-block;width:120px;height:120px;border-radius:50%;
    border:8px solid {score_color};line-height:104px;text-align:center;font-size:32px;
    font-weight:bold;color:{score_color};margin:16px 0}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px;margin-bottom:24px}}
  .card{{background:#1e293b;border-radius:12px;padding:20px;border:1px solid #334155}}
  .card h2{{font-size:16px;color:#94a3b8;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px}}
  .metric{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #334155}}
  .metric:last-child{{border-bottom:none}}
  .metric .label{{color:#94a3b8}}
  .metric .value{{font-weight:600}}
  table{{width:100%;border-collapse:collapse;margin-top:8px}}
  th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
  th{{color:#94a3b8;font-size:12px;text-transform:uppercase}}
  svg{{background:#1e293b;border-radius:8px;border:1px solid #334155}}
  ul{{list-style:none;padding:0}}
  li{{padding:6px 0;border-bottom:1px solid #334155}}
  li:before{{content:"→ ";color:#3b82f6}}
</style></head><body>
<div class="header">
  <h1>🧬 Swarm Chemotaxis Report</h1>
  <p>Grid: {report.grid_size}×{report.grid_size} | Agents: {len(report.trajectories)} |
     Ticks: {report.total_ticks} | Sources: {report.num_sources}</p>
  <div class="score-ring">{h.score:.0f}</div>
  <p style="color:{score_color};font-size:14px">Navigation Health Score</p>
</div>
<div class="grid">
  <div class="card">
    <h2>Fleet Metrics</h2>
    <div class="metric"><span class="label">Chemotactic Index</span><span class="value">{h.avg_chemotactic_index:.3f}</span></div>
    <div class="metric"><span class="label">Receptor Health</span><span class="value">{h.receptor_health:.3f}</span></div>
    <div class="metric"><span class="label">Source Coverage</span><span class="value">{h.source_coverage:.1%}</span></div>
    <div class="metric"><span class="label">Coordination</span><span class="value">{h.coordination_score:.3f}</span></div>
    <div class="metric"><span class="label">Nav Efficiency</span><span class="value">{h.navigation_efficiency:.3f}</span></div>
    <div class="metric"><span class="label">Fleet CI</span><span class="value">{report.fleet_ci:.3f}</span></div>
  </div>
  <div class="card">
    <h2>Recommendations</h2>
    <ul>{recs_html}</ul>
  </div>
</div>
<div class="grid">
  <div class="card" style="grid-column:1/-1">
    <h2>Agent Trajectories & Detected Sources</h2>
    <svg width="{svg_size}" height="{svg_size}" viewBox="0 0 {svg_size} {svg_size}">
      <rect width="{svg_size}" height="{svg_size}" fill="#0f172a"/>
      {trails_svg}
      {sources_svg}
    </svg>
  </div>
</div>
<div class="grid">
  <div class="card" style="grid-column:1/-1">
    <h2>Agent Performance</h2>
    <table>
      <tr><th>Agent</th><th>CI</th><th>Distance</th><th>Displacement</th><th>Run/Tumble</th><th>Sources</th></tr>
      {agent_rows}
    </table>
  </div>
</div>
<div class="grid">
  <div class="card" style="grid-column:1/-1">
    <h2>Detected Sources (Collective Localization)</h2>
    <table>
      <tr><th>Chemical</th><th>Est. Position</th><th>Est. Strength</th><th>Confidence</th><th>Agents</th><th>Error Dist</th></tr>
      {det_rows}
    </table>
  </div>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class SwarmChemotaxisEngine:
    """Orchestrates chemotactic simulation, localization, and health assessment."""

    def __init__(
        self,
        grid_size: int = 30,
        num_agents: int = 8,
        diffusion_rates: Optional[Dict[ChemicalType, float]] = None,
        decay_rates: Optional[Dict[ChemicalType, float]] = None,
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.grid = ChemicalGrid(grid_size, diffusion_rates, decay_rates)
        self.sources: List[ChemicalSource] = []
        self.agents: List[ChemotacticAgent] = []
        self._tick = 0
        self._report: Optional[ChemotaxisReport] = None

        # Spawn agents at random positions
        for i in range(num_agents):
            ax = random.uniform(0, grid_size)
            ay = random.uniform(0, grid_size)
            self.agents.append(ChemotacticAgent(f"agent-{i+1}", ax, ay, grid_size))

    def add_source(
        self,
        x: int,
        y: int,
        chemical: ChemicalType = ChemicalType.ATTRACTANT,
        strength: float = 5.0,
    ) -> ChemicalSource:
        """Add a chemical source to the environment."""
        src = ChemicalSource(x=x % self.grid_size, y=y % self.grid_size,
                             chemical=chemical, strength=strength)
        self.sources.append(src)
        return src

    def tick(self) -> None:
        """Advance simulation by one step."""
        self._tick += 1
        self.grid.step(self.sources)
        for agent in self.agents:
            agent.move(self.grid, self._tick)
            agent.check_sources(self.sources)

    def simulate(self, steps: int = 200) -> ChemotaxisReport:
        """Run full simulation and generate report."""
        for _ in range(steps):
            self.tick()

        # Collect results
        trajectories = [a.trajectory() for a in self.agents]
        all_measurements = []
        for a in self.agents:
            all_measurements.extend(a.measurements)

        localizer = SourceLocalizer(self.grid_size)
        localization = localizer.localize(all_measurements, self.sources)

        assessor = HealthAssessor()
        health = assessor.assess(trajectories, localization, self.grid, self.sources)

        fleet_ci = statistics.mean(t.chemotactic_index for t in trajectories) if trajectories else 0.0

        self._report = ChemotaxisReport(
            health=health,
            localization=localization,
            trajectories=trajectories,
            fleet_ci=round(fleet_ci, 3),
            grid_size=self.grid_size,
            total_ticks=self._tick,
            num_sources=len(self.sources),
            config={
                "grid_size": self.grid_size,
                "num_agents": self.num_agents,
                "sources": len(self.sources),
                "steps": steps,
            },
        )
        return self._report

    def export_html(self, path: str) -> None:
        """Export interactive HTML dashboard."""
        report = self._report
        if report is None:
            report = self.simulate()
        html_str = _generate_html(report)
        Path(path).write_text(html_str, encoding="utf-8")

    def export_json(self, path: str) -> None:
        """Export report as JSON."""
        report = self._report
        if report is None:
            report = self.simulate()

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return obj

        data = asdict(report)
        Path(path).write_text(
            json.dumps(data, indent=2, default=_serialize),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Demo Simulation
# ---------------------------------------------------------------------------

def run_demo(
    grid_size: int = 30,
    num_agents: int = 8,
    steps: int = 200,
) -> ChemotaxisReport:
    """Run a demo simulation with preset sources."""
    engine = SwarmChemotaxisEngine(grid_size=grid_size, num_agents=num_agents)

    # Place interesting sources
    engine.add_source(x=int(grid_size * 0.8), y=int(grid_size * 0.8),
                      chemical=ChemicalType.ATTRACTANT, strength=5.0)
    engine.add_source(x=int(grid_size * 0.2), y=int(grid_size * 0.2),
                      chemical=ChemicalType.REPELLENT, strength=3.0)
    engine.add_source(x=int(grid_size * 0.5), y=int(grid_size * 0.7),
                      chemical=ChemicalType.NUTRIENT, strength=4.0)
    engine.add_source(x=int(grid_size * 0.7), y=int(grid_size * 0.3),
                      chemical=ChemicalType.BEACON, strength=6.0)
    engine.add_source(x=int(grid_size * 0.3), y=int(grid_size * 0.6),
                      chemical=ChemicalType.SIGNALING, strength=2.0)

    return engine.simulate(steps=steps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm Chemotaxis Engine — autonomous chemical gradient navigation"
    )
    parser.add_argument("--grid", type=int, default=30, help="Grid size (default: 30)")
    parser.add_argument("--agents", type=int, default=8, help="Number of agents (default: 8)")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps (default: 200)")
    parser.add_argument("--out", type=str, help="Output HTML file path")
    parser.add_argument("--json", type=str, help="Output JSON file path")
    args = parser.parse_args()

    print("🧬 Swarm Chemotaxis Engine")
    print("=" * 50)
    print(f"Grid: {args.grid}×{args.grid} | Agents: {args.agents} | Steps: {args.steps}")
    print()

    engine = SwarmChemotaxisEngine(grid_size=args.grid, num_agents=args.agents)

    # Place demo sources
    engine.add_source(x=int(args.grid * 0.8), y=int(args.grid * 0.8),
                      chemical=ChemicalType.ATTRACTANT, strength=5.0)
    engine.add_source(x=int(args.grid * 0.2), y=int(args.grid * 0.2),
                      chemical=ChemicalType.REPELLENT, strength=3.0)
    engine.add_source(x=int(args.grid * 0.5), y=int(args.grid * 0.7),
                      chemical=ChemicalType.NUTRIENT, strength=4.0)
    engine.add_source(x=int(args.grid * 0.7), y=int(args.grid * 0.3),
                      chemical=ChemicalType.BEACON, strength=6.0)
    engine.add_source(x=int(args.grid * 0.3), y=int(args.grid * 0.6),
                      chemical=ChemicalType.SIGNALING, strength=2.0)

    report = engine.simulate(steps=args.steps)

    h = report.health
    score_icon = "✅" if h.score >= 70 else "⚠️" if h.score >= 40 else "❌"
    print(f"{score_icon} Navigation Health: {h.score:.0f}/100")
    print(f"   Chemotactic Index: {h.avg_chemotactic_index:.3f}")
    print(f"   Receptor Health: {h.receptor_health:.3f}")
    print(f"   Source Coverage: {h.source_coverage:.1%}")
    print(f"   Coordination: {h.coordination_score:.3f}")
    print(f"   Nav Efficiency: {h.navigation_efficiency:.3f}")
    print()

    print("📍 Detected Sources:")
    for det in report.localization.sources:
        print(f"   {det.chemical.value} at ({det.estimated_x:.1f}, {det.estimated_y:.1f}) "
              f"conf={det.confidence:.1%} agents={det.contributing_agents} "
              f"error={det.actual_distance}")
    print(f"   Convergence: {report.localization.convergence_score:.3f}")
    print(f"   Triangulation Accuracy: {report.localization.triangulation_accuracy:.1f}")
    print()

    print("🧬 Agent Performance:")
    for traj in report.trajectories:
        print(f"   {traj.agent_id}: CI={traj.chemotactic_index:.3f} "
              f"dist={traj.total_distance:.1f} disp={traj.net_displacement:.1f} "
              f"R/T={traj.run_tumble_ratio:.2f} sources={traj.sources_reached}")
    print()

    print("🧭 Recommendations:")
    for rec in h.recommendations:
        print(f"   • {rec}")

    if args.out:
        engine.export_html(args.out)
        print(f"\n📄 HTML report: {args.out}")

    if args.json:
        engine.export_json(args.json)
        print(f"📄 JSON report: {args.json}")


if __name__ == "__main__":
    main()
