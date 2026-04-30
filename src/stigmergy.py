"""Swarm Stigmergy Engine — indirect coordination through environmental traces.

Biologically-inspired by how social insects (ants, termites) coordinate
complex collective behavior without direct communication.  Agents deposit
*pheromone traces* in a shared digital environment.  Traces evaporate over
time (exponential decay), amplify through positive feedback when multiple
agents reinforce the same path, and provide gradient information to guide
future agent decisions.

Capabilities:

- **Pheromone Environment** — shared spatial/topical grid where agents
  deposit typed pheromones (attraction, repulsion, exploration, danger,
  success, resource markers).
- **Evaporation Dynamics** — configurable exponential decay with per-type
  half-lives; prevents stale trace accumulation.
- **Positive Feedback Amplifier** — superlinear reinforcement when multiple
  agents independently confirm a path/decision.
- **Gradient Navigator** — agents sense local pheromone gradients to bias
  movement toward promising regions of solution space.
- **Trace Archaeology** — historical analysis of pheromone deposition
  patterns to discover emergent highways, dead zones, and oscillations.
- **Anti-Pheromone Support** — negative markers that repel agents from
  known-bad regions, enabling collective avoidance learning.
- **Stigmergic Memory** — long-lived "monument" traces that persist across
  sessions, encoding institutional knowledge.
- **Environment Health Score** — measures trace diversity, coverage, and
  freshness to detect stagnation or chaos.
- **Interactive HTML Dashboard** — visualizes pheromone landscapes, decay
  curves, agent trails, gradient fields, and archaeology findings.

Usage (Python API)::

    from src.stigmergy import StigmergyEngine, PheromoneType

    engine = StigmergyEngine(grid_size=20, evaporation_rate=0.05)

    # Agents deposit pheromones at locations
    engine.deposit("agent-1", x=5, y=3, ptype=PheromoneType.ATTRACTION, intensity=1.0)
    engine.deposit("agent-2", x=5, y=4, ptype=PheromoneType.SUCCESS, intensity=0.8)
    engine.deposit("agent-3", x=10, y=10, ptype=PheromoneType.DANGER, intensity=1.5)

    # Advance time (triggers evaporation)
    engine.tick(steps=10)

    # Agent senses gradient to navigate
    gradient = engine.sense_gradient("agent-4", x=4, y=3)
    print(gradient)  # direction + magnitude toward strongest attraction

    # Full analysis
    report = engine.analyze()
    print(report.highways)       # emergent high-traffic corridors
    print(report.dead_zones)     # regions with no activity
    print(report.health_score)   # 0-100 environment health

    engine.export_html("stigmergy_report.html")

CLI::

    python -m src.stigmergy                       # demo with simulated agents
    python -m src.stigmergy --grid 30 --agents 12 # larger environment
    python -m src.stigmergy --steps 200           # longer simulation
    python -m src.stigmergy --out report.html --json stigmergy.json
"""
from __future__ import annotations

import argparse
import hashlib
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
# Enums & Data Models
# ---------------------------------------------------------------------------

class PheromoneType(str, Enum):
    """Types of pheromone traces agents can deposit."""
    ATTRACTION = "attraction"      # draws agents toward a location
    REPULSION = "repulsion"        # pushes agents away (anti-pheromone)
    EXPLORATION = "exploration"    # marks areas as explored
    DANGER = "danger"              # warns of hazards/failures
    SUCCESS = "success"            # marks successful outcomes
    RESOURCE = "resource"          # indicates resource availability
    MONUMENT = "monument"          # long-lived institutional knowledge


# Default half-lives (in ticks) per pheromone type
DEFAULT_HALF_LIVES: Dict[PheromoneType, float] = {
    PheromoneType.ATTRACTION: 20.0,
    PheromoneType.REPULSION: 15.0,
    PheromoneType.EXPLORATION: 30.0,
    PheromoneType.DANGER: 25.0,
    PheromoneType.SUCCESS: 40.0,
    PheromoneType.RESOURCE: 35.0,
    PheromoneType.MONUMENT: 200.0,  # very long-lived
}


@dataclass
class PheromoneDeposit:
    """A single pheromone deposit at a location."""
    agent_id: str
    ptype: PheromoneType
    intensity: float
    x: int
    y: int
    tick_deposited: int
    tick_last_reinforced: int = 0
    reinforcement_count: int = 1

    @property
    def deposit_id(self) -> str:
        return hashlib.md5(
            f"{self.agent_id}:{self.ptype.value}:{self.x},{self.y}:{self.tick_deposited}".encode()
        ).hexdigest()[:12]


@dataclass
class GradientVector:
    """Directional pheromone gradient sensed by an agent."""
    dx: float
    dy: float
    magnitude: float
    dominant_type: PheromoneType
    attractors: int
    repulsors: int

    @property
    def direction_degrees(self) -> float:
        return math.degrees(math.atan2(self.dy, self.dx)) % 360


@dataclass
class Highway:
    """An emergent high-traffic corridor discovered in trace archaeology."""
    path: List[Tuple[int, int]]
    total_intensity: float
    unique_agents: int
    dominant_type: PheromoneType
    persistence_ticks: int


@dataclass
class DeadZone:
    """A region with no or negligible pheromone activity."""
    center_x: int
    center_y: int
    radius: float
    cells: int
    isolation_ticks: int


@dataclass
class Oscillation:
    """A detected oscillation in pheromone levels at a location."""
    x: int
    y: int
    ptype: PheromoneType
    period_ticks: float
    amplitude: float
    cycle_count: int


@dataclass
class ArchaeologyReport:
    """Results of trace archaeology analysis."""
    highways: List[Highway]
    dead_zones: List[DeadZone]
    oscillations: List[Oscillation]
    total_deposits: int
    active_deposits: int
    evaporated_deposits: int
    coverage_pct: float  # % of grid with any active trace
    diversity_score: float  # Shannon entropy of type distribution
    freshness_score: float  # avg recency of active deposits


@dataclass
class EnvironmentHealth:
    """Overall environment health assessment."""
    score: float  # 0-100
    coverage_pct: float
    diversity_score: float
    freshness_score: float
    stagnation_risk: float  # 0-1
    chaos_risk: float  # 0-1
    recommendations: List[str]


@dataclass
class AgentTrail:
    """Complete movement trail of an agent through the environment."""
    agent_id: str
    positions: List[Tuple[int, int, int]]  # (x, y, tick)
    deposits: List[PheromoneDeposit]
    total_distance: float
    exploration_coverage: float  # % of grid visited


@dataclass
class StigmergyReport:
    """Complete analysis report."""
    archaeology: ArchaeologyReport
    health: EnvironmentHealth
    trails: List[AgentTrail]
    grid_size: int
    current_tick: int
    config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Pheromone Grid
# ---------------------------------------------------------------------------

class PheromoneGrid:
    """2D grid storing pheromone concentrations with evaporation."""

    def __init__(self, size: int, half_lives: Optional[Dict[PheromoneType, float]] = None):
        self.size = size
        self.half_lives = half_lives or dict(DEFAULT_HALF_LIVES)
        # grid[y][x][ptype] = current_intensity
        self._grid: List[List[Dict[PheromoneType, float]]] = [
            [{} for _ in range(size)] for _ in range(size)
        ]
        self._deposits: List[PheromoneDeposit] = []
        self._history: List[List[Dict[PheromoneType, List[float]]]] = [
            [{} for _ in range(size)] for _ in range(size)
        ]

    def deposit(self, dep: PheromoneDeposit) -> None:
        """Add a pheromone deposit to the grid."""
        x, y = dep.x % self.size, dep.y % self.size
        cell = self._grid[y][x]
        current = cell.get(dep.ptype, 0.0)
        # Superlinear reinforcement: if already present, amplify
        if current > 0:
            boost = 1.0 + 0.3 * math.log1p(current)
            cell[dep.ptype] = current + dep.intensity * boost
            dep.reinforcement_count += 1
            dep.tick_last_reinforced = dep.tick_deposited
        else:
            cell[dep.ptype] = dep.intensity
        self._deposits.append(dep)
        # Record in history
        hist = self._history[y][x]
        if dep.ptype not in hist:
            hist[dep.ptype] = []
        hist[dep.ptype].append(cell[dep.ptype])

    def evaporate(self, ticks: int = 1) -> int:
        """Apply exponential decay to all pheromones. Returns count of fully evaporated cells."""
        evaporated = 0
        for y in range(self.size):
            for x in range(self.size):
                cell = self._grid[y][x]
                to_remove = []
                for ptype, intensity in cell.items():
                    half_life = self.half_lives.get(ptype, 20.0)
                    decay = math.exp(-0.693 * ticks / half_life)
                    new_val = intensity * decay
                    if new_val < 0.01:
                        to_remove.append(ptype)
                        evaporated += 1
                    else:
                        cell[ptype] = new_val
                for pt in to_remove:
                    del cell[pt]
        return evaporated

    def read(self, x: int, y: int) -> Dict[PheromoneType, float]:
        """Read all pheromone concentrations at a cell."""
        return dict(self._grid[y % self.size][x % self.size])

    def read_type(self, x: int, y: int, ptype: PheromoneType) -> float:
        """Read a specific pheromone type at a cell."""
        return self._grid[y % self.size][x % self.size].get(ptype, 0.0)

    def total_intensity_at(self, x: int, y: int) -> float:
        """Sum of all pheromone intensities at a cell."""
        return sum(self._grid[y % self.size][x % self.size].values())

    def coverage(self) -> float:
        """Fraction of grid cells with any active pheromone."""
        active = sum(
            1 for y in range(self.size) for x in range(self.size)
            if self._grid[y][x]
        )
        return active / (self.size * self.size)

    def type_distribution(self) -> Dict[PheromoneType, float]:
        """Total intensity per pheromone type across entire grid."""
        dist: Dict[PheromoneType, float] = defaultdict(float)
        for y in range(self.size):
            for x in range(self.size):
                for pt, val in self._grid[y][x].items():
                    dist[pt] += val
        return dict(dist)

    @property
    def deposits(self) -> List[PheromoneDeposit]:
        return self._deposits

    @property
    def history(self):
        return self._history


# ---------------------------------------------------------------------------
# Gradient Computation
# ---------------------------------------------------------------------------

class GradientComputer:
    """Computes pheromone gradients for agent navigation."""

    def __init__(self, grid: PheromoneGrid):
        self.grid = grid

    def compute(self, x: int, y: int, sense_radius: int = 3) -> GradientVector:
        """Compute combined gradient at position considering all nearby pheromones."""
        dx_total = 0.0
        dy_total = 0.0
        attractors = 0
        repulsors = 0
        type_magnitudes: Dict[PheromoneType, float] = defaultdict(float)

        for dy in range(-sense_radius, sense_radius + 1):
            for ddx in range(-sense_radius, sense_radius + 1):
                if dy == 0 and ddx == 0:
                    continue
                nx = (x + ddx) % self.grid.size
                ny = (y + dy) % self.grid.size
                dist = math.sqrt(ddx * ddx + dy * dy)
                if dist > sense_radius:
                    continue

                cell = self.grid.read(nx, ny)
                for ptype, intensity in cell.items():
                    # Falloff with distance
                    effective = intensity / (1 + dist)
                    type_magnitudes[ptype] += effective

                    # Determine direction contribution
                    if ptype in (PheromoneType.REPULSION, PheromoneType.DANGER):
                        # Repel: push away from source
                        dx_total -= (ddx / dist) * effective
                        dy_total -= (dy / dist) * effective
                        repulsors += 1
                    else:
                        # Attract: pull toward source
                        dx_total += (ddx / dist) * effective
                        dy_total += (dy / dist) * effective
                        attractors += 1

        magnitude = math.sqrt(dx_total * dx_total + dy_total * dy_total)
        # Normalize direction
        if magnitude > 0:
            dx_total /= magnitude
            dy_total /= magnitude

        # Determine dominant type
        dominant = PheromoneType.ATTRACTION
        if type_magnitudes:
            dominant = max(type_magnitudes, key=lambda k: type_magnitudes[k])

        return GradientVector(
            dx=dx_total,
            dy=dy_total,
            magnitude=magnitude,
            dominant_type=dominant,
            attractors=attractors,
            repulsors=repulsors,
        )


# ---------------------------------------------------------------------------
# Trace Archaeology
# ---------------------------------------------------------------------------

class TraceArchaeologist:
    """Analyzes historical pheromone patterns to discover emergent structures."""

    def __init__(self, grid: PheromoneGrid):
        self.grid = grid

    # -- Shared BFS clustering ---------------------------------------------

    def _bfs_clusters(
        self,
        seed_cells: Set[Tuple[int, int]],
        neighbor_predicate: Optional[Any] = None,
        min_size: int = 1,
    ) -> List[List[Tuple[int, int]]]:
        """BFS-cluster a set of seed cells on the toroidal grid.

        Parameters
        ----------
        seed_cells : set of (x, y) coordinates eligible for clustering.
        neighbor_predicate : optional callable (x, y) -> bool applied to
            neighbors before adding them to the frontier.  If *None*, only
            membership in *seed_cells* is checked.
        min_size : discard clusters smaller than this.

        Returns a list of clusters (each a list of (x, y) tuples).
        """
        size = self.grid.size
        visited: Set[Tuple[int, int]] = set()
        clusters: List[List[Tuple[int, int]]] = []

        for cell in seed_cells:
            if cell in visited:
                continue
            cluster: List[Tuple[int, int]] = []
            queue: List[Tuple[int, int]] = [cell]
            while queue:
                c = queue.pop(0)
                if c in visited:
                    continue
                # If a neighbor predicate is provided, apply it;
                # otherwise require membership in seed_cells.
                if neighbor_predicate is not None:
                    if not neighbor_predicate(c[0], c[1]):
                        continue
                elif c not in seed_cells:
                    continue
                visited.add(c)
                cluster.append(c)
                cx, cy = c
                for ndx, ndy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = (cx + ndx) % size, (cy + ndy) % size
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))
            if len(cluster) >= min_size:
                clusters.append(cluster)
        return clusters

    # -- Public analysis methods -------------------------------------------

    def find_highways(self, threshold: float = 3.0) -> List[Highway]:
        """Discover high-intensity corridors (connected cells above threshold)."""
        size = self.grid.size

        # Seeds: cells meeting the full threshold
        seeds: Set[Tuple[int, int]] = set()
        for y in range(size):
            for x in range(size):
                if self.grid.total_intensity_at(x, y) >= threshold:
                    seeds.add((x, y))

        # BFS expands into neighbors above half-threshold (corridor edges)
        half_thresh = threshold * 0.5

        def _highway_pred(x: int, y: int) -> bool:
            return self.grid.total_intensity_at(x, y) >= half_thresh

        clusters = self._bfs_clusters(seeds, neighbor_predicate=_highway_pred, min_size=3)

        highways: List[Highway] = []
        for path in clusters:
            total_int = sum(self.grid.total_intensity_at(px, py) for px, py in path)
            path_set = set(path)
            agents: Set[str] = set()
            dom_types: Dict[PheromoneType, float] = defaultdict(float)
            for dep in self.grid.deposits:
                if (dep.x % size, dep.y % size) in path_set:
                    agents.add(dep.agent_id)
                    dom_types[dep.ptype] += dep.intensity
            dominant = max(dom_types, key=lambda k: dom_types[k]) if dom_types else PheromoneType.ATTRACTION
            highways.append(Highway(
                path=path,
                total_intensity=total_int,
                unique_agents=len(agents),
                dominant_type=dominant,
                persistence_ticks=len(path) * 5,
            ))
        return highways

    def find_dead_zones(self, min_cells: int = 4) -> List[DeadZone]:
        """Find regions with no pheromone activity."""
        size = self.grid.size
        dead_cells: Set[Tuple[int, int]] = set()
        for y in range(size):
            for x in range(size):
                if self.grid.total_intensity_at(x, y) < 0.01:
                    dead_cells.add((x, y))

        clusters = self._bfs_clusters(dead_cells, min_size=min_cells)

        zones: List[DeadZone] = []
        for cluster in clusters:
            avg_x = sum(c[0] for c in cluster) / len(cluster)
            avg_y = sum(c[1] for c in cluster) / len(cluster)
            radius = max(math.sqrt((c[0] - avg_x) ** 2 + (c[1] - avg_y) ** 2) for c in cluster)
            zones.append(DeadZone(
                center_x=int(avg_x),
                center_y=int(avg_y),
                radius=radius,
                cells=len(cluster),
                isolation_ticks=0,
            ))
        return zones

    def detect_oscillations(self, min_cycles: int = 2) -> List[Oscillation]:
        """Detect periodic fluctuations in pheromone levels."""
        oscillations: List[Oscillation] = []
        size = self.grid.size
        history = self.grid.history

        for y in range(size):
            for x in range(size):
                hist = history[y][x]
                for ptype, values in hist.items():
                    if len(values) < min_cycles * 4:
                        continue
                    # Simple peak detection
                    peaks = []
                    for i in range(1, len(values) - 1):
                        if values[i] > values[i - 1] and values[i] > values[i + 1]:
                            peaks.append(i)
                    if len(peaks) >= min_cycles:
                        periods = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
                        if periods:
                            avg_period = statistics.mean(periods)
                            amplitude = max(values) - min(values)
                            oscillations.append(Oscillation(
                                x=x, y=y, ptype=ptype,
                                period_ticks=avg_period,
                                amplitude=amplitude,
                                cycle_count=len(peaks),
                            ))
        return oscillations

    def full_report(self, current_tick: int) -> ArchaeologyReport:
        """Generate complete archaeology report."""
        highways = self.find_highways()
        dead_zones = self.find_dead_zones()
        oscillations = self.detect_oscillations()

        active = sum(
            1 for y in range(self.grid.size) for x in range(self.grid.size)
            if self.grid.total_intensity_at(x, y) >= 0.01
        )
        total_cells = self.grid.size * self.grid.size

        # Diversity: Shannon entropy of type distribution
        dist = self.grid.type_distribution()
        total_val = sum(dist.values()) or 1.0
        entropy = 0.0
        for val in dist.values():
            p = val / total_val
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(len(PheromoneType)) if len(PheromoneType) > 1 else 1.0
        diversity = entropy / max_entropy

        # Freshness: average recency of deposits
        if self.grid.deposits:
            avg_age = statistics.mean(current_tick - d.tick_deposited for d in self.grid.deposits)
            freshness = max(0.0, 1.0 - avg_age / max(current_tick, 1))
        else:
            freshness = 0.0

        return ArchaeologyReport(
            highways=highways,
            dead_zones=dead_zones,
            oscillations=oscillations,
            total_deposits=len(self.grid.deposits),
            active_deposits=active,
            evaporated_deposits=len(self.grid.deposits) - active,
            coverage_pct=self.grid.coverage() * 100,
            diversity_score=diversity,
            freshness_score=freshness,
        )


# ---------------------------------------------------------------------------
# Environment Health Assessor
# ---------------------------------------------------------------------------

class HealthAssessor:
    """Assesses overall environment health."""

    def assess(self, archaeology: ArchaeologyReport, grid: PheromoneGrid) -> EnvironmentHealth:
        coverage = archaeology.coverage_pct
        diversity = archaeology.diversity_score
        freshness = archaeology.freshness_score

        # Stagnation: low coverage + low freshness
        stagnation = max(0.0, 1.0 - (coverage / 100 + freshness) / 2)

        # Chaos: very high coverage + low diversity (everyone doing same thing)
        chaos = max(0.0, (coverage / 100) * (1.0 - diversity))

        # Overall score
        score = (
            coverage * 0.3 +
            diversity * 100 * 0.3 +
            freshness * 100 * 0.2 +
            (1 - stagnation) * 100 * 0.1 +
            (1 - chaos) * 100 * 0.1
        )
        score = max(0.0, min(100.0, score))

        recommendations = []
        if stagnation > 0.6:
            recommendations.append("Environment is stagnating — inject exploration pheromones or spawn scout agents")
        if chaos > 0.6:
            recommendations.append("Environment is chaotic — increase evaporation rates or reduce deposit intensity")
        if coverage < 20:
            recommendations.append("Low coverage — encourage agents to explore unexplored regions")
        if diversity < 0.3:
            recommendations.append("Low diversity — agents are using too few pheromone types")
        if freshness < 0.3:
            recommendations.append("Low freshness — deposits are aging; increase agent activity")
        if not recommendations:
            recommendations.append("Environment is healthy — maintain current parameters")

        return EnvironmentHealth(
            score=score,
            coverage_pct=coverage,
            diversity_score=diversity,
            freshness_score=freshness,
            stagnation_risk=stagnation,
            chaos_risk=chaos,
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Simulated Agent
# ---------------------------------------------------------------------------

class StigmergicAgent:
    """An agent that navigates using pheromone gradients."""

    def __init__(self, agent_id: str, x: int, y: int, grid_size: int):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.grid_size = grid_size
        self.trail: List[Tuple[int, int, int]] = []
        self.deposits_made: List[PheromoneDeposit] = []
        self.explore_bias: float = random.uniform(0.1, 0.4)
        self.success_count: int = 0
        self.steps_taken: int = 0

    def move(self, gradient: GradientVector, tick: int) -> Tuple[int, int]:
        """Move based on gradient with some randomness."""
        self.trail.append((self.x, self.y, tick))
        self.steps_taken += 1

        if random.random() < self.explore_bias:
            # Random exploration
            self.x = (self.x + random.randint(-1, 1)) % self.grid_size
            self.y = (self.y + random.randint(-1, 1)) % self.grid_size
        elif gradient.magnitude > 0.1:
            # Follow gradient
            self.x = (self.x + int(round(gradient.dx))) % self.grid_size
            self.y = (self.y + int(round(gradient.dy))) % self.grid_size
        else:
            # Weak gradient: random walk
            self.x = (self.x + random.randint(-1, 1)) % self.grid_size
            self.y = (self.y + random.randint(-1, 1)) % self.grid_size

        return self.x, self.y

    def decide_deposit(self, tick: int) -> Optional[PheromoneDeposit]:
        """Decide whether and what to deposit at current location."""
        # Simple heuristic: deposit based on state
        if random.random() > 0.6:
            return None

        if self.steps_taken % 10 == 0:
            ptype = PheromoneType.EXPLORATION
            intensity = 0.5
        elif random.random() < 0.2:
            ptype = PheromoneType.DANGER if random.random() < 0.3 else PheromoneType.ATTRACTION
            intensity = random.uniform(0.3, 1.2)
        else:
            ptype = random.choice([PheromoneType.ATTRACTION, PheromoneType.SUCCESS, PheromoneType.RESOURCE])
            intensity = random.uniform(0.4, 1.0)

        dep = PheromoneDeposit(
            agent_id=self.agent_id,
            ptype=ptype,
            intensity=intensity,
            x=self.x,
            y=self.y,
            tick_deposited=tick,
        )
        self.deposits_made.append(dep)
        return dep

    def to_trail(self) -> AgentTrail:
        """Export agent trail data."""
        if len(self.trail) < 2:
            total_dist = 0.0
        else:
            total_dist = sum(
                math.sqrt((self.trail[i][0] - self.trail[i - 1][0]) ** 2 +
                           (self.trail[i][1] - self.trail[i - 1][1]) ** 2)
                for i in range(1, len(self.trail))
            )
        visited = set((t[0], t[1]) for t in self.trail)
        coverage = len(visited) / (self.grid_size * self.grid_size)
        return AgentTrail(
            agent_id=self.agent_id,
            positions=self.trail,
            deposits=self.deposits_made,
            total_distance=total_dist,
            exploration_coverage=coverage,
        )


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class StigmergyEngine:
    """Orchestrates the stigmergic environment and agent interactions."""

    def __init__(
        self,
        grid_size: int = 20,
        evaporation_rate: float = 0.05,
        half_lives: Optional[Dict[PheromoneType, float]] = None,
        sense_radius: int = 3,
    ):
        self.grid_size = grid_size
        self.evaporation_rate = evaporation_rate
        hl = half_lives or dict(DEFAULT_HALF_LIVES)
        # Apply global rate modifier
        modified_hl = {k: v / max(evaporation_rate * 20, 0.1) for k, v in hl.items()}
        self.grid = PheromoneGrid(grid_size, modified_hl)
        self.gradient_computer = GradientComputer(self.grid)
        self.archaeologist = TraceArchaeologist(self.grid)
        self.health_assessor = HealthAssessor()
        self.sense_radius = sense_radius
        self.current_tick = 0
        self.agents: List[StigmergicAgent] = []

    def deposit(self, agent_id: str, x: int, y: int,
                ptype: PheromoneType, intensity: float = 1.0) -> PheromoneDeposit:
        """Manually deposit a pheromone at a location."""
        dep = PheromoneDeposit(
            agent_id=agent_id,
            ptype=ptype,
            intensity=intensity,
            x=x,
            y=y,
            tick_deposited=self.current_tick,
        )
        self.grid.deposit(dep)
        return dep

    def tick(self, steps: int = 1) -> int:
        """Advance environment by N ticks, applying evaporation."""
        total_evaporated = 0
        for _ in range(steps):
            self.current_tick += 1
            total_evaporated += self.grid.evaporate(1)
        return total_evaporated

    def sense_gradient(self, agent_id: str, x: int, y: int) -> GradientVector:
        """Sense the pheromone gradient at a position."""
        return self.gradient_computer.compute(x, y, self.sense_radius)

    def add_agent(self, agent_id: Optional[str] = None) -> StigmergicAgent:
        """Add a new agent at a random position."""
        aid = agent_id or f"agent-{len(self.agents) + 1}"
        agent = StigmergicAgent(
            agent_id=aid,
            x=random.randint(0, self.grid_size - 1),
            y=random.randint(0, self.grid_size - 1),
            grid_size=self.grid_size,
        )
        self.agents.append(agent)
        return agent

    def simulate(self, steps: int = 100, num_agents: int = 8) -> StigmergyReport:
        """Run a full simulation with agents."""
        # Initialize agents if none exist
        if not self.agents:
            for i in range(num_agents):
                self.add_agent()

        for step in range(steps):
            self.current_tick += 1
            # Each agent: sense, move, possibly deposit
            for agent in self.agents:
                gradient = self.gradient_computer.compute(agent.x, agent.y, self.sense_radius)
                agent.move(gradient, self.current_tick)
                dep = agent.decide_deposit(self.current_tick)
                if dep:
                    dep.tick_deposited = self.current_tick
                    self.grid.deposit(dep)

            # Evaporate
            self.grid.evaporate(1)

        return self.analyze()

    def analyze(self) -> StigmergyReport:
        """Generate complete analysis report."""
        archaeology = self.archaeologist.full_report(self.current_tick)
        health = self.health_assessor.assess(archaeology, self.grid)
        trails = [a.to_trail() for a in self.agents]

        return StigmergyReport(
            archaeology=archaeology,
            health=health,
            trails=trails,
            grid_size=self.grid_size,
            current_tick=self.current_tick,
            config={
                "evaporation_rate": self.evaporation_rate,
                "sense_radius": self.sense_radius,
                "half_lives": {k.value: v for k, v in self.grid.half_lives.items()},
            },
        )

    def export_json(self, path: str) -> None:
        """Export report as JSON."""
        report = self.analyze()

        def serialize(obj: Any) -> Any:
            if hasattr(obj, '__dict__'):
                d = {}
                for k, v in obj.__dict__.items():
                    if k.startswith('_'):
                        continue
                    d[k] = serialize(v)
                return d
            if isinstance(obj, list):
                return [serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {str(k): serialize(v) for k, v in obj.items()}
            if isinstance(obj, Enum):
                return obj.value
            return obj

        data = serialize(report)
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def export_html(self, path: str) -> None:
        """Export interactive HTML dashboard."""
        report = self.analyze()
        html_content = _generate_html(report, self.grid)
        Path(path).write_text(html_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML Dashboard Generator
# ---------------------------------------------------------------------------

def _generate_html(report: StigmergyReport, grid: PheromoneGrid) -> str:
    """Generate interactive HTML dashboard."""
    # Build heatmap data
    size = report.grid_size
    heatmap_data = []
    for y in range(size):
        row = []
        for x in range(size):
            row.append(round(grid.total_intensity_at(x, y), 2))
        heatmap_data.append(row)

    # Type distribution
    dist = grid.type_distribution()
    type_labels = [t.value for t in PheromoneType]
    type_values = [round(dist.get(t, 0), 2) for t in PheromoneType]

    # Trail data (simplified)
    trail_data = []
    for trail in report.trails[:10]:
        trail_data.append({
            "id": trail.agent_id,
            "positions": trail.positions[:50],
            "distance": round(trail.total_distance, 1),
            "coverage": round(trail.exploration_coverage * 100, 1),
        })

    health = report.health
    arch = report.archaeology

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Stigmergy Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 20px; }}
h1 {{ text-align: center; color: #00ff88; margin-bottom: 5px; font-size: 1.8em; }}
.subtitle {{ text-align: center; color: #888; margin-bottom: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 16px; max-width: 1400px; margin: 0 auto; }}
.card {{ background: #1a1a2e; border-radius: 12px; padding: 20px; border: 1px solid #333; }}
.card h2 {{ color: #00cc6a; margin-bottom: 12px; font-size: 1.1em; }}
.score-big {{ font-size: 3em; font-weight: bold; text-align: center; margin: 10px 0; }}
.score-good {{ color: #00ff88; }}
.score-warn {{ color: #ffaa00; }}
.score-bad {{ color: #ff4444; }}
.metric {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #222; }}
.metric-label {{ color: #999; }}
.metric-value {{ font-weight: bold; }}
.bar {{ height: 8px; border-radius: 4px; background: #333; margin-top: 4px; }}
.bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
.heatmap {{ display: grid; gap: 1px; margin: 10px 0; }}
.heatmap-cell {{ aspect-ratio: 1; border-radius: 2px; min-width: 4px; }}
.tag {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; margin: 2px; background: #2a2a4e; }}
.rec {{ padding: 8px; margin: 4px 0; background: #1e2a1e; border-left: 3px solid #00cc6a; border-radius: 4px; font-size: 0.9em; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
th {{ text-align: left; padding: 8px; color: #00cc6a; border-bottom: 1px solid #333; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #1a1a2e; }}
.tabs {{ display: flex; gap: 4px; margin-bottom: 12px; flex-wrap: wrap; }}
.tab {{ padding: 6px 14px; border-radius: 6px; cursor: pointer; background: #222; border: 1px solid #444; font-size: 0.85em; }}
.tab.active {{ background: #00cc6a; color: #000; border-color: #00cc6a; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
</style>
</head>
<body>
<h1>🐜 Swarm Stigmergy Engine</h1>
<p class="subtitle">Indirect Coordination Through Environmental Traces — Tick {report.current_tick}</p>

<div class="grid">
  <div class="card">
    <h2>🏥 Environment Health</h2>
    <div class="score-big {'score-good' if health.score >= 70 else 'score-warn' if health.score >= 40 else 'score-bad'}">{health.score:.0f}</div>
    <div class="metric"><span class="metric-label">Coverage</span><span class="metric-value">{health.coverage_pct:.1f}%</span></div>
    <div class="bar"><div class="bar-fill" style="width:{health.coverage_pct}%;background:#00cc6a"></div></div>
    <div class="metric"><span class="metric-label">Diversity</span><span class="metric-value">{health.diversity_score:.2f}</span></div>
    <div class="bar"><div class="bar-fill" style="width:{health.diversity_score*100}%;background:#4488ff"></div></div>
    <div class="metric"><span class="metric-label">Freshness</span><span class="metric-value">{health.freshness_score:.2f}</span></div>
    <div class="bar"><div class="bar-fill" style="width:{health.freshness_score*100}%;background:#ff8844"></div></div>
    <div class="metric"><span class="metric-label">Stagnation Risk</span><span class="metric-value">{health.stagnation_risk:.2f}</span></div>
    <div class="metric"><span class="metric-label">Chaos Risk</span><span class="metric-value">{health.chaos_risk:.2f}</span></div>
  </div>

  <div class="card">
    <h2>📊 Archaeology Summary</h2>
    <div class="metric"><span class="metric-label">Highways Found</span><span class="metric-value">{len(arch.highways)}</span></div>
    <div class="metric"><span class="metric-label">Dead Zones</span><span class="metric-value">{len(arch.dead_zones)}</span></div>
    <div class="metric"><span class="metric-label">Oscillations</span><span class="metric-value">{len(arch.oscillations)}</span></div>
    <div class="metric"><span class="metric-label">Total Deposits</span><span class="metric-value">{arch.total_deposits}</span></div>
    <div class="metric"><span class="metric-label">Active Cells</span><span class="metric-value">{arch.active_deposits}</span></div>
    <div class="metric"><span class="metric-label">Evaporated</span><span class="metric-value">{arch.evaporated_deposits}</span></div>
  </div>

  <div class="card">
    <h2>🧭 Recommendations</h2>
    {''.join(f'<div class="rec">{html_mod.escape(r)}</div>' for r in health.recommendations)}
  </div>

  <div class="card">
    <h2>🐜 Agent Trails</h2>
    <table>
      <tr><th>Agent</th><th>Distance</th><th>Coverage</th><th>Deposits</th></tr>
      {''.join(f'<tr><td>{html_mod.escape(t["id"])}</td><td>{t["distance"]}</td><td>{t["coverage"]}%</td><td>{len([d for d in report.trails[i].deposits]) if i < len(report.trails) else 0}</td></tr>' for i, t in enumerate(trail_data))}
    </table>
  </div>

  <div class="card" style="grid-column: span 2">
    <h2>🗺️ Pheromone Heatmap</h2>
    <div class="heatmap" style="grid-template-columns: repeat({size}, 1fr)">
      {''.join(f'<div class="heatmap-cell" style="background:rgba(0,255,136,{min(1.0, v/5)});"></div>' for row in heatmap_data for v in row)}
    </div>
    <p style="color:#666;font-size:0.8em;margin-top:6px">Brightness = pheromone intensity. Grid: {size}×{size}</p>
  </div>

  <div class="card">
    <h2>📈 Type Distribution</h2>
    {''.join(f'<div class="metric"><span class="metric-label">{html_mod.escape(type_labels[i])}</span><span class="metric-value">{type_values[i]:.1f}</span></div><div class="bar"><div class="bar-fill" style="width:{min(100, type_values[i] / max(max(type_values), 1) * 100)}%;background:hsl({i*50},70%,50%)"></div></div>' for i in range(len(type_labels)))}
  </div>

  <div class="card">
    <h2>⚙️ Configuration</h2>
    <div class="metric"><span class="metric-label">Grid Size</span><span class="metric-value">{report.grid_size}×{report.grid_size}</span></div>
    <div class="metric"><span class="metric-label">Evaporation Rate</span><span class="metric-value">{report.config['evaporation_rate']}</span></div>
    <div class="metric"><span class="metric-label">Sense Radius</span><span class="metric-value">{report.config['sense_radius']}</span></div>
    <div class="metric"><span class="metric-label">Agents</span><span class="metric-value">{len(report.trails)}</span></div>
    <div class="metric"><span class="metric-label">Ticks Elapsed</span><span class="metric-value">{report.current_tick}</span></div>
  </div>
</div>

<script>
document.querySelectorAll('.tabs').forEach(tabGroup => {{
  tabGroup.querySelectorAll('.tab').forEach(tab => {{
    tab.addEventListener('click', () => {{
      tabGroup.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const card = tabGroup.closest('.card');
      card.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      card.querySelector('#' + tab.dataset.target)?.classList.add('active');
    }});
  }});
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _demo() -> StigmergyReport:
    """Run demonstration simulation."""
    engine = StigmergyEngine(grid_size=20, evaporation_rate=0.05)
    report = engine.simulate(steps=150, num_agents=10)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm Stigmergy Engine — indirect coordination through environmental traces"
    )
    parser.add_argument("--grid", type=int, default=20, help="Grid size (default: 20)")
    parser.add_argument("--agents", type=int, default=10, help="Number of agents (default: 10)")
    parser.add_argument("--steps", type=int, default=150, help="Simulation steps (default: 150)")
    parser.add_argument("--evaporation", type=float, default=0.05, help="Evaporation rate (default: 0.05)")
    parser.add_argument("--sense-radius", type=int, default=3, help="Sensing radius (default: 3)")
    parser.add_argument("--out", type=str, help="Output HTML file path")
    parser.add_argument("--json", type=str, help="Output JSON file path")
    args = parser.parse_args()

    print("🐜 Swarm Stigmergy Engine")
    print("=" * 50)
    print(f"Grid: {args.grid}×{args.grid} | Agents: {args.agents} | Steps: {args.steps}")
    print(f"Evaporation: {args.evaporation} | Sense radius: {args.sense_radius}")
    print()

    engine = StigmergyEngine(
        grid_size=args.grid,
        evaporation_rate=args.evaporation,
        sense_radius=args.sense_radius,
    )
    report = engine.simulate(steps=args.steps, num_agents=args.agents)

    # Print summary
    health = report.health
    arch = report.archaeology

    score_icon = "✅" if health.score >= 70 else "⚠️" if health.score >= 40 else "❌"
    print(f"{score_icon} Environment Health: {health.score:.0f}/100")
    print(f"   Coverage: {health.coverage_pct:.1f}%")
    print(f"   Diversity: {health.diversity_score:.2f}")
    print(f"   Freshness: {health.freshness_score:.2f}")
    print(f"   Stagnation Risk: {health.stagnation_risk:.2f}")
    print(f"   Chaos Risk: {health.chaos_risk:.2f}")
    print()

    print("📊 Archaeology:")
    print(f"   Highways: {len(arch.highways)}")
    print(f"   Dead Zones: {len(arch.dead_zones)}")
    print(f"   Oscillations: {len(arch.oscillations)}")
    print(f"   Total Deposits: {arch.total_deposits}")
    print(f"   Active Cells: {arch.active_deposits}/{report.grid_size**2}")
    print()

    print("🐜 Agent Performance:")
    for trail in report.trails:
        print(f"   {trail.agent_id}: dist={trail.total_distance:.1f}, "
              f"coverage={trail.exploration_coverage*100:.1f}%, "
              f"deposits={len(trail.deposits)}")
    print()

    print("🧭 Recommendations:")
    for rec in health.recommendations:
        print(f"   • {rec}")

    if args.out:
        engine.export_html(args.out)
        print(f"\n📄 HTML report: {args.out}")

    if args.json:
        engine.export_json(args.json)
        print(f"📄 JSON report: {args.json}")


if __name__ == "__main__":
    main()
