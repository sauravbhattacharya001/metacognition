"""Swarm Proprioception Engine — autonomous body-schema awareness for swarm topology.

Biologically inspired by the proprioceptive system: the sense of body position
and movement without visual input.  While nociception detects damage and
neuroplasticity adapts connections, proprioception provides **continuous
awareness of the swarm's own structural configuration** — where each agent
is relative to others, how the topology is arranged, and what movements
(reconfigurations) are occurring.

Capabilities:

- **Body Schema Builder** — constructs a dynamic internal model of the swarm's
  topology: connection distances, agent roles (core/limb/joint/endpoint),
  neighborhood density, and structural symmetry.
- **Kinesthetic Tracker** — detects movement in the topology: agents joining,
  leaving, connections forming or breaking.  Computes velocity and acceleration
  of structural change.
- **Joint Angle Sensor** — measures "angles" at junction agents (those bridging
  sub-groups), detecting flexion/extension of swarm limbs.  Identifies
  over-extension (fragile elongation) and over-compression (congestion).
- **Balance Detector** — evaluates structural balance across the swarm: center
  of mass, tilt direction, load distribution asymmetry.  Detects instability
  before collapse.
- **Postural Memory** — records stable configurations (postures) and detects
  deviations from known-good topologies.  Enables return-to-baseline reflexes.
- **Coordination Feedback** — provides real-time proprioceptive feedback loops
  that enable coordinated movement: synchronized expansion/contraction,
  formation maintenance, and shape morphing.
- **Proprioceptive Health Scorer** — composite 0-100 score from schema
  accuracy, kinesthetic responsiveness, balance quality, postural stability,
  and coordination effectiveness.

Usage (Python API)::

    from src.proprioception import SwarmProprioceptionEngine

    engine = SwarmProprioceptionEngine(num_agents=8)
    engine.add_connection("agent-0", "agent-1")
    engine.add_connection("agent-1", "agent-2")
    engine.tick()
    report = engine.get_report()
    print(report.health.score)       # 0-100
    print(report.health.tier)        # ALIGNED / AWARE / DRIFTING / ...
    engine.export_html("proprioception.html")

CLI::

    python -m src.proprioception                         # demo with defaults
    python -m src.proprioception --agents 10             # more agents
    python -m src.proprioception --ticks 80              # longer simulation
    python -m src.proprioception --scenario reconfiguration
    python -m src.proprioception --out report.html --json proprioception.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import random
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.stats_utils import gini as _gini_shared


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentRole(str, Enum):
    """Structural role of an agent in the swarm body."""
    CORE = "core"               # Central hub, high connectivity
    JOINT = "joint"             # Bridge between sub-groups (articulation point)
    LIMB = "limb"               # Mid-chain agent in an extension
    ENDPOINT = "endpoint"       # Leaf agent at the periphery
    ISOLATED = "isolated"       # Disconnected agent


class MovementType(str, Enum):
    """Types of topological movement detected."""
    EXPANSION = "expansion"         # Swarm growing outward
    CONTRACTION = "contraction"     # Swarm pulling inward
    ROTATION = "rotation"           # Structural rearrangement
    FRAGMENTATION = "fragmentation" # Breaking apart
    MERGING = "merging"             # Sub-groups joining
    DRIFT = "drift"                 # Slow unintentional shift


class PostureState(str, Enum):
    """Overall swarm posture classification."""
    COMPACT = "compact"         # Tightly connected, low diameter
    EXTENDED = "extended"       # Stretched out, high diameter
    BRANCHED = "branched"       # Multiple arms/limbs
    RING = "ring"               # Circular topology
    SCATTERED = "scattered"     # Loosely connected fragments
    CLUSTERED = "clustered"     # Dense clumps with gaps


class HealthTier(str, Enum):
    """Proprioceptive system health classification."""
    ALIGNED = "aligned"         # 80-100: Perfect body awareness
    AWARE = "aware"             # 60-79: Good proprioception
    DRIFTING = "drifting"       # 40-59: Losing track of structure
    DISORIENTED = "disoriented" # 20-39: Poor structural awareness
    DISSOCIATED = "dissociated" # 0-19: Complete loss of body schema


class InsightSeverity(str, Enum):
    """Severity level of generated insights."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class BalanceAxis(str, Enum):
    """Axes along which balance is measured."""
    CONNECTIVITY = "connectivity"       # Degree distribution balance
    CENTRALITY = "centrality"           # Centrality distribution balance
    LOAD = "load"                       # Task/message load balance
    DENSITY = "density"                 # Local density balance


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class BodySchemaEntry:
    """Proprioceptive model entry for a single agent."""
    agent_id: str
    role: AgentRole
    degree: int
    local_density: float            # Fraction of possible local connections
    eccentricity: float             # Max distance to any other agent
    betweenness: float              # Normalized betweenness centrality (approx)
    neighbors: List[str] = field(default_factory=list)


@dataclass
class KinestheticEvent:
    """A detected topological movement event."""
    tick: int
    movement_type: MovementType
    agents_involved: List[str]
    magnitude: float                # 0-1 scale
    description: str


@dataclass
class JointAngle:
    """Angular measurement at a junction agent."""
    joint_agent: str
    limb_a: List[str]               # Agents on one side
    limb_b: List[str]               # Agents on other side
    angle: float                    # 0-180 degrees conceptual
    flexion_state: str              # "extended", "neutral", "flexed", "overextended"


@dataclass
class BalanceReport:
    """Balance assessment across axes."""
    axis: BalanceAxis
    gini_coefficient: float         # 0 = perfect balance, 1 = total imbalance
    skew_direction: str             # Which side is heavier
    stability: float                # 0-1 stability score


@dataclass
class PostureSnapshot:
    """A saved posture configuration."""
    posture_id: str
    tick_recorded: int
    state: PostureState
    num_agents: int
    num_connections: int
    diameter: int
    avg_degree: float
    signature: str                  # Hash-like structural fingerprint


@dataclass
class CoordinationFeedback:
    """Real-time coordination guidance."""
    agent_id: str
    recommended_action: str
    urgency: float                  # 0-1
    reason: str


@dataclass
class HealthScore:
    """Composite proprioceptive health assessment."""
    score: float
    tier: HealthTier
    schema_accuracy: float
    kinesthetic_responsiveness: float
    balance_quality: float
    postural_stability: float
    coordination_effectiveness: float
    anomalies: List[str] = field(default_factory=list)


@dataclass
class Insight:
    """Generated autonomous insight."""
    severity: InsightSeverity
    category: str
    message: str
    tick: int
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProprioceptionReport:
    """Complete proprioceptive status report."""
    tick: int
    body_schema: List[BodySchemaEntry]
    kinesthetic_events: List[KinestheticEvent]
    joint_angles: List[JointAngle]
    balance_reports: List[BalanceReport]
    posture_history: List[PostureSnapshot]
    current_posture: PostureState
    coordination_feedback: List[CoordinationFeedback]
    health: HealthScore
    insights: List[Insight]
    movement_velocity: float
    movement_acceleration: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SwarmProprioceptionEngine:
    """Autonomous proprioceptive awareness engine for swarm topology."""

    def __init__(self, num_agents: int = 6, seed: Optional[int] = None):
        self._num_agents = num_agents
        self._rng = random.Random(seed)
        self._tick = 0

        # Topology state
        self._agents: Set[str] = {f"agent-{i}" for i in range(num_agents)}
        self._connections: Dict[str, Set[str]] = defaultdict(set)

        # Body schema
        self._body_schema: Dict[str, BodySchemaEntry] = {}

        # Kinesthetic tracking
        self._kinesthetic_events: List[KinestheticEvent] = []
        self._prev_connection_count = 0
        self._prev_agent_count = num_agents
        self._movement_history: List[float] = []  # magnitude per tick

        # Joint angles
        self._joint_angles: List[JointAngle] = []

        # Balance
        self._balance_reports: List[BalanceReport] = []

        # Posture memory
        self._posture_history: List[PostureSnapshot] = []
        self._known_postures: List[PostureSnapshot] = []
        self._current_posture = PostureState.SCATTERED

        # Coordination
        self._coordination_feedback: List[CoordinationFeedback] = []

        # Insights
        self._insights: List[Insight] = []

        # Health tracking
        self._health_history: List[float] = []

    # -----------------------------------------------------------------------
    # Public API — Topology Manipulation
    # -----------------------------------------------------------------------

    def add_agent(self, agent_id: str) -> None:
        """Add a new agent to the swarm."""
        self._agents.add(agent_id)
        if agent_id not in self._connections:
            self._connections[agent_id] = set()

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the swarm."""
        self._agents.discard(agent_id)
        # Remove all connections involving this agent
        if agent_id in self._connections:
            for neighbor in list(self._connections[agent_id]):
                self._connections[neighbor].discard(agent_id)
            del self._connections[agent_id]
        if agent_id in self._body_schema:
            del self._body_schema[agent_id]

    def add_connection(self, agent_a: str, agent_b: str) -> None:
        """Add a bidirectional connection between two agents."""
        self._agents.add(agent_a)
        self._agents.add(agent_b)
        self._connections[agent_a].add(agent_b)
        self._connections[agent_b].add(agent_a)

    def remove_connection(self, agent_a: str, agent_b: str) -> None:
        """Remove a connection between two agents."""
        self._connections[agent_a].discard(agent_b)
        self._connections[agent_b].discard(agent_a)

    # -----------------------------------------------------------------------
    # Public API — Tick & Report
    # -----------------------------------------------------------------------

    def tick(self, dt: float = 1.0) -> None:
        """Advance one time step: rebuild schema, detect movement, assess."""
        self._tick += 1
        self._rebuild_body_schema()
        self._detect_kinesthetic_events()
        self._compute_joint_angles()
        self._assess_balance()
        self._classify_posture()
        self._generate_coordination_feedback()
        self._generate_insights()
        self._prev_connection_count = self._total_connections()
        self._prev_agent_count = len(self._agents)

    def do_tick(self, dt: float = 1.0) -> None:
        """Alias for tick()."""
        self.tick(dt)

    def get_report(self) -> ProprioceptionReport:
        """Get the current proprioceptive status report."""
        health = self._compute_health()
        vel = self._compute_velocity()
        acc = self._compute_acceleration()
        return ProprioceptionReport(
            tick=self._tick,
            body_schema=list(self._body_schema.values()),
            kinesthetic_events=self._kinesthetic_events[-50:],
            joint_angles=self._joint_angles,
            balance_reports=self._balance_reports,
            posture_history=self._posture_history[-20:],
            current_posture=self._current_posture,
            coordination_feedback=self._coordination_feedback,
            health=health,
            insights=self._insights[-30:],
            movement_velocity=vel,
            movement_acceleration=acc,
        )

    def get_health(self) -> HealthScore:
        """Get just the health score."""
        return self._compute_health()

    def save_posture(self, posture_id: Optional[str] = None) -> PostureSnapshot:
        """Save the current topology as a known-good posture."""
        snap = self._take_posture_snapshot(posture_id or f"saved-{self._tick}")
        self._known_postures.append(snap)
        return snap

    # -----------------------------------------------------------------------
    # Internal — Body Schema
    # -----------------------------------------------------------------------

    def _rebuild_body_schema(self) -> None:
        """Reconstruct the body schema from current topology."""
        self._body_schema.clear()
        distances = self._compute_all_distances()

        for agent in self._agents:
            degree = len(self._connections.get(agent, set()))
            neighbors = sorted(self._connections.get(agent, set()))

            # Local density: actual connections / possible local connections
            local_possible = max(1, degree * (degree - 1) // 2)
            local_actual = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 < n2 and n2 in self._connections.get(n1, set()):
                        local_actual += 1
            local_density = local_actual / local_possible if local_possible > 0 else 0.0

            # Eccentricity
            agent_dists = distances.get(agent, {})
            eccentricity = max(agent_dists.values()) if agent_dists else 0.0

            # Approximate betweenness
            betweenness = self._approx_betweenness(agent, distances)

            # Role classification
            role = self._classify_role(agent, degree, betweenness, neighbors)

            self._body_schema[agent] = BodySchemaEntry(
                agent_id=agent,
                role=role,
                degree=degree,
                local_density=local_density,
                eccentricity=eccentricity,
                betweenness=betweenness,
                neighbors=neighbors,
            )

    def _classify_role(self, agent: str, degree: int, betweenness: float,
                       neighbors: List[str]) -> AgentRole:
        """Classify an agent's structural role."""
        if degree == 0:
            return AgentRole.ISOLATED
        if degree == 1:
            return AgentRole.ENDPOINT

        avg_degree = self._average_degree()
        # High-degree hubs are cores even if they're also articulation points
        if degree >= avg_degree * 1.5 and betweenness > 0.1:
            return AgentRole.CORE

        # Check if articulation point (simplified)
        if self._is_articulation_point(agent):
            return AgentRole.JOINT

        return AgentRole.LIMB

    def _is_articulation_point(self, agent: str) -> bool:
        """Check if removing this agent disconnects the graph."""
        if len(self._agents) <= 2:
            return False
        # BFS without this agent
        remaining = self._agents - {agent}
        if not remaining:
            return False
        start = next(iter(remaining))
        visited = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self._connections.get(current, set()):
                if neighbor != agent and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        # If some remaining agents weren't reached, it's an articulation point
        return len(visited) < len(remaining)

    # -----------------------------------------------------------------------
    # Internal — Kinesthetic Detection
    # -----------------------------------------------------------------------

    def _detect_kinesthetic_events(self) -> None:
        """Detect topological movements by comparing current vs previous state."""
        current_conns = self._total_connections()
        current_agents = len(self._agents)

        # Connection changes
        conn_delta = current_conns - self._prev_connection_count
        agent_delta = current_agents - self._prev_agent_count

        magnitude = 0.0
        events_this_tick: List[KinestheticEvent] = []

        if agent_delta > 0:
            mag = min(1.0, agent_delta / max(1, self._prev_agent_count))
            magnitude += mag
            events_this_tick.append(KinestheticEvent(
                tick=self._tick,
                movement_type=MovementType.EXPANSION,
                agents_involved=[a for a in self._agents],
                magnitude=mag,
                description=f"{agent_delta} agent(s) joined — swarm expanding",
            ))
        elif agent_delta < 0:
            mag = min(1.0, abs(agent_delta) / max(1, self._prev_agent_count))
            magnitude += mag
            events_this_tick.append(KinestheticEvent(
                tick=self._tick,
                movement_type=MovementType.CONTRACTION if conn_delta <= 0 else MovementType.FRAGMENTATION,
                agents_involved=[a for a in self._agents],
                magnitude=mag,
                description=f"{abs(agent_delta)} agent(s) departed",
            ))

        if conn_delta > 2 and agent_delta == 0:
            mag = min(1.0, conn_delta / max(1, current_conns))
            magnitude += mag
            events_this_tick.append(KinestheticEvent(
                tick=self._tick,
                movement_type=MovementType.MERGING,
                agents_involved=[a for a in self._agents],
                magnitude=mag,
                description=f"{conn_delta} new connections — sub-groups merging",
            ))
        elif conn_delta < -2 and agent_delta == 0:
            mag = min(1.0, abs(conn_delta) / max(1, self._prev_connection_count))
            magnitude += mag
            events_this_tick.append(KinestheticEvent(
                tick=self._tick,
                movement_type=MovementType.FRAGMENTATION,
                agents_involved=[a for a in self._agents],
                magnitude=mag,
                description=f"{abs(conn_delta)} connections lost — fragmenting",
            ))
        elif abs(conn_delta) > 0 and agent_delta == 0:
            mag = min(1.0, abs(conn_delta) / max(1, current_conns + 1))
            magnitude += mag
            mt = MovementType.ROTATION if abs(conn_delta) <= 2 else MovementType.DRIFT
            events_this_tick.append(KinestheticEvent(
                tick=self._tick,
                movement_type=mt,
                agents_involved=[a for a in self._agents],
                magnitude=mag,
                description=f"Topology rearrangement ({conn_delta:+d} connections)",
            ))

        self._kinesthetic_events.extend(events_this_tick)
        self._movement_history.append(min(1.0, magnitude))

    # -----------------------------------------------------------------------
    # Internal — Joint Angles
    # -----------------------------------------------------------------------

    def _compute_joint_angles(self) -> None:
        """Compute angular measurements at junction agents."""
        self._joint_angles.clear()
        for agent_id, entry in self._body_schema.items():
            if entry.role == AgentRole.JOINT and entry.degree >= 2:
                neighbors = entry.neighbors
                # Partition neighbors into two groups by connectivity
                group_a, group_b = self._partition_neighbors(agent_id, neighbors)
                if group_a and group_b:
                    # "Angle" based on relative sizes and connectivity
                    size_ratio = len(group_a) / (len(group_a) + len(group_b))
                    # More balanced = wider angle (180 = perfectly balanced)
                    angle = 180.0 * (1.0 - abs(size_ratio - 0.5) * 2)

                    if angle > 150:
                        state = "extended"
                    elif angle > 90:
                        state = "neutral"
                    elif angle > 45:
                        state = "flexed"
                    else:
                        state = "overextended"

                    self._joint_angles.append(JointAngle(
                        joint_agent=agent_id,
                        limb_a=group_a,
                        limb_b=group_b,
                        angle=angle,
                        flexion_state=state,
                    ))

    def _partition_neighbors(self, joint: str, neighbors: List[str]) -> Tuple[List[str], List[str]]:
        """Partition neighbors of a joint into two limb groups."""
        if len(neighbors) < 2:
            return neighbors, []

        # Simple partition: BFS from each neighbor without going through joint
        visited_from_first: Set[str] = set()
        queue = deque([neighbors[0]])
        visited_from_first.add(neighbors[0])
        while queue:
            current = queue.popleft()
            for n in self._connections.get(current, set()):
                if n != joint and n not in visited_from_first and n in self._agents:
                    visited_from_first.add(n)
                    queue.append(n)

        group_a = [n for n in neighbors if n in visited_from_first]
        group_b = [n for n in neighbors if n not in visited_from_first]
        return group_a, group_b

    # -----------------------------------------------------------------------
    # Internal — Balance Assessment
    # -----------------------------------------------------------------------

    def _assess_balance(self) -> None:
        """Assess structural balance across multiple axes."""
        self._balance_reports.clear()

        # Connectivity balance (degree distribution)
        degrees = [len(self._connections.get(a, set())) for a in self._agents]
        if degrees:
            gini = self._gini_coefficient(degrees)
            mean_d = statistics.mean(degrees)
            skew = "left-heavy" if statistics.median(degrees) < mean_d else "right-heavy"
            self._balance_reports.append(BalanceReport(
                axis=BalanceAxis.CONNECTIVITY,
                gini_coefficient=gini,
                skew_direction=skew,
                stability=1.0 - gini,
            ))

        # Centrality balance
        betweenness_vals = [e.betweenness for e in self._body_schema.values()]
        if betweenness_vals:
            gini = self._gini_coefficient(betweenness_vals)
            self._balance_reports.append(BalanceReport(
                axis=BalanceAxis.CENTRALITY,
                gini_coefficient=gini,
                skew_direction="core-heavy" if gini > 0.5 else "distributed",
                stability=1.0 - gini,
            ))

        # Density balance
        densities = [e.local_density for e in self._body_schema.values()]
        if densities:
            gini = self._gini_coefficient(densities)
            self._balance_reports.append(BalanceReport(
                axis=BalanceAxis.DENSITY,
                gini_coefficient=gini,
                skew_direction="clustered" if gini > 0.5 else "uniform",
                stability=1.0 - gini,
            ))

    # -----------------------------------------------------------------------
    # Internal — Posture Classification
    # -----------------------------------------------------------------------

    def _classify_posture(self) -> None:
        """Classify the current overall posture."""
        if not self._agents:
            self._current_posture = PostureState.SCATTERED
            return

        n = len(self._agents)
        self._total_connections()
        avg_degree = self._average_degree()
        diameter = self._compute_diameter()

        # Count connected components
        components = self._count_components()

        if components > 1 and components > n * 0.3:
            self._current_posture = PostureState.SCATTERED
        elif components > 1:
            self._current_posture = PostureState.CLUSTERED
        elif self._is_ring_like():
            self._current_posture = PostureState.RING
        elif diameter <= 2 and avg_degree >= n * 0.4:
            self._current_posture = PostureState.COMPACT
        elif diameter >= n * 0.7:
            self._current_posture = PostureState.EXTENDED
        else:
            # Check for branching
            endpoints = sum(1 for e in self._body_schema.values() if e.role == AgentRole.ENDPOINT)
            joints = sum(1 for e in self._body_schema.values() if e.role == AgentRole.JOINT)
            if endpoints >= 3 and joints >= 1:
                self._current_posture = PostureState.BRANCHED
            else:
                self._current_posture = PostureState.COMPACT

        # Save snapshot
        snap = self._take_posture_snapshot(f"tick-{self._tick}")
        self._posture_history.append(snap)

    def _is_ring_like(self) -> bool:
        """Check if topology is ring-like (all degree ~2, single cycle)."""
        if len(self._agents) < 3:
            return False
        degrees = [len(self._connections.get(a, set())) for a in self._agents]
        return all(d == 2 for d in degrees)

    def _take_posture_snapshot(self, posture_id: str) -> PostureSnapshot:
        """Create a snapshot of current posture."""
        n = len(self._agents)
        conns = self._total_connections()
        diameter = self._compute_diameter()
        avg_deg = self._average_degree()
        # Simple structural fingerprint
        degrees_sorted = sorted(len(self._connections.get(a, set())) for a in self._agents)
        sig = f"{n}-{conns}-{diameter}-{''.join(str(d) for d in degrees_sorted[:10])}"
        return PostureSnapshot(
            posture_id=posture_id,
            tick_recorded=self._tick,
            state=self._current_posture,
            num_agents=n,
            num_connections=conns,
            diameter=diameter,
            avg_degree=avg_deg,
            signature=sig,
        )

    # -----------------------------------------------------------------------
    # Internal — Coordination Feedback
    # -----------------------------------------------------------------------

    def _generate_coordination_feedback(self) -> None:
        """Generate actionable coordination feedback for agents."""
        self._coordination_feedback.clear()

        for agent_id, entry in self._body_schema.items():
            # Isolated agents should connect
            if entry.role == AgentRole.ISOLATED:
                self._coordination_feedback.append(CoordinationFeedback(
                    agent_id=agent_id,
                    recommended_action="connect_to_nearest",
                    urgency=0.9,
                    reason="Agent is isolated — no proprioceptive signal",
                ))
            # Overloaded cores should delegate
            elif entry.role == AgentRole.CORE and entry.degree > len(self._agents) * 0.6:
                self._coordination_feedback.append(CoordinationFeedback(
                    agent_id=agent_id,
                    recommended_action="delegate_connections",
                    urgency=0.7,
                    reason="Core overloaded — risk of single-point-of-failure",
                ))
            # Joints under stress
            elif entry.role == AgentRole.JOINT:
                angle = next((j.angle for j in self._joint_angles if j.joint_agent == agent_id), 90.0)
                if angle < 45:
                    self._coordination_feedback.append(CoordinationFeedback(
                        agent_id=agent_id,
                        recommended_action="recruit_parallel_joint",
                        urgency=0.8,
                        reason="Joint over-compressed — structural stress",
                    ))

    # -----------------------------------------------------------------------
    # Internal — Insight Generation
    # -----------------------------------------------------------------------

    def _generate_insights(self) -> None:
        """Generate autonomous insights about proprioceptive state."""
        # Balance insight
        for br in self._balance_reports:
            if br.gini_coefficient > 0.7:
                self._insights.append(Insight(
                    severity=InsightSeverity.CRITICAL,
                    category="balance",
                    message=f"Severe {br.axis.value} imbalance (Gini={br.gini_coefficient:.2f}) — structural collapse risk",
                    tick=self._tick,
                    data={"axis": br.axis.value, "gini": br.gini_coefficient},
                ))
            elif br.gini_coefficient > 0.5:
                self._insights.append(Insight(
                    severity=InsightSeverity.WARNING,
                    category="balance",
                    message=f"{br.axis.value} imbalance detected (Gini={br.gini_coefficient:.2f})",
                    tick=self._tick,
                    data={"axis": br.axis.value, "gini": br.gini_coefficient},
                ))

        # Movement velocity insight
        vel = self._compute_velocity()
        if vel > 0.5:
            self._insights.append(Insight(
                severity=InsightSeverity.WARNING,
                category="kinesthetic",
                message=f"High structural change velocity ({vel:.2f}) — topology shifting rapidly",
                tick=self._tick,
                data={"velocity": vel},
            ))

        # Posture deviation from known-good
        if self._known_postures:
            current_sig = self._posture_history[-1].signature if self._posture_history else ""
            matches = any(p.signature == current_sig for p in self._known_postures)
            if not matches and self._tick > 5:
                self._insights.append(Insight(
                    severity=InsightSeverity.INFO,
                    category="posture",
                    message=f"Current posture ({self._current_posture.value}) deviates from all known-good configurations",
                    tick=self._tick,
                    data={"posture": self._current_posture.value},
                ))

        # Isolated agents
        isolated = sum(1 for e in self._body_schema.values() if e.role == AgentRole.ISOLATED)
        if isolated > 0:
            sev = InsightSeverity.CRITICAL if isolated > len(self._agents) * 0.3 else InsightSeverity.WARNING
            self._insights.append(Insight(
                severity=sev,
                category="schema",
                message=f"{isolated} agent(s) isolated — proprioceptive blind spots",
                tick=self._tick,
                data={"isolated_count": isolated},
            ))

    # -----------------------------------------------------------------------
    # Internal — Health Scoring
    # -----------------------------------------------------------------------

    def _compute_health(self) -> HealthScore:
        """Compute composite proprioceptive health score."""
        # Schema accuracy: how well-connected and role-diverse is the swarm
        if not self._agents:
            return HealthScore(
                score=0.0, tier=HealthTier.DISSOCIATED,
                schema_accuracy=0, kinesthetic_responsiveness=0,
                balance_quality=0, postural_stability=0,
                coordination_effectiveness=0, anomalies=["No agents"],
            )

        # 1. Schema accuracy (0-100): penalize isolated agents and low connectivity
        isolated_ratio = sum(1 for e in self._body_schema.values() if e.role == AgentRole.ISOLATED) / len(self._agents)
        connected_ratio = min(1.0, self._total_connections() / max(1, len(self._agents) - 1))
        schema_accuracy = (1.0 - isolated_ratio) * 50 + connected_ratio * 50

        # 2. Kinesthetic responsiveness (0-100): ability to detect movement
        recent_moves = self._movement_history[-10:]
        if recent_moves:
            responsiveness = min(100.0, len([m for m in recent_moves if m > 0]) * 20)
        else:
            responsiveness = 50.0  # Neutral if no history

        # 3. Balance quality (0-100): average stability across axes
        if self._balance_reports:
            avg_stability = statistics.mean(br.stability for br in self._balance_reports)
            balance_quality = avg_stability * 100
        else:
            balance_quality = 50.0

        # 4. Postural stability (0-100): consistency of posture over time
        if len(self._posture_history) >= 3:
            recent = self._posture_history[-5:]
            posture_changes = sum(1 for i in range(1, len(recent)) if recent[i].state != recent[i-1].state)
            postural_stability = max(0, 100 - posture_changes * 25)
        else:
            postural_stability = 75.0  # Assume stable initially

        # 5. Coordination effectiveness (0-100): fewer urgent issues = better
        if self._coordination_feedback:
            avg_urgency = statistics.mean(f.urgency for f in self._coordination_feedback)
            coordination_effectiveness = (1.0 - avg_urgency) * 100
        else:
            coordination_effectiveness = 90.0  # No issues = good

        # Composite weighted score
        score = (
            schema_accuracy * 0.30 +
            responsiveness * 0.15 +
            balance_quality * 0.25 +
            postural_stability * 0.15 +
            coordination_effectiveness * 0.15
        )
        score = max(0.0, min(100.0, score))

        # Tier classification
        if score >= 80:
            tier = HealthTier.ALIGNED
        elif score >= 60:
            tier = HealthTier.AWARE
        elif score >= 40:
            tier = HealthTier.DRIFTING
        elif score >= 20:
            tier = HealthTier.DISORIENTED
        else:
            tier = HealthTier.DISSOCIATED

        # Anomalies
        anomalies: List[str] = []
        if isolated_ratio > 0.3:
            anomalies.append("high_isolation")
        if balance_quality < 30:
            anomalies.append("severe_imbalance")
        vel = self._compute_velocity()
        if vel > 0.7:
            anomalies.append("rapid_reconfiguration")

        self._health_history.append(score)

        return HealthScore(
            score=round(score, 1),
            tier=tier,
            schema_accuracy=round(schema_accuracy, 1),
            kinesthetic_responsiveness=round(responsiveness, 1),
            balance_quality=round(balance_quality, 1),
            postural_stability=round(postural_stability, 1),
            coordination_effectiveness=round(coordination_effectiveness, 1),
            anomalies=anomalies,
        )

    # -----------------------------------------------------------------------
    # Internal — Graph Utilities
    # -----------------------------------------------------------------------

    def _total_connections(self) -> int:
        """Count total undirected connections."""
        return sum(len(v) for v in self._connections.values()) // 2

    def _average_degree(self) -> float:
        """Average degree of agents."""
        if not self._agents:
            return 0.0
        return sum(len(self._connections.get(a, set())) for a in self._agents) / len(self._agents)

    def _compute_diameter(self) -> int:
        """Compute graph diameter (longest shortest path)."""
        distances = self._compute_all_distances()
        max_dist = 0
        for dists in distances.values():
            if dists:
                max_dist = max(max_dist, max(dists.values()))
        return max_dist

    def _compute_all_distances(self) -> Dict[str, Dict[str, float]]:
        """BFS-based all-pairs shortest paths."""
        distances: Dict[str, Dict[str, float]] = {}
        for source in self._agents:
            dist: Dict[str, float] = {}
            visited = {source}
            queue = deque([(source, 0)])
            while queue:
                node, d = queue.popleft()
                dist[node] = d
                for neighbor in self._connections.get(node, set()):
                    if neighbor not in visited and neighbor in self._agents:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))
            distances[source] = dist
        return distances

    def _approx_betweenness(self, agent: str, distances: Dict[str, Dict[str, float]]) -> float:
        """Approximate betweenness centrality."""
        if len(self._agents) <= 2:
            return 0.0
        count = 0
        total_pairs = 0
        for s in self._agents:
            if s == agent:
                continue
            for t in self._agents:
                if t == agent or t <= s:
                    continue
                total_pairs += 1
                # Check if agent is on shortest path from s to t
                d_st = distances.get(s, {}).get(t, float('inf'))
                d_sa = distances.get(s, {}).get(agent, float('inf'))
                d_at = distances.get(agent, {}).get(t, float('inf'))
                if d_st < float('inf') and abs(d_sa + d_at - d_st) < 0.001:
                    count += 1
        return count / max(1, total_pairs)

    def _count_components(self) -> int:
        """Count connected components."""
        visited: Set[str] = set()
        components = 0
        for agent in self._agents:
            if agent not in visited:
                components += 1
                queue = deque([agent])
                visited.add(agent)
                while queue:
                    current = queue.popleft()
                    for neighbor in self._connections.get(current, set()):
                        if neighbor not in visited and neighbor in self._agents:
                            visited.add(neighbor)
                            queue.append(neighbor)
        return components

    def _gini_coefficient(self, values: List[float]) -> float:
        """Compute Gini coefficient for a list of values."""
        return _gini_shared(values)

    def _compute_velocity(self) -> float:
        """Compute recent movement velocity."""
        if len(self._movement_history) < 2:
            return 0.0
        recent = self._movement_history[-5:]
        return statistics.mean(recent)

    def _compute_acceleration(self) -> float:
        """Compute movement acceleration."""
        if len(self._movement_history) < 4:
            return 0.0
        recent = self._movement_history[-6:]
        mid = len(recent) // 2
        first_half = statistics.mean(recent[:mid]) if recent[:mid] else 0
        second_half = statistics.mean(recent[mid:]) if recent[mid:] else 0
        return second_half - first_half

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        """Export report as JSON."""
        report = self.get_report()
        data = {
            "tick": report.tick,
            "current_posture": report.current_posture.value,
            "movement_velocity": report.movement_velocity,
            "movement_acceleration": report.movement_acceleration,
            "health": {
                "score": report.health.score,
                "tier": report.health.tier.value,
                "schema_accuracy": report.health.schema_accuracy,
                "kinesthetic_responsiveness": report.health.kinesthetic_responsiveness,
                "balance_quality": report.health.balance_quality,
                "postural_stability": report.health.postural_stability,
                "coordination_effectiveness": report.health.coordination_effectiveness,
                "anomalies": report.health.anomalies,
            },
            "body_schema": [
                {"agent_id": e.agent_id, "role": e.role.value, "degree": e.degree,
                 "local_density": round(e.local_density, 3), "eccentricity": e.eccentricity,
                 "betweenness": round(e.betweenness, 3)}
                for e in report.body_schema
            ],
            "joint_angles": [
                {"joint": j.joint_agent, "angle": round(j.angle, 1), "state": j.flexion_state}
                for j in report.joint_angles
            ],
            "balance": [
                {"axis": b.axis.value, "gini": round(b.gini_coefficient, 3),
                 "stability": round(b.stability, 3)}
                for b in report.balance_reports
            ],
            "kinesthetic_events": [
                {"tick": k.tick, "type": k.movement_type.value, "magnitude": round(k.magnitude, 3),
                 "description": k.description}
                for k in report.kinesthetic_events[-20:]
            ],
            "insights": [
                {"severity": i.severity.value, "category": i.category, "message": i.message}
                for i in report.insights[-20:]
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def export_html(self, path: str) -> None:
        """Export interactive HTML dashboard."""
        report = self.get_report()
        html_content = self._render_html(report)
        Path(path).write_text(html_content, encoding="utf-8")

    def _render_html(self, report: ProprioceptionReport) -> str:
        """Render HTML dashboard."""
        h = report.health

        # Body schema table
        schema_rows = ""
        for entry in sorted(report.body_schema, key=lambda e: e.degree, reverse=True):
            role_colors = {
                "core": "#4CAF50", "joint": "#FF9800", "limb": "#2196F3",
                "endpoint": "#9C27B0", "isolated": "#F44336",
            }
            color = role_colors.get(entry.role.value, "#666")
            schema_rows += f"""<tr>
                <td>{html_mod.escape(entry.agent_id)}</td>
                <td><span style="color:{color};font-weight:bold">{entry.role.value}</span></td>
                <td>{entry.degree}</td>
                <td>{entry.local_density:.2f}</td>
                <td>{entry.betweenness:.3f}</td>
            </tr>"""

        # Joint angles
        joint_html = ""
        for j in report.joint_angles:
            state_colors = {"extended": "#4CAF50", "neutral": "#2196F3",
                           "flexed": "#FF9800", "overextended": "#F44336"}
            color = state_colors.get(j.flexion_state, "#666")
            joint_html += f"""<div style="margin:4px 0;padding:6px;background:#f5f5f5;border-radius:4px">
                <strong>{html_mod.escape(j.joint_agent)}</strong>: {j.angle:.0f}°
                <span style="color:{color}">({j.flexion_state})</span>
            </div>"""
        if not joint_html:
            joint_html = "<p style='color:#999'>No joints detected</p>"

        # Balance bars
        balance_html = ""
        for br in report.balance_reports:
            pct = int(br.stability * 100)
            color = "#4CAF50" if pct >= 70 else "#FF9800" if pct >= 40 else "#F44336"
            balance_html += f"""<div style="margin:8px 0">
                <div style="display:flex;justify-content:space-between">
                    <span>{br.axis.value}</span>
                    <span>Gini: {br.gini_coefficient:.2f}</span>
                </div>
                <div style="background:#eee;border-radius:4px;height:12px;overflow:hidden">
                    <div style="width:{pct}%;background:{color};height:100%"></div>
                </div>
            </div>"""

        # Events
        events_html = ""
        for ev in report.kinesthetic_events[-10:]:
            events_html += f"<div style='margin:3px 0'>⚡ [tick {ev.tick}] {html_mod.escape(ev.description)}</div>"
        if not events_html:
            events_html = "<p style='color:#999'>No movement detected yet</p>"

        # Insights
        insight_html = ""
        for ins in report.insights[-10:]:
            icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(ins.severity.value, "•")
            insight_html += f"<div style='margin:4px 0'>{icon} [{ins.category}] {html_mod.escape(ins.message)}</div>"
        if not insight_html:
            insight_html = "<p style='color:#999'>No insights yet</p>"

        # Coordination
        coord_html = ""
        for cf in report.coordination_feedback[:10]:
            urgency_color = "#F44336" if cf.urgency > 0.7 else "#FF9800" if cf.urgency > 0.4 else "#4CAF50"
            coord_html += f"""<div style="margin:4px 0;padding:6px;border-left:3px solid {urgency_color};background:#fafafa">
                <strong>{html_mod.escape(cf.agent_id)}</strong>: {html_mod.escape(cf.recommended_action)}
                <br><small style="color:#666">{html_mod.escape(cf.reason)}</small>
            </div>"""
        if not coord_html:
            coord_html = "<p style='color:#999'>All agents well-coordinated</p>"

        tier_colors = {
            "aligned": "#4CAF50", "aware": "#8BC34A", "drifting": "#FF9800",
            "disoriented": "#FF5722", "dissociated": "#F44336",
        }
        tier_color = tier_colors.get(h.tier.value, "#666")

        return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Swarm Proprioception Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #f8f9fa; }}
  .header {{ text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white; margin-bottom: 20px; }}
  .score {{ font-size: 48px; font-weight: bold; }}
  .tier {{ font-size: 18px; text-transform: uppercase; letter-spacing: 2px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px; }}
  .card {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .card h2 {{ margin-top: 0; color: #333; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 6px 8px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  .metric {{ display: inline-block; margin: 4px 8px; padding: 6px 12px; background: #f0f0f0; border-radius: 16px; font-size: 13px; }}
</style></head><body>
<div class="header">
  <div class="score">{h.score:.0f}</div>
  <div class="tier" style="color:{tier_color}">{h.tier.value}</div>
  <div style="margin-top:8px">Posture: {report.current_posture.value} | Velocity: {report.movement_velocity:.2f} | Tick: {report.tick}</div>
</div>

<div style="text-align:center;margin-bottom:16px">
  <span class="metric">Schema: {h.schema_accuracy:.0f}</span>
  <span class="metric">Kinesthetic: {h.kinesthetic_responsiveness:.0f}</span>
  <span class="metric">Balance: {h.balance_quality:.0f}</span>
  <span class="metric">Posture: {h.postural_stability:.0f}</span>
  <span class="metric">Coordination: {h.coordination_effectiveness:.0f}</span>
</div>

<div class="grid">
<div class="card">
  <h2>🦴 Body Schema</h2>
  <table><tr><th>Agent</th><th>Role</th><th>Degree</th><th>Density</th><th>Betweenness</th></tr>
  {schema_rows}
  </table>
</div>

<div class="card">
  <h2>🔗 Joint Angles</h2>
  {joint_html}
</div>

<div class="card">
  <h2>⚖️ Balance Assessment</h2>
  {balance_html}
</div>

<div class="card">
  <h2>🏃 Kinesthetic Events</h2>
  {events_html}
</div>

<div class="card">
  <h2>🎯 Coordination Feedback</h2>
  {coord_html}
</div>

<div class="card">
  <h2>💡 Insights</h2>
  {insight_html}
</div>

</div></body></html>"""


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "description": "Stable connected topology with gradual growth",
        "connections": [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3),
        ],
        "events": [],
    },
    "reconfiguration": {
        "description": "Rapid structural changes: connections breaking and reforming",
        "connections": [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5),
        ],
        "events": [
            {"tick": 10, "action": "remove_connection", "a": 2, "b": 3},
            {"tick": 15, "action": "add_connection", "a": 2, "b": 5},
            {"tick": 25, "action": "remove_connection", "a": 0, "b": 1},
            {"tick": 30, "action": "add_connection", "a": 0, "b": 4},
            {"tick": 40, "action": "add_agent", "id": 6},
            {"tick": 41, "action": "add_connection", "a": 6, "b": 3},
            {"tick": 50, "action": "remove_agent", "id": 5},
        ],
    },
    "fragmentation": {
        "description": "Connected topology gradually breaking apart",
        "connections": [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4),
        ],
        "events": [
            {"tick": 10, "action": "remove_connection", "a": 2, "b": 3},
            {"tick": 20, "action": "remove_connection", "a": 1, "b": 4},
            {"tick": 30, "action": "remove_connection", "a": 0, "b": 5},
            {"tick": 40, "action": "remove_agent", "id": 3},
            {"tick": 50, "action": "remove_agent", "id": 4},
        ],
    },
    "growth": {
        "description": "Swarm rapidly expanding and connecting",
        "connections": [
            (0, 1), (1, 2),
        ],
        "events": [
            {"tick": 5, "action": "add_agent", "id": 6},
            {"tick": 6, "action": "add_connection", "a": 6, "b": 2},
            {"tick": 10, "action": "add_agent", "id": 7},
            {"tick": 11, "action": "add_connection", "a": 7, "b": 0},
            {"tick": 15, "action": "add_connection", "a": 3, "b": 4},
            {"tick": 20, "action": "add_connection", "a": 4, "b": 5},
            {"tick": 25, "action": "add_agent", "id": 8},
            {"tick": 26, "action": "add_connection", "a": 8, "b": 1},
            {"tick": 26, "action": "add_connection", "a": 8, "b": 6},
            {"tick": 35, "action": "add_connection", "a": 7, "b": 3},
            {"tick": 40, "action": "add_connection", "a": 0, "b": 5},
        ],
    },
    "imbalance": {
        "description": "Star topology with extreme centralization then rebalancing",
        "connections": [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        ],
        "events": [
            {"tick": 20, "action": "add_connection", "a": 1, "b": 2},
            {"tick": 25, "action": "add_connection", "a": 3, "b": 4},
            {"tick": 30, "action": "add_connection", "a": 4, "b": 5},
            {"tick": 35, "action": "add_connection", "a": 2, "b": 3},
            {"tick": 40, "action": "remove_connection", "a": 0, "b": 3},
            {"tick": 45, "action": "remove_connection", "a": 0, "b": 4},
        ],
    },
}


# ---------------------------------------------------------------------------
# Demo Runner
# ---------------------------------------------------------------------------


def run_demo(num_agents: int = 6, ticks: int = 60,
             scenario: str = "baseline", seed: Optional[int] = 42,
             out_html: Optional[str] = None, out_json: Optional[str] = None) -> ProprioceptionReport:
    """Run a proprioception demo with a predefined scenario."""
    engine = SwarmProprioceptionEngine(num_agents=num_agents, seed=seed)

    scenario_data = SCENARIOS.get(scenario, SCENARIOS["baseline"])

    # Set up initial connections
    for a, b in scenario_data.get("connections", []):
        engine.add_connection(f"agent-{a}", f"agent-{b}")

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  🦴 Swarm Proprioception Engine — Demo                      ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Scenario: {scenario:<20} Agents: {num_agents:<5} Ticks: {ticks:<5}  ║")
    print(f"║  {scenario_data['description']:<56}  ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()

    events = scenario_data.get("events", [])

    for t in range(1, ticks + 1):
        # Apply scheduled events
        for ev in events:
            if ev.get("tick") == t:
                action = ev.get("action", "")
                if action == "add_connection":
                    a_id = f"agent-{ev['a']}"
                    b_id = f"agent-{ev['b']}"
                    engine.add_connection(a_id, b_id)
                    print(f"  [tick {t:3d}] 🔗 {a_id} ↔ {b_id} connected")
                elif action == "remove_connection":
                    a_id = f"agent-{ev['a']}"
                    b_id = f"agent-{ev['b']}"
                    engine.remove_connection(a_id, b_id)
                    print(f"  [tick {t:3d}] ✂️  {a_id} ↔ {b_id} disconnected")
                elif action == "add_agent":
                    aid = f"agent-{ev['id']}"
                    engine.add_agent(aid)
                    print(f"  [tick {t:3d}] ➕ {aid} joined")
                elif action == "remove_agent":
                    aid = f"agent-{ev['id']}"
                    engine.remove_agent(aid)
                    print(f"  [tick {t:3d}] ➖ {aid} departed")

        engine.tick()

    report = engine.get_report()

    # Print summary
    print()
    print(f"┌─────────────────────────────────────────────────────────┐")
    print(f"│  Health Score: {report.health.score:5.1f} / 100  [{report.health.tier.value}]")
    print(f"│  Posture: {report.current_posture.value:<12}  Velocity: {report.movement_velocity:.3f}")
    print(f"│  Agents: {len(report.body_schema):3d}   Joints: {sum(1 for e in report.body_schema if e.role == AgentRole.JOINT):3d}")
    print(f"│  Events: {len(report.kinesthetic_events):3d}   Insights: {len(report.insights):3d}")
    print(f"│  Anomalies: {', '.join(report.health.anomalies) or 'None'}")
    print(f"└─────────────────────────────────────────────────────────┘")

    if report.insights:
        print(f"\n  Insights:")
        for ins in report.insights[-8:]:
            icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(ins.severity.value, "•")
            print(f"    {icon} [{ins.category}] {ins.message}")

    if out_html:
        engine.export_html(out_html)
        print(f"\n  📄 HTML report: {out_html}")
    if out_json:
        engine.export_json(out_json)
        print(f"  📄 JSON report: {out_json}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Swarm Proprioception Engine — autonomous body-schema awareness"
    )
    parser.add_argument("--agents", type=int, default=6, help="Number of agents")
    parser.add_argument("--ticks", type=int, default=60, help="Simulation ticks")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="baseline",
                        help="Scenario preset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="HTML output path")
    parser.add_argument("--json", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    run_demo(
        num_agents=args.agents,
        ticks=args.ticks,
        scenario=args.scenario,
        seed=args.seed,
        out_html=args.out,
        out_json=args.json,
    )


if __name__ == "__main__":
    main()
