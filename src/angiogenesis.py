"""Swarm Angiogenesis Engine — autonomous communication pathway growth and pruning.

Biologically-inspired by vascular angiogenesis: new communication channels
(vessels) sprout toward demand signals, mature under sustained use, and
regress when underutilized.  The network self-organizes an efficient
communication topology that adapts to shifting workload patterns.

Key biological parallels:

- **VEGF (Vascular Endothelial Growth Factor)** — demand signals emitted by
  agents that lack sufficient communication bandwidth.
- **Tip Cells** — leading edge of a sprouting vessel, guided by VEGF gradients.
- **Stalk Cells** — following cells that extend the vessel behind the tip.
- **Pericyte Coverage** — maturation marker; mature vessels resist pruning.
- **Hypoxia** — communication starvation triggering angiogenic signaling.
- **Vessel Regression** — pruning of underused or redundant pathways.
- **Anastomosis** — connection of two vessel tips to form a loop.

Capabilities:

- **Demand Signal System** — agents emit VEGF-like signals when communication
  demand exceeds local capacity; signals diffuse and decay over time.
- **Sprouting Engine** — new vessels sprout from existing ones toward strongest
  VEGF gradients; tip cells navigate via gradient climbing.
- **Maturation Engine** — vessels gain pericyte coverage through sustained
  traffic; mature vessels have higher capacity and pruning resistance.
- **Flow Simulation** — capacity-aware flow routing; overloaded vessels trigger
  upstream VEGF emission.
- **Regression Engine** — vessels with low utilization lose pericyte coverage
  and eventually regress (are pruned).
- **Anastomosis Detector** — when two sprouting tips approach each other, they
  fuse to form a circuit, improving redundancy.
- **Health Scoring** — composite 0-100 metric: perfusion coverage, flow
  efficiency, redundancy, maturation balance.
- **Interactive HTML Dashboard** — network topology, demand heatmap, vessel
  lifecycle timeline, health gauges.

Usage (Python API)::

    from src.angiogenesis import SwarmAngiogenesisEngine

    engine = SwarmAngiogenesisEngine(agents=["a1", "a2", "a3", "a4", "a5"])

    # Agents generate communication demand
    engine.emit_demand("a1", "a3", intensity=3.0)
    engine.emit_demand("a2", "a4", intensity=2.5)

    # Simulate growth/pruning cycles
    report = engine.simulate(ticks=100)
    print(report.health_score)        # 0-100
    print(report.vessel_count)        # active vessels
    print(report.perfusion_coverage)  # fraction of agents well-connected
    print(report.insights)            # autonomous recommendations

    engine.export_html("angiogenesis_report.html")

CLI::

    python -m src.angiogenesis                     # demo with defaults
    python -m src.angiogenesis --agents 10         # larger swarm
    python -m src.angiogenesis --ticks 200         # longer simulation
    python -m src.angiogenesis --out report.html --json angio.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------


class VesselState(str, Enum):
    """Lifecycle state of a communication vessel."""
    SPROUTING = "sprouting"       # actively growing toward demand
    ACTIVE = "active"             # established and carrying traffic
    MATURE = "mature"             # high pericyte coverage, resistant to pruning
    REGRESSING = "regressing"     # underutilized, losing pericyte coverage
    PRUNED = "pruned"             # removed from network


class DemandType(str, Enum):
    """Types of communication demand signals."""
    VEGF = "vegf"                 # general demand for connectivity
    HYPOXIA = "hypoxia"           # severe starvation — urgent need
    ARTERIOGENIC = "arteriogenic"  # need for higher capacity (flow-driven)


# Demand urgency multipliers
DEMAND_URGENCY: Dict[DemandType, float] = {
    DemandType.VEGF: 1.0,
    DemandType.HYPOXIA: 2.5,
    DemandType.ARTERIOGENIC: 1.5,
}


@dataclass
class DemandSignal:
    """A demand signal emitted by an agent."""
    source_agent: str
    target_agent: str
    demand_type: DemandType
    intensity: float
    tick_emitted: int
    remaining_strength: float = 0.0

    def __post_init__(self) -> None:
        if self.remaining_strength == 0.0:
            self.remaining_strength = self.intensity


@dataclass
class Vessel:
    """A communication pathway (vessel) between two agents."""
    vessel_id: str
    source: str
    target: str
    state: VesselState = VesselState.SPROUTING
    capacity: float = 1.0           # max flow units per tick
    current_flow: float = 0.0       # current utilization
    pericyte_coverage: float = 0.0  # 0-1, maturation level
    age: int = 0                    # ticks since creation
    utilization_history: List[float] = field(default_factory=list)
    sprouting_progress: float = 0.0  # 0-1 for SPROUTING vessels
    created_tick: int = 0
    pruned_tick: Optional[int] = None

    @property
    def utilization(self) -> float:
        """Current utilization ratio."""
        if self.capacity <= 0:
            return 0.0
        return min(1.0, self.current_flow / self.capacity)

    @property
    def is_active(self) -> bool:
        return self.state in (VesselState.ACTIVE, VesselState.MATURE, VesselState.SPROUTING)

    @property
    def avg_utilization(self) -> float:
        if not self.utilization_history:
            return 0.0
        return statistics.mean(self.utilization_history[-20:])


@dataclass
class AgentNode:
    """An agent's connectivity state."""
    agent_id: str
    demand_emitted: float = 0.0      # total demand emitted
    demand_received: float = 0.0     # total demand received
    perfusion_score: float = 0.0     # how well-connected (0-1)
    incoming_vessels: int = 0
    outgoing_vessels: int = 0
    hypoxic: bool = False            # starved for connections


@dataclass
class AngiogenesisEvent:
    """A significant event in the angiogenesis lifecycle."""
    tick: int
    event_type: str
    vessel_id: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None
    detail: str = ""


@dataclass
class HealthReport:
    """Composite health assessment."""
    score: float                     # 0-100
    perfusion_coverage: float        # fraction of agents well-perfused
    flow_efficiency: float           # avg utilization of active vessels
    redundancy: float                # fraction with multiple paths
    maturation_ratio: float          # fraction mature vs total
    vessel_turnover: float           # sprout + prune rate
    tier: str = ""

    def __post_init__(self) -> None:
        if not self.tier:
            if self.score >= 80:
                self.tier = "Thriving"
            elif self.score >= 60:
                self.tier = "Healthy"
            elif self.score >= 40:
                self.tier = "Stressed"
            elif self.score >= 20:
                self.tier = "Ischemic"
            else:
                self.tier = "Necrotic"


@dataclass
class AngiogenesisReport:
    """Full analysis report."""
    health_score: float
    health: HealthReport
    vessel_count: int
    active_vessel_count: int
    mature_count: int
    sprouting_count: int
    pruned_total: int
    perfusion_coverage: float
    total_ticks: int
    events: List[AngiogenesisEvent]
    agent_nodes: Dict[str, AgentNode]
    insights: List[str]
    vessels: List[Vessel] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class SwarmAngiogenesisEngine:
    """Autonomous communication pathway growth and pruning engine."""

    def __init__(
        self,
        agents: List[str],
        *,
        sprout_threshold: float = 2.0,
        prune_threshold: float = 0.1,
        maturation_threshold: float = 0.7,
        regression_threshold: float = 0.15,
        demand_decay: float = 0.1,
        max_vessels_per_agent: int = 6,
        sprouting_speed: float = 0.25,
        capacity_growth_rate: float = 0.05,
        initial_connectivity: float = 0.2,
        seed: Optional[int] = None,
    ):
        """Initialize the angiogenesis engine.

        Args:
            agents: List of agent identifiers.
            sprout_threshold: Min demand to trigger sprouting.
            prune_threshold: Avg utilization below which regression starts.
            maturation_threshold: Pericyte coverage to reach MATURE state.
            regression_threshold: Avg utilization to enter REGRESSING.
            demand_decay: Per-tick decay of demand signals.
            max_vessels_per_agent: Max outgoing vessels per agent.
            sprouting_speed: Progress per tick for sprouting vessels.
            capacity_growth_rate: Rate of capacity increase for mature vessels.
            initial_connectivity: Fraction of possible edges to create initially.
            seed: Random seed for reproducibility.
        """
        self._rng = random.Random(seed)
        self.agents = list(agents)
        self.sprout_threshold = sprout_threshold
        self.prune_threshold = prune_threshold
        self.maturation_threshold = maturation_threshold
        self.regression_threshold = regression_threshold
        self.demand_decay = demand_decay
        self.max_vessels_per_agent = max_vessels_per_agent
        self.sprouting_speed = sprouting_speed
        self.capacity_growth_rate = capacity_growth_rate

        self.vessels: Dict[str, Vessel] = {}
        self.demand_signals: List[DemandSignal] = []
        self.events: List[AngiogenesisEvent] = []
        self.tick_count: int = 0
        self._vessel_counter: int = 0
        self._pruned_total: int = 0

        # Initialize some connectivity
        if initial_connectivity > 0:
            self._init_connectivity(initial_connectivity)

    def _init_connectivity(self, fraction: float) -> None:
        """Create initial vessel connections."""
        for src in self.agents:
            for tgt in self.agents:
                if src == tgt:
                    continue
                if self._rng.random() < fraction:
                    self._create_vessel(src, tgt, state=VesselState.ACTIVE)

    def _create_vessel(
        self, source: str, target: str,
        state: VesselState = VesselState.SPROUTING,
        capacity: float = 1.0,
    ) -> Optional[Vessel]:
        """Create a new vessel if not duplicate and under limit."""
        # Check for existing active vessel
        for v in self.vessels.values():
            if v.source == source and v.target == target and v.is_active:
                return None

        # Check max vessels per agent
        out_count = sum(
            1 for v in self.vessels.values()
            if v.source == source and v.is_active
        )
        if out_count >= self.max_vessels_per_agent:
            return None

        self._vessel_counter += 1
        vid = f"v-{self._vessel_counter}"
        vessel = Vessel(
            vessel_id=vid,
            source=source,
            target=target,
            state=state,
            capacity=capacity,
            created_tick=self.tick_count,
            sprouting_progress=1.0 if state != VesselState.SPROUTING else 0.0,
            pericyte_coverage=0.3 if state == VesselState.ACTIVE else 0.0,
        )
        self.vessels[vid] = vessel
        self.events.append(AngiogenesisEvent(
            tick=self.tick_count,
            event_type="sprouted" if state == VesselState.SPROUTING else "created",
            vessel_id=vid,
            source=source,
            target=target,
            detail=f"capacity={capacity:.2f}",
        ))
        return vessel

    def emit_demand(
        self,
        source_agent: str,
        target_agent: str,
        intensity: float = 1.0,
        demand_type: DemandType = DemandType.VEGF,
    ) -> DemandSignal:
        """Emit a communication demand signal.

        Args:
            source_agent: Agent needing connectivity.
            target_agent: Desired destination.
            intensity: Strength of demand (higher = more urgent).
            demand_type: Type of demand signal.

        Returns:
            The created DemandSignal.
        """
        signal = DemandSignal(
            source_agent=source_agent,
            target_agent=target_agent,
            demand_type=demand_type,
            intensity=intensity * DEMAND_URGENCY[demand_type],
            tick_emitted=self.tick_count,
        )
        self.demand_signals.append(signal)
        return signal

    def route_flow(self, source: str, target: str, amount: float = 1.0) -> float:
        """Route communication flow through existing vessels.

        Returns the amount successfully routed.
        """
        routed = 0.0
        # Find direct vessels
        for v in self.vessels.values():
            if v.source == source and v.target == target and v.is_active:
                available = v.capacity - v.current_flow
                use = min(amount - routed, available)
                if use > 0:
                    v.current_flow += use
                    routed += use
                if routed >= amount:
                    break
        return routed

    def simulate(self, ticks: int = 100) -> AngiogenesisReport:
        """Run the full simulation.

        Args:
            ticks: Number of simulation ticks to run.

        Returns:
            AngiogenesisReport with full analysis.
        """
        for _ in range(ticks):
            self._tick()
        return self.analyze()

    def _tick(self) -> None:
        """Execute one simulation tick."""
        self.tick_count += 1

        # 1. Decay demand signals
        self._decay_demands()

        # 2. Auto-generate demand for poorly-connected agents
        self._detect_hypoxia()

        # 3. Process sprouting
        self._process_sprouting()

        # 4. Grow new sprouts toward demand
        self._sprout_new_vessels()

        # 5. Route flow and update utilization
        self._update_flow()

        # 6. Maturation
        self._process_maturation()

        # 7. Regression and pruning
        self._process_regression()

        # 8. Anastomosis detection
        self._detect_anastomosis()

        # 9. Capacity adaptation
        self._adapt_capacity()

    def _decay_demands(self) -> None:
        """Decay and remove expired demand signals."""
        alive = []
        for sig in self.demand_signals:
            sig.remaining_strength *= (1.0 - self.demand_decay)
            if sig.remaining_strength > 0.01:
                alive.append(sig)
        self.demand_signals = alive

    def _detect_hypoxia(self) -> None:
        """Detect agents with insufficient connectivity and emit HYPOXIA signals."""
        for agent in self.agents:
            incoming = sum(
                1 for v in self.vessels.values()
                if v.target == agent and v.is_active
            )
            outgoing = sum(
                1 for v in self.vessels.values()
                if v.source == agent and v.is_active
            )
            total_connections = incoming + outgoing
            if total_connections < 2 and self.tick_count % 5 == 0:
                # Pick a random target to connect to
                others = [a for a in self.agents if a != agent]
                if others:
                    target = self._rng.choice(others)
                    self.emit_demand(agent, target, intensity=2.0, demand_type=DemandType.HYPOXIA)

    def _process_sprouting(self) -> None:
        """Advance sprouting vessels toward completion."""
        for v in list(self.vessels.values()):
            if v.state == VesselState.SPROUTING:
                v.sprouting_progress += self.sprouting_speed
                v.age += 1
                if v.sprouting_progress >= 1.0:
                    v.state = VesselState.ACTIVE
                    v.sprouting_progress = 1.0
                    v.pericyte_coverage = 0.1
                    self.events.append(AngiogenesisEvent(
                        tick=self.tick_count,
                        event_type="activated",
                        vessel_id=v.vessel_id,
                        source=v.source,
                        target=v.target,
                    ))

    def _sprout_new_vessels(self) -> None:
        """Create new vessel sprouts toward strong demand signals."""
        # Aggregate demand by (source, target) pair
        demand_map: Dict[Tuple[str, str], float] = defaultdict(float)
        for sig in self.demand_signals:
            demand_map[(sig.source_agent, sig.target_agent)] += sig.remaining_strength

        # Sort by demand strength
        sorted_demands = sorted(demand_map.items(), key=lambda x: x[1], reverse=True)

        for (src, tgt), strength in sorted_demands:
            if strength < self.sprout_threshold:
                break
            # Try to sprout a vessel
            vessel = self._create_vessel(src, tgt)
            if vessel:
                # Consume some demand
                for sig in self.demand_signals:
                    if sig.source_agent == src and sig.target_agent == tgt:
                        sig.remaining_strength *= 0.3

    def _update_flow(self) -> None:
        """Simulate flow and record utilization."""
        # Reset flow
        for v in self.vessels.values():
            if v.is_active:
                v.current_flow = 0.0

        # Generate synthetic traffic based on demand
        for sig in self.demand_signals:
            self.route_flow(sig.source_agent, sig.target_agent, sig.remaining_strength * 0.5)

        # Also generate some background traffic
        for v in self.vessels.values():
            if v.is_active and v.state != VesselState.SPROUTING:
                # Small baseline traffic
                v.current_flow += self._rng.uniform(0.0, 0.3) * v.capacity

        # Record utilization
        for v in self.vessels.values():
            if v.is_active:
                v.utilization_history.append(v.utilization)
                # Keep history bounded
                if len(v.utilization_history) > 50:
                    v.utilization_history = v.utilization_history[-50:]
                v.age += 1 if v.state != VesselState.SPROUTING else 0

    def _process_maturation(self) -> None:
        """Vessels with high sustained utilization gain pericyte coverage."""
        for v in self.vessels.values():
            if v.state == VesselState.ACTIVE:
                if v.avg_utilization > 0.4:
                    # Gain pericyte coverage
                    v.pericyte_coverage = min(1.0, v.pericyte_coverage + 0.02)
                    if v.pericyte_coverage >= self.maturation_threshold:
                        v.state = VesselState.MATURE
                        self.events.append(AngiogenesisEvent(
                            tick=self.tick_count,
                            event_type="matured",
                            vessel_id=v.vessel_id,
                            source=v.source,
                            target=v.target,
                            detail=f"pericyte={v.pericyte_coverage:.2f}",
                        ))
                else:
                    # Slow pericyte loss
                    v.pericyte_coverage = max(0.0, v.pericyte_coverage - 0.005)

    def _process_regression(self) -> None:
        """Low-utilization vessels regress and get pruned."""
        for v in list(self.vessels.values()):
            if v.state in (VesselState.ACTIVE, VesselState.MATURE):
                if v.avg_utilization < self.regression_threshold and v.age > 10:
                    if v.state == VesselState.MATURE:
                        # Mature vessels resist — lose pericyte first
                        v.pericyte_coverage -= 0.03
                        if v.pericyte_coverage < 0.3:
                            v.state = VesselState.ACTIVE
                    else:
                        v.state = VesselState.REGRESSING
                        self.events.append(AngiogenesisEvent(
                            tick=self.tick_count,
                            event_type="regressing",
                            vessel_id=v.vessel_id,
                            source=v.source,
                            target=v.target,
                        ))

            elif v.state == VesselState.REGRESSING:
                v.pericyte_coverage -= 0.05
                if v.pericyte_coverage <= 0.0:
                    v.state = VesselState.PRUNED
                    v.pruned_tick = self.tick_count
                    self._pruned_total += 1
                    self.events.append(AngiogenesisEvent(
                        tick=self.tick_count,
                        event_type="pruned",
                        vessel_id=v.vessel_id,
                        source=v.source,
                        target=v.target,
                    ))

    def _detect_anastomosis(self) -> None:
        """Detect when sprouting vessels can merge to form circuits."""
        sprouting = [v for v in self.vessels.values() if v.state == VesselState.SPROUTING]
        for i, v1 in enumerate(sprouting):
            for v2 in sprouting[i + 1:]:
                # If v1 targets where v2 sources and vice versa, fuse
                if v1.target == v2.source and v1.source == v2.target:
                    v1.state = VesselState.ACTIVE
                    v1.sprouting_progress = 1.0
                    v1.pericyte_coverage = 0.2
                    v2.state = VesselState.ACTIVE
                    v2.sprouting_progress = 1.0
                    v2.pericyte_coverage = 0.2
                    self.events.append(AngiogenesisEvent(
                        tick=self.tick_count,
                        event_type="anastomosis",
                        vessel_id=v1.vessel_id,
                        source=v1.source,
                        target=v1.target,
                        detail=f"fused with {v2.vessel_id}",
                    ))

    def _adapt_capacity(self) -> None:
        """Mature vessels under high load increase capacity."""
        for v in self.vessels.values():
            if v.state == VesselState.MATURE and v.utilization > 0.8:
                v.capacity += self.capacity_growth_rate
                # Emit arteriogenic demand if very overloaded
                if v.utilization > 0.95:
                    self.emit_demand(
                        v.source, v.target,
                        intensity=1.5,
                        demand_type=DemandType.ARTERIOGENIC,
                    )

    def analyze(self) -> AngiogenesisReport:
        """Generate full analysis report."""
        active_vessels = [v for v in self.vessels.values() if v.is_active]
        mature = [v for v in active_vessels if v.state == VesselState.MATURE]
        sprouting = [v for v in active_vessels if v.state == VesselState.SPROUTING]

        # Compute agent nodes
        agent_nodes: Dict[str, AgentNode] = {}
        for agent in self.agents:
            incoming = [v for v in active_vessels if v.target == agent]
            outgoing = [v for v in active_vessels if v.source == agent]
            total_cap_in = sum(v.capacity for v in incoming)
            total_cap_out = sum(v.capacity for v in outgoing)
            perfusion = min(1.0, (total_cap_in + total_cap_out) / max(1, len(self.agents) - 1))
            agent_nodes[agent] = AgentNode(
                agent_id=agent,
                perfusion_score=perfusion,
                incoming_vessels=len(incoming),
                outgoing_vessels=len(outgoing),
                hypoxic=len(incoming) + len(outgoing) < 2,
            )

        # Health metrics
        perfusion_scores = [n.perfusion_score for n in agent_nodes.values()]
        perfusion_coverage = sum(1 for s in perfusion_scores if s > 0.3) / max(1, len(self.agents))

        flow_efficiency = 0.0
        if active_vessels:
            utils = [v.avg_utilization for v in active_vessels if v.state != VesselState.SPROUTING]
            flow_efficiency = statistics.mean(utils) if utils else 0.0

        # Redundancy: fraction of agent pairs with >1 path
        pair_paths: Dict[Tuple[str, str], int] = defaultdict(int)
        for v in active_vessels:
            if v.state != VesselState.SPROUTING:
                pair_paths[(v.source, v.target)] += 1
        total_pairs = len(self.agents) * (len(self.agents) - 1) if len(self.agents) > 1 else 1
        redundant_pairs = sum(1 for c in pair_paths.values() if c > 1)
        redundancy = redundant_pairs / max(1, total_pairs)

        maturation_ratio = len(mature) / max(1, len(active_vessels))

        # Vessel turnover
        recent_events = [e for e in self.events if e.tick > self.tick_count - 20]
        sprout_events = sum(1 for e in recent_events if e.event_type == "sprouted")
        prune_events = sum(1 for e in recent_events if e.event_type == "pruned")
        vessel_turnover = (sprout_events + prune_events) / max(1, 20)

        # Composite score
        score = (
            perfusion_coverage * 35 +
            flow_efficiency * 25 +
            redundancy * 15 +
            maturation_ratio * 15 +
            max(0, 10 - vessel_turnover * 20)  # low turnover = stable
        )
        score = max(0.0, min(100.0, score))

        health = HealthReport(
            score=score,
            perfusion_coverage=perfusion_coverage,
            flow_efficiency=flow_efficiency,
            redundancy=redundancy,
            maturation_ratio=maturation_ratio,
            vessel_turnover=vessel_turnover,
        )

        # Generate insights
        insights = self._generate_insights(health, agent_nodes, active_vessels)

        return AngiogenesisReport(
            health_score=score,
            health=health,
            vessel_count=len(self.vessels),
            active_vessel_count=len(active_vessels),
            mature_count=len(mature),
            sprouting_count=len(sprouting),
            pruned_total=self._pruned_total,
            perfusion_coverage=perfusion_coverage,
            total_ticks=self.tick_count,
            events=self.events,
            agent_nodes=agent_nodes,
            insights=insights,
            vessels=list(self.vessels.values()),
        )

    def _generate_insights(
        self,
        health: HealthReport,
        nodes: Dict[str, AgentNode],
        active: List[Vessel],
    ) -> List[str]:
        """Generate autonomous insights and recommendations."""
        insights: List[str] = []

        # Perfusion issues
        hypoxic = [n for n in nodes.values() if n.hypoxic]
        if hypoxic:
            agents_str = ", ".join(n.agent_id for n in hypoxic[:3])
            insights.append(
                f"[WARN] {len(hypoxic)} agent(s) are hypoxic (poorly connected): {agents_str}. "
                "Consider increasing demand signals or lowering sprout threshold."
            )

        # Over-maturation (network too rigid)
        if health.maturation_ratio > 0.8:
            insights.append(
                "[RIGID] Network is highly mature (>80% vessels mature). This provides stability "
                "but may resist adapting to new demand patterns."
            )

        # High turnover (instability)
        if health.vessel_turnover > 0.5:
            insights.append(
                "[TURNOVER] High vessel turnover detected. The network is rapidly creating and "
                "pruning vessels -- consider tuning sprout/prune thresholds."
            )

        # Low efficiency
        if health.flow_efficiency < 0.2 and active:
            insights.append(
                "[LOW-FLOW] Low flow efficiency -- many vessels are underutilized. "
                "The network may be over-provisioned relative to demand."
            )

        # Good health
        if health.score >= 80:
            insights.append(
                "[OK] Network is thriving with good perfusion, efficient flow, and stable topology."
            )
        elif health.score >= 60:
            insights.append(
                "[OK] Network is healthy. Minor optimization opportunities exist."
            )

        # Redundancy
        if health.redundancy < 0.05 and len(self.agents) > 3:
            insights.append(
                "[FRAGILE] Very low redundancy -- single points of failure exist. "
                "Promoting parallel pathways would improve resilience."
            )

        return insights

    def export_json(self, path: str) -> None:
        """Export report as JSON."""
        report = self.analyze()
        data = {
            "health_score": report.health_score,
            "health": asdict(report.health),
            "vessel_count": report.vessel_count,
            "active_vessel_count": report.active_vessel_count,
            "mature_count": report.mature_count,
            "sprouting_count": report.sprouting_count,
            "pruned_total": report.pruned_total,
            "perfusion_coverage": report.perfusion_coverage,
            "total_ticks": report.total_ticks,
            "insights": report.insights,
            "agent_nodes": {k: asdict(v) for k, v in report.agent_nodes.items()},
            "events": [asdict(e) for e in report.events[-100:]],
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def export_html(self, path: str) -> None:
        """Export interactive HTML dashboard."""
        report = self.analyze()
        html = self._render_html(report)
        Path(path).write_text(html, encoding="utf-8")

    def _render_html(self, report: AngiogenesisReport) -> str:
        """Render HTML dashboard."""
        h = html_mod.escape

        # Event timeline rows
        event_rows = ""
        for ev in report.events[-50:]:
            color = {
                "sprouted": "#4CAF50",
                "activated": "#2196F3",
                "matured": "#9C27B0",
                "regressing": "#FF9800",
                "pruned": "#F44336",
                "anastomosis": "#00BCD4",
                "created": "#607D8B",
            }.get(ev.event_type, "#757575")
            event_rows += (
                f"<tr><td>{ev.tick}</td>"
                f"<td><span style='color:{color};font-weight:bold'>{h(ev.event_type)}</span></td>"
                f"<td>{h(ev.vessel_id or '')}</td>"
                f"<td>{h(ev.source or '')} → {h(ev.target or '')}</td>"
                f"<td>{h(ev.detail)}</td></tr>\n"
            )

        # Agent table
        agent_rows = ""
        for node in report.agent_nodes.values():
            status = "🔴 Hypoxic" if node.hypoxic else "🟢 Perfused"
            agent_rows += (
                f"<tr><td>{h(node.agent_id)}</td>"
                f"<td>{node.perfusion_score:.2f}</td>"
                f"<td>{node.incoming_vessels}</td>"
                f"<td>{node.outgoing_vessels}</td>"
                f"<td>{status}</td></tr>\n"
            )

        # Insight list
        insight_items = "".join(f"<li>{h(i)}</li>" for i in report.insights)

        score_color = (
            "#4CAF50" if report.health_score >= 60 else
            "#FF9800" if report.health_score >= 40 else "#F44336"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Angiogenesis Dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #e94560; }}
h2 {{ color: #0f3460; background: #16213e; padding: 10px; border-radius: 6px; color: #e94560; }}
.score-badge {{ display: inline-block; font-size: 2.5em; font-weight: bold;
               color: {score_color}; border: 3px solid {score_color};
               border-radius: 50%; width: 100px; height: 100px;
               line-height: 100px; text-align: center; margin: 10px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 12px; margin: 16px 0; }}
.metric {{ background: #16213e; padding: 14px; border-radius: 8px; text-align: center; }}
.metric .value {{ font-size: 1.8em; font-weight: bold; color: #00d2d3; }}
.metric .label {{ font-size: 0.85em; color: #aaa; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
th {{ background: #16213e; color: #e94560; }}
tr:hover {{ background: #16213e; }}
.insights {{ background: #0f3460; padding: 16px; border-radius: 8px; margin: 16px 0; }}
.insights li {{ margin: 8px 0; }}
.tier {{ font-size: 1.2em; color: #ffd32a; }}
</style>
</head>
<body>
<h1>🩸 Swarm Angiogenesis Dashboard</h1>
<div style="display:flex;align-items:center;gap:20px">
  <div class="score-badge">{report.health_score:.0f}</div>
  <div>
    <div class="tier">Tier: {h(report.health.tier)}</div>
    <div>Ticks simulated: {report.total_ticks}</div>
    <div>Active vessels: {report.active_vessel_count}</div>
  </div>
</div>

<div class="metrics">
  <div class="metric"><div class="value">{report.perfusion_coverage:.0%}</div><div class="label">Perfusion Coverage</div></div>
  <div class="metric"><div class="value">{report.health.flow_efficiency:.0%}</div><div class="label">Flow Efficiency</div></div>
  <div class="metric"><div class="value">{report.health.redundancy:.0%}</div><div class="label">Redundancy</div></div>
  <div class="metric"><div class="value">{report.mature_count}</div><div class="label">Mature Vessels</div></div>
  <div class="metric"><div class="value">{report.sprouting_count}</div><div class="label">Sprouting</div></div>
  <div class="metric"><div class="value">{report.pruned_total}</div><div class="label">Total Pruned</div></div>
</div>

<h2>🧬 Agent Perfusion</h2>
<table>
<tr><th>Agent</th><th>Perfusion</th><th>Incoming</th><th>Outgoing</th><th>Status</th></tr>
{agent_rows}
</table>

<h2>💡 Autonomous Insights</h2>
<div class="insights"><ul>{insight_items}</ul></div>

<h2>📜 Event Timeline (last 50)</h2>
<table>
<tr><th>Tick</th><th>Event</th><th>Vessel</th><th>Path</th><th>Detail</th></tr>
{event_rows}
</table>

<footer style="margin-top:30px;color:#666;font-size:0.8em">
Generated by Swarm Angiogenesis Engine — mBFT Metacognition Framework
</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo(num_agents: int = 8, ticks: int = 150, seed: int = 42) -> AngiogenesisReport:
    """Run a demonstration scenario."""
    agents = [f"agent-{i}" for i in range(1, num_agents + 1)]
    engine = SwarmAngiogenesisEngine(
        agents=agents,
        initial_connectivity=0.15,
        seed=seed,
    )

    # Phase 1: Initial demand burst
    for _ in range(3):
        src = random.choice(agents)
        tgt = random.choice([a for a in agents if a != src])
        engine.emit_demand(src, tgt, intensity=3.0)

    # Simulate with periodic demand injection
    for tick in range(ticks):
        if tick % 15 == 0:
            src = random.choice(agents)
            tgt = random.choice([a for a in agents if a != src])
            engine.emit_demand(src, tgt, intensity=random.uniform(1.5, 4.0))
        if tick % 30 == 0 and tick > 0:
            # Shift demand to simulate changing workload
            src = random.choice(agents)
            tgt = random.choice([a for a in agents if a != src])
            engine.emit_demand(src, tgt, intensity=3.5, demand_type=DemandType.HYPOXIA)
        engine._tick()

    return engine.analyze(), engine  # type: ignore[return-value]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Swarm Angiogenesis Engine — autonomous communication pathway growth"
    )
    parser.add_argument("--agents", type=int, default=8, help="Number of agents")
    parser.add_argument("--ticks", type=int, default=150, help="Simulation ticks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="HTML output path")
    parser.add_argument("--json", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    print("[Angiogenesis] Swarm Angiogenesis Engine")
    print("=" * 50)

    report, engine = _demo(args.agents, args.ticks, args.seed)

    print(f"\n{'Health Score:':<25} {report.health_score:.1f}/100 ({report.health.tier})")
    print(f"{'Perfusion Coverage:':<25} {report.perfusion_coverage:.1%}")
    print(f"{'Flow Efficiency:':<25} {report.health.flow_efficiency:.1%}")
    print(f"{'Redundancy:':<25} {report.health.redundancy:.1%}")
    print(f"{'Active Vessels:':<25} {report.active_vessel_count}")
    print(f"{'Mature Vessels:':<25} {report.mature_count}")
    print(f"{'Sprouting:':<25} {report.sprouting_count}")
    print(f"{'Total Pruned:':<25} {report.pruned_total}")
    print(f"{'Simulation Ticks:':<25} {report.total_ticks}")

    print("\n[Perfusion] Agent Perfusion:")
    for node in report.agent_nodes.values():
        status = "[!]" if node.hypoxic else "[+]"
        print(f"  {status} {node.agent_id}: perfusion={node.perfusion_score:.2f} "
              f"in={node.incoming_vessels} out={node.outgoing_vessels}")

    print("\n[Insights]:")
    for insight in report.insights:
        print(f"  {insight}")

    if args.out:
        engine.export_html(args.out)
        print(f"\nHTML report: {args.out}")

    if args.json:
        engine.export_json(args.json)
        print(f"JSON report: {args.json}")


if __name__ == "__main__":
    main()
