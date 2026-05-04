"""Swarm Hibernation Engine — autonomous energy conservation with torpor states.

Biologically inspired by mammalian hibernation (bears, ground squirrels,
bats).  Agents enter low-energy torpor states during resource scarcity to
conserve energy, with arousal mechanisms that wake them when conditions
improve or emergencies occur.

Capabilities:

- **Energy Budget Tracker** — each agent has an energy level (0-100) that
  depletes with activity and recovers slowly during rest.  Tracks metabolic
  rate, energy reserves, and consumption patterns.
- **Torpor State Manager** — agents transition through states:
  ACTIVE → DROWSY → LIGHT_TORPOR → DEEP_TORPOR → AROUSING → ACTIVE.
  Each state has different metabolic rates (deep torpor uses ~5% of active
  metabolism).  Minimum bout durations prevent premature arousal.
- **Scarcity Detector** — monitors swarm-wide resource levels.  When
  resources drop below thresholds, broadcasts hibernation signals.  Uses
  exponential moving average to smooth noise.
- **Arousal Trigger Engine** — detects conditions requiring emergency
  arousal: critical tasks, threat signals, and periodic arousal bouts
  (like real hibernators who periodically warm up).
- **Hibernaculum Manager** — manages hibernation clusters.  Agents
  hibernate in groups for warmth (mutual benefit).  Ensures minimum
  active agents for swarm survival.
- **Health Scorer** — composite 0-100 score from energy reserves, torpor
  efficiency, arousal responsiveness, cluster utilization, active ratio,
  and sustainability.  5 tiers: Thriving/Conserving/Strained/Critical/Collapsed.
- **Insight Generator** — autonomous observations about hibernation
  patterns, energy efficiency, arousal responsiveness, cluster dynamics.

Usage (Python API)::

    from src.hibernation import SwarmHibernationEngine

    engine = SwarmHibernationEngine(num_agents=12, seed=42)
    report = engine.simulate(cycles=80)
    print(report.health.score)       # 0-100
    print(report.health.tier)        # Thriving/Conserving/Strained/...
    print(report.insights)           # autonomous observations
    engine.export_html("hibernation_report.html")

CLI::

    python -m src.hibernation                         # demo with defaults
    python -m src.hibernation --agents 15 --cycles 100
    python -m src.hibernation --scenario deep_freeze
    python -m src.hibernation --out report.html --json hibernation.json
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
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------


class HibernationState(str, Enum):
    """Agent torpor lifecycle states."""
    ACTIVE = "active"
    DROWSY = "drowsy"
    LIGHT_TORPOR = "light_torpor"
    DEEP_TORPOR = "deep_torpor"
    AROUSING = "arousing"


class ArousalTrigger(str, Enum):
    """Reasons for waking from torpor."""
    RESOURCE_RECOVERY = "resource_recovery"
    EMERGENCY_TASK = "emergency_task"
    THREAT_SIGNAL = "threat_signal"
    PERIODIC_BOUT = "periodic_bout"
    MANUAL = "manual"
    CLUSTER_BREAK = "cluster_break"


class ScarcityLevel(str, Enum):
    """Resource scarcity classifications."""
    ABUNDANT = "abundant"
    ADEQUATE = "adequate"
    SCARCE = "scarce"
    CRITICAL = "critical"
    DEPLETED = "depleted"


# Metabolic rate multipliers per state
_METABOLIC_MULTIPLIER: Dict[HibernationState, float] = {
    HibernationState.ACTIVE: 1.0,
    HibernationState.DROWSY: 0.6,
    HibernationState.LIGHT_TORPOR: 0.2,
    HibernationState.DEEP_TORPOR: 0.05,
    HibernationState.AROUSING: 0.8,
}


@dataclass
class AgentEnergyState:
    """Energy and torpor state of a single agent."""
    agent_id: str
    energy: float = 100.0
    metabolic_rate: float = 1.0
    state: HibernationState = HibernationState.ACTIVE
    torpor_bouts: int = 0
    total_torpor_cycles: int = 0
    cluster_id: Optional[str] = None
    arousal_count: int = 0
    last_arousal_trigger: Optional[ArousalTrigger] = None
    cycles_in_state: int = 0
    energy_at_torpor_entry: float = 0.0


@dataclass
class TorporBout:
    """Record of a single torpor episode."""
    agent_id: str
    start_cycle: int
    end_cycle: int
    depth: str  # "light" or "deep"
    energy_saved: float
    trigger: Optional[ArousalTrigger] = None


@dataclass
class HibernationCluster:
    """A group of co-hibernating agents."""
    cluster_id: str
    member_ids: List[str] = field(default_factory=list)
    formation_cycle: int = 0
    thermal_benefit: float = 0.0
    is_active: bool = True


@dataclass
class ScarcityEvent:
    """Record of a scarcity level change."""
    cycle: int
    level: ScarcityLevel
    resource_level: float
    trigger_count: int


@dataclass
class ArousalEvent:
    """Record of an agent being aroused from torpor."""
    agent_id: str
    cycle: int
    trigger: ArousalTrigger
    from_state: HibernationState
    latency_cycles: int


@dataclass
class HealthScore:
    """Composite health assessment."""
    score: float
    tier: str
    energy_reserves: float
    torpor_efficiency: float
    arousal_responsiveness: float
    cluster_utilization: float
    active_ratio: float
    sustainability: float


@dataclass
class HibernationReport:
    """Full simulation report."""
    agents: List[AgentEnergyState]
    clusters: List[HibernationCluster]
    torpor_bouts: List[TorporBout]
    arousal_events: List[ArousalEvent]
    scarcity_events: List[ScarcityEvent]
    health: HealthScore
    insights: List[str]
    cycle_history: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "mild_winter": {
        "num_agents": 12,
        "cycles": 80,
        "resource_schedule": "gradual_drop",
        "min_resource": 0.45,
        "description": "Moderate scarcity — most agents stay active",
    },
    "deep_freeze": {
        "num_agents": 15,
        "cycles": 120,
        "resource_schedule": "severe_drop",
        "min_resource": 0.08,
        "description": "Severe scarcity — prolonged deep torpor",
    },
    "intermittent_scarcity": {
        "num_agents": 12,
        "cycles": 100,
        "resource_schedule": "oscillating",
        "min_resource": 0.20,
        "description": "Alternating abundance and scarcity",
    },
    "emergency_arousal": {
        "num_agents": 10,
        "cycles": 80,
        "resource_schedule": "severe_drop",
        "min_resource": 0.10,
        "inject_threats": True,
        "description": "Deep hibernation interrupted by threats",
    },
    "cluster_survival": {
        "num_agents": 20,
        "cycles": 100,
        "resource_schedule": "gradual_drop",
        "min_resource": 0.15,
        "cluster_thermal_bonus": 0.5,
        "description": "Tests hibernaculum clustering benefits",
    },
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SwarmHibernationEngine:
    """Autonomous swarm energy conservation engine."""

    def __init__(
        self,
        num_agents: int = 12,
        base_metabolic_rate: float = 1.0,
        torpor_metabolic_fraction: float = 0.05,
        scarcity_threshold: float = 0.4,
        critical_threshold: float = 0.15,
        min_active_ratio: float = 0.2,
        cluster_thermal_bonus: float = 0.3,
        periodic_arousal_interval: int = 20,
        drowsy_energy_threshold: float = 60.0,
        torpor_energy_threshold: float = 40.0,
        min_torpor_bout: int = 5,
        seed: Optional[int] = None,
    ):
        self.num_agents = num_agents
        self.base_metabolic_rate = base_metabolic_rate
        self.torpor_metabolic_fraction = torpor_metabolic_fraction
        self.scarcity_threshold = scarcity_threshold
        self.critical_threshold = critical_threshold
        self.min_active_ratio = min_active_ratio
        self.cluster_thermal_bonus = cluster_thermal_bonus
        self.periodic_arousal_interval = periodic_arousal_interval
        self.drowsy_energy_threshold = drowsy_energy_threshold
        self.torpor_energy_threshold = torpor_energy_threshold
        self.min_torpor_bout = min_torpor_bout

        self._rng = random.Random(seed)
        self._agents: List[AgentEnergyState] = []
        self._clusters: List[HibernationCluster] = []
        self._torpor_bouts: List[TorporBout] = []
        self._arousal_events: List[ArousalEvent] = []
        self._scarcity_events: List[ScarcityEvent] = []
        self._cycle_history: List[Dict[str, Any]] = []

        self._resource_level: float = 1.0
        self._resource_ema: float = 1.0
        self._ema_alpha: float = 0.15
        self._current_scarcity: ScarcityLevel = ScarcityLevel.ABUNDANT
        self._cycle: int = 0
        self._threat_level: float = 0.0
        self._active_bout_starts: Dict[str, int] = {}  # agent_id -> start cycle
        self._active_bout_energy: Dict[str, float] = {}  # energy at entry
        self._next_cluster_id: int = 0

        self._init_agents()

    def _init_agents(self) -> None:
        """Initialize agent population."""
        self._agents = []
        for i in range(self.num_agents):
            agent = AgentEnergyState(
                agent_id=f"agent_{i:03d}",
                energy=self._rng.uniform(70.0, 100.0),
                metabolic_rate=self.base_metabolic_rate * self._rng.uniform(0.8, 1.2),
                state=HibernationState.ACTIVE,
            )
            self._agents.append(agent)

    @classmethod
    def from_scenario(cls, name: str, seed: Optional[int] = None) -> "SwarmHibernationEngine":
        """Create engine from a named scenario preset."""
        if name not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")
        cfg = SCENARIOS[name]
        kwargs: Dict[str, Any] = {"seed": seed}
        if "num_agents" in cfg:
            kwargs["num_agents"] = cfg["num_agents"]
        if "cluster_thermal_bonus" in cfg:
            kwargs["cluster_thermal_bonus"] = cfg["cluster_thermal_bonus"]
        engine = cls(**kwargs)
        engine._scenario_cfg = cfg
        return engine

    # ------------------------------------------------------------------
    # Public mutation methods
    # ------------------------------------------------------------------

    def set_resource_level(self, level: float) -> None:
        """Set current resource level (0-1)."""
        self._resource_level = max(0.0, min(1.0, level))

    def inject_threat(self, severity: float = 0.8) -> None:
        """Inject an emergency threat signal."""
        self._threat_level = max(0.0, min(1.0, severity))

    # ------------------------------------------------------------------
    # Engine 1: Energy Budget Tracker
    # ------------------------------------------------------------------

    def _update_energy(self, agent: AgentEnergyState) -> None:
        """Update agent energy based on current state and resources."""
        multiplier = _METABOLIC_MULTIPLIER[agent.state]
        # Cluster thermal benefit reduces torpor cost
        if agent.cluster_id and agent.state in (
            HibernationState.LIGHT_TORPOR,
            HibernationState.DEEP_TORPOR,
        ):
            cluster = self._find_cluster(agent.cluster_id)
            if cluster:
                multiplier *= (1.0 - cluster.thermal_benefit)

        cost = agent.metabolic_rate * multiplier * self._rng.uniform(0.8, 1.2)

        # Active agents consume more when resources are scarce (working harder)
        if agent.state == HibernationState.ACTIVE and self._resource_level < 0.5:
            cost *= 1.0 + (0.5 - self._resource_level)

        # Resource availability provides some recovery for active agents
        if agent.state == HibernationState.ACTIVE and self._resource_level > 0.7:
            recovery = (self._resource_level - 0.7) * 2.0 * self._rng.uniform(0.5, 1.0)
            agent.energy = min(100.0, agent.energy + recovery)

        agent.energy = max(0.0, agent.energy - cost)

    def _find_cluster(self, cluster_id: str) -> Optional[HibernationCluster]:
        """Find a cluster by ID."""
        for c in self._clusters:
            if c.cluster_id == cluster_id and c.is_active:
                return c
        return None

    # ------------------------------------------------------------------
    # Engine 2: Torpor State Manager
    # ------------------------------------------------------------------

    def _check_torpor_entry(self, agent: AgentEnergyState) -> None:
        """Check if agent should enter or deepen torpor."""
        if agent.state == HibernationState.AROUSING:
            # Let arousal complete
            agent.cycles_in_state += 1
            if agent.cycles_in_state >= 3:
                agent.state = HibernationState.ACTIVE
                agent.cycles_in_state = 0
            return

        is_scarce = self._current_scarcity in (
            ScarcityLevel.SCARCE, ScarcityLevel.CRITICAL, ScarcityLevel.DEPLETED
        )

        if agent.state == HibernationState.ACTIVE:
            # Consider entering drowsy if energy dropping or scarcity detected
            if (agent.energy < self.drowsy_energy_threshold and is_scarce) or \
               agent.energy < self.drowsy_energy_threshold * 0.6:
                agent.state = HibernationState.DROWSY
                agent.cycles_in_state = 0
            else:
                agent.cycles_in_state += 1

        elif agent.state == HibernationState.DROWSY:
            agent.cycles_in_state += 1
            if agent.energy < self.torpor_energy_threshold or \
               (is_scarce and agent.cycles_in_state >= 3):
                # Check we maintain minimum active agents
                active_count = sum(
                    1 for a in self._agents
                    if a.state in (HibernationState.ACTIVE, HibernationState.DROWSY)
                    and a.agent_id != agent.agent_id
                )
                min_needed = max(1, int(self.num_agents * self.min_active_ratio))
                if active_count >= min_needed:
                    agent.state = HibernationState.LIGHT_TORPOR
                    agent.cycles_in_state = 0
                    agent.energy_at_torpor_entry = agent.energy
                    self._active_bout_starts[agent.agent_id] = self._cycle
                    self._active_bout_energy[agent.agent_id] = agent.energy
                    agent.torpor_bouts += 1
            elif agent.energy > self.drowsy_energy_threshold + 10 and not is_scarce:
                agent.state = HibernationState.ACTIVE
                agent.cycles_in_state = 0

        elif agent.state == HibernationState.LIGHT_TORPOR:
            agent.cycles_in_state += 1
            agent.total_torpor_cycles += 1
            if agent.cycles_in_state >= self.min_torpor_bout and \
               self._current_scarcity in (ScarcityLevel.SCARCE, ScarcityLevel.CRITICAL, ScarcityLevel.DEPLETED):
                agent.state = HibernationState.DEEP_TORPOR
                agent.cycles_in_state = 0

        elif agent.state == HibernationState.DEEP_TORPOR:
            agent.cycles_in_state += 1
            agent.total_torpor_cycles += 1

    # ------------------------------------------------------------------
    # Engine 3: Scarcity Detector
    # ------------------------------------------------------------------

    def _detect_scarcity(self) -> None:
        """Assess resource scarcity using EMA smoothing."""
        self._resource_ema = (
            self._ema_alpha * self._resource_level
            + (1.0 - self._ema_alpha) * self._resource_ema
        )

        prev = self._current_scarcity
        ema = self._resource_ema
        if ema >= 0.7:
            self._current_scarcity = ScarcityLevel.ABUNDANT
        elif ema >= self.scarcity_threshold:
            self._current_scarcity = ScarcityLevel.ADEQUATE
        elif ema >= self.critical_threshold + 0.05:
            self._current_scarcity = ScarcityLevel.SCARCE
        elif ema >= self.critical_threshold * 0.5:
            self._current_scarcity = ScarcityLevel.CRITICAL
        else:
            self._current_scarcity = ScarcityLevel.DEPLETED

        if self._current_scarcity != prev:
            self._scarcity_events.append(ScarcityEvent(
                cycle=self._cycle,
                level=self._current_scarcity,
                resource_level=self._resource_level,
                trigger_count=len(self._scarcity_events) + 1,
            ))

    # ------------------------------------------------------------------
    # Engine 4: Arousal Trigger Engine
    # ------------------------------------------------------------------

    def _check_arousal(self, agent: AgentEnergyState) -> None:
        """Check if torpid agent should be aroused."""
        if agent.state not in (
            HibernationState.LIGHT_TORPOR,
            HibernationState.DEEP_TORPOR,
        ):
            return

        trigger: Optional[ArousalTrigger] = None
        latency = 1

        # Emergency threat
        if self._threat_level > 0.5:
            trigger = ArousalTrigger.THREAT_SIGNAL
            latency = 1 if self._threat_level > 0.8 else 2

        # Resource recovery
        elif self._current_scarcity in (ScarcityLevel.ABUNDANT, ScarcityLevel.ADEQUATE) \
                and agent.cycles_in_state >= self.min_torpor_bout:
            trigger = ArousalTrigger.RESOURCE_RECOVERY
            latency = 3

        # Periodic arousal bout
        elif agent.total_torpor_cycles > 0 and \
                agent.total_torpor_cycles % self.periodic_arousal_interval == 0:
            trigger = ArousalTrigger.PERIODIC_BOUT
            latency = 2

        if trigger is not None:
            self._begin_arousal(agent, trigger, latency)

    def _begin_arousal(
        self, agent: AgentEnergyState,
        trigger: ArousalTrigger,
        latency: int,
    ) -> None:
        """Initiate arousal from torpor."""
        from_state = agent.state

        # Record torpor bout
        start = self._active_bout_starts.pop(agent.agent_id, self._cycle - 1)
        entry_energy = self._active_bout_energy.pop(agent.agent_id, agent.energy)
        # Energy saved = what would have been consumed at active rate minus actual
        active_cost = (self._cycle - start) * agent.metabolic_rate
        actual_cost = entry_energy - agent.energy
        energy_saved = max(0.0, active_cost - actual_cost)

        depth = "deep" if from_state == HibernationState.DEEP_TORPOR else "light"
        self._torpor_bouts.append(TorporBout(
            agent_id=agent.agent_id,
            start_cycle=start,
            end_cycle=self._cycle,
            depth=depth,
            energy_saved=energy_saved,
            trigger=trigger,
        ))

        agent.state = HibernationState.AROUSING
        agent.cycles_in_state = 0
        agent.arousal_count += 1
        agent.last_arousal_trigger = trigger

        # Remove from cluster
        if agent.cluster_id:
            self._remove_from_cluster(agent)

        self._arousal_events.append(ArousalEvent(
            agent_id=agent.agent_id,
            cycle=self._cycle,
            trigger=trigger,
            from_state=from_state,
            latency_cycles=latency,
        ))

    # ------------------------------------------------------------------
    # Engine 5: Hibernaculum Manager
    # ------------------------------------------------------------------

    def _manage_clusters(self) -> None:
        """Form and dissolve hibernation clusters."""
        # Collect torpid agents not yet clustered
        unclustered = [
            a for a in self._agents
            if a.state in (HibernationState.LIGHT_TORPOR, HibernationState.DEEP_TORPOR)
            and a.cluster_id is None
        ]

        # Try to add to existing clusters or form new ones (groups of 2-4)
        while len(unclustered) >= 2:
            group_size = min(self._rng.randint(2, 4), len(unclustered))
            group = unclustered[:group_size]
            unclustered = unclustered[group_size:]

            cluster_id = f"hib_{self._next_cluster_id:03d}"
            self._next_cluster_id += 1

            # Thermal benefit scales with group size
            thermal = min(
                self.cluster_thermal_bonus * (group_size / 4.0) * 1.2,
                0.6,
            )
            cluster = HibernationCluster(
                cluster_id=cluster_id,
                member_ids=[a.agent_id for a in group],
                formation_cycle=self._cycle,
                thermal_benefit=thermal,
                is_active=True,
            )
            self._clusters.append(cluster)
            for a in group:
                a.cluster_id = cluster_id

        # Dissolve clusters with no torpid members
        for cluster in self._clusters:
            if not cluster.is_active:
                continue
            torpid_members = [
                a for a in self._agents
                if a.agent_id in cluster.member_ids
                and a.state in (HibernationState.LIGHT_TORPOR, HibernationState.DEEP_TORPOR)
            ]
            if len(torpid_members) < 2:
                cluster.is_active = False
                for a in self._agents:
                    if a.cluster_id == cluster.cluster_id:
                        a.cluster_id = None

    def _remove_from_cluster(self, agent: AgentEnergyState) -> None:
        """Remove agent from its cluster."""
        if not agent.cluster_id:
            return
        cluster = self._find_cluster(agent.cluster_id)
        if cluster:
            if agent.agent_id in cluster.member_ids:
                cluster.member_ids.remove(agent.agent_id)
            if len(cluster.member_ids) < 2:
                cluster.is_active = False
                # Wake other members (cluster break)
                for mid in cluster.member_ids:
                    for a in self._agents:
                        if a.agent_id == mid and a.state in (
                            HibernationState.LIGHT_TORPOR,
                            HibernationState.DEEP_TORPOR,
                        ):
                            self._begin_arousal(a, ArousalTrigger.CLUSTER_BREAK, 2)
        agent.cluster_id = None

    # ------------------------------------------------------------------
    # Engine 6: Health Scorer
    # ------------------------------------------------------------------

    def _score_health(self) -> HealthScore:
        """Compute composite health score."""
        if not self._agents:
            return HealthScore(0.0, "Collapsed", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Energy reserves: average energy / 100
        avg_energy = statistics.mean(a.energy for a in self._agents)
        energy_reserves = avg_energy / 100.0

        # Torpor efficiency: energy saved per torpor cycle
        total_saved = sum(b.energy_saved for b in self._torpor_bouts)
        total_torpor = sum(a.total_torpor_cycles for a in self._agents)
        torpor_efficiency = min(1.0, total_saved / max(1, total_torpor) / 2.0)

        # Arousal responsiveness: arousals that happened vs needed
        threat_arousals = sum(
            1 for e in self._arousal_events
            if e.trigger in (ArousalTrigger.THREAT_SIGNAL, ArousalTrigger.EMERGENCY_TASK)
        )
        total_arousals = len(self._arousal_events)
        arousal_responsiveness = min(1.0, (total_arousals + 1) / (total_arousals + 5))
        if threat_arousals > 0:
            arousal_responsiveness = min(1.0, arousal_responsiveness + 0.2)

        # Cluster utilization
        active_clusters = [c for c in self._clusters if c.is_active]
        torpid_agents = [
            a for a in self._agents
            if a.state in (HibernationState.LIGHT_TORPOR, HibernationState.DEEP_TORPOR)
        ]
        clustered = sum(1 for a in torpid_agents if a.cluster_id)
        cluster_utilization = clustered / max(1, len(torpid_agents))

        # Active ratio: proportion of agents active (not torpid)
        active = sum(
            1 for a in self._agents
            if a.state in (HibernationState.ACTIVE, HibernationState.AROUSING)
        )
        active_ratio = active / len(self._agents)

        # Sustainability: do we have enough active agents and energy
        min_ratio = self.min_active_ratio
        ratio_health = min(1.0, active_ratio / max(0.01, min_ratio))
        low_energy = sum(1 for a in self._agents if a.energy < 15.0)
        low_ratio = 1.0 - (low_energy / len(self._agents))
        sustainability = (ratio_health + low_ratio) / 2.0

        # Weighted composite
        score = (
            energy_reserves * 25
            + torpor_efficiency * 20
            + arousal_responsiveness * 15
            + cluster_utilization * 10
            + active_ratio * 10
            + sustainability * 20
        )
        score = max(0.0, min(100.0, score))

        if score >= 80:
            tier = "Thriving"
        elif score >= 60:
            tier = "Conserving"
        elif score >= 40:
            tier = "Strained"
        elif score >= 20:
            tier = "Critical"
        else:
            tier = "Collapsed"

        return HealthScore(
            score=round(score, 1),
            tier=tier,
            energy_reserves=round(energy_reserves, 3),
            torpor_efficiency=round(torpor_efficiency, 3),
            arousal_responsiveness=round(arousal_responsiveness, 3),
            cluster_utilization=round(cluster_utilization, 3),
            active_ratio=round(active_ratio, 3),
            sustainability=round(sustainability, 3),
        )

    # ------------------------------------------------------------------
    # Engine 7: Insight Generator
    # ------------------------------------------------------------------

    def _generate_insights(self) -> List[str]:
        """Generate autonomous observations."""
        insights: List[str] = []

        # Energy insights
        energies = [a.energy for a in self._agents]
        avg_e = statistics.mean(energies) if energies else 0
        if avg_e < 25:
            insights.append(
                f"⚠️ Critical energy reserves: average {avg_e:.1f}/100 — "
                "swarm approaching exhaustion."
            )
        elif avg_e > 75:
            insights.append(
                f"✅ Healthy energy reserves: average {avg_e:.1f}/100."
            )

        # Torpor insights
        torpid = [
            a for a in self._agents
            if a.state in (HibernationState.LIGHT_TORPOR, HibernationState.DEEP_TORPOR)
        ]
        deep_torpid = [a for a in torpid if a.state == HibernationState.DEEP_TORPOR]
        if len(torpid) > len(self._agents) * 0.6:
            insights.append(
                f"🧊 {len(torpid)}/{len(self._agents)} agents in torpor — "
                "swarm in deep conservation mode."
            )
        if deep_torpid:
            insights.append(
                f"❄️ {len(deep_torpid)} agents in deep torpor — "
                "maximum energy conservation."
            )

        # Cluster insights
        active_clusters = [c for c in self._clusters if c.is_active]
        if active_clusters:
            total_members = sum(len(c.member_ids) for c in active_clusters)
            avg_benefit = statistics.mean(c.thermal_benefit for c in active_clusters)
            insights.append(
                f"🏠 {len(active_clusters)} hibernacula active "
                f"({total_members} agents, avg thermal benefit {avg_benefit:.1%})."
            )

        # Scarcity insights
        if self._current_scarcity in (ScarcityLevel.CRITICAL, ScarcityLevel.DEPLETED):
            insights.append(
                f"🔴 Resource scarcity at {self._current_scarcity.value} level — "
                f"EMA: {self._resource_ema:.2f}."
            )
        elif self._current_scarcity == ScarcityLevel.SCARCE:
            insights.append(
                "🟡 Resources scarce — hibernation signals active."
            )

        # Arousal patterns
        if self._arousal_events:
            trigger_counts: Dict[str, int] = defaultdict(int)
            for e in self._arousal_events:
                trigger_counts[e.trigger.value] += 1
            top = max(trigger_counts, key=trigger_counts.get)  # type: ignore[arg-type]
            insights.append(
                f"🔔 Most common arousal trigger: {top} "
                f"({trigger_counts[top]} events)."
            )

        # Energy variance
        if len(energies) > 1:
            std_e = statistics.stdev(energies)
            if std_e > 25:
                insights.append(
                    f"⚡ High energy inequality (σ={std_e:.1f}) — "
                    "consider load redistribution."
                )

        # Bout efficiency
        if self._torpor_bouts:
            avg_saved = statistics.mean(b.energy_saved for b in self._torpor_bouts)
            if avg_saved > 5:
                insights.append(
                    f"💤 Torpor bouts saving avg {avg_saved:.1f} energy units — "
                    "effective conservation."
                )

        return insights

    # ------------------------------------------------------------------
    # Tick & Simulate
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance one cycle."""
        self._cycle += 1

        # Detect scarcity
        self._detect_scarcity()

        # Update each agent
        for agent in self._agents:
            self._update_energy(agent)
            self._check_torpor_entry(agent)
            self._check_arousal(agent)

        # Manage clusters
        self._manage_clusters()

        # Decay threat
        self._threat_level *= 0.7

        # Record cycle history
        active_count = sum(
            1 for a in self._agents
            if a.state in (HibernationState.ACTIVE, HibernationState.AROUSING, HibernationState.DROWSY)
        )
        torpid_count = sum(
            1 for a in self._agents
            if a.state in (HibernationState.LIGHT_TORPOR, HibernationState.DEEP_TORPOR)
        )
        energies = [a.energy for a in self._agents]
        self._cycle_history.append({
            "cycle": self._cycle,
            "avg_energy": round(statistics.mean(energies), 2) if energies else 0,
            "min_energy": round(min(energies), 2) if energies else 0,
            "active_count": active_count,
            "torpid_count": torpid_count,
            "resource_level": round(self._resource_level, 3),
            "scarcity": self._current_scarcity.value,
            "cluster_count": sum(1 for c in self._clusters if c.is_active),
        })

    def _apply_resource_schedule(self, cycle: int, total: int) -> None:
        """Apply resource schedule from scenario config."""
        cfg = getattr(self, "_scenario_cfg", {})
        schedule = cfg.get("resource_schedule", "constant")
        min_res = cfg.get("min_resource", 0.2)
        inject = cfg.get("inject_threats", False)

        progress = cycle / max(1, total)

        if schedule == "gradual_drop":
            self._resource_level = max(min_res, 1.0 - progress * (1.0 - min_res))
        elif schedule == "severe_drop":
            if progress < 0.2:
                self._resource_level = 1.0 - progress * 4.0 * (1.0 - min_res)
            else:
                self._resource_level = min_res + self._rng.uniform(-0.05, 0.05)
            self._resource_level = max(min_res * 0.5, min(1.0, self._resource_level))
        elif schedule == "oscillating":
            # Sinusoidal oscillation
            self._resource_level = max(
                min_res,
                0.5 + 0.4 * math.sin(progress * math.pi * 4),
            )
        else:
            self._resource_level = 0.8

        if inject and cycle % 25 == 0 and cycle > 10:
            self.inject_threat(self._rng.uniform(0.6, 0.95))

    def simulate(self, cycles: int = 80) -> HibernationReport:
        """Run full simulation."""
        has_schedule = hasattr(self, "_scenario_cfg")

        for c in range(cycles):
            if has_schedule:
                self._apply_resource_schedule(c, cycles)
            self.tick()

        return self.get_report()

    def get_report(self) -> HibernationReport:
        """Generate current report."""
        return HibernationReport(
            agents=list(self._agents),
            clusters=list(self._clusters),
            torpor_bouts=list(self._torpor_bouts),
            arousal_events=list(self._arousal_events),
            scarcity_events=list(self._scarcity_events),
            health=self._score_health(),
            insights=self._generate_insights(),
            cycle_history=list(self._cycle_history),
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self, report: Optional[HibernationReport] = None) -> Dict[str, Any]:
        """Serialize report to dict."""
        if report is None:
            report = self.get_report()
        return {
            "health": asdict(report.health),
            "insights": report.insights,
            "torpor_bout_count": len(report.torpor_bouts),
            "arousal_event_count": len(report.arousal_events),
            "scarcity_event_count": len(report.scarcity_events),
            "cluster_count": len([c for c in report.clusters if c.is_active]),
            "cycle_history": report.cycle_history,
            "agents": [
                {
                    "id": a.agent_id,
                    "energy": round(a.energy, 1),
                    "state": a.state.value,
                    "torpor_bouts": a.torpor_bouts,
                    "total_torpor_cycles": a.total_torpor_cycles,
                    "cluster_id": a.cluster_id,
                    "arousal_count": a.arousal_count,
                }
                for a in report.agents
            ],
        }

    def export_html(self, path: str, report: Optional[HibernationReport] = None) -> None:
        """Export interactive HTML dashboard."""
        if report is None:
            report = self.get_report()

        health = report.health
        insights_html = "".join(
            f"<li>{html_mod.escape(i)}</li>" for i in report.insights
        )

        state_counts: Dict[str, int] = defaultdict(int)
        for a in report.agents:
            state_counts[a.state.value] += 1

        # Cycle chart data (sample every 5 cycles)
        step = max(1, len(report.cycle_history) // 30)
        cycle_labels = [str(c["cycle"]) for c in report.cycle_history[::step]]
        energy_data = [c["avg_energy"] for c in report.cycle_history[::step]]
        torpid_data = [c["torpid_count"] for c in report.cycle_history[::step]]

        # Cluster summary
        active_clusters = [c for c in report.clusters if c.is_active]
        cluster_rows = "".join(
            f"<tr><td>{html_mod.escape(c.cluster_id)}</td>"
            f"<td>{len(c.member_ids)}</td>"
            f"<td>{c.thermal_benefit:.1%}</td></tr>"
            for c in active_clusters
        )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Swarm Hibernation Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f0f23; color: #e0e0e0; padding: 20px; }}
.header {{ text-align: center; padding: 30px; }}
.header h1 {{ font-size: 2em; color: #4fc3f7; }}
.header .tier {{ font-size: 1.5em; margin-top: 10px; }}
.score-gauge {{ font-size: 3em; font-weight: bold; color: #4fc3f7; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
         gap: 20px; margin: 20px 0; }}
.card {{ background: #1a1a3e; border-radius: 12px; padding: 20px;
         border: 1px solid #333366; }}
.card h2 {{ color: #4fc3f7; margin-bottom: 15px; font-size: 1.1em; }}
.metric {{ display: flex; justify-content: space-between; padding: 8px 0;
           border-bottom: 1px solid #2a2a4e; }}
.metric:last-child {{ border-bottom: none; }}
.bar {{ height: 8px; background: #2a2a4e; border-radius: 4px; margin-top: 4px; }}
.bar-fill {{ height: 100%; border-radius: 4px; background: #4fc3f7; }}
ul {{ list-style: none; }}
ul li {{ padding: 8px 0; border-bottom: 1px solid #2a2a4e; }}
ul li:last-child {{ border-bottom: none; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #2a2a4e; }}
th {{ color: #4fc3f7; }}
.state-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
               font-size: 0.8em; margin-left: 5px; }}
.state-active {{ background: #004d40; color: #00e676; }}
.state-drowsy {{ background: #4a148c; color: #ce93d8; }}
.state-light_torpor {{ background: #1a237e; color: #448aff; }}
.state-deep_torpor {{ background: #0d47a1; color: #80d8ff; }}
.state-arousing {{ background: #e65100; color: #ffab40; }}
</style>
</head>
<body>
<div class="header">
    <h1>🧊 Swarm Hibernation Report</h1>
    <div class="score-gauge">{health.score}</div>
    <div class="tier">{health.tier}</div>
</div>

<div class="grid">
    <div class="card">
        <h2>📊 Health Dimensions</h2>
        <div class="metric"><span>Energy Reserves</span><span>{health.energy_reserves:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.energy_reserves*100:.0f}%"></div></div>
        <div class="metric"><span>Torpor Efficiency</span><span>{health.torpor_efficiency:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.torpor_efficiency*100:.0f}%"></div></div>
        <div class="metric"><span>Arousal Responsiveness</span><span>{health.arousal_responsiveness:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.arousal_responsiveness*100:.0f}%"></div></div>
        <div class="metric"><span>Cluster Utilization</span><span>{health.cluster_utilization:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.cluster_utilization*100:.0f}%"></div></div>
        <div class="metric"><span>Active Ratio</span><span>{health.active_ratio:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.active_ratio*100:.0f}%"></div></div>
        <div class="metric"><span>Sustainability</span><span>{health.sustainability:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.sustainability*100:.0f}%"></div></div>
    </div>

    <div class="card">
        <h2>🏷️ Agent States</h2>
        {"".join(
            f'<div class="metric"><span>{s}</span><span>{state_counts.get(s, 0)}</span></div>'
            for s in ["active", "drowsy", "light_torpor", "deep_torpor", "arousing"]
        )}
    </div>

    <div class="card">
        <h2>🏠 Active Hibernacula</h2>
        {"<table><tr><th>Cluster</th><th>Members</th><th>Thermal</th></tr>"
         + cluster_rows + "</table>" if cluster_rows else "<p>No active clusters</p>"}
    </div>

    <div class="card">
        <h2>📈 Simulation Stats</h2>
        <div class="metric"><span>Total Torpor Bouts</span><span>{len(report.torpor_bouts)}</span></div>
        <div class="metric"><span>Arousal Events</span><span>{len(report.arousal_events)}</span></div>
        <div class="metric"><span>Scarcity Transitions</span><span>{len(report.scarcity_events)}</span></div>
    </div>

    <div class="card" style="grid-column: 1 / -1;">
        <h2>💡 Autonomous Insights</h2>
        <ul>{insights_html if insights_html else "<li>No insights generated</li>"}</ul>
    </div>
</div>
</body>
</html>"""

        Path(path).write_text(html_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Swarm Hibernation Engine — autonomous energy conservation",
    )
    parser.add_argument("--agents", type=int, default=12)
    parser.add_argument("--cycles", type=int, default=80)
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="HTML output path")
    parser.add_argument("--json", type=str, default=None, help="JSON output path")
    args = parser.parse_args(argv)

    if args.scenario:
        engine = SwarmHibernationEngine.from_scenario(args.scenario, seed=args.seed)
        cycles = SCENARIOS[args.scenario].get("cycles", args.cycles)
    else:
        engine = SwarmHibernationEngine(num_agents=args.agents, seed=args.seed)
        cycles = args.cycles

    report = engine.simulate(cycles=cycles)
    health = report.health

    print(f"\n🧊 Swarm Hibernation Report")
    print(f"{'=' * 50}")
    print(f"  Score: {health.score}/100 — {health.tier}")
    print(f"  Energy Reserves:       {health.energy_reserves:.1%}")
    print(f"  Torpor Efficiency:     {health.torpor_efficiency:.1%}")
    print(f"  Arousal Responsiveness:{health.arousal_responsiveness:.1%}")
    print(f"  Cluster Utilization:   {health.cluster_utilization:.1%}")
    print(f"  Active Ratio:          {health.active_ratio:.1%}")
    print(f"  Sustainability:        {health.sustainability:.1%}")
    print(f"  Torpor Bouts:          {len(report.torpor_bouts)}")
    print(f"  Arousal Events:        {len(report.arousal_events)}")
    print(f"  Scarcity Transitions:  {len(report.scarcity_events)}")
    print()

    for insight in report.insights:
        print(f"  {insight}")

    if args.out:
        engine.export_html(args.out, report)
        print(f"\n  📄 HTML report: {args.out}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(engine.to_dict(report), indent=2),
            encoding="utf-8",
        )
        print(f"  📄 JSON report: {args.json}")


if __name__ == "__main__":
    _main()
