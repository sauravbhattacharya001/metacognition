"""Swarm Senescence Engine — autonomous agent aging and rejuvenation.

Biologically inspired by cellular senescence (Hayflick limit, telomere
shortening, senescence-associated secretory phenotype SASP, stem cell
rejuvenation, autophagy-mediated renewal).  Agents age with each task
cycle, their telomeres shorten, and eventually they enter senescence —
unless caught early enough for rejuvenation.

Capabilities:

- **Telomere Tracker** — each agent has telomere length (0-100) that
  shortens with each task cycle; critical shortening triggers senescence.
- **SASP Detector** — senescent agents emit inflammatory signals that
  accelerate aging in neighbors (bystander effect) within a configurable
  radius.
- **Rejuvenation Engine** — stem-cell-like renewal; agents in
  PRE_SENESCENT state can be rejuvenated (telomere extension, stress
  reset) up to a Hayflick limit of total rejuvenations.
- **Retirement Scheduler** — identifies agents past recovery, schedules
  graceful retirement with knowledge transfer to younger agents.
- **Longevity Optimizer** — analyzes workload patterns to find optimal
  task distribution that maximizes swarm lifespan.
- **Health Scorer** — composite 0-100 metric: telomere reserves, SASP
  burden, rejuvenation success, retirement efficiency, age diversity,
  population sustainability.
- **Insight Generator** — autonomous observations about aging patterns,
  rejuvenation effectiveness, population sustainability.

Usage (Python API)::

    from src.senescence import SwarmSenescenceEngine

    engine = SwarmSenescenceEngine(num_agents=15)
    report = engine.simulate(cycles=100)
    print(report.health.score)       # 0-100
    print(report.health.tier)        # Immortal/Thriving/Aging/Declining/Collapsing
    print(report.insights)           # autonomous observations
    engine.export_html("senescence_report.html")

CLI::

    python -m src.senescence                          # demo with defaults
    python -m src.senescence --agents 20 --cycles 100
    python -m src.senescence --scenario aging_crisis
    python -m src.senescence --out report.html --json senescence.json
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


class SenescenceState(str, Enum):
    """Agent lifecycle states."""
    YOUNG = "young"
    MATURE = "mature"
    PRE_SENESCENT = "pre_senescent"
    SENESCENT = "senescent"
    RETIRED = "retired"


class RetirementReason(str, Enum):
    """Why an agent was retired."""
    TELOMERE_EXHAUSTION = "telomere_exhaustion"
    HAYFLICK_LIMIT = "hayflick_limit"
    SASP_OVERLOAD = "sasp_overload"
    VOLUNTARY = "voluntary"


class RejuvenationMethod(str, Enum):
    """Methods of rejuvenation."""
    TELOMERE_EXTENSION = "telomere_extension"
    STRESS_RESET = "stress_reset"
    EPIGENETIC_REPROGRAMMING = "epigenetic_reprogramming"


@dataclass
class AgentBiology:
    """Biological state of a single swarm agent."""
    agent_id: str
    telomere_length: float = 100.0
    age: int = 0
    state: SenescenceState = SenescenceState.YOUNG
    rejuvenation_count: int = 0
    knowledge_store: float = 0.0
    stress_level: float = 0.0
    sasp_exposure: float = 0.0
    total_tasks: int = 0


@dataclass
class SASPSignal:
    """Senescence-associated secretory phenotype signal."""
    source_agent: str
    strength: float
    radius: float
    decay_rate: float
    age: int = 0


@dataclass
class RejuvenationEvent:
    """Record of a rejuvenation attempt."""
    agent_id: str
    cycle: int
    old_telomere: float
    new_telomere: float
    method: RejuvenationMethod
    success: bool


@dataclass
class RetirementRecord:
    """Record of an agent retirement."""
    agent_id: str
    cycle: int
    age_at_retirement: int
    knowledge_transferred_to: List[str]
    reason: RetirementReason
    knowledge_amount: float


@dataclass
class PopulationStats:
    """Population-level statistics."""
    total_agents: int
    active_agents: int
    retired_agents: int
    senescent_agents: int
    avg_telomere: float
    avg_age: float
    birth_rate: float
    retirement_rate: float
    sustainability_ratio: float


@dataclass
class HealthScore:
    """Composite health assessment."""
    score: float
    tier: str
    telomere_reserve: float
    sasp_burden: float
    rejuvenation_rate: float
    retirement_efficiency: float
    age_diversity: float
    sustainability: float


@dataclass
class SenescenceReport:
    """Full simulation report."""
    agents: List[AgentBiology]
    sasp_signals: List[SASPSignal]
    rejuvenations: List[RejuvenationEvent]
    retirements: List[RetirementRecord]
    health: HealthScore
    insights: List[str]
    population_stats: PopulationStats
    cycle_history: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "healthy_swarm": {
        "num_agents": 15,
        "cycles": 80,
        "base_shortening": 0.5,
        "sasp_strength": 0.3,
        "rejuvenation_chance": 0.8,
        "hayflick_limit": 5,
        "description": "Well-maintained swarm with good rejuvenation",
    },
    "aging_crisis": {
        "num_agents": 15,
        "cycles": 120,
        "base_shortening": 1.2,
        "sasp_strength": 0.8,
        "rejuvenation_chance": 0.3,
        "hayflick_limit": 2,
        "description": "Rapidly aging swarm with poor rejuvenation",
    },
    "sasp_cascade": {
        "num_agents": 20,
        "cycles": 100,
        "base_shortening": 0.7,
        "sasp_strength": 1.5,
        "rejuvenation_chance": 0.5,
        "hayflick_limit": 3,
        "description": "High SASP propagation causing cascade aging",
    },
    "rejuvenation_success": {
        "num_agents": 12,
        "cycles": 150,
        "base_shortening": 0.8,
        "sasp_strength": 0.4,
        "rejuvenation_chance": 0.95,
        "hayflick_limit": 8,
        "description": "Excellent rejuvenation keeps population young",
    },
    "population_collapse": {
        "num_agents": 10,
        "cycles": 200,
        "base_shortening": 1.5,
        "sasp_strength": 1.2,
        "rejuvenation_chance": 0.1,
        "hayflick_limit": 1,
        "description": "Rapid aging with no recovery leads to collapse",
    },
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SwarmSenescenceEngine:
    """Autonomous swarm agent aging and rejuvenation engine."""

    def __init__(
        self,
        num_agents: int = 15,
        base_shortening: float = 0.8,
        sasp_strength: float = 0.5,
        sasp_radius: float = 3.0,
        sasp_decay: float = 0.1,
        rejuvenation_chance: float = 0.6,
        rejuvenation_boost: float = 30.0,
        hayflick_limit: int = 4,
        critical_telomere: float = 20.0,
        senescent_telomere: float = 10.0,
        max_sasp_tolerable: int = 10,
        seed: Optional[int] = None,
    ):
        self.num_agents = num_agents
        self.base_shortening = base_shortening
        self.sasp_strength = sasp_strength
        self.sasp_radius = sasp_radius
        self.sasp_decay = sasp_decay
        self.rejuvenation_chance = rejuvenation_chance
        self.rejuvenation_boost = rejuvenation_boost
        self.hayflick_limit = hayflick_limit
        self.critical_telomere = critical_telomere
        self.senescent_telomere = senescent_telomere
        self.max_sasp_tolerable = max_sasp_tolerable

        self._rng = random.Random(seed)
        self._agents: List[AgentBiology] = []
        self._sasp_signals: List[SASPSignal] = []
        self._rejuvenations: List[RejuvenationEvent] = []
        self._retirements: List[RetirementRecord] = []
        self._cycle_history: List[Dict[str, Any]] = []
        self._total_births: int = 0
        self._total_retirements: int = 0
        self._graceful_retirements: int = 0

        self._init_agents()

    def _init_agents(self) -> None:
        """Initialize agent population."""
        self._agents = []
        for i in range(self.num_agents):
            agent = AgentBiology(
                agent_id=f"agent_{i:03d}",
                telomere_length=self._rng.uniform(80.0, 100.0),
                age=self._rng.randint(0, 10),
                state=SenescenceState.YOUNG,
                knowledge_store=self._rng.uniform(0.0, 5.0),
            )
            self._agents.append(agent)
        self._total_births = self.num_agents

    def _get_active_agents(self) -> List[AgentBiology]:
        """Return agents that are not retired."""
        return [a for a in self._agents if a.state != SenescenceState.RETIRED]

    # --- Engine 1: Telomere Tracker ---

    def _shorten_telomeres(self, cycle: int) -> None:
        """Apply telomere shortening to all active agents."""
        for agent in self._get_active_agents():
            if agent.state == SenescenceState.SENESCENT:
                continue  # Already senescent, no further shortening

            # Base shortening + stress factor + SASP exposure
            stress_factor = 1.0 + agent.stress_level * 0.5
            sasp_factor = 1.0 + agent.sasp_exposure * 0.3
            shortening = self.base_shortening * stress_factor * sasp_factor

            # Add task-based variation
            shortening *= self._rng.uniform(0.8, 1.2)

            agent.telomere_length = max(0.0, agent.telomere_length - shortening)
            agent.age += 1
            agent.total_tasks += 1

            # Accumulate knowledge
            agent.knowledge_store += self._rng.uniform(0.1, 0.5)

            # Random stress fluctuation
            agent.stress_level = max(0.0, min(1.0,
                agent.stress_level + self._rng.uniform(-0.1, 0.15)))

            # Update state based on telomere length
            self._update_agent_state(agent)

    def _update_agent_state(self, agent: AgentBiology) -> None:
        """Update agent state based on telomere length."""
        if agent.state == SenescenceState.RETIRED:
            return
        if agent.telomere_length <= self.senescent_telomere:
            agent.state = SenescenceState.SENESCENT
        elif agent.telomere_length <= self.critical_telomere:
            agent.state = SenescenceState.PRE_SENESCENT
        elif agent.telomere_length <= 60.0:
            agent.state = SenescenceState.MATURE
        else:
            agent.state = SenescenceState.YOUNG

    # --- Engine 2: SASP Detector ---

    def _emit_sasp(self, cycle: int) -> None:
        """Senescent agents emit SASP signals."""
        # Decay existing signals
        self._sasp_signals = [
            s for s in self._sasp_signals
            if s.strength > 0.1
        ]
        for signal in self._sasp_signals:
            signal.strength *= (1.0 - signal.decay_rate)
            signal.age += 1

        # Emit new signals from senescent agents
        for agent in self._get_active_agents():
            if agent.state == SenescenceState.SENESCENT:
                signal = SASPSignal(
                    source_agent=agent.agent_id,
                    strength=self.sasp_strength * (1.0 + agent.stress_level),
                    radius=self.sasp_radius,
                    decay_rate=self.sasp_decay,
                )
                self._sasp_signals.append(signal)

    def _apply_sasp_bystander(self) -> None:
        """Apply bystander effect from SASP signals to nearby agents."""
        active = self._get_active_agents()
        # Simple proximity model: agents are on a 1D line by index
        agent_positions = {a.agent_id: i for i, a in enumerate(active)}

        for agent in active:
            if agent.state in (SenescenceState.SENESCENT, SenescenceState.RETIRED):
                continue
            exposure = 0.0
            pos = agent_positions.get(agent.agent_id, 0)
            for signal in self._sasp_signals:
                src_pos = agent_positions.get(signal.source_agent, -999)
                distance = abs(pos - src_pos)
                if distance <= signal.radius and signal.source_agent != agent.agent_id:
                    # Inverse distance weighting
                    weight = 1.0 / (1.0 + distance)
                    exposure += signal.strength * weight
            agent.sasp_exposure = min(2.0, exposure)

    # --- Engine 3: Rejuvenation Engine ---

    def _attempt_rejuvenation(self, cycle: int) -> None:
        """Attempt to rejuvenate pre-senescent agents."""
        for agent in self._get_active_agents():
            if agent.state != SenescenceState.PRE_SENESCENT:
                continue
            if agent.rejuvenation_count >= self.hayflick_limit:
                # Past Hayflick limit, cannot rejuvenate
                continue

            # Attempt rejuvenation
            if self._rng.random() < self.rejuvenation_chance:
                old_telomere = agent.telomere_length
                boost = self.rejuvenation_boost * self._rng.uniform(0.7, 1.0)
                agent.telomere_length = min(100.0, agent.telomere_length + boost)
                agent.stress_level *= 0.5  # Stress reset
                agent.sasp_exposure *= 0.3
                agent.rejuvenation_count += 1

                method = self._rng.choice(list(RejuvenationMethod))
                self._update_agent_state(agent)

                self._rejuvenations.append(RejuvenationEvent(
                    agent_id=agent.agent_id,
                    cycle=cycle,
                    old_telomere=old_telomere,
                    new_telomere=agent.telomere_length,
                    method=method,
                    success=True,
                ))
            else:
                self._rejuvenations.append(RejuvenationEvent(
                    agent_id=agent.agent_id,
                    cycle=cycle,
                    old_telomere=agent.telomere_length,
                    new_telomere=agent.telomere_length,
                    method=RejuvenationMethod.TELOMERE_EXTENSION,
                    success=False,
                ))

    # --- Engine 4: Retirement Scheduler ---

    def _schedule_retirements(self, cycle: int) -> None:
        """Retire agents that are past recovery."""
        to_retire: List[AgentBiology] = []
        for agent in self._get_active_agents():
            if agent.state == SenescenceState.SENESCENT:
                # Check if past Hayflick limit or telomere fully depleted
                if (agent.rejuvenation_count >= self.hayflick_limit or
                        agent.telomere_length <= 2.0):
                    to_retire.append(agent)
            # Also retire if SASP-overloaded
            elif agent.sasp_exposure >= 1.8 and agent.state == SenescenceState.PRE_SENESCENT:
                if agent.rejuvenation_count >= self.hayflick_limit:
                    to_retire.append(agent)

        for agent in to_retire:
            self._retire_agent(agent, cycle)

    def _retire_agent(self, agent: AgentBiology, cycle: int) -> None:
        """Execute graceful retirement with knowledge transfer."""
        # Find young agents to transfer knowledge to
        young_agents = [
            a for a in self._get_active_agents()
            if a.state in (SenescenceState.YOUNG, SenescenceState.MATURE)
            and a.agent_id != agent.agent_id
        ]

        recipients: List[str] = []
        knowledge_per_recipient = agent.knowledge_store / max(1, min(3, len(young_agents)))

        for recipient in young_agents[:3]:
            recipient.knowledge_store += knowledge_per_recipient
            recipients.append(recipient.agent_id)

        # Determine reason
        if agent.sasp_exposure >= 1.8:
            reason = RetirementReason.SASP_OVERLOAD
        elif agent.rejuvenation_count >= self.hayflick_limit:
            reason = RetirementReason.HAYFLICK_LIMIT
        else:
            reason = RetirementReason.TELOMERE_EXHAUSTION

        agent.state = SenescenceState.RETIRED
        self._total_retirements += 1
        if recipients:
            self._graceful_retirements += 1

        self._retirements.append(RetirementRecord(
            agent_id=agent.agent_id,
            cycle=cycle,
            age_at_retirement=agent.age,
            knowledge_transferred_to=recipients,
            reason=reason,
            knowledge_amount=agent.knowledge_store,
        ))

        # Spawn replacement agent
        self._spawn_replacement(cycle)

    def _spawn_replacement(self, cycle: int) -> None:
        """Spawn a new agent to replace a retired one."""
        new_id = f"agent_{len(self._agents):03d}"
        new_agent = AgentBiology(
            agent_id=new_id,
            telomere_length=self._rng.uniform(90.0, 100.0),
            age=0,
            state=SenescenceState.YOUNG,
            knowledge_store=0.0,
        )
        self._agents.append(new_agent)
        self._total_births += 1

    # --- Engine 5: Longevity Optimizer ---

    def _optimize_longevity(self) -> Dict[str, Any]:
        """Analyze workload patterns for optimal lifespan distribution."""
        active = self._get_active_agents()
        if not active:
            return {"recommendation": "no_active_agents", "actions": []}

        avg_stress = statistics.mean(a.stress_level for a in active)
        avg_telomere = statistics.mean(a.telomere_length for a in active)
        high_stress_count = sum(1 for a in active if a.stress_level > 0.7)

        actions: List[str] = []
        if avg_stress > 0.5:
            actions.append("reduce_global_workload")
        if high_stress_count > len(active) * 0.3:
            actions.append("redistribute_tasks_from_stressed_agents")
        if avg_telomere < 40.0:
            actions.append("prioritize_rejuvenation_batch")

        # Find agents that should rest
        rest_candidates = [
            a.agent_id for a in active
            if a.stress_level > 0.6 and a.state == SenescenceState.MATURE
        ]
        if rest_candidates:
            actions.append(f"rest_agents:{','.join(rest_candidates[:3])}")

        return {
            "avg_stress": round(avg_stress, 3),
            "avg_telomere": round(avg_telomere, 1),
            "high_stress_ratio": round(high_stress_count / max(1, len(active)), 3),
            "actions": actions,
        }

    # --- Engine 6: Health Scorer ---

    def _compute_health(self) -> HealthScore:
        """Compute composite senescence health score."""
        active = self._get_active_agents()

        # Telomere reserve
        if active:
            telomere_reserve = statistics.mean(a.telomere_length for a in active) / 100.0
        else:
            telomere_reserve = 0.0

        # SASP burden
        active_sasp = len([s for s in self._sasp_signals if s.strength > 0.1])
        sasp_burden = max(0.0, 1.0 - active_sasp / max(1, self.max_sasp_tolerable))

        # Rejuvenation rate
        successful = sum(1 for r in self._rejuvenations if r.success)
        total_attempts = len(self._rejuvenations)
        rejuvenation_rate = successful / max(1, total_attempts)

        # Retirement efficiency (graceful/total)
        retirement_efficiency = (
            self._graceful_retirements / max(1, self._total_retirements)
        )

        # Age diversity (Shannon entropy of age buckets)
        if active:
            age_buckets: Dict[str, int] = defaultdict(int)
            for a in active:
                bucket = f"{a.age // 20 * 20}-{a.age // 20 * 20 + 19}"
                age_buckets[bucket] += 1
            total = len(active)
            entropy = 0.0
            for count in age_buckets.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
            # Normalize: max entropy for 5 buckets
            max_entropy = math.log2(max(2, len(age_buckets)))
            age_diversity = min(1.0, entropy / max(0.01, max_entropy))
        else:
            age_diversity = 0.0

        # Sustainability
        if self._total_retirements > 0:
            sustainability = min(1.0, self._total_births / max(1, self._total_retirements + self._total_births) * 2)
        else:
            sustainability = 1.0

        # Composite score
        score = (
            telomere_reserve * 0.25 +
            sasp_burden * 0.20 +
            rejuvenation_rate * 0.15 +
            retirement_efficiency * 0.15 +
            age_diversity * 0.15 +
            sustainability * 0.10
        ) * 100.0
        score = max(0.0, min(100.0, score))

        # Tier
        if score >= 80:
            tier = "Immortal"
        elif score >= 60:
            tier = "Thriving"
        elif score >= 40:
            tier = "Aging"
        elif score >= 20:
            tier = "Declining"
        else:
            tier = "Collapsing"

        return HealthScore(
            score=round(score, 1),
            tier=tier,
            telomere_reserve=round(telomere_reserve, 3),
            sasp_burden=round(sasp_burden, 3),
            rejuvenation_rate=round(rejuvenation_rate, 3),
            retirement_efficiency=round(retirement_efficiency, 3),
            age_diversity=round(age_diversity, 3),
            sustainability=round(sustainability, 3),
        )

    # --- Engine 7: Insight Generator ---

    def _generate_insights(self) -> List[str]:
        """Generate autonomous observations about aging patterns."""
        insights: List[str] = []
        active = self._get_active_agents()

        if not active:
            insights.append("⚠️ CRITICAL: No active agents remaining — population has collapsed.")
            return insights

        avg_telomere = statistics.mean(a.telomere_length for a in active)
        senescent_count = sum(1 for a in active if a.state == SenescenceState.SENESCENT)
        pre_sen_count = sum(1 for a in active if a.state == SenescenceState.PRE_SENESCENT)

        if avg_telomere < 30:
            insights.append(
                f"🔴 Average telomere length critically low ({avg_telomere:.1f}/100) — "
                "swarm aging rapidly."
            )
        elif avg_telomere > 70:
            insights.append(
                f"🟢 Excellent telomere reserves ({avg_telomere:.1f}/100) — "
                "population is biologically young."
            )

        if senescent_count > len(active) * 0.3:
            insights.append(
                f"⚠️ {senescent_count}/{len(active)} agents are senescent — "
                "SASP cascade risk is HIGH."
            )

        if pre_sen_count > len(active) * 0.4:
            insights.append(
                f"🟡 {pre_sen_count} agents in pre-senescent state — "
                "rejuvenation window closing soon."
            )

        # SASP analysis
        active_sasp = [s for s in self._sasp_signals if s.strength > 0.3]
        if len(active_sasp) > 5:
            insights.append(
                f"🔥 {len(active_sasp)} active SASP signals — bystander "
                "effect is accelerating population aging."
            )

        # Rejuvenation effectiveness
        successful = sum(1 for r in self._rejuvenations if r.success)
        total = len(self._rejuvenations)
        if total > 5:
            rate = successful / total
            if rate > 0.8:
                insights.append(
                    f"💚 Rejuvenation success rate excellent ({rate:.0%}) — "
                    "population sustainability is strong."
                )
            elif rate < 0.3:
                insights.append(
                    f"💔 Rejuvenation success rate poor ({rate:.0%}) — "
                    "consider improving rejuvenation capacity."
                )

        # Knowledge preservation
        if self._retirements:
            graceful_pct = self._graceful_retirements / len(self._retirements)
            if graceful_pct < 0.5:
                insights.append(
                    "📚 Knowledge loss risk: many retirements lack knowledge "
                    "transfer recipients."
                )

        # Longevity optimization
        high_stress = [a for a in active if a.stress_level > 0.7]
        if len(high_stress) > len(active) * 0.4:
            insights.append(
                f"😰 {len(high_stress)} agents under high stress — workload "
                "redistribution recommended."
            )

        return insights

    # --- Population Stats ---

    def _compute_population_stats(self) -> PopulationStats:
        """Compute population-level statistics."""
        active = self._get_active_agents()
        retired = [a for a in self._agents if a.state == SenescenceState.RETIRED]
        senescent = [a for a in active if a.state == SenescenceState.SENESCENT]

        avg_telomere = statistics.mean(a.telomere_length for a in active) if active else 0.0
        avg_age = statistics.mean(a.age for a in active) if active else 0.0

        total_cycles = max(1, len(self._cycle_history))
        birth_rate = self._total_births / total_cycles
        retirement_rate = self._total_retirements / total_cycles
        sustainability = birth_rate / max(0.01, retirement_rate)

        return PopulationStats(
            total_agents=len(self._agents),
            active_agents=len(active),
            retired_agents=len(retired),
            senescent_agents=len(senescent),
            avg_telomere=round(avg_telomere, 1),
            avg_age=round(avg_age, 1),
            birth_rate=round(birth_rate, 3),
            retirement_rate=round(retirement_rate, 3),
            sustainability_ratio=round(sustainability, 3),
        )

    # --- Simulation ---

    def simulate(self, cycles: int = 100) -> SenescenceReport:
        """Run the full senescence simulation."""
        for cycle in range(cycles):
            self._shorten_telomeres(cycle)
            self._emit_sasp(cycle)
            self._apply_sasp_bystander()
            self._attempt_rejuvenation(cycle)
            self._schedule_retirements(cycle)

            # Record cycle state
            active = self._get_active_agents()
            self._cycle_history.append({
                "cycle": cycle,
                "active_count": len(active),
                "avg_telomere": round(
                    statistics.mean(a.telomere_length for a in active), 1
                ) if active else 0.0,
                "senescent_count": sum(
                    1 for a in active if a.state == SenescenceState.SENESCENT
                ),
                "sasp_count": len([s for s in self._sasp_signals if s.strength > 0.1]),
            })

        health = self._compute_health()
        insights = self._generate_insights()
        pop_stats = self._compute_population_stats()

        return SenescenceReport(
            agents=list(self._agents),
            sasp_signals=list(self._sasp_signals),
            rejuvenations=list(self._rejuvenations),
            retirements=list(self._retirements),
            health=health,
            insights=insights,
            population_stats=pop_stats,
            cycle_history=list(self._cycle_history),
        )

    @classmethod
    def from_scenario(cls, name: str, seed: Optional[int] = None) -> "SwarmSenescenceEngine":
        """Create engine from a named scenario preset."""
        if name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
        cfg = SCENARIOS[name]
        return cls(
            num_agents=cfg["num_agents"],
            base_shortening=cfg["base_shortening"],
            sasp_strength=cfg["sasp_strength"],
            rejuvenation_chance=cfg["rejuvenation_chance"],
            hayflick_limit=cfg["hayflick_limit"],
            cycles=cfg.get("cycles", 100),
            seed=seed,
        ) if False else cls(
            num_agents=cfg["num_agents"],
            base_shortening=cfg["base_shortening"],
            sasp_strength=cfg["sasp_strength"],
            rejuvenation_chance=cfg["rejuvenation_chance"],
            hayflick_limit=cfg["hayflick_limit"],
            seed=seed,
        )

    def scenario_cycles(self, name: str) -> int:
        """Get recommended cycles for a scenario."""
        if name in SCENARIOS:
            return SCENARIOS[name].get("cycles", 100)
        return 100

    # --- Export ---

    def export_json(self, report: SenescenceReport) -> Dict[str, Any]:
        """Export report as JSON-serializable dict."""
        return {
            "health": asdict(report.health),
            "population_stats": asdict(report.population_stats),
            "insights": report.insights,
            "rejuvenation_count": len(report.rejuvenations),
            "retirement_count": len(report.retirements),
            "cycle_history": report.cycle_history,
            "agents": [
                {
                    "id": a.agent_id,
                    "telomere": round(a.telomere_length, 1),
                    "age": a.age,
                    "state": a.state.value,
                    "rejuvenations": a.rejuvenation_count,
                    "knowledge": round(a.knowledge_store, 2),
                }
                for a in report.agents
            ],
        }

    def export_html(self, path: str, report: Optional[SenescenceReport] = None) -> None:
        """Export interactive HTML dashboard."""
        if report is None:
            report = self.simulate()

        health = report.health
        pop = report.population_stats
        insights_html = "".join(
            f"<li>{html_mod.escape(i)}</li>" for i in report.insights
        )

        # Age pyramid data
        active = [a for a in report.agents if a.state != SenescenceState.RETIRED]
        state_counts = defaultdict(int)
        for a in active:
            state_counts[a.state.value] += 1

        # Cycle chart data
        cycle_labels = [str(c["cycle"]) for c in report.cycle_history[::5]]
        telomere_data = [c["avg_telomere"] for c in report.cycle_history[::5]]
        senescent_data = [c["senescent_count"] for c in report.cycle_history[::5]]

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Swarm Senescence Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f0f23; color: #e0e0e0; padding: 20px; }}
.header {{ text-align: center; padding: 30px; }}
.header h1 {{ font-size: 2em; color: #00d4aa; }}
.header .tier {{ font-size: 1.5em; margin-top: 10px; }}
.score-gauge {{ font-size: 3em; font-weight: bold; color: #00d4aa; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
         gap: 20px; margin: 20px 0; }}
.card {{ background: #1a1a3e; border-radius: 12px; padding: 20px;
         border: 1px solid #333366; }}
.card h2 {{ color: #00d4aa; margin-bottom: 15px; font-size: 1.1em; }}
.metric {{ display: flex; justify-content: space-between; padding: 8px 0;
           border-bottom: 1px solid #2a2a4e; }}
.metric:last-child {{ border-bottom: none; }}
.bar {{ height: 8px; background: #2a2a4e; border-radius: 4px; margin-top: 4px; }}
.bar-fill {{ height: 100%; border-radius: 4px; background: #00d4aa; }}
ul {{ list-style: none; }}
ul li {{ padding: 8px 0; border-bottom: 1px solid #2a2a4e; }}
ul li:last-child {{ border-bottom: none; }}
.state-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
               font-size: 0.8em; margin-left: 5px; }}
.state-young {{ background: #004d40; color: #00e676; }}
.state-mature {{ background: #1a237e; color: #448aff; }}
.state-pre_senescent {{ background: #e65100; color: #ffab40; }}
.state-senescent {{ background: #b71c1c; color: #ff5252; }}
.state-retired {{ background: #37474f; color: #90a4ae; }}
</style>
</head>
<body>
<div class="header">
    <h1>🧬 Swarm Senescence Report</h1>
    <div class="score-gauge">{health.score}</div>
    <div class="tier">{health.tier}</div>
</div>

<div class="grid">
    <div class="card">
        <h2>📊 Health Dimensions</h2>
        <div class="metric"><span>Telomere Reserve</span><span>{health.telomere_reserve:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.telomere_reserve*100:.0f}%"></div></div>
        <div class="metric"><span>SASP Burden</span><span>{health.sasp_burden:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.sasp_burden*100:.0f}%"></div></div>
        <div class="metric"><span>Rejuvenation Rate</span><span>{health.rejuvenation_rate:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.rejuvenation_rate*100:.0f}%"></div></div>
        <div class="metric"><span>Retirement Efficiency</span><span>{health.retirement_efficiency:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.retirement_efficiency*100:.0f}%"></div></div>
        <div class="metric"><span>Age Diversity</span><span>{health.age_diversity:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.age_diversity*100:.0f}%"></div></div>
        <div class="metric"><span>Sustainability</span><span>{health.sustainability:.1%}</span></div>
        <div class="bar"><div class="bar-fill" style="width:{health.sustainability*100:.0f}%"></div></div>
    </div>

    <div class="card">
        <h2>👥 Population</h2>
        <div class="metric"><span>Total Created</span><span>{pop.total_agents}</span></div>
        <div class="metric"><span>Active</span><span>{pop.active_agents}</span></div>
        <div class="metric"><span>Retired</span><span>{pop.retired_agents}</span></div>
        <div class="metric"><span>Senescent</span><span>{pop.senescent_agents}</span></div>
        <div class="metric"><span>Avg Telomere</span><span>{pop.avg_telomere}</span></div>
        <div class="metric"><span>Avg Age</span><span>{pop.avg_age}</span></div>
        <div class="metric"><span>Sustainability</span><span>{pop.sustainability_ratio:.2f}x</span></div>
    </div>

    <div class="card">
        <h2>🔬 State Distribution</h2>
        {"".join(f'<div class="metric"><span class="state-badge state-{k}">{k}</span><span>{v}</span></div>' for k, v in state_counts.items())}
    </div>

    <div class="card">
        <h2>💡 Insights</h2>
        <ul>{insights_html if insights_html else "<li>No insights generated.</li>"}</ul>
    </div>

    <div class="card">
        <h2>♻️ Rejuvenation Events</h2>
        <div class="metric"><span>Total Attempts</span><span>{len(report.rejuvenations)}</span></div>
        <div class="metric"><span>Successful</span><span>{sum(1 for r in report.rejuvenations if r.success)}</span></div>
        <div class="metric"><span>Failed</span><span>{sum(1 for r in report.rejuvenations if not r.success)}</span></div>
    </div>

    <div class="card">
        <h2>🪦 Retirements</h2>
        <div class="metric"><span>Total</span><span>{len(report.retirements)}</span></div>
        <div class="metric"><span>Graceful (w/ transfer)</span><span>{self._graceful_retirements}</span></div>
        {"".join(f'<div class="metric"><span>{r.reason.value}</span><span>{r.agent_id} @ cycle {r.cycle}</span></div>' for r in report.retirements[-5:])}
    </div>
</div>

</body>
</html>"""

        Path(path).write_text(html_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Swarm Senescence Engine — autonomous agent aging simulation"
    )
    parser.add_argument("--agents", type=int, default=15, help="Number of agents")
    parser.add_argument("--cycles", type=int, default=100, help="Simulation cycles")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(SCENARIOS.keys()),
                        help="Run a preset scenario")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="HTML output path")
    parser.add_argument("--json", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    if args.scenario:
        engine = SwarmSenescenceEngine.from_scenario(args.scenario, seed=args.seed)
        cycles = engine.scenario_cycles(args.scenario)
    else:
        engine = SwarmSenescenceEngine(num_agents=args.agents, seed=args.seed)
        cycles = args.cycles

    print(f"🧬 Swarm Senescence Engine")
    print(f"   Agents: {engine.num_agents} | Cycles: {cycles}")
    if args.scenario:
        print(f"   Scenario: {args.scenario} — {SCENARIOS[args.scenario]['description']}")
    print()

    report = engine.simulate(cycles=cycles)

    print(f"═══ Health Score: {report.health.score}/100 ({report.health.tier}) ═══")
    print(f"  Telomere Reserve:      {report.health.telomere_reserve:.1%}")
    print(f"  SASP Burden:           {report.health.sasp_burden:.1%}")
    print(f"  Rejuvenation Rate:     {report.health.rejuvenation_rate:.1%}")
    print(f"  Retirement Efficiency: {report.health.retirement_efficiency:.1%}")
    print(f"  Age Diversity:         {report.health.age_diversity:.1%}")
    print(f"  Sustainability:        {report.health.sustainability:.1%}")
    print()
    print(f"═══ Population ═══")
    print(f"  Active: {report.population_stats.active_agents} | "
          f"Retired: {report.population_stats.retired_agents} | "
          f"Senescent: {report.population_stats.senescent_agents}")
    print(f"  Avg Telomere: {report.population_stats.avg_telomere} | "
          f"Avg Age: {report.population_stats.avg_age}")
    print(f"  Rejuvenations: {len(report.rejuvenations)} | "
          f"Retirements: {len(report.retirements)}")
    print()

    if report.insights:
        print("═══ Insights ═══")
        for insight in report.insights:
            print(f"  {insight}")
        print()

    if args.out:
        engine.export_html(args.out, report)
        print(f"📄 HTML report: {args.out}")

    if args.json:
        data = engine.export_json(report)
        Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"📄 JSON report: {args.json}")


if __name__ == "__main__":
    main()
