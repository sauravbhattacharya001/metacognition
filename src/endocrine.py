"""Swarm Endocrine Engine — autonomous hormonal signaling for global swarm state regulation.

Biologically-inspired by the mammalian endocrine system.  Unlike stigmergy
(local pheromone traces) or quorum sensing (density-dependent), the endocrine
system provides **global broadcast signaling** via hormones in a shared
bloodstream with receptor-mediated responses, feedback loops, and cascading
hormone chains.

Capabilities:

- **Gland Controller** — agents have glands that produce hormones based on
  events (stress triggers cortisol, success triggers dopamine, etc.).
  Production rates adapt over time via habituation.
- **Bloodstream Simulator** — shared hormone pool where hormones are released,
  diffuse globally, and decay based on configurable half-lives.  Tracks
  concentration history for time-series analysis.
- **Receptor Binding Engine** — agents have typed receptors that bind hormones
  following Hill equation kinetics (cooperative binding).  Receptor sensitivity
  adapts via upregulation/downregulation.
- **Feedback Loop Regulator** — negative feedback (high cortisol suppresses
  further cortisol production) and positive feedback (oxytocin triggers more
  oxytocin during collaboration bursts).
- **Hormonal Cascade Engine** — hormone-to-hormone triggering chains.
  E.g. stress → cortisol → adrenaline cascade.  Tracks cascade depth and
  amplification factors.
- **Endocrine Health Scorer** — composite 0-100 score from hormone balance,
  receptor health, feedback responsiveness, and cascade stability.
- **Insight Generator** — autonomous pattern detection: chronic stress,
  reward deficiency, bonding gaps, energy dysregulation, growth stalls.

Usage (Python API)::

    from src.endocrine import SwarmEndocrineEngine, EventType

    engine = SwarmEndocrineEngine(num_agents=6)
    engine.inject_event("agent-0", EventType.TASK_FAILURE, magnitude=0.8)
    engine.inject_event("agent-1", EventType.TASK_SUCCESS, magnitude=1.0)
    engine.tick(dt=1.0)
    report = engine.get_report()
    print(report.health.score)       # 0-100
    print(report.health.tier)        # OPTIMAL / BALANCED / STRESSED / ...
    engine.export_html("endocrine.html")

CLI::

    python -m src.endocrine                          # demo with defaults
    python -m src.endocrine --agents 10              # more agents
    python -m src.endocrine --ticks 200              # longer simulation
    python -m src.endocrine --scenario stress        # stress scenario
    python -m src.endocrine --out report.html --json endocrine.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HormoneType(str, Enum):
    """Types of hormones in the swarm endocrine system."""
    CORTISOL = "cortisol"
    ADRENALINE = "adrenaline"
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    OXYTOCIN = "oxytocin"
    INSULIN = "insulin"
    GROWTH_HORMONE = "growth_hormone"


class EventType(str, Enum):
    """External events that trigger hormone production."""
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    HIGH_LOAD = "high_load"
    COLLABORATION = "collaboration"
    LEARNING = "learning"
    IDLE = "idle"
    RESOURCE_SHORTAGE = "resource_shortage"


class FeedbackType(str, Enum):
    """Feedback loop types."""
    NEGATIVE = "negative"
    POSITIVE = "positive"


class HealthTier(str, Enum):
    """Health classification tiers."""
    OPTIMAL = "optimal"
    BALANCED = "balanced"
    STRESSED = "stressed"
    DYSREGULATED = "dysregulated"
    CRITICAL = "critical"


class InsightSeverity(str, Enum):
    """Severity of generated insights."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORMONE_PROFILES: Dict[HormoneType, Dict[str, float]] = {
    HormoneType.CORTISOL:       {"half_life": 8.0, "potency": 1.0, "decay_rate": 0.087, "baseline": 0.3},
    HormoneType.ADRENALINE:     {"half_life": 3.0, "potency": 1.5, "decay_rate": 0.231, "baseline": 0.1},
    HormoneType.DOPAMINE:       {"half_life": 5.0, "potency": 1.0, "decay_rate": 0.139, "baseline": 0.4},
    HormoneType.SEROTONIN:      {"half_life": 10.0, "potency": 0.8, "decay_rate": 0.069, "baseline": 0.5},
    HormoneType.OXYTOCIN:       {"half_life": 6.0, "potency": 0.9, "decay_rate": 0.116, "baseline": 0.2},
    HormoneType.INSULIN:        {"half_life": 4.0, "potency": 1.2, "decay_rate": 0.173, "baseline": 0.3},
    HormoneType.GROWTH_HORMONE: {"half_life": 7.0, "potency": 0.7, "decay_rate": 0.099, "baseline": 0.2},
}

# Which events trigger which hormones and at what base rate
EVENT_HORMONE_MAP: Dict[EventType, List[Tuple[HormoneType, float]]] = {
    EventType.TASK_SUCCESS:     [(HormoneType.DOPAMINE, 1.0), (HormoneType.SEROTONIN, 0.5)],
    EventType.TASK_FAILURE:     [(HormoneType.CORTISOL, 1.0), (HormoneType.ADRENALINE, 0.6)],
    EventType.HIGH_LOAD:        [(HormoneType.CORTISOL, 0.7), (HormoneType.ADRENALINE, 1.0), (HormoneType.INSULIN, 0.4)],
    EventType.COLLABORATION:    [(HormoneType.OXYTOCIN, 1.0), (HormoneType.DOPAMINE, 0.3)],
    EventType.LEARNING:         [(HormoneType.GROWTH_HORMONE, 1.0), (HormoneType.DOPAMINE, 0.4)],
    EventType.IDLE:             [(HormoneType.SEROTONIN, 0.3)],
    EventType.RESOURCE_SHORTAGE: [(HormoneType.CORTISOL, 0.8), (HormoneType.INSULIN, 1.0)],
}

# Cascade rules: trigger_hormone -> [(triggered_hormone, amplification, threshold)]
CASCADE_RULES: Dict[HormoneType, List[Tuple[HormoneType, float, float]]] = {
    HormoneType.CORTISOL:   [(HormoneType.ADRENALINE, 0.5, 0.6)],
    HormoneType.ADRENALINE: [(HormoneType.CORTISOL, 0.3, 0.8)],
    HormoneType.DOPAMINE:   [(HormoneType.SEROTONIN, 0.2, 0.7)],
    HormoneType.OXYTOCIN:   [(HormoneType.DOPAMINE, 0.15, 0.5)],
}

# Feedback rules: (hormone, feedback_type, strength_factor)
FEEDBACK_RULES: Dict[HormoneType, List[Tuple[FeedbackType, float, float]]] = {
    HormoneType.CORTISOL:   [(FeedbackType.NEGATIVE, 0.3, 0.7)],   # high cortisol suppresses production
    HormoneType.ADRENALINE: [(FeedbackType.NEGATIVE, 0.4, 0.6)],
    HormoneType.OXYTOCIN:   [(FeedbackType.POSITIVE, 0.2, 0.5)],   # oxytocin begets oxytocin
    HormoneType.DOPAMINE:   [(FeedbackType.NEGATIVE, 0.2, 0.8)],   # habituation
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class HormoneLevel:
    """Current state of a hormone in the bloodstream."""
    hormone_type: HormoneType
    concentration: float = 0.0
    production_rate: float = 0.0
    decay_rate: float = 0.0
    baseline: float = 0.0


@dataclass
class ReceptorState:
    """State of an agent's receptor for a specific hormone."""
    hormone_type: HormoneType
    sensitivity: float = 1.0       # 0.0 to 2.0 (1.0 = normal)
    bound_amount: float = 0.0
    hill_coefficient: float = 2.0  # cooperativity
    kd: float = 0.5               # dissociation constant


@dataclass
class AgentEndocrineState:
    """Full endocrine state of a single agent."""
    agent_id: str
    glands: Dict[str, float] = field(default_factory=dict)      # hormone -> production_rate
    receptors: Dict[str, ReceptorState] = field(default_factory=dict)
    bound_hormones: Dict[str, float] = field(default_factory=dict)
    total_produced: float = 0.0
    total_bound: float = 0.0


@dataclass
class BloodstreamSnapshot:
    """Snapshot of bloodstream at one point in time."""
    tick: int = 0
    concentrations: Dict[str, float] = field(default_factory=dict)


@dataclass
class CascadeEvent:
    """Record of a hormone cascade event."""
    trigger_hormone: str
    triggered_hormone: str
    amplification: float = 0.0
    depth: int = 1
    tick: int = 0


@dataclass
class FeedbackEvent:
    """Record of a feedback loop activation."""
    hormone: str
    feedback_type: str
    magnitude: float = 0.0
    tick: int = 0


@dataclass
class Insight:
    """Autonomous insight generated by the engine."""
    category: str
    message: str
    severity: InsightSeverity = InsightSeverity.INFO
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthScore:
    """Composite health assessment."""
    score: float = 100.0
    tier: HealthTier = HealthTier.OPTIMAL
    components: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class EndocrineReport:
    """Complete report from the endocrine engine."""
    health: HealthScore = field(default_factory=HealthScore)
    bloodstream_history: List[BloodstreamSnapshot] = field(default_factory=list)
    current_concentrations: Dict[str, float] = field(default_factory=dict)
    agent_states: List[AgentEndocrineState] = field(default_factory=list)
    cascades: List[CascadeEvent] = field(default_factory=list)
    feedback_events: List[FeedbackEvent] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)
    tick_count: int = 0
    num_agents: int = 0


# ---------------------------------------------------------------------------
# Engine 1: Gland Controller
# ---------------------------------------------------------------------------

class GlandController:
    """Manages hormone production by agent glands.

    Each agent has glands for each hormone type.  Production is triggered by
    events and modulated by habituation (repeated identical stimuli decrease
    production over time).
    """

    def __init__(self) -> None:
        self.agent_glands: Dict[str, Dict[HormoneType, float]] = {}  # agent -> hormone -> production_rate
        self.habituation: Dict[str, Dict[HormoneType, float]] = {}    # agent -> hormone -> habituation_factor
        self._habituation_rate: float = 0.05
        self._recovery_rate: float = 0.02

    def register_agent(self, agent_id: str) -> None:
        """Register a new agent with default gland configuration."""
        self.agent_glands[agent_id] = {h: 0.0 for h in HormoneType}
        self.habituation[agent_id] = {h: 1.0 for h in HormoneType}

    def produce(self, agent_id: str, event: EventType, magnitude: float = 1.0) -> Dict[HormoneType, float]:
        """Calculate hormone production for an event.

        Returns dict of hormone_type -> amount produced.
        """
        if agent_id not in self.agent_glands:
            self.register_agent(agent_id)

        produced: Dict[HormoneType, float] = {}
        mappings = EVENT_HORMONE_MAP.get(event, [])

        for hormone, base_rate in mappings:
            hab = self.habituation[agent_id].get(hormone, 1.0)
            amount = base_rate * magnitude * hab
            produced[hormone] = amount
            self.agent_glands[agent_id][hormone] = amount

            # Habituation: decrease sensitivity to repeated stimuli
            self.habituation[agent_id][hormone] = max(
                0.1, hab - self._habituation_rate
            )

        return produced

    def apply_feedback_suppression(self, agent_id: str, hormone: HormoneType, factor: float) -> None:
        """Apply feedback-based suppression to gland production."""
        if agent_id in self.agent_glands:
            current = self.agent_glands[agent_id].get(hormone, 0.0)
            self.agent_glands[agent_id][hormone] = current * max(0.0, 1.0 - factor)

    def recover_habituation(self, dt: float = 1.0) -> None:
        """Slowly recover habituation levels toward 1.0."""
        for agent_id in self.habituation:
            for hormone in HormoneType:
                current = self.habituation[agent_id].get(hormone, 1.0)
                self.habituation[agent_id][hormone] = min(
                    1.0, current + self._recovery_rate * dt
                )

    def get_agent_production(self, agent_id: str) -> Dict[HormoneType, float]:
        """Get current production rates for an agent."""
        return dict(self.agent_glands.get(agent_id, {}))


# ---------------------------------------------------------------------------
# Engine 2: Bloodstream Simulator
# ---------------------------------------------------------------------------

class BloodstreamSimulator:
    """Shared hormone pool with diffusion and decay dynamics.

    Hormones released by agent glands enter the shared bloodstream,
    decay according to half-life kinetics, and are tracked over time.
    """

    def __init__(self) -> None:
        self.concentrations: Dict[HormoneType, float] = {h: HORMONE_PROFILES[h]["baseline"] for h in HormoneType}
        self.history: List[BloodstreamSnapshot] = []
        self._tick: int = 0

    def release(self, hormone: HormoneType, amount: float) -> None:
        """Release hormone into the bloodstream."""
        self.concentrations[hormone] = self.concentrations.get(hormone, 0.0) + amount

    def decay(self, dt: float = 1.0) -> None:
        """Apply exponential decay based on hormone half-lives."""
        for hormone in HormoneType:
            rate = HORMONE_PROFILES[hormone]["decay_rate"]
            self.concentrations[hormone] *= math.exp(-rate * dt)

    def tick(self, dt: float = 1.0) -> BloodstreamSnapshot:
        """Advance one time step: decay, then snapshot."""
        self.decay(dt)
        self._tick += 1
        snapshot = BloodstreamSnapshot(
            tick=self._tick,
            concentrations={h.value: c for h, c in self.concentrations.items()},
        )
        self.history.append(snapshot)
        return snapshot

    def get_concentration(self, hormone: HormoneType) -> float:
        """Get current concentration of a hormone."""
        return self.concentrations.get(hormone, 0.0)

    def get_all_concentrations(self) -> Dict[str, float]:
        """Get all current concentrations as string-keyed dict."""
        return {h.value: c for h, c in self.concentrations.items()}

    def deviation_from_baseline(self) -> Dict[HormoneType, float]:
        """Calculate deviation of each hormone from baseline."""
        devs: Dict[HormoneType, float] = {}
        for h in HormoneType:
            baseline = HORMONE_PROFILES[h]["baseline"]
            devs[h] = abs(self.concentrations[h] - baseline)
        return devs


# ---------------------------------------------------------------------------
# Engine 3: Receptor Binding Engine
# ---------------------------------------------------------------------------

class ReceptorBindingEngine:
    """Models receptor-hormone binding using Hill equation kinetics.

    Each agent has receptors for each hormone type.  Binding affinity
    follows the Hill equation:  bound = (C^n) / (Kd^n + C^n)
    Receptor sensitivity adapts via upregulation (low exposure) and
    downregulation (high exposure).
    """

    def __init__(self) -> None:
        self.agent_receptors: Dict[str, Dict[HormoneType, ReceptorState]] = {}
        self._adaptation_rate: float = 0.03

    def register_agent(self, agent_id: str) -> None:
        """Register agent with default receptors."""
        self.agent_receptors[agent_id] = {
            h: ReceptorState(hormone_type=h) for h in HormoneType
        }

    def bind(self, agent_id: str, concentrations: Dict[HormoneType, float]) -> Dict[HormoneType, float]:
        """Calculate binding for all receptors given bloodstream concentrations.

        Returns dict of hormone -> bound_amount.
        """
        if agent_id not in self.agent_receptors:
            self.register_agent(agent_id)

        bound: Dict[HormoneType, float] = {}
        for hormone in HormoneType:
            receptor = self.agent_receptors[agent_id][hormone]
            conc = concentrations.get(hormone, 0.0)
            # Hill equation
            n = receptor.hill_coefficient
            kd = receptor.kd / max(receptor.sensitivity, 0.01)  # sensitivity modulates Kd
            if conc <= 0:
                binding = 0.0
            else:
                binding = (conc ** n) / (kd ** n + conc ** n)
            receptor.bound_amount = binding
            bound[hormone] = binding
        return bound

    def adapt_receptors(self, agent_id: str, concentrations: Dict[HormoneType, float]) -> None:
        """Adapt receptor sensitivity based on exposure.

        High concentration -> downregulation (decreased sensitivity).
        Low concentration -> upregulation (increased sensitivity).
        """
        if agent_id not in self.agent_receptors:
            return

        for hormone in HormoneType:
            receptor = self.agent_receptors[agent_id][hormone]
            conc = concentrations.get(hormone, 0.0)
            baseline = HORMONE_PROFILES[hormone]["baseline"]

            if conc > baseline * 1.5:
                # Downregulate
                receptor.sensitivity = max(0.1, receptor.sensitivity - self._adaptation_rate)
            elif conc < baseline * 0.5:
                # Upregulate
                receptor.sensitivity = min(2.0, receptor.sensitivity + self._adaptation_rate)
            else:
                # Drift back toward normal
                if receptor.sensitivity > 1.0:
                    receptor.sensitivity -= self._adaptation_rate * 0.5
                elif receptor.sensitivity < 1.0:
                    receptor.sensitivity += self._adaptation_rate * 0.5

    def get_receptor_states(self, agent_id: str) -> Dict[HormoneType, ReceptorState]:
        """Get all receptor states for an agent."""
        return dict(self.agent_receptors.get(agent_id, {}))


# ---------------------------------------------------------------------------
# Engine 4: Feedback Loop Regulator
# ---------------------------------------------------------------------------

class FeedbackLoopRegulator:
    """Manages negative and positive feedback loops.

    Negative feedback: high concentration suppresses further production.
    Positive feedback: concentration above threshold amplifies production.
    """

    def __init__(self) -> None:
        self.events: List[FeedbackEvent] = []
        self._tick: int = 0

    def evaluate(self, concentrations: Dict[HormoneType, float], tick: int = 0) -> List[FeedbackEvent]:
        """Evaluate all feedback rules against current concentrations.

        Returns list of feedback events triggered.
        """
        self._tick = tick
        triggered: List[FeedbackEvent] = []

        for hormone, rules in FEEDBACK_RULES.items():
            conc = concentrations.get(hormone, 0.0)
            for fb_type, strength, threshold in rules:
                if conc >= threshold:
                    magnitude = strength * (conc - threshold)
                    evt = FeedbackEvent(
                        hormone=hormone.value,
                        feedback_type=fb_type.value,
                        magnitude=magnitude,
                        tick=tick,
                    )
                    triggered.append(evt)
                    self.events.append(evt)

        return triggered


# ---------------------------------------------------------------------------
# Engine 5: Hormonal Cascade Engine
# ---------------------------------------------------------------------------

class HormonalCascadeEngine:
    """Detects and triggers hormone-to-hormone cascades.

    When a hormone exceeds its cascade threshold, it triggers production
    of downstream hormones with amplification factors.
    """

    MAX_CASCADE_DEPTH = 3

    def __init__(self) -> None:
        self.events: List[CascadeEvent] = []
        self._tick: int = 0

    def evaluate(
        self, concentrations: Dict[HormoneType, float], tick: int = 0
    ) -> List[CascadeEvent]:
        """Evaluate cascade rules.  Returns list of cascade events with amounts to inject."""
        self._tick = tick
        triggered: List[CascadeEvent] = []
        self._cascade_recursive(concentrations, triggered, depth=1, tick=tick)
        return triggered

    def _cascade_recursive(
        self,
        concentrations: Dict[HormoneType, float],
        triggered: List[CascadeEvent],
        depth: int,
        tick: int,
    ) -> None:
        if depth > self.MAX_CASCADE_DEPTH:
            return

        new_releases: Dict[HormoneType, float] = {}

        for trigger_hormone, rules in CASCADE_RULES.items():
            conc = concentrations.get(trigger_hormone, 0.0)
            for target_hormone, amplification, threshold in rules:
                if conc >= threshold:
                    amount = amplification * (conc - threshold)
                    evt = CascadeEvent(
                        trigger_hormone=trigger_hormone.value,
                        triggered_hormone=target_hormone.value,
                        amplification=amplification,
                        depth=depth,
                        tick=tick,
                    )
                    triggered.append(evt)
                    self.events.append(evt)
                    new_releases[target_hormone] = new_releases.get(target_hormone, 0.0) + amount

        if new_releases:
            # Update concentrations for next cascade depth
            updated = dict(concentrations)
            for h, amount in new_releases.items():
                updated[h] = updated.get(h, 0.0) + amount
            self._cascade_recursive(updated, triggered, depth + 1, tick)


# ---------------------------------------------------------------------------
# Engine 6: Endocrine Health Scorer
# ---------------------------------------------------------------------------

class EndocrineHealthScorer:
    """Computes a composite 0-100 health score.

    Components:
    - hormone_balance: how close concentrations are to baselines (25%)
    - receptor_health: average receptor sensitivity drift (25%)
    - feedback_responsiveness: whether feedback loops are firing appropriately (25%)
    - cascade_stability: whether cascades are controlled, not runaway (25%)
    """

    @staticmethod
    def score(
        concentrations: Dict[HormoneType, float],
        agent_receptors: Dict[str, Dict[HormoneType, ReceptorState]],
        feedback_events: List[FeedbackEvent],
        cascade_events: List[CascadeEvent],
        tick_count: int,
    ) -> HealthScore:
        """Calculate composite health score."""
        # 1. Hormone balance (0-100)
        deviations = []
        for h in HormoneType:
            baseline = HORMONE_PROFILES[h]["baseline"]
            conc = concentrations.get(h, 0.0)
            if baseline > 0:
                rel_dev = abs(conc - baseline) / baseline
            else:
                rel_dev = abs(conc)
            deviations.append(min(rel_dev, 3.0))  # cap at 3x
        avg_dev = statistics.mean(deviations) if deviations else 0.0
        hormone_balance = max(0.0, 100.0 * (1.0 - avg_dev / 3.0))

        # 2. Receptor health (0-100)
        sensitivity_drifts = []
        for agent_id, receptors in agent_receptors.items():
            for h, receptor in receptors.items():
                drift = abs(receptor.sensitivity - 1.0)
                sensitivity_drifts.append(drift)
        avg_drift = statistics.mean(sensitivity_drifts) if sensitivity_drifts else 0.0
        receptor_health = max(0.0, 100.0 * (1.0 - avg_drift))

        # 3. Feedback responsiveness (0-100)
        # Good: some feedback events. Bad: zero (no regulation) or too many (dysregulation)
        if tick_count > 0:
            fb_rate = len(feedback_events) / tick_count
            if fb_rate < 0.1:
                feedback_score = 60.0  # under-regulated
            elif fb_rate > 2.0:
                feedback_score = max(0.0, 100.0 - (fb_rate - 2.0) * 20)  # over-firing
            else:
                feedback_score = 100.0
        else:
            feedback_score = 80.0

        # 4. Cascade stability (0-100)
        deep_cascades = sum(1 for c in cascade_events if c.depth >= 3)
        total_cascades = len(cascade_events)
        if total_cascades > 0:
            deep_ratio = deep_cascades / total_cascades
            cascade_score = max(0.0, 100.0 * (1.0 - deep_ratio))
        else:
            cascade_score = 100.0
        if tick_count > 0 and total_cascades / max(tick_count, 1) > 3.0:
            cascade_score = max(0.0, cascade_score - 30.0)

        # Composite
        score = (
            hormone_balance * 0.25
            + receptor_health * 0.25
            + feedback_score * 0.25
            + cascade_score * 0.25
        )
        score = max(0.0, min(100.0, score))

        # Tier
        if score >= 80:
            tier = HealthTier.OPTIMAL
        elif score >= 60:
            tier = HealthTier.BALANCED
        elif score >= 40:
            tier = HealthTier.STRESSED
        elif score >= 20:
            tier = HealthTier.DYSREGULATED
        else:
            tier = HealthTier.CRITICAL

        # Recommendations
        recs: List[str] = []
        if hormone_balance < 50:
            recs.append("Hormone levels deviate significantly from baselines — consider reducing stressor events")
        if receptor_health < 50:
            recs.append("Receptor sensitivity drift detected — allow recovery time between stimuli")
        if feedback_score < 50:
            recs.append("Feedback loops are over-firing — system may be in dysregulated oscillation")
        if cascade_score < 50:
            recs.append("Deep hormone cascades detected — consider circuit breaker mechanisms")

        return HealthScore(
            score=round(score, 1),
            tier=tier,
            components={
                "hormone_balance": round(hormone_balance, 1),
                "receptor_health": round(receptor_health, 1),
                "feedback_responsiveness": round(feedback_score, 1),
                "cascade_stability": round(cascade_score, 1),
            },
            recommendations=recs,
        )


# ---------------------------------------------------------------------------
# Engine 7: Insight Generator
# ---------------------------------------------------------------------------

# Declarative insight rules for threshold-based hormone checks.
# Each rule: (category, hormone, direction, threshold_multiplier, severity, escalation_multiplier, message_template)
# direction: "above" triggers when conc > baseline * threshold, "below" when conc < baseline * threshold.
# escalation_multiplier: if set and conc crosses baseline * escalation, severity upgrades to CRITICAL.
_INSIGHT_RULES: List[Tuple[str, HormoneType, str, float, InsightSeverity, Optional[float], str]] = [
    ("chronic_stress", HormoneType.CORTISOL, "above", 2.5, InsightSeverity.WARNING, 4.0,
     "{name} at {conc:.2f} — {ratio:.1f}x baseline. Chronic stress detected."),
    ("reward_deficiency", HormoneType.DOPAMINE, "below", 0.3, InsightSeverity.WARNING, None,
     "{name} at {conc:.2f} — well below baseline {baseline:.2f}. Reward signaling impaired."),
    ("bonding_gap", HormoneType.OXYTOCIN, "below", 0.3, InsightSeverity.WARNING, None,
     "{name} at {conc:.2f} — collaboration/bonding signals are weak."),
    ("energy_dysregulation", HormoneType.INSULIN, "above", 3.0, InsightSeverity.WARNING, None,
     "{name} at {conc:.2f} — resource regulation may be overactive."),
    ("growth_stall", HormoneType.GROWTH_HORMONE, "below", 0.3, InsightSeverity.INFO, None,
     "{name} at {conc:.2f} — learning/development signals may be insufficient."),
    ("adrenaline_surge", HormoneType.ADRENALINE, "above", 5.0, InsightSeverity.CRITICAL, None,
     "{name} at {conc:.2f} — sustained fight-or-flight state."),
]


class InsightGenerator:
    """Detects patterns and generates autonomous insights.

    Hormone-threshold insights are driven by ``_INSIGHT_RULES`` — a
    declarative table that replaces per-hormone if/else blocks.  To add a
    new hormone-level insight, append a tuple to the table instead of
    writing another manual block.
    """

    @staticmethod
    def _evaluate_hormone_rules(
        concentrations: Dict[HormoneType, float],
    ) -> List[Insight]:
        """Evaluate all declarative hormone threshold rules."""
        insights: List[Insight] = []
        for category, hormone, direction, threshold_mul, severity, escalation_mul, msg_tpl in _INSIGHT_RULES:
            conc = concentrations.get(hormone, 0.0)
            baseline = HORMONE_PROFILES[hormone]["baseline"]
            triggered = (
                conc > baseline * threshold_mul if direction == "above"
                else conc < baseline * threshold_mul
            )
            if not triggered:
                continue
            # Escalate severity if an escalation multiplier is defined and crossed
            final_severity = severity
            if escalation_mul is not None and direction == "above" and conc > baseline * escalation_mul:
                final_severity = InsightSeverity.CRITICAL
            ratio = conc / baseline if baseline > 0 else 0.0
            name = hormone.value.replace("_", " ").title()
            insights.append(Insight(
                category=category,
                message=msg_tpl.format(name=name, conc=conc, baseline=baseline, ratio=ratio),
                severity=final_severity,
                details={hormone.value: conc, "baseline": baseline},
            ))
        return insights

    @staticmethod
    def generate(
        concentrations: Dict[HormoneType, float],
        history: List[BloodstreamSnapshot],
        feedback_events: List[FeedbackEvent],
        cascade_events: List[CascadeEvent],
        health: HealthScore,
    ) -> List[Insight]:
        """Generate insights from current endocrine state."""
        # 1-6. Hormone threshold insights (data-driven)
        insights = InsightGenerator._evaluate_hormone_rules(concentrations)

        # 7. Cascade runaway detection
        deep = [c for c in cascade_events if c.depth >= 3]
        if len(deep) > 5:
            insights.append(Insight(
                category="cascade_runaway",
                message=f"{len(deep)} deep cascades detected — hormone chains may be self-amplifying.",
                severity=InsightSeverity.CRITICAL,
                details={"deep_cascades": len(deep)},
            ))

        # 8. Overall health insight
        if health.score < 30:
            insights.append(Insight(
                category="system_health",
                message=f"Endocrine health critically low at {health.score:.0f}/100. Immediate intervention recommended.",
                severity=InsightSeverity.CRITICAL,
                details={"score": health.score, "tier": health.tier.value},
            ))

        return insights


# ---------------------------------------------------------------------------
# Main Orchestrator: SwarmEndocrineEngine
# ---------------------------------------------------------------------------

class SwarmEndocrineEngine:
    """Autonomous hormonal signaling engine for swarm state regulation.

    Orchestrates all seven sub-engines to provide a complete endocrine
    simulation for a multi-agent swarm.
    """

    def __init__(self, num_agents: int = 5, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)

        self.num_agents = max(1, num_agents)
        self.agent_ids = [f"agent-{i}" for i in range(self.num_agents)]
        self.tick_count = 0

        # Engines
        self.gland_controller = GlandController()
        self.bloodstream = BloodstreamSimulator()
        self.receptor_engine = ReceptorBindingEngine()
        self.feedback_regulator = FeedbackLoopRegulator()
        self.cascade_engine = HormonalCascadeEngine()

        # Register agents
        for agent_id in self.agent_ids:
            self.gland_controller.register_agent(agent_id)
            self.receptor_engine.register_agent(agent_id)

        # Event log
        self._pending_productions: Dict[HormoneType, float] = defaultdict(float)

    def inject_event(self, agent_id: str, event: EventType, magnitude: float = 1.0) -> Dict[HormoneType, float]:
        """Inject an external event that triggers hormone production.

        Returns the hormones produced.
        """
        produced = self.gland_controller.produce(agent_id, event, magnitude)
        for hormone, amount in produced.items():
            self._pending_productions[hormone] += amount
        return produced

    def tick(self, dt: float = 1.0) -> BloodstreamSnapshot:
        """Advance the simulation by one time step.

        1. Release pending productions into bloodstream
        2. Evaluate cascades
        3. Evaluate feedback loops
        4. Apply receptor binding for all agents
        5. Decay and snapshot bloodstream
        6. Recover habituation
        """
        self.tick_count += 1

        # 1. Release pending productions
        for hormone, amount in self._pending_productions.items():
            self.bloodstream.release(hormone, amount)
        self._pending_productions.clear()

        # 2. Evaluate cascades
        cascade_events = self.cascade_engine.evaluate(
            self.bloodstream.concentrations, tick=self.tick_count
        )
        for evt in cascade_events:
            triggered_h = HormoneType(evt.triggered_hormone)
            amount = evt.amplification * max(
                0, self.bloodstream.get_concentration(HormoneType(evt.trigger_hormone))
                - 0.5
            )
            self.bloodstream.release(triggered_h, max(0, amount * 0.1))

        # 3. Evaluate feedback loops
        fb_events = self.feedback_regulator.evaluate(
            self.bloodstream.concentrations, tick=self.tick_count
        )
        for evt in fb_events:
            hormone = HormoneType(evt.hormone)
            if evt.feedback_type == FeedbackType.NEGATIVE.value:
                # Suppress production for all agents
                for agent_id in self.agent_ids:
                    self.gland_controller.apply_feedback_suppression(
                        agent_id, hormone, evt.magnitude * 0.1
                    )
            elif evt.feedback_type == FeedbackType.POSITIVE.value:
                # Amplify: small release
                self.bloodstream.release(hormone, evt.magnitude * 0.05)

        # 4. Receptor binding
        for agent_id in self.agent_ids:
            self.receptor_engine.bind(agent_id, self.bloodstream.concentrations)
            self.receptor_engine.adapt_receptors(agent_id, self.bloodstream.concentrations)

        # 5. Decay and snapshot
        snapshot = self.bloodstream.tick(dt)

        # 6. Recover habituation
        self.gland_controller.recover_habituation(dt)

        return snapshot

    def get_report(self) -> EndocrineReport:
        """Generate a comprehensive endocrine report."""
        concentrations = self.bloodstream.concentrations

        # Collect agent states
        agent_states = []
        for agent_id in self.agent_ids:
            glands = self.gland_controller.get_agent_production(agent_id)
            receptors = self.receptor_engine.get_receptor_states(agent_id)
            bound = {h.value: r.bound_amount for h, r in receptors.items()}
            total_produced = sum(glands.values())
            total_bound = sum(bound.values())
            agent_states.append(AgentEndocrineState(
                agent_id=agent_id,
                glands={h.value: v for h, v in glands.items()},
                receptors={h.value: r for h, r in receptors.items()},
                bound_hormones=bound,
                total_produced=total_produced,
                total_bound=total_bound,
            ))

        # Health score
        health = EndocrineHealthScorer.score(
            concentrations,
            self.receptor_engine.agent_receptors,
            self.feedback_regulator.events,
            self.cascade_engine.events,
            self.tick_count,
        )

        # Insights
        insights = InsightGenerator.generate(
            concentrations,
            self.bloodstream.history,
            self.feedback_regulator.events,
            self.cascade_engine.events,
            health,
        )

        return EndocrineReport(
            health=health,
            bloodstream_history=self.bloodstream.history,
            current_concentrations=self.bloodstream.get_all_concentrations(),
            agent_states=agent_states,
            cascades=self.cascade_engine.events,
            feedback_events=self.feedback_regulator.events,
            insights=insights,
            tick_count=self.tick_count,
            num_agents=self.num_agents,
        )

    # -- Export ---------------------------------------------------------------

    def export_json(self, path: str) -> str:
        """Export report as JSON."""
        report = self.get_report()

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return str(obj)

        data = {
            "tick_count": report.tick_count,
            "num_agents": report.num_agents,
            "health": {
                "score": report.health.score,
                "tier": report.health.tier.value,
                "components": report.health.components,
                "recommendations": report.health.recommendations,
            },
            "current_concentrations": report.current_concentrations,
            "insights": [
                {"category": i.category, "message": i.message, "severity": i.severity.value}
                for i in report.insights
            ],
            "cascades_count": len(report.cascades),
            "feedback_events_count": len(report.feedback_events),
        }

        json_str = json.dumps(data, indent=2, default=_serialize)
        Path(path).write_text(json_str, encoding="utf-8")
        return json_str

    def export_html(self, path: str) -> None:
        """Export interactive HTML dashboard."""
        report = self.get_report()
        h = report.health
        conc = report.current_concentrations

        # Hormone bars
        bars_html = ""
        for hormone in HormoneType:
            val = conc.get(hormone.value, 0.0)
            baseline = HORMONE_PROFILES[hormone]["baseline"]
            pct = min(100, val / max(baseline * 3, 0.01) * 100)
            color = "#4CAF50" if abs(val - baseline) / max(baseline, 0.01) < 1.0 else "#FF9800" if abs(val - baseline) / max(baseline, 0.01) < 2.0 else "#f44336"
            bars_html += f"""
            <div style="margin:6px 0;">
                <div style="display:flex;justify-content:space-between;font-size:13px;">
                    <span>{hormone.value.replace('_',' ').title()}</span>
                    <span>{val:.3f} (baseline: {baseline})</span>
                </div>
                <div style="background:#333;border-radius:4px;height:18px;overflow:hidden;">
                    <div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:4px;transition:width 0.3s;"></div>
                </div>
            </div>"""

        # Timeline data (last 50 snapshots)
        recent = report.bloodstream_history[-50:]
        timeline_labels = [str(s.tick) for s in recent]
        timeline_datasets = ""
        colors = ["#f44336", "#FF9800", "#4CAF50", "#2196F3", "#E91E63", "#9C27B0", "#00BCD4"]
        for i, hormone in enumerate(HormoneType):
            vals = [s.concentrations.get(hormone.value, 0.0) for s in recent]
            vals_str = ",".join(f"{v:.4f}" for v in vals)
            timeline_datasets += f'{{label:"{hormone.value}",data:[{vals_str}],borderColor:"{colors[i]}",fill:false,tension:0.3}},'

        # Insights
        insights_html = ""
        for ins in report.insights:
            icon = "🔴" if ins.severity == InsightSeverity.CRITICAL else "🟡" if ins.severity == InsightSeverity.WARNING else "🔵"
            insights_html += f'<div style="padding:8px;margin:4px 0;background:#1e1e1e;border-radius:6px;border-left:3px solid {"#f44336" if ins.severity == InsightSeverity.CRITICAL else "#FF9800" if ins.severity == InsightSeverity.WARNING else "#2196F3"}">{icon} <strong>{ins.category}</strong>: {html_mod.escape(ins.message)}</div>'

        if not insights_html:
            insights_html = '<div style="padding:8px;color:#888;">No insights — system operating normally.</div>'

        # Agent receptor heatmap (text-based)
        receptor_html = '<table style="width:100%;border-collapse:collapse;font-size:12px;"><tr><th style="text-align:left;padding:4px;">Agent</th>'
        for hormone in HormoneType:
            receptor_html += f'<th style="padding:4px;">{hormone.value[:4]}</th>'
        receptor_html += "</tr>"
        for state in report.agent_states[:10]:
            receptor_html += f'<tr><td style="padding:4px;">{state.agent_id}</td>'
            for hormone in HormoneType:
                rs = state.receptors.get(hormone.value)
                if rs and hasattr(rs, 'sensitivity'):
                    sens = rs.sensitivity
                elif isinstance(rs, dict):
                    sens = rs.get("sensitivity", 1.0)
                else:
                    sens = 1.0
                bg = f"rgba(76,175,80,{min(sens/2,1.0):.2f})" if sens >= 1.0 else f"rgba(244,67,54,{min((2-sens)/2,1.0):.2f})"
                receptor_html += f'<td style="padding:4px;text-align:center;background:{bg};border-radius:3px;">{sens:.2f}</td>'
            receptor_html += "</tr>"
        receptor_html += "</table>"

        # Health gauge
        gauge_color = "#4CAF50" if h.score >= 80 else "#FF9800" if h.score >= 60 else "#f44336" if h.score >= 40 else "#b71c1c"
        tier_badge = h.tier.value.upper()

        # Cascade summary
        cascade_html = f"<p>Total cascades: {len(report.cascades)}</p>"
        if report.cascades:
            depth_counts: Dict[int, int] = defaultdict(int)
            for c in report.cascades:
                depth_counts[c.depth] += 1
            for depth in sorted(depth_counts):
                cascade_html += f'<div style="margin:2px 0;">Depth {depth}: {depth_counts[depth]} events</div>'

        # Feedback summary
        fb_html = f"<p>Total feedback events: {len(report.feedback_events)}</p>"
        fb_by_type: Dict[str, int] = defaultdict(int)
        for fe in report.feedback_events:
            fb_by_type[fe.feedback_type] += 1
        for ft, count in fb_by_type.items():
            fb_html += f'<div style="margin:2px 0;">{ft}: {count}</div>'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swarm Endocrine Engine Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#e6edf3;padding:20px}}
h1{{text-align:center;margin-bottom:20px;font-size:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px}}
.card h2{{font-size:16px;margin-bottom:12px;color:#58a6ff}}
.gauge{{text-align:center;padding:20px}}
.gauge .score{{font-size:64px;font-weight:bold;color:{gauge_color}}}
.gauge .tier{{font-size:20px;color:{gauge_color};margin-top:4px}}
.recs li{{margin:4px 0;font-size:13px;color:#f0ad4e}}
</style>
</head>
<body>
<h1>🧬 Swarm Endocrine Engine Dashboard</h1>
<p style="text-align:center;color:#8b949e;margin-bottom:20px;">
{report.num_agents} agents · {report.tick_count} ticks
</p>
<div class="grid">
<div class="card">
<div class="gauge">
<div class="score">{h.score:.0f}</div>
<div class="tier">{tier_badge}</div>
<p style="margin-top:8px;font-size:13px;color:#8b949e;">Endocrine Health Score</p>
</div>
<ul class="recs">{''.join(f'<li>{r}</li>' for r in h.recommendations)}</ul>
</div>
<div class="card"><h2>🧪 Hormone Levels</h2>{bars_html}</div>
<div class="card"><h2>📊 Concentration Timeline</h2><canvas id="timeline" height="200"></canvas></div>
<div class="card"><h2>🔬 Receptor Sensitivity</h2>{receptor_html}</div>
<div class="card"><h2>⚡ Cascades</h2>{cascade_html}</div>
<div class="card"><h2>🔄 Feedback Loops</h2>{fb_html}</div>
<div class="card" style="grid-column:1/-1"><h2>💡 Insights</h2>{insights_html}</div>
</div>
<script>
new Chart(document.getElementById('timeline'),{{
type:'line',
data:{{labels:{json.dumps(timeline_labels)},datasets:[{timeline_datasets}]}},
options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#e6edf3',font:{{size:11}}}}}}}},scales:{{x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},y:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}}}}}}
}});
</script>
</body>
</html>"""

        Path(path).write_text(html_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, List[Tuple[str, EventType, float]]] = {
    "default": [
        ("agent-0", EventType.TASK_SUCCESS, 1.0),
        ("agent-1", EventType.TASK_FAILURE, 0.8),
        ("agent-2", EventType.COLLABORATION, 1.0),
        ("agent-3", EventType.LEARNING, 0.9),
        ("agent-0", EventType.HIGH_LOAD, 0.6),
        ("agent-1", EventType.IDLE, 0.5),
        ("agent-2", EventType.RESOURCE_SHORTAGE, 0.7),
        ("agent-4", EventType.TASK_SUCCESS, 1.0),
        ("agent-3", EventType.COLLABORATION, 0.8),
        ("agent-0", EventType.LEARNING, 1.0),
    ],
    "stress": [
        ("agent-0", EventType.TASK_FAILURE, 1.0),
        ("agent-1", EventType.TASK_FAILURE, 0.9),
        ("agent-2", EventType.HIGH_LOAD, 1.0),
        ("agent-3", EventType.RESOURCE_SHORTAGE, 1.0),
        ("agent-0", EventType.TASK_FAILURE, 0.8),
        ("agent-1", EventType.HIGH_LOAD, 1.0),
        ("agent-2", EventType.TASK_FAILURE, 0.7),
        ("agent-3", EventType.RESOURCE_SHORTAGE, 0.9),
        ("agent-4", EventType.TASK_FAILURE, 1.0),
        ("agent-0", EventType.HIGH_LOAD, 0.8),
    ],
    "reward": [
        ("agent-0", EventType.TASK_SUCCESS, 1.0),
        ("agent-1", EventType.TASK_SUCCESS, 1.0),
        ("agent-2", EventType.LEARNING, 1.0),
        ("agent-3", EventType.TASK_SUCCESS, 0.9),
        ("agent-4", EventType.LEARNING, 0.8),
        ("agent-0", EventType.COLLABORATION, 1.0),
        ("agent-1", EventType.TASK_SUCCESS, 1.0),
        ("agent-2", EventType.LEARNING, 0.9),
        ("agent-3", EventType.COLLABORATION, 0.7),
        ("agent-4", EventType.TASK_SUCCESS, 1.0),
    ],
    "collaboration": [
        ("agent-0", EventType.COLLABORATION, 1.0),
        ("agent-1", EventType.COLLABORATION, 1.0),
        ("agent-2", EventType.COLLABORATION, 0.9),
        ("agent-3", EventType.COLLABORATION, 0.8),
        ("agent-4", EventType.COLLABORATION, 1.0),
        ("agent-0", EventType.TASK_SUCCESS, 0.5),
        ("agent-1", EventType.COLLABORATION, 1.0),
        ("agent-2", EventType.TASK_SUCCESS, 0.6),
        ("agent-3", EventType.COLLABORATION, 0.9),
        ("agent-4", EventType.LEARNING, 0.4),
    ],
    "dysregulated": [
        ("agent-0", EventType.TASK_FAILURE, 1.0),
        ("agent-0", EventType.TASK_SUCCESS, 1.0),
        ("agent-1", EventType.HIGH_LOAD, 1.0),
        ("agent-1", EventType.IDLE, 1.0),
        ("agent-2", EventType.RESOURCE_SHORTAGE, 1.0),
        ("agent-2", EventType.COLLABORATION, 1.0),
        ("agent-3", EventType.TASK_FAILURE, 0.9),
        ("agent-3", EventType.LEARNING, 0.9),
        ("agent-4", EventType.HIGH_LOAD, 0.8),
        ("agent-4", EventType.TASK_SUCCESS, 0.8),
    ],
}


def run_demo(
    num_agents: int = 5,
    ticks: int = 100,
    scenario: str = "default",
    seed: Optional[int] = None,
) -> EndocrineReport:
    """Run a demo simulation and return the report."""
    engine = SwarmEndocrineEngine(num_agents=num_agents, seed=seed)
    events = SCENARIOS.get(scenario, SCENARIOS["default"])

    for t in range(ticks):
        # Inject events periodically
        if t % 10 == 0:
            for agent_id, event, mag in events:
                idx = int(agent_id.split("-")[1]) if "-" in agent_id else 0
                if idx < num_agents:
                    real_id = f"agent-{idx}"
                    engine.inject_event(real_id, event, mag)
        engine.tick()

    return engine.get_report()


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Swarm Endocrine Engine — autonomous hormonal signaling"
    )
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--ticks", type=int, default=100, help="Simulation ticks")
    parser.add_argument("--scenario", default="default", choices=list(SCENARIOS.keys()),
                        help="Scenario preset")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out", default=None, help="HTML output path")
    parser.add_argument("--json", default=None, help="JSON output path")
    args = parser.parse_args(argv)

    engine = SwarmEndocrineEngine(num_agents=args.agents, seed=args.seed)
    events = SCENARIOS.get(args.scenario, SCENARIOS["default"])

    print(f"🧬 Swarm Endocrine Engine")
    print(f"   Agents: {args.agents} | Ticks: {args.ticks} | Scenario: {args.scenario}")
    print()

    for t in range(args.ticks):
        if t % 10 == 0:
            for agent_id, event, mag in events:
                idx = int(agent_id.split("-")[1]) if "-" in agent_id else 0
                if idx < args.agents:
                    real_id = f"agent-{idx}"
                    engine.inject_event(real_id, event, mag)
        engine.tick()

    report = engine.get_report()
    h = report.health

    score_icon = "✅" if h.score >= 70 else "⚠️" if h.score >= 40 else "❌"
    print(f"{score_icon} Endocrine Health: {h.score:.0f}/100 ({h.tier.value})")
    print(f"   Hormone Balance: {h.components.get('hormone_balance', 0):.0f}")
    print(f"   Receptor Health: {h.components.get('receptor_health', 0):.0f}")
    print(f"   Feedback Response: {h.components.get('feedback_responsiveness', 0):.0f}")
    print(f"   Cascade Stability: {h.components.get('cascade_stability', 0):.0f}")
    print()

    print("🧪 Hormone Levels:")
    for hormone in HormoneType:
        conc = report.current_concentrations.get(hormone.value, 0.0)
        baseline = HORMONE_PROFILES[hormone]["baseline"]
        ratio = conc / baseline if baseline > 0 else 0.0
        indicator = "▲" if ratio > 1.5 else "▼" if ratio < 0.5 else "●"
        print(f"   {indicator} {hormone.value:16s}: {conc:.4f} ({ratio:.1f}x baseline)")
    print()

    print(f"⚡ Cascades: {len(report.cascades)} total")
    print(f"🔄 Feedback events: {len(report.feedback_events)} total")
    print()

    if report.insights:
        print("💡 Insights:")
        for ins in report.insights:
            icon = "🔴" if ins.severity == InsightSeverity.CRITICAL else "🟡" if ins.severity == InsightSeverity.WARNING else "🔵"
            print(f"   {icon} [{ins.category}] {ins.message}")
    else:
        print("💡 No insights — system operating normally.")
    print()

    print("🧬 Agent Summary:")
    for state in report.agent_states[:10]:
        print(f"   {state.agent_id}: produced={state.total_produced:.3f} bound={state.total_bound:.3f}")

    if args.out:
        engine.export_html(args.out)
        print(f"\n📄 HTML report: {args.out}")

    if args.json:
        engine.export_json(args.json)
        print(f"📄 JSON report: {args.json}")


if __name__ == "__main__":
    main()
