"""Swarm Nociception Engine — autonomous pain/damage signaling for swarm protection.

Biologically inspired by the mammalian nociceptive system.  Unlike the endocrine
system (slow, global hormonal signaling) or neuroplasticity (structural
adaptation), nociception provides **rapid damage detection and protective
response** — the swarm's alarm system that prevents injury from escalating.

Capabilities:

- **Nociceptor Array** — each agent has typed nociceptors (mechanical, thermal,
  chemical, polymodal) that detect harmful stimuli with configurable thresholds.
  Sensitization and desensitization adapt thresholds over time.
- **Pain Signal Propagator** — detected noxious signals propagate through the
  swarm via fast (A-delta) and slow (C-fiber) pathways with different latencies
  and persistence.  Spatial/topological neighbors receive referred pain.
- **Protective Reflex Engine** — automatic protective responses triggered by
  pain: withdrawal, guarding, splinting, avoidance, and alert broadcasting.
  Reflexes fire before conscious processing (preemptive protection).
- **Pain Memory Engine** — noxious experiences are stored in a pain memory with
  context (stimulus type, source, intensity).  Enables anticipatory avoidance
  and threat pattern recognition across the swarm.
- **Tolerance Adaptation Engine** — repeated sub-lethal stimuli build tolerance
  (habituation), while novel intense stimuli cause sensitization.  Tracks
  per-agent and swarm-wide tolerance curves.
- **Gate Control Modulator** — implements Melzack-Wall gate control theory:
  non-noxious signals can inhibit pain transmission (counter-stimulation),
  and descending modulation from high-level goals can suppress or amplify pain.
- **Nociceptive Health Scorer** — composite 0-100 score from acute pain load,
  chronic pain burden, reflex responsiveness, memory utility, and tolerance
  balance.  Detects pathological states (allodynia, hyperalgesia, analgesia).

Usage (Python API)::

    from src.nociception import SwarmNociceptionEngine, StimulusType

    engine = SwarmNociceptionEngine(num_agents=6)
    engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, intensity=0.8)
    engine.apply_stimulus("agent-2", StimulusType.THERMAL, intensity=0.5)
    engine.tick(dt=1.0)
    report = engine.get_report()
    print(report.health.score)       # 0-100
    print(report.health.tier)        # PROTECTED / VIGILANT / STRESSED / ...
    engine.export_html("nociception.html")

CLI::

    python -m src.nociception                         # demo with defaults
    python -m src.nociception --agents 8              # more agents
    python -m src.nociception --ticks 100             # longer simulation
    python -m src.nociception --scenario injury       # injury scenario
    python -m src.nociception --out report.html --json nociception.json
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
# Enums
# ---------------------------------------------------------------------------


class StimulusType(str, Enum):
    """Types of noxious stimuli detected by nociceptors."""
    MECHANICAL = "mechanical"       # Physical damage: overload, collision, crush
    THERMAL = "thermal"             # Temperature extremes: overheating, freezing
    CHEMICAL = "chemical"           # Toxic inputs: corrupted data, poison messages
    POLYMODAL = "polymodal"         # Multiple modalities: compound failures
    ISCHEMIC = "ischemic"           # Resource starvation: no CPU, memory, network
    INFLAMMATORY = "inflammatory"   # Cascading damage: error propagation


class FiberType(str, Enum):
    """Signal fiber types with different transmission characteristics."""
    A_DELTA = "a_delta"     # Fast, sharp, well-localized pain
    C_FIBER = "c_fiber"     # Slow, dull, diffuse pain


class ReflexType(str, Enum):
    """Protective reflex responses."""
    WITHDRAWAL = "withdrawal"       # Pull away from stimulus source
    GUARDING = "guarding"           # Protect damaged area
    SPLINTING = "splinting"         # Immobilize to prevent further damage
    AVOIDANCE = "avoidance"         # Preemptive route-around
    ALERT = "alert"                 # Broadcast danger to neighbors
    FREEZE = "freeze"               # Stop all activity to assess threat


class PainPhase(str, Enum):
    """Temporal phases of pain experience."""
    ACUTE = "acute"             # Immediate, sharp, protective
    SUBACUTE = "subacute"       # Settling but still present
    CHRONIC = "chronic"         # Persistent, potentially pathological
    RESOLVED = "resolved"       # Healed, only memory remains


class HealthTier(str, Enum):
    """Overall nociceptive system health classification."""
    PROTECTED = "protected"     # 80-100: Healthy pain response
    VIGILANT = "vigilant"       # 60-79: Heightened but functional
    STRESSED = "stressed"       # 40-59: Overloaded but coping
    SUFFERING = "suffering"     # 20-39: Significant impairment
    CRITICAL = "critical"       # 0-19: System breakdown


class PathologyType(str, Enum):
    """Detectable pathological pain states."""
    ALLODYNIA = "allodynia"             # Pain from non-noxious stimuli
    HYPERALGESIA = "hyperalgesia"       # Exaggerated pain response
    ANALGESIA = "analgesia"             # Inability to feel pain (dangerous)
    CHRONIC_PAIN = "chronic_pain"       # Pain without ongoing stimulus
    REFERRED_PAIN = "referred_pain"     # Pain felt far from source
    PHANTOM_PAIN = "phantom_pain"       # Pain from removed component


class InsightSeverity(str, Enum):
    """Severity level of generated insights."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

STIMULUS_PROFILES: Dict[str, Dict[str, Any]] = {
    StimulusType.MECHANICAL: {
        "base_threshold": 0.4,
        "fiber": FiberType.A_DELTA,
        "decay_rate": 0.3,
        "spread_factor": 0.2,
        "description": "Physical force damage",
    },
    StimulusType.THERMAL: {
        "base_threshold": 0.35,
        "fiber": FiberType.A_DELTA,
        "decay_rate": 0.25,
        "spread_factor": 0.3,
        "description": "Temperature extreme damage",
    },
    StimulusType.CHEMICAL: {
        "base_threshold": 0.3,
        "fiber": FiberType.C_FIBER,
        "decay_rate": 0.15,
        "spread_factor": 0.5,
        "description": "Toxic/corrupted input damage",
    },
    StimulusType.POLYMODAL: {
        "base_threshold": 0.25,
        "fiber": FiberType.C_FIBER,
        "decay_rate": 0.2,
        "spread_factor": 0.4,
        "description": "Multi-modal compound damage",
    },
    StimulusType.ISCHEMIC: {
        "base_threshold": 0.5,
        "fiber": FiberType.C_FIBER,
        "decay_rate": 0.1,
        "spread_factor": 0.6,
        "description": "Resource starvation damage",
    },
    StimulusType.INFLAMMATORY: {
        "base_threshold": 0.2,
        "fiber": FiberType.C_FIBER,
        "decay_rate": 0.12,
        "spread_factor": 0.7,
        "description": "Cascading propagation damage",
    },
}

REFLEX_PROFILES: Dict[str, Dict[str, Any]] = {
    ReflexType.WITHDRAWAL: {
        "threshold": 0.3,
        "latency": 0.1,
        "cost": 0.2,
        "effectiveness": 0.8,
    },
    ReflexType.GUARDING: {
        "threshold": 0.4,
        "latency": 0.3,
        "cost": 0.3,
        "effectiveness": 0.6,
    },
    ReflexType.SPLINTING: {
        "threshold": 0.6,
        "latency": 0.5,
        "cost": 0.5,
        "effectiveness": 0.7,
    },
    ReflexType.AVOIDANCE: {
        "threshold": 0.2,
        "latency": 0.2,
        "cost": 0.1,
        "effectiveness": 0.9,
    },
    ReflexType.ALERT: {
        "threshold": 0.5,
        "latency": 0.1,
        "cost": 0.05,
        "effectiveness": 0.5,
    },
    ReflexType.FREEZE: {
        "threshold": 0.7,
        "latency": 0.05,
        "cost": 0.4,
        "effectiveness": 0.6,
    },
}

# Fiber transmission characteristics
FIBER_PROFILES: Dict[str, Dict[str, float]] = {
    FiberType.A_DELTA: {
        "speed": 1.0,       # Fast transmission
        "persistence": 0.3, # Short-lived signal
        "precision": 0.9,   # Well-localized
    },
    FiberType.C_FIBER: {
        "speed": 0.3,       # Slow transmission
        "persistence": 0.8, # Long-lived signal
        "precision": 0.4,   # Diffuse
    },
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class PainSignal:
    """A pain signal propagating through the swarm."""
    source_agent: str
    stimulus_type: StimulusType
    fiber_type: FiberType
    intensity: float
    original_intensity: float
    tick_created: int
    tick_expires: int
    phase: PainPhase = PainPhase.ACUTE
    hops: int = 0
    referred: bool = False


@dataclass
class Nociceptor:
    """A single nociceptor on an agent."""
    stimulus_type: StimulusType
    threshold: float
    base_threshold: float
    sensitization: float = 0.0  # Positive = sensitized, negative = desensitized
    activation_count: int = 0
    last_activation_tick: int = -1


@dataclass
class ReflexEvent:
    """A protective reflex that was triggered."""
    agent_id: str
    reflex_type: ReflexType
    trigger_signal: StimulusType
    intensity: float
    tick: int
    effectiveness: float


@dataclass
class PainMemory:
    """A stored pain experience for future avoidance."""
    agent_id: str
    stimulus_type: StimulusType
    intensity: float
    source_context: str
    tick_recorded: int
    times_recalled: int = 0
    avoidance_learned: bool = False


@dataclass
class ToleranceProfile:
    """Tolerance adaptation profile for an agent."""
    agent_id: str
    exposures: Dict[str, int] = field(default_factory=dict)
    tolerance_levels: Dict[str, float] = field(default_factory=dict)
    peak_tolerance: Dict[str, float] = field(default_factory=dict)
    sensitization_events: int = 0
    habituation_events: int = 0


@dataclass
class GateState:
    """Gate control modulator state."""
    agent_id: str
    gate_openness: float = 0.5  # 0=fully closed, 1=fully open
    inhibitory_input: float = 0.0
    excitatory_input: float = 0.0
    descending_modulation: float = 0.0  # Negative=suppression, positive=amplification


@dataclass
class Insight:
    """An autonomous insight about nociceptive patterns."""
    category: str
    severity: InsightSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthScore:
    """Composite nociceptive health assessment."""
    score: float
    tier: HealthTier
    acute_load: float
    chronic_burden: float
    reflex_responsiveness: float
    memory_utility: float
    tolerance_balance: float
    gate_effectiveness: float
    pathologies: List[PathologyType] = field(default_factory=list)


@dataclass
class AgentPainState:
    """Complete pain state for one agent."""
    agent_id: str
    nociceptors: List[Nociceptor]
    active_signals: List[PainSignal]
    current_pain_level: float
    pain_history: List[float]
    reflexes_triggered: int
    tolerance: ToleranceProfile
    gate: GateState
    memories: List[PainMemory]


@dataclass
class NociceptionReport:
    """Full swarm nociception report."""
    num_agents: int
    total_ticks: int
    agent_states: Dict[str, AgentPainState]
    all_signals: List[PainSignal]
    all_reflexes: List[ReflexEvent]
    all_memories: List[PainMemory]
    health: HealthScore
    insights: List[Insight]
    pathologies_detected: List[Dict[str, Any]]
    swarm_pain_timeline: List[float]


# ---------------------------------------------------------------------------
# Engine: Nociceptor Array
# ---------------------------------------------------------------------------


class NociceptorArray:
    """Manages nociceptors across all agents — detects noxious stimuli."""

    def __init__(self, agent_ids: List[str]):
        self.agents: Dict[str, List[Nociceptor]] = {}
        for aid in agent_ids:
            self.agents[aid] = [
                Nociceptor(
                    stimulus_type=st,
                    threshold=STIMULUS_PROFILES[st]["base_threshold"],
                    base_threshold=STIMULUS_PROFILES[st]["base_threshold"],
                )
                for st in StimulusType
            ]

    def detect(self, agent_id: str, stimulus_type: StimulusType,
               intensity: float, tick: int) -> Optional[float]:
        """Test stimulus against nociceptor threshold. Returns pain intensity or None."""
        if agent_id not in self.agents:
            return None
        for noc in self.agents[agent_id]:
            if noc.stimulus_type == stimulus_type:
                effective_threshold = max(0.05, noc.threshold - noc.sensitization)
                if intensity >= effective_threshold:
                    pain_intensity = (intensity - effective_threshold) / (1.0 - effective_threshold + 1e-9)
                    pain_intensity = min(1.0, max(0.0, pain_intensity))
                    noc.activation_count += 1
                    noc.last_activation_tick = tick
                    return pain_intensity
                return None
        return None

    def sensitize(self, agent_id: str, stimulus_type: StimulusType, amount: float):
        """Increase sensitivity (lower threshold) — e.g. after injury."""
        for noc in self.agents.get(agent_id, []):
            if noc.stimulus_type == stimulus_type:
                noc.sensitization = min(0.3, noc.sensitization + amount)

    def desensitize(self, agent_id: str, stimulus_type: StimulusType, amount: float):
        """Decrease sensitivity (raise threshold) — habituation."""
        for noc in self.agents.get(agent_id, []):
            if noc.stimulus_type == stimulus_type:
                noc.sensitization = max(-0.2, noc.sensitization - amount)

    def get_state(self, agent_id: str) -> List[Nociceptor]:
        return self.agents.get(agent_id, [])


# ---------------------------------------------------------------------------
# Engine: Pain Signal Propagator
# ---------------------------------------------------------------------------


class PainSignalPropagator:
    """Propagates pain signals through the swarm via fast/slow pathways."""

    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.active_signals: Dict[str, List[PainSignal]] = {aid: [] for aid in agent_ids}
        self.all_signals: List[PainSignal] = []

    def emit_signal(self, agent_id: str, stimulus_type: StimulusType,
                    intensity: float, tick: int) -> PainSignal:
        """Create and emit a new pain signal from a nociceptor activation."""
        profile = STIMULUS_PROFILES[stimulus_type]
        fiber = profile["fiber"]
        fiber_profile = FIBER_PROFILES[fiber]
        duration = int(5 + 15 * fiber_profile["persistence"])
        signal = PainSignal(
            source_agent=agent_id,
            stimulus_type=stimulus_type,
            fiber_type=fiber,
            intensity=intensity,
            original_intensity=intensity,
            tick_created=tick,
            tick_expires=tick + duration,
        )
        self.active_signals[agent_id].append(signal)
        self.all_signals.append(signal)
        return signal

    def propagate(self, tick: int):
        """Propagate signals to neighbors (referred pain) and decay existing signals."""
        new_referred: List[Tuple[str, PainSignal]] = []
        for agent_id, signals in self.active_signals.items():
            surviving = []
            for sig in signals:
                if tick >= sig.tick_expires:
                    sig.phase = PainPhase.RESOLVED
                    continue
                # Decay
                age = tick - sig.tick_created
                profile = STIMULUS_PROFILES[sig.stimulus_type]
                sig.intensity = sig.original_intensity * math.exp(-profile["decay_rate"] * age)
                # Phase transition
                if age > 10:
                    sig.phase = PainPhase.CHRONIC
                elif age > 4:
                    sig.phase = PainPhase.SUBACUTE
                surviving.append(sig)
                # Spread to neighbors (referred pain) with probability
                if not sig.referred and sig.intensity > 0.3 and sig.hops < 2:
                    spread_factor = profile["spread_factor"]
                    for other_id in self.agent_ids:
                        if other_id != agent_id and random.random() < spread_factor * 0.3:
                            referred = PainSignal(
                                source_agent=sig.source_agent,
                                stimulus_type=sig.stimulus_type,
                                fiber_type=sig.fiber_type,
                                intensity=sig.intensity * 0.4,
                                original_intensity=sig.intensity * 0.4,
                                tick_created=tick,
                                tick_expires=tick + 5,
                                phase=PainPhase.ACUTE,
                                hops=sig.hops + 1,
                                referred=True,
                            )
                            new_referred.append((other_id, referred))
            self.active_signals[agent_id] = surviving

        for agent_id, sig in new_referred:
            self.active_signals[agent_id].append(sig)
            self.all_signals.append(sig)

    def get_pain_level(self, agent_id: str) -> float:
        """Get current aggregate pain level for an agent."""
        signals = self.active_signals.get(agent_id, [])
        if not signals:
            return 0.0
        # Sum with diminishing returns
        intensities = sorted([s.intensity for s in signals], reverse=True)
        total = 0.0
        for i, intensity in enumerate(intensities):
            total += intensity * (0.7 ** i)
        return min(1.0, total)


# ---------------------------------------------------------------------------
# Engine: Protective Reflex Engine
# ---------------------------------------------------------------------------


class ProtectiveReflexEngine:
    """Triggers automatic protective responses based on pain signals."""

    def __init__(self):
        self.events: List[ReflexEvent] = []
        self.agent_cooldowns: Dict[str, Dict[str, int]] = defaultdict(dict)

    def evaluate(self, agent_id: str, pain_level: float,
                 stimulus_type: StimulusType, tick: int) -> List[ReflexEvent]:
        """Evaluate which reflexes should fire given current pain level."""
        triggered = []
        for reflex_type, profile in REFLEX_PROFILES.items():
            # Check cooldown
            cooldown_key = f"{reflex_type}"
            last_fire = self.agent_cooldowns[agent_id].get(cooldown_key, -100)
            if tick - last_fire < 3:
                continue
            if pain_level >= profile["threshold"]:
                event = ReflexEvent(
                    agent_id=agent_id,
                    reflex_type=reflex_type,
                    trigger_signal=stimulus_type,
                    intensity=pain_level,
                    tick=tick,
                    effectiveness=profile["effectiveness"],
                )
                triggered.append(event)
                self.events.append(event)
                self.agent_cooldowns[agent_id][cooldown_key] = tick
        return triggered


# ---------------------------------------------------------------------------
# Engine: Pain Memory Engine
# ---------------------------------------------------------------------------


class PainMemoryEngine:
    """Stores and retrieves pain experiences for avoidance learning."""

    def __init__(self, max_memories_per_agent: int = 50):
        self.memories: Dict[str, List[PainMemory]] = defaultdict(list)
        self.max_per_agent = max_memories_per_agent

    def record(self, agent_id: str, stimulus_type: StimulusType,
               intensity: float, context: str, tick: int) -> PainMemory:
        """Record a new pain experience."""
        memory = PainMemory(
            agent_id=agent_id,
            stimulus_type=stimulus_type,
            intensity=intensity,
            source_context=context,
            tick_recorded=tick,
        )
        self.memories[agent_id].append(memory)
        # Evict oldest if over capacity
        if len(self.memories[agent_id]) > self.max_per_agent:
            self.memories[agent_id] = self.memories[agent_id][-self.max_per_agent:]
        return memory

    def recall(self, agent_id: str, stimulus_type: Optional[StimulusType] = None) -> List[PainMemory]:
        """Recall pain memories, optionally filtered by stimulus type."""
        memories = self.memories.get(agent_id, [])
        if stimulus_type:
            memories = [m for m in memories if m.stimulus_type == stimulus_type]
        for m in memories:
            m.times_recalled += 1
        return memories

    def has_learned_avoidance(self, agent_id: str, stimulus_type: StimulusType) -> bool:
        """Check if agent has learned to avoid a particular stimulus type."""
        memories = [m for m in self.memories.get(agent_id, [])
                    if m.stimulus_type == stimulus_type]
        # Avoidance learned after 3+ painful experiences of same type
        if len(memories) >= 3:
            for m in memories:
                m.avoidance_learned = True
            return True
        return False

    def get_all(self) -> List[PainMemory]:
        """Get all memories across all agents."""
        result = []
        for mems in self.memories.values():
            result.extend(mems)
        return result


# ---------------------------------------------------------------------------
# Engine: Tolerance Adaptation Engine
# ---------------------------------------------------------------------------


class ToleranceAdaptationEngine:
    """Manages tolerance curves — habituation and sensitization."""

    def __init__(self, agent_ids: List[str]):
        self.profiles: Dict[str, ToleranceProfile] = {
            aid: ToleranceProfile(agent_id=aid)
            for aid in agent_ids
        }

    def process_exposure(self, agent_id: str, stimulus_type: StimulusType,
                         intensity: float) -> Tuple[str, float]:
        """Process a stimulus exposure and update tolerance. Returns (effect, adjusted_intensity)."""
        profile = self.profiles.get(agent_id)
        if not profile:
            return ("none", intensity)

        st_key = stimulus_type.value
        profile.exposures[st_key] = profile.exposures.get(st_key, 0) + 1
        current_tolerance = profile.tolerance_levels.get(st_key, 0.0)

        if intensity < 0.6:
            # Sub-lethal: habituation (build tolerance)
            new_tolerance = min(0.5, current_tolerance + 0.03)
            profile.tolerance_levels[st_key] = new_tolerance
            profile.habituation_events += 1
            effect = "habituation"
        else:
            # High intensity: sensitization (lower tolerance)
            new_tolerance = max(-0.2, current_tolerance - 0.05)
            profile.tolerance_levels[st_key] = new_tolerance
            profile.sensitization_events += 1
            effect = "sensitization"

        # Track peak
        if new_tolerance > profile.peak_tolerance.get(st_key, 0.0):
            profile.peak_tolerance[st_key] = new_tolerance

        # Adjust perceived intensity based on tolerance
        adjusted = max(0.0, intensity - current_tolerance)
        return (effect, adjusted)


# ---------------------------------------------------------------------------
# Engine: Gate Control Modulator
# ---------------------------------------------------------------------------


class GateControlModulator:
    """Implements Melzack-Wall gate control theory for pain modulation."""

    def __init__(self, agent_ids: List[str]):
        self.gates: Dict[str, GateState] = {
            aid: GateState(agent_id=aid) for aid in agent_ids
        }

    def apply_inhibition(self, agent_id: str, amount: float):
        """Apply non-noxious inhibitory input (closes gate)."""
        gate = self.gates.get(agent_id)
        if gate:
            gate.inhibitory_input = min(1.0, gate.inhibitory_input + amount)

    def apply_excitation(self, agent_id: str, amount: float):
        """Apply excitatory input (opens gate)."""
        gate = self.gates.get(agent_id)
        if gate:
            gate.excitatory_input = min(1.0, gate.excitatory_input + amount)

    def set_descending_modulation(self, agent_id: str, modulation: float):
        """Set top-down modulation (-1 to +1). Negative suppresses pain."""
        gate = self.gates.get(agent_id)
        if gate:
            gate.descending_modulation = max(-1.0, min(1.0, modulation))

    def compute_gate(self, agent_id: str) -> float:
        """Compute effective gate openness (0=fully closed, 1=fully open)."""
        gate = self.gates.get(agent_id)
        if not gate:
            return 0.5
        # Gate openness: excitation opens, inhibition closes, modulation adjusts
        raw = 0.5 + 0.3 * gate.excitatory_input - 0.3 * gate.inhibitory_input
        raw += 0.2 * gate.descending_modulation
        gate.gate_openness = max(0.0, min(1.0, raw))
        return gate.gate_openness

    def modulate_pain(self, agent_id: str, raw_pain: float) -> float:
        """Apply gate modulation to raw pain signal."""
        openness = self.compute_gate(agent_id)
        return raw_pain * openness

    def decay(self):
        """Decay inputs toward baseline each tick."""
        for gate in self.gates.values():
            gate.inhibitory_input *= 0.8
            gate.excitatory_input *= 0.8
            gate.descending_modulation *= 0.9


# ---------------------------------------------------------------------------
# Engine: Nociceptive Health Scorer
# ---------------------------------------------------------------------------


class NociceptiveHealthScorer:
    """Computes composite nociceptive health score."""

    def score(self, agent_states: Dict[str, AgentPainState],
              reflexes: List[ReflexEvent], memories: List[PainMemory],
              timeline: List[float]) -> HealthScore:
        """Compute health score from all subsystems."""
        if not agent_states:
            return HealthScore(
                score=100.0, tier=HealthTier.PROTECTED,
                acute_load=0.0, chronic_burden=0.0,
                reflex_responsiveness=1.0, memory_utility=1.0,
                tolerance_balance=1.0, gate_effectiveness=1.0,
            )

        # Acute pain load (current pain levels)
        pain_levels = [s.current_pain_level for s in agent_states.values()]
        acute_load = statistics.mean(pain_levels) if pain_levels else 0.0

        # Chronic burden (how much pain persists over time)
        chronic_signals = 0
        total_signals = 0
        for state in agent_states.values():
            for sig in state.active_signals:
                total_signals += 1
                if sig.phase in (PainPhase.CHRONIC, PainPhase.SUBACUTE):
                    chronic_signals += 1
        chronic_burden = chronic_signals / max(1, total_signals)

        # Reflex responsiveness (are reflexes firing when needed?)
        high_pain_agents = sum(1 for p in pain_levels if p > 0.4)
        agents_with_reflexes = len(set(r.agent_id for r in reflexes[-20:]))
        reflex_responsiveness = min(1.0, agents_with_reflexes / max(1, high_pain_agents)) if high_pain_agents > 0 else 1.0

        # Memory utility (are memories leading to avoidance?)
        avoidance_memories = sum(1 for m in memories if m.avoidance_learned)
        memory_utility = min(1.0, avoidance_memories / max(1, len(memories)) * 3) if memories else 0.5

        # Tolerance balance (not too habituated, not too sensitized)
        tolerance_scores = []
        for state in agent_states.values():
            tol_vals = list(state.tolerance.tolerance_levels.values())
            if tol_vals:
                avg_tol = statistics.mean(tol_vals)
                # Ideal tolerance is moderate (0.1-0.3)
                if 0.1 <= avg_tol <= 0.3:
                    tolerance_scores.append(1.0)
                elif avg_tol < 0.0:
                    tolerance_scores.append(0.5)  # Over-sensitized
                else:
                    tolerance_scores.append(0.7)  # Over-habituated
        tolerance_balance = statistics.mean(tolerance_scores) if tolerance_scores else 0.8

        # Gate effectiveness
        gate_scores = []
        for state in agent_states.values():
            g = state.gate.gate_openness
            # Ideal gate is responsive (0.3-0.7)
            if 0.3 <= g <= 0.7:
                gate_scores.append(1.0)
            else:
                gate_scores.append(0.6)
        gate_effectiveness = statistics.mean(gate_scores) if gate_scores else 0.8

        # Detect pathologies
        pathologies = self._detect_pathologies(agent_states, timeline)

        # Composite score
        score = 100.0
        score -= acute_load * 30  # High acute pain hurts score
        score -= chronic_burden * 25  # Chronic pain is bad
        score -= (1.0 - reflex_responsiveness) * 15
        score -= (1.0 - memory_utility) * 10
        score -= (1.0 - tolerance_balance) * 10
        score -= (1.0 - gate_effectiveness) * 10
        score -= len(pathologies) * 5  # Pathologies penalize
        score = max(0.0, min(100.0, score))

        # Determine tier
        if score >= 80:
            tier = HealthTier.PROTECTED
        elif score >= 60:
            tier = HealthTier.VIGILANT
        elif score >= 40:
            tier = HealthTier.STRESSED
        elif score >= 20:
            tier = HealthTier.SUFFERING
        else:
            tier = HealthTier.CRITICAL

        return HealthScore(
            score=round(score, 1),
            tier=tier,
            acute_load=round(acute_load, 3),
            chronic_burden=round(chronic_burden, 3),
            reflex_responsiveness=round(reflex_responsiveness, 3),
            memory_utility=round(memory_utility, 3),
            tolerance_balance=round(tolerance_balance, 3),
            gate_effectiveness=round(gate_effectiveness, 3),
            pathologies=pathologies,
        )

    def _detect_pathologies(self, agent_states: Dict[str, AgentPainState],
                            timeline: List[float]) -> List[PathologyType]:
        """Detect pathological pain states."""
        pathologies = []

        # Check for allodynia (low-threshold activations)
        for state in agent_states.values():
            for noc in state.nociceptors:
                if noc.sensitization > 0.2 and noc.activation_count > 5:
                    if PathologyType.ALLODYNIA not in pathologies:
                        pathologies.append(PathologyType.ALLODYNIA)

        # Check for analgesia (no pain despite stimuli)
        agents_with_signals = sum(1 for s in agent_states.values() if s.active_signals)
        agents_with_pain = sum(1 for s in agent_states.values() if s.current_pain_level > 0.1)
        if agents_with_signals > 0 and agents_with_pain == 0:
            pathologies.append(PathologyType.ANALGESIA)

        # Check for chronic pain
        if timeline and len(timeline) > 10:
            recent = timeline[-10:]
            if all(p > 0.3 for p in recent):
                pathologies.append(PathologyType.CHRONIC_PAIN)

        # Check for hyperalgesia
        for state in agent_states.values():
            if state.current_pain_level > 0.8 and len(state.active_signals) <= 2:
                if PathologyType.HYPERALGESIA not in pathologies:
                    pathologies.append(PathologyType.HYPERALGESIA)

        return pathologies


# ---------------------------------------------------------------------------
# Engine: Insight Generator
# ---------------------------------------------------------------------------


class NociceptionInsightGenerator:
    """Generates autonomous insights about nociceptive patterns."""

    def generate(self, report_data: Dict[str, Any]) -> List[Insight]:
        """Generate insights from current nociception state."""
        insights = []
        agent_states = report_data.get("agent_states", {})
        reflexes = report_data.get("reflexes", [])
        memories = report_data.get("memories", [])
        timeline = report_data.get("timeline", [])
        health = report_data.get("health")

        # High pain load warning
        pain_levels = [s.current_pain_level for s in agent_states.values()]
        if pain_levels:
            avg_pain = statistics.mean(pain_levels)
            if avg_pain > 0.6:
                insights.append(Insight(
                    category="acute_overload",
                    severity=InsightSeverity.CRITICAL,
                    message=f"Swarm experiencing severe pain overload (avg={avg_pain:.2f})",
                    details={"average_pain": avg_pain, "agents_in_pain": sum(1 for p in pain_levels if p > 0.3)},
                ))
            elif avg_pain > 0.3:
                insights.append(Insight(
                    category="elevated_pain",
                    severity=InsightSeverity.WARNING,
                    message=f"Elevated pain across swarm (avg={avg_pain:.2f})",
                    details={"average_pain": avg_pain},
                ))

        # Reflex exhaustion
        if reflexes:
            recent_reflexes = [r for r in reflexes if r.tick > (report_data.get("tick", 0) - 10)]
            if len(recent_reflexes) > len(agent_states) * 3:
                insights.append(Insight(
                    category="reflex_exhaustion",
                    severity=InsightSeverity.WARNING,
                    message="Protective reflexes firing excessively — possible exhaustion",
                    details={"recent_count": len(recent_reflexes)},
                ))

        # Memory saturation
        for aid, state in agent_states.items():
            if len(state.memories) >= 45:
                insights.append(Insight(
                    category="memory_saturation",
                    severity=InsightSeverity.INFO,
                    message=f"Agent {aid} approaching pain memory capacity",
                    details={"agent": aid, "memory_count": len(state.memories)},
                ))
                break

        # Tolerance imbalance
        for aid, state in agent_states.items():
            tol = state.tolerance
            if tol.sensitization_events > tol.habituation_events * 2 and tol.sensitization_events > 5:
                insights.append(Insight(
                    category="chronic_sensitization",
                    severity=InsightSeverity.WARNING,
                    message=f"Agent {aid} chronically sensitized — overreacting to stimuli",
                    details={"agent": aid, "sensitization": tol.sensitization_events, "habituation": tol.habituation_events},
                ))
                break

        # Pain persistence
        if timeline and len(timeline) > 15:
            recent = timeline[-15:]
            if statistics.mean(recent) > 0.2 and all(p > 0.1 for p in recent):
                insights.append(Insight(
                    category="persistent_pain",
                    severity=InsightSeverity.WARNING,
                    message="Swarm pain has not resolved — check for ongoing damage sources",
                    details={"mean_recent": statistics.mean(recent)},
                ))

        # Gate stuck
        for aid, state in agent_states.items():
            if state.gate.gate_openness > 0.85:
                insights.append(Insight(
                    category="gate_stuck_open",
                    severity=InsightSeverity.WARNING,
                    message=f"Agent {aid} pain gate stuck open — amplifying all signals",
                    details={"agent": aid, "openness": state.gate.gate_openness},
                ))
                break
            elif state.gate.gate_openness < 0.15:
                insights.append(Insight(
                    category="gate_stuck_closed",
                    severity=InsightSeverity.INFO,
                    message=f"Agent {aid} pain gate stuck closed — may miss real threats",
                    details={"agent": aid, "openness": state.gate.gate_openness},
                ))
                break

        return insights


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "description": "Normal operation with occasional minor stimuli",
        "events": [
            {"tick": 5, "agent": 0, "type": StimulusType.MECHANICAL, "intensity": 0.3},
            {"tick": 15, "agent": 2, "type": StimulusType.THERMAL, "intensity": 0.4},
            {"tick": 30, "agent": 1, "type": StimulusType.CHEMICAL, "intensity": 0.35},
            {"tick": 45, "agent": 3, "type": StimulusType.MECHANICAL, "intensity": 0.25},
        ],
    },
    "injury": {
        "description": "Sudden severe injury to one agent with spreading damage",
        "events": [
            {"tick": 5, "agent": 0, "type": StimulusType.MECHANICAL, "intensity": 0.9},
            {"tick": 6, "agent": 0, "type": StimulusType.INFLAMMATORY, "intensity": 0.7},
            {"tick": 10, "agent": 1, "type": StimulusType.INFLAMMATORY, "intensity": 0.5},
            {"tick": 15, "agent": 2, "type": StimulusType.CHEMICAL, "intensity": 0.6},
            {"tick": 25, "agent": 0, "type": StimulusType.ISCHEMIC, "intensity": 0.4},
        ],
    },
    "chronic": {
        "description": "Persistent low-grade damage that builds over time",
        "events": [
            {"tick": i * 5, "agent": i % 4, "type": StimulusType.CHEMICAL, "intensity": 0.35}
            for i in range(15)
        ],
    },
    "adaptation": {
        "description": "Repeated stimuli showing tolerance development",
        "events": [
            {"tick": i * 4, "agent": 0, "type": StimulusType.MECHANICAL, "intensity": 0.45}
            for i in range(12)
        ] + [
            {"tick": 60, "agent": 0, "type": StimulusType.MECHANICAL, "intensity": 0.9},
        ],
    },
    "cascade": {
        "description": "Inflammatory cascade spreading across the swarm",
        "events": [
            {"tick": 3, "agent": 0, "type": StimulusType.POLYMODAL, "intensity": 0.8},
            {"tick": 8, "agent": 1, "type": StimulusType.INFLAMMATORY, "intensity": 0.7},
            {"tick": 12, "agent": 2, "type": StimulusType.INFLAMMATORY, "intensity": 0.65},
            {"tick": 16, "agent": 3, "type": StimulusType.INFLAMMATORY, "intensity": 0.6},
            {"tick": 20, "agent": 4, "type": StimulusType.INFLAMMATORY, "intensity": 0.55},
            {"tick": 25, "agent": 5, "type": StimulusType.ISCHEMIC, "intensity": 0.7},
        ],
    },
}


# ---------------------------------------------------------------------------
# Main Engine: SwarmNociceptionEngine
# ---------------------------------------------------------------------------


class SwarmNociceptionEngine:
    """Unified swarm nociception engine orchestrating all subsystems."""

    def __init__(self, num_agents: int = 6, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.num_agents = num_agents
        self.agent_ids = [f"agent-{i}" for i in range(num_agents)]
        self.tick = 0

        # Initialize subsystems
        self.nociceptors = NociceptorArray(self.agent_ids)
        self.propagator = PainSignalPropagator(self.agent_ids)
        self.reflexes = ProtectiveReflexEngine()
        self.memory = PainMemoryEngine()
        self.tolerance = ToleranceAdaptationEngine(self.agent_ids)
        self.gate = GateControlModulator(self.agent_ids)
        self.scorer = NociceptiveHealthScorer()
        self.insight_gen = NociceptionInsightGenerator()

        # State tracking
        self.pain_timeline: List[float] = []
        self.pain_history: Dict[str, List[float]] = {aid: [] for aid in self.agent_ids}

    def apply_stimulus(self, agent_id: str, stimulus_type: StimulusType,
                       intensity: float, context: str = ""):
        """Apply a noxious stimulus to an agent."""
        if agent_id not in self.agent_ids:
            return

        # Tolerance adjustment
        effect, adjusted = self.tolerance.process_exposure(agent_id, stimulus_type, intensity)

        # Nociceptor detection
        pain = self.nociceptors.detect(agent_id, stimulus_type, adjusted, self.tick)
        if pain is None:
            return

        # Gate modulation
        self.gate.apply_excitation(agent_id, pain * 0.5)
        modulated_pain = self.gate.modulate_pain(agent_id, pain)

        # Emit pain signal
        self.propagator.emit_signal(agent_id, stimulus_type, modulated_pain, self.tick)

        # Record memory
        self.memory.record(agent_id, stimulus_type, modulated_pain,
                          context or f"{stimulus_type.value}@tick{self.tick}", self.tick)

        # Check for avoidance learning
        self.memory.has_learned_avoidance(agent_id, stimulus_type)

        # Sensitization/desensitization
        if effect == "sensitization":
            self.nociceptors.sensitize(agent_id, stimulus_type, 0.02)
        elif effect == "habituation" and intensity < 0.4:
            self.nociceptors.desensitize(agent_id, stimulus_type, 0.01)

        # Evaluate reflexes
        pain_level = self.propagator.get_pain_level(agent_id)
        self.reflexes.evaluate(agent_id, pain_level, stimulus_type, self.tick)

    def apply_inhibition(self, agent_id: str, amount: float = 0.3):
        """Apply counter-stimulation (gate closure) to an agent."""
        self.gate.apply_inhibition(agent_id, amount)

    def set_descending_modulation(self, agent_id: str, modulation: float):
        """Set top-down goal-based pain modulation for an agent."""
        self.gate.set_descending_modulation(agent_id, modulation)

    def do_tick(self, dt: float = 1.0):
        """Advance simulation by one tick."""
        self.tick += 1
        self.propagator.propagate(self.tick)
        self.gate.decay()

        # Record pain levels
        pain_sum = 0.0
        for aid in self.agent_ids:
            level = self.propagator.get_pain_level(aid)
            self.pain_history[aid].append(level)
            pain_sum += level
        avg_pain = pain_sum / max(1, len(self.agent_ids))
        self.pain_timeline.append(avg_pain)

    def get_report(self) -> NociceptionReport:
        """Generate comprehensive nociception report."""
        agent_states: Dict[str, AgentPainState] = {}
        for aid in self.agent_ids:
            agent_states[aid] = AgentPainState(
                agent_id=aid,
                nociceptors=self.nociceptors.get_state(aid),
                active_signals=self.propagator.active_signals.get(aid, []),
                current_pain_level=self.propagator.get_pain_level(aid),
                pain_history=self.pain_history[aid],
                reflexes_triggered=sum(1 for r in self.reflexes.events if r.agent_id == aid),
                tolerance=self.tolerance.profiles[aid],
                gate=self.gate.gates[aid],
                memories=self.memory.memories.get(aid, []),
            )

        all_memories = self.memory.get_all()
        health = self.scorer.score(agent_states, self.reflexes.events, all_memories, self.pain_timeline)

        # Generate insights
        insight_data = {
            "agent_states": agent_states,
            "reflexes": self.reflexes.events,
            "memories": all_memories,
            "timeline": self.pain_timeline,
            "health": health,
            "tick": self.tick,
        }
        insights = self.insight_gen.generate(insight_data)

        # Pathology details
        pathology_details = [{"type": p.value, "detected_at_tick": self.tick} for p in health.pathologies]

        return NociceptionReport(
            num_agents=self.num_agents,
            total_ticks=self.tick,
            agent_states=agent_states,
            all_signals=self.propagator.all_signals,
            all_reflexes=self.reflexes.events,
            all_memories=all_memories,
            health=health,
            insights=insights,
            pathologies_detected=pathology_details,
            swarm_pain_timeline=self.pain_timeline,
        )

    def export_json(self, path: str):
        """Export report as JSON."""
        report = self.get_report()
        data = {
            "num_agents": report.num_agents,
            "total_ticks": report.total_ticks,
            "health": {
                "score": report.health.score,
                "tier": report.health.tier.value,
                "acute_load": report.health.acute_load,
                "chronic_burden": report.health.chronic_burden,
                "reflex_responsiveness": report.health.reflex_responsiveness,
                "memory_utility": report.health.memory_utility,
                "tolerance_balance": report.health.tolerance_balance,
                "gate_effectiveness": report.health.gate_effectiveness,
                "pathologies": [p.value for p in report.health.pathologies],
            },
            "insights": [
                {"category": i.category, "severity": i.severity.value, "message": i.message}
                for i in report.insights
            ],
            "pain_timeline": report.swarm_pain_timeline,
            "total_signals": len(report.all_signals),
            "total_reflexes": len(report.all_reflexes),
            "total_memories": len(report.all_memories),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def export_html(self, path: str):
        """Export interactive HTML dashboard."""
        report = self.get_report()
        html = _generate_html_dashboard(report)
        Path(path).write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------


def _generate_html_dashboard(report: NociceptionReport) -> str:
    """Generate interactive HTML dashboard for nociception report."""
    health = report.health
    tier_colors = {
        HealthTier.PROTECTED: "#22c55e",
        HealthTier.VIGILANT: "#eab308",
        HealthTier.STRESSED: "#f97316",
        HealthTier.SUFFERING: "#ef4444",
        HealthTier.CRITICAL: "#7f1d1d",
    }
    color = tier_colors.get(health.tier, "#6b7280")

    # Pain timeline sparkline as SVG
    timeline = report.swarm_pain_timeline
    sparkline_points = ""
    if timeline:
        max_val = max(max(timeline), 0.01)
        step = max(1, len(timeline) // 200)
        sampled = timeline[::step]
        points = []
        for i, v in enumerate(sampled):
            x = i * (600 / max(1, len(sampled) - 1))
            y = 80 - (v / max_val) * 70
            points.append(f"{x:.1f},{y:.1f}")
        sparkline_points = " ".join(points)

    # Agent pain bars
    agent_bars = ""
    for aid in sorted(report.agent_states.keys()):
        state = report.agent_states[aid]
        pct = min(100, int(state.current_pain_level * 100))
        bar_color = "#22c55e" if pct < 30 else "#eab308" if pct < 60 else "#ef4444"
        agent_bars += f"""
        <div style="margin:4px 0;display:flex;align-items:center;gap:8px">
          <span style="width:70px;font-size:12px">{html_mod.escape(aid)}</span>
          <div style="flex:1;height:16px;background:#1e293b;border-radius:4px;overflow:hidden">
            <div style="width:{pct}%;height:100%;background:{bar_color};transition:width 0.3s"></div>
          </div>
          <span style="width:40px;font-size:12px;text-align:right">{pct}%</span>
        </div>"""

    # Insights list
    insight_html = ""
    severity_icons = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
    for ins in report.insights:
        icon = severity_icons.get(ins.severity.value, "•")
        insight_html += f"<div style='margin:4px 0;padding:6px 10px;background:#1e293b;border-radius:4px;font-size:13px'>{icon} {html_mod.escape(ins.message)}</div>"
    if not insight_html:
        insight_html = "<div style='color:#6b7280;font-size:13px'>No insights generated</div>"

    # Reflex summary
    reflex_counts: Dict[str, int] = defaultdict(int)
    for r in report.all_reflexes:
        reflex_counts[r.reflex_type.value] += 1
    reflex_html = ""
    for rt, count in sorted(reflex_counts.items(), key=lambda x: -x[1]):
        reflex_html += f"<div style='display:inline-block;margin:3px;padding:4px 10px;background:#1e293b;border-radius:12px;font-size:12px'>{rt}: {count}</div>"
    if not reflex_html:
        reflex_html = "<div style='color:#6b7280;font-size:13px'>No reflexes triggered</div>"

    # Pathology badges
    pathology_html = ""
    for p in health.pathologies:
        pathology_html += f"<span style='display:inline-block;margin:3px;padding:4px 10px;background:#7f1d1d;border-radius:12px;font-size:12px;color:#fca5a5'>{p.value}</span>"
    if not pathology_html:
        pathology_html = "<span style='color:#6b7280;font-size:13px'>None detected ✓</span>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swarm Nociception Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}}
.card{{background:#1e293b;border-radius:12px;padding:20px;margin:12px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
h1{{font-size:24px;margin-bottom:8px}}
h2{{font-size:16px;color:#94a3b8;margin-bottom:12px}}
.score{{font-size:64px;font-weight:700;color:{color}}}
.tier{{font-size:18px;color:{color};text-transform:uppercase;letter-spacing:2px}}
.metric{{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #334155}}
.metric-label{{color:#94a3b8;font-size:13px}}
.metric-value{{font-weight:600;font-size:13px}}
</style>
</head>
<body>
<h1>🔥 Swarm Nociception Dashboard</h1>
<p style="color:#94a3b8;margin-bottom:16px">{report.num_agents} agents · {report.total_ticks} ticks · {len(report.all_signals)} signals · {len(report.all_reflexes)} reflexes</p>

<div class="grid">
  <div class="card" style="text-align:center">
    <h2>Health Score</h2>
    <div class="score">{health.score}</div>
    <div class="tier">{health.tier.value}</div>
  </div>
  <div class="card">
    <h2>Metrics</h2>
    <div class="metric"><span class="metric-label">Acute Load</span><span class="metric-value">{health.acute_load}</span></div>
    <div class="metric"><span class="metric-label">Chronic Burden</span><span class="metric-value">{health.chronic_burden}</span></div>
    <div class="metric"><span class="metric-label">Reflex Responsiveness</span><span class="metric-value">{health.reflex_responsiveness}</span></div>
    <div class="metric"><span class="metric-label">Memory Utility</span><span class="metric-value">{health.memory_utility}</span></div>
    <div class="metric"><span class="metric-label">Tolerance Balance</span><span class="metric-value">{health.tolerance_balance}</span></div>
    <div class="metric"><span class="metric-label">Gate Effectiveness</span><span class="metric-value">{health.gate_effectiveness}</span></div>
  </div>
</div>

<div class="card">
  <h2>Pain Timeline</h2>
  <svg width="100%" viewBox="0 0 600 90" preserveAspectRatio="none" style="max-height:100px">
    <polyline points="{sparkline_points}" fill="none" stroke="{color}" stroke-width="2"/>
  </svg>
</div>

<div class="grid">
  <div class="card">
    <h2>Agent Pain Levels</h2>
    {agent_bars}
  </div>
  <div class="card">
    <h2>Protective Reflexes</h2>
    {reflex_html}
  </div>
</div>

<div class="card">
  <h2>Pathologies</h2>
  {pathology_html}
</div>

<div class="card">
  <h2>Insights</h2>
  {insight_html}
</div>

</body></html>"""
    return html


# ---------------------------------------------------------------------------
# Demo Runner
# ---------------------------------------------------------------------------


def run_demo(num_agents: int = 6, ticks: int = 60,
             scenario: str = "baseline", seed: Optional[int] = 42,
             out_html: Optional[str] = None, out_json: Optional[str] = None) -> NociceptionReport:
    """Run a nociception demo with a predefined scenario."""
    engine = SwarmNociceptionEngine(num_agents=num_agents, seed=seed)

    scenario_data = SCENARIOS.get(scenario, SCENARIOS["baseline"])
    events = scenario_data["events"]

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  🔥 Swarm Nociception Engine — Demo                        ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Scenario: {scenario:<20} Agents: {num_agents:<5} Ticks: {ticks:<5}  ║")
    print(f"║  {scenario_data['description']:<56}  ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()

    for t in range(1, ticks + 1):
        # Apply scheduled events
        for ev in events:
            if ev["tick"] == t:
                agent_idx = ev["agent"] % num_agents
                aid = f"agent-{agent_idx}"
                engine.apply_stimulus(aid, ev["type"], ev["intensity"])
                print(f"  [tick {t:3d}] ⚡ {aid} ← {ev['type'].value} (intensity={ev['intensity']:.2f})")

        engine.do_tick()

    report = engine.get_report()

    # Print summary
    print()
    print(f"┌─────────────────────────────────────────┐")
    print(f"│  Health Score: {report.health.score:5.1f} / 100  [{report.health.tier.value}]")
    print(f"│  Total Signals: {len(report.all_signals):4d}   Reflexes: {len(report.all_reflexes):4d}")
    print(f"│  Pain Memories: {len(report.all_memories):4d}")
    print(f"│  Pathologies:   {', '.join(p.value for p in report.health.pathologies) or 'None'}")
    print(f"└─────────────────────────────────────────┘")

    if report.insights:
        print(f"\n  Insights:")
        for ins in report.insights:
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
        description="Swarm Nociception Engine — autonomous pain/damage signaling"
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
