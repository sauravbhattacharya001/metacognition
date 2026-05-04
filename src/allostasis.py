"""Swarm Allostasis Engine — autonomous predictive regulation for mBFT.

Biologically inspired by **allostasis** ("stability through change"), this
engine complements the reactive homeostasis controller with **predictive
regulation**.  Instead of waiting for deviations and correcting via PID
loops, allostasis anticipates future states from contextual cues and
pre-adjusts effectors *before* disruptions materialise.

Real biological systems use allostasis constantly — the brain pre-releases
cortisol before a known stressor, the body pre-warms before waking, and
athletes' cardiovascular systems ramp up before the starting gun.  This
engine brings the same principle to swarm metacognition.

Capabilities:

- **Predictive Model Engine** — per-vital sliding-window linear regression.
  Forecasts future vital values N steps ahead.  Tracks prediction accuracy
  (MAE) and adjusts confidence dynamically.
- **Context Cue Detector** — learns cue→outcome associations from history
  via co-occurrence counting within a configurable time window.  Recognises
  situational patterns that precede disruptions.
- **Anticipatory Adjuster** — generates confidence-scaled pre-emptive
  effector adjustments when predictions or cues signal upcoming deviation.
- **Allostatic Load Tracker** — measures cumulative stress across 5
  dimensions: prediction_burden, adjustment_frequency, false_alarm_rate,
  recovery_debt, cue_saturation.  High load = system fatigue.
- **Adaptation Scheduler** — manages anticipatory-vs-reactive control mode
  transitions based on prediction reliability.
- **Health Scorer** — composite 0-100 score from prediction accuracy, load
  levels, anticipation success rate, false alarm rate, and adaptation balance.
- **Insight Generator** — autonomous pattern detection: chronic fatigue,
  prediction drift, cue obsolescence, load accumulation warnings, adaptation
  recommendations.

Usage (Python API)::

    from src.allostasis import SwarmAllostasisEngine

    engine = SwarmAllostasisEngine(num_agents=5)

    # Feed vital readings each cycle
    engine.record_vitals({
        "consensus_latency": 1.2,
        "throughput": 8.5,
        "failure_rate": 0.1,
        "agent_utilization": 0.75,
        "opinion_entropy": 1.4,
        "quorum_margin": 0.25,
    })
    engine.tick()

    report = engine.get_report()
    print(report.health.score, report.health.tier)

    insights = engine.get_insights()
    for i in insights:
        print(i.category, i.message)

    engine.export_html("allostasis_dashboard.html")
    engine.save("allostasis.json")
    engine = SwarmAllostasisEngine.load("allostasis.json")

CLI::

    python -m src.allostasis                          # demo with defaults
    python -m src.allostasis --cycles 100             # longer simulation
    python -m src.allostasis --agents 8               # more agents
    python -m src.allostasis --scenario volatile       # scenario preset
    python -m src.allostasis --export html -o dash.html
    python -m src.allostasis --json state.json
    python -m src.allostasis --health
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
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VITAL_NAMES = [
    "consensus_latency",
    "throughput",
    "failure_rate",
    "agent_utilization",
    "opinion_entropy",
    "quorum_margin",
]

# Default setpoints (same as homeostasis for compatibility)
DEFAULT_SETPOINTS: Dict[str, float] = {
    "consensus_latency": 1.0,
    "throughput": 10.0,
    "failure_rate": 0.05,
    "agent_utilization": 0.8,
    "opinion_entropy": 1.5,
    "quorum_margin": 0.3,
}

# Whether lower is better for each vital
LOWER_IS_BETTER: Dict[str, bool] = {
    "consensus_latency": True,
    "throughput": False,
    "failure_rate": True,
    "agent_utilization": False,
    "opinion_entropy": False,
    "quorum_margin": False,
}

# Vital-to-effector mapping
VITAL_EFFECTOR_MAP: Dict[str, str] = {
    "consensus_latency": "timeout_multiplier",
    "throughput": "concurrency_limit",
    "failure_rate": "threshold_adjustment",
    "agent_utilization": "quorum_size_target",
    "opinion_entropy": "threshold_adjustment",
    "quorum_margin": "retry_delay",
}

EFFECTOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "threshold_adjustment": (-0.5, 0.5),
    "timeout_multiplier": (0.5, 3.0),
    "concurrency_limit": (1.0, 20.0),
    "retry_delay": (0.1, 10.0),
    "quorum_size_target": (3.0, 50.0),
}

# Deviation thresholds (fraction of setpoint)
WARNING_BAND: Dict[str, float] = {
    "consensus_latency": 0.5,
    "throughput": 0.4,
    "failure_rate": 0.1,
    "agent_utilization": 0.2,
    "opinion_entropy": 0.4,
    "quorum_margin": 0.15,
}

# Load dimension weights for composite load score
LOAD_WEIGHTS: Dict[str, float] = {
    "prediction_burden": 0.20,
    "adjustment_frequency": 0.25,
    "false_alarm_rate": 0.25,
    "recovery_debt": 0.15,
    "cue_saturation": 0.15,
}

# Health tiers
HEALTH_TIERS = [
    (80, "OPTIMAL"),
    (60, "BALANCED"),
    (40, "STRAINED"),
    (20, "FATIGUED"),
    (0, "EXHAUSTED"),
]

# Adaptation modes
MODE_ANTICIPATORY = "anticipatory"
MODE_REACTIVE = "reactive"
MODE_MIXED = "mixed"

# Insight categories
INSIGHT_CHRONIC_FATIGUE = "chronic_fatigue"
INSIGHT_PREDICTION_DRIFT = "prediction_drift"
INSIGHT_CUE_OBSOLESCENCE = "cue_obsolescence"
INSIGHT_LOAD_WARNING = "load_warning"
INSIGHT_ADAPTATION_REC = "adaptation_recommendation"
INSIGHT_FALSE_ALARM = "false_alarm_pattern"
INSIGHT_ANTICIPATION_SUCCESS = "anticipation_success"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class VitalReading:
    """A single vital sign measurement."""
    name: str
    value: float
    timestamp: float
    predicted: Optional[float] = None
    prediction_error: Optional[float] = None


@dataclass
class PredictionModel:
    """Linear regression model for one vital sign."""
    vital_name: str
    slope: float = 0.0
    intercept: float = 0.0
    confidence: float = 0.5
    mae: float = 0.0
    predictions_made: int = 0
    correct_anticipations: int = 0
    false_alarms: int = 0


@dataclass
class ContextCue:
    """A learned cue→outcome association."""
    cue_vital: str
    outcome_vital: str
    cue_direction: str  # "rising" or "falling"
    outcome_direction: str  # "rising" or "falling"
    occurrences: int = 0
    strength: float = 0.0
    last_seen: float = 0.0


@dataclass
class AnticipatoryAdjustment:
    """A pre-emptive effector adjustment."""
    effector: str
    value: float
    reason: str
    confidence: float
    triggered_by: str  # "prediction" or "cue"
    timestamp: float = 0.0


@dataclass
class AllostasisLoad:
    """Cumulative allostatic load across 5 dimensions."""
    prediction_burden: float = 0.0
    adjustment_frequency: float = 0.0
    false_alarm_rate: float = 0.0
    recovery_debt: float = 0.0
    cue_saturation: float = 0.0

    @property
    def composite(self) -> float:
        """Weighted composite load 0-100."""
        vals = {
            "prediction_burden": self.prediction_burden,
            "adjustment_frequency": self.adjustment_frequency,
            "false_alarm_rate": self.false_alarm_rate,
            "recovery_debt": self.recovery_debt,
            "cue_saturation": self.cue_saturation,
        }
        return sum(vals[k] * LOAD_WEIGHTS[k] for k in LOAD_WEIGHTS) * 100


@dataclass
class HealthScore:
    """Composite health assessment."""
    score: float
    tier: str
    prediction_accuracy: float
    load_level: float
    anticipation_success_rate: float
    false_alarm_rate: float
    adaptation_balance: float
    mode: str


@dataclass
class Insight:
    """An autonomous insight."""
    category: str
    message: str
    severity: str  # "info", "warning", "critical"
    timestamp: float = 0.0


@dataclass
class AllostasisReport:
    """Full engine report."""
    health: HealthScore
    load: AllostasisLoad
    mode: str
    predictions: Dict[str, Optional[float]]
    active_cues: List[Dict[str, Any]]
    recent_adjustments: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    cycle_count: int
    per_vital: Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Linear regression helper
# ---------------------------------------------------------------------------

def _linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Simple OLS linear regression. Returns (slope, intercept)."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[-1] if ys else 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0, mean_y
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class SwarmAllostasisEngine:
    """Autonomous predictive regulation engine for mBFT swarms."""

    def __init__(
        self,
        num_agents: int = 5,
        history_window: int = 50,
        forecast_horizon: int = 5,
        setpoints: Optional[Dict[str, float]] = None,
    ) -> None:
        self.num_agents = num_agents
        self.history_window = history_window
        self.forecast_horizon = forecast_horizon
        self.setpoints = dict(setpoints or DEFAULT_SETPOINTS)

        # Vital history: name -> deque of VitalReading
        self.vital_history: Dict[str, deque] = {
            name: deque(maxlen=history_window) for name in VITAL_NAMES
        }

        # Prediction models
        self.models: Dict[str, PredictionModel] = {
            name: PredictionModel(vital_name=name) for name in VITAL_NAMES
        }

        # Context cues (cue_vital, outcome_vital) -> ContextCue
        self.cues: Dict[Tuple[str, str], ContextCue] = {}

        # Cue detection window
        self.cue_window: int = 5

        # Allostatic load
        self.load = AllostasisLoad()

        # Adaptation mode
        self.mode: str = MODE_MIXED

        # Adjustment history
        self.adjustment_history: List[AnticipatoryAdjustment] = []

        # Tracking
        self.cycle_count: int = 0
        self._current_time: float = 0.0
        self._mode_history: List[Tuple[float, str]] = []
        self._adjustment_count_window: deque = deque(maxlen=50)
        self._false_alarm_window: deque = deque(maxlen=50)
        self._anticipation_window: deque = deque(maxlen=50)

        # Snapshots for save/load
        self._snapshots: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------

    def record_vitals(self, vitals: Dict[str, float]) -> None:
        """Record a set of vital sign readings."""
        ts = self._current_time
        for name in VITAL_NAMES:
            if name not in vitals:
                continue
            val = vitals[name]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue

            # Get prediction from model before recording
            predicted = self._predict(name)
            pred_error = None
            if predicted is not None:
                pred_error = abs(val - predicted)

            reading = VitalReading(
                name=name,
                value=val,
                timestamp=ts,
                predicted=predicted,
                prediction_error=pred_error,
            )
            self.vital_history[name].append(reading)

    # -------------------------------------------------------------------
    # Tick (main processing cycle)
    # -------------------------------------------------------------------

    def tick(self) -> List[AnticipatoryAdjustment]:
        """Run one allostatic processing cycle. Returns adjustments."""
        self.cycle_count += 1
        self._current_time += 1.0

        # 1. Update prediction models
        self._update_models()

        # 2. Detect context cues
        self._detect_cues()

        # 3. Evaluate prediction accuracy & update confidence
        self._evaluate_predictions()

        # 4. Generate anticipatory adjustments
        adjustments = self._generate_adjustments()

        # 5. Update allostatic load
        self._update_load(adjustments)

        # 6. Update adaptation mode
        self._update_mode()

        # 7. Record snapshot
        self._record_snapshot()

        return adjustments

    # -------------------------------------------------------------------
    # Prediction Model Engine
    # -------------------------------------------------------------------

    def _predict(self, vital_name: str) -> Optional[float]:
        """Predict future value for a vital using linear regression."""
        history = self.vital_history[vital_name]
        if len(history) < 3:
            return None

        model = self.models[vital_name]
        xs = list(range(len(history)))
        ys = [r.value for r in history]
        slope, intercept = _linear_regression(xs, ys)
        model.slope = slope
        model.intercept = intercept

        # Forecast: extrapolate forecast_horizon steps
        future_x = len(history) + self.forecast_horizon
        predicted = slope * future_x + intercept
        model.predictions_made += 1
        return predicted

    def _update_models(self) -> None:
        """Refit all prediction models on current history."""
        for name in VITAL_NAMES:
            history = self.vital_history[name]
            if len(history) < 3:
                continue

            model = self.models[name]
            xs = list(range(len(history)))
            ys = [r.value for r in history]
            slope, intercept = _linear_regression(xs, ys)
            model.slope = slope
            model.intercept = intercept

            # Compute MAE over recent predictions
            errors = [r.prediction_error for r in history if r.prediction_error is not None]
            if errors:
                recent_errors = errors[-20:]
                model.mae = sum(recent_errors) / len(recent_errors)

    def _evaluate_predictions(self) -> None:
        """Evaluate prediction accuracy and update confidence."""
        for name in VITAL_NAMES:
            model = self.models[name]
            history = self.vital_history[name]
            if len(history) < 5:
                model.confidence = 0.3
                continue

            errors = [r.prediction_error for r in history if r.prediction_error is not None]
            if not errors:
                model.confidence = 0.3
                continue

            recent = errors[-10:]
            mae = sum(recent) / len(recent)
            model.mae = mae

            # Confidence: inversely proportional to MAE relative to setpoint
            sp = self.setpoints.get(name, 1.0)
            if sp == 0:
                sp = 1.0
            normalised_mae = mae / abs(sp)
            model.confidence = max(0.1, min(0.95, 1.0 - normalised_mae))

    def get_prediction(self, vital_name: str) -> Optional[float]:
        """Public accessor: predict future value for a vital."""
        return self._predict(vital_name)

    def get_predictions(self) -> Dict[str, Optional[float]]:
        """Get predictions for all vitals."""
        return {name: self._predict(name) for name in VITAL_NAMES}

    # -------------------------------------------------------------------
    # Context Cue Detector
    # -------------------------------------------------------------------

    def _detect_cues(self) -> None:
        """Detect co-occurrence patterns between vital changes."""
        if self.cycle_count < self.cue_window + 2:
            return

        for cue_vital in VITAL_NAMES:
            cue_hist = self.vital_history[cue_vital]
            if len(cue_hist) < self.cue_window + 1:
                continue

            # Check if cue_vital had a significant change cue_window ago
            old_val = cue_hist[-(self.cue_window + 1)].value
            mid_val = cue_hist[-self.cue_window].value
            sp = self.setpoints.get(cue_vital, 1.0)
            if sp == 0:
                sp = 1.0
            delta_cue = (mid_val - old_val) / abs(sp)
            if abs(delta_cue) < 0.1:
                continue
            cue_dir = "rising" if delta_cue > 0 else "falling"

            for outcome_vital in VITAL_NAMES:
                if outcome_vital == cue_vital:
                    continue
                out_hist = self.vital_history[outcome_vital]
                if len(out_hist) < 2:
                    continue

                # Check if outcome_vital deviated recently
                recent_val = out_hist[-1].value
                prev_val = out_hist[-2].value
                out_sp = self.setpoints.get(outcome_vital, 1.0)
                if out_sp == 0:
                    out_sp = 1.0
                delta_out = (recent_val - prev_val) / abs(out_sp)
                if abs(delta_out) < 0.05:
                    continue
                out_dir = "rising" if delta_out > 0 else "falling"

                key = (cue_vital, outcome_vital)
                if key not in self.cues:
                    self.cues[key] = ContextCue(
                        cue_vital=cue_vital,
                        outcome_vital=outcome_vital,
                        cue_direction=cue_dir,
                        outcome_direction=out_dir,
                    )
                cue = self.cues[key]
                cue.occurrences += 1
                cue.strength = min(1.0, cue.occurrences / 10.0)
                cue.last_seen = self._current_time
                cue.cue_direction = cue_dir
                cue.outcome_direction = out_dir

    def get_active_cues(self, min_strength: float = 0.2) -> List[ContextCue]:
        """Return cues above minimum strength."""
        return [c for c in self.cues.values() if c.strength >= min_strength]

    # -------------------------------------------------------------------
    # Anticipatory Adjuster
    # -------------------------------------------------------------------

    def _generate_adjustments(self) -> List[AnticipatoryAdjustment]:
        """Generate pre-emptive adjustments based on predictions and cues."""
        adjustments: List[AnticipatoryAdjustment] = []

        if self.mode == MODE_REACTIVE:
            return adjustments

        # Prediction-based adjustments
        for name in VITAL_NAMES:
            model = self.models[name]
            if model.confidence < 0.3:
                continue

            predicted = self._predict(name)
            if predicted is None:
                continue

            sp = self.setpoints[name]
            band = WARNING_BAND[name]
            deviation = predicted - sp

            # Check if predicted to leave acceptable band
            if LOWER_IS_BETTER[name]:
                # For lower-is-better, worry about predicted going too high
                if deviation > band * abs(sp if sp != 0 else 1.0):
                    effector = VITAL_EFFECTOR_MAP[name]
                    bounds = EFFECTOR_BOUNDS[effector]
                    adj_magnitude = -0.1 * model.confidence * min(1.0, abs(deviation) / max(abs(sp), 0.01))
                    adj_value = max(bounds[0], min(bounds[1], adj_magnitude))
                    adjustments.append(AnticipatoryAdjustment(
                        effector=effector,
                        value=adj_value,
                        reason=f"Predicted {name} rising to {predicted:.3f} (setpoint {sp:.2f})",
                        confidence=model.confidence,
                        triggered_by="prediction",
                        timestamp=self._current_time,
                    ))
                    self._anticipation_window.append(1)
            else:
                # For higher-is-better, worry about predicted going too low
                if deviation < -band * abs(sp if sp != 0 else 1.0):
                    effector = VITAL_EFFECTOR_MAP[name]
                    bounds = EFFECTOR_BOUNDS[effector]
                    adj_magnitude = 0.1 * model.confidence * min(1.0, abs(deviation) / max(abs(sp), 0.01))
                    adj_value = max(bounds[0], min(bounds[1], adj_magnitude))
                    adjustments.append(AnticipatoryAdjustment(
                        effector=effector,
                        value=adj_value,
                        reason=f"Predicted {name} dropping to {predicted:.3f} (setpoint {sp:.2f})",
                        confidence=model.confidence,
                        triggered_by="prediction",
                        timestamp=self._current_time,
                    ))
                    self._anticipation_window.append(1)

        # Cue-based adjustments
        for cue in self.get_active_cues(min_strength=0.3):
            outcome = cue.outcome_vital
            effector = VITAL_EFFECTOR_MAP.get(outcome)
            if not effector:
                continue

            bounds = EFFECTOR_BOUNDS[effector]
            sp = self.setpoints[outcome]
            # Determine adjustment direction based on whether outcome deviation is bad
            is_bad = (
                (cue.outcome_direction == "rising" and LOWER_IS_BETTER[outcome])
                or (cue.outcome_direction == "falling" and not LOWER_IS_BETTER[outcome])
            )
            if is_bad:
                adj_mag = -0.05 * cue.strength
            else:
                continue  # outcome direction is not harmful

            adj_value = max(bounds[0], min(bounds[1], adj_mag))
            adjustments.append(AnticipatoryAdjustment(
                effector=effector,
                value=adj_value,
                reason=f"Cue: {cue.cue_vital} {cue.cue_direction} → {outcome} {cue.outcome_direction}",
                confidence=cue.strength,
                triggered_by="cue",
                timestamp=self._current_time,
            ))

        self.adjustment_history.extend(adjustments)
        self._adjustment_count_window.append(len(adjustments))

        # Track false alarms: if we adjusted but vital stayed in band
        if self.cycle_count > 5:
            for name in VITAL_NAMES:
                hist = self.vital_history[name]
                if len(hist) < 2:
                    continue
                sp = self.setpoints[name]
                band = WARNING_BAND[name]
                val = hist[-1].value
                deviation = abs(val - sp) / max(abs(sp), 0.01)
                # If predicted deviation but actual is fine = false alarm
                pred = hist[-1].predicted
                if pred is not None:
                    pred_dev = abs(pred - sp) / max(abs(sp), 0.01)
                    if pred_dev > band and deviation < band * 0.5:
                        self._false_alarm_window.append(1)
                        self.models[name].false_alarms += 1
                    elif pred_dev > band and deviation > band * 0.5:
                        self.models[name].correct_anticipations += 1

        return adjustments

    # -------------------------------------------------------------------
    # Allostatic Load Tracker
    # -------------------------------------------------------------------

    def _update_load(self, adjustments: List[AnticipatoryAdjustment]) -> None:
        """Update cumulative allostatic load."""
        decay = 0.95  # slow decay each cycle

        # Prediction burden: higher if models are strained (high MAE)
        maes = [m.mae for m in self.models.values() if m.predictions_made > 0]
        if maes:
            avg_mae = sum(maes) / len(maes)
            self.load.prediction_burden = (
                self.load.prediction_burden * decay + avg_mae * 0.1
            )
        self.load.prediction_burden = min(1.0, self.load.prediction_burden)

        # Adjustment frequency: more adjustments = more load
        recent_adj = list(self._adjustment_count_window)[-10:]
        if recent_adj:
            freq = sum(recent_adj) / max(len(recent_adj), 1)
            self.load.adjustment_frequency = (
                self.load.adjustment_frequency * decay + freq * 0.05
            )
        self.load.adjustment_frequency = min(1.0, self.load.adjustment_frequency)

        # False alarm rate
        recent_fa = list(self._false_alarm_window)[-20:]
        total_preds = sum(m.predictions_made for m in self.models.values())
        if total_preds > 0:
            total_fa = sum(m.false_alarms for m in self.models.values())
            self.load.false_alarm_rate = min(1.0, total_fa / max(total_preds, 1))
        self.load.false_alarm_rate = min(1.0, self.load.false_alarm_rate)

        # Recovery debt: accumulates when load is high
        if self.load.composite > 50:
            self.load.recovery_debt = min(
                1.0, self.load.recovery_debt + 0.02
            )
        else:
            self.load.recovery_debt = max(0.0, self.load.recovery_debt * decay - 0.01)

        # Cue saturation: more active cues = higher saturation
        active = len(self.get_active_cues(min_strength=0.1))
        max_cues = len(VITAL_NAMES) * (len(VITAL_NAMES) - 1)
        self.load.cue_saturation = min(1.0, active / max(max_cues, 1))

    # -------------------------------------------------------------------
    # Adaptation Scheduler
    # -------------------------------------------------------------------

    def _update_mode(self) -> None:
        """Update the adaptation mode based on prediction reliability."""
        avg_confidence = statistics.mean(
            m.confidence for m in self.models.values()
        )

        old_mode = self.mode
        if avg_confidence >= 0.6 and self.load.composite < 60:
            self.mode = MODE_ANTICIPATORY
        elif avg_confidence < 0.3 or self.load.composite > 80:
            self.mode = MODE_REACTIVE
        else:
            self.mode = MODE_MIXED

        if self.mode != old_mode:
            self._mode_history.append((self._current_time, self.mode))

    def get_mode(self) -> str:
        """Get current adaptation mode."""
        return self.mode

    # -------------------------------------------------------------------
    # Health Scorer
    # -------------------------------------------------------------------

    def get_health(self) -> HealthScore:
        """Compute composite health score 0-100."""
        # 1. Prediction accuracy (0-100)
        confidences = [m.confidence for m in self.models.values()]
        prediction_accuracy = statistics.mean(confidences) * 100 if confidences else 50.0

        # 2. Load level (inverted: lower load = higher score)
        load_score = max(0, 100 - self.load.composite)

        # 3. Anticipation success rate
        total_correct = sum(m.correct_anticipations for m in self.models.values())
        total_attempts = total_correct + sum(m.false_alarms for m in self.models.values())
        if total_attempts > 0:
            anticipation_success = (total_correct / total_attempts) * 100
        else:
            anticipation_success = 50.0

        # 4. False alarm rate (inverted)
        false_alarm_score = max(0, 100 - self.load.false_alarm_rate * 100)

        # 5. Adaptation balance: mixed is ideal, extremes less so
        if self.mode == MODE_MIXED:
            adaptation_balance = 80.0
        elif self.mode == MODE_ANTICIPATORY:
            adaptation_balance = 70.0
        else:
            adaptation_balance = 50.0

        # Weighted composite
        score = (
            prediction_accuracy * 0.25
            + load_score * 0.25
            + anticipation_success * 0.20
            + false_alarm_score * 0.15
            + adaptation_balance * 0.15
        )
        score = max(0, min(100, score))

        # Determine tier
        tier = "EXHAUSTED"
        for threshold, tier_name in HEALTH_TIERS:
            if score >= threshold:
                tier = tier_name
                break

        return HealthScore(
            score=score,
            tier=tier,
            prediction_accuracy=prediction_accuracy,
            load_level=self.load.composite,
            anticipation_success_rate=anticipation_success,
            false_alarm_rate=self.load.false_alarm_rate * 100,
            adaptation_balance=adaptation_balance,
            mode=self.mode,
        )

    # -------------------------------------------------------------------
    # Insight Generator
    # -------------------------------------------------------------------

    def get_insights(self) -> List[Insight]:
        """Generate autonomous insights."""
        insights: List[Insight] = []
        ts = self._current_time

        # Chronic fatigue
        if self.load.composite > 70:
            insights.append(Insight(
                category=INSIGHT_CHRONIC_FATIGUE,
                message=f"Allostatic load critically high ({self.load.composite:.0f}%). "
                        f"Consider reducing anticipatory adjustments or increasing recovery time.",
                severity="critical",
                timestamp=ts,
            ))
        elif self.load.composite > 50:
            insights.append(Insight(
                category=INSIGHT_CHRONIC_FATIGUE,
                message=f"Allostatic load elevated ({self.load.composite:.0f}%). "
                        f"Monitor for fatigue accumulation.",
                severity="warning",
                timestamp=ts,
            ))

        # Prediction drift
        for name, model in self.models.items():
            sp = self.setpoints.get(name, 1.0)
            if sp == 0:
                sp = 1.0
            if model.mae > abs(sp) * 0.5 and model.predictions_made > 10:
                insights.append(Insight(
                    category=INSIGHT_PREDICTION_DRIFT,
                    message=f"Prediction model for {name} has high error "
                            f"(MAE={model.mae:.3f}, setpoint={sp:.2f}). "
                            f"Model may need retraining window adjustment.",
                    severity="warning",
                    timestamp=ts,
                ))

        # Cue obsolescence
        for key, cue in self.cues.items():
            age = self._current_time - cue.last_seen
            if age > 30 and cue.strength > 0.3:
                insights.append(Insight(
                    category=INSIGHT_CUE_OBSOLESCENCE,
                    message=f"Cue [{cue.cue_vital} → {cue.outcome_vital}] not seen "
                            f"in {age:.0f} cycles. May be obsolete.",
                    severity="info",
                    timestamp=ts,
                ))

        # Load warnings per dimension
        if self.load.adjustment_frequency > 0.7:
            insights.append(Insight(
                category=INSIGHT_LOAD_WARNING,
                message="Adjustment frequency is very high. "
                        "System may be over-anticipating.",
                severity="warning",
                timestamp=ts,
            ))

        if self.load.recovery_debt > 0.5:
            insights.append(Insight(
                category=INSIGHT_LOAD_WARNING,
                message=f"Recovery debt accumulating ({self.load.recovery_debt:.0%}). "
                        f"System needs a calm period to recover.",
                severity="warning",
                timestamp=ts,
            ))

        # False alarm pattern
        total_fa = sum(m.false_alarms for m in self.models.values())
        total_correct = sum(m.correct_anticipations for m in self.models.values())
        if total_fa > 5 and total_fa > total_correct:
            insights.append(Insight(
                category=INSIGHT_FALSE_ALARM,
                message=f"False alarms ({total_fa}) exceed correct anticipations "
                        f"({total_correct}). Consider reducing anticipatory sensitivity.",
                severity="warning",
                timestamp=ts,
            ))

        # Adaptation recommendation
        avg_conf = statistics.mean(m.confidence for m in self.models.values())
        if avg_conf > 0.7 and self.mode != MODE_ANTICIPATORY:
            insights.append(Insight(
                category=INSIGHT_ADAPTATION_REC,
                message="Prediction confidence is high. Could benefit from "
                        "more aggressive anticipatory control.",
                severity="info",
                timestamp=ts,
            ))
        elif avg_conf < 0.3 and self.mode != MODE_REACTIVE:
            insights.append(Insight(
                category=INSIGHT_ADAPTATION_REC,
                message="Prediction confidence is low. Recommend falling back "
                        "to reactive control until models stabilise.",
                severity="info",
                timestamp=ts,
            ))

        # Anticipation success
        if total_correct > 10:
            rate = total_correct / max(total_correct + total_fa, 1)
            if rate > 0.8:
                insights.append(Insight(
                    category=INSIGHT_ANTICIPATION_SUCCESS,
                    message=f"Anticipation success rate excellent ({rate:.0%}). "
                            f"Predictive models are well calibrated.",
                    severity="info",
                    timestamp=ts,
                ))

        return insights

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------

    def get_report(self) -> AllostasisReport:
        """Generate comprehensive report."""
        health = self.get_health()
        predictions = self.get_predictions()
        active_cues = [
            {
                "cue_vital": c.cue_vital,
                "outcome_vital": c.outcome_vital,
                "cue_direction": c.cue_direction,
                "outcome_direction": c.outcome_direction,
                "strength": c.strength,
                "occurrences": c.occurrences,
            }
            for c in self.get_active_cues()
        ]
        recent_adj = [
            {
                "effector": a.effector,
                "value": a.value,
                "reason": a.reason,
                "confidence": a.confidence,
                "triggered_by": a.triggered_by,
            }
            for a in self.adjustment_history[-10:]
        ]
        insights = [
            {"category": i.category, "message": i.message, "severity": i.severity}
            for i in self.get_insights()
        ]
        per_vital: Dict[str, Dict[str, Any]] = {}
        for name in VITAL_NAMES:
            hist = self.vital_history[name]
            model = self.models[name]
            per_vital[name] = {
                "current": hist[-1].value if hist else None,
                "predicted": predictions.get(name),
                "confidence": model.confidence,
                "mae": model.mae,
                "slope": model.slope,
                "readings": len(hist),
            }

        return AllostasisReport(
            health=health,
            load=self.load,
            mode=self.mode,
            predictions=predictions,
            active_cues=active_cues,
            recent_adjustments=recent_adj,
            insights=insights,
            cycle_count=self.cycle_count,
            per_vital=per_vital,
        )

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save engine state to JSON."""
        state: Dict[str, Any] = {
            "num_agents": self.num_agents,
            "history_window": self.history_window,
            "forecast_horizon": self.forecast_horizon,
            "setpoints": self.setpoints,
            "cycle_count": self.cycle_count,
            "current_time": self._current_time,
            "mode": self.mode,
            "load": asdict(self.load),
            "models": {
                name: asdict(m) for name, m in self.models.items()
            },
            "vital_history": {
                name: [asdict(r) for r in readings]
                for name, readings in self.vital_history.items()
            },
            "cues": {
                f"{k[0]}|{k[1]}": asdict(v) for k, v in self.cues.items()
            },
            "mode_history": self._mode_history,
        }
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: str) -> "SwarmAllostasisEngine":
        """Load engine state from JSON."""
        data = json.loads(Path(path).read_text())
        engine = cls(
            num_agents=data.get("num_agents", 5),
            history_window=data.get("history_window", 50),
            forecast_horizon=data.get("forecast_horizon", 5),
            setpoints=data.get("setpoints"),
        )
        engine.cycle_count = data.get("cycle_count", 0)
        engine._current_time = data.get("current_time", 0)
        engine.mode = data.get("mode", MODE_MIXED)

        # Restore load
        load_data = data.get("load", {})
        engine.load = AllostasisLoad(**load_data)

        # Restore models
        for name, mdata in data.get("models", {}).items():
            if name in engine.models:
                engine.models[name] = PredictionModel(**mdata)

        # Restore vital history
        for name, readings in data.get("vital_history", {}).items():
            if name in engine.vital_history:
                for r in readings:
                    engine.vital_history[name].append(VitalReading(**r))

        # Restore cues
        for key_str, cdata in data.get("cues", {}).items():
            parts = key_str.split("|")
            if len(parts) == 2:
                key = (parts[0], parts[1])
                engine.cues[key] = ContextCue(**cdata)

        engine._mode_history = data.get("mode_history", [])
        return engine

    # -------------------------------------------------------------------
    # HTML Dashboard
    # -------------------------------------------------------------------

    def export_html(self, path: str) -> None:
        """Export interactive HTML dashboard."""
        html = self._render_html()
        Path(path).write_text(html, encoding="utf-8")

    def _render_html(self) -> str:
        """Render the HTML dashboard string."""
        report = self.get_report()
        health = report.health
        score = health.score
        tier = health.tier

        tier_colors = {
            "OPTIMAL": "#4caf50",
            "BALANCED": "#8bc34a",
            "STRAINED": "#ff9800",
            "FATIGUED": "#ff5722",
            "EXHAUSTED": "#f44336",
        }
        tier_color = tier_colors.get(tier, "#666")

        # Vital rows
        vital_rows = ""
        for name in VITAL_NAMES:
            info = report.per_vital.get(name, {})
            current = info.get("current")
            predicted = info.get("predicted")
            confidence = info.get("confidence", 0)
            mae = info.get("mae", 0)
            slope_val = info.get("slope", 0)
            trend = "↑" if slope_val > 0.01 else ("↓" if slope_val < -0.01 else "→")
            sparkline_data = [r.value for r in self.vital_history.get(name, [])][-30:]
            sparkline_js = json.dumps(sparkline_data)
            conf_pct = confidence * 100

            cur_str = f"{current:.3f}" if current is not None else "—"
            pred_str = f"{predicted:.3f}" if predicted is not None else "—"

            vital_rows += f"""
            <tr>
                <td><strong>{html_mod.escape(name.replace('_', ' ').title())}</strong></td>
                <td>{cur_str}</td>
                <td>{pred_str}</td>
                <td>{trend}</td>
                <td>{conf_pct:.0f}%</td>
                <td>{mae:.4f}</td>
                <td><canvas class="spark" data-values='{sparkline_js}' width="120" height="30"></canvas></td>
            </tr>"""

        # Load dimensions
        load_rows = ""
        for dim in ["prediction_burden", "adjustment_frequency", "false_alarm_rate",
                     "recovery_debt", "cue_saturation"]:
            val = getattr(report.load, dim, 0)
            pct = val * 100
            bar_color = "#4caf50" if pct < 30 else ("#ff9800" if pct < 60 else "#f44336")
            load_rows += f"""
            <tr>
                <td>{html_mod.escape(dim.replace('_', ' ').title())}</td>
                <td>{pct:.1f}%</td>
                <td><div style="background:#333;border-radius:4px;height:16px;width:120px">
                    <div style="background:{bar_color};height:16px;border-radius:4px;width:{min(pct, 100):.0f}%"></div>
                </div></td>
            </tr>"""

        # Cue rows
        cue_rows = ""
        for c in report.active_cues[:10]:
            cue_rows += f"""
            <tr>
                <td>{html_mod.escape(c['cue_vital'])}</td>
                <td>{html_mod.escape(c['cue_direction'])}</td>
                <td>{html_mod.escape(c['outcome_vital'])}</td>
                <td>{html_mod.escape(c['outcome_direction'])}</td>
                <td>{c['strength']:.2f}</td>
                <td>{c['occurrences']}</td>
            </tr>"""

        # Insight rows
        insight_html = ""
        sev_colors = {"info": "#4fc3f7", "warning": "#ff9800", "critical": "#f44336"}
        for ins in report.insights:
            col = sev_colors.get(ins["severity"], "#888")
            insight_html += (
                f'<li style="color:{col}">'
                f'<strong>[{html_mod.escape(ins["severity"].upper())}]</strong> '
                f'{html_mod.escape(ins["message"])}</li>'
            )

        # Recent adjustments
        adj_html = ""
        for a in report.recent_adjustments:
            adj_html += (
                f'<li><strong>{html_mod.escape(a["effector"])}</strong>: '
                f'{a["value"]:.4f} ({html_mod.escape(a["triggered_by"])}) — '
                f'{html_mod.escape(a["reason"])}</li>'
            )

        mode_colors = {
            MODE_ANTICIPATORY: "#4caf50",
            MODE_MIXED: "#ff9800",
            MODE_REACTIVE: "#2196f3",
        }
        mode_color = mode_colors.get(report.mode, "#666")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Allostasis Dashboard</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#1a1a2e; color:#e0e0e0; padding:24px; }}
.header {{ text-align:center; margin-bottom:32px; }}
.header h1 {{ font-size:2em; color:#fff; }}
.header .mode {{ display:inline-block; padding:6px 16px; border-radius:20px; background:{mode_color}; color:#fff; font-weight:bold; margin-top:8px; }}
.header .tier {{ display:inline-block; padding:6px 16px; border-radius:20px; background:{tier_color}; color:#fff; font-weight:bold; margin:8px 4px; }}
.gauge-container {{ text-align:center; margin:24px 0; }}
.score-label {{ font-size:2.5em; font-weight:bold; color:#fff; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; max-width:1400px; margin:0 auto; }}
.card {{ background:#16213e; border-radius:12px; padding:20px; }}
.card.full {{ grid-column:1/-1; }}
.card h2 {{ color:#4fc3f7; margin-bottom:12px; font-size:1.2em; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #2a2a4a; }}
th {{ color:#888; font-size:0.85em; text-transform:uppercase; }}
ul {{ list-style:none; }}
ul li {{ padding:6px 0; border-bottom:1px solid #2a2a4a; }}
ul li:last-child {{ border-bottom:none; }}
.spark {{ display:block; }}
</style>
</head>
<body>
<div class="header">
    <h1>🔮 Swarm Allostasis Engine</h1>
    <div class="mode">{html_mod.escape(report.mode.upper())}</div>
    <div class="tier">{html_mod.escape(tier)}</div>
</div>

<div class="gauge-container">
    <svg viewBox="0 0 200 120" width="200" height="120">
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#333" stroke-width="12" stroke-linecap="round"/>
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#grad)" stroke-width="12" stroke-linecap="round"
              stroke-dasharray="{score * 2.51} 251" />
        <defs><linearGradient id="grad"><stop offset="0%" stop-color="#f44336"/><stop offset="50%" stop-color="#ff9800"/><stop offset="100%" stop-color="#4caf50"/></linearGradient></defs>
    </svg>
    <div class="score-label">{score:.0f}</div>
    <div style="color:#888">Allostasis Score</div>
</div>

<div class="grid">
    <div class="card full">
        <h2>📊 Vital Predictions</h2>
        <table>
            <tr><th>Vital</th><th>Current</th><th>Predicted</th><th>Trend</th><th>Confidence</th><th>MAE</th><th>History</th></tr>
            {vital_rows}
        </table>
    </div>
    <div class="card">
        <h2>⚡ Allostatic Load</h2>
        <table>
            <tr><th>Dimension</th><th>Level</th><th>Bar</th></tr>
            {load_rows}
        </table>
        <p style="margin-top:12px;color:#888">Composite Load: {report.load.composite:.1f}%</p>
    </div>
    <div class="card">
        <h2>🔗 Context Cues</h2>
        <table>
            <tr><th>Cue Vital</th><th>Dir</th><th>Outcome</th><th>Dir</th><th>Strength</th><th>Count</th></tr>
            {cue_rows if cue_rows else '<tr><td colspan="6" style="color:#666">No active cues yet</td></tr>'}
        </table>
    </div>
    <div class="card">
        <h2>🎯 Recent Adjustments</h2>
        <ul>{adj_html if adj_html else '<li style="color:#666">No recent adjustments</li>'}</ul>
    </div>
    <div class="card">
        <h2>💡 Insights</h2>
        <ul>{insight_html if insight_html else '<li style="color:#666">No insights yet</li>'}</ul>
    </div>
    <div class="card full">
        <h2>ℹ️ System Info</h2>
        <table>
            <tr><td>Mode</td><td style="color:{mode_color};font-weight:bold">{html_mod.escape(report.mode.upper())}</td></tr>
            <tr><td>Health Tier</td><td style="color:{tier_color};font-weight:bold">{html_mod.escape(tier)}</td></tr>
            <tr><td>Prediction Accuracy</td><td>{health.prediction_accuracy:.1f}%</td></tr>
            <tr><td>Anticipation Success</td><td>{health.anticipation_success_rate:.1f}%</td></tr>
            <tr><td>False Alarm Rate</td><td>{health.false_alarm_rate:.1f}%</td></tr>
            <tr><td>Cycles</td><td>{report.cycle_count}</td></tr>
            <tr><td>Active Cues</td><td>{len(report.active_cues)}</td></tr>
        </table>
    </div>
</div>

<script>
document.querySelectorAll('.spark').forEach(canvas => {{
    const ctx = canvas.getContext('2d');
    const values = JSON.parse(canvas.dataset.values || '[]');
    if (!values.length) return;
    const w = canvas.width, h = canvas.height;
    const min = Math.min(...values), max = Math.max(...values);
    const range = max - min || 1;
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    values.forEach((v, i) => {{
        const x = (i / (values.length - 1)) * w;
        const y = h - ((v - min) / range) * (h - 4) - 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.stroke();
}});
</script>
</body>
</html>"""
        return html

    # -------------------------------------------------------------------
    # Snapshot tracking
    # -------------------------------------------------------------------

    def _record_snapshot(self) -> None:
        """Record a lightweight snapshot."""
        self._snapshots.append({
            "cycle": self.cycle_count,
            "mode": self.mode,
            "load": self.load.composite,
            "health": self.get_health().score,
        })
        # Keep bounded
        if len(self._snapshots) > self.history_window * 2:
            self._snapshots = self._snapshots[-self.history_window:]


# ---------------------------------------------------------------------------
# Simulation / CLI
# ---------------------------------------------------------------------------

SCENARIOS = {
    "calm": {
        "desc": "Stable environment — low noise, no disruptions",
        "noise": 0.05,
        "disruption": None,
    },
    "volatile": {
        "desc": "High-variance environment — large swings",
        "noise": 0.4,
        "disruption": None,
    },
    "chronic_stress": {
        "desc": "Sustained elevated latency and failure rate",
        "noise": 0.1,
        "disruption": {"start": 10, "end": 80, "latency_add": 1.5, "fail_add": 0.2},
    },
    "recovery": {
        "desc": "Stress burst followed by gradual recovery",
        "noise": 0.1,
        "disruption": {"start": 15, "end": 35, "latency_add": 3.0, "fail_add": 0.4},
    },
    "cue_rich": {
        "desc": "Repeating patterns with detectable cues",
        "noise": 0.08,
        "disruption": "periodic",
    },
}


def _simulate(
    cycles: int = 60,
    num_agents: int = 5,
    scenario: str = "calm",
) -> SwarmAllostasisEngine:
    """Run a demonstration simulation."""
    cfg = SCENARIOS.get(scenario, SCENARIOS["calm"])
    engine = SwarmAllostasisEngine(num_agents=num_agents)
    noise = cfg["noise"]
    disruption = cfg["disruption"]

    print(f"🔮 Swarm Allostasis Simulation — {scenario}")
    print(f"   {cfg['desc']}")
    print("=" * 60)

    for i in range(cycles):
        # Base readings near setpoints
        latency = 1.0 + random.gauss(0, noise)
        throughput = 10.0 + random.gauss(0, noise * 5)
        failure_rate = 0.05 + random.gauss(0, noise * 0.2)
        utilization = 0.8 + random.gauss(0, noise * 0.3)
        entropy = 1.5 + random.gauss(0, noise)
        margin = 0.3 + random.gauss(0, noise * 0.3)

        # Apply disruption
        if isinstance(disruption, dict):
            s, e = disruption["start"], disruption["end"]
            if s <= i < e:
                latency += disruption.get("latency_add", 0)
                failure_rate += disruption.get("fail_add", 0)
                throughput -= disruption.get("latency_add", 0) * 2
        elif disruption == "periodic":
            # Periodic bursts every 15 cycles
            if i % 15 < 5:
                latency += 1.5
                failure_rate += 0.15

        # Clamp
        latency = max(0.1, latency)
        throughput = max(0.0, throughput)
        failure_rate = max(0.0, min(1.0, failure_rate))
        utilization = max(0.0, min(1.0, utilization))
        entropy = max(0.0, entropy)
        margin = max(-0.5, margin)

        engine.record_vitals({
            "consensus_latency": latency,
            "throughput": throughput,
            "failure_rate": failure_rate,
            "agent_utilization": utilization,
            "opinion_entropy": entropy,
            "quorum_margin": margin,
        })

        adjustments = engine.tick()

        if i % 15 == 0 or adjustments:
            h = engine.get_health()
            mode = engine.get_mode()
            adj_str = f" | Adjustments: {len(adjustments)}" if adjustments else ""
            print(
                f"  Cycle {i:3d} | Mode: {mode:13s} | Health: {h.score:5.1f} "
                f"| Tier: {h.tier:10s} | Load: {engine.load.composite:5.1f}%{adj_str}"
            )

    print("\n" + "=" * 60)
    health = engine.get_health()
    print(f"\n  Final Health: {health.score:.1f}/100 | Tier: {health.tier}")
    print(f"  Mode: {health.mode}")
    print(f"  Prediction Accuracy: {health.prediction_accuracy:.1f}%")
    print(f"  Anticipation Success: {health.anticipation_success_rate:.1f}%")
    print(f"  Allostatic Load: {engine.load.composite:.1f}%")

    insights = engine.get_insights()
    if insights:
        print(f"\n  Insights ({len(insights)}):")
        for ins in insights:
            icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(ins.severity, "•")
            print(f"    {icon} [{ins.category}] {ins.message}")

    return engine


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Swarm Allostasis Engine — autonomous predictive regulation"
    )
    parser.add_argument("--cycles", type=int, default=60, help="Simulation cycles")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="calm",
        help="Scenario preset",
    )
    parser.add_argument("--export", choices=["html"], help="Export format")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--json", help="Export JSON state to file")
    parser.add_argument("--health", action="store_true", help="Print health report")
    parser.add_argument("--load-state", help="Load state from JSON file")

    args = parser.parse_args()

    if args.load_state:
        engine = SwarmAllostasisEngine.load(args.load_state)
    else:
        engine = _simulate(
            cycles=args.cycles,
            num_agents=args.agents,
            scenario=args.scenario,
        )

    if args.health:
        h = engine.get_health()
        print(f"\n{'='*40}")
        print(f"  Health Score: {h.score:.1f}/100")
        print(f"  Tier: {h.tier}")
        print(f"  Mode: {h.mode}")
        print(f"  Load: {engine.load.composite:.1f}%")
        for ins in engine.get_insights():
            print(f"  • [{ins.category}] {ins.message}")

    if args.export == "html":
        out = args.output or "allostasis_dashboard.html"
        engine.export_html(out)
        print(f"\n  📊 Dashboard exported to {out}")

    if args.json:
        engine.save(args.json)
        print(f"\n  💾 State saved to {args.json}")


if __name__ == "__main__":
    main()
