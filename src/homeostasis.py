"""Swarm Homeostasis Controller — autonomous bio-inspired regulation for mBFT.

Maintains optimal swarm operating conditions through continuous PID-style
feedback loops.  Like a biological homeostatic system (thermoregulation,
blood pressure), the controller monitors vital signs, detects deviations
from setpoints, and autonomously adjusts effectors to restore balance.

Capabilities:

- **6 Vital Signs** — consensus latency, throughput, failure rate, agent
  utilization, opinion entropy, quorum margin — continuously monitored.
- **5 Effectors** — threshold adjustment, timeout multiplier, concurrency
  limit, retry delay, quorum size target — autonomously tuned.
- **PID Control Loops** — proportional, integral, derivative gains per
  vital with anti-windup and output clamping.
- **4 Operating Modes** — normal, stressed, emergency, recovery —
  with automatic transitions based on vital sign health.
- **Oscillation Detection** — dampens gains when effectors flip-flop.
- **Health Score** — composite 0-100 reflecting homeostatic balance.
- **Persistence** — JSON save/load for cross-session continuity.
- **Interactive HTML Dashboard** — gauges, sparklines, mode indicators.

Usage (Python API)::

    from src.homeostasis import HomeostasisController

    ctrl = HomeostasisController()

    # Feed vital readings after each consensus round
    ctrl.record_vitals({
        "consensus_latency": 1.2,
        "throughput": 8.5,
        "failure_rate": 0.1,
        "agent_utilization": 0.75,
        "opinion_entropy": 1.4,
        "quorum_margin": 0.25,
    })

    # Get recommended adjustments
    adjustments = ctrl.compute_adjustments()
    print(adjustments)
    # {'threshold_adjustment': -0.02, 'timeout_multiplier': 1.1, ...}

    # Check health
    report = ctrl.get_health()
    print(report.score, report.mode)

    # Persistence
    ctrl.save("homeostasis.json")
    ctrl = HomeostasisController.load("homeostasis.json")

    # Dashboard
    ctrl.export_html("homeostasis_dashboard.html")

CLI::

    python -m src.homeostasis                    # demo simulation
    python -m src.homeostasis --cycles 100       # longer sim
    python -m src.homeostasis --perturbation     # inject stress event
    python -m src.homeostasis --export html -o dash.html
    python -m src.homeostasis --json state.json
    python -m src.homeostasis --health
"""
from __future__ import annotations

import argparse
import json
import random
import time
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

EFFECTOR_NAMES = [
    "threshold_adjustment",
    "timeout_multiplier",
    "concurrency_limit",
    "retry_delay",
    "quorum_size_target",
]

# Default setpoints
DEFAULT_SETPOINTS: Dict[str, float] = {
    "consensus_latency": 1.0,
    "throughput": 10.0,
    "failure_rate": 0.05,
    "agent_utilization": 0.8,
    "opinion_entropy": 1.5,
    "quorum_margin": 0.3,
}

# Acceptable bands (fraction of setpoint deviation)
ACCEPTABLE_BAND: Dict[str, float] = {
    "consensus_latency": 0.5,
    "throughput": 0.4,
    "failure_rate": 0.1,
    "agent_utilization": 0.2,
    "opinion_entropy": 0.4,
    "quorum_margin": 0.15,
}

# Critical thresholds (absolute deviations that trigger emergency)
CRITICAL_THRESHOLDS: Dict[str, float] = {
    "consensus_latency": 5.0,
    "throughput": 1.0,
    "failure_rate": 0.7,
    "agent_utilization": 0.2,
    "opinion_entropy": 0.3,
    "quorum_margin": 0.0,
}

# Effector bounds
EFFECTOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "threshold_adjustment": (-0.5, 0.5),
    "timeout_multiplier": (0.5, 3.0),
    "concurrency_limit": (1.0, 20.0),
    "retry_delay": (0.1, 10.0),
    "quorum_size_target": (3.0, 50.0),
}

# Default effector values
DEFAULT_EFFECTORS: Dict[str, float] = {
    "threshold_adjustment": 0.0,
    "timeout_multiplier": 1.0,
    "concurrency_limit": 5.0,
    "retry_delay": 1.0,
    "quorum_size_target": 7.0,
}

# Vital-to-effector mapping (which effector each vital primarily influences)
VITAL_EFFECTOR_MAP: Dict[str, str] = {
    "consensus_latency": "timeout_multiplier",
    "throughput": "concurrency_limit",
    "failure_rate": "threshold_adjustment",
    "agent_utilization": "quorum_size_target",
    "opinion_entropy": "threshold_adjustment",
    "quorum_margin": "retry_delay",
}

# Vital importance weights (for health score)
VITAL_WEIGHTS: Dict[str, float] = {
    "consensus_latency": 0.2,
    "throughput": 0.2,
    "failure_rate": 0.25,
    "agent_utilization": 0.1,
    "opinion_entropy": 0.1,
    "quorum_margin": 0.15,
}

# Whether lower is better (True) or target is the setpoint
LOWER_IS_BETTER: Dict[str, bool] = {
    "consensus_latency": True,
    "throughput": False,
    "failure_rate": True,
    "agent_utilization": False,
    "opinion_entropy": False,
    "quorum_margin": False,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class VitalReading:
    """A single vital sign measurement."""
    name: str
    value: float
    timestamp: float
    error: float = 0.0  # deviation from setpoint
    in_band: bool = True
    critical: bool = False


@dataclass
class ControlLoop:
    """PID control loop state for one vital sign."""
    vital_name: str
    setpoint: float
    kp: float = 0.3
    ki: float = 0.05
    kd: float = 0.1
    integral: float = 0.0
    prev_error: float = 0.0
    output: float = 0.0
    dampened: bool = False
    integral_limit: float = 5.0  # anti-windup


@dataclass
class EffectorState:
    """Current state of an effector."""
    name: str
    value: float
    history: List[float] = field(default_factory=list)
    oscillation_count: int = 0


@dataclass
class HomeostasisSnapshot:
    """Point-in-time snapshot of the entire homeostatic system."""
    timestamp: float
    vitals: Dict[str, float]
    effectors: Dict[str, float]
    mode: str
    health_score: float
    errors: Dict[str, float]


@dataclass
class HealthReport:
    """Overall health assessment."""
    score: float  # 0-100
    mode: str
    vitals_in_band: int
    vitals_critical: int
    oscillating_effectors: List[str]
    recommendations: List[str]
    per_vital: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main Controller
# ---------------------------------------------------------------------------

class HomeostasisController:
    """Autonomous swarm homeostasis controller with PID feedback loops."""

    def __init__(
        self,
        setpoints: Optional[Dict[str, float]] = None,
        gains: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ):
        self.setpoints = dict(DEFAULT_SETPOINTS)
        if setpoints:
            self.setpoints.update(setpoints)

        # Initialize control loops
        self.loops: Dict[str, ControlLoop] = {}
        for vital in VITAL_NAMES:
            kp, ki, kd = 0.3, 0.05, 0.1
            if gains and vital in gains:
                kp, ki, kd = gains[vital]
            self.loops[vital] = ControlLoop(
                vital_name=vital,
                setpoint=self.setpoints[vital],
                kp=kp, ki=ki, kd=kd,
            )

        # Effector states
        self.effectors: Dict[str, EffectorState] = {}
        for eff in EFFECTOR_NAMES:
            self.effectors[eff] = EffectorState(
                name=eff, value=DEFAULT_EFFECTORS[eff]
            )

        # Vital reading history
        self.vital_history: Dict[str, List[VitalReading]] = {v: [] for v in VITAL_NAMES}
        self.snapshots: List[HomeostasisSnapshot] = []

        # Mode tracking
        self._mode: str = "normal"
        self._mode_since: float = time.time()
        self._stressed_count: int = 0
        self._recovery_steps: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_vitals(self, readings: Dict[str, float]) -> None:
        """Record a new set of vital sign measurements."""
        ts = time.time()
        for name, value in readings.items():
            if name not in VITAL_NAMES:
                continue
            setpoint = self.setpoints[name]
            error = self._compute_error(name, value, setpoint)
            in_band = abs(error) <= ACCEPTABLE_BAND[name]
            critical = self._is_critical(name, value)
            reading = VitalReading(
                name=name, value=value, timestamp=ts,
                error=error, in_band=in_band, critical=critical,
            )
            self.vital_history[name].append(reading)
            # Keep last 200 readings per vital
            if len(self.vital_history[name]) > 200:
                self.vital_history[name] = self.vital_history[name][-200:]

        self._update_mode()

    def compute_adjustments(self) -> Dict[str, float]:
        """Run all PID loops and return effector adjustments."""
        adjustments: Dict[str, float] = {}

        for vital_name, loop in self.loops.items():
            history = self.vital_history[vital_name]
            if not history:
                continue

            current = history[-1]
            error = current.error

            # PID computation
            # For "lower is better" vitals, positive error means too high
            if LOWER_IS_BETTER[vital_name]:
                error = current.value - loop.setpoint
            else:
                error = loop.setpoint - current.value

            # Proportional
            p_term = loop.kp * error

            # Integral (with anti-windup)
            loop.integral += error
            loop.integral = max(-loop.integral_limit, min(loop.integral_limit, loop.integral))
            i_term = loop.ki * loop.integral

            # Derivative
            d_term = loop.kd * (error - loop.prev_error)
            loop.prev_error = error

            # Combined output
            output = p_term + i_term + d_term

            # Apply mode multiplier
            if self._mode == "stressed":
                output *= 1.5
            elif self._mode == "emergency":
                output *= 2.0
            elif self._mode == "recovery":
                output *= 0.7

            # Apply dampening if oscillating
            if loop.dampened:
                output *= 0.5

            loop.output = output

            # Map to effector
            effector_name = VITAL_EFFECTOR_MAP[vital_name]
            if effector_name not in adjustments:
                adjustments[effector_name] = 0.0
            adjustments[effector_name] += output

        # Clamp effectors and update state
        for eff_name in adjustments:
            bounds = EFFECTOR_BOUNDS[eff_name]
            base = DEFAULT_EFFECTORS[eff_name]
            new_value = base + adjustments[eff_name]
            new_value = max(bounds[0], min(bounds[1], new_value))
            adjustments[eff_name] = new_value

            # Track history for oscillation detection
            eff = self.effectors[eff_name]
            eff.history.append(new_value)
            if len(eff.history) > 20:
                eff.history = eff.history[-20:]
            eff.value = new_value

            # Oscillation detection
            self._detect_oscillation(eff_name)

        # Record snapshot
        self._record_snapshot()

        return adjustments

    def get_health(self) -> HealthReport:
        """Compute current homeostatic health report."""
        vitals_in_band = 0
        vitals_critical = 0
        per_vital: Dict[str, Dict[str, Any]] = {}
        weighted_score = 0.0

        for name in VITAL_NAMES:
            history = self.vital_history[name]
            if not history:
                per_vital[name] = {"status": "no_data", "value": None, "score": 50.0}
                weighted_score += VITAL_WEIGHTS[name] * 50.0
                continue

            latest = history[-1]
            if latest.in_band:
                vitals_in_band += 1
            if latest.critical:
                vitals_critical += 1

            # Per-vital score: 100 when at setpoint, 0 when critical
            band = ACCEPTABLE_BAND[name]
            deviation = abs(latest.error)
            if band > 0:
                vital_score = max(0.0, 100.0 * (1.0 - deviation / (band * 3)))
            else:
                vital_score = 100.0 if deviation == 0 else 0.0

            status = "critical" if latest.critical else ("ok" if latest.in_band else "warning")
            per_vital[name] = {
                "status": status,
                "value": latest.value,
                "error": latest.error,
                "score": round(vital_score, 1),
            }
            weighted_score += VITAL_WEIGHTS[name] * vital_score

        # Oscillation warnings
        oscillating = [
            name for name, eff in self.effectors.items()
            if eff.oscillation_count >= 4
        ]

        # Recommendations
        recommendations = self._generate_recommendations(per_vital, oscillating)

        score = max(0.0, min(100.0, weighted_score))

        return HealthReport(
            score=round(score, 1),
            mode=self._mode,
            vitals_in_band=vitals_in_band,
            vitals_critical=vitals_critical,
            oscillating_effectors=oscillating,
            recommendations=recommendations,
            per_vital=per_vital,
        )

    def get_mode(self) -> str:
        """Return current operating mode."""
        return self._mode

    def get_history(self) -> List[HomeostasisSnapshot]:
        """Return all recorded snapshots."""
        return list(self.snapshots)

    def reset(self) -> None:
        """Reset integral terms, history, and mode."""
        for loop in self.loops.values():
            loop.integral = 0.0
            loop.prev_error = 0.0
            loop.output = 0.0
            loop.dampened = False
        for eff in self.effectors.values():
            eff.history.clear()
            eff.oscillation_count = 0
            eff.value = DEFAULT_EFFECTORS[eff.name]
        self.vital_history = {v: [] for v in VITAL_NAMES}
        self.snapshots.clear()
        self._mode = "normal"
        self._stressed_count = 0
        self._recovery_steps = 0

    def to_json(self) -> Dict[str, Any]:
        """Serialize controller state to JSON-compatible dict."""
        return {
            "setpoints": self.setpoints,
            "mode": self._mode,
            "loops": {
                name: {
                    "integral": loop.integral,
                    "prev_error": loop.prev_error,
                    "output": loop.output,
                    "dampened": loop.dampened,
                    "kp": loop.kp,
                    "ki": loop.ki,
                    "kd": loop.kd,
                }
                for name, loop in self.loops.items()
            },
            "effectors": {
                name: {
                    "value": eff.value,
                    "history": eff.history[-20:],
                    "oscillation_count": eff.oscillation_count,
                }
                for name, eff in self.effectors.items()
            },
            "snapshots": [asdict(s) for s in self.snapshots[-100:]],
            "vital_history": {
                name: [{"value": r.value, "error": r.error, "timestamp": r.timestamp}
                       for r in readings[-50:]]
                for name, readings in self.vital_history.items()
            },
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> HomeostasisController:
        """Restore controller from JSON dict."""
        ctrl = cls(setpoints=data.get("setpoints"))
        ctrl._mode = data.get("mode", "normal")

        for name, loop_data in data.get("loops", {}).items():
            if name in ctrl.loops:
                ctrl.loops[name].integral = loop_data.get("integral", 0.0)
                ctrl.loops[name].prev_error = loop_data.get("prev_error", 0.0)
                ctrl.loops[name].output = loop_data.get("output", 0.0)
                ctrl.loops[name].dampened = loop_data.get("dampened", False)
                ctrl.loops[name].kp = loop_data.get("kp", 0.3)
                ctrl.loops[name].ki = loop_data.get("ki", 0.05)
                ctrl.loops[name].kd = loop_data.get("kd", 0.1)

        for name, eff_data in data.get("effectors", {}).items():
            if name in ctrl.effectors:
                ctrl.effectors[name].value = eff_data.get("value", DEFAULT_EFFECTORS.get(name, 0))
                ctrl.effectors[name].history = eff_data.get("history", [])
                ctrl.effectors[name].oscillation_count = eff_data.get("oscillation_count", 0)

        return ctrl

    def save(self, path: str) -> None:
        """Save state to JSON file."""
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str) -> HomeostasisController:
        """Load state from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_json(data)

    def export_html(self, path: str) -> None:
        """Generate interactive HTML dashboard."""
        report = self.get_health()
        html = _generate_dashboard_html(self, report)
        Path(path).write_text(html, encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_error(self, name: str, value: float, setpoint: float) -> float:
        """Compute signed error. Positive = above setpoint."""
        if LOWER_IS_BETTER[name]:
            return value - setpoint  # positive means worse
        else:
            return setpoint - value  # positive means below target (worse)

    def _is_critical(self, name: str, value: float) -> bool:
        """Check if a vital reading is in critical territory."""
        threshold = CRITICAL_THRESHOLDS[name]
        if LOWER_IS_BETTER[name]:
            return value >= threshold
        else:
            return value <= threshold

    def _update_mode(self) -> None:
        """Update operating mode based on current vital readings."""
        out_of_band = 0
        any_critical = False

        for name in VITAL_NAMES:
            history = self.vital_history[name]
            if not history:
                continue
            latest = history[-1]
            if not latest.in_band:
                out_of_band += 1
            if latest.critical:
                any_critical = True

        if any_critical:
            if self._mode != "emergency":
                self._mode = "emergency"
                self._mode_since = time.time()
                self._apply_emergency_config()
        elif self._mode == "emergency":
            # Transition to recovery
            self._mode = "recovery"
            self._mode_since = time.time()
            self._recovery_steps = 0
        elif self._mode == "recovery":
            self._recovery_steps += 1
            if self._recovery_steps >= 5 and out_of_band <= 1:
                self._mode = "normal"
                self._mode_since = time.time()
        elif out_of_band >= 3:
            self._stressed_count += 1
            if self._stressed_count >= 2:
                self._mode = "stressed"
                self._mode_since = time.time()
        else:
            self._stressed_count = max(0, self._stressed_count - 1)
            if self._mode == "stressed" and out_of_band <= 1:
                self._mode = "normal"
                self._mode_since = time.time()

    def _apply_emergency_config(self) -> None:
        """Apply emergency preset configuration."""
        self.effectors["threshold_adjustment"].value = 0.3
        self.effectors["timeout_multiplier"].value = 2.0
        self.effectors["concurrency_limit"].value = 2.0
        self.effectors["retry_delay"].value = 3.0

    def _detect_oscillation(self, eff_name: str) -> None:
        """Detect oscillation in an effector's recent history."""
        eff = self.effectors[eff_name]
        history = eff.history
        if len(history) < 8:
            eff.oscillation_count = 0
            return

        # Count direction reversals in last 8 values
        recent = history[-8:]
        reversals = 0
        for i in range(2, len(recent)):
            d1 = recent[i - 1] - recent[i - 2]
            d2 = recent[i] - recent[i - 1]
            if d1 * d2 < 0:  # sign change
                reversals += 1

        eff.oscillation_count = reversals

        # Dampen associated loop if oscillating
        if reversals >= 4:
            for vital_name, mapped_eff in VITAL_EFFECTOR_MAP.items():
                if mapped_eff == eff_name:
                    self.loops[vital_name].dampened = True
        else:
            for vital_name, mapped_eff in VITAL_EFFECTOR_MAP.items():
                if mapped_eff == eff_name:
                    self.loops[vital_name].dampened = False

    def _record_snapshot(self) -> None:
        """Record a homeostasis snapshot."""
        vitals = {}
        errors = {}
        for name in VITAL_NAMES:
            history = self.vital_history[name]
            if history:
                vitals[name] = history[-1].value
                errors[name] = history[-1].error
            else:
                vitals[name] = 0.0
                errors[name] = 0.0

        effector_values = {name: eff.value for name, eff in self.effectors.items()}
        report = self.get_health()

        snapshot = HomeostasisSnapshot(
            timestamp=time.time(),
            vitals=vitals,
            effectors=effector_values,
            mode=self._mode,
            health_score=report.score,
            errors=errors,
        )
        self.snapshots.append(snapshot)
        if len(self.snapshots) > 500:
            self.snapshots = self.snapshots[-500:]

    def _generate_recommendations(
        self, per_vital: Dict[str, Dict[str, Any]], oscillating: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs: List[str] = []

        for name, info in per_vital.items():
            if info.get("status") == "critical":
                recs.append(f"CRITICAL: {name} at {info.get('value', '?')} — immediate attention required")
            elif info.get("status") == "warning":
                recs.append(f"WARNING: {name} drifting from setpoint (error={info.get('error', 0):.3f})")

        for eff in oscillating:
            recs.append(f"OSCILLATION: {eff} is flip-flopping — gains dampened automatically")

        if self._mode == "emergency":
            recs.append("EMERGENCY MODE active — conservative config applied, monitor for stabilization")
        elif self._mode == "stressed":
            recs.append("STRESSED: multiple vitals out of band — increased control effort active")

        if not recs:
            recs.append("All systems nominal — homeostasis maintained")

        return recs


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

def _generate_dashboard_html(ctrl: HomeostasisController, report: HealthReport) -> str:
    """Generate single-file interactive HTML dashboard."""
    score = report.score
    mode = report.mode
    mode_colors = {"normal": "#4caf50", "stressed": "#ff9800", "emergency": "#f44336", "recovery": "#2196f3"}
    mode_color = mode_colors.get(mode, "#666")

    # Build vital rows
    vital_rows = ""
    for name in VITAL_NAMES:
        info = report.per_vital.get(name, {})
        status = info.get("status", "no_data")
        value = info.get("value", "—")
        info.get("error", 0)
        v_score = info.get("score", 50)
        status_color = {"ok": "#4caf50", "warning": "#ff9800", "critical": "#f44336"}.get(status, "#999")
        sparkline_data = [r.value for r in ctrl.vital_history.get(name, [])[-30:]]
        sparkline_js = json.dumps(sparkline_data)
        vital_rows += f"""
        <tr>
            <td><strong>{name.replace('_', ' ').title()}</strong></td>
            <td>{value if value is not None else '—':.3f}</td>
            <td>{ctrl.setpoints[name]:.2f}</td>
            <td style="color:{status_color};font-weight:bold">{status.upper()}</td>
            <td>{v_score:.0f}/100</td>
            <td><canvas class="spark" data-values='{sparkline_js}' width="120" height="30"></canvas></td>
        </tr>"""

    # Build effector rows
    effector_rows = ""
    for name in EFFECTOR_NAMES:
        eff = ctrl.effectors[name]
        bounds = EFFECTOR_BOUNDS[name]
        osc = "⚠️" if eff.oscillation_count >= 4 else "✓"
        effector_rows += f"""
        <tr>
            <td><strong>{name.replace('_', ' ').title()}</strong></td>
            <td>{eff.value:.3f}</td>
            <td>[{bounds[0]}, {bounds[1]}]</td>
            <td>{osc}</td>
        </tr>"""

    # Recommendations
    rec_html = "".join(f"<li>{r}</li>" for r in report.recommendations)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Homeostasis Dashboard</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#1a1a2e; color:#e0e0e0; padding:24px; }}
.header {{ text-align:center; margin-bottom:32px; }}
.header h1 {{ font-size:2em; color:#fff; }}
.header .mode {{ display:inline-block; padding:6px 16px; border-radius:20px; background:{mode_color}; color:#fff; font-weight:bold; margin-top:8px; }}
.gauge-container {{ text-align:center; margin:24px 0; }}
.gauge {{ width:200px; height:120px; margin:0 auto; }}
.score-label {{ font-size:2.5em; font-weight:bold; color:#fff; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; max-width:1200px; margin:0 auto; }}
.card {{ background:#16213e; border-radius:12px; padding:20px; }}
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
    <h1>🧬 Swarm Homeostasis Controller</h1>
    <div class="mode">{mode.upper()}</div>
</div>

<div class="gauge-container">
    <svg class="gauge" viewBox="0 0 200 120">
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#333" stroke-width="12" stroke-linecap="round"/>
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#grad)" stroke-width="12" stroke-linecap="round"
              stroke-dasharray="{score * 2.51} 251" />
        <defs><linearGradient id="grad"><stop offset="0%" stop-color="#f44336"/><stop offset="50%" stop-color="#ff9800"/><stop offset="100%" stop-color="#4caf50"/></linearGradient></defs>
    </svg>
    <div class="score-label">{score:.0f}</div>
    <div style="color:#888">Homeostasis Score</div>
</div>

<div class="grid">
    <div class="card">
        <h2>Vital Signs</h2>
        <table>
            <tr><th>Vital</th><th>Value</th><th>Setpoint</th><th>Status</th><th>Score</th><th>Trend</th></tr>
            {vital_rows}
        </table>
    </div>
    <div class="card">
        <h2>Effectors</h2>
        <table>
            <tr><th>Effector</th><th>Value</th><th>Bounds</th><th>Stable</th></tr>
            {effector_rows}
        </table>
    </div>
    <div class="card">
        <h2>Recommendations</h2>
        <ul>{rec_html}</ul>
    </div>
    <div class="card">
        <h2>System Info</h2>
        <table>
            <tr><td>Mode</td><td style="color:{mode_color};font-weight:bold">{mode.upper()}</td></tr>
            <tr><td>Vitals In Band</td><td>{report.vitals_in_band}/{len(VITAL_NAMES)}</td></tr>
            <tr><td>Critical Vitals</td><td>{report.vitals_critical}</td></tr>
            <tr><td>Oscillating Effectors</td><td>{len(report.oscillating_effectors)}</td></tr>
            <tr><td>Snapshots Recorded</td><td>{len(ctrl.snapshots)}</td></tr>
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


# ---------------------------------------------------------------------------
# Simulation / CLI
# ---------------------------------------------------------------------------

def _simulate(cycles: int = 50, perturbation: bool = False) -> HomeostasisController:
    """Run a simulation demonstrating the controller stabilizing."""
    ctrl = HomeostasisController()

    print(f"🧬 Swarm Homeostasis Simulation ({cycles} cycles)")
    print("=" * 60)

    for i in range(cycles):
        # Base vital readings (near-setpoint with noise)
        latency = 1.0 + random.gauss(0, 0.2)
        throughput = 10.0 + random.gauss(0, 1.5)
        failure_rate = 0.05 + random.gauss(0, 0.02)
        utilization = 0.8 + random.gauss(0, 0.05)
        entropy = 1.5 + random.gauss(0, 0.2)
        margin = 0.3 + random.gauss(0, 0.05)

        # Inject perturbation midway
        if perturbation and 20 <= i < 35:
            latency += 2.0 + random.random()
            failure_rate += 0.3
            throughput -= 5.0
            utilization -= 0.3

        # Clamp to reasonable ranges
        latency = max(0.1, latency)
        throughput = max(0.0, throughput)
        failure_rate = max(0.0, min(1.0, failure_rate))
        utilization = max(0.0, min(1.0, utilization))
        entropy = max(0.0, entropy)
        margin = max(-0.5, margin)

        ctrl.record_vitals({
            "consensus_latency": latency,
            "throughput": throughput,
            "failure_rate": failure_rate,
            "agent_utilization": utilization,
            "opinion_entropy": entropy,
            "quorum_margin": margin,
        })

        ctrl.compute_adjustments()
        mode = ctrl.get_mode()

        if i % 10 == 0 or mode != "normal":
            health = ctrl.get_health()
            print(f"\n  Cycle {i:3d} | Mode: {mode:10s} | Health: {health.score:5.1f}")
            print(f"           | Latency: {latency:.2f}s | Throughput: {throughput:.1f}/min | Fail: {failure_rate:.2%}")

    print("\n" + "=" * 60)
    report = ctrl.get_health()
    print(f"\n  Final Health: {report.score:.1f}/100 | Mode: {report.mode}")
    print(f"  Vitals in band: {report.vitals_in_band}/{len(VITAL_NAMES)}")
    print(f"  Recommendations:")
    for r in report.recommendations:
        print(f"    • {r}")

    return ctrl


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Swarm Homeostasis Controller — autonomous PID-based regulation"
    )
    parser.add_argument("--cycles", type=int, default=50, help="Simulation cycles")
    parser.add_argument("--perturbation", action="store_true", help="Inject stress event")
    parser.add_argument("--export", choices=["html"], help="Export format")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--json", help="Export JSON state to file")
    parser.add_argument("--health", action="store_true", help="Print health report")
    parser.add_argument("--load", help="Load state from JSON file")

    args = parser.parse_args()

    if args.load:
        ctrl = HomeostasisController.load(args.load)
    else:
        ctrl = _simulate(cycles=args.cycles, perturbation=args.perturbation)

    if args.health:
        report = ctrl.get_health()
        print(f"\n{'='*40}")
        print(f"  Health Score: {report.score:.1f}/100")
        print(f"  Mode: {report.mode}")
        print(f"  Vitals OK: {report.vitals_in_band}/{len(VITAL_NAMES)}")
        for r in report.recommendations:
            print(f"  • {r}")

    if args.export == "html":
        out = args.output or "homeostasis_dashboard.html"
        ctrl.export_html(out)
        print(f"\n  📊 Dashboard exported to {out}")

    if args.json:
        ctrl.save(args.json)
        print(f"\n  💾 State saved to {args.json}")


if __name__ == "__main__":
    main()
