"""Tests for Swarm Homeostasis Controller."""
from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

import pytest

from src.homeostasis import (
    ACCEPTABLE_BAND,
    CRITICAL_THRESHOLDS,
    DEFAULT_EFFECTORS,
    DEFAULT_SETPOINTS,
    EFFECTOR_BOUNDS,
    EFFECTOR_NAMES,
    LOWER_IS_BETTER,
    VITAL_NAMES,
    VITAL_WEIGHTS,
    ControlLoop,
    EffectorState,
    HealthReport,
    HomeostasisController,
    HomeostasisSnapshot,
    VitalReading,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nominal_vitals() -> dict:
    """Return vitals at perfect setpoints."""
    return dict(DEFAULT_SETPOINTS)


def _stressed_vitals() -> dict:
    """Return vitals that are out of band but not critical."""
    return {
        "consensus_latency": 2.5,
        "throughput": 4.0,
        "failure_rate": 0.25,
        "agent_utilization": 0.4,
        "opinion_entropy": 0.8,
        "quorum_margin": 0.1,
    }


def _critical_vitals() -> dict:
    """Return vitals with at least one critical reading."""
    return {
        "consensus_latency": 6.0,  # critical threshold is 5.0
        "throughput": 0.5,         # critical threshold is 1.0
        "failure_rate": 0.8,       # critical threshold is 0.7
        "agent_utilization": 0.1,  # critical threshold is 0.2
        "opinion_entropy": 0.2,    # critical threshold is 0.3
        "quorum_margin": -0.1,     # critical threshold is 0.0
    }


# ---------------------------------------------------------------------------
# Basic recording and retrieval
# ---------------------------------------------------------------------------

class TestBasicRecording:
    def test_record_vitals_stores_history(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_nominal_vitals())
        for name in VITAL_NAMES:
            assert len(ctrl.vital_history[name]) == 1

    def test_multiple_recordings(self):
        ctrl = HomeostasisController()
        for _ in range(5):
            ctrl.record_vitals(_nominal_vitals())
        for name in VITAL_NAMES:
            assert len(ctrl.vital_history[name]) == 5

    def test_history_capped_at_200(self):
        ctrl = HomeostasisController()
        for _ in range(250):
            ctrl.record_vitals(_nominal_vitals())
        for name in VITAL_NAMES:
            assert len(ctrl.vital_history[name]) <= 200

    def test_unknown_vitals_ignored(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals({"unknown_vital": 99.0, "consensus_latency": 1.0})
        assert "unknown_vital" not in ctrl.vital_history
        assert len(ctrl.vital_history["consensus_latency"]) == 1

    def test_vital_reading_error_computed(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals({"consensus_latency": 2.0})  # setpoint is 1.0
        reading = ctrl.vital_history["consensus_latency"][-1]
        assert reading.error == pytest.approx(1.0, abs=0.01)

    def test_vital_in_band_flag(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals({"consensus_latency": 1.1})  # band is 0.5
        assert ctrl.vital_history["consensus_latency"][-1].in_band is True
        ctrl.record_vitals({"consensus_latency": 3.0})  # way out
        assert ctrl.vital_history["consensus_latency"][-1].in_band is False

    def test_vital_critical_flag(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals({"failure_rate": 0.8})  # threshold 0.7
        assert ctrl.vital_history["failure_rate"][-1].critical is True
        ctrl.record_vitals({"failure_rate": 0.1})
        assert ctrl.vital_history["failure_rate"][-1].critical is False


# ---------------------------------------------------------------------------
# PID computation
# ---------------------------------------------------------------------------

class TestPIDComputation:
    def test_compute_adjustments_returns_effectors(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_nominal_vitals())
        adj = ctrl.compute_adjustments()
        assert isinstance(adj, dict)
        for name in adj:
            assert name in EFFECTOR_NAMES

    def test_nominal_vitals_produce_near_default_effectors(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_nominal_vitals())
        adj = ctrl.compute_adjustments()
        # At setpoint, error ~0, adjustments should be near defaults
        for name, val in adj.items():
            default = DEFAULT_EFFECTORS[name]
            assert abs(val - default) < 1.0, f"{name}: {val} far from {default}"

    def test_high_latency_increases_timeout(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals({"consensus_latency": 3.0})
        adj = ctrl.compute_adjustments()
        # High latency → increase timeout multiplier
        assert adj["timeout_multiplier"] > DEFAULT_EFFECTORS["timeout_multiplier"]

    def test_high_failure_rate_adjusts_threshold(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals({"failure_rate": 0.4})
        adj = ctrl.compute_adjustments()
        # High failure rate → threshold adjustment changes
        assert adj["threshold_adjustment"] != DEFAULT_EFFECTORS["threshold_adjustment"]

    def test_effectors_clamped_to_bounds(self):
        ctrl = HomeostasisController()
        # Extreme values
        ctrl.record_vitals({
            "consensus_latency": 100.0,
            "throughput": 0.0,
            "failure_rate": 1.0,
            "agent_utilization": 0.0,
            "opinion_entropy": 0.0,
            "quorum_margin": -1.0,
        })
        adj = ctrl.compute_adjustments()
        for name, val in adj.items():
            lo, hi = EFFECTOR_BOUNDS[name]
            assert lo <= val <= hi, f"{name}={val} outside [{lo},{hi}]"

    def test_integral_accumulates(self):
        ctrl = HomeostasisController()
        # Record same error multiple times
        for _ in range(10):
            ctrl.record_vitals({"consensus_latency": 3.0})
            ctrl.compute_adjustments()
        loop = ctrl.loops["consensus_latency"]
        assert loop.integral != 0.0

    def test_anti_windup_limits_integral(self):
        ctrl = HomeostasisController()
        for _ in range(1000):
            ctrl.record_vitals({"consensus_latency": 100.0})
            ctrl.compute_adjustments()
        loop = ctrl.loops["consensus_latency"]
        assert abs(loop.integral) <= loop.integral_limit

    def test_derivative_responds_to_change(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals({"consensus_latency": 1.0})
        ctrl.compute_adjustments()
        ctrl.record_vitals({"consensus_latency": 3.0})  # sudden spike
        adj = ctrl.compute_adjustments()
        # Should produce stronger response due to derivative
        assert adj["timeout_multiplier"] > DEFAULT_EFFECTORS["timeout_multiplier"]


# ---------------------------------------------------------------------------
# Mode transitions
# ---------------------------------------------------------------------------

class TestModeTransitions:
    def test_starts_in_normal_mode(self):
        ctrl = HomeostasisController()
        assert ctrl.get_mode() == "normal"

    def test_transitions_to_stressed(self):
        ctrl = HomeostasisController()
        # Need 3+ out of band and 2+ consecutive recordings
        for _ in range(3):
            ctrl.record_vitals(_stressed_vitals())
        assert ctrl.get_mode() == "stressed"

    def test_transitions_to_emergency(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_critical_vitals())
        assert ctrl.get_mode() == "emergency"

    def test_emergency_to_recovery(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_critical_vitals())
        assert ctrl.get_mode() == "emergency"
        # Record non-critical readings
        ctrl.record_vitals(_stressed_vitals())
        assert ctrl.get_mode() == "recovery"

    def test_recovery_to_normal(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_critical_vitals())
        ctrl.record_vitals(_stressed_vitals())  # → recovery
        assert ctrl.get_mode() == "recovery"
        # 5+ recovery steps with vitals mostly in band
        for _ in range(6):
            ctrl.record_vitals(_nominal_vitals())
        assert ctrl.get_mode() == "normal"

    def test_stressed_back_to_normal(self):
        ctrl = HomeostasisController()
        for _ in range(3):
            ctrl.record_vitals(_stressed_vitals())
        assert ctrl.get_mode() == "stressed"
        # Recover
        for _ in range(3):
            ctrl.record_vitals(_nominal_vitals())
        assert ctrl.get_mode() == "normal"


# ---------------------------------------------------------------------------
# Oscillation detection
# ---------------------------------------------------------------------------

class TestOscillation:
    def test_no_oscillation_initially(self):
        ctrl = HomeostasisController()
        for eff in ctrl.effectors.values():
            assert eff.oscillation_count == 0

    def test_detects_oscillation(self):
        ctrl = HomeostasisController()
        # Alternate between high and low latency to cause flip-flopping
        for i in range(12):
            lat = 3.0 if i % 2 == 0 else 0.5
            ctrl.record_vitals({"consensus_latency": lat})
            ctrl.compute_adjustments()
        # timeout_multiplier should show oscillation
        eff = ctrl.effectors["timeout_multiplier"]
        assert eff.oscillation_count >= 4

    def test_oscillation_dampens_loop(self):
        ctrl = HomeostasisController()
        for i in range(12):
            lat = 4.0 if i % 2 == 0 else 0.3
            ctrl.record_vitals({"consensus_latency": lat})
            ctrl.compute_adjustments()
        assert ctrl.loops["consensus_latency"].dampened is True

    def test_no_oscillation_with_stable_input(self):
        ctrl = HomeostasisController()
        for _ in range(12):
            ctrl.record_vitals({"consensus_latency": 1.5})
            ctrl.compute_adjustments()
        eff = ctrl.effectors["timeout_multiplier"]
        assert eff.oscillation_count < 4


# ---------------------------------------------------------------------------
# Health score
# ---------------------------------------------------------------------------

class TestHealthScore:
    def test_perfect_health_near_100(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_nominal_vitals())
        report = ctrl.get_health()
        assert report.score >= 80.0

    def test_critical_health_low(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_critical_vitals())
        report = ctrl.get_health()
        assert report.score < 30.0

    def test_health_has_per_vital_info(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_nominal_vitals())
        report = ctrl.get_health()
        for name in VITAL_NAMES:
            assert name in report.per_vital
            assert "status" in report.per_vital[name]

    def test_health_recommendations_present(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_critical_vitals())
        report = ctrl.get_health()
        assert len(report.recommendations) > 0

    def test_no_data_gives_neutral_score(self):
        ctrl = HomeostasisController()
        report = ctrl.get_health()
        assert 40.0 <= report.score <= 60.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_json_returns_dict(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_nominal_vitals())
        ctrl.compute_adjustments()
        data = ctrl.to_json()
        assert isinstance(data, dict)
        assert "setpoints" in data
        assert "loops" in data
        assert "effectors" in data

    def test_roundtrip_json(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_stressed_vitals())
        ctrl.compute_adjustments()
        data = ctrl.to_json()
        ctrl2 = HomeostasisController.from_json(data)
        assert ctrl2.get_mode() == ctrl.get_mode()
        assert ctrl2.loops["consensus_latency"].integral == pytest.approx(
            ctrl.loops["consensus_latency"].integral
        )

    def test_save_load_file(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_nominal_vitals())
        ctrl.compute_adjustments()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        ctrl.save(path)
        ctrl2 = HomeostasisController.load(path)
        assert ctrl2.effectors["timeout_multiplier"].value == pytest.approx(
            ctrl.effectors["timeout_multiplier"].value
        )
        Path(path).unlink()

    def test_json_serializable(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_nominal_vitals())
        ctrl.compute_adjustments()
        # Should not raise
        json_str = json.dumps(ctrl.to_json())
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_compute_with_no_readings(self):
        ctrl = HomeostasisController()
        adj = ctrl.compute_adjustments()
        assert adj == {}

    def test_single_reading(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals({"consensus_latency": 2.0})
        adj = ctrl.compute_adjustments()
        assert "timeout_multiplier" in adj

    def test_all_vitals_critical_simultaneously(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_critical_vitals())
        assert ctrl.get_mode() == "emergency"
        report = ctrl.get_health()
        assert report.vitals_critical >= 3

    def test_reset_clears_state(self):
        ctrl = HomeostasisController()
        for _ in range(10):
            ctrl.record_vitals(_stressed_vitals())
            ctrl.compute_adjustments()
        ctrl.reset()
        assert ctrl.get_mode() == "normal"
        assert all(len(h) == 0 for h in ctrl.vital_history.values())
        assert len(ctrl.snapshots) == 0

    def test_custom_setpoints(self):
        ctrl = HomeostasisController(setpoints={"consensus_latency": 2.0})
        assert ctrl.setpoints["consensus_latency"] == 2.0
        assert ctrl.loops["consensus_latency"].setpoint == 2.0

    def test_custom_gains(self):
        ctrl = HomeostasisController(gains={"consensus_latency": (0.5, 0.1, 0.2)})
        loop = ctrl.loops["consensus_latency"]
        assert loop.kp == 0.5
        assert loop.ki == 0.1
        assert loop.kd == 0.2

    def test_get_history_returns_snapshots(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_nominal_vitals())
        ctrl.compute_adjustments()
        history = ctrl.get_history()
        assert len(history) == 1
        assert isinstance(history[0], HomeostasisSnapshot)

    def test_export_html_creates_file(self):
        ctrl = HomeostasisController()
        ctrl.record_vitals(_nominal_vitals())
        ctrl.compute_adjustments()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        ctrl.export_html(path)
        content = Path(path).read_text(encoding="utf-8")
        assert "Swarm Homeostasis" in content
        assert "gauge" in content
        Path(path).unlink()
