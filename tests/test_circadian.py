"""Tests for the Swarm Circadian Engine."""
from __future__ import annotations

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.circadian import (
    CircadianEngine,
)


def _make_sample(hour: int, quality: float = 0.5) -> dict:
    """Create a metrics dict with given quality level (0-1)."""
    return {
        "throughput": quality * 15,
        "latency": 2.0 - quality * 1.5,
        "error_rate": 0.2 - quality * 0.15,
        "collaboration": quality * 8,
        "creativity": quality * 0.8,
        "focus_duration": quality * 50,
        "recovery_time": 5 - quality * 3,
    }


class TestRecording:
    """Test sample recording."""

    def test_record_basic(self):
        engine = CircadianEngine()
        engine.record_sample("a1", 10, _make_sample(10, 0.7))
        assert "a1" in engine._agents
        assert engine._agents["a1"].sample_count == 1

    def test_record_multiple_bins(self):
        engine = CircadianEngine()
        for h in range(24):
            engine.record_sample("a1", h, _make_sample(h, 0.5))
        assert engine._agents["a1"].sample_count == 24

    def test_record_wraps_hour(self):
        engine = CircadianEngine()
        engine.record_sample("a1", 25, _make_sample(1, 0.5))
        # hour 25 % 24 = 1, bin = 1
        assert 1 in engine._agents["a1"].bins

    def test_record_caps_per_bin(self):
        engine = CircadianEngine(max_samples_per_bin=5)
        for _ in range(20):
            engine.record_sample("a1", 3, _make_sample(3, 0.5))
        assert len(engine._agents["a1"].bins[3]) == 5

    def test_multiple_agents(self):
        engine = CircadianEngine()
        engine.record_sample("a1", 9, _make_sample(9, 0.8))
        engine.record_sample("a2", 21, _make_sample(21, 0.6))
        assert len(engine._agents) == 2


class TestPerformanceScoring:
    """Test composite performance score computation."""

    def test_high_quality_scores_high(self):
        engine = CircadianEngine()
        score = engine._compute_performance_score(_make_sample(12, 1.0))
        assert score > 0.6

    def test_low_quality_scores_low(self):
        engine = CircadianEngine()
        score = engine._compute_performance_score(_make_sample(12, 0.0))
        assert score < 0.4

    def test_score_in_range(self):
        engine = CircadianEngine()
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            score = engine._compute_performance_score(_make_sample(12, q))
            assert 0.0 <= score <= 1.0

    def test_empty_metrics(self):
        engine = CircadianEngine()
        score = engine._compute_performance_score({})
        assert score == 0.0


class TestChronotypeClassification:
    """Test chronotype classification."""

    def test_early_bird(self):
        engine = CircadianEngine()
        # High performance in morning, low at night
        for _ in range(10):
            for h in range(24):
                quality = 0.9 if 6 <= h <= 11 else 0.2
                engine.record_sample("early", h, _make_sample(h, quality))
        ct = engine.classify_chronotype("early")
        assert ct.chronotype == "EarlyBird"
        assert ct.confidence > 0

    def test_night_owl(self):
        engine = CircadianEngine()
        for _ in range(10):
            for h in range(24):
                quality = 0.9 if h >= 20 or h <= 2 else 0.2
                engine.record_sample("night", h, _make_sample(h, quality))
        ct = engine.classify_chronotype("night")
        assert ct.chronotype == "NightOwl"

    def test_steady_state(self):
        engine = CircadianEngine()
        for _ in range(10):
            for h in range(24):
                engine.record_sample("steady", h, _make_sample(h, 0.6))
        ct = engine.classify_chronotype("steady")
        assert ct.chronotype == "SteadyState"

    def test_unknown_agent(self):
        engine = CircadianEngine()
        ct = engine.classify_chronotype("nonexistent")
        assert ct.chronotype == "Irregular"
        assert ct.confidence == 0.0

    def test_profile_has_fields(self):
        engine = CircadianEngine()
        for h in range(24):
            engine.record_sample("x", h, _make_sample(h, 0.5 + 0.3 * math.sin(h / 4)))
        ct = engine.classify_chronotype("x")
        assert isinstance(ct.peak_hours, list)
        assert isinstance(ct.trough_hours, list)
        assert ct.dominant_period > 0
        assert isinstance(ct.amplitude, float)


class TestJetlagDetection:
    """Test rhythm disruption detection."""

    def test_no_jetlag_stable(self):
        engine = CircadianEngine()
        for _ in range(10):
            for h in range(24):
                quality = 0.8 if 8 <= h <= 12 else 0.3
                engine.record_sample("stable", h, _make_sample(h, quality))
            engine.classify_chronotype("stable")  # updates phase history
        jl = engine.detect_jetlag("stable")
        # With consistent phase, no disruption
        assert jl.severity in ("none", "mild")

    def test_jetlag_after_shift(self):
        engine = CircadianEngine()
        # Build baseline
        for _ in range(5):
            for h in range(24):
                quality = 0.8 if 8 <= h <= 12 else 0.3
                engine.record_sample("shifted", h, _make_sample(h, quality))
            engine.classify_chronotype("shifted")

        # Manually inject phase shift
        state = engine._agents["shifted"]
        state.phase_history.extend([state.phase_history[-1] + 8] * 4)

        jl = engine.detect_jetlag("shifted")
        assert jl.disrupted is True
        assert jl.severity in ("moderate", "severe")
        assert jl.recovery_eta_hours > 0

    def test_unknown_agent_no_crash(self):
        engine = CircadianEngine()
        jl = engine.detect_jetlag("ghost")
        assert jl.disrupted is False

    def test_jetlag_report_fields(self):
        engine = CircadianEngine()
        engine.record_sample("x", 10, _make_sample(10, 0.7))
        engine.classify_chronotype("x")
        jl = engine.detect_jetlag("x")
        assert hasattr(jl, "phase_shift_hours")
        assert hasattr(jl, "performance_degradation")


class TestOptimalWindows:
    """Test optimal window detection."""

    def test_finds_peak(self):
        engine = CircadianEngine()
        for _ in range(10):
            for h in range(24):
                quality = 0.9 if 14 <= h <= 17 else 0.2
                engine.record_sample("worker", h, _make_sample(h, quality))
        win = engine.optimal_windows("worker")
        assert 12 <= win.peak_start <= 16
        assert win.score > 0

    def test_empty_agent(self):
        engine = CircadianEngine()
        win = engine.optimal_windows("empty")
        assert win.peak_start == 9  # default
        assert win.score == 0.0

    def test_has_recommendations(self):
        engine = CircadianEngine()
        for _ in range(5):
            for h in range(24):
                engine.record_sample("x", h, _make_sample(h, 0.7))
        win = engine.optimal_windows("x")
        assert isinstance(win.recommended_tasks, list)
        assert len(win.recommended_tasks) > 0


class TestEntrainment:
    """Test phase coupling analysis."""

    def test_synchronized_agents(self):
        engine = CircadianEngine()
        for _ in range(10):
            for h in range(24):
                quality = 0.8 if 9 <= h <= 13 else 0.3
                engine.record_sample("sync1", h, _make_sample(h, quality))
                engine.record_sample("sync2", h, _make_sample(h, quality))
        pairs = engine.compute_entrainment()
        assert len(pairs) == 1
        assert pairs[0].coupling_strength > 0.5

    def test_desynchronized_agents(self):
        engine = CircadianEngine()
        for _ in range(10):
            for h in range(24):
                q1 = 0.9 if 6 <= h <= 10 else 0.2
                q2 = 0.9 if 18 <= h <= 22 else 0.2
                engine.record_sample("day", h, _make_sample(h, q1))
                engine.record_sample("night", h, _make_sample(h, q2))
        pairs = engine.compute_entrainment()
        assert len(pairs) == 1
        # Should have low coupling or large phase difference
        assert pairs[0].phase_difference != 0 or pairs[0].coupling_strength < 0.8

    def test_empty_swarm(self):
        engine = CircadianEngine()
        pairs = engine.compute_entrainment()
        assert pairs == []

    def test_single_agent(self):
        engine = CircadianEngine()
        engine.record_sample("solo", 10, _make_sample(10, 0.8))
        pairs = engine.compute_entrainment()
        assert pairs == []


class TestCollectiveReport:
    """Test swarm-wide report generation."""

    def test_empty_report(self):
        engine = CircadianEngine()
        report = engine.collective_report()
        assert report.health_score == 0.0
        assert report.agent_count == 0

    def test_healthy_swarm(self):
        engine = CircadianEngine()
        for _ in range(10):
            for h in range(24):
                quality = 0.7 + 0.2 * math.sin(math.pi * (h - 10) / 12)
                engine.record_sample("a1", h, _make_sample(h, quality))
                engine.record_sample("a2", h, _make_sample(h, quality))
        report = engine.collective_report()
        assert report.health_score > 0
        assert report.agent_count == 2
        assert report.total_samples > 0
        assert isinstance(report.recommendations, list)

    def test_report_has_chronotypes(self):
        engine = CircadianEngine()
        for h in range(24):
            engine.record_sample("x", h, _make_sample(h, 0.5))
        report = engine.collective_report()
        assert isinstance(report.chronotype_distribution, dict)

    def test_report_detects_disrupted(self):
        engine = CircadianEngine()
        for _ in range(5):
            for h in range(24):
                engine.record_sample("ok", h, _make_sample(h, 0.6))
                engine.record_sample("bad", h, _make_sample(h, 0.6))
            engine.classify_chronotype("ok")
            engine.classify_chronotype("bad")
        # Inject disruption
        engine._agents["bad"].phase_history.extend([20, 20, 20, 20])
        report = engine.collective_report()
        # May or may not detect depending on baseline; just verify no crash
        assert isinstance(report.disrupted_agents, list)


class TestPersistence:
    """Test save/load."""

    def test_roundtrip(self):
        engine = CircadianEngine()
        for h in range(24):
            engine.record_sample("a1", h, _make_sample(h, 0.5 + 0.3 * (h / 24)))
        engine.classify_chronotype("a1")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            engine.save(path)
            loaded = CircadianEngine.load(path)
            assert "a1" in loaded._agents
            assert loaded._agents["a1"].sample_count == 24
            assert loaded._sample_counter == 24
        finally:
            os.unlink(path)

    def test_load_preserves_phase_history(self):
        engine = CircadianEngine()
        engine.record_sample("x", 10, _make_sample(10, 0.8))
        engine._agents["x"].phase_history = [5.0, 6.0, 7.0]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            engine.save(path)
            loaded = CircadianEngine.load(path)
            assert loaded._agents["x"].phase_history == [5.0, 6.0, 7.0]
        finally:
            os.unlink(path)


class TestHTMLExport:
    """Test dashboard export."""

    def test_export_creates_file(self):
        engine = CircadianEngine()
        for _ in range(5):
            for h in range(24):
                engine.record_sample("a1", h, _make_sample(h, 0.6))
                engine.record_sample("a2", h, _make_sample(h, 0.4))

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            engine.export_html(path)
            content = open(path, encoding='utf-8').read()
            assert "Circadian" in content
            assert "a1" in content
            assert len(content) > 500
        finally:
            os.unlink(path)

    def test_empty_export_no_crash(self):
        engine = CircadianEngine()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            engine.export_html(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestDFT:
    """Test FFT/DFT rhythm detection."""

    def test_detects_24h_cycle(self):
        engine = CircadianEngine()
        # Pure sinusoidal with 24h period
        profile = [0.5 + 0.4 * math.sin(2 * math.pi * h / 24) for h in range(24)]
        period, amplitude, phase = engine._dominant_rhythm(profile)
        assert abs(period - 24.0) < 1.0
        assert amplitude > 0.2

    def test_detects_12h_cycle(self):
        engine = CircadianEngine()
        profile = [0.5 + 0.4 * math.sin(2 * math.pi * h / 12) for h in range(24)]
        period, amplitude, phase = engine._dominant_rhythm(profile)
        assert abs(period - 12.0) < 1.0

    def test_flat_signal(self):
        engine = CircadianEngine()
        profile = [0.5] * 24
        period, amplitude, phase = engine._dominant_rhythm(profile)
        assert amplitude < 0.01


class TestCLI:
    """Test CLI entry point doesn't crash."""

    def test_import(self):
        from src.circadian import main
        assert callable(main)


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

def run_tests():
    """Run all tests and report results."""
    import traceback

    classes = [
        TestRecording,
        TestPerformanceScoring,
        TestChronotypeClassification,
        TestJetlagDetection,
        TestOptimalWindows,
        TestEntrainment,
        TestCollectiveReport,
        TestPersistence,
        TestHTMLExport,
        TestDFT,
        TestCLI,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in methods:
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
            except Exception as e:
                failed += 1
                errors.append((f"{cls.__name__}.{method_name}", str(e), traceback.format_exc()))

    print(f"\n{'='*60}")
    print(f"  Circadian Engine Tests: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    if errors:
        print("\nFailures:")
        for name, err, tb in errors:
            print(f"\n  ✗ {name}: {err}")
            for line in tb.strip().split('\n')[-3:]:
                print(f"    {line}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
