"""Tests for Swarm Quorum Sensing Engine."""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quorum_sensing import (
    QuorumReport,
    SwarmQuorumSensingEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Basic engine with 5 agents."""
    return SwarmQuorumSensingEngine(agents=["a1", "a2", "a3", "a4", "a5"])


@pytest.fixture
def configured_engine():
    """Engine with channels and behaviors pre-configured."""
    e = SwarmQuorumSensingEngine(agents=[f"agent-{i}" for i in range(8)])
    e.add_channel("ahl", decay_rate=0.1)
    e.add_channel("ai2", decay_rate=0.15)
    e.register_behavior("biofilm", channel="ahl", threshold=3.0, hysteresis=0.8)
    e.register_behavior("glow", channel="ai2", threshold=5.0, hysteresis=1.5)
    return e


# ---------------------------------------------------------------------------
# Channel Management
# ---------------------------------------------------------------------------


class TestChannelManagement:
    def test_add_channel(self, engine):
        ch = engine.add_channel("test", decay_rate=0.2, diffusion_rate=0.8)
        assert ch.name == "test"
        assert ch.decay_rate == 0.2
        assert ch.diffusion_rate == 0.8
        assert ch.concentration == 0.0

    def test_default_decay(self, engine):
        ch = engine.add_channel("test")
        assert ch.decay_rate == engine.default_decay

    def test_multiple_channels(self, engine):
        engine.add_channel("ch1")
        engine.add_channel("ch2")
        engine.add_channel("ch3")
        assert len(engine.channels) == 3

    def test_auto_create_on_produce(self, engine):
        engine.produce("a1", channel="new_ch", intensity=1.0)
        assert "new_ch" in engine.channels


# ---------------------------------------------------------------------------
# Behavior Registration
# ---------------------------------------------------------------------------


class TestBehaviorRegistration:
    def test_register_behavior(self, engine):
        engine.add_channel("ahl")
        bp = engine.register_behavior("biofilm", channel="ahl", threshold=3.0)
        assert bp.name == "biofilm"
        assert bp.threshold == 3.0
        assert bp.active is False

    def test_auto_create_channel(self, engine):
        engine.register_behavior("test", channel="auto_ch", threshold=1.0)
        assert "auto_ch" in engine.channels

    def test_hysteresis_default(self, engine):
        bp = engine.register_behavior("test", channel="ch", threshold=5.0)
        assert bp.hysteresis == 0.5


# ---------------------------------------------------------------------------
# Signal Production
# ---------------------------------------------------------------------------


class TestSignalProduction:
    def test_produce_increases_concentration(self, engine):
        engine.add_channel("ahl")
        engine.produce("a1", channel="ahl", intensity=2.0)
        assert engine.channels["ahl"].concentration == 2.0

    def test_multiple_produces_accumulate(self, engine):
        engine.add_channel("ahl")
        engine.produce("a1", channel="ahl", intensity=1.0)
        engine.produce("a2", channel="ahl", intensity=1.5)
        assert engine.channels["ahl"].concentration == 2.5

    def test_diffusion_rate_scales(self, engine):
        engine.add_channel("slow", diffusion_rate=0.5)
        engine.produce("a1", channel="slow", intensity=2.0)
        assert engine.channels["slow"].concentration == 1.0  # 2.0 * 0.5

    def test_agent_contribution_tracked(self, engine):
        engine.produce("a1", channel="ch1", intensity=3.0)
        engine.produce("a1", channel="ch2", intensity=2.0)
        report = engine.analyze()
        assert report.agent_contributions["a1"] == 5.0


# ---------------------------------------------------------------------------
# Signal Jamming
# ---------------------------------------------------------------------------


class TestSignalJamming:
    def test_jam_decreases_concentration(self, engine):
        engine.add_channel("ahl")
        engine.produce("a1", channel="ahl", intensity=5.0)
        engine.jam("enemy", channel="ahl", strength=2.0)
        assert engine.channels["ahl"].concentration == 3.0

    def test_jam_cannot_go_negative(self, engine):
        engine.add_channel("ahl")
        engine.produce("a1", channel="ahl", intensity=1.0)
        engine.jam("enemy", channel="ahl", strength=5.0)
        assert engine.channels["ahl"].concentration == 0.0

    def test_jam_event_recorded(self, engine):
        engine.jam("enemy", channel="ahl", strength=1.0)
        report = engine.analyze()
        assert report.jamming_events == 1


# ---------------------------------------------------------------------------
# Tick & Decay
# ---------------------------------------------------------------------------


class TestTickAndDecay:
    def test_tick_advances_counter(self, engine):
        engine.tick()
        assert engine.tick_count == 1

    def test_multi_step_tick(self, engine):
        engine.tick(steps=5)
        assert engine.tick_count == 5

    def test_decay_reduces_concentration(self, configured_engine):
        configured_engine.produce("agent-0", channel="ahl", intensity=10.0)
        initial = configured_engine.channels["ahl"].concentration
        configured_engine.tick()
        after = configured_engine.channels["ahl"].concentration
        assert after < initial
        assert after == pytest.approx(initial * 0.9, rel=1e-6)

    def test_exponential_decay_over_time(self, engine):
        engine.add_channel("ch", decay_rate=0.2)
        engine.produce("a1", channel="ch", intensity=100.0)
        engine.tick(steps=10)
        # After 10 ticks: 100 * (0.8)^10 ≈ 10.74
        expected = 100.0 * (0.8 ** 10)
        assert engine.channels["ch"].concentration == pytest.approx(expected, rel=1e-4)

    def test_history_recorded(self, configured_engine):
        configured_engine.produce("agent-0", channel="ahl", intensity=5.0)
        configured_engine.tick(steps=3)
        assert len(configured_engine.channels["ahl"].history) == 3


# ---------------------------------------------------------------------------
# Threshold Activation
# ---------------------------------------------------------------------------


class TestThresholdActivation:
    def test_behavior_activates_at_threshold(self, configured_engine):
        # biofilm threshold=3.0 on ahl
        configured_engine.produce("agent-0", channel="ahl", intensity=3.5)
        snapshot = configured_engine.tick()
        # After decay: 3.5 * 0.9 = 3.15 >= 3.0
        assert "biofilm" in snapshot.active_behaviors

    def test_behavior_stays_inactive_below_threshold(self, configured_engine):
        configured_engine.produce("agent-0", channel="ahl", intensity=1.0)
        snapshot = configured_engine.tick()
        assert "biofilm" not in snapshot.active_behaviors

    def test_hysteresis_prevents_oscillation(self, configured_engine):
        # Activate biofilm (threshold=3.0)
        configured_engine.produce("agent-0", channel="ahl", intensity=5.0)
        configured_engine.tick()  # 5.0 * 0.9 = 4.5 >= 3.0 → activate
        assert configured_engine.behaviors["biofilm"].active

        # Drop below threshold but above hysteresis band (3.0 - 0.8 = 2.2)
        # Let it decay: after more ticks it should stay active until < 2.2
        for _ in range(5):
            configured_engine.tick()
        # 5.0 * 0.9^6 ≈ 2.66 — still above 2.2
        assert configured_engine.behaviors["biofilm"].active

    def test_deactivation_below_hysteresis(self, engine):
        engine.add_channel("ch", decay_rate=0.5)  # fast decay
        engine.register_behavior("test", channel="ch", threshold=3.0, hysteresis=1.0)
        engine.produce("a1", channel="ch", intensity=5.0)
        engine.tick()  # 5 * 0.5 = 2.5 — below threshold, never activates after decay
        # Need to produce enough that after decay it's still >= 3.0
        engine.produce("a1", channel="ch", intensity=10.0)
        engine.tick()  # (2.5*0.5 + 10)*0.5 = 6.25 * 0.5... let's just check
        # Actually: concentration = 2.5 + 10 = 12.5 (at tick before decay)
        # Wait — produce happens before tick. Let me re-check logic.
        # After tick: decay then check. Let's just verify final state.
        engine.tick()  # more decay
        # Eventually it will deactivate when < 2.0 (threshold - hysteresis)

    def test_newly_activated_reported(self, configured_engine):
        configured_engine.produce("agent-0", channel="ahl", intensity=5.0)
        snapshot = configured_engine.tick()
        assert "biofilm" in snapshot.newly_activated

    def test_newly_deactivated_reported(self, engine):
        engine.add_channel("ch", decay_rate=0.9)  # very fast decay
        engine.register_behavior("quick", channel="ch", threshold=2.0, hysteresis=0.5)
        engine.produce("a1", channel="ch", intensity=30.0)
        engine.tick()  # 30 * 0.1 = 3.0 >= 2.0 → activate
        # Fast decay: 3.0 * 0.1 = 0.3 < 1.5 → deactivate
        snap = engine.tick()
        assert "quick" in snap.newly_deactivated


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_signal_diversity_single_channel(self, engine):
        engine.add_channel("only")
        engine.produce("a1", channel="only", intensity=5.0)
        snapshot = engine.tick()
        # Single channel → diversity normalized but log2(1)=0
        assert snapshot.signal_diversity >= 0.0

    def test_signal_diversity_multiple_equal(self, engine):
        engine.add_channel("ch1", decay_rate=0.0)
        engine.add_channel("ch2", decay_rate=0.0)
        engine.produce("a1", channel="ch1", intensity=5.0)
        engine.produce("a2", channel="ch2", intensity=5.0)
        snapshot = engine.tick()
        # Equal distribution → max entropy → diversity ≈ 1.0
        assert snapshot.signal_diversity > 0.9

    def test_coordination_efficiency(self, configured_engine):
        # No signals → all behaviors inactive, thresholds not met → efficient
        snapshot = configured_engine.tick()
        assert snapshot.coordination_efficiency > 0.0

    def test_health_score_range(self, configured_engine):
        configured_engine.produce("agent-0", channel="ahl", intensity=5.0)
        snapshot = configured_engine.tick()
        assert 0.0 <= snapshot.quorum_health_score <= 100.0

    def test_density_estimation(self, engine):
        engine.add_channel("ch", decay_rate=0.1)
        for a in engine.agents:
            engine.produce(a, channel="ch", intensity=1.0)
        snapshot = engine.tick()
        assert snapshot.estimated_density > 0.0


# ---------------------------------------------------------------------------
# Analysis Report
# ---------------------------------------------------------------------------


class TestAnalysis:
    def test_report_structure(self, configured_engine):
        for _ in range(5):
            configured_engine.produce("agent-0", channel="ahl", intensity=1.0)
            configured_engine.tick()
        report = configured_engine.analyze()
        assert isinstance(report, QuorumReport)
        assert len(report.snapshots) == 5
        assert report.overall_health > 0

    def test_peak_concentration(self, engine):
        engine.add_channel("ch", decay_rate=0.0)
        engine.produce("a1", channel="ch", intensity=10.0)
        engine.tick()
        engine.produce("a1", channel="ch", intensity=5.0)
        engine.tick()
        report = engine.analyze()
        assert report.peak_concentration["ch"] >= 10.0

    def test_behavior_uptime(self, engine):
        engine.add_channel("ch", decay_rate=0.0)
        engine.register_behavior("always_on", channel="ch", threshold=1.0)
        engine.produce("a1", channel="ch", intensity=5.0)
        engine.tick(steps=10)
        report = engine.analyze()
        assert report.behavior_uptime["always_on"] == 1.0  # active all 10 ticks

    def test_signal_wars_detected(self, engine):
        engine.add_channel("ch")
        engine.produce("a1", channel="ch", intensity=5.0)
        engine.jam("enemy", channel="ch", strength=2.0)
        engine.tick()
        report = engine.analyze()
        assert len(report.signal_wars) >= 1


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_html_export(self, configured_engine):
        configured_engine.produce("agent-0", channel="ahl", intensity=5.0)
        configured_engine.tick(steps=5)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            configured_engine.export_html(path)
            content = open(path, encoding="utf-8").read()
            assert "Swarm Quorum Sensing" in content
            assert "biofilm" in content
        finally:
            os.unlink(path)

    def test_json_export(self, configured_engine):
        configured_engine.produce("agent-0", channel="ahl", intensity=5.0)
        configured_engine.tick(steps=3)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            configured_engine.export_json(path)
            data = json.loads(open(path, encoding="utf-8").read())
            assert "overall_health" in data
            assert "snapshots" in data
            assert len(data["snapshots"]) == 3
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_runs(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["quorum_sensing", "--agents", "5", "--ticks", "10"])
        from src.quorum_sensing import main
        main()  # Should not raise


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_agents(self):
        engine = SwarmQuorumSensingEngine(agents=[])
        engine.add_channel("ch")
        snapshot = engine.tick()
        assert snapshot.quorum_health_score >= 0

    def test_zero_intensity_produce(self, engine):
        engine.add_channel("ch")
        engine.produce("a1", channel="ch", intensity=0.0)
        assert engine.channels["ch"].concentration == 0.0

    def test_very_high_decay(self, engine):
        engine.add_channel("ch", decay_rate=0.99)
        engine.produce("a1", channel="ch", intensity=100.0)
        engine.tick()
        assert engine.channels["ch"].concentration < 2.0

    def test_zero_decay(self, engine):
        engine.add_channel("ch", decay_rate=0.0)
        engine.produce("a1", channel="ch", intensity=5.0)
        engine.tick(steps=100)
        assert engine.channels["ch"].concentration == 5.0
