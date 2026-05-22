"""Tests for Swarm Nociception Engine."""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nociception import (
    FiberType,
    FIBER_PROFILES,
    GateControlModulator,
    GateState,
    HealthTier,
    InsightSeverity,
    NociceptiveHealthScorer,
    NociceptionReport,
    NociceptorArray,
    PainMemoryEngine,
    PainPhase,
    PainSignalPropagator,
    PathologyType,
    ProtectiveReflexEngine,
    REFLEX_PROFILES,
    ReflexType,
    SCENARIOS,
    STIMULUS_PROFILES,
    StimulusType,
    SwarmNociceptionEngine,
    ToleranceAdaptationEngine,
    ToleranceProfile,
    AgentPainState,
    run_demo,
    main,
)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_stimulus_types(self):
        assert len(StimulusType) == 6
        assert StimulusType.MECHANICAL.value == "mechanical"
        assert StimulusType.INFLAMMATORY.value == "inflammatory"

    def test_fiber_types(self):
        assert len(FiberType) == 2
        assert FiberType.A_DELTA.value == "a_delta"
        assert FiberType.C_FIBER.value == "c_fiber"

    def test_reflex_types(self):
        assert len(ReflexType) == 6
        assert ReflexType.WITHDRAWAL.value == "withdrawal"
        assert ReflexType.FREEZE.value == "freeze"

    def test_pain_phases(self):
        assert len(PainPhase) == 4
        assert PainPhase.ACUTE.value == "acute"
        assert PainPhase.RESOLVED.value == "resolved"

    def test_health_tiers(self):
        assert len(HealthTier) == 5
        assert HealthTier.PROTECTED.value == "protected"
        assert HealthTier.CRITICAL.value == "critical"

    def test_pathology_types(self):
        assert len(PathologyType) == 6
        assert PathologyType.ALLODYNIA.value == "allodynia"

    def test_insight_severity(self):
        assert len(InsightSeverity) == 3


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_stimulus_profiles_complete(self):
        for st in StimulusType:
            assert st in STIMULUS_PROFILES or st.value in [s.value for s in STIMULUS_PROFILES]

    def test_reflex_profiles_complete(self):
        for rt in ReflexType:
            assert rt in REFLEX_PROFILES

    def test_fiber_profiles_complete(self):
        for ft in FiberType:
            assert ft in FIBER_PROFILES

    def test_stimulus_profile_fields(self):
        for st, profile in STIMULUS_PROFILES.items():
            assert "base_threshold" in profile
            assert "fiber" in profile
            assert "decay_rate" in profile
            assert "spread_factor" in profile
            assert 0 < profile["base_threshold"] < 1
            assert 0 < profile["decay_rate"] < 1

    def test_reflex_profile_fields(self):
        for rt, profile in REFLEX_PROFILES.items():
            assert "threshold" in profile
            assert "effectiveness" in profile
            assert 0 < profile["threshold"] < 1
            assert 0 < profile["effectiveness"] <= 1


# ---------------------------------------------------------------------------
# NociceptorArray Tests
# ---------------------------------------------------------------------------


class TestNociceptorArray:
    def setup_method(self):
        self.agents = ["agent-0", "agent-1", "agent-2"]
        self.array = NociceptorArray(self.agents)

    def test_initialization(self):
        for aid in self.agents:
            nocs = self.array.get_state(aid)
            assert len(nocs) == len(StimulusType)

    def test_detect_below_threshold(self):
        result = self.array.detect("agent-0", StimulusType.MECHANICAL, 0.1, tick=1)
        assert result is None

    def test_detect_above_threshold(self):
        result = self.array.detect("agent-0", StimulusType.MECHANICAL, 0.8, tick=1)
        assert result is not None
        assert 0.0 < result <= 1.0

    def test_detect_unknown_agent(self):
        result = self.array.detect("unknown", StimulusType.MECHANICAL, 0.9, tick=1)
        assert result is None

    def test_activation_count_increments(self):
        self.array.detect("agent-0", StimulusType.MECHANICAL, 0.8, tick=1)
        self.array.detect("agent-0", StimulusType.MECHANICAL, 0.9, tick=2)
        nocs = self.array.get_state("agent-0")
        mech = [n for n in nocs if n.stimulus_type == StimulusType.MECHANICAL][0]
        assert mech.activation_count == 2

    def test_sensitize(self):
        self.array.sensitize("agent-0", StimulusType.MECHANICAL, 0.1)
        nocs = self.array.get_state("agent-0")
        mech = [n for n in nocs if n.stimulus_type == StimulusType.MECHANICAL][0]
        assert mech.sensitization > 0

    def test_desensitize(self):
        self.array.sensitize("agent-0", StimulusType.MECHANICAL, 0.1)
        self.array.desensitize("agent-0", StimulusType.MECHANICAL, 0.05)
        nocs = self.array.get_state("agent-0")
        mech = [n for n in nocs if n.stimulus_type == StimulusType.MECHANICAL][0]
        assert mech.sensitization == pytest.approx(0.05)

    def test_sensitization_lowers_threshold(self):
        # Before sensitization, 0.35 shouldn't trigger MECHANICAL (threshold=0.4)
        r1 = self.array.detect("agent-0", StimulusType.MECHANICAL, 0.35, tick=1)
        assert r1 is None
        # Sensitize
        self.array.sensitize("agent-0", StimulusType.MECHANICAL, 0.15)
        # Now 0.35 should trigger
        r2 = self.array.detect("agent-0", StimulusType.MECHANICAL, 0.35, tick=2)
        assert r2 is not None


# ---------------------------------------------------------------------------
# PainSignalPropagator Tests
# ---------------------------------------------------------------------------


class TestPainSignalPropagator:
    def setup_method(self):
        self.agents = ["agent-0", "agent-1", "agent-2"]
        self.prop = PainSignalPropagator(self.agents)

    def test_emit_signal(self):
        sig = self.prop.emit_signal("agent-0", StimulusType.MECHANICAL, 0.7, tick=1)
        assert sig.source_agent == "agent-0"
        assert sig.intensity == 0.7
        assert sig.phase == PainPhase.ACUTE

    def test_pain_level_empty(self):
        assert self.prop.get_pain_level("agent-0") == 0.0

    def test_pain_level_after_signal(self):
        self.prop.emit_signal("agent-0", StimulusType.MECHANICAL, 0.7, tick=1)
        assert self.prop.get_pain_level("agent-0") > 0.0

    def test_signal_decay(self):
        self.prop.emit_signal("agent-0", StimulusType.MECHANICAL, 0.7, tick=1)
        initial = self.prop.get_pain_level("agent-0")
        for t in range(2, 10):
            self.prop.propagate(t)
        later = self.prop.get_pain_level("agent-0")
        assert later < initial

    def test_signal_expires(self):
        self.prop.emit_signal("agent-0", StimulusType.MECHANICAL, 0.5, tick=1)
        for t in range(2, 50):
            self.prop.propagate(t)
        assert self.prop.get_pain_level("agent-0") == 0.0

    def test_multiple_signals_stack(self):
        self.prop.emit_signal("agent-0", StimulusType.MECHANICAL, 0.5, tick=1)
        single = self.prop.get_pain_level("agent-0")
        self.prop.emit_signal("agent-0", StimulusType.THERMAL, 0.5, tick=1)
        double = self.prop.get_pain_level("agent-0")
        assert double > single


# ---------------------------------------------------------------------------
# ProtectiveReflexEngine Tests
# ---------------------------------------------------------------------------


class TestProtectiveReflexEngine:
    def setup_method(self):
        self.engine = ProtectiveReflexEngine()

    def test_no_reflex_below_threshold(self):
        reflexes = self.engine.evaluate("agent-0", 0.1, StimulusType.MECHANICAL, tick=1)
        assert len(reflexes) == 0

    def test_reflex_fires_above_threshold(self):
        reflexes = self.engine.evaluate("agent-0", 0.8, StimulusType.MECHANICAL, tick=1)
        assert len(reflexes) > 0

    def test_cooldown_prevents_rapid_fire(self):
        self.engine.evaluate("agent-0", 0.8, StimulusType.MECHANICAL, tick=1)
        r2 = self.engine.evaluate("agent-0", 0.8, StimulusType.MECHANICAL, tick=2)
        assert len(r2) == 0  # Cooldown active

    def test_cooldown_expires(self):
        self.engine.evaluate("agent-0", 0.8, StimulusType.MECHANICAL, tick=1)
        r2 = self.engine.evaluate("agent-0", 0.8, StimulusType.MECHANICAL, tick=10)
        assert len(r2) > 0

    def test_higher_pain_triggers_more_reflexes(self):
        low = self.engine.evaluate("agent-0", 0.35, StimulusType.MECHANICAL, tick=1)
        high = self.engine.evaluate("agent-1", 0.9, StimulusType.MECHANICAL, tick=1)
        assert len(high) >= len(low)


# ---------------------------------------------------------------------------
# PainMemoryEngine Tests
# ---------------------------------------------------------------------------


class TestPainMemoryEngine:
    def setup_method(self):
        self.mem = PainMemoryEngine(max_memories_per_agent=10)

    def test_record_memory(self):
        m = self.mem.record("agent-0", StimulusType.MECHANICAL, 0.6, "test", tick=1)
        assert m.agent_id == "agent-0"
        assert m.intensity == 0.6

    def test_recall(self):
        self.mem.record("agent-0", StimulusType.MECHANICAL, 0.6, "ctx", tick=1)
        memories = self.mem.recall("agent-0")
        assert len(memories) == 1
        assert memories[0].times_recalled == 1

    def test_recall_filtered(self):
        self.mem.record("agent-0", StimulusType.MECHANICAL, 0.6, "a", tick=1)
        self.mem.record("agent-0", StimulusType.THERMAL, 0.5, "b", tick=2)
        mech = self.mem.recall("agent-0", StimulusType.MECHANICAL)
        assert len(mech) == 1

    def test_avoidance_learning(self):
        for i in range(3):
            self.mem.record("agent-0", StimulusType.CHEMICAL, 0.7, f"ctx{i}", tick=i)
        assert self.mem.has_learned_avoidance("agent-0", StimulusType.CHEMICAL)

    def test_no_avoidance_without_repetition(self):
        self.mem.record("agent-0", StimulusType.CHEMICAL, 0.7, "ctx", tick=1)
        assert not self.mem.has_learned_avoidance("agent-0", StimulusType.CHEMICAL)

    def test_capacity_eviction(self):
        for i in range(15):
            self.mem.record("agent-0", StimulusType.MECHANICAL, 0.5, f"ctx{i}", tick=i)
        assert len(self.mem.memories["agent-0"]) == 10

    def test_get_all(self):
        self.mem.record("agent-0", StimulusType.MECHANICAL, 0.5, "a", tick=1)
        self.mem.record("agent-1", StimulusType.THERMAL, 0.6, "b", tick=2)
        all_mems = self.mem.get_all()
        assert len(all_mems) == 2


# ---------------------------------------------------------------------------
# ToleranceAdaptationEngine Tests
# ---------------------------------------------------------------------------


class TestToleranceAdaptation:
    def setup_method(self):
        self.engine = ToleranceAdaptationEngine(["agent-0", "agent-1"])

    def test_low_intensity_builds_tolerance(self):
        effect, adjusted = self.engine.process_exposure("agent-0", StimulusType.MECHANICAL, 0.4)
        assert effect == "habituation"
        profile = self.engine.profiles["agent-0"]
        assert profile.tolerance_levels[StimulusType.MECHANICAL.value] > 0

    def test_high_intensity_sensitizes(self):
        effect, adjusted = self.engine.process_exposure("agent-0", StimulusType.MECHANICAL, 0.8)
        assert effect == "sensitization"
        profile = self.engine.profiles["agent-0"]
        assert profile.tolerance_levels[StimulusType.MECHANICAL.value] < 0

    def test_adjusted_intensity_reduced_by_tolerance(self):
        # Build some tolerance first
        self.engine.process_exposure("agent-0", StimulusType.MECHANICAL, 0.3)
        self.engine.process_exposure("agent-0", StimulusType.MECHANICAL, 0.3)
        _, adjusted = self.engine.process_exposure("agent-0", StimulusType.MECHANICAL, 0.5)
        assert adjusted < 0.5

    def test_unknown_agent(self):
        effect, adjusted = self.engine.process_exposure("unknown", StimulusType.MECHANICAL, 0.5)
        assert effect == "none"
        assert adjusted == 0.5

    def test_exposure_count_tracks(self):
        self.engine.process_exposure("agent-0", StimulusType.MECHANICAL, 0.4)
        self.engine.process_exposure("agent-0", StimulusType.MECHANICAL, 0.4)
        profile = self.engine.profiles["agent-0"]
        assert profile.exposures[StimulusType.MECHANICAL.value] == 2


# ---------------------------------------------------------------------------
# GateControlModulator Tests
# ---------------------------------------------------------------------------


class TestGateControlModulator:
    def setup_method(self):
        self.gate = GateControlModulator(["agent-0", "agent-1"])

    def test_default_gate(self):
        openness = self.gate.compute_gate("agent-0")
        assert 0.4 <= openness <= 0.6  # Near baseline

    def test_inhibition_closes_gate(self):
        self.gate.apply_inhibition("agent-0", 0.8)
        openness = self.gate.compute_gate("agent-0")
        assert openness < 0.5

    def test_excitation_opens_gate(self):
        self.gate.apply_excitation("agent-0", 0.8)
        openness = self.gate.compute_gate("agent-0")
        assert openness > 0.5

    def test_modulate_pain(self):
        self.gate.apply_inhibition("agent-0", 0.8)
        modulated = self.gate.modulate_pain("agent-0", 0.7)
        assert modulated < 0.7

    def test_descending_modulation_suppresses(self):
        self.gate.set_descending_modulation("agent-0", -0.8)
        openness = self.gate.compute_gate("agent-0")
        assert openness < 0.4

    def test_decay_toward_baseline(self):
        self.gate.apply_excitation("agent-0", 1.0)
        for _ in range(20):
            self.gate.decay()
        openness = self.gate.compute_gate("agent-0")
        assert 0.4 <= openness <= 0.6

    def test_gate_bounds(self):
        self.gate.apply_excitation("agent-0", 10.0)
        openness = self.gate.compute_gate("agent-0")
        assert openness <= 1.0
        self.gate.apply_inhibition("agent-1", 10.0)
        openness = self.gate.compute_gate("agent-1")
        assert openness >= 0.0


# ---------------------------------------------------------------------------
# NociceptiveHealthScorer Tests
# ---------------------------------------------------------------------------


class TestHealthScorer:
    def setup_method(self):
        self.scorer = NociceptiveHealthScorer()

    def test_empty_state_is_healthy(self):
        score = self.scorer.score({}, [], [], [])
        assert score.score == 100.0
        assert score.tier == HealthTier.PROTECTED

    def test_high_pain_reduces_score(self):
        states = {
            "agent-0": AgentPainState(
                agent_id="agent-0", nociceptors=[], active_signals=[],
                current_pain_level=0.9, pain_history=[0.9],
                reflexes_triggered=0,
                tolerance=ToleranceProfile(agent_id="agent-0"),
                gate=GateState(agent_id="agent-0"),
                memories=[],
            )
        }
        score = self.scorer.score(states, [], [], [])
        assert score.score < 80
        assert score.acute_load > 0.5


# ---------------------------------------------------------------------------
# SwarmNociceptionEngine Integration Tests
# ---------------------------------------------------------------------------


class TestSwarmNociceptionEngine:
    def setup_method(self):
        self.engine = SwarmNociceptionEngine(num_agents=4, seed=42)

    def test_initialization(self):
        assert self.engine.num_agents == 4
        assert len(self.engine.agent_ids) == 4
        assert self.engine.tick == 0

    def test_apply_stimulus(self):
        self.engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.8)
        self.engine.do_tick()
        report = self.engine.get_report()
        assert report.total_ticks == 1

    def test_pain_increases_after_stimulus(self):
        self.engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.9)
        self.engine.do_tick()
        report = self.engine.get_report()
        state = report.agent_states["agent-0"]
        assert state.current_pain_level > 0

    def test_pain_decays_over_time(self):
        self.engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.8)
        self.engine.do_tick()
        r1 = self.engine.get_report()
        for _ in range(20):
            self.engine.do_tick()
        r2 = self.engine.get_report()
        assert r2.agent_states["agent-0"].current_pain_level <= r1.agent_states["agent-0"].current_pain_level

    def test_reflexes_triggered(self):
        self.engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.9)
        self.engine.do_tick()
        report = self.engine.get_report()
        assert len(report.all_reflexes) > 0

    def test_memory_recorded(self):
        self.engine.apply_stimulus("agent-0", StimulusType.CHEMICAL, 0.7)
        self.engine.do_tick()
        report = self.engine.get_report()
        assert len(report.all_memories) > 0

    def test_inhibition_reduces_pain(self):
        self.engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.8)
        self.engine.do_tick()
        r1 = self.engine.get_report()
        # Apply fresh with inhibition
        engine2 = SwarmNociceptionEngine(num_agents=4, seed=42)
        engine2.apply_inhibition("agent-0", 0.8)
        engine2.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.8)
        engine2.do_tick()
        r2 = engine2.get_report()
        assert r2.agent_states["agent-0"].current_pain_level <= r1.agent_states["agent-0"].current_pain_level

    def test_descending_modulation(self):
        self.engine.set_descending_modulation("agent-0", -0.8)
        self.engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.8)
        self.engine.do_tick()
        report = self.engine.get_report()
        # Pain should be reduced due to descending suppression
        assert report.agent_states["agent-0"].current_pain_level < 0.8

    def test_timeline_grows(self):
        for _ in range(10):
            self.engine.do_tick()
        assert len(self.engine.pain_timeline) == 10

    def test_health_score_generated(self):
        self.engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.7)
        self.engine.do_tick()
        report = self.engine.get_report()
        assert 0 <= report.health.score <= 100
        assert report.health.tier in HealthTier

    def test_multiple_stimuli_compound(self):
        self.engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.7)
        self.engine.apply_stimulus("agent-0", StimulusType.THERMAL, 0.6)
        self.engine.do_tick()
        report = self.engine.get_report()
        # Should have higher pain than single stimulus
        assert report.agent_states["agent-0"].current_pain_level > 0

    def test_ignore_unknown_agent(self):
        # Should not raise
        self.engine.apply_stimulus("unknown-agent", StimulusType.MECHANICAL, 0.9)
        self.engine.do_tick()

    def test_subthreshold_no_pain(self):
        self.engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.1)
        self.engine.do_tick()
        report = self.engine.get_report()
        assert report.agent_states["agent-0"].current_pain_level == 0.0


# ---------------------------------------------------------------------------
# Scenario Tests
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_all_scenarios_defined(self):
        assert "baseline" in SCENARIOS
        assert "injury" in SCENARIOS
        assert "chronic" in SCENARIOS
        assert "adaptation" in SCENARIOS
        assert "cascade" in SCENARIOS

    def test_baseline_scenario(self):
        report = run_demo(num_agents=4, ticks=50, scenario="baseline", seed=42)
        assert report.health.score > 50

    def test_injury_scenario(self):
        report = run_demo(num_agents=4, ticks=40, scenario="injury", seed=42)
        # Injury should cause lower health
        assert report.health.score < 90

    def test_chronic_scenario(self):
        report = run_demo(num_agents=4, ticks=80, scenario="chronic", seed=42)
        assert len(report.all_memories) > 5

    def test_adaptation_scenario(self):
        report = run_demo(num_agents=4, ticks=70, scenario="adaptation", seed=42)
        # Should see tolerance building
        agent0 = report.agent_states["agent-0"]
        assert agent0.tolerance.habituation_events > 0


# ---------------------------------------------------------------------------
# Export Tests
# ---------------------------------------------------------------------------


class TestExports:
    def test_export_json(self):
        engine = SwarmNociceptionEngine(num_agents=3, seed=42)
        engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.7)
        engine.do_tick()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            engine.export_json(path)
            data = json.loads(Path(path).read_text())
            assert "health" in data
            assert "pain_timeline" in data
            assert data["health"]["score"] <= 100
        finally:
            os.unlink(path)

    def test_export_html(self):
        engine = SwarmNociceptionEngine(num_agents=3, seed=42)
        engine.apply_stimulus("agent-0", StimulusType.MECHANICAL, 0.7)
        engine.do_tick()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            engine.export_html(path)
            content = Path(path).read_text(encoding="utf-8")
            assert "Swarm Nociception Dashboard" in content
            assert "Health Score" in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_run_demo_returns_report(self):
        report = run_demo(num_agents=3, ticks=30, scenario="baseline", seed=42)
        assert isinstance(report, NociceptionReport)
        assert report.num_agents == 3
        assert report.total_ticks == 30

    def test_main_runs(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["nociception", "--agents", "3", "--ticks", "20", "--seed", "1"])
        main()  # Should not raise
