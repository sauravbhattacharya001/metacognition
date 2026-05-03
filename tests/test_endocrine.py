"""Tests for Swarm Endocrine Engine."""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.endocrine import (
    BloodstreamSimulator,
    BloodstreamSnapshot,
    CascadeEvent,
    EndocrineHealthScorer,
    EndocrineReport,
    EventType,
    FeedbackEvent,
    FeedbackLoopRegulator,
    FeedbackType,
    GlandController,
    HealthScore,
    HealthTier,
    HormoneLevel,
    HormoneType,
    HormonalCascadeEngine,
    HORMONE_PROFILES,
    EVENT_HORMONE_MAP,
    CASCADE_RULES,
    FEEDBACK_RULES,
    Insight,
    InsightGenerator,
    InsightSeverity,
    ReceptorBindingEngine,
    ReceptorState,
    SCENARIOS,
    SwarmEndocrineEngine,
    AgentEndocrineState,
    run_demo,
    main,
)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_hormone_types(self):
        assert len(HormoneType) == 7
        assert HormoneType.CORTISOL.value == "cortisol"
        assert HormoneType.GROWTH_HORMONE.value == "growth_hormone"

    def test_event_types(self):
        assert len(EventType) == 7
        assert EventType.TASK_SUCCESS.value == "task_success"
        assert EventType.RESOURCE_SHORTAGE.value == "resource_shortage"

    def test_feedback_types(self):
        assert FeedbackType.NEGATIVE.value == "negative"
        assert FeedbackType.POSITIVE.value == "positive"

    def test_health_tiers(self):
        assert len(HealthTier) == 5
        assert HealthTier.OPTIMAL.value == "optimal"
        assert HealthTier.CRITICAL.value == "critical"

    def test_insight_severity(self):
        assert InsightSeverity.INFO.value == "info"
        assert InsightSeverity.CRITICAL.value == "critical"


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_hormone_profiles_complete(self):
        for h in HormoneType:
            assert h in HORMONE_PROFILES
            profile = HORMONE_PROFILES[h]
            assert "half_life" in profile
            assert "potency" in profile
            assert "decay_rate" in profile
            assert "baseline" in profile
            assert profile["half_life"] > 0
            assert profile["decay_rate"] > 0

    def test_event_hormone_map_complete(self):
        for e in EventType:
            assert e in EVENT_HORMONE_MAP
            mappings = EVENT_HORMONE_MAP[e]
            assert len(mappings) > 0
            for hormone, rate in mappings:
                assert isinstance(hormone, HormoneType)
                assert rate > 0

    def test_scenarios_exist(self):
        assert "default" in SCENARIOS
        assert "stress" in SCENARIOS
        assert "reward" in SCENARIOS
        assert "collaboration" in SCENARIOS
        assert "dysregulated" in SCENARIOS


# ---------------------------------------------------------------------------
# Gland Controller Tests
# ---------------------------------------------------------------------------

class TestGlandController:
    def test_register_agent(self):
        gc = GlandController()
        gc.register_agent("a1")
        assert "a1" in gc.agent_glands
        assert len(gc.agent_glands["a1"]) == 7

    def test_produce_returns_hormones(self):
        gc = GlandController()
        gc.register_agent("a1")
        produced = gc.produce("a1", EventType.TASK_SUCCESS, 1.0)
        assert HormoneType.DOPAMINE in produced
        assert produced[HormoneType.DOPAMINE] > 0

    def test_produce_auto_registers(self):
        gc = GlandController()
        produced = gc.produce("new_agent", EventType.TASK_FAILURE, 0.5)
        assert "new_agent" in gc.agent_glands
        assert HormoneType.CORTISOL in produced

    def test_magnitude_scales_production(self):
        gc = GlandController()
        p1 = gc.produce("a1", EventType.TASK_SUCCESS, 0.5)
        gc2 = GlandController()
        p2 = gc2.produce("a1", EventType.TASK_SUCCESS, 1.0)
        assert p2[HormoneType.DOPAMINE] > p1[HormoneType.DOPAMINE]

    def test_habituation_decreases_production(self):
        gc = GlandController()
        gc.register_agent("a1")
        p1 = gc.produce("a1", EventType.TASK_SUCCESS, 1.0)
        p2 = gc.produce("a1", EventType.TASK_SUCCESS, 1.0)
        # Second production should be lower due to habituation
        assert p2[HormoneType.DOPAMINE] < p1[HormoneType.DOPAMINE]

    def test_habituation_recovery(self):
        gc = GlandController()
        gc.register_agent("a1")
        gc.produce("a1", EventType.TASK_SUCCESS, 1.0)
        initial_hab = gc.habituation["a1"][HormoneType.DOPAMINE]
        gc.recover_habituation(dt=10.0)
        recovered_hab = gc.habituation["a1"][HormoneType.DOPAMINE]
        assert recovered_hab > initial_hab

    def test_feedback_suppression(self):
        gc = GlandController()
        gc.register_agent("a1")
        gc.produce("a1", EventType.TASK_FAILURE, 1.0)
        before = gc.agent_glands["a1"][HormoneType.CORTISOL]
        gc.apply_feedback_suppression("a1", HormoneType.CORTISOL, 0.5)
        after = gc.agent_glands["a1"][HormoneType.CORTISOL]
        assert after < before

    def test_get_agent_production(self):
        gc = GlandController()
        gc.register_agent("a1")
        gc.produce("a1", EventType.LEARNING, 1.0)
        prod = gc.get_agent_production("a1")
        assert HormoneType.GROWTH_HORMONE in prod


# ---------------------------------------------------------------------------
# Bloodstream Simulator Tests
# ---------------------------------------------------------------------------

class TestBloodstreamSimulator:
    def test_initial_concentrations_at_baseline(self):
        bs = BloodstreamSimulator()
        for h in HormoneType:
            assert bs.get_concentration(h) == pytest.approx(HORMONE_PROFILES[h]["baseline"])

    def test_release_increases_concentration(self):
        bs = BloodstreamSimulator()
        before = bs.get_concentration(HormoneType.CORTISOL)
        bs.release(HormoneType.CORTISOL, 1.0)
        after = bs.get_concentration(HormoneType.CORTISOL)
        assert after == pytest.approx(before + 1.0)

    def test_decay_reduces_concentration(self):
        bs = BloodstreamSimulator()
        bs.release(HormoneType.ADRENALINE, 5.0)
        before = bs.get_concentration(HormoneType.ADRENALINE)
        bs.tick(dt=1.0)
        after = bs.get_concentration(HormoneType.ADRENALINE)
        assert after < before

    def test_exponential_decay(self):
        bs = BloodstreamSimulator()
        bs.concentrations[HormoneType.CORTISOL] = 10.0
        rate = HORMONE_PROFILES[HormoneType.CORTISOL]["decay_rate"]
        bs.tick(dt=1.0)
        expected = 10.0 * math.exp(-rate)
        assert bs.get_concentration(HormoneType.CORTISOL) == pytest.approx(expected, rel=1e-4)

    def test_history_recorded(self):
        bs = BloodstreamSimulator()
        bs.tick()
        bs.tick()
        bs.tick()
        assert len(bs.history) == 3

    def test_snapshot_has_concentrations(self):
        bs = BloodstreamSimulator()
        snap = bs.tick()
        assert isinstance(snap, BloodstreamSnapshot)
        assert len(snap.concentrations) == 7

    def test_get_all_concentrations(self):
        bs = BloodstreamSimulator()
        concs = bs.get_all_concentrations()
        assert len(concs) == 7
        assert all(isinstance(k, str) for k in concs)

    def test_deviation_from_baseline(self):
        bs = BloodstreamSimulator()
        bs.release(HormoneType.CORTISOL, 5.0)
        devs = bs.deviation_from_baseline()
        assert devs[HormoneType.CORTISOL] > 0
        assert devs[HormoneType.DOPAMINE] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Receptor Binding Engine Tests
# ---------------------------------------------------------------------------

class TestReceptorBindingEngine:
    def test_register_agent(self):
        rbe = ReceptorBindingEngine()
        rbe.register_agent("a1")
        assert "a1" in rbe.agent_receptors
        assert len(rbe.agent_receptors["a1"]) == 7

    def test_bind_returns_values(self):
        rbe = ReceptorBindingEngine()
        rbe.register_agent("a1")
        concs = {h: 1.0 for h in HormoneType}
        bound = rbe.bind("a1", concs)
        assert len(bound) == 7
        for h, val in bound.items():
            assert 0.0 <= val <= 1.0

    def test_hill_equation_zero_concentration(self):
        rbe = ReceptorBindingEngine()
        rbe.register_agent("a1")
        concs = {h: 0.0 for h in HormoneType}
        bound = rbe.bind("a1", concs)
        for h, val in bound.items():
            assert val == 0.0

    def test_hill_equation_high_concentration(self):
        rbe = ReceptorBindingEngine()
        rbe.register_agent("a1")
        concs = {h: 100.0 for h in HormoneType}
        bound = rbe.bind("a1", concs)
        for h, val in bound.items():
            assert val > 0.9  # should be near saturation

    def test_sensitivity_downregulation(self):
        rbe = ReceptorBindingEngine()
        rbe.register_agent("a1")
        initial = rbe.agent_receptors["a1"][HormoneType.CORTISOL].sensitivity
        # High concentration
        concs = {h: 0.0 for h in HormoneType}
        concs[HormoneType.CORTISOL] = 10.0  # way above baseline * 1.5
        rbe.adapt_receptors("a1", concs)
        after = rbe.agent_receptors["a1"][HormoneType.CORTISOL].sensitivity
        assert after < initial

    def test_sensitivity_upregulation(self):
        rbe = ReceptorBindingEngine()
        rbe.register_agent("a1")
        initial = rbe.agent_receptors["a1"][HormoneType.CORTISOL].sensitivity
        # Very low concentration (below baseline * 0.5)
        concs = {h: 0.0 for h in HormoneType}
        concs[HormoneType.CORTISOL] = 0.01
        rbe.adapt_receptors("a1", concs)
        after = rbe.agent_receptors["a1"][HormoneType.CORTISOL].sensitivity
        assert after > initial

    def test_auto_register_on_bind(self):
        rbe = ReceptorBindingEngine()
        concs = {h: 0.5 for h in HormoneType}
        bound = rbe.bind("new_agent", concs)
        assert "new_agent" in rbe.agent_receptors

    def test_get_receptor_states(self):
        rbe = ReceptorBindingEngine()
        rbe.register_agent("a1")
        states = rbe.get_receptor_states("a1")
        assert len(states) == 7
        for h, s in states.items():
            assert isinstance(s, ReceptorState)


# ---------------------------------------------------------------------------
# Feedback Loop Regulator Tests
# ---------------------------------------------------------------------------

class TestFeedbackLoopRegulator:
    def test_no_feedback_at_low_concentrations(self):
        flr = FeedbackLoopRegulator()
        concs = {h: 0.0 for h in HormoneType}
        events = flr.evaluate(concs)
        assert len(events) == 0

    def test_negative_feedback_cortisol(self):
        flr = FeedbackLoopRegulator()
        concs = {h: 0.0 for h in HormoneType}
        concs[HormoneType.CORTISOL] = 2.0  # well above threshold 0.7
        events = flr.evaluate(concs)
        cortisol_fb = [e for e in events if e.hormone == "cortisol"]
        assert len(cortisol_fb) > 0
        assert cortisol_fb[0].feedback_type == "negative"

    def test_positive_feedback_oxytocin(self):
        flr = FeedbackLoopRegulator()
        concs = {h: 0.0 for h in HormoneType}
        concs[HormoneType.OXYTOCIN] = 1.5  # above threshold 0.5
        events = flr.evaluate(concs)
        oxy_fb = [e for e in events if e.hormone == "oxytocin"]
        assert len(oxy_fb) > 0
        assert oxy_fb[0].feedback_type == "positive"

    def test_events_accumulated(self):
        flr = FeedbackLoopRegulator()
        concs = {h: 5.0 for h in HormoneType}
        flr.evaluate(concs, tick=1)
        flr.evaluate(concs, tick=2)
        assert len(flr.events) > 0

    def test_tick_recorded(self):
        flr = FeedbackLoopRegulator()
        concs = {h: 5.0 for h in HormoneType}
        events = flr.evaluate(concs, tick=42)
        for e in events:
            assert e.tick == 42


# ---------------------------------------------------------------------------
# Hormonal Cascade Engine Tests
# ---------------------------------------------------------------------------

class TestHormonalCascadeEngine:
    def test_no_cascade_at_low_levels(self):
        hce = HormonalCascadeEngine()
        concs = {h: 0.0 for h in HormoneType}
        events = hce.evaluate(concs)
        assert len(events) == 0

    def test_cortisol_triggers_adrenaline_cascade(self):
        hce = HormonalCascadeEngine()
        concs = {h: 0.0 for h in HormoneType}
        concs[HormoneType.CORTISOL] = 2.0  # above threshold 0.6
        events = hce.evaluate(concs)
        cort_cascades = [e for e in events if e.trigger_hormone == "cortisol"
                         and e.triggered_hormone == "adrenaline"]
        assert len(cort_cascades) > 0

    def test_cascade_depth_limited(self):
        hce = HormonalCascadeEngine()
        concs = {h: 5.0 for h in HormoneType}  # high everything
        events = hce.evaluate(concs)
        for e in events:
            assert e.depth <= HormonalCascadeEngine.MAX_CASCADE_DEPTH

    def test_cascade_events_accumulated(self):
        hce = HormonalCascadeEngine()
        concs = {h: 3.0 for h in HormoneType}
        hce.evaluate(concs, tick=1)
        hce.evaluate(concs, tick=2)
        assert len(hce.events) > 0

    def test_cascade_tick_recorded(self):
        hce = HormonalCascadeEngine()
        concs = {h: 3.0 for h in HormoneType}
        events = hce.evaluate(concs, tick=99)
        for e in events:
            assert e.tick == 99


# ---------------------------------------------------------------------------
# Health Scorer Tests
# ---------------------------------------------------------------------------

class TestHealthScorer:
    def test_perfect_health(self):
        concs = {h: HORMONE_PROFILES[h]["baseline"] for h in HormoneType}
        receptors = {"a1": {h: ReceptorState(hormone_type=h) for h in HormoneType}}
        health = EndocrineHealthScorer.score(concs, receptors, [], [], 10)
        assert health.score >= 70
        assert health.tier in (HealthTier.OPTIMAL, HealthTier.BALANCED)

    def test_stressed_health(self):
        concs = {h: HORMONE_PROFILES[h]["baseline"] * 4 for h in HormoneType}
        receptors = {"a1": {h: ReceptorState(hormone_type=h, sensitivity=0.3) for h in HormoneType}}
        fb = [FeedbackEvent("cortisol", "negative", 1.0, t) for t in range(30)]
        cascades = [CascadeEvent("cortisol", "adrenaline", 0.5, 3, t) for t in range(20)]
        health = EndocrineHealthScorer.score(concs, receptors, fb, cascades, 10)
        assert health.score < 60

    def test_tier_boundaries(self):
        # Verify tier classification
        h = HealthScore(score=85, tier=HealthTier.OPTIMAL)
        assert h.tier == HealthTier.OPTIMAL
        h2 = HealthScore(score=15, tier=HealthTier.CRITICAL)
        assert h2.tier == HealthTier.CRITICAL

    def test_recommendations_generated(self):
        concs = {h: HORMONE_PROFILES[h]["baseline"] * 5 for h in HormoneType}
        receptors = {"a1": {h: ReceptorState(hormone_type=h, sensitivity=0.2) for h in HormoneType}}
        health = EndocrineHealthScorer.score(concs, receptors, [], [], 10)
        assert len(health.recommendations) > 0

    def test_score_in_range(self):
        concs = {h: 0.0 for h in HormoneType}
        health = EndocrineHealthScorer.score(concs, {}, [], [], 0)
        assert 0.0 <= health.score <= 100.0


# ---------------------------------------------------------------------------
# Insight Generator Tests
# ---------------------------------------------------------------------------

class TestInsightGenerator:
    def test_chronic_stress_detection(self):
        concs = {h: HORMONE_PROFILES[h]["baseline"] for h in HormoneType}
        concs[HormoneType.CORTISOL] = 2.0  # > baseline * 2.5 (0.3 * 2.5 = 0.75)
        health = HealthScore(score=50)
        insights = InsightGenerator.generate(concs, [], [], [], health)
        stress = [i for i in insights if i.category == "chronic_stress"]
        assert len(stress) > 0

    def test_reward_deficiency_detection(self):
        concs = {h: HORMONE_PROFILES[h]["baseline"] for h in HormoneType}
        concs[HormoneType.DOPAMINE] = 0.01  # < baseline * 0.3
        health = HealthScore(score=50)
        insights = InsightGenerator.generate(concs, [], [], [], health)
        reward = [i for i in insights if i.category == "reward_deficiency"]
        assert len(reward) > 0

    def test_bonding_gap_detection(self):
        concs = {h: HORMONE_PROFILES[h]["baseline"] for h in HormoneType}
        concs[HormoneType.OXYTOCIN] = 0.01
        health = HealthScore(score=50)
        insights = InsightGenerator.generate(concs, [], [], [], health)
        bonding = [i for i in insights if i.category == "bonding_gap"]
        assert len(bonding) > 0

    def test_no_insights_at_baseline(self):
        concs = {h: HORMONE_PROFILES[h]["baseline"] for h in HormoneType}
        health = HealthScore(score=80)
        insights = InsightGenerator.generate(concs, [], [], [], health)
        # Should have no insights at baseline levels with good health
        assert len(insights) == 0

    def test_adrenaline_surge_detection(self):
        concs = {h: HORMONE_PROFILES[h]["baseline"] for h in HormoneType}
        concs[HormoneType.ADRENALINE] = 1.0  # > baseline (0.1) * 5
        health = HealthScore(score=50)
        insights = InsightGenerator.generate(concs, [], [], [], health)
        surge = [i for i in insights if i.category == "adrenaline_surge"]
        assert len(surge) > 0

    def test_cascade_runaway_detection(self):
        concs = {h: HORMONE_PROFILES[h]["baseline"] for h in HormoneType}
        cascades = [CascadeEvent("cortisol", "adrenaline", 0.5, 3, t) for t in range(10)]
        health = HealthScore(score=50)
        insights = InsightGenerator.generate(concs, [], [], cascades, health)
        runaway = [i for i in insights if i.category == "cascade_runaway"]
        assert len(runaway) > 0

    def test_critical_health_insight(self):
        concs = {h: HORMONE_PROFILES[h]["baseline"] for h in HormoneType}
        health = HealthScore(score=20)
        insights = InsightGenerator.generate(concs, [], [], [], health)
        sys_health = [i for i in insights if i.category == "system_health"]
        assert len(sys_health) > 0


# ---------------------------------------------------------------------------
# Swarm Endocrine Engine Tests
# ---------------------------------------------------------------------------

class TestSwarmEndocrineEngine:
    def test_creation(self):
        engine = SwarmEndocrineEngine(num_agents=3)
        assert engine.num_agents == 3
        assert len(engine.agent_ids) == 3

    def test_min_agents(self):
        engine = SwarmEndocrineEngine(num_agents=0)
        assert engine.num_agents == 1  # minimum 1

    def test_inject_event(self):
        engine = SwarmEndocrineEngine(num_agents=3)
        produced = engine.inject_event("agent-0", EventType.TASK_SUCCESS, 1.0)
        assert HormoneType.DOPAMINE in produced

    def test_tick_returns_snapshot(self):
        engine = SwarmEndocrineEngine(num_agents=3)
        engine.inject_event("agent-0", EventType.TASK_FAILURE, 1.0)
        snap = engine.tick()
        assert isinstance(snap, BloodstreamSnapshot)
        assert snap.tick == 1

    def test_multiple_ticks(self):
        engine = SwarmEndocrineEngine(num_agents=3)
        for _ in range(10):
            engine.tick()
        assert engine.tick_count == 10

    def test_get_report(self):
        engine = SwarmEndocrineEngine(num_agents=3, seed=42)
        engine.inject_event("agent-0", EventType.TASK_SUCCESS, 1.0)
        engine.inject_event("agent-1", EventType.TASK_FAILURE, 0.8)
        for _ in range(20):
            engine.tick()
        report = engine.get_report()
        assert isinstance(report, EndocrineReport)
        assert report.tick_count == 20
        assert report.num_agents == 3
        assert 0 <= report.health.score <= 100
        assert len(report.agent_states) == 3

    def test_stress_scenario_high_cortisol(self):
        engine = SwarmEndocrineEngine(num_agents=3, seed=42)
        for _ in range(5):
            engine.inject_event("agent-0", EventType.TASK_FAILURE, 1.0)
            engine.inject_event("agent-1", EventType.HIGH_LOAD, 1.0)
            engine.tick()
        report = engine.get_report()
        cortisol = report.current_concentrations.get("cortisol", 0)
        assert cortisol > HORMONE_PROFILES[HormoneType.CORTISOL]["baseline"]

    def test_collaboration_scenario_high_oxytocin(self):
        engine = SwarmEndocrineEngine(num_agents=3, seed=42)
        for _ in range(5):
            engine.inject_event("agent-0", EventType.COLLABORATION, 1.0)
            engine.inject_event("agent-1", EventType.COLLABORATION, 1.0)
            engine.tick()
        report = engine.get_report()
        oxytocin = report.current_concentrations.get("oxytocin", 0)
        assert oxytocin > HORMONE_PROFILES[HormoneType.OXYTOCIN]["baseline"]

    def test_export_html(self, tmp_path):
        engine = SwarmEndocrineEngine(num_agents=3, seed=42)
        for _ in range(10):
            engine.inject_event("agent-0", EventType.TASK_SUCCESS, 1.0)
            engine.tick()
        out = str(tmp_path / "endocrine.html")
        engine.export_html(out)
        assert Path(out).exists()
        content = Path(out).read_text(encoding="utf-8")
        assert "Swarm Endocrine Engine Dashboard" in content
        assert "chart.js" in content.lower() or "Chart" in content

    def test_export_json(self, tmp_path):
        engine = SwarmEndocrineEngine(num_agents=3, seed=42)
        for _ in range(10):
            engine.inject_event("agent-0", EventType.LEARNING, 1.0)
            engine.tick()
        out = str(tmp_path / "endocrine.json")
        engine.export_json(out)
        assert Path(out).exists()
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert "health" in data
        assert "current_concentrations" in data
        assert data["tick_count"] == 10

    def test_seed_reproducibility(self):
        engine1 = SwarmEndocrineEngine(num_agents=3, seed=123)
        engine2 = SwarmEndocrineEngine(num_agents=3, seed=123)
        for _ in range(5):
            engine1.inject_event("agent-0", EventType.TASK_SUCCESS, 1.0)
            engine2.inject_event("agent-0", EventType.TASK_SUCCESS, 1.0)
            engine1.tick()
            engine2.tick()
        r1 = engine1.get_report()
        r2 = engine2.get_report()
        assert r1.health.score == r2.health.score


# ---------------------------------------------------------------------------
# Run Demo Tests
# ---------------------------------------------------------------------------

class TestRunDemo:
    def test_default_demo(self):
        report = run_demo(num_agents=3, ticks=50, seed=42)
        assert isinstance(report, EndocrineReport)
        assert report.tick_count == 50

    def test_stress_demo(self):
        report = run_demo(num_agents=3, ticks=50, scenario="stress", seed=42)
        assert report.health.score < 90  # stress should lower health

    def test_reward_demo(self):
        report = run_demo(num_agents=3, ticks=50, scenario="reward", seed=42)
        assert isinstance(report, EndocrineReport)

    def test_collaboration_demo(self):
        report = run_demo(num_agents=3, ticks=50, scenario="collaboration", seed=42)
        assert isinstance(report, EndocrineReport)

    def test_dysregulated_demo(self):
        report = run_demo(num_agents=3, ticks=50, scenario="dysregulated", seed=42)
        assert isinstance(report, EndocrineReport)


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_main_default(self, capsys):
        main(["--agents", "3", "--ticks", "20", "--seed", "42"])
        captured = capsys.readouterr()
        assert "Endocrine Health" in captured.out
        assert "Hormone Levels" in captured.out

    def test_main_stress_scenario(self, capsys):
        main(["--agents", "3", "--ticks", "20", "--scenario", "stress", "--seed", "42"])
        captured = capsys.readouterr()
        assert "Endocrine Health" in captured.out

    def test_main_with_html_output(self, tmp_path, capsys):
        out = str(tmp_path / "test.html")
        main(["--agents", "3", "--ticks", "20", "--out", out, "--seed", "42"])
        assert Path(out).exists()

    def test_main_with_json_output(self, tmp_path, capsys):
        out = str(tmp_path / "test.json")
        main(["--agents", "3", "--ticks", "20", "--json", out, "--seed", "42"])
        assert Path(out).exists()
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert "health" in data
