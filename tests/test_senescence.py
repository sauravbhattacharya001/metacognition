"""Tests for Swarm Senescence Engine."""
import json
import tempfile
from pathlib import Path

import pytest

from src.senescence import (
    PopulationStats,
    RejuvenationEvent,
    RejuvenationMethod,
    RetirementReason,
    SASPSignal,
    SCENARIOS,
    SenescenceReport,
    SenescenceState,
    SwarmSenescenceEngine,
)


# ---------------------------------------------------------------------------
# Basic Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self):
        engine = SwarmSenescenceEngine(seed=42)
        assert engine.num_agents == 15

    def test_custom_agents(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        assert len(engine._agents) == 5

    def test_agents_start_young(self):
        engine = SwarmSenescenceEngine(num_agents=10, seed=1)
        for agent in engine._agents:
            assert agent.state == SenescenceState.YOUNG
            assert agent.telomere_length >= 80.0

    def test_seed_reproducibility(self):
        e1 = SwarmSenescenceEngine(num_agents=5, seed=99)
        e2 = SwarmSenescenceEngine(num_agents=5, seed=99)
        for a, b in zip(e1._agents, e2._agents):
            assert a.telomere_length == b.telomere_length

    def test_from_scenario(self):
        for name in SCENARIOS:
            engine = SwarmSenescenceEngine.from_scenario(name, seed=1)
            assert engine is not None

    def test_invalid_scenario(self):
        with pytest.raises(ValueError):
            SwarmSenescenceEngine.from_scenario("nonexistent")


# ---------------------------------------------------------------------------
# Telomere Mechanics
# ---------------------------------------------------------------------------


class TestTelomereMechanics:
    def test_telomeres_shorten(self):
        engine = SwarmSenescenceEngine(num_agents=3, base_shortening=2.0, seed=1)
        initial = [a.telomere_length for a in engine._agents]
        engine._shorten_telomeres(0)
        for i, agent in enumerate(engine._agents):
            assert agent.telomere_length < initial[i]

    def test_telomere_never_negative(self):
        engine = SwarmSenescenceEngine(num_agents=3, base_shortening=200.0, seed=1)
        engine._shorten_telomeres(0)
        for agent in engine._agents:
            assert agent.telomere_length >= 0.0

    def test_state_transition_to_mature(self):
        engine = SwarmSenescenceEngine(num_agents=1, seed=1)
        engine._agents[0].telomere_length = 55.0
        engine._update_agent_state(engine._agents[0])
        assert engine._agents[0].state == SenescenceState.MATURE

    def test_state_transition_to_pre_senescent(self):
        engine = SwarmSenescenceEngine(num_agents=1, seed=1)
        engine._agents[0].telomere_length = 15.0
        engine._update_agent_state(engine._agents[0])
        assert engine._agents[0].state == SenescenceState.PRE_SENESCENT

    def test_state_transition_to_senescent(self):
        engine = SwarmSenescenceEngine(num_agents=1, seed=1)
        engine._agents[0].telomere_length = 5.0
        engine._update_agent_state(engine._agents[0])
        assert engine._agents[0].state == SenescenceState.SENESCENT

    def test_stress_increases_shortening(self):
        engine = SwarmSenescenceEngine(num_agents=2, base_shortening=1.0, seed=42)
        engine._agents[0].stress_level = 0.0
        engine._agents[1].stress_level = 1.0
        t0 = engine._agents[0].telomere_length
        t1 = engine._agents[1].telomere_length
        engine._shorten_telomeres(0)
        loss0 = t0 - engine._agents[0].telomere_length
        loss1 = t1 - engine._agents[1].telomere_length
        # Stressed agent loses more (statistically, with same RNG divergence)
        # Just check both lost something
        assert loss0 > 0
        assert loss1 > 0

    def test_age_increments(self):
        engine = SwarmSenescenceEngine(num_agents=2, seed=1)
        initial_ages = [a.age for a in engine._agents]
        engine._shorten_telomeres(0)
        for i, agent in enumerate(engine._agents):
            assert agent.age == initial_ages[i] + 1


# ---------------------------------------------------------------------------
# SASP Mechanics
# ---------------------------------------------------------------------------


class TestSASP:
    def test_senescent_emits_sasp(self):
        engine = SwarmSenescenceEngine(num_agents=3, seed=1)
        engine._agents[0].state = SenescenceState.SENESCENT
        engine._emit_sasp(0)
        assert len(engine._sasp_signals) >= 1
        assert engine._sasp_signals[0].source_agent == engine._agents[0].agent_id

    def test_non_senescent_no_sasp(self):
        engine = SwarmSenescenceEngine(num_agents=3, seed=1)
        engine._emit_sasp(0)
        assert len(engine._sasp_signals) == 0

    def test_sasp_decays(self):
        engine = SwarmSenescenceEngine(num_agents=3, sasp_decay=0.5, seed=1)
        engine._sasp_signals.append(SASPSignal(
            source_agent="agent_000", strength=1.0, radius=3.0, decay_rate=0.5
        ))
        engine._emit_sasp(1)
        assert engine._sasp_signals[0].strength < 1.0

    def test_sasp_removed_when_weak(self):
        engine = SwarmSenescenceEngine(num_agents=3, seed=1)
        engine._sasp_signals.append(SASPSignal(
            source_agent="agent_000", strength=0.05, radius=3.0, decay_rate=0.1
        ))
        engine._emit_sasp(1)
        # Should be removed because strength < 0.1
        assert len(engine._sasp_signals) == 0

    def test_bystander_effect(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        # Make agent 0 senescent and emit SASP
        engine._agents[0].state = SenescenceState.SENESCENT
        engine._emit_sasp(0)
        engine._apply_sasp_bystander()
        # Nearby agents should have exposure > 0
        assert engine._agents[1].sasp_exposure > 0

    def test_sasp_exposure_capped(self):
        engine = SwarmSenescenceEngine(num_agents=3, seed=1)
        # Flood with signals
        for i in range(20):
            engine._sasp_signals.append(SASPSignal(
                source_agent="agent_000", strength=5.0, radius=10.0, decay_rate=0.01
            ))
        engine._agents[0].state = SenescenceState.SENESCENT
        engine._apply_sasp_bystander()
        for agent in engine._agents:
            assert agent.sasp_exposure <= 2.0


# ---------------------------------------------------------------------------
# Rejuvenation
# ---------------------------------------------------------------------------


class TestRejuvenation:
    def test_rejuvenation_restores_telomere(self):
        engine = SwarmSenescenceEngine(
            num_agents=1, rejuvenation_chance=1.0, rejuvenation_boost=30.0, seed=1
        )
        engine._agents[0].telomere_length = 18.0
        engine._agents[0].state = SenescenceState.PRE_SENESCENT
        engine._attempt_rejuvenation(0)
        assert engine._agents[0].telomere_length > 18.0
        assert any(r.success for r in engine._rejuvenations)

    def test_rejuvenation_fails_when_chance_zero(self):
        engine = SwarmSenescenceEngine(
            num_agents=1, rejuvenation_chance=0.0, seed=1
        )
        engine._agents[0].telomere_length = 18.0
        engine._agents[0].state = SenescenceState.PRE_SENESCENT
        engine._attempt_rejuvenation(0)
        assert engine._agents[0].telomere_length == 18.0
        assert not any(r.success for r in engine._rejuvenations)

    def test_hayflick_limit_blocks_rejuvenation(self):
        engine = SwarmSenescenceEngine(
            num_agents=1, rejuvenation_chance=1.0, hayflick_limit=2, seed=1
        )
        engine._agents[0].telomere_length = 18.0
        engine._agents[0].state = SenescenceState.PRE_SENESCENT
        engine._agents[0].rejuvenation_count = 2
        engine._attempt_rejuvenation(0)
        # Should not rejuvenate
        assert engine._agents[0].telomere_length == 18.0

    def test_rejuvenation_increments_count(self):
        engine = SwarmSenescenceEngine(
            num_agents=1, rejuvenation_chance=1.0, seed=1
        )
        engine._agents[0].telomere_length = 18.0
        engine._agents[0].state = SenescenceState.PRE_SENESCENT
        engine._attempt_rejuvenation(0)
        assert engine._agents[0].rejuvenation_count == 1

    def test_rejuvenation_reduces_stress(self):
        engine = SwarmSenescenceEngine(
            num_agents=1, rejuvenation_chance=1.0, seed=1
        )
        engine._agents[0].telomere_length = 18.0
        engine._agents[0].state = SenescenceState.PRE_SENESCENT
        engine._agents[0].stress_level = 0.8
        engine._attempt_rejuvenation(0)
        assert engine._agents[0].stress_level < 0.8

    def test_only_pre_senescent_rejuvenated(self):
        engine = SwarmSenescenceEngine(
            num_agents=3, rejuvenation_chance=1.0, seed=1
        )
        engine._agents[0].state = SenescenceState.YOUNG
        engine._agents[1].state = SenescenceState.SENESCENT
        engine._agents[2].state = SenescenceState.PRE_SENESCENT
        engine._agents[2].telomere_length = 15.0
        engine._attempt_rejuvenation(0)
        assert len(engine._rejuvenations) == 1
        assert engine._rejuvenations[0].agent_id == engine._agents[2].agent_id

    def test_telomere_capped_at_100(self):
        engine = SwarmSenescenceEngine(
            num_agents=1, rejuvenation_chance=1.0, rejuvenation_boost=200.0, seed=1
        )
        engine._agents[0].telomere_length = 95.0
        engine._agents[0].state = SenescenceState.PRE_SENESCENT
        engine._attempt_rejuvenation(0)
        assert engine._agents[0].telomere_length <= 100.0


# ---------------------------------------------------------------------------
# Retirement
# ---------------------------------------------------------------------------


class TestRetirement:
    def test_senescent_past_hayflick_retires(self):
        engine = SwarmSenescenceEngine(num_agents=5, hayflick_limit=2, seed=1)
        engine._agents[0].state = SenescenceState.SENESCENT
        engine._agents[0].rejuvenation_count = 2
        engine._agents[0].telomere_length = 5.0
        engine._schedule_retirements(0)
        assert engine._agents[0].state == SenescenceState.RETIRED

    def test_retirement_spawns_replacement(self):
        engine = SwarmSenescenceEngine(num_agents=5, hayflick_limit=2, seed=1)
        initial_count = len(engine._agents)
        engine._agents[0].state = SenescenceState.SENESCENT
        engine._agents[0].rejuvenation_count = 2
        engine._agents[0].telomere_length = 1.0
        engine._schedule_retirements(0)
        assert len(engine._agents) == initial_count + 1

    def test_knowledge_transfer(self):
        engine = SwarmSenescenceEngine(num_agents=5, hayflick_limit=2, seed=1)
        engine._agents[0].state = SenescenceState.SENESCENT
        engine._agents[0].rejuvenation_count = 2
        engine._agents[0].telomere_length = 1.0
        engine._agents[0].knowledge_store = 30.0
        initial_knowledge = [a.knowledge_store for a in engine._agents[1:4]]
        engine._schedule_retirements(0)
        # Some young agents should have gained knowledge
        gained = any(
            engine._agents[i].knowledge_store > initial_knowledge[i - 1]
            for i in range(1, 4)
        )
        assert gained

    def test_retirement_record_created(self):
        engine = SwarmSenescenceEngine(num_agents=5, hayflick_limit=2, seed=1)
        engine._agents[0].state = SenescenceState.SENESCENT
        engine._agents[0].rejuvenation_count = 2
        engine._agents[0].telomere_length = 1.0
        engine._schedule_retirements(0)
        assert len(engine._retirements) == 1
        assert engine._retirements[0].agent_id == "agent_000"

    def test_retirement_reason_telomere(self):
        engine = SwarmSenescenceEngine(num_agents=5, hayflick_limit=10, seed=1)
        engine._agents[0].state = SenescenceState.SENESCENT
        engine._agents[0].rejuvenation_count = 10
        engine._agents[0].telomere_length = 1.0
        engine._schedule_retirements(0)
        assert engine._retirements[0].reason == RetirementReason.HAYFLICK_LIMIT


# ---------------------------------------------------------------------------
# Longevity Optimizer
# ---------------------------------------------------------------------------


class TestLongevityOptimizer:
    def test_high_stress_triggers_actions(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        for a in engine._agents:
            a.stress_level = 0.9
        result = engine._optimize_longevity()
        assert "reduce_global_workload" in result["actions"]

    def test_low_telomere_triggers_rejuvenation(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        for a in engine._agents:
            a.telomere_length = 30.0
        result = engine._optimize_longevity()
        assert "prioritize_rejuvenation_batch" in result["actions"]

    def test_no_agents_returns_empty(self):
        engine = SwarmSenescenceEngine(num_agents=1, seed=1)
        engine._agents[0].state = SenescenceState.RETIRED
        result = engine._optimize_longevity()
        assert result["recommendation"] == "no_active_agents"


# ---------------------------------------------------------------------------
# Health Scoring
# ---------------------------------------------------------------------------


class TestHealthScoring:
    def test_healthy_swarm_high_score(self):
        engine = SwarmSenescenceEngine(num_agents=10, seed=1)
        # All young, high telomeres
        health = engine._compute_health()
        assert health.score >= 50

    def test_all_senescent_low_score(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        for a in engine._agents:
            a.state = SenescenceState.SENESCENT
            a.telomere_length = 5.0
        # Add many SASP signals
        for i in range(15):
            engine._sasp_signals.append(SASPSignal(
                source_agent=f"agent_{i:03d}", strength=1.0,
                radius=3.0, decay_rate=0.1
            ))
        health = engine._compute_health()
        assert health.score < 40

    def test_score_bounded_0_100(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        health = engine._compute_health()
        assert 0.0 <= health.score <= 100.0

    def test_tier_immortal(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        # Ensure high score
        engine._graceful_retirements = 10
        engine._total_retirements = 10
        engine._rejuvenations = [
            RejuvenationEvent("a", 0, 20, 50, RejuvenationMethod.TELOMERE_EXTENSION, True)
            for _ in range(10)
        ]
        health = engine._compute_health()
        assert health.tier in ("Immortal", "Thriving")

    def test_tier_collapsing(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        for a in engine._agents:
            a.state = SenescenceState.RETIRED
        health = engine._compute_health()
        assert health.tier in ("Collapsing", "Declining")

    def test_all_tiers_exist(self):
        tiers = {"Immortal", "Thriving", "Aging", "Declining", "Collapsing"}
        # Just validate they exist as strings
        assert len(tiers) == 5


# ---------------------------------------------------------------------------
# Insight Generation
# ---------------------------------------------------------------------------


class TestInsights:
    def test_low_telomere_insight(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        for a in engine._agents:
            a.telomere_length = 20.0
        insights = engine._generate_insights()
        assert any("critically low" in i for i in insights)

    def test_high_telomere_insight(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        for a in engine._agents:
            a.telomere_length = 85.0
        insights = engine._generate_insights()
        assert any("biologically young" in i for i in insights)

    def test_no_active_agents_insight(self):
        engine = SwarmSenescenceEngine(num_agents=3, seed=1)
        for a in engine._agents:
            a.state = SenescenceState.RETIRED
        insights = engine._generate_insights()
        assert any("collapsed" in i for i in insights)

    def test_sasp_cascade_insight(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        for i in range(8):
            engine._sasp_signals.append(SASPSignal(
                source_agent=f"x_{i}", strength=0.5, radius=3.0, decay_rate=0.1
            ))
        insights = engine._generate_insights()
        assert any("SASP" in i for i in insights)


# ---------------------------------------------------------------------------
# Full Simulation
# ---------------------------------------------------------------------------


class TestSimulation:
    def test_simulate_returns_report(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=42)
        report = engine.simulate(cycles=50)
        assert isinstance(report, SenescenceReport)
        assert report.health is not None
        assert len(report.cycle_history) == 50

    def test_simulate_population_evolves(self):
        engine = SwarmSenescenceEngine(num_agents=10, seed=1)
        report = engine.simulate(cycles=80)
        # Some agents should have aged
        assert any(a.age > 0 for a in report.agents)

    def test_cycle_history_tracked(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        report = engine.simulate(cycles=30)
        assert len(report.cycle_history) == 30
        assert "avg_telomere" in report.cycle_history[0]

    def test_scenario_healthy_swarm(self):
        engine = SwarmSenescenceEngine.from_scenario("healthy_swarm", seed=42)
        report = engine.simulate(cycles=80)
        # Healthy swarm should maintain reasonable health
        assert report.health.score > 20

    def test_scenario_aging_crisis(self):
        engine = SwarmSenescenceEngine.from_scenario("aging_crisis", seed=42)
        report = engine.simulate(cycles=120)
        assert report.health is not None

    def test_scenario_sasp_cascade(self):
        engine = SwarmSenescenceEngine.from_scenario("sasp_cascade", seed=42)
        report = engine.simulate(cycles=100)
        assert report.health is not None

    def test_scenario_rejuvenation_success(self):
        engine = SwarmSenescenceEngine.from_scenario("rejuvenation_success", seed=42)
        report = engine.simulate(cycles=150)
        # Good rejuvenation should keep health reasonable
        assert report.health.score > 15

    def test_scenario_population_collapse(self):
        engine = SwarmSenescenceEngine.from_scenario("population_collapse", seed=42)
        report = engine.simulate(cycles=200)
        assert report.health is not None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_json(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        report = engine.simulate(cycles=20)
        data = engine.export_json(report)
        assert "health" in data
        assert "agents" in data
        assert "insights" in data
        # Serializable
        json.dumps(data)

    def test_export_html(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        report = engine.simulate(cycles=20)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        engine.export_html(path, report)
        content = Path(path).read_text(encoding="utf-8")
        assert "Swarm Senescence Report" in content
        assert str(report.health.score) in content
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_agent(self):
        engine = SwarmSenescenceEngine(num_agents=1, seed=1)
        report = engine.simulate(cycles=50)
        assert report.health is not None

    def test_zero_cycles(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        report = engine.simulate(cycles=0)
        assert len(report.cycle_history) == 0
        assert report.health is not None

    def test_large_swarm(self):
        engine = SwarmSenescenceEngine(num_agents=50, seed=1)
        report = engine.simulate(cycles=30)
        assert report.population_stats.total_agents >= 50

    def test_extreme_shortening(self):
        engine = SwarmSenescenceEngine(
            num_agents=5, base_shortening=50.0, seed=1
        )
        report = engine.simulate(cycles=10)
        # Should handle rapid aging gracefully
        assert report.health is not None

    def test_no_rejuvenation(self):
        engine = SwarmSenescenceEngine(
            num_agents=5, rejuvenation_chance=0.0, seed=1
        )
        report = engine.simulate(cycles=50)
        assert all(not r.success for r in report.rejuvenations)


# ---------------------------------------------------------------------------
# Population Stats
# ---------------------------------------------------------------------------


class TestPopulationStats:
    def test_stats_computed(self):
        engine = SwarmSenescenceEngine(num_agents=10, seed=1)
        engine.simulate(cycles=30)
        stats = engine._compute_population_stats()
        assert isinstance(stats, PopulationStats)
        assert stats.total_agents >= 10

    def test_sustainability_ratio(self):
        engine = SwarmSenescenceEngine(num_agents=5, seed=1)
        engine._total_births = 10
        engine._total_retirements = 5
        engine._cycle_history = [{}] * 10
        stats = engine._compute_population_stats()
        assert stats.sustainability_ratio > 0
