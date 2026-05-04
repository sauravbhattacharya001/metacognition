"""Tests for Swarm Hibernation Engine."""
import json
import math
import tempfile
from pathlib import Path

import pytest

from src.hibernation import (
    AgentEnergyState,
    ArousalEvent,
    ArousalTrigger,
    HealthScore,
    HibernationCluster,
    HibernationReport,
    HibernationState,
    SCENARIOS,
    ScarcityEvent,
    ScarcityLevel,
    SwarmHibernationEngine,
    TorporBout,
    _METABOLIC_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Basic Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self):
        engine = SwarmHibernationEngine(seed=42)
        assert engine.num_agents == 12

    def test_custom_agents(self):
        engine = SwarmHibernationEngine(num_agents=5, seed=1)
        assert len(engine._agents) == 5

    def test_agents_start_active(self):
        engine = SwarmHibernationEngine(num_agents=10, seed=1)
        for agent in engine._agents:
            assert agent.state == HibernationState.ACTIVE
            assert agent.energy >= 70.0

    def test_seed_reproducibility(self):
        e1 = SwarmHibernationEngine(num_agents=5, seed=99)
        e2 = SwarmHibernationEngine(num_agents=5, seed=99)
        for a, b in zip(e1._agents, e2._agents):
            assert a.energy == b.energy
            assert a.metabolic_rate == b.metabolic_rate

    def test_from_scenario(self):
        for name in SCENARIOS:
            engine = SwarmHibernationEngine.from_scenario(name, seed=1)
            assert engine is not None

    def test_invalid_scenario(self):
        with pytest.raises(ValueError):
            SwarmHibernationEngine.from_scenario("nonexistent")

    def test_initial_energy_range(self):
        engine = SwarmHibernationEngine(num_agents=20, seed=1)
        for a in engine._agents:
            assert 70.0 <= a.energy <= 100.0

    def test_initial_metabolic_rate(self):
        engine = SwarmHibernationEngine(num_agents=10, base_metabolic_rate=2.0, seed=1)
        for a in engine._agents:
            assert 1.6 <= a.metabolic_rate <= 2.4


# ---------------------------------------------------------------------------
# Energy Budget
# ---------------------------------------------------------------------------


class TestEnergyBudget:
    def test_energy_decreases_over_time(self):
        engine = SwarmHibernationEngine(num_agents=3, seed=42)
        initial = [a.energy for a in engine._agents]
        for _ in range(10):
            engine.tick()
        for i, a in enumerate(engine._agents):
            assert a.energy < initial[i]

    def test_energy_never_negative(self):
        engine = SwarmHibernationEngine(num_agents=3, base_metabolic_rate=5.0, seed=1)
        for _ in range(200):
            engine.tick()
        for a in engine._agents:
            assert a.energy >= 0.0

    def test_high_resources_provide_recovery(self):
        engine = SwarmHibernationEngine(num_agents=3, seed=42)
        engine.set_resource_level(0.95)
        initial = engine._agents[0].energy
        # Run a few ticks — high resources help active agents
        for _ in range(5):
            engine.tick()
        # With high resources, energy loss should be modest
        assert engine._agents[0].energy > initial * 0.5

    def test_low_resources_increase_cost(self):
        engine = SwarmHibernationEngine(num_agents=3, seed=42)
        # Low resources
        engine.set_resource_level(0.1)
        initial = engine._agents[0].energy
        for _ in range(10):
            engine.tick()
        loss_low = initial - engine._agents[0].energy

        # Compare with high resources
        engine2 = SwarmHibernationEngine(num_agents=3, seed=42)
        engine2.set_resource_level(0.9)
        initial2 = engine2._agents[0].energy
        for _ in range(10):
            engine2.tick()
        loss_high = initial2 - engine2._agents[0].energy

        assert loss_low > loss_high

    def test_torpor_reduces_energy_cost(self):
        engine = SwarmHibernationEngine(num_agents=1, seed=42)
        agent = engine._agents[0]
        agent.state = HibernationState.DEEP_TORPOR
        initial = agent.energy
        engine._update_energy(agent)
        torpor_cost = initial - agent.energy

        engine2 = SwarmHibernationEngine(num_agents=1, seed=42)
        agent2 = engine2._agents[0]
        initial2 = agent2.energy
        engine2._update_energy(agent2)
        active_cost = initial2 - agent2.energy

        assert torpor_cost < active_cost

    def test_set_resource_level_clamped(self):
        engine = SwarmHibernationEngine(seed=1)
        engine.set_resource_level(1.5)
        assert engine._resource_level == 1.0
        engine.set_resource_level(-0.5)
        assert engine._resource_level == 0.0


# ---------------------------------------------------------------------------
# Torpor States
# ---------------------------------------------------------------------------


class TestTorporStates:
    def test_state_enum_values(self):
        assert HibernationState.ACTIVE.value == "active"
        assert HibernationState.DEEP_TORPOR.value == "deep_torpor"

    def test_agents_enter_drowsy_on_low_energy(self):
        engine = SwarmHibernationEngine(num_agents=3, seed=42)
        engine.set_resource_level(0.1)
        for _ in range(100):
            engine.tick()
        states = {a.state for a in engine._agents}
        # At least some should have left ACTIVE
        assert states != {HibernationState.ACTIVE}

    def test_drowsy_before_torpor(self):
        """Agents must pass through drowsy before torpor."""
        engine = SwarmHibernationEngine(num_agents=5, seed=42)
        engine.set_resource_level(0.1)
        seen_drowsy = False
        for _ in range(100):
            engine.tick()
            for a in engine._agents:
                if a.state == HibernationState.DROWSY:
                    seen_drowsy = True
        assert seen_drowsy

    def test_deep_torpor_reachable(self):
        engine = SwarmHibernationEngine(
            num_agents=10, base_metabolic_rate=2.0,
            periodic_arousal_interval=100, seed=42,
        )
        for _ in range(100):
            engine.set_resource_level(0.01)
            engine.tick()
        deep = [a for a in engine._agents if a.state == HibernationState.DEEP_TORPOR]
        assert len(deep) > 0

    def test_min_active_ratio_enforced(self):
        engine = SwarmHibernationEngine(num_agents=10, min_active_ratio=0.3, seed=42)
        engine.set_resource_level(0.05)
        for _ in range(200):
            engine.tick()
        active = sum(
            1 for a in engine._agents
            if a.state in (HibernationState.ACTIVE, HibernationState.DROWSY, HibernationState.AROUSING)
        )
        # Should maintain at least some active
        assert active >= 1

    def test_torpor_bout_counter_increments(self):
        engine = SwarmHibernationEngine(num_agents=5, seed=42)
        engine.set_resource_level(0.05)
        for _ in range(200):
            engine.tick()
        total_bouts = sum(a.torpor_bouts for a in engine._agents)
        assert total_bouts > 0

    def test_metabolic_multipliers_ordered(self):
        assert _METABOLIC_MULTIPLIER[HibernationState.DEEP_TORPOR] < \
               _METABOLIC_MULTIPLIER[HibernationState.LIGHT_TORPOR] < \
               _METABOLIC_MULTIPLIER[HibernationState.DROWSY] < \
               _METABOLIC_MULTIPLIER[HibernationState.ACTIVE]

    def test_arousing_transitions_to_active(self):
        engine = SwarmHibernationEngine(num_agents=3, seed=42)
        agent = engine._agents[0]
        agent.state = HibernationState.AROUSING
        agent.cycles_in_state = 0
        for _ in range(5):
            engine._check_torpor_entry(agent)
        assert agent.state == HibernationState.ACTIVE

    def test_total_torpor_cycles_accumulate(self):
        engine = SwarmHibernationEngine(num_agents=5, seed=42)
        engine.set_resource_level(0.05)
        for _ in range(100):
            engine.tick()
        total = sum(a.total_torpor_cycles for a in engine._agents)
        assert total > 0


# ---------------------------------------------------------------------------
# Scarcity Detection
# ---------------------------------------------------------------------------


class TestScarcityDetection:
    def test_abundant_at_start(self):
        engine = SwarmHibernationEngine(seed=1)
        assert engine._current_scarcity == ScarcityLevel.ABUNDANT

    def test_scarcity_detects_low_resources(self):
        engine = SwarmHibernationEngine(seed=1)
        for _ in range(30):
            engine.set_resource_level(0.1)
            engine._detect_scarcity()
        assert engine._current_scarcity in (
            ScarcityLevel.CRITICAL, ScarcityLevel.DEPLETED, ScarcityLevel.SCARCE
        )

    def test_ema_smoothing(self):
        engine = SwarmHibernationEngine(seed=1)
        engine.set_resource_level(0.0)
        engine._detect_scarcity()
        # EMA should not immediately drop to 0
        assert engine._resource_ema > 0.0

    def test_scarcity_events_recorded(self):
        engine = SwarmHibernationEngine(seed=1)
        for _ in range(20):
            engine.set_resource_level(0.05)
            engine._detect_scarcity()
        assert len(engine._scarcity_events) > 0

    def test_recovery_from_scarcity(self):
        engine = SwarmHibernationEngine(seed=1)
        for _ in range(30):
            engine.set_resource_level(0.05)
            engine._detect_scarcity()
        for _ in range(50):
            engine.set_resource_level(0.95)
            engine._detect_scarcity()
        assert engine._current_scarcity in (
            ScarcityLevel.ABUNDANT, ScarcityLevel.ADEQUATE
        )

    def test_scarcity_levels_all_valid(self):
        for level in ScarcityLevel:
            assert isinstance(level.value, str)

    def test_depleted_detection(self):
        engine = SwarmHibernationEngine(seed=1, critical_threshold=0.15)
        for _ in range(50):
            engine.set_resource_level(0.0)
            engine._detect_scarcity()
        assert engine._current_scarcity == ScarcityLevel.DEPLETED


# ---------------------------------------------------------------------------
# Arousal Triggers
# ---------------------------------------------------------------------------


class TestArousalTriggers:
    def test_threat_triggers_arousal(self):
        engine = SwarmHibernationEngine(num_agents=5, seed=42)
        # Put agents in torpor manually
        for a in engine._agents[:3]:
            a.state = HibernationState.DEEP_TORPOR
            a.cycles_in_state = 10
        engine.inject_threat(0.9)
        for a in engine._agents[:3]:
            engine._check_arousal(a)
        aroused = [a for a in engine._agents[:3] if a.state == HibernationState.AROUSING]
        assert len(aroused) > 0

    def test_resource_recovery_triggers_arousal(self):
        engine = SwarmHibernationEngine(num_agents=5, seed=42)
        agent = engine._agents[0]
        agent.state = HibernationState.LIGHT_TORPOR
        agent.cycles_in_state = 10
        engine._current_scarcity = ScarcityLevel.ABUNDANT
        engine._check_arousal(agent)
        assert agent.state == HibernationState.AROUSING

    def test_periodic_arousal(self):
        engine = SwarmHibernationEngine(num_agents=3, periodic_arousal_interval=5, seed=42)
        agent = engine._agents[0]
        agent.state = HibernationState.LIGHT_TORPOR
        agent.total_torpor_cycles = 5
        agent.cycles_in_state = 10
        engine._current_scarcity = ScarcityLevel.SCARCE
        engine._check_arousal(agent)
        assert agent.state == HibernationState.AROUSING

    def test_arousal_event_recorded(self):
        engine = SwarmHibernationEngine(num_agents=3, seed=42)
        agent = engine._agents[0]
        agent.state = HibernationState.DEEP_TORPOR
        agent.cycles_in_state = 10
        engine.inject_threat(0.9)
        engine._check_arousal(agent)
        assert len(engine._arousal_events) > 0
        assert engine._arousal_events[0].trigger == ArousalTrigger.THREAT_SIGNAL

    def test_torpor_bout_recorded_on_arousal(self):
        engine = SwarmHibernationEngine(num_agents=3, seed=42)
        agent = engine._agents[0]
        agent.state = HibernationState.LIGHT_TORPOR
        agent.cycles_in_state = 10
        engine._active_bout_starts[agent.agent_id] = 0
        engine._active_bout_energy[agent.agent_id] = 80.0
        engine._cycle = 10
        engine._current_scarcity = ScarcityLevel.ABUNDANT
        engine._check_arousal(agent)
        assert len(engine._torpor_bouts) > 0

    def test_inject_threat_clamped(self):
        engine = SwarmHibernationEngine(seed=1)
        engine.inject_threat(1.5)
        assert engine._threat_level == 1.0
        engine.inject_threat(-0.5)
        assert engine._threat_level == 0.0

    def test_threat_decays(self):
        engine = SwarmHibernationEngine(seed=1)
        engine.inject_threat(1.0)
        engine.tick()
        assert engine._threat_level < 1.0

    def test_arousal_count_increments(self):
        engine = SwarmHibernationEngine(num_agents=3, seed=42)
        agent = engine._agents[0]
        agent.state = HibernationState.DEEP_TORPOR
        agent.cycles_in_state = 10
        engine.inject_threat(0.9)
        engine._check_arousal(agent)
        assert agent.arousal_count == 1

    def test_active_agents_not_aroused(self):
        engine = SwarmHibernationEngine(num_agents=3, seed=42)
        agent = engine._agents[0]
        assert agent.state == HibernationState.ACTIVE
        engine.inject_threat(0.9)
        engine._check_arousal(agent)
        assert agent.state == HibernationState.ACTIVE  # unchanged


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------


class TestClusters:
    def test_clusters_form_during_torpor(self):
        engine = SwarmHibernationEngine(num_agents=10, seed=42)
        for a in engine._agents[:6]:
            a.state = HibernationState.LIGHT_TORPOR
        engine._manage_clusters()
        assert len(engine._clusters) > 0

    def test_cluster_members_assigned(self):
        engine = SwarmHibernationEngine(num_agents=6, seed=42)
        for a in engine._agents[:4]:
            a.state = HibernationState.DEEP_TORPOR
        engine._manage_clusters()
        clustered = [a for a in engine._agents if a.cluster_id is not None]
        assert len(clustered) >= 2

    def test_cluster_thermal_benefit(self):
        engine = SwarmHibernationEngine(num_agents=6, cluster_thermal_bonus=0.4, seed=42)
        for a in engine._agents[:4]:
            a.state = HibernationState.DEEP_TORPOR
        engine._manage_clusters()
        for c in engine._clusters:
            if c.is_active:
                assert c.thermal_benefit > 0

    def test_cluster_dissolves_when_members_wake(self):
        engine = SwarmHibernationEngine(num_agents=6, seed=42)
        for a in engine._agents[:4]:
            a.state = HibernationState.DEEP_TORPOR
        engine._manage_clusters()
        # Wake all members
        for a in engine._agents[:4]:
            a.state = HibernationState.ACTIVE
        engine._manage_clusters()
        active_clusters = [c for c in engine._clusters if c.is_active]
        assert len(active_clusters) == 0

    def test_cluster_break_arousal(self):
        engine = SwarmHibernationEngine(num_agents=4, seed=42)
        for a in engine._agents[:2]:
            a.state = HibernationState.DEEP_TORPOR
            a.cycles_in_state = 10
        engine._manage_clusters()
        # Now remove one agent from cluster
        clustered = [a for a in engine._agents if a.cluster_id is not None]
        if len(clustered) >= 2:
            engine._remove_from_cluster(clustered[0])
            # Other member should get cluster break arousal
            assert any(
                e.trigger == ArousalTrigger.CLUSTER_BREAK
                for e in engine._arousal_events
            )

    def test_min_cluster_size(self):
        """Clusters need at least 2 members."""
        engine = SwarmHibernationEngine(num_agents=3, seed=42)
        # Only 1 torpid — shouldn't form cluster
        engine._agents[0].state = HibernationState.DEEP_TORPOR
        engine._manage_clusters()
        clustered = [a for a in engine._agents if a.cluster_id is not None]
        assert len(clustered) == 0


# ---------------------------------------------------------------------------
# Health Scoring
# ---------------------------------------------------------------------------


class TestHealthScoring:
    def test_score_range(self):
        engine = SwarmHibernationEngine(seed=42)
        report = engine.simulate(cycles=50)
        assert 0 <= report.health.score <= 100

    def test_tier_assigned(self):
        engine = SwarmHibernationEngine(seed=42)
        report = engine.simulate(cycles=50)
        assert report.health.tier in ("Thriving", "Conserving", "Strained", "Critical", "Collapsed")

    def test_health_dimensions_valid(self):
        engine = SwarmHibernationEngine(seed=42)
        report = engine.simulate(cycles=50)
        h = report.health
        assert 0 <= h.energy_reserves <= 1
        assert 0 <= h.torpor_efficiency <= 1
        assert 0 <= h.arousal_responsiveness <= 1
        assert 0 <= h.cluster_utilization <= 1
        assert 0 <= h.active_ratio <= 1
        assert 0 <= h.sustainability <= 1

    def test_healthy_start_scores_well(self):
        engine = SwarmHibernationEngine(num_agents=10, seed=42)
        engine.set_resource_level(0.9)
        report = engine.simulate(cycles=10)
        assert report.health.score >= 40

    def test_depleted_scores_low(self):
        engine = SwarmHibernationEngine(num_agents=10, base_metabolic_rate=3.0, seed=42)
        engine.set_resource_level(0.01)
        report = engine.simulate(cycles=200)
        assert report.health.score < 80

    def test_empty_agents_collapsed(self):
        engine = SwarmHibernationEngine(num_agents=1, seed=1)
        engine._agents = []
        h = engine._score_health()
        assert h.tier == "Collapsed"


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------


class TestInsights:
    def test_insights_generated(self):
        engine = SwarmHibernationEngine(num_agents=10, seed=42)
        engine.set_resource_level(0.1)
        report = engine.simulate(cycles=100)
        assert len(report.insights) > 0

    def test_critical_energy_insight(self):
        engine = SwarmHibernationEngine(num_agents=5, base_metabolic_rate=4.0, seed=42)
        engine.set_resource_level(0.01)
        for _ in range(200):
            engine.tick()
        insights = engine._generate_insights()
        assert any("energy" in i.lower() for i in insights)

    def test_torpor_insight(self):
        engine = SwarmHibernationEngine(
            num_agents=10, base_metabolic_rate=2.0,
            periodic_arousal_interval=100, seed=42,
        )
        for _ in range(100):
            engine.set_resource_level(0.01)
            engine.tick()
        insights = engine._generate_insights()
        # Should mention torpor or conservation
        assert any("torpor" in i.lower() or "conservation" in i.lower() for i in insights)

    def test_scarcity_insight(self):
        engine = SwarmHibernationEngine(num_agents=5, seed=42)
        engine.set_resource_level(0.02)
        for _ in range(50):
            engine.tick()
        insights = engine._generate_insights()
        assert any("scarcity" in i.lower() or "resource" in i.lower() for i in insights)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_mild_winter(self):
        engine = SwarmHibernationEngine.from_scenario("mild_winter", seed=42)
        report = engine.simulate()
        assert report.health.score > 0

    def test_deep_freeze(self):
        engine = SwarmHibernationEngine.from_scenario("deep_freeze", seed=42)
        report = engine.simulate()
        assert report.health.score > 0
        # Should have significant torpor
        assert any(a.torpor_bouts > 0 for a in report.agents)

    def test_intermittent_scarcity(self):
        engine = SwarmHibernationEngine.from_scenario("intermittent_scarcity", seed=42)
        report = engine.simulate()
        # Should have scarcity events (oscillating resources)
        assert len(report.scarcity_events) > 0

    def test_emergency_arousal(self):
        engine = SwarmHibernationEngine.from_scenario("emergency_arousal", seed=42)
        report = engine.simulate()
        # Should have threat-triggered arousals
        threats = [e for e in report.arousal_events if e.trigger == ArousalTrigger.THREAT_SIGNAL]
        assert len(threats) > 0

    def test_cluster_survival(self):
        engine = SwarmHibernationEngine.from_scenario("cluster_survival", seed=42)
        report = engine.simulate()
        assert len(report.clusters) > 0

    def test_all_scenarios_complete(self):
        for name in SCENARIOS:
            engine = SwarmHibernationEngine.from_scenario(name, seed=1)
            report = engine.simulate()
            assert isinstance(report, HibernationReport)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_to_dict(self):
        engine = SwarmHibernationEngine(seed=42)
        report = engine.simulate(cycles=20)
        d = engine.to_dict(report)
        assert "health" in d
        assert "agents" in d
        assert "cycle_history" in d

    def test_to_dict_json_serializable(self):
        engine = SwarmHibernationEngine(seed=42)
        report = engine.simulate(cycles=20)
        d = engine.to_dict(report)
        s = json.dumps(d)
        assert len(s) > 0

    def test_export_html(self):
        engine = SwarmHibernationEngine(seed=42)
        report = engine.simulate(cycles=20)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        engine.export_html(path, report)
        content = Path(path).read_text(encoding="utf-8")
        assert "Swarm Hibernation" in content
        assert "Health Dimensions" in content
        Path(path).unlink()

    def test_export_html_no_report(self):
        engine = SwarmHibernationEngine(seed=42)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        engine.export_html(path)
        content = Path(path).read_text(encoding="utf-8")
        assert len(content) > 100
        Path(path).unlink()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_default(self, capsys):
        from src.hibernation import _main
        _main(["--seed", "42", "--cycles", "20"])
        out = capsys.readouterr().out
        assert "Hibernation" in out

    def test_main_scenario(self, capsys):
        from src.hibernation import _main
        _main(["--scenario", "mild_winter", "--seed", "1"])
        out = capsys.readouterr().out
        assert "Score" in out

    def test_main_json_output(self):
        from src.hibernation import _main
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        _main(["--seed", "42", "--cycles", "10", "--json", path])
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "health" in data
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_lifecycle(self):
        """Agents go through full lifecycle: active -> torpor -> arousal -> active."""
        engine = SwarmHibernationEngine(num_agents=8, base_metabolic_rate=2.0, seed=42)
        # Phase 1: Scarcity drives torpor
        for _ in range(120):
            engine.set_resource_level(0.01)
            engine.tick()
        torpid = sum(
            1 for a in engine._agents
            if a.state in (HibernationState.LIGHT_TORPOR, HibernationState.DEEP_TORPOR,
                           HibernationState.DROWSY)
        )
        assert torpid > 0

        # Phase 2: Resources recover, agents wake
        for _ in range(100):
            engine.set_resource_level(0.95)
            engine.tick()
        active = sum(
            1 for a in engine._agents
            if a.state == HibernationState.ACTIVE
        )
        assert active > 0

    def test_threat_during_hibernation(self):
        engine = SwarmHibernationEngine(num_agents=10, seed=42)
        engine.set_resource_level(0.05)
        for _ in range(80):
            engine.tick()
        # Inject threat
        engine.inject_threat(0.95)
        for _ in range(5):
            engine.tick()
        # Some agents should have been aroused
        assert len(engine._arousal_events) > 0

    def test_cycle_history_length(self):
        engine = SwarmHibernationEngine(seed=42)
        report = engine.simulate(cycles=50)
        assert len(report.cycle_history) == 50

    def test_report_structure(self):
        engine = SwarmHibernationEngine(seed=42)
        report = engine.simulate(cycles=30)
        assert isinstance(report, HibernationReport)
        assert isinstance(report.health, HealthScore)
        assert isinstance(report.agents, list)
        assert isinstance(report.insights, list)
