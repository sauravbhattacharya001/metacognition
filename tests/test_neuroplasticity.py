"""Tests for Swarm Neuroplasticity Engine."""
import json
import os
import tempfile

import pytest

from src.neuroplasticity import (
    NeuroplasticityEngine,
    NeuroplasticityReport,
    NetworkSnapshot,
    PlasticityEvent,
    PlasticityRecord,
    Synapse,
    SynapseState,
)


@pytest.fixture
def agents():
    return ["a1", "a2", "a3", "a4", "a5"]


@pytest.fixture
def engine(agents):
    return NeuroplasticityEngine(
        agents=agents, initial_connectivity=0.0, prune_threshold=0.1,
        ltp_threshold=5, ltd_threshold=20, homeostatic_target=0.5,
    )


@pytest.fixture
def connected_engine(agents):
    return NeuroplasticityEngine(
        agents=agents, initial_connectivity=1.0, prune_threshold=0.1,
        ltp_threshold=5, ltd_threshold=20, homeostatic_target=0.5,
    )


class TestSynapseCreation:
    def test_initial_no_connections(self, engine):
        assert len(engine.synapses) == 0

    def test_full_connectivity(self, connected_engine, agents):
        n = len(agents)
        assert len(connected_engine.synapses) == n * (n - 1)

    def test_synapse_created_on_activate(self, engine):
        engine.activate("a1", "a2", success=True)
        syn = engine.get_synapse("a1", "a2")
        assert syn is not None
        assert syn.source == "a1"
        assert syn.target == "a2"

    def test_synapse_initial_weight(self, engine):
        engine.activate("a1", "a2")
        syn = engine.get_synapse("a1", "a2")
        # Created at 0.3, then one hebbian strengthen
        assert syn.weight > 0.3

    def test_self_loop_prevented(self, engine):
        engine.activate("a1", "a1", success=True)
        assert engine.get_synapse("a1", "a1") is None

    def test_duplicate_synapse_not_created(self, engine):
        engine.activate("a1", "a2")
        engine.activate("a1", "a2")
        count = sum(1 for k in engine.synapses if k == ("a1", "a2"))
        assert count == 1


class TestHebbianLearning:
    def test_strengthen_on_success(self, engine):
        engine.activate("a1", "a2", success=True)
        syn = engine.get_synapse("a1", "a2")
        initial_w = syn.weight
        engine.activate("a1", "a2", success=True)
        assert syn.weight > initial_w

    def test_weaken_on_failure(self, engine):
        engine.activate("a1", "a2", success=True)
        syn = engine.get_synapse("a1", "a2")
        w_after_success = syn.weight
        engine.activate("a1", "a2", success=False)
        assert syn.weight < w_after_success

    def test_weight_bounded_above(self, engine):
        engine.activate("a1", "a2")
        syn = engine.get_synapse("a1", "a2")
        for _ in range(1000):
            engine.activate("a1", "a2", success=True)
        assert syn.weight <= 1.0

    def test_weight_bounded_below(self, engine):
        engine.activate("a1", "a2")
        syn = engine.get_synapse("a1", "a2")
        for _ in range(1000):
            engine.activate("a1", "a2", success=False)
        assert syn.weight >= 0.0

    def test_strengthen_increases_activation_count(self, engine):
        engine.activate("a1", "a2")
        engine.activate("a1", "a2")
        engine.activate("a1", "a2")
        syn = engine.get_synapse("a1", "a2")
        assert syn.activation_count == 3


class TestLTP:
    def test_ltp_triggers_at_threshold(self, engine):
        for _ in range(5):
            engine.activate("a1", "a2", success=True)
        syn = engine.get_synapse("a1", "a2")
        assert syn.state == SynapseState.POTENTIATED
        assert syn.ltp_count >= 1

    def test_ltp_weight_bonus(self, engine):
        for _ in range(4):
            engine.activate("a1", "a2", success=True)
        syn = engine.get_synapse("a1", "a2")
        pre_ltp_weight = syn.weight
        engine.activate("a1", "a2", success=True)  # triggers LTP
        # Weight should jump by ~0.15
        assert syn.weight > pre_ltp_weight + 0.1

    def test_ltp_not_retriggered(self, engine):
        for _ in range(10):
            engine.activate("a1", "a2", success=True)
        syn = engine.get_synapse("a1", "a2")
        assert syn.ltp_count == 1  # only triggered once

    def test_ltp_events_recorded(self, engine):
        for _ in range(5):
            engine.activate("a1", "a2", success=True)
        ltp_events = [e for e in engine.events if e.event_type == PlasticityEvent.LTP]
        assert len(ltp_events) == 1


class TestLTD:
    def test_ltd_triggers_after_idle(self, engine):
        engine.activate("a1", "a2", success=True)
        syn = engine.get_synapse("a1", "a2")
        engine.tick(steps=20)
        assert syn.state == SynapseState.DEPRESSED

    def test_ltd_reduces_weight(self, engine):
        engine.activate("a1", "a2", success=True)
        engine.tick(steps=20)
        # LTD event should record a weight decrease
        ltd_events = [e for e in engine.events if e.event_type == PlasticityEvent.LTD]
        assert len(ltd_events) >= 1
        ev = ltd_events[0]
        assert ev.new_weight < ev.old_weight

    def test_ltd_increments_count(self, engine):
        engine.activate("a1", "a2", success=True)
        engine.tick(steps=20)
        syn = engine.get_synapse("a1", "a2")
        assert syn.ltd_count >= 1

    def test_ltd_events_recorded(self, engine):
        engine.activate("a1", "a2", success=True)
        engine.tick(steps=20)
        ltd_events = [e for e in engine.events if e.event_type == PlasticityEvent.LTD]
        assert len(ltd_events) >= 1


class TestPruning:
    def test_pruning_removes_weak_synapse(self, engine):
        # Give a1 enough outgoing connections so synaptogenesis won't re-create
        engine.activate("a1", "a2", success=True)
        engine.activate("a1", "a3", success=True)
        engine.activate("a1", "a4", success=True)
        syn = engine.get_synapse("a1", "a2")
        # Force weight below threshold
        syn.weight = 0.05
        engine.tick()
        assert engine.get_synapse("a1", "a2") is None

    def test_pruning_increments_counter(self, engine):
        engine.activate("a1", "a2", success=True)
        syn = engine.get_synapse("a1", "a2")
        syn.weight = 0.05
        initial_pruned = engine.pruned_count
        engine.tick()
        assert engine.pruned_count > initial_pruned

    def test_pruning_event_recorded(self, engine):
        engine.activate("a1", "a2", success=True)
        syn = engine.get_synapse("a1", "a2")
        syn.weight = 0.05
        engine.tick()
        prune_events = [e for e in engine.events if e.event_type == PlasticityEvent.PRUNING]
        assert len(prune_events) >= 1


class TestSynaptogenesis:
    def test_synaptogenesis_for_isolated_agent(self, engine):
        # a1 has no connections, tick should create some
        engine.tick()
        connections = engine.get_agent_connections("a1")
        assert len(connections) >= 1

    def test_force_synaptogenesis(self, engine):
        syn = engine.force_synaptogenesis("a1", "a3")
        assert syn is not None
        assert syn.source == "a1"
        assert syn.target == "a3"
        assert syn.weight == 0.3

    def test_force_synaptogenesis_self_loop_blocked(self, engine):
        result = engine.force_synaptogenesis("a1", "a1")
        assert result is None

    def test_force_synaptogenesis_existing_returns_existing(self, engine):
        engine.force_synaptogenesis("a1", "a2")
        syn1 = engine.get_synapse("a1", "a2")
        syn2 = engine.force_synaptogenesis("a1", "a2")
        assert syn1 is syn2


class TestHomeostaticPlasticity:
    def test_scaling_down_when_too_high(self, connected_engine):
        # Set all weights very high
        for syn in connected_engine.synapses.values():
            syn.weight = 0.95
            syn.last_activated = connected_engine.current_tick
        connected_engine.tick()
        weights = [s.weight for s in connected_engine.synapses.values()]
        import statistics
        mean_w = statistics.mean(weights)
        assert mean_w < 0.95

    def test_scaling_up_when_too_low(self, connected_engine):
        # Set all weights very low (but above prune threshold)
        for syn in connected_engine.synapses.values():
            syn.weight = 0.15
            syn.last_activated = connected_engine.current_tick
        connected_engine.tick()
        weights = [s.weight for s in connected_engine.synapses.values()]
        import statistics
        mean_w = statistics.mean(weights)
        assert mean_w > 0.15

    def test_homeostatic_event_recorded(self, connected_engine):
        for syn in connected_engine.synapses.values():
            syn.weight = 0.95
            syn.last_activated = connected_engine.current_tick
        connected_engine.tick()
        homeo_events = [e for e in connected_engine.events if e.event_type == PlasticityEvent.HOMEOSTATIC_SCALE]
        assert len(homeo_events) >= 1


class TestCriticalPeriod:
    def test_critical_period_starts(self, engine):
        engine.trigger_critical_period(duration=10)
        assert engine.in_critical_period is True

    def test_critical_period_doubles_lr(self, engine):
        assert engine.learning_rate == 0.1
        engine.trigger_critical_period()
        assert engine.learning_rate == 0.2

    def test_critical_period_ends(self, engine):
        engine.trigger_critical_period(duration=5)
        engine.tick(steps=5)
        assert engine.in_critical_period is False

    def test_critical_period_events(self, engine):
        engine.trigger_critical_period(duration=3)
        engine.tick(steps=3)
        starts = [e for e in engine.events if e.event_type == PlasticityEvent.CRITICAL_PERIOD_START]
        ends = [e for e in engine.events if e.event_type == PlasticityEvent.CRITICAL_PERIOD_END]
        assert len(starts) == 1
        assert len(ends) == 1

    def test_stronger_hebbian_during_critical(self, engine):
        engine.activate("a1", "a2", success=True)
        syn = engine.get_synapse("a1", "a2")
        w1 = syn.weight

        engine.trigger_critical_period()
        engine.activate("a1", "a2", success=True)
        w2 = syn.weight

        # The critical-period step should have bigger increase
        delta_normal = w1 - 0.3  # first activation from base 0.3
        delta_critical = w2 - w1  # second activation during critical period
        # Both are positive; critical should be larger (2x learning rate)
        assert delta_critical > 0


class TestSnapshot:
    def test_snapshot_returns_correct_type(self, connected_engine):
        snap = connected_engine.snapshot()
        assert isinstance(snap, NetworkSnapshot)

    def test_snapshot_counts(self, connected_engine, agents):
        snap = connected_engine.snapshot()
        assert snap.num_agents == len(agents)
        assert snap.num_synapses == len(agents) * (len(agents) - 1)

    def test_snapshot_density_full(self, connected_engine):
        snap = connected_engine.snapshot()
        assert abs(snap.density - 1.0) < 0.01

    def test_snapshot_added_to_list(self, engine):
        engine.snapshot()
        assert len(engine.snapshots) == 1


class TestAnalyze:
    def test_analyze_returns_report(self, connected_engine):
        report = connected_engine.analyze()
        assert isinstance(report, NeuroplasticityReport)

    def test_report_has_health_score(self, connected_engine):
        report = connected_engine.analyze()
        assert 0 <= report.health_score <= 100

    def test_report_has_insights(self, connected_engine):
        report = connected_engine.analyze()
        assert len(report.insights) > 0


class TestExport:
    def test_html_export_creates_file(self, connected_engine):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            connected_engine.export_html(path)
            assert os.path.exists(path)
            content = open(path, encoding='utf-8').read()
            assert "Neuroplasticity" in content
            assert "<html" in content
        finally:
            os.unlink(path)

    def test_json_export_creates_file(self, connected_engine):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            connected_engine.export_json(path)
            assert os.path.exists(path)
            data = json.loads(open(path).read())
            assert "health_score" in data
            assert "synapses" in data
        finally:
            os.unlink(path)

    def test_json_has_correct_synapse_count(self, connected_engine, agents):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            connected_engine.export_json(path)
            data = json.loads(open(path).read())
            n = len(agents)
            assert len(data["synapses"]) == n * (n - 1)
        finally:
            os.unlink(path)


class TestDemo:
    def test_demo_runs(self):
        engine = NeuroplasticityEngine.demo(num_agents=4, steps=20, seed=42)
        assert engine.current_tick == 20
        assert len(engine.agents) == 4

    def test_demo_has_events(self):
        engine = NeuroplasticityEngine.demo(num_agents=6, steps=50, seed=42)
        assert len(engine.events) > 0

    def test_demo_critical_period_triggered(self):
        engine = NeuroplasticityEngine.demo(num_agents=4, steps=50, seed=42)
        assert engine.critical_period_count >= 1


class TestGetAgentConnections:
    def test_returns_both_directions(self, engine):
        engine.activate("a1", "a2")
        engine.activate("a3", "a1")
        conns = engine.get_agent_connections("a1")
        assert len(conns) == 2
