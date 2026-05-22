"""Tests for Swarm Angiogenesis Engine."""
import json
import os
import tempfile

import pytest

from src.angiogenesis import (
    AngiogenesisReport,
    DemandSignal,
    DemandType,
    DEMAND_URGENCY,
    HealthReport,
    SwarmAngiogenesisEngine,
    Vessel,
    VesselState,
)


@pytest.fixture
def agents():
    return ["a1", "a2", "a3", "a4", "a5"]


@pytest.fixture
def engine(agents):
    return SwarmAngiogenesisEngine(
        agents=agents,
        initial_connectivity=0.0,
        seed=42,
    )


@pytest.fixture
def connected_engine(agents):
    return SwarmAngiogenesisEngine(
        agents=agents,
        initial_connectivity=1.0,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Vessel Creation Tests
# ---------------------------------------------------------------------------


class TestVesselCreation:
    def test_no_initial_vessels(self, engine):
        assert len(engine.vessels) == 0

    def test_full_initial_connectivity(self, connected_engine, agents):
        n = len(agents)
        max_possible = n * (n - 1)
        assert len(connected_engine.vessels) == max_possible

    def test_create_vessel_via_demand(self, engine):
        engine.emit_demand("a1", "a2", intensity=5.0)
        engine._tick()
        engine._tick()
        # Should have sprouted something
        active = [v for v in engine.vessels.values() if v.is_active]
        assert len(active) >= 1

    def test_no_duplicate_vessels(self, engine):
        engine._create_vessel("a1", "a2")
        result = engine._create_vessel("a1", "a2")
        assert result is None

    def test_max_vessels_per_agent(self, engine):
        engine.max_vessels_per_agent = 2
        engine._create_vessel("a1", "a2")
        engine._create_vessel("a1", "a3")
        result = engine._create_vessel("a1", "a4")
        assert result is None

    def test_vessel_initial_state_sprouting(self, engine):
        v = engine._create_vessel("a1", "a2")
        assert v is not None
        assert v.state == VesselState.SPROUTING
        assert v.sprouting_progress == 0.0

    def test_vessel_initial_state_active(self, engine):
        v = engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        assert v is not None
        assert v.state == VesselState.ACTIVE


# ---------------------------------------------------------------------------
# Demand Signal Tests
# ---------------------------------------------------------------------------


class TestDemandSignals:
    def test_emit_demand_creates_signal(self, engine):
        sig = engine.emit_demand("a1", "a2", intensity=3.0)
        assert sig.source_agent == "a1"
        assert sig.target_agent == "a2"
        assert sig.remaining_strength == 3.0

    def test_demand_urgency_multiplier(self, engine):
        sig = engine.emit_demand("a1", "a2", intensity=2.0, demand_type=DemandType.HYPOXIA)
        assert sig.intensity == 2.0 * DEMAND_URGENCY[DemandType.HYPOXIA]

    def test_demand_decay(self, engine):
        engine.emit_demand("a1", "a2", intensity=5.0)
        initial = engine.demand_signals[0].remaining_strength
        engine._decay_demands()
        assert engine.demand_signals[0].remaining_strength < initial

    def test_demand_removal_when_weak(self, engine):
        engine.emit_demand("a1", "a2", intensity=0.005)
        # After multiple decays it should vanish
        for _ in range(50):
            engine._decay_demands()
        assert len(engine.demand_signals) == 0

    def test_hypoxia_detection(self, engine):
        # Agent with no connections should trigger hypoxia demand
        engine.tick_count = 5  # align to %5 == 0
        engine._detect_hypoxia()
        assert len(engine.demand_signals) > 0
        assert any(s.demand_type == DemandType.HYPOXIA for s in engine.demand_signals)


# ---------------------------------------------------------------------------
# Sprouting Tests
# ---------------------------------------------------------------------------


class TestSprouting:
    def test_sprouting_progresses(self, engine):
        v = engine._create_vessel("a1", "a2")
        engine._process_sprouting()
        assert v.sprouting_progress == engine.sprouting_speed

    def test_sprouting_completes(self, engine):
        v = engine._create_vessel("a1", "a2")
        # Advance enough ticks
        for _ in range(int(1.0 / engine.sprouting_speed) + 1):
            engine._process_sprouting()
        assert v.state == VesselState.ACTIVE

    def test_sprout_triggered_by_strong_demand(self, engine):
        engine.emit_demand("a1", "a3", intensity=5.0)
        engine._sprout_new_vessels()
        sprouting = [v for v in engine.vessels.values() if v.state == VesselState.SPROUTING]
        assert len(sprouting) >= 1
        assert any(v.source == "a1" and v.target == "a3" for v in sprouting)

    def test_no_sprout_below_threshold(self, engine):
        engine.emit_demand("a1", "a3", intensity=0.5)  # below threshold
        engine._sprout_new_vessels()
        assert len(engine.vessels) == 0


# ---------------------------------------------------------------------------
# Flow Routing Tests
# ---------------------------------------------------------------------------


class TestFlowRouting:
    def test_route_flow_success(self, engine):
        engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        routed = engine.route_flow("a1", "a2", 0.5)
        assert routed == 0.5

    def test_route_flow_capacity_limit(self, engine):
        v = engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        v.capacity = 0.3
        routed = engine.route_flow("a1", "a2", 1.0)
        assert routed == pytest.approx(0.3)

    def test_route_flow_no_path(self, engine):
        routed = engine.route_flow("a1", "a2", 1.0)
        assert routed == 0.0

    def test_utilization_recorded(self, engine):
        engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        engine.emit_demand("a1", "a2", intensity=2.0)
        engine._update_flow()
        v = list(engine.vessels.values())[0]
        assert len(v.utilization_history) == 1


# ---------------------------------------------------------------------------
# Maturation Tests
# ---------------------------------------------------------------------------


class TestMaturation:
    def test_vessel_matures_with_utilization(self, engine):
        v = engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        # Simulate high utilization history
        v.utilization_history = [0.6] * 20
        v.pericyte_coverage = 0.65
        engine._process_maturation()
        # Should gain pericyte coverage
        assert v.pericyte_coverage > 0.65

    def test_vessel_reaches_mature_state(self, engine):
        v = engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        v.utilization_history = [0.7] * 20
        v.pericyte_coverage = 0.69
        engine._process_maturation()
        assert v.state == VesselState.MATURE

    def test_low_utilization_loses_pericyte(self, engine):
        v = engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        v.utilization_history = [0.1] * 20
        v.pericyte_coverage = 0.5
        engine._process_maturation()
        assert v.pericyte_coverage < 0.5


# ---------------------------------------------------------------------------
# Regression & Pruning Tests
# ---------------------------------------------------------------------------


class TestRegression:
    def test_low_utilization_triggers_regression(self, engine):
        v = engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        v.utilization_history = [0.05] * 20
        v.age = 15
        v.pericyte_coverage = 0.1
        engine._process_regression()
        assert v.state == VesselState.REGRESSING

    def test_regressing_vessel_gets_pruned(self, engine):
        v = engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        v.state = VesselState.REGRESSING
        v.pericyte_coverage = 0.03
        engine._process_regression()
        assert v.state == VesselState.PRUNED
        assert v.pruned_tick is not None

    def test_mature_vessel_resists_pruning(self, engine):
        v = engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        v.state = VesselState.MATURE
        v.utilization_history = [0.05] * 20
        v.age = 15
        v.pericyte_coverage = 0.8
        engine._process_regression()
        # Should lose pericyte but not immediately regress
        assert v.pericyte_coverage < 0.8
        assert v.state in (VesselState.MATURE, VesselState.ACTIVE)


# ---------------------------------------------------------------------------
# Anastomosis Tests
# ---------------------------------------------------------------------------


class TestAnastomosis:
    def test_anastomosis_detected(self, engine):
        v1 = engine._create_vessel("a1", "a2")
        v2 = engine._create_vessel("a2", "a1")
        engine._detect_anastomosis()
        assert v1.state == VesselState.ACTIVE
        assert v2.state == VesselState.ACTIVE

    def test_anastomosis_event_logged(self, engine):
        engine._create_vessel("a1", "a2")
        engine._create_vessel("a2", "a1")
        engine._detect_anastomosis()
        anast_events = [e for e in engine.events if e.event_type == "anastomosis"]
        assert len(anast_events) >= 1


# ---------------------------------------------------------------------------
# Capacity Adaptation Tests
# ---------------------------------------------------------------------------


class TestCapacityAdaptation:
    def test_mature_overloaded_gains_capacity(self, engine):
        v = engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        v.state = VesselState.MATURE
        v.capacity = 1.0
        v.current_flow = 0.9  # 90% utilization
        engine._adapt_capacity()
        assert v.capacity > 1.0

    def test_severely_overloaded_emits_demand(self, engine):
        v = engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        v.state = VesselState.MATURE
        v.capacity = 1.0
        v.current_flow = 0.96
        # After adapt_capacity, capacity grows so we need to check utilization
        # before adaptation: 0.96/1.0 = 0.96 > 0.95 triggers demand
        # But capacity grows first in the loop, so set flow higher
        v.current_flow = 1.0
        engine._adapt_capacity()
        arteriogenic = [
            s for s in engine.demand_signals
            if s.demand_type == DemandType.ARTERIOGENIC
        ]
        assert len(arteriogenic) >= 1


# ---------------------------------------------------------------------------
# Full Simulation Tests
# ---------------------------------------------------------------------------


class TestSimulation:
    def test_simulate_returns_report(self, engine):
        engine.emit_demand("a1", "a3", intensity=4.0)
        report = engine.simulate(ticks=50)
        assert isinstance(report, AngiogenesisReport)
        assert report.total_ticks == 50

    def test_simulate_creates_vessels(self, engine):
        engine.emit_demand("a1", "a3", intensity=5.0)
        engine.emit_demand("a2", "a4", intensity=4.0)
        report = engine.simulate(ticks=30)
        assert report.active_vessel_count > 0

    def test_health_score_range(self, engine):
        engine.emit_demand("a1", "a2", intensity=3.0)
        report = engine.simulate(ticks=50)
        assert 0 <= report.health_score <= 100

    def test_longer_simulation_improves_maturation(self, agents):
        engine = SwarmAngiogenesisEngine(agents=agents, initial_connectivity=0.5, seed=42)
        # Emit strong sustained demand to drive utilization
        for a in agents:
            others = [x for x in agents if x != a]
            for o in others:
                engine.emit_demand(a, o, intensity=5.0)
        report = engine.simulate(ticks=300)
        assert report.mature_count > 0


# ---------------------------------------------------------------------------
# Analysis & Insights Tests
# ---------------------------------------------------------------------------


class TestAnalysis:
    def test_agent_nodes_populated(self, connected_engine, agents):
        report = connected_engine.analyze()
        assert set(report.agent_nodes.keys()) == set(agents)

    def test_perfusion_coverage_full_connectivity(self, connected_engine):
        report = connected_engine.analyze()
        assert report.perfusion_coverage > 0.5

    def test_insights_generated(self, engine):
        # Hypoxic agents should generate insight
        report = engine.analyze()
        assert len(report.insights) > 0

    def test_health_tier_assignment(self, engine):
        report = engine.analyze()
        assert report.health.tier in ("Thriving", "Healthy", "Stressed", "Ischemic", "Necrotic")

    def test_redundancy_zero_without_parallel(self, engine):
        engine._create_vessel("a1", "a2", state=VesselState.ACTIVE)
        report = engine.analyze()
        assert report.health.redundancy == 0.0


# ---------------------------------------------------------------------------
# Export Tests
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_json(self, connected_engine):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            connected_engine.simulate(ticks=10)
            connected_engine.export_json(path)
            data = json.loads(open(path).read())
            assert "health_score" in data
            assert "insights" in data
            assert "agent_nodes" in data
        finally:
            os.unlink(path)

    def test_export_html(self, connected_engine):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            connected_engine.simulate(ticks=10)
            connected_engine.export_html(path)
            content = open(path, encoding="utf-8").read()
            assert "Swarm Angiogenesis Dashboard" in content
            assert "Perfusion Coverage" in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Data Model Tests
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_vessel_utilization(self):
        v = Vessel(vessel_id="v1", source="a", target="b", capacity=2.0, current_flow=1.0)
        assert v.utilization == 0.5

    def test_vessel_utilization_zero_capacity(self):
        v = Vessel(vessel_id="v1", source="a", target="b", capacity=0.0, current_flow=0.0)
        assert v.utilization == 0.0

    def test_vessel_is_active(self):
        v = Vessel(vessel_id="v1", source="a", target="b", state=VesselState.ACTIVE)
        assert v.is_active is True
        v.state = VesselState.PRUNED
        assert v.is_active is False

    def test_health_report_tiers(self):
        assert HealthReport(score=85, perfusion_coverage=0.9, flow_efficiency=0.8,
                          redundancy=0.5, maturation_ratio=0.6, vessel_turnover=0.1).tier == "Thriving"
        assert HealthReport(score=15, perfusion_coverage=0.1, flow_efficiency=0.1,
                          redundancy=0.0, maturation_ratio=0.0, vessel_turnover=0.9).tier == "Necrotic"

    def test_demand_signal_defaults(self):
        sig = DemandSignal(source_agent="a", target_agent="b",
                          demand_type=DemandType.VEGF, intensity=3.0, tick_emitted=0)
        assert sig.remaining_strength == 3.0

    def test_vessel_state_enum(self):
        assert VesselState.SPROUTING.value == "sprouting"
        assert VesselState.PRUNED.value == "pruned"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_agent(self):
        engine = SwarmAngiogenesisEngine(agents=["solo"], seed=1)
        report = engine.simulate(ticks=10)
        assert report.active_vessel_count == 0

    def test_two_agents(self):
        engine = SwarmAngiogenesisEngine(agents=["a", "b"], initial_connectivity=0.0, seed=1)
        engine.emit_demand("a", "b", intensity=5.0)
        report = engine.simulate(ticks=20)
        assert report.vessel_count > 0

    def test_large_swarm(self):
        agents = [f"node-{i}" for i in range(20)]
        engine = SwarmAngiogenesisEngine(agents=agents, initial_connectivity=0.1, seed=1)
        report = engine.simulate(ticks=50)
        assert report.health_score >= 0

    def test_zero_ticks(self, engine):
        report = engine.simulate(ticks=0)
        assert report.total_ticks == 0
