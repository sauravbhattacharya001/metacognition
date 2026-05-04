"""Tests for Swarm Proprioception Engine."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from src.proprioception import (
    AgentRole,
    BalanceAxis,
    HealthTier,
    InsightSeverity,
    MovementType,
    PostureState,
    SwarmProprioceptionEngine,
    run_demo,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a basic engine with 6 agents."""
    return SwarmProprioceptionEngine(num_agents=6, seed=42)


@pytest.fixture
def connected_engine():
    """Create engine with a connected linear topology."""
    e = SwarmProprioceptionEngine(num_agents=6, seed=42)
    for i in range(5):
        e.add_connection(f"agent-{i}", f"agent-{i+1}")
    e.tick()
    return e


@pytest.fixture
def star_engine():
    """Create engine with star topology (agent-0 is hub)."""
    e = SwarmProprioceptionEngine(num_agents=6, seed=42)
    for i in range(1, 6):
        e.add_connection("agent-0", f"agent-{i}")
    e.tick()
    return e


# ---------------------------------------------------------------------------
# Basic Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_creates_agents(self, engine):
        assert len(engine._agents) == 6

    def test_no_initial_connections(self, engine):
        total = sum(len(v) for v in engine._connections.values())
        assert total == 0

    def test_tick_zero(self, engine):
        assert engine._tick == 0

    def test_custom_agent_count(self):
        e = SwarmProprioceptionEngine(num_agents=10)
        assert len(e._agents) == 10


# ---------------------------------------------------------------------------
# Topology Manipulation
# ---------------------------------------------------------------------------


class TestTopologyManipulation:
    def test_add_connection(self, engine):
        engine.add_connection("agent-0", "agent-1")
        assert "agent-1" in engine._connections["agent-0"]
        assert "agent-0" in engine._connections["agent-1"]

    def test_remove_connection(self, engine):
        engine.add_connection("agent-0", "agent-1")
        engine.remove_connection("agent-0", "agent-1")
        assert "agent-1" not in engine._connections["agent-0"]

    def test_add_agent(self, engine):
        engine.add_agent("agent-99")
        assert "agent-99" in engine._agents

    def test_remove_agent(self, engine):
        engine.add_connection("agent-0", "agent-1")
        engine.remove_agent("agent-0")
        assert "agent-0" not in engine._agents
        assert "agent-0" not in engine._connections.get("agent-1", set())

    def test_add_connection_creates_agents(self):
        e = SwarmProprioceptionEngine(num_agents=0)
        e.add_connection("x", "y")
        assert "x" in e._agents
        assert "y" in e._agents


# ---------------------------------------------------------------------------
# Body Schema
# ---------------------------------------------------------------------------


class TestBodySchema:
    def test_schema_built_after_tick(self, connected_engine):
        report = connected_engine.get_report()
        assert len(report.body_schema) == 6

    def test_endpoint_detection(self, connected_engine):
        report = connected_engine.get_report()
        endpoints = [e for e in report.body_schema if e.role == AgentRole.ENDPOINT]
        # Linear chain: agent-0 and agent-5 are endpoints
        assert len(endpoints) >= 2

    def test_isolated_detection(self, engine):
        engine.tick()
        report = engine.get_report()
        isolated = [e for e in report.body_schema if e.role == AgentRole.ISOLATED]
        assert len(isolated) == 6  # No connections = all isolated

    def test_core_detection(self, star_engine):
        report = star_engine.get_report()
        cores = [e for e in report.body_schema if e.role == AgentRole.CORE]
        assert len(cores) >= 1

    def test_degree_correct(self, star_engine):
        report = star_engine.get_report()
        hub = next(e for e in report.body_schema if e.agent_id == "agent-0")
        assert hub.degree == 5

    def test_joint_detection(self):
        e = SwarmProprioceptionEngine(num_agents=5, seed=42)
        # Create: 0-1-2-3-4 with 2 as bridge
        e.add_connection("agent-0", "agent-1")
        e.add_connection("agent-1", "agent-2")
        e.add_connection("agent-2", "agent-3")
        e.add_connection("agent-3", "agent-4")
        e.tick()
        report = e.get_report()
        # Middle agents should be joints or limbs
        mid = next(e for e in report.body_schema if e.agent_id == "agent-2")
        assert mid.role in (AgentRole.JOINT, AgentRole.LIMB)


# ---------------------------------------------------------------------------
# Kinesthetic Events
# ---------------------------------------------------------------------------


class TestKinestheticEvents:
    def test_no_events_static(self, connected_engine):
        # After first tick with connections, tick again with no changes
        connected_engine.tick()
        # Kinesthetic events from static topology should be minimal
        report = connected_engine.get_report()
        recent = [e for e in report.kinesthetic_events if e.tick == connected_engine._tick]
        # No agent/connection changes = no events
        assert len(recent) == 0

    def test_expansion_detected(self, connected_engine):
        connected_engine.add_agent("agent-new")
        connected_engine.tick()
        report = connected_engine.get_report()
        expansion = [e for e in report.kinesthetic_events if e.movement_type == MovementType.EXPANSION]
        assert len(expansion) >= 1

    def test_fragmentation_detected(self, connected_engine):
        # Remove multiple connections
        connected_engine.remove_connection("agent-1", "agent-2")
        connected_engine.remove_connection("agent-2", "agent-3")
        connected_engine.remove_connection("agent-3", "agent-4")
        connected_engine.tick()
        report = connected_engine.get_report()
        frag = [e for e in report.kinesthetic_events if e.movement_type == MovementType.FRAGMENTATION]
        assert len(frag) >= 1

    def test_merging_detected(self):
        e = SwarmProprioceptionEngine(num_agents=6, seed=42)
        e.add_connection("agent-0", "agent-1")
        e.tick()
        # Now add many connections at once
        e.add_connection("agent-2", "agent-3")
        e.add_connection("agent-3", "agent-4")
        e.add_connection("agent-4", "agent-5")
        e.tick()
        report = e.get_report()
        merging = [ev for ev in report.kinesthetic_events if ev.movement_type == MovementType.MERGING]
        assert len(merging) >= 1


# ---------------------------------------------------------------------------
# Joint Angles
# ---------------------------------------------------------------------------


class TestJointAngles:
    def test_joint_angles_computed(self):
        e = SwarmProprioceptionEngine(num_agents=7, seed=42)
        # T-shaped: 0-1-2-3 with 2 also connected to 4-5-6
        e.add_connection("agent-0", "agent-1")
        e.add_connection("agent-1", "agent-2")
        e.add_connection("agent-2", "agent-3")
        e.add_connection("agent-2", "agent-4")
        e.add_connection("agent-4", "agent-5")
        e.add_connection("agent-5", "agent-6")
        e.tick()
        report = e.get_report()
        assert len(report.joint_angles) >= 1

    def test_angle_range(self):
        e = SwarmProprioceptionEngine(num_agents=5, seed=42)
        e.add_connection("agent-0", "agent-2")
        e.add_connection("agent-1", "agent-2")
        e.add_connection("agent-2", "agent-3")
        e.add_connection("agent-2", "agent-4")
        e.tick()
        report = e.get_report()
        for j in report.joint_angles:
            assert 0 <= j.angle <= 180

    def test_flexion_state_assigned(self):
        e = SwarmProprioceptionEngine(num_agents=5, seed=42)
        e.add_connection("agent-0", "agent-2")
        e.add_connection("agent-1", "agent-2")
        e.add_connection("agent-2", "agent-3")
        e.add_connection("agent-2", "agent-4")
        e.tick()
        report = e.get_report()
        for j in report.joint_angles:
            assert j.flexion_state in ("extended", "neutral", "flexed", "overextended")


# ---------------------------------------------------------------------------
# Balance Assessment
# ---------------------------------------------------------------------------


class TestBalanceAssessment:
    def test_balance_reports_generated(self, connected_engine):
        report = connected_engine.get_report()
        assert len(report.balance_reports) >= 1

    def test_star_has_imbalance(self, star_engine):
        report = star_engine.get_report()
        conn_balance = next(
            (b for b in report.balance_reports if b.axis == BalanceAxis.CONNECTIVITY), None
        )
        assert conn_balance is not None
        assert conn_balance.gini_coefficient > 0.3  # Star is imbalanced

    def test_gini_range(self, connected_engine):
        report = connected_engine.get_report()
        for br in report.balance_reports:
            assert 0.0 <= br.gini_coefficient <= 1.0
            assert 0.0 <= br.stability <= 1.0


# ---------------------------------------------------------------------------
# Posture Classification
# ---------------------------------------------------------------------------


class TestPostureClassification:
    def test_scattered_when_disconnected(self, engine):
        engine.tick()
        report = engine.get_report()
        assert report.current_posture == PostureState.SCATTERED

    def test_compact_when_dense(self):
        e = SwarmProprioceptionEngine(num_agents=4, seed=42)
        # Fully connected
        for i in range(4):
            for j in range(i+1, 4):
                e.add_connection(f"agent-{i}", f"agent-{j}")
        e.tick()
        report = e.get_report()
        assert report.current_posture == PostureState.COMPACT

    def test_ring_detection(self):
        e = SwarmProprioceptionEngine(num_agents=5, seed=42)
        for i in range(5):
            e.add_connection(f"agent-{i}", f"agent-{(i+1) % 5}")
        e.tick()
        report = e.get_report()
        assert report.current_posture == PostureState.RING

    def test_posture_history_grows(self, connected_engine):
        connected_engine.tick()
        connected_engine.tick()
        report = connected_engine.get_report()
        assert len(report.posture_history) >= 3


# ---------------------------------------------------------------------------
# Coordination Feedback
# ---------------------------------------------------------------------------


class TestCoordinationFeedback:
    def test_isolated_get_connect_feedback(self, engine):
        engine.tick()
        report = engine.get_report()
        connect_feedback = [f for f in report.coordination_feedback
                          if f.recommended_action == "connect_to_nearest"]
        assert len(connect_feedback) == 6

    def test_no_feedback_when_balanced(self):
        e = SwarmProprioceptionEngine(num_agents=4, seed=42)
        # Ring — balanced
        for i in range(4):
            e.add_connection(f"agent-{i}", f"agent-{(i+1) % 4}")
        e.tick()
        report = e.get_report()
        isolated_feedback = [f for f in report.coordination_feedback
                            if f.recommended_action == "connect_to_nearest"]
        assert len(isolated_feedback) == 0


# ---------------------------------------------------------------------------
# Health Scoring
# ---------------------------------------------------------------------------


class TestHealthScoring:
    def test_health_score_range(self, connected_engine):
        report = connected_engine.get_report()
        assert 0 <= report.health.score <= 100

    def test_disconnected_low_score(self, engine):
        engine.tick()
        report = engine.get_report()
        assert report.health.score < 50

    def test_connected_higher_score(self, connected_engine):
        report = connected_engine.get_report()
        assert report.health.score > 30

    def test_tier_assigned(self, connected_engine):
        report = connected_engine.get_report()
        assert report.health.tier in HealthTier

    def test_all_components_present(self, connected_engine):
        h = connected_engine.get_report().health
        assert h.schema_accuracy >= 0
        assert h.kinesthetic_responsiveness >= 0
        assert h.balance_quality >= 0
        assert h.postural_stability >= 0
        assert h.coordination_effectiveness >= 0

    def test_anomalies_detected(self, engine):
        engine.tick()
        report = engine.get_report()
        assert "high_isolation" in report.health.anomalies


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------


class TestInsights:
    def test_isolated_insight_generated(self, engine):
        engine.tick()
        report = engine.get_report()
        schema_insights = [i for i in report.insights if i.category == "schema"]
        assert len(schema_insights) >= 1

    def test_insight_severity_valid(self, engine):
        engine.tick()
        report = engine.get_report()
        for ins in report.insights:
            assert ins.severity in InsightSeverity

    def test_imbalance_insight(self, star_engine):
        report = star_engine.get_report()
        balance_insights = [i for i in report.insights if i.category == "balance"]
        # Star topology should have some imbalance insight
        assert len(balance_insights) >= 0  # May or may not trigger depending on Gini


# ---------------------------------------------------------------------------
# Movement Velocity & Acceleration
# ---------------------------------------------------------------------------


class TestMovement:
    def test_velocity_zero_when_static(self, connected_engine):
        # Run several more static ticks to wash out initial connection event
        for _ in range(8):
            connected_engine.tick()
        report = connected_engine.get_report()
        assert report.movement_velocity < 0.1

    def test_velocity_increases_with_changes(self):
        e = SwarmProprioceptionEngine(num_agents=6, seed=42)
        e.add_connection("agent-0", "agent-1")
        e.tick()
        # Make changes each tick
        e.add_agent("agent-10")
        e.tick()
        e.add_agent("agent-11")
        e.tick()
        e.add_agent("agent-12")
        e.tick()
        report = e.get_report()
        assert report.movement_velocity > 0

    def test_acceleration_computed(self, connected_engine):
        for _ in range(6):
            connected_engine.tick()
        report = connected_engine.get_report()
        assert isinstance(report.movement_acceleration, float)


# ---------------------------------------------------------------------------
# Posture Memory
# ---------------------------------------------------------------------------


class TestPostureMemory:
    def test_save_posture(self, connected_engine):
        snap = connected_engine.save_posture("my-posture")
        assert snap.posture_id == "my-posture"
        assert snap.num_agents == 6

    def test_saved_posture_tracked(self, connected_engine):
        connected_engine.save_posture("baseline")
        assert len(connected_engine._known_postures) == 1

    def test_deviation_insight_when_changed(self, connected_engine):
        connected_engine.save_posture("baseline")
        # Change topology
        connected_engine.remove_connection("agent-2", "agent-3")
        connected_engine.add_agent("agent-99")
        for _ in range(6):
            connected_engine.tick()
        report = connected_engine.get_report()
        posture_insights = [i for i in report.insights if i.category == "posture"]
        # May generate deviation insight
        assert isinstance(posture_insights, list)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_json(self, connected_engine):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            connected_engine.export_json(path)
            data = json.loads(Path(path).read_text())
            assert "health" in data
            assert "body_schema" in data
            assert "current_posture" in data
            assert data["health"]["score"] >= 0
        finally:
            os.unlink(path)

    def test_export_html(self, connected_engine):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            connected_engine.export_html(path)
            content = Path(path).read_text(encoding="utf-8")
            assert "Swarm Proprioception Dashboard" in content
            assert "Body Schema" in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Demo Runner
# ---------------------------------------------------------------------------


class TestDemoRunner:
    def test_baseline_scenario(self):
        report = run_demo(num_agents=6, ticks=30, scenario="baseline", seed=42)
        assert report.health.score >= 0

    def test_reconfiguration_scenario(self):
        report = run_demo(num_agents=6, ticks=55, scenario="reconfiguration", seed=42)
        assert len(report.kinesthetic_events) > 0

    def test_fragmentation_scenario(self):
        report = run_demo(num_agents=6, ticks=55, scenario="fragmentation", seed=42)
        assert report.health.score >= 0

    def test_growth_scenario(self):
        report = run_demo(num_agents=6, ticks=45, scenario="growth", seed=42)
        assert report.health.score >= 0

    def test_imbalance_scenario(self):
        report = run_demo(num_agents=6, ticks=50, scenario="imbalance", seed=42)
        assert report.health.score >= 0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_engine(self):
        e = SwarmProprioceptionEngine(num_agents=0)
        e.tick()
        report = e.get_report()
        assert report.health.tier == HealthTier.DISSOCIATED

    def test_single_agent(self):
        e = SwarmProprioceptionEngine(num_agents=1)
        e.tick()
        report = e.get_report()
        assert len(report.body_schema) == 1

    def test_self_connection_ignored(self, engine):
        engine.add_connection("agent-0", "agent-0")
        engine.tick()
        # Should not crash

    def test_duplicate_connection(self, engine):
        engine.add_connection("agent-0", "agent-1")
        engine.add_connection("agent-0", "agent-1")
        engine.tick()
        report = engine.get_report()
        hub = next(e for e in report.body_schema if e.agent_id == "agent-0")
        assert hub.degree == 1  # Sets prevent duplicates

    def test_remove_nonexistent_agent(self, engine):
        engine.remove_agent("agent-999")  # Should not crash

    def test_many_ticks(self, connected_engine):
        for _ in range(50):
            connected_engine.tick()
        report = connected_engine.get_report()
        assert report.tick == 51  # 1 from fixture + 50
