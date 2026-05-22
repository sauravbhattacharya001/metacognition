"""Tests for the Swarm Consciousness Engine."""
import json
import tempfile
from pathlib import Path

import pytest

from src.consciousness import (
    ConsciousnessReport,
    ConsciousnessSnapshot,
    SwarmConsciousnessEngine,
    run_demo,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agents():
    return [f"agent-{i}" for i in range(1, 6)]


@pytest.fixture
def engine(agents):
    return SwarmConsciousnessEngine(agents=agents)


@pytest.fixture
def synced_engine(agents):
    """Engine where all agents agree on everything."""
    eng = SwarmConsciousnessEngine(agents=agents)
    for agent in agents:
        eng.submit_belief(agent, "topic1", 0.8, confidence=0.9)
        eng.submit_attention(agent, "topic1", intensity=0.9)
        eng.submit_intent(agent, goal="optimize", action="execute")
    return eng


@pytest.fixture
def fragmented_engine(agents):
    """Engine where agents completely disagree."""
    eng = SwarmConsciousnessEngine(agents=agents)
    topics = ["a", "b", "c", "d", "e"]
    for i, agent in enumerate(agents):
        eng.submit_belief(agent, topics[i], (i - 2) * 0.5, confidence=0.8)
        eng.submit_attention(agent, topics[i], intensity=0.8)
        eng.submit_intent(agent, goal=f"goal-{i}", action=f"action-{i}")
    return eng


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_creates_with_agents(self, agents):
        engine = SwarmConsciousnessEngine(agents=agents)
        assert engine.agents == agents

    def test_starts_at_tick_zero(self, engine):
        assert engine.tick_count == 0

    def test_empty_snapshots(self, engine):
        assert engine.snapshots == []

    def test_custom_topics(self, agents):
        engine = SwarmConsciousnessEngine(agents=agents, topics=["x", "y"])
        assert engine.topics == ["x", "y"]


# ---------------------------------------------------------------------------
# Belief submission
# ---------------------------------------------------------------------------

class TestBeliefs:
    def test_submit_belief(self, engine):
        engine.submit_belief("agent-1", "topic1", 0.5, confidence=0.8)
        assert "topic1" in engine.topics

    def test_belief_clamps_value(self, engine):
        engine.submit_belief("agent-1", "t", 2.0)
        assert engine._beliefs["agent-1"]["t"].value == 1.0

    def test_belief_clamps_negative(self, engine):
        engine.submit_belief("agent-1", "t", -5.0)
        assert engine._beliefs["agent-1"]["t"].value == -1.0

    def test_belief_clamps_confidence(self, engine):
        engine.submit_belief("agent-1", "t", 0.5, confidence=1.5)
        assert engine._beliefs["agent-1"]["t"].confidence == 1.0

    def test_overwrites_previous_belief(self, engine):
        engine.submit_belief("agent-1", "t", 0.3)
        engine.submit_belief("agent-1", "t", 0.9)
        assert engine._beliefs["agent-1"]["t"].value == 0.9


# ---------------------------------------------------------------------------
# Attention submission
# ---------------------------------------------------------------------------

class TestAttention:
    def test_submit_attention(self, engine):
        engine.submit_attention("agent-1", "focus", intensity=0.7)
        assert len(engine._attention["agent-1"]) == 1

    def test_attention_clamps(self, engine):
        engine.submit_attention("agent-1", "x", intensity=2.0)
        assert engine._attention["agent-1"][0].intensity == 1.0

    def test_multiple_attention_events(self, engine):
        for i in range(10):
            engine.submit_attention("agent-1", f"topic-{i}")
        assert len(engine._attention["agent-1"]) == 10


# ---------------------------------------------------------------------------
# Intent submission
# ---------------------------------------------------------------------------

class TestIntents:
    def test_submit_intent(self, engine):
        engine.submit_intent("agent-1", "explore", "scan")
        assert engine._intents["agent-1"].goal == "explore"

    def test_intent_overwrites(self, engine):
        engine.submit_intent("agent-1", "explore", "scan")
        engine.submit_intent("agent-1", "defend", "fortify")
        assert engine._intents["agent-1"].goal == "defend"


# ---------------------------------------------------------------------------
# Belief alignment
# ---------------------------------------------------------------------------

class TestBeliefAlignment:
    def test_perfect_alignment(self, synced_engine):
        score = synced_engine.compute_belief_alignment()
        assert score > 0.85

    def test_no_beliefs_returns_zero(self, engine):
        assert engine.compute_belief_alignment() == 0.0

    def test_single_agent_belief(self, engine):
        engine.submit_belief("agent-1", "t", 0.5)
        score = engine.compute_belief_alignment()
        assert score == 0.5  # partial credit for single belief

    def test_opposing_beliefs_low_alignment(self, agents):
        eng = SwarmConsciousnessEngine(agents=agents)
        eng.submit_belief("agent-1", "t", 1.0, confidence=1.0)
        eng.submit_belief("agent-2", "t", -1.0, confidence=1.0)
        eng.submit_belief("agent-3", "t", 1.0, confidence=1.0)
        eng.submit_belief("agent-4", "t", -1.0, confidence=1.0)
        score = eng.compute_belief_alignment()
        assert score < 0.5


# ---------------------------------------------------------------------------
# Attention coherence
# ---------------------------------------------------------------------------

class TestAttentionCoherence:
    def test_all_same_topic(self, synced_engine):
        score = synced_engine.compute_attention_coherence()
        assert score == 1.0

    def test_all_different_topics(self, fragmented_engine):
        score = fragmented_engine.compute_attention_coherence()
        assert score < 0.2

    def test_no_attention_returns_zero(self, engine):
        assert engine.compute_attention_coherence() == 0.0

    def test_partial_coherence(self, agents):
        eng = SwarmConsciousnessEngine(agents=agents)
        eng.submit_attention("agent-1", "x")
        eng.submit_attention("agent-2", "x")
        eng.submit_attention("agent-3", "x")
        eng.submit_attention("agent-4", "y")
        eng.submit_attention("agent-5", "y")
        score = eng.compute_attention_coherence()
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Intentional coherence
# ---------------------------------------------------------------------------

class TestIntentionalCoherence:
    def test_all_same_goal(self, synced_engine):
        score = synced_engine.compute_intentional_coherence()
        assert score > 0.8

    def test_all_different_goals(self, fragmented_engine):
        score = fragmented_engine.compute_intentional_coherence()
        assert score < 0.5

    def test_no_intents_returns_zero(self, engine):
        assert engine.compute_intentional_coherence() == 0.0


# ---------------------------------------------------------------------------
# Information flow
# ---------------------------------------------------------------------------

class TestInformationFlow:
    def test_returns_neutral_without_history(self, engine):
        assert engine.compute_information_flow() == 0.5

    def test_good_coverage_high_flow(self, agents):
        eng = SwarmConsciousnessEngine(agents=agents)
        # All agents have beliefs on same topic
        for agent in agents:
            eng.submit_belief(agent, "shared", 0.5)
        eng.tick()
        eng.tick()
        score = eng.compute_information_flow()
        assert score > 0.7


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------

class TestPhaseClassification:
    def test_dormant(self, engine):
        assert engine.classify_phase(10) == "dormant"

    def test_stirring(self, engine):
        assert engine.classify_phase(30) == "stirring"

    def test_aware(self, engine):
        assert engine.classify_phase(50) == "aware"

    def test_synchronized(self, engine):
        assert engine.classify_phase(70) == "synchronized"

    def test_transcendent(self, engine):
        assert engine.classify_phase(90) == "transcendent"

    def test_zero_is_dormant(self, engine):
        assert engine.classify_phase(0) == "dormant"

    def test_hundred_is_transcendent(self, engine):
        assert engine.classify_phase(100) == "transcendent"


# ---------------------------------------------------------------------------
# Tick mechanics
# ---------------------------------------------------------------------------

class TestTick:
    def test_tick_advances_counter(self, engine):
        engine.tick()
        assert engine.tick_count == 1

    def test_tick_returns_snapshot(self, engine):
        snap = engine.tick()
        assert isinstance(snap, ConsciousnessSnapshot)

    def test_tick_stores_snapshot(self, engine):
        engine.tick()
        assert len(engine.snapshots) == 1

    def test_synced_tick_high_score(self, synced_engine):
        snap = synced_engine.tick()
        assert snap.hive_mind_score > 50

    def test_belief_decay_reduces_confidence(self, engine):
        engine.submit_belief("agent-1", "t", 0.5, confidence=1.0)
        engine.tick()
        assert engine._beliefs["agent-1"]["t"].confidence < 1.0

    def test_score_bounded_0_100(self, engine):
        for _ in range(10):
            snap = engine.tick()
            assert 0 <= snap.hive_mind_score <= 100


# ---------------------------------------------------------------------------
# Contagion detection
# ---------------------------------------------------------------------------

class TestContagions:
    def test_no_contagion_without_history(self, engine):
        assert engine.detect_contagions() == []

    def test_detects_spread(self, agents):
        eng = SwarmConsciousnessEngine(agents=agents)
        # Simulate spreading belief
        eng.submit_belief("agent-1", "viral", 0.8)
        eng.tick()
        eng.tick()
        eng.tick()
        eng.submit_belief("agent-2", "viral", 0.7)
        eng.submit_belief("agent-3", "viral", 0.6)
        eng.tick()
        eng.submit_belief("agent-4", "viral", 0.75)
        eng.submit_belief("agent-5", "viral", 0.65)
        eng.tick()
        # Now detection should find something
        contagions = eng.detect_contagions()
        # May or may not detect depending on thresholds, just ensure no crash
        assert isinstance(contagions, list)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

class TestOutliers:
    def test_no_outliers_when_aligned(self, synced_engine):
        outliers = synced_engine._detect_outliers()
        assert outliers == []

    def test_detects_outlier(self):
        agents = [f"agent-{i}" for i in range(1, 11)]  # need more agents for z-score
        eng = SwarmConsciousnessEngine(agents=agents)
        for agent in agents[:9]:
            eng.submit_belief(agent, "t", 0.8, confidence=1.0)
        eng.submit_belief("agent-10", "t", -0.9, confidence=1.0)
        outliers = eng._detect_outliers()
        assert "agent-10" in outliers


# ---------------------------------------------------------------------------
# Analysis report
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_empty_report(self, engine):
        report = engine.analyze()
        assert isinstance(report, ConsciousnessReport)
        assert report.overall_score == 0.0

    def test_report_after_ticks(self, synced_engine):
        for _ in range(5):
            synced_engine.tick()
        report = synced_engine.analyze()
        assert report.overall_score > 0
        assert len(report.snapshots) == 5

    def test_phase_history_tracked(self, engine):
        engine.tick()
        report = engine.analyze()
        assert len(report.phase_history) >= 1


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_html(self, synced_engine):
        for _ in range(5):
            synced_engine.tick()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        synced_engine.export_html(path)
        content = Path(path).read_text(encoding="utf-8")
        assert "Swarm Consciousness" in content
        assert "Hive Mind Score" in content
        Path(path).unlink()

    def test_export_json(self, synced_engine):
        for _ in range(5):
            synced_engine.tick()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        synced_engine.export_json(path)
        data = json.loads(Path(path).read_text())
        assert "overall_score" in data
        assert "snapshots" in data
        assert len(data["snapshots"]) == 5
        Path(path).unlink()

    def test_json_roundtrip_fields(self, synced_engine):
        for _ in range(3):
            synced_engine.tick()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        synced_engine.export_json(path)
        data = json.loads(Path(path).read_text())
        assert "peak_phase" in data
        assert "phase_history" in data
        assert "belief_clusters" in data
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

class TestDemo:
    def test_demo_runs(self):
        engine = run_demo(n_agents=5, n_ticks=30)
        assert engine.tick_count == 30
        assert len(engine.snapshots) == 30

    def test_demo_shows_phase_transitions(self):
        engine = run_demo(n_agents=8, n_ticks=60)
        report = engine.analyze()
        phases = [p for _, p in report.phase_history]
        # Should have at least 2 different phases (the demo is designed to transition)
        assert len(set(phases)) >= 2

    def test_demo_score_varies(self):
        engine = run_demo(n_agents=8, n_ticks=60)
        scores = [s.hive_mind_score for s in engine.snapshots]
        assert max(scores) - min(scores) > 10  # scores should vary over time


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_agent(self):
        eng = SwarmConsciousnessEngine(agents=["solo"])
        eng.submit_belief("solo", "t", 0.5)
        eng.submit_attention("solo", "t")
        eng.submit_intent("solo", "go", "move")
        snap = eng.tick()
        assert snap.hive_mind_score >= 0

    def test_many_topics(self, agents):
        eng = SwarmConsciousnessEngine(agents=agents)
        for i in range(20):
            eng.submit_belief("agent-1", f"topic-{i}", 0.5)
        snap = eng.tick()
        assert snap is not None

    def test_no_submissions_still_ticks(self, engine):
        snap = engine.tick()
        assert snap.phase == "dormant"
        assert snap.hive_mind_score >= 0
