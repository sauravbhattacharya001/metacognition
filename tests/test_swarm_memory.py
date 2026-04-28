"""Tests for Swarm Memory module."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from src.swarm_memory import (
    Episode,
    MemoryHealth,
    Pattern,
    Recommendation,
    SwarmMemory,
    _agent_band,
    _jaccard,
    _threshold_band,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ep(
    task: str = "test task",
    agents: int = 5,
    byz: int = 1,
    threshold: float = 1.5,
    committed: bool = True,
    rounds: int = 1,
    weight: float = 2.0,
    **kw,
) -> Episode:
    return Episode(
        task=task,
        agent_count=agents,
        byzantine_count=byz,
        threshold=threshold,
        committed=committed,
        solution="answer" if committed else None,
        rounds_used=rounds,
        aggregate_weight=weight,
        timestamp=kw.get("timestamp", time.time()),
        tags=kw.get("tags", []),
    )


# ---------------------------------------------------------------------------
# Episode tests
# ---------------------------------------------------------------------------

class TestEpisode:
    def test_task_tokens(self):
        ep = _ep(task="What is the capital of France?")
        tokens = ep.task_tokens()
        assert "what" in tokens
        assert "france" in tokens

    def test_byzantine_fraction(self):
        ep = _ep(agents=10, byz=3)
        assert ep.byzantine_fraction() == pytest.approx(0.3)

    def test_byzantine_fraction_zero(self):
        ep = _ep(agents=0, byz=0)
        assert ep.byzantine_fraction() == 0.0

    def test_efficiency_committed(self):
        ep = _ep(committed=True, rounds=1, weight=2.0, threshold=1.5)
        assert ep.efficiency() > 0

    def test_efficiency_failed(self):
        ep = _ep(committed=False)
        assert ep.efficiency() == 0.0

    def test_efficiency_more_rounds_lower(self):
        e1 = _ep(rounds=1, weight=2.0)
        e2 = _ep(rounds=3, weight=2.0)
        assert e1.efficiency() > e2.efficiency()


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_jaccard_identical(self):
        assert _jaccard(["a", "b"], ["a", "b"]) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard(["a"], ["b"]) == 0.0

    def test_jaccard_empty(self):
        assert _jaccard([], []) == 1.0

    def test_jaccard_partial(self):
        assert _jaccard(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(0.5)

    def test_threshold_band(self):
        assert _threshold_band(0.5) == "low"
        assert _threshold_band(1.5) == "medium"
        assert _threshold_band(2.5) == "high"
        assert _threshold_band(4.0) == "very_high"

    def test_agent_band(self):
        assert _agent_band(2) == "small"
        assert _agent_band(5) == "medium"
        assert _agent_band(10) == "large"
        assert _agent_band(20) == "xlarge"


# ---------------------------------------------------------------------------
# SwarmMemory core tests
# ---------------------------------------------------------------------------

class TestSwarmMemory:
    def test_record(self):
        mem = SwarmMemory()
        mem.record(_ep())
        assert len(mem.episodes) == 1

    def test_record_from_result(self):
        mem = SwarmMemory()

        class FakeResult:
            committed = True
            committed_solution = "42"
            round_index = 0
            aggregate_weight = 2.5

        ep = mem.record_from_result(
            task="test", result=FakeResult(),
            agent_count=5, byzantine_count=1,
            threshold=1.5,
        )
        assert ep.committed is True
        assert ep.solution == "42"
        assert len(mem.episodes) == 1

    def test_record_from_none_result(self):
        mem = SwarmMemory()
        ep = mem.record_from_result(
            task="test", result=None,
            agent_count=5, byzantine_count=1,
            threshold=1.5,
        )
        assert ep.committed is False

    def test_find_similar_empty(self):
        mem = SwarmMemory()
        assert mem.find_similar("anything") == []

    def test_find_similar_returns_matches(self):
        mem = SwarmMemory()
        mem.record(_ep(task="What is the capital of France"))
        mem.record(_ep(task="What is the capital of Germany"))
        mem.record(_ep(task="Solve the quadratic equation"))
        results = mem.find_similar("What is the capital of Spain")
        # Should find the two capital questions as more similar
        assert len(results) >= 1
        assert "capital" in results[0][0].task.lower()


# ---------------------------------------------------------------------------
# Pattern extraction tests
# ---------------------------------------------------------------------------

class TestPatterns:
    def test_no_patterns_from_empty(self):
        mem = SwarmMemory()
        assert mem.extract_patterns() == []

    def test_no_patterns_from_single(self):
        mem = SwarmMemory()
        mem.record(_ep())
        # Need at least 2 episodes per bucket
        patterns = mem.extract_patterns()
        assert len(patterns) == 0

    def test_extracts_pattern(self):
        mem = SwarmMemory()
        for _ in range(5):
            mem.record(_ep(agents=5, threshold=1.5, committed=True))
        patterns = mem.extract_patterns()
        assert len(patterns) >= 1
        assert patterns[0].episode_count == 5
        assert patterns[0].success_rate == 1.0

    def test_separate_success_failure_patterns(self):
        mem = SwarmMemory()
        for _ in range(3):
            mem.record(_ep(agents=5, threshold=1.5, committed=True))
        for _ in range(3):
            mem.record(_ep(agents=5, threshold=1.5, committed=False))
        patterns = mem.extract_patterns()
        assert len(patterns) == 2


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------

class TestPrediction:
    def test_predict_empty(self):
        mem = SwarmMemory()
        assert mem.predict_success(5, 1.5) == 0.5

    def test_predict_all_success(self):
        mem = SwarmMemory()
        for _ in range(10):
            mem.record(_ep(agents=5, threshold=1.5, committed=True))
        prob = mem.predict_success(5, 1.5)
        assert prob > 0.7

    def test_predict_all_failure(self):
        mem = SwarmMemory()
        for _ in range(10):
            mem.record(_ep(agents=5, threshold=1.5, committed=False))
        prob = mem.predict_success(5, 1.5)
        assert prob < 0.3

    def test_predict_mixed(self):
        mem = SwarmMemory()
        for _ in range(5):
            mem.record(_ep(agents=5, threshold=1.5, committed=True))
        for _ in range(5):
            mem.record(_ep(agents=5, threshold=1.5, committed=False))
        prob = mem.predict_success(5, 1.5)
        assert 0.3 < prob < 0.7


# ---------------------------------------------------------------------------
# Recommendation tests
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_recommend_empty(self):
        mem = SwarmMemory()
        rec = mem.recommend("anything")
        assert rec.suggested_agents == 5
        assert rec.confidence == pytest.approx(0.1)

    def test_recommend_with_history(self):
        mem = SwarmMemory()
        for i in range(10):
            mem.record(_ep(
                task=f"What is the answer to question {i}",
                agents=7, threshold=2.0, committed=True,
            ))
        rec = mem.recommend("What is the answer to question 99")
        assert rec.suggested_agents >= 3
        assert rec.confidence > 0.1
        assert rec.similar_episodes > 0

    def test_recommend_low_success_conservative(self):
        mem = SwarmMemory()
        for i in range(10):
            mem.record(_ep(
                task=f"hard problem variant {i}",
                agents=3, threshold=0.8, committed=False,
            ))
        rec = mem.recommend("hard problem variant 99")
        # Should recommend more agents for safety
        assert rec.suggested_agents >= 5


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_empty(self):
        mem = SwarmMemory()
        h = mem.health_check()
        assert h.total_episodes == 0
        assert h.reliability_score == 0.0
        assert len(h.warnings) > 0

    def test_health_small_memory(self):
        mem = SwarmMemory()
        for _ in range(5):
            mem.record(_ep())
        h = mem.health_check()
        assert h.total_episodes == 5
        assert any("Small memory" in w for w in h.warnings)

    def test_health_good_coverage(self):
        mem = SwarmMemory()
        # Add diverse configs
        for agents in [2, 5, 10, 20]:
            for thresh in [0.5, 1.5, 2.5, 4.0]:
                mem.record(_ep(agents=agents, threshold=thresh))
        h = mem.health_check()
        assert h.coverage_score > 50


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self):
        mem = SwarmMemory(half_life_days=14.0)
        mem.record(_ep(task="saved task", agents=7, threshold=2.0))
        mem.record(_ep(task="another task", committed=False))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        mem.save(path)
        loaded = SwarmMemory.load(path)

        assert len(loaded.episodes) == 2
        assert loaded.episodes[0].task == "saved task"
        assert loaded.episodes[1].committed is False
        assert loaded.half_life_days == 14.0

        Path(path).unlink()

    def test_save_creates_valid_json(self):
        mem = SwarmMemory()
        mem.record(_ep())

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        mem.save(path)
        data = json.loads(Path(path).read_text())
        assert "episodes" in data
        assert len(data["episodes"]) == 1

        Path(path).unlink()


# ---------------------------------------------------------------------------
# HTML export tests
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_html_empty(self):
        mem = SwarmMemory()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        mem.export_html(path)
        html = Path(path).read_text(encoding='utf-8')
        assert "Swarm Memory Dashboard" in html
        Path(path).unlink()

    def test_export_html_with_data(self):
        mem = SwarmMemory()
        for i in range(10):
            mem.record(_ep(task=f"task {i}", agents=5, threshold=1.5))
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        mem.export_html(path)
        html = Path(path).read_text(encoding='utf-8')
        assert "10 episodes" in html
        assert "Committed" in html
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_empty(self):
        mem = SwarmMemory()
        assert "empty" in mem.summary().lower()

    def test_summary_with_data(self):
        mem = SwarmMemory()
        for _ in range(5):
            mem.record(_ep())
        s = mem.summary()
        assert "5 episodes" in s
        assert "committed" in s.lower()
