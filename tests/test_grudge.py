"""Tests for src.grudge — Consensus Memory & Grudge System.

Covers Relationship properties, AgentMemory bookkeeping, GrudgeEngine
forgiveness/interaction processing, full run pipeline, and HTML report
generation.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grudge import (
    AgentMemory,
    GrudgeEngine,
    Interaction,
    Relationship,
    generate_html_report,
)
from src.core.state import RoundResult, Vote


# ── Relationship dataclass ─────────────────────────────────────────────


class TestRelationship:
    """Validate Relationship computed properties."""

    def test_default_sentiment_is_zero(self):
        rel = Relationship()
        assert rel.sentiment == 0.0

    def test_positive_sentiment(self):
        rel = Relationship(trust=0.8, grudge=0.2)
        assert rel.sentiment == pytest.approx(0.6)

    def test_negative_sentiment(self):
        rel = Relationship(trust=0.1, grudge=0.9)
        assert rel.sentiment == pytest.approx(-0.8)

    def test_is_grudge_true(self):
        rel = Relationship(trust=0.1, grudge=0.7)
        assert rel.is_grudge is True

    def test_is_grudge_false_when_trust_higher(self):
        rel = Relationship(trust=0.8, grudge=0.6)
        assert rel.is_grudge is False

    def test_is_grudge_false_when_grudge_below_threshold(self):
        rel = Relationship(trust=0.1, grudge=0.3)
        assert rel.is_grudge is False

    def test_is_alliance_true(self):
        rel = Relationship(trust=0.8, grudge=0.2)
        assert rel.is_alliance is True

    def test_is_alliance_false_when_grudge_higher(self):
        rel = Relationship(trust=0.6, grudge=0.7)
        assert rel.is_alliance is False

    def test_is_alliance_false_when_trust_below_threshold(self):
        rel = Relationship(trust=0.3, grudge=0.1)
        assert rel.is_alliance is False

    def test_neither_grudge_nor_alliance(self):
        rel = Relationship(trust=0.2, grudge=0.2)
        assert rel.is_grudge is False
        assert rel.is_alliance is False


# ── AgentMemory ────────────────────────────────────────────────────────


class TestAgentMemory:
    """Validate AgentMemory construction and get_relationship."""

    def test_default_counters(self):
        mem = AgentMemory(agent_id="a1")
        assert mem.times_led == 0
        assert mem.times_slashed == 0
        assert mem.betrayals_committed == 0
        assert mem.betrayals_received == 0
        assert mem.history == []
        assert mem.relationships == {}

    def test_get_relationship_creates_new(self):
        mem = AgentMemory(agent_id="a1")
        rel = mem.get_relationship("a2")
        assert isinstance(rel, Relationship)
        assert rel.trust == 0.0
        assert rel.grudge == 0.0

    def test_get_relationship_returns_same_instance(self):
        mem = AgentMemory(agent_id="a1")
        r1 = mem.get_relationship("a2")
        r1.trust = 0.5
        r2 = mem.get_relationship("a2")
        assert r2.trust == 0.5
        assert r1 is r2


# ── Interaction dataclass ──────────────────────────────────────────────


class TestInteraction:
    def test_fields(self):
        inter = Interaction(
            scenario=1,
            round_idx=2,
            agent_a="leader",
            agent_b="voter",
            kind="betrayal",
            intensity=0.75,
            timestamp=42,
        )
        assert inter.kind == "betrayal"
        assert inter.intensity == 0.75
        assert inter.timestamp == 42


# ── GrudgeEngine: forgiveness decay ───────────────────────────────────


class TestGrudgeEngineForgiveness:
    """Test _apply_forgiveness independently."""

    def _make_engine(self, forgiveness_rate: float = 0.1) -> GrudgeEngine:
        eng = GrudgeEngine(n_agents=2, forgiveness_rate=forgiveness_rate)
        eng.memories = {
            "a0": AgentMemory(agent_id="a0"),
            "a1": AgentMemory(agent_id="a1"),
        }
        return eng

    def test_grudge_decays(self):
        eng = self._make_engine(forgiveness_rate=0.1)
        rel = eng.memories["a0"].get_relationship("a1")
        rel.grudge = 0.5
        eng._apply_forgiveness()
        assert rel.grudge == pytest.approx(0.4)

    def test_grudge_does_not_go_negative(self):
        eng = self._make_engine(forgiveness_rate=0.5)
        rel = eng.memories["a0"].get_relationship("a1")
        rel.grudge = 0.1
        eng._apply_forgiveness()
        assert rel.grudge == 0.0

    def test_trust_decays_at_lower_rate(self):
        eng = self._make_engine(forgiveness_rate=0.1)
        rel = eng.memories["a0"].get_relationship("a1")
        rel.trust = 0.5
        eng._apply_forgiveness()
        assert rel.trust == pytest.approx(0.5 - 0.1 * 0.3)

    def test_trust_does_not_go_negative(self):
        eng = self._make_engine(forgiveness_rate=1.0)
        rel = eng.memories["a0"].get_relationship("a1")
        rel.trust = 0.01
        eng._apply_forgiveness()
        assert rel.trust == 0.0


# ── GrudgeEngine: _process_round ──────────────────────────────────────


class TestProcessRound:
    """Unit-test the round processing logic with hand-crafted RoundResults."""

    def _make_engine(self) -> GrudgeEngine:
        eng = GrudgeEngine(n_agents=3, forgiveness_rate=0.0)
        for i in range(3):
            eng.memories[f"agent_{i}"] = AgentMemory(agent_id=f"agent_{i}")
        eng.global_step = 0
        return eng

    def test_support_increases_leader_trust(self):
        eng = self._make_engine()
        rr = RoundResult(
            round_index=0,
            leader_id="agent_0",
            committed_solution="sol",
            aggregate_weight=1.5,
            threshold=1.0,
            votes=[
                Vote(voter_id="agent_1", target_proposal_id="p", weight=0.8),
            ],
            slashed=[],
        )
        eng._process_round(0, rr, {})
        # Leader's view of voter should have increased trust
        rel = eng.memories["agent_0"].get_relationship("agent_1")
        assert rel.trust > 0
        assert rel.alliance_streak == 1
        assert rel.grudge_streak == 0

    def test_rejection_increases_leader_grudge(self):
        eng = self._make_engine()
        rr = RoundResult(
            round_index=0,
            leader_id="agent_0",
            committed_solution=None,
            aggregate_weight=-0.5,
            threshold=1.0,
            votes=[
                Vote(voter_id="agent_1", target_proposal_id="p", weight=-0.9),
            ],
            slashed=[],
        )
        eng._process_round(0, rr, {})
        rel = eng.memories["agent_0"].get_relationship("agent_1")
        assert rel.grudge > 0
        assert rel.grudge_streak == 1
        assert rel.alliance_streak == 0
        assert eng.memories["agent_0"].betrayals_received == 1
        assert eng.memories["agent_1"].betrayals_committed == 1

    def test_leader_times_led_increments(self):
        eng = self._make_engine()
        rr = RoundResult(
            round_index=0, leader_id="agent_2",
            committed_solution="sol", aggregate_weight=1.0, threshold=1.0,
            votes=[], slashed=[],
        )
        eng._process_round(0, rr, {})
        assert eng.memories["agent_2"].times_led == 1

    def test_slashed_agents_tracked(self):
        eng = self._make_engine()
        rr = RoundResult(
            round_index=0, leader_id="agent_0",
            committed_solution=None, aggregate_weight=-1.0, threshold=1.0,
            votes=[], slashed=["agent_0"],
        )
        eng._process_round(0, rr, {})
        assert eng.memories["agent_0"].times_slashed == 1

    def test_interactions_recorded_in_both_memories(self):
        eng = self._make_engine()
        rr = RoundResult(
            round_index=0, leader_id="agent_0",
            committed_solution="sol", aggregate_weight=1.5, threshold=1.0,
            votes=[
                Vote(voter_id="agent_1", target_proposal_id="p", weight=0.7),
            ],
            slashed=[],
        )
        eng._process_round(0, rr, {})
        assert len(eng.memories["agent_0"].history) == 1
        assert len(eng.memories["agent_1"].history) == 1
        assert eng.all_interactions[-1].kind == "support"

    def test_multiple_votes_in_round(self):
        eng = self._make_engine()
        rr = RoundResult(
            round_index=0, leader_id="agent_0",
            committed_solution=None, aggregate_weight=0.0, threshold=1.0,
            votes=[
                Vote(voter_id="agent_1", target_proposal_id="p", weight=0.5),
                Vote(voter_id="agent_2", target_proposal_id="p", weight=-0.8),
            ],
            slashed=[],
        )
        eng._process_round(0, rr, {})
        # Agent_1 supported, agent_2 betrayed
        rel1 = eng.memories["agent_0"].get_relationship("agent_1")
        rel2 = eng.memories["agent_0"].get_relationship("agent_2")
        assert rel1.trust > 0
        assert rel2.grudge > 0


# ── GrudgeEngine: full run pipeline ───────────────────────────────────


class TestGrudgeEngineFullRun:
    """Integration tests for the complete run() pipeline."""

    def test_small_run_produces_valid_results(self):
        eng = GrudgeEngine(
            n_agents=3, n_rounds=2, n_scenarios=3,
            threshold=1.5, seed=42,
        )
        results = asyncio.get_event_loop().run_until_complete(eng.run())

        # Structure checks
        assert "matrix" in results
        assert "profiles" in results
        assert "grudges" in results
        assert "alliances" in results
        assert "stability" in results
        assert "recommendations" in results
        assert "timeline" in results
        assert "config" in results

        # Right number of agents
        assert len(results["profiles"]) == 3
        assert len(results["matrix"]) == 3

        # Matrix is square and self-sentiment is 0
        for aid in results["matrix"]:
            assert results["matrix"][aid][aid] == 0.0

        # Stability in [0, 1]
        assert 0.0 <= results["stability"] <= 1.0

        # At least one recommendation
        assert len(results["recommendations"]) >= 1

        # Total interactions > 0
        assert results["total_interactions"] > 0

    def test_timeline_has_one_snapshot_per_scenario(self):
        eng = GrudgeEngine(
            n_agents=3, n_rounds=2, n_scenarios=5, seed=7,
        )
        results = asyncio.get_event_loop().run_until_complete(eng.run())
        assert len(results["timeline"]) == 5
        for i, snap in enumerate(results["timeline"]):
            assert snap["scenario"] == i

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        def do_run():
            eng = GrudgeEngine(n_agents=4, n_rounds=2, n_scenarios=5, seed=123)
            return asyncio.get_event_loop().run_until_complete(eng.run())

        r1 = do_run()
        r2 = do_run()
        assert r1["stability"] == r2["stability"]
        assert r1["total_interactions"] == r2["total_interactions"]
        assert r1["profiles"] == r2["profiles"]

    def test_high_forgiveness_reduces_grudges(self):
        eng_low = GrudgeEngine(
            n_agents=4, n_rounds=3, n_scenarios=10,
            forgiveness_rate=0.01, seed=55,
        )
        eng_high = GrudgeEngine(
            n_agents=4, n_rounds=3, n_scenarios=10,
            forgiveness_rate=0.5, seed=55,
        )
        r_low = asyncio.get_event_loop().run_until_complete(eng_low.run())
        r_high = asyncio.get_event_loop().run_until_complete(eng_high.run())
        # Higher forgiveness → higher stability (fewer grudges)
        assert r_high["stability"] >= r_low["stability"]

    def test_config_captured_in_results(self):
        eng = GrudgeEngine(
            n_agents=5, n_rounds=3, n_scenarios=2,
            threshold=2.5, grudge_threshold=0.6, forgiveness_rate=0.08, seed=1,
        )
        results = asyncio.get_event_loop().run_until_complete(eng.run())
        cfg = results["config"]
        assert cfg["n_agents"] == 5
        assert cfg["n_scenarios"] == 2
        assert cfg["n_rounds"] == 3
        assert cfg["threshold"] == 2.5
        assert cfg["grudge_threshold"] == 0.6
        assert cfg["forgiveness_rate"] == 0.08


# ── GrudgeEngine: _snapshot ───────────────────────────────────────────


class TestSnapshot:
    def test_snapshot_structure(self):
        eng = GrudgeEngine(n_agents=2, seed=1)
        eng.memories = {
            "a0": AgentMemory(agent_id="a0"),
            "a1": AgentMemory(agent_id="a1"),
        }
        rel = eng.memories["a0"].get_relationship("a1")
        rel.trust = 0.8
        rel.grudge = 0.1

        snap = eng._snapshot(5)
        assert snap["scenario"] == 5
        assert "a0->a1" in snap["relationships"]
        pair = snap["relationships"]["a0->a1"]
        assert pair["trust"] == 0.8
        assert pair["grudge"] == 0.1
        assert pair["sentiment"] == pytest.approx(0.7)
        assert pair["is_alliance"] is True
        assert pair["is_grudge"] is False


# ── HTML report generation ────────────────────────────────────────────


class TestHtmlReport:
    def test_generates_valid_html(self):
        eng = GrudgeEngine(
            n_agents=3, n_rounds=2, n_scenarios=3, seed=42,
        )
        results = asyncio.get_event_loop().run_until_complete(eng.run())
        html = generate_html_report(results)

        assert "<!DOCTYPE html>" in html
        assert "Consensus Memory" in html
        assert "Relationship Heatmap" in html
        assert "agent_0" in html
        assert "Recommendations" in html

    def test_html_contains_agent_profiles(self):
        eng = GrudgeEngine(
            n_agents=4, n_rounds=2, n_scenarios=2, seed=99,
        )
        results = asyncio.get_event_loop().run_until_complete(eng.run())
        html = generate_html_report(results)
        for i in range(4):
            assert f"agent_{i}" in html

    def test_html_contains_stability(self):
        eng = GrudgeEngine(
            n_agents=3, n_rounds=2, n_scenarios=2, seed=10,
        )
        results = asyncio.get_event_loop().run_until_complete(eng.run())
        html = generate_html_report(results)
        assert str(results["stability"]) in html


# ── Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_scenario(self):
        eng = GrudgeEngine(n_agents=2, n_rounds=1, n_scenarios=1, seed=42)
        results = asyncio.get_event_loop().run_until_complete(eng.run())
        assert results["total_interactions"] > 0
        assert len(results["timeline"]) == 1

    def test_many_agents(self):
        eng = GrudgeEngine(n_agents=10, n_rounds=2, n_scenarios=2, seed=7)
        results = asyncio.get_event_loop().run_until_complete(eng.run())
        assert len(results["profiles"]) == 10
        assert len(results["matrix"]) == 10
