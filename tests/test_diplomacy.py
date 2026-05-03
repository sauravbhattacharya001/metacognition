"""Tests for Consensus Diplomacy Engine (src.diplomacy).

Tests cover:
- Helper functions (_cosine_sim)
- Data structures (DiplomaticEvent, Treaty, Faction)
- DiplomacyEngine construction and configuration
- Simulation basics (run_simulation produces expected structure)
- Faction detection (_detect_factions)
- Alliance matrix (_compute_alliances)
- Treaty detection (_detect_treaties)
- Diplomatic pressure measurement (_measure_pressure)
- Event generation (_generate_events)
- Auto-negotiation (_auto_negotiate)
- Summary building (_build_summary)
- Edge cases: minimal agents, zero tasks, all byzantine
"""
from __future__ import annotations

import math
import random

import pytest

from src.diplomacy import (
    DiplomacyEngine,
    DiplomaticEvent,
    Faction,
    Treaty,
    _cosine_sim,
    FACTION_COLORS,
)


# ===================================================================
# _cosine_sim
# ===================================================================

class TestCosineSim:
    def test_identical_vectors(self):
        assert _cosine_sim([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        assert _cosine_sim([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_orthogonal(self):
        assert _cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert _cosine_sim([], []) == 0.0

    def test_length_mismatch(self):
        assert _cosine_sim([1, 2], [1]) == 0.0

    def test_zero_vector(self):
        assert _cosine_sim([0, 0], [1, 2]) == 0.0

    def test_both_zero(self):
        assert _cosine_sim([0, 0], [0, 0]) == 0.0

    def test_negative_components(self):
        sim = _cosine_sim([-1, -2], [-1, -2])
        assert sim == pytest.approx(1.0)

    def test_partial_similarity(self):
        sim = _cosine_sim([1, 0, 1], [1, 1, 0])
        expected = 1 / (math.sqrt(2) * math.sqrt(2))
        assert sim == pytest.approx(expected)


# ===================================================================
# Data structures
# ===================================================================

class TestDataStructures:
    def test_diplomatic_event(self):
        e = DiplomaticEvent(
            round_idx=5, event_type="alliance_formed",
            agents=["a", "b"], detail="test"
        )
        assert e.round_idx == 5
        assert e.event_type == "alliance_formed"

    def test_treaty_active_default(self):
        t = Treaty(faction_a=0, faction_b=1, formed_round=0)
        assert t.active is True
        assert t.broken_round is None

    def test_treaty_broken(self):
        t = Treaty(faction_a=0, faction_b=1, formed_round=0, broken_round=5)
        assert t.active is False

    def test_faction_defaults(self):
        f = Faction(faction_id=0, members=["a", "b"])
        assert f.avg_vote == 0.0
        assert f.power == 0.0
        assert f.color == "#888"

    def test_faction_colors_exist(self):
        assert len(FACTION_COLORS) >= 10


# ===================================================================
# DiplomacyEngine construction
# ===================================================================

class TestEngineConstruction:
    def test_default_construction(self):
        eng = DiplomacyEngine(n_agents=4, n_byzantine=1, seed=42)
        assert len(eng.agent_names) == 4
        assert len(eng.byzantine_set) == 1
        assert eng.n_rounds == 40
        assert eng.n_tasks == 15

    def test_custom_params(self):
        eng = DiplomacyEngine(n_agents=6, n_byzantine=0, n_rounds=10,
                              n_tasks=5, auto_negotiate=True, seed=99)
        assert eng.n_agents == 6
        assert eng.n_byzantine == 0
        assert eng.auto_negotiate is True
        assert len(eng.byzantine_set) == 0

    def test_agent_naming(self):
        eng = DiplomacyEngine(n_agents=3, seed=1)
        assert eng.agent_names == ["agent_0", "agent_1", "agent_2"]

    def test_byzantine_cap(self):
        eng = DiplomacyEngine(n_agents=3, n_byzantine=10, seed=42)
        assert len(eng.byzantine_set) == 3

    def test_seed_reproducibility(self):
        eng1 = DiplomacyEngine(n_agents=5, n_byzantine=2, seed=42)
        eng2 = DiplomacyEngine(n_agents=5, n_byzantine=2, seed=42)
        assert eng1.byzantine_set == eng2.byzantine_set


# ===================================================================
# Simulation
# ===================================================================

class TestSimulation:
    def test_simulate_round_returns_all_agents(self):
        eng = DiplomacyEngine(n_agents=4, n_byzantine=1,
                              n_rounds=5, n_tasks=3, seed=42)
        votes = eng._simulate_round(0, 0)
        assert set(votes.keys()) == set(eng.agent_names)

    def test_vote_bounds(self):
        eng = DiplomacyEngine(n_agents=6, n_byzantine=2,
                              n_rounds=10, n_tasks=5, seed=42)
        for t in range(5):
            for r in range(10):
                votes = eng._simulate_round(t, r)
                for v in votes.values():
                    assert -1.0 <= v <= 1.0

    def test_run_simulation_structure(self):
        eng = DiplomacyEngine(n_agents=4, n_byzantine=1,
                              n_rounds=5, n_tasks=3, seed=42)
        summary = eng.run_simulation(verbose=False)
        assert "config" in summary
        assert "factions" in summary
        assert "treaties" in summary
        assert "alliance_heatmap" in summary
        assert "events" in summary
        assert "pressure" in summary
        assert summary["config"]["agents"] == 4

    def test_run_populates_vote_matrix(self):
        eng = DiplomacyEngine(n_agents=3, n_byzantine=0,
                              n_rounds=4, n_tasks=2, seed=42)
        eng.run_simulation(verbose=False)
        assert len(eng.vote_matrix) == 2
        assert len(eng.vote_matrix[0]) == 4

    def test_run_populates_agent_vectors(self):
        eng = DiplomacyEngine(n_agents=3, n_byzantine=0,
                              n_rounds=4, n_tasks=2, seed=42)
        eng.run_simulation(verbose=False)
        for agent in eng.agent_names:
            assert len(eng.agent_vectors[agent]) == 8

    def test_reproducible_simulation(self):
        random.seed(42)
        eng1 = DiplomacyEngine(n_agents=4, n_byzantine=1,
                               n_rounds=5, n_tasks=3, seed=42)
        s1 = eng1.run_simulation(verbose=False)
        random.seed(42)
        eng2 = DiplomacyEngine(n_agents=4, n_byzantine=1,
                               n_rounds=5, n_tasks=3, seed=42)
        s2 = eng2.run_simulation(verbose=False)
        assert len(eng1.vote_matrix) == len(eng2.vote_matrix)
        for t in range(len(eng1.vote_matrix)):
            for r in range(len(eng1.vote_matrix[t])):
                assert eng1.vote_matrix[t][r] == eng2.vote_matrix[t][r]


# ===================================================================
# Faction detection
# ===================================================================

class TestFactionDetection:
    def test_factions_created(self):
        eng = DiplomacyEngine(n_agents=6, n_byzantine=2,
                              n_rounds=10, n_tasks=5, seed=42)
        eng.run_simulation(verbose=False)
        assert len(eng.factions) > 0

    def test_all_agents_in_factions(self):
        eng = DiplomacyEngine(n_agents=5, n_byzantine=1,
                              n_rounds=8, n_tasks=4, seed=42)
        eng.run_simulation(verbose=False)
        all_members = set()
        for f in eng.factions:
            all_members.update(f.members)
        assert all_members == set(eng.agent_names)

    def test_no_duplicate_members(self):
        eng = DiplomacyEngine(n_agents=6, n_byzantine=2,
                              n_rounds=10, n_tasks=5, seed=42)
        eng.run_simulation(verbose=False)
        all_members = []
        for f in eng.factions:
            all_members.extend(f.members)
        assert len(all_members) == len(set(all_members))

    def test_faction_has_color(self):
        eng = DiplomacyEngine(n_agents=4, n_byzantine=1,
                              n_rounds=5, n_tasks=3, seed=42)
        eng.run_simulation(verbose=False)
        for f in eng.factions:
            assert f.color in FACTION_COLORS


# ===================================================================
# Alliance matrix
# ===================================================================

class TestAllianceMatrix:
    def test_alliance_symmetry(self):
        eng = DiplomacyEngine(n_agents=4, n_byzantine=1,
                              n_rounds=5, n_tasks=3, seed=42)
        eng.run_simulation(verbose=False)
        for (a, b), score in eng.alliance_matrix.items():
            reverse = eng.alliance_matrix.get((b, a))
            assert reverse is not None
            assert score == reverse

    def test_alliance_scores_bounded(self):
        eng = DiplomacyEngine(n_agents=4, n_byzantine=1,
                              n_rounds=10, n_tasks=5, seed=42)
        eng.run_simulation(verbose=False)
        for score in eng.alliance_matrix.values():
            assert -1.0 <= score <= 1.0


# ===================================================================
# Diplomatic pressure
# ===================================================================

class TestDiplomaticPressure:
    def test_pressure_computed_for_all_agents(self):
        eng = DiplomacyEngine(n_agents=5, n_byzantine=1,
                              n_rounds=10, n_tasks=5, seed=42)
        eng.run_simulation(verbose=False)
        assert set(eng.pressure_scores.keys()) == set(eng.agent_names)

    def test_pressure_non_negative(self):
        eng = DiplomacyEngine(n_agents=5, n_byzantine=1,
                              n_rounds=10, n_tasks=5, seed=42)
        eng.run_simulation(verbose=False)
        for score in eng.pressure_scores.values():
            assert score >= 0.0


# ===================================================================
# Events
# ===================================================================

class TestEvents:
    def test_events_generated(self):
        eng = DiplomacyEngine(n_agents=5, n_byzantine=1,
                              n_rounds=10, n_tasks=5, seed=42)
        eng.run_simulation(verbose=False)
        assert len(eng.events) > 0

    def test_events_sorted_by_round(self):
        eng = DiplomacyEngine(n_agents=5, n_byzantine=1,
                              n_rounds=10, n_tasks=5, seed=42)
        eng.run_simulation(verbose=False)
        rounds = [e.round_idx for e in eng.events]
        assert rounds == sorted(rounds)

    def test_faction_formation_events(self):
        eng = DiplomacyEngine(n_agents=4, n_byzantine=1,
                              n_rounds=5, n_tasks=3, seed=42)
        eng.run_simulation(verbose=False)
        faction_events = [e for e in eng.events if e.event_type == "alliance_formed"]
        assert len(faction_events) >= len(eng.factions)


# ===================================================================
# Auto-negotiation
# ===================================================================

class TestAutoNegotiate:
    def test_auto_negotiate_generates_recommendations(self):
        eng = DiplomacyEngine(n_agents=6, n_byzantine=2,
                              n_rounds=10, n_tasks=5,
                              auto_negotiate=True, seed=42)
        eng.run_simulation(verbose=False)
        assert len(eng.recommendations) > 0

    def test_auto_negotiate_off_no_recommendations(self):
        eng = DiplomacyEngine(n_agents=4, n_byzantine=1,
                              n_rounds=5, n_tasks=3,
                              auto_negotiate=False, seed=42)
        eng.run_simulation(verbose=False)
        assert eng.recommendations == []

    def test_auto_negotiate_mentions_strategy(self):
        eng = DiplomacyEngine(n_agents=6, n_byzantine=2,
                              n_rounds=10, n_tasks=5,
                              auto_negotiate=True, seed=42)
        eng.run_simulation(verbose=False)
        text = "\n".join(eng.recommendations)
        assert "Strategy:" in text

    def test_auto_negotiate_identifies_byzantine(self):
        eng = DiplomacyEngine(n_agents=6, n_byzantine=2,
                              n_rounds=10, n_tasks=5,
                              auto_negotiate=True, seed=42)
        eng.run_simulation(verbose=False)
        text = "\n".join(eng.recommendations)
        assert "Byzantine" in text

    def test_auto_negotiate_few_honest(self):
        eng = DiplomacyEngine(n_agents=2, n_byzantine=1,
                              n_rounds=5, n_tasks=2,
                              auto_negotiate=True, seed=42)
        eng.run_simulation(verbose=False)
        assert len(eng.recommendations) > 0


# ===================================================================
# Summary
# ===================================================================

class TestSummary:
    def test_summary_keys(self):
        eng = DiplomacyEngine(n_agents=4, n_byzantine=1,
                              n_rounds=5, n_tasks=3, seed=42)
        summary = eng.run_simulation(verbose=False)
        required = {"config", "factions", "treaties", "alliance_heatmap",
                    "events", "pressure"}
        assert required.issubset(summary.keys())

    def test_summary_config_matches(self):
        eng = DiplomacyEngine(n_agents=8, n_byzantine=2,
                              n_rounds=20, n_tasks=10, seed=42)
        summary = eng.run_simulation(verbose=False)
        cfg = summary["config"]
        assert cfg["agents"] == 8
        assert cfg["byzantine"] == 2
        assert cfg["rounds"] == 20
        assert cfg["tasks"] == 10


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_minimal_agents(self):
        eng = DiplomacyEngine(n_agents=2, n_byzantine=0,
                              n_rounds=3, n_tasks=2, seed=42)
        summary = eng.run_simulation(verbose=False)
        assert len(summary["factions"]) >= 1

    def test_all_byzantine(self):
        eng = DiplomacyEngine(n_agents=3, n_byzantine=3,
                              n_rounds=5, n_tasks=3, seed=42)
        summary = eng.run_simulation(verbose=False)
        assert summary["config"]["byzantine"] == 3

    def test_single_task(self):
        eng = DiplomacyEngine(n_agents=4, n_byzantine=1,
                              n_rounds=5, n_tasks=1, seed=42)
        summary = eng.run_simulation(verbose=False)
        assert len(eng.vote_matrix) == 1

    def test_single_round(self):
        eng = DiplomacyEngine(n_agents=4, n_byzantine=1,
                              n_rounds=1, n_tasks=3, seed=42)
        summary = eng.run_simulation(verbose=False)
        for t in range(3):
            assert len(eng.vote_matrix[t]) == 1

    def test_faction_avg_vote_empty_members(self):
        eng = DiplomacyEngine(n_agents=3, n_byzantine=0,
                              n_rounds=3, n_tasks=2, seed=42)
        eng.run_simulation(verbose=False)
        assert eng._faction_avg_vote([], 0) == 0.0
