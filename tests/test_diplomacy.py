"""Tests for the Consensus Diplomacy Engine (src.diplomacy).

Covers: _cosine_sim helper, DiplomacyEngine construction, simulation phases,
faction detection, alliance computation, treaty detection, pressure measurement,
event generation, auto-negotiation, summary building, and HTML report generation.
"""
from __future__ import annotations

import math
import pytest
from unittest.mock import patch

from src.diplomacy import (
    _cosine_sim,
    DiplomacyEngine,
    DiplomaticEvent,
    Faction,
    Treaty,
    generate_html_report,
    _alliance_color,
    _esc,
)


# ---------------------------------------------------------------------------
# _cosine_sim
# ---------------------------------------------------------------------------

class TestCosineSim:
    def test_identical_vectors(self):
        assert _cosine_sim([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        assert _cosine_sim([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert _cosine_sim([], []) == 0.0

    def test_mismatched_length(self):
        assert _cosine_sim([1, 2], [1]) == 0.0

    def test_zero_vector(self):
        assert _cosine_sim([0, 0, 0], [1, 2, 3]) == 0.0

    def test_near_zero_magnitude(self):
        assert _cosine_sim([1e-12, 0], [1, 1]) == 0.0

    def test_negative_vectors(self):
        sim = _cosine_sim([-1, -2, -3], [-2, -4, -6])
        assert sim == pytest.approx(1.0)

    def test_known_angle(self):
        sim = _cosine_sim([1, 0], [1, 1])
        assert sim == pytest.approx(1 / math.sqrt(2), abs=1e-6)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class TestTreaty:
    def test_active_when_not_broken(self):
        t = Treaty(faction_a=0, faction_b=1, formed_round=0, strength=0.8)
        assert t.active is True

    def test_inactive_when_broken(self):
        t = Treaty(faction_a=0, faction_b=1, formed_round=0, broken_round=10, strength=0.5)
        assert t.active is False

    def test_defaults(self):
        t = Treaty(faction_a=2, faction_b=3, formed_round=5)
        assert t.broken_round is None
        assert t.strength == 0.0
        assert t.active is True


class TestDiplomaticEvent:
    def test_construction(self):
        e = DiplomaticEvent(round_idx=3, event_type="alliance_formed",
                            agents=["a", "b"], detail="test")
        assert e.round_idx == 3
        assert e.event_type == "alliance_formed"
        assert e.agents == ["a", "b"]


class TestFaction:
    def test_defaults(self):
        f = Faction(faction_id=0, members=["agent_0"])
        assert f.avg_vote == 0.0
        assert f.power == 0.0
        assert f.color == "#888"


# ---------------------------------------------------------------------------
# DiplomacyEngine construction
# ---------------------------------------------------------------------------

class TestEngineConstruction:
    def test_default_construction(self):
        engine = DiplomacyEngine(seed=42)
        assert engine.n_agents == 8
        assert engine.n_byzantine == 2
        assert engine.n_rounds == 40
        assert engine.n_tasks == 15
        assert len(engine.agent_names) == 8
        assert len(engine.byzantine_set) == 2
        assert all(b in engine.agent_names for b in engine.byzantine_set)

    def test_custom_construction(self):
        engine = DiplomacyEngine(n_agents=4, n_byzantine=1, n_rounds=10,
                                 n_tasks=5, auto_negotiate=True, seed=99)
        assert engine.n_agents == 4
        assert engine.n_byzantine == 1
        assert engine.auto_negotiate is True
        assert len(engine.agent_names) == 4

    def test_byzantine_capped_at_agent_count(self):
        engine = DiplomacyEngine(n_agents=3, n_byzantine=10, seed=1)
        assert len(engine.byzantine_set) == 3

    def test_zero_byzantine(self):
        engine = DiplomacyEngine(n_agents=5, n_byzantine=0, seed=7)
        assert len(engine.byzantine_set) == 0

    def test_seed_determinism(self):
        e1 = DiplomacyEngine(n_agents=6, n_byzantine=2, seed=123)
        e2 = DiplomacyEngine(n_agents=6, n_byzantine=2, seed=123)
        assert e1.byzantine_set == e2.byzantine_set

    def test_initial_state_empty(self):
        engine = DiplomacyEngine(seed=1)
        assert engine.vote_matrix == []
        assert engine.factions == []
        assert engine.treaties == []
        assert engine.events == []
        assert engine.alliance_matrix == {}
        assert engine.pressure_scores == {}
        assert engine.recommendations == []


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_simulate_round_returns_all_agents(self):
        engine = DiplomacyEngine(n_agents=5, n_byzantine=1, seed=42)
        votes = engine._simulate_round(0, 0)
        assert set(votes.keys()) == set(engine.agent_names)

    def test_votes_bounded(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=3, seed=42)
        for _ in range(50):
            votes = engine._simulate_round(0, 0)
            for v in votes.values():
                assert -1.0 <= v <= 1.0

    def test_run_simulation_populates_state(self):
        engine = DiplomacyEngine(n_agents=4, n_byzantine=1,
                                 n_rounds=5, n_tasks=3, seed=42)
        summary = engine.run_simulation(verbose=False)
        assert len(engine.vote_matrix) == 3
        assert all(len(rounds) == 5 for rounds in engine.vote_matrix)
        assert len(engine.factions) > 0
        assert len(engine.agent_vectors) == 4
        for vec in engine.agent_vectors.values():
            assert len(vec) == 15

    def test_run_simulation_summary_keys(self):
        engine = DiplomacyEngine(n_agents=4, n_byzantine=1,
                                 n_rounds=5, n_tasks=3, seed=42)
        summary = engine.run_simulation(verbose=False)
        expected_keys = {"config", "factions", "treaties", "alliance_heatmap",
                         "events", "pressure", "recommendations", "byzantine_agents"}
        assert set(summary.keys()) == expected_keys

    def test_config_in_summary(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=2,
                                 n_rounds=10, n_tasks=8, seed=1)
        summary = engine.run_simulation(verbose=False)
        cfg = summary["config"]
        assert cfg["agents"] == 6
        assert cfg["byzantine"] == 2
        assert cfg["rounds"] == 10
        assert cfg["tasks"] == 8

    def test_deterministic_with_seed(self):
        def run(seed):
            e = DiplomacyEngine(n_agents=5, n_byzantine=1,
                                n_rounds=8, n_tasks=4, seed=seed)
            return e.run_simulation(verbose=False)
        s1 = run(77)
        s2 = run(77)
        assert s1["factions"] == s2["factions"]
        assert s1["pressure"] == s2["pressure"]
        assert s1["byzantine_agents"] == s2["byzantine_agents"]


# ---------------------------------------------------------------------------
# Faction detection
# ---------------------------------------------------------------------------

class TestFactionDetection:
    def test_factions_cover_all_agents(self):
        engine = DiplomacyEngine(n_agents=8, n_byzantine=2,
                                 n_rounds=10, n_tasks=5, seed=42)
        engine.run_simulation(verbose=False)
        all_members = []
        for f in engine.factions:
            all_members.extend(f.members)
        assert sorted(all_members) == sorted(engine.agent_names)

    def test_no_duplicate_membership(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=1,
                                 n_rounds=10, n_tasks=5, seed=42)
        engine.run_simulation(verbose=False)
        all_members = []
        for f in engine.factions:
            all_members.extend(f.members)
        assert len(all_members) == len(set(all_members))

    def test_faction_ids_sequential(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=1,
                                 n_rounds=10, n_tasks=5, seed=42)
        engine.run_simulation(verbose=False)
        ids = [f.faction_id for f in engine.factions]
        assert ids == list(range(len(ids)))

    def test_faction_power_positive(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=1,
                                 n_rounds=10, n_tasks=5, seed=42)
        engine.run_simulation(verbose=False)
        for f in engine.factions:
            assert f.power > 0


# ---------------------------------------------------------------------------
# Alliance computation
# ---------------------------------------------------------------------------

class TestAlliances:
    def test_alliance_symmetric(self):
        engine = DiplomacyEngine(n_agents=4, n_byzantine=1,
                                 n_rounds=10, n_tasks=5, seed=42)
        engine.run_simulation(verbose=False)
        for (a, b), score in engine.alliance_matrix.items():
            assert engine.alliance_matrix.get((b, a)) == score

    def test_alliance_bounded(self):
        engine = DiplomacyEngine(n_agents=5, n_byzantine=1,
                                 n_rounds=10, n_tasks=5, seed=42)
        engine.run_simulation(verbose=False)
        for score in engine.alliance_matrix.values():
            assert -1.0 <= score <= 1.0

    def test_alliance_heatmap_in_summary(self):
        engine = DiplomacyEngine(n_agents=4, n_byzantine=1,
                                 n_rounds=5, n_tasks=3, seed=42)
        summary = engine.run_simulation(verbose=False)
        hm = summary["alliance_heatmap"]
        assert "agents" in hm
        assert "scores" in hm
        assert len(hm["agents"]) == 4


# ---------------------------------------------------------------------------
# Treaty detection
# ---------------------------------------------------------------------------

class TestTreaties:
    def test_treaty_strength_bounded(self):
        engine = DiplomacyEngine(n_agents=8, n_byzantine=2,
                                 n_rounds=15, n_tasks=10, seed=42)
        engine.run_simulation(verbose=False)
        for t in engine.treaties:
            assert 0.0 <= t.strength <= 1.0

    def test_treaty_faction_ids_valid(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=1,
                                 n_rounds=10, n_tasks=8, seed=42)
        engine.run_simulation(verbose=False)
        faction_ids = {f.faction_id for f in engine.factions}
        for t in engine.treaties:
            assert t.faction_a in faction_ids
            assert t.faction_b in faction_ids
            assert t.faction_a != t.faction_b

    def test_no_treaties_with_single_faction(self):
        engine = DiplomacyEngine(n_agents=3, n_byzantine=0,
                                 n_rounds=5, n_tasks=3, seed=42)
        engine.run_simulation(verbose=False)
        if len(engine.factions) <= 1:
            assert engine.treaties == []

    def test_treaty_summary_format(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=2,
                                 n_rounds=10, n_tasks=8, seed=42)
        summary = engine.run_simulation(verbose=False)
        for t in summary["treaties"]:
            assert "faction_a" in t
            assert "faction_b" in t
            assert "strength" in t
            assert "active" in t
            assert isinstance(t["active"], bool)


# ---------------------------------------------------------------------------
# Pressure measurement
# ---------------------------------------------------------------------------

class TestPressure:
    def test_pressure_for_all_agents(self):
        engine = DiplomacyEngine(n_agents=5, n_byzantine=1,
                                 n_rounds=10, n_tasks=5, seed=42)
        engine.run_simulation(verbose=False)
        assert set(engine.pressure_scores.keys()) == set(engine.agent_names)

    def test_pressure_non_negative(self):
        engine = DiplomacyEngine(n_agents=5, n_byzantine=1,
                                 n_rounds=10, n_tasks=5, seed=42)
        engine.run_simulation(verbose=False)
        for score in engine.pressure_scores.values():
            assert score >= 0.0

    def test_pressure_in_summary(self):
        engine = DiplomacyEngine(n_agents=4, n_byzantine=1,
                                 n_rounds=5, n_tasks=3, seed=42)
        summary = engine.run_simulation(verbose=False)
        assert "pressure" in summary
        assert len(summary["pressure"]) == 4


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------

class TestEvents:
    def test_events_sorted_by_round(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=2,
                                 n_rounds=15, n_tasks=8, seed=42)
        engine.run_simulation(verbose=False)
        rounds = [e.round_idx for e in engine.events]
        assert rounds == sorted(rounds)

    def test_faction_formation_events(self):
        engine = DiplomacyEngine(n_agents=5, n_byzantine=1,
                                 n_rounds=10, n_tasks=5, seed=42)
        engine.run_simulation(verbose=False)
        alliance_events = [e for e in engine.events if e.event_type == "alliance_formed"]
        assert len(alliance_events) >= len(engine.factions)

    def test_event_types_valid(self):
        valid_types = {"alliance_formed", "treaty_broken", "faction_merged",
                       "capitulation", "diplomatic_isolation"}
        engine = DiplomacyEngine(n_agents=8, n_byzantine=3,
                                 n_rounds=15, n_tasks=10, seed=42)
        engine.run_simulation(verbose=False)
        for e in engine.events:
            assert e.event_type in valid_types

    def test_events_in_summary(self):
        engine = DiplomacyEngine(n_agents=4, n_byzantine=1,
                                 n_rounds=5, n_tasks=3, seed=42)
        summary = engine.run_simulation(verbose=False)
        for e in summary["events"]:
            assert "round" in e
            assert "type" in e
            assert "detail" in e


# ---------------------------------------------------------------------------
# Auto-negotiation
# ---------------------------------------------------------------------------

class TestAutoNegotiate:
    def test_recommendations_populated(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=2,
                                 n_rounds=10, n_tasks=5,
                                 auto_negotiate=True, seed=42)
        engine.run_simulation(verbose=False)
        assert len(engine.recommendations) > 0

    def test_no_recommendations_without_flag(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=2,
                                 n_rounds=10, n_tasks=5,
                                 auto_negotiate=False, seed=42)
        engine.run_simulation(verbose=False)
        assert engine.recommendations == []

    def test_recommendations_mention_byzantine(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=2,
                                 n_rounds=10, n_tasks=5,
                                 auto_negotiate=True, seed=42)
        engine.run_simulation(verbose=False)
        text = "\n".join(engine.recommendations)
        assert "Byzantine" in text

    def test_coalition_health_assessment(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=2,
                                 n_rounds=10, n_tasks=5,
                                 auto_negotiate=True, seed=42)
        engine.run_simulation(verbose=False)
        text = "\n".join(engine.recommendations)
        assert any(k in text for k in ["STRONG", "MODERATE", "WEAK"])

    def test_too_few_honest_agents(self):
        engine = DiplomacyEngine(n_agents=2, n_byzantine=2,
                                 n_rounds=5, n_tasks=3,
                                 auto_negotiate=True, seed=42)
        engine.run_simulation(verbose=False)
        if len(engine.byzantine_set) == 2:
            text = "\n".join(engine.recommendations)
            assert "Too few" in text or "Strongest" in text


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

class TestHTMLReport:
    def test_generates_valid_html(self):
        engine = DiplomacyEngine(n_agents=4, n_byzantine=1,
                                 n_rounds=5, n_tasks=3, seed=42)
        summary = engine.run_simulation(verbose=False)
        html = generate_html_report(summary)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "Consensus Diplomacy Engine" in html

    def test_html_contains_faction_data(self):
        engine = DiplomacyEngine(n_agents=4, n_byzantine=1,
                                 n_rounds=5, n_tasks=3, seed=42)
        summary = engine.run_simulation(verbose=False)
        html = generate_html_report(summary)
        assert "Faction" in html
        assert "Alliance Heatmap" in html
        assert "Treaties" in html
        assert "Diplomatic Pressure" in html

    def test_html_with_auto_negotiate(self):
        engine = DiplomacyEngine(n_agents=4, n_byzantine=1,
                                 n_rounds=5, n_tasks=3,
                                 auto_negotiate=True, seed=42)
        summary = engine.run_simulation(verbose=False)
        html = generate_html_report(summary)
        assert "Autonomous Diplomat" in html

    def test_html_escapes_special_chars(self):
        assert _esc("<script>") == "&lt;script&gt;"
        assert _esc('"test"') == "&quot;test&quot;"


# ---------------------------------------------------------------------------
# _alliance_color
# ---------------------------------------------------------------------------

class TestAllianceColor:
    def test_strong_positive(self):
        assert _alliance_color(0.8) == "#3fb950"

    def test_moderate_positive(self):
        assert _alliance_color(0.4) == "#56d364"

    def test_weak_positive(self):
        assert _alliance_color(0.1) == "#2ea04380"

    def test_weak_negative(self):
        assert _alliance_color(-0.1) == "#f8514930"

    def test_moderate_negative(self):
        assert _alliance_color(-0.4) == "#f85149"

    def test_strong_negative(self):
        assert _alliance_color(-0.8) == "#da3633"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_agent(self):
        engine = DiplomacyEngine(n_agents=1, n_byzantine=0,
                                 n_rounds=5, n_tasks=3, seed=42)
        summary = engine.run_simulation(verbose=False)
        assert len(engine.factions) == 1
        assert engine.factions[0].members == ["agent_0"]

    def test_all_byzantine(self):
        engine = DiplomacyEngine(n_agents=4, n_byzantine=4,
                                 n_rounds=5, n_tasks=3, seed=42)
        summary = engine.run_simulation(verbose=False)
        assert len(engine.byzantine_set) == 4
        assert set(summary["byzantine_agents"]) == set(engine.agent_names)

    def test_minimal_config(self):
        engine = DiplomacyEngine(n_agents=2, n_byzantine=0,
                                 n_rounds=1, n_tasks=1, seed=42)
        summary = engine.run_simulation(verbose=False)
        assert summary["config"]["agents"] == 2

    def test_large_simulation(self):
        engine = DiplomacyEngine(n_agents=12, n_byzantine=4,
                                 n_rounds=20, n_tasks=10, seed=42)
        summary = engine.run_simulation(verbose=False)
        assert len(summary["factions"]) > 0
        assert len(summary["pressure"]) == 12

    def test_byzantine_agents_in_summary(self):
        engine = DiplomacyEngine(n_agents=6, n_byzantine=2, seed=42)
        summary = engine.run_simulation(verbose=False)
        assert set(summary["byzantine_agents"]) == engine.byzantine_set
