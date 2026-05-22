"""Tests for the Swarm Speciation Engine."""
from __future__ import annotations

import asyncio
import json
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.speciation import (
    EcosystemSummary,
    FitnessProfile,
    RoutingRecommendation,
    SpeciationEngine,
    SpeciationEvent,
    SpeciesModel,
    TaskRecord,
    _centroid,
    _elbow_k,
    _euclidean,
    _kmeans,
    run_simulation,
)
from src.core.state import RoundResult, Vote


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_result(
    leader_id: str = "a1",
    committed: bool = True,
    solution: str = "yes",
    votes: list | None = None,
    slashed: list | None = None,
    round_index: int = 0,
) -> RoundResult:
    """Build a RoundResult for testing."""
    return RoundResult(
        round_index=round_index,
        leader_id=leader_id,
        committed_solution=solution if committed else None,
        aggregate_weight=3.0,
        threshold=2.0,
        votes=votes or [
            Vote(voter_id="a1", target_proposal_id="p1", weight=0.9),
            Vote(voter_id="a2", target_proposal_id="p1", weight=0.8),
            Vote(voter_id="a3", target_proposal_id="p1", weight=0.7),
        ],
        slashed=slashed or [],
    )


async def _run(coro):
    return await coro


# ── K-Means Tests ────────────────────────────────────────────────────────

class TestKMeansInternals:
    def test_euclidean_zero(self):
        assert _euclidean([0, 0], [0, 0]) == 0.0

    def test_euclidean_basic(self):
        assert abs(_euclidean([0, 0], [3, 4]) - 5.0) < 1e-9

    def test_centroid_single(self):
        assert _centroid([[1.0, 2.0]]) == [1.0, 2.0]

    def test_centroid_multiple(self):
        c = _centroid([[0.0, 0.0], [2.0, 4.0]])
        assert abs(c[0] - 1.0) < 1e-9
        assert abs(c[1] - 2.0) < 1e-9

    def test_centroid_empty(self):
        assert _centroid([]) == []

    def test_kmeans_two_clusters(self):
        import random
        vectors = [[0.0, 0.0], [0.1, 0.1], [0.0, 0.1],
                   [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]]
        labels, centroids, wcss = _kmeans(vectors, 2, rng=random.Random(42))
        # two groups: should cluster low values together and high values together
        low_labels = {labels[0], labels[1], labels[2]}
        high_labels = {labels[3], labels[4], labels[5]}
        assert len(low_labels) == 1
        assert len(high_labels) == 1
        assert low_labels != high_labels

    def test_kmeans_single_point(self):
        labels, centroids, wcss = _kmeans([[1.0, 2.0]], 1)
        assert labels == [0]
        assert wcss == 0.0

    def test_kmeans_empty(self):
        labels, centroids, wcss = _kmeans([], 2)
        assert labels == []

    def test_kmeans_k_exceeds_n(self):
        """k > n should be clamped."""
        vectors = [[1.0], [2.0]]
        labels, centroids, wcss = _kmeans(vectors, 10)
        assert len(labels) == 2

    def test_elbow_k_small_data(self):
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        k = _elbow_k(vectors, max_k=5)
        assert 1 <= k <= 2

    def test_elbow_k_clear_clusters(self):
        import random
        rng = random.Random(99)
        vectors = [[rng.gauss(0, 0.1), rng.gauss(0, 0.1)] for _ in range(10)]
        vectors += [[rng.gauss(5, 0.1), rng.gauss(5, 0.1)] for _ in range(10)]
        vectors += [[rng.gauss(10, 0.1), rng.gauss(10, 0.1)] for _ in range(10)]
        k = _elbow_k(vectors, max_k=8, rng=rng)
        assert 2 <= k <= 5  # should find ~3 clusters


# ── Data Model Tests ─────────────────────────────────────────────────────

class TestDataModels:
    def test_task_record_defaults(self):
        rec = TaskRecord(task_type="analytical", round_index=0)
        assert rec.task_type == "analytical"
        assert rec.committed is False
        assert rec.task_id  # auto-generated

    def test_fitness_profile_vector(self):
        fp = FitnessProfile(
            agent_id="a1",
            task_scores={"analytical": 0.8, "creative": 0.3},
        )
        v = fp.vector(["analytical", "creative", "rapid"])
        assert v == [0.8, 0.3, 0.0]

    def test_species_model_defaults(self):
        sp = SpeciesModel(name="Test")
        assert sp.population == 0
        assert sp.species_id

    def test_speciation_event_defaults(self):
        ev = SpeciationEvent(event_type="formation")
        assert ev.event_type == "formation"
        assert ev.event_id

    def test_routing_recommendation_defaults(self):
        rec = RoutingRecommendation(
            task_type="analytical",
            recommended_species_id="s1",
            recommended_species_name="Alpha",
        )
        assert rec.confidence == 0.0

    def test_ecosystem_summary_defaults(self):
        eco = EcosystemSummary()
        assert eco.num_species == 0
        assert eco.stability_score == 0.0


# ── Engine Core Tests ────────────────────────────────────────────────────

class TestSpeciationEngine:
    def test_init(self):
        eng = SpeciationEngine(seed=1)
        assert eng.task_records == []
        assert eng.species == []

    def test_record_round(self):
        eng = SpeciationEngine()
        result = _make_result()
        rec = eng.record_round("analytical", result)
        assert rec.task_type == "analytical"
        assert rec.committed is True
        assert len(eng.task_records) == 1
        assert "analytical" in eng.active_task_types

    def test_record_multiple_types(self):
        eng = SpeciationEngine()
        eng.record_round("analytical", _make_result())
        eng.record_round("creative", _make_result())
        assert set(eng.active_task_types) == {"analytical", "creative"}

    def test_update_fitness_profiles(self):
        eng = SpeciationEngine()
        for i in range(5):
            eng.record_round("analytical", _make_result(round_index=i))
        profiles = eng.update_fitness_profiles()
        assert "a1" in profiles
        assert "a2" in profiles
        assert "a3" in profiles
        assert profiles["a1"].overall > 0

    def test_fitness_leadership_rate(self):
        eng = SpeciationEngine()
        for i in range(5):
            eng.record_round("analytical", _make_result(leader_id="a1", committed=True, round_index=i))
        profiles = eng.update_fitness_profiles()
        assert profiles["a1"].leadership_rate["analytical"] == 1.0

    def test_fitness_slashing(self):
        eng = SpeciationEngine()
        eng.record_round("analytical", _make_result(slashed=["a3"]))
        profiles = eng.update_fitness_profiles()
        assert profiles["a3"].calibration["analytical"] < 1.0

    def test_detect_species_single_agent(self):
        eng = SpeciationEngine(seed=1)
        votes = [Vote(voter_id="solo", target_proposal_id="p1", weight=0.9)]
        eng.record_round("analytical", _make_result(leader_id="solo", votes=votes))
        eng.update_fitness_profiles()
        species = eng.detect_species()
        assert len(species) == 1
        assert "solo" in species[0].member_ids

    def test_detect_species_multiple(self):
        eng = SpeciationEngine(seed=42)
        # Create distinct groups via different task types
        # Group 1: a1, a2 strong in analytical
        for i in range(10):
            v1 = [Vote(voter_id="a1", target_proposal_id="p1", weight=0.9),
                   Vote(voter_id="a2", target_proposal_id="p1", weight=0.85)]
            eng.record_round("analytical", RoundResult(
                round_index=i, leader_id="a1", committed_solution="yes",
                aggregate_weight=3, threshold=2, votes=v1))
        # Group 2: a3, a4 strong in creative
        for i in range(10):
            v2 = [Vote(voter_id="a3", target_proposal_id="p2", weight=0.9),
                   Vote(voter_id="a4", target_proposal_id="p2", weight=0.85)]
            eng.record_round("creative", RoundResult(
                round_index=10 + i, leader_id="a3", committed_solution="yes",
                aggregate_weight=3, threshold=2, votes=v2))

        eng.update_fitness_profiles()
        species = eng.detect_species()
        assert len(species) >= 2

    def test_detect_species_forced_k(self):
        eng = SpeciationEngine(seed=42)
        for i in range(5):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        species = eng.detect_species(k=2)
        assert len(species) == 2

    def test_species_health_metrics(self):
        eng = SpeciationEngine(seed=42)
        for i in range(5):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        health = eng.get_species_health()
        assert len(health) >= 1
        assert "population" in health[0]
        assert "avg_fitness" in health[0]


# ── Speciation Event Tests ───────────────────────────────────────────────

class TestSpeciationEvents:
    def test_formation_event(self):
        eng = SpeciationEngine(seed=42)
        for i in range(5):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        # First detection always creates formation events
        events = eng.detect_speciation_events()
        formations = [e for e in events if e.event_type == "formation"]
        assert len(formations) >= 1

    def test_extinction_event(self):
        eng = SpeciationEngine(seed=42)
        # Phase 1: detect species with distinct groups
        for i in range(10):
            v1 = [Vote(voter_id="a1", target_proposal_id="p1", weight=0.9),
                  Vote(voter_id="a2", target_proposal_id="p1", weight=0.85)]
            eng.record_round("analytical", RoundResult(
                round_index=i, leader_id="a1", committed_solution="yes",
                aggregate_weight=3, threshold=2, votes=v1))
        for i in range(10):
            v2 = [Vote(voter_id="a3", target_proposal_id="p2", weight=0.9),
                  Vote(voter_id="a4", target_proposal_id="p2", weight=0.85)]
            eng.record_round("creative", RoundResult(
                round_index=10+i, leader_id="a3", committed_solution="yes",
                aggregate_weight=3, threshold=2, votes=v2))
        for i in range(10):
            v3 = [Vote(voter_id="a5", target_proposal_id="p3", weight=0.9),
                  Vote(voter_id="a6", target_proposal_id="p3", weight=0.85)]
            eng.record_round("rapid", RoundResult(
                round_index=20+i, leader_id="a5", committed_solution="yes",
                aggregate_weight=3, threshold=2, votes=v3))
        eng.update_fitness_profiles()
        eng.detect_species(k=3)
        eng.detect_speciation_events()
        {sp.name for sp in eng.species}

        # Phase 2: radically different data - only a5/a6 active, others gone
        for i in range(30, 60):
            v = [Vote(voter_id="a5", target_proposal_id="p3", weight=0.95),
                 Vote(voter_id="a6", target_proposal_id="p3", weight=0.9)]
            eng.record_round("rapid", RoundResult(
                round_index=i, leader_id="a5", committed_solution="yes",
                aggregate_weight=3, threshold=2, votes=v))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        events = eng.detect_speciation_events()
        # Should detect formation of new species names or extinction of old ones
        {e.event_type for e in events}
        assert len(events) > 0

    def test_adaptation_or_change_event(self):
        eng = SpeciationEngine(seed=42)
        for i in range(10):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        first_events = eng.detect_speciation_events()
        # First detection always triggers formations
        assert any(e.event_type == "formation" for e in first_events)

        # Phase 2: dramatically shift fitness - add many rounds of different task
        for i in range(10, 50):
            v = [Vote(voter_id="a1", target_proposal_id="p1", weight=-0.9),
                 Vote(voter_id="a2", target_proposal_id="p1", weight=-0.5),
                 Vote(voter_id="a3", target_proposal_id="p1", weight=0.95)]
            eng.record_round("rapid", RoundResult(
                round_index=i, leader_id="a3", committed_solution="yes",
                aggregate_weight=3, threshold=2, votes=v, slashed=["a1", "a2"]))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        events = eng.detect_speciation_events()
        # Should detect adaptation (centroid drift) or formation/extinction of species
        all_events = first_events + events
        assert len(all_events) >= 2  # at least the initial formations + some change


# ── Routing Tests ────────────────────────────────────────────────────────

class TestRouting:
    def test_routing_no_species(self):
        eng = SpeciationEngine()
        rec = eng.get_routing_recommendation("analytical")
        assert rec.confidence == 0.0
        assert rec.recommended_species_name == "none"

    def test_routing_with_species(self):
        eng = SpeciationEngine(seed=42)
        for i in range(10):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        rec = eng.get_routing_recommendation("analytical")
        assert rec.recommended_species_name != "none"
        assert rec.confidence > 0
        assert len(rec.recommended_agents) > 0

    def test_routing_unknown_task_type(self):
        eng = SpeciationEngine(seed=42)
        for i in range(5):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        rec = eng.get_routing_recommendation("unknown_type")
        # Should still return a recommendation based on overall fitness
        assert rec.recommended_species_name != "none"


# ── Ecosystem Summary Tests ──────────────────────────────────────────────

class TestEcosystemSummary:
    def test_empty_ecosystem(self):
        eng = SpeciationEngine()
        eco = eng.get_ecosystem_summary()
        assert eco.num_species == 0
        assert eco.num_agents == 0
        assert eco.diversity_index == 0.0

    def test_populated_ecosystem(self):
        eng = SpeciationEngine(seed=42)
        for i in range(10):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        eco = eng.get_ecosystem_summary()
        assert eco.num_species == 2
        assert eco.num_agents == 3
        assert eco.total_rounds == 10

    def test_diversity_index(self):
        eng = SpeciationEngine(seed=42)
        for i in range(10):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        eco = eng.get_ecosystem_summary()
        assert eco.diversity_index >= 0.0

    def test_stability_score(self):
        eng = SpeciationEngine(seed=42)
        eco = eng.get_ecosystem_summary()
        # no events = max stability
        assert eco.stability_score == 1.0


# ── Export Tests ─────────────────────────────────────────────────────────

class TestExport:
    def test_json_export(self):
        eng = SpeciationEngine(seed=42)
        for i in range(5):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        data = eng.export_json()
        assert "fitness_profiles" in data
        assert "species" in data
        assert "events" in data
        assert "ecosystem" in data
        assert "task_types" in data
        # ensure it's JSON serializable
        serialized = json.dumps(data, default=str)
        assert len(serialized) > 0

    def test_json_roundtrip(self):
        eng = SpeciationEngine(seed=42)
        for i in range(5):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        data = eng.export_json()
        text = json.dumps(data, default=str)
        loaded = json.loads(text)
        assert loaded["task_types"] == data["task_types"]
        assert loaded["total_rounds"] == data["total_rounds"]


# ── HTML Report Tests ────────────────────────────────────────────────────

class TestHTMLReport:
    def test_html_contains_title(self):
        eng = SpeciationEngine(seed=42)
        html_str = eng.render_html()
        assert "Swarm Speciation Dashboard" in html_str

    def test_html_contains_species_map(self):
        eng = SpeciationEngine(seed=42)
        for i in range(5):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        eng.detect_species(k=2)
        html_str = eng.render_html()
        assert "Species Map" in html_str

    def test_html_contains_heatmap(self):
        eng = SpeciationEngine(seed=42)
        for i in range(5):
            eng.record_round("analytical", _make_result(round_index=i))
        eng.update_fitness_profiles()
        html_str = eng.render_html()
        assert "Fitness Heatmap" in html_str

    def test_html_contains_routing(self):
        eng = SpeciationEngine(seed=42)
        html_str = eng.render_html()
        assert "Task Routing" in html_str

    def test_html_contains_ecosystem(self):
        eng = SpeciationEngine(seed=42)
        html_str = eng.render_html()
        assert "Ecosystem Summary" in html_str


# ── Simulation Tests ─────────────────────────────────────────────────────

class TestSimulation:
    def test_simulation_runs(self):
        engine = asyncio.run(run_simulation(n_agents=6, n_rounds=20, n_task_types=3, seed=42))
        assert engine._round_counter > 0
        assert len(engine.species) > 0

    def test_simulation_byzantine(self):
        engine = asyncio.run(run_simulation(n_agents=8, n_byzantine=3, n_rounds=30, seed=99))
        eco = engine.get_ecosystem_summary()
        assert eco.num_agents > 0
        assert eco.num_species > 0

    def test_simulation_deterministic(self):
        e1 = asyncio.run(run_simulation(n_agents=6, n_rounds=20, seed=42))
        e2 = asyncio.run(run_simulation(n_agents=6, n_rounds=20, seed=42))
        assert e1.get_ecosystem_summary().num_species == e2.get_ecosystem_summary().num_species
        assert len(e1.events) == len(e2.events)

    def test_simulation_html(self):
        engine = asyncio.run(run_simulation(n_agents=6, n_rounds=15, seed=42))
        html_str = engine.render_html()
        assert "🧬" in html_str
        assert len(html_str) > 500


# ── Edge Cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_rounds(self):
        eng = SpeciationEngine()
        profiles = eng.update_fitness_profiles()
        assert profiles == {}
        species = eng.detect_species()
        assert species == []

    def test_all_identical_agents(self):
        eng = SpeciationEngine(seed=42)
        votes = [
            Vote(voter_id=f"a{i}", target_proposal_id="p1", weight=0.5)
            for i in range(5)
        ]
        for r in range(10):
            eng.record_round("analytical", RoundResult(
                round_index=r, leader_id="a0", committed_solution="yes",
                aggregate_weight=3, threshold=2, votes=votes))
        eng.update_fitness_profiles()
        species = eng.detect_species()
        assert len(species) >= 1

    def test_single_round(self):
        eng = SpeciationEngine(seed=42)
        eng.record_round("analytical", _make_result())
        eng.update_fitness_profiles()
        species = eng.detect_species()
        assert len(species) >= 1

    def test_all_slashed(self):
        eng = SpeciationEngine()
        eng.record_round("analytical", _make_result(slashed=["a1", "a2", "a3"]))
        profiles = eng.update_fitness_profiles()
        for aid in ["a1", "a2", "a3"]:
            assert profiles[aid].calibration["analytical"] < 1.0
