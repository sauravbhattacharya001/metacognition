"""Tests for Swarm Mitosis Engine."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.mitosis import (
    AgentCell,
    ApoptosisEvent,
    CellCyclePhase,
    DivisionEvent,
    GrowthFactor,
    MitosisReport,
    MitosisSnapshot,
    SwarmMitosisEngine,
    SCENARIOS,
    main,
)


# ── Initialization ──────────────────────────────────────────────────────


class TestInitialization:
    def test_default_init(self):
        e = SwarmMitosisEngine()
        assert e.population == 10
        assert e.carrying_capacity == 50

    def test_custom_init(self):
        e = SwarmMitosisEngine(num_agents=5, carrying_capacity=20, max_divisions=7)
        assert e.population == 5
        assert e.carrying_capacity == 20

    def test_agents_are_g0(self):
        e = SwarmMitosisEngine(num_agents=3)
        for cell in e.agents.values():
            assert cell.phase == CellCyclePhase.G0_QUIESCENT

    def test_agents_have_traits(self):
        e = SwarmMitosisEngine(num_agents=2)
        for cell in e.agents.values():
            assert "growth_rate" in cell.traits
            assert "resilience" in cell.traits

    def test_zero_agents(self):
        e = SwarmMitosisEngine(num_agents=0)
        assert e.population == 0

    def test_tick_count_starts_zero(self):
        e = SwarmMitosisEngine()
        assert e.tick_count == 0

    def test_all_generation_zero(self):
        e = SwarmMitosisEngine(num_agents=5)
        for cell in e.agents.values():
            assert cell.generation == 0

    def test_telomere_starts_high(self):
        e = SwarmMitosisEngine(num_agents=3)
        for cell in e.agents.values():
            assert cell.telomere_length == 100.0

    def test_division_count_starts_zero(self):
        e = SwarmMitosisEngine(num_agents=3)
        for cell in e.agents.values():
            assert cell.division_count == 0


# ── Growth Factors ──────────────────────────────────────────────────────


class TestGrowthFactors:
    def test_add_growth_factor(self):
        e = SwarmMitosisEngine(num_agents=2)
        gf = e.add_growth_factor("EGF", potency=0.8)
        assert gf.name == "EGF"
        assert gf.potency == 0.8

    def test_default_growth_factor_created_on_tick(self):
        e = SwarmMitosisEngine(num_agents=3)
        e.tick()
        assert len(e._growth_factors) > 0

    def test_growth_factor_concentration_increases(self):
        e = SwarmMitosisEngine(num_agents=5)
        e.add_growth_factor("mitogen", potency=0.7, decay_rate=0.01)
        e.tick()
        assert e._growth_factors["mitogen"].concentration > 0

    def test_growth_factor_decays(self):
        e = SwarmMitosisEngine(num_agents=2)
        gf = e.add_growth_factor("test", potency=0.5, decay_rate=0.5)
        gf.concentration = 10.0
        e._decay_growth_factors()
        assert gf.concentration == 5.0


# ── Checkpoints ─────────────────────────────────────────────────────────


class TestCheckpoints:
    def test_g1s_fails_low_nutrients(self):
        e = SwarmMitosisEngine(num_agents=1)
        cell = list(e.agents.values())[0]
        cell.nutrient_level = 10
        e._total_growth_concentration = 5.0
        assert not e._check_g1s(cell)

    def test_g1s_fails_low_dna(self):
        e = SwarmMitosisEngine(num_agents=1)
        cell = list(e.agents.values())[0]
        cell.dna_integrity = 50
        cell.nutrient_level = 60
        e._total_growth_concentration = 5.0
        assert not e._check_g1s(cell)

    def test_g1s_fails_low_growth_factors(self):
        e = SwarmMitosisEngine(num_agents=1)
        cell = list(e.agents.values())[0]
        cell.nutrient_level = 60
        cell.dna_integrity = 95
        e._total_growth_concentration = 0.1
        assert not e._check_g1s(cell)

    def test_g1s_passes(self):
        e = SwarmMitosisEngine(num_agents=1)
        cell = list(e.agents.values())[0]
        cell.nutrient_level = 60
        cell.dna_integrity = 95
        cell.division_count = 0
        e._total_growth_concentration = 5.0
        assert e._check_g1s(cell)
        assert "G1/S" in cell.checkpoints_passed

    def test_g1s_fails_at_hayflick(self):
        e = SwarmMitosisEngine(num_agents=1, max_divisions=5)
        cell = list(e.agents.values())[0]
        cell.division_count = 5
        cell.nutrient_level = 60
        cell.dna_integrity = 95
        e._total_growth_concentration = 5.0
        assert not e._check_g1s(cell)

    def test_g2m_fails_incomplete_synthesis(self):
        e = SwarmMitosisEngine(num_agents=1)
        cell = list(e.agents.values())[0]
        cell._s_phase_ticks = 0
        cell._s_phase_target = 2
        assert not e._check_g2m(cell)

    def test_g2m_passes(self):
        e = SwarmMitosisEngine(num_agents=1)
        cell = list(e.agents.values())[0]
        cell._s_phase_ticks = 3
        cell._s_phase_target = 2
        cell.dna_integrity = 95
        cell.fitness = 70
        assert e._check_g2m(cell)


# ── Division ────────────────────────────────────────────────────────────


class TestDivision:
    def test_symmetric_division(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10)
        cell = list(e.agents.values())[0]
        event = e._divide(cell, "symmetric")
        assert event is not None
        assert event.division_type == "symmetric"
        assert e.population == 2

    def test_asymmetric_division(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10)
        cell = list(e.agents.values())[0]
        event = e._divide(cell, "asymmetric")
        assert event is not None
        child = e.get_agent(event.child_id)
        assert child.specialization == "stem"

    def test_division_blocked_at_capacity(self):
        e = SwarmMitosisEngine(num_agents=5, carrying_capacity=5)
        cell = list(e.agents.values())[0]
        event = e._divide(cell, "symmetric")
        assert event is None

    def test_telomere_shortens_on_division(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10, telomere_loss_per_division=10.0)
        cell = list(e.agents.values())[0]
        orig_telo = cell.telomere_length
        e._divide(cell, "symmetric")
        assert cell.telomere_length < orig_telo

    def test_child_generation_incremented(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10)
        cell = list(e.agents.values())[0]
        event = e._divide(cell, "symmetric")
        child = e.get_agent(event.child_id)
        assert child.generation == 1

    def test_parent_division_count_incremented(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10)
        cell = list(e.agents.values())[0]
        e._divide(cell, "symmetric")
        assert cell.division_count == 1

    def test_child_inherits_traits(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10, mutation_rate=0.0)
        cell = list(e.agents.values())[0]
        event = e._divide(cell, "symmetric")
        child = e.get_agent(event.child_id)
        # With mutation_rate 0, traits should be very close
        for k in cell.traits:
            assert k in child.traits

    def test_force_division(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10)
        aid = list(e.agents.keys())[0]
        event = e.force_division(aid)
        assert event is not None
        assert e.population == 2

    def test_force_division_nonexistent(self):
        e = SwarmMitosisEngine(num_agents=1)
        event = e.force_division("nonexistent")
        assert event is None

    def test_trait_mutations_recorded(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10, mutation_rate=0.1)
        cell = list(e.agents.values())[0]
        event = e._divide(cell, "symmetric")
        assert isinstance(event.trait_mutations, dict)

    def test_parent_nutrients_halved(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10)
        cell = list(e.agents.values())[0]
        cell.nutrient_level = 80.0
        e._divide(cell, "symmetric")
        assert cell.nutrient_level == 40.0


# ── Contact Inhibition ──────────────────────────────────────────────────


class TestContactInhibition:
    def test_low_pressure(self):
        e = SwarmMitosisEngine(num_agents=5, carrying_capacity=50)
        assert e._contact_inhibition_pressure() == pytest.approx(0.1)

    def test_full_capacity(self):
        e = SwarmMitosisEngine(num_agents=50, carrying_capacity=50)
        assert e._contact_inhibition_pressure() == pytest.approx(1.0)

    def test_over_capacity(self):
        e = SwarmMitosisEngine(num_agents=60, carrying_capacity=50)
        assert e._contact_inhibition_pressure() == 1.0


# ── Apoptosis ───────────────────────────────────────────────────────────


class TestApoptosis:
    def test_telomere_exhaustion(self):
        e = SwarmMitosisEngine(num_agents=3, carrying_capacity=10)
        cell = list(e.agents.values())[0]
        cell.telomere_length = 2.0
        deaths = e._run_apoptosis()
        assert any(d.reason == "telomere_exhaustion" for d in deaths)
        assert cell.agent_id not in e.agents

    def test_dna_damage(self):
        e = SwarmMitosisEngine(num_agents=3, carrying_capacity=10)
        cell = list(e.agents.values())[0]
        cell.dna_integrity = 10.0
        deaths = e._run_apoptosis()
        assert any(d.reason == "dna_damage" for d in deaths)

    def test_low_fitness(self):
        e = SwarmMitosisEngine(num_agents=3, carrying_capacity=10)
        cell = list(e.agents.values())[0]
        cell.fitness = 5.0
        deaths = e._run_apoptosis()
        assert any(d.reason == "low_fitness" for d in deaths)

    def test_kill_agent(self):
        e = SwarmMitosisEngine(num_agents=3)
        aid = list(e.agents.keys())[0]
        ev = e.kill_agent(aid, reason="test")
        assert ev is not None
        assert ev.reason == "test"
        assert aid not in e.agents

    def test_kill_nonexistent(self):
        e = SwarmMitosisEngine(num_agents=1)
        ev = e.kill_agent("fake")
        assert ev is None


# ── Tick & Simulation ───────────────────────────────────────────────────


class TestTickAndSimulation:
    def test_tick_increments(self):
        e = SwarmMitosisEngine(num_agents=3)
        e.tick()
        assert e.tick_count == 1

    def test_tick_returns_snapshot(self):
        e = SwarmMitosisEngine(num_agents=3)
        snap = e.tick()
        assert isinstance(snap, MitosisSnapshot)
        assert snap.tick == 1

    def test_simulate_runs(self):
        e = SwarmMitosisEngine(num_agents=5, carrying_capacity=30)
        report = e.simulate(ticks=20)
        assert isinstance(report, MitosisReport)
        assert len(report.snapshots) == 20

    def test_population_changes_over_time(self):
        e = SwarmMitosisEngine(num_agents=5, carrying_capacity=40)
        report = e.simulate(ticks=50)
        pops = [s.population_size for s in report.snapshots]
        # Population should change at least once
        assert len(set(pops)) > 1

    def test_divisions_happen(self):
        e = SwarmMitosisEngine(num_agents=5, carrying_capacity=40)
        report = e.simulate(ticks=50)
        assert report.total_divisions > 0

    def test_phase_distribution_in_snapshot(self):
        e = SwarmMitosisEngine(num_agents=5)
        snap = e.tick()
        assert isinstance(snap.phase_distribution, dict)


# ── Lineage ─────────────────────────────────────────────────────────────


class TestLineage:
    def test_lineage_tree_initial(self):
        e = SwarmMitosisEngine(num_agents=3)
        tree = e.get_lineage_tree()
        assert len(tree) == 3
        for children in tree.values():
            assert children == []

    def test_lineage_after_division(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10)
        aid = list(e.agents.keys())[0]
        event = e.force_division(aid)
        tree = e.get_lineage_tree()
        assert event.child_id in tree[aid]

    def test_count_descendants(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10)
        aid = list(e.agents.keys())[0]
        e.force_division(aid)
        assert e._count_descendants(aid) == 1

    def test_get_agent(self):
        e = SwarmMitosisEngine(num_agents=2)
        aid = list(e.agents.keys())[0]
        cell = e.get_agent(aid)
        assert cell is not None
        assert cell.agent_id == aid

    def test_get_nonexistent_agent(self):
        e = SwarmMitosisEngine(num_agents=1)
        assert e.get_agent("nope") is None


# ── Health Score ────────────────────────────────────────────────────────


class TestHealthScore:
    def test_health_range(self):
        e = SwarmMitosisEngine(num_agents=10)
        e.simulate(ticks=10)
        score = e._compute_health_score()
        assert 0 <= score <= 100

    def test_health_tier_mapping(self):
        assert SwarmMitosisEngine._health_tier(90) == "Thriving"
        assert SwarmMitosisEngine._health_tier(65) == "Stable"
        assert SwarmMitosisEngine._health_tier(45) == "Stressed"
        assert SwarmMitosisEngine._health_tier(25) == "Declining"
        assert SwarmMitosisEngine._health_tier(10) == "Collapsing"

    def test_empty_population_score_zero(self):
        e = SwarmMitosisEngine(num_agents=0)
        assert e._compute_health_score() == 0.0

    def test_report_includes_tier(self):
        e = SwarmMitosisEngine(num_agents=5)
        report = e.simulate(ticks=10)
        assert report.health_tier in ("Thriving", "Stable", "Stressed", "Declining", "Collapsing")


# ── Insights ────────────────────────────────────────────────────────────


class TestInsights:
    def test_insights_list(self):
        e = SwarmMitosisEngine(num_agents=5, carrying_capacity=30)
        e.simulate(ticks=30)
        insights = e._generate_insights()
        assert isinstance(insights, list)

    def test_extinct_insight(self):
        e = SwarmMitosisEngine(num_agents=0)
        insights = e._generate_insights()
        assert any("extinct" in i.lower() for i in insights)

    def test_carrying_capacity_insight(self):
        e = SwarmMitosisEngine(num_agents=48, carrying_capacity=50)
        e._tick = 1
        insights = e._generate_insights()
        assert any("capacity" in i.lower() for i in insights)


# ── Scenarios ───────────────────────────────────────────────────────────


class TestScenarios:
    def test_all_scenarios_exist(self):
        assert "default" in SCENARIOS
        assert "population_explosion" in SCENARIOS
        assert "stem_cell" in SCENARIOS
        assert "aging_crisis" in SCENARIOS
        assert "bottleneck" in SCENARIOS

    def test_bottleneck_causes_deaths(self):
        cfg = dict(SCENARIOS["bottleneck"])
        ticks = cfg.pop("ticks", 50)
        cfg.pop("growth_factor_threshold", None)
        cfg.pop("telomere_loss_per_division", None)
        e = SwarmMitosisEngine(**cfg)
        report = e.simulate(ticks=ticks)
        assert report.total_deaths > 0

    def test_aging_crisis_limited_divisions(self):
        """Aging crisis scenario has constrained max_divisions."""
        cfg = dict(SCENARIOS["aging_crisis"])
        ticks = cfg.pop("ticks", 50)
        cfg.pop("growth_factor_threshold", None)
        telo = cfg.pop("telomere_loss_per_division", None)
        if telo:
            cfg["telomere_loss_per_division"] = telo
        e = SwarmMitosisEngine(**cfg)
        report = e.simulate(ticks=ticks)
        # The max_divisions=4 constraint limits how many times each cell divides
        assert cfg["max_divisions"] == 4
        assert isinstance(report, MitosisReport)


# ── Export ──────────────────────────────────────────────────────────────


class TestExport:
    def test_export_json(self):
        e = SwarmMitosisEngine(num_agents=3)
        e.simulate(ticks=10)
        data = e.export_json()
        assert isinstance(data, dict)
        assert "overall_health" in data
        assert "total_divisions" in data

    def test_export_html(self):
        e = SwarmMitosisEngine(num_agents=3)
        e.simulate(ticks=10)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            e.export_html(path)
            content = open(path, encoding="utf-8").read()
            assert "Swarm Mitosis" in content
            assert "Health Score" in content
        finally:
            os.unlink(path)

    def test_json_serializable(self):
        e = SwarmMitosisEngine(num_agents=3)
        e.simulate(ticks=10)
        data = e.export_json()
        dumped = json.dumps(data, default=str)
        assert len(dumped) > 0


# ── CLI ─────────────────────────────────────────────────────────────────


class TestCLI:
    def test_main_default(self, capsys):
        main(["--ticks", "5"])
        out = capsys.readouterr().out
        assert "Swarm Mitosis Engine" in out

    def test_main_scenario(self, capsys):
        main(["--scenario", "bottleneck", "--ticks", "5"])
        out = capsys.readouterr().out
        assert "bottleneck" in out

    def test_main_with_html(self, capsys):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            main(["--ticks", "5", "--out", path])
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_main_with_json(self, capsys):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            main(["--ticks", "5", "--json", path])
            assert os.path.exists(path)
            data = json.loads(open(path).read())
            assert "overall_health" in data
        finally:
            os.unlink(path)

    def test_main_custom_agents(self, capsys):
        main(["--agents", "3", "--ticks", "5"])
        out = capsys.readouterr().out
        assert "3" in out


# ── Edge Cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_agent_simulation(self):
        e = SwarmMitosisEngine(num_agents=1, carrying_capacity=10)
        report = e.simulate(ticks=20)
        assert isinstance(report, MitosisReport)

    def test_many_ticks(self):
        e = SwarmMitosisEngine(num_agents=3, carrying_capacity=15)
        report = e.simulate(ticks=200)
        assert len(report.snapshots) == 200

    def test_report_generation_distribution(self):
        e = SwarmMitosisEngine(num_agents=3, carrying_capacity=20)
        report = e.simulate(ticks=30)
        assert isinstance(report.generation_distribution, dict)

    def test_population_peak_tracked(self):
        e = SwarmMitosisEngine(num_agents=3, carrying_capacity=20)
        report = e.simulate(ticks=30)
        assert report.population_peak >= 3

    def test_analyze_without_simulation(self):
        e = SwarmMitosisEngine(num_agents=5)
        report = e.analyze()
        assert report.total_divisions == 0
        assert report.overall_health >= 0
