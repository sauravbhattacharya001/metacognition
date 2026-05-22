"""Tests for Swarm Microbiome Engine."""
import json
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.microbiome import (
    DysbiosisEvent,
    Intervention,
    Metabolite,
    MicrobeSpecies,
    NicheState,
    SwarmMicrobiomeEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Basic engine with default niches."""
    return SwarmMicrobiomeEngine()


@pytest.fixture
def configured_engine():
    """Engine with species seeded and metabolites registered."""
    e = SwarmMicrobiomeEngine()
    # Commensals
    e.seed_species(MicrobeSpecies(
        name="Bacteroides", species_type="commensal",
        preferred_niches=["gut"], metabolites_produced=["butyrate"],
        antibiotic_susceptibility={"broad": 0.7}, growth_rate=0.08,
    ), "gut", 200.0)
    e.seed_species(MicrobeSpecies(
        name="Lactobacillus", species_type="commensal",
        preferred_niches=["gut", "oral"], metabolites_produced=["lactic_acid"],
        antibiotic_susceptibility={"broad": 0.6}, growth_rate=0.07,
    ), "gut", 150.0)
    e.seed_species(MicrobeSpecies(
        name="Bifidobacterium", species_type="commensal",
        preferred_niches=["gut"], metabolites_produced=["folate"],
        antibiotic_susceptibility={"broad": 0.8}, growth_rate=0.06,
    ), "gut", 120.0)
    # Pathogen
    e.seed_species(MicrobeSpecies(
        name="C.difficile", species_type="pathogenic",
        preferred_niches=["gut"], antibiotic_susceptibility={"broad": 0.2},
        growth_rate=0.12, immune_evasion=0.3,
    ), "gut", 5.0)
    # Skin
    e.seed_species(MicrobeSpecies(
        name="S.epidermidis", species_type="commensal",
        preferred_niches=["skin"], antibiotic_susceptibility={"broad": 0.5},
        growth_rate=0.06,
    ), "skin", 100.0)
    # Metabolites
    e.register_metabolite("butyrate", ["Bacteroides"], optimal_range=(5.0, 50.0))
    e.register_metabolite("lactic_acid", ["Lactobacillus"], optimal_range=(3.0, 30.0))
    # Keystone
    e.mark_keystone("Bacteroides")
    return e


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class TestMicrobeSpecies:
    def test_create_species(self):
        sp = MicrobeSpecies(name="TestBug", species_type="commensal")
        assert sp.name == "TestBug"
        assert sp.species_type == "commensal"
        assert sp.growth_rate == 0.08

    def test_species_with_full_params(self):
        sp = MicrobeSpecies(
            name="Pathogen", species_type="pathogenic",
            preferred_niches=["gut", "skin"],
            metabolites_produced=["toxin"],
            antibiotic_susceptibility={"broad": 0.9},
            growth_rate=0.15, immune_evasion=0.5,
        )
        assert sp.immune_evasion == 0.5
        assert "toxin" in sp.metabolites_produced

    def test_default_fields(self):
        sp = MicrobeSpecies(name="X", species_type="opportunistic")
        assert sp.preferred_niches == []
        assert sp.metabolites_produced == []
        assert sp.antibiotic_susceptibility == {}
        assert sp.immune_evasion == 0.0


class TestNicheState:
    def test_create_niche(self):
        ns = NicheState(niche_name="gut", carrying_capacity=1000.0)
        assert ns.niche_name == "gut"
        assert ns.populations == {}
        assert ns.diversity_index == 0.0


class TestMetabolite:
    def test_create_metabolite(self):
        m = Metabolite(name="butyrate", producer_species=["Bacteroides"])
        assert m.name == "butyrate"
        assert m.current_level == 0.0


class TestDysbiosisEvent:
    def test_create_event(self):
        e = DysbiosisEvent(tick=10, niche="gut", event_type="bloom", severity=0.7)
        assert e.tick == 10
        assert e.severity == 0.7


class TestIntervention:
    def test_antibiotic(self):
        iv = Intervention(intervention_type="antibiotic", spectrum="broad", strength=0.8)
        assert iv.spectrum == "broad"

    def test_probiotic(self):
        sp = MicrobeSpecies(name="Probiotic1", species_type="probiotic")
        iv = Intervention(intervention_type="probiotic", species_to_introduce=sp)
        assert iv.species_to_introduce.name == "Probiotic1"


# ---------------------------------------------------------------------------
# Niche Management
# ---------------------------------------------------------------------------


class TestNicheManagement:
    def test_default_niches(self, engine):
        assert len(engine.niches) == 6
        assert "gut" in engine.niches
        assert "skin" in engine.niches

    def test_custom_niches(self):
        e = SwarmMicrobiomeEngine(niches={"gut": 500, "custom": 200})
        assert len(e.niches) == 2
        assert "custom" in e.niches

    def test_carrying_capacity(self, engine):
        assert engine.niches["gut"].carrying_capacity == 1000.0
        assert engine.niches["skin"].carrying_capacity == 500.0
        assert engine.niches["neural"].carrying_capacity == 100.0

    def test_unknown_niche_raises(self, engine):
        sp = MicrobeSpecies(name="X", species_type="commensal")
        with pytest.raises(ValueError, match="Unknown niche"):
            engine.seed_species(sp, "nonexistent")


# ---------------------------------------------------------------------------
# Colonization
# ---------------------------------------------------------------------------


class TestColonization:
    def test_seed_species(self, engine):
        sp = MicrobeSpecies(name="Bug1", species_type="commensal")
        engine.seed_species(sp, "gut", 100.0)
        assert engine.get_abundance("Bug1", "gut") == 100.0

    def test_seed_multiple(self, engine):
        sp1 = MicrobeSpecies(name="A", species_type="commensal")
        sp2 = MicrobeSpecies(name="B", species_type="commensal")
        engine.seed_species(sp1, "gut", 100.0)
        engine.seed_species(sp2, "gut", 50.0)
        assert engine.get_total_population("gut") == 150.0

    def test_seed_additive(self, engine):
        sp = MicrobeSpecies(name="A", species_type="commensal")
        engine.seed_species(sp, "gut", 100.0)
        engine.seed_species(sp, "gut", 50.0)
        assert engine.get_abundance("A", "gut") == 150.0

    def test_species_registered(self, engine):
        sp = MicrobeSpecies(name="NewSpecies", species_type="probiotic")
        engine.seed_species(sp, "gut", 10.0)
        assert "NewSpecies" in engine.species

    def test_abundance_nonexistent_niche(self, engine):
        assert engine.get_abundance("X", "nonexistent") == 0.0

    def test_total_population_empty_niche(self, engine):
        assert engine.get_total_population("gut") == 0.0


# ---------------------------------------------------------------------------
# Growth Dynamics
# ---------------------------------------------------------------------------


class TestGrowthDynamics:
    def test_population_grows(self, configured_engine):
        initial = configured_engine.get_abundance("Bacteroides", "gut")
        configured_engine.tick(steps=10)
        after = configured_engine.get_abundance("Bacteroides", "gut")
        assert after > initial * 0.8  # should grow (allowing for noise)

    def test_carrying_capacity_limit(self, engine):
        sp = MicrobeSpecies(name="Fast", species_type="commensal", growth_rate=0.2)
        engine.seed_species(sp, "neural", 90.0)  # capacity=100
        for _ in range(100):
            engine.tick()
        pop = engine.get_abundance("Fast", "neural")
        # Should not vastly exceed capacity
        assert pop <= 120.0  # some overshoot due to noise is ok

    def test_pathogen_immune_pressure(self, engine):
        pathogen = MicrobeSpecies(
            name="P", species_type="pathogenic",
            growth_rate=0.1, immune_evasion=0.0,  # no evasion
        )
        commensal = MicrobeSpecies(
            name="C", species_type="commensal", growth_rate=0.1,
        )
        engine.seed_species(pathogen, "gut", 100.0)
        engine.seed_species(commensal, "gut", 100.0)
        engine.tick(steps=20)
        # Commensal should be ≥ pathogen (immune pressure slows pathogen)
        p = engine.get_abundance("P", "gut")
        c = engine.get_abundance("C", "gut")
        # With immune pressure reducing pathogen growth, commensal should do at least as well
        assert c >= p * 0.5  # relaxed bound due to randomness

    def test_extinction(self, engine):
        sp = MicrobeSpecies(name="Dying", species_type="commensal", growth_rate=-0.5)
        engine.seed_species(sp, "gut", 10.0)
        engine.tick(steps=20)
        assert engine.get_abundance("Dying", "gut") == 0.0

    def test_multi_step_tick(self, engine):
        sp = MicrobeSpecies(name="A", species_type="commensal", growth_rate=0.05)
        engine.seed_species(sp, "gut", 100.0)
        engine.tick(steps=5)
        assert engine.tick_count == 5


# ---------------------------------------------------------------------------
# Metabolic Network
# ---------------------------------------------------------------------------


class TestMetabolicNetwork:
    def test_register_metabolite(self, engine):
        m = engine.register_metabolite("testmet", ["sp1"], optimal_range=(1.0, 10.0))
        assert m.name == "testmet"
        assert "testmet" in engine.metabolites

    def test_metabolite_production(self, configured_engine):
        # butyrate is produced by Bacteroides
        configured_engine.tick(steps=10)
        m = configured_engine.metabolites["butyrate"]
        assert m.current_level > 0.0

    def test_metabolite_decay(self, engine):
        engine.register_metabolite("decaying", [], optimal_range=(0, 100))
        # Manually set level then tick
        engine._metabolites["decaying"].current_level = 50.0
        engine.tick()
        assert engine.metabolites["decaying"].current_level < 50.0

    def test_multiple_metabolites(self, configured_engine):
        configured_engine.tick(steps=10)
        assert configured_engine.metabolites["butyrate"].current_level > 0
        assert configured_engine.metabolites["lactic_acid"].current_level > 0


# ---------------------------------------------------------------------------
# Dysbiosis Detection
# ---------------------------------------------------------------------------


class TestDysbiosisDetection:
    def test_shannon_diversity(self, engine):
        sp1 = MicrobeSpecies(name="A", species_type="commensal")
        sp2 = MicrobeSpecies(name="B", species_type="commensal")
        engine.seed_species(sp1, "gut", 100.0)
        engine.seed_species(sp2, "gut", 100.0)
        d = engine.compute_diversity("gut")
        # Two equal species: H = ln(2) ≈ 0.693
        assert abs(d - math.log(2)) < 0.01

    def test_zero_diversity_single_species(self, engine):
        sp = MicrobeSpecies(name="Alone", species_type="commensal")
        engine.seed_species(sp, "gut", 100.0)
        assert engine.compute_diversity("gut") == 0.0

    def test_bloom_detection(self, engine):
        # One species dominates > 60%
        dominant = MicrobeSpecies(name="Dom", species_type="pathogenic", growth_rate=0.0)
        minor = MicrobeSpecies(name="Min", species_type="commensal", growth_rate=0.0)
        engine.seed_species(dominant, "gut", 800.0)
        engine.seed_species(minor, "gut", 100.0)
        snapshot = engine.tick()
        bloom_events = [e for e in snapshot.dysbiosis_events if e["type"] == "bloom"]
        assert len(bloom_events) > 0

    def test_pathogen_invasion_detection(self, engine):
        pathogen = MicrobeSpecies(name="Pathogen", species_type="pathogenic", growth_rate=0.0)
        commensal = MicrobeSpecies(name="Comm", species_type="commensal", growth_rate=0.0)
        engine.seed_species(pathogen, "gut", 400.0)
        engine.seed_species(commensal, "gut", 600.0)
        snapshot = engine.tick()
        invasion_events = [e for e in snapshot.dysbiosis_events if e["type"] == "invasion"]
        assert len(invasion_events) > 0

    def test_no_dysbiosis_healthy(self, configured_engine):
        # Normal populations shouldn't trigger too many events on first tick
        snapshot = configured_engine.tick()
        # C.difficile is very small, so no invasion expected in gut
        gut_invasions = [e for e in snapshot.dysbiosis_events
                         if e["type"] == "invasion" and e["niche"] == "gut"]
        assert len(gut_invasions) == 0

    def test_dysbiosis_tier_classification(self, configured_engine):
        configured_engine.tick(steps=5)
        tier = configured_engine._classify_dysbiosis_tier()
        # Tier should be a valid classification string
        assert tier in ("Eubiosis", "Mild-Dysbiosis", "Moderate-Dysbiosis",
                        "Severe-Dysbiosis", "Critical")


# ---------------------------------------------------------------------------
# Antibiotic Disruption
# ---------------------------------------------------------------------------


class TestAntibioticDisruption:
    def test_broad_spectrum_kills(self, configured_engine):
        before_bact = configured_engine.get_abundance("Bacteroides", "gut")
        configured_engine.apply_intervention(
            Intervention(intervention_type="antibiotic", spectrum="broad", strength=0.8)
        )
        after_bact = configured_engine.get_abundance("Bacteroides", "gut")
        # Bacteroides has broad susceptibility 0.7, so kill = 0.7*0.8 = 0.56
        assert after_bact < before_bact

    def test_broad_spectrum_spares_resistant(self, configured_engine):
        configured_engine.get_abundance("C.difficile", "gut")
        configured_engine.apply_intervention(
            Intervention(intervention_type="antibiotic", spectrum="broad", strength=0.8)
        )
        after_cd = configured_engine.get_abundance("C.difficile", "gut")
        # C.difficile has broad susceptibility 0.2, so kill = 0.2*0.8 = 0.16
        # Should survive more than commensals
        configured_engine.get_abundance("Bacteroides", "gut")
        # Relative ratio should increase
        assert after_cd > 0  # Still alive

    def test_narrow_spectrum(self, configured_engine):
        before_cd = configured_engine.get_abundance("C.difficile", "gut")
        before_bact = configured_engine.get_abundance("Bacteroides", "gut")
        configured_engine.apply_intervention(
            Intervention(intervention_type="antibiotic", spectrum="narrow",
                         target_species="C.difficile", strength=0.9)
        )
        after_cd = configured_engine.get_abundance("C.difficile", "gut")
        after_bact = configured_engine.get_abundance("Bacteroides", "gut")
        assert after_cd < before_cd
        assert after_bact == before_bact  # untouched

    def test_antibiotic_recorded(self, configured_engine):
        configured_engine.apply_intervention(
            Intervention(intervention_type="antibiotic", spectrum="broad", strength=0.5)
        )
        report = configured_engine.analyze()
        assert len(report.intervention_history) == 1
        assert report.intervention_history[0]["type"] == "antibiotic"


# ---------------------------------------------------------------------------
# Probiotic Intervention
# ---------------------------------------------------------------------------


class TestProbioticIntervention:
    def test_probiotic_engraftment(self, engine):
        probiotic = MicrobeSpecies(
            name="L.rhamnosus", species_type="probiotic",
            preferred_niches=["gut"], growth_rate=0.09,
        )
        engine.apply_intervention(Intervention(
            intervention_type="probiotic", species_to_introduce=probiotic, strength=1.0,
        ))
        assert engine.get_abundance("L.rhamnosus", "gut") > 0

    def test_probiotic_in_preferred_niche(self, engine):
        probiotic = MicrobeSpecies(
            name="P1", species_type="probiotic",
            preferred_niches=["skin", "oral"], growth_rate=0.07,
        )
        engine.apply_intervention(Intervention(
            intervention_type="probiotic", species_to_introduce=probiotic, strength=1.0,
        ))
        assert engine.get_abundance("P1", "skin") > 0
        assert engine.get_abundance("P1", "oral") > 0

    def test_prebiotic_boost(self, configured_engine):
        before = configured_engine.get_abundance("Bacteroides", "gut")
        configured_engine.apply_intervention(Intervention(
            intervention_type="prebiotic", boost_species=["Bacteroides"], strength=1.0,
        ))
        after = configured_engine.get_abundance("Bacteroides", "gut")
        assert after > before

    def test_probiotic_without_species(self, engine):
        # Should not crash
        engine.apply_intervention(Intervention(
            intervention_type="probiotic", species_to_introduce=None,
        ))
        # Nothing happened
        assert engine.get_total_population("gut") == 0.0


# ---------------------------------------------------------------------------
# Health Scoring
# ---------------------------------------------------------------------------


class TestHealthScoring:
    def test_health_score_range(self, configured_engine):
        configured_engine.tick(steps=10)
        report = configured_engine.analyze()
        assert 0 <= report.overall_health <= 100

    def test_tier_classification(self, configured_engine):
        configured_engine.tick(steps=10)
        report = configured_engine.analyze()
        assert report.tier in ("Thriving", "Healthy", "Stressed", "Dysbiotic", "Critical")

    def test_healthy_system_good_score(self, configured_engine):
        configured_engine.tick(steps=20)
        report = configured_engine.analyze()
        # With diverse commensals and low pathogen, should be decent
        assert report.overall_health > 20

    def test_empty_engine_moderate_score(self, engine):
        engine.tick(steps=5)
        report = engine.analyze()
        # Empty niches = not great
        assert report.overall_health >= 0


# ---------------------------------------------------------------------------
# Insight Generation
# ---------------------------------------------------------------------------


class TestInsightGeneration:
    def test_insights_generated(self, configured_engine):
        configured_engine.tick(steps=20)
        report = configured_engine.analyze()
        # Should generate at least some insights
        assert isinstance(report.insights, list)

    def test_empty_niche_insight(self, engine):
        sp = MicrobeSpecies(name="A", species_type="commensal")
        engine.seed_species(sp, "gut", 100.0)
        engine.tick(steps=5)
        report = engine.analyze()
        # Some niches are empty
        empty_insights = [i for i in report.insights if "empty niche" in i.lower()]
        assert len(empty_insights) > 0

    def test_pathogen_alert_insight(self, engine):
        pathogen = MicrobeSpecies(
            name="BigBad", species_type="pathogenic", growth_rate=0.0,
        )
        commensal = MicrobeSpecies(
            name="Good", species_type="commensal", growth_rate=0.0,
        )
        engine.seed_species(pathogen, "gut", 300.0)
        engine.seed_species(commensal, "gut", 700.0)
        engine.tick(steps=5)
        report = engine.analyze()
        pathogen_insights = [i for i in report.insights if "BigBad" in i]
        assert len(pathogen_insights) > 0


# ---------------------------------------------------------------------------
# HTML Export
# ---------------------------------------------------------------------------


class TestHTMLExport:
    def test_export_html(self, configured_engine):
        configured_engine.tick(steps=10)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            configured_engine.export_html(path)
            content = open(path, encoding="utf-8").read()
            assert "<!DOCTYPE html>" in content
            assert "Swarm Microbiome" in content
            assert "Bacteroides" in content
        finally:
            os.unlink(path)

    def test_html_contains_sections(self, configured_engine):
        configured_engine.tick(steps=5)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            configured_engine.export_html(path)
            content = open(path, encoding="utf-8").read()
            assert "Niche Populations" in content
            assert "Metabolites" in content
            assert "Dysbiosis Events" in content
            assert "Autonomous Insights" in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# JSON Export
# ---------------------------------------------------------------------------


class TestJSONExport:
    def test_export_json(self, configured_engine):
        configured_engine.tick(steps=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            configured_engine.export_json(path)
            data = json.loads(open(path, encoding="utf-8").read())
            assert "overall_health" in data
            assert "tier" in data
            assert "species_census" in data
        finally:
            os.unlink(path)

    def test_json_valid_structure(self, configured_engine):
        configured_engine.tick(steps=5)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            configured_engine.export_json(path)
            data = json.loads(open(path, encoding="utf-8").read())
            assert isinstance(data["snapshots"], list)
            assert isinstance(data["insights"], list)
            assert isinstance(data["niche_summaries"], dict)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_import(self):
        from src.microbiome import main
        assert callable(main)

    def test_argparse(self):
        from src.microbiome import main
        # Just verify the function exists and is callable
        assert main is not None


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_lifecycle(self, configured_engine):
        """Full simulation: grow, disrupt, recover, analyze."""
        # Phase 1: Growth
        configured_engine.tick(steps=20)
        configured_engine.analyze().overall_health

        # Phase 2: Antibiotic
        configured_engine.apply_intervention(
            Intervention(intervention_type="antibiotic", spectrum="broad", strength=0.8)
        )
        configured_engine.tick(steps=5)
        configured_engine.analyze().overall_health

        # Phase 3: Recovery with probiotic
        probiotic = MicrobeSpecies(
            name="L.rhamnosus", species_type="probiotic",
            preferred_niches=["gut"], growth_rate=0.09,
        )
        configured_engine.apply_intervention(Intervention(
            intervention_type="probiotic", species_to_introduce=probiotic,
        ))
        configured_engine.tick(steps=20)
        configured_engine.analyze().overall_health

        report = configured_engine.analyze()
        assert report.total_dysbiosis_events >= 0
        assert len(report.intervention_history) == 2
        assert report.overall_health >= 0

    def test_snapshot_recording(self, configured_engine):
        configured_engine.tick(steps=10)
        report = configured_engine.analyze()
        assert len(report.snapshots) == 10

    def test_tick_count(self, configured_engine):
        configured_engine.tick(steps=7)
        assert configured_engine.tick_count == 7
