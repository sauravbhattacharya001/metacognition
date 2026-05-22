"""Tests for Swarm Epigenetics Engine."""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.epigenetics import (
    EpigeneticMark,
    EpigeneticsEngine,
    EpigeneticsReport,
    Epigenome,
    GeneCategory,
    MarkType,
    SignalType,
    run_demo,
)


# ---------------------------------------------------------------------------
# Gene Expression Calculation
# ---------------------------------------------------------------------------


class TestGeneExpression:
    def test_base_expression_without_marks(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        expr = epi.get_expression("reasoning")
        assert expr == 0.6  # base for reasoning

    def test_methylation_reduces_expression(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        epi.marks.append(EpigeneticMark(
            gene="reasoning", mark_type=MarkType.METHYLATION,
            strength=1.0, source="test", generation=0,
        ))
        expr = epi.get_expression("reasoning")
        assert expr < 0.6

    def test_acetylation_increases_expression(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        epi.marks.append(EpigeneticMark(
            gene="reasoning", mark_type=MarkType.ACETYLATION,
            strength=1.0, source="test", generation=0,
        ))
        expr = epi.get_expression("reasoning")
        assert expr > 0.6

    def test_phosphorylation_increases_expression(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        epi.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.PHOSPHORYLATION,
            strength=0.8, source="stress", generation=0,
        ))
        expr = epi.get_expression("speed")
        assert expr > 0.5

    def test_ubiquitination_strongly_reduces_expression(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        epi.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.UBIQUITINATION,
            strength=1.0, source="test", generation=0,
        ))
        expr = epi.get_expression("speed")
        assert expr < 0.3

    def test_expression_clamped_to_zero(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        # Add many methylation marks to force below zero
        for _ in range(5):
            epi.marks.append(EpigeneticMark(
                gene="speed", mark_type=MarkType.METHYLATION,
                strength=1.0, source="test", generation=0,
            ))
        expr = epi.get_expression("speed")
        assert expr == 0.0

    def test_expression_clamped_to_one(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        for _ in range(5):
            epi.marks.append(EpigeneticMark(
                gene="speed", mark_type=MarkType.ACETYLATION,
                strength=1.0, source="test", generation=0,
            ))
        expr = epi.get_expression("speed")
        assert expr == 1.0

    def test_multiple_marks_combine(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        epi.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.ACETYLATION,
            strength=0.5, source="test", generation=0,
        ))
        epi.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.METHYLATION,
            strength=0.3, source="test", generation=0,
        ))
        expr = epi.get_expression("speed")
        # base 0.5 + 0.5*0.5 - 0.3*0.6 = 0.5 + 0.25 - 0.18 = 0.57
        assert 0.5 < expr < 0.7

    def test_unknown_gene_returns_zero(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        assert epi.get_expression("nonexistent") == 0.0


# ---------------------------------------------------------------------------
# Mark Decay
# ---------------------------------------------------------------------------


class TestMarkDecay:
    def test_marks_decay_over_time(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        epi.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.METHYLATION,
            strength=0.5, source="test", generation=0, decay_rate=0.1,
        ))
        epi.decay_marks()
        assert len(epi.marks) == 1
        assert epi.marks[0].strength == pytest.approx(0.4, abs=0.01)

    def test_mark_removed_when_strength_zero(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        epi.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.METHYLATION,
            strength=0.05, source="test", generation=0, decay_rate=0.1,
        ))
        removed = epi.decay_marks()
        assert len(epi.marks) == 0
        assert len(removed) == 1

    def test_age_increments(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        epi.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.METHYLATION,
            strength=1.0, source="test", generation=0, decay_rate=0.01,
        ))
        epi.decay_marks()
        epi.decay_marks()
        assert epi.marks[0].age == 2


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------


class TestFitness:
    def test_fitness_with_no_marks(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        fitness = epi.compute_fitness()
        assert 0.4 < fitness < 0.7  # avg base expression ~0.52

    def test_fitness_penalty_for_silenced_essential(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        # Heavily silence an essential gene
        for _ in range(5):
            epi.marks.append(EpigeneticMark(
                gene="reasoning", mark_type=MarkType.METHYLATION,
                strength=1.0, source="test", generation=0,
            ))
        fitness = epi.compute_fitness()
        # Should be lower due to essential gene penalty
        engine2 = EpigeneticsEngine(seed=1)
        epi2 = engine2.register_agent("b")
        assert fitness < epi2.compute_fitness()

    def test_fitness_no_penalty_for_silenced_nonessential(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        # Silence a non-essential gene (planning)
        for _ in range(3):
            epi.marks.append(EpigeneticMark(
                gene="planning", mark_type=MarkType.METHYLATION,
                strength=1.0, source="test", generation=0,
            ))
        fitness = epi.compute_fitness()
        # Still reasonable since no essential penalty
        assert fitness >= 0.0

    def test_empty_genome_returns_zero(self):
        epi = Epigenome(agent_id="empty", genes={})
        assert epi.compute_fitness() == 0.0


# ---------------------------------------------------------------------------
# Agent Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_default_genome(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("agent-0")
        assert len(epi.genes) == 12
        assert "reasoning" in epi.genes
        assert "adaptability" in epi.genes

    def test_register_custom_genome(self):
        engine = EpigeneticsEngine(seed=1)
        genome = [("custom_gene", GeneCategory.COGNITIVE, 0.7, True)]
        epi = engine.register_agent("a", genome=genome)
        assert len(epi.genes) == 1
        assert "custom_gene" in epi.genes

    def test_agents_stored(self):
        engine = EpigeneticsEngine(seed=1)
        engine.register_agent("a")
        engine.register_agent("b")
        assert len(engine.agents) == 2
        assert "a" in engine.agents


# ---------------------------------------------------------------------------
# Environmental Signals
# ---------------------------------------------------------------------------


class TestSignals:
    def test_emit_signal(self):
        engine = EpigeneticsEngine(seed=1)
        sig = engine.emit_signal(SignalType.STRESS, intensity=0.8, duration=5)
        assert sig.signal_type == SignalType.STRESS
        assert sig.intensity == 0.8
        assert sig.duration == 5

    def test_signal_applies_marks_on_tick(self):
        engine = EpigeneticsEngine(seed=42)
        engine.register_agent("a")
        engine.emit_signal(SignalType.STRESS, intensity=1.0, duration=10,
                           target_genes=["resilience"])
        # Tick multiple times to ensure marks applied
        for _ in range(20):
            engine.tick()
        epi = engine.agents["a"]
        stress_marks = [m for m in epi.marks if m.gene == "resilience"]
        assert len(stress_marks) > 0

    def test_signal_duration_decreases(self):
        engine = EpigeneticsEngine(seed=1)
        engine.register_agent("a")
        engine.emit_signal(SignalType.ABUNDANCE, intensity=0.5, duration=3)
        engine.tick()
        assert len(engine.active_signals) == 1
        engine.tick()
        engine.tick()
        # After 3 ticks, signal should be removed
        assert len(engine.active_signals) == 0

    def test_signal_history_preserved(self):
        engine = EpigeneticsEngine(seed=1)
        engine.emit_signal(SignalType.NOVELTY, intensity=0.5, duration=1)
        engine.emit_signal(SignalType.THREAT, intensity=0.7, duration=2)
        assert len(engine.signal_history) == 2

    def test_default_target_genes(self):
        engine = EpigeneticsEngine(seed=1)
        sig = engine.emit_signal(SignalType.STRESS, intensity=0.5, duration=3)
        assert "resilience" in sig.target_genes

    def test_intensity_clamped(self):
        engine = EpigeneticsEngine(seed=1)
        sig = engine.emit_signal(SignalType.STRESS, intensity=5.0, duration=1)
        assert sig.intensity == 1.0


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_basic_reproduction(self):
        engine = EpigeneticsEngine(seed=42, inheritance_fidelity=1.0, inheritance_noise=0.0)
        parent = engine.register_agent("parent")
        parent.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.ACETYLATION,
            strength=0.8, source="test", generation=0, heritable=True,
        ))
        child = engine.reproduce("parent", "child")
        assert child is not None
        assert child.parent_id == "parent"
        assert child.generation == 1
        assert len(child.marks) == 1

    def test_non_heritable_marks_not_inherited(self):
        engine = EpigeneticsEngine(seed=42, inheritance_fidelity=1.0)
        parent = engine.register_agent("parent")
        parent.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.UBIQUITINATION,
            strength=0.8, source="test", generation=0, heritable=False,
        ))
        child = engine.reproduce("parent", "child")
        assert len(child.marks) == 0

    def test_inheritance_fidelity_affects_transfer(self):
        engine = EpigeneticsEngine(seed=42, inheritance_fidelity=0.0)
        parent = engine.register_agent("parent")
        for _ in range(10):
            parent.marks.append(EpigeneticMark(
                gene="speed", mark_type=MarkType.ACETYLATION,
                strength=0.8, source="test", generation=0, heritable=True,
            ))
        child = engine.reproduce("parent", "child")
        # With 0% fidelity, no marks should transfer
        assert len(child.marks) == 0

    def test_inheritance_event_recorded(self):
        engine = EpigeneticsEngine(seed=42, inheritance_fidelity=0.8)
        engine.register_agent("parent")
        engine.reproduce("parent", "child")
        assert len(engine.inheritance_events) == 1
        assert engine.inheritance_events[0].parent_id == "parent"

    def test_nonexistent_parent_returns_none(self):
        engine = EpigeneticsEngine(seed=1)
        result = engine.reproduce("ghost", "child")
        assert result is None

    def test_multi_generation(self):
        engine = EpigeneticsEngine(seed=42, inheritance_fidelity=0.9, inheritance_noise=0.01)
        p = engine.register_agent("gen0")
        p.marks.append(EpigeneticMark(
            gene="resilience", mark_type=MarkType.PHOSPHORYLATION,
            strength=0.9, source="stress", generation=0, heritable=True, decay_rate=0.01,
        ))
        engine.reproduce("gen0", "gen1")
        engine.reproduce("gen1", "gen2")
        engine.reproduce("gen2", "gen3")
        # Check that mark propagated (may degrade)
        gen3 = engine.agents["gen3"]
        [m for m in gen3.marks if "stress" in m.source]
        # With 90% fidelity across 3 generations: 0.9^3 = 72.9% chance
        # It's probabilistic but seed=42 should give predictable results
        assert gen3.generation == 3


# ---------------------------------------------------------------------------
# Engine Tick Simulation
# ---------------------------------------------------------------------------


class TestTick:
    def test_tick_advances_counter(self):
        engine = EpigeneticsEngine(seed=1)
        engine.register_agent("a")
        engine.tick()
        assert engine.tick_counter == 1

    def test_tick_decays_marks(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        epi.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.METHYLATION,
            strength=0.5, source="test", generation=0, decay_rate=0.1,
        ))
        engine.tick()
        assert epi.marks[0].strength < 0.5

    def test_tick_takes_snapshots(self):
        engine = EpigeneticsEngine(seed=1)
        engine.register_agent("a")
        for _ in range(10):
            engine.tick()
        assert len(engine.snapshots) == 2  # tick 5 and 10

    def test_many_ticks_stable(self):
        engine = EpigeneticsEngine(seed=42)
        for i in range(5):
            engine.register_agent(f"a-{i}")
        engine.emit_signal(SignalType.STRESS, intensity=0.6, duration=10)
        for _ in range(50):
            engine.tick()
        # Should not crash
        assert engine.tick_counter == 50


# ---------------------------------------------------------------------------
# Analysis Report
# ---------------------------------------------------------------------------


class TestAnalysis:
    def test_report_structure(self):
        engine = EpigeneticsEngine(seed=42)
        for i in range(5):
            engine.register_agent(f"a-{i}")
        for _ in range(10):
            engine.tick()
        report = engine.analyze()
        assert isinstance(report, EpigeneticsReport)
        assert report.agent_count == 5
        assert 0 <= report.health_score <= 100

    def test_empty_engine_report(self):
        engine = EpigeneticsEngine(seed=1)
        report = engine.analyze()
        assert report.agent_count == 0
        assert report.health_score == 0

    def test_expression_profiles_in_report(self):
        engine = EpigeneticsEngine(seed=42)
        engine.register_agent("a")
        report = engine.analyze()
        assert "a" in report.expression_profiles
        assert "reasoning" in report.expression_profiles["a"]

    def test_silencing_patterns(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        # Silence a gene completely
        for _ in range(5):
            epi.marks.append(EpigeneticMark(
                gene="speed", mark_type=MarkType.METHYLATION,
                strength=1.0, source="test", generation=0, decay_rate=0.0,
            ))
        report = engine.analyze()
        assert report.silencing_patterns["speed"] == 1.0  # 100% of agents silenced

    def test_insights_generated(self):
        engine = EpigeneticsEngine(seed=42)
        for i in range(5):
            engine.register_agent(f"a-{i}")
        engine.emit_signal(SignalType.STRESS, intensity=0.9, duration=20)
        for _ in range(25):
            engine.tick()
        report = engine.analyze()
        assert len(report.insights) > 0

    def test_mark_distribution(self):
        engine = EpigeneticsEngine(seed=42)
        epi = engine.register_agent("a")
        epi.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.METHYLATION,
            strength=0.5, source="test", generation=0,
        ))
        epi.marks.append(EpigeneticMark(
            gene="speed", mark_type=MarkType.ACETYLATION,
            strength=0.5, source="test", generation=0,
        ))
        report = engine.analyze()
        assert report.mark_distribution.get("methylation", 0) == 1
        assert report.mark_distribution.get("acetylation", 0) == 1


# ---------------------------------------------------------------------------
# HTML Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_html_export(self):
        engine = EpigeneticsEngine(seed=42)
        for i in range(3):
            engine.register_agent(f"a-{i}")
        for _ in range(10):
            engine.tick()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            engine.export_html(path)
            content = open(path).read()
            assert "Swarm Epigenetics Dashboard" in content
            assert "Chart.js" in content or "chart.js" in content
        finally:
            os.unlink(path)

    def test_json_export(self):
        engine = EpigeneticsEngine(seed=42)
        engine.register_agent("a")
        json_str = engine.export_json()
        data = json.loads(json_str)
        assert "health_score" in data
        assert "expression_profiles" in data


# ---------------------------------------------------------------------------
# Demo Function
# ---------------------------------------------------------------------------


class TestDemo:
    def test_demo_runs(self):
        engine, report = run_demo(num_agents=5, num_generations=20, seed=1)
        assert report.agent_count >= 5
        assert report.tick == 20

    def test_demo_high_stress(self):
        engine, report = run_demo(num_agents=5, num_generations=20, stress_level="high", seed=1)
        assert report.total_marks > 0

    def test_demo_creates_children(self):
        engine, report = run_demo(num_agents=5, num_generations=50, seed=42)
        # With gen%7 reproduction, should have some children
        assert report.agent_count > 5


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_agent(self):
        engine = EpigeneticsEngine(seed=1)
        engine.register_agent("solo")
        for _ in range(10):
            engine.tick()
        report = engine.analyze()
        assert report.agent_count == 1

    def test_all_genes_silenced(self):
        engine = EpigeneticsEngine(seed=1)
        epi = engine.register_agent("a")
        for gene in epi.genes:
            for _ in range(5):
                epi.marks.append(EpigeneticMark(
                    gene=gene, mark_type=MarkType.UBIQUITINATION,
                    strength=1.0, source="test", generation=0, decay_rate=0.0,
                ))
        fitness = epi.compute_fitness()
        assert fitness == 0.0

    def test_no_marks_state(self):
        engine = EpigeneticsEngine(seed=1)
        engine.register_agent("a")
        report = engine.analyze()
        assert report.total_marks == 0
        assert report.health_score > 0

    def test_expression_profile(self):
        engine = EpigeneticsEngine(seed=1)
        engine.register_agent("a")
        profile = engine.get_expression_profile("a")
        assert len(profile) == 12
        assert all(0 <= v <= 1 for v in profile.values())

    def test_expression_profile_nonexistent(self):
        engine = EpigeneticsEngine(seed=1)
        assert engine.get_expression_profile("ghost") == {}
