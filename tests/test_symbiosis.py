"""Tests for Swarm Symbiosis Engine."""
import json
import math
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.symbiosis import (
    DependencyType,
    Interaction,
    MutualismOpportunity,
    ParasiteAlert,
    RelationshipType,
    SymbiosisEngine,
    SymbiosisReport,
    SymbioticRelationship,
    run_demo,
)


# ---------------------------------------------------------------------------
# Interaction Recording
# ---------------------------------------------------------------------------

class TestInteractionRecording:
    def test_record_basic_interaction(self):
        engine = SymbiosisEngine()
        ix = engine.record_interaction("a", "b", 0.5, 0.3, "test")
        assert ix.agent_a == "a"
        assert ix.agent_b == "b"
        assert ix.benefit_a == 0.5
        assert ix.benefit_b == 0.3
        assert ix.context == "test"

    def test_benefit_clamping(self):
        ix = Interaction("a", "b", 0.0, 2.5, -3.0)
        assert ix.benefit_a == 1.0
        assert ix.benefit_b == -1.0

    def test_custom_timestamp(self):
        engine = SymbiosisEngine()
        ix = engine.record_interaction("a", "b", 0.5, 0.5, timestamp=12345.0)
        assert ix.timestamp == 12345.0

    def test_interactions_accumulate(self):
        engine = SymbiosisEngine()
        for i in range(10):
            engine.record_interaction("a", "b", 0.5, 0.5)
        assert len(engine.interactions) == 10

    def test_get_interactions_between(self):
        engine = SymbiosisEngine()
        engine.record_interaction("a", "b", 0.5, 0.3)
        engine.record_interaction("b", "a", 0.4, 0.6)
        engine.record_interaction("a", "c", 0.1, 0.2)
        result = engine.get_interactions_between("a", "b")
        assert len(result) == 2

    def test_get_interactions_between_order_independent(self):
        engine = SymbiosisEngine()
        engine.record_interaction("x", "y", 0.5, 0.3)
        assert len(engine.get_interactions_between("y", "x")) == 1

    def test_get_agent_ids(self):
        engine = SymbiosisEngine()
        engine.record_interaction("alpha", "beta", 0.5, 0.3)
        engine.record_interaction("gamma", "alpha", 0.2, 0.1)
        agents = engine.get_agent_ids()
        assert agents == {"alpha", "beta", "gamma"}

    def test_solo_performance_recording(self):
        engine = SymbiosisEngine()
        engine.record_solo_performance("agent-1", 0.7)
        engine.record_solo_performance("agent-1", 0.8)
        assert len(engine._agent_solo_performance["agent-1"]) == 2


# ---------------------------------------------------------------------------
# Relationship Classification
# ---------------------------------------------------------------------------

class TestRelationshipClassification:
    def _build_engine_with_pattern(self, benefit_a: float, benefit_b: float, count: int = 5):
        engine = SymbiosisEngine()
        for i in range(count):
            engine.record_interaction("a", "b", benefit_a, benefit_b, timestamp=float(i))
        return engine

    def test_mutualism_classification(self):
        engine = self._build_engine_with_pattern(0.7, 0.6)
        rel = engine.classify_relationship("a", "b")
        assert rel is not None
        assert rel.relationship_type == RelationshipType.MUTUALISM

    def test_parasitism_classification(self):
        engine = self._build_engine_with_pattern(0.8, -0.5)
        rel = engine.classify_relationship("a", "b")
        assert rel is not None
        assert rel.relationship_type == RelationshipType.PARASITISM

    def test_competition_classification(self):
        engine = self._build_engine_with_pattern(-0.4, -0.5)
        rel = engine.classify_relationship("a", "b")
        assert rel is not None
        assert rel.relationship_type == RelationshipType.COMPETITION

    def test_commensalism_classification(self):
        engine = self._build_engine_with_pattern(0.6, 0.05)
        rel = engine.classify_relationship("a", "b")
        assert rel is not None
        assert rel.relationship_type == RelationshipType.COMMENSALISM

    def test_amensalism_classification(self):
        engine = self._build_engine_with_pattern(0.0, -0.5)
        rel = engine.classify_relationship("a", "b")
        assert rel is not None
        assert rel.relationship_type == RelationshipType.AMENSALISM

    def test_insufficient_interactions_returns_none(self):
        engine = SymbiosisEngine(min_interactions=5)
        engine.record_interaction("a", "b", 0.5, 0.5)
        engine.record_interaction("a", "b", 0.5, 0.5)
        rel = engine.classify_relationship("a", "b")
        assert rel is None

    def test_confidence_increases_with_interactions(self):
        engine1 = self._build_engine_with_pattern(0.7, 0.6, count=3)
        engine2 = self._build_engine_with_pattern(0.7, 0.6, count=10)
        rel1 = engine1.classify_relationship("a", "b")
        rel2 = engine2.classify_relationship("a", "b")
        assert rel2.confidence >= rel1.confidence

    def test_relationship_has_timestamps(self):
        engine = self._build_engine_with_pattern(0.5, 0.5)
        rel = engine.classify_relationship("a", "b")
        assert rel.first_seen == 0.0
        assert rel.last_seen == 4.0


# ---------------------------------------------------------------------------
# Parasitism Detection
# ---------------------------------------------------------------------------

class TestParasitismDetection:
    def test_detects_parasite(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("parasite", "host", 0.8, -0.6, "drain")
        alerts = engine.detect_parasites()
        assert len(alerts) == 1
        assert alerts[0].parasite_id == "parasite"
        assert alerts[0].host_id == "host"

    def test_severity_calculation(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("p", "h", 0.9, -0.9, "severe")
        alerts = engine.detect_parasites()
        assert alerts[0].severity > 0.7

    def test_no_false_positives_for_mutualism(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", 0.7, 0.6)
        alerts = engine.detect_parasites()
        assert len(alerts) == 0

    def test_quarantine_recommendation_for_severe(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("p", "h", 0.9, -0.9)
        alerts = engine.detect_parasites()
        assert alerts[0].recommended_action == "quarantine"

    def test_monitor_recommendation_for_mild(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("p", "h", 0.4, -0.3)
        alerts = engine.detect_parasites()
        if alerts:
            assert alerts[0].recommended_action == "monitor"

    def test_evidence_contexts_captured(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("p", "h", 0.8, -0.6, f"drain-{i}")
        alerts = engine.detect_parasites()
        assert len(alerts[0].evidence_contexts) > 0


# ---------------------------------------------------------------------------
# Mutualism Opportunities
# ---------------------------------------------------------------------------

class TestMutualismOpportunities:
    def test_finds_strengthening_opportunity(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", 0.4, 0.3, timestamp=float(i))
        opps = engine.find_mutualism_opportunities()
        assert len(opps) >= 1
        assert opps[0].agent_a == "a" or opps[0].agent_b == "a"

    def test_no_opportunities_for_competition(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", -0.5, -0.5, timestamp=float(i))
        opps = engine.find_mutualism_opportunities()
        assert len(opps) == 0

    def test_sorted_by_potential(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", 0.3, 0.3, timestamp=float(i))
            engine.record_interaction("c", "d", 0.8, 0.7, timestamp=float(i + 100))
        opps = engine.find_mutualism_opportunities()
        if len(opps) >= 2:
            assert opps[0].potential_benefit >= opps[1].potential_benefit


# ---------------------------------------------------------------------------
# Dependency Analysis
# ---------------------------------------------------------------------------

class TestDependencyAnalysis:
    def test_obligate_dependency_detected(self):
        engine = SymbiosisEngine()
        # Low solo performance + high collaborative benefit
        for _ in range(5):
            engine.record_solo_performance("a", 0.1)
        for i in range(5):
            engine.record_interaction("a", "b", 0.8, 0.7, timestamp=float(i))
        rel = engine.classify_relationship("a", "b")
        assert rel is not None
        assert rel.dependency_type_a == DependencyType.OBLIGATE

    def test_no_dependency_without_solo_data(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", 0.5, 0.5, timestamp=float(i))
        rel = engine.classify_relationship("a", "b")
        assert rel.dependency_type_a == DependencyType.NONE

    def test_dependency_graph_built(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", 0.7, 0.6, timestamp=float(i))
        graph = engine.build_dependency_graph()
        assert "a" in graph
        assert "b" in graph


# ---------------------------------------------------------------------------
# Ecosystem Health Scoring
# ---------------------------------------------------------------------------

class TestEcosystemHealth:
    def test_all_mutualism_high_score(self):
        engine = SymbiosisEngine()
        rels = [
            SymbioticRelationship("a", "b", RelationshipType.MUTUALISM, 0.9, 5, 0.7, 0.6),
            SymbioticRelationship("c", "d", RelationshipType.MUTUALISM, 0.8, 5, 0.6, 0.5),
        ]
        score = engine.compute_ecosystem_health(rels)
        assert score > 60

    def test_all_parasitism_low_score(self):
        engine = SymbiosisEngine()
        rels = [
            SymbioticRelationship("a", "b", RelationshipType.PARASITISM, 0.9, 5, 0.7, -0.5),
            SymbioticRelationship("c", "d", RelationshipType.PARASITISM, 0.8, 5, 0.8, -0.6),
        ]
        score = engine.compute_ecosystem_health(rels)
        assert score < 40

    def test_empty_relationships_neutral(self):
        engine = SymbiosisEngine()
        score = engine.compute_ecosystem_health([])
        assert score == 50.0

    def test_diverse_ecosystem_moderate_score(self):
        engine = SymbiosisEngine()
        rels = [
            SymbioticRelationship("a", "b", RelationshipType.MUTUALISM, 0.9, 5, 0.7, 0.6),
            SymbioticRelationship("c", "d", RelationshipType.COMMENSALISM, 0.8, 5, 0.5, 0.0),
            SymbioticRelationship("e", "f", RelationshipType.COMPETITION, 0.7, 5, -0.3, -0.4),
        ]
        score = engine.compute_ecosystem_health(rels)
        assert 30 < score < 80

    def test_score_bounded_0_100(self):
        engine = SymbiosisEngine()
        rels = [SymbioticRelationship("a", "b", RelationshipType.MUTUALISM, 1.0, 100, 1.0, 1.0)]
        score = engine.compute_ecosystem_health(rels)
        assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# Full Analysis
# ---------------------------------------------------------------------------

class TestFullAnalysis:
    def test_analyze_returns_report(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", 0.7, 0.6, timestamp=float(i))
        report = engine.analyze()
        assert isinstance(report, SymbiosisReport)
        assert report.agent_count == 2
        assert report.interaction_count == 5

    def test_analyze_empty_engine(self):
        engine = SymbiosisEngine()
        report = engine.analyze()
        assert report.agent_count == 0
        assert report.ecosystem_health_score == 50.0

    def test_insights_generated(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", 0.7, 0.6, timestamp=float(i))
        report = engine.analyze()
        assert len(report.insights) > 0

    def test_relationship_distribution(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", 0.7, 0.6, timestamp=float(i))
            engine.record_interaction("c", "d", -0.4, -0.5, timestamp=float(i + 100))
        report = engine.analyze()
        assert "mutualism" in report.relationship_distribution
        assert "competition" in report.relationship_distribution


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_json(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", 0.7, 0.6, timestamp=float(i))
        report = engine.analyze()
        json_str = engine.export_json(report)
        data = json.loads(json_str)
        assert "ecosystem_health_score" in data
        assert "relationships" in data
        assert "insights" in data

    def test_export_html(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", 0.7, 0.6, timestamp=float(i))
        report = engine.analyze()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            engine.export_html(path, report)
            content = open(path, encoding="utf-8").read()
            assert "Swarm Symbiosis Dashboard" in content
            assert "Ecosystem Health Score" in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Demo & CLI
# ---------------------------------------------------------------------------

class TestDemoAndCLI:
    def test_run_demo_default(self):
        report = run_demo()
        assert report.agent_count == 8
        assert report.interaction_count > 0
        assert len(report.relationships) > 0

    def test_run_demo_custom_params(self):
        report = run_demo(num_agents=5, num_interactions=50)
        assert report.agent_count == 5

    def test_demo_produces_insights(self):
        report = run_demo()
        assert len(report.insights) > 0

    def test_demo_health_score_valid(self):
        report = run_demo()
        assert 0 <= report.ecosystem_health_score <= 100


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_agent_no_classification(self):
        engine = SymbiosisEngine()
        engine.record_interaction("a", "a", 0.5, 0.5)  # self-interaction
        report = engine.analyze()
        # Self-interaction recorded but may not produce valid relationships
        assert report.ecosystem_health_score >= 0

    def test_all_neutral_interactions(self):
        engine = SymbiosisEngine()
        for i in range(5):
            engine.record_interaction("a", "b", 0.0, 0.0, timestamp=float(i))
        rel = engine.classify_relationship("a", "b")
        # Both neutral → commensalism fallback
        assert rel.relationship_type == RelationshipType.COMMENSALISM

    def test_many_agents_few_interactions(self):
        engine = SymbiosisEngine()
        for i in range(20):
            engine.record_interaction(f"agent-{i}", f"agent-{i+1}", 0.5, 0.5, timestamp=float(i))
        report = engine.analyze()
        # Most pairs won't have enough interactions for classification
        assert report.agent_count == 21
