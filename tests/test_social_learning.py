"""Tests for Swarm Social Learning Engine."""
import json
import sys
from pathlib import Path
from unittest.mock import patch


from src.social_learning import (
    AgentProfile,
    LearningMode,
    Skill,
    SkillComplexity,
    SocialLearningEngine,
    COMPLEXITY_LEVELS,
)


class TestInitialization:
    def test_engine_creates_agents(self):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        assert len(engine.agents) == 10

    def test_engine_creates_initial_skills(self):
        engine = SocialLearningEngine(num_initial_skills=8, seed=42)
        assert len(engine.skills) == 8

    def test_agents_have_starting_skills(self):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        for agent in engine.agents.values():
            assert len(agent.skills) >= 1

    def test_initial_skills_have_complexity_progression(self):
        engine = SocialLearningEngine(num_initial_skills=10, seed=42)
        complexities = [s.complexity for s in engine.skills.values()]
        assert SkillComplexity.TRIVIAL in complexities

    def test_minimum_agents_enforced(self):
        engine = SocialLearningEngine(num_agents=1, seed=42)
        assert len(engine.agents) == 2

    def test_minimum_skills_enforced(self):
        engine = SocialLearningEngine(num_initial_skills=0, seed=42)
        assert len(engine.skills) == 1

    def test_seed_produces_deterministic_results(self):
        e1 = SocialLearningEngine(num_agents=5, seed=123)
        e2 = SocialLearningEngine(num_agents=5, seed=123)
        assert list(e1.agents.keys()) == list(e2.agents.keys())
        assert list(e1.skills.keys()) == list(e2.skills.keys())


class TestSkillModel:
    def test_skill_has_required_fields(self):
        skill = Skill(skill_id="s1", name="test", complexity=SkillComplexity.TRIVIAL)
        assert skill.skill_id == "s1"
        assert skill.fitness_value == 0.5

    def test_complexity_levels_ordered(self):
        assert COMPLEXITY_LEVELS[SkillComplexity.TRIVIAL] < COMPLEXITY_LEVELS[SkillComplexity.MASTERWORK]

    def test_skill_prerequisites_default_empty(self):
        skill = Skill(skill_id="s1", name="x", complexity=SkillComplexity.SIMPLE)
        assert skill.prerequisites == []

    def test_skill_parent_skills_tracked(self):
        skill = Skill(skill_id="s1", name="x", complexity=SkillComplexity.MODERATE,
                      parent_skills=["s0a", "s0b"])
        assert len(skill.parent_skills) == 2


class TestPrerequisites:
    def test_has_prerequisites_true(self):
        engine = SocialLearningEngine(num_agents=5, num_initial_skills=5, seed=42)
        agent = list(engine.agents.values())[0]
        prereq_id = list(agent.skills.keys())[0]
        skill = Skill(skill_id="test", name="t", complexity=SkillComplexity.SIMPLE,
                      prerequisites=[prereq_id])
        assert engine._has_prerequisites(agent, skill)

    def test_has_prerequisites_false(self):
        engine = SocialLearningEngine(num_agents=5, num_initial_skills=5, seed=42)
        agent = list(engine.agents.values())[0]
        skill = Skill(skill_id="test", name="t", complexity=SkillComplexity.SIMPLE,
                      prerequisites=["nonexistent-skill"])
        assert not engine._has_prerequisites(agent, skill)

    def test_low_proficiency_prerequisite_fails(self):
        engine = SocialLearningEngine(num_agents=5, seed=42)
        agent = list(engine.agents.values())[0]
        agent.skills["low-skill"] = 0.1
        skill = Skill(skill_id="test", name="t", complexity=SkillComplexity.SIMPLE,
                      prerequisites=["low-skill"])
        assert not engine._has_prerequisites(agent, skill)


class TestObservationImitation:
    def test_observation_probability_decreases_with_complexity(self):
        engine = SocialLearningEngine(seed=42)
        trivial = Skill(skill_id="t", name="t", complexity=SkillComplexity.TRIVIAL)
        master = Skill(skill_id="m", name="m", complexity=SkillComplexity.MASTERWORK)
        assert engine._observation_probability(trivial) > engine._observation_probability(master)

    def test_imitation_probability_zero_without_prereqs(self):
        engine = SocialLearningEngine(num_agents=5, seed=42)
        agent = AgentProfile(agent_id="test")
        skill = Skill(skill_id="s", name="s", complexity=SkillComplexity.MODERATE,
                      prerequisites=["missing"])
        assert engine._imitation_probability(agent, skill) == 0.0

    def test_imitation_probability_positive_with_prereqs(self):
        engine = SocialLearningEngine(num_agents=5, seed=42)
        agent = AgentProfile(agent_id="test", skills={"prereq1": 0.7})
        skill = Skill(skill_id="s", name="s", complexity=SkillComplexity.SIMPLE,
                      prerequisites=["prereq1"])
        assert engine._imitation_probability(agent, skill) > 0.0

    def test_teaching_probability_higher_than_base(self):
        engine = SocialLearningEngine(seed=42)
        teacher = AgentProfile(agent_id="t", skills={"s1": 0.9}, teacher_reputation=0.8)
        skill = Skill(skill_id="s1", name="s", complexity=SkillComplexity.SIMPLE)
        teach_prob = engine._teaching_probability(teacher, skill)
        assert teach_prob > 0.5


class TestSimulation:
    def test_tick_advances_time(self):
        engine = SocialLearningEngine(num_agents=5, seed=42)
        engine.tick()
        assert engine.current_tick == 1

    def test_simulation_produces_events(self):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        report = engine.simulate(steps=50)
        assert report.total_events > 0

    def test_simulation_creates_innovations(self):
        engine = SocialLearningEngine(num_agents=15, innovation_rate=0.1, seed=42)
        report = engine.simulate(steps=100)
        assert report.innovation_count > 0

    def test_skills_grow_over_time(self):
        engine = SocialLearningEngine(num_agents=10, num_initial_skills=5,
                                      innovation_rate=0.05, seed=42)
        initial_skills = len(engine.skills)
        engine.simulate(steps=100)
        assert len(engine.skills) >= initial_skills

    def test_complexity_grows_with_innovation(self):
        engine = SocialLearningEngine(num_agents=10, innovation_rate=0.1, seed=42)
        engine.simulate(steps=150)
        assert engine._max_complexity_depth() >= 1

    def test_agents_acquire_skills(self):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        initial_total = sum(len(a.skills) for a in engine.agents.values())
        engine.simulate(steps=100)
        final_total = sum(len(a.skills) for a in engine.agents.values())
        assert final_total >= initial_total

    def test_teaching_events_recorded(self):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        engine.simulate(steps=100)
        teach_events = [e for e in engine.events if e.mode == LearningMode.TEACHING]
        assert len(teach_events) > 0

    def test_timeline_recorded(self):
        engine = SocialLearningEngine(num_agents=5, seed=42)
        engine.simulate(steps=50)
        assert len(engine._complexity_timeline) == 50
        assert len(engine._gini_timeline) == 50


class TestCulturalHealth:
    def test_health_score_in_range(self):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        engine.simulate(steps=50)
        health = engine.analyze()
        assert 0 <= health.score <= 100

    def test_gini_in_range(self):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        engine.simulate(steps=50)
        health = engine.analyze()
        assert 0 <= health.knowledge_gini <= 1.0

    def test_diversity_in_range(self):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        engine.simulate(steps=50)
        health = engine.analyze()
        assert 0 <= health.skill_diversity <= 1.0

    def test_stagnation_risk_in_range(self):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        engine.simulate(steps=100)
        health = engine.analyze()
        assert 0 <= health.stagnation_risk <= 1.0

    def test_recommendations_generated(self):
        engine = SocialLearningEngine(num_agents=3, num_initial_skills=2,
                                      innovation_rate=0.0, seed=42)
        engine.simulate(steps=100)
        health = engine.analyze()
        assert len(health.recommendations) > 0


class TestInsights:
    def test_insights_generated(self):
        engine = SocialLearningEngine(num_agents=15, seed=42)
        engine.simulate(steps=100)
        insights = engine.get_insights()
        assert isinstance(insights, list)

    def test_insight_categories(self):
        engine = SocialLearningEngine(num_agents=15, innovation_rate=0.05, seed=42)
        engine.simulate(steps=150)
        insights = engine.get_insights()
        categories = {i.category for i in insights}
        assert len(categories) > 0

    def test_insight_has_severity(self):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        engine.simulate(steps=100)
        for ins in engine.get_insights():
            assert ins.severity in ("info", "warning", "critical")


class TestLineage:
    def test_lineages_created_for_initial_skills(self):
        engine = SocialLearningEngine(num_initial_skills=5, seed=42)
        assert len(engine.lineages) == 5

    def test_lineage_tracks_transmissions(self):
        engine = SocialLearningEngine(num_agents=15, seed=42)
        engine.simulate(steps=100)
        total = sum(l.total_transmissions for l in engine.lineages.values())
        assert total > 0

    def test_get_lineage_returns_none_for_unknown(self):
        engine = SocialLearningEngine(seed=42)
        assert engine.get_lineage("nonexistent") is None

    def test_innovation_creates_lineage(self):
        engine = SocialLearningEngine(num_agents=10, innovation_rate=0.1, seed=42)
        initial = len(engine.lineages)
        engine.simulate(steps=100)
        assert len(engine.lineages) >= initial


class TestAgentAccess:
    def test_get_agent(self):
        engine = SocialLearningEngine(num_agents=5, seed=42)
        agent = engine.get_agent("agent-000")
        assert agent is not None
        assert agent.agent_id == "agent-000"

    def test_get_agent_none_for_unknown(self):
        engine = SocialLearningEngine(seed=42)
        assert engine.get_agent("nonexistent") is None

    def test_get_skill(self):
        engine = SocialLearningEngine(num_initial_skills=5, seed=42)
        skill = engine.get_skill("skill-0001")
        assert skill is not None

    def test_get_skill_none_for_unknown(self):
        engine = SocialLearningEngine(seed=42)
        assert engine.get_skill("nonexistent") is None


class TestExport:
    def test_export_html(self, tmp_path):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        engine.simulate(steps=50)
        out = str(tmp_path / "report.html")
        engine.export_html(out)
        content = Path(out).read_text(encoding="utf-8")
        assert "Cultural Health Score" in content
        assert "<!DOCTYPE html>" in content

    def test_export_json(self, tmp_path):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        engine.simulate(steps=50)
        out = str(tmp_path / "report.json")
        engine.export_json(out)
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert "health" in data
        assert "skills" in data
        assert "agents" in data
        assert data["num_agents"] == 10

    def test_json_health_fields(self, tmp_path):
        engine = SocialLearningEngine(num_agents=10, seed=42)
        engine.simulate(steps=50)
        out = str(tmp_path / "report.json")
        engine.export_json(out)
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        health = data["health"]
        assert "score" in health
        assert "skill_diversity" in health
        assert "knowledge_gini" in health


class TestEdgeCases:
    def test_isolated_agent_no_crash(self):
        engine = SocialLearningEngine(num_agents=2, num_initial_skills=3, seed=42)
        engine.observation_range = 0
        engine.simulate(steps=20)

    def test_zero_innovation_rate(self):
        engine = SocialLearningEngine(num_agents=5, innovation_rate=0.0, seed=42)
        engine.simulate(steps=50)
        innovations = sum(1 for e in engine.events if e.mode == LearningMode.INNOVATION)
        assert innovations == 0

    def test_high_innovation_rate(self):
        engine = SocialLearningEngine(num_agents=10, innovation_rate=0.5, seed=42)
        engine.simulate(steps=50)
        assert len(engine.skills) > 10

    def test_many_agents(self):
        engine = SocialLearningEngine(num_agents=50, seed=42)
        engine.simulate(steps=30)
        assert engine.current_tick == 30

    def test_decay_does_not_crash(self):
        engine = SocialLearningEngine(num_agents=5, seed=42)
        agent = list(engine.agents.values())[0]
        agent.skills["test-decay"] = 0.06
        engine.skills["test-decay"] = Skill(
            skill_id="test-decay", name="decay-test",
            complexity=SkillComplexity.TRIVIAL
        )
        for _ in range(200):
            engine.tick()


class TestCLI:
    def test_cli_runs(self):
        with patch.object(sys, 'argv', ['social_learning', '--steps', '10',
                                        '--agents', '5', '--seed', '42']):
            from src.social_learning import main
            main()
