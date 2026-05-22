"""Tests for Swarm Dreaming Engine."""
import json
from pathlib import Path


from src.dreaming import (
    AnticipationEngine,
    ConsolidationEngine,
    DreamJournal,
    Episode,
    EpisodicMemory,
    Hypothesis,
    RecombinationEngine,
    ReplayEngine,
    Schema,
    SwarmDreamEngine,
)
import random


# ---------------------------------------------------------------------------
# EpisodicMemory Tests
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    def _make_episode(self, task="test", outcome="success", solution="sol",
                      agents=5, rounds=2, **kw):
        return Episode(task=task, outcome=outcome, solution=solution,
                       agents=agents, rounds=rounds, **kw)

    def test_record_and_len(self):
        mem = EpisodicMemory()
        assert len(mem) == 0
        mem.record(self._make_episode())
        assert len(mem) == 1

    def test_successes_and_failures(self):
        mem = EpisodicMemory()
        mem.record(self._make_episode(outcome="success"))
        mem.record(self._make_episode(outcome="failure"))
        mem.record(self._make_episode(outcome="success"))
        assert len(mem.successes()) == 2
        assert len(mem.failures()) == 1

    def test_recent(self):
        mem = EpisodicMemory()
        for i in range(10):
            mem.record(self._make_episode(task=f"task-{i}",
                                          timestamp=float(i)))
        recent = mem.recent(3)
        assert len(recent) == 3
        assert recent[0].task == "task-9"

    def test_by_tag(self):
        mem = EpisodicMemory()
        mem.record(self._make_episode(tags=["routing"]))
        mem.record(self._make_episode(tags=["scheduling"]))
        mem.record(self._make_episode(tags=["routing"]))
        assert len(mem.by_tag("routing")) == 2

    def test_token_frequency(self):
        mem = EpisodicMemory()
        mem.record(self._make_episode(task="route optimization"))
        mem.record(self._make_episode(task="route planning"))
        freq = mem.token_frequency()
        assert freq["route"] == 2

    def test_save_and_load(self, tmp_path):
        mem = EpisodicMemory()
        mem.record(self._make_episode(task="save-test"))
        path = str(tmp_path / "mem.json")
        mem.save(path)
        loaded = EpisodicMemory.load(path)
        assert len(loaded) == 1
        assert loaded.episodes[0].task == "save-test"


# ---------------------------------------------------------------------------
# Episode Tests
# ---------------------------------------------------------------------------

class TestEpisode:
    def test_tokens(self):
        ep = Episode(task="Route optimization for fleet",
                     outcome="success", solution="greedy", agents=3, rounds=1)
        tokens = ep.tokens()
        assert "route" in tokens
        assert "optimization" in tokens
        assert "for" not in tokens  # 3 chars, filtered by > 3

    def test_defaults(self):
        ep = Episode(task="t", outcome="success", solution=None, agents=1, rounds=1)
        assert ep.context == {}
        assert ep.tags == []
        assert ep.timestamp > 0


# ---------------------------------------------------------------------------
# ReplayEngine Tests
# ---------------------------------------------------------------------------

class TestReplayEngine:
    def test_replay_produces_results(self):
        engine = ReplayEngine(noise_level=0.2)
        episodes = [
            Episode(task="test", outcome="success", solution="x", agents=5, rounds=2),
            Episode(task="test2", outcome="failure", solution=None, agents=3, rounds=4),
        ]
        replays = engine.replay(episodes, random.Random(42))
        assert len(replays) == 2
        assert "stability" in replays[0]
        assert "robust" in replays[0]

    def test_replay_empty(self):
        engine = ReplayEngine()
        assert engine.replay([], random.Random(42)) == []


# ---------------------------------------------------------------------------
# ConsolidationEngine Tests
# ---------------------------------------------------------------------------

class TestConsolidationEngine:
    def _episodes(self):
        return [
            Episode(task="route optimization", outcome="success", solution="greedy",
                    agents=5, rounds=2),
            Episode(task="route planning", outcome="success", solution="astar",
                    agents=4, rounds=1),
            Episode(task="load balancing", outcome="failure", solution=None,
                    agents=3, rounds=5),
            Episode(task="route scheduling", outcome="success", solution="dp",
                    agents=6, rounds=2),
            Episode(task="conflict resolve", outcome="failure", solution=None,
                    agents=2, rounds=4),
        ]

    def test_produces_schemas(self):
        engine = ConsolidationEngine()
        schemas = engine.consolidate(self._episodes())
        assert len(schemas) > 0
        assert all(isinstance(s, Schema) for s in schemas)

    def test_agent_threshold_schema(self):
        engine = ConsolidationEngine()
        schemas = engine.consolidate(self._episodes())
        rules = [s.rule for s in schemas]
        assert any("agent" in r.lower() for r in rules)

    def test_failure_pattern_schema(self):
        engine = ConsolidationEngine()
        schemas = engine.consolidate(self._episodes())
        rules = [s.rule for s in schemas]
        assert any("failure" in r.lower() or "prolonged" in r.lower() for r in rules)

    def test_empty_episodes(self):
        engine = ConsolidationEngine()
        assert engine.consolidate([]) == []

    def test_confidence_bounded(self):
        engine = ConsolidationEngine()
        schemas = engine.consolidate(self._episodes())
        for s in schemas:
            assert 0 <= s.confidence <= 1.0


# ---------------------------------------------------------------------------
# RecombinationEngine Tests
# ---------------------------------------------------------------------------

class TestRecombinationEngine:
    def test_generates_hypotheses(self):
        engine = RecombinationEngine()
        episodes = [
            Episode(task="route optimization", outcome="success",
                    solution="greedy-refine", agents=5, rounds=2),
            Episode(task="load balancing", outcome="success",
                    solution="round-robin", agents=4, rounds=1),
            Episode(task="scheduling", outcome="success",
                    solution="priority-queue", agents=3, rounds=2),
        ]
        hypotheses = engine.recombine(episodes, random.Random(42))
        assert len(hypotheses) > 0
        assert all(isinstance(h, Hypothesis) for h in hypotheses)

    def test_lucidity_and_novelty_bounded(self):
        engine = RecombinationEngine()
        episodes = [
            Episode(task="A task", outcome="success", solution="method-one",
                    agents=4, rounds=2),
            Episode(task="B task", outcome="success", solution="method-two",
                    agents=5, rounds=1),
        ]
        hypotheses = engine.recombine(episodes, random.Random(1))
        for h in hypotheses:
            assert 0 <= h.lucidity <= 1.0
            assert 0 <= h.novelty <= 1.0

    def test_insufficient_solutions(self):
        engine = RecombinationEngine()
        episodes = [Episode(task="only one", outcome="success",
                            solution="solo", agents=3, rounds=1)]
        assert engine.recombine(episodes, random.Random(42)) == []


# ---------------------------------------------------------------------------
# AnticipationEngine Tests
# ---------------------------------------------------------------------------

class TestAnticipationEngine:
    def test_detects_agent_growth(self):
        engine = AnticipationEngine()
        episodes = [
            Episode(task=f"task-{i}", outcome="success", solution="x",
                    agents=3 + i, rounds=2, timestamp=float(i))
            for i in range(5)
        ]
        anticipations = engine.anticipate(episodes)
        scenarios = [a.scenario for a in anticipations]
        assert any("agent" in s.lower() for s in scenarios)

    def test_detects_high_failure_rate(self):
        engine = AnticipationEngine()
        episodes = [
            Episode(task=f"task-{i}", outcome="failure" if i % 2 == 0 else "success",
                    solution="x" if i % 2 != 0 else None,
                    agents=4, rounds=3, timestamp=float(i))
            for i in range(10)
        ]
        anticipations = engine.anticipate(episodes)
        scenarios = [a.scenario for a in anticipations]
        assert any("failure" in s.lower() for s in scenarios)

    def test_detects_round_inflation(self):
        engine = AnticipationEngine()
        episodes = [
            Episode(task=f"task-{i}", outcome="success", solution="x",
                    agents=4, rounds=1 + i, timestamp=float(i))
            for i in range(6)
        ]
        anticipations = engine.anticipate(episodes)
        scenarios = [a.scenario for a in anticipations]
        assert any("round" in s.lower() for s in scenarios)

    def test_empty_episodes(self):
        engine = AnticipationEngine()
        assert engine.anticipate([]) == []

    def test_probability_bounded(self):
        engine = AnticipationEngine()
        episodes = [
            Episode(task=f"task-{i}", outcome="success", solution="x",
                    agents=3+i, rounds=2, timestamp=float(i))
            for i in range(10)
        ]
        for a in engine.anticipate(episodes):
            assert 0 <= a.probability <= 1.0


# ---------------------------------------------------------------------------
# SwarmDreamEngine Integration Tests
# ---------------------------------------------------------------------------

class TestSwarmDreamEngine:
    def _make_memory(self):
        mem = EpisodicMemory()
        tasks = [
            ("Route optimization", "success", "greedy-refine", 5, 2),
            ("Load balancing", "success", "round-robin", 4, 1),
            ("Conflict resolution", "failure", None, 3, 5),
            ("Schedule planning", "success", "constraint-prop", 6, 2),
            ("Risk assessment", "failure", None, 3, 4),
            ("Cache update", "success", "ttl-adaptive", 4, 1),
            ("Network recovery", "success", "quorum-reconcile", 5, 3),
            ("Task delegation", "success", "skill-auction", 7, 2),
        ]
        for i, (task, outcome, sol, agents, rounds) in enumerate(tasks):
            mem.record(Episode(task=task, outcome=outcome, solution=sol,
                               agents=agents, rounds=rounds, timestamp=float(i)))
        return mem

    def test_dream_returns_journal(self):
        engine = SwarmDreamEngine(memory=self._make_memory(), seed=42)
        journal = engine.dream(cycles=2)
        assert isinstance(journal, DreamJournal)
        assert len(journal.cycles) == 2

    def test_dream_produces_all_outputs(self):
        engine = SwarmDreamEngine(memory=self._make_memory(), seed=42)
        journal = engine.dream(cycles=3)
        assert len(journal.schemas) > 0
        assert len(journal.anticipations) > 0

    def test_lucidity_threshold_filters(self):
        mem = self._make_memory()
        # High threshold should filter more hypotheses
        engine_strict = SwarmDreamEngine(memory=mem, lucidity_threshold=0.9, seed=42)
        engine_loose = SwarmDreamEngine(memory=mem, lucidity_threshold=0.1, seed=42)
        j_strict = engine_strict.dream(cycles=2)
        j_loose = engine_loose.dream(cycles=2)
        assert len(j_strict.hypotheses) <= len(j_loose.hypotheses)

    def test_empty_memory_graceful(self):
        engine = SwarmDreamEngine(memory=EpisodicMemory(), seed=42)
        journal = engine.dream(cycles=1)
        assert len(journal.cycles) == 0
        assert journal.schemas == []
        assert journal.hypotheses == []

    def test_deduplicates_schemas(self):
        engine = SwarmDreamEngine(memory=self._make_memory(), seed=42)
        journal = engine.dream(cycles=5)
        rules = [s.rule for s in journal.schemas]
        assert len(rules) == len(set(rules))

    def test_deduplicates_anticipations(self):
        engine = SwarmDreamEngine(memory=self._make_memory(), seed=42)
        journal = engine.dream(cycles=5)
        scenarios = [a.scenario for a in journal.anticipations]
        assert len(scenarios) == len(set(scenarios))

    def test_overall_lucidity_computed(self):
        engine = SwarmDreamEngine(memory=self._make_memory(),
                                  lucidity_threshold=0.0, seed=42)
        journal = engine.dream(cycles=2)
        if journal.hypotheses:
            assert journal.overall_lucidity > 0

    def test_export_json(self, tmp_path):
        engine = SwarmDreamEngine(memory=self._make_memory(), seed=42)
        engine.dream(cycles=2)
        path = str(tmp_path / "dream.json")
        engine.export_json(path)
        data = json.loads(Path(path).read_text())
        assert "schemas" in data
        assert "hypotheses" in data
        assert "anticipations" in data

    def test_export_html(self, tmp_path):
        engine = SwarmDreamEngine(memory=self._make_memory(), seed=42)
        engine.dream(cycles=2)
        path = str(tmp_path / "dream.html")
        engine.export_html(path)
        content = Path(path).read_text()
        assert "Swarm Dream Report" in content
        assert "chart.js" in content

    def test_validate_hypothesis(self):
        engine = SwarmDreamEngine(memory=self._make_memory(), seed=42,
                                  lucidity_threshold=0.0)
        engine.dream(cycles=2)
        if engine._journal and engine._journal.hypotheses:
            h = engine._journal.hypotheses[0]
            # Validate with matching text
            result = engine.validate_hypothesis(h, h.description)
            assert result is True

    def test_dream_cycle_phases(self):
        engine = SwarmDreamEngine(memory=self._make_memory(), seed=42)
        journal = engine.dream(cycles=1)
        cycle = journal.cycles[0]
        phase_types = [p.phase_type for p in cycle.phases]
        assert "replay" in phase_types
        assert "consolidation" in phase_types
        assert "recombination" in phase_types
        assert "anticipation" in phase_types

    def test_total_duration_positive(self):
        engine = SwarmDreamEngine(memory=self._make_memory(), seed=42)
        journal = engine.dream(cycles=2)
        assert journal.total_duration_ms > 0


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_main_runs(self, tmp_path, monkeypatch):
        import asyncio
        from src.dreaming import _main
        monkeypatch.setattr("sys.argv", [
            "dreaming", "--cycles", "2", "--seed", "1",
            "--out", str(tmp_path / "out.html"),
            "--json", str(tmp_path / "out.json"),
        ])
        asyncio.run(_main())
        assert (tmp_path / "out.html").exists()
        assert (tmp_path / "out.json").exists()
