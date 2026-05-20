"""Tests for the Network Partition Simulator (src/partition.py).

Covers:
- Partition dataclass auto-label behavior
- PartitionStrategy: random_split, isolate_leader, isolate_byzantine,
  minority_split (including degenerate edges: all-honest / all-byzantine /
  single agent / odd sizing).
- NetworkPartitionSimulator.simulate: split-brain detection, healing,
  quorum bookkeeping, empty-partition handling.
- sweep_partitions covers every named strategy.
- _get_partitions error path on unknown strategy.
- build_agents byzantine count + confidence ranges.
- generate_html_report: shape + presence of key markers; sweep + healed flow.
- CLI main() with --json and --report (smoke).
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from src.agents.metacognitive import MockAgent
from src.core.state import RoundResult
from src.partition import (
    NetworkPartitionSimulator,
    Partition,
    PartitionEvent,
    PartitionResult,
    PartitionStrategy,
    build_agents,
    generate_html_report,
    main,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _honest_swarm(answer: str = "X", n: int = 5) -> list[MockAgent]:
    return [MockAgent(f"a{i+1}", answer, 0.7 + 0.05 * i) for i in range(n)]


def _mixed_swarm() -> list[MockAgent]:
    """3 honest agree on X + 2 disagreeing/byzantine."""
    return [
        MockAgent("h1", "X", 0.80),
        MockAgent("h2", "X", 0.75),
        MockAgent("h3", "X", 0.70),
        MockAgent("d1", "Y", 0.65),  # honest but disagrees
        MockAgent("b1", "Z", 0.95, byzantine=True),
    ]


# --------------------------------------------------------------------------- #
# Partition dataclass
# --------------------------------------------------------------------------- #

class TestPartition:
    def test_auto_label_when_blank(self) -> None:
        p = Partition(partition_id=3, agent_ids={"a", "b"})
        assert p.label == "P3"

    def test_explicit_label_kept(self) -> None:
        p = Partition(partition_id=0, agent_ids={"a"}, label="Custom")
        assert p.label == "Custom"


# --------------------------------------------------------------------------- #
# PartitionStrategy
# --------------------------------------------------------------------------- #

class TestPartitionStrategy:
    def test_random_split_partitions_all_agents(self) -> None:
        ids = [f"a{i}" for i in range(7)]
        parts = PartitionStrategy.random_split(ids, num_partitions=3, seed=42)
        # Every agent appears exactly once across all partitions.
        all_assigned: set[str] = set()
        for p in parts:
            assert p.agent_ids.isdisjoint(all_assigned)
            all_assigned |= p.agent_ids
        assert all_assigned == set(ids)
        # Up to num_partitions buckets (empty ones are dropped).
        assert 1 <= len(parts) <= 3

    def test_random_split_seed_is_deterministic(self) -> None:
        ids = [f"a{i}" for i in range(8)]
        a = PartitionStrategy.random_split(ids, num_partitions=2, seed=7)
        b = PartitionStrategy.random_split(ids, num_partitions=2, seed=7)
        assert [p.agent_ids for p in a] == [p.agent_ids for p in b]

    def test_random_split_single_partition_keeps_everyone(self) -> None:
        ids = [f"a{i}" for i in range(4)]
        parts = PartitionStrategy.random_split(ids, num_partitions=1, seed=1)
        assert len(parts) == 1
        assert parts[0].agent_ids == set(ids)

    def test_isolate_leader_separates_highest_confidence(self) -> None:
        agents = [
            MockAgent("low", "X", 0.30),
            MockAgent("mid", "X", 0.60),
            MockAgent("high", "X", 0.95),
        ]
        parts = PartitionStrategy.isolate_leader(agents, threshold=1.5)
        assert len(parts) == 2
        leader_part = next(p for p in parts if p.label == "Leader-Isolated")
        majority_part = next(p for p in parts if p.label == "Majority")
        assert leader_part.agent_ids == {"high"}
        assert majority_part.agent_ids == {"low", "mid"}

    def test_isolate_byzantine_separates_attackers(self) -> None:
        agents = [
            MockAgent("h1", "X", 0.7),
            MockAgent("h2", "X", 0.6),
            MockAgent("b1", "Y", 0.9, byzantine=True),
            MockAgent("b2", "Z", 0.85, byzantine=True),
        ]
        parts = PartitionStrategy.isolate_byzantine(agents)
        labels = {p.label: p.agent_ids for p in parts}
        assert labels["Byzantine"] == {"b1", "b2"}
        assert labels["Honest"] == {"h1", "h2"}

    def test_isolate_byzantine_all_honest_returns_single_honest(self) -> None:
        agents = _honest_swarm(n=3)
        parts = PartitionStrategy.isolate_byzantine(agents)
        assert len(parts) == 1
        assert parts[0].label == "Honest"
        assert parts[0].agent_ids == {a.id for a in agents}

    def test_isolate_byzantine_all_byzantine_returns_single_byzantine(self) -> None:
        agents = [
            MockAgent("b1", "X", 0.9, byzantine=True),
            MockAgent("b2", "Y", 0.8, byzantine=True),
        ]
        parts = PartitionStrategy.isolate_byzantine(agents)
        assert len(parts) == 1
        assert parts[0].label == "Byzantine"
        assert parts[0].agent_ids == {"b1", "b2"}

    def test_minority_split_one_third_two_thirds(self) -> None:
        agents = _honest_swarm(n=6)
        parts = PartitionStrategy.minority_split(agents)
        minority = next(p for p in parts if p.label == "Minority")
        majority = next(p for p in parts if p.label == "Majority")
        assert len(minority.agent_ids) == 2  # 6 // 3
        assert len(majority.agent_ids) == 4
        assert minority.agent_ids.isdisjoint(majority.agent_ids)

    def test_minority_split_tiny_swarm_keeps_at_least_one_in_minority(self) -> None:
        agents = _honest_swarm(n=2)
        parts = PartitionStrategy.minority_split(agents)
        assert sum(len(p.agent_ids) for p in parts) == 2
        assert all(len(p.agent_ids) >= 1 for p in parts)


# --------------------------------------------------------------------------- #
# NetworkPartitionSimulator.simulate
# --------------------------------------------------------------------------- #

class TestSimulate:
    @pytest.mark.asyncio
    async def test_unanimous_within_partition_commits(self) -> None:
        # Use high confidences so each 2-agent partition clears threshold=1.5.
        agents = [MockAgent(f"a{i+1}", "X", 0.9) for i in range(4)]
        sim = NetworkPartitionSimulator(agents, threshold=1.5, max_rounds=2)
        parts = [Partition(0, {a.id for a in agents[:2]}, label="A"),
                 Partition(1, {a.id for a in agents[2:]}, label="B")]
        result = await sim.simulate(parts, task="t")

        assert isinstance(result, PartitionResult)
        # Both partitions agree on "X" -> no split brain.
        assert result.split_brain is False
        assert set(result.conflicting_solutions) == {"X"}
        # Both partitions reached a committed result (sum of confidences > 1.5).
        assert all(result.quorum_achieved.values())
        # At least the initial "split" event is logged.
        assert result.events[0].event_type == "split"
        assert len(result.events[0].partitions) == 2

    @pytest.mark.asyncio
    async def test_split_brain_detected_when_partitions_disagree(self) -> None:
        agents = [
            MockAgent("x1", "X", 0.9),
            MockAgent("x2", "X", 0.9),
            MockAgent("y1", "Y", 0.9),
            MockAgent("y2", "Y", 0.9),
        ]
        sim = NetworkPartitionSimulator(agents, threshold=1.5, max_rounds=2)
        parts = [Partition(0, {"x1", "x2"}, label="X-side"),
                 Partition(1, {"y1", "y2"}, label="Y-side")]
        result = await sim.simulate(parts, task="t")

        assert result.split_brain is True
        assert set(result.conflicting_solutions) == {"X", "Y"}
        # A "detect" event is appended whenever split_brain triggers.
        assert any(e.event_type == "detect" for e in result.events)

    @pytest.mark.asyncio
    async def test_empty_partition_records_no_quorum_and_no_result(self) -> None:
        agents = _honest_swarm(n=3)
        sim = NetworkPartitionSimulator(agents, threshold=1.5, max_rounds=1)
        parts = [
            Partition(0, {a.id for a in agents}, label="All"),
            Partition(1, {"ghost"}, label="Phantom"),  # ghost not in agent map
        ]
        result = await sim.simulate(parts, task="t")
        assert result.sub_results[1] is None
        assert result.quorum_achieved[1] is False

    @pytest.mark.asyncio
    async def test_healing_runs_full_network_consensus(self) -> None:
        agents = _honest_swarm(answer="OK", n=4)
        sim = NetworkPartitionSimulator(
            agents, threshold=1.5, max_rounds=2, heal_after=3,
        )
        parts = [Partition(0, {agents[0].id}, label="Solo"),
                 Partition(1, {a.id for a in agents[1:]}, label="Rest")]
        result = await sim.simulate(parts, task="t")
        # Healed result stored under key -1.
        assert -1 in result.sub_results
        healed = result.sub_results[-1]
        assert healed is not None and healed.committed
        assert healed.committed_solution == "OK"
        assert result.quorum_achieved[-1] is True
        # Heal event recorded.
        heal_events = [e for e in result.events if e.event_type == "heal"]
        assert heal_events and heal_events[0].round_index == 3


# --------------------------------------------------------------------------- #
# sweep_partitions + _get_partitions
# --------------------------------------------------------------------------- #

class TestSweep:
    @pytest.mark.asyncio
    async def test_sweep_runs_all_strategies_by_default(self) -> None:
        agents = _mixed_swarm()
        sim = NetworkPartitionSimulator(agents, threshold=1.5, max_rounds=2)
        results = await sim.sweep_partitions(task="t")
        names = [s for s, _ in results]
        assert names == ["random", "isolate_leader", "isolate_byzantine", "minority"]
        for _, r in results:
            assert isinstance(r, PartitionResult)
            # Every result has at least one event (the initial split).
            assert any(e.event_type == "split" for e in r.events)

    @pytest.mark.asyncio
    async def test_sweep_custom_subset(self) -> None:
        agents = _mixed_swarm()
        sim = NetworkPartitionSimulator(agents, threshold=1.5, max_rounds=2)
        results = await sim.sweep_partitions(task="t", strategies=["minority"])
        assert [s for s, _ in results] == ["minority"]

    def test_get_partitions_unknown_strategy_raises(self) -> None:
        sim = NetworkPartitionSimulator(_honest_swarm(n=3))
        with pytest.raises(ValueError, match="Unknown strategy"):
            sim._get_partitions("does-not-exist")


# --------------------------------------------------------------------------- #
# build_agents
# --------------------------------------------------------------------------- #

class TestBuildAgents:
    def test_byzantine_count_respected(self) -> None:
        agents = build_agents(num_agents=6, num_byzantine=2, seed=1)
        assert len(agents) == 6
        byz = [a for a in agents if a.byzantine]
        assert len(byz) == 2
        # Byzantine agents land at the tail.
        assert byz == agents[-2:]

    def test_deterministic_with_seed(self) -> None:
        a = build_agents(5, 1, seed=99)
        b = build_agents(5, 1, seed=99)
        assert [(x.id, x.answer, x.confidence, x.byzantine) for x in a] == \
               [(x.id, x.answer, x.confidence, x.byzantine) for x in b]

    def test_confidence_within_expected_ranges(self) -> None:
        agents = build_agents(5, 2, seed=2)
        for a in agents:
            assert 0.0 <= a.confidence <= 1.0
            if a.byzantine:
                assert 0.8 <= a.confidence <= 0.99
            else:
                assert 0.5 <= a.confidence <= 0.95


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

class TestHtmlReport:
    @pytest.mark.asyncio
    async def test_html_report_contains_strategy_and_safe_badge(self) -> None:
        agents = _honest_swarm(n=4)
        sim = NetworkPartitionSimulator(agents, threshold=1.5, max_rounds=2)
        parts = sim._get_partitions("minority")
        result = await sim.simulate(parts, task="t")
        html = generate_html_report([("minority", result)], agents)
        assert "<!DOCTYPE html>" in html
        assert "minority" in html
        assert "Safe" in html  # no split-brain -> safe badge
        assert "SPLIT-BRAIN" not in html
        # Agent fleet table renders one row per agent.
        for a in agents:
            assert a.id in html

    @pytest.mark.asyncio
    async def test_html_report_flags_split_brain(self) -> None:
        agents = [
            MockAgent("x1", "X", 0.9),
            MockAgent("x2", "X", 0.9),
            MockAgent("y1", "Y", 0.9),
            MockAgent("y2", "Y", 0.9),
        ]
        sim = NetworkPartitionSimulator(agents, threshold=1.5, max_rounds=2)
        parts = [Partition(0, {"x1", "x2"}), Partition(1, {"y1", "y2"})]
        result = await sim.simulate(parts, task="t")
        html = generate_html_report([("random", result)], agents)
        assert "SPLIT-BRAIN" in html


# --------------------------------------------------------------------------- #
# CLI smoke
# --------------------------------------------------------------------------- #

class TestCli:
    @pytest.mark.asyncio
    async def test_main_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        await main([
            "--agents", "4", "--byzantine", "1",
            "--strategy", "minority", "--threshold", "1.5",
            "--seed", "7", "--json",
        ])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list) and data
        assert data[0]["strategy"] == "minority"
        assert "partitions" in data[0]
        assert "events" in data[0]

    @pytest.mark.asyncio
    async def test_main_writes_html_report(self, tmp_path: Path) -> None:
        report = tmp_path / "report.html"
        await main([
            "--agents", "4", "--byzantine", "1",
            "--strategy", "sweep", "--threshold", "1.5",
            "--seed", "7", "--report", str(report),
        ])
        assert report.exists()
        body = report.read_text(encoding="utf-8")
        assert "mBFT Network Partition Analysis" in body
        assert "Agent Fleet" in body

    @pytest.mark.asyncio
    async def test_main_heal_after_runs_full_network(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        await main([
            "--agents", "4", "--byzantine", "0",
            "--strategy", "minority", "--heal-after", "2",
            "--threshold", "1.5", "--seed", "3",
        ])
        out = capsys.readouterr().out
        # Either healed succeeded or honestly failed; the post-heal banner must show.
        assert re.search(r"Healed network", out)
