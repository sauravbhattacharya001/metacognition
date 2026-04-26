"""Behavioural tests for the mBFT engine."""
from __future__ import annotations

import pytest

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine


@pytest.mark.asyncio
async def test_unanimous_swarm_commits_first_round() -> None:
    agents = [
        MockAgent("a1", "X", 0.9),
        MockAgent("a2", "X", 0.8),
        MockAgent("a3", "X", 0.7),
    ]
    engine = MBFTEngine(agents, threshold=1.5)
    result = await engine.run("task")

    assert result is not None
    assert result.committed
    assert result.committed_solution == "X"
    assert result.round_index == 0
    assert result.leader_id == "a1"


@pytest.mark.asyncio
async def test_counter_proof_blocks_commit_and_triggers_view_change() -> None:
    agents = [
        MockAgent("a1", "X", 0.95),
        MockAgent("a2", "Y", 0.90),
        MockAgent("a3", "Y", 0.85),
    ]
    engine = MBFTEngine(agents, threshold=1.5, max_rounds=3, slash_factor=0.1)
    result = await engine.run("task")

    assert result is not None
    assert engine.history[0].committed is False
    assert "a1" in engine.history[0].slashed
    assert result.committed
    assert result.committed_solution == "Y"
    assert result.leader_id == "a2"


@pytest.mark.asyncio
async def test_byzantine_minority_cannot_force_bad_commit() -> None:
    agents = [
        MockAgent("a1", "WRONG", 0.99, byzantine=True),
        MockAgent("a2", "RIGHT", 0.80),
        MockAgent("a3", "RIGHT", 0.75),
        MockAgent("a4", "RIGHT", 0.70),
    ]
    engine = MBFTEngine(agents, threshold=1.5, max_rounds=4, slash_factor=0.05)
    result = await engine.run("task")

    assert result is not None
    assert result.committed
    assert result.committed_solution == "RIGHT"
    assert engine.reputation["a1"] < 1.0


@pytest.mark.asyncio
async def test_weight_below_threshold_does_not_commit() -> None:
    agents = [
        MockAgent("a1", "X", 0.30),
        MockAgent("a2", "X", 0.20),
    ]
    engine = MBFTEngine(agents, threshold=2.0, max_rounds=1)
    result = await engine.run("task")

    assert result is not None
    assert not result.committed
    assert result.aggregate_weight < result.threshold


def test_engine_requires_agents() -> None:
    with pytest.raises(ValueError):
        MBFTEngine(agents=[], threshold=1.0)
