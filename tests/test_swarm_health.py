"""Tests for the SwarmHealthMonitor agentic observability layer."""
from __future__ import annotations

import json

import pytest

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine
from src.core.state import RoundResult, Vote
from src.swarm_health import (
    SwarmHealthMonitor,
)


def _make_round(
    *,
    round_index: int,
    leader_id: str,
    solution: str | None,
    aggregate: float,
    threshold: float,
    votes: list[Vote],
    slashed: list[str] | None = None,
) -> RoundResult:
    return RoundResult(
        round_index=round_index,
        leader_id=leader_id,
        committed_solution=solution,
        aggregate_weight=aggregate,
        threshold=threshold,
        votes=votes,
        slashed=slashed or [],
    )


def _vote(voter: str, weight: float, target: str = "p1") -> Vote:
    return Vote(voter_id=voter, target_proposal_id=target, weight=weight)


@pytest.mark.asyncio
async def test_monitor_summarises_real_engine_run() -> None:
    agents = [
        MockAgent("a1", "X", 0.9),
        MockAgent("a2", "X", 0.8),
        MockAgent("a3", "X", 0.7),
    ]
    engine = MBFTEngine(agents, threshold=1.5)
    await engine.run("task")

    report = SwarmHealthMonitor().analyze(
        history=engine.history,
        reputation=engine.reputation,
        threshold=engine.threshold,
        agent_ids=[a.id for a in engine.agents],
    )

    assert report.rounds_observed == 1
    assert report.rounds_committed == 1
    assert report.commit_rate == pytest.approx(1.0)
    assert report.avg_margin_to_threshold > 0
    assert {a.agent_id for a in report.agents} == {"a1", "a2", "a3"}
    # Leader was a1; only followers cast votes.
    leader = next(a for a in report.agents if a.agent_id == "a1")
    assert leader.leader_rounds == 1
    assert leader.leader_commits == 1
    assert leader.leader_success_rate == pytest.approx(1.0)


def test_calibration_neutral_when_no_commits() -> None:
    monitor = SwarmHealthMonitor()
    rnd = _make_round(
        round_index=0,
        leader_id="a1",
        solution=None,
        aggregate=0.4,
        threshold=1.5,
        votes=[_vote("a2", 0.3)],
        slashed=["a1"],
    )
    report = monitor.analyze(
        history=[rnd],
        reputation={"a1": 0.5, "a2": 1.0},
        threshold=1.5,
    )
    a2 = next(a for a in report.agents if a.agent_id == "a2")
    assert a2.calibration_score == pytest.approx(0.5)


def test_persistent_dissenter_flagged_as_suspect() -> None:
    rounds = []
    for i in range(4):
        rounds.append(
            _make_round(
                round_index=i,
                leader_id="a1",
                solution="X",
                aggregate=2.0,
                threshold=1.5,
                votes=[
                    _vote("a2", 0.9),
                    _vote("a3", -0.95),
                ],
            )
        )
    report = SwarmHealthMonitor().analyze(
        history=rounds,
        reputation={"a1": 1.0, "a2": 1.0, "a3": 1.0},
        threshold=1.5,
    )

    a3 = next(a for a in report.agents if a.agent_id == "a3")
    assert a3.status == "suspect"
    assert a3.rejections_cast == 4
    assert a3.calibration_score == pytest.approx(0.0)

    flagged = [r for r in report.recommendations if r.target_agent == "a3"]
    assert flagged, "expected an investigate recommendation for a3"
    assert flagged[0].kind == "investigate"
    assert flagged[0].severity == "critical"


def test_threshold_raised_when_commits_too_easy() -> None:
    rounds = [
        _make_round(
            round_index=i,
            leader_id="a1",
            solution="X",
            aggregate=3.0,  # margin = 1.5 vs threshold 1.5
            threshold=1.5,
            votes=[_vote("a2", 0.9)],
        )
        for i in range(5)
    ]
    report = SwarmHealthMonitor().analyze(
        history=rounds,
        reputation={"a1": 1.0, "a2": 1.0},
        threshold=1.5,
    )
    threshold_recs = [r for r in report.recommendations if r.kind == "threshold"]
    assert any(
        r.suggested_value and r.suggested_value > 1.5 for r in threshold_recs
    ), threshold_recs


def test_threshold_lowered_when_commits_rare() -> None:
    rounds = [
        _make_round(
            round_index=i,
            leader_id="a1",
            solution=None,
            aggregate=0.3,
            threshold=1.5,
            votes=[_vote("a2", -0.4)],
            slashed=["a1"],
        )
        for i in range(3)
    ]
    report = SwarmHealthMonitor().analyze(
        history=rounds,
        reputation={"a1": 0.125, "a2": 1.0},
        threshold=1.5,
        slash_factor=0.5,
    )
    threshold_recs = [r for r in report.recommendations if r.kind == "threshold"]
    assert any(
        r.suggested_value and r.suggested_value < 1.5 for r in threshold_recs
    ), threshold_recs


def test_softer_slash_factor_suggested_when_leaders_collapse() -> None:
    rounds = [
        _make_round(
            round_index=i,
            leader_id=f"a{i % 2 + 1}",
            solution=None,
            aggregate=0.5,
            threshold=1.5,
            votes=[_vote("a3", -0.5)],
            slashed=[f"a{i % 2 + 1}"],
        )
        for i in range(4)
    ]
    report = SwarmHealthMonitor().analyze(
        history=rounds,
        reputation={"a1": 0.0625, "a2": 0.0625, "a3": 1.0},
        threshold=1.5,
        slash_factor=0.5,
    )
    slash_recs = [r for r in report.recommendations if r.kind == "slash_factor"]
    assert slash_recs, "expected a slash_factor recommendation"
    assert slash_recs[0].suggested_value is not None
    assert slash_recs[0].suggested_value < 0.5


def test_collapsed_reputation_marks_agent_slashed_out() -> None:
    rnd = _make_round(
        round_index=0,
        leader_id="a2",
        solution="Y",
        aggregate=1.8,
        threshold=1.5,
        votes=[_vote("a3", 0.9)],
    )
    report = SwarmHealthMonitor().analyze(
        history=[rnd],
        reputation={"a1": 0.01, "a2": 1.0, "a3": 1.0},
        threshold=1.5,
    )
    a1 = next(a for a in report.agents if a.agent_id == "a1")
    assert a1.status == "slashed_out"
    targeted = [r for r in report.recommendations if r.target_agent == "a1"]
    assert targeted and targeted[0].kind == "investigate"


def test_exports_render_without_error() -> None:
    rnd = _make_round(
        round_index=0,
        leader_id="a1",
        solution="X",
        aggregate=2.0,
        threshold=1.5,
        votes=[_vote("a2", 0.8)],
    )
    report = SwarmHealthMonitor().analyze(
        history=[rnd],
        reputation={"a1": 1.0, "a2": 1.0},
        threshold=1.5,
    )

    md = report.to_markdown()
    assert "# Swarm Health Report" in md
    assert "a1" in md and "a2" in md

    text = report.to_text()
    assert "SWARM HEALTH REPORT" in text

    csv = report.to_csv()
    assert csv.splitlines()[0].startswith("agent_id,")
    # Header + 2 agent rows
    assert len(csv.strip().splitlines()) == 3

    parsed = json.loads(report.to_json())
    assert parsed["rounds_observed"] == 1
    assert parsed["threshold"] == 1.5


def test_invalid_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        SwarmHealthMonitor().analyze(
            history=[],
            reputation={"a1": 1.0},
            threshold=0.0,
        )


def test_empty_history_returns_neutral_report_with_advice() -> None:
    report = SwarmHealthMonitor().analyze(
        history=[],
        reputation={"a1": 1.0, "a2": 1.0},
        threshold=1.5,
    )
    assert report.rounds_observed == 0
    assert report.commit_rate == 0.0
    assert report.recommendations
    assert report.recommendations[0].kind == "none"
