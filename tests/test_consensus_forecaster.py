"""Tests for :mod:`src.consensus_forecaster`."""
from __future__ import annotations

import json

import pytest

from src.consensus_forecaster import (
    AgentForecast,
    ConsensusForecast,
    ConsensusForecaster,
    Intervention,
)
from src.core.state import RoundResult, Vote


def _round(
    idx: int,
    leader: str,
    committed: bool,
    aggregate: float,
    threshold: float,
    votes: list[Vote],
    slashed: list[str] | None = None,
) -> RoundResult:
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution="ok" if committed else None,
        aggregate_weight=aggregate,
        threshold=threshold,
        votes=votes,
        slashed=list(slashed or []),
    )


def _vote(voter: str, target: str, weight: float) -> Vote:
    return Vote(voter_id=voter, target_proposal_id=target, weight=weight)


def test_empty_history_returns_neutral_forecast():
    f = ConsensusForecaster().forecast(
        history=[],
        reputation={"a": 1.0, "b": 1.0, "c": 1.0},
        threshold=1.5,
        slash_factor=0.5,
        agent_ids=["a", "b", "c"],
    )
    assert f.rounds_observed == 0
    assert f.predicted_leader_id in {"a", "b", "c"}
    assert 0.0 <= f.p_commit <= 1.0
    assert any("No prior rounds" in n for n in f.notes)


def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        ConsensusForecaster().forecast(
            history=[],
            reputation={"a": 1.0},
            threshold=0.0,
            slash_factor=0.5,
        )


def test_leader_probabilities_sum_to_one():
    f = ConsensusForecaster().forecast(
        history=[],
        reputation={"a": 1.0, "b": 1.0, "c": 1.0},
        threshold=1.5,
        slash_factor=0.5,
    )
    total = sum(a.p_leader for a in f.leaders_ranked)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_higher_reputation_predicted_as_leader():
    f = ConsensusForecaster().forecast(
        history=[],
        reputation={"a": 1.0, "b": 0.3, "c": 0.2},
        threshold=1.5,
        slash_factor=0.5,
        agent_ids=["a", "b", "c"],
    )
    assert f.predicted_leader_id == "a"
    a = next(x for x in f.leaders_ranked if x.agent_id == "a")
    b = next(x for x in f.leaders_ranked if x.agent_id == "b")
    assert a.p_leader > b.p_leader


def test_chronic_rejector_triggers_p0_pause():
    # b1 rejects every time; the forecaster should P0-pause it.
    # a1 is the consistent (slightly higher-reputation) leader so the
    # P0 target is unambiguously b1, not the predicted leader.
    votes = [_vote("b1", "p", -0.9), _vote("c1", "p", 0.8)]
    history = [
        _round(0, "a1", False, 0.8, 1.5, votes),
        _round(1, "a1", False, 0.7, 1.5, votes),
        _round(2, "a1", False, 0.7, 1.5, votes),
    ]
    f = ConsensusForecaster().forecast(
        history=history,
        reputation={"a1": 1.5, "b1": 1.0, "c1": 1.0},
        threshold=1.5,
        slash_factor=0.5,
        agent_ids=["a1", "b1", "c1"],
    )
    assert f.predicted_leader_id == "a1"
    pauses = [iv for iv in f.interventions if iv.kind == "pause_agent"]
    assert any(iv.target_agent == "b1" and iv.priority == "P0" for iv in pauses)
    assert f.p_unrefuted_rejection > 0.5


def test_near_threshold_triggers_threshold_recommendation():
    # Everyone agrees; aggregate hovers just under threshold.
    # a1 carries most of the aggregate so it is unambiguously the predicted leader.
    votes = [_vote("b1", "p", 0.3), _vote("c1", "p", 0.3)]
    history = [
        _round(0, "a1", False, 1.4, 1.5, votes),
        _round(1, "a1", False, 1.4, 1.5, votes),
        _round(2, "a1", False, 1.4, 1.5, votes),
    ]
    f = ConsensusForecaster().forecast(
        history=history,
        reputation={"a1": 1.5, "b1": 1.0, "c1": 1.0},
        threshold=1.5,
        slash_factor=0.5,
        agent_ids=["a1", "b1", "c1"],
    )
    assert f.predicted_leader_id == "a1"
    thresh_recs = [iv for iv in f.interventions if iv.kind == "threshold"]
    assert thresh_recs, "expected a threshold intervention"
    assert thresh_recs[0].suggested_value is not None
    assert thresh_recs[0].suggested_value < 1.5


def test_healthy_swarm_returns_on_track_message():
    votes = [_vote("b1", "p", 0.6), _vote("c1", "p", 0.6), _vote("d1", "p", 0.6)]
    history = [
        _round(0, "a1", True, 2.6, 1.5, votes),
        _round(1, "a1", True, 2.6, 1.5, votes),
        _round(2, "a1", True, 2.6, 1.5, votes),
    ]
    f = ConsensusForecaster().forecast(
        history=history,
        reputation={"a1": 1.0, "b1": 1.0, "c1": 1.0, "d1": 1.0},
        threshold=1.5,
        slash_factor=0.5,
        agent_ids=["a1", "b1", "c1", "d1"],
    )
    assert f.p_commit > 0.6
    kinds = {iv.kind for iv in f.interventions}
    assert "none" in kinds
    assert all(iv.priority in {"P2", "P3"} for iv in f.interventions)


def test_renderers_return_nonempty_strings():
    f = ConsensusForecaster().forecast(
        history=[],
        reputation={"a": 1.0, "b": 1.0},
        threshold=1.5,
        slash_factor=0.5,
    )
    assert "CONSENSUS FORECAST" in f.to_text()
    md = f.to_markdown()
    assert md.startswith("# Consensus Forecast")
    payload = json.loads(f.to_json())
    assert payload["threshold"] == 1.5
    assert "leaders_ranked" in payload


def test_aggregate_weight_range_brackets_point_estimate():
    f = ConsensusForecaster().forecast(
        history=[],
        reputation={"a": 1.0, "b": 1.0, "c": 1.0},
        threshold=1.5,
        slash_factor=0.5,
    )
    assert f.aggregate_weight_low <= f.predicted_aggregate_weight <= f.aggregate_weight_high


def test_collapsed_reputation_noted():
    f = ConsensusForecaster().forecast(
        history=[_round(0, "a", False, 0.1, 1.5, [])],
        reputation={"a": 0.05, "b": 1.0, "c": 1.0},
        threshold=1.5,
        slash_factor=0.5,
        agent_ids=["a", "b", "c"],
    )
    assert any("collapsed reputation" in n for n in f.notes)


def test_failure_streak_noted():
    history = [
        _round(0, "a", False, 0.1, 1.5, []),
        _round(1, "a", False, 0.1, 1.5, []),
        _round(2, "a", False, 0.1, 1.5, []),
    ]
    f = ConsensusForecaster().forecast(
        history=history,
        reputation={"a": 1.0, "b": 1.0},
        threshold=1.5,
        slash_factor=0.5,
        agent_ids=["a", "b"],
    )
    assert any("failed to commit" in n for n in f.notes)


def test_determinism_same_inputs_same_output():
    args = dict(
        history=[
            _round(
                0,
                "a",
                True,
                2.0,
                1.5,
                [_vote("b", "p", 0.5), _vote("c", "p", 0.5)],
            )
        ],
        reputation={"a": 1.0, "b": 1.0, "c": 1.0},
        threshold=1.5,
        slash_factor=0.5,
        agent_ids=["a", "b", "c"],
    )
    f1 = ConsensusForecaster().forecast(**args)
    f2 = ConsensusForecaster().forecast(**args)
    assert f1.to_json() == f2.to_json()
