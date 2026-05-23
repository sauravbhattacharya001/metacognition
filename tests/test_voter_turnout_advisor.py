"""Tests for VoterTurnoutAdvisor."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src.core.state import RoundResult, Vote
from src.voter_turnout_advisor import (
    VoterTurnoutAdvisor,
    to_json,
    to_markdown,
    to_text,
)


FIXED_NOW = lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)


def _round(idx, leader, committed, votes, aggregate=None, threshold=1.5, slashed=None):
    if aggregate is None:
        aggregate = sum(v.weight for v in votes)
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution="x" if committed else None,
        aggregate_weight=aggregate,
        threshold=threshold,
        votes=list(votes),
        slashed=list(slashed or []),
    )


def _v(vid, w, target="p"):
    return Vote(voter_id=vid, target_proposal_id=target, weight=w)


# ---------------------------------------------------------------------------


def test_empty_history_returns_clean_report():
    rep = VoterTurnoutAdvisor(now_fn=FIXED_NOW).analyze([])
    assert rep.portfolio.rounds_observed == 0
    assert rep.portfolio.grade == "A"
    assert rep.agents == []
    assert any(a.id == "HEALTHY_PARTICIPATION" for a in rep.playbook)


def test_full_turnout_healthy_grade():
    history = [
        _round(i, "a1", True, [_v("a1", 0.9), _v("a2", 0.9), _v("a3", 0.9)])
        for i in range(5)
    ]
    rep = VoterTurnoutAdvisor(now_fn=FIXED_NOW).analyze(history)
    assert rep.portfolio.avg_turnout == 1.0
    assert rep.portfolio.grade == "A"
    assert rep.portfolio.chronic_absentee_count == 0
    assert rep.portfolio.rounds_at_risk == 0


def test_chronic_absentee_detected():
    # a3 votes in 1/5 rounds; a1+a2 in all
    rounds = []
    for i in range(5):
        votes = [_v("a1", 0.9), _v("a2", 0.9)]
        if i == 0:
            votes.append(_v("a3", 0.9))
        rounds.append(_round(i, "a1", True, votes))
    rep = VoterTurnoutAdvisor(now_fn=FIXED_NOW).analyze(rounds)
    a3 = next(a for a in rep.agents if a.agent_id == "a3")
    assert a3.status == "CHRONIC_ABSENTEE"
    assert any(act.id.startswith("REMOVE_CHRONIC_ABSENTEE") for act in rep.playbook)


def test_low_turnout_rounds_flagged_at_risk():
    # roster: a1..a5 seeded in round 0; rounds 1,2 only 2 voters
    seed = _round(
        0,
        "a1",
        True,
        [_v(f"a{i}", 0.7) for i in range(1, 6)],
        threshold=1.5,
    )
    sparse1 = _round(1, "a1", False, [_v("a1", 0.4), _v("a2", 0.3)])
    sparse2 = _round(2, "a1", False, [_v("a1", 0.4), _v("a2", 0.3)])
    rep = VoterTurnoutAdvisor(min_acceptable_turnout=0.6, now_fn=FIXED_NOW).analyze(
        [seed, sparse1, sparse2]
    )
    assert rep.portfolio.rounds_at_risk >= 2
    assert rep.portfolio.grade in {"D", "F", "C"}
    assert any(act.id == "RECRUIT_QUORUM_BACKUPS" for act in rep.playbook)


def test_phantom_dissent_commit_detected():
    # a4 is a known rejector (history of negative votes), then absent on a commit
    seed = _round(
        0,
        "a1",
        False,
        [
            _v("a1", 0.4),
            _v("a2", 0.4),
            _v("a4", -0.8),
        ],
        threshold=1.5,
    )
    seed2 = _round(
        1,
        "a1",
        False,
        [
            _v("a1", 0.4),
            _v("a2", 0.4),
            _v("a4", -0.8),
        ],
        threshold=1.5,
    )
    # Now a4 is absent and round just barely commits.
    phantom = _round(
        2,
        "a2",
        True,
        [_v("a1", 0.9), _v("a2", 0.9)],
        aggregate=1.8,
        threshold=1.5,
    )
    rep = VoterTurnoutAdvisor(now_fn=FIXED_NOW).analyze([seed, seed2, phantom])
    phantom_round = next(r for r in rep.rounds if r.round_index == 2)
    assert phantom_round.only_committed_because_absent_dissent is True
    assert phantom_round.priority == "P0"
    assert any(act.id == "ROLLBACK_PHANTOM_COMMITS" for act in rep.playbook)
    assert rep.portfolio.grade in {"D", "F"}


def test_decaying_voter_detected():
    # a3 votes in first 2 rounds, then nothing for 4 rounds
    rounds = []
    for i in range(6):
        votes = [_v("a1", 0.9), _v("a2", 0.9)]
        if i < 2:
            votes.append(_v("a3", 0.9))
        rounds.append(_round(i, "a1", True, votes))
    rep = VoterTurnoutAdvisor(now_fn=FIXED_NOW).analyze(rounds)
    a3 = next(a for a in rep.agents if a.agent_id == "a3")
    # 2 voted / 6 eligible = 0.33 absentee_rate=0.67 -> CHRONIC; tail_inactive=4
    # either CHRONIC_ABSENTEE wins over DECAYING which is fine
    assert a3.status in {"DECAYING", "CHRONIC_ABSENTEE"}
    assert a3.rounds_inactive_tail == 4


def test_emerged_voter_detected():
    rounds = []
    for i in range(6):
        votes = [_v("a1", 0.9), _v("a2", 0.9)]
        if i >= 5:  # a3 joins only at the last round
            votes.append(_v("a3", 0.9))
        rounds.append(_round(i, "a1", True, votes))
    rep = VoterTurnoutAdvisor(now_fn=FIXED_NOW).analyze(rounds)
    a3 = next(a for a in rep.agents if a.agent_id == "a3")
    assert a3.status == "EMERGED"
    # rounds before a3 joined should not count them as absent
    r0 = next(r for r in rep.rounds if r.round_index == 0)
    assert "a3" not in r0.absent_agents


def test_cautious_appetite_raises_quorum_floor():
    seed = _round(
        0, "a1", True, [_v(f"a{i}", 0.7) for i in range(1, 6)], threshold=1.5
    )
    mid = _round(1, "a1", True, [_v("a1", 0.6), _v("a2", 0.6), _v("a3", 0.6)])
    history = [seed, mid]
    rep_balanced = VoterTurnoutAdvisor(
        min_acceptable_turnout=0.6, risk_appetite="balanced", now_fn=FIXED_NOW
    ).analyze(history)
    rep_cautious = VoterTurnoutAdvisor(
        min_acceptable_turnout=0.6, risk_appetite="cautious", now_fn=FIXED_NOW
    ).analyze(history)
    assert rep_cautious.portfolio.turnout_score <= rep_balanced.portfolio.turnout_score


def test_aggressive_trims_p3_when_actions_present():
    # Phantom-dissent scenario produces P0 actions; aggressive should drop P3 fallback
    seed = _round(0, "a1", False, [_v("a1", 0.3), _v("a2", 0.3), _v("a4", -0.8)])
    phantom = _round(1, "a2", True, [_v("a1", 0.9), _v("a2", 0.9)], aggregate=1.8)
    rep = VoterTurnoutAdvisor(risk_appetite="aggressive", now_fn=FIXED_NOW).analyze(
        [seed, phantom]
    )
    assert not any(a.priority == "P3" for a in rep.playbook)


def test_insights_chronic_cluster():
    rounds = []
    for i in range(6):
        votes = [_v("a1", 0.9), _v("a2", 0.9)]
        if i == 0:
            votes.extend([_v("a3", 0.9), _v("a4", 0.9), _v("a5", 0.9)])
        rounds.append(_round(i, "a1", True, votes))
    rep = VoterTurnoutAdvisor(now_fn=FIXED_NOW).analyze(rounds)
    assert "CHRONIC_ABSENTEE_CLUSTER" in rep.insights
    assert any(act.id == "REMOVE_CHRONIC_ABSENTEE_CLUSTER" for act in rep.playbook)


def test_render_text_markdown_json():
    rounds = [
        _round(i, "a1", True, [_v("a1", 0.9), _v("a2", 0.9), _v("a3", 0.9)])
        for i in range(3)
    ]
    rep = VoterTurnoutAdvisor(now_fn=FIXED_NOW).analyze(rounds)
    txt = to_text(rep)
    md = to_markdown(rep)
    js = to_json(rep)
    assert "VERDICT:" in txt
    assert "## Summary" in md
    assert "## Per-round turnout" in md
    parsed = json.loads(js)
    assert "portfolio" in parsed
    assert parsed["portfolio"]["grade"] == "A"


def test_json_byte_stable():
    rounds = [
        _round(i, "a1", True, [_v("a1", 0.9), _v("a2", 0.9)]) for i in range(3)
    ]
    advisor = VoterTurnoutAdvisor(now_fn=FIXED_NOW)
    a = to_json(advisor.analyze(rounds))
    b = to_json(advisor.analyze(rounds))
    assert a == b


def test_never_mutates_inputs():
    rounds = [
        _round(i, "a1", True, [_v("a1", 0.9), _v("a2", 0.9)]) for i in range(3)
    ]
    snapshot = [r.model_dump() for r in rounds]
    VoterTurnoutAdvisor(now_fn=FIXED_NOW).analyze(rounds)
    after = [r.model_dump() for r in rounds]
    assert snapshot == after


def test_invalid_risk_appetite_raises():
    import pytest

    with pytest.raises(ValueError):
        VoterTurnoutAdvisor(risk_appetite="reckless")


def test_priority_sort_p0_first():
    seed = _round(0, "a1", False, [_v("a1", 0.3), _v("a2", 0.3), _v("a4", -0.8)])
    phantom = _round(1, "a2", True, [_v("a1", 0.9), _v("a2", 0.9)], aggregate=1.8)
    rep = VoterTurnoutAdvisor(now_fn=FIXED_NOW).analyze([seed, phantom])
    priorities = [a.priority for a in rep.playbook]
    rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    assert priorities == sorted(priorities, key=lambda p: rank[p])
