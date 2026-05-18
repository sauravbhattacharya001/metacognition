"""Tests for LeaderRotationAdvisor."""
from __future__ import annotations

import copy
from datetime import datetime, timezone

from src.core.state import RoundResult, Vote
from src.leader_rotation_advisor import (
    LeaderRotationAdvisor,
    LeaderRotationReport,
)


FIXED_NOW = lambda: datetime(2026, 5, 17, tzinfo=timezone.utc)


def _round(
    idx: int,
    leader: str,
    committed: bool,
    aggregate: float,
    threshold: float,
    votes,
    slashed=None,
) -> RoundResult:
    if slashed is None:
        slashed = [] if committed else [leader]
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution="42" if committed else None,
        aggregate_weight=aggregate,
        threshold=threshold,
        votes=list(votes),
        slashed=list(slashed),
    )


def _vote(voter: str, weight: float, counter=None) -> Vote:
    return Vote(
        voter_id=voter,
        target_proposal_id="p1",
        weight=weight,
        counter_proof=counter,
    )


def _healthy_history():
    """3 successful rounds, alice/bob/carol all rotate as leaders."""
    return [
        _round(0, "alice", True, 2.5, 1.0, [_vote("bob", 0.8), _vote("carol", 0.9)]),
        _round(1, "bob", True, 2.4, 1.0, [_vote("alice", 0.9), _vote("carol", 0.8)]),
        _round(2, "carol", True, 2.6, 1.0, [_vote("alice", 0.9), _vote("bob", 0.85)]),
    ]


# ---------------------------------------------------------------------------
# 1. empty history
# ---------------------------------------------------------------------------


def test_empty_history_insufficient_history_insight():
    advisor = LeaderRotationAdvisor(horizon=3, now_fn=FIXED_NOW)
    rep = advisor.recommend(history=[], reputation={}, agents=[])
    assert isinstance(rep, LeaderRotationReport)
    assert rep.rounds_analyzed == 0
    assert any(i.startswith("INSUFFICIENT_HISTORY") for i in rep.insights)
    # Headline still produced
    assert rep.summary_headline
    assert rep.horizon == 3


# ---------------------------------------------------------------------------
# 2. healthy 3-agent rotation -> grade A and diverse queue
# ---------------------------------------------------------------------------


def test_healthy_rotation_grade_high_and_diverse_picks():
    advisor = LeaderRotationAdvisor(horizon=3, now_fn=FIXED_NOW)
    history = _healthy_history()
    rep = advisor.recommend(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        agents=["alice", "bob", "carol"],
    )
    assert rep.overall_grade in ("A", "B")
    assert len(rep.rotation_queue) == 3
    picked = {s.agent_id for s in rep.rotation_queue}
    # Three distinct picks under diversity dampener.
    assert len(picked) == 3
    assert rep.leader_capture_agent is None


# ---------------------------------------------------------------------------
# 3. leader capture detected -> BREAK_LEADER_CAPTURE playbook
# ---------------------------------------------------------------------------


def test_leader_capture_triggers_break_playbook():
    # alice leads 4 of last 5 rounds
    history = [
        _round(0, "alice", True, 2.5, 1.0, [_vote("bob", 0.8), _vote("carol", 0.9)]),
        _round(1, "alice", True, 2.5, 1.0, [_vote("bob", 0.8), _vote("carol", 0.9)]),
        _round(2, "bob", True, 2.5, 1.0, [_vote("alice", 0.8), _vote("carol", 0.9)]),
        _round(3, "alice", True, 2.5, 1.0, [_vote("bob", 0.8), _vote("carol", 0.9)]),
        _round(4, "alice", True, 2.5, 1.0, [_vote("bob", 0.8), _vote("carol", 0.9)]),
    ]
    advisor = LeaderRotationAdvisor(horizon=4, now_fn=FIXED_NOW)
    rep = advisor.recommend(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        agents=["alice", "bob", "carol"],
    )
    assert rep.leader_capture_agent == "alice"
    patterns = [p.pattern for p in rep.playbook]
    assert "BREAK_LEADER_CAPTURE" in patterns
    # In the first half of the queue alice should be dampened.
    first_half = rep.rotation_queue[: max(1, rep.horizon // 2)]
    assert all(s.agent_id != "alice" for s in first_half)


# ---------------------------------------------------------------------------
# 4. all agents slashed -> EMERGENCY_BACKUP_LEADER + grade F
# ---------------------------------------------------------------------------


def test_all_slashed_emergency_backup_and_grade_f():
    # Five failed rounds; every leader gets slashed; voters never agree
    history = [
        _round(
            i,
            leader,
            False,
            0.3,
            1.0,
            [_vote(other, -0.9, counter="bad") for other in others],
            slashed=[leader],
        )
        for i, (leader, others) in enumerate(
            [
                ("alice", ["bob", "carol"]),
                ("bob", ["alice", "carol"]),
                ("carol", ["alice", "bob"]),
                ("alice", ["bob", "carol"]),
                ("bob", ["alice", "carol"]),
            ]
        )
    ]
    advisor = LeaderRotationAdvisor(horizon=3, now_fn=FIXED_NOW)
    rep = advisor.recommend(
        history=history,
        reputation={"alice": 0.1, "bob": 0.1, "carol": 0.1},
        agents=["alice", "bob", "carol"],
    )
    patterns = [p.pattern for p in rep.playbook]
    assert "EMERGENCY_BACKUP_LEADER" in patterns
    assert rep.overall_grade == "F"


# ---------------------------------------------------------------------------
# 5. chronic blocker -> ROTATE_OUT_BLOCKER targeted at right agent
# ---------------------------------------------------------------------------


def test_chronic_blocker_rotate_out():
    # bob vetoes (with full weight) several rounds where the aggregate met
    # the threshold but committed=False (unrefuted blocker).
    history = [
        _round(
            i,
            "alice",
            False,
            1.5,
            1.0,
            [_vote("bob", -1.0, counter="x"), _vote("carol", 0.9)],
            slashed=[],
        )
        for i in range(3)
    ] + [
        _round(3, "carol", True, 2.5, 1.0, [_vote("alice", 0.8), _vote("bob", 0.5)]),
    ]
    advisor = LeaderRotationAdvisor(horizon=3, now_fn=FIXED_NOW)
    rep = advisor.recommend(
        history=history,
        reputation={"alice": 0.9, "bob": 0.8, "carol": 1.0},
        agents=["alice", "bob", "carol"],
    )
    rotate_outs = [p for p in rep.playbook if p.pattern == "ROTATE_OUT_BLOCKER"]
    assert any("bob" in p.targets for p in rotate_outs)


# ---------------------------------------------------------------------------
# 6. rising star -> PROMOTE_RISING_STAR in playbook when not in top-3
# ---------------------------------------------------------------------------


def test_rising_star_promote_playbook():
    # dave never led but has perfect calibration across many votes.
    # alice/bob/carol carry leadership.
    history = [
        _round(
            i,
            leader,
            True,
            3.0,
            1.0,
            [
                _vote("dave", 0.95),
                _vote(co, 0.8),
            ],
        )
        for i, (leader, co) in enumerate(
            [
                ("alice", "bob"),
                ("bob", "carol"),
                ("alice", "carol"),
                ("carol", "bob"),
                ("alice", "bob"),
            ]
        )
    ]
    advisor = LeaderRotationAdvisor(horizon=2, now_fn=FIXED_NOW)
    rep = advisor.recommend(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0, "dave": 0.9},
        agents=["alice", "bob", "carol", "dave"],
    )
    dave = next(a for a in rep.agents if a.agent_id == "dave")
    assert "RISING_STAR" in dave.reasons
    # horizon=2, so top-3 in queue is just 2 slots; if dave not in queue,
    # promote playbook should fire.
    queue_ids = {s.agent_id for s in rep.rotation_queue}
    if "dave" not in queue_ids:
        assert any(
            p.pattern == "PROMOTE_RISING_STAR" and "dave" in p.targets
            for p in rep.playbook
        )


# ---------------------------------------------------------------------------
# 7. risk_appetite monotonicity: aggressive favors fresh blood vs cautious
# ---------------------------------------------------------------------------


def test_risk_appetite_changes_fresh_blood_scoring():
    # alice has led tons; dave is fresh but well-calibrated.
    history = [
        _round(
            i,
            "alice",
            True,
            3.0,
            1.0,
            [_vote("bob", 0.9), _vote("dave", 0.9)],
        )
        for i in range(5)
    ]
    rep_repo = {"alice": 1.0, "bob": 0.9, "dave": 0.8}
    cautious = LeaderRotationAdvisor(
        horizon=1, risk_appetite="cautious", now_fn=FIXED_NOW
    ).recommend(
        history=history, reputation=rep_repo, agents=["alice", "bob", "dave"]
    )
    aggressive = LeaderRotationAdvisor(
        horizon=1, risk_appetite="aggressive", now_fn=FIXED_NOW
    ).recommend(
        history=history, reputation=rep_repo, agents=["alice", "bob", "dave"]
    )
    dave_c = next(a for a in cautious.agents if a.agent_id == "dave")
    dave_a = next(a for a in aggressive.agents if a.agent_id == "dave")
    # Aggressive should not punish dave less or equally compared to cautious.
    assert dave_a.lead_fitness >= dave_c.lead_fitness


# ---------------------------------------------------------------------------
# 8. text renderer non-empty and contains header
# ---------------------------------------------------------------------------


def test_text_renderer_contains_header():
    advisor = LeaderRotationAdvisor(horizon=2, now_fn=FIXED_NOW)
    rep = advisor.recommend(
        history=_healthy_history(),
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        agents=["alice", "bob", "carol"],
    )
    txt = rep.to_text()
    assert "Leader Rotation Advisor" in txt
    assert "Rotation queue" in txt


# ---------------------------------------------------------------------------
# 9. markdown renderer contains slot header
# ---------------------------------------------------------------------------


def test_markdown_renderer_contains_slot_header():
    advisor = LeaderRotationAdvisor(horizon=2, now_fn=FIXED_NOW)
    rep = advisor.recommend(
        history=_healthy_history(),
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        agents=["alice", "bob", "carol"],
    )
    md = rep.to_markdown()
    assert "| slot |" in md
    assert "## Rotation queue" in md


# ---------------------------------------------------------------------------
# 10. JSON renderer byte-deterministic
# ---------------------------------------------------------------------------


def test_json_renderer_deterministic():
    advisor = LeaderRotationAdvisor(horizon=3, now_fn=FIXED_NOW)
    history = _healthy_history()
    rep1 = advisor.recommend(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        agents=["alice", "bob", "carol"],
    )
    rep2 = advisor.recommend(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        agents=["alice", "bob", "carol"],
    )
    assert rep1.to_json() == rep2.to_json()


# ---------------------------------------------------------------------------
# 11. horizon=1 returns exactly 1 slot
# ---------------------------------------------------------------------------


def test_horizon_one_returns_one_slot():
    advisor = LeaderRotationAdvisor(horizon=1, now_fn=FIXED_NOW)
    rep = advisor.recommend(
        history=_healthy_history(),
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        agents=["alice", "bob", "carol"],
    )
    assert len(rep.rotation_queue) == 1
    assert rep.rotation_queue[0].slot_index == 0


# ---------------------------------------------------------------------------
# 12. horizon larger than roster -> queue still length=horizon
# ---------------------------------------------------------------------------


def test_horizon_larger_than_roster_reuses_with_penalty():
    advisor = LeaderRotationAdvisor(horizon=6, now_fn=FIXED_NOW)
    rep = advisor.recommend(
        history=_healthy_history(),
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        agents=["alice", "bob", "carol"],
    )
    assert len(rep.rotation_queue) == 6
    # No two consecutive picks are identical (diversity dampener works).
    for i in range(1, len(rep.rotation_queue)):
        assert (
            rep.rotation_queue[i].agent_id
            != rep.rotation_queue[i - 1].agent_id
        )


# ---------------------------------------------------------------------------
# 13. INSUFFICIENT_DATA verdict for under-observed agent
# ---------------------------------------------------------------------------


def test_insufficient_data_verdict_for_unobserved_agent():
    # eve is in the roster but never appears in history or reputation
    advisor = LeaderRotationAdvisor(
        horizon=3, min_observations=3, now_fn=FIXED_NOW
    )
    rep = advisor.recommend(
        history=_healthy_history(),
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        agents=["alice", "bob", "carol", "eve"],
    )
    eve = next(a for a in rep.agents if a.agent_id == "eve")
    assert eve.verdict == "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# 14. advisor never mutates inputs
# ---------------------------------------------------------------------------


def test_advisor_never_mutates_inputs():
    history = _healthy_history()
    reputation = {"alice": 1.0, "bob": 1.0, "carol": 1.0}
    agents = ["alice", "bob", "carol"]

    history_snap = copy.deepcopy(history)
    reputation_snap = copy.deepcopy(reputation)
    agents_snap = list(agents)

    advisor = LeaderRotationAdvisor(horizon=4, now_fn=FIXED_NOW)
    advisor.recommend(history=history, reputation=reputation, agents=agents)

    assert [r.model_dump() for r in history] == [
        r.model_dump() for r in history_snap
    ]
    assert reputation == reputation_snap
    assert agents == agents_snap


# ---------------------------------------------------------------------------
# Bonus: invalid risk raises
# ---------------------------------------------------------------------------


def test_invalid_risk_appetite_raises():
    import pytest

    with pytest.raises(ValueError):
        LeaderRotationAdvisor(risk_appetite="reckless")


def test_invalid_horizon_raises():
    import pytest

    with pytest.raises(ValueError):
        LeaderRotationAdvisor(horizon=0)
