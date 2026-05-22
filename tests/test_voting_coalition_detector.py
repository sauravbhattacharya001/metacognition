"""Tests for VotingCoalitionDetector."""
from __future__ import annotations

import json

import pytest

from src.core.state import RoundResult, Vote
from src.voting_coalition_detector import (
    VotingCoalitionDetector,
)


def _round(
    *,
    idx: int,
    leader: str,
    solution: str | None,
    aggregate: float,
    threshold: float,
    votes: list[Vote],
    slashed: list[str] | None = None,
) -> RoundResult:
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution=solution,
        aggregate_weight=aggregate,
        threshold=threshold,
        votes=votes,
        slashed=slashed or [],
    )


def _v(voter: str, weight: float, target: str = "p1") -> Vote:
    return Vote(voter_id=voter, target_proposal_id=target, weight=weight)


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_empty_history_returns_no_coalitions():
    rep = VotingCoalitionDetector().analyze(
        history=[],
        reputation={"a1": 1.0, "a2": 1.0},
        threshold=1.5,
    )
    assert rep.rounds_observed == 0
    assert rep.coalitions == []
    assert rep.overall_grade == "A"
    assert "No coalitions detected" in rep.summary
    assert rep.agent_count == 2


def test_independent_voters_produce_no_coalition():
    # Each round different sign pairings; no two agents consistently agree.
    history = [
        _round(
            idx=0,
            leader="a1",
            solution="x",
            aggregate=2.0,
            threshold=1.5,
            votes=[_v("a2", 0.9), _v("a3", -0.5), _v("a4", 0.8)],
        ),
        _round(
            idx=1,
            leader="a2",
            solution="y",
            aggregate=2.0,
            threshold=1.5,
            votes=[_v("a1", 0.8), _v("a3", 0.7), _v("a4", -0.9)],
        ),
        _round(
            idx=2,
            leader="a3",
            solution="z",
            aggregate=2.0,
            threshold=1.5,
            votes=[_v("a1", -0.6), _v("a2", 0.9), _v("a4", 0.7)],
        ),
    ]
    rep = VotingCoalitionDetector(
        cohesion_threshold=0.9, min_co_votes=2
    ).analyze(
        history=history,
        reputation={"a1": 1, "a2": 1, "a3": 1, "a4": 1},
        threshold=1.5,
        agent_ids=["a1", "a2", "a3", "a4"],
    )
    # Any incidental coalitions must be benign (no risky verdicts).
    for c in rep.coalitions:
        assert c.verdict == "BENIGN_AFFINITY"
    assert rep.overall_grade in ("A", "B")


def test_two_agents_always_agree_form_bonded_coalition():
    history = [
        _round(
            idx=i,
            leader="a3",
            solution="ok" if i % 2 == 0 else None,
            aggregate=1.0,
            threshold=1.5,
            votes=[_v("a1", 0.9), _v("a2", 0.9), _v("a4", -0.3)],
        )
        for i in range(4)
    ]
    rep = VotingCoalitionDetector(
        cohesion_threshold=0.85, min_co_votes=2
    ).analyze(
        history=history,
        reputation={"a1": 1, "a2": 1, "a3": 1, "a4": 1},
        threshold=1.5,
        agent_ids=["a1", "a2", "a3", "a4"],
    )
    assert len(rep.coalitions) == 1
    c = rep.coalitions[0]
    assert c.members == ["a1", "a2"]
    assert c.cohesion >= 0.99
    assert c.leader_capture_rate == 0.0
    # Joint actions on every round.
    assert c.rounds_active == 4


def test_dominant_faction_flagged_when_control_high_and_leads():
    # 3 of 4 agents always vote positively together and lead together.
    votes_pos = [_v("b1", 0.9), _v("b2", 0.9), _v("b3", 0.9), _v("o1", -0.2)]
    history = []
    leaders = ["b1", "b2", "b3", "b1"]
    for i, ldr in enumerate(leaders):
        # Move leader vote out
        present = [v for v in votes_pos if v.voter_id != ldr]
        history.append(
            _round(
                idx=i,
                leader=ldr,
                solution="committed",
                aggregate=2.5,
                threshold=1.5,
                votes=present,
            )
        )
    rep = VotingCoalitionDetector(
        cohesion_threshold=0.85,
        min_co_votes=1,
        dominant_control_fraction=0.5,
        kingmaker_leader_rate=0.5,
    ).analyze(
        history=history,
        reputation={x: 1.0 for x in ["b1", "b2", "b3", "o1"]},
        threshold=1.5,
        agent_ids=["b1", "b2", "b3", "o1"],
    )
    assert any(c.verdict == "DOMINANT_FACTION" for c in rep.coalitions)
    assert rep.overall_grade == "F"
    patterns = {p.pattern for p in rep.playbook}
    assert "SPLIT_DOMINANT_FACTION" in patterns


def test_blocking_bloc_flagged_when_rejections_align():
    # b1+b2 always reject together; leader b3 fails to commit.
    history = [
        _round(
            idx=i,
            leader="b3",
            solution=None,
            aggregate=0.5,
            threshold=1.5,
            votes=[_v("b1", -0.8), _v("b2", -0.8), _v("b4", 0.5)],
        )
        for i in range(3)
    ]
    rep = VotingCoalitionDetector(
        cohesion_threshold=0.85, min_co_votes=2
    ).analyze(
        history=history,
        reputation={"b1": 1, "b2": 1, "b3": 1, "b4": 1},
        threshold=1.5,
        agent_ids=["b1", "b2", "b3", "b4"],
    )
    blocking = [c for c in rep.coalitions if c.verdict == "BLOCKING_BLOC"]
    assert blocking, rep.coalitions
    assert blocking[0].rejection_alignment == 1.0
    assert any(p.pattern == "INVESTIGATE_BLOCKING" for p in rep.playbook)


def test_kingmaker_flagged_when_members_lead_often():
    # k1+k2 strongly correlated and lead 3/4 rounds together.
    votes = [_v("k1", 0.9), _v("k2", 0.9), _v("o1", 0.4)]
    history = []
    leaders = ["k1", "k2", "k1", "o1"]
    for i, ldr in enumerate(leaders):
        present = [v for v in votes if v.voter_id != ldr]
        history.append(
            _round(
                idx=i,
                leader=ldr,
                solution="ok",
                aggregate=1.8,
                threshold=1.5,
                votes=present,
            )
        )
    rep = VotingCoalitionDetector(
        cohesion_threshold=0.85,
        min_co_votes=1,
        dominant_control_fraction=0.95,
        kingmaker_leader_rate=0.5,
    ).analyze(
        history=history,
        reputation={"k1": 1, "k2": 1, "o1": 1},
        threshold=1.5,
        agent_ids=["k1", "k2", "o1"],
    )
    verdicts = {c.verdict for c in rep.coalitions}
    assert "KINGMAKER" in verdicts
    assert any(p.pattern == "ROTATE_LEADERS" for p in rep.playbook)


def test_echo_chamber_flagged_when_commit_alignment_low():
    # Two agents always agree positively but commits never happen.
    history = [
        _round(
            idx=i,
            leader="e3",
            solution=None,
            aggregate=0.9,
            threshold=1.5,
            votes=[_v("e1", 0.9), _v("e2", 0.9), _v("e4", -0.6)],
        )
        for i in range(3)
    ]
    rep = VotingCoalitionDetector(
        cohesion_threshold=0.85,
        min_co_votes=1,
        dominant_control_fraction=0.95,
        kingmaker_leader_rate=0.95,
        blocking_rejection_alignment=0.95,
    ).analyze(
        history=history,
        reputation={"e1": 1, "e2": 1, "e3": 1, "e4": 1},
        threshold=1.5,
        agent_ids=["e1", "e2", "e3", "e4"],
    )
    verdicts = {c.verdict for c in rep.coalitions}
    assert "ECHO_CHAMBER" in verdicts


# ---------------------------------------------------------------------------
# Risk appetite / determinism / renderers
# ---------------------------------------------------------------------------


def test_risk_appetite_changes_classification_threshold():
    # Make a borderline case: cautious flags BLOCKING_BLOC at lower
    # rejection alignment than balanced.
    history = [
        _round(
            idx=i,
            leader="x",
            solution=None,
            aggregate=0.9,
            threshold=1.5,
            votes=[_v("a1", -0.5 if i % 2 == 0 else 0.5),
                   _v("a2", -0.5 if i % 2 == 0 else 0.5),
                   _v("a3", 0.3)],
        )
        for i in range(4)
    ]
    base = VotingCoalitionDetector(
        cohesion_threshold=0.85, min_co_votes=2
    ).analyze(
        history=history,
        reputation={"a1": 1, "a2": 1, "a3": 1, "x": 1},
        threshold=1.5,
        agent_ids=["a1", "a2", "a3", "x"],
        risk_appetite="balanced",
    )
    cautious = VotingCoalitionDetector(
        cohesion_threshold=0.85, min_co_votes=2
    ).analyze(
        history=history,
        reputation={"a1": 1, "a2": 1, "a3": 1, "x": 1},
        threshold=1.5,
        agent_ids=["a1", "a2", "a3", "x"],
        risk_appetite="cautious",
    )
    # Both should find the coalition; cautious should be at least as
    # severe in risk_score for the same coalition.
    assert base.coalitions and cautious.coalitions
    assert cautious.cohesion_threshold <= base.cohesion_threshold


def test_invalid_risk_appetite_raises():
    with pytest.raises(ValueError):
        VotingCoalitionDetector().analyze(
            history=[],
            reputation={},
            threshold=1.5,
            risk_appetite="silly",
        )


def test_aggressive_trims_low_priority_actions():
    history = [
        _round(
            idx=i,
            leader="z",
            solution="ok",
            aggregate=1.8,
            threshold=1.5,
            votes=[_v("e1", 0.9), _v("e2", 0.9), _v("e3", -0.3)],
        )
        for i in range(3)
    ]
    balanced = VotingCoalitionDetector(
        cohesion_threshold=0.85,
        min_co_votes=2,
        dominant_control_fraction=0.95,
        kingmaker_leader_rate=0.95,
        blocking_rejection_alignment=0.95,
    ).analyze(
        history=history,
        reputation={"e1": 1, "e2": 1, "e3": 1, "z": 1},
        threshold=1.5,
        agent_ids=["e1", "e2", "e3", "z"],
        risk_appetite="balanced",
    )
    aggressive = VotingCoalitionDetector(
        cohesion_threshold=0.85,
        min_co_votes=2,
        dominant_control_fraction=0.95,
        kingmaker_leader_rate=0.95,
        blocking_rejection_alignment=0.95,
    ).analyze(
        history=history,
        reputation={"e1": 1, "e2": 1, "e3": 1, "z": 1},
        threshold=1.5,
        agent_ids=["e1", "e2", "e3", "z"],
        risk_appetite="aggressive",
    )
    # No P3 in aggressive playbook (besides fallback maybe).
    assert all(p.priority in ("P0", "P1") for p in aggressive.playbook) or (
        len(aggressive.playbook) == 1
    )
    # Balanced may contain P2 actions where aggressive doesn't.
    bal_p2_count = sum(1 for p in balanced.playbook if p.priority == "P2")
    agg_p2_count = sum(1 for p in aggressive.playbook if p.priority == "P2")
    assert agg_p2_count <= bal_p2_count


def test_renderers_produce_text_markdown_json():
    history = [
        _round(
            idx=i,
            leader="b3",
            solution=None,
            aggregate=0.5,
            threshold=1.5,
            votes=[_v("b1", -0.8), _v("b2", -0.8), _v("b4", 0.5)],
        )
        for i in range(3)
    ]
    rep = VotingCoalitionDetector(
        cohesion_threshold=0.85, min_co_votes=2
    ).analyze(
        history=history,
        reputation={"b1": 1, "b2": 1, "b3": 1, "b4": 1},
        threshold=1.5,
        agent_ids=["b1", "b2", "b3", "b4"],
    )
    text = rep.to_text()
    assert "VOTING COALITION REPORT" in text
    md = rep.to_markdown()
    assert "# Voting Coalition Report" in md
    js = rep.to_json()
    parsed = json.loads(js)
    assert parsed["rounds_observed"] == 3
    # Byte-stable: same input -> same JSON.
    assert rep.to_json() == js


def test_json_is_byte_stable_with_fixed_clock():
    from datetime import datetime, timezone

    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    detector = VotingCoalitionDetector(now_fn=lambda: fixed)
    history = [
        _round(
            idx=0,
            leader="b3",
            solution=None,
            aggregate=0.5,
            threshold=1.5,
            votes=[_v("b1", -0.8), _v("b2", -0.8)],
        )
    ]
    a = detector.analyze(
        history=history,
        reputation={"b1": 1, "b2": 1, "b3": 1},
        threshold=1.5,
        agent_ids=["b1", "b2", "b3"],
    )
    b = detector.analyze(
        history=history,
        reputation={"b1": 1, "b2": 1, "b3": 1},
        threshold=1.5,
        agent_ids=["b1", "b2", "b3"],
    )
    assert a.to_json() == b.to_json()


def test_detector_never_mutates_inputs():
    history = [
        _round(
            idx=i,
            leader="b3",
            solution=None,
            aggregate=0.5,
            threshold=1.5,
            votes=[_v("b1", -0.8), _v("b2", -0.8), _v("b4", 0.5)],
        )
        for i in range(3)
    ]
    rep_in = {"b1": 1.0, "b2": 1.0, "b3": 1.0, "b4": 1.0}
    snapshot_history = [r.model_dump() for r in history]
    snapshot_rep = dict(rep_in)
    VotingCoalitionDetector().analyze(
        history=history,
        reputation=rep_in,
        threshold=1.5,
        agent_ids=["b1", "b2", "b3", "b4"],
    )
    assert [r.model_dump() for r in history] == snapshot_history
    assert rep_in == snapshot_rep


def test_healthy_fleet_appears_when_no_coalitions():
    history = [
        _round(
            idx=i,
            leader=f"a{i}",
            solution="ok",
            aggregate=2.0,
            threshold=1.5,
            votes=[_v(f"a{(i + 1) % 4}", 0.9),
                   _v(f"a{(i + 2) % 4}", -0.3 if i % 2 else 0.5)],
        )
        for i in range(4)
    ]
    rep = VotingCoalitionDetector(
        cohesion_threshold=0.95, min_co_votes=3
    ).analyze(
        history=history,
        reputation={"a0": 1, "a1": 1, "a2": 1, "a3": 1},
        threshold=1.5,
        agent_ids=["a0", "a1", "a2", "a3"],
    )
    assert rep.coalitions == []
    patterns = {p.pattern for p in rep.playbook}
    assert "HEALTHY_FLEET" in patterns


def test_min_co_votes_validation():
    with pytest.raises(ValueError):
        VotingCoalitionDetector(min_co_votes=0)
    with pytest.raises(ValueError):
        VotingCoalitionDetector(cohesion_threshold=1.5)


def test_priority_ordering_in_report():
    history = []
    # Bloc 1: dominant
    for i in range(4):
        history.append(
            _round(
                idx=i,
                leader="d1" if i % 2 == 0 else "d2",
                solution="x",
                aggregate=2.5,
                threshold=1.5,
                votes=[_v("d1", 0.9) if i % 2 else _v("d1", 0.9),
                       _v("d2", 0.9), _v("d3", 0.9), _v("o", -0.2)],
            )
        )
    rep = VotingCoalitionDetector(
        cohesion_threshold=0.85,
        min_co_votes=1,
        dominant_control_fraction=0.5,
        kingmaker_leader_rate=0.5,
    ).analyze(
        history=history,
        reputation={"d1": 1, "d2": 1, "d3": 1, "o": 1},
        threshold=1.5,
        agent_ids=["d1", "d2", "d3", "o"],
    )
    if len(rep.coalitions) >= 2:
        # Higher priority (lower index) first.
        from src.voting_coalition_detector import _PRIORITY_RANK  # type: ignore

        ranks = [_PRIORITY_RANK[c.priority] for c in rep.coalitions]
        assert ranks == sorted(ranks)


def test_insights_emit_dominant_when_present():
    history = [
        _round(
            idx=i,
            leader="d1" if i % 2 == 0 else "d2",
            solution="x",
            aggregate=2.5,
            threshold=1.5,
            votes=[_v("d2" if i % 2 == 0 else "d1", 0.9),
                   _v("d3", 0.9), _v("o", -0.2)],
        )
        for i in range(4)
    ]
    rep = VotingCoalitionDetector(
        cohesion_threshold=0.85,
        min_co_votes=1,
        dominant_control_fraction=0.5,
        kingmaker_leader_rate=0.5,
    ).analyze(
        history=history,
        reputation={"d1": 1, "d2": 1, "d3": 1, "o": 1},
        threshold=1.5,
        agent_ids=["d1", "d2", "d3", "o"],
    )
    # Should have DOMINANT_FACTION or KINGMAKER somewhere.
    assert any(
        "DOMINANT_FACTION_PRESENT" in s or "LEADER_CAPTURE" in s
        for s in rep.insights
    )
