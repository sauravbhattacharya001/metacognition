"""Tests for ProposalRiskScorer."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.core.state import Proposal, RoundResult, Vote
from src.proposal_risk_scorer import (
    PlaybookAction,
    PredictedVoter,
    ProposalRiskReport,
    ProposalRiskScorer,
    RiskFactor,
)


FIXED_NOW = lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)


def _vote(voter, weight, counter=None):
    return Vote(
        voter_id=voter,
        target_proposal_id="p1",
        weight=weight,
        counter_proof=counter,
    )


def _round(idx, leader, committed, agg, thr, votes, sol="42"):
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution=sol if committed else None,
        aggregate_weight=agg,
        threshold=thr,
        votes=list(votes),
        slashed=[] if committed else [leader],
    )


def _good_proposal(**kw):
    return Proposal(
        agent_id=kw.get("agent_id", "a1"),
        solution=kw.get("solution", "the answer is 42 based on prior work"),
        proof=kw.get(
            "proof",
            "Because step 1 follows from [1] and therefore by lemma 2 we have the result. See https://example.com",
        ),
        confidence=kw.get("confidence", 0.7),
    )


def test_empty_history_thin_roster_is_safe_or_low():
    scorer = ProposalRiskScorer(threshold=3.0, now_fn=FIXED_NOW)
    rep = scorer.score(_good_proposal(), leader_reputation=0.8, history=[], roster={})
    assert rep.verdict in ("SAFE", "LOW", "ELEVATED")
    assert rep.grade in ("A", "B", "C")
    assert not any(a.priority == "P0" for a in rep.playbook)
    assert "EMPTY_HISTORY" in rep.insights
    assert "THIN_ROSTER" in rep.insights


def test_placeholder_in_solution_blocks_submission():
    scorer = ProposalRiskScorer(threshold=3.0, now_fn=FIXED_NOW)
    p = _good_proposal(solution="The result is {{TODO_FILL_IN}} maybe")
    rep = scorer.score(p, leader_reputation=0.9, history=[], roster={})
    assert rep.verdict == "BLOCK_SUBMISSION"
    assert rep.grade == "F"
    ids = [a.id for a in rep.playbook]
    assert "REMOVE_PLACEHOLDER" in ids
    assert any(a.priority == "P0" and a.id == "REMOVE_PLACEHOLDER" for a in rep.playbook)


def test_low_quality_proof_triggers_rewrite_proof():
    scorer = ProposalRiskScorer(threshold=3.0, now_fn=FIXED_NOW)
    p = _good_proposal(proof="ok")
    rep = scorer.score(p, leader_reputation=0.5, history=[], roster={})
    proof_factor = next(f for f in rep.factors if f.dimension == "proof_quality")
    assert proof_factor.severity >= 70
    assert any(a.id == "REWRITE_PROOF" for a in rep.playbook)


def test_overconfident_leader_lowers_confidence():
    history = [
        _round(0, "a1", False, 0.5, 3.0, [_vote("a2", 0.2)]),
        _round(1, "a1", False, 0.7, 3.0, [_vote("a2", 0.3)]),
        _round(2, "a1", False, 0.4, 3.0, [_vote("a2", 0.1)]),
    ]
    scorer = ProposalRiskScorer(threshold=3.0, now_fn=FIXED_NOW)
    p = _good_proposal(confidence=0.95)
    rep = scorer.score(p, leader_reputation=0.2, history=history, roster={"a2": 1.0})
    assert "OVERCONFIDENT_LEADER" in rep.insights
    assert any(a.id == "LOWER_CONFIDENCE" for a in rep.playbook)


def test_stale_rejected_replay_abandon():
    # last round had a rejection with a counter_proof solution; we re-submit
    # the same idea
    history = [
        _round(
            0,
            "a1",
            False,
            0.5,
            3.0,
            [_vote("a2", -0.5, counter="the answer is definitely 7 not 42")],
        ),
    ]
    scorer = ProposalRiskScorer(threshold=3.0, now_fn=FIXED_NOW)
    p = _good_proposal(solution="the answer is definitely 7 not 42")
    rep = scorer.score(
        p, leader_reputation=0.5, history=history, roster={"a2": 1.0}
    )
    assert rep.verdict == "BLOCK_SUBMISSION"
    assert "STALE_REJECTED_REPLAY" in rep.insights
    assert any(a.id == "ABANDON_STALE_REPLAY" for a in rep.playbook)


def test_chronic_rejectors_trigger_add_counter_proof():
    history = [
        _round(0, "a1", False, 0.5, 3.0, [_vote("a2", -0.7), _vote("a3", -0.6)]),
        _round(1, "a1", False, 0.5, 3.0, [_vote("a2", -0.7), _vote("a3", -0.6)]),
        _round(2, "a1", False, 0.5, 3.0, [_vote("a2", -0.7), _vote("a3", -0.6)]),
    ]
    scorer = ProposalRiskScorer(threshold=3.0, now_fn=FIXED_NOW)
    p = _good_proposal()
    rep = scorer.score(
        p,
        leader_reputation=0.5,
        history=history,
        roster={"a2": 1.0, "a3": 1.0},
    )
    assert "CHRONIC_BLOCKER_AUDIENCE" in rep.insights
    assert any(a.id == "ADD_COUNTER_PROOF" for a in rep.playbook)


def test_predicted_aggregate_below_threshold_shrink_scope():
    # tiny roster, low predicted aggregate vs threshold 5.0
    scorer = ProposalRiskScorer(threshold=5.0, now_fn=FIXED_NOW)
    p = _good_proposal(confidence=0.5)
    rep = scorer.score(
        p,
        leader_reputation=0.4,
        history=[],
        roster={"a2": 0.3, "a3": 0.3},
    )
    assert "LIKELY_BELOW_THRESHOLD" in rep.insights
    shortfall = next(
        f for f in rep.factors if f.dimension == "predicted_aggregate_shortfall"
    )
    assert shortfall.severity >= 30
    assert any(a.id == "SHRINK_SCOPE" for a in rep.playbook)


def test_risk_appetite_monotonicity():
    history = [
        _round(0, "a1", False, 1.0, 3.0, [_vote("a2", -0.3), _vote("a3", 0.2)]),
    ]
    roster = {"a2": 0.8, "a3": 0.7}
    p = _good_proposal(proof="short proof", confidence=0.9)

    def run(appetite):
        s = ProposalRiskScorer(
            threshold=3.0, risk_appetite=appetite, now_fn=FIXED_NOW
        )
        return s.score(p, leader_reputation=0.4, history=history, roster=roster)

    cautious = run("cautious").overall_risk_score
    balanced = run("balanced").overall_risk_score
    aggressive = run("aggressive").overall_risk_score
    assert cautious >= balanced >= aggressive


def test_json_byte_stability():
    scorer = ProposalRiskScorer(threshold=3.0, now_fn=FIXED_NOW)
    p = _good_proposal()
    rep = scorer.score(p, leader_reputation=0.6, history=[], roster={"a2": 0.6})
    assert rep.to_json() == rep.to_json()


def test_markdown_contains_required_sections():
    scorer = ProposalRiskScorer(threshold=3.0, now_fn=FIXED_NOW)
    p = _good_proposal()
    rep = scorer.score(p, leader_reputation=0.6, history=[], roster={"a2": 0.6})
    md = rep.to_markdown()
    assert "## Summary" in md
    assert "## Playbook" in md or "## Risk factors" in md  # playbook may be empty


def test_does_not_mutate_input_proposal():
    scorer = ProposalRiskScorer(threshold=3.0, now_fn=FIXED_NOW)
    p = _good_proposal()
    before = p.model_dump()
    scorer.score(p, leader_reputation=0.6, history=[], roster={"a2": 0.6})
    after = p.model_dump()
    assert before == after


def test_predicted_voters_sorted_desc():
    scorer = ProposalRiskScorer(threshold=3.0, now_fn=FIXED_NOW)
    p = _good_proposal()
    rep = scorer.score(
        p,
        leader_reputation=0.6,
        history=[],
        roster={"a2": 0.9, "a3": 0.3, "a4": 0.7},
    )
    weights = [v.predicted_weight for v in rep.predicted_voters]
    assert weights == sorted(weights, reverse=True)


def test_aggressive_trims_p3_proposal_ready():
    # induce a clean A/B grade where PROPOSAL_READY P3 would normally appear
    scorer_b = ProposalRiskScorer(
        threshold=3.0, risk_appetite="balanced", now_fn=FIXED_NOW
    )
    scorer_a = ProposalRiskScorer(
        threshold=3.0, risk_appetite="aggressive", now_fn=FIXED_NOW
    )
    p = _good_proposal()
    rep_b = scorer_b.score(p, leader_reputation=0.95, history=[], roster={})
    rep_a = scorer_a.score(p, leader_reputation=0.95, history=[], roster={})
    # balanced should have PROPOSAL_READY when grade A/B and no other actions
    if not any(a.id != "PROPOSAL_READY" for a in rep_b.playbook):
        assert not any(a.id == "PROPOSAL_READY" for a in rep_a.playbook)


def test_cautious_adds_second_reviewer_when_grade_low():
    scorer = ProposalRiskScorer(
        threshold=3.0, risk_appetite="cautious", now_fn=FIXED_NOW
    )
    # set up an ELEVATED/HIGH risk case
    p = _good_proposal(proof="short", confidence=0.9)
    rep = scorer.score(
        p,
        leader_reputation=0.3,
        history=[],
        roster={"a2": 0.3, "a3": 0.3},
    )
    assert rep.grade in ("C", "D", "F")
    assert any(a.id == "SECOND_REVIEWER" for a in rep.playbook)
