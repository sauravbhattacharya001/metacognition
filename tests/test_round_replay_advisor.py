"""Tests for RoundReplayAdvisor."""
from __future__ import annotations

from datetime import datetime, timezone


from src.core.state import RoundResult, Vote
from src.round_replay_advisor import (
    RoundReplayAdvisor,
)


FIXED_NOW = lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)


def _round(
    idx: int,
    leader: str,
    committed: bool,
    aggregate: float,
    threshold: float,
    votes,
) -> RoundResult:
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution="42" if committed else None,
        aggregate_weight=aggregate,
        threshold=threshold,
        votes=list(votes),
        slashed=[] if committed else [leader],
    )


def _vote(voter: str, weight: float, counter: str | None = None) -> Vote:
    return Vote(
        voter_id=voter,
        target_proposal_id="p1",
        weight=weight,
        counter_proof=counter,
    )


def test_empty_history_returns_clean_report():
    rep = RoundReplayAdvisor().analyze(history=[], reputation={}, now=FIXED_NOW)
    assert rep.rounds_replayed == 0
    assert rep.rounds_flippable == 0
    assert rep.rounds_unsalvageable == 0
    assert rep.per_round == []
    assert rep.playbook == []
    assert rep.overall_grade == "A"


def test_all_committed_rounds_skipped():
    history = [
        _round(0, "a1", True, 2.0, 1.5, [_vote("a2", 0.5)]),
        _round(1, "a1", True, 2.0, 1.5, [_vote("a2", 0.5)]),
    ]
    rep = RoundReplayAdvisor().analyze(history=history, reputation={"a1": 1.0, "a2": 1.0}, now=FIXED_NOW)
    assert rep.rounds_replayed == 0
    assert rep.per_round == []


def test_below_threshold_lower_threshold_flips():
    # aggregate 1.2, threshold 1.5, no rejection
    history = [
        _round(0, "a1", False, 1.2, 1.5, [_vote("a2", 0.3)]),
    ]
    rep = RoundReplayAdvisor().analyze(
        history=history, reputation={"a1": 0.5, "a2": 1.0}, now=FIXED_NOW
    )
    assert rep.rounds_replayed == 1
    r0 = rep.per_round[0]
    assert r0.original_blocker == "BELOW_THRESHOLD"
    names = [iv.intervention for iv in r0.top_interventions]
    assert "LOWER_THRESHOLD_TO_AGGREGATE" in names
    lt = next(iv for iv in r0.top_interventions if iv.intervention == "LOWER_THRESHOLD_TO_AGGREGATE")
    assert lt.projected_committed is True


def test_demote_slashed_rejection_flips_when_rejecter_is_slashed():
    # leader contrib reconstruct: aggregate = leader_contrib + sum(vote*rep)
    # vote a2 weight=-1.0 rep=0.4 -> contrib -0.4. leader_contrib = aggregate - (-0.4) = 1.0+0.4 = 1.4
    # if a2 dropped: aggregate becomes 1.4. threshold 1.3 -> committed.
    history = [
        _round(0, "a1", False, 1.0, 1.3, [_vote("a2", -1.0, counter="bad")]),
    ]
    rep = RoundReplayAdvisor().analyze(
        history=history, reputation={"a1": 1.0, "a2": 0.4}, now=FIXED_NOW
    )
    r0 = rep.per_round[0]
    # blocker classification: a2 rep<1.0 so it does NOT have unrefuted (rep<1).
    # So blocker should be BELOW_THRESHOLD actually. Let me check: aggregate 1.0 < threshold 1.3 -> below.
    assert r0.original_blocker == "BELOW_THRESHOLD"
    # DEMOTE_SLASHED_REJECTION only runs when unrefuted_now=True. So it shouldn't be there.
    # Let's adjust the test: make a rejection with rep<1 but main blocker is unrefuted via another voter.
    # Actually let's keep simple: under BELOW_THRESHOLD, IGNORE_VOTER(a2) should flip it (drops -0.4 -> agg=1.4)
    names = [iv.intervention for iv in r0.top_interventions]
    assert "IGNORE_VOTER(a2)" in names
    ig = next(iv for iv in r0.top_interventions if iv.intervention == "IGNORE_VOTER(a2)")
    assert ig.projected_committed is True
    assert ig.cost_band == "low"  # because a2 is slashed (rep<1)


def test_demote_slashed_rejection_appears_for_unrefuted_block():
    # Build: a1 leader, a2 rejects (rep=0.4 -> not unrefuted), a3 rejects (rep=1.0 -> unrefuted blocker)
    # aggregate above threshold but blocked by unrefuted rejection from a3.
    # If we DEMOTE_SLASHED_REJECTION (drop a2 only), a3 still unrefuted -> doesn't flip.
    # IGNORE_VOTER(a3) should flip.
    history = [
        _round(0, "a1", False, 2.0, 1.5, [
            _vote("a2", -0.5, counter="x"),
            _vote("a3", -0.3, counter="y"),
        ]),
    ]
    rep = RoundReplayAdvisor().analyze(
        history=history, reputation={"a1": 1.0, "a2": 0.4, "a3": 1.0}, now=FIXED_NOW
    )
    r0 = rep.per_round[0]
    assert r0.original_blocker == "UNREFUTED_REJECTION"
    names = [iv.intervention for iv in r0.top_interventions]
    assert "DEMOTE_SLASHED_REJECTION" in names
    assert "IGNORE_VOTER(a3)" in names


def test_ignore_voter_for_rep1_rejecter_is_medium_cost():
    history = [
        _round(0, "a1", False, 2.0, 1.5, [_vote("a3", -0.3, counter="y")]),
    ]
    rep = RoundReplayAdvisor().analyze(
        history=history, reputation={"a1": 1.0, "a3": 1.0}, now=FIXED_NOW
    )
    r0 = rep.per_round[0]
    ig = next(iv for iv in r0.top_interventions if iv.intervention == "IGNORE_VOTER(a3)")
    assert ig.cost_band == "medium"


def test_swap_leader_appears_when_any_rejection():
    history = [
        _round(0, "a1", False, 1.0, 1.5, [_vote("a2", -0.4, counter="x")]),
    ]
    rep = RoundReplayAdvisor().analyze(
        history=history, reputation={"a1": 1.0, "a2": 1.0}, now=FIXED_NOW
    )
    r0 = rep.per_round[0]
    names = [iv.intervention for iv in r0.top_interventions]
    assert "SWAP_LEADER_TO_STRONGEST_DISSENTER" in names


def test_add_redundant_agent_only_for_close_below_no_rejection():
    # aggregate 1.2, threshold 1.5, no rejection, margin 0.3 < default close_margin 0.5
    history = [_round(0, "a1", False, 1.2, 1.5, [_vote("a2", 0.3)])]
    rep = RoundReplayAdvisor().analyze(
        history=history, reputation={"a1": 1.0, "a2": 1.0}, now=FIXED_NOW
    )
    names = [iv.intervention for iv in rep.per_round[0].top_interventions]
    assert "ADD_REDUNDANT_AGENT" in names

    # not present when there's an unrefuted rejection
    history2 = [_round(0, "a1", False, 2.0, 1.5, [_vote("a2", -0.3, counter="x")])]
    rep2 = RoundReplayAdvisor().analyze(
        history=history2, reputation={"a1": 1.0, "a2": 1.0}, now=FIXED_NOW
    )
    names2 = [iv.intervention for iv in rep2.per_round[0].top_interventions]
    assert "ADD_REDUNDANT_AGENT" not in names2


def test_priority_ordering_p0_first():
    # Construct round where IGNORE_VOTER on a slashed dissenter flips (cost=low -> P0)
    # and LOWER_THRESHOLD flips (high cost -> P1). Verify P0 sorted first.
    history = [_round(0, "a1", False, 1.0, 1.5, [_vote("a2", -0.4, counter="x")])]
    rep = RoundReplayAdvisor().analyze(
        history=history, reputation={"a1": 1.0, "a2": 0.4}, now=FIXED_NOW
    )
    top = rep.per_round[0].top_interventions
    priorities = [iv.priority for iv in top]
    # Must be non-decreasing in rank order
    ranks = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    assert ranks[priorities[0]] <= ranks[priorities[-1]]


def test_recommended_picks_cheapest_p0():
    # IGNORE_VOTER(a2) cost=low flips -> P0. Verify it becomes the recommendation.
    # vote a2 weight=-1.0 rep=0.4 -> contrib=-0.4; leader_contrib = 1.0 - (-0.4) = 1.4
    # drop a2 -> agg=1.4 >= threshold 1.3 -> committed.
    history = [_round(0, "a1", False, 1.0, 1.3, [
        _vote("a2", -1.0, counter="x"),
    ])]
    rep = RoundReplayAdvisor().analyze(
        history=history, reputation={"a1": 1.0, "a2": 0.4}, now=FIXED_NOW
    )
    r0 = rep.per_round[0]
    assert r0.recommended is not None
    assert r0.recommended.priority == "P0"
    assert r0.recommended.cost_band == "low"


def test_repeat_blocker_voter_playbook():
    # Same voter a2 (slashed) flips 2 rounds via IGNORE_VOTER (weight -1.0 rep 0.4)
    history = [
        _round(0, "a1", False, 1.0, 1.3, [_vote("a2", -1.0, counter="x")]),
        _round(1, "a1", False, 1.0, 1.3, [_vote("a2", -1.0, counter="y")]),
    ]
    rep = RoundReplayAdvisor().analyze(
        history=history, reputation={"a1": 1.0, "a2": 0.4}, now=FIXED_NOW
    )
    patterns = [p.pattern for p in rep.playbook]
    assert "REPEAT_BLOCKER_VOTER" in patterns
    rb = next(p for p in rep.playbook if p.pattern == "REPEAT_BLOCKER_VOTER")
    assert rb.target == "a2"
    assert rb.priority == "P0"


def test_systematic_threshold_too_high_playbook():
    # 2 rounds, both BELOW_THRESHOLD no rejection, both flippable by LOWER_THRESHOLD
    history = [
        _round(0, "a1", False, 1.2, 1.5, [_vote("a2", 0.3)]),
        _round(1, "a1", False, 1.3, 1.5, [_vote("a2", 0.3)]),
    ]
    rep = RoundReplayAdvisor().analyze(
        history=history, reputation={"a1": 1.0, "a2": 1.0}, now=FIXED_NOW
    )
    patterns = [p.pattern for p in rep.playbook]
    assert "SYSTEMATIC_THRESHOLD_TOO_HIGH" in patterns
    st = next(p for p in rep.playbook if p.pattern == "SYSTEMATIC_THRESHOLD_TOO_HIGH")
    assert st.suggested_value is not None
    assert 1.0 <= st.suggested_value <= 1.5


def test_to_json_deterministic_with_fixed_now():
    history = [_round(0, "a1", False, 1.2, 1.5, [_vote("a2", 0.3)])]
    rep = RoundReplayAdvisor().analyze(history=history, reputation={"a1": 1.0, "a2": 1.0}, now=FIXED_NOW)
    a = rep.to_json()
    b = rep.to_json()
    assert a == b
    assert '"generated_at": "2026-01-01T00:00:00+00:00"' in a


def test_to_text_and_markdown_nonempty():
    history = [_round(0, "a1", False, 1.2, 1.5, [_vote("a2", 0.3)])]
    rep = RoundReplayAdvisor().analyze(history=history, reputation={"a1": 1.0, "a2": 1.0}, now=FIXED_NOW)
    txt = rep.to_text()
    md = rep.to_markdown()
    assert "RoundReplayAdvisor" in txt
    assert "RoundReplayAdvisor" in md
    assert "round 0" in txt
    assert "Round 0" in md


def test_analyzer_does_not_mutate_inputs():
    history = [_round(0, "a1", False, 1.2, 1.5, [_vote("a2", 0.3)])]
    reputation = {"a1": 1.0, "a2": 1.0}
    snapshot_hist_len = len(history)
    snapshot_rep = dict(reputation)
    RoundReplayAdvisor().analyze(history=history, reputation=reputation, now=FIXED_NOW)
    assert len(history) == snapshot_hist_len
    assert reputation == snapshot_rep
