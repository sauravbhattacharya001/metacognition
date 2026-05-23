"""Tests for src.counter_proof_quality_advisor."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.core.state import RoundResult, Vote
from src.counter_proof_quality_advisor import CounterProofQualityAdvisor


def _vote(voter: str, weight: float, counter: str | None = None) -> Vote:
    return Vote(
        voter_id=voter,
        target_proposal_id="p",
        weight=weight,
        counter_proof=counter,
    )


def _round(
    idx: int,
    leader: str = "L",
    aggregate: float = 1.0,
    threshold: float = 1.5,
    committed: bool = False,
    votes: list[Vote] | None = None,
    sol: str | None = None,
) -> RoundResult:
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution=sol if committed else None,
        aggregate_weight=aggregate,
        threshold=threshold,
        votes=votes or [],
    )


def _adv() -> CounterProofQualityAdvisor:
    return CounterProofQualityAdvisor(now_fn=lambda: datetime(2026, 1, 1, 0, 0, 0))


# ---------------------------------------------------------------------------


def test_empty_history_grade_a() -> None:
    r = _adv().analyze([])
    assert r.rounds_observed == 0
    assert r.rejections_observed == 0
    assert r.grade == "A"
    assert r.portfolio_quality_score == 100.0
    assert any(i.startswith("NO_REJECTIONS_OBSERVED") for i in r.insights)


def test_no_rejections_in_history_grade_a() -> None:
    h = [_round(0, votes=[_vote("a", 0.8)])]
    r = _adv().analyze(h)
    assert r.rejections_observed == 0
    assert r.grade == "A"


def test_missing_counter_proof_p0_finding() -> None:
    h = [_round(0, votes=[_vote("a", -0.7)])]
    r = _adv().analyze(h)
    codes = [f.code for f in r.findings]
    assert "MISSING_COUNTER_PROOF" in codes
    f = next(x for x in r.findings if x.code == "MISSING_COUNTER_PROOF")
    assert f.priority == "P0"
    assert f.severity >= 60


def test_vague_counter_proof_flagged() -> None:
    h = [_round(0, votes=[_vote("a", -0.7, "too short")])]
    r = _adv().analyze(h)
    assert any(f.code == "VAGUE_COUNTER_PROOF" for f in r.findings)


def test_template_repetition_detected_across_rounds() -> None:
    text = "the proof is wrong because of step three"
    h = [
        _round(0, votes=[_vote("a", -0.7, text)]),
        _round(1, votes=[_vote("a", -0.7, text)]),
    ]
    r = _adv().analyze(h)
    template_findings = [f for f in r.findings if f.code == "TEMPLATE_REPETITION"]
    assert len(template_findings) >= 2  # both rounds flagged once count>=2


def test_generic_phrase_detected() -> None:
    h = [_round(0, votes=[_vote("a", -0.7, "disagree")])]
    r = _adv().analyze(h)
    assert any(f.code == "GENERIC_PHRASE" for f in r.findings)


def test_low_information_detected() -> None:
    # Mostly stopwords with a few content tokens, no contradiction marker
    h = [_round(0, votes=[_vote("a", -0.7, "it is the answer for the team of this is the result that they agree")])]
    r = _adv().analyze(h)
    assert any(f.code == "LOW_INFORMATION" for f in r.findings)


def test_contradictory_rejecter_detected() -> None:
    # voter "a" first votes positively on a committed solution "S"
    h = [
        _round(0, committed=True, sol="S", votes=[_vote("a", 0.8)]),
        _round(
            1,
            committed=False,
            sol="S",
            votes=[_vote("a", -0.7, "this violates axiom 2 because of step 4")],
        ),
    ]
    r = _adv().analyze(h)
    assert any(f.code == "CONTRADICTORY_REJECTER" for f in r.findings)


def test_high_quality_counter_proof_flagged_positive() -> None:
    text = (
        "The proof violates the associativity axiom on line 3 because "
        "(a + b) + c was replaced with a + (b * c), which is a counterexample."
    )
    h = [_round(0, votes=[_vote("a", -0.8, text)])]
    r = _adv().analyze(h)
    assert any(f.code == "HIGH_QUALITY" for f in r.findings)


def test_voter_quality_aggregation_and_status() -> None:
    h = [
        _round(0, votes=[_vote("lazy", -0.7), _vote("lazy", -0.7)]),
        _round(1, votes=[_vote("lazy", -0.7)]),
    ]
    r = _adv().analyze(h)
    v = next(v for v in r.voters if v.voter_id == "lazy")
    assert v.missing_proof_count == 3
    assert v.status == "slash_candidate"
    assert r.grade == "F"


def test_slash_playbook_action_emitted_for_proofless_rejecter() -> None:
    h = [
        _round(0, votes=[_vote("lazy", -0.7), _vote("lazy", -0.7)]),
    ]
    r = _adv().analyze(h)
    ids = [a.id for a in r.playbook]
    assert "SLASH_NO_PROOF_REJECTERS" in ids
    a = next(a for a in r.playbook if a.id == "SLASH_NO_PROOF_REJECTERS")
    assert a.priority == "P0"
    assert "lazy" in a.target_voters


def test_coach_vague_action_emitted() -> None:
    h = [
        _round(0, votes=[_vote("v", -0.7, "no good")]),
        _round(1, votes=[_vote("v", -0.7, "bad")]),
    ]
    r = _adv().analyze(h)
    ids = [a.id for a in r.playbook]
    assert "COACH_VAGUE_REJECTERS" in ids


def test_reward_high_quality_action_emitted() -> None:
    text = (
        "The proof violates the associativity axiom on line 3 because "
        "(a + b) + c was replaced with a + (b * c)."
    )
    h = [_round(0, votes=[_vote("star", -0.8, text)])]
    r = _adv().analyze(h)
    ids = [a.id for a in r.playbook]
    assert "REWARD_HIGH_QUALITY_REJECTERS" in ids


def test_aggressive_trims_p3_when_higher_priority_present() -> None:
    h = [_round(0, votes=[_vote("lazy", -0.7), _vote("lazy", -0.7)])]
    r = _adv().analyze(h, risk_appetite="aggressive")
    assert all(a.priority != "P3" for a in r.playbook)


def test_cautious_schedules_audit_when_grade_low() -> None:
    h = [_round(0, votes=[_vote("lazy", -0.7), _vote("lazy", -0.7)])]
    r = _adv().analyze(h, risk_appetite="cautious")
    ids = [a.id for a in r.playbook]
    assert "SCHEDULE_REJECTION_AUDIT" in ids


def test_healthy_fallback_action_when_no_findings() -> None:
    text = (
        "The proof violates the associativity axiom on line 3 because "
        "(a + b) + c was replaced with a + (b * c)."
    )
    h = [_round(0, votes=[_vote("star", -0.8, text)])]
    r = _adv().analyze(h)
    # high quality only -> no negative playbook actions, healthy fallback fires
    ids = [a.id for a in r.playbook]
    assert "HEALTHY_REJECTION_DISCIPLINE" not in ids  # reward is present, so not "no actions"
    # But if literally nothing fires (no rejections at all):
    r2 = _adv().analyze([_round(0, votes=[_vote("p", 0.5)])])
    ids2 = [a.id for a in r2.playbook]
    assert "HEALTHY_REJECTION_DISCIPLINE" in ids2


def test_renderers_produce_nonempty_strings() -> None:
    h = [_round(0, votes=[_vote("v", -0.7, "no")])]
    r = _adv().analyze(h)
    assert "Counter-Proof Quality Report" in r.to_markdown()
    assert "VERDICT" in r.to_text()
    obj = json.loads(r.to_json())
    assert obj["rejections_observed"] == 1
    # byte-stable: json renderer uses sort_keys -> keys must be sorted
    text = r.to_json()
    assert text.index('"findings"') < text.index('"grade"')


def test_invalid_risk_appetite_raises() -> None:
    with pytest.raises(ValueError):
        _adv().analyze([], risk_appetite="bogus")


def test_never_mutates_inputs() -> None:
    votes = [_vote("v", -0.7, "no")]
    h = [_round(0, votes=votes)]
    snapshot = [r.model_dump() for r in h]
    _adv().analyze(h)
    assert [r.model_dump() for r in h] == snapshot


def test_deterministic_findings_order() -> None:
    h = [
        _round(0, votes=[_vote("b", -0.7), _vote("a", -0.7, "no")]),
        _round(1, votes=[_vote("a", -0.7, "wrong")]),
    ]
    r1 = _adv().analyze(h)
    r2 = _adv().analyze(h)
    assert [(f.round_index, f.code, f.voter_id) for f in r1.findings] == [
        (f.round_index, f.code, f.voter_id) for f in r2.findings
    ]
