"""Tests for src.disagreement_forensics."""
from __future__ import annotations

import json

import pytest

from src.core.state import RoundResult, Vote
from src.disagreement_forensics import DisagreementForensics


def _vote(voter: str, weight: float, counter: str | None = None) -> Vote:
    return Vote(
        voter_id=voter,
        target_proposal_id="p",
        weight=weight,
        counter_proof=counter,
    )


def _round(
    idx: int,
    leader: str,
    aggregate: float,
    threshold: float,
    committed: bool,
    votes: list[Vote] | None = None,
) -> RoundResult:
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution="X" if committed else None,
        aggregate_weight=aggregate,
        threshold=threshold,
        votes=votes or [],
    )


def test_empty_history_returns_empty_report() -> None:
    report = DisagreementForensics().analyze([])
    assert report.total_rounds == 0
    assert report.committed_rounds == 0
    assert report.failed_rounds == 0
    assert report.rounds == []
    assert report.patterns == []
    assert "no rounds" in report.headline


def test_clean_commit_round_yields_info_severity_no_action() -> None:
    history = [_round(0, "a1", 2.0, 1.5, committed=True, votes=[_vote("a2", 0.8)])]
    report = DisagreementForensics().analyze(history)
    assert report.committed_rounds == 1
    assert report.failed_rounds == 0
    fr = report.rounds[0]
    assert fr.blocker == "COMMITTED"
    assert fr.severity == "info"
    assert fr.recommendation_priority == "none"
    assert fr.primary_blame is None


def test_unrefuted_rejection_classifies_and_blames_strongest_dissenter() -> None:
    votes = [
        _vote("a2", 0.5),
        _vote("a3", -0.9, counter="contradicts axiom"),
        _vote("a4", -0.3, counter="weak doubt"),
    ]
    history = [_round(0, "a1", 2.0, 1.5, committed=False, votes=votes)]
    report = DisagreementForensics().analyze(history)
    fr = report.rounds[0]
    assert fr.blocker == "UNREFUTED_REJECTION"
    assert fr.primary_blame == "a3"
    assert fr.next_view_leader == "a3"
    assert fr.severity == "high"  # |weight| 0.9 >= 0.7
    assert fr.recommendation_priority == "P1"
    assert any(cp.voter_id == "a3" for cp in fr.counter_proofs)


def test_unrefuted_rejection_without_counter_proof_is_p0() -> None:
    votes = [_vote("a2", -0.8)]  # no written counter
    history = [_round(0, "a1", 2.0, 1.5, committed=False, votes=votes)]
    report = DisagreementForensics().analyze(history)
    fr = report.rounds[0]
    assert fr.blocker == "UNREFUTED_REJECTION"
    assert fr.recommendation_priority == "P0"
    assert "without a written counter-proof" in fr.recommendation


def test_below_threshold_no_rejections_near_miss_is_p2_low_severity() -> None:
    history = [_round(0, "a1", 1.4, 1.5, committed=False, votes=[_vote("a2", 0.4)])]
    report = DisagreementForensics().analyze(history)
    fr = report.rounds[0]
    assert fr.blocker == "BELOW_THRESHOLD"
    assert fr.severity == "low"
    assert fr.recommendation_priority == "P2"
    assert fr.primary_blame == "a1"


def test_below_threshold_with_rejections_promotes_next_view_leader() -> None:
    votes = [_vote("a2", -0.6, counter="bad proof"), _vote("a3", -0.3)]
    history = [_round(0, "a1", 0.5, 1.5, committed=False, votes=votes)]
    report = DisagreementForensics().analyze(history)
    fr = report.rounds[0]
    assert fr.blocker == "BELOW_THRESHOLD"
    assert fr.next_view_leader == "a2"
    assert fr.primary_blame == "a2"
    assert fr.recommendation_priority == "P1"


def test_below_threshold_far_miss_no_rejections_is_p0_calibration_issue() -> None:
    history = [_round(0, "a1", 0.2, 1.5, committed=False, votes=[_vote("a2", 0.1)])]
    report = DisagreementForensics().analyze(history)
    fr = report.rounds[0]
    assert fr.blocker == "BELOW_THRESHOLD"
    assert fr.severity == "high"
    assert fr.recommendation_priority == "P0"
    assert "confidence collapsed" in fr.recommendation


def test_chronic_blocker_pattern_fires_when_same_voter_rejects_twice() -> None:
    votes_a = [_vote("a2", -0.7, counter="c")]
    votes_b = [_vote("a2", -0.6, counter="c2")]
    history = [
        _round(0, "a1", 1.6, 1.5, committed=False, votes=votes_a),
        _round(1, "a3", 1.7, 1.5, committed=False, votes=votes_b),
    ]
    report = DisagreementForensics().analyze(history)
    codes = {p.code for p in report.patterns}
    assert "CHRONIC_BLOCKER" in codes
    chronic = next(p for p in report.patterns if p.code == "CHRONIC_BLOCKER")
    assert chronic.priority == "P0"
    assert "a2" in chronic.suspects


def test_calibration_collapse_fires_on_below_threshold_no_rejection_majority() -> None:
    history = [
        _round(0, "a1", 1.0, 1.5, committed=False, votes=[_vote("a2", 0.3)]),
        _round(1, "a2", 0.9, 1.5, committed=False, votes=[_vote("a1", 0.2)]),
    ]
    report = DisagreementForensics().analyze(history)
    codes = {p.code for p in report.patterns}
    assert "CALIBRATION_COLLAPSE" in codes


def test_threshold_too_high_fires_on_near_misses_no_rejections() -> None:
    history = [
        _round(0, "a1", 1.4, 1.5, committed=False, votes=[_vote("a2", 0.3)]),
        _round(1, "a2", 1.45, 1.5, committed=False, votes=[_vote("a1", 0.3)]),
    ]
    report = DisagreementForensics().analyze(history)
    codes = {p.code for p in report.patterns}
    assert "THRESHOLD_TOO_HIGH" in codes


def test_threshold_too_high_suppressed_when_any_rejection_exists() -> None:
    history = [
        _round(0, "a1", 1.4, 1.5, committed=False, votes=[_vote("a2", -0.3)]),
        _round(1, "a2", 1.45, 1.5, committed=False, votes=[_vote("a1", 0.3)]),
    ]
    report = DisagreementForensics().analyze(history)
    codes = {p.code for p in report.patterns}
    assert "THRESHOLD_TOO_HIGH" not in codes


def test_all_commits_yields_clean_headline_and_no_patterns() -> None:
    history = [
        _round(0, "a1", 2.0, 1.5, committed=True),
        _round(1, "a2", 2.0, 1.5, committed=True),
    ]
    report = DisagreementForensics().analyze(history)
    assert report.failed_rounds == 0
    assert report.patterns == []
    assert "committed cleanly" in report.headline


def test_renderers_are_deterministic_and_contain_key_info() -> None:
    votes = [_vote("a2", -0.9, counter="contradiction")]
    history = [_round(0, "a1", 0.4, 1.5, committed=False, votes=votes)]
    report = DisagreementForensics().analyze(history)

    text1 = report.to_text()
    text2 = report.to_text()
    assert text1 == text2
    assert "a1" in text1 and "BELOW_THRESHOLD" in text1

    md = report.to_markdown()
    assert "Disagreement Forensics" in md
    assert "Round 0" in md
    assert "contradiction" in md

    js = report.to_json()
    parsed = json.loads(js)
    assert parsed["total_rounds"] == 1
    assert parsed["failed_rounds"] == 1
    assert parsed["rounds"][0]["leader_id"] == "a1"


def test_invalid_constructor_args_raise() -> None:
    with pytest.raises(ValueError):
        DisagreementForensics(chronic_blocker_min_rounds=0)
    with pytest.raises(ValueError):
        DisagreementForensics(calibration_collapse_overconfidence=0.0)
    with pytest.raises(ValueError):
        DisagreementForensics(calibration_collapse_overconfidence=1.5)
    with pytest.raises(ValueError):
        DisagreementForensics(threshold_too_high_close_margin=-0.1)


@pytest.mark.asyncio
async def test_integrates_with_real_engine_history() -> None:
    from src.core.protocol import MBFTEngine
    from src.network.simulator import build_demo_swarm

    engine = MBFTEngine(
        agents=build_demo_swarm(),
        threshold=3.0,  # intentionally high
        max_rounds=3,
        slash_factor=0.5,
    )
    await engine.run("demo")
    report = DisagreementForensics().analyze(engine.history)
    assert report.total_rounds == len(engine.history)
    assert report.total_rounds >= 1
    # should at least produce per-round verdicts
    assert len(report.rounds) == report.total_rounds
