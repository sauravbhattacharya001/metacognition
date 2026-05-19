"""Tests for ThresholdTuningAdvisor."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.core.state import RoundResult, Vote
from src.threshold_tuning_advisor import ThresholdTuningAdvisor


FIXED_NOW = datetime(2026, 5, 19, 8, 5, 0)


def _now():
    return FIXED_NOW


def _round(idx, leader, agg, thr, committed=True, votes=None, slashed=None):
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution="ans" if committed else None,
        aggregate_weight=agg,
        threshold=thr,
        votes=votes or [],
        slashed=slashed or [],
    )


def _vote(voter, prop_id, weight, counter=None):
    return Vote(voter_id=voter, target_proposal_id=prop_id, weight=weight, counter_proof=counter)


# ---------------------------------------------------------------------------
# Empty / minimal history
# ---------------------------------------------------------------------------


def test_empty_history_keeps_settings():
    advisor = ThresholdTuningAdvisor(now_fn=_now)
    report = advisor.analyze([], threshold=1.5, slash_factor=0.5)
    assert report.rounds_observed == 0
    assert report.recommendation.recommended_threshold == 1.5
    assert report.recommendation.recommended_slash_factor == 0.5
    assert report.grade == "A"
    assert any("INSUFFICIENT_HISTORY" in i for i in report.insights)


def test_invalid_risk_appetite_raises():
    advisor = ThresholdTuningAdvisor(now_fn=_now)
    with pytest.raises(ValueError):
        advisor.analyze([], threshold=1.0, slash_factor=0.5, risk_appetite="paranoid")


# ---------------------------------------------------------------------------
# Healthy baseline
# ---------------------------------------------------------------------------


def test_healthy_history_suggests_no_change():
    # Include rejection votes so NO_REJECTIONS_OBSERVED does not fire.
    history = [
        _round(
            i,
            "A",
            agg=2.5,
            thr=1.5,
            committed=True,
            votes=[_vote("B", "p", -0.2, counter="cp")] if i % 2 == 0 else [],
        )
        for i in range(6)
    ]
    advisor = ThresholdTuningAdvisor(now_fn=_now)
    report = advisor.analyze(history, threshold=1.5, slash_factor=0.5)
    assert report.grade in {"A", "B"}
    # No change to threshold or slash on healthy
    assert report.recommendation.threshold_delta == 0.0
    assert report.recommendation.slash_factor_delta == 0.0
    assert any(a.id == "HOLD_CURRENT_TUNING" for a in report.playbook)


# ---------------------------------------------------------------------------
# Threshold too high — many close failures
# ---------------------------------------------------------------------------


def test_close_failures_recommend_lower_threshold():
    history = [
        _round(i, "A", agg=1.2, thr=1.5, committed=False)
        for i in range(5)
    ] + [
        _round(5, "A", agg=2.0, thr=1.5, committed=True),
    ]
    advisor = ThresholdTuningAdvisor(now_fn=_now)
    report = advisor.analyze(history, threshold=1.5, slash_factor=0.5)
    codes = {f.code for f in report.findings}
    assert "THRESHOLD_LIKELY_TOO_HIGH" in codes
    assert report.recommendation.recommended_threshold < 1.5
    assert any(a.id == "LOWER_THRESHOLD" for a in report.playbook)


# ---------------------------------------------------------------------------
# Threshold too low — narrow passes with rejections
# ---------------------------------------------------------------------------


def test_narrow_passes_with_rejection_recommend_raise():
    # 5 commits within 0.5 of threshold AND each has a rejection
    history = [
        _round(
            i,
            "A",
            agg=1.6,
            thr=1.5,
            committed=True,
            votes=[_vote("B", "p", -0.5, counter="cp")],
        )
        for i in range(5)
    ]
    advisor = ThresholdTuningAdvisor(now_fn=_now)
    report = advisor.analyze(history, threshold=1.5, slash_factor=0.5)
    codes = {f.code for f in report.findings}
    assert "THRESHOLD_LIKELY_TOO_LOW" in codes
    assert report.recommendation.recommended_threshold > 1.5
    assert any(a.id == "RAISE_THRESHOLD" for a in report.playbook)


# ---------------------------------------------------------------------------
# Unrefuted rejection vetoes — raise slash factor
# ---------------------------------------------------------------------------


def test_unrefuted_rejection_raises_slash_factor():
    # aggregate >= threshold but uncommitted (rejection blocked it)
    history = [
        _round(
            i,
            "A",
            agg=2.0,
            thr=1.5,
            committed=False,
            votes=[_vote("B", "p", -0.5, counter="cp")],
        )
        for i in range(3)
    ]
    advisor = ThresholdTuningAdvisor(now_fn=_now)
    report = advisor.analyze(history, threshold=1.5, slash_factor=0.5)
    codes = {f.code for f in report.findings}
    assert "UNREFUTED_REJECTION_VETOES" in codes
    assert report.recommendation.recommended_slash_factor > 0.5
    raise_action = next(a for a in report.playbook if a.id == "RAISE_SLASH_FACTOR")
    assert raise_action.priority == "P0"


# ---------------------------------------------------------------------------
# Slash runaway — lower slash_factor
# ---------------------------------------------------------------------------


def test_slash_runaway_lowers_slash_factor():
    history = [
        _round(i, "A", agg=2.0, thr=1.5, committed=True, slashed=["X", "Y"])
        for i in range(4)
    ]
    rep = {"X": 0.0, "Y": -0.1, "Z": 1.0}
    advisor = ThresholdTuningAdvisor(now_fn=_now)
    report = advisor.analyze(
        history, threshold=1.5, slash_factor=0.8, reputation=rep
    )
    codes = {f.code for f in report.findings}
    assert "SLASH_RUNAWAY" in codes
    assert report.recommendation.recommended_slash_factor < 0.8
    lower = next(a for a in report.playbook if a.id == "LOWER_SLASH_FACTOR")
    assert lower.priority == "P0"


# ---------------------------------------------------------------------------
# No rejections at all
# ---------------------------------------------------------------------------


def test_no_rejections_flags_echo_chamber():
    history = [
        _round(i, "A", agg=2.0, thr=1.5, committed=True)
        for i in range(6)
    ]
    advisor = ThresholdTuningAdvisor(now_fn=_now)
    report = advisor.analyze(history, threshold=1.5, slash_factor=0.5)
    codes = {f.code for f in report.findings}
    # healthy may also fire; both can coexist
    assert "NO_REJECTIONS_OBSERVED" in codes
    assert report.recommendation.recommended_slash_factor <= 0.5


# ---------------------------------------------------------------------------
# Risk appetite monotonicity: cautious moves less than aggressive
# ---------------------------------------------------------------------------


def test_risk_appetite_monotonicity():
    history = [
        _round(i, "A", agg=1.2, thr=1.5, committed=False)
        for i in range(5)
    ] + [_round(5, "A", agg=2.0, thr=1.5, committed=True)]

    advisor = ThresholdTuningAdvisor(now_fn=_now)
    cautious = advisor.analyze(
        history, threshold=1.5, slash_factor=0.5, risk_appetite="cautious"
    )
    balanced = advisor.analyze(
        history, threshold=1.5, slash_factor=0.5, risk_appetite="balanced"
    )
    aggressive = advisor.analyze(
        history, threshold=1.5, slash_factor=0.5, risk_appetite="aggressive"
    )

    # The recommended threshold goes DOWN, so cautious should be closer to current.
    assert cautious.recommendation.recommended_threshold >= balanced.recommendation.recommended_threshold
    assert balanced.recommendation.recommended_threshold >= aggressive.recommendation.recommended_threshold


# ---------------------------------------------------------------------------
# Recommendation bounded to +/- 50%
# ---------------------------------------------------------------------------


def test_recommendation_bounded_per_call():
    # Extreme failures: every round at 0.1 vs threshold 10
    history = [
        _round(i, "A", agg=0.1, thr=10.0, committed=False)
        for i in range(10)
    ]
    advisor = ThresholdTuningAdvisor(now_fn=_now)
    report = advisor.analyze(history, threshold=10.0, slash_factor=0.5)
    assert report.recommendation.recommended_threshold >= 5.0  # at most -50%
    assert report.recommendation.recommended_threshold <= 10.0


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def test_text_render_includes_verdict():
    history = [_round(i, "A", agg=2.0, thr=1.5, committed=True) for i in range(5)]
    report = ThresholdTuningAdvisor(now_fn=_now).analyze(
        history, threshold=1.5, slash_factor=0.5
    )
    txt = report.to_text()
    assert "VERDICT" in txt
    assert "Recommendation" in txt


def test_markdown_render_has_sections():
    history = [_round(i, "A", agg=2.0, thr=1.5, committed=True) for i in range(5)]
    report = ThresholdTuningAdvisor(now_fn=_now).analyze(
        history, threshold=1.5, slash_factor=0.5
    )
    md = report.to_markdown()
    assert "# Threshold Tuning Report" in md
    assert "## Summary" in md
    assert "## Recommendation" in md
    assert "## Playbook" in md
    assert "## Insights" in md


def test_json_render_is_byte_stable_and_parseable():
    history = [_round(i, "A", agg=2.0, thr=1.5, committed=True) for i in range(5)]
    advisor = ThresholdTuningAdvisor(now_fn=_now)
    r1 = advisor.analyze(history, threshold=1.5, slash_factor=0.5).to_json()
    r2 = advisor.analyze(history, threshold=1.5, slash_factor=0.5).to_json()
    assert r1 == r2
    parsed = json.loads(r1)
    assert parsed["rounds_observed"] == 5
    assert "recommendation" in parsed
    assert "playbook" in parsed


# ---------------------------------------------------------------------------
# Determinism: never mutates input
# ---------------------------------------------------------------------------


def test_does_not_mutate_history_or_reputation():
    history = [_round(i, "A", agg=1.2, thr=1.5, committed=False) for i in range(5)]
    rep = {"A": 1.0, "B": 0.0}
    original_history_dump = [r.model_dump() for r in history]
    original_rep = dict(rep)

    advisor = ThresholdTuningAdvisor(now_fn=_now)
    advisor.analyze(history, threshold=1.5, slash_factor=0.5, reputation=rep)

    assert [r.model_dump() for r in history] == original_history_dump
    assert rep == original_rep
