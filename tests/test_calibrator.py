"""Unit tests for src.calibrator pure metric functions.

Covers ``compute_calibration`` (binning, ECE/MCE, Brier score, edge cases)
and ``diagnose_agent`` (label/recommendation ladder for the well-calibrated,
over/underconfident, noisy, and degenerate cases). These functions had no
direct coverage before this file - the rest of the calibrator module is
exercised end-to-end via the benchmark runner, which makes regressions in
the binning math hard to attribute.
"""
from __future__ import annotations

import math
import random

import pytest

from src.calibrator import (
    AgentCalibration,
    CalibrationBin,
    TrialRecord,
    compute_calibration,
    diagnose_agent,
)


def _record(agent_id: str, conf: float, correct: bool) -> TrialRecord:
    return TrialRecord(
        agent_id=agent_id,
        confidence=conf,
        is_correct=correct,
        answer="x" if correct else "y",
        ground_truth="x",
    )


# ---------------------------------------------------------------- #
# compute_calibration
# ---------------------------------------------------------------- #


class TestComputeCalibration:
    def test_empty_records_returns_zero_metrics_and_full_bin_skeleton(self) -> None:
        bins, ece, mce, brier = compute_calibration([], n_bins=10)
        assert ece == 0.0
        assert mce == 0.0
        assert brier == 0.0
        assert len(bins) == 10
        # Every bin should be empty but well-formed (lo/hi, count=0).
        assert [b.count for b in bins] == [0] * 10
        # Bin edges should partition [0, 1] exactly.
        assert bins[0].bin_start == pytest.approx(0.0)
        assert bins[-1].bin_end == pytest.approx(1.0)
        for i in range(len(bins) - 1):
            assert bins[i].bin_end == pytest.approx(bins[i + 1].bin_start)

    def test_perfect_calibration_yields_zero_ece_and_low_brier(self) -> None:
        # 100 trials at conf=0.7, 70 correct - calibration is exact.
        records = [_record("a", 0.7, True) for _ in range(70)]
        records += [_record("a", 0.7, False) for _ in range(30)]
        _, ece, mce, brier = compute_calibration(records, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-12)
        assert mce == pytest.approx(0.0, abs=1e-12)
        # Brier = mean((0.7 - y)^2) = 0.7*0.3^2 + 0.3*0.7^2 = 0.21
        assert brier == pytest.approx(0.21, abs=1e-12)

    def test_systematic_overconfidence_shows_up_in_ece(self) -> None:
        # Always reports 0.9 confidence but is only right 50% of the time.
        records = [_record("a", 0.9, True) for _ in range(50)]
        records += [_record("a", 0.9, False) for _ in range(50)]
        bins, ece, mce, brier = compute_calibration(records, n_bins=10)
        # All 100 trials land in the [0.9, 1.0) bin.
        nonempty = [b for b in bins if b.count > 0]
        assert len(nonempty) == 1
        assert nonempty[0].count == 100
        assert nonempty[0].mean_confidence == pytest.approx(0.9)
        assert nonempty[0].accuracy == pytest.approx(0.5)
        # ECE = |0.9 - 0.5| weighted = 0.4. MCE equals it (one bin).
        assert ece == pytest.approx(0.4, abs=1e-12)
        assert mce == pytest.approx(0.4, abs=1e-12)
        assert brier == pytest.approx(0.41, abs=1e-12)

    def test_confidence_equal_to_one_lands_in_last_bin(self) -> None:
        # int(1.0 * n_bins) == n_bins, which would be out of range; the
        # implementation clamps to the last bin. Pin that contract.
        records = [_record("a", 1.0, True), _record("a", 1.0, False)]
        bins, _, _, _ = compute_calibration(records, n_bins=10)
        assert bins[-1].count == 2
        assert sum(b.count for b in bins[:-1]) == 0

    def test_confidence_zero_lands_in_first_bin(self) -> None:
        records = [_record("a", 0.0, False), _record("a", 0.0, False)]
        bins, ece, _, brier = compute_calibration(records, n_bins=10)
        assert bins[0].count == 2
        assert bins[0].accuracy == pytest.approx(0.0)
        # ECE should be ~0 (confidence and accuracy both 0).
        assert ece == pytest.approx(0.0, abs=1e-12)
        assert brier == pytest.approx(0.0, abs=1e-12)

    def test_brier_score_matches_manual_computation_on_mixed_records(self) -> None:
        records = [
            _record("a", 0.9, True),   # (0.9-1)^2 = 0.01
            _record("a", 0.1, False),  # (0.1-0)^2 = 0.01
            _record("a", 0.6, True),   # (0.6-1)^2 = 0.16
            _record("a", 0.4, False),  # (0.4-0)^2 = 0.16
        ]
        expected = (0.01 + 0.01 + 0.16 + 0.16) / 4
        _, _, _, brier = compute_calibration(records, n_bins=10)
        assert brier == pytest.approx(expected, abs=1e-12)

    def test_mce_picks_the_worst_bin_not_the_average(self) -> None:
        # Bin A (0.0-0.1): perfectly calibrated (ECE contribution 0).
        # Bin B (0.9-1.0): badly miscalibrated (gap 0.9).
        records = [_record("a", 0.05, False) for _ in range(90)]
        records += [_record("a", 0.95, False) for _ in range(10)]
        _, ece, mce, _ = compute_calibration(records, n_bins=10)
        # MCE is the worst per-bin |conf - acc|, here ~0.95.
        assert mce == pytest.approx(0.95, abs=1e-12)
        # ECE is the count-weighted mean: 90*0.05 + 10*0.95 over 100 = 0.14
        assert ece == pytest.approx(0.14, abs=1e-12)

    def test_n_bins_parameter_is_respected(self) -> None:
        records = [_record("a", 0.5, True) for _ in range(5)]
        bins5, _, _, _ = compute_calibration(records, n_bins=5)
        bins20, _, _, _ = compute_calibration(records, n_bins=20)
        assert len(bins5) == 5
        assert len(bins20) == 20
        # All records have conf=0.5; they should all land in the same bin.
        assert sum(b.count for b in bins5) == 5
        assert sum(b.count for b in bins20) == 5

    def test_randomized_data_brier_in_unit_interval(self) -> None:
        rng = random.Random(42)
        records = []
        for _ in range(500):
            conf = rng.random()
            correct = rng.random() < conf  # roughly well-calibrated
            records.append(_record("a", conf, correct))
        bins, ece, mce, brier = compute_calibration(records, n_bins=10)
        # Sanity guards on metric ranges.
        assert 0.0 <= ece <= 1.0
        assert 0.0 <= mce <= 1.0
        assert 0.0 <= brier <= 1.0
        # All records accounted for.
        assert sum(b.count for b in bins) == 500
        # Well-calibrated synthetic data shouldn't blow up MCE.
        assert ece < 0.15

    def test_single_record_does_not_crash(self) -> None:
        records = [_record("a", 0.42, True)]
        bins, ece, mce, brier = compute_calibration(records, n_bins=10)
        assert sum(b.count for b in bins) == 1
        # Single record: brier = (0.42-1)^2 = 0.3364
        assert brier == pytest.approx(0.3364, abs=1e-12)
        # Only one populated bin so MCE == ECE (single nonzero contribution).
        assert mce == pytest.approx(ece, abs=1e-12)


# ---------------------------------------------------------------- #
# diagnose_agent
# ---------------------------------------------------------------- #


def _empty_bins(n: int = 10) -> list[CalibrationBin]:
    return [
        CalibrationBin(i / n, (i + 1) / n, (i + 0.5) / n, 0.0, 0)
        for i in range(n)
    ]


class TestDiagnoseAgent:
    def test_no_records_returns_no_data_diagnosis(self) -> None:
        result = diagnose_agent("a1", [], _empty_bins(), 0.0, 0.0, 0.0)
        assert isinstance(result, AgentCalibration)
        assert result.agent_id == "a1"
        assert result.diagnosis == "no-data"
        assert result.recommendations == ["No data."]
        assert result.accuracy == 0
        assert result.mean_confidence == 0

    def test_well_calibrated_agent_has_no_action_recommendation(self) -> None:
        # Mean conf ~= accuracy, low ECE/MCE/Brier.
        records = [_record("a", 0.8, True) for _ in range(80)]
        records += [_record("a", 0.8, False) for _ in range(20)]
        bins, ece, mce, brier = compute_calibration(records, n_bins=10)
        result = diagnose_agent("a", records, bins, ece, mce, brier)
        assert result.diagnosis == "well-calibrated"
        assert result.accuracy == pytest.approx(0.8)
        assert result.mean_confidence == pytest.approx(0.8)
        assert any("well-calibrated" in r.lower() for r in result.recommendations)

    def test_overconfident_agent_labeled_overconfident(self) -> None:
        # Confidence 0.9 but only 50% accuracy - 40pp gap >> 15pp threshold.
        records = [_record("a", 0.9, True) for _ in range(50)]
        records += [_record("a", 0.9, False) for _ in range(50)]
        bins, ece, mce, brier = compute_calibration(records, n_bins=10)
        result = diagnose_agent("a", records, bins, ece, mce, brier)
        assert result.diagnosis == "overconfident"
        # Should explicitly recommend temperature scaling / Platt calibration.
        assert any(
            "temperature scaling" in r.lower() or "platt" in r.lower()
            for r in result.recommendations
        )
        # Every trial has conf > accuracy+0.1 (0.9 > 0.6), so over_ratio == 1.0.
        assert result.overconfidence_ratio == pytest.approx(1.0)
        assert result.underconfidence_ratio == pytest.approx(0.0)

    def test_underconfident_agent_labeled_underconfident(self) -> None:
        # Confidence 0.3 but 90% accuracy - cautious agent.
        records = [_record("a", 0.3, True) for _ in range(90)]
        records += [_record("a", 0.3, False) for _ in range(10)]
        bins, ece, mce, brier = compute_calibration(records, n_bins=10)
        result = diagnose_agent("a", records, bins, ece, mce, brier)
        assert result.diagnosis == "underconfident"
        assert result.underconfidence_ratio == pytest.approx(1.0)
        assert result.overconfidence_ratio == pytest.approx(0.0)

    def test_poor_ece_triggers_poorly_calibrated_warning(self) -> None:
        # Force a high ECE via the parameter to verify the recommendation
        # ladder reads ECE without recomputing it.
        records = [_record("a", 0.8, True), _record("a", 0.8, False)]
        bins = _empty_bins()
        # Pre-set the ECE that diagnose_agent will report on.
        result = diagnose_agent("a", records, bins, ece=0.25, mce=0.1, brier=0.1)
        assert any("ECE=" in r and "calibration tuning" in r.lower()
                   for r in result.recommendations)

    def test_high_brier_adds_brier_recommendation(self) -> None:
        records = [_record("a", 0.5, True)]
        result = diagnose_agent("a", records, _empty_bins(),
                                ece=0.01, mce=0.01, brier=0.45)
        assert any("brier" in r.lower() for r in result.recommendations)

    def test_high_mce_adds_extreme_bin_recommendation(self) -> None:
        records = [_record("a", 0.5, True)]
        result = diagnose_agent("a", records, _empty_bins(),
                                ece=0.05, mce=0.5, brier=0.1)
        assert any("MCE=" in r and "extreme" in r.lower()
                   for r in result.recommendations)

    def test_recommendations_are_unique_per_distinct_warning(self) -> None:
        # An agent that's overconfident + has high ECE + high Brier should get
        # multiple distinct recommendations, not the generic "no action needed".
        records = [_record("a", 0.95, True) for _ in range(30)]
        records += [_record("a", 0.95, False) for _ in range(70)]
        bins, ece, mce, brier = compute_calibration(records, n_bins=10)
        result = diagnose_agent("a", records, bins, ece, mce, brier)
        # At least an ECE warning + the overconfidence warning.
        assert len(result.recommendations) >= 2
        joined = " ".join(result.recommendations).lower()
        assert "no action needed" not in joined

    def test_mean_confidence_matches_simple_average(self) -> None:
        # Verifies the in-function single-pass aggregation matches a naive
        # mean - catches off-by-one accumulator bugs.
        confs = [0.1, 0.4, 0.7, 0.9]
        records = [_record("a", c, True) for c in confs]
        result = diagnose_agent("a", records, _empty_bins(),
                                ece=0.0, mce=0.0, brier=0.0)
        assert result.mean_confidence == pytest.approx(sum(confs) / len(confs))
        assert result.accuracy == pytest.approx(1.0)

    def test_returned_object_carries_through_metrics(self) -> None:
        # Wiring test: ece/mce/brier and bins should be returned verbatim.
        records = [_record("a", 0.5, True)]
        bins = _empty_bins(5)
        result = diagnose_agent("a", records, bins,
                                ece=0.123, mce=0.456, brier=0.789)
        assert result.bins is bins
        assert result.ece == 0.123
        assert result.mce == 0.456
        assert result.brier == 0.789
        assert math.isfinite(result.mean_confidence)
