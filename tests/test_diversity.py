"""Tests for the Consensus Diversity Index module.

Covers:
- Shannon, Simpson, confidence-spread, Gini, herd-behavior metrics
- Composite groupthink score across bounds
- DiversityAnalysis aggregation, diagnosis, dict serialization
- analyze_rounds end-to-end against synthetic RoundResult sequences
- Recommendation generation across groupthink regimes
- HTML report rendering (non-empty, well-formed)
- Agent classification helper
"""
from __future__ import annotations

import math

import pytest

from src.core.state import RoundResult, Vote
from src.diversity import (
    DiversityAnalysis,
    _classify_agent,
    _generate_recommendations,
    analyze_rounds,
    confidence_spread,
    generate_html_report,
    gini_coefficient,
    groupthink_score,
    herd_behavior_index,
    shannon_diversity,
    simpson_diversity,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _vote(voter: str, weight: float, target: str = "p1") -> Vote:
    return Vote(voter_id=voter, target_proposal_id=target, weight=weight)


def _round(
    *,
    index: int,
    leader: str,
    committed: bool,
    aggregate: float,
    votes: list[Vote],
    threshold: float = 1.0,
) -> RoundResult:
    return RoundResult(
        round_index=index,
        leader_id=leader,
        committed_solution="X" if committed else None,
        aggregate_weight=aggregate,
        threshold=threshold,
        votes=votes,
    )


# --------------------------------------------------------------------------- #
# Pure-metric tests
# --------------------------------------------------------------------------- #

class TestShannon:
    def test_empty_returns_zero(self) -> None:
        assert shannon_diversity([]) == 0.0

    def test_single_bin_zero_entropy(self) -> None:
        # All weights collapse into the same "moderate_accept" bin.
        h = shannon_diversity([0.5, 0.55, 0.6, 0.5])
        assert h == pytest.approx(0.0, abs=1e-9)

    def test_uniform_across_two_bins(self) -> None:
        # Two distinct bins, 50/50 -> entropy = 1 bit.
        h = shannon_diversity([0.5, 0.5, -0.5, -0.5])
        assert h == pytest.approx(1.0, abs=1e-9)

    def test_uniform_five_bins_is_log2_5(self) -> None:
        # One weight per bin.
        weights = [-0.9, -0.2, 0.1, 0.5, 0.9]
        h = shannon_diversity(weights)
        assert h == pytest.approx(math.log2(5), abs=1e-9)


class TestSimpson:
    def test_too_few_returns_zero(self) -> None:
        assert simpson_diversity([]) == 0.0
        assert simpson_diversity([0.4]) == 0.0

    def test_identical_values(self) -> None:
        # All bin to the same key -> 1 - 1 = 0.
        assert simpson_diversity([0.4, 0.4, 0.4, 0.4]) == pytest.approx(0.0)

    def test_all_distinct_max(self) -> None:
        # n distinct bins, each count 1 -> sum(c*(c-1)) == 0.
        assert simpson_diversity([0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)

    def test_in_unit_interval(self) -> None:
        v = simpson_diversity([0.1, 0.1, 0.5, 0.5, 0.9])
        assert 0.0 <= v <= 1.0


class TestConfidenceSpread:
    def test_too_few_returns_zero(self) -> None:
        assert confidence_spread([]) == 0.0
        assert confidence_spread([0.5]) == 0.0

    def test_zero_mean_returns_zero(self) -> None:
        # Mean is zero -> avoid div-by-zero, return 0.
        assert confidence_spread([0.0, 0.0, 0.0]) == 0.0

    def test_constant_values_zero_spread(self) -> None:
        assert confidence_spread([0.7, 0.7, 0.7]) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        # values=[0.2, 0.4, 0.6, 0.8] mean=0.5
        # var = (0.09+0.01+0.01+0.09)/4 = 0.05; std=sqrt(0.05); cv=std/0.5
        cv = confidence_spread([0.2, 0.4, 0.6, 0.8])
        assert cv == pytest.approx(math.sqrt(0.05) / 0.5, abs=1e-9)


class TestGiniCoefficient:
    def test_perfect_equality_zero(self) -> None:
        # Delegates to shared gini helper.
        assert gini_coefficient([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-9)

    def test_maximum_inequality(self) -> None:
        # All zero except one large value -> high gini, in (0, 1].
        v = gini_coefficient([0.0, 0.0, 0.0, 10.0])
        assert v > 0.5


class TestHerdBehavior:
    def test_empty_zero(self) -> None:
        assert herd_behavior_index([]) == 0.0

    def test_skips_empty_round(self) -> None:
        assert herd_behavior_index([[]]) == 0.0

    def test_unanimous_positive_is_one(self) -> None:
        # Everyone matches the majority direction.
        assert herd_behavior_index([[0.5, 0.7, 0.9]]) == pytest.approx(1.0)

    def test_one_dissenter(self) -> None:
        # 3 positive + 1 negative, majority positive -> 3/4.
        assert herd_behavior_index([[0.5, 0.5, 0.5, -0.5]]) == pytest.approx(0.75)

    def test_tie_treated_as_majority_positive(self) -> None:
        # positives == negatives -> code treats this as majority_positive=True.
        # 2 positive + 2 negative -> 2/4 = 0.5.
        assert herd_behavior_index([[0.5, 0.5, -0.5, -0.5]]) == pytest.approx(0.5)

    def test_aggregates_across_rounds(self) -> None:
        rounds = [[0.5, 0.5], [-0.5, -0.5]]
        # round1: majority positive, both match -> 2/2
        # round2: majority negative, both match -> 2/2
        assert herd_behavior_index(rounds) == pytest.approx(1.0)


class TestGroupthinkScore:
    def test_perfect_echo_chamber(self) -> None:
        # Zero diversity, zero dissent, zero spread, max herd.
        s = groupthink_score(0.0, 0.0, 0.0, 0.0, 1.0)
        # 0.25*1 + 0.20*1 + 0.15*1 + 0.20*1 + 0.20*1 = 1.0
        assert s == pytest.approx(1.0)

    def test_perfectly_diverse(self) -> None:
        max_shannon = math.log2(5)
        # 0.25*0 + 0.20*0 + 0.15*0 + 0.20*0 + 0.20*0 = 0.0
        s = groupthink_score(max_shannon, 1.0, 1.0, 1.0, 0.0)
        assert s == pytest.approx(0.0, abs=1e-9)

    def test_clipped_to_unit_interval(self) -> None:
        s_low = groupthink_score(-100.0, -1.0, -1.0, -1.0, -1.0)
        s_high = groupthink_score(0.0, 0.0, 0.0, 0.0, 5.0)
        assert 0.0 <= s_low <= 1.0
        assert 0.0 <= s_high <= 1.0

    def test_returns_rounded_to_four_dp(self) -> None:
        s = groupthink_score(0.123456, 0.5, 0.5, 0.5, 0.5)
        # Verify at most 4 decimal places worth of precision.
        assert s == pytest.approx(round(s, 4))


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #

class TestClassifyAgent:
    @pytest.mark.parametrize(
        "avg,rej,expected",
        [
            (0.0, 0.6, "Contrarian"),
            (0.5, 0.3, "Skeptic"),
            (0.8, 0.05, "Enthusiast"),
            (0.5, 0.05, "Moderate"),
            (0.1, 0.05, "Cautious"),
            # Boundary: rejection > 0.5 wins even with positive avg vote.
            (0.9, 0.6, "Contrarian"),
        ],
    )
    def test_classification(self, avg: float, rej: float, expected: str) -> None:
        assert _classify_agent(avg, rej) == expected


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #

class TestRecommendations:
    def test_healthy_swarm(self) -> None:
        # gt<0.3, sh>1.0, dr>0.2, lg<0.6, hi<0.8, balanced roles
        profiles = {
            "a1": {"role": "Contrarian"},
            "a2": {"role": "Enthusiast"},
            "a3": {"role": "Moderate"},
            "a4": {"role": "Skeptic"},
        }
        recs = _generate_recommendations(0.1, 1.5, 0.4, 0.3, 0.5, profiles)
        # All gates closed -> single "looks healthy" message.
        assert len(recs) == 1
        assert "healthy" in recs[0].lower()

    def test_critical_echo_chamber(self) -> None:
        profiles = {"a1": {"role": "Enthusiast"}, "a2": {"role": "Enthusiast"}}
        recs = _generate_recommendations(0.85, 0.4, 0.05, 0.8, 0.95, profiles)
        # Many warnings, no healthy line.
        joined = " ".join(recs).lower()
        assert "echo chamber" in joined
        assert "leadership" in joined
        assert "herd" in joined
        assert "low role diversity" in joined
        assert not any("looks healthy" in r.lower() for r in recs)

    def test_low_role_diversity_warning(self) -> None:
        # Only 2 distinct roles -> "Low role diversity"
        profiles = {f"a{i}": {"role": "Moderate"} for i in range(4)}
        profiles["b"] = {"role": "Skeptic"}
        recs = _generate_recommendations(0.2, 1.5, 0.4, 0.3, 0.5, profiles)
        assert any("low role diversity" in r.lower() for r in recs)


# --------------------------------------------------------------------------- #
# analyze_rounds end-to-end
# --------------------------------------------------------------------------- #

class TestAnalyzeRounds:
    def test_empty_history(self) -> None:
        analysis = analyze_rounds([], reputation={})
        assert analysis.shannon == 0.0
        assert analysis.simpson == 0.0
        assert analysis.dissent_ratio == 0.0
        assert analysis.per_round == []
        assert analysis.agent_profiles == {}
        # Even with empty data, we should get a recommendations list.
        assert isinstance(analysis.recommendations, list)
        assert analysis.recommendations  # not empty

    def test_synthetic_consensus(self) -> None:
        # 3 rounds, same leader, unanimous positive -> high groupthink.
        rounds = [
            _round(
                index=i,
                leader="leader",
                committed=True,
                aggregate=2.4,
                votes=[
                    _vote("a1", 0.8),
                    _vote("a2", 0.8),
                    _vote("a3", 0.8),
                ],
            )
            for i in range(3)
        ]
        analysis = analyze_rounds(rounds, reputation={"a1": 1.0, "a2": 1.0, "a3": 1.0})
        assert analysis.shannon == pytest.approx(0.0)
        assert analysis.dissent_ratio == 0.0
        assert analysis.leader_gini == pytest.approx(0.0)  # only one leader, even distribution
        assert analysis.groupthink > 0.5
        assert len(analysis.per_round) == 3
        # Every agent should appear in the profiles.
        assert set(analysis.agent_profiles.keys()) == {"a1", "a2", "a3"}
        for prof in analysis.agent_profiles.values():
            assert prof["avg_vote"] == pytest.approx(0.8)
            assert prof["rejection_rate"] == 0.0
            assert prof["reputation"] == 1.0

    def test_dissent_counted_when_any_negative(self) -> None:
        rounds = [
            _round(
                index=0,
                leader="L1",
                committed=False,
                aggregate=0.5,
                votes=[_vote("a1", 0.6), _vote("a2", -0.4)],
            ),
            _round(
                index=1,
                leader="L2",
                committed=True,
                aggregate=1.2,
                votes=[_vote("a1", 0.6), _vote("a2", 0.6)],
            ),
        ]
        analysis = analyze_rounds(rounds, reputation={})
        assert analysis.dissent_ratio == pytest.approx(0.5)

    def test_per_round_records_leader_and_committed(self) -> None:
        rounds = [
            _round(
                index=7,
                leader="lead-x",
                committed=True,
                aggregate=1.234,
                votes=[_vote("v1", 0.4), _vote("v2", -0.2)],
                threshold=1.0,
            ),
        ]
        analysis = analyze_rounds(rounds, reputation={})
        row = analysis.per_round[0]
        assert row["round"] == 7
        assert row["leader"] == "lead-x"
        assert row["committed"] is True
        assert row["aggregate"] == 1.234
        # vote_spread: max(0.4, -0.2) - min(...) = 0.6
        assert row["vote_spread"] == pytest.approx(0.6)

    def test_default_reputation_when_missing(self) -> None:
        # An agent with no reputation entry should default to 1.0.
        rounds = [
            _round(
                index=0,
                leader="L",
                committed=True,
                aggregate=1.0,
                votes=[_vote("ghost", 0.5)],
            )
        ]
        analysis = analyze_rounds(rounds, reputation={})
        assert analysis.agent_profiles["ghost"]["reputation"] == 1.0


# --------------------------------------------------------------------------- #
# DiversityAnalysis container
# --------------------------------------------------------------------------- #

class TestDiversityAnalysisContainer:
    def _make(self, gt: float) -> DiversityAnalysis:
        return DiversityAnalysis(
            shannon=1.0,
            simpson=0.5,
            conf_spread=0.2,
            dissent_ratio=0.3,
            leader_gini=0.4,
            herd_index=0.5,
            groupthink=gt,
            per_round=[{"round": 0}],
            recommendations=["rec1"],
            agent_profiles={"a": {"role": "Moderate"}},
        )

    @pytest.mark.parametrize(
        "gt,fragment",
        [
            (0.75, "CRITICAL"),
            (0.55, "WARNING"),
            (0.35, "HEALTHY"),
            (0.10, "EXCELLENT"),
        ],
    )
    def test_diagnosis_thresholds(self, gt: float, fragment: str) -> None:
        analysis = self._make(gt)
        assert fragment in analysis.to_dict()["diagnosis"]

    def test_to_dict_shape(self) -> None:
        data = self._make(0.4).to_dict()
        assert set(data.keys()) >= {
            "metrics", "diagnosis", "per_round",
            "agent_profiles", "recommendations",
        }
        assert data["metrics"]["groupthink_score"] == 0.4
        assert data["metrics"]["shannon_diversity"] == 1.0


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

class TestHtmlReport:
    def test_html_contains_expected_sections(self) -> None:
        rounds = [
            _round(
                index=0,
                leader="leader<&>",  # exercises html escaping
                committed=True,
                aggregate=1.5,
                votes=[_vote("a1", 0.6), _vote("a2", -0.4)],
            )
        ]
        analysis = analyze_rounds(rounds, reputation={"a1": 1.0})
        report = generate_html_report(analysis)
        assert report.startswith("<!DOCTYPE html>")
        assert "Diversity" in report
        # Leader name appears escaped in the rendered table rows.
        assert "leader&lt;&amp;&gt;" in report
        # Each recommendation surfaces in the report (first token is enough,
        # since html.escape may transform punctuation but leaves words intact).
        for rec in analysis.recommendations:
            assert rec.split()[0] in report or rec in report
