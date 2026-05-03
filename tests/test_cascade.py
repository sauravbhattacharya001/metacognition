"""Tests for Consensus Information Cascade Detector (src.cascade).

Tests cover:
- Data structures (CascadeSignal, AgentCascadeProfile, CascadeReport)
- Pure analysis functions (_diversity_ratio, _confidence_spread, _echo_index,
  _detect_flip_flops, _signal_abandonment, _cascade_velocity)
- Full cascade analysis (analyze_cascade)
- Signal generation (_generate_signals) for all 6 detection channels
- Edge cases: empty inputs, single-agent, single-round, all-identical
"""
from __future__ import annotations

import math
import statistics

import pytest

from src.core.state import Proposal, RoundResult, Vote
from src.cascade import (
    AgentCascadeProfile,
    CascadeReport,
    CascadeSignal,
    _cascade_velocity,
    _confidence_spread,
    _detect_flip_flops,
    _diversity_ratio,
    _echo_index,
    _generate_signals,
    _signal_abandonment,
    analyze_cascade,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vote(voter: str, weight: float, target: str = "p1",
          counter_proof: str | None = None) -> Vote:
    return Vote(voter_id=voter, target_proposal_id=target,
                weight=weight, counter_proof=counter_proof)


def _round(idx: int, leader: str, votes: list[Vote],
           committed: str | None = None,
           threshold: float = 0.5) -> RoundResult:
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution=committed,
        aggregate_weight=sum(v.weight for v in votes),
        threshold=threshold,
        votes=votes,
    )


def _proposal(agent: str, solution: str, conf: float = 0.7) -> Proposal:
    return Proposal(agent_id=agent, solution=solution,
                    proof=f"proof-{agent}", confidence=conf)


# ===================================================================
# _diversity_ratio
# ===================================================================

class TestDiversityRatio:
    def test_empty_proposals(self):
        assert _diversity_ratio([]) == 1.0

    def test_all_unique(self):
        props = [_proposal("a", "sol-A"), _proposal("b", "sol-B"),
                 _proposal("c", "sol-C")]
        assert _diversity_ratio(props) == pytest.approx(1.0)

    def test_all_identical(self):
        props = [_proposal("a", "X"), _proposal("b", "X"),
                 _proposal("c", "X")]
        assert _diversity_ratio(props) == pytest.approx(1 / 3)

    def test_partial_overlap(self):
        props = [_proposal("a", "A"), _proposal("b", "A"),
                 _proposal("c", "B")]
        assert _diversity_ratio(props) == pytest.approx(2 / 3)

    def test_single_proposal(self):
        assert _diversity_ratio([_proposal("a", "X")]) == 1.0


# ===================================================================
# _confidence_spread
# ===================================================================

class TestConfidenceSpread:
    def test_empty(self):
        assert _confidence_spread([]) == 0.0

    def test_single_value(self):
        assert _confidence_spread([0.5]) == 0.0

    def test_identical_values(self):
        assert _confidence_spread([0.7, 0.7, 0.7]) == 0.0

    def test_spread_calculation(self):
        vals = [0.1, 0.5, 0.9]
        assert _confidence_spread(vals) == pytest.approx(statistics.stdev(vals))

    def test_two_values(self):
        vals = [0.2, 0.8]
        assert _confidence_spread(vals) == pytest.approx(statistics.stdev(vals))


# ===================================================================
# _echo_index
# ===================================================================

class TestEchoIndex:
    def test_empty(self):
        assert _echo_index([]) == 0.0

    def test_all_positive(self):
        assert _echo_index([0.5, 0.3, 0.8]) == 1.0

    def test_all_negative(self):
        assert _echo_index([-0.5, -0.3, -0.8]) == 0.0

    def test_mixed(self):
        assert _echo_index([0.5, -0.3, 0.8, -0.1]) == 0.5

    def test_zero_treated_as_non_positive(self):
        assert _echo_index([0.0, 0.0, 0.5]) == pytest.approx(1 / 3)


# ===================================================================
# _detect_flip_flops
# ===================================================================

class TestDetectFlipFlops:
    def test_empty_history(self):
        assert _detect_flip_flops([]) == {}

    def test_single_round_no_flips(self):
        r = _round(0, "leader", [_vote("a", 0.5), _vote("b", -0.3)])
        assert _detect_flip_flops([r]) == {}

    def test_no_flip_same_sign(self):
        r1 = _round(0, "L", [_vote("a", 0.5)])
        r2 = _round(1, "L", [_vote("a", 0.7)])
        assert _detect_flip_flops([r1, r2]) == {}

    def test_flip_reject_to_accept_without_counter(self):
        r1 = _round(0, "L", [_vote("a", -0.5, counter_proof="cp")])
        r2 = _round(1, "L", [_vote("a", 0.5)])
        result = _detect_flip_flops([r1, r2])
        assert result == {"a": 1}

    def test_no_flip_if_counter_proof_present(self):
        r1 = _round(0, "L", [_vote("a", -0.5)])
        r2 = _round(1, "L", [_vote("a", 0.5, counter_proof="new-proof")])
        assert _detect_flip_flops([r1, r2]) == {}

    def test_accept_to_reject_not_counted(self):
        r1 = _round(0, "L", [_vote("a", 0.5)])
        r2 = _round(1, "L", [_vote("a", -0.5)])
        assert _detect_flip_flops([r1, r2]) == {}

    def test_multiple_flips(self):
        r1 = _round(0, "L", [_vote("a", -0.3)])
        r2 = _round(1, "L", [_vote("a", 0.3)])
        r3 = _round(2, "L", [_vote("a", -0.2)])
        r4 = _round(3, "L", [_vote("a", 0.4)])
        result = _detect_flip_flops([r1, r2, r3, r4])
        assert result == {"a": 2}

    def test_multiple_agents(self):
        r1 = _round(0, "L", [_vote("a", -0.3), _vote("b", 0.5)])
        r2 = _round(1, "L", [_vote("a", 0.3), _vote("b", -0.3)])
        r3 = _round(2, "L", [_vote("a", 0.4), _vote("b", 0.6)])
        flips = _detect_flip_flops([r1, r2, r3])
        assert flips.get("a", 0) == 1
        assert flips.get("b", 0) == 1


# ===================================================================
# _signal_abandonment
# ===================================================================

class TestSignalAbandonment:
    def test_no_proposals(self):
        assert _signal_abandonment([], "leader", []) == []

    def test_leader_not_found(self):
        props = [_proposal("a", "sol-A")]
        votes = [_vote("a", 0.5)]
        assert _signal_abandonment(props, "nonexistent", votes) == []

    def test_no_dissenters(self):
        props = [_proposal("leader", "X"), _proposal("a", "X")]
        votes = [_vote("a", 0.5)]
        assert _signal_abandonment(props, "leader", votes) == []

    def test_dissenter_votes_positive(self):
        props = [_proposal("leader", "X"), _proposal("a", "Y")]
        votes = [_vote("a", 0.7)]
        result = _signal_abandonment(props, "leader", votes)
        assert result == ["a"]

    def test_dissenter_votes_negative(self):
        props = [_proposal("leader", "X"), _proposal("a", "Y")]
        votes = [_vote("a", -0.5)]
        assert _signal_abandonment(props, "leader", votes) == []

    def test_leader_not_abandoner(self):
        props = [_proposal("leader", "X"), _proposal("a", "Y")]
        votes = [_vote("leader", 0.9), _vote("a", 0.5)]
        result = _signal_abandonment(props, "leader", votes)
        assert "leader" not in result
        assert "a" in result

    def test_multiple_abandoners(self):
        props = [_proposal("L", "X"), _proposal("a", "Y"),
                 _proposal("b", "Z"), _proposal("c", "X")]
        votes = [_vote("a", 0.5), _vote("b", 0.3), _vote("c", 0.8)]
        result = _signal_abandonment(props, "L", votes)
        assert set(result) == {"a", "b"}


# ===================================================================
# _cascade_velocity
# ===================================================================

class TestCascadeVelocity:
    def test_empty(self):
        assert _cascade_velocity([]) == 0.0

    def test_single_point(self):
        assert _cascade_velocity([0.5]) == 0.0

    def test_constant(self):
        assert _cascade_velocity([0.5, 0.5, 0.5]) == pytest.approx(0.0)

    def test_increasing(self):
        v = _cascade_velocity([0.0, 0.5, 1.0])
        assert v > 0

    def test_decreasing(self):
        v = _cascade_velocity([1.0, 0.5, 0.0])
        assert v < 0

    def test_linear_slope(self):
        timeline = [0.1 * i for i in range(10)]
        v = _cascade_velocity(timeline)
        assert v == pytest.approx(0.1)


# ===================================================================
# analyze_cascade — integration
# ===================================================================

class TestAnalyzeCascade:
    def test_empty_histories(self):
        report = analyze_cascade([])
        assert report.total_runs == 0
        assert report.total_rounds == 0
        assert report.overall_cascade_risk == 0.0

    def test_single_run_single_round(self):
        votes = [_vote("a", 0.5), _vote("b", 0.3), _vote("c", -0.2)]
        r = _round(0, "a", votes, committed="sol")
        report = analyze_cascade([[r]])
        assert report.total_runs == 1
        assert report.total_rounds == 1
        assert len(report.echo_index_timeline) == 1
        assert len(report.agent_profiles) == 3

    def test_with_proposals(self):
        votes = [_vote("a", 0.8), _vote("b", 0.7)]
        r = _round(0, "a", votes)
        props = [_proposal("a", "X"), _proposal("b", "Y")]
        report = analyze_cascade([[r]], [[props]])
        assert len(report.diversity_timeline) == 1
        assert report.diversity_timeline[0] == 1.0

    def test_cascade_herding_detected(self):
        votes = [_vote("leader", 0.9), _vote("a", 0.9),
                 _vote("b", 0.9), _vote("c", 0.9)]
        rounds = [_round(i, "leader", votes) for i in range(5)]
        report = analyze_cascade([rounds])
        assert all(ei == 1.0 for ei in report.echo_index_timeline)
        assert all(s == pytest.approx(0.0) for s in report.confidence_spread_timeline)

    def test_no_proposals_skips_diversity(self):
        votes = [_vote("a", 0.5)]
        r = _round(0, "a", votes)
        report = analyze_cascade([[r]], None)
        assert report.diversity_timeline == []

    def test_agent_profiles_composite_score(self):
        votes = [_vote("a", 0.8), _vote("b", -0.3)]
        rounds = [_round(i, "a", votes) for i in range(3)]
        report = analyze_cascade([rounds])
        for p in report.agent_profiles:
            assert 0.0 <= p.cascade_susceptibility <= 1.0

    def test_multiple_runs(self):
        v1 = [_vote("a", 0.5), _vote("b", -0.3)]
        v2 = [_vote("a", -0.2), _vote("b", 0.8)]
        run1 = [_round(0, "a", v1)]
        run2 = [_round(0, "b", v2)]
        report = analyze_cascade([run1, run2])
        assert report.total_runs == 2
        assert report.total_rounds == 2

    def test_flip_flops_recorded_in_profiles(self):
        r1 = _round(0, "L", [_vote("a", -0.5), _vote("L", 0.8)])
        r2 = _round(1, "L", [_vote("a", 0.5), _vote("L", 0.8)])
        report = analyze_cascade([[r1, r2]])
        a_profile = next(p for p in report.agent_profiles if p.agent_id == "a")
        assert a_profile.flip_flop_count == 1


# ===================================================================
# _generate_signals — targeted tests for each detection channel
# ===================================================================

class TestGenerateSignals:
    def test_no_data_returns_healthy(self):
        report = CascadeReport()
        signals = _generate_signals(report)
        assert len(signals) == 1
        assert signals[0].name == "No Cascade Detected"
        assert signals[0].severity == "info"

    def test_diversity_collapse_critical(self):
        report = CascadeReport(diversity_timeline=[0.1, 0.2, 0.15])
        signals = _generate_signals(report)
        names = [s.name for s in signals]
        assert "Diversity Collapse" in names
        sig = next(s for s in signals if s.name == "Diversity Collapse")
        assert sig.severity == "critical"

    def test_diversity_decline_warning(self):
        report = CascadeReport(diversity_timeline=[0.4, 0.5, 0.45])
        signals = _generate_signals(report)
        names = [s.name for s in signals]
        assert "Diversity Decline" in names

    def test_confidence_herding_critical(self):
        report = CascadeReport(confidence_spread_timeline=[0.01, 0.02, 0.03])
        signals = _generate_signals(report)
        names = [s.name for s in signals]
        assert "Confidence Herding" in names

    def test_confidence_clustering_warning(self):
        report = CascadeReport(confidence_spread_timeline=[0.10, 0.12, 0.11])
        signals = _generate_signals(report)
        names = [s.name for s in signals]
        assert "Confidence Clustering" in names

    def test_echo_chamber_critical(self):
        report = CascadeReport(echo_index_timeline=[0.95, 0.92, 0.98])
        signals = _generate_signals(report)
        names = [s.name for s in signals]
        assert "Echo Chamber" in names

    def test_echo_high_agreement_warning(self):
        report = CascadeReport(echo_index_timeline=[0.78, 0.80, 0.76])
        signals = _generate_signals(report)
        names = [s.name for s in signals]
        assert "High Agreement" in names

    def test_rapid_convergence(self):
        report = CascadeReport(cascade_velocity=0.25)
        signals = _generate_signals(report)
        names = [s.name for s in signals]
        assert "Rapid Convergence" in names

    def test_susceptible_agents(self):
        report = CascadeReport(agent_profiles=[
            AgentCascadeProfile(agent_id="a", cascade_susceptibility=0.9),
            AgentCascadeProfile(agent_id="b", cascade_susceptibility=0.1),
        ])
        signals = _generate_signals(report)
        names = [s.name for s in signals]
        assert "Susceptible Agents" in names

    def test_signal_abandonment_critical(self):
        report = CascadeReport(
            total_rounds=10,
            agent_profiles=[
                AgentCascadeProfile(agent_id="a", signal_abandonment_count=8),
                AgentCascadeProfile(agent_id="b", signal_abandonment_count=5),
            ],
        )
        signals = _generate_signals(report)
        names = [s.name for s in signals]
        assert "Private Signal Abandonment" in names

    def test_signal_weakening_warning(self):
        report = CascadeReport(
            total_rounds=10,
            agent_profiles=[
                AgentCascadeProfile(agent_id="a", signal_abandonment_count=2),
                AgentCascadeProfile(agent_id="b", signal_abandonment_count=1),
            ],
        )
        signals = _generate_signals(report)
        names = [s.name for s in signals]
        assert "Signal Weakening" in names

    def test_all_signals_have_recommendations(self):
        report = CascadeReport(
            diversity_timeline=[0.1],
            confidence_spread_timeline=[0.01],
            echo_index_timeline=[0.95],
            cascade_velocity=0.3,
            total_rounds=10,
            agent_profiles=[
                AgentCascadeProfile(agent_id="x", cascade_susceptibility=0.9,
                                    signal_abandonment_count=8),
            ],
        )
        signals = _generate_signals(report)
        for s in signals:
            assert s.recommendation, f"{s.name} has no recommendation"
            assert s.description, f"{s.name} has no description"


# ===================================================================
# Data structures
# ===================================================================

class TestDataStructures:
    def test_cascade_signal_fields(self):
        s = CascadeSignal("test", "warning", 0.5, "desc", "rec")
        assert s.name == "test"
        assert s.severity == "warning"
        assert s.score == 0.5

    def test_agent_cascade_profile_defaults(self):
        p = AgentCascadeProfile(agent_id="a")
        assert p.flip_flop_count == 0
        assert p.echo_count == 0
        assert p.signal_abandonment_count == 0
        assert p.cascade_susceptibility == 0.0

    def test_cascade_report_defaults(self):
        r = CascadeReport()
        assert r.signals == []
        assert r.agent_profiles == []
        assert r.overall_cascade_risk == 0.0
        assert r.total_rounds == 0
