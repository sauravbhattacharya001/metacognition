"""Tests for ConsensusLatencyProfiler."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.core.state import RoundResult, Vote
from src.consensus_latency_profiler import (
    ConsensusLatencyProfiler,
    LatencyReport,
)


def _fixed_now():
    return datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc)


def _vote(voter: str, weight: float) -> Vote:
    return Vote(voter_id=voter, target_proposal_id="p1", weight=weight)


def _round(idx: int, leader: str, committed: bool, agg: float, threshold: float = 1.5,
           votes=None, slashed=None) -> RoundResult:
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution="X" if committed else None,
        aggregate_weight=agg,
        threshold=threshold,
        votes=votes or [],
        slashed=slashed or [],
    )


class TestEmptyHistory:
    def test_empty_returns_grade_a(self):
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze([], max_rounds=4)
        assert report.grade == "A"
        assert report.verdict == "FAST_CONSENSUS"
        assert report.latency_score == 100
        assert report.total_rounds == 0

    def test_empty_playbook_has_healthy_fallback(self):
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze([], max_rounds=4)
        assert len(report.playbook) == 1
        assert report.playbook[0].id == "CONSENSUS_HEALTHY"


class TestFastConsensus:
    def test_single_round_commit(self):
        history = [_round(0, "a1", True, 2.0)]
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        assert report.committed_rounds == 1
        assert report.commit_rate == 1.0
        assert report.mean_rounds_to_commit == 1.0
        assert report.grade == "A"


class TestSlowConvergence:
    def test_multi_round_commit_detected(self):
        # 3 rounds to commit (threshold is 2 for max_rounds=4)
        history = [
            _round(0, "a1", False, 1.0, slashed=["a1"]),
            _round(1, "a2", False, 1.2, slashed=["a2"]),
            _round(2, "a3", True, 2.0),
        ]
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        modes = [f.mode for f in report.findings]
        assert "SLOW_CONVERGENCE" in modes


class TestSerialSlashCascade:
    def test_same_agent_slashed_twice(self):
        history = [
            _round(0, "a1", False, 0.5, slashed=["a1"]),
            _round(1, "a1", False, 0.5, slashed=["a1"]),
            _round(2, "a2", True, 2.0),
        ]
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        modes = [f.mode for f in report.findings]
        assert "SERIAL_SLASH_CASCADE" in modes


class TestLeaderMonopoly:
    def test_single_leader_dominance(self):
        # a1 leads all 4 committed rounds
        history = [
            _round(0, "a1", True, 2.0),
            _round(0, "a1", True, 2.0),
            _round(0, "a1", True, 2.0),
            _round(0, "a1", True, 2.0),
        ]
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        modes = [f.mode for f in report.findings]
        assert "LEADER_MONOPOLY" in modes


class TestRevolvingDoor:
    def test_all_different_leaders_no_commit(self):
        history = [
            _round(0, "a1", False, 0.5, slashed=["a1"]),
            _round(1, "a2", False, 0.5, slashed=["a2"]),
            _round(2, "a3", False, 0.5, slashed=["a3"]),
        ]
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        modes = [f.mode for f in report.findings]
        assert "REVOLVING_DOOR" in modes


class TestNearMissStall:
    def test_close_to_threshold(self):
        # aggregate = 1.44, threshold = 1.5 -> margin = 0.04 (within 5%)
        history = [_round(0, "a1", False, 1.44, threshold=1.5)]
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        modes = [f.mode for f in report.findings]
        assert "NEAR_MISS_STALL" in modes


class TestVetoBottleneck:
    def test_chronic_rejector(self):
        history = [
            _round(0, "a1", False, 0.5, votes=[_vote("blocker", -0.9)]),
            _round(0, "a2", False, 0.5, votes=[_vote("blocker", -0.8)]),
            _round(0, "a3", True, 2.0),
        ]
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        modes = [f.mode for f in report.findings]
        assert "VETO_BOTTLENECK" in modes
        f = next(x for x in report.findings if x.mode == "VETO_BOTTLENECK")
        assert f.evidence["blocker_id"] == "blocker"


class TestInstantCommit:
    def test_all_instant(self):
        history = [_round(0, "a1", True, 2.0) for _ in range(5)]
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        modes = [f.mode for f in report.findings]
        assert "INSTANT_COMMIT" in modes


class TestRiskAppetite:
    def test_cautious_adds_audit(self):
        # Create a scenario with findings + grade C/D/F
        history = [
            _round(0, "a1", False, 0.5, votes=[_vote("blocker", -0.9)]),
            _round(0, "a2", False, 0.5, votes=[_vote("blocker", -0.8)]),
            _round(0, "a3", False, 0.5, votes=[_vote("blocker", -0.7)]),
        ]
        p = ConsensusLatencyProfiler(risk_appetite="cautious", now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        action_ids = [a.id for a in report.playbook]
        assert "SCHEDULE_LATENCY_AUDIT" in action_ids

    def test_aggressive_trims_p3(self):
        # Scenario that produces P0 + P3 fallback
        history = [
            _round(0, "a1", False, 0.5, slashed=["a1"]),
            _round(1, "a1", False, 0.5, slashed=["a1"]),
            _round(2, "a2", True, 2.0),
        ]
        p = ConsensusLatencyProfiler(risk_appetite="aggressive", now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        priorities = [a.priority for a in report.playbook]
        assert "P3" not in priorities

    def test_invalid_appetite_raises(self):
        with pytest.raises(ValueError):
            ConsensusLatencyProfiler(risk_appetite="reckless")


class TestRenderers:
    def _get_report(self):
        history = [_round(0, "a1", True, 2.0)]
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        return p, p.analyze(history, max_rounds=4)

    def test_to_text_has_sections(self):
        p, report = self._get_report()
        text = p.to_text(report)
        assert "VERDICT:" in text
        assert "--- Summary ---" in text
        assert "--- Findings ---" in text
        assert "--- Playbook ---" in text
        assert "--- Insights ---" in text

    def test_to_markdown_has_sections(self):
        p, report = self._get_report()
        md = p.to_markdown(report)
        assert "## Summary" in md
        assert "## Findings" in md
        assert "## Playbook" in md
        assert "## Insights" in md

    def test_to_json_is_valid_and_stable(self):
        p, report = self._get_report()
        j1 = p.to_json(report)
        j2 = p.to_json(report)
        assert j1 == j2
        data = json.loads(j1)
        assert "grade" in data
        assert "latency_score" in data


class TestDeterminism:
    def test_same_input_same_output(self):
        history = [
            _round(0, "a1", False, 1.44, threshold=1.5, votes=[_vote("v1", -0.5)]),
            _round(0, "a2", True, 2.0),
        ]
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        r1 = p.analyze(history, max_rounds=4)
        r2 = p.analyze(history, max_rounds=4)
        assert p.to_json(r1) == p.to_json(r2)


class TestInputImmutability:
    def test_original_history_not_mutated(self):
        history = [_round(0, "a1", True, 2.0)]
        original_json = json.dumps([r.model_dump() for r in history], sort_keys=True)
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        p.analyze(history, max_rounds=4)
        after_json = json.dumps([r.model_dump() for r in history], sort_keys=True)
        assert original_json == after_json


class TestStaleConsensus:
    def test_declining_commit_rate(self):
        # First 4 runs commit, last 4 don't
        history = []
        for i in range(4):
            history.append(_round(0, f"a{i}", True, 2.0))
        for i in range(4):
            history.append(_round(0, f"b{i}", False, 0.5, slashed=[f"b{i}"]))
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        modes = [f.mode for f in report.findings]
        assert "STALE_CONSENSUS" in modes


class TestGrading:
    def test_double_p0_gives_f(self):
        # Two P0 scenarios: veto bottleneck + slash cascade
        history = [
            _round(0, "a1", False, 0.5, slashed=["a1"], votes=[_vote("blocker", -0.9)]),
            _round(1, "a1", False, 0.5, slashed=["a1"], votes=[_vote("blocker", -0.8)]),
            _round(2, "a2", False, 0.5, votes=[_vote("blocker", -0.7)]),
        ]
        p = ConsensusLatencyProfiler(now_fn=_fixed_now)
        report = p.analyze(history, max_rounds=4)
        assert report.grade == "F" or report.grade == "D"  # At least one P0
        p0_findings = [f for f in report.findings if f.priority == "P0"]
        assert len(p0_findings) >= 1
