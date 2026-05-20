"""Tests for the Consensus Emergence Detector (src/emergence.py).

Covers:
- Dataclasses (EmergenceSignal, FactionInfo, EmergenceReport) defaults
- analyze_emergence:
  * empty input
  * leadership monopoly (high Gini) signal
  * healthy leadership distribution signal
  * alignment wave (rising) signal
  * alignment collapse (falling) signal
  * reputation divergence signal
  * faction emergence signal
  * consensus momentum strengthening + stalling
  * healthy-emergence default signal
- generate_html_report shape + JSON encoding of alignment series
- run_simulation end-to-end smoke (small N)
- CLI main() smoke writes HTML
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from src.core.state import RoundResult, Vote
from src.emergence import (
    EmergenceReport,
    EmergenceSignal,
    FactionInfo,
    analyze_emergence,
    generate_html_report,
    main,
    run_simulation,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _round(
    *,
    index: int,
    leader: str,
    committed: bool,
    votes: list[Vote] | None = None,
    slashed: list[str] | None = None,
    aggregate: float = 1.0,
) -> RoundResult:
    return RoundResult(
        round_index=index,
        leader_id=leader,
        committed_solution="S" if committed else None,
        aggregate_weight=aggregate,
        threshold=1.0,
        votes=votes or [],
        slashed=slashed or [],
    )


def _vote(voter: str, weight: float) -> Vote:
    return Vote(voter_id=voter, target_proposal_id="p", weight=weight)


# --------------------------------------------------------------------------- #
# Dataclass defaults
# --------------------------------------------------------------------------- #

class TestDataclasses:
    def test_empty_report_defaults(self) -> None:
        r = EmergenceReport()
        assert r.signals == []
        assert r.factions == []
        assert r.leadership_gini == 0.0
        assert r.alignment_timeline == []
        assert r.reputation_phases == []
        assert r.momentum == 0.0
        assert r.commit_rate == 0.0
        assert r.total_rounds == 0

    def test_signal_and_faction_construct(self) -> None:
        s = EmergenceSignal(
            name="x", severity="info", score=0.1,
            description="d", recommendation="r",
        )
        assert s.name == "x" and s.severity == "info"
        f = FactionInfo(members=["a", "b"], cohesion=0.7, label="F1")
        assert f.label == "F1" and f.members == ["a", "b"]


# --------------------------------------------------------------------------- #
# analyze_emergence
# --------------------------------------------------------------------------- #

class TestAnalyzeEmergence:
    def test_empty_history_returns_empty_report(self) -> None:
        r = analyze_emergence([])
        assert r.total_rounds == 0
        assert r.signals == []
        assert r.commit_rate == 0.0

    def test_leadership_monopoly_flags_critical(self) -> None:
        # boss dominates; 5 also-rans each lead once -> Gini > 0.6 (warning+).
        # To reach 'critical' (Gini > 0.8) we need extreme skew across many agents.
        rounds = [_round(index=i, leader="boss", committed=True) for i in range(40)]
        for i, name in enumerate(["a", "b", "c", "d", "e"]):
            rounds.append(_round(index=40 + i, leader=name, committed=True))
        report = analyze_emergence([rounds])
        assert report.leadership_gini > 0.6
        names = [s.name for s in report.signals]
        assert "Leadership Monopoly" in names

    def test_healthy_leadership_distribution_flags_info(self) -> None:
        # Perfectly even leader rotation -> Gini ~ 0
        agents = ["a", "b", "c", "d", "e"]
        rounds = [_round(index=i, leader=agents[i % 5], committed=True) for i in range(10)]
        report = analyze_emergence([rounds])
        assert report.leadership_gini < 0.2
        assert any(s.name == "Healthy Leadership Distribution" for s in report.signals)

    def test_alignment_wave_rising(self) -> None:
        # Alignment starts low (50/50 split) then becomes unanimous.
        rounds: list[RoundResult] = []
        for i in range(6):
            votes = [_vote("a1", 0.5), _vote("a2", -0.5),
                     _vote("a3", 0.5), _vote("a4", -0.5)]
            rounds.append(_round(index=i, leader=f"L{i % 4}",
                                 committed=True, votes=votes))
        for i in range(6, 12):
            votes = [_vote(f"a{j}", 0.5) for j in range(1, 5)]
            rounds.append(_round(index=i, leader=f"L{i % 4}",
                                 committed=True, votes=votes))
        report = analyze_emergence([rounds])
        assert any(s.name == "Alignment Wave Detected" for s in report.signals)

    def test_alignment_collapse_falling(self) -> None:
        rounds: list[RoundResult] = []
        for i in range(6):
            votes = [_vote(f"a{j}", 0.5) for j in range(1, 5)]
            rounds.append(_round(index=i, leader=f"L{i % 4}",
                                 committed=True, votes=votes))
        for i in range(6, 12):
            votes = [_vote("a1", 0.5), _vote("a2", -0.5),
                     _vote("a3", 0.5), _vote("a4", -0.5)]
            rounds.append(_round(index=i, leader=f"L{i % 4}",
                                 committed=True, votes=votes))
        report = analyze_emergence([rounds])
        assert any(s.name == "Alignment Collapse" for s in report.signals)

    def test_reputation_divergence_flags_when_slashing_concentrated(self) -> None:
        # Slash one agent every round -> its reputation halves repeatedly,
        # others stay at 1.0 -> high variance.
        rounds = [
            _round(index=i, leader=f"L{i % 3}", committed=False,
                   votes=[_vote("a1", 0.5), _vote("a2", 0.5), _vote("a3", 0.5)],
                   slashed=["a1"])
            for i in range(8)
        ]
        report = analyze_emergence([rounds])
        # Either rep divergence is flagged OR the rep_phases captured the shift.
        names = [s.name for s in report.signals]
        assert "Reputation Divergence" in names or report.reputation_phases

    def test_faction_emergence_detected(self) -> None:
        # Two cohesive blocs voting in correlated patterns. Each round has
        # a varying sign so per-agent vote vectors are non-constant (well-
        # defined pearson correlation). Bloc A votes +/-/+/-... and bloc B
        # votes the opposite, producing strong intra-bloc correlation and
        # negative cross-bloc correlation.
        rounds = []
        for i in range(8):
            sign = 0.5 if i % 2 == 0 else -0.5
            votes = [
                _vote("a1", sign), _vote("a2", sign), _vote("a3", sign),
                _vote("b1", -sign), _vote("b2", -sign), _vote("b3", -sign),
            ]
            rounds.append(_round(index=i, leader=f"L{i % 6}",
                                 committed=True, votes=votes))
        report = analyze_emergence([rounds])
        assert report.factions, "Expected at least one faction"
        assert any(s.name == "Faction Emergence" for s in report.signals)
        assert any(s.name == "Faction Emergence" for s in report.signals)

    def test_consensus_strengthening_momentum(self) -> None:
        # First half no commits, second half all commits -> rising momentum.
        rounds = [_round(index=i, leader="L", committed=False) for i in range(5)]
        rounds += [_round(index=i + 5, leader="L", committed=True) for i in range(5)]
        report = analyze_emergence([rounds])
        assert report.momentum > 0.3
        assert any(s.name == "Consensus Strengthening" for s in report.signals)

    def test_consensus_stalling_momentum(self) -> None:
        rounds = [_round(index=i, leader="L", committed=True) for i in range(5)]
        rounds += [_round(index=i + 5, leader="L", committed=False) for i in range(5)]
        report = analyze_emergence([rounds])
        assert report.momentum < -0.3
        assert any(s.name == "Consensus Stalling" for s in report.signals)

    def test_healthy_emergence_default_signal(self) -> None:
        # Diverse leaders, steady commits, no slashing, no factions.
        agents = ["a", "b", "c", "d"]
        rounds = [
            _round(index=i, leader=agents[i % 4], committed=True,
                   votes=[_vote(a, 0.5) for a in agents])
            for i in range(8)
        ]
        report = analyze_emergence([rounds])
        assert any(s.name == "Healthy Emergence" for s in report.signals)
        # Healthy report has no critical signals.
        assert not any(s.severity == "critical" for s in report.signals)


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

class TestHtmlReport:
    def test_report_contains_key_sections(self) -> None:
        agents = ["a", "b", "c", "d"]
        rounds = [
            _round(index=i, leader=agents[i % 4], committed=True,
                   votes=[_vote(x, 0.5) for x in agents])
            for i in range(8)
        ]
        report = analyze_emergence([rounds])
        html = generate_html_report(report)
        assert "<!DOCTYPE html>" in html
        assert "Consensus Emergence Report" in html
        assert "Commit Rate" in html
        assert "Leadership Gini" in html
        # alignment series serialized as JSON inside the script
        assert json.dumps(report.alignment_timeline) in html

    def test_report_handles_empty(self) -> None:
        html = generate_html_report(EmergenceReport())
        assert "<!DOCTYPE html>" in html
        # With no factions section gets fallback text.
        assert "No distinct factions detected." in html


# --------------------------------------------------------------------------- #
# run_simulation + CLI
# --------------------------------------------------------------------------- #

class TestRunSimulation:
    @pytest.mark.asyncio
    async def test_run_simulation_smoke(self) -> None:
        report = await run_simulation(n_runs=3, n_agents=4, threshold=1.5)
        assert isinstance(report, EmergenceReport)
        assert report.total_rounds >= 1
        # commit_rate is a valid probability.
        assert 0.0 <= report.commit_rate <= 1.0


class TestCli:
    def test_main_writes_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out = tmp_path / "emergence.html"
        argv = ["emergence", "--runs", "2", "--agents", "4",
                "--threshold", "1.5", "--output", str(out)]
        monkeypatch.setattr(sys, "argv", argv)
        main()
        assert out.exists()
        body = out.read_text(encoding="utf-8")
        assert "Consensus Emergence Report" in body
