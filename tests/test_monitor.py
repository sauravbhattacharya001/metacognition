"""Behavioural tests for `src/monitor.py` – the consensus resilience monitor.

Covers the pure helpers (`_build_swarm`, `_to_dict`, `_render_html`,
`_generate_recommendations`), the async scenario runner (`run_scenario`,
`analyze_resilience` with and without the threshold sweep), and the CLI
(`main`) for both stdout and the JSON/HTML export modes.

The MBFT engine is real – no mocking – so these tests also act as a smoke
suite for the engine + monitor integration.
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from src import monitor
from src.monitor import (
    ResilienceReport,
    ScenarioResult,
    _build_swarm,
    _generate_recommendations,
    _print_report,
    _render_html,
    _to_dict,
    analyze_resilience,
    run_scenario,
)


# ---------------------------------------------------------------------------
# _build_swarm
# ---------------------------------------------------------------------------

class TestBuildSwarm:
    def test_returns_n_agents_with_correct_ids(self) -> None:
        agents = _build_swarm(5, byzantine_count=0)
        assert len(agents) == 5
        assert [a.id for a in agents] == ["a1", "a2", "a3", "a4", "a5"]

    def test_zero_byzantine_all_honest(self) -> None:
        agents = _build_swarm(4, byzantine_count=0)
        assert all(not a.byzantine for a in agents)
        # Honest agents share the same "correct" answer
        assert {a.answer for a in agents} == {"correct"}

    def test_byzantine_placed_at_tail(self) -> None:
        agents = _build_swarm(6, byzantine_count=2)
        assert [a.byzantine for a in agents] == [False, False, False, False, True, True]
        # Byzantines get unique answers so they cannot accidentally agree
        byz_answers = {a.answer for a in agents if a.byzantine}
        assert len(byz_answers) == 2
        assert all(a.startswith("byz-") for a in byz_answers)

    def test_all_byzantine(self) -> None:
        agents = _build_swarm(3, byzantine_count=3)
        assert all(a.byzantine for a in agents)
        assert all(a.confidence == 0.95 for a in agents)

    def test_custom_confidence_levels(self) -> None:
        agents = _build_swarm(
            4, byzantine_count=1, honest_confidence=0.42, byzantine_confidence=0.88
        )
        honest = [a for a in agents if not a.byzantine]
        byz = [a for a in agents if a.byzantine]
        assert all(a.confidence == 0.42 for a in honest)
        assert all(a.confidence == 0.88 for a in byz)


# ---------------------------------------------------------------------------
# run_scenario
# ---------------------------------------------------------------------------

class TestRunScenario:
    @pytest.mark.asyncio
    async def test_zero_byzantine_commits_correct(self) -> None:
        result = await run_scenario(n=5, byzantine_count=0, threshold=1.5)
        assert isinstance(result, ScenarioResult)
        assert result.total_agents == 5
        assert result.byzantine_count == 0
        assert result.committed is True
        assert result.solution == "correct"
        assert result.aggregate_weight >= 1.5
        assert result.rounds_used >= 1
        assert set(result.reputation_after.keys()) == {f"a{i+1}" for i in range(5)}

    @pytest.mark.asyncio
    async def test_byzantine_majority_scenario_structure(self) -> None:
        # Mostly-byzantine swarm: scenario still runs cleanly and reports.
        # NOTE: in the engine, byzantine voters happily endorse the leader,
        # so a commit can still happen – we only assert structural sanity.
        result = await run_scenario(n=5, byzantine_count=4, threshold=2.5)
        assert result.total_agents == 5
        assert result.byzantine_count == 4
        assert result.threshold == 2.5
        assert isinstance(result.committed, bool)
        assert len(result.reputation_after) == 5

    @pytest.mark.asyncio
    async def test_high_threshold_blocks_commit(self) -> None:
        # 3 honest agents at 0.80 confidence cannot reach threshold 5.0
        result = await run_scenario(n=3, byzantine_count=0, threshold=5.0)
        assert result.committed is False
        assert result.aggregate_weight < 5.0

    @pytest.mark.asyncio
    async def test_round_count_bounded_by_max_rounds(self) -> None:
        result = await run_scenario(n=4, byzantine_count=0, threshold=1.0)
        # MBFTEngine in monitor uses max_rounds=4
        assert 1 <= result.rounds_used <= 4


# ---------------------------------------------------------------------------
# analyze_resilience
# ---------------------------------------------------------------------------

class TestAnalyzeResilience:
    @pytest.mark.asyncio
    async def test_basic_analysis_returns_report(self) -> None:
        report = await analyze_resilience(swarm_size=4, threshold=1.5)
        assert isinstance(report, ResilienceReport)
        assert report.swarm_size == 4
        assert report.threshold == 1.5
        assert len(report.scenarios) == 4  # byz in 0..n-1
        assert 0.0 <= report.fault_tolerance_ratio <= 1.0
        assert report.threshold_sweep is None
        assert report.recommendations  # non-empty

    @pytest.mark.asyncio
    async def test_max_byzantine_tolerated_monotonic_setup(self) -> None:
        report = await analyze_resilience(swarm_size=5, threshold=1.5)
        # Byzantine counts tested were 0..4 inclusive
        assert {s.byzantine_count for s in report.scenarios} == {0, 1, 2, 3, 4}
        # ratio is consistent with max_tolerated / size
        assert report.fault_tolerance_ratio == pytest.approx(
            report.max_byzantine_tolerated / 5
        )

    @pytest.mark.asyncio
    async def test_threshold_sweep_populated(self) -> None:
        report = await analyze_resilience(
            swarm_size=3, threshold=1.5, sweep_thresholds=True
        )
        assert report.threshold_sweep is not None
        # Sweep covers thresholds 0.5..3.0 in steps of 0.5
        assert set(report.threshold_sweep.keys()) == {
            f"{t/10:.1f}" for t in range(5, 35, 5)
        }
        for entry in report.threshold_sweep.values():
            assert "max_byzantine" in entry
            assert "ratio" in entry
            assert 0.0 <= entry["ratio"] <= 1.0

    @pytest.mark.asyncio
    async def test_zero_swarm_size_safe(self) -> None:
        report = await analyze_resilience(swarm_size=0, threshold=1.5)
        assert report.swarm_size == 0
        assert report.scenarios == []
        assert report.fault_tolerance_ratio == 0
        assert report.max_byzantine_tolerated == 0


# ---------------------------------------------------------------------------
# _generate_recommendations
# ---------------------------------------------------------------------------

class TestGenerateRecommendations:
    def _make_report(
        self,
        size: int = 7,
        max_tol: int = 2,
        sweep: dict | None = None,
        scenarios: list | None = None,
    ) -> ResilienceReport:
        return ResilienceReport(
            swarm_size=size,
            threshold=1.5,
            max_byzantine_tolerated=max_tol,
            fault_tolerance_ratio=max_tol / size if size else 0,
            scenarios=scenarios or [],
            threshold_sweep=sweep,
        )

    def test_critical_when_below_20pct(self) -> None:
        r = self._make_report(size=10, max_tol=1)  # 10%
        recs = _generate_recommendations(r)
        joined = "\n".join(recs)
        assert "CRITICAL" in joined

    def test_warning_when_between_20_and_33pct(self) -> None:
        r = self._make_report(size=10, max_tol=3)  # 30%
        recs = _generate_recommendations(r)
        joined = "\n".join(recs)
        assert "Fault tolerance is below the classic BFT" in joined

    def test_ok_when_at_or_above_33pct(self) -> None:
        r = self._make_report(size=9, max_tol=3)  # 33.3%
        recs = _generate_recommendations(r)
        joined = "\n".join(recs)
        assert "meets or exceeds" in joined

    def test_small_swarm_warning(self) -> None:
        r = self._make_report(size=4, max_tol=1)
        recs = _generate_recommendations(r)
        joined = "\n".join(recs)
        assert "fewer than 5 agents" in joined

    def test_no_small_swarm_warning_for_5(self) -> None:
        r = self._make_report(size=5, max_tol=2)
        recs = _generate_recommendations(r)
        joined = "\n".join(recs)
        assert "fewer than 5 agents" not in joined

    def test_threshold_sweep_better_setting_suggestion(self) -> None:
        sweep = {
            "0.5": {"max_byzantine": 5, "ratio": 0.50},
            "1.5": {"max_byzantine": 2, "ratio": 0.20},
        }
        r = self._make_report(size=10, max_tol=2, sweep=sweep)
        recs = _generate_recommendations(r)
        joined = "\n".join(recs)
        assert "Threshold" in joined and "achieves better fault tolerance" in joined

    def test_threshold_sweep_no_better_setting(self) -> None:
        sweep = {
            "0.5": {"max_byzantine": 1, "ratio": 0.10},
            "1.5": {"max_byzantine": 5, "ratio": 0.50},
        }
        r = self._make_report(size=10, max_tol=5, sweep=sweep)
        recs = _generate_recommendations(r)
        joined = "\n".join(recs)
        assert "achieves better fault tolerance" not in joined

    def test_safety_alert_on_incorrect_commit(self) -> None:
        bad_scenario = ScenarioResult(
            total_agents=5, byzantine_count=2, threshold=1.0,
            committed=True, rounds_used=1, aggregate_weight=2.0,
            solution="WRONG", reputation_after={},
        )
        r = self._make_report(size=5, max_tol=2, scenarios=[bad_scenario])
        recs = _generate_recommendations(r)
        joined = "\n".join(recs)
        assert "SAFETY ALERT" in joined and "1 scenario" in joined


# ---------------------------------------------------------------------------
# _to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    @pytest.mark.asyncio
    async def test_to_dict_is_json_serialisable(self) -> None:
        report = await analyze_resilience(swarm_size=3, threshold=1.5)
        data = _to_dict(report)
        # round-trip through JSON to prove serialisability
        roundtripped = json.loads(json.dumps(data))
        assert roundtripped["swarm_size"] == 3
        assert roundtripped["threshold"] == 1.5
        assert "scenarios" in roundtripped
        assert len(roundtripped["scenarios"]) == 3
        for sc in roundtripped["scenarios"]:
            assert set(sc.keys()) == {
                "byzantine_count", "committed", "solution",
                "rounds_used", "aggregate_weight",
            }
        assert "recommendations" in roundtripped
        assert roundtripped["threshold_sweep"] is None

    @pytest.mark.asyncio
    async def test_to_dict_includes_sweep(self) -> None:
        report = await analyze_resilience(
            swarm_size=3, threshold=1.5, sweep_thresholds=True
        )
        data = _to_dict(report)
        assert data["threshold_sweep"] is not None
        assert all(
            "max_byzantine" in v and "ratio" in v
            for v in data["threshold_sweep"].values()
        )


# ---------------------------------------------------------------------------
# _render_html
# ---------------------------------------------------------------------------

class TestRenderHtml:
    @pytest.mark.asyncio
    async def test_render_html_contains_key_markup(self) -> None:
        report = await analyze_resilience(swarm_size=4, threshold=1.5)
        html = _render_html(report)
        assert html.startswith("<!DOCTYPE html>")
        assert "mBFT Resilience Report" in html
        assert "<canvas id=\"chart\"" in html
        # Each recommendation should appear as an <li>
        for rec in report.recommendations:
            assert rec in html
        # Embedded JSON should parse: extract scenarios array
        marker = "const scenarios = "
        idx = html.index(marker) + len(marker)
        end = html.index(";", idx)
        scenarios = json.loads(html[idx:end])
        assert len(scenarios) == len(report.scenarios)

    @pytest.mark.asyncio
    async def test_render_html_handles_empty_sweep(self) -> None:
        report = await analyze_resilience(swarm_size=3, threshold=1.5)
        html = _render_html(report)
        assert "const sweep = {}" in html


# ---------------------------------------------------------------------------
# _print_report  (smoke – mostly just shouldn't raise)
# ---------------------------------------------------------------------------

class TestPrintReport:
    @pytest.mark.asyncio
    async def test_print_report_without_sweep(self) -> None:
        report = await analyze_resilience(swarm_size=3, threshold=1.5)

        # _print_report wraps sys.stdout.buffer in a TextIOWrapper that
        # closes the underlying buffer on GC, so capture before exiting
        # the patch context.
        class UnclosableBytesIO(io.BytesIO):
            def close(self) -> None:  # noqa: D401
                # no-op so the wrapper's close doesn't kill our buffer
                pass

        fake = UnclosableBytesIO()

        class FakeStdout:
            buffer = fake

        with patch.object(sys, "stdout", FakeStdout()):
            _print_report(report)
            out = fake.getvalue().decode("utf-8", errors="replace")

        assert "mBFT CONSENSUS RESILIENCE REPORT" in out
        assert "Swarm Size" in out
        assert "SCENARIO RESULTS" in out
        assert "RECOMMENDATIONS" in out

    @pytest.mark.asyncio
    async def test_print_report_with_sweep(self) -> None:
        report = await analyze_resilience(
            swarm_size=3, threshold=1.5, sweep_thresholds=True
        )

        class UnclosableBytesIO(io.BytesIO):
            def close(self) -> None:
                pass

        fake = UnclosableBytesIO()

        class FakeStdout:
            buffer = fake

        with patch.object(sys, "stdout", FakeStdout()):
            _print_report(report)
            out = fake.getvalue().decode("utf-8", errors="replace")

        assert "THRESHOLD SWEEP" in out


# ---------------------------------------------------------------------------
# main / CLI
# ---------------------------------------------------------------------------

class TestMainCli:
    @pytest.mark.asyncio
    async def test_main_default_stdout(self) -> None:
        class UnclosableBytesIO(io.BytesIO):
            def close(self) -> None:
                pass

        fake = UnclosableBytesIO()

        class FakeStdout:
            buffer = fake

        with patch.object(sys, "argv", ["monitor", "--agents", "3"]):
            with patch.object(sys, "stdout", FakeStdout()):
                await monitor.main()
                out = fake.getvalue().decode("utf-8", errors="replace")

        assert "mBFT CONSENSUS RESILIENCE REPORT" in out

    @pytest.mark.asyncio
    async def test_main_export_json_to_stdout(self, capsys) -> None:
        with patch.object(
            sys, "argv", ["monitor", "--agents", "3", "--export", "json"]
        ):
            await monitor.main()

        captured = capsys.readouterr().out
        # Output should be valid JSON
        data = json.loads(captured)
        assert data["swarm_size"] == 3

    @pytest.mark.asyncio
    async def test_main_export_json_to_file(self, tmp_path, capsys) -> None:
        out_file = tmp_path / "report.json"
        with patch.object(
            sys, "argv",
            ["monitor", "--agents", "3", "--export", "json", "--output", str(out_file)],
        ):
            await monitor.main()

        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["swarm_size"] == 3
        captured = capsys.readouterr().out
        assert "Report written to" in captured

    @pytest.mark.asyncio
    async def test_main_export_html_default_path(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.chdir(tmp_path)
        with patch.object(
            sys, "argv", ["monitor", "--agents", "3", "--export", "html"]
        ):
            await monitor.main()

        out_file = tmp_path / "resilience_report.html"
        assert out_file.exists()
        html = out_file.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        captured = capsys.readouterr().out
        assert "Interactive report written to" in captured

    @pytest.mark.asyncio
    async def test_main_export_html_custom_path(self, tmp_path, capsys) -> None:
        out_file = tmp_path / "custom.html"
        with patch.object(
            sys, "argv",
            ["monitor", "--agents", "3", "--export", "html", "--output", str(out_file)],
        ):
            await monitor.main()

        assert out_file.exists()
        assert "<!DOCTYPE html>" in out_file.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_main_with_sweep_flag(self, capsys) -> None:
        with patch.object(
            sys, "argv",
            ["monitor", "--agents", "3", "--sweep-thresholds", "--export", "json"],
        ):
            await monitor.main()

        captured = capsys.readouterr().out
        data = json.loads(captured)
        assert data["threshold_sweep"] is not None
