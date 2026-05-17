"""Tests for the Trust Evolution Tracker (src.trust_tracker).

Covers:
- Data structure defaults and behaviour (ReputationSnapshot, TrustAnomaly,
  AgentProfile including trust_grade boundary mapping)
- Pure helpers (_random_agent_id, generate_scenario)
- Anomaly detection (_detect_anomalies) for sudden_drop, stagnation,
  rehabilitation, and the no-anomaly happy path
- Tracker run end-to-end with a small deterministic scenario count
- JSON and HTML rendering (structure + key invariants, no fragile snapshot)
- CLI entry point (_main) with --json output and seeded determinism
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from src.agents.metacognitive import MockAgent
from src.trust_tracker import (
    AgentProfile,
    ReputationSnapshot,
    TrustAnomaly,
    TrustEvolutionTracker,
    _main,
    _random_agent_id,
    generate_scenario,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TestReputationSnapshot:
    def test_basic_construction(self):
        snap = ReputationSnapshot(scenario=2, round_index=4, reputations={"a": 0.5})
        assert snap.scenario == 2
        assert snap.round_index == 4
        assert snap.reputations == {"a": 0.5}

    def test_empty_reputations_allowed(self):
        snap = ReputationSnapshot(0, 0, {})
        assert snap.reputations == {}


class TestTrustAnomaly:
    def test_fields_round_trip(self):
        a = TrustAnomaly("agent_1", "sudden_drop", 3, "fell hard", "high")
        assert a.agent_id == "agent_1"
        assert a.kind == "sudden_drop"
        assert a.scenario == 3
        assert a.detail == "fell hard"
        assert a.severity == "high"


class TestAgentProfile:
    def _make(self, rep: float) -> AgentProfile:
        return AgentProfile(
            agent_id="x",
            final_reputation=rep,
            min_reputation=rep,
            max_reputation=rep,
            times_slashed=0,
            times_leader=0,
        )

    def test_default_collections_independent(self):
        p1 = self._make(0.5)
        p2 = self._make(0.5)
        p1.anomalies.append(TrustAnomaly("x", "sudden_drop", 0, "d", "low"))
        # default_factory must give each instance its own list
        assert p2.anomalies == []
        assert p2.trajectory == []

    @pytest.mark.parametrize(
        "rep, grade",
        [
            (1.0, "A"),
            (0.9, "A"),
            (0.89, "B"),
            (0.7, "B"),
            (0.69, "C"),
            (0.5, "C"),
            (0.49, "D"),
            (0.25, "D"),
            (0.24, "F"),
            (0.0, "F"),
        ],
    )
    def test_trust_grade_boundaries(self, rep, grade):
        assert self._make(rep).trust_grade == grade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestRandomAgentId:
    def test_format(self):
        assert _random_agent_id(0) == "agent_0"
        assert _random_agent_id(42) == "agent_42"


class TestGenerateScenario:
    def test_default_size_and_threshold_range(self):
        agents, threshold = generate_scenario(0)
        assert len(agents) == 5
        assert all(isinstance(a, MockAgent) for a in agents)
        assert 1.0 <= threshold <= 2.0

    def test_custom_n_agents(self):
        agents, _ = generate_scenario(1, n_agents=8)
        assert len(agents) == 8
        # ids must be unique and follow pattern
        ids = [a.id for a in agents]
        assert len(set(ids)) == 8
        assert all(i.startswith("agent_") for i in ids)

    def test_byzantine_minority_with_seeded_random(self):
        # With many agents and a seeded RNG, byzantine fraction stays well
        # under half (the generator probability is ~0.2).
        import random as _r
        _r.seed(123)
        agents, _ = generate_scenario(0, n_agents=50)
        byz = [a for a in agents if a.byzantine]
        assert 0 <= len(byz) <= 25  # never the majority for p=0.2, n=50, seed=123
        # honest agents share the same answer
        honest = [a for a in agents if not a.byzantine]
        if len(honest) >= 2:
            assert len({a.answer for a in honest}) == 1

    def test_byzantine_confidence_bounds(self):
        import random as _r
        _r.seed(7)
        agents, _ = generate_scenario(0, n_agents=20)
        for a in agents:
            assert 0.4 <= a.confidence <= 1.0


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def _profile_with_trajectory(traj: list[float], agent_id: str = "a") -> AgentProfile:
    return AgentProfile(
        agent_id=agent_id,
        final_reputation=traj[-1] if traj else 0.0,
        min_reputation=min(traj) if traj else 0.0,
        max_reputation=max(traj) if traj else 0.0,
        times_slashed=0,
        times_leader=0,
        trajectory=list(traj),
    )


class TestAnomalyDetection:
    def _detect(self, profiles: dict[str, AgentProfile]) -> TrustEvolutionTracker:
        t = TrustEvolutionTracker(n_scenarios=0)
        t.profiles = profiles
        t._detect_anomalies()
        return t

    def test_no_anomalies_on_smooth_trajectory(self):
        t = self._detect({"a": _profile_with_trajectory([0.6, 0.61, 0.62, 0.65])})
        assert t.anomalies == []
        assert t.profiles["a"].anomalies == []

    def test_sudden_drop_detected(self):
        t = self._detect({"a": _profile_with_trajectory([0.9, 0.5])})
        kinds = [a.kind for a in t.anomalies]
        assert "sudden_drop" in kinds
        drop = next(a for a in t.anomalies if a.kind == "sudden_drop")
        assert drop.agent_id == "a"
        assert drop.severity == "high"

    def test_drop_below_threshold_not_anomaly(self):
        # Drop of 0.25 should not trigger (strict > 0.3).
        t = self._detect({"a": _profile_with_trajectory([0.8, 0.55])})
        assert all(a.kind != "sudden_drop" for a in t.anomalies)

    def test_stagnation_low_three_scenarios(self):
        t = self._detect({"a": _profile_with_trajectory([0.1, 0.2, 0.25])})
        kinds = [a.kind for a in t.anomalies]
        assert "stagnation" in kinds
        stag = next(a for a in t.anomalies if a.kind == "stagnation")
        assert stag.severity == "medium"

    def test_no_stagnation_when_above_threshold(self):
        t = self._detect({"a": _profile_with_trajectory([0.1, 0.2, 0.31])})
        assert all(a.kind != "stagnation" for a in t.anomalies)

    def test_rehabilitation_detected(self):
        t = self._detect({"a": _profile_with_trajectory([0.1, 0.75])})
        kinds = [a.kind for a in t.anomalies]
        assert "rehabilitation" in kinds
        rehab = next(a for a in t.anomalies if a.kind == "rehabilitation")
        assert rehab.severity == "low"

    def test_multiple_agents_isolated(self):
        good = _profile_with_trajectory([0.8, 0.85, 0.9])
        bad = _profile_with_trajectory([0.9, 0.4])  # sudden drop (0.5 > 0.3)
        t = self._detect({"good": good, "bad": bad})
        assert good.anomalies == []
        assert any(a.kind == "sudden_drop" for a in bad.anomalies)
        # tracker-level total matches per-agent sum
        assert len(t.anomalies) == sum(len(p.anomalies) for p in t.profiles.values())


# ---------------------------------------------------------------------------
# Tracker run
# ---------------------------------------------------------------------------

class TestTrackerRun:
    def test_seeded_run_is_deterministic(self):
        t1 = TrustEvolutionTracker(n_scenarios=2, seed=42)
        asyncio.run(t1.run())
        t2 = TrustEvolutionTracker(n_scenarios=2, seed=42)
        asyncio.run(t2.run())
        assert sorted(t1.profiles) == sorted(t2.profiles)
        for aid in t1.profiles:
            assert t1.profiles[aid].trajectory == t2.profiles[aid].trajectory
        assert len(t1.snapshots) == len(t2.snapshots)
        assert len(t1.scenario_results) == len(t2.scenario_results) == 2

    def test_profiles_populated(self):
        t = TrustEvolutionTracker(n_scenarios=2, seed=7)
        asyncio.run(t.run())
        assert t.profiles, "expected at least one agent profile"
        for p in t.profiles.values():
            assert p.trajectory, "trajectory should be non-empty"
            assert p.min_reputation <= p.final_reputation <= p.max_reputation
            assert p.times_slashed >= 0
            assert p.times_leader >= 0
            assert p.trust_grade in {"A", "B", "C", "D", "F"}

    def test_scenario_results_shape(self):
        t = TrustEvolutionTracker(n_scenarios=3, seed=1)
        asyncio.run(t.run())
        assert len(t.scenario_results) == 3
        for i, sr in enumerate(t.scenario_results):
            assert sr["scenario"] == i
            assert isinstance(sr["committed"], bool)
            assert sr["rounds"] >= 0
            assert 1.0 <= sr["threshold"] <= 2.0
            assert sr["agents"]

    def test_snapshots_have_consistent_scenario_indices(self):
        t = TrustEvolutionTracker(n_scenarios=2, seed=11)
        asyncio.run(t.run())
        # every snapshot's scenario index lies in [0, n_scenarios)
        for snap in t.snapshots:
            assert 0 <= snap.scenario < 2
            assert snap.round_index >= 0
            assert isinstance(snap.reputations, dict)

    def test_zero_scenarios_is_inert(self):
        t = TrustEvolutionTracker(n_scenarios=0, seed=1)
        asyncio.run(t.run())
        assert t.profiles == {}
        assert t.snapshots == []
        assert t.scenario_results == []
        assert t.anomalies == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def _seeded_tracker(self) -> TrustEvolutionTracker:
        t = TrustEvolutionTracker(n_scenarios=2, seed=3)
        asyncio.run(t.run())
        return t

    def test_to_json_structure(self):
        t = self._seeded_tracker()
        data = t.to_json()
        assert set(data.keys()) == {"scenarios", "profiles", "anomalies_total"}
        assert data["anomalies_total"] == len(t.anomalies)
        for aid, p in data["profiles"].items():
            assert {
                "final_reputation", "min_reputation", "max_reputation",
                "times_slashed", "times_leader", "trust_grade",
                "trajectory", "anomalies",
            } <= set(p.keys())
            assert p["trust_grade"] in {"A", "B", "C", "D", "F"}
            assert p["min_reputation"] <= p["max_reputation"]

    def test_to_json_is_serializable(self):
        t = self._seeded_tracker()
        # round-trip via the JSON encoder to confirm no numpy / unhashable types
        s = json.dumps(t.to_json())
        assert json.loads(s) == t.to_json()

    def test_to_html_contains_key_markup(self):
        t = self._seeded_tracker()
        html = t.to_html()
        assert "<!DOCTYPE html>" in html
        assert "Trust Evolution Report" in html
        # Chart.js is referenced for the line chart
        assert "chart.js" in html.lower() or "Chart(" in html
        # Profile table headers
        for header in ("Grade", "Final Rep", "Min", "Max", "Slashed", "Anomalies"):
            assert header in html
        # Every agent id from the JSON view appears in the HTML
        data = t.to_json()
        for aid in data["profiles"]:
            assert aid in html

    def test_to_html_handles_empty_tracker(self):
        t = TrustEvolutionTracker(n_scenarios=0, seed=1)
        # No run; ensure rendering does not crash on empty profiles/scenarios.
        html = t.to_html()
        assert "Trust Evolution Report" in html
        assert "Agents Tracked" in html

    def test_html_anomaly_badges_when_present(self):
        # Force a sudden-drop profile so the anomaly badge branch renders.
        t = TrustEvolutionTracker(n_scenarios=0, seed=1)
        t.profiles = {"a": _profile_with_trajectory([0.9, 0.5])}
        t._detect_anomalies()
        html = t.to_html()
        assert "sudden_drop" in html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_main_writes_html_and_json(self, tmp_path: Path, capsys, monkeypatch):
        html_out = tmp_path / "report.html"
        json_out = tmp_path / "report.json"
        monkeypatch.setattr(
            sys, "argv",
            [
                "trust_tracker",
                "--scenarios", "2",
                "--seed", "5",
                "--out", str(html_out),
                "--json", str(json_out),
            ],
        )
        asyncio.run(_main())
        captured = capsys.readouterr().out
        assert "Trust Evolution Report" in captured
        assert html_out.exists() and html_out.stat().st_size > 0
        assert json_out.exists()
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert "scenarios" in payload and "profiles" in payload
        assert len(payload["scenarios"]) == 2

    def test_main_without_json_flag(self, tmp_path: Path, monkeypatch, capsys):
        html_out = tmp_path / "report.html"
        monkeypatch.setattr(
            sys, "argv",
            [
                "trust_tracker",
                "--scenarios", "1",
                "--seed", "9",
                "--out", str(html_out),
            ],
        )
        asyncio.run(_main())
        out = capsys.readouterr().out
        assert html_out.exists()
        # No "JSON export" line when --json is omitted
        assert "JSON export" not in out
