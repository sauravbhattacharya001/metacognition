"""Tests for Swarm Autophagy Engine."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.autophagy import (
    AutophagyEngine,
    AutophagyReport,
    RecycleEntry,
    MODES,
    STALE_THRESHOLD_ROUNDS,
    MEMORY_ZOMBIE_THRESHOLD,
    WASTE_AGE_THRESHOLD,
    SENESCENCE_WINDOW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine(mode="recycle", cooldown=0, stress=0.0):
    return AutophagyEngine(mode=mode, cooldown_rounds=cooldown, stress_level=stress)


def feed_active_round(engine, agents, round_num):
    """Feed a round where all agents are active."""
    activity = {a: {"votes": 3, "memory_accesses": 4, "state_updates": 2, "performance_score": 0.9} for a in agents}
    engine.record_round(round_num=round_num, agent_activity=activity)


def feed_stale_rounds(engine, stale_agents, active_agents, rounds):
    """Feed multiple rounds where stale_agents do nothing."""
    for r in rounds:
        activity = {}
        for a in stale_agents:
            activity[a] = {"votes": 0, "memory_accesses": 0, "state_updates": 0}
        for a in active_agents:
            activity[a] = {"votes": 2, "memory_accesses": 3, "state_updates": 1}
        engine.record_round(round_num=r, agent_activity=activity)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_mode(self):
        e = AutophagyEngine()
        assert e.mode == "monitor"

    def test_custom_mode(self):
        e = AutophagyEngine(mode="recycle")
        assert e.mode == "recycle"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            AutophagyEngine(mode="destroy")

    def test_stress_clamped(self):
        e = AutophagyEngine(stress_level=2.0)
        assert e.stress_level == 1.0
        e2 = AutophagyEngine(stress_level=-1.0)
        assert e2.stress_level == 0.0

    def test_initial_state_empty(self):
        e = AutophagyEngine()
        assert e.current_round == 0
        assert len(e.dysfunctions) == 0
        assert len(e.lysosome_queue) == 0
        assert len(e.recycling_ledger) == 0


# ---------------------------------------------------------------------------
# Stale Agent Detection
# ---------------------------------------------------------------------------

class TestStaleAgentDetector:
    def test_detects_stale_agent(self):
        e = make_engine()
        feed_stale_rounds(e, ["lazy"], ["active"], range(1, STALE_THRESHOLD_ROUNDS + 2))
        dys = e.detect()
        stale = [d for d in dys if d.category == "stale_agent"]
        assert len(stale) >= 1
        assert "lazy" in stale[0].targets

    def test_no_false_positive_for_active(self):
        e = make_engine()
        for r in range(1, 10):
            feed_active_round(e, ["a1", "a2"], r)
        dys = e.detect()
        stale = [d for d in dys if d.category == "stale_agent"]
        assert len(stale) == 0

    def test_not_triggered_below_threshold(self):
        e = make_engine()
        feed_stale_rounds(e, ["lazy"], ["active"], range(1, STALE_THRESHOLD_ROUNDS))
        dys = e.detect()
        stale = [d for d in dys if d.category == "stale_agent"]
        assert len(stale) == 0


# ---------------------------------------------------------------------------
# Zombie Memory Detection
# ---------------------------------------------------------------------------

class TestZombieMemoryDetector:
    def test_detects_zombie_memory(self):
        e = make_engine()
        # Create memory, never access again
        e.record_round(1, {"a1": {"votes": 1, "memory_accesses": 1, "state_updates": 1}},
                       memory_accesses={"mem_old": False})
        # Advance rounds
        for r in range(2, MEMORY_ZOMBIE_THRESHOLD + 3):
            e.record_round(r, {"a1": {"votes": 1, "memory_accesses": 1, "state_updates": 1}})
        dys = e.detect()
        zombies = [d for d in dys if d.category == "zombie_memory"]
        assert len(zombies) >= 1
        assert "mem_old" in zombies[0].targets

    def test_no_zombie_if_accessed(self):
        e = make_engine()
        for r in range(1, 10):
            e.record_round(r, {"a1": {"votes": 1, "memory_accesses": 1, "state_updates": 1}},
                           memory_accesses={"mem_active": True})
        dys = e.detect()
        zombies = [d for d in dys if d.category == "zombie_memory"]
        assert all("mem_active" not in d.targets for d in zombies)


# ---------------------------------------------------------------------------
# Circular Dependency Detection
# ---------------------------------------------------------------------------

class TestCircularDependencyDetector:
    def test_detects_echo_chamber(self):
        e = make_engine()
        # Build strong mutual interaction pattern
        for r in range(1, 6):
            e.record_round(r, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1},
                               "b": {"votes": 1, "memory_accesses": 1, "state_updates": 1}},
                           interactions={"a": ["b"] * 10, "b": ["a"] * 10})
        dys = e.detect()
        circular = [d for d in dys if d.category == "circular_dependency"]
        assert len(circular) >= 1

    def test_no_detection_with_diverse_interactions(self):
        e = make_engine()
        agents = [f"a{i}" for i in range(5)]
        for r in range(1, 6):
            activity = {a: {"votes": 1, "memory_accesses": 1, "state_updates": 1} for a in agents}
            interactions = {a: [agents[(i + 1) % 5], agents[(i + 2) % 5]] for i, a in enumerate(agents)}
            e.record_round(r, activity, interactions=interactions)
        dys = e.detect()
        circular = [d for d in dys if d.category == "circular_dependency"]
        assert len(circular) == 0


# ---------------------------------------------------------------------------
# Metabolic Waste Detection
# ---------------------------------------------------------------------------

class TestMetabolicWasteDetector:
    def test_detects_old_temp_state(self):
        e = make_engine()
        e.record_round(1, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}},
                       temp_state_ids=["tmp_1", "tmp_2"])
        for r in range(2, WASTE_AGE_THRESHOLD + 3):
            e.record_round(r, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}})
        dys = e.detect()
        waste = [d for d in dys if d.category == "metabolic_waste"]
        assert len(waste) >= 2

    def test_no_waste_if_recent(self):
        e = make_engine()
        e.record_round(1, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}},
                       temp_state_ids=["tmp_new"])
        e.record_round(2, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}})
        dys = e.detect()
        waste = [d for d in dys if d.category == "metabolic_waste"]
        assert len(waste) == 0


# ---------------------------------------------------------------------------
# Senescent Agent Detection
# ---------------------------------------------------------------------------

class TestSenescentAgentDetector:
    def test_detects_declining_performance(self):
        e = make_engine()
        for r in range(1, SENESCENCE_WINDOW + 2):
            perf = max(0.1, 1.0 - r * 0.08)
            e.record_round(r, {"declining": {"votes": 1, "memory_accesses": 1, "state_updates": 1, "performance_score": perf}})
        dys = e.detect()
        senescent = [d for d in dys if d.category == "senescent_agent"]
        assert len(senescent) >= 1
        assert "declining" in senescent[0].targets

    def test_no_detection_stable_performance(self):
        e = make_engine()
        for r in range(1, SENESCENCE_WINDOW + 2):
            e.record_round(r, {"stable": {"votes": 2, "memory_accesses": 2, "state_updates": 1, "performance_score": 0.85}})
        dys = e.detect()
        senescent = [d for d in dys if d.category == "senescent_agent"]
        assert len(senescent) == 0


# ---------------------------------------------------------------------------
# Protein Misfolding Detection
# ---------------------------------------------------------------------------

class TestProteinMisfoldingDetector:
    def test_detects_conflicting_beliefs(self):
        e = make_engine()
        e.record_round(1, {"confused": {
            "votes": 1, "memory_accesses": 1, "state_updates": 1,
            "beliefs": {"up": 5.0, "down": -5.0, "left": 3.0, "right": -3.0}
        }})
        dys = e.detect()
        misfolded = [d for d in dys if d.category == "protein_misfolding"]
        assert len(misfolded) >= 1

    def test_no_detection_consistent_beliefs(self):
        e = make_engine()
        e.record_round(1, {"coherent": {
            "votes": 2, "memory_accesses": 2, "state_updates": 1,
            "beliefs": {"growth": 5.0, "expansion": 3.0, "positive": True}
        }})
        dys = e.detect()
        misfolded = [d for d in dys if d.category == "protein_misfolding"]
        assert len(misfolded) == 0

    def test_no_beliefs_no_detection(self):
        e = make_engine()
        e.record_round(1, {"simple": {"votes": 2, "memory_accesses": 2, "state_updates": 1}})
        dys = e.detect()
        misfolded = [d for d in dys if d.category == "protein_misfolding"]
        assert len(misfolded) == 0


# ---------------------------------------------------------------------------
# Organelle Dysfunction Detection
# ---------------------------------------------------------------------------

class TestOrganelleDysfunctionDetector:
    def test_detects_degraded_subsystem(self):
        e = make_engine()
        e.record_round(1, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}},
                       subsystem_health={"voting": 0.1, "memory": 0.9})
        dys = e.detect()
        organelle = [d for d in dys if d.category == "organelle_dysfunction"]
        assert len(organelle) >= 1
        assert "voting" in organelle[0].targets

    def test_no_detection_healthy_subsystems(self):
        e = make_engine()
        e.record_round(1, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}},
                       subsystem_health={"voting": 0.8, "memory": 0.9})
        dys = e.detect()
        organelle = [d for d in dys if d.category == "organelle_dysfunction"]
        assert len(organelle) == 0


# ---------------------------------------------------------------------------
# Mode & Processing
# ---------------------------------------------------------------------------

class TestModes:
    def test_monitor_no_tagging(self):
        e = make_engine(mode="monitor")
        feed_stale_rounds(e, ["lazy"], ["active"], range(1, 6))
        dys = e.detect()
        assert all(not d.tagged for d in dys)
        assert len(e.lysosome_queue) == 0

    def test_tag_mode_tags(self):
        e = make_engine(mode="tag")
        feed_stale_rounds(e, ["lazy"], ["active"], range(1, 6))
        dys = e.detect()
        tagged = [d for d in dys if d.tagged]
        assert len(tagged) > 0
        assert len(e.lysosome_queue) > 0

    def test_recycle_mode_recycles(self):
        e = make_engine(mode="recycle", cooldown=0)
        feed_stale_rounds(e, ["lazy"], ["active"], range(1, 6))
        e.detect()
        results = e.process_queue()
        assert len(results) > 0
        assert all(isinstance(r, RecycleEntry) for r in results)

    def test_degrade_mode_degrades(self):
        e = make_engine(mode="degrade", cooldown=0)
        feed_stale_rounds(e, ["lazy"], ["active"], range(1, 6))
        e.detect()
        e.process_queue()
        degraded = [d for d in e.lysosome_queue if d.degraded]
        assert len(degraded) > 0


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_cooldown_prevents_processing(self):
        e = make_engine(mode="recycle", cooldown=3)
        feed_stale_rounds(e, ["lazy"], ["active"], range(1, 6))
        e.detect()
        first = e.process_queue()
        assert len(first) > 0
        # Add more to queue
        feed_stale_rounds(e, ["lazy2"], ["active"], range(6, 12))
        e.detect()
        second = e.process_queue()
        assert len(second) == 0  # blocked by cooldown

    def test_cooldown_decrements(self):
        e = make_engine(mode="recycle", cooldown=2)
        e.cooldown_remaining = 2
        e.process_queue()
        assert e.cooldown_remaining == 1
        e.process_queue()
        assert e.cooldown_remaining == 0


# ---------------------------------------------------------------------------
# Stress Escalation
# ---------------------------------------------------------------------------

class TestStressEscalation:
    def test_escalates_mode_on_high_stress(self):
        e = make_engine(mode="monitor", stress=0.0)
        e.stress_level = 0.8
        e.record_round(1, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}})
        assert e.mode == "tag"

    def test_no_escalation_below_threshold(self):
        e = make_engine(mode="monitor", stress=0.5)
        e.record_round(1, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}})
        assert e.mode == "monitor"

    def test_escalation_caps_at_recycle(self):
        e = make_engine(mode="recycle", stress=0.9)
        e.record_round(1, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}})
        assert e.mode == "recycle"


# ---------------------------------------------------------------------------
# Queue Prioritization
# ---------------------------------------------------------------------------

class TestQueuePrioritization:
    def test_queue_sorted_by_severity(self):
        e = make_engine(mode="tag")
        # Create mix of severities
        e.record_round(1, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}},
                       subsystem_health={"critical": 0.05, "mild": 0.25})
        e.record_round(1, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}},
                       temp_state_ids=["waste1"])
        for r in range(2, WASTE_AGE_THRESHOLD + 3):
            e.record_round(r, {"a": {"votes": 1, "memory_accesses": 1, "state_updates": 1}},
                           subsystem_health={"critical": 0.05, "mild": 0.25})
        e.detect()
        if len(e.lysosome_queue) >= 2:
            severities = [d.severity for d in e.lysosome_queue]
            assert severities == sorted(severities, reverse=True)


# ---------------------------------------------------------------------------
# Recycling Ledger
# ---------------------------------------------------------------------------

class TestRecyclingLedger:
    def test_ledger_records_recycled(self):
        e = make_engine(mode="recycle", cooldown=0)
        feed_stale_rounds(e, ["x"], ["y"], range(1, 6))
        e.detect()
        e.process_queue()
        assert len(e.recycling_ledger) > 0
        entry = e.recycling_ledger[0]
        assert entry.resources_recovered > 0
        assert entry.recycled_at > 0

    def test_ledger_empty_without_recycle(self):
        e = make_engine(mode="tag")
        feed_stale_rounds(e, ["x"], ["y"], range(1, 6))
        e.detect()
        e.process_queue()
        assert len(e.recycling_ledger) == 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_structure(self):
        e = make_engine()
        report = e.get_report()
        assert isinstance(report, AutophagyReport)
        assert 0 <= report.score <= 100
        assert report.mode in MODES
        assert isinstance(report.dysfunction_counts, dict)

    def test_clean_swarm_high_score(self):
        e = make_engine()
        for r in range(1, 5):
            feed_active_round(e, ["a", "b", "c"], r)
        report = e.get_report()
        assert report.score >= 80

    def test_dysfunctional_swarm_low_score(self):
        e = make_engine(mode="tag", stress=0.8)
        feed_stale_rounds(e, ["s1", "s2", "s3"], ["a1"], range(1, 8))
        e.record_round(8, {"a1": {"votes": 1, "memory_accesses": 1, "state_updates": 1}},
                       subsystem_health={"voting": 0.1, "memory": 0.1})
        e.detect()
        report = e.get_report()
        assert report.score < 50


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self):
        e = make_engine(mode="tag", cooldown=3, stress=0.4)
        feed_stale_rounds(e, ["lazy"], ["active"], range(1, 6))
        e.detect()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        e.save(path)
        loaded = AutophagyEngine.load(path)

        assert loaded.mode == "tag"
        assert loaded.cooldown_rounds == 3
        assert loaded.stress_level == pytest.approx(0.4)
        assert len(loaded.dysfunctions) == len(e.dysfunctions)
        assert len(loaded.lysosome_queue) == len(e.lysosome_queue)
        Path(path).unlink()

    def test_save_creates_file(self):
        e = make_engine()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        e.save(path)
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert "mode" in data
        Path(path).unlink()


# ---------------------------------------------------------------------------
# HTML Export
# ---------------------------------------------------------------------------

class TestHtmlExport:
    def test_export_creates_file(self):
        e = make_engine(mode="recycle", cooldown=0)
        feed_stale_rounds(e, ["lazy"], ["active"], range(1, 6))
        e.detect()
        e.process_queue()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        e.export_html(path)
        content = Path(path).read_text()
        assert "Swarm Autophagy Dashboard" in content
        assert "autophagy" in content.lower()
        Path(path).unlink()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_cli_runs(self, capsys):
        AutophagyEngine.run_cli(["--agents", "5", "--rounds", "10", "--mode", "recycle"])
        captured = capsys.readouterr()
        assert "Swarm Autophagy Simulation" in captured.out
        assert "Final Report" in captured.out

    def test_cli_json_output(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        AutophagyEngine.run_cli(["--agents", "5", "--rounds", "10", "--json", path])
        data = json.loads(Path(path).read_text())
        assert "report" in data
        Path(path).unlink()

    def test_cli_html_output(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        AutophagyEngine.run_cli(["--agents", "5", "--rounds", "10", "--output", path])
        content = Path(path).read_text()
        assert "<!DOCTYPE html>" in content
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_swarm(self):
        e = make_engine()
        dys = e.detect()
        assert len(dys) == 0
        report = e.get_report()
        assert report.score == 100.0

    def test_all_agents_dysfunctional(self):
        e = make_engine(mode="recycle", cooldown=0)
        agents = [f"a{i}" for i in range(5)]
        feed_stale_rounds(e, agents, [], range(1, 8))
        dys = e.detect()
        assert len(dys) > 0

    def test_single_agent_swarm(self):
        e = make_engine()
        for r in range(1, 5):
            e.record_round(r, {"solo": {"votes": 1, "memory_accesses": 1, "state_updates": 1}})
        dys = e.detect()
        # No dysfunction for an active solo agent
        stale = [d for d in dys if d.category == "stale_agent"]
        assert len(stale) == 0

    def test_set_stress(self):
        e = make_engine()
        e.set_stress(0.9)
        assert e.stress_level == 0.9
        e.set_stress(5.0)
        assert e.stress_level == 1.0
        e.set_stress(-1.0)
        assert e.stress_level == 0.0

    def test_score_history_grows(self):
        e = make_engine()
        e.get_report()
        e.get_report()
        e.get_report()
        assert len(e.score_history) == 3


# ---------------------------------------------------------------------------
# Protein Misfolding Detection — O(K) refactor invariants (run #4443)
# ---------------------------------------------------------------------------
# These tests pin down the semantics that the O(n_keys**2) -> O(n_keys)
# refactor of _detect_protein_misfolding relies on. They are deliberately
# property-style so any future change that drifts from the bucket-count
# formula will trip at least one assertion.

class TestProteinMisfoldingBucketSemantics:
    """The O(K) implementation derives `conflicts` from bucket counts.

    Conflict count formula (must hold for any belief dict):
        conflicts = (#True * #False) + ((#True + #pos_num) * #neg_num)
    where `#pos_num` / `#neg_num` count *non-bool* numerics with
    strictly positive / strictly negative values, and #True / #False
    count bools (which in Python are also int but get the bool arm).
    """

    @staticmethod
    def _detect(beliefs):
        e = make_engine()
        e.record_round(1, {"agent": {
            "votes": 1, "memory_accesses": 1, "state_updates": 1,
            "beliefs": beliefs,
        }})
        return [d for d in e.detect() if d.category == "protein_misfolding"]

    def test_all_bools_one_class_no_conflict(self):
        # 4 `True` and 0 `False` -> bool*bool conflicts = 0, no numerics,
        # so total_pairs = 6, conflicts = 0, ratio = 0 -> nothing flagged.
        assert self._detect({"a": True, "b": True, "c": True, "d": True}) == []

    def test_bool_split_above_threshold(self):
        # 2 True + 2 False -> conflicts = 2*2 = 4 out of C(4,2)=6 = 0.666
        # which exceeds MISFOLDING_CONFLICT_THRESHOLD (0.5).
        out = self._detect({"a": True, "b": True, "c": False, "d": False})
        assert len(out) == 1
        # severity is min(1.0, conflict_ratio); 4/6 ≈ 0.667.
        assert 0.6 < out[0].severity < 0.7

    def test_numeric_signs_only(self):
        # 2 pos + 2 neg numerics -> 2*2 = 4 conflicts / 6 pairs = 0.667.
        out = self._detect({"a": 1.0, "b": 2.0, "c": -1.0, "d": -3.0})
        assert len(out) == 1

    def test_zero_numeric_never_conflicts(self):
        # 0.0 is neither >0 nor <0; pairs with it never count as conflict.
        # 1 pos + 1 zero + 1 neg -> 1*1 = 1 conflict / 3 pairs = 0.333 < 0.5.
        assert self._detect({"a": 1.0, "b": 0.0, "c": -1.0}) == []

    def test_true_bool_acts_as_positive_against_negative_number(self):
        # `True` is a bool (so it's not in #pos_num), but it is also an
        # int, and the original elif arm fired for (bool, number) pairs
        # when their signs differed. The bucket formula encodes this via
        # the `(#True + #pos_num) * #neg_num` term.
        # Setup: 1 True, 0 False, 0 #pos_num, 3 #neg_num. Pairs = C(4,2)=6.
        # bool*bool conflicts = 1*0 = 0.
        # mixed-sign conflicts = (1+0) * 3 = 3 -> ratio = 3/6 = 0.5 (>=0.5).
        out = self._detect({"flag": True, "x": -1.0, "y": -2.0, "z": -3.0})
        assert len(out) == 1

    def test_false_bool_never_creates_sign_conflict(self):
        # `False` is bool, so it's excluded from #pos_num and #neg_num.
        # 1 False + 3 negatives -> bool*bool = 0, mixed = (0+0)*3 = 0.
        assert self._detect({"flag": False, "x": -1.0, "y": -2.0, "z": -3.0}) == []

    def test_non_numeric_values_ignored_but_count_in_pairs(self):
        # Strings / None never go into any bucket but they still take up
        # a slot in total_pairs = C(n,2). 2 pos + 2 neg = 4 conflicts;
        # adding 2 strings pushes total_pairs from 6 to 15, ratio = 4/15
        # ≈ 0.267 < 0.5, so nothing should be flagged.
        beliefs = {"a": 1.0, "b": 2.0, "c": -1.0, "d": -2.0,
                   "e": "hello", "f": None}
        assert self._detect(beliefs) == []

    def test_large_belief_set_is_fast(self):
        # Sanity check that the O(K) path doesn't accidentally regress
        # back to O(K**2). With K=500 the old code did 124k pair
        # comparisons + isinstance checks per agent per detect(); the
        # bucket form is a single 500-element pass. Wall-clock budget is
        # generous (1s) so this is not flaky, but the old implementation
        # already creeps into 10s of ms here, so any reintroduced
        # quadratic walk would show up as a regression on CI.
        import time
        beliefs = {f"k{i}": (1.0 if i % 2 == 0 else -1.0) for i in range(500)}
        e = make_engine()
        e.record_round(1, {"agent": {
            "votes": 1, "memory_accesses": 1, "state_updates": 1,
            "beliefs": beliefs,
        }})
        t0 = time.perf_counter()
        dys = e.detect()
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"detect() took {elapsed:.3f}s for K=500 beliefs"
        misfolded = [d for d in dys if d.category == "protein_misfolding"]
        # 250 pos + 250 neg numerics -> 250*250 = 62500 conflicts /
        # C(500,2) = 124750 pairs ≈ 0.5008 -> just over threshold.
        assert len(misfolded) == 1
