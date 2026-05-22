"""Tests for the Consensus Protocol Fuzzer (`src.fuzzer`).

Covers the pure / deterministic surface area:

* ``MutationPlan.fingerprint`` stability + collision behaviour
* ``generate_mutation_plan`` parameter contract + reproducibility under seed
* ``FuzzOutcome.classify`` priority ordering
* ``is_interesting`` boundary cases
* ``FuzzableAgent`` mutation application & confidence/weight clamping
* End-to-end ``fuzz()`` smoke run (tiny budget, fixed seed) + ``FuzzerStats`` accounting
* JSON / HTML report rendering (well-formed, contains key data)
* Recommendation generator branches
* CLI ``main`` round-trip (exports both HTML and JSON, autopilot path)
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import fuzzer  # noqa: E402
from src.fuzzer import (  # noqa: E402
    MUTATION_OPS,
    FuzzableAgent,
    FuzzOutcome,
    FuzzerStats,
    MutationPlan,
    _build_json_report,
    _generate_html,
    _generate_recommendations,
    _tag,
    fuzz,
    generate_mutation_plan,
    is_interesting,
    main,
    run_one_iteration,
)


# ---------------------------------------------------------------------------
# MutationPlan / fingerprint
# ---------------------------------------------------------------------------


def test_mutation_plan_fingerprint_is_stable_and_order_independent():
    p1 = MutationPlan(ops=["a", "b", "c"], params={}, affected_agents=[])
    p2 = MutationPlan(ops=["c", "a", "b"], params={"unused": 1}, affected_agents=["x"])
    assert p1.fingerprint == p2.fingerprint
    assert len(p1.fingerprint) == 8
    # Hex digest only.
    int(p1.fingerprint, 16)


def test_mutation_plan_fingerprint_distinguishes_different_op_sets():
    a = MutationPlan(ops=["a", "b"], params={}, affected_agents=[])
    b = MutationPlan(ops=["a", "b", "c"], params={}, affected_agents=[])
    assert a.fingerprint != b.fingerprint


# ---------------------------------------------------------------------------
# generate_mutation_plan
# ---------------------------------------------------------------------------


def test_generate_mutation_plan_respects_seed():
    energy = {op: 1.0 for op in MUTATION_OPS}
    agents = [f"agent-{i}" for i in range(5)]
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    plan1 = generate_mutation_plan(agents, rng1, energy)
    plan2 = generate_mutation_plan(agents, rng2, energy)
    assert sorted(plan1.ops) == sorted(plan2.ops)
    assert plan1.params == plan2.params
    assert plan1.affected_agents == plan2.affected_agents


def test_generate_mutation_plan_only_uses_known_ops_and_known_agents():
    energy = {op: 1.0 for op in MUTATION_OPS}
    agents = [f"agent-{i}" for i in range(7)]
    rng = random.Random(0)
    for _ in range(50):
        plan = generate_mutation_plan(agents, rng, energy)
        assert 1 <= len(plan.ops) <= 3
        assert set(plan.ops).issubset(set(MUTATION_OPS))
        assert set(plan.affected_agents).issubset(set(agents))
        assert 1 <= len(plan.affected_agents) <= max(1, len(agents) // 2)
        # Every chosen op must appear as a key in params (either real value or sentinel True).
        for op in plan.ops:
            assert op in plan.params


def test_generate_mutation_plan_param_value_ranges():
    energy = {op: 1.0 for op in MUTATION_OPS}
    rng = random.Random(42)
    # Force ops by mocking the chosen_ops path: run many trials and check each
    # op's params land in the documented range whenever it appears.
    seen = set()
    for _ in range(500):
        plan = generate_mutation_plan(["a", "b", "c", "d"], rng, energy)
        for op in plan.ops:
            seen.add(op)
            v = plan.params[op]
            if op == "confidence_noise":
                assert 0.1 <= v <= 0.5
            elif op == "vote_weight_noise":
                assert 0.1 <= v <= 0.6
            elif op == "threshold_perturb":
                assert -0.3 <= v <= 0.3
            elif op == "quorum_shrink":
                assert isinstance(v, int) and v >= 1
            elif op == "quorum_bloat":
                assert isinstance(v, int) and 1 <= v <= 3
            elif op == "late_vote_drop":
                assert 0.2 <= v <= 0.6
            elif op == "duplicate_proposal":
                assert v == "duplicated-answer"
            else:
                assert v is True
    # Sanity: we should have exercised most mutation ops in 500 draws.
    assert len(seen) >= len(MUTATION_OPS) // 2


# ---------------------------------------------------------------------------
# FuzzOutcome.classify / is_interesting
# ---------------------------------------------------------------------------


def _bare_plan() -> MutationPlan:
    return MutationPlan(ops=["confidence_noise"], params={"confidence_noise": 0.1},
                        affected_agents=["agent-0"])


def test_classify_crash_takes_priority():
    o = FuzzOutcome(iteration=0, mutation_plan=_bare_plan(), committed=True,
                    rounds_used=1, aggregate_weight=0.5, slashed_count=2,
                    exception="boom")
    assert o.classify() == "crash"


def test_classify_commit_with_slash_then_no_commit_then_trivial_then_slow_then_normal():
    p = _bare_plan()
    # commit_with_slash wins over trivial/slow.
    assert FuzzOutcome(0, p, True, 5, 0.99, 1).classify() == "commit_with_slash"
    assert FuzzOutcome(0, p, False, 0, 0.0, 0).classify() == "no_commit"
    assert FuzzOutcome(0, p, True, 1, 0.99, 0).classify() == "trivial_commit"
    assert FuzzOutcome(0, p, True, 3, 0.5, 0).classify() == "slow_commit"
    assert FuzzOutcome(0, p, True, 1, 0.5, 0).classify() == "normal"


def test_is_interesting_matrix():
    p = _bare_plan()
    truth = {
        "crash": True,
        "commit_with_slash": True,
        "no_commit": True,
        "slow_commit": True,
        "trivial_commit": False,
        "normal": False,
    }
    for cls, expected in truth.items():
        o = FuzzOutcome(0, p, True, 1, 0.5, 0)
        o.classification = cls
        assert is_interesting(o) is expected, cls


# ---------------------------------------------------------------------------
# FuzzableAgent behaviour
# ---------------------------------------------------------------------------


def test_fuzzable_agent_generate_proposal_honours_mutations_and_clamps():
    random.seed(7)
    agent = FuzzableAgent("agent-0", honest=True)
    agent.apply_mutations({
        "confidence_clamp_high": True,
        "proof_scramble": True,
        "duplicate_proposal": "dup-answer",
    })
    prop = asyncio.run(agent.generate_proposal("task"))
    assert prop.confidence == 1.0
    assert prop.solution == "dup-answer"
    assert prop.proof.startswith("garbled-")
    agent.clear_mutations()
    assert agent._mutations == {}


def test_fuzzable_agent_verify_proposal_clamps_weight():
    random.seed(11)
    agent = FuzzableAgent("agent-1", honest=False)
    agent.apply_mutations({"vote_weight_noise": 5.0, "vote_flip_random": True})

    class _Stub:
        proposal_id = "p-1"

    vote = asyncio.run(agent.verify_proposal(_Stub()))
    assert -1.0 <= vote.weight <= 1.0
    assert vote.voter_id == "agent-1"
    if vote.weight < 0:
        assert vote.counter_proof == "counter-agent-1"
    else:
        assert vote.counter_proof is None


# ---------------------------------------------------------------------------
# Fuzz run + stats accounting
# ---------------------------------------------------------------------------


def test_fuzz_smoke_run_is_deterministic_and_accounting_consistent():
    outcomes_a, stats_a = asyncio.run(fuzz(n_agents=3, iterations=15, seed=99))
    outcomes_b, stats_b = asyncio.run(fuzz(n_agents=3, iterations=15, seed=99))

    # Determinism: same seed -> same classification sequence + same stats counters.
    assert [o.classification for o in outcomes_a] == [o.classification for o in outcomes_b]
    assert stats_a.total_iterations == stats_b.total_iterations == len(outcomes_a) == 15
    assert dict(stats_a.classifications) == dict(stats_b.classifications)
    assert stats_a.unique_fingerprints == stats_b.unique_fingerprints

    # Accounting invariants.
    assert sum(stats_a.classifications.values()) == stats_a.total_iterations
    # interesting_count <= total
    assert stats_a.interesting_count <= stats_a.total_iterations
    # Every counted op_interesting hit must also be reflected in op_hit_count.
    for op, n in stats_a.op_interesting_count.items():
        assert stats_a.op_hit_count.get(op, 0) >= n
    # crash_count matches outcomes carrying exceptions.
    assert stats_a.crash_count == sum(1 for o in outcomes_a if o.exception)


def test_fuzz_autopilot_can_short_circuit_after_boring_streak():
    # With 0 mutations forced "interesting" the loop still completes; we just
    # assert it terminates within the budget when autopilot=True.
    outcomes, stats = asyncio.run(fuzz(n_agents=3, iterations=120, seed=1, autopilot=True))
    assert stats.total_iterations <= 120
    assert len(outcomes) == stats.total_iterations


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _tiny_run():
    return asyncio.run(fuzz(n_agents=3, iterations=10, seed=7))


def test_build_json_report_shape():
    outcomes, stats = _tiny_run()
    report = _build_json_report(outcomes, stats)
    assert set(report.keys()) == {"summary", "mutation_effectiveness", "findings"}
    s = report["summary"]
    assert s["total_iterations"] == stats.total_iterations
    assert s["unique_fingerprints"] == stats.unique_fingerprints
    assert s["interesting_count"] == stats.interesting_count
    assert s["crash_count"] == stats.crash_count
    # findings only carry interesting outcomes
    assert len(report["findings"]) == stats.interesting_count
    # mutation_effectiveness rate is a percentage 0..100
    for op, d in report["mutation_effectiveness"].items():
        assert 0.0 <= d["rate"] <= 100.0
        assert d["interesting"] <= d["total"]


def test_generate_html_contains_charts_and_summary_numbers():
    outcomes, stats = _tiny_run()
    html = _generate_html(outcomes, stats)
    assert html.startswith("<!DOCTYPE html>")
    assert "Chart" in html  # chart.js call present
    assert "Consensus Protocol Fuzzer" in html
    # Token replacement happened (no raw placeholders left).
    for placeholder in ["ITER_TOTAL", "UNIQUE_FP", "CLS_LABELS", "OP_TOTAL", "RECS"]:
        assert placeholder not in html, placeholder
    assert str(stats.total_iterations) in html


def test_tag_html_escapes_user_text():
    out = _tag("<script>alert(1)</script>", "tag-crash")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert 'class="tag tag-crash"' in out


def test_generate_recommendations_branches():
    plan = _bare_plan()
    stats = FuzzerStats()
    stats.total_iterations = 10
    stats.crash_count = 2
    stats.classifications = Counter({"no_commit": 5, "slow_commit": 12})
    stats.interesting_count = 7
    stats.op_hit_count = Counter({"confidence_noise": 10, "vote_flip_random": 5})
    stats.op_interesting_count = Counter({"confidence_noise": 8, "vote_flip_random": 1})
    crash_outcome = FuzzOutcome(1, plan, False, 0, 0.0, 0, exception="kaboom")
    recs = _generate_recommendations(stats, [crash_outcome])
    assert "2 crashes detected" in recs
    assert "confidence_noise" in recs
    assert "no_commit" not in recs  # we render human-readable text, not the key
    # Slow-commit branch
    assert "slow commits" in recs
    # No-commit threshold branch (5/10 = 50% > 30%)
    assert "failed to commit" in recs


def test_generate_recommendations_no_concerns_branch():
    recs = _generate_recommendations(FuzzerStats(), [])
    assert "No major concerns" in recs


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".json", ".html"])
def test_cli_main_exports_report(tmp_path, monkeypatch, capsys, ext):
    out = tmp_path / f"report{ext}"
    monkeypatch.setattr(
        sys, "argv",
        ["fuzzer", "--iterations", "8", "--agents", "3", "--seed", "5",
         "--autopilot", "--export", str(out)],
    )
    asyncio.run(main())
    captured = capsys.readouterr().out
    assert "Completed" in captured
    assert "Classification breakdown" in captured
    assert out.exists() and out.stat().st_size > 0
    if ext == ".json":
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "summary" in data and data["summary"]["total_iterations"] >= 1
    else:
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_cli_main_without_export_still_runs(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv",
        ["fuzzer", "--iterations", "5", "--agents", "3", "--seed", "1"],
    )
    asyncio.run(main())
    out = capsys.readouterr().out
    assert "Iterations" in out


# ---------------------------------------------------------------------------
# run_one_iteration direct invocation (covers exception path)
# ---------------------------------------------------------------------------


def test_run_one_iteration_records_exception(monkeypatch):
    plan = MutationPlan(
        ops=["confidence_noise"],
        params={"confidence_noise": 0.1},
        affected_agents=["agent-0"],
    )
    agents = [FuzzableAgent("agent-0", honest=True),
              FuzzableAgent("agent-1", honest=True),
              FuzzableAgent("agent-2", honest=False)]

    class _Boom:
        def __init__(self, *_a, **_kw):
            raise RuntimeError("synthetic engine failure")

    monkeypatch.setattr(fuzzer, "MBFTEngine", _Boom)
    out = asyncio.run(run_one_iteration(0, agents, plan, base_threshold=0.6))
    assert out.exception is not None
    assert "synthetic engine failure" in out.exception
    assert out.classify() == "crash"
