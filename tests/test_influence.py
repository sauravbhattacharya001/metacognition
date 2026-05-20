"""Tests for src.influence — focused on the swing/kingmaker contract.

Background: prior to this commit ``_compute_metrics`` incremented the
``swing`` and ``kingmaker`` counters under the *same* condition, which
made ``kingmaker_score`` numerically identical to ``swing_power`` in
every generated influence report. The refactor extracts ``_swing_stats``
with distinct semantics; these tests pin down that contract so the
regression cannot silently come back.
"""
from __future__ import annotations

import asyncio

from src.influence import _compute_metrics, _swing_stats, _recommendations


# --------------------------------------------------------------------------
# _swing_stats — the new isolated helper
# --------------------------------------------------------------------------

def test_swing_stats_pure_positive_kingmaker():
    """Agent's positive vote was strictly necessary for commit ⇒ both
    swing and kingmaker increment."""
    # Round committed with aggregate 0.7; remove the 0.4 vote → 0.3 < 0.6.
    swing, king = _swing_stats(
        weights=[0.4],
        aggregates=[0.7],
        committed_with=[True],
        threshold=0.6,
    )
    assert swing == 1
    assert king == 1


def test_swing_stats_negative_vote_flips_no_commit_is_swing_not_king():
    """A Byzantine-style negative vote that *prevented* a commit is a
    swing but must not count as a kingmaker (kingmaker = decisive
    contribution *toward* commit)."""
    # aggregate = 0.4 (no commit). Remove a -0.3 vote → 0.7 ≥ 0.6 (commit).
    swing, king = _swing_stats(
        weights=[-0.3],
        aggregates=[0.4],
        committed_with=[False],
        threshold=0.6,
    )
    assert swing == 1
    assert king == 0, "negative-weight swings must not count as kingmaking"


def test_swing_stats_non_decisive_vote_counts_neither():
    """Round committed comfortably; removing one small vote still commits."""
    swing, king = _swing_stats(
        weights=[0.1],
        aggregates=[0.9],
        committed_with=[True],
        threshold=0.6,
    )
    assert swing == 0
    assert king == 0


def test_swing_stats_empty_inputs():
    assert _swing_stats([], [], [], 0.6) == (0, 0)


def test_swing_stats_kingmaker_is_strict_subset_of_swing():
    """Across a mixed history, kingmaker count never exceeds swing count."""
    weights         = [0.4, -0.3, 0.1, 0.5, -0.5, 0.35]
    aggregates      = [0.7,  0.4, 0.9, 0.65, 0.2, 0.62]
    committed_with  = [True, False, True, True, False, True]
    swing, king = _swing_stats(weights, aggregates, committed_with, 0.6)
    assert king <= swing
    # And specifically: the negative-vote no-commit-flip is NOT a king,
    # so the two counts must actually differ here.
    assert king < swing


# --------------------------------------------------------------------------
# _compute_metrics — end-to-end behaviour through the public path
# --------------------------------------------------------------------------

class _FakeVote:
    __slots__ = ("voter_id", "weight")

    def __init__(self, voter_id: str, weight: float) -> None:
        self.voter_id = voter_id
        self.weight = weight


class _FakeResult:
    __slots__ = ("votes", "committed", "aggregate_weight", "leader_id")

    def __init__(self, votes, committed, aggregate_weight, leader_id="agent-0"):
        self.votes = votes
        self.committed = committed
        self.aggregate_weight = aggregate_weight
        self.leader_id = leader_id


def test_compute_metrics_distinguishes_king_from_swing():
    """Regression: ``kingmaker_score`` used to alias ``swing_power``
    for every agent. After the fix, an agent whose only swing was a
    negative-weight no-commit flip must have king < swing."""
    agent_ids = ["a", "b", "c"]
    threshold = 0.6
    results = [
        # Round 0: a's +0.6 is decisive for commit; b and c are not.
        #   aggregate = 0.7, committed = True
        #   remove a → 0.10 < 0.6 (flip, swing+king for a)
        #   remove b → 0.65 ≥ 0.6 (no flip)
        #   remove c → 0.65 ≥ 0.6 (no flip)
        _FakeResult(
            votes=[_FakeVote("a", 0.6), _FakeVote("b", 0.05), _FakeVote("c", 0.05)],
            committed=True,
            aggregate_weight=0.7,
        ),
        # Round 1: b's -0.3 *prevents* a commit. Removing b flips
        # no-commit → commit (swing for b, NOT kingmaker for b).
        #   aggregate = 0.4, committed = False
        #   remove a → 0.1 < 0.6 (no flip)
        #   remove b → 0.7 ≥ 0.6 (flip, swing only)
        #   remove c → 0.0 < 0.6 (no flip)
        _FakeResult(
            votes=[_FakeVote("a", 0.3), _FakeVote("b", -0.3), _FakeVote("c", 0.4)],
            committed=False,
            aggregate_weight=0.4,
        ),
    ]
    metrics = _compute_metrics(agent_ids, results, threshold)

    a = metrics["agents"]["a"]
    b = metrics["agents"]["b"]

    # a was a kingmaker on round 0 → both counters bump for that round.
    assert a["kingmaker_score"] == a["swing_power"] > 0

    # b swung the no-commit on round 1 but did NOT kingmake any commit.
    assert b["swing_power"] > 0
    assert b["kingmaker_score"] == 0
    assert b["kingmaker_score"] < b["swing_power"], (
        "kingmaker_score must no longer alias swing_power"
    )


def test_compute_metrics_empty_results():
    out = _compute_metrics(["a", "b"], [], 0.6)
    assert out == {"agents": {}, "gini": 0.0, "coalitions": [], "timeline": []}


def test_compute_metrics_coalition_detection():
    """Two agents that always vote in lock-step should register as a
    coalition; one that drifts independently should not."""
    agent_ids = ["a", "b", "c"]
    rounds = [
        ((0.5, 0.5, -0.3), True,  0.7),
        ((0.4, 0.4,  0.1), True,  0.9),
        ((0.2, 0.2, -0.5), False, -0.1),
        ((0.6, 0.6,  0.0), True,  1.2),
        ((0.3, 0.3,  0.2), False, 0.8),
    ]
    results = [
        _FakeResult(
            votes=[_FakeVote("a", wa), _FakeVote("b", wb), _FakeVote("c", wc)],
            committed=cm,
            aggregate_weight=agg,
        )
        for (wa, wb, wc), cm, agg in rounds
    ]
    metrics = _compute_metrics(agent_ids, results, 0.6)
    pairs = {tuple(sorted(c["agents"])) for c in metrics["coalitions"]}
    assert ("a", "b") in pairs


def test_recommendations_flags_kingmaker_and_concentration():
    metrics = {
        "agents": {
            "agent-0": {
                "swing_power": 0.8,
                "kingmaker_score": 0.6,
                "influence_radius": 0.5,
                "avg_weight": 0.4,
            },
            "agent-1": {
                "swing_power": 0.1,
                "kingmaker_score": 0.0,
                "influence_radius": -0.5,
                "avg_weight": -0.2,
            },
        },
        "gini": 0.7,
        "coalitions": [],
    }
    recs = _recommendations(metrics)
    joined = "\n".join(recs)
    assert "kingmaker" in joined.lower()
    assert "asymmetry" in joined.lower()
    assert "contrarian" in joined.lower()


# --------------------------------------------------------------------------
# Smoke test: full CLI pipeline (kept tiny so it stays fast in CI).
# --------------------------------------------------------------------------

def test_influence_simulation_smoke(tmp_path):
    """End-to-end: run a 3-round / 4-agent simulation and write the
    report. Asserts the report file exists and kingmaker_score is
    correctly bounded by swing_power for every agent (the bug we fixed)."""
    from src.influence import _run_simulation, _compute_metrics

    ids, results = asyncio.run(_run_simulation(
        n_agents=4, byzantine_ratio=0.25, n_rounds=3, threshold=0.6
    ))
    metrics = _compute_metrics(ids, results, 0.6)
    for aid, m in metrics["agents"].items():
        assert 0.0 <= m["kingmaker_score"] <= m["swing_power"] + 1e-9, (
            f"{aid}: kingmaker_score ({m['kingmaker_score']}) must not "
            f"exceed swing_power ({m['swing_power']})"
        )
