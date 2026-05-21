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
import json

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


# --------------------------------------------------------------------------
# Extracted helpers — pin their contracts so the refactor stays honest.
# --------------------------------------------------------------------------

def test_build_vote_matrix_fills_missing_voters_with_zero():
    """An agent that didn't vote in a round must show up as 0.0 weight,
    not be silently skipped — otherwise downstream correlations get
    misaligned with the outcome series."""
    from src.influence import _build_vote_matrix

    results = [
        _FakeResult(
            votes=[_FakeVote("a", 0.5)],  # b absent
            committed=True,
            aggregate_weight=0.5,
        ),
        _FakeResult(
            votes=[_FakeVote("a", 0.1), _FakeVote("b", -0.4)],
            committed=False,
            aggregate_weight=-0.3,
        ),
    ]
    matrix, outcomes = _build_vote_matrix(["a", "b"], results)
    assert matrix == {"a": [0.5, 0.1], "b": [0.0, -0.4]}
    assert outcomes == [1, 0]


def test_detect_coalitions_respects_threshold_and_skips_self_pairs():
    """Only ordered (i<j) pairs above the absolute correlation threshold
    should be returned. Self-pairs and duplicates must not appear."""
    from src.influence import _detect_coalitions

    matrix = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.0, 2.0, 3.0, 4.0],   # perfectly correlated with a
        "c": [4.0, 1.0, 3.0, 2.0],   # noisy
    }
    coalitions = _detect_coalitions(["a", "b", "c"], matrix, corr_threshold=0.6)
    pairs = [tuple(c["agents"]) for c in coalitions]
    assert ("a", "b") in pairs
    # No self-pair, no reversed duplicate.
    assert all(p[0] < p[1] for p in pairs)
    assert len(pairs) == len(set(pairs))


def test_build_timeline_preserves_order_and_keys():
    from src.influence import _build_timeline

    results = [
        _FakeResult(votes=[], committed=True, aggregate_weight=0.71, leader_id="agent-2"),
        _FakeResult(votes=[], committed=False, aggregate_weight=0.10, leader_id="agent-1"),
    ]
    timeline = _build_timeline(results)
    assert [t["round"] for t in timeline] == [0, 1]
    assert timeline[0] == {
        "round": 0, "committed": True, "aggregate": 0.71, "leader": "agent-2",
    }
    assert timeline[1]["leader"] == "agent-1"


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

# --------------------------------------------------------------------------
# _recommendations — branch coverage for moderate-gini, coalitions, healthy
# --------------------------------------------------------------------------

def _bland_agent():
    return {
        "swing_power": 0.05,
        "kingmaker_score": 0.0,
        "influence_radius": 0.1,
        "avg_weight": 0.1,
    }


def test_recommendations_moderate_asymmetry_branch():
    """Gini in the (0.3, 0.5] band should produce the *moderate* warning,
    not the high-concentration one."""
    metrics = {
        "agents": {"a": _bland_agent()},
        "gini": 0.4,
        "coalitions": [],
    }
    recs = _recommendations(metrics)
    joined = "\n".join(recs).lower()
    assert "moderate power asymmetry" in joined
    assert "high power asymmetry" not in joined


def test_recommendations_high_coalition_count_flags_collusion():
    """More than 3 coalitions => an explicit collusion-risk warning."""
    metrics = {
        "agents": {"a": _bland_agent()},
        "gini": 0.1,
        "coalitions": [
            {"agents": ["a", "b"], "correlation": 0.9},
            {"agents": ["a", "c"], "correlation": 0.8},
            {"agents": ["b", "c"], "correlation": 0.7},
            {"agents": ["a", "d"], "correlation": 0.65},
        ],
    }
    recs = _recommendations(metrics)
    joined = "\n".join(recs).lower()
    assert "coalitions detected" in joined
    assert "collusion" in joined


def test_recommendations_healthy_distribution_fallback():
    """Low gini, no kingmakers, no contrarians, no coalitions => the
    single 'looks healthy' message is the only output."""
    metrics = {
        "agents": {"a": _bland_agent(), "b": _bland_agent()},
        "gini": 0.05,
        "coalitions": [],
    }
    recs = _recommendations(metrics)
    assert len(recs) == 1
    assert "healthy" in recs[0].lower()


# --------------------------------------------------------------------------
# _generate_html — basic structural assertions (cheap, no browser needed)
# --------------------------------------------------------------------------

def test_generate_html_embeds_metrics_and_escapes_recommendations():
    """The HTML payload must inline the metrics JSON and HTML-escape any
    user-visible recommendation text."""
    from src.influence import _generate_html

    metrics = {
        "agents": {
            "agent-0": {
                "swing_power": 0.25,
                "kingmaker_score": 0.1,
                "influence_radius": 0.0,
                "avg_weight": 0.2,
            },
        },
        "gini": 0.42,
        "coalitions": [],
        "timeline": [
            {"round": 0, "committed": True, "aggregate": 0.7, "leader": "agent-0"},
        ],
    }
    recs = ["<script>alert(1)</script> watch this"]
    cfg = {"agents": 1, "rounds": 1, "byzantine": 0.0, "threshold": 0.6}

    html = _generate_html(metrics, recs, cfg)

    # Structure: must be a full HTML document with the canvases the JS
    # block references — if any are renamed we want CI to scream.
    assert html.startswith("<!DOCTYPE html>")
    for canvas_id in ("giniGauge", "powerChart", "coalitionGraph", "timeline"):
        assert f'id="{canvas_id}"' in html

    # Metrics JSON inlined (no string slicing surprises).
    assert '"swing_power": 0.25' in html
    assert '"kingmaker_score": 0.1' in html

    # Recommendation text must be escaped: raw < > must not leak into the
    # DOM, the escaped form must.
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


# --------------------------------------------------------------------------
# main — CLI smoke (covers argparse + HTML + JSON export paths)
# --------------------------------------------------------------------------

def test_main_writes_html_and_json(tmp_path, capsys, monkeypatch):
    """Run the CLI end-to-end with --json and assert both files land on
    disk with the expected shape."""
    from src import influence

    # Deterministic agent assignment so the smoke test is reproducible.
    import random as _random
    _random.seed(7)

    monkeypatch.chdir(tmp_path)
    html_path = tmp_path / "out.html"
    json_path = tmp_path / "out.json"

    influence.main([
        "--agents", "4",
        "--rounds", "3",
        "--byzantine", "0.25",
        "--threshold", "0.6",
        "--output", str(html_path),
        "--json",
    ])

    captured = capsys.readouterr().out
    assert "Consensus Influence Mapper" in captured
    assert "Gini coefficient" in captured

    assert html_path.exists(), "HTML report must be written"
    html = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html

    assert json_path.exists(), "--json flag must also write the JSON sibling"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"config", "metrics", "recommendations"}
    assert data["config"]["agents"] == 4
    assert data["config"]["rounds"] == 3
    assert isinstance(data["recommendations"], list) and data["recommendations"]

    # Kingmaker/swing invariant must hold in the rendered metrics too.
    for aid, m in data["metrics"]["agents"].items():
        assert m["kingmaker_score"] <= m["swing_power"] + 1e-9, aid


def test_main_html_only_skips_json(tmp_path, monkeypatch):
    """Without --json no sibling JSON file should be written."""
    from src import influence

    import random as _random
    _random.seed(11)

    monkeypatch.chdir(tmp_path)
    html_path = tmp_path / "only.html"

    influence.main([
        "--agents", "3",
        "--rounds", "2",
        "--byzantine", "0.34",
        "--threshold", "0.6",
        "--output", str(html_path),
    ])

    assert html_path.exists()
    # The CLI replaces .html with .json; that sibling must NOT exist.
    assert not (tmp_path / "only.json").exists()


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
