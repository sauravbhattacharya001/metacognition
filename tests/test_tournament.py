"""Tests for the Consensus Tournament Arena (src.tournament).

Covers:
  * ``_build_agents`` — deterministic seeding, Byzantine ratio, accept_set, diversity.
  * ``_elo_update`` — symmetry, zero-sum delta, draw stability, ordering invariants.
  * ``_run_match`` — winner logic for every branch (commit/no-commit, faster rounds,
    higher aggregate, draw on near-tie).
  * ``run_tournament`` — match counts, aggregate ELO conservation, leaderboard order,
    determinism with the same seed.
  * ``print_report`` — produces output, mentions every team and "DRAW" when applicable.
  * ``export_json`` / ``export_html`` — round-trip on disk, schema, embedded data.
  * ``main`` — CLI smoke test via ``sys.argv`` with JSON export.
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path
from typing import List

import pytest

from src import tournament as tour
from src.tournament import (
    DEFAULT_TEAMS,
    MatchResult,
    TeamConfig,
    TeamStats,
    _build_agents,
    _elo_update,
    _leaderboard,
    _run_match,
    export_html,
    export_json,
    print_report,
    run_tournament,
)


# ---------------------------------------------------------------------------
# _build_agents
# ---------------------------------------------------------------------------


def test_build_agents_count_and_ids() -> None:
    cfg = TeamConfig("Alpha", 6, (0.5, 0.9), 0.0, 1)
    agents = _build_agents(cfg, seed=42)
    assert len(agents) == 6
    assert [a.id for a in agents] == [f"Alpha_{i}" for i in range(6)]


def test_build_agents_deterministic_with_seed() -> None:
    cfg = TeamConfig("Det", 5, (0.4, 0.95), 0.2, 3)
    a = _build_agents(cfg, seed=123)
    b = _build_agents(cfg, seed=123)
    assert [(x.id, x.answer, x.confidence, x.byzantine) for x in a] == \
           [(x.id, x.answer, x.confidence, x.byzantine) for x in b]


def test_build_agents_different_seeds_differ() -> None:
    cfg = TeamConfig("S", 8, (0.3, 0.9), 0.0, 5)
    a = _build_agents(cfg, seed=1)
    b = _build_agents(cfg, seed=999)
    # Extremely unlikely all 8 confidences match across two seeds.
    assert [x.confidence for x in a] != [x.confidence for x in b]


def test_build_agents_byzantine_ratio_floor() -> None:
    # 40% of 5 = 2.0 → exactly 2 Byzantine (floor via int()).
    cfg = TeamConfig("Sab", 5, (0.6, 0.85), 0.4, 1)
    agents = _build_agents(cfg, seed=7)
    byz = [a for a in agents if a.byzantine]
    assert len(byz) == 2
    # Byzantine agents are the first `n_byz` by index (deterministic).
    assert all(int(a.id.split("_")[-1]) < 2 for a in byz)


def test_build_agents_zero_byzantine() -> None:
    cfg = TeamConfig("Clean", 4, (0.7, 0.9), 0.0, 1)
    agents = _build_agents(cfg, seed=5)
    assert all(not a.byzantine for a in agents)


def test_build_agents_accept_set_singleton_when_diversity_one() -> None:
    cfg = TeamConfig("Mono", 4, (0.8, 0.9), 0.0, 1)
    agents = _build_agents(cfg, seed=11)
    # All agents share the same single answer; accept_set is the singleton.
    answers = {a.answer for a in agents}
    assert len(answers) == 1
    for a in agents:
        assert a.accept_set == {a.answer}


def test_build_agents_accept_set_matches_sampled_pool_when_diverse() -> None:
    cfg = TeamConfig("Div", 6, (0.5, 0.9), 0.0, 3)
    agents = _build_agents(cfg, seed=13)
    # Every agent's accept_set is the same shared pool of <=3 answers.
    pools = {frozenset(a.accept_set) for a in agents}
    assert len(pools) == 1
    shared = next(iter(pools))
    assert 1 <= len(shared) <= 3
    # And every chosen answer is from that pool.
    assert all(a.answer in shared for a in agents)


def test_build_agents_diversity_capped_by_pool_size() -> None:
    big = TeamConfig("Big", 3, (0.5, 0.9), 0.0, 9999)
    agents = _build_agents(big, seed=3)
    shared = next(iter({frozenset(a.accept_set) for a in agents}))
    # Capped at the size of POSSIBLE_ANSWERS.
    assert len(shared) == len(tour.POSSIBLE_ANSWERS)


def test_build_agents_confidence_in_range() -> None:
    lo, hi = 0.42, 0.78
    cfg = TeamConfig("R", 20, (lo, hi), 0.0, 1)
    agents = _build_agents(cfg, seed=2)
    for a in agents:
        assert lo - 1e-9 <= a.confidence <= hi + 1e-9


# ---------------------------------------------------------------------------
# _elo_update
# ---------------------------------------------------------------------------


def test_elo_update_zero_sum_delta() -> None:
    ra, rb = 1500.0, 1500.0
    na, nb = _elo_update(ra, rb, 1.0)
    # ELO is zero-sum: change to A == -change to B.
    assert math.isclose((na - ra) + (nb - rb), 0.0, abs_tol=1e-9)


def test_elo_update_draw_between_equals_is_stable() -> None:
    ra, rb = 1500.0, 1500.0
    na, nb = _elo_update(ra, rb, 0.5)
    assert math.isclose(na, 1500.0, abs_tol=1e-9)
    assert math.isclose(nb, 1500.0, abs_tol=1e-9)


def test_elo_update_winner_gains_more_when_underdog() -> None:
    # Lower-rated A beats higher-rated B → A should gain more than the standard 16.
    na, nb = _elo_update(1400.0, 1600.0, 1.0)
    assert na - 1400.0 > 16.0
    assert nb - 1600.0 < -16.0


def test_elo_update_favorite_gains_less_when_winning() -> None:
    na, _ = _elo_update(1700.0, 1300.0, 1.0)
    # Big favorite winning gains far less than k/2.
    assert 0.0 < (na - 1700.0) < 16.0


def test_elo_update_k_factor_scales_delta() -> None:
    a1, _ = _elo_update(1500.0, 1500.0, 1.0, k=32.0)
    a2, _ = _elo_update(1500.0, 1500.0, 1.0, k=16.0)
    # Halving k halves the delta from a balanced match.
    assert math.isclose((a1 - 1500.0) / (a2 - 1500.0), 2.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# _run_match
# ---------------------------------------------------------------------------


def _solo(name: str, conf: float, byz_ratio: float = 0.0) -> TeamConfig:
    """A tiny single-agent team for predictable matches."""
    return TeamConfig(name, 1, (conf, conf), byz_ratio, 1)


def test_run_match_commit_beats_no_commit() -> None:
    # A has confidence 0.9 → aggregate 0.9 >= threshold 0.5 → commits.
    # B has confidence 0.1 → aggregate 0.1 < 0.5 → never commits.
    ta = _solo("A_committer", 0.9)
    tb = _solo("B_quiet", 0.1)
    result = asyncio.run(_run_match(ta, tb, "task", threshold=0.5, seed=1))
    assert result.a_committed
    assert not result.b_committed
    assert result.winner == "A_committer"


def test_run_match_no_commit_either_side_is_draw() -> None:
    ta = _solo("A_low", 0.1)
    tb = _solo("B_low", 0.1)
    result = asyncio.run(_run_match(ta, tb, "task", threshold=5.0, seed=4))
    assert not result.a_committed
    assert not result.b_committed
    assert result.winner is None


def test_run_match_records_round_counts_and_task() -> None:
    ta = _solo("X", 0.9)
    tb = _solo("Y", 0.9)
    r = asyncio.run(_run_match(ta, tb, "the-task", threshold=0.5, seed=2))
    assert r.team_a == "X" and r.team_b == "Y"
    assert r.task == "the-task"
    assert r.a_rounds >= 1 and r.b_rounds >= 1


def test_run_match_higher_aggregate_breaks_round_tie() -> None:
    # Both single-agent teams commit in round 1 (tied rounds);
    # A has strictly higher confidence so higher aggregate → A wins.
    ta = _solo("Strong", 0.95)
    tb = _solo("Weaker", 0.60)
    r = asyncio.run(_run_match(ta, tb, "task", threshold=0.5, seed=10))
    assert r.a_committed and r.b_committed
    assert r.a_rounds == r.b_rounds
    assert r.a_aggregate > r.b_aggregate
    assert r.winner == "Strong"


def test_run_match_near_tied_aggregates_is_draw() -> None:
    # Both commit at the same confidence in the same round → aggregates within 0.01 → DRAW.
    ta = _solo("Twin1", 0.8)
    tb = _solo("Twin2", 0.8)
    r = asyncio.run(_run_match(ta, tb, "task", threshold=0.5, seed=8))
    assert r.a_committed and r.b_committed
    assert r.a_rounds == r.b_rounds
    assert abs(r.a_aggregate - r.b_aggregate) <= 0.01
    assert r.winner is None


# ---------------------------------------------------------------------------
# run_tournament
# ---------------------------------------------------------------------------


@pytest.fixture
def small_teams() -> List[TeamConfig]:
    return [
        _solo("Alpha", 0.9),
        _solo("Bravo", 0.85),
        _solo("Charlie", 0.4),  # rarely commits at threshold 0.7
    ]


def test_run_tournament_match_count(small_teams: List[TeamConfig]) -> None:
    stats, matches = asyncio.run(
        run_tournament(small_teams, n_rounds=2, threshold=0.7, seed=0)
    )
    # 3 teams choose 2 = 3 pairs, * 2 rounds = 6 matches.
    assert len(matches) == 6
    # Every team is tracked, every match recorded both ways.
    for s in stats.values():
        assert s.total_matches == 4  # each team faces 2 others * 2 rounds
        assert len(s.history) == 4


def test_run_tournament_elo_is_zero_sum(small_teams: List[TeamConfig]) -> None:
    stats, _ = asyncio.run(
        run_tournament(small_teams, n_rounds=2, threshold=0.7, seed=0)
    )
    total = sum(s.elo for s in stats.values())
    # 3 teams starting at 1500 → total stays at 4500 because every match is zero-sum.
    assert math.isclose(total, 1500.0 * len(small_teams), abs_tol=1e-6)


def test_run_tournament_deterministic_with_seed(small_teams: List[TeamConfig]) -> None:
    s1, m1 = asyncio.run(run_tournament(small_teams, n_rounds=2, threshold=0.7, seed=99))
    s2, m2 = asyncio.run(run_tournament(small_teams, n_rounds=2, threshold=0.7, seed=99))
    assert [m.winner for m in m1] == [m.winner for m in m2]
    assert [(n, round(stat.elo, 4)) for n, stat in s1.items()] == \
           [(n, round(stat.elo, 4)) for n, stat in s2.items()]


def test_run_tournament_uses_defaults_when_teams_none() -> None:
    stats, matches = asyncio.run(run_tournament(teams=None, n_rounds=1, threshold=2.0, seed=1))
    assert set(stats) == {t.name for t in DEFAULT_TEAMS}
    # 6 default teams * 5 / 2 = 15 pairs * 1 round = 15 matches.
    expected_pairs = len(DEFAULT_TEAMS) * (len(DEFAULT_TEAMS) - 1) // 2
    assert len(matches) == expected_pairs


def test_run_tournament_wins_losses_draws_sum_to_matches(small_teams: List[TeamConfig]) -> None:
    stats, _ = asyncio.run(
        run_tournament(small_teams, n_rounds=3, threshold=0.7, seed=2)
    )
    for s in stats.values():
        assert s.wins + s.losses + s.draws == s.total_matches


def test_leaderboard_sorted_by_elo_descending(small_teams: List[TeamConfig]) -> None:
    stats, _ = asyncio.run(
        run_tournament(small_teams, n_rounds=2, threshold=0.7, seed=0)
    )
    lb = _leaderboard(stats)
    elos = [s.elo for s in lb]
    assert elos == sorted(elos, reverse=True)


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------


def test_print_report_includes_all_teams_and_match_log(
    capsys: pytest.CaptureFixture[str],
    small_teams: List[TeamConfig],
) -> None:
    stats, matches = asyncio.run(
        run_tournament(small_teams, n_rounds=1, threshold=0.7, seed=0)
    )
    print_report(stats, matches)
    out = capsys.readouterr().out
    assert "LEADERBOARD" in out
    assert "MATCH LOG" in out
    for t in small_teams:
        assert t.name in out


def test_print_report_marks_draws(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Two identical low-confidence teams → all matches will fail to commit (DRAW).
    teams = [_solo("T1", 0.1), _solo("T2", 0.1)]
    stats, matches = asyncio.run(
        run_tournament(teams, n_rounds=1, threshold=5.0, seed=0)
    )
    assert all(m.winner is None for m in matches)
    print_report(stats, matches)
    assert "DRAW" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# export_json / export_html
# ---------------------------------------------------------------------------


def test_export_json_roundtrip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    small_teams: List[TeamConfig],
) -> None:
    stats, matches = asyncio.run(
        run_tournament(small_teams, n_rounds=1, threshold=0.7, seed=0)
    )
    out = tmp_path / "report.json"
    export_json(stats, matches, str(out))
    data = json.loads(out.read_text(encoding="utf-8"))

    assert "leaderboard" in data and "matches" in data
    assert len(data["leaderboard"]) == len(small_teams)
    assert len(data["matches"]) == len(matches)

    # Schema checks on first entries.
    first = data["leaderboard"][0]
    for k in ("rank", "team", "elo", "wins", "losses", "draws",
              "commits", "total_matches", "avg_aggregate", "elo_history"):
        assert k in first
    assert first["rank"] == 1

    m0 = data["matches"][0]
    for k in ("team_a", "team_b", "task", "winner",
              "a_committed", "b_committed", "a_rounds", "b_rounds",
              "a_aggregate", "b_aggregate"):
        assert k in m0

    # Confirms the success log line was printed (silences capsys noise too).
    assert "JSON exported" in capsys.readouterr().out


def test_export_html_writes_valid_skeleton(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    small_teams: List[TeamConfig],
) -> None:
    stats, matches = asyncio.run(
        run_tournament(small_teams, n_rounds=1, threshold=0.7, seed=0)
    )
    out = tmp_path / "report.html"
    export_html(stats, matches, str(out))
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    # Leaderboard rows mention every team.
    for t in small_teams:
        assert t.name in html
    # Match rows mention each task at least once.
    assert all(m.task[:40] in html for m in matches)
    # Embedded ELO chart data is valid JSON.
    start = html.index("const data=") + len("const data=")
    end = html.index(";", start)
    chart = json.loads(html[start:end])
    assert set(chart.keys()) == {t.name for t in small_teams}
    assert "HTML exported" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CLI: main()
# ---------------------------------------------------------------------------


def test_main_cli_with_json_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "cli.json"
    argv = [
        "tournament",
        "--teams", "2",
        "--rounds", "1",
        "--threshold", "0.5",
        "--seed", "7",
        "--export", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    tour.main()
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["leaderboard"]) == 2
    # 2 teams * 1 round → exactly 1 match.
    assert len(data["matches"]) == 1
    captured = capsys.readouterr().out
    assert "LEADERBOARD" in captured
    assert "JSON exported" in captured


def test_main_cli_html_export_default_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "cli.html"
    monkeypatch.setattr(sys, "argv", [
        "tournament",
        "--teams", "2",
        "--rounds", "1",
        "--threshold", "0.5",
        "--seed", "1",
        "--export", str(out),
    ])
    tour.main()
    assert out.exists()
    assert "<!DOCTYPE html>" in out.read_text(encoding="utf-8")
    assert "HTML exported" in capsys.readouterr().out


def test_main_cli_no_export_prints_report_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "tournament",
        "--teams", "2",
        "--rounds", "1",
        "--threshold", "0.5",
        "--seed", "0",
    ])
    tour.main()
    captured = capsys.readouterr().out
    assert "LEADERBOARD" in captured
    assert "exported" not in captured


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


def test_team_stats_defaults() -> None:
    s = TeamStats(name="X")
    assert s.elo == 1500.0
    assert s.wins == s.losses == s.draws == s.commits == s.total_matches == 0
    assert s.total_aggregate == 0.0
    assert s.history == []


def test_match_result_fields_preserved() -> None:
    m = MatchResult(
        team_a="A", team_b="B", task="t", winner="A",
        a_committed=True, b_committed=False,
        a_rounds=2, b_rounds=4,
        a_aggregate=1.5, b_aggregate=0.7,
    )
    assert m.team_a == "A" and m.team_b == "B" and m.winner == "A"
    assert m.a_committed and not m.b_committed
    assert (m.a_rounds, m.b_rounds) == (2, 4)
    assert (m.a_aggregate, m.b_aggregate) == (1.5, 0.7)
