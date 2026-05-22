"""Tests for src.replay - Consensus Replay Animator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src import replay as replay_mod
from src.replay import (
    AgentSnapshot,
    ReplayData,
    RoundEvent,
    VoteEvent,
    _build_swarm,
    _serialize,
    record_replay,
    render_html,
)


# --------------------------------------------------------------------------- #
# Dataclass smoke tests                                                       #
# --------------------------------------------------------------------------- #


def test_agent_snapshot_fields():
    snap = AgentSnapshot(
        agent_id="a1",
        reputation=1.0,
        is_byzantine=False,
        confidence=0.8,
        answer="correct",
    )
    assert snap.agent_id == "a1"
    assert snap.reputation == 1.0
    assert snap.is_byzantine is False


def test_vote_event_optional_counter_proof():
    ev = VoteEvent(
        voter_id="a2",
        weight=0.7,
        effective_weight=0.7,
        is_rejection=False,
        counter_proof=None,
    )
    assert ev.counter_proof is None
    ev2 = VoteEvent("a3", -0.5, -0.5, True, "axiom mismatch")
    assert ev2.is_rejection is True
    assert ev2.counter_proof == "axiom mismatch"


def test_round_event_required_fields():
    r = RoundEvent(
        round_index=0,
        leader_id="a1",
        leader_solution="correct",
        leader_confidence=0.8,
        votes=[],
        aggregate_weight=2.4,
        threshold=1.5,
        committed=True,
        slashed=[],
        reputations_after={"a1": 1.0},
    )
    assert r.committed is True
    assert r.reputations_after["a1"] == 1.0


def test_replay_data_defaults():
    rd = ReplayData(
        swarm_size=5,
        threshold=1.5,
        byzantine_count=1,
        agents=[],
    )
    assert rd.rounds == []
    assert rd.final_committed is False
    assert rd.final_solution is None


# --------------------------------------------------------------------------- #
# _build_swarm                                                                #
# --------------------------------------------------------------------------- #


def test_build_swarm_sizes_and_byzantine_placement():
    agents = _build_swarm(n=5, byzantine_count=2)
    assert len(agents) == 5
    # Honest agents come first, byzantine last
    assert all(not a.byzantine for a in agents[:3])
    assert all(a.byzantine for a in agents[3:])
    # IDs are a1..a5
    assert [a.id for a in agents] == [f"a{i+1}" for i in range(5)]
    # Confidence matches defaults
    assert agents[0].confidence == pytest.approx(0.80)
    assert agents[-1].confidence == pytest.approx(0.95)
    # Honest agents answer "correct"; byzantine answer "byz-i"
    assert agents[0].answer == "correct"
    assert agents[-1].answer.startswith("byz-")


def test_build_swarm_custom_confidences():
    agents = _build_swarm(n=3, byzantine_count=0, honest_conf=0.5, byz_conf=0.99)
    assert len(agents) == 3
    assert all(a.confidence == pytest.approx(0.5) for a in agents)
    assert all(not a.byzantine for a in agents)


def test_build_swarm_all_byzantine():
    agents = _build_swarm(n=2, byzantine_count=2)
    assert all(a.byzantine for a in agents)


# --------------------------------------------------------------------------- #
# record_replay                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_record_replay_default_commits():
    rd = await record_replay(swarm_size=5, byzantine_count=1, threshold=1.5)
    assert rd.swarm_size == 5
    assert rd.byzantine_count == 1
    assert rd.threshold == 1.5
    assert len(rd.agents) == 5
    # With 4 honest @0.8 and 1 byzantine @0.95, we expect to commit
    assert rd.final_committed is True
    assert rd.final_solution == "correct"
    assert len(rd.rounds) >= 1
    # First round should have leader and votes recorded
    r0 = rd.rounds[0]
    assert r0.leader_id.startswith("a")
    assert isinstance(r0.votes, list)
    assert r0.threshold == pytest.approx(1.5)
    # Reputations tracked for every agent
    assert set(r0.reputations_after.keys()) == {a.agent_id for a in rd.agents}


@pytest.mark.asyncio
async def test_record_replay_high_threshold_may_fail():
    """A threshold above max possible aggregate weight should not commit."""
    rd = await record_replay(swarm_size=3, byzantine_count=0, threshold=100.0)
    # No round should hit threshold 100
    assert rd.final_committed is False
    assert rd.final_solution is None
    for r in rd.rounds:
        assert r.committed is False


# --------------------------------------------------------------------------- #
# _serialize                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_serialize_round_trips_to_json():
    rd = await record_replay(swarm_size=4, byzantine_count=1, threshold=1.2)
    s = _serialize(rd)
    parsed = json.loads(s)
    assert parsed["swarm_size"] == 4
    assert parsed["byzantine_count"] == 1
    assert "rounds" in parsed
    assert "agents" in parsed
    assert len(parsed["agents"]) == 4


def test_serialize_handles_plain_dict():
    rd = ReplayData(
        swarm_size=1,
        threshold=0.5,
        byzantine_count=0,
        agents=[AgentSnapshot("a1", 1.0, False, 0.9, "ok")],
    )
    parsed = json.loads(_serialize(rd))
    assert parsed["agents"][0]["agent_id"] == "a1"
    assert parsed["rounds"] == []


# --------------------------------------------------------------------------- #
# render_html                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_render_html_contains_data_and_speed():
    rd = await record_replay(swarm_size=4, byzantine_count=1, threshold=1.2)
    html = render_html(rd, speed="fast")
    assert "<!DOCTYPE html>" in html
    assert "mBFT Consensus Replay" in html
    assert '"swarm_size": 4' in html
    # Fast speed maps to 300ms
    assert "const SPEED = 300;" in html


@pytest.mark.asyncio
async def test_render_html_unknown_speed_falls_back_to_normal():
    rd = await record_replay(swarm_size=3, byzantine_count=0, threshold=0.5)
    html = render_html(rd, speed="bogus")
    assert "const SPEED = 700;" in html


@pytest.mark.asyncio
async def test_render_html_all_speeds_supported():
    rd = await record_replay(swarm_size=3, byzantine_count=0, threshold=0.5)
    for name, ms in [("slow", 1200), ("normal", 700), ("fast", 300)]:
        html = render_html(rd, speed=name)
        assert f"const SPEED = {ms};" in html


# --------------------------------------------------------------------------- #
# CLI main()                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_main_writes_html_file(tmp_path, monkeypatch, capsys):
    out = tmp_path / "replay.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay",
            "--agents", "4",
            "--byzantine", "1",
            "--threshold", "1.2",
            "--speed", "fast",
            "--output", str(out),
        ],
    )
    await replay_mod.main()
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "mBFT Consensus Replay" in content
    captured = capsys.readouterr()
    assert f"Replay written to {out}" in captured.out
    assert "Rounds:" in captured.out


@pytest.mark.asyncio
async def test_main_export_json(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay", "--agents", "3", "--byzantine", "0",
         "--threshold", "0.5", "--export", "json"],
    )
    await replay_mod.main()
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["swarm_size"] == 3
    assert parsed["byzantine_count"] == 0


@pytest.mark.asyncio
async def test_main_rejects_byzantine_ge_agents(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay", "--agents", "3", "--byzantine", "3"],
    )
    await replay_mod.main()
    captured = capsys.readouterr()
    assert "Error" in captured.out
    assert "Byzantine" in captured.out


@pytest.mark.asyncio
async def test_main_defaults(tmp_path, monkeypatch, capsys):
    """Run main() with only --output to exercise default arg values."""
    out = tmp_path / "default.html"
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay", "--output", str(out)],
    )
    await replay_mod.main()
    assert out.exists()
    captured = capsys.readouterr()
    assert "Replay written to" in captured.out
