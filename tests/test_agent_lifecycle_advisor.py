"""Tests for AgentLifecycleAdvisor."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.agent_lifecycle_advisor import (
    AgentLifecycle,
    AgentLifecycleAdvisor,
    LifecycleAdvisorReport,
    PlaybookItem,
)
from src.core.state import RoundResult, Vote


FIXED_NOW = lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)


def _round(
    idx: int,
    leader: str,
    committed: bool,
    aggregate: float,
    threshold: float,
    votes,
    slashed=None,
) -> RoundResult:
    if slashed is None:
        slashed = [] if committed else [leader]
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution="42" if committed else None,
        aggregate_weight=aggregate,
        threshold=threshold,
        votes=list(votes),
        slashed=list(slashed),
    )


def _vote(voter: str, weight: float, counter=None) -> Vote:
    return Vote(
        voter_id=voter,
        target_proposal_id="p1",
        weight=weight,
        counter_proof=counter,
    )


# ---------------------------------------------------------------------------


def test_empty_history_returns_clean_report():
    rep = AgentLifecycleAdvisor().analyze(
        history=[], reputation={}, now=FIXED_NOW
    )
    assert rep.rounds_analyzed == 0
    assert rep.agents == []
    assert rep.playbook == []  # HEALTHY_FLEET requires rounds>=3
    assert rep.overall_grade == "A"


def test_all_committed_history_produces_keep_and_healthy_fleet():
    # 3 successful rounds, leader rotates, voters agree positively.
    history = [
        _round(0, "alice", True, 2.5, 1.0, [_vote("bob", 0.8), _vote("carol", 0.7)]),
        _round(1, "bob", True, 2.4, 1.0, [_vote("alice", 0.8), _vote("carol", 0.7)]),
        _round(2, "carol", True, 2.6, 1.0, [_vote("alice", 0.9), _vote("bob", 0.8)]),
    ]
    rep = AgentLifecycleAdvisor().analyze(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        now=FIXED_NOW,
    )
    verdicts = {a.agent_id: a.verdict for a in rep.agents}
    assert verdicts == {"alice": "KEEP", "bob": "KEEP", "carol": "KEEP"}
    assert any(item.pattern == "HEALTHY_FLEET" for item in rep.playbook)
    assert rep.overall_grade == "A"


def test_chronic_blocker_detected_and_byzantine_cluster_pattern():
    # 3 rounds, aggregate meets threshold but blockers (bob+eve) veto each.
    # Each blocker vetoes 2 rounds => chronic_blocker_count = 2 => DEMOTE.
    history = [
        _round(0, "alice", False, 1.5, 1.0, [
            _vote("bob", -0.5, "counter"),
            _vote("eve", -0.4, "counter"),
            _vote("carol", 0.6),
        ]),
        _round(1, "carol", False, 1.5, 1.0, [
            _vote("bob", -0.5, "counter"),
            _vote("eve", -0.4, "counter"),
            _vote("alice", 0.6),
        ]),
    ]
    rep = AgentLifecycleAdvisor().analyze(
        history=history,
        reputation={"alice": 0.5, "bob": 1.0, "carol": 1.0, "eve": 1.0},
        now=FIXED_NOW,
    )
    by_id = {a.agent_id: a for a in rep.agents}
    assert by_id["bob"].verdict == "DEMOTE"
    assert "CHRONIC_BLOCKER" in by_id["bob"].reasons
    assert by_id["eve"].verdict == "DEMOTE"
    patterns = [item.pattern for item in rep.playbook]
    assert "BYZANTINE_CLUSTER" in patterns


def test_bad_leader_demoted():
    # alice leads 3 rounds, none commit. bob/carol vote positively each
    # round (no rejections) so alice is the leader-fail blocker.
    history = [
        _round(0, "alice", False, 0.4, 1.0, [_vote("bob", 0.2), _vote("carol", 0.1)]),
        _round(1, "alice", False, 0.4, 1.0, [_vote("bob", 0.2), _vote("carol", 0.1)]),
        _round(2, "alice", False, 0.4, 1.0, [_vote("bob", 0.2), _vote("carol", 0.1)]),
    ]
    rep = AgentLifecycleAdvisor().analyze(
        history=history,
        reputation={"alice": 0.5, "bob": 1.0, "carol": 1.0},
        now=FIXED_NOW,
    )
    alice = next(a for a in rep.agents if a.agent_id == "alice")
    assert alice.verdict == "DEMOTE"
    assert "BAD_LEADER" in alice.reasons


def test_low_reputation_and_chronic_blocker_evict_grade_f():
    history = [
        _round(0, "alice", False, 1.2, 1.0, [
            _vote("eve", -0.6, "counter"),
            _vote("bob", 0.5),
        ]),
        _round(1, "alice", False, 1.2, 1.0, [
            _vote("eve", -0.6, "counter"),
            _vote("bob", 0.5),
        ]),
    ]
    rep = AgentLifecycleAdvisor().analyze(
        history=history,
        reputation={"alice": 0.25, "bob": 1.0, "eve": 0.1},
        now=FIXED_NOW,
    )
    eve = next(a for a in rep.agents if a.agent_id == "eve")
    assert eve.verdict == "EVICT"
    assert rep.overall_grade == "F"


def test_star_performer_detected():
    # alice leads 2 rounds, both commit; calibration high (agrees on bob's commit).
    history = [
        _round(0, "alice", True, 2.4, 1.0, [_vote("bob", 0.8), _vote("carol", 0.7)]),
        _round(1, "alice", True, 2.5, 1.0, [_vote("bob", 0.8), _vote("carol", 0.7)]),
        _round(2, "bob", True, 2.4, 1.0, [_vote("alice", 0.9), _vote("carol", 0.7)]),
    ]
    rep = AgentLifecycleAdvisor().analyze(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        now=FIXED_NOW,
    )
    alice = next(a for a in rep.agents if a.agent_id == "alice")
    assert alice.verdict == "KEEP"
    assert "STAR_PERFORMER" in alice.reasons
    patterns = [item.pattern for item in rep.playbook]
    assert "STAR_PROMOTION" in patterns


def test_probe_for_underobserved_agent():
    # only 2 rounds; carol only votes once -> observations=1 < min=3 -> PROBE
    history = [
        _round(0, "alice", True, 2.0, 1.0, [_vote("bob", 0.9), _vote("carol", 0.5)]),
        _round(1, "bob", True, 2.0, 1.0, [_vote("alice", 0.9)]),
    ]
    rep = AgentLifecycleAdvisor().analyze(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0, "dave": 1.0},
        now=FIXED_NOW,
    )
    by_id = {a.agent_id: a for a in rep.agents}
    assert by_id["carol"].verdict == "PROBE"
    assert by_id["dave"].verdict == "PROBE"
    assert "INSUFFICIENT_DATA" in by_id["dave"].reasons


def test_reinstate_for_recovering_agent():
    # bob was slashed in round 0 but agrees positively on the committed
    # rounds 1 and 2 and has recovered to rep 0.6.
    history = [
        _round(0, "bob", False, 0.4, 1.0, [_vote("alice", 0.2), _vote("carol", 0.1)]),
        _round(1, "alice", True, 2.4, 1.0, [_vote("bob", 0.8), _vote("carol", 0.7)]),
        _round(2, "carol", True, 2.5, 1.0, [_vote("alice", 0.9), _vote("bob", 0.7)]),
    ]
    rep = AgentLifecycleAdvisor().analyze(
        history=history,
        reputation={"alice": 1.0, "bob": 0.6, "carol": 1.0},
        now=FIXED_NOW,
    )
    bob = next(a for a in rep.agents if a.agent_id == "bob")
    assert bob.verdict == "REINSTATE"
    assert "REPUTATION_RECOVERING" in bob.reasons


def test_echo_chamber_pattern():
    # 5 committed rounds with zero rejections from non-slashed voters.
    history = []
    leaders = ["alice", "bob", "carol", "alice", "bob"]
    for i, ld in enumerate(leaders):
        others = [a for a in ("alice", "bob", "carol") if a != ld]
        history.append(_round(
            i, ld, True, 2.4, 1.0,
            [_vote(o, 0.7) for o in others],
        ))
    rep = AgentLifecycleAdvisor().analyze(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        now=FIXED_NOW,
    )
    patterns = [item.pattern for item in rep.playbook]
    assert "ECHO_CHAMBER" in patterns


def test_capacity_loss_risk_when_many_evict():
    # 5-agent swarm, 2 of them evictable => 40% >= 30%
    history = [
        _round(0, "alice", False, 1.2, 1.0, [
            _vote("eve1", -0.5, "c"), _vote("eve2", -0.5, "c"),
            _vote("bob", 0.5), _vote("carol", 0.4),
        ]),
        _round(1, "alice", False, 1.2, 1.0, [
            _vote("eve1", -0.5, "c"), _vote("eve2", -0.5, "c"),
            _vote("bob", 0.5), _vote("carol", 0.4),
        ]),
    ]
    rep = AgentLifecycleAdvisor().analyze(
        history=history,
        reputation={
            "alice": 1.0, "bob": 1.0, "carol": 1.0,
            "eve1": 0.1, "eve2": 0.1,
        },
        now=FIXED_NOW,
    )
    patterns = [item.pattern for item in rep.playbook]
    assert "CAPACITY_LOSS_RISK" in patterns
    assert "BYZANTINE_CLUSTER" in patterns


def test_risk_appetite_orders_scores_monotonically():
    # Build a history with a mildly suspect voter (1 chronic block).
    history = [
        _round(0, "alice", False, 1.2, 1.0, [
            _vote("eve", -0.5, "c"),
            _vote("bob", 0.6),
        ]),
        _round(1, "alice", True, 2.4, 1.0, [
            _vote("eve", 0.6),
            _vote("bob", 0.8),
        ]),
        _round(2, "bob", True, 2.4, 1.0, [
            _vote("eve", 0.6),
            _vote("alice", 0.8),
        ]),
    ]
    rep_args = dict(
        history=history,
        reputation={"alice": 0.7, "bob": 1.0, "eve": 1.0},
        now=FIXED_NOW,
    )
    cautious = AgentLifecycleAdvisor().analyze(risk_appetite="cautious", **rep_args)
    balanced = AgentLifecycleAdvisor().analyze(risk_appetite="balanced", **rep_args)
    aggressive = AgentLifecycleAdvisor().analyze(risk_appetite="aggressive", **rep_args)

    def _eve_risk(rep):
        return next(a.lifecycle_risk for a in rep.agents if a.agent_id == "eve")

    assert _eve_risk(aggressive) <= _eve_risk(balanced) <= _eve_risk(cautious)


def test_renderers_produce_nonempty_output_and_valid_json():
    history = [
        _round(0, "alice", True, 2.4, 1.0, [_vote("bob", 0.8), _vote("carol", 0.7)]),
        _round(1, "bob", True, 2.5, 1.0, [_vote("alice", 0.9), _vote("carol", 0.7)]),
        _round(2, "carol", True, 2.4, 1.0, [_vote("alice", 0.8), _vote("bob", 0.8)]),
    ]
    rep = AgentLifecycleAdvisor().analyze(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        now=FIXED_NOW,
    )
    text = rep.to_text()
    md = rep.to_markdown()
    js = rep.to_json()
    assert "Agent Lifecycle Advisor" in text
    assert "# Agent Lifecycle Advisor" in md
    parsed = json.loads(js)
    assert parsed["overall_grade"] == "A"
    assert isinstance(parsed["agents"], list)


def test_to_json_byte_stable_with_fixed_clock():
    history = [
        _round(0, "alice", True, 2.4, 1.0, [_vote("bob", 0.8), _vote("carol", 0.7)]),
        _round(1, "bob", True, 2.5, 1.0, [_vote("alice", 0.9), _vote("carol", 0.7)]),
        _round(2, "carol", True, 2.4, 1.0, [_vote("alice", 0.8), _vote("bob", 0.8)]),
    ]
    args = dict(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0},
        now=FIXED_NOW,
    )
    a = AgentLifecycleAdvisor().analyze(**args).to_json()
    b = AgentLifecycleAdvisor().analyze(**args).to_json()
    assert a == b


def test_inactive_agent_sweep_pattern():
    # 6 rounds, only alice + bob active; carol/dave inactive throughout.
    history = []
    for i in range(6):
        ld = "alice" if i % 2 == 0 else "bob"
        other = "bob" if ld == "alice" else "alice"
        history.append(_round(
            i, ld, True, 2.4, 1.0, [_vote(other, 0.8)]
        ))
    rep = AgentLifecycleAdvisor().analyze(
        history=history,
        reputation={"alice": 1.0, "bob": 1.0, "carol": 1.0, "dave": 1.0},
        now=FIXED_NOW,
    )
    patterns = [item.pattern for item in rep.playbook]
    assert "INACTIVE_AGENT_SWEEP" in patterns
    by_id = {a.agent_id: a for a in rep.agents}
    assert "INACTIVE" in by_id["carol"].reasons
    assert "INACTIVE" in by_id["dave"].reasons
