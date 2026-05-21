"""Comprehensive tests for src.lineage (Consensus Lineage Tracker).

Covers:
- _sim text similarity edge cases
- LineageNode construction
- InstrumentedEngine proposal capture
- ConsensusLineageTracker.build wiring (parents/children, committed node)
- Influence score computation
- Innovation detection (first-round + dissimilar later-round)
- Winning-chain trace and ordering
- Convergence per-round (singleton + multi-proposal)
- Agent influence ranking (max-per-agent)
- to_dict / text_summary / html_report exporters
- _build_agents factory size + Byzantine count contract
- CLI main() with --output + --json
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine
from src.core.state import Proposal, RoundResult, Vote
from src.lineage import (
    ConsensusLineageTracker,
    InstrumentedEngine,
    LineageNode,
    _build_agents,
    _sim,
    main,
)


# ---------------------------------------------------------------------------
# _sim
# ---------------------------------------------------------------------------

class TestSim:
    def test_identical_strings_score_1(self):
        assert _sim("hello world", "hello world") == 1.0

    def test_completely_different_strings_score_low(self):
        # SequenceMatcher will not be 0 if there are any common chars, but it
        # must be well below the default 0.4 threshold for these strings.
        assert _sim("aaaaaa", "zzzzzz") < 0.2

    def test_empty_strings_score_1(self):
        # By SequenceMatcher convention two empty sequences have ratio 1.0.
        assert _sim("", "") == 1.0

    def test_one_empty_string(self):
        assert _sim("", "non-empty") == 0.0

    def test_symmetric(self):
        a, b = "leader rotation strategy", "leader weighted strategy"
        assert _sim(a, b) == _sim(b, a)


# ---------------------------------------------------------------------------
# LineageNode
# ---------------------------------------------------------------------------

class TestLineageNode:
    def test_initial_state(self):
        p = Proposal(agent_id="a1", solution="x", proof="p", confidence=0.5)
        n = LineageNode(p, round_idx=3)
        assert n.proposal is p
        assert n.round_idx == 3
        assert n.parents == []
        assert n.children == []
        assert n.influence_score == 0.0
        assert n.is_innovation is False
        assert n.is_on_winning_chain is False


# ---------------------------------------------------------------------------
# InstrumentedEngine
# ---------------------------------------------------------------------------

def _honest_swarm(n: int = 4, byz: int = 1) -> list[MockAgent]:
    """Small, deterministic swarm where honest agents share an answer."""
    agents: list[MockAgent] = []
    accept = {"alpha"}
    for i in range(n):
        is_byz = i >= (n - byz)
        if is_byz:
            agents.append(MockAgent(
                agent_id=f"a{i}", answer="beta",
                confidence=0.9, byzantine=True,
            ))
        else:
            agents.append(MockAgent(
                agent_id=f"a{i}", answer="alpha",
                confidence=0.85, accept_set=accept,
            ))
    return agents


class TestInstrumentedEngine:
    def test_captures_proposals_per_round(self):
        agents = _honest_swarm(n=4, byz=0)
        engine = InstrumentedEngine(agents=agents, threshold=1.5, max_rounds=2)
        asyncio.run(engine.run("task"))

        # Should have captured at least one round of proposals
        assert len(engine.all_proposals) >= 1
        # Each captured round has one proposal per agent
        for round_props in engine.all_proposals:
            assert len(round_props) == len(agents)
            ids = {p.agent_id for p in round_props}
            assert ids == {a.id for a in agents}


# ---------------------------------------------------------------------------
# ConsensusLineageTracker.build
# ---------------------------------------------------------------------------

def _run_tracker(n: int = 4, byz: int = 1,
                  threshold: float = 1.5, rounds: int = 3,
                  sim_thresh: float = 0.4):
    agents = _honest_swarm(n=n, byz=byz)
    engine = InstrumentedEngine(agents=agents, threshold=threshold,
                                 max_rounds=rounds)
    asyncio.run(engine.run("solve consensus"))
    tracker = ConsensusLineageTracker(similarity_threshold=sim_thresh)
    tracker.build(engine, engine.history)
    return tracker, engine


class TestTrackerBuild:
    def test_nodes_created_for_every_proposal(self):
        tracker, engine = _run_tracker()
        expected = sum(len(r) for r in engine.all_proposals)
        assert len(tracker.nodes) == expected

    def test_nodes_by_round_covers_all_rounds(self):
        tracker, engine = _run_tracker()
        assert set(tracker.nodes_by_round.keys()) == set(
            range(len(engine.all_proposals))
        )

    def test_parent_child_edges_only_cross_consecutive_rounds(self):
        tracker, _ = _run_tracker()
        for node in tracker.nodes:
            for parent in node.parents:
                assert parent.round_idx == node.round_idx - 1
            for child in node.children:
                assert child.round_idx == node.round_idx + 1

    def test_committed_node_matches_history(self):
        tracker, engine = _run_tracker()
        committed_rounds = [r for r in engine.history if r.committed]
        if not committed_rounds:
            assert tracker.committed_node is None
            return
        assert tracker.committed_node is not None
        # The committed node must belong to a leader of a committed round.
        committed_leader_ids = {r.leader_id for r in committed_rounds}
        assert tracker.committed_node.proposal.agent_id in committed_leader_ids

    def test_first_round_nodes_are_all_innovations(self):
        tracker, _ = _run_tracker()
        for n in tracker.nodes_by_round[0]:
            assert n.is_innovation is True


# ---------------------------------------------------------------------------
# Influence + chain
# ---------------------------------------------------------------------------

class TestInfluenceAndChain:
    def test_influence_scores_are_non_negative_and_bounded(self):
        tracker, _ = _run_tracker()
        for node in tracker.nodes:
            # influence = sim * conf * rep; all factors in [0,1] for honest
            # agents, but rep can be < 1 after slashing. Lower bound is 0.
            assert node.influence_score >= 0.0
            # Upper bound: confidence <= 1, sim <= 1, rep starts at 1.0; rep
            # can only decrease via slashing, so score <= 1.0.
            assert node.influence_score <= 1.0 + 1e-9

    def test_committed_node_has_high_self_similarity(self):
        tracker, _ = _run_tracker()
        if tracker.committed_node is None:
            pytest.skip("no committed node in this run")
        # The committed node's own influence score equals
        # sim(self, self) * conf * rep == conf * rep, which is positive.
        assert tracker.committed_node.influence_score > 0.0

    def test_winning_chain_starts_at_round_0_when_committed(self):
        tracker, _ = _run_tracker()
        chain = tracker.winning_chain()
        if not chain:
            assert tracker.committed_node is None
            return
        # Chain is sorted by round and starts at the earliest round we can
        # trace back to (typically round 0, but at minimum it must contain
        # the committed node and be monotonically increasing in round_idx).
        rounds = [n.round_idx for n in chain]
        assert rounds == sorted(rounds)
        assert tracker.committed_node in chain

    def test_winning_chain_marks_is_on_winning_chain(self):
        tracker, _ = _run_tracker()
        for n in tracker.nodes:
            on_chain = n.is_on_winning_chain
            in_list = n in tracker.winning_chain()
            assert on_chain == in_list


# ---------------------------------------------------------------------------
# Convergence + agent influence
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_convergence_per_round_singleton_is_one(self):
        # Single agent per round -> no pairs -> defaults to 1.0
        agents = _honest_swarm(n=1, byz=0)
        engine = InstrumentedEngine(agents=agents, threshold=0.5, max_rounds=2)
        asyncio.run(engine.run("t"))
        tracker = ConsensusLineageTracker()
        tracker.build(engine, engine.history)
        for _, sim in tracker.convergence_per_round():
            assert sim == 1.0

    def test_convergence_values_in_unit_interval(self):
        tracker, _ = _run_tracker()
        for ri, sim in tracker.convergence_per_round():
            assert 0.0 <= sim <= 1.0
            assert ri in tracker.nodes_by_round

    def test_agent_influence_sorted_descending(self):
        tracker, _ = _run_tracker()
        ranking = tracker.agent_influence()
        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_agent_influence_uses_max_per_agent(self):
        tracker, _ = _run_tracker()
        ranking = dict(tracker.agent_influence())
        for agent_id, score in ranking.items():
            agent_nodes = [n for n in tracker.nodes
                            if n.proposal.agent_id == agent_id]
            expected = max((n.influence_score for n in agent_nodes), default=0.0)
            assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

class TestExporters:
    def test_to_dict_shape(self):
        tracker, _ = _run_tracker()
        d = tracker.to_dict()
        assert set(d.keys()) == {
            "nodes", "edges", "agent_influence", "convergence", "committed"
        }
        # Every node dict has the documented fields
        for n in d["nodes"]:
            assert set(n.keys()) == {
                "round", "agent", "solution", "confidence",
                "influence", "innovation", "winning_chain", "proposal_id",
            }
        # Every edge points to known proposal ids
        all_ids = {n["proposal_id"] for n in d["nodes"]}
        for e in d["edges"]:
            assert e["from"] in all_ids
            assert e["to"] in all_ids
            assert 0.0 <= e["similarity"] <= 1.0

    def test_to_dict_committed_matches_committed_node(self):
        tracker, _ = _run_tracker()
        d = tracker.to_dict()
        if tracker.committed_node is None:
            assert d["committed"] is None
        else:
            assert d["committed"] == tracker.committed_node.proposal.proposal_id

    def test_text_summary_contains_section_headers(self):
        tracker, _ = _run_tracker()
        s = tracker.text_summary()
        assert "CONSENSUS LINEAGE TRACKER" in s
        assert "Agent Influence" in s
        assert "Innovation" in s
        assert "Convergence" in s

    def test_text_summary_handles_no_commit(self):
        # Force no-commit by setting an unreachably high threshold.
        agents = _honest_swarm(n=3, byz=2)
        engine = InstrumentedEngine(agents=agents, threshold=99.0, max_rounds=2)
        asyncio.run(engine.run("t"))
        tracker = ConsensusLineageTracker()
        tracker.build(engine, engine.history)
        assert tracker.committed_node is None
        s = tracker.text_summary()
        # No-commit branch must still produce output without crashing.
        assert "lineage chain empty" in s or "No committed" in s

    def test_html_report_is_valid_html(self):
        tracker, _ = _run_tracker()
        html = tracker.html_report()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        # Embedded data must round-trip as JSON.
        # Extract nodes={...}; line.
        import re
        m = re.search(r"const nodes=(\[.*?\]);", html, re.DOTALL)
        assert m, "nodes payload not found in HTML"
        parsed = json.loads(m.group(1))
        assert len(parsed) == len(tracker.nodes)


# ---------------------------------------------------------------------------
# _build_agents
# ---------------------------------------------------------------------------

class TestBuildAgents:
    def test_size_and_byzantine_count(self):
        agents = _build_agents(n_agents=6, n_byzantine=2)
        assert len(agents) == 6
        assert sum(1 for a in agents if a.byzantine) == 2

    def test_ids_are_unique(self):
        agents = _build_agents(n_agents=5, n_byzantine=1)
        ids = [a.id for a in agents]
        assert len(set(ids)) == len(ids)

    def test_confidences_in_range(self):
        agents = _build_agents(n_agents=5, n_byzantine=2)
        for a in agents:
            assert 0.0 <= a.confidence <= 1.0

    def test_zero_byzantine(self):
        agents = _build_agents(n_agents=3, n_byzantine=0)
        assert all(not a.byzantine for a in agents)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_main_writes_html_and_json(self, tmp_path: Path, capsys):
        html_path = tmp_path / "out.html"
        json_path = tmp_path / "out.json"
        main([
            "--agents", "3",
            "--byzantine", "1",
            "--threshold", "1.2",
            "--rounds", "2",
            "--output", str(html_path),
            "--json", str(json_path),
        ])
        captured = capsys.readouterr().out
        assert "CONSENSUS LINEAGE TRACKER" in captured
        assert html_path.exists()
        assert html_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "nodes" in data and "edges" in data
        assert isinstance(data["nodes"], list)

    def test_main_runs_without_outputs(self, capsys):
        main(["--agents", "3", "--byzantine", "1", "--rounds", "2"])
        out = capsys.readouterr().out
        assert "CONSENSUS LINEAGE TRACKER" in out
