"""CLI demo for ConsensusLatencyProfiler.

Simulates a network of agents with various latency pathologies and
demonstrates profiler diagnostics.

Usage:
    python -m src.network.latency_demo
    python -m src.network.latency_demo --scenario stall
    python -m src.network.latency_demo --scenario healthy
    python -m src.network.latency_demo --scenario cascade
    python -m src.network.latency_demo --format json
"""
from __future__ import annotations

import argparse
import sys

from src.core.state import RoundResult, Vote
from src.consensus_latency_profiler import ConsensusLatencyProfiler


def _vote(voter: str, weight: float) -> Vote:
    return Vote(voter_id=voter, target_proposal_id="p1", weight=weight)


def _round(idx, leader, committed, agg, threshold=1.5, votes=None, slashed=None):
    return RoundResult(
        round_index=idx,
        leader_id=leader,
        committed_solution="X" if committed else None,
        aggregate_weight=agg,
        threshold=threshold,
        votes=votes or [],
        slashed=slashed or [],
    )


def scenario_healthy():
    """4 agents, fast commits."""
    return [
        _round(0, "alpha", True, 2.5),
        _round(0, "beta", True, 2.3),
        _round(0, "gamma", True, 2.1),
        _round(0, "alpha", True, 2.4),
        _round(0, "delta", True, 1.9),
    ]


def scenario_stall():
    """Chronic veto bottleneck + near-miss stalls."""
    return [
        _round(0, "alpha", False, 1.44, votes=[_vote("veto_agent", -0.9)]),
        _round(1, "beta", False, 1.43, votes=[_vote("veto_agent", -0.8)]),
        _round(2, "gamma", False, 1.42, votes=[_vote("veto_agent", -0.7)]),
        _round(3, "delta", True, 2.0),
        _round(0, "alpha", False, 1.44, votes=[_vote("veto_agent", -0.9)]),
        _round(1, "beta", False, 1.45, votes=[_vote("veto_agent", -0.85)]),
        _round(2, "gamma", True, 1.8),
        _round(0, "alpha", False, 0.8, votes=[_vote("veto_agent", -0.9)]),
        _round(1, "beta", False, 0.9, votes=[_vote("veto_agent", -0.8)]),
        _round(2, "gamma", False, 1.0),
    ]


def scenario_cascade():
    """Serial slash cascade + revolving door."""
    return [
        _round(0, "alpha", False, 0.5, slashed=["alpha"]),
        _round(1, "alpha", False, 0.4, slashed=["alpha"]),
        _round(2, "beta", False, 0.6, slashed=["beta"]),
        _round(3, "gamma", False, 0.5),
        _round(0, "delta", False, 0.5, slashed=["delta"]),
        _round(1, "epsilon", False, 0.4, slashed=["epsilon", "delta"]),
        _round(2, "zeta", False, 0.5, slashed=["zeta"]),
        _round(0, "alpha", True, 2.0),
    ]


def scenario_degrading():
    """Commit rate declines over time."""
    history = []
    # First 4 commit fast
    for i, leader in enumerate(["a1", "a2", "a3", "a4"]):
        history.append(_round(0, leader, True, 2.0))
    # Last 4 stall
    for i, leader in enumerate(["b1", "b2", "b3", "b4"]):
        history.append(_round(0, leader, False, 0.5, slashed=[leader]))
    return history


SCENARIOS = {
    "healthy": scenario_healthy,
    "stall": scenario_stall,
    "cascade": scenario_cascade,
    "degrading": scenario_degrading,
}


def main():
    parser = argparse.ArgumentParser(description="Consensus Latency Profiler Demo")
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()), default="stall",
        help="Demo scenario (default: stall)",
    )
    parser.add_argument(
        "--format", choices=["text", "markdown", "json"], default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--risk", choices=["cautious", "balanced", "aggressive"], default="balanced",
        help="Risk appetite (default: balanced)",
    )
    args = parser.parse_args()

    history = SCENARIOS[args.scenario]()
    profiler = ConsensusLatencyProfiler(risk_appetite=args.risk)
    report = profiler.analyze(history, max_rounds=4)

    print(f"\n{'='*60}")
    print(f"  CONSENSUS LATENCY PROFILER DEMO")
    print(f"  Scenario: {args.scenario} | Risk: {args.risk} | Format: {args.format}")
    print(f"{'='*60}\n")

    if args.format == "text":
        print(profiler.to_text(report))
    elif args.format == "markdown":
        print(profiler.to_markdown(report))
    else:
        print(profiler.to_json(report))


if __name__ == "__main__":
    main()
