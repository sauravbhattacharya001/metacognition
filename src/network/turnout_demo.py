"""CLI entry point: audit voter turnout / absenteeism / phantom commits.

Usage::

    python -m src.network.turnout_demo
    python -m src.network.turnout_demo --format markdown
    python -m src.network.turnout_demo --risk cautious
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from src.core.protocol import MBFTEngine
from src.core.state import RoundResult, Vote
from src.network.simulator import build_demo_swarm
from src.voter_turnout_advisor import (
    VoterTurnoutAdvisor,
    to_json,
    to_markdown,
    to_text,
)


def _augment_history(history):
    """Append rounds that exercise low-turnout + chronic absentee +
    phantom-dissent commit + a decayed voter."""
    base_idx = (history[-1].round_index + 1) if history else 0
    low_turnout = RoundResult(
        round_index=base_idx,
        leader_id="a1",
        committed_solution="42",
        aggregate_weight=1.8,
        threshold=1.5,
        votes=[
            Vote(voter_id="a1", target_proposal_id="p", weight=0.9),
            Vote(voter_id="a2", target_proposal_id="p", weight=0.9),
        ],
    )
    # phantom: a4 historically rejects, but is absent here, so commit clears
    phantom_commit = RoundResult(
        round_index=base_idx + 1,
        leader_id="a2",
        committed_solution="42",
        aggregate_weight=1.6,
        threshold=1.5,
        votes=[
            Vote(voter_id="a1", target_proposal_id="q", weight=0.8),
            Vote(voter_id="a2", target_proposal_id="q", weight=0.8),
            Vote(voter_id="a3", target_proposal_id="q", weight=0.0),
        ],
    )
    # a4 votes early so it has a history of rejection
    seed_a4 = RoundResult(
        round_index=base_idx + 2,
        leader_id="a3",
        committed_solution=None,
        aggregate_weight=0.4,
        threshold=1.5,
        votes=[
            Vote(voter_id="a1", target_proposal_id="r", weight=0.3),
            Vote(voter_id="a2", target_proposal_id="r", weight=0.3),
            Vote(voter_id="a4", target_proposal_id="r", weight=-0.8, counter_proof="nope"),
        ],
    )
    healthy = RoundResult(
        round_index=base_idx + 3,
        leader_id="a1",
        committed_solution="42",
        aggregate_weight=3.4,
        threshold=1.5,
        votes=[
            Vote(voter_id="a1", target_proposal_id="s", weight=0.85),
            Vote(voter_id="a2", target_proposal_id="s", weight=0.9),
            Vote(voter_id="a3", target_proposal_id="s", weight=0.75),
            Vote(voter_id="a5", target_proposal_id="s", weight=0.9),
        ],
    )
    # reorder so a4 is seeded before the phantom round
    return list(history) + [seed_a4, low_turnout, phantom_commit, healthy]


async def _run(threshold: float, fmt: str, risk: str, quorum: float) -> str:
    agents = build_demo_swarm()
    engine = MBFTEngine(agents=agents, threshold=threshold, max_rounds=4)
    await engine.run("What is the answer to life?")
    history = _augment_history(engine.history)
    advisor = VoterTurnoutAdvisor(
        min_acceptable_turnout=quorum, risk_appetite=risk
    )
    report = advisor.analyze(history, reputation=engine.reputation)
    if fmt == "json":
        return to_json(report)
    if fmt == "markdown":
        return to_markdown(report)
    return to_text(report)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--quorum", type=float, default=0.6)
    parser.add_argument(
        "--format", choices=("text", "markdown", "json"), default="text"
    )
    parser.add_argument(
        "--risk",
        choices=("cautious", "balanced", "aggressive"),
        default="balanced",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    out = asyncio.run(_run(args.threshold, args.format, args.risk, args.quorum))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
