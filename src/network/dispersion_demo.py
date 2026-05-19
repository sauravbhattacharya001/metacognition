"""CLI entry point: detect vote dispersion / groupthink patterns.

Usage::

    python -m src.network.dispersion_demo
    python -m src.network.dispersion_demo --format markdown
    python -m src.network.dispersion_demo --threshold 1.5 --risk cautious
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from src.core.protocol import MBFTEngine
from src.core.state import RoundResult, Vote
from src.network.simulator import build_demo_swarm
from src.vote_dispersion_advisor import VoteDispersionAdvisor, to_json, to_markdown, to_text


def _augment_history(history):
    """Append a hand-crafted groupthink + polarized round so the demo
    surfaces a mix of verdicts even on a tiny live engine run."""
    base_idx = (history[-1].round_index + 1) if history else 0
    groupthink = RoundResult(
        round_index=base_idx,
        leader_id="a1",
        committed_solution="42",
        aggregate_weight=4.0,
        threshold=1.5,
        votes=[
            Vote(voter_id="a1", target_proposal_id="p", weight=0.8),
            Vote(voter_id="a2", target_proposal_id="p", weight=0.82),
            Vote(voter_id="a3", target_proposal_id="p", weight=0.79),
            Vote(voter_id="a4", target_proposal_id="p", weight=0.81),
            Vote(voter_id="a5", target_proposal_id="p", weight=0.80),
        ],
    )
    polarized = RoundResult(
        round_index=base_idx + 1,
        leader_id="a2",
        committed_solution=None,
        aggregate_weight=0.6,
        threshold=1.5,
        votes=[
            Vote(voter_id="a1", target_proposal_id="q", weight=0.85),
            Vote(voter_id="a2", target_proposal_id="q", weight=0.80),
            Vote(voter_id="a3", target_proposal_id="q", weight=-0.6),
            Vote(voter_id="a4", target_proposal_id="q", weight=-0.5),
        ],
    )
    hedged = RoundResult(
        round_index=base_idx + 2,
        leader_id="a3",
        committed_solution=None,
        aggregate_weight=1.3,
        threshold=1.5,
        votes=[
            Vote(voter_id="a1", target_proposal_id="r", weight=0.1),
            Vote(voter_id="a2", target_proposal_id="r", weight=0.15),
            Vote(voter_id="a3", target_proposal_id="r", weight=-0.05),
            Vote(voter_id="a4", target_proposal_id="r", weight=0.18),
            Vote(voter_id="a5", target_proposal_id="r", weight=0.12),
        ],
    )
    return list(history) + [groupthink, polarized, hedged]


async def _run(threshold: float, fmt: str, risk: str) -> str:
    agents = build_demo_swarm()
    engine = MBFTEngine(agents=agents, threshold=threshold, max_rounds=4)
    await engine.run("What is the answer to life?")
    history = _augment_history(engine.history)
    advisor = VoteDispersionAdvisor(
        engine_threshold=threshold,
        risk_appetite=risk,
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
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
    )
    parser.add_argument(
        "--risk",
        choices=("cautious", "balanced", "aggressive"),
        default="balanced",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    out = asyncio.run(_run(args.threshold, args.format, args.risk))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
