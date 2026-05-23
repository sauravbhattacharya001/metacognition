"""CLI: run the demo swarm and print a counter-proof quality report.

Usage::

    python -m src.network.counter_proof_demo
    python -m src.network.counter_proof_demo --format markdown --risk cautious
"""
from __future__ import annotations

import argparse
import asyncio

from src.core.protocol import MBFTEngine
from src.core.state import RoundResult, Vote
from src.counter_proof_quality_advisor import CounterProofQualityAdvisor
from src.network.simulator import build_demo_swarm


async def _run(threshold: float, slash_factor: float, risk: str, fmt: str) -> str:
    agents = build_demo_swarm()
    engine = MBFTEngine(
        agents=agents,
        threshold=threshold,
        max_rounds=4,
        slash_factor=slash_factor,
    )
    await engine.run("What is the answer to life?")

    # Augment with synthetic counter-proof variety so the demo is illustrative.
    extra = [
        RoundResult(
            round_index=100,
            leader_id="leader-x",
            committed_solution=None,
            aggregate_weight=1.0,
            threshold=threshold,
            votes=[
                Vote(voter_id="lazy-1", target_proposal_id="p100", weight=-0.7),
                Vote(voter_id="lazy-1", target_proposal_id="p100", weight=-0.7),
                Vote(voter_id="vague-1", target_proposal_id="p100", weight=-0.5, counter_proof="no"),
                Vote(
                    voter_id="careful-1",
                    target_proposal_id="p100",
                    weight=-0.8,
                    counter_proof=(
                        "The proof violates the associativity axiom on line 3 because "
                        "(a + b) + c is replaced with a + (b * c)."
                    ),
                ),
            ],
        ),
        RoundResult(
            round_index=101,
            leader_id="leader-x",
            committed_solution=None,
            aggregate_weight=1.0,
            threshold=threshold,
            votes=[
                Vote(voter_id="lazy-1", target_proposal_id="p101", weight=-0.7),
                Vote(voter_id="vague-1", target_proposal_id="p101", weight=-0.5, counter_proof="wrong"),
                Vote(
                    voter_id="template-1",
                    target_proposal_id="p101",
                    weight=-0.4,
                    counter_proof="proof is wrong",
                ),
            ],
        ),
        RoundResult(
            round_index=102,
            leader_id="leader-x",
            committed_solution=None,
            aggregate_weight=1.0,
            threshold=threshold,
            votes=[
                Vote(
                    voter_id="template-1",
                    target_proposal_id="p102",
                    weight=-0.4,
                    counter_proof="proof is wrong",
                ),
            ],
        ),
    ]
    history = list(engine.history) + extra

    report = CounterProofQualityAdvisor().analyze(
        history,
        reputation=engine.reputation,
        risk_appetite=risk,
    )

    if fmt == "json":
        return report.to_json()
    if fmt == "markdown":
        return report.to_markdown()
    return report.to_text()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--slash-factor", type=float, default=0.5)
    parser.add_argument(
        "--risk",
        choices=("cautious", "balanced", "aggressive"),
        default="balanced",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
    )
    args = parser.parse_args()
    print(asyncio.run(_run(args.threshold, args.slash_factor, args.risk, args.format)))


if __name__ == "__main__":
    main()
