"""CLI entry point: detect voting coalitions in a swarm history.

Usage::

    python -m src.network.coalition_demo
    python -m src.network.coalition_demo --format markdown
    python -m src.network.coalition_demo --threshold 1.5 --risk cautious
"""
from __future__ import annotations

import argparse
import asyncio

from src.core.protocol import MBFTEngine
from src.network.simulator import build_demo_swarm
from src.voting_coalition_detector import VotingCoalitionDetector


async def _run(threshold: float, fmt: str, risk: str) -> str:
    agents = build_demo_swarm()
    engine = MBFTEngine(
        agents=agents,
        threshold=threshold,
        max_rounds=4,
    )
    await engine.run("What is the answer to life?")
    detector = VotingCoalitionDetector(cohesion_threshold=0.6, min_co_votes=1)
    report = detector.analyze(
        history=engine.history,
        reputation=engine.reputation,
        threshold=engine.threshold,
        agent_ids=[a.id for a in engine.agents],
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
    args = parser.parse_args()
    print(asyncio.run(_run(args.threshold, args.format, args.risk)))


if __name__ == "__main__":
    main()
