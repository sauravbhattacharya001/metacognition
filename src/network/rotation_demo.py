"""CLI entry point: leader rotation advisor over a demo swarm.

Usage::

    python -m src.network.rotation_demo
    python -m src.network.rotation_demo --format markdown
    python -m src.network.rotation_demo --horizon 5 --risk aggressive --format json
"""
from __future__ import annotations

import argparse
import asyncio

from src.core.protocol import MBFTEngine
from src.leader_rotation_advisor import LeaderRotationAdvisor
from src.network.simulator import build_demo_swarm


async def _run(
    threshold: float,
    slash_factor: float,
    fmt: str,
    horizon: int,
    risk: str,
) -> str:
    agents = build_demo_swarm()
    engine = MBFTEngine(
        agents=agents,
        threshold=threshold,
        max_rounds=4,
        slash_factor=slash_factor,
    )
    await engine.run("What is the answer to life?")

    advisor = LeaderRotationAdvisor(horizon=horizon, risk_appetite=risk)
    report = advisor.recommend(
        history=engine.history,
        reputation=engine.reputation,
        agents=[a.id for a in agents],
    )

    if fmt == "json":
        return report.to_json()
    if fmt == "markdown":
        return report.to_markdown()
    return report.to_text()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--slash-factor", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=5)
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
    print(
        asyncio.run(
            _run(
                args.threshold,
                args.slash_factor,
                args.format,
                args.horizon,
                args.risk,
            )
        )
    )


if __name__ == "__main__":
    main()
