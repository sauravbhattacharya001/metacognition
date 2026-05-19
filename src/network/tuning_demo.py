"""CLI: run the demo swarm and print a threshold tuning report.

Usage::

    python -m src.network.tuning_demo
    python -m src.network.tuning_demo --format markdown --risk cautious
"""
from __future__ import annotations

import argparse
import asyncio

from src.core.protocol import MBFTEngine
from src.network.simulator import build_demo_swarm
from src.threshold_tuning_advisor import ThresholdTuningAdvisor


async def _run(threshold: float, slash_factor: float, risk: str, fmt: str) -> str:
    agents = build_demo_swarm()
    engine = MBFTEngine(
        agents=agents,
        threshold=threshold,
        max_rounds=4,
        slash_factor=slash_factor,
    )
    await engine.run("What is the answer to life?")

    report = ThresholdTuningAdvisor().analyze(
        engine.history,
        threshold=engine.threshold,
        slash_factor=slash_factor,
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
