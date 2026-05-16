"""CLI entry point: run the demo swarm and print a health report.

Usage::

    python -m src.network.health_demo
    python -m src.network.health_demo --format json
    python -m src.network.health_demo --threshold 1.5 --format markdown

The intent is to give operators a one-liner that exercises the engine and
shows what the :class:`~src.swarm_health.SwarmHealthMonitor` would surface
on a real run — a quick "is my swarm OK?" smoke test.
"""
from __future__ import annotations

import argparse
import asyncio

from src.core.protocol import MBFTEngine
from src.network.simulator import build_demo_swarm
from src.swarm_health import SwarmHealthMonitor


async def _run(threshold: float, fmt: str, slash_factor: float) -> str:
    agents = build_demo_swarm()
    engine = MBFTEngine(
        agents=agents,
        threshold=threshold,
        max_rounds=4,
        slash_factor=slash_factor,
    )
    await engine.run("What is the answer to life?")

    report = SwarmHealthMonitor().analyze(
        history=engine.history,
        reputation=engine.reputation,
        threshold=engine.threshold,
        agent_ids=[a.id for a in engine.agents],
        slash_factor=slash_factor,
    )

    if fmt == "json":
        return report.to_json()
    if fmt == "markdown":
        return report.to_markdown()
    if fmt == "csv":
        return report.to_csv()
    return report.to_text()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--slash-factor", type=float, default=0.5)
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json", "csv"),
        default="text",
    )
    args = parser.parse_args()
    print(asyncio.run(_run(args.threshold, args.format, args.slash_factor)))


if __name__ == "__main__":
    main()
