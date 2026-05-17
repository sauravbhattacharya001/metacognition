"""CLI entry point: run the demo swarm and print a forecast for round N+1.

Usage::

    python -m src.network.forecast_demo
    python -m src.network.forecast_demo --format markdown
    python -m src.network.forecast_demo --threshold 1.5 --format json

Pairs with :mod:`src.network.health_demo` — health is "what happened",
forecast is "what's about to happen and what should I do about it".
"""
from __future__ import annotations

import argparse
import asyncio

from src.consensus_forecaster import ConsensusForecaster
from src.core.protocol import MBFTEngine
from src.network.simulator import build_demo_swarm


async def _run(threshold: float, fmt: str, slash_factor: float) -> str:
    agents = build_demo_swarm()
    engine = MBFTEngine(
        agents=agents,
        threshold=threshold,
        max_rounds=4,
        slash_factor=slash_factor,
    )
    await engine.run("What is the answer to life?")

    forecast = ConsensusForecaster().forecast(
        history=engine.history,
        reputation=engine.reputation,
        threshold=engine.threshold,
        slash_factor=engine.slash_factor,
        agent_ids=[a.id for a in engine.agents],
    )

    if fmt == "json":
        return forecast.to_json()
    if fmt == "markdown":
        return forecast.to_markdown()
    return forecast.to_text()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--slash-factor", type=float, default=0.5)
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
    )
    args = parser.parse_args()
    print(asyncio.run(_run(args.threshold, args.format, args.slash_factor)))


if __name__ == "__main__":
    main()
