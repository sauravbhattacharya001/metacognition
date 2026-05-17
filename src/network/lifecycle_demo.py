"""CLI entry point: lifecycle advisor over a demo swarm.

Usage::

    python -m src.network.lifecycle_demo
    python -m src.network.lifecycle_demo --format markdown
    python -m src.network.lifecycle_demo --threshold 3.0 --risk cautious --format json

The demo intentionally runs with a high consensus threshold so the engine
struggles, surfacing the kinds of agent-lifecycle decisions
:class:`~src.agent_lifecycle_advisor.AgentLifecycleAdvisor` exists to
support.
"""
from __future__ import annotations

import argparse
import asyncio

from src.agent_lifecycle_advisor import AgentLifecycleAdvisor
from src.core.protocol import MBFTEngine
from src.network.simulator import build_demo_swarm


async def _run(
    threshold: float,
    slash_factor: float,
    fmt: str,
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

    report = AgentLifecycleAdvisor().analyze(
        history=engine.history,
        reputation=engine.reputation,
        slash_factor=slash_factor,
        risk_appetite=risk,
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
    print(asyncio.run(_run(args.threshold, args.slash_factor, args.format, args.risk)))


if __name__ == "__main__":
    main()
