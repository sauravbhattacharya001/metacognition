"""CLI entry point: run the demo swarm and print a disagreement forensics report.

Usage::

    python -m src.network.forensics_demo
    python -m src.network.forensics_demo --format markdown
    python -m src.network.forensics_demo --threshold 3.0 --format json

The intent: a one-liner that exercises the engine *under conditions where it
struggles* (artificially-high threshold + a Byzantine voter) so operators
can see what :class:`~src.disagreement_forensics.DisagreementForensics`
would surface on a real degraded run.
"""
from __future__ import annotations

import argparse
import asyncio

from src.core.protocol import MBFTEngine
from src.disagreement_forensics import DisagreementForensics
from src.network.simulator import build_demo_swarm


async def _run(threshold: float, slash_factor: float, fmt: str) -> str:
    agents = build_demo_swarm()
    engine = MBFTEngine(
        agents=agents,
        threshold=threshold,
        max_rounds=4,
        slash_factor=slash_factor,
    )
    await engine.run("What is the answer to life?")

    report = DisagreementForensics().analyze(engine.history)

    if fmt == "json":
        return report.to_json()
    if fmt == "markdown":
        return report.to_markdown()
    return report.to_text()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="consensus threshold (default 3.0, intentionally high to "
        "trigger failures for the demo)",
    )
    parser.add_argument("--slash-factor", type=float, default=0.5)
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
    )
    args = parser.parse_args()
    print(asyncio.run(_run(args.threshold, args.slash_factor, args.format)))


if __name__ == "__main__":
    main()
