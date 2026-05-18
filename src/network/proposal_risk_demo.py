"""CLI entry point: pre-submission proposal risk report.

Usage::

    python -m src.network.proposal_risk_demo
    python -m src.network.proposal_risk_demo --format markdown --risk cautious
    python -m src.network.proposal_risk_demo --threshold 3.0 --format json

Mirrors :mod:`src.network.replay_demo` / :mod:`src.network.coalition_demo`:
runs the demo swarm to populate an engine ``history`` and ``reputation``
map, then synthesizes the *next* leader's hypothetical proposal and asks
:class:`~src.proposal_risk_scorer.ProposalRiskScorer` to score it.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import sys
from typing import List

from src.core.protocol import MBFTEngine
from src.core.state import Proposal, RoundResult
from src.network.simulator import build_demo_swarm
from src.proposal_risk_scorer import ProposalRiskScorer


def _ensure_utf8_stdout() -> None:
    # On Windows the default stdout encoding can be cp1252; sibling demos
    # use this same shim so markdown tables with unicode survive piping.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8"
            )  # type: ignore[assignment]
        except Exception:
            pass


def _synthesize_next_proposal(history: List[RoundResult]) -> Proposal:
    """Take the most recent leader and synthesize a 'next-round' proposal."""
    if not history:
        return Proposal(
            agent_id="a1",
            solution="42",
            proof="Because 42 is the canonical answer per prior literature.",
            confidence=0.7,
        )
    last = history[-1]
    # if last failed, the leader will retry with similar content -- this is
    # exactly the kind of risky proposal the scorer should flag.
    solution = last.committed_solution or "42"
    confidence = 0.85
    if last.committed:
        proof = (
            "Therefore the prior committed solution still holds for this round "
            "because no contradictory evidence has emerged. See [1]."
        )
    else:
        proof = "ok"  # intentionally weak to make the demo interesting
        confidence = 0.95
    return Proposal(
        agent_id=last.leader_id,
        solution=solution,
        proof=proof,
        confidence=confidence,
    )


async def _run(threshold: float, slash_factor: float, fmt: str, risk: str) -> str:
    agents = build_demo_swarm()
    engine = MBFTEngine(
        agents=agents,
        threshold=threshold,
        max_rounds=4,
        slash_factor=slash_factor,
    )
    await engine.run("What is the answer to life?")

    next_proposal = _synthesize_next_proposal(list(engine.history))
    leader_id = next_proposal.agent_id
    leader_rep = float(engine.reputation.get(leader_id, 0.5))
    roster = {
        aid: float(rep)
        for aid, rep in engine.reputation.items()
        if aid != leader_id
    }

    scorer = ProposalRiskScorer(threshold=threshold, risk_appetite=risk)
    report = scorer.score(
        next_proposal,
        leader_reputation=leader_rep,
        history=engine.history,
        roster=roster,
    )

    if fmt == "json":
        return report.to_json()
    if fmt == "markdown":
        return report.to_markdown()
    return report.to_text()


def main() -> None:
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="consensus threshold (default 3.0)",
    )
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
