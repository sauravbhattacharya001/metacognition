"""Asyncio-based demo simulator.

Runs an mBFT round over a small mock swarm so you can see the protocol in
action without provisioning any LLM credentials::

    python -m src.network.simulator
"""
from __future__ import annotations

import asyncio

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine


def build_demo_swarm() -> list[MockAgent]:
    return [
        MockAgent("a1", answer="42", confidence=0.92),
        MockAgent("a2", answer="42", confidence=0.78),
        MockAgent("a3", answer="42", confidence=0.65),
        MockAgent("a4", answer="41", confidence=0.40),
        MockAgent("a5", answer="999", confidence=0.99, byzantine=True),
    ]


async def main() -> None:
    agents = build_demo_swarm()
    engine = MBFTEngine(agents=agents, threshold=1.5, max_rounds=4)
    result = await engine.run("What is the answer to life?")

    print("=" * 60)
    if result and result.committed:
        print(f"COMMITTED: {result.committed_solution!r}")
        print(f"  leader: {result.leader_id}")
        print(f"  Σ V_i: {result.aggregate_weight:.3f} >= θ={result.threshold}")
    else:
        print("NO CONSENSUS within max_rounds.")
    print(f"reputation after run: {engine.reputation}")
    print(f"rounds executed: {len(engine.history)}")


if __name__ == "__main__":
    asyncio.run(main())
