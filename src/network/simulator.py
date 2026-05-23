"""Asyncio-based demo simulator.

Runs an mBFT round over a small mock swarm so you can see the protocol in
action without provisioning any LLM credentials::

    python -m src.network.simulator

The output is ASCII-only so that this command (the very first one in
``docs/getting-started.md``) works on a default Windows ``cp1252`` shell
without ``PYTHONIOENCODING=utf-8``. See issue #19 for the regression
this guards against.
"""
from __future__ import annotations

import asyncio
import sys

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine


def _make_stdout_robust() -> None:
    """Best-effort: switch stdout to UTF-8 so future glyphs do not crash.

    This is defensive - the demo itself is already ASCII-safe - but it
    means any downstream code (or REPL exploration) inheriting this
    interpreter's stdout will not blow up on Greek/math characters under
    the default Windows ``cp1252`` console.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is None:  # pragma: no cover - Python < 3.7
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - exotic streams (capture, pipe)
        pass


def build_demo_swarm() -> list[MockAgent]:
    return [
        MockAgent("a1", answer="42", confidence=0.92),
        MockAgent("a2", answer="42", confidence=0.78),
        MockAgent("a3", answer="42", confidence=0.65),
        MockAgent("a4", answer="41", confidence=0.40),
        MockAgent("a5", answer="999", confidence=0.99, byzantine=True),
    ]


async def main() -> None:
    _make_stdout_robust()
    agents = build_demo_swarm()
    engine = MBFTEngine(agents=agents, threshold=1.5, max_rounds=4)
    result = await engine.run("What is the answer to life?")

    print("=" * 60)
    if result and result.committed:
        print(f"COMMITTED: {result.committed_solution!r}")
        print(f"  leader: {result.leader_id}")
        print(
            f"  sum V_i: {result.aggregate_weight:.3f} "
            f">= theta={result.threshold}"
        )
    else:
        print("NO CONSENSUS within max_rounds.")
    print(f"reputation after run: {engine.reputation}")
    print(f"rounds executed: {len(engine.history)}")


if __name__ == "__main__":
    asyncio.run(main())
