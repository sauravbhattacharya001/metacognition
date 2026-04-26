"""Concrete agents.

``MockAgent`` lets the simulator and tests run with no external dependencies
and provides deterministic Byzantine-injection knobs. ``MetacognitiveAgent``
shows how to wire a real LLM client in: it expects an object exposing
``async complete(prompt: str) -> str`` that returns JSON-shaped output.
"""
from __future__ import annotations

import json
from typing import Optional, Protocol

from src.agents.base import BaseAgent
from src.core.state import Proposal, Vote


class MockAgent(BaseAgent):
    """Deterministic agent for tests and local demos."""

    def __init__(
        self,
        agent_id: str,
        answer: str,
        confidence: float,
        byzantine: bool = False,
        accept_set: Optional[set[str]] = None,
    ) -> None:
        super().__init__(agent_id)
        self.answer = answer
        self.confidence = confidence
        self.byzantine = byzantine
        self.accept_set = accept_set if accept_set is not None else {answer}

    async def generate_proposal(self, task: str) -> Proposal:
        return Proposal(
            agent_id=self.id,
            solution=self.answer,
            proof=f"[{self.id}] proof-trace for task={task!r} -> {self.answer}",
            confidence=self.confidence,
        )

    async def verify_proposal(self, leader_proposal: Proposal) -> Vote:
        if self.byzantine:
            return Vote(
                voter_id=self.id,
                target_proposal_id=leader_proposal.proposal_id,
                weight=self.confidence,
            )

        if leader_proposal.solution in self.accept_set:
            return Vote(
                voter_id=self.id,
                target_proposal_id=leader_proposal.proposal_id,
                weight=self.confidence,
            )

        return Vote(
            voter_id=self.id,
            target_proposal_id=leader_proposal.proposal_id,
            weight=-self.confidence,
            counter_proof=(
                f"[{self.id}] derives ¬({leader_proposal.solution}); "
                f"axiom set yields {self.answer}."
            ),
        )


class LLMClient(Protocol):
    async def complete(self, prompt: str) -> str: ...  # pragma: no cover


_PROPOSE_PROMPT = """You are agent {agent_id} in an mBFT consensus swarm.
Task:
{task}

Respond with strict JSON: {{"solution": str, "proof": str, "confidence": float}}
where confidence ∈ [0, 1] reflects your calibrated certainty.
"""

_VERIFY_PROMPT = """You are agent {agent_id} verifying another agent's proposal.

Leader solution: {solution}
Leader proof:
{proof}

If the proof is sound, accept. If you can produce a logical counter-proof,
reject. Respond with strict JSON:
{{"accept": bool, "confidence": float, "counter_proof": str | null}}
"""


class MetacognitiveAgent(BaseAgent):
    """Real-LLM agent. Expects an ``LLMClient`` that returns JSON."""

    def __init__(self, agent_id: str, llm: LLMClient) -> None:
        super().__init__(agent_id)
        self.llm = llm

    async def generate_proposal(self, task: str) -> Proposal:
        raw = await self.llm.complete(
            _PROPOSE_PROMPT.format(agent_id=self.id, task=task)
        )
        data = json.loads(raw)
        return Proposal(
            agent_id=self.id,
            solution=str(data["solution"]),
            proof=str(data["proof"]),
            confidence=float(data["confidence"]),
        )

    async def verify_proposal(self, leader_proposal: Proposal) -> Vote:
        raw = await self.llm.complete(
            _VERIFY_PROMPT.format(
                agent_id=self.id,
                solution=leader_proposal.solution,
                proof=leader_proposal.proof,
            )
        )
        data = json.loads(raw)
        confidence = float(data["confidence"])
        accept = bool(data["accept"])
        counter = data.get("counter_proof")
        return Vote(
            voter_id=self.id,
            target_proposal_id=leader_proposal.proposal_id,
            weight=confidence if accept else -confidence,
            counter_proof=None if accept else (str(counter) if counter else None),
        )
