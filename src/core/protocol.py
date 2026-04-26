"""mBFT consensus engine.

Executable specification for the three state transitions described in the
paper: epistemic leader election, semantic verification, and
confidence-weighted finality.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Dict, List, Optional

from .state import Proposal, RoundResult, Vote

if TYPE_CHECKING:
    from src.agents.base import BaseAgent


class MBFTEngine:
    """Run mBFT rounds over a fixed set of agents."""

    def __init__(
        self,
        agents: List["BaseAgent"],
        threshold: float,
        max_rounds: int = 4,
        slash_factor: float = 0.5,
    ) -> None:
        if not agents:
            raise ValueError("MBFTEngine requires at least one agent.")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        self.agents = agents
        self.threshold = threshold
        self.max_rounds = max_rounds
        self.slash_factor = slash_factor
        self._reputation: Dict[str, float] = {a.id: 1.0 for a in agents}
        self.history: List[RoundResult] = []

    async def run(self, task_prompt: str) -> Optional[RoundResult]:
        """Drive rounds until commit or ``max_rounds`` is reached."""
        forced_leader: Optional[str] = None
        for r in range(self.max_rounds):
            result = await self._run_round(task_prompt, r, forced_leader)
            self.history.append(result)
            if result.committed:
                return result
            forced_leader = self._pick_counter_leader(result)
        return self.history[-1] if self.history else None

    async def _run_round(
        self,
        task_prompt: str,
        round_index: int,
        forced_leader: Optional[str],
    ) -> RoundResult:
        proposals = await self._gather_proposals(task_prompt)
        leader = self._elect_leader(proposals, forced_leader)
        votes = await self._collect_votes(leader)

        aggregate = leader.confidence * self._reputation[leader.agent_id]
        weighted_votes: List[tuple[Vote, float]] = []
        for v in votes:
            effective = v.weight * self._reputation[v.voter_id]
            aggregate += effective
            weighted_votes.append((v, effective))

        # A rejection only vetoes if the voter still carries full reputation.
        # Slashed (previously-faulty) agents lose veto power but keep voice.
        has_unrefuted_rejection = any(
            v.is_rejection and self._reputation[v.voter_id] >= 1.0
            for v, _ in weighted_votes
        )
        committed = aggregate >= self.threshold and not has_unrefuted_rejection

        slashed: List[str] = []
        if not committed:
            self._reputation[leader.agent_id] *= self.slash_factor
            slashed.append(leader.agent_id)

        return RoundResult(
            round_index=round_index,
            leader_id=leader.agent_id,
            committed_solution=leader.solution if committed else None,
            aggregate_weight=aggregate,
            threshold=self.threshold,
            votes=votes,
            slashed=slashed,
        )

    async def _gather_proposals(self, task_prompt: str) -> List[Proposal]:
        return list(
            await asyncio.gather(
                *(a.generate_proposal(task_prompt) for a in self.agents)
            )
        )

    def _elect_leader(
        self,
        proposals: List[Proposal],
        forced_leader: Optional[str],
    ) -> Proposal:
        """``L_r = argmax_i τ_i(S_i) * reputation_i``.

        A forced leader (set after a successful counter-proof) overrides the
        argmax and implements the view-change rule.
        """
        if forced_leader is not None:
            for p in proposals:
                if p.agent_id == forced_leader:
                    return p
        return max(
            proposals,
            key=lambda p: p.confidence * self._reputation[p.agent_id],
        )

    async def _collect_votes(self, leader: Proposal) -> List[Vote]:
        followers = [a for a in self.agents if a.id != leader.agent_id]
        return list(
            await asyncio.gather(
                *(a.verify_proposal(leader) for a in followers)
            )
        )

    def _pick_counter_leader(self, result: RoundResult) -> Optional[str]:
        rejections = [v for v in result.votes if v.is_rejection]
        if not rejections:
            return None
        # Pick the rejector with the strongest reputation-weighted rebuttal.
        strongest = min(
            rejections,
            key=lambda v: v.weight * self._reputation[v.voter_id],
        )
        return strongest.voter_id

    @property
    def reputation(self) -> Dict[str, float]:
        return dict(self._reputation)
