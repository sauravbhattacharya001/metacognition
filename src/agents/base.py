"""Abstract agent contract used by the mBFT engine."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.state import Proposal, Vote


class BaseAgent(ABC):
    def __init__(self, agent_id: str) -> None:
        self.id = agent_id

    @abstractmethod
    async def generate_proposal(self, task: str) -> Proposal:
        """Produce a (solution, proof, τ_i) tuple for ``task``."""

    @abstractmethod
    async def verify_proposal(self, leader_proposal: Proposal) -> Vote:
        """Inspect ``leader_proposal``; return a signed/weighted vote."""
