"""Pydantic state models corresponding to the mBFT formalism.

Mapping to the paper:
- ``Proposal.confidence`` is the metacognitive weight ``τ_i(S_i) ∈ [0, 1]``.
- ``Vote.weight`` is ``V_i(S_L) ∈ [-1, 1]`` — negative encodes a defeasible
  rejection backed by a counter-proof.
"""
from __future__ import annotations

from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Proposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: uuid4().hex)
    agent_id: str
    solution: str
    proof: str
    confidence: float = Field(ge=0.0, le=1.0)


class Vote(BaseModel):
    voter_id: str
    target_proposal_id: str
    weight: float = Field(ge=-1.0, le=1.0)
    counter_proof: Optional[str] = None

    @property
    def is_rejection(self) -> bool:
        return self.weight < 0.0


class RoundResult(BaseModel):
    round_index: int
    leader_id: str
    committed_solution: Optional[str]
    aggregate_weight: float
    threshold: float
    votes: List[Vote] = Field(default_factory=list)
    slashed: List[str] = Field(default_factory=list)

    @property
    def committed(self) -> bool:
        return self.committed_solution is not None
