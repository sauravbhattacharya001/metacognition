"""Bayesian variant of the mBFT consensus engine (FUTURE WORK).

This module is a research sketch accompanying the paper's "Future Work"
section. Where ``protocol.py`` treats agent confidence as a discrete,
defeasible logical state (``τ_i ∈ [0, 1]`` with hard veto on counter-proof),
``BayesianMBFTEngine`` treats each agent's report as a likelihood signal
over a finite hypothesis space and performs *Bayesian belief updating*
across rounds.

Formal sketch
-------------
Let ``H = {h_1, … h_k}`` be the set of distinct candidate solutions
proposed in a round. We maintain a posterior ``P(h | E_r)`` over ``H``
where ``E_r`` is the evidence accumulated through round ``r``.

For each agent ``a_i`` reporting ``(s_i, τ_i)`` we treat ``τ_i`` as a
calibrated likelihood::

    P(report_i | h) = τ_i           if  s_i == h
                      (1 - τ_i)/(k-1) otherwise

Agent reputation ``ρ_i ∈ (0, 1]`` (slashed across rounds, as in mBFT) acts
as a likelihood-tempering exponent::

    L_i(h) = P(report_i | h) ** ρ_i

The posterior update for round ``r`` is the standard product rule::

    P(h | E_r) ∝ P(h | E_{r-1}) · ∏_i L_i(h)

Commit rule
-----------
Commit ``h*`` iff ``P(h* | E_r) ≥ θ_post`` (a posterior-mass threshold) AND
the Bayes-factor margin over the runner-up exceeds ``log_bf_min``.

Caveats
-------
- Independence assumption between agent reports is strong; a future
  refinement should model peer-influence correlations explicitly.
- Calibration of ``τ_i`` from raw LLM token log-probabilities is an open
  problem; we treat it as a black-box input here.
- This implementation is intentionally minimal — it is a reference,
  not a benchmarked production engine.
"""
from __future__ import annotations

import asyncio
import math
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .state import Proposal

class BayesianRoundResult(BaseModel):
    round_index: int
    posterior: Dict[str, float]
    committed_solution: Optional[str]
    log_bayes_factor: float
    proposals: List[Proposal] = Field(default_factory=list)

    @property
    def committed(self) -> bool:
        return self.committed_solution is not None


class BayesianMBFTEngine:
    """Posterior-driven variant of mBFT.

    Parameters
    ----------
    agents:
        Same agent contract as the logic-based engine.
    posterior_threshold:
        Minimum posterior mass on the MAP hypothesis to commit (``θ_post``).
    log_bf_min:
        Minimum natural-log Bayes factor over the runner-up required to
        commit. Guards against committing when two hypotheses are nearly
        tied even if both exceed ``θ_post`` collectively.
    max_rounds:
        Cap on rounds before giving up.
    slash_factor:
        Exponent reduction applied to an agent's reputation when it
        consistently disagrees with the emerging posterior majority.
    """

    def __init__(
        self,
        agents: List["BaseAgent"],
        posterior_threshold: float = 0.85,
        log_bf_min: float = math.log(10.0),
        max_rounds: int = 5,
        slash_factor: float = 0.7,
    ) -> None:
        if not agents:
            raise ValueError("BayesianMBFTEngine requires at least one agent.")
        if not 0.0 < posterior_threshold < 1.0:
            raise ValueError("posterior_threshold must be in (0, 1).")
        self.agents = agents
        self.posterior_threshold = posterior_threshold
        self.log_bf_min = log_bf_min
        self.max_rounds = max_rounds
        self.slash_factor = slash_factor
        self._reputation: Dict[str, float] = {a.id: 1.0 for a in agents}
        self._posterior: Dict[str, float] = {}
        self.history: List[BayesianRoundResult] = []

    async def run(self, task_prompt: str) -> Optional[BayesianRoundResult]:
        for r in range(self.max_rounds):
            proposals = await self._gather_proposals(task_prompt)
            self._update_posterior(proposals)
            result = self._evaluate_commit(r, proposals)
            self.history.append(result)
            if result.committed:
                return result
            self._slash_dissenters(proposals)
        return self.history[-1] if self.history else None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _gather_proposals(self, task_prompt: str) -> List[Proposal]:
        return list(
            await asyncio.gather(
                *(a.generate_proposal(task_prompt) for a in self.agents)
            )
        )

    def _update_posterior(self, proposals: List[Proposal]) -> None:
        hypotheses = {p.solution for p in proposals} | set(self._posterior)
        k = max(len(hypotheses), 2)

        if not self._posterior:
            prior = 1.0 / len(hypotheses)
            self._posterior = {h: prior for h in hypotheses}
        else:
            for h in hypotheses:
                self._posterior.setdefault(h, 1e-9)

        log_post = {h: math.log(self._posterior[h]) for h in hypotheses}

        for p in proposals:
            tau = min(max(p.confidence, 1e-6), 1.0 - 1e-6)
            rho = self._reputation[p.agent_id]
            for h in hypotheses:
                like = tau if h == p.solution else (1.0 - tau) / (k - 1)
                log_post[h] += rho * math.log(like)

        max_lp = max(log_post.values())
        unnorm = {h: math.exp(lp - max_lp) for h, lp in log_post.items()}
        z = sum(unnorm.values()) or 1.0
        self._posterior = {h: v / z for h, v in unnorm.items()}

    def _evaluate_commit(
        self, round_index: int, proposals: List[Proposal]
    ) -> BayesianRoundResult:
        ranked = sorted(self._posterior.items(), key=lambda kv: -kv[1])
        top, top_p = ranked[0]
        runner_p = ranked[1][1] if len(ranked) > 1 else 1e-12
        log_bf = math.log(max(top_p, 1e-12)) - math.log(max(runner_p, 1e-12))

        commit = (
            top_p >= self.posterior_threshold and log_bf >= self.log_bf_min
        )
        return BayesianRoundResult(
            round_index=round_index,
            posterior=dict(self._posterior),
            committed_solution=top if commit else None,
            log_bayes_factor=log_bf,
            proposals=proposals,
        )

    def _slash_dissenters(self, proposals: List[Proposal]) -> None:
        if not self._posterior:
            return
        map_hypothesis = max(self._posterior.items(), key=lambda kv: kv[1])[0]
        for p in proposals:
            if p.solution != map_hypothesis:
                self._reputation[p.agent_id] *= self.slash_factor

    @property
    def reputation(self) -> Dict[str, float]:
        return dict(self._reputation)

    @property
    def posterior(self) -> Dict[str, float]:
        return dict(self._posterior)
