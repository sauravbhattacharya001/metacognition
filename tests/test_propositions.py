"""Proposition tests.

These are not benchmark experiments — they are executable encodings of
properties the paper claims about mBFT. Each test asserts a *theorem*
(safety, liveness, calibration-monotonicity, etc.) over a parameterised
family of swarms. Failing any of them means the paper's claim does not
hold for the reference implementation.

Conventions
-----------
- "Honest" agents propose the ground-truth answer with calibrated τ.
- "Byzantine" agents (``byzantine=True``) propose a wrong answer with
  high τ and rubber-stamp the leader regardless.
- Honest dissent is modelled as an honest agent with a *different*
  answer — they will issue a counter-proof rather than rubber-stamp.
"""
from __future__ import annotations

from typing import List

import pytest

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine
from src.core.protocol_bayesian import BayesianMBFTEngine


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_swarm(
    n_honest: int,
    n_byzantine: int,
    truth: str = "TRUE",
    lie: str = "FALSE",
    honest_tau: float = 0.8,
    byz_tau: float = 0.99,
) -> List[MockAgent]:
    swarm: List[MockAgent] = []
    for i in range(n_honest):
        swarm.append(MockAgent(f"h{i}", truth, honest_tau))
    for j in range(n_byzantine):
        swarm.append(MockAgent(f"b{j}", lie, byz_tau, byzantine=True))
    return swarm


# --------------------------------------------------------------------------- #
# Proposition 1 — Safety under f < n/3 Byzantine
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_honest, n_byz", [(3, 1), (5, 1), (7, 2), (10, 3)])
@pytest.mark.asyncio
async def test_safety_below_byzantine_third(n_honest: int, n_byz: int) -> None:
    """Proposition: with f < n/3 Byzantine nodes and θ tuned to honest mass,
    mBFT never commits the Byzantine answer."""
    n = n_honest + n_byz
    assert n_byz < n / 3, "test setup violates f < n/3"

    agents = make_swarm(n_honest, n_byz)
    # Threshold = honest_mass * 0.6 — comfortably reachable by the honest
    # majority but unreachable by Byzantine reputation alone after slashing.
    threshold = n_honest * 0.8 * 0.6
    engine = MBFTEngine(agents, threshold=threshold, max_rounds=6, slash_factor=0.05)
    result = await engine.run("task")

    assert result is not None
    if result.committed:
        assert result.committed_solution == "TRUE", (
            "SAFETY VIOLATION: committed Byzantine answer"
        )


# --------------------------------------------------------------------------- #
# Proposition 2 — Liveness for honest unanimity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 3, 5, 10, 25])
@pytest.mark.asyncio
async def test_liveness_honest_unanimity(n: int) -> None:
    """Proposition: a fully honest, unanimous swarm commits in round 0."""
    agents = [MockAgent(f"a{i}", "X", 0.8) for i in range(n)]
    threshold = n * 0.8 * 0.5
    engine = MBFTEngine(agents, threshold=threshold)
    result = await engine.run("task")

    assert result is not None and result.committed
    assert result.committed_solution == "X"
    assert result.round_index == 0


# --------------------------------------------------------------------------- #
# Proposition 3 — Slashing monotonicity
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_slashing_is_monotone() -> None:
    """Proposition: an agent's reputation is non-increasing across rounds."""
    agents = [
        MockAgent("a1", "WRONG", 0.99, byzantine=True),
        MockAgent("a2", "RIGHT", 0.80),
        MockAgent("a3", "RIGHT", 0.75),
        MockAgent("a4", "RIGHT", 0.70),
    ]
    engine = MBFTEngine(agents, threshold=1.5, max_rounds=4, slash_factor=0.5)
    rep_before = engine.reputation.copy()
    await engine.run("task")
    rep_after = engine.reputation

    for aid in rep_before:
        assert rep_after[aid] <= rep_before[aid] + 1e-12, (
            f"reputation of {aid} increased: {rep_before[aid]} -> {rep_after[aid]}"
        )


# --------------------------------------------------------------------------- #
# Proposition 4 — No commit below threshold
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0, 3.0, 5.0])
@pytest.mark.asyncio
async def test_no_commit_below_threshold(threshold: float) -> None:
    """Proposition: every committed round satisfies Σ V_i ≥ θ."""
    agents = [MockAgent(f"a{i}", "X", 0.6) for i in range(4)]
    engine = MBFTEngine(agents, threshold=threshold, max_rounds=2)
    result = await engine.run("task")

    if result and result.committed:
        assert result.aggregate_weight >= threshold


# --------------------------------------------------------------------------- #
# Proposition 5 — Wisdom-of-the-swarm: more honest agents ⇒ at least
# as confident an aggregate at commit
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_swarm_wisdom_monotonicity() -> None:
    """Proposition: holding τ fixed, increasing the number of honest agents
    does not decrease the committed aggregate weight."""
    aggregates: List[float] = []
    for n in (3, 5, 7, 9):
        agents = [MockAgent(f"a{i}", "X", 0.7) for i in range(n)]
        engine = MBFTEngine(agents, threshold=n * 0.7 * 0.5)
        result = await engine.run("task")
        assert result is not None and result.committed
        aggregates.append(result.aggregate_weight)

    for prev, nxt in zip(aggregates, aggregates[1:]):
        assert nxt >= prev - 1e-9


# --------------------------------------------------------------------------- #
# Proposition 6 — Bayesian variant: posterior concentration with calibrated
# honest majority
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_bayesian_posterior_concentrates_on_truth() -> None:
    """Proposition: with a calibrated honest majority, the Bayesian engine's
    posterior on the truth strictly exceeds that of any other hypothesis."""
    agents = [
        MockAgent("h1", "T", 0.85),
        MockAgent("h2", "T", 0.80),
        MockAgent("h3", "T", 0.78),
        MockAgent("h4", "T", 0.75),
        MockAgent("b1", "F", 0.90, byzantine=True),
    ]
    engine = BayesianMBFTEngine(
        agents,
        posterior_threshold=0.85,
        max_rounds=3,
    )
    result = await engine.run("task")

    assert result is not None
    posterior = engine.posterior
    assert posterior["T"] > posterior.get("F", 0.0), (
        f"posterior failed to concentrate on truth: {posterior}"
    )


# --------------------------------------------------------------------------- #
# Proposition 7 — Bayesian commit implies bounded Bayes factor
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_bayesian_commit_requires_bayes_factor() -> None:
    """Proposition: any committed round in the Bayesian engine has
    log-Bayes-factor ≥ log_bf_min over the runner-up."""
    agents = [MockAgent(f"a{i}", "T", 0.9) for i in range(6)]
    engine = BayesianMBFTEngine(
        agents,
        posterior_threshold=0.85,
        max_rounds=3,
    )
    result = await engine.run("task")

    if result and result.committed:
        assert result.log_bayes_factor >= engine.log_bf_min - 1e-9
