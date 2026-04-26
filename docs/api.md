# API Reference

## Core Protocol

### `MBFTEngine`

::: src.core.protocol.MBFTEngine

The main consensus engine. Drives mBFT rounds over a fixed set of agents.

```python
from src.core.protocol import MBFTEngine

engine = MBFTEngine(
    agents=[agent1, agent2, agent3],
    threshold=0.6,
    max_rounds=4,
    slash_factor=0.5,
)
result = await engine.run("What is the capital of France?")
```

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `agents` | `List[BaseAgent]` | Participating agents (at least 1) |
| `threshold` | `float` | Minimum aggregate vote sum for commit (must be > 0) |
| `max_rounds` | `int` | Maximum rounds before giving up (default: 4) |
| `slash_factor` | `float` | Trust reduction on failed leadership (default: 0.5) |

**Methods:**

- `async run(task_prompt: str) -> Optional[RoundResult]` — Drive rounds until commit or `max_rounds` reached.

---

## Data Models (`src.core.state`)

### `Proposal`

Represents an agent's proposed solution with proof and confidence score.

### `Vote`

A follower's vote on a leader's proposal, in range `[-τ_i, τ_i]`.

### `RoundResult`

The outcome of a single consensus round, including proposal, votes, and commit status.

---

## Agents

### `BaseAgent` (abstract)

```python
from src.agents.base import BaseAgent
```

Abstract base class defining the agent contract:

- `async generate_proposal(task: str) -> Proposal` — Produce a (solution, proof, τ_i) tuple.
- `async verify_proposal(leader_proposal: Proposal) -> Vote` — Inspect a proposal and return a signed/weighted vote.

### `MetacognitiveAgent`

Full implementation with confidence self-assessment, counter-proof generation, and Bayesian belief revision.

---

## Network

### `simulator`

Asyncio-based network simulator (`src.network.simulator`):

- Configurable latency, message loss, and reordering
- Byzantine behavior injection
- Run with: `python -m src.network.simulator`

---

## Monitoring & Analysis

| Module | Purpose |
|--------|---------|
| `calibrator.py` | Calibration curve tracking — measures whether confidence matches accuracy |
| `monitor.py` | Consensus resilience monitoring — liveness, agreement rate, round duration |
| `trust_tracker.py` | Per-agent trust score evolution across rounds |
| `partition.py` | Network partition simulation for resilience testing |
| `replay.py` | Round recording and replay for debugging |
