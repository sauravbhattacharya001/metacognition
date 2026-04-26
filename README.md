# mBFT — Metacognitive Byzantine Fault Tolerance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation for the paper **"Consensus-Driven Metacognition in
Multi-Agent Systems."** mBFT is a semantic-consensus protocol for swarms of
LLM-style agents that combines Byzantine Fault Tolerance with defeasible,
epistemic-logic reasoning.

## Why mBFT?

Classical CFT protocols (Paxos, Raft) assume a non-faulty node tells the truth.
LLM agents violate that assumption: a confidently hallucinating agent is a
Byzantine fault, not a crash. mBFT replaces flat node-counting with
**confidence-weighted, defeasible voting** over logical proofs.

## Protocol summary

For round `r` over agent set `A = {a_1 … a_n}`:

1. **Epistemic Leader Election** — `L_r = argmax_i τ_i(S_i)`
2. **Semantic Verification** — each follower returns a vote
   `V_i ∈ [-τ_i, τ_i]`; a valid counter-proof yields `V_i < 0`.
3. **Confidence-Weighted Finality** —
   `Commit(S_L) ⇔ (Σ V_i ≥ θ_meta) ∧ (min V_i ≥ 0)`
4. On failure, the leader's weight is slashed and the counter-proposer leads
   round `r+1`.

See `src/core/protocol.py` for the executable specification.

## Layout

```
metacognition/
├── src/
│   ├── core/            # State models + mBFT engine
│   ├── agents/          # Base + metacognitive agent (mock + LLM)
│   └── network/         # Asyncio simulator / message bus
└── tests/               # Including Byzantine fault-injection tests
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
python -m src.network.simulator   # demo run with mock agents
```

The default demo uses `MockAgent`, so no API key is required. To use a real
LLM, implement an `LLMClient` and pass it to `MetacognitiveAgent`.

## Status

Theoretical-framework reference impl. Future work: Bayesian belief updates,
HotStuff-style pipelined view changes, calibration benchmarks.
