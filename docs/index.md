# mBFT — Metacognitive Byzantine Fault Tolerance

Reference implementation for the paper **"Consensus-Driven Metacognition in Multi-Agent Systems."**

mBFT is a semantic-consensus protocol for swarms of LLM-style agents that combines Byzantine Fault Tolerance with defeasible, epistemic-logic reasoning.

## Why mBFT?

Classical CFT protocols (Paxos, Raft) assume a non-faulty node tells the truth.
LLM agents violate that assumption: a confidently hallucinating agent is a
Byzantine fault, not a crash. mBFT replaces flat node-counting with
**confidence-weighted, defeasible voting** over logical proofs.

## Key Features

- **Epistemic leader election** — leaders are chosen by confidence, not round-robin
- **Semantic verification** — followers produce counter-proofs, not just ack/nack
- **Confidence-weighted finality** — commit requires both aggregate and minimum vote thresholds
- **Bayesian belief updates** — agents refine confidence through posterior calibration
- **Async network simulation** — test consensus under partitions, delays, and Byzantine faults

## Quick Links

- [Protocol Details](protocol.md)
- [Architecture Overview](architecture.md)
- [API Reference](api.md)
- [Getting Started](getting-started.md)
- [GitHub Repository](https://github.com/sauravbhattacharya001/metacognition)
