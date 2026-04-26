# Protocol Details

## Overview

mBFT operates in rounds. Each round proceeds through four phases:

### 1. Epistemic Leader Election

```
L_r = argmax_i τ_i(S_i)
```

The agent with the highest confidence in its proposal becomes the leader for round `r`. This replaces the arbitrary round-robin rotation of classical BFT protocols with a meritocratic selection based on epistemic state.

### 2. Semantic Verification

Each follower evaluates the leader's proposal and returns a vote `V_i ∈ [-τ_i, τ_i]`:

- **Positive vote**: follower's reasoning supports the proposal
- **Negative vote**: follower produces a valid counter-proof
- **Zero vote**: follower abstains (insufficient evidence)

### 3. Confidence-Weighted Finality

A proposal commits if and only if:

```
Commit(S_L) ⇔ (Σ V_i ≥ θ_meta) ∧ (min V_i ≥ 0)
```

This dual threshold prevents both low-confidence consensus and single-agent vetoes without proof.

### 4. Failure Recovery

On failed consensus, the leader's trust weight is slashed and the highest-voted counter-proposer leads round `r+1`.

## Bayesian Extension

The `protocol_bayesian.py` module extends the base protocol with posterior belief updates. After each round, agents update their confidence priors using observed voting outcomes:

- Agents whose votes aligned with the final outcome gain confidence
- Agents whose votes diverged lose confidence proportionally
- The calibrator module (`calibrator.py`) tracks long-term calibration curves

## Trust Tracking

The `trust_tracker.py` module maintains per-agent trust scores that evolve across rounds:

- Trust increases for consistent, well-calibrated voters
- Trust decreases for Byzantine behavior or persistent miscalibration
- Trust scores feed back into leader election weights

## Network Partitions

The `partition.py` module simulates network partitions to test protocol resilience:

- Symmetric and asymmetric partition scenarios
- Gradual healing with message reordering
- Verifies safety (no conflicting commits) under all partition modes
