# Architecture

## Project Structure

```
metacognition/
├── src/
│   ├── core/                        # Protocol engine
│   │   ├── protocol.py              # Base mBFT consensus protocol
│   │   ├── protocol_bayesian.py     # Bayesian belief-update extension
│   │   └── state.py                 # Round state, proposal, vote models
│   ├── agents/                      # Agent implementations
│   │   ├── base.py                  # Abstract agent interface
│   │   └── metacognitive.py         # Full metacognitive agent
│   ├── network/                     # Communication layer
│   │   └── simulator.py             # Asyncio network simulator with fault injection
│   │
│   │ ── Self-Regulation ──
│   ├── homeostasis.py               # Vital-sign monitoring + corrective actions
│   ├── autophagy.py                 # Dysfunctional agent detection + removal
│   ├── autopilot.py                 # Autonomous parameter tuning
│   ├── immune.py                    # Bio-inspired adversarial defense
│   ├── circadian.py                 # Agent performance rhythms
│   ├── endocrine.py                 # Hormonal signaling for global regulation
│   ├── nociception.py               # Pain/damage signaling + protective reflex
│   ├── senescence.py                # Agent aging and rejuvenation
│   │
│   │ ── Governance & Economics ──
│   ├── governance.py                # Constitutional governance (amendments)
│   ├── diplomacy.py                 # Inter-agent negotiation + alliances
│   ├── prediction_market.py         # Internal prediction markets
│   ├── economy.py                   # Resource budgets + fiscal policy
│   ├── accountability.py            # Immutable audit ledger
│   │
│   │ ── Analysis & Forensics ──
│   ├── forensics.py                 # Post-mortem behavioral profiling
│   ├── deadlock.py                  # Voting cycle detection + resolution
│   ├── lineage.py                   # Decision lineage tracing
│   ├── trust_tracker.py             # Trust score evolution
│   ├── calibrator.py                # Confidence calibration curves
│   ├── monitor.py                   # Consensus resilience metrics
│   │
│   │ ── Bio-Inspired ──
│   ├── morphogenesis.py             # Reaction-diffusion role differentiation
│   ├── epigenetics.py               # Heritable behavioral marks
│   ├── neuroplasticity.py           # Dynamic topology rewiring
│   ├── speciation.py                # Behavioral divergence tracking
│   ├── stigmergy.py                 # Pheromone-based indirect coordination
│   ├── quorum_sensing.py            # Density-dependent state transitions
│   ├── symbiosis.py                 # Agent relationship dynamics
│   ├── angiogenesis.py              # Communication pathway growth/pruning
│   ├── chemotaxis.py                # Chemical gradient navigation
│   ├── mitosis.py                   # Agent replication with cell-cycle phases
│   ├── proprioception.py            # Swarm body-schema awareness
│   │
│   │ ── Collective Intelligence ──
│   ├── consciousness.py             # Collective self-awareness metrics
│   ├── dreaming.py                  # Hypothetical scenario rehearsal
│   ├── swarm_memory.py              # Episodic collective memory
│   ├── quorum_predict.py            # Pre-round outcome prediction
│   ├── social_learning.py           # Cultural evolution via social learning
│   │
│   │ ── Dynamics & Topology ──
│   ├── grudge.py                    # Persistent inter-agent conflicts
│   ├── influence.py                 # Influence centrality metrics
│   ├── spectral.py                  # Spectral correlation analysis
│   ├── emergence.py                 # Emergent behavior detection
│   ├── landscape.py                 # Consensus fitness landscapes
│   ├── regime.py                    # Phase transition early warnings
│   ├── cascade.py                   # Failure cascade modeling
│   ├── diversity.py                 # Behavioral diversity metrics
│   ├── learning_curve.py            # Collective learning analysis
│   ├── tournament.py                # Competitive strategy evaluation
│   │
│   │ ── Testing & Debugging ──
│   ├── adversarial_trainer.py       # Attack scenario generation
│   ├── fuzzer.py                    # Protocol fuzzing
│   ├── partition.py                 # Network partition simulation
│   ├── decomposer.py               # Task decomposition strategies
│   ├── replay.py                    # Deterministic round replay
│   ├── monitor.py                   # Consensus resilience stress-testing
│   │
│   │ ── Agentic Advisors (read-only recommenders) ──
│   ├── swarm_health.py              # Aggregate health monitor + recommendations
│   ├── disagreement_forensics.py    # Per-round failure root-cause analyzer
│   ├── round_replay_advisor.py      # Counterfactual "what-if" round replay
│   ├── agent_lifecycle_advisor.py   # Roster-level keep/watch/evict planner
│   ├── leader_rotation_advisor.py   # Next-N-rounds leader schedule planner
│   ├── voting_coalition_detector.py # Voting-bloc / faction detector
│   ├── proposal_risk_scorer.py      # Pre-submission commit-risk advisor
│   ├── threshold_tuning_advisor.py  # Threshold + slash-factor auto-tuner
│   ├── vote_dispersion_advisor.py   # Vote-weight distribution pattern classifier
│   │
│   │ ── Adaptive Energy & Ecosystem ──
│   ├── allostasis.py                # Predictive ("stability through change") regulation
│   ├── hibernation.py               # Torpor/arousal energy-conservation engine
│   ├── microbiome.py                # Commensal-agent ecosystem manager
│   └── stats_utils.py               # Shared statistical helpers
│
├── tests/                           # Test suite with fault injection
├── paper/                           # LaTeX source for the research paper
├── docs/                            # MkDocs Material documentation
├── requirements.txt
└── pyproject.toml
```

## Core Components

### Protocol Engine (`src/core/`)

The heart of mBFT. `protocol.py` implements the four-phase consensus loop:

1. **Leader Election** — Confidence-weighted selection (not round-robin)
2. **Proposal** — Leader generates solution + proof + confidence
3. **Verification** — Followers produce counter-proofs
4. **Commit/Abort** — Confidence-weighted finality check

`protocol_bayesian.py` adds posterior confidence updates. `state.py` defines the data models for rounds, proposals, and votes.

### Agents (`src/agents/`)

`base.py` defines the abstract agent interface (propose, verify, vote). `metacognitive.py` implements the full metacognitive agent with confidence self-assessment, counter-proof generation, and belief revision.

### Network (`src/network/`)

`simulator.py` provides an asyncio-based network simulator that supports configurable latency, message loss, reordering, and Byzantine behavior injection.

## Engine Subsystems

The project has grown to 60+ specialized engines organized into six functional areas:

| Area | Engines | Purpose |
|------|---------|---------|
| **Self-Regulation** | Homeostasis, Autophagy, Autopilot, Immune, Circadian, Endocrine, Nociception, Senescence | Keep the swarm stable and self-correcting |
| **Governance** | Governance, Diplomacy, PredictionMarket, Economy, Audit | Decentralized decision-making and resources |
| **Analysis** | Forensics, Deadlock, Lineage, Trust, Calibrator, Monitor | Deep inspection of behavior and failures |
| **Bio-Inspired** | Morphogenesis, Epigenetics, Neuroplasticity, Speciation, Stigmergy, Quorum, Symbiosis, Angiogenesis, Chemotaxis, Mitosis, Proprioception | Biological system analogies |
| **Collective Intelligence** | Consciousness, Dreaming, SwarmMemory, QuorumPredict, SocialLearning | Higher-order reasoning and anticipation |
| **Dynamics** | Grudge, Influence, Monitor, Spectral, Emergence, Landscape, Regime, Cascade, Diversity, Learning, Tournament | Structural and temporal analysis |
| **Agentic Advisors** | SwarmHealth, DisagreementForensics, RoundReplayAdvisor, AgentLifecycleAdvisor, LeaderRotationAdvisor, VotingCoalitionDetector, ProposalRiskScorer, ThresholdTuningAdvisor, VoteDispersionAdvisor | Read-only recommenders that observe history and emit prioritised playbooks |
| **Adaptive Energy** | Allostasis, Hibernation, Microbiome | Predictive regulation, energy conservation, commensal-agent ecosystem |

See the **[Engine Catalog](engines.md)** for detailed documentation of every engine, its module path, key classes, and purpose.

## Shared Utilities

- **`stats_utils.py`** — Statistical helper functions (mean, stddev, percentiles) used across engines
- **`__init__.py`** — Package exports for clean imports

