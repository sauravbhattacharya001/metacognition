# Engine Catalog

mBFT includes 40+ specialized engines beyond the core consensus protocol. These engines extend the swarm with self-healing, governance, analysis, and bio-inspired behaviors.

---

## Self-Regulation & Health

Engines that keep the swarm stable and self-correcting.

### HomeostasisController
**Module:** `src/homeostasis.py`

Monitors vital signs (latency, agreement rate, throughput) and triggers corrective actions when readings drift outside safe bands. Produces `HealthReport` snapshots.

### AutophagyEngine
**Module:** `src/autophagy.py`

Detects and removes dysfunctional agents or subsystems. Identifies `Dysfunction` entries (stale agents, chronically wrong voters) and triggers quarantine or removal.

### ConsensusAutopilot
**Module:** `src/autopilot.py`

Autonomous operator that adjusts engine parameters (threshold, max rounds, slash factor) in real-time based on health telemetry. Maintains `QuarantineRecord` for agents placed on hold.

### ImmuneSystem
**Module:** `src/immune.py`

Bio-inspired defense against adversarial agents. Classifies threats as `Pathogen` instances, builds `ImmuneMemory` of past attacks, and mounts adaptive responses to novel Byzantine behaviors.

### CircadianEngine
**Module:** `src/circadian.py`

Models agent performance variation over time. Tracks `PerformanceSample` data to learn each agent's peak and trough periods, enabling smarter leader election scheduling.

---

## Governance & Economics

Decentralized decision-making and resource management.

### GovernanceEngine
**Module:** `src/governance.py`

Constitutional governance layer. Agents can propose and vote on `Amendment` changes to protocol parameters, consensus rules, or trust formulas.

### DiplomacyEngine
**Module:** `src/diplomacy.py`

Manages inter-agent negotiations and alliances. Records `DiplomaticEvent` history (treaties, conflicts, mediation) to influence voting coalitions.

### PredictionMarketEngine
**Module:** `src/prediction_market.py`

Internal prediction markets where agents bet on consensus outcomes. `Market` instances aggregate crowd wisdom to forecast round success probability.

### EconAgent / FiscalPolicy
**Module:** `src/economy.py`

Economic simulation layer. `EconAgent` instances manage resource budgets; `FiscalPolicy` controls inflation/deflation of trust tokens across the swarm.

### AuditEngine
**Module:** `src/accountability.py`

Immutable audit log. Records every proposal, vote, and parameter change as `LedgerEntry` items for post-hoc accountability analysis.

---

## Analysis & Forensics

Deep inspection of swarm behavior and failure modes.

### ForensicsAnalyzer
**Module:** `src/forensics.py`

Post-mortem analysis of failed consensus rounds. Builds `AgentProfile` behavioral fingerprints to identify collusion, incompetence, or sabotage.

### DeadlockDetector / DeadlockResolver
**Module:** `src/deadlock.py`

Detects voting cycles and irreconcilable disagreements via `VetoEdge` graph analysis. `DeadlockResolver` applies tie-breaking strategies.

### ConsensusLineageTracker
**Module:** `src/lineage.py`

Tracks the causal chain of decisions across rounds. `InstrumentedEngine` wraps the core protocol to record which proposals influenced later outcomes.

### TrustEvolutionTracker
**Module:** `src/trust_tracker.py`

Visualizes trust score trajectories over time. Produces `ReputationSnapshot` timelines for each agent.

### AgentCalibration / CalibrationReport
**Module:** `src/calibrator.py`

Measures whether agents' confidence scores predict their actual accuracy. `CalibrationReport` includes reliability diagrams and Brier scores.

---

## Bio-Inspired Dynamics

Engines inspired by biological systems.

### MorphogenesisEngine
**Module:** `src/morphogenesis.py`

Models agent role differentiation using reaction-diffusion dynamics. `CellFate` assignments emerge from morphogen gradients across the agent network.

### EpigeneticsEngine
**Module:** `src/epigenetics.py`

Heritable behavioral modifications without changing core agent logic. `MarkType` tags alter agent behavior across consensus generations.

### NeuroplasticityEngine
**Module:** `src/neuroplasticity.py`

Dynamic rewiring of agent communication topology. Records `PlasticityEvent` instances as connections strengthen (Hebbian) or weaken based on co-voting patterns.

### SpeciationEngine
**Module:** `src/speciation.py`

Tracks agent behavioral divergence. When subpopulations develop distinct strategies (`TaskRecord` profiles), the engine identifies potential speciation events.

### StigmergyEngine
**Module:** `src/stigmergy.py`

Indirect coordination via `PheromoneType` signals in shared memory. Agents leave traces that guide future decisions without direct communication.

### SwarmQuorumSensingEngine
**Module:** `src/quorum_sensing.py`

Density-dependent behavior switching. `SignalChannel` broadcasts aggregate when enough agents emit similar signals, triggering collective state transitions.

### SymbiosisEngine
**Module:** `src/symbiosis.py`

Models mutualistic, commensal, and parasitic agent `RelationshipType` interactions. Tracks fitness effects of inter-agent dependencies.

---

## Simulation & Testing

Controlled environments for stress-testing swarms.

### AdversarialMockAgent / TrainingHistory
**Module:** `src/adversarial_trainer.py`

Generates `AttackScenario` sequences to stress-test swarm resilience. `TrainingHistory` records which attack patterns the swarm learned to resist.

### FuzzableAgent / FuzzerStats
**Module:** `src/fuzzer.py`

Protocol fuzzer that mutates proposals, votes, and timing. `FuzzOutcome` results identify edge cases; `FuzzerStats` summarizes coverage.

### NetworkPartitionSimulator
**Module:** `src/partition.py`

Creates and manages `Partition` splits in the agent network. Tests consensus behavior under network fragmentation and healing.

### SwarmTaskDecomposer
**Module:** `src/decomposer.py`

Breaks complex tasks into `Subtask` graphs with dependency ordering. Supports multiple `DecompositionStrategy` patterns (parallel, pipeline, hierarchical).

---

## Collective Intelligence

Higher-order reasoning and emergent behavior.

### SwarmConsciousnessEngine
**Module:** `src/consciousness.py`

Measures collective self-awareness. Tracks `AgentBelief` models about the swarm's own state, enabling meta-level reasoning about group performance.

### SwarmDreamEngine / AnticipationEngine
**Module:** `src/dreaming.py`

Offline simulation of hypothetical scenarios. `Episode` logs record "dreams" where the swarm rehearses responses to anticipated challenges.

### SwarmMemory
**Module:** `src/swarm_memory.py`

Collective episodic memory. Stores and retrieves `Episode` records with `MemoryHealth` monitoring for decay, interference, and consolidation.

### QuorumPredictor
**Module:** `src/quorum_predict.py`

Forecasts consensus outcomes before rounds complete. `AgentProfile` features feed `PredictionReport` estimates of commit probability.

---

## Dynamics & Topology

Structural analysis of swarm behavior.

### GrudgeEngine
**Module:** `src/grudge.py`

Tracks persistent inter-agent conflicts. `Interaction` history identifies grudges that bias future voting patterns.

### Influence Analysis
**Module:** `src/influence.py`

Computes agent influence metrics (centrality, vote leverage, persuasion effectiveness) over the communication graph.

### Spectral Analysis
**Module:** `src/spectral.py`

Spectral decomposition of the swarm's voting correlation matrix. Identifies clusters, factions, and latent structure.

### Emergence Detection
**Module:** `src/emergence.py`

Detects `EmergenceSignal` patterns — behaviors that arise at the swarm level but aren't programmed into individual agents. `FactionInfo` tracks emergent subgroups.

### Landscape Analysis
**Module:** `src/landscape.py`

Maps the consensus fitness landscape. `CellResult` sampling identifies `TippingPoint` regions where small perturbations cause phase transitions.

### Regime Detection
**Module:** `src/regime.py`

Identifies regime changes in swarm dynamics. `SignalVector` time series analysis detects `EarlyWarning` indicators of impending phase shifts.

### Cascade Analysis
**Module:** `src/cascade.py`

Models information and failure cascades. `CascadeSignal` propagation through the network is tracked to predict and contain chain reactions.

### DiversityAnalysis
**Module:** `src/diversity.py`

Measures behavioral, strategic, and opinion diversity across agents. Low diversity scores flag groupthink risk.

### Learning Curve Analysis
**Module:** `src/learning_curve.py`

Tracks collective learning over time. `DifficultyLevel` progression and `BreakthroughEvent` detection reveal how fast the swarm acquires new competencies.

### Tournament
**Module:** `src/tournament.py`

Competitive evaluation framework. `TeamConfig` pairings produce `MatchResult` data comparing agent strategies head-to-head.

---

## Replay & Debugging

### RoundEvent / ReplayData
**Module:** `src/replay.py`

Records consensus rounds as `RoundEvent` sequences. `ReplayData` bundles enable deterministic replay for debugging and analysis.
