# Testing & Simulation Guide

How to verify swarm correctness, stress-test under adversarial conditions, and debug consensus failures.

---

## Running the Test Suite

```bash
# Full test suite
pytest -q

# Specific engine tests
pytest tests/test_immune.py -v
pytest tests/test_consensus.py -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

### Test Categories

| Test File | What It Covers |
|-----------|----------------|
| `test_consensus.py` | Core protocol: commit/reject, threshold behavior, multi-round convergence |
| `test_immune.py` | Threat detection, immune memory, adaptive response |
| `test_homeostasis.py` | Vital sign monitoring, corrective action triggers |
| `test_autophagy.py` | Dysfunctional agent detection and removal |
| `test_morphogenesis.py` | Role differentiation, morphogen gradients |
| `test_epigenetics.py` | Heritable behavioral marks across generations |
| `test_neuroplasticity.py` | Connection rewiring based on co-voting |
| `test_speciation.py` | Behavioral divergence and subpopulation detection |
| `test_stigmergy.py` | Pheromone-based indirect coordination |
| `test_quorum_sensing.py` | Density-dependent collective state transitions |
| `test_symbiosis.py` | Agent relationship modeling (mutualism, parasitism) |
| `test_consciousness.py` | Collective self-awareness metrics |
| `test_dreaming.py` | Offline scenario simulation |
| `test_swarm_memory.py` | Episodic memory storage and retrieval |
| `test_prediction_market.py` | Internal prediction market mechanics |
| `test_grudge.py` | Persistent inter-agent conflict tracking |
| `test_circadian.py` | Performance rhythm modeling |
| `test_chemotaxis.py` | Gradient-following agent navigation |
| `test_decomposer.py` | Task decomposition strategies |
| `test_propositions.py` | Proposal and vote data model validation |

---

## Network Simulation

The built-in simulator (`src/network/simulator.py`) supports configurable network conditions.

### Basic Simulation

```bash
# Default: 5 agents, 3 rounds, no faults
python -m src.network.simulator

# Custom simulation
python -m src.network.simulator --agents 7 --rounds 10 --byzantine 2
```

### Fault Injection

The simulator supports injecting Byzantine behaviors:

```python
from src.network.simulator import NetworkSimulator

sim = NetworkSimulator(
    agent_count=7,
    byzantine_count=2,     # 2 agents will behave adversarially
    latency_ms=(10, 100),  # Random latency range
    message_loss=0.05,     # 5% message drop rate
)
results = await sim.run(rounds=20)
```

### Network Partitions

Use `NetworkPartitionSimulator` to test split-brain scenarios:

```python
from src.partition import NetworkPartitionSimulator

partitioner = NetworkPartitionSimulator()

# Create a partition splitting the swarm
partition = partitioner.create_partition(
    agents=all_agents,
    groups=[[0, 1, 2], [3, 4, 5, 6]],  # Two disconnected groups
)

# Run consensus in each partition
for group in partition.groups:
    result = await engine.run(task, agents=group)
    # Only the majority partition should be able to commit
```

---

## Adversarial Testing

### Fuzzing

The `FuzzableAgent` generates random mutations to test protocol robustness:

```python
from src.fuzzer import FuzzableAgent, FuzzerStats

fuzzer = FuzzableAgent(base_agent)
stats = FuzzerStats()

for _ in range(1000):
    # Fuzz proposals, votes, confidence values
    mutated = fuzzer.fuzz()
    outcome = await engine.run_with(mutated)
    stats.record(outcome)

print(stats.summary())
# Shows: edge cases found, crash triggers, unexpected commits
```

### Adversarial Training

Train the swarm against known attack patterns:

```python
from src.adversarial_trainer import AdversarialMockAgent, TrainingHistory

trainer = AdversarialMockAgent(attack_type="confidence_manipulation")
history = TrainingHistory()

# Run attack scenarios
for scenario in trainer.generate_scenarios(count=50):
    result = await engine.run(scenario.task)
    history.record(scenario, result)

# Analyze resilience
print(history.resilience_score())
```

### Common Attack Patterns

| Attack | Description | Detection |
|--------|-------------|-----------|
| **Confidence manipulation** | Byzantine agent reports inflated confidence | `AgentCalibration` detects miscalibration |
| **Collusion** | Multiple agents coordinate to push incorrect answers | `ForensicsAnalyzer.detect_collusion()` |
| **Vote buying** | Agent changes votes based on other votes | `AuditEngine` detects vote pattern anomalies |
| **Sybil** | Single entity controls multiple agent identities | `ForensicsAnalyzer.detect_sybil()` |
| **Slowloris** | Agent intentionally delays responses | `CircadianEngine` detects latency outliers |

---

## Debugging Consensus Failures

### Step 1: Enable Replay Recording

```python
from src.lineage import InstrumentedEngine

instrumented = InstrumentedEngine(engine)
result = await instrumented.run(task)

# Save replay for offline analysis
replay = instrumented.export_replay()
replay.save("debug_round.json")
```

### Step 2: Analyze the Failure

```python
from src.forensics import ForensicsAnalyzer
from src.replay import ReplayData

# Load replay data
replay = ReplayData.load("debug_round.json")
forensics = ForensicsAnalyzer()

# Get behavioral profiles
for profile in forensics.build_profiles(replay):
    print(f"Agent {profile.agent_id}:")
    print(f"  Accuracy: {profile.accuracy:.1%}")
    print(f"  Confidence calibration: {profile.calibration:.3f}")
    print(f"  Collusion score: {profile.collusion_score:.3f}")
```

### Step 3: Check for Deadlocks

```python
from src.deadlock import DeadlockDetector

detector = DeadlockDetector()
deadlocks = detector.analyze(replay)

for dl in deadlocks:
    print(f"Deadlock type: {dl.type}")
    print(f"Agents involved: {dl.agents}")
    print(f"Resolution: {dl.suggested_resolution}")
```

### Common Debugging Scenarios

!!! info "All agents vote positively but no commit"
    Check that the aggregate vote sum exceeds `threshold`.
    Remember: `Commit ⇔ (Σ V_i ≥ θ_meta) ∧ (min V_i ≥ 0)`.
    Even one zero vote can prevent commit if the remaining votes don't reach threshold.

!!! info "Same agent always becomes leader"
    This is expected if one agent consistently has the highest confidence.
    If it's undesirable, lower `slash_factor` to punish failed leaders more aggressively,
    or enable `NeuroplasticityEngine` to diversify leadership rotation.

!!! info "Swarm converges to wrong answer"
    Run `AgentCalibration` — overconfident agents can dominate consensus.
    Consider using `BayesianMBFTEngine` for better posterior calibration,
    or add more diverse agents (different models/providers).

---

## Continuous Integration

The project includes a CI workflow (`.github/workflows/ci.yml`) that runs tests on every push.

### Adding New Tests

Follow the existing conventions:

```python
# tests/test_my_engine.py
import pytest
from src.my_engine import MyEngine

class TestMyEngine:
    def test_basic_behavior(self):
        engine = MyEngine()
        result = engine.process(input_data)
        assert result.status == "success"

    def test_edge_case(self):
        engine = MyEngine()
        with pytest.raises(ValueError):
            engine.process(None)

    @pytest.mark.asyncio
    async def test_async_operation(self):
        engine = MyEngine()
        result = await engine.run_async(task)
        assert result is not None
```

### Test Fixtures

Use fixtures for common setup:

```python
@pytest.fixture
def swarm():
    """Create a standard 5-agent swarm for testing."""
    agents = [MockAgent(f"test-{i}") for i in range(5)]
    return MBFTEngine(agents=agents, threshold=0.6)

def test_commit(swarm):
    result = asyncio.run(swarm.run("test task"))
    assert result.committed
```
