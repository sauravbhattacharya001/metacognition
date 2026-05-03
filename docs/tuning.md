# Tuning & Operations Guide

Practical guidance for configuring, deploying, and operating mBFT swarms in production.

---

## Core Protocol Parameters

The `MBFTEngine` constructor exposes three parameters that control consensus behavior.
Choosing the right values depends on your swarm size, trust model, and latency tolerance.

### Commitment Threshold (`threshold`)

The minimum aggregate vote sum required to commit a proposal.

```python
engine = MBFTEngine(agents=agents, threshold=0.6)
```

| Value Range | Behavior | Use When |
|-------------|----------|----------|
| **0.3 – 0.5** | Permissive — commits easily | Low-stakes tasks, fast iteration, trusted agents |
| **0.5 – 0.7** | Balanced — reasonable safety | General-purpose production workloads |
| **0.7 – 0.9** | Conservative — hard to commit | High-stakes decisions, untrusted agents |
| **> 0.9** | Very strict — frequent stalls | Safety-critical domains (medical, legal, financial) |

!!! tip "Rule of Thumb"
    Start with `threshold = 0.6` and adjust based on your commit rate.
    If commits happen too easily (false positives), raise it.
    If rounds stall too often, lower it.

### Maximum Rounds (`max_rounds`)

How many rounds the engine attempts before giving up on a task.

| Value | Trade-off |
|-------|-----------|
| **2 – 3** | Fast failure, low latency, but may miss eventual agreement |
| **4** (default) | Balanced — enough retries for leader rotation |
| **6 – 10** | Patient — useful when agents have high variance |

!!! warning
    Each round involves a full proposal + verification cycle across all agents.
    High `max_rounds` multiplied by high agent count can be expensive with real LLM backends.

### Slash Factor (`slash_factor`)

How much to reduce an agent's reputation after a failed leadership round.

```python
engine = MBFTEngine(agents=agents, threshold=0.6, slash_factor=0.5)
```

| Value | Effect |
|-------|--------|
| **1.0** | No punishment — reputation unchanged after failure |
| **0.5** (default) | Halve reputation — aggressive penalty |
| **0.7 – 0.9** | Mild penalty — appropriate for noisy environments |
| **0.1 – 0.3** | Severe — effectively bans agents after 1–2 failures |

!!! note
    Low slash factors create a "winner-take-all" dynamic where early leaders dominate.
    High slash factors are more forgiving but slower to converge on reliable leaders.

---

## Autopilot Configuration

The `ConsensusAutopilot` (`src/autopilot.py`) runs continuous consensus loops with self-tuning.

### CLI Options

```bash
# Basic demo
python -m src.autopilot

# Custom configuration
python -m src.autopilot \
    --agents 7 \
    --cycles 50 \
    --threshold 0.65 \
    --quarantine-floor 0.15 \
    --quarantine-cooldown 3 \
    --export html -o dashboard.html
```

### Adaptive Threshold

The Autopilot adjusts `threshold` automatically based on outcomes:

- **After a false commit** (detected via post-hoc verification): threshold increases
- **After a stall** (no commit within `max_rounds`): threshold decreases

This feedback loop finds the sweet spot between safety and liveness without manual tuning.

### Agent Quarantine

Agents whose reputation drops below `quarantine_floor` are benched for `quarantine_cooldown` rounds,
then reinstated on probation. This prevents chronically bad agents from disrupting consensus
while giving them a chance to recover.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quarantine_floor` | 0.15 | Reputation threshold for quarantine |
| `quarantine_cooldown` | 3 | Rounds before reinstatement |

---

## Swarm Sizing

### How Many Agents?

The classic BFT formula requires `n ≥ 3f + 1` agents to tolerate `f` Byzantine faults.
mBFT's confidence-weighted voting relaxes this somewhat, but the principle holds:

| Agents | Byzantine Tolerance | Notes |
|--------|-------------------|-------|
| **3** | 0 | Minimum viable — no fault tolerance |
| **4** | 1 | One faulty agent tolerated |
| **5** | 1 | Comfortable with one fault |
| **7** | 2 | Good for production |
| **10+** | 3+ | High-reliability deployments |

!!! tip
    With mBFT, even `n=3` provides value via epistemic leader election and semantic verification.
    You get better decisions even without Byzantine fault tolerance.

### Cost Estimation

Each consensus round involves:

- 1 proposal generation (leader only)
- `n-1` verification calls (all followers)
- Up to `max_rounds` rounds per task

**Per-task LLM calls** = `max_rounds × n` (worst case) to `n` (best case, single-round commit).

For a 7-agent swarm with `max_rounds=4`:

- **Best case**: 7 LLM calls
- **Worst case**: 28 LLM calls
- **Typical**: 10–14 LLM calls (commit in 1–2 rounds)

---

## Engine Composition

mBFT engines are designed to compose. Enable only what you need.

### Minimal Setup (Core Only)

```python
from src.core.protocol import MBFTEngine
from src.agents.metacognitive import MetacognitiveAgent

agents = [MetacognitiveAgent(f"agent-{i}") for i in range(5)]
engine = MBFTEngine(agents=agents, threshold=0.6)
result = await engine.run("Your task here")
```

### Production Setup (Core + Health + Governance)

```python
from src.homeostasis import HomeostasisController
from src.immune import ImmuneSystem
from src.autopilot import ConsensusAutopilot
from src.accountability import AuditEngine

# Wrap the core engine with health monitoring
health = HomeostasisController(engine)
immune = ImmuneSystem(engine)
audit = AuditEngine()

# Run with autopilot for continuous operation
autopilot = ConsensusAutopilot(
    agents=agents,
    threshold=0.6,
    quarantine_floor=0.15,
)
```

### Analysis Setup (Core + Forensics + Replay)

```python
from src.lineage import InstrumentedEngine
from src.replay import ReplayData
from src.forensics import ForensicsAnalyzer

# Wrap for full observability
instrumented = InstrumentedEngine(engine)
result = await instrumented.run(task)

# Post-hoc analysis
forensics = ForensicsAnalyzer()
report = forensics.analyze(instrumented.lineage)
```

---

## Monitoring & Observability

### Health Metrics

The `HomeostasisController` tracks vital signs:

| Metric | Healthy Range | Action on Violation |
|--------|--------------|-------------------|
| Agreement rate | > 60% | Lower threshold or investigate agent quality |
| Average rounds | < 3 | No action needed |
| Stall rate | < 20% | Raise agent count or lower threshold |
| Quarantine rate | < 30% | Replace bad agents or retrain |

### Trust Score Monitoring

Use `TrustEvolutionTracker` to watch reputation trajectories:

```python
from src.trust_tracker import TrustEvolutionTracker

tracker = TrustEvolutionTracker()
# After each round:
tracker.record(engine)
# Visualize:
tracker.summary()
```

Healthy swarms show stable or slowly converging trust scores.
Diverging trust scores suggest adversarial behavior or miscalibrated agents.

### Calibration Audits

Run `AgentCalibration` periodically to check whether agent confidence
predicts accuracy:

```python
from src.calibrator import AgentCalibration

cal = AgentCalibration()
report = cal.evaluate(agent, history)
print(f"Brier score: {report.brier_score:.3f}")
# < 0.1 = well calibrated, > 0.3 = poorly calibrated
```

---

## Failure Modes & Recovery

### Stall (No Commit)

**Symptoms**: Every round exhausts `max_rounds` without committing.

**Diagnosis**:

1. Check if agents disagree fundamentally (diverse training data / prompts)
2. Check if threshold is too high for agent quality
3. Check for deadlocks via `DeadlockDetector`

**Recovery**:

- Lower `threshold` temporarily
- Increase `max_rounds`
- Add more agents for broader perspective

### False Commits

**Symptoms**: Swarm commits to incorrect answers.

**Diagnosis**:

1. Run `ForensicsAnalyzer` on failed rounds
2. Check `AgentCalibration` — overconfident agents cause false commits
3. Look for collusion via `ForensicsAnalyzer.detect_collusion()`

**Recovery**:

- Raise `threshold`
- Enable `ImmuneSystem` for adversarial detection
- Increase `slash_factor` (punish failures less, since aggressive slashing may be silencing correct but minority agents)

### Agent Cascade Failure

**Symptoms**: Multiple agents quarantined simultaneously, swarm degrades.

**Diagnosis**:

1. Check `autopilot.dashboard()` for quarantine events
2. Review `CascadeAnalyzer` for failure propagation

**Recovery**:

- Lower `quarantine_floor` temporarily
- Add reserve agents
- Review and fix the root cause (often a shared upstream dependency)

---

## Performance Tuning

### Reduce Latency

1. **Parallelize verification**: mBFT verification is embarrassingly parallel — use async agents
2. **Cache proposals**: If agents produce similar proposals, enable `PromptCache`-style deduplication
3. **Lower `max_rounds`**: Accept faster failure over patient retry

### Reduce Cost

1. **Smaller swarms**: Fewer agents = fewer LLM calls per round
2. **Fast-path commit**: If the first round commits, skip remaining rounds (default behavior)
3. **Confidence gating**: Skip verification for high-confidence proposals (trade safety for speed)
4. **Model tiering**: Use cheaper models for followers, expensive models for leaders

### Improve Accuracy

1. **Diverse agents**: Use different LLM providers/models for different agents
2. **Higher threshold**: Require stronger agreement
3. **Bayesian updates**: Enable `BayesianMBFTEngine` for posterior calibration
4. **Adversarial training**: Run `AdversarialMockAgent` scenarios to identify weaknesses
