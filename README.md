# mBFT — Metacognitive Byzantine Fault Tolerance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**What if AI agents could know when they don't know?**

mBFT is a consensus protocol for multi-agent AI systems that treats confident hallucinations as what they really are: **Byzantine faults.** Instead of trusting any single LLM's output, mBFT makes agents debate, verify each other's reasoning, and only commit when the group reaches genuine agreement.

> Think of it as "peer review for AI agents" — no answer gets through unless the swarm can back it up.

---

## The Problem

A single LLM can't tell the difference between *knowing* something, *inferring* something, and *making something up*. It has no epistemic self-awareness. Ask it a question and it answers with the same confidence whether it's right or hallucinating.

Classical consensus protocols (Paxos, Raft) don't help — they assume a responding node tells the truth. But an LLM that confidently returns a wrong answer **is** a Byzantine fault, not a crash.

## The Idea

Put multiple agents in a room. Make them propose answers *with proofs*. Make them verify each other's proofs. Weight votes by calibrated confidence. Require consensus to commit.

The result: **metacognition emerges from the protocol itself.** Disagreement becomes the system's version of doubt. Agreement becomes a real confidence signal. The "observer" isn't inside any single model — it lives in the space between them.

## How It Works

Each consensus round has three phases:

```
┌─────────────────────────────────────────────────────┐
│  1. PROPOSE  — Every agent generates a solution     │
│                with a proof and confidence score     │
│                                                     │
│  2. VERIFY   — Agents review the leader's proof     │
│                and vote: agree (+weight) or          │
│                reject with counter-proof (-weight)   │
│                                                     │
│  3. COMMIT   — Solution commits only if:            │
│                • Total weighted votes ≥ threshold    │
│                • No unrefuted rejections             │
│                                                     │
│  On failure → leader's reputation is slashed,       │
│               strongest dissenter leads next round   │
└─────────────────────────────────────────────────────┘
```

**Key properties:**
- **Confidence-weighted voting** — not all agents count equally; calibrated confidence matters
- **Counter-proofs required** — you can't just say "no," you have to show why
- **Reputation tracking** — agents that propose bad solutions lose influence over time
- **Byzantine tolerance** — adversarial or hallucinating agents get naturally filtered out

## Quick Start

**No API keys needed** — the demo runs with mock agents out of the box.

```bash
# Clone and set up
git clone https://github.com/sauravbhattacharya001/metacognition.git
cd metacognition
python -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\Activate.ps1

# Install and run
pip install -r requirements.txt
python -m src.network.simulator
```

**Expected output:**
```
============================================================
COMMITTED: '42'
  leader: a1
  Σ V_i: 2.270 >= θ=1.5
reputation after run: {'a1': 1.0, 'a2': 1.0, 'a3': 1.0, 'a4': 1.0, 'a5': 1.0}
rounds executed: 1
```

The demo swarm has 5 agents — 3 honest agents agreeing on "42", 1 dissenter, and 1 Byzantine (adversarial) agent. The protocol reaches consensus on the correct answer despite the noise.

**Run the tests:**
```bash
pytest -q
```

## Using Real LLMs

To plug in an actual language model, implement the `LLMClient` protocol and pass it to `MetacognitiveAgent`:

```python
from src.agents.metacognitive import MetacognitiveAgent
from src.core.protocol import MBFTEngine

class MyLLMClient:
    async def complete(self, prompt: str) -> str:
        # Call your LLM (OpenAI, Anthropic, local model, etc.)
        # Must return JSON with: solution, proof, confidence
        ...

agents = [
    MetacognitiveAgent("agent-1", llm=MyLLMClient()),
    MetacognitiveAgent("agent-2", llm=MyLLMClient()),
    MetacognitiveAgent("agent-3", llm=MyLLMClient()),
]

engine = MBFTEngine(agents=agents, threshold=1.5)
result = await engine.run("What caused the 2008 financial crisis?")
```

**Tip:** Use diverse models (e.g., mix GPT-4, Claude, Gemini) — homogeneous agents may agree on the same mistakes, defeating the purpose.

## Consensus Autopilot

The **Autopilot** is an autonomous swarm governor that processes task queues through continuous consensus rounds while self-tuning the protocol:

- **Adaptive threshold** — automatically raises θ after successful commits (tightening safety) and lowers it after consecutive stalls (restoring liveness)
- **Agent quarantine** — benches agents whose reputation drops below a configurable floor, reinstating them on probation after a cooldown period
- **Health monitoring** — tracks commit rate, average rounds, stall streaks, quarantine events, and threshold adjustments
- **Pluggable task source** — accepts task files, lists, or any async iterator

```bash
# Run the demo (10 tasks through a 7-agent swarm)
python -m src.autopilot

# Larger swarm, more cycles
python -m src.autopilot --agents 9 --cycles 20

# Custom task file (one task per line)
python -m src.autopilot --tasks my_tasks.txt

# Export JSON report or interactive HTML dashboard
python -m src.autopilot --export json -o report.json
python -m src.autopilot --export html -o dashboard.html

# Tune quarantine sensitivity
python -m src.autopilot --quarantine-floor 0.4 --quarantine-cooldown 60
```

**Programmatic usage:**

```python
from src.autopilot import ConsensusAutopilot, tasks_from_list
from src.agents.metacognitive import MockAgent

agents = [
    MockAgent("a1", answer="correct", confidence=0.9),
    MockAgent("a2", answer="correct", confidence=0.8),
    MockAgent("a3", answer="wrong", confidence=0.5),
    MockAgent("a4", answer="byz", confidence=0.95, byzantine=True),
]

pilot = ConsensusAutopilot(agents, initial_threshold=1.5)
tasks = tasks_from_list(["task 1", "task 2", "task 3"])
health = await pilot.run_queue(tasks)
print(f"Commit rate: {health.commit_rate:.0%}")
print(pilot.status_summary())
```

The Autopilot embodies the "agency" direction for mBFT — it doesn't just run consensus, it *governs* the swarm autonomously.

## Project Structure

```
metacognition/
├── src/
│   ├── core/               # Protocol engine + state models
│   │   ├── protocol.py     # MBFTEngine — the consensus loop
│   │   └── state.py        # Proposal, Vote, RoundResult
│   ├── agents/             # Agent implementations
│   │   ├── base.py         # Abstract BaseAgent contract
│   │   └── metacognitive.py # MockAgent + MetacognitiveAgent (LLM)
│   ├── network/            # Async simulator
│   ├── autopilot.py        # Consensus Autopilot — autonomous swarm governor
│   ├── adversarial_trainer.py  # Red-team testing for agent swarms
│   ├── calibrator.py       # Confidence calibration utilities
│   ├── diversity.py        # Model diversity analysis
│   ├── governance.py       # Governance policy enforcement
│   ├── trust_tracker.py    # Long-term agent reputation tracking
│   └── ...                 # Additional analysis modules
├── tests/                  # Consensus + Byzantine fault-injection tests
├── paper/                  # LaTeX source for the research paper
└── docs/                   # MkDocs documentation site
```

## Key Concepts

| Term | What it means |
|---|---|
| **Proposal** | An agent's answer + logical proof + confidence score (τ ∈ [0,1]) |
| **Vote** | A follower's weighted judgment on the leader's proposal (V ∈ [-1, 1]) |
| **Counter-proof** | A logical refutation — required to cast a negative vote |
| **Threshold (θ)** | Minimum aggregate weight needed to commit a solution |
| **Reputation** | Multiplicative weight per agent; slashed on failed leadership |
| **Byzantine agent** | An agent that lies, hallucinates, or acts adversarially |

## Research Paper

The accompanying paper — *"Consensus-Driven Metacognition in Multi-Agent Systems"* — formalizes the protocol and explores:

1. Can multi-agent disagreement approximate epistemic uncertainty?
2. What consensus protocol best supports safe autonomous agency?
3. Does model diversity improve the metacognitive signal?
4. Can consensus serve as governance for autonomous AI agents?

LaTeX source is in [`paper/`](paper/).

## Contributing

Contributions welcome! Some areas where help would be especially valuable:

- **Bayesian belief update integration** — replacing flat confidence with posterior updates
- **Benchmarks** — calibration tests across different LLM combinations
- **HotStuff-style pipelining** — optimizing view changes for lower latency
- **Real-world eval datasets** — testing mBFT on factual QA, reasoning, and code generation

## License

MIT — see [LICENSE](LICENSE).

---

*Built by [Saurav Bhattacharya](https://github.com/sauravbhattacharya001) as part of research into AI agent governance and safety.*
