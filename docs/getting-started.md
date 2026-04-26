# Getting Started

## Prerequisites

- Python 3.10+
- pip

## Installation

```bash
# Clone the repository
git clone https://github.com/sauravbhattacharya001/metacognition.git
cd metacognition

# Create a virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Demo

The default demo uses `MockAgent`, so no API key is required:

```bash
python -m src.network.simulator
```

This runs a multi-agent consensus simulation with configurable fault injection.

## Running Tests

```bash
pytest -q
```

The test suite includes Byzantine fault-injection scenarios to verify protocol safety and liveness.

## Using Your Own LLM

To use a real LLM backend, implement the `BaseAgent` interface:

```python
from src.agents.base import BaseAgent
from src.core.state import Proposal, Vote

class MyLLMAgent(BaseAgent):
    async def generate_proposal(self, task: str) -> Proposal:
        # Call your LLM API here
        response = await my_llm.complete(task)
        return Proposal(
            agent_id=self.id,
            solution=response.text,
            proof=response.reasoning,
            confidence=response.confidence,
        )

    async def verify_proposal(self, leader_proposal: Proposal) -> Vote:
        # Evaluate the leader's proposal
        evaluation = await my_llm.evaluate(leader_proposal)
        return Vote(
            voter_id=self.id,
            value=evaluation.score,
            counter_proof=evaluation.counter_proof,
        )
```

Then pass your agents to the engine:

```python
from src.core.protocol import MBFTEngine

engine = MBFTEngine(
    agents=[MyLLMAgent("agent-1"), MyLLMAgent("agent-2"), MyLLMAgent("agent-3")],
    threshold=0.6,
)
result = await engine.run("Solve this problem...")
```
