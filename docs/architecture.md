# Architecture

## Project Structure

```
metacognition/
├── src/
│   ├── core/                    # Protocol engine
│   │   ├── protocol.py          # Base mBFT consensus protocol
│   │   ├── protocol_bayesian.py # Bayesian belief-update extension
│   │   └── state.py             # Round state, proposal, vote models
│   ├── agents/                  # Agent implementations
│   │   ├── base.py              # Abstract agent interface
│   │   └── metacognitive.py     # Metacognitive agent with confidence tracking
│   ├── network/                 # Communication layer
│   │   └── simulator.py         # Asyncio network simulator with fault injection
│   ├── calibrator.py            # Calibration curve tracking and analysis
│   ├── monitor.py               # Consensus resilience monitoring
│   ├── partition.py             # Network partition simulation
│   ├── replay.py                # Round replay and debugging
│   └── trust_tracker.py         # Per-agent trust score evolution
├── tests/                       # Test suite with Byzantine fault injection
├── paper/                       # LaTeX source for the research paper
├── requirements.txt
└── pyproject.toml
```

## Core Components

### Protocol Engine (`src/core/`)

The heart of mBFT. `protocol.py` implements the four-phase consensus loop. `protocol_bayesian.py` adds posterior confidence updates. `state.py` defines the data models for rounds, proposals, and votes.

### Agents (`src/agents/`)

`base.py` defines the abstract agent interface (propose, verify, vote). `metacognitive.py` implements the full metacognitive agent with confidence self-assessment, counter-proof generation, and belief revision.

### Network (`src/network/`)

`simulator.py` provides an asyncio-based network simulator that supports configurable latency, message loss, reordering, and Byzantine behavior injection.

### Monitoring & Analysis

- **`monitor.py`** — Real-time consensus health metrics (liveness, agreement rate, round duration)
- **`calibrator.py`** — Tracks whether agents' confidence scores match their actual accuracy
- **`trust_tracker.py`** — Evolves trust scores based on voting history
- **`partition.py`** — Simulates network partitions for resilience testing
- **`replay.py`** — Records and replays rounds for debugging and analysis
