# Paper

LaTeX source for *Consensus-Driven Metacognition in Multi-Agent Systems:
A Logic-Based Byzantine Fault-Tolerant Protocol.*

## Build

```powershell
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Files

- `main.tex` — IEEEtran-formatted manuscript with the full mBFT formalism,
  Algorithm 1, and Propositions 1–4 (each encoded as a property test in
  `../tests/test_propositions.py`).
- `refs.bib` — bibliography (Paxos, Raft, PBFT, HotStuff).

The propositions in §IV are mechanically verified by the companion test
suite; running `pytest -q` from the repo root re-checks all theorems.
