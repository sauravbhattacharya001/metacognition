# Agent Calibration

A consensus is only as trustworthy as the confidence values the agents bring to
the table. If every agent claims `confidence=0.99` regardless of whether it is
correct, the protocol degenerates to majority vote — and the *metacognitive*
property mBFT is supposed to provide quietly disappears.

The **Agent Calibration Benchmarker** (`src/calibrator.py`) measures, scores,
and diagnoses how well each agent's reported confidence (`τᵢ`) corresponds to
its actual probability of being correct. It is the single most important
diagnostic tool in this repo for tuning a swarm before deployment.

---

## Why calibration matters in mBFT

In mBFT, votes are weighted by trust × confidence. A poorly calibrated agent
distorts the aggregate `Σ wᵢ · τᵢ` in two pathological ways:

| Failure mode | Symptom in production |
|--------------|------------------------|
| **Overconfidence** | Wrong answers pass the threshold θ because the bad proposer voted strongly for itself |
| **Underconfidence** | Correct answers fail to commit; consensus rate collapses; rounds repeat unnecessarily |
| **High noise (ECE)** | Threshold tuning becomes impossible — the same θ yields different commit rates on different days |

The benchmarker quantifies all three using standard reliability metrics:

- **ECE** — Expected Calibration Error (binned, weighted mean gap between confidence and accuracy)
- **MCE** — Maximum Calibration Error (worst-bin gap)
- **Brier score** — squared-error of probabilistic predictions
- **Reliability diagram** — confidence vs. accuracy per bin

Lower is better for all four.

---

## Quick start

```bash
# Default: 5 agents × 100 trials
python -m src.calibrator

# Larger swarm, more trials, deterministic seed
python -m src.calibrator --agents 9 --trials 500 --seed 1234

# Interactive HTML report with per-agent reliability diagrams
python -m src.calibrator --export html --output calibration.html

# Machine-readable JSON for dashboards / CI gates
python -m src.calibrator --export json --output calibration.json
```

A typical text report looks like this:

```
=================================================================
  mBFT AGENT CALIBRATION BENCHMARK
=================================================================
  Swarm: 5 agents x 100 trials
  Swarm ECE:            0.0612
  Swarm Brier:          0.1843
  Swarm Accuracy:       82.0%
  Consensus Accuracy:   91.0%
  Consensus Rate:       78.0%

  PER-AGENT CALIBRATION:
     Agent     ECE   Brier     Acc    Conf  Diagnosis
     --------------------------------------------------
        a1   0.041   0.120     90%    0.89  well-calibrated
        a2   0.182   0.291     74%    0.91  overconfident
        a3   0.094   0.178     85%    0.68  underconfident
        a4   0.243   0.348     61%    0.86  high-overconfident
        a5   0.071   0.184     81%    0.79  well-calibrated
```

---

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--agents`, `-n` | `5`     | Swarm size. Each agent is assigned a profile (well-calibrated, overconfident, …) round-robin from a fixed catalog. |
| `--trials`, `-t` | `100`   | Number of independent consensus trials. More trials = tighter ECE bounds. |
| `--threshold`    | `1.5`   | Consensus threshold θ passed to `MBFTEngine`. |
| `--seed`         | `42`    | Master RNG seed. Per-agent seeds are derived from it so runs are reproducible. |
| `--diagnose`     | off     | Print extended per-agent diagnoses and recommendations. |
| `--export`       | —       | `json` or `html`. With no flag, the benchmarker prints a UTF-8 plain-text report to stdout. |
| `--output`, `-o` | —       | Destination file for `--export`. Defaults to stdout (json) or `calibration_report.html` (html). |

---

## Reading the output

### Per-agent diagnosis labels

| Label | Meaning | First thing to try |
|-------|---------|---------------------|
| `well-calibrated`       | ECE < 0.08, no gross over/underconfidence | Keep the current profile |
| `moderately calibrated` | 0.08 ≤ ECE < 0.15 | Light temperature scaling |
| `overconfident`         | mean confidence > accuracy + 15 pp | Lower base confidence, apply Platt scaling |
| `underconfident`        | mean confidence < accuracy − 15 pp | Raise base confidence, trust-rank boost |
| `poorly calibrated`     | ECE ≥ 0.15 | Re-train confidence head or replace agent |

### Swarm-level metrics

- **Swarm ECE / Brier** — aggregate calibration of *all* records pooled together.
  Useful as a single dashboard number.
- **Consensus accuracy** — among trials that *did* commit, fraction that committed the correct answer. This is the metric you care about in production.
- **Consensus rate** — fraction of trials that committed at all. Too low (< 50%) means θ is too high or the agents disagree too often.

### Recommendations

The benchmarker emits both per-agent and global recommendations. They are
deterministic functions of the metrics — there is no LLM in the loop — so
they're safe to gate CI on. Treat any line beginning with `🔴` or `🚨` as a
hard failure for production swarms.

---

## Using calibration in CI

A useful pattern is to fail CI if swarm calibration regresses:

```yaml
- name: Calibration gate
  run: |
    python -m src.calibrator --export json --output cal.json --seed 0
    python - <<'PY'
    import json, sys
    data = json.load(open("cal.json"))
    if data["swarm_ece"] > 0.12:
        sys.exit(f"❌ Swarm ECE {data['swarm_ece']} above 0.12 budget")
    if data["consensus_accuracy"] < 0.80:
        sys.exit(f"❌ Consensus accuracy {data['consensus_accuracy']} below 0.80")
    print("✅ Calibration within budget")
    PY
```

Because the benchmarker uses a fixed seed and synthetic ground truths, the
output is reproducible across machines.

---

## Implementation notes

- **Synthetic ground truths.** Each trial uses a fresh `f"answer-{trial}"`
  string as the correct answer. Agents either return it (with their assigned
  `correct_rate`) or a randomly-tagged wrong answer.
- **Agent profiles.** Seven calibration archetypes (`well-calibrated`,
  `overconfident`, `underconfident`, `high-overconfident`, `noisy`,
  `slightly-biased`, `expert-cautious`) cycle through the swarm. Swarm sizes
  beyond 7 simply reuse profiles round-robin.
- **Binning.** Confidences are bucketed into 10 equal-width bins in `[0, 1]`
  in a single O(N+B) pass. The previous quadratic implementation was rewritten
  in 4146.
- **Determinism.** All randomness flows from `--seed`; each agent gets a
  derived seed so adding agents doesn't perturb earlier ones.

---

## See also

- [`src/calibrator.py`](https://github.com/sauravbhattacharya001/metacognition/blob/main/src/calibrator.py) — implementation
- [Protocol](protocol.md) — how confidence enters the vote weight
- [Tuning & Operations](tuning.md) — choosing θ, slash factor, max rounds
- [`tests/test_calibrator.py`](https://github.com/sauravbhattacharya001/metacognition/tree/main/tests) — calibration math unit tests (if present)
