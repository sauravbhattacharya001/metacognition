"""Autonomous Agent Calibration Benchmarker.

Measures how well agent confidence (τ_i) correlates with actual correctness
by running controlled experiments with known ground truths. Produces
calibration curves, Expected Calibration Error (ECE), Brier scores, and
per-agent recommendations for improving reliability.

Usage::

    python -m src.calibrator                    # default benchmark
    python -m src.calibrator --agents 9         # custom swarm size
    python -m src.calibrator --trials 200       # more trials
    python -m src.calibrator --export html      # interactive HTML report
    python -m src.calibrator --export json      # JSON export
    python -m src.calibrator --diagnose         # per-agent diagnosis mode
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine


# ------------------------------------------------------------------ #
# Calibration-aware mock agent with tunable miscalibration
# ------------------------------------------------------------------ #

class CalibrationTestAgent(MockAgent):
    """Mock agent with configurable calibration profile.

    ``bias`` shifts confidence up/down from ground truth.
    ``noise`` adds random jitter to confidence.
    ``overconfidence_rate`` is the fraction of incorrect answers
    where the agent still reports high confidence.
    """

    def __init__(
        self,
        agent_id: str,
        ground_truth: str,
        correct_rate: float = 0.8,
        bias: float = 0.0,
        noise: float = 0.05,
        overconfidence_rate: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._rng = rng or random.Random()
        self._ground_truth = ground_truth
        self._correct_rate = correct_rate
        self._bias = bias
        self._noise = noise
        self._overconfidence_rate = overconfidence_rate

        # Will be set per-trial
        self._trial_answer = ground_truth
        self._trial_confidence = 0.8

        super().__init__(
            agent_id=agent_id,
            answer=ground_truth,
            confidence=0.8,
        )

    def prepare_trial(self, ground_truth: str) -> Tuple[bool, float]:
        """Set up this agent for a single trial. Returns (is_correct, confidence)."""
        is_correct = self._rng.random() < self._correct_rate

        if is_correct:
            self.answer = ground_truth
            raw_conf = self._correct_rate + self._bias
        else:
            self.answer = f"wrong-{self.id}-{self._rng.randint(0,999)}"
            if self._rng.random() < self._overconfidence_rate:
                # Overconfident on wrong answer
                raw_conf = 0.85 + self._bias
            else:
                raw_conf = (1.0 - self._correct_rate) + self._bias

        raw_conf += self._rng.gauss(0, self._noise)
        self.confidence = max(0.01, min(0.99, raw_conf))
        self._trial_confidence = self.confidence
        self._trial_answer = self.answer

        # Update accept_set for voting
        self.accept_set = {self.answer}

        return is_correct, self.confidence


# ------------------------------------------------------------------ #
# Calibration metrics
# ------------------------------------------------------------------ #

@dataclass
class TrialRecord:
    agent_id: str
    confidence: float
    is_correct: bool
    answer: str
    ground_truth: str


@dataclass
class CalibrationBin:
    bin_start: float
    bin_end: float
    mean_confidence: float
    accuracy: float
    count: int


@dataclass
class AgentCalibration:
    agent_id: str
    bins: List[CalibrationBin]
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier: float  # Brier score
    accuracy: float
    mean_confidence: float
    overconfidence_ratio: float  # fraction of trials where conf > accuracy
    underconfidence_ratio: float
    diagnosis: str
    recommendations: List[str]


@dataclass
class CalibrationReport:
    total_trials: int
    swarm_size: int
    agents: List[AgentCalibration]
    swarm_ece: float
    swarm_brier: float
    swarm_accuracy: float
    consensus_accuracy: float  # accuracy when consensus is reached
    consensus_rate: float  # fraction of trials reaching consensus
    best_calibrated: str
    worst_calibrated: str
    recommendations: List[str]


def compute_calibration(
    records: List[TrialRecord], n_bins: int = 10
) -> Tuple[List[CalibrationBin], float, float, float]:
    """Compute binned calibration, ECE, MCE, and Brier score."""
    bins: List[CalibrationBin] = []
    bin_width = 1.0 / n_bins

    total_ece = 0.0
    max_ce = 0.0
    total_brier = 0.0
    total = len(records)

    for i in range(n_bins):
        lo = i * bin_width
        hi = lo + bin_width
        in_bin = [r for r in records if lo <= r.confidence < hi or (i == n_bins - 1 and r.confidence == 1.0 and lo <= r.confidence <= hi)]

        if not in_bin:
            bins.append(CalibrationBin(lo, hi, (lo + hi) / 2, 0.0, 0))
            continue

        mean_conf = statistics.mean(r.confidence for r in in_bin)
        acc = sum(1 for r in in_bin if r.is_correct) / len(in_bin)
        ce = abs(acc - mean_conf)

        bins.append(CalibrationBin(lo, hi, mean_conf, acc, len(in_bin)))
        total_ece += ce * len(in_bin)
        max_ce = max(max_ce, ce)

    for r in records:
        total_brier += (r.confidence - (1.0 if r.is_correct else 0.0)) ** 2

    ece = total_ece / total if total > 0 else 0.0
    brier = total_brier / total if total > 0 else 0.0

    return bins, ece, max_ce, brier


def diagnose_agent(
    agent_id: str, records: List[TrialRecord], bins: List[CalibrationBin],
    ece: float, mce: float, brier: float,
) -> AgentCalibration:
    """Produce per-agent diagnosis and recommendations."""
    accuracy = sum(1 for r in records if r.is_correct) / len(records) if records else 0
    mean_conf = statistics.mean(r.confidence for r in records) if records else 0

    # Count overconfident trials (confidence > bin accuracy)
    over = sum(1 for r in records if r.confidence > accuracy + 0.1)
    under = sum(1 for r in records if r.confidence < accuracy - 0.1)
    over_ratio = over / len(records) if records else 0
    under_ratio = under / len(records) if records else 0

    recs: List[str] = []
    diagnosis = "well-calibrated"

    if ece > 0.15:
        diagnosis = "poorly calibrated"
        recs.append(f"ECE={ece:.3f} is high. Agent needs calibration tuning.")
    elif ece > 0.08:
        diagnosis = "moderately calibrated"
        recs.append(f"ECE={ece:.3f} is moderate. Minor calibration adjustments recommended.")

    if mean_conf > accuracy + 0.15:
        diagnosis = "overconfident"
        recs.append(
            f"Mean confidence ({mean_conf:.2f}) exceeds accuracy ({accuracy:.2f}) "
            "by >15pp. Apply temperature scaling or Platt calibration to reduce confidence."
        )
    elif mean_conf < accuracy - 0.15:
        diagnosis = "underconfident"
        recs.append(
            f"Mean confidence ({mean_conf:.2f}) is below accuracy ({accuracy:.2f}) "
            "by >15pp. Agent is too cautious — increase confidence or trust scores."
        )

    if mce > 0.25:
        recs.append(
            f"MCE={mce:.3f}: worst bin is severely miscalibrated. "
            "Check confidence distribution in the extreme ranges."
        )

    if brier > 0.3:
        recs.append(
            f"Brier score {brier:.3f} is high. Both accuracy and calibration "
            "need improvement."
        )

    if over_ratio > 0.4:
        recs.append(
            f"{over_ratio:.0%} of trials are overconfident. Consider adding "
            "self-doubt mechanisms or epistemic uncertainty estimation."
        )

    if not recs:
        recs.append("Agent is well-calibrated. No action needed.")

    return AgentCalibration(
        agent_id=agent_id,
        bins=bins,
        ece=ece,
        mce=mce,
        brier=brier,
        accuracy=accuracy,
        mean_confidence=mean_conf,
        overconfidence_ratio=over_ratio,
        underconfidence_ratio=under_ratio,
        diagnosis=diagnosis,
        recommendations=recs,
    )


# ------------------------------------------------------------------ #
# Benchmark runner
# ------------------------------------------------------------------ #

AGENT_PROFILES = [
    # (correct_rate, bias, noise, overconf_rate, label)
    (0.90, 0.0, 0.05, 0.0, "well-calibrated"),
    (0.75, 0.15, 0.05, 0.3, "overconfident"),
    (0.85, -0.20, 0.08, 0.0, "underconfident"),
    (0.60, 0.25, 0.10, 0.5, "high-overconfident"),
    (0.80, 0.0, 0.15, 0.1, "noisy"),
    (0.70, 0.05, 0.03, 0.1, "slightly-biased"),
    (0.95, -0.05, 0.02, 0.0, "expert-cautious"),
]


async def run_benchmark(
    swarm_size: int = 5,
    n_trials: int = 100,
    threshold: float = 1.5,
    seed: int = 42,
) -> CalibrationReport:
    rng = random.Random(seed)
    profiles = [AGENT_PROFILES[i % len(AGENT_PROFILES)] for i in range(swarm_size)]

    agents: List[CalibrationTestAgent] = []
    for i, (cr, bias, noise, oc, _label) in enumerate(profiles):
        agents.append(CalibrationTestAgent(
            agent_id=f"a{i+1}",
            ground_truth="correct",
            correct_rate=cr,
            bias=bias,
            noise=noise,
            overconfidence_rate=oc,
            rng=random.Random(rng.randint(0, 2**32)),
        ))

    all_records: Dict[str, List[TrialRecord]] = {a.id: [] for a in agents}
    consensus_correct = 0
    consensus_total = 0

    for trial in range(n_trials):
        ground_truth = f"answer-{trial}"

        # Prepare each agent for this trial
        for agent in agents:
            is_correct, conf = agent.prepare_trial(ground_truth)
            all_records[agent.id].append(TrialRecord(
                agent_id=agent.id,
                confidence=conf,
                is_correct=is_correct,
                answer=agent.answer,
                ground_truth=ground_truth,
            ))

        # Run consensus
        engine = MBFTEngine(agents=list(agents), threshold=threshold, max_rounds=4)
        result = await engine.run(f"trial-{trial}")
        if result and result.committed:
            consensus_total += 1
            if result.committed_solution == ground_truth:
                consensus_correct += 1

    # Compute per-agent calibration
    agent_cals: List[AgentCalibration] = []
    for agent in agents:
        recs = all_records[agent.id]
        bins, ece, mce, brier = compute_calibration(recs)
        cal = diagnose_agent(agent.id, recs, bins, ece, mce, brier)
        agent_cals.append(cal)

    # Swarm-level metrics
    all_recs = [r for recs in all_records.values() for r in recs]
    _, swarm_ece, _, swarm_brier = compute_calibration(all_recs)
    swarm_acc = sum(1 for r in all_recs if r.is_correct) / len(all_recs) if all_recs else 0

    best = min(agent_cals, key=lambda a: a.ece)
    worst = max(agent_cals, key=lambda a: a.ece)

    consensus_acc = consensus_correct / consensus_total if consensus_total > 0 else 0
    consensus_rate = consensus_total / n_trials if n_trials > 0 else 0

    # Global recommendations
    global_recs: List[str] = []
    overconf_agents = [a for a in agent_cals if a.diagnosis == "overconfident" or a.diagnosis == "high-overconfident"]
    if overconf_agents:
        global_recs.append(
            f"⚠️ {len(overconf_agents)} agent(s) are overconfident: "
            f"{', '.join(a.agent_id for a in overconf_agents)}. "
            "This can corrupt consensus by inflating bad proposals."
        )

    if swarm_ece > 0.12:
        global_recs.append(
            f"🔴 Swarm ECE={swarm_ece:.3f} is high. Collective calibration "
            "is poor — consensus decisions may be unreliable."
        )
    elif swarm_ece > 0.06:
        global_recs.append(
            f"🟡 Swarm ECE={swarm_ece:.3f} is moderate. Room for improvement."
        )
    else:
        global_recs.append(
            f"✅ Swarm ECE={swarm_ece:.3f} is good. Collective calibration is reliable."
        )

    if consensus_acc < 0.7:
        global_recs.append(
            f"🚨 Consensus accuracy is only {consensus_acc:.0%}. "
            "The swarm is committing to wrong answers too often."
        )

    if consensus_rate < 0.5:
        global_recs.append(
            f"📉 Consensus rate is only {consensus_rate:.0%}. "
            "Consider lowering threshold θ or improving agent agreement."
        )

    global_recs.append(
        f"💡 Best calibrated: {best.agent_id} (ECE={best.ece:.3f}). "
        f"Worst: {worst.agent_id} (ECE={worst.ece:.3f})."
    )

    return CalibrationReport(
        total_trials=n_trials,
        swarm_size=swarm_size,
        agents=agent_cals,
        swarm_ece=swarm_ece,
        swarm_brier=swarm_brier,
        swarm_accuracy=swarm_acc,
        consensus_accuracy=consensus_acc,
        consensus_rate=consensus_rate,
        best_calibrated=best.agent_id,
        worst_calibrated=worst.agent_id,
        recommendations=global_recs,
    )


# ------------------------------------------------------------------ #
# Output rendering
# ------------------------------------------------------------------ #

def _to_dict(report: CalibrationReport) -> dict:
    return {
        "total_trials": report.total_trials,
        "swarm_size": report.swarm_size,
        "swarm_ece": round(report.swarm_ece, 4),
        "swarm_brier": round(report.swarm_brier, 4),
        "swarm_accuracy": round(report.swarm_accuracy, 4),
        "consensus_accuracy": round(report.consensus_accuracy, 4),
        "consensus_rate": round(report.consensus_rate, 4),
        "best_calibrated": report.best_calibrated,
        "worst_calibrated": report.worst_calibrated,
        "recommendations": report.recommendations,
        "agents": [
            {
                "agent_id": a.agent_id,
                "ece": round(a.ece, 4),
                "mce": round(a.mce, 4),
                "brier": round(a.brier, 4),
                "accuracy": round(a.accuracy, 4),
                "mean_confidence": round(a.mean_confidence, 4),
                "overconfidence_ratio": round(a.overconfidence_ratio, 4),
                "diagnosis": a.diagnosis,
                "recommendations": a.recommendations,
                "bins": [
                    {
                        "range": f"{b.bin_start:.1f}-{b.bin_end:.1f}",
                        "mean_confidence": round(b.mean_confidence, 3),
                        "accuracy": round(b.accuracy, 3),
                        "count": b.count,
                    }
                    for b in a.bins
                ],
            }
            for a in report.agents
        ],
    }


def _render_html(report: CalibrationReport) -> str:
    data = _to_dict(report)
    agents_json = json.dumps(data["agents"])
    recs_html = "".join(f"<li>{r}</li>" for r in report.recommendations)

    agent_cards = ""
    for a in report.agents:
        color = "#3fb950" if a.ece < 0.08 else "#d29922" if a.ece < 0.15 else "#f85149"
        agent_recs = "".join(f"<li>{r}</li>" for r in a.recommendations)
        agent_cards += f"""
        <div class="agent-card">
          <h3>{a.agent_id} <span style="color:{color}">● {a.diagnosis}</span></h3>
          <div class="metrics">
            <span>ECE: <b>{a.ece:.3f}</b></span>
            <span>Brier: <b>{a.brier:.3f}</b></span>
            <span>Accuracy: <b>{a.accuracy:.0%}</b></span>
            <span>Confidence: <b>{a.mean_confidence:.2f}</b></span>
          </div>
          <canvas id="cal-{a.agent_id}" width="300" height="300"></canvas>
          <ul class="agent-recs">{agent_recs}</ul>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>mBFT Agent Calibration Report</title>
<style>
  :root {{ --bg:#0d1117; --fg:#c9d1d9; --accent:#58a6ff; --red:#f85149; --green:#3fb950; --yellow:#d29922; --card:#161b22; }}
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--fg);padding:2rem}}
  h1{{color:var(--accent);margin-bottom:.5rem}}
  h2{{color:var(--fg);margin:1.5rem 0 1rem}}
  .subtitle{{color:#8b949e;margin-bottom:2rem}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:2rem}}
  .card{{background:var(--card);border-radius:8px;padding:1.2rem}}
  .card h3{{font-size:.8rem;color:#8b949e;text-transform:uppercase}}
  .card .value{{font-size:1.8rem;font-weight:700;margin-top:.3rem}}
  .green{{color:var(--green)}} .red{{color:var(--red)}} .yellow{{color:var(--yellow)}}
  .recs{{background:var(--card);border-radius:8px;padding:1.5rem;margin-bottom:2rem}}
  .recs li{{margin:.5rem 0;line-height:1.5}}
  .agent-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1.5rem}}
  .agent-card{{background:var(--card);border-radius:8px;padding:1.5rem}}
  .agent-card h3{{margin-bottom:.8rem;font-size:1.1rem}}
  .metrics{{display:flex;flex-wrap:wrap;gap:.8rem;margin-bottom:1rem;font-size:.85rem;color:#8b949e}}
  .metrics b{{color:var(--fg)}}
  .agent-recs{{margin-top:1rem;font-size:.85rem;padding-left:1.2rem}}
  .agent-recs li{{margin:.3rem 0;color:#8b949e}}
  canvas{{display:block;margin:0 auto}}
</style></head><body>
<h1>🎯 mBFT Agent Calibration Report</h1>
<p class="subtitle">{report.swarm_size} agents &times; {report.total_trials} trials</p>

<div class="grid">
  <div class="card"><h3>Swarm ECE</h3>
    <div class="value {'green' if report.swarm_ece < 0.08 else 'yellow' if report.swarm_ece < 0.15 else 'red'}">{report.swarm_ece:.3f}</div></div>
  <div class="card"><h3>Swarm Brier</h3>
    <div class="value">{report.swarm_brier:.3f}</div></div>
  <div class="card"><h3>Swarm Accuracy</h3>
    <div class="value">{report.swarm_accuracy:.0%}</div></div>
  <div class="card"><h3>Consensus Accuracy</h3>
    <div class="value {'green' if report.consensus_accuracy >= 0.8 else 'yellow' if report.consensus_accuracy >= 0.6 else 'red'}">{report.consensus_accuracy:.0%}</div></div>
  <div class="card"><h3>Consensus Rate</h3>
    <div class="value">{report.consensus_rate:.0%}</div></div>
  <div class="card"><h3>Best Calibrated</h3>
    <div class="value green" style="font-size:1.2rem">{report.best_calibrated}</div></div>
</div>

<h2>📋 Recommendations</h2>
<div class="recs"><ul>{recs_html}</ul></div>

<h2>🔬 Per-Agent Calibration</h2>
<div class="agent-grid">{agent_cards}</div>

<script>
const agents = {agents_json};

// Draw reliability diagram for each agent
agents.forEach(agent => {{
  const canvas = document.getElementById('cal-' + agent.agent_id);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = {{l:45, r:15, t:15, b:40}};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  // Perfect calibration line
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + ph);
  ctx.lineTo(pad.l + pw, pad.t);
  ctx.stroke(); ctx.setLineDash([]);

  // Axes
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ph);
  ctx.lineTo(pad.l + pw, pad.t + ph); ctx.stroke();

  // Labels
  ctx.fillStyle = '#8b949e'; ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Mean Confidence', W/2, H - 3);
  ctx.save(); ctx.translate(10, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Accuracy', 0, 0); ctx.restore();

  // Tick marks
  for (let i = 0; i <= 10; i += 2) {{
    const v = i / 10;
    const x = pad.l + v * pw;
    const y = pad.t + ph - v * ph;
    ctx.fillStyle = '#8b949e'; ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(v.toFixed(1), x, pad.t + ph + 14);
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(1), pad.l - 5, y + 3);
  }}

  // Bars + points
  const bins = agent.bins.filter(b => b.count > 0);
  const maxCount = Math.max(...bins.map(b => b.count), 1);

  bins.forEach(bin => {{
    const x = pad.l + bin.mean_confidence * pw;
    const y = pad.t + ph - bin.accuracy * ph;

    // Gap bar (background)
    const barW = pw / 12;
    const barH = Math.max(2, (bin.count / maxCount) * 20);
    ctx.fillStyle = 'rgba(88,166,255,0.15)';
    ctx.fillRect(x - barW/2, pad.t + ph - barH, barW, barH);

    // Calibration point
    const gap = Math.abs(bin.accuracy - bin.mean_confidence);
    const r = Math.min(8, Math.max(3, bin.count / maxCount * 8));
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = gap < 0.1 ? '#3fb950' : gap < 0.2 ? '#d29922' : '#f85149';
    ctx.fill();
  }});

  // Title
  ctx.fillStyle = '#8b949e'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText('Reliability Diagram', pad.l, pad.t + 10);
}});
</script></body></html>"""


def _print_report(report: CalibrationReport) -> None:
    import io
    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    p = lambda *a, **kw: print(*a, **kw, file=out)

    p("=" * 65)
    p("  mBFT AGENT CALIBRATION BENCHMARK")
    p("=" * 65)
    p(f"  Swarm: {report.swarm_size} agents x {report.total_trials} trials")
    p(f"  Swarm ECE:            {report.swarm_ece:.4f}")
    p(f"  Swarm Brier:          {report.swarm_brier:.4f}")
    p(f"  Swarm Accuracy:       {report.swarm_accuracy:.1%}")
    p(f"  Consensus Accuracy:   {report.consensus_accuracy:.1%}")
    p(f"  Consensus Rate:       {report.consensus_rate:.1%}")
    p()

    p("  PER-AGENT CALIBRATION:")
    p(f"  {'Agent':>8}  {'ECE':>6}  {'Brier':>6}  {'Acc':>6}  {'Conf':>6}  {'Diagnosis'}")
    p("  " + "-" * 58)
    for a in report.agents:
        p(f"  {a.agent_id:>8}  {a.ece:>6.3f}  {a.brier:>6.3f}  "
          f"{a.accuracy:>5.0%}  {a.mean_confidence:>6.2f}  {a.diagnosis}")

    p()
    p("  RECOMMENDATIONS:")
    for r in report.recommendations:
        clean = r.encode("ascii", "ignore").decode("ascii").strip()
        p(f"    {clean}")

    p()
    p("  PER-AGENT RECOMMENDATIONS:")
    for a in report.agents:
        p(f"    [{a.agent_id}]")
        for r in a.recommendations:
            clean = r.encode("ascii", "ignore").decode("ascii").strip()
            p(f"      - {clean}")

    p("=" * 65)
    out.flush()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Agent Calibration Benchmarker"
    )
    parser.add_argument(
        "--agents", "-n", type=int, default=5,
        help="Number of agents in the swarm (default: 5)",
    )
    parser.add_argument(
        "--trials", "-t", type=int, default=100,
        help="Number of benchmark trials (default: 100)",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.5,
        help="Consensus threshold theta (default: 1.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Show detailed per-agent diagnosis",
    )
    parser.add_argument(
        "--export", choices=["json", "html"],
        help="Export report as JSON or interactive HTML",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output file path",
    )
    args = parser.parse_args()

    report = await run_benchmark(
        swarm_size=args.agents,
        n_trials=args.trials,
        threshold=args.threshold,
        seed=args.seed,
    )

    if args.export == "json":
        data = json.dumps(_to_dict(report), indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(data)
            print(f"Report written to {args.output}")
        else:
            print(data)
    elif args.export == "html":
        html = _render_html(report)
        out_path = args.output or "calibration_report.html"
        with open(out_path, "w") as f:
            f.write(html)
        print(f"Interactive report written to {out_path}")
    else:
        _print_report(report)


if __name__ == "__main__":
    asyncio.run(main())
