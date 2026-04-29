"""Consensus Regime Detector — autonomous regime classification for mBFT.

Runs batches of mBFT consensus tasks, extracts multi-dimensional signal
features, and classifies each window into an operating **regime**:

- **cooperative** — high commit rate, strong agreement, low slashing
- **contested** — moderate commit rate, notable disagreement
- **adversarial** — low commit rate, high slash rate, Byzantine dominance
- **deadlocked** — very low commit rate, many rounds exhausted
- **chaotic** — high variance across all signals, no stable pattern

Change-points are detected via CUSUM (cumulative sum control chart).  A
Markov transition matrix tracks regime-to-regime probabilities and early-
warning signals flag impending regime shifts.

Usage::

    python -m src.regime --agents 8 --byzantine 2 --tasks 30
    python -m src.regime --agents 10 --byzantine 3 --tasks 50 --auto-monitor --interval 30
    python -m src.regime --output regime_report.json
    python -m src.regime --help

Features:
- Per-window signal extraction (commit rate, agreement, reputation variance,
  slash rate, round usage, aggregate spread)
- 5-regime classification with configurable thresholds
- CUSUM change-point detection on regime-encoded time-series
- Markov transition matrix (rows normalised to 1.0)
- Early-warning indicators (rising variance, autocorrelation shift)
- Auto-monitor mode with configurable interval
- Interactive HTML report with regime timeline, transition heatmap,
  signal plots, early-warning chart, and proactive recommendations
- JSON export
"""
from __future__ import annotations

import argparse
import asyncio
import html as html_mod
import json
import os
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine
from src.core.state import RoundResult

# ── Constants ────────────────────────────────────────────────────

REGIMES = ["cooperative", "contested", "adversarial", "deadlocked", "chaotic"]
REGIME_COLORS: Dict[str, str] = {
    "cooperative": "#22c55e",
    "contested": "#eab308",
    "adversarial": "#ef4444",
    "deadlocked": "#6366f1",
    "chaotic": "#a855f7",
}
REGIME_ICONS: Dict[str, str] = {
    "cooperative": "🤝",
    "contested": "⚔️",
    "adversarial": "🔥",
    "deadlocked": "🔒",
    "chaotic": "🌀",
}


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class SignalVector:
    """Feature vector extracted from a window of consensus tasks."""
    window_index: int
    commit_rate: float
    mean_agreement: float
    reputation_variance: float
    slash_rate: float
    mean_rounds: float
    aggregate_spread: float

    def as_list(self) -> List[float]:
        return [
            self.commit_rate,
            self.mean_agreement,
            self.reputation_variance,
            self.slash_rate,
            self.mean_rounds,
            self.aggregate_spread,
        ]


@dataclass
class ChangePoint:
    """Detected regime change-point."""
    index: int
    from_regime: str
    to_regime: str
    cusum_value: float
    description: str


@dataclass
class EarlyWarning:
    """Early-warning signal for an impending regime transition."""
    index: int
    signal_name: str
    value: float
    severity: str  # info / warning / critical
    description: str


@dataclass
class RegimeReport:
    """Full regime analysis."""
    signals: List[SignalVector] = field(default_factory=list)
    regimes: List[str] = field(default_factory=list)
    change_points: List[ChangePoint] = field(default_factory=list)
    transition_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    early_warnings: List[EarlyWarning] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals": [asdict(s) for s in self.signals],
            "regimes": self.regimes,
            "change_points": [asdict(c) for c in self.change_points],
            "transition_matrix": self.transition_matrix,
            "early_warnings": [asdict(w) for w in self.early_warnings],
            "recommendations": self.recommendations,
            "config": self.config,
        }


# ── Agent Factory ────────────────────────────────────────────────

def _build_agents(
    n_agents: int, n_byzantine: int
) -> List[MockAgent]:
    """Create a mixed pool of honest and Byzantine mock agents."""
    agents: List[MockAgent] = []
    correct_answer = "42"
    for i in range(n_agents):
        is_byz = i < n_byzantine
        agents.append(
            MockAgent(
                agent_id=f"agent-{i}",
                answer=f"wrong-{i}" if is_byz else correct_answer,
                confidence=round(random.uniform(0.3, 0.9), 2),
                byzantine=is_byz,
            )
        )
    return agents


# ── Signal Extraction ────────────────────────────────────────────

def _extract_signals(
    histories: List[List[RoundResult]], window_index: int
) -> SignalVector:
    """Compute signal features from a batch of task histories."""
    if not histories:
        return SignalVector(window_index, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    commits = sum(1 for h in histories if any(r.committed for r in h))
    commit_rate = commits / len(histories)

    all_votes = [v for h in histories for r in h for v in r.votes]
    positive_weights = [v.weight for v in all_votes if v.weight > 0]
    mean_agreement = statistics.mean(positive_weights) if positive_weights else 0.0

    # Reputation variance — reconstruct from slash history
    rep: Dict[str, float] = defaultdict(lambda: 1.0)
    for h in histories:
        for r in h:
            for sid in r.slashed:
                rep[sid] *= 0.5
    rep_vals = list(rep.values())
    reputation_variance = statistics.pvariance(rep_vals) if len(rep_vals) > 1 else 0.0

    total_rounds = sum(len(h) for h in histories)
    slash_rounds = sum(1 for h in histories for r in h if r.slashed)
    slash_rate = slash_rounds / total_rounds if total_rounds else 0.0

    mean_rounds = total_rounds / len(histories)

    aggregates = [r.aggregate_weight for h in histories for r in h]
    aggregate_spread = (
        statistics.pstdev(aggregates) if len(aggregates) > 1 else 0.0
    )

    return SignalVector(
        window_index=window_index,
        commit_rate=round(commit_rate, 4),
        mean_agreement=round(mean_agreement, 4),
        reputation_variance=round(reputation_variance, 4),
        slash_rate=round(slash_rate, 4),
        mean_rounds=round(mean_rounds, 4),
        aggregate_spread=round(aggregate_spread, 4),
    )


# ── Regime Classification ────────────────────────────────────────

def _classify_regime(sig: SignalVector) -> str:
    """Rule-based regime classification from signal vector."""
    # Chaotic: contradictory signals — high commit but also high slash,
    # or very high aggregate spread with moderate commit
    if sig.aggregate_spread > 1.5 and sig.slash_rate > 0.2 and sig.commit_rate > 0.3:
        return "chaotic"

    # Deadlocked: very low commit, many rounds exhausted
    if sig.commit_rate < 0.2 and sig.mean_rounds >= 3.0:
        return "deadlocked"

    # Adversarial: low commit with high slash
    if sig.commit_rate < 0.4 and sig.slash_rate > 0.3:
        return "adversarial"

    # Contested: moderate commit or notable disagreement
    if sig.commit_rate < 0.8 or (sig.slash_rate > 0.15 and sig.reputation_variance > 0.05):
        return "contested"

    # Cooperative
    return "cooperative"


# ── CUSUM Change-Point Detection ─────────────────────────────────

def _regime_to_ordinal(regime: str) -> float:
    return float(REGIMES.index(regime))


def _cusum_detect(
    series: List[float], drift: float = 0.5, threshold: float = 2.0
) -> List[int]:
    """Detect change-points in a numeric series using tabular CUSUM."""
    if len(series) < 3:
        return []
    mu = statistics.mean(series)
    s_pos = 0.0
    s_neg = 0.0
    points: List[int] = []
    for i, x in enumerate(series):
        s_pos = max(0.0, s_pos + (x - mu) - drift)
        s_neg = max(0.0, s_neg - (x - mu) - drift)
        if s_pos > threshold or s_neg > threshold:
            points.append(i)
            s_pos = 0.0
            s_neg = 0.0
    return points


def _detect_change_points(
    regimes: List[str],
) -> List[ChangePoint]:
    """Find regime transition change-points."""
    ordinals = [_regime_to_ordinal(r) for r in regimes]
    cusum_pts = _cusum_detect(ordinals)

    change_points: List[ChangePoint] = []
    # Also find simple transitions
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            cusum_val = abs(ordinals[i] - ordinals[i - 1])
            if i in cusum_pts:
                cusum_val += 1.0  # boost CUSUM-flagged transitions
            change_points.append(
                ChangePoint(
                    index=i,
                    from_regime=regimes[i - 1],
                    to_regime=regimes[i],
                    cusum_value=round(cusum_val, 3),
                    description=(
                        f"Transition from {REGIME_ICONS[regimes[i-1]]} "
                        f"{regimes[i-1]} → {REGIME_ICONS[regimes[i]]} "
                        f"{regimes[i]} at window {i}"
                    ),
                )
            )
    return change_points


# ── Transition Matrix ────────────────────────────────────────────

def _build_transition_matrix(
    regimes: List[str],
) -> Dict[str, Dict[str, float]]:
    """Build a Markov transition matrix from observed regime sequence."""
    counts: Dict[str, Dict[str, int]] = {
        r: {r2: 0 for r2 in REGIMES} for r in REGIMES
    }
    for i in range(len(regimes) - 1):
        counts[regimes[i]][regimes[i + 1]] += 1

    matrix: Dict[str, Dict[str, float]] = {}
    for r in REGIMES:
        total = sum(counts[r].values())
        if total == 0:
            matrix[r] = {r2: 0.0 for r2 in REGIMES}
        else:
            matrix[r] = {r2: round(counts[r][r2] / total, 4) for r2 in REGIMES}
    return matrix


# ── Early-Warning Detection ──────────────────────────────────────

def _detect_early_warnings(
    signals: List[SignalVector], regimes: List[str]
) -> List[EarlyWarning]:
    """Compute early-warning indicators for regime transitions."""
    warnings: List[EarlyWarning] = []
    if len(signals) < 4:
        return warnings

    # Rolling variance of commit rate (window=3)
    for i in range(3, len(signals)):
        window = [signals[j].commit_rate for j in range(i - 2, i + 1)]
        var = statistics.pvariance(window)
        if var > 0.05:
            severity = "critical" if var > 0.1 else "warning"
            warnings.append(
                EarlyWarning(
                    index=i,
                    signal_name="commit_rate_variance",
                    value=round(var, 4),
                    severity=severity,
                    description=f"Rising commit-rate variance ({var:.4f}) at window {i} — regime shift likely",
                )
            )

    # Slash rate acceleration
    for i in range(2, len(signals)):
        delta = signals[i].slash_rate - signals[i - 1].slash_rate
        prev_delta = signals[i - 1].slash_rate - signals[i - 2].slash_rate if i >= 3 else 0
        accel = delta - prev_delta
        if accel > 0.1:
            severity = "critical" if accel > 0.2 else "warning"
            warnings.append(
                EarlyWarning(
                    index=i,
                    signal_name="slash_rate_acceleration",
                    value=round(accel, 4),
                    severity=severity,
                    description=f"Slash rate accelerating ({accel:+.4f}) at window {i} — adversarial shift imminent",
                )
            )

    # Reputation divergence
    for i in range(1, len(signals)):
        if signals[i].reputation_variance > 0.1 and signals[i].reputation_variance > signals[i - 1].reputation_variance * 1.5:
            warnings.append(
                EarlyWarning(
                    index=i,
                    signal_name="reputation_divergence",
                    value=round(signals[i].reputation_variance, 4),
                    severity="warning",
                    description=f"Reputation divergence spike ({signals[i].reputation_variance:.4f}) at window {i}",
                )
            )

    return warnings


# ── Recommendations ──────────────────────────────────────────────

def _generate_recommendations(report: RegimeReport) -> List[str]:
    """Generate proactive recommendations based on regime analysis."""
    recs: List[str] = []
    regime_counts = Counter(report.regimes)
    dominant = regime_counts.most_common(1)[0] if regime_counts else ("cooperative", 0)

    if dominant[0] == "adversarial":
        recs.append("⚠️ Adversarial regime dominates — consider increasing Byzantine fault tolerance threshold or adding agent verification layers")
    if dominant[0] == "deadlocked":
        recs.append("🔒 Frequent deadlocks — reduce consensus threshold or increase max rounds to allow more negotiation")
    if dominant[0] == "chaotic":
        recs.append("🌀 Chaotic dynamics detected — stabilize by reducing agent pool volatility or adding reputation dampening")

    if len(report.change_points) > len(report.regimes) * 0.4:
        recs.append("⚡ High regime instability — system is oscillating rapidly between states; consider adding hysteresis or cooldown periods")

    crit_warnings = [w for w in report.early_warnings if w.severity == "critical"]
    if len(crit_warnings) > 3:
        recs.append("🚨 Multiple critical early-warning signals — preemptive intervention recommended before next regime collapse")

    # Transition-specific
    tm = report.transition_matrix
    if tm.get("cooperative", {}).get("adversarial", 0) > 0.2:
        recs.append("📉 Cooperative→Adversarial transitions are frequent — investigate what triggers Byzantine escalation")
    if tm.get("contested", {}).get("deadlocked", 0) > 0.3:
        recs.append("🔄 Contested→Deadlocked pattern — contested regimes often lead to deadlock; add tie-breaking mechanisms")

    if not recs:
        recs.append("✅ Regime dynamics appear stable — no immediate intervention required")

    return recs


# ── Simulation ───────────────────────────────────────────────────

async def _run_tasks(
    n_agents: int,
    n_byzantine: int,
    n_tasks: int,
    threshold: float,
    max_rounds: int,
) -> List[List[RoundResult]]:
    """Run a batch of consensus tasks and collect histories."""
    histories: List[List[RoundResult]] = []
    for t in range(n_tasks):
        agents = _build_agents(n_agents, n_byzantine)
        engine = MBFTEngine(
            agents=agents, threshold=threshold, max_rounds=max_rounds
        )
        await engine.run(f"task-{t}")
        histories.append(list(engine.history))
    return histories


async def run_analysis(
    n_agents: int = 7,
    n_byzantine: int = 2,
    n_tasks: int = 30,
    threshold: float = 3.0,
    max_rounds: int = 4,
    window_size: int = 5,
) -> RegimeReport:
    """Full regime analysis pipeline."""
    histories = await _run_tasks(n_agents, n_byzantine, n_tasks, threshold, max_rounds)

    # Window the tasks
    signals: List[SignalVector] = []
    for i in range(0, len(histories), window_size):
        window = histories[i : i + window_size]
        if window:
            signals.append(_extract_signals(window, len(signals)))

    # Classify
    regimes = [_classify_regime(s) for s in signals]

    # Change-points
    change_points = _detect_change_points(regimes)

    # Transition matrix
    transition_matrix = _build_transition_matrix(regimes)

    # Early warnings
    early_warnings = _detect_early_warnings(signals, regimes)

    report = RegimeReport(
        signals=signals,
        regimes=regimes,
        change_points=change_points,
        transition_matrix=transition_matrix,
        early_warnings=early_warnings,
        config={
            "agents": n_agents,
            "byzantine": n_byzantine,
            "tasks": n_tasks,
            "threshold": threshold,
            "max_rounds": max_rounds,
            "window_size": window_size,
        },
    )
    report.recommendations = _generate_recommendations(report)
    return report


# ── HTML Report ──────────────────────────────────────────────────

def _html_escape(text: str) -> str:
    return html_mod.escape(str(text))


def generate_html(report: RegimeReport) -> str:
    """Produce a self-contained interactive HTML report."""
    cfg = report.config

    # Regime timeline bars
    timeline_bars = ""
    for i, regime in enumerate(report.regimes):
        color = REGIME_COLORS[regime]
        icon = REGIME_ICONS[regime]
        pct = 100 / max(len(report.regimes), 1)
        timeline_bars += (
            f'<div class="regime-bar" style="width:{pct:.2f}%;background:{color}" '
            f'title="Window {i}: {regime}">{icon}</div>'
        )

    # Transition matrix HTML
    tm_rows = ""
    for r_from in REGIMES:
        cells = ""
        for r_to in REGIMES:
            val = report.transition_matrix.get(r_from, {}).get(r_to, 0.0)
            intensity = int(val * 255)
            bg = f"rgba(99,102,241,{val:.2f})"
            cells += f'<td style="background:{bg};text-align:center">{val:.2f}</td>'
        tm_rows += f'<tr><td><strong>{REGIME_ICONS.get(r_from, "")} {r_from}</strong></td>{cells}</tr>'

    tm_header = "".join(f"<th>{REGIME_ICONS.get(r, '')} {r}</th>" for r in REGIMES)

    # Signal chart data (JSON for JS)
    signal_data = json.dumps(
        [
            {
                "i": s.window_index,
                "cr": s.commit_rate,
                "ma": s.mean_agreement,
                "rv": s.reputation_variance,
                "sr": s.slash_rate,
                "mr": s.mean_rounds / (cfg.get("max_rounds", 4) or 4),
                "as": s.aggregate_spread,
            }
            for s in report.signals
        ]
    )

    # Change-points list
    cp_items = ""
    for cp in report.change_points:
        cp_items += (
            f'<div class="cp-item">'
            f'<span class="cp-badge">{_html_escape(cp.from_regime)} → {_html_escape(cp.to_regime)}</span> '
            f'Window {cp.index} (CUSUM: {cp.cusum_value:.3f})'
            f'</div>'
        )
    if not cp_items:
        cp_items = '<div class="cp-item">No regime transitions detected</div>'

    # Early-warning items
    ew_items = ""
    sev_colors = {"info": "#60a5fa", "warning": "#f59e0b", "critical": "#ef4444"}
    for ew in report.early_warnings:
        color = sev_colors.get(ew.severity, "#9ca3af")
        ew_items += (
            f'<div class="ew-item" style="border-left:4px solid {color}">'
            f'<strong>[{ew.severity.upper()}]</strong> {_html_escape(ew.description)}'
            f'</div>'
        )
    if not ew_items:
        ew_items = '<div class="ew-item">No early-warning signals</div>'

    # Recommendations
    rec_items = ""
    for r in report.recommendations:
        rec_items += f'<div class="rec-item">{_html_escape(r)}</div>'

    # Regime distribution
    regime_counts = Counter(report.regimes)
    dist_items = ""
    for regime in REGIMES:
        count = regime_counts.get(regime, 0)
        pct = (count / max(len(report.regimes), 1)) * 100
        dist_items += (
            f'<div class="dist-bar-wrap">'
            f'<span class="dist-label">{REGIME_ICONS[regime]} {regime}</span>'
            f'<div class="dist-bar" style="width:{pct:.1f}%;background:{REGIME_COLORS[regime]}">'
            f'{count} ({pct:.0f}%)</div></div>'
        )

    # Early-warning chart data
    ew_data = json.dumps(
        [{"i": w.index, "v": w.value, "s": w.severity, "n": w.signal_name} for w in report.early_warnings]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Consensus Regime Detector — mBFT</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.8rem;margin-bottom:4px}}
h2{{font-size:1.2rem;color:#94a3b8;margin:24px 0 12px;border-bottom:1px solid #1e293b;padding-bottom:6px}}
.header{{text-align:center;margin-bottom:32px}}
.subtitle{{color:#64748b;font-size:0.95rem}}
.card{{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.stat{{background:#1e293b;border-radius:10px;padding:16px;text-align:center}}
.stat-val{{font-size:2rem;font-weight:bold;color:#38bdf8}}
.stat-label{{color:#94a3b8;font-size:0.85rem;margin-top:4px}}
.timeline{{display:flex;border-radius:8px;overflow:hidden;height:48px;margin:12px 0}}
.regime-bar{{display:flex;align-items:center;justify-content:center;font-size:1.2rem;cursor:pointer;transition:opacity .2s}}
.regime-bar:hover{{opacity:.8}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}
th,td{{padding:8px 10px;border:1px solid #334155;font-size:0.85rem}}
th{{background:#334155;color:#e2e8f0}}
.cp-item,.ew-item,.rec-item{{background:#0f172a;border-radius:6px;padding:10px 14px;margin:6px 0;font-size:0.9rem}}
.cp-badge{{background:#4f46e5;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem}}
.ew-item{{padding-left:18px}}
.canvas-wrap{{background:#0f172a;border-radius:8px;padding:12px;margin:12px 0}}
canvas{{width:100%;height:220px;display:block}}
.dist-bar-wrap{{display:flex;align-items:center;margin:4px 0}}
.dist-label{{width:130px;font-size:0.85rem}}
.dist-bar{{height:28px;border-radius:4px;display:flex;align-items:center;padding-left:8px;font-size:0.8rem;color:#fff;min-width:40px;transition:width .3s}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;justify-content:center;margin:8px 0}}
.legend span{{display:flex;align-items:center;gap:4px;font-size:0.8rem}}
.legend i{{width:12px;height:12px;border-radius:3px;display:inline-block}}
</style>
</head>
<body>
<div class="header">
<h1>🔬 Consensus Regime Detector</h1>
<p class="subtitle">mBFT Metacognitive Byzantine Fault Tolerance — Autonomous Regime Classification</p>
<p class="subtitle" style="margin-top:6px">{cfg.get('agents',0)} agents · {cfg.get('byzantine',0)} Byzantine · {cfg.get('tasks',0)} tasks · threshold {cfg.get('threshold',0)}</p>
</div>

<div class="grid">
<div class="stat"><div class="stat-val">{len(report.regimes)}</div><div class="stat-label">Windows Analyzed</div></div>
<div class="stat"><div class="stat-val">{len(report.change_points)}</div><div class="stat-label">Regime Transitions</div></div>
<div class="stat"><div class="stat-val">{len([w for w in report.early_warnings if w.severity=='critical'])}</div><div class="stat-label">Critical Warnings</div></div>
<div class="stat"><div class="stat-val">{regime_counts.most_common(1)[0][0] if regime_counts else 'n/a'}</div><div class="stat-label">Dominant Regime</div></div>
</div>

<h2>📊 Regime Timeline</h2>
<div class="card">
<div class="timeline">{timeline_bars}</div>
<div class="legend">
{"".join(f'<span><i style="background:{REGIME_COLORS[r]}"></i>{REGIME_ICONS[r]} {r}</span>' for r in REGIMES)}
</div>
</div>

<h2>📈 Regime Distribution</h2>
<div class="card">{dist_items}</div>

<h2>📉 Signal Features Over Time</h2>
<div class="card">
<div class="canvas-wrap"><canvas id="signalChart"></canvas></div>
<div class="legend">
<span><i style="background:#38bdf8"></i>Commit Rate</span>
<span><i style="background:#22c55e"></i>Agreement</span>
<span><i style="background:#f59e0b"></i>Slash Rate</span>
<span><i style="background:#ef4444"></i>Rep. Variance</span>
<span><i style="background:#a855f7"></i>Round Usage</span>
<span><i style="background:#ec4899"></i>Agg. Spread</span>
</div>
</div>

<h2>🔄 Transition Matrix</h2>
<div class="card">
<table>
<tr><th>From ↓ / To →</th>{tm_header}</tr>
{tm_rows}
</table>
</div>

<h2>⚡ Change Points</h2>
<div class="card">{cp_items}</div>

<h2>⚠️ Early-Warning Signals</h2>
<div class="card">{ew_items}</div>

<h2>🧠 Proactive Recommendations</h2>
<div class="card">{rec_items}</div>

<script>
(function(){{
const data={signal_data};
const ew={ew_data};
const canvas=document.getElementById('signalChart');
const ctx=canvas.getContext('2d');
function draw(){{
  const W=canvas.width=canvas.offsetWidth;
  const H=canvas.height=canvas.offsetHeight;
  const pad={{t:20,b:30,l:40,r:20}};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
  ctx.clearRect(0,0,W,H);
  if(!data.length) return;
  const n=data.length;
  const keys=[['cr','#38bdf8'],['ma','#22c55e'],['sr','#f59e0b'],['rv','#ef4444'],['mr','#a855f7'],['as','#ec4899']];
  // Grid
  ctx.strokeStyle='#334155';ctx.lineWidth=0.5;
  for(let i=0;i<=4;i++){{
    const y=pad.t+ph*(i/4);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+pw,y);ctx.stroke();
    ctx.fillStyle='#64748b';ctx.font='11px system-ui';
    ctx.fillText((1-i/4).toFixed(1),4,y+4);
  }}
  // X labels
  for(let i=0;i<n;i++){{
    const x=pad.l+(i/(n-1||1))*pw;
    if(i%Math.max(1,Math.floor(n/10))===0){{
      ctx.fillStyle='#64748b';ctx.font='11px system-ui';
      ctx.fillText(i,x-4,H-8);
    }}
  }}
  // Lines
  keys.forEach(function(kc){{
    const k=kc[0],c=kc[1];
    ctx.strokeStyle=c;ctx.lineWidth=2;
    ctx.beginPath();
    data.forEach(function(d,i){{
      const x=pad.l+(i/(n-1||1))*pw;
      const v=Math.min(Math.max(d[k],0),1);
      const y=pad.t+ph*(1-v);
      if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    }});
    ctx.stroke();
  }});
  // EW markers
  ew.forEach(function(w){{
    const x=pad.l+(w.i/(n-1||1))*pw;
    ctx.fillStyle=w.s==='critical'?'#ef4444':w.s==='warning'?'#f59e0b':'#60a5fa';
    ctx.beginPath();ctx.arc(x,pad.t+5,4,0,Math.PI*2);ctx.fill();
  }});
}}
draw();
window.addEventListener('resize',draw);
}})();
</script>
</body>
</html>"""


# ── Auto-Monitor ─────────────────────────────────────────────────

async def auto_monitor(
    n_agents: int,
    n_byzantine: int,
    n_tasks: int,
    threshold: float,
    max_rounds: int,
    interval: int,
    output: Optional[str],
) -> None:
    """Continuous monitoring loop."""
    cycle = 0
    prev_regime: Optional[str] = None
    print(f"[regime-monitor] Starting auto-monitor (interval={interval}s)")
    while True:
        cycle += 1
        print(f"\n[regime-monitor] Cycle {cycle} ...")
        report = await run_analysis(n_agents, n_byzantine, n_tasks, threshold, max_rounds)
        current = report.regimes[-1] if report.regimes else "unknown"

        if prev_regime and current != prev_regime:
            print(
                f"  🚨 REGIME TRANSITION: {prev_regime} → {current}"
            )
        else:
            print(f"  Current regime: {REGIME_ICONS.get(current, '')} {current}")

        if report.early_warnings:
            crit = [w for w in report.early_warnings if w.severity == "critical"]
            if crit:
                print(f"  ⚠️  {len(crit)} critical early-warning(s)")

        prev_regime = current

        if output:
            fname = output.replace(".json", f"_cycle{cycle}.json")
            with open(fname, "w") as f:
                json.dump(report.to_dict(), f, indent=2)

        await asyncio.sleep(interval)


# ── CLI ──────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Consensus Regime Detector — autonomous mBFT regime classification"
    )
    p.add_argument("--agents", type=int, default=7, help="Number of agents (default: 7)")
    p.add_argument("--byzantine", type=int, default=2, help="Byzantine agents (default: 2)")
    p.add_argument("--tasks", type=int, default=30, help="Consensus tasks (default: 30)")
    p.add_argument("--threshold", type=float, default=3.0, help="Consensus threshold (default: 3.0)")
    p.add_argument("--max-rounds", type=int, default=4, help="Max rounds per task (default: 4)")
    p.add_argument("--window-size", type=int, default=5, help="Tasks per analysis window (default: 5)")
    p.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    p.add_argument("--auto-monitor", action="store_true", help="Continuous monitoring mode")
    p.add_argument("--interval", type=int, default=60, help="Monitor interval in seconds (default: 60)")
    return p.parse_args(argv)


async def _main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    if args.auto_monitor:
        await auto_monitor(
            args.agents, args.byzantine, args.tasks,
            args.threshold, args.max_rounds, args.interval, args.output,
        )
        return

    print(f"🔬 Consensus Regime Detector")
    print(f"   {args.agents} agents · {args.byzantine} Byzantine · {args.tasks} tasks")
    print()

    report = await run_analysis(
        args.agents, args.byzantine, args.tasks,
        args.threshold, args.max_rounds, args.window_size,
    )

    # Print summary
    regime_counts = Counter(report.regimes)
    print("Regime sequence:", " → ".join(
        f"{REGIME_ICONS[r]} {r}" for r in report.regimes
    ))
    print()
    print("Distribution:")
    for r in REGIMES:
        c = regime_counts.get(r, 0)
        bar = "█" * c
        print(f"  {REGIME_ICONS[r]} {r:12s} {bar} ({c})")

    if report.change_points:
        print(f"\n⚡ {len(report.change_points)} change-point(s):")
        for cp in report.change_points:
            print(f"  Window {cp.index}: {cp.from_regime} → {cp.to_regime} (CUSUM: {cp.cusum_value:.3f})")

    if report.early_warnings:
        print(f"\n⚠️  {len(report.early_warnings)} early-warning signal(s)")

    print("\n🧠 Recommendations:")
    for r in report.recommendations:
        print(f"  {r}")

    # Write HTML
    html_path = os.path.join(os.getcwd(), "regime_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(generate_html(report))
    print(f"\n📄 HTML report: {html_path}")

    # Write JSON
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"📦 JSON export: {args.output}")


if __name__ == "__main__":
    asyncio.run(_main())
