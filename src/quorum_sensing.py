"""Swarm Quorum Sensing Engine — autonomous density-dependent behavior coordination.

Biologically-inspired by bacterial quorum sensing (V. fischeri bioluminescence,
P. aeruginosa biofilm formation, S. aureus virulence activation).  Agents
produce *autoinducer* signals at a constitutive rate.  When local signal
concentration exceeds a density-dependent threshold, collective behaviors
activate — enabling the swarm to coordinate actions only when sufficient
participants are present.

Capabilities:

- **Signal Production & Decay** — agents constitutively emit typed autoinducer
  molecules that accumulate in a shared signal pool with exponential decay.
- **Threshold Activation** — configurable quorum thresholds per behavior;
  collective programs activate when concentration crosses threshold.
- **Multi-Channel Signaling** — supports multiple orthogonal signal channels
  (like bacterial AI-1/AI-2/AHL systems) for independent behavior control.
- **Density Estimation** — real-time population density inference from signal
  concentration using inverse-decay mathematics.
- **Behavioral Programs** — registered collective behaviors that activate/
  deactivate based on quorum state (biofilm, bioluminescence, virulence,
  competence, sporulation, swarming).
- **Signal Interference** — agents can jam or amplify signals, modeling
  quorum quenching enzymes and signal boosting.
- **Hysteresis & Memory** — once activated, behaviors require signal to drop
  below a lower threshold to deactivate (prevents oscillation).
- **Quorum Health Score** — composite 0-100 metric measuring signal diversity,
  responsiveness, and coordination efficiency.
- **Interactive HTML Dashboard** — visualizes signal concentrations, threshold
  crossings, behavior state timeline, and population dynamics.

Usage (Python API)::

    from src.quorum_sensing import SwarmQuorumSensingEngine, SignalChannel

    engine = SwarmQuorumSensingEngine(agents=["a1", "a2", "a3", "a4", "a5"])

    # Register collective behaviors with activation thresholds
    engine.register_behavior("biofilm", channel="ahl", threshold=3.0, hysteresis=0.5)
    engine.register_behavior("bioluminescence", channel="ai1", threshold=5.0)
    engine.register_behavior("swarming", channel="ai2", threshold=2.0)

    # Agents produce signals
    engine.produce("a1", channel="ahl", intensity=1.0)
    engine.produce("a2", channel="ahl", intensity=1.2)
    engine.produce("a3", channel="ahl", intensity=0.9)

    # Advance time (accumulates signals, applies decay, checks thresholds)
    snapshot = engine.tick()
    print(snapshot.active_behaviors)  # [] or ["biofilm"] depending on concentration

    # Signal jamming (quorum quenching)
    engine.jam("enemy-agent", channel="ahl", strength=2.0)

    # Full analysis
    report = engine.analyze()
    engine.export_html("quorum_report.html")

CLI::

    python -m src.quorum_sensing                     # demo with 10 agents
    python -m src.quorum_sensing --agents 20         # larger population
    python -m src.quorum_sensing --ticks 80          # longer simulation
    python -m src.quorum_sensing --out report.html --json quorum.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SignalChannel:
    """An orthogonal signaling channel (like AHL, AI-1, AI-2)."""
    name: str
    decay_rate: float = 0.1  # exponential decay per tick
    diffusion_rate: float = 1.0  # how fast signals spread
    concentration: float = 0.0  # current signal level
    history: List[float] = field(default_factory=list)


@dataclass
class BehaviorProgram:
    """A collective behavior that activates at quorum."""
    name: str
    channel: str  # which signal channel controls this
    threshold: float  # activation threshold
    hysteresis: float = 0.5  # deactivation = threshold - hysteresis
    active: bool = False
    activation_tick: Optional[int] = None
    deactivation_tick: Optional[int] = None
    total_active_ticks: int = 0


@dataclass
class SignalEvent:
    """A single signal production or jamming event."""
    agent_id: str
    channel: str
    intensity: float
    event_type: str  # "produce" or "jam"
    tick: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class QuorumSnapshot:
    """State at a single tick."""
    tick: int
    concentrations: Dict[str, float]  # channel -> concentration
    active_behaviors: List[str]
    estimated_density: float  # inferred population density
    signal_diversity: float  # Shannon entropy of channel usage
    coordination_efficiency: float  # 0-1
    quorum_health_score: float  # 0-100
    events_this_tick: int
    newly_activated: List[str] = field(default_factory=list)
    newly_deactivated: List[str] = field(default_factory=list)


@dataclass
class QuorumReport:
    """Full analysis across all ticks."""
    snapshots: List[QuorumSnapshot] = field(default_factory=list)
    overall_health: float = 0.0
    total_activations: int = 0
    total_deactivations: int = 0
    peak_concentration: Dict[str, float] = field(default_factory=dict)
    behavior_uptime: Dict[str, float] = field(default_factory=dict)  # fraction of time active
    agent_contributions: Dict[str, float] = field(default_factory=dict)
    jamming_events: int = 0
    signal_wars: List[Dict[str, Any]] = field(default_factory=list)
    density_timeline: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SwarmQuorumSensingEngine:
    """Autonomous density-dependent behavior coordination for multi-agent swarms."""

    def __init__(
        self,
        agents: List[str],
        default_decay: float = 0.1,
        density_calibration: float = 1.0,
    ) -> None:
        self.agents = list(agents)
        self.default_decay = default_decay
        self.density_calibration = density_calibration

        # State
        self._channels: Dict[str, SignalChannel] = {}
        self._behaviors: Dict[str, BehaviorProgram] = {}
        self._events: List[SignalEvent] = []
        self._tick: int = 0
        self._snapshots: List[QuorumSnapshot] = []
        self._agent_total_production: Dict[str, float] = defaultdict(float)
        self._tick_events: List[SignalEvent] = []

    @property
    def tick_count(self) -> int:
        return self._tick

    @property
    def channels(self) -> Dict[str, SignalChannel]:
        return dict(self._channels)

    @property
    def behaviors(self) -> Dict[str, BehaviorProgram]:
        return dict(self._behaviors)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_channel(self, name: str, decay_rate: Optional[float] = None,
                    diffusion_rate: float = 1.0) -> SignalChannel:
        """Add a signaling channel."""
        ch = SignalChannel(
            name=name,
            decay_rate=decay_rate if decay_rate is not None else self.default_decay,
            diffusion_rate=diffusion_rate,
        )
        self._channels[name] = ch
        return ch

    def register_behavior(self, name: str, channel: str, threshold: float,
                          hysteresis: float = 0.5) -> BehaviorProgram:
        """Register a collective behavior program."""
        if channel not in self._channels:
            self.add_channel(channel)
        bp = BehaviorProgram(
            name=name, channel=channel, threshold=threshold, hysteresis=hysteresis
        )
        self._behaviors[name] = bp
        return bp

    # ------------------------------------------------------------------
    # Signal Production & Jamming
    # ------------------------------------------------------------------

    def produce(self, agent_id: str, channel: str, intensity: float = 1.0) -> None:
        """Agent produces autoinducer signal."""
        if channel not in self._channels:
            self.add_channel(channel)
        evt = SignalEvent(
            agent_id=agent_id, channel=channel, intensity=intensity,
            event_type="produce", tick=self._tick
        )
        self._events.append(evt)
        self._tick_events.append(evt)
        self._channels[channel].concentration += intensity * self._channels[channel].diffusion_rate
        self._agent_total_production[agent_id] += intensity

    def jam(self, agent_id: str, channel: str, strength: float = 1.0) -> None:
        """Agent jams (quorum quenches) a signal channel."""
        if channel not in self._channels:
            self.add_channel(channel)
        evt = SignalEvent(
            agent_id=agent_id, channel=channel, intensity=-strength,
            event_type="jam", tick=self._tick
        )
        self._events.append(evt)
        self._tick_events.append(evt)
        self._channels[channel].concentration = max(
            0.0, self._channels[channel].concentration - strength
        )

    # ------------------------------------------------------------------
    # Tick & Threshold Logic
    # ------------------------------------------------------------------

    def tick(self, steps: int = 1) -> QuorumSnapshot:
        """Advance simulation by N steps. Returns final snapshot."""
        snapshot = None
        for _ in range(steps):
            snapshot = self._do_tick()
        return snapshot  # type: ignore[return-value]

    def _do_tick(self) -> QuorumSnapshot:
        """Single tick: decay, check thresholds, record snapshot."""
        self._tick += 1
        self._tick_events = []

        # Apply exponential decay to all channels
        for ch in self._channels.values():
            ch.concentration *= (1.0 - ch.decay_rate)
            ch.history.append(ch.concentration)

        # Check behavior thresholds (with hysteresis)
        newly_activated: List[str] = []
        newly_deactivated: List[str] = []

        for bp in self._behaviors.values():
            ch = self._channels.get(bp.channel)
            if ch is None:
                continue
            conc = ch.concentration

            if not bp.active and conc >= bp.threshold:
                bp.active = True
                bp.activation_tick = self._tick
                newly_activated.append(bp.name)
            elif bp.active and conc < (bp.threshold - bp.hysteresis):
                bp.active = False
                bp.deactivation_tick = self._tick
                newly_deactivated.append(bp.name)

            if bp.active:
                bp.total_active_ticks += 1

        # Compute metrics
        concentrations = {name: ch.concentration for name, ch in self._channels.items()}
        active_behaviors = [bp.name for bp in self._behaviors.values() if bp.active]
        density = self._estimate_density()
        diversity = self._signal_diversity()
        efficiency = self._coordination_efficiency()
        health = self._quorum_health(diversity, efficiency, active_behaviors)

        snapshot = QuorumSnapshot(
            tick=self._tick,
            concentrations=concentrations,
            active_behaviors=active_behaviors,
            estimated_density=density,
            signal_diversity=diversity,
            coordination_efficiency=efficiency,
            quorum_health_score=health,
            events_this_tick=len(self._tick_events),
            newly_activated=newly_activated,
            newly_deactivated=newly_deactivated,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def _estimate_density(self) -> float:
        """Estimate population density from signal concentrations."""
        if not self._channels:
            return 0.0
        total_conc = sum(ch.concentration for ch in self._channels.values())
        # Inverse decay estimation: concentration ≈ n_agents * production / decay_rate
        avg_decay = statistics.mean(ch.decay_rate for ch in self._channels.values()) or 0.1
        estimated = (total_conc * avg_decay) / self.density_calibration
        return max(0.0, estimated)

    def _signal_diversity(self) -> float:
        """Shannon entropy of signal distribution across channels."""
        if not self._channels:
            return 0.0
        total = sum(max(ch.concentration, 0.001) for ch in self._channels.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for ch in self._channels.values():
            p = max(ch.concentration, 0.001) / total
            if p > 0:
                entropy -= p * math.log2(p)
        # Normalize to 0-1
        max_entropy = math.log2(len(self._channels)) if len(self._channels) > 1 else 1.0
        return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0

    def _coordination_efficiency(self) -> float:
        """How well the swarm coordinates (behaviors activate when expected)."""
        if not self._behaviors:
            return 0.0
        # Fraction of behaviors that are appropriately activated given signal levels
        correct = 0
        for bp in self._behaviors.values():
            ch = self._channels.get(bp.channel)
            if ch is None:
                continue
            if bp.active and ch.concentration >= bp.threshold:
                correct += 1
            elif not bp.active and ch.concentration < bp.threshold:
                correct += 1
        return correct / len(self._behaviors)

    def _quorum_health(self, diversity: float, efficiency: float,
                       active: List[str]) -> float:
        """Composite health score 0-100."""
        # Weighted: diversity (25%), efficiency (35%), activation ratio (20%), signal strength (20%)
        if not self._behaviors:
            return 50.0
        activation_ratio = len(active) / len(self._behaviors) if self._behaviors else 0.0
        avg_conc = statistics.mean(
            ch.concentration for ch in self._channels.values()
        ) if self._channels else 0.0
        # Normalize concentration to 0-1 scale (sigmoid)
        conc_norm = 1.0 / (1.0 + math.exp(-avg_conc + 2.0))

        score = (diversity * 25 + efficiency * 35 + activation_ratio * 20 + conc_norm * 20)
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self) -> QuorumReport:
        """Generate full analysis report."""
        report = QuorumReport()
        report.snapshots = list(self._snapshots)

        if self._snapshots:
            report.overall_health = statistics.mean(s.quorum_health_score for s in self._snapshots)
            report.density_timeline = [s.estimated_density for s in self._snapshots]

        # Peak concentrations
        for name, ch in self._channels.items():
            report.peak_concentration[name] = max(ch.history) if ch.history else 0.0

        # Behavior uptime
        for name, bp in self._behaviors.items():
            report.behavior_uptime[name] = (
                bp.total_active_ticks / self._tick if self._tick > 0 else 0.0
            )

        # Activations/deactivations
        for s in self._snapshots:
            report.total_activations += len(s.newly_activated)
            report.total_deactivations += len(s.newly_deactivated)

        # Agent contributions
        report.agent_contributions = dict(self._agent_total_production)

        # Jamming count
        report.jamming_events = sum(1 for e in self._events if e.event_type == "jam")

        # Signal wars: ticks where both produce and jam happened on same channel
        tick_channel_events: Dict[Tuple[int, str], Dict[str, int]] = defaultdict(
            lambda: {"produce": 0, "jam": 0}
        )
        for e in self._events:
            tick_channel_events[(e.tick, e.channel)][e.event_type] += 1
        for (t, ch), counts in tick_channel_events.items():
            if counts["produce"] > 0 and counts["jam"] > 0:
                report.signal_wars.append({"tick": t, "channel": ch, **counts})

        return report

    # ------------------------------------------------------------------
    # HTML Export
    # ------------------------------------------------------------------

    def export_html(self, filepath: str) -> str:
        """Export interactive HTML dashboard."""
        report = self.analyze()
        html_content = self._render_html(report)
        Path(filepath).write_text(html_content, encoding="utf-8")
        return filepath

    def _render_html(self, report: QuorumReport) -> str:
        """Render HTML dashboard."""
        # Concentration timeline data
        conc_data = {}
        for name, ch in self._channels.items():
            conc_data[name] = ch.history

        # Behavior activation timeline
        behavior_timeline = []
        for s in self._snapshots:
            behavior_timeline.append({
                "tick": s.tick,
                "active": s.active_behaviors,
                "health": s.quorum_health_score,
            })

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Quorum Sensing — Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0a0a0f; color: #e0e0e0; padding: 24px; }}
h1 {{ color: #00e5ff; margin-bottom: 8px; }}
h2 {{ color: #76ff03; margin: 24px 0 12px; font-size: 1.2em; }}
.subtitle {{ color: #888; margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.card {{ background: #1a1a2e; border-radius: 12px; padding: 20px; border: 1px solid #333; }}
.card h3 {{ color: #ffab40; margin-bottom: 8px; font-size: 0.95em; }}
.metric {{ font-size: 2em; font-weight: bold; color: #00e5ff; }}
.metric-sm {{ font-size: 1.3em; color: #76ff03; }}
.bar {{ height: 8px; border-radius: 4px; background: #333; margin-top: 8px; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
th {{ color: #ffab40; font-size: 0.85em; text-transform: uppercase; }}
td {{ font-size: 0.9em; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 0.8em; margin: 2px; }}
.tag-active {{ background: #1b5e20; color: #76ff03; }}
.tag-inactive {{ background: #333; color: #888; }}
.timeline {{ display: flex; flex-wrap: wrap; gap: 2px; margin-top: 8px; }}
.tick-dot {{ width: 6px; height: 6px; border-radius: 50%; }}
.canvas {{ background: #111; border-radius: 8px; padding: 16px; margin-top: 16px; overflow-x: auto; }}
.ascii {{ font-family: monospace; font-size: 11px; white-space: pre; line-height: 1.4; color: #aaa; }}
</style>
</head>
<body>
<h1>🧬 Swarm Quorum Sensing Dashboard</h1>
<p class="subtitle">Density-dependent collective behavior coordination — {len(self.agents)} agents, {self._tick} ticks, {len(self._channels)} channels</p>

<div class="grid">
  <div class="card">
    <h3>Quorum Health Score</h3>
    <div class="metric">{report.overall_health:.1f}</div>
    <div class="bar"><div class="bar-fill" style="width:{report.overall_health}%;background:{'#76ff03' if report.overall_health > 60 else '#ffab40' if report.overall_health > 30 else '#ff5252'}"></div></div>
  </div>
  <div class="card">
    <h3>Total Activations</h3>
    <div class="metric">{report.total_activations}</div>
    <p style="color:#888;margin-top:4px">{report.total_deactivations} deactivations</p>
  </div>
  <div class="card">
    <h3>Signal Wars</h3>
    <div class="metric">{len(report.signal_wars)}</div>
    <p style="color:#888;margin-top:4px">{report.jamming_events} jam events</p>
  </div>
  <div class="card">
    <h3>Active Behaviors</h3>
    <div class="metric-sm">{', '.join(bp.name for bp in self._behaviors.values() if bp.active) or 'None'}</div>
  </div>
</div>

<h2>Signal Channels</h2>
<div class="card">
<table>
<tr><th>Channel</th><th>Concentration</th><th>Peak</th><th>Decay Rate</th></tr>
{''.join(f'<tr><td>{html_mod.escape(name)}</td><td>{ch.concentration:.3f}</td><td>{report.peak_concentration.get(name, 0):.3f}</td><td>{ch.decay_rate:.2f}</td></tr>' for name, ch in self._channels.items())}
</table>
</div>

<h2>Behavior Programs</h2>
<div class="card">
<table>
<tr><th>Behavior</th><th>Channel</th><th>Threshold</th><th>Status</th><th>Uptime</th></tr>
{''.join(f'<tr><td>{html_mod.escape(bp.name)}</td><td>{bp.channel}</td><td>{bp.threshold:.1f}</td><td><span class="tag {"tag-active" if bp.active else "tag-inactive"}">{"ACTIVE" if bp.active else "inactive"}</span></td><td>{report.behavior_uptime.get(bp.name, 0)*100:.1f}%</td></tr>' for bp in self._behaviors.values())}
</table>
</div>

<h2>Agent Contributions</h2>
<div class="card">
<table>
<tr><th>Agent</th><th>Total Signal Produced</th><th>Contribution %</th></tr>
{''.join(f'<tr><td>{html_mod.escape(a)}</td><td>{report.agent_contributions.get(a, 0):.2f}</td><td>{(report.agent_contributions.get(a, 0) / max(sum(report.agent_contributions.values()), 0.001)) * 100:.1f}%</td></tr>' for a in sorted(self.agents, key=lambda x: report.agent_contributions.get(x, 0), reverse=True)[:15])}
</table>
</div>

<h2>Concentration History</h2>
<div class="canvas">
<pre class="ascii">{self._render_ascii_chart(report)}</pre>
</div>

<h2>Health Timeline</h2>
<div class="canvas">
<pre class="ascii">{self._render_health_timeline(report)}</pre>
</div>

</body>
</html>"""

    def _render_ascii_chart(self, report: QuorumReport) -> str:
        """Render ASCII concentration chart."""
        if not self._channels:
            return "No channels configured."
        height = 12
        width = min(60, self._tick)
        if width == 0:
            return "No data yet."

        lines = []
        for name, ch in list(self._channels.items())[:4]:
            lines.append(f"  Channel: {name} (peak: {report.peak_concentration.get(name, 0):.2f})")
            history = ch.history[-width:] if ch.history else []
            if not history:
                lines.append("  [no data]")
                continue
            max_val = max(max(history), 0.01)
            for row in range(height, 0, -1):
                threshold = max_val * row / height
                line = "  "
                for val in history:
                    line += "█" if val >= threshold else " "
                if row == height:
                    line += f" {max_val:.2f}"
                lines.append(line)
            lines.append("  " + "─" * len(history) + f"  ({len(history)} ticks)")
            # Show threshold line
            for bp in self._behaviors.values():
                if bp.channel == name:
                    int((bp.threshold / max_val) * height) if max_val > 0 else 0
                    lines.append(f"  ↑ Threshold '{bp.name}': {bp.threshold:.1f}")
            lines.append("")
        return "\n".join(lines)

    def _render_health_timeline(self, report: QuorumReport) -> str:
        """Render ASCII health score timeline."""
        if not report.snapshots:
            return "No snapshots yet."
        width = min(60, len(report.snapshots))
        samples = report.snapshots[-width:]
        height = 8
        lines = []
        lines.append("  Health Score (0-100)")
        for row in range(height, 0, -1):
            threshold = 100.0 * row / height
            line = f"  {threshold:5.0f}│"
            for s in samples:
                line += "█" if s.quorum_health_score >= threshold else " "
            lines.append(line)
        lines.append("       └" + "─" * len(samples))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def export_json(self, filepath: str) -> str:
        """Export analysis as JSON."""
        report = self.analyze()
        data = {
            "overall_health": report.overall_health,
            "total_activations": report.total_activations,
            "total_deactivations": report.total_deactivations,
            "peak_concentration": report.peak_concentration,
            "behavior_uptime": report.behavior_uptime,
            "agent_contributions": report.agent_contributions,
            "jamming_events": report.jamming_events,
            "signal_wars": report.signal_wars,
            "density_timeline": report.density_timeline,
            "snapshots": [
                {
                    "tick": s.tick,
                    "concentrations": s.concentrations,
                    "active_behaviors": s.active_behaviors,
                    "estimated_density": s.estimated_density,
                    "health": s.quorum_health_score,
                }
                for s in report.snapshots
            ],
        }
        Path(filepath).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return filepath


# ---------------------------------------------------------------------------
# CLI Demo
# ---------------------------------------------------------------------------


def _run_demo(args: argparse.Namespace) -> None:
    """Run an interactive demo simulation."""
    agents = [f"agent-{i+1}" for i in range(args.agents)]
    engine = SwarmQuorumSensingEngine(agents=agents, default_decay=0.08)

    # Register channels and behaviors (mimicking bacterial systems)
    engine.add_channel("ahl", decay_rate=0.08)  # acyl-homoserine lactone
    engine.add_channel("ai2", decay_rate=0.12)  # autoinducer-2 (universal)
    engine.add_channel("aip", decay_rate=0.06)  # autoinducing peptide

    engine.register_behavior("biofilm_formation", channel="ahl", threshold=4.0, hysteresis=1.0)
    engine.register_behavior("bioluminescence", channel="ai2", threshold=6.0, hysteresis=1.5)
    engine.register_behavior("virulence_activation", channel="aip", threshold=8.0, hysteresis=2.0)
    engine.register_behavior("competence", channel="ahl", threshold=2.5, hysteresis=0.5)
    engine.register_behavior("sporulation", channel="ai2", threshold=10.0, hysteresis=3.0)
    engine.register_behavior("swarming_motility", channel="aip", threshold=3.0, hysteresis=0.8)

    print(f"🧬 Swarm Quorum Sensing Simulation")
    print(f"   Agents: {args.agents} | Ticks: {args.ticks} | Channels: 3")
    print(f"   Behaviors: {len(engine.behaviors)}")
    print("─" * 60)

    # Simulation phases
    for t in range(args.ticks):
        # Phase 1: Growth (first third) — increasing signal production
        if t < args.ticks // 3:
            # Population growing — more agents producing
            active_count = min(len(agents), int(len(agents) * (t / (args.ticks // 3))))
            for a in agents[:active_count]:
                ch = random.choice(["ahl", "ai2", "aip"])
                engine.produce(a, channel=ch, intensity=random.uniform(0.3, 1.2))

        # Phase 2: Peak density (middle third) — all agents active
        elif t < 2 * args.ticks // 3:
            for a in agents:
                ch = random.choice(["ahl", "ai2", "aip"])
                engine.produce(a, channel=ch, intensity=random.uniform(0.5, 1.5))
            # Occasional signal boost (positive feedback)
            if random.random() < 0.2:
                boosted_ch = random.choice(["ahl", "ai2", "aip"])
                engine.produce(random.choice(agents), channel=boosted_ch, intensity=3.0)

        # Phase 3: Decline + jamming (final third)
        else:
            # Reduced production
            for a in agents[:len(agents) // 2]:
                ch = random.choice(["ahl", "ai2", "aip"])
                engine.produce(a, channel=ch, intensity=random.uniform(0.2, 0.8))
            # Quorum quenching attacks
            if random.random() < 0.3:
                jammed_ch = random.choice(["ahl", "ai2", "aip"])
                engine.jam(f"quencher-{random.randint(1,3)}", channel=jammed_ch, strength=2.0)

        snapshot = engine.tick()

        # Print key events
        if snapshot.newly_activated:
            print(f"  ✅ Tick {t+1}: ACTIVATED → {', '.join(snapshot.newly_activated)}")
        if snapshot.newly_deactivated:
            print(f"  ⛔ Tick {t+1}: DEACTIVATED → {', '.join(snapshot.newly_deactivated)}")

    # Final report
    report = engine.analyze()
    print("\n" + "─" * 60)
    print(f"📊 Final Report")
    print(f"   Overall Health Score: {report.overall_health:.1f}/100")
    print(f"   Total Activations: {report.total_activations}")
    print(f"   Total Deactivations: {report.total_deactivations}")
    print(f"   Jamming Events: {report.jamming_events}")
    print(f"   Signal Wars: {len(report.signal_wars)}")
    print(f"\n   Behavior Uptime:")
    for name, uptime in sorted(report.behavior_uptime.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(uptime * 30) + "░" * (30 - int(uptime * 30))
        print(f"     {name:25s} {bar} {uptime*100:.1f}%")

    print(f"\n   Peak Concentrations:")
    for ch, peak in report.peak_concentration.items():
        print(f"     {ch}: {peak:.2f}")

    # Export
    if args.out:
        engine.export_html(args.out)
        print(f"\n   📄 HTML report: {args.out}")
    if args.json_out:
        engine.export_json(args.json_out)
        print(f"   📄 JSON export: {args.json_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm Quorum Sensing Engine — density-dependent behavior coordination"
    )
    parser.add_argument("--agents", type=int, default=10, help="Number of agents")
    parser.add_argument("--ticks", type=int, default=60, help="Simulation ticks")
    parser.add_argument("--out", type=str, default=None, help="HTML output path")
    parser.add_argument("--json", dest="json_out", type=str, default=None, help="JSON output path")
    args = parser.parse_args()
    _run_demo(args)


if __name__ == "__main__":
    main()
