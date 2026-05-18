"""Swarm Neuroplasticity Engine — autonomous structural adaptation of agent communication networks.

Biologically-inspired by how neural networks physically restructure
themselves based on activity patterns.  Agent-to-agent connections
(synapses) strengthen when agents successfully collaborate (Hebbian
learning) and weaken when interactions fail or cease.  The network
self-optimizes its topology through Long-Term Potentiation (LTP),
Long-Term Depression (LTD), synaptic pruning, synaptogenesis, homeostatic
scaling, and critical periods of heightened plasticity.

Capabilities:

- **Hebbian Learning** — "agents that fire together wire together":
  successful co-activation strengthens synaptic weight.
- **Long-Term Potentiation (LTP)** — sustained high-frequency
  co-activation causes persistent strengthening and POTENTIATED state.
- **Long-Term Depression (LTD)** — prolonged inactivity causes persistent
  weakening and DEPRESSED state.
- **Synaptic Pruning** — connections below a threshold are removed,
  simplifying the network.
- **Synaptogenesis** — new connections form when agents have too few
  outgoing synapses.
- **Critical Periods** — time windows of 2x learning rate where the
  network reshapes rapidly.
- **Homeostatic Plasticity** — global weight scaling to keep the network
  in a healthy operating range.
- **Network Snapshots** — periodic topology metrics (density, clustering,
  hub identification).
- **Interactive HTML Dashboard** — visualizes plasticity timeline, weight
  distributions, hub rankings, and insights.

Usage (Python API)::

    from src.neuroplasticity import NeuroplasticityEngine

    engine = NeuroplasticityEngine(agents=["a1", "a2", "a3", "a4"])

    # Agents interact
    engine.activate("a1", "a2", success=True)
    engine.activate("a1", "a3", success=True)
    engine.activate("a2", "a3", success=False)

    # Advance time (triggers LTD, pruning, homeostatic scaling)
    engine.tick(steps=10)

    # Trigger heightened plasticity
    engine.trigger_critical_period(duration=20)

    # Full analysis
    report = engine.analyze()
    print(report.health_score)
    print(report.insights)

    engine.export_html("neuroplasticity_report.html")

CLI::

    python -m src.neuroplasticity                    # demo with 8 agents
    python -m src.neuroplasticity --agents 12        # larger swarm
    python -m src.neuroplasticity --steps 200        # longer simulation
    python -m src.neuroplasticity --out report.html --json data.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------


class PlasticityEvent(str, Enum):
    """Types of plasticity events in the network."""
    LTP = "ltp"
    LTD = "ltd"
    PRUNING = "pruning"
    SYNAPTOGENESIS = "synaptogenesis"
    HEBBIAN_STRENGTHEN = "hebbian_strengthen"
    HEBBIAN_WEAKEN = "hebbian_weaken"
    HOMEOSTATIC_SCALE = "homeostatic_scale"
    CRITICAL_PERIOD_START = "critical_period_start"
    CRITICAL_PERIOD_END = "critical_period_end"


class SynapseState(str, Enum):
    """Lifecycle states of a synapse."""
    NASCENT = "nascent"
    STABLE = "stable"
    POTENTIATED = "potentiated"
    DEPRESSED = "depressed"
    PRUNING_CANDIDATE = "pruning_candidate"


@dataclass
class Synapse:
    """A directed weighted connection between two agents."""
    source: str
    target: str
    weight: float = 0.5
    state: SynapseState = SynapseState.NASCENT
    created_tick: int = 0
    last_activated: int = 0
    activation_count: int = 0
    ltp_count: int = 0
    ltd_count: int = 0


@dataclass
class PlasticityRecord:
    """Record of a single plasticity event."""
    tick: int
    event_type: PlasticityEvent
    source: str
    target: str
    old_weight: float
    new_weight: float
    reason: str = ""


@dataclass
class NetworkSnapshot:
    """Snapshot of network topology at a point in time."""
    tick: int
    num_agents: int
    num_synapses: int
    avg_weight: float
    density: float
    clustering_coefficient: float
    top_hubs: List[Tuple[str, int]]


@dataclass
class NeuroplasticityReport:
    """Full analysis report."""
    snapshots: List[NetworkSnapshot] = field(default_factory=list)
    events: List[PlasticityRecord] = field(default_factory=list)
    health_score: float = 0.0
    pruned_count: int = 0
    formed_count: int = 0
    ltp_events: int = 0
    ltd_events: int = 0
    critical_periods: int = 0
    insights: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class NeuroplasticityEngine:
    """Autonomous structural adaptation of agent communication networks."""

    def __init__(
        self,
        agents: List[str],
        initial_connectivity: float = 0.3,
        prune_threshold: float = 0.1,
        ltp_threshold: int = 5,
        ltd_threshold: int = 20,
        homeostatic_target: float = 0.5,
        critical_period_duration: int = 50,
    ):
        self.agents = list(agents)
        self.prune_threshold = prune_threshold
        self.ltp_threshold = ltp_threshold
        self.ltd_threshold = ltd_threshold
        self.homeostatic_target = homeostatic_target
        self.critical_period_duration = critical_period_duration
        self.base_learning_rate = 0.1

        # State
        self.current_tick: int = 0
        self.synapses: Dict[Tuple[str, str], Synapse] = {}
        self.events: List[PlasticityRecord] = []
        self.snapshots: List[NetworkSnapshot] = []
        self.pruned_count: int = 0
        self.formed_count: int = 0

        # Critical period state
        self.in_critical_period: bool = False
        self.critical_period_end_tick: int = 0
        self.critical_period_count: int = 0

        # Initialize random connections
        self._initialize_connections(initial_connectivity)

    @property
    def learning_rate(self) -> float:
        """Current learning rate (doubled during critical periods)."""
        return self.base_learning_rate * (2.0 if self.in_critical_period else 1.0)

    def _initialize_connections(self, connectivity: float) -> None:
        """Create initial random connections between agents."""
        for src in self.agents:
            for tgt in self.agents:
                if src == tgt:
                    continue
                if random.random() < connectivity:
                    weight = 0.3 + random.random() * 0.4  # 0.3-0.7
                    self.synapses[(src, tgt)] = Synapse(
                        source=src,
                        target=tgt,
                        weight=weight,
                        state=SynapseState.NASCENT,
                        created_tick=0,
                        last_activated=0,
                    )
                    self.formed_count += 1

    def activate(self, source: str, target: str, success: bool = True) -> None:
        """Record an interaction between agents and apply Hebbian update."""
        if source == target:
            return
        key = (source, target)

        # Create synapse if it doesn't exist
        if key not in self.synapses:
            self.synapses[key] = Synapse(
                source=source,
                target=target,
                weight=0.3,
                state=SynapseState.NASCENT,
                created_tick=self.current_tick,
                last_activated=self.current_tick,
            )
            self.formed_count += 1
            self._record_event(
                PlasticityEvent.SYNAPTOGENESIS, source, target, 0.0, 0.3,
                "auto-created on first interaction"
            )

        syn = self.synapses[key]
        old_weight = syn.weight
        syn.last_activated = self.current_tick
        syn.activation_count += 1

        if success:
            # Hebbian strengthening: bounded growth
            delta = self.learning_rate * (1.0 - syn.weight) * 0.1
            syn.weight = min(1.0, syn.weight + delta)
            self._record_event(
                PlasticityEvent.HEBBIAN_STRENGTHEN, source, target,
                old_weight, syn.weight, "successful interaction"
            )
        else:
            # Hebbian weakening: bounded decay
            delta = self.learning_rate * syn.weight * 0.05
            syn.weight = max(0.0, syn.weight - delta)
            self._record_event(
                PlasticityEvent.HEBBIAN_WEAKEN, source, target,
                old_weight, syn.weight, "failed interaction"
            )

        # Check for LTP
        if syn.activation_count >= self.ltp_threshold and syn.state != SynapseState.POTENTIATED:
            old_w = syn.weight
            syn.weight = min(1.0, syn.weight + 0.15)
            syn.state = SynapseState.POTENTIATED
            syn.ltp_count += 1
            self._record_event(
                PlasticityEvent.LTP, source, target, old_w, syn.weight,
                f"activation count {syn.activation_count} >= threshold {self.ltp_threshold}"
            )

        # Update state based on weight
        if syn.state == SynapseState.NASCENT and syn.activation_count >= 3:
            syn.state = SynapseState.STABLE

    def tick(self, steps: int = 1) -> None:
        """Advance time: apply LTD, pruning, synaptogenesis, homeostatic scaling."""
        for _ in range(steps):
            self.current_tick += 1

            # Check critical period end
            if self.in_critical_period and self.current_tick >= self.critical_period_end_tick:
                self.in_critical_period = False
                self._record_event(
                    PlasticityEvent.CRITICAL_PERIOD_END, "", "", 0, 0,
                    f"critical period ended at tick {self.current_tick}"
                )

            # Apply LTD to idle synapses
            to_prune: List[Tuple[str, str]] = []
            for key, syn in list(self.synapses.items()):
                idle_ticks = self.current_tick - syn.last_activated
                if idle_ticks >= self.ltd_threshold and syn.state != SynapseState.DEPRESSED:
                    old_w = syn.weight
                    syn.weight = max(0.0, syn.weight - 0.1)
                    syn.state = SynapseState.DEPRESSED
                    syn.ltd_count += 1
                    self._record_event(
                        PlasticityEvent.LTD, syn.source, syn.target,
                        old_w, syn.weight,
                        f"idle for {idle_ticks} ticks >= {self.ltd_threshold}"
                    )

                # Check for pruning
                if syn.weight < self.prune_threshold:
                    syn.state = SynapseState.PRUNING_CANDIDATE
                    to_prune.append(key)

            # Prune
            for key in to_prune:
                syn = self.synapses.pop(key)
                self.pruned_count += 1
                self._record_event(
                    PlasticityEvent.PRUNING, syn.source, syn.target,
                    syn.weight, 0.0,
                    f"weight {syn.weight:.3f} < threshold {self.prune_threshold}"
                )

            # Synaptogenesis for poorly-connected agents
            for agent in self.agents:
                outgoing = [s for s in self.synapses.values() if s.source == agent]
                if len(outgoing) < 2:
                    # Try to form a new connection
                    candidates = [a for a in self.agents if a != agent and (agent, a) not in self.synapses]
                    if candidates:
                        target = random.choice(candidates)
                        new_syn = Synapse(
                            source=agent,
                            target=target,
                            weight=0.3,
                            state=SynapseState.NASCENT,
                            created_tick=self.current_tick,
                            last_activated=self.current_tick,
                        )
                        self.synapses[(agent, target)] = new_syn
                        self.formed_count += 1
                        self._record_event(
                            PlasticityEvent.SYNAPTOGENESIS, agent, target,
                            0.0, 0.3,
                            f"agent {agent} had <2 outgoing connections"
                        )

            # Homeostatic scaling
            if self.synapses:
                weights = [s.weight for s in self.synapses.values()]
                mean_w = statistics.mean(weights)
                upper = self.homeostatic_target * 1.2
                lower = self.homeostatic_target * 0.8

                if mean_w > upper or mean_w < lower:
                    scale_factor = self.homeostatic_target / mean_w if mean_w > 0 else 1.0
                    # Gentle scaling (move 10% toward target)
                    scale_factor = 1.0 + (scale_factor - 1.0) * 0.1
                    for syn in self.synapses.values():
                        old_w = syn.weight
                        syn.weight = max(0.0, min(1.0, syn.weight * scale_factor))
                    self._record_event(
                        PlasticityEvent.HOMEOSTATIC_SCALE, "", "", mean_w,
                        mean_w * scale_factor,
                        f"mean weight {mean_w:.3f} outside [{lower:.3f}, {upper:.3f}]"
                    )

    def trigger_critical_period(self, duration: Optional[int] = None) -> None:
        """Enter heightened plasticity mode (2x learning rate)."""
        dur = duration or self.critical_period_duration
        self.in_critical_period = True
        self.critical_period_end_tick = self.current_tick + dur
        self.critical_period_count += 1
        self._record_event(
            PlasticityEvent.CRITICAL_PERIOD_START, "", "", 0, 0,
            f"critical period started, duration={dur} ticks"
        )

    def force_synaptogenesis(self, source: str, target: str) -> Optional[Synapse]:
        """Manually create a new synapse between agents."""
        if source == target:
            return None
        key = (source, target)
        if key in self.synapses:
            return self.synapses[key]
        syn = Synapse(
            source=source,
            target=target,
            weight=0.3,
            state=SynapseState.NASCENT,
            created_tick=self.current_tick,
            last_activated=self.current_tick,
        )
        self.synapses[key] = syn
        self.formed_count += 1
        self._record_event(
            PlasticityEvent.SYNAPTOGENESIS, source, target, 0.0, 0.3,
            "manual synaptogenesis"
        )
        return syn

    def get_synapse(self, source: str, target: str) -> Optional[Synapse]:
        """Get synapse between two agents if it exists."""
        return self.synapses.get((source, target))

    def get_agent_connections(self, agent: str) -> List[Synapse]:
        """Get all connections (in and out) for an agent."""
        return [
            s for s in self.synapses.values()
            if s.source == agent or s.target == agent
        ]

    def snapshot(self) -> NetworkSnapshot:
        """Capture current network topology metrics."""
        num_syn = len(self.synapses)
        n = len(self.agents)
        max_possible = n * (n - 1) if n > 1 else 1
        density = num_syn / max_possible if max_possible > 0 else 0.0

        weights = [s.weight for s in self.synapses.values()]
        avg_w = statistics.mean(weights) if weights else 0.0

        # Clustering coefficient (simplified: fraction of agent triads that are connected)
        cc = self._compute_clustering_coefficient()

        # Top hubs by connection count
        conn_count: Dict[str, int] = defaultdict(int)
        for syn in self.synapses.values():
            conn_count[syn.source] += 1
            conn_count[syn.target] += 1
        top_hubs = sorted(conn_count.items(), key=lambda x: x[1], reverse=True)[:5]

        snap = NetworkSnapshot(
            tick=self.current_tick,
            num_agents=n,
            num_synapses=num_syn,
            avg_weight=round(avg_w, 4),
            density=round(density, 4),
            clustering_coefficient=round(cc, 4),
            top_hubs=top_hubs,
        )
        self.snapshots.append(snap)
        return snap

    def _compute_clustering_coefficient(self) -> float:
        """Compute average local clustering coefficient."""
        if len(self.agents) < 3:
            return 0.0

        # Build undirected adjacency
        neighbors: Dict[str, Set[str]] = defaultdict(set)
        for (s, t) in self.synapses:
            neighbors[s].add(t)
            neighbors[t].add(s)

        coefficients = []
        for agent in self.agents:
            nbrs = neighbors[agent]
            k = len(nbrs)
            if k < 2:
                coefficients.append(0.0)
                continue
            # Count edges between neighbors
            links = 0
            nbr_list = list(nbrs)
            for i in range(len(nbr_list)):
                for j in range(i + 1, len(nbr_list)):
                    a, b = nbr_list[i], nbr_list[j]
                    if (a, b) in self.synapses or (b, a) in self.synapses:
                        links += 1
            max_links = k * (k - 1) / 2
            coefficients.append(links / max_links)

        return statistics.mean(coefficients) if coefficients else 0.0

    def analyze(self) -> NeuroplasticityReport:
        """Generate full analysis report with insights."""
        # Take a snapshot
        self.snapshot()

        ltp_events = sum(1 for e in self.events if e.event_type == PlasticityEvent.LTP)
        ltd_events = sum(1 for e in self.events if e.event_type == PlasticityEvent.LTD)

        insights = self._generate_insights()
        health = self._compute_health_score()

        return NeuroplasticityReport(
            snapshots=list(self.snapshots),
            events=list(self.events),
            health_score=round(health, 1),
            pruned_count=self.pruned_count,
            formed_count=self.formed_count,
            ltp_events=ltp_events,
            ltd_events=ltd_events,
            critical_periods=self.critical_period_count,
            insights=insights,
        )

    def _compute_health_score(self) -> float:
        """Compute network health score 0-100."""
        if not self.synapses:
            return 0.0

        scores = []
        weights = [s.weight for s in self.synapses.values()]
        n = len(self.agents)
        max_possible = n * (n - 1) if n > 1 else 1

        # Density score (prefer 0.2-0.6)
        density = len(self.synapses) / max_possible if max_possible > 0 else 0
        if 0.2 <= density <= 0.6:
            scores.append(100)
        elif density < 0.1 or density > 0.8:
            scores.append(30)
        else:
            scores.append(70)

        # Weight distribution (prefer moderate variance)
        if len(weights) >= 2:
            std = statistics.stdev(weights)
            if 0.1 <= std <= 0.3:
                scores.append(100)
            elif std < 0.05 or std > 0.4:
                scores.append(40)
            else:
                scores.append(70)
        else:
            scores.append(50)

        # Mean weight near homeostatic target
        mean_w = statistics.mean(weights)
        dist = abs(mean_w - self.homeostatic_target)
        scores.append(max(0, 100 - dist * 200))

        # Pruning/formation balance
        if self.formed_count > 0:
            ratio = self.pruned_count / self.formed_count
            if 0.2 <= ratio <= 0.8:
                scores.append(100)
            else:
                scores.append(50)
        else:
            scores.append(50)

        # No isolated agents
        connected_agents = set()
        for syn in self.synapses.values():
            connected_agents.add(syn.source)
            connected_agents.add(syn.target)
        connectivity_ratio = len(connected_agents) / len(self.agents) if self.agents else 1
        scores.append(connectivity_ratio * 100)

        return statistics.mean(scores)

    def _generate_insights(self) -> List[str]:
        """Generate human-readable insights about the network."""
        insights: List[str] = []

        # Hub detection
        conn_count: Dict[str, int] = defaultdict(int)
        for syn in self.synapses.values():
            conn_count[syn.source] += 1
            conn_count[syn.target] += 1

        hubs = [a for a, c in conn_count.items() if c >= 5]
        if hubs:
            insights.append(f"Hub agents detected ({len(hubs)}): {', '.join(hubs[:5])} — high connectivity may indicate coordination bottlenecks")

        # Isolated agents
        connected = set()
        for syn in self.synapses.values():
            connected.add(syn.source)
            connected.add(syn.target)
        isolated = [a for a in self.agents if a not in connected]
        if isolated:
            insights.append(f"Isolated agents ({len(isolated)}): {', '.join(isolated[:5])} — no active connections, consider triggering synaptogenesis")

        # Potentiation clusters
        potentiated = [(s.source, s.target) for s in self.synapses.values() if s.state == SynapseState.POTENTIATED]
        if len(potentiated) >= 3:
            insights.append(f"{len(potentiated)} potentiated synapses form strong collaboration clusters")

        # Pruning rate
        if self.pruned_count > self.formed_count * 0.8:
            insights.append("High pruning rate — network may be over-simplifying; consider lowering prune threshold")
        elif self.pruned_count < self.formed_count * 0.1:
            insights.append("Low pruning rate — network may accumulate weak connections; consider raising prune threshold")

        # LTP/LTD balance
        ltp = sum(1 for e in self.events if e.event_type == PlasticityEvent.LTP)
        ltd = sum(1 for e in self.events if e.event_type == PlasticityEvent.LTD)
        if ltp > 0 and ltd > 0:
            ratio = ltp / ltd
            if ratio > 3:
                insights.append("Network strongly favoring potentiation over depression — may become rigid")
            elif ratio < 0.3:
                insights.append("Network strongly favoring depression — connections degrading faster than strengthening")

        # Density insight
        n = len(self.agents)
        max_possible = n * (n - 1) if n > 1 else 1
        density = len(self.synapses) / max_possible if max_possible > 0 else 0
        if density > 0.7:
            insights.append(f"Very high network density ({density:.1%}) — almost fully connected, limited pruning")
        elif density < 0.15:
            insights.append(f"Sparse network ({density:.1%}) — agents poorly connected, may limit collective capability")

        if not insights:
            insights.append("Network is in a balanced, healthy state with no anomalies detected")

        return insights

    def _record_event(self, event_type: PlasticityEvent, source: str, target: str,
                      old_weight: float, new_weight: float, reason: str) -> None:
        """Record a plasticity event."""
        self.events.append(PlasticityRecord(
            tick=self.current_tick,
            event_type=event_type,
            source=source,
            target=target,
            old_weight=round(old_weight, 4),
            new_weight=round(new_weight, 4),
            reason=reason,
        ))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        """Export analysis data to JSON."""
        report = self.analyze()
        data = {
            "health_score": report.health_score,
            "pruned_count": report.pruned_count,
            "formed_count": report.formed_count,
            "ltp_events": report.ltp_events,
            "ltd_events": report.ltd_events,
            "critical_periods": report.critical_periods,
            "insights": report.insights,
            "num_agents": len(self.agents),
            "num_synapses": len(self.synapses),
            "current_tick": self.current_tick,
            "synapses": [
                {
                    "source": s.source,
                    "target": s.target,
                    "weight": round(s.weight, 4),
                    "state": s.state.value,
                    "activation_count": s.activation_count,
                    "ltp_count": s.ltp_count,
                    "ltd_count": s.ltd_count,
                }
                for s in self.synapses.values()
            ],
            "events_summary": {
                et.value: sum(1 for e in self.events if e.event_type == et)
                for et in PlasticityEvent
            },
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')

    def export_html(self, path: str) -> None:
        """Export interactive HTML dashboard."""
        report = self.analyze()

        # Event counts by type
        event_counts: Dict[str, int] = defaultdict(int)
        for e in self.events:
            event_counts[e.event_type.value] += 1

        # Weight distribution bins
        weights = [s.weight for s in self.synapses.values()]
        hist_bins = [0] * 10
        for w in weights:
            idx = min(9, int(w * 10))
            hist_bins[idx] += 1

        # Top hubs
        conn_count: Dict[str, int] = defaultdict(int)
        for syn in self.synapses.values():
            conn_count[syn.source] += 1
            conn_count[syn.target] += 1
        top_hubs = sorted(conn_count.items(), key=lambda x: x[1], reverse=True)[:8]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Neuroplasticity Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; padding: 24px; }}
h1 {{ font-size: 1.8rem; margin-bottom: 8px; color: #a78bfa; }}
h2 {{ font-size: 1.2rem; margin-bottom: 12px; color: #818cf8; border-bottom: 1px solid #2d2d44; padding-bottom: 6px; }}
.subtitle {{ color: #94a3b8; margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.card {{ background: #1e1e2e; border-radius: 12px; padding: 20px; border: 1px solid #2d2d44; }}
.stat {{ font-size: 2rem; font-weight: bold; color: #a78bfa; }}
.label {{ font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }}
.bar-container {{ display: flex; align-items: center; margin: 6px 0; }}
.bar-label {{ width: 120px; font-size: 0.8rem; color: #94a3b8; }}
.bar {{ height: 20px; background: #6366f1; border-radius: 4px; min-width: 2px; transition: width 0.3s; }}
.bar-value {{ margin-left: 8px; font-size: 0.8rem; color: #c4b5fd; }}
.insight {{ background: #1a1a2e; border-left: 3px solid #a78bfa; padding: 10px 14px; margin: 8px 0; border-radius: 0 8px 8px 0; font-size: 0.9rem; }}
.health-score {{ font-size: 3rem; font-weight: bold; color: {'#10b981' if report.health_score >= 70 else '#f59e0b' if report.health_score >= 40 else '#ef4444'}; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #2d2d44; }}
th {{ color: #a78bfa; }}
</style>
</head>
<body>
<h1>🧠 Swarm Neuroplasticity Dashboard</h1>
<p class="subtitle">Autonomous structural adaptation · Tick {self.current_tick} · {len(self.agents)} agents</p>

<div class="grid">
  <div class="card"><div class="health-score">{report.health_score}</div><div class="label">Network Health Score (0-100)</div></div>
  <div class="card"><div class="stat">{len(self.synapses)}</div><div class="label">Active Synapses</div></div>
  <div class="card"><div class="stat">{report.formed_count}</div><div class="label">Total Formed</div></div>
  <div class="card"><div class="stat">{report.pruned_count}</div><div class="label">Total Pruned</div></div>
  <div class="card"><div class="stat">{report.ltp_events}</div><div class="label">LTP Events</div></div>
  <div class="card"><div class="stat">{report.ltd_events}</div><div class="label">LTD Events</div></div>
  <div class="card"><div class="stat">{report.critical_periods}</div><div class="label">Critical Periods</div></div>
  <div class="card"><div class="stat">{statistics.mean(weights):.3f}</div><div class="label">Mean Synapse Weight</div></div>
</div>

<div class="grid">
  <div class="card">
    <h2>Plasticity Event Distribution</h2>
    {''.join(f'<div class="bar-container"><div class="bar-label">{html_mod.escape(k)}</div><div class="bar" style="width:{min(200, v*3)}px"></div><div class="bar-value">{v}</div></div>' for k, v in sorted(event_counts.items(), key=lambda x: -x[1]))}
  </div>
  <div class="card">
    <h2>Weight Distribution</h2>
    {''.join(f'<div class="bar-container"><div class="bar-label">{i*10}%-{(i+1)*10}%</div><div class="bar" style="width:{min(200, c*8)}px"></div><div class="bar-value">{c}</div></div>' for i, c in enumerate(hist_bins))}
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Top Hub Agents</h2>
    <table><tr><th>Agent</th><th>Connections</th></tr>
    {''.join(f'<tr><td>{html_mod.escape(a)}</td><td>{c}</td></tr>' for a, c in top_hubs)}
    </table>
  </div>
  <div class="card">
    <h2>Insights</h2>
    {''.join(f'<div class="insight">{html_mod.escape(ins)}</div>' for ins in report.insights)}
  </div>
</div>

</body>
</html>"""
        Path(path).write_text(html, encoding='utf-8')

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @classmethod
    def demo(cls, num_agents: int = 8, steps: int = 100, seed: int | None = None) -> "NeuroplasticityEngine":
        """Run a demonstration simulation."""
        if seed is not None:
            random.seed(seed)

        agents = [f"agent-{i}" for i in range(num_agents)]
        engine = cls(agents=agents, initial_connectivity=0.3)

        for tick in range(steps):
            # Random activations (2-4 per tick)
            num_interactions = random.randint(2, 4)
            for _ in range(num_interactions):
                src = random.choice(agents)
                tgt = random.choice([a for a in agents if a != src])
                success = random.random() < 0.8
                engine.activate(src, tgt, success=success)

            # Trigger critical period at tick 30
            if tick == 30:
                engine.trigger_critical_period(duration=20)

            engine.tick()

        return engine


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Swarm Neuroplasticity Engine — autonomous network adaptation"
    )
    parser.add_argument("--agents", type=int, default=8, help="Number of agents (default: 8)")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps (default: 100)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default=None, help="Output HTML report path")
    parser.add_argument("--json", type=str, default=None, help="Output JSON data path")
    args = parser.parse_args()

    print(f"🧠 Swarm Neuroplasticity Engine")
    print(f"   Agents: {args.agents} | Steps: {args.steps}")
    print(f"   Simulating autonomous network adaptation...\n")

    engine = NeuroplasticityEngine.demo(
        num_agents=args.agents, steps=args.steps, seed=args.seed
    )

    report = engine.analyze()

    print(f"═══ Network Health Score: {report.health_score}/100 ═══\n")
    print(f"  Active Synapses: {len(engine.synapses)}")
    print(f"  Total Formed:    {report.formed_count}")
    print(f"  Total Pruned:    {report.pruned_count}")
    print(f"  LTP Events:      {report.ltp_events}")
    print(f"  LTD Events:      {report.ltd_events}")
    print(f"  Critical Periods: {report.critical_periods}")
    print(f"\n═══ Insights ═══\n")
    for ins in report.insights:
        print(f"  • {ins}")

    if args.out:
        engine.export_html(args.out)
        print(f"\n  📊 HTML report: {args.out}")
    if args.json:
        engine.export_json(args.json)
        print(f"  📦 JSON data: {args.json}")

    print()


if __name__ == "__main__":
    main()
