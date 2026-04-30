"""Swarm Consciousness Engine — autonomous collective awareness detection.

Biologically-inspired by theories of collective consciousness in social
organisms (ant supercolonies, neural assemblies, flocking birds).  Measures
emergent cognitive phenomena across multi-agent swarms: shared mental models,
collective attention, group intentionality, and thought contagion dynamics.

Capabilities:

- **Shared Mental Model Tracker** — measures belief convergence/divergence
  across agents, detects consensus formation and outlier beliefs.
- **Collective Attention Monitor** — tracks swarm-wide focus, detects
  attention fragmentation, hyperfocus (tunnel vision), and shifts.
- **Group Intentionality Detector** — identifies coordinated purposeful
  behavior without explicit communication (emergent alignment).
- **Hive Mind Coherence Score** — composite 0-100 score: belief alignment +
  attention coordination + intentional coherence + information flow.
- **Consciousness Phase Classifier** — classifies swarm phase: dormant,
  stirring, aware, synchronized, transcendent.
- **Thought Contagion Tracker** — measures belief spreading patterns,
  identifies viral ideas, mutation rates, and ideological clusters.
- **Interactive HTML Dashboard** — visualizes coherence timeline, phase
  transitions, belief heatmaps, and contagion networks.

Usage (Python API)::

    from src.consciousness import SwarmConsciousnessEngine

    engine = SwarmConsciousnessEngine(
        agents=["agent-1", "agent-2", "agent-3", "agent-4", "agent-5"]
    )

    # Agents submit beliefs about topics
    engine.submit_belief("agent-1", "strategy", 0.8, confidence=0.9)
    engine.submit_belief("agent-2", "strategy", 0.7, confidence=0.8)
    engine.submit_belief("agent-3", "strategy", -0.3, confidence=0.6)

    # Agents report attention focus
    engine.submit_attention("agent-1", "optimization", intensity=0.9)
    engine.submit_attention("agent-2", "optimization", intensity=0.8)

    # Agents declare intent
    engine.submit_intent("agent-1", goal="minimize_latency", action="route_optimize")
    engine.submit_intent("agent-2", goal="minimize_latency", action="cache_warm")

    # Advance time and get snapshot
    snapshot = engine.tick()
    print(snapshot.hive_mind_score)   # 0-100
    print(snapshot.phase)             # dormant/stirring/aware/synchronized/transcendent

    # Full report
    report = engine.analyze()
    engine.export_html("consciousness_report.html")

CLI::

    python -m src.consciousness                    # demo with 8 agents
    python -m src.consciousness --agents 12        # more agents
    python -m src.consciousness --ticks 100        # longer simulation
    python -m src.consciousness --out report.html --json consciousness.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AgentBelief:
    """An agent's stated belief about a topic."""
    agent_id: str
    topic: str
    value: float  # -1.0 to 1.0 (opinion spectrum)
    confidence: float  # 0-1
    timestamp: float = field(default_factory=time.time)


@dataclass
class AttentionEvent:
    """An agent reporting what it's focusing on."""
    agent_id: str
    focus_topic: str
    intensity: float  # 0-1
    timestamp: float = field(default_factory=time.time)


@dataclass
class IntentSignal:
    """An agent declaring its current goal and action."""
    agent_id: str
    goal: str
    action: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ThoughtContagion:
    """A belief that has spread through the swarm."""
    belief_topic: str
    origin_agent: str
    infected_agents: List[str] = field(default_factory=list)
    spread_rate: float = 0.0  # agents/tick
    virality: float = 0.0  # 0-1
    mutations: int = 0  # how much belief changed during spread


@dataclass
class ConsciousnessSnapshot:
    """State of collective consciousness at a single tick."""
    tick: int
    belief_alignment: float  # 0-1
    attention_coherence: float  # 0-1
    intentional_coherence: float  # 0-1
    information_flow: float  # 0-1
    hive_mind_score: float  # 0-100
    phase: str  # dormant/stirring/aware/synchronized/transcendent
    dominant_topics: List[str] = field(default_factory=list)
    outlier_agents: List[str] = field(default_factory=list)
    contagions: List[ThoughtContagion] = field(default_factory=list)


@dataclass
class ConsciousnessReport:
    """Full analysis report across all ticks."""
    snapshots: List[ConsciousnessSnapshot] = field(default_factory=list)
    overall_score: float = 0.0
    peak_phase: str = "dormant"
    phase_history: List[Tuple[int, str]] = field(default_factory=list)
    top_contagions: List[ThoughtContagion] = field(default_factory=list)
    belief_clusters: Dict[str, List[str]] = field(default_factory=dict)
    attention_fragmentation_events: int = 0
    synchronization_events: int = 0


# ---------------------------------------------------------------------------
# Phase thresholds
# ---------------------------------------------------------------------------

PHASE_THRESHOLDS: List[Tuple[float, str]] = [
    (85.0, "transcendent"),
    (65.0, "synchronized"),
    (45.0, "aware"),
    (25.0, "stirring"),
    (0.0, "dormant"),
]

PHASE_ORDER = ["dormant", "stirring", "aware", "synchronized", "transcendent"]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SwarmConsciousnessEngine:
    """Autonomous collective awareness detection for multi-agent swarms."""

    def __init__(
        self,
        agents: List[str],
        topics: Optional[List[str]] = None,
        belief_decay: float = 0.95,
        attention_window: int = 5,
    ) -> None:
        self.agents = list(agents)
        self.topics = topics or []
        self.belief_decay = belief_decay
        self.attention_window = attention_window

        # State stores
        self._beliefs: Dict[str, Dict[str, AgentBelief]] = defaultdict(dict)  # agent -> topic -> belief
        self._attention: Dict[str, List[AttentionEvent]] = defaultdict(list)  # agent -> events
        self._intents: Dict[str, IntentSignal] = {}  # agent -> latest intent
        self._belief_history: List[Dict[str, Dict[str, float]]] = []  # tick -> agent -> topic -> value
        self._contagions: List[ThoughtContagion] = []

        # Time
        self._tick: int = 0
        self._snapshots: List[ConsciousnessSnapshot] = []

    @property
    def tick_count(self) -> int:
        return self._tick

    @property
    def snapshots(self) -> List[ConsciousnessSnapshot]:
        return list(self._snapshots)

    def submit_belief(self, agent_id: str, topic: str, value: float, confidence: float = 1.0) -> None:
        """Agent declares a belief about a topic (-1.0 to 1.0)."""
        value = max(-1.0, min(1.0, value))
        confidence = max(0.0, min(1.0, confidence))
        belief = AgentBelief(
            agent_id=agent_id,
            topic=topic,
            value=value,
            confidence=confidence,
            timestamp=time.time(),
        )
        self._beliefs[agent_id][topic] = belief
        if topic not in self.topics:
            self.topics.append(topic)

    def submit_attention(self, agent_id: str, topic: str, intensity: float = 1.0) -> None:
        """Agent reports what it's currently focused on."""
        intensity = max(0.0, min(1.0, intensity))
        event = AttentionEvent(
            agent_id=agent_id,
            focus_topic=topic,
            intensity=intensity,
            timestamp=time.time(),
        )
        self._attention[agent_id].append(event)
        # Keep only recent window
        if len(self._attention[agent_id]) > self.attention_window * 3:
            self._attention[agent_id] = self._attention[agent_id][-self.attention_window * 3:]

    def submit_intent(self, agent_id: str, goal: str, action: str) -> None:
        """Agent declares its current goal and action."""
        self._intents[agent_id] = IntentSignal(
            agent_id=agent_id,
            goal=goal,
            action=action,
            timestamp=time.time(),
        )

    def tick(self) -> ConsciousnessSnapshot:
        """Advance one time step, compute all metrics, return snapshot."""
        self._tick += 1

        # Decay old beliefs slightly
        for agent_id in self._beliefs:
            for topic in self._beliefs[agent_id]:
                b = self._beliefs[agent_id][topic]
                b.confidence *= self.belief_decay

        # Compute metrics
        belief_align = self.compute_belief_alignment()
        attention_coh = self.compute_attention_coherence()
        intent_coh = self.compute_intentional_coherence()
        info_flow = self.compute_information_flow()

        # Composite score (weighted average)
        hive_score = (
            belief_align * 30.0
            + attention_coh * 25.0
            + intent_coh * 25.0
            + info_flow * 20.0
        )
        hive_score = max(0.0, min(100.0, hive_score))

        phase = self.classify_phase(hive_score)
        dominant = self._get_dominant_topics()
        outliers = self._detect_outliers()
        contagions = self.detect_contagions()

        snapshot = ConsciousnessSnapshot(
            tick=self._tick,
            belief_alignment=round(belief_align, 4),
            attention_coherence=round(attention_coh, 4),
            intentional_coherence=round(intent_coh, 4),
            information_flow=round(info_flow, 4),
            hive_mind_score=round(hive_score, 2),
            phase=phase,
            dominant_topics=dominant,
            outlier_agents=outliers,
            contagions=contagions,
        )
        self._snapshots.append(snapshot)

        # Record belief state for contagion tracking
        tick_beliefs: Dict[str, Dict[str, float]] = {}
        for agent_id, topics_map in self._beliefs.items():
            tick_beliefs[agent_id] = {t: b.value for t, b in topics_map.items()}
        self._belief_history.append(tick_beliefs)

        return snapshot

    def compute_belief_alignment(self) -> float:
        """How aligned are agents' beliefs? High = consensus, Low = disagreement."""
        if not self._beliefs:
            return 0.0

        topic_scores: List[float] = []
        all_topics = set()
        for agent_beliefs in self._beliefs.values():
            all_topics.update(agent_beliefs.keys())

        for topic in all_topics:
            values = []
            for agent_id in self.agents:
                if agent_id in self._beliefs and topic in self._beliefs[agent_id]:
                    b = self._beliefs[agent_id][topic]
                    values.append(b.value * b.confidence)
            if len(values) >= 2:
                # Alignment = 1 - normalized_std_dev
                std = statistics.stdev(values)
                # Max possible std for [-1,1] range is 1.0
                alignment = max(0.0, 1.0 - std)
                topic_scores.append(alignment)
            elif len(values) == 1:
                topic_scores.append(0.5)  # single belief, partial alignment

        if not topic_scores:
            return 0.0
        return statistics.mean(topic_scores)

    def compute_attention_coherence(self) -> float:
        """How focused is the swarm collectively? (entropy-based, low entropy = high coherence)."""
        # Get latest attention for each agent
        current_focus: List[str] = []
        for agent_id in self.agents:
            events = self._attention.get(agent_id, [])
            if events:
                latest = events[-1]
                current_focus.append(latest.focus_topic)

        if not current_focus:
            return 0.0

        # Compute normalized entropy
        counts = Counter(current_focus)
        total = len(current_focus)
        n_topics = len(counts)

        if n_topics <= 1:
            return 1.0  # all focused on same thing

        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(n_topics)
        if max_entropy == 0:
            return 1.0

        # Coherence = 1 - normalized_entropy
        coherence = 1.0 - (entropy / max_entropy)
        return max(0.0, min(1.0, coherence))

    def compute_intentional_coherence(self) -> float:
        """Are agents pursuing the same goals? High = shared purpose."""
        if not self._intents:
            return 0.0

        goals = [sig.goal for sig in self._intents.values()]
        if not goals:
            return 0.0

        # Frequency of most common goal
        counts = Counter(goals)
        most_common_count = counts.most_common(1)[0][1]
        total = len(goals)

        # Also consider action alignment within same-goal groups
        goal_groups: Dict[str, List[str]] = defaultdict(list)
        for sig in self._intents.values():
            goal_groups[sig.goal].append(sig.action)

        action_alignment = 0.0
        for goal, actions in goal_groups.items():
            if len(actions) >= 2:
                action_counts = Counter(actions)
                dominant_action = action_counts.most_common(1)[0][1]
                action_alignment += dominant_action / len(actions)
            else:
                action_alignment += 0.5
        action_alignment /= max(1, len(goal_groups))

        goal_coherence = most_common_count / total
        # Blend goal coherence and action alignment
        return 0.6 * goal_coherence + 0.4 * action_alignment

    def compute_information_flow(self) -> float:
        """How well does information propagate? Based on belief diversity reaching agents."""
        if len(self._belief_history) < 2:
            return 0.5  # neutral when not enough history

        # Measure: how many agents have beliefs on the same topics (coverage)
        topic_coverage: Dict[str, int] = defaultdict(int)
        for agent_beliefs in self._beliefs.values():
            for topic in agent_beliefs:
                topic_coverage[topic] += 1

        if not topic_coverage:
            return 0.0

        n_agents = len(self.agents)
        coverage_scores = [count / n_agents for count in topic_coverage.values()]
        avg_coverage = statistics.mean(coverage_scores)

        # Also measure belief convergence speed (are beliefs getting more similar over time?)
        convergence_bonus = 0.0
        if len(self._belief_history) >= 3:
            recent = self._belief_history[-1]
            older = self._belief_history[-3] if len(self._belief_history) >= 3 else self._belief_history[0]
            # Check if variance decreased
            for topic in self.topics:
                recent_vals = [recent.get(a, {}).get(topic, 0) for a in self.agents if a in recent and topic in recent.get(a, {})]
                older_vals = [older.get(a, {}).get(topic, 0) for a in self.agents if a in older and topic in older.get(a, {})]
                if len(recent_vals) >= 2 and len(older_vals) >= 2:
                    recent_std = statistics.stdev(recent_vals)
                    older_std = statistics.stdev(older_vals)
                    if older_std > 0 and recent_std < older_std:
                        convergence_bonus += 0.1

        convergence_bonus = min(0.3, convergence_bonus)
        return min(1.0, avg_coverage + convergence_bonus)

    def detect_contagions(self) -> List[ThoughtContagion]:
        """Detect belief spreading patterns across agents."""
        if len(self._belief_history) < 3:
            return []

        contagions: List[ThoughtContagion] = []

        for topic in self.topics:
            # Track which agents adopted this belief over time
            adoption_timeline: List[Set[str]] = []
            for tick_beliefs in self._belief_history[-10:]:  # last 10 ticks
                adopters = set()
                for agent_id, beliefs in tick_beliefs.items():
                    if topic in beliefs and abs(beliefs[topic]) > 0.3:
                        adopters.add(agent_id)
                adoption_timeline.append(adopters)

            if len(adoption_timeline) < 2:
                continue

            # Check if adoption grew
            first_adopters = adoption_timeline[0]
            last_adopters = adoption_timeline[-1]
            new_adopters = last_adopters - first_adopters

            if len(new_adopters) >= 2:
                # Find origin (earliest adopter)
                origin = None
                for adopters in adoption_timeline:
                    if adopters:
                        origin = sorted(adopters)[0]
                        break

                n_ticks = len(adoption_timeline)
                spread_rate = len(new_adopters) / max(1, n_ticks)
                virality = min(1.0, len(last_adopters) / max(1, len(self.agents)))

                # Check mutations (value drift from origin)
                mutations = 0
                if origin and origin in self._beliefs and topic in self._beliefs[origin]:
                    origin_val = self._beliefs[origin][topic].value
                    for agent_id in new_adopters:
                        if agent_id in self._beliefs and topic in self._beliefs[agent_id]:
                            if abs(self._beliefs[agent_id][topic].value - origin_val) > 0.3:
                                mutations += 1

                contagion = ThoughtContagion(
                    belief_topic=topic,
                    origin_agent=origin or "unknown",
                    infected_agents=sorted(new_adopters),
                    spread_rate=round(spread_rate, 3),
                    virality=round(virality, 3),
                    mutations=mutations,
                )
                contagions.append(contagion)

        self._contagions.extend(contagions)
        return contagions

    def classify_phase(self, score: float) -> str:
        """Map hive mind score to consciousness phase."""
        for threshold, phase in PHASE_THRESHOLDS:
            if score >= threshold:
                return phase
        return "dormant"

    def _get_dominant_topics(self) -> List[str]:
        """Get topics with most attention/belief activity."""
        topic_activity: Counter = Counter()
        for agent_events in self._attention.values():
            for event in agent_events[-self.attention_window:]:
                topic_activity[event.focus_topic] += event.intensity
        for agent_beliefs in self._beliefs.values():
            for topic, belief in agent_beliefs.items():
                topic_activity[topic] += belief.confidence

        return [t for t, _ in topic_activity.most_common(3)]

    def _detect_outliers(self) -> List[str]:
        """Detect agents whose beliefs diverge significantly from the swarm."""
        outliers: Set[str] = set()

        for topic in self.topics:
            values = []
            agent_vals: List[Tuple[str, float]] = []
            for agent_id in self.agents:
                if agent_id in self._beliefs and topic in self._beliefs[agent_id]:
                    v = self._beliefs[agent_id][topic].value
                    values.append(v)
                    agent_vals.append((agent_id, v))

            if len(values) < 3:
                continue

            mean = statistics.mean(values)
            std = statistics.stdev(values)
            if std < 0.01:
                continue

            for agent_id, val in agent_vals:
                z_score = abs(val - mean) / std
                if z_score > 1.8:
                    outliers.add(agent_id)

        return sorted(outliers)

    def analyze(self) -> ConsciousnessReport:
        """Generate full analysis report across all recorded snapshots."""
        if not self._snapshots:
            return ConsciousnessReport()

        scores = [s.hive_mind_score for s in self._snapshots]
        overall = statistics.mean(scores) if scores else 0.0

        # Peak phase
        peak_score = max(scores) if scores else 0.0
        peak_phase = self.classify_phase(peak_score)

        # Phase history
        phase_history: List[Tuple[int, str]] = []
        prev_phase = ""
        for s in self._snapshots:
            if s.phase != prev_phase:
                phase_history.append((s.tick, s.phase))
                prev_phase = s.phase

        # Belief clusters
        belief_clusters: Dict[str, List[str]] = {}
        for topic in self.topics:
            aligned: List[str] = []
            values = []
            for agent_id in self.agents:
                if agent_id in self._beliefs and topic in self._beliefs[agent_id]:
                    values.append(self._beliefs[agent_id][topic].value)

            if len(values) < 2:
                continue
            mean_val = statistics.mean(values)
            for agent_id in self.agents:
                if agent_id in self._beliefs and topic in self._beliefs[agent_id]:
                    if abs(self._beliefs[agent_id][topic].value - mean_val) < 0.3:
                        aligned.append(agent_id)
            if aligned:
                belief_clusters[topic] = aligned

        # Top contagions by virality
        all_contagions = sorted(self._contagions, key=lambda c: c.virality, reverse=True)
        top_contagions = all_contagions[:5]

        # Count events
        frag_events = sum(1 for s in self._snapshots if s.attention_coherence < 0.3)
        sync_events = sum(1 for s in self._snapshots if s.phase in ("synchronized", "transcendent"))

        return ConsciousnessReport(
            snapshots=self._snapshots,
            overall_score=round(overall, 2),
            peak_phase=peak_phase,
            phase_history=phase_history,
            top_contagions=top_contagions,
            belief_clusters=belief_clusters,
            attention_fragmentation_events=frag_events,
            synchronization_events=sync_events,
        )

    def export_json(self, path: str) -> None:
        """Export analysis as JSON."""
        report = self.analyze()
        data = {
            "overall_score": report.overall_score,
            "peak_phase": report.peak_phase,
            "phase_history": [{"tick": t, "phase": p} for t, p in report.phase_history],
            "attention_fragmentation_events": report.attention_fragmentation_events,
            "synchronization_events": report.synchronization_events,
            "belief_clusters": report.belief_clusters,
            "top_contagions": [asdict(c) for c in report.top_contagions],
            "snapshots": [asdict(s) for s in report.snapshots],
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def export_html(self, path: str) -> None:
        """Generate interactive HTML dashboard."""
        report = self.analyze()
        e = html_mod.escape

        # Phase colors
        phase_colors = {
            "dormant": "#6b7280",
            "stirring": "#f59e0b",
            "aware": "#3b82f6",
            "synchronized": "#10b981",
            "transcendent": "#8b5cf6",
        }

        # Build score timeline SVG
        chart_w, chart_h = 700, 200
        if report.snapshots:
            n = len(report.snapshots)
            x_step = chart_w / max(1, n - 1) if n > 1 else chart_w
            points = []
            for i, s in enumerate(report.snapshots):
                x = i * x_step
                y = chart_h - (s.hive_mind_score / 100.0 * chart_h)
                points.append(f"{x:.1f},{y:.1f}")
            polyline = " ".join(points)
        else:
            polyline = "0,200"

        # Build phase timeline
        phase_blocks = ""
        if report.phase_history:
            n_ticks = self._tick or 1
            for i, (tick, phase) in enumerate(report.phase_history):
                end_tick = report.phase_history[i + 1][0] if i + 1 < len(report.phase_history) else n_ticks
                x_start = (tick / n_ticks) * 100
                width = ((end_tick - tick) / n_ticks) * 100
                color = phase_colors.get(phase, "#6b7280")
                phase_blocks += (
                    f'<div style="position:absolute;left:{x_start:.1f}%;width:{width:.1f}%;'
                    f'height:100%;background:{color};opacity:0.7" '
                    f'title="{e(phase)} (tick {tick}-{end_tick})"></div>'
                )

        # Build belief alignment table
        belief_rows = ""
        for topic in self.topics[:10]:
            agents_in = []
            for agent_id in self.agents:
                if agent_id in self._beliefs and topic in self._beliefs[agent_id]:
                    v = self._beliefs[agent_id][topic].value
                    agents_in.append(f"{agent_id}: {v:.2f}")
            belief_rows += f"<tr><td>{e(topic)}</td><td>{', '.join(agents_in[:5])}</td></tr>"

        # Build contagion table
        contagion_rows = ""
        for c in report.top_contagions[:5]:
            contagion_rows += (
                f"<tr><td>{e(c.belief_topic)}</td><td>{e(c.origin_agent)}</td>"
                f"<td>{len(c.infected_agents)}</td><td>{c.virality:.2f}</td>"
                f"<td>{c.mutations}</td></tr>"
            )

        # Current state
        current = report.snapshots[-1] if report.snapshots else None
        current_score = current.hive_mind_score if current else 0
        current_phase = current.phase if current else "dormant"
        phase_color = phase_colors.get(current_phase, "#6b7280")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Consciousness Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #0f172a; color: #e2e8f0; padding: 24px; }}
.header {{ text-align: center; margin-bottom: 32px; }}
.header h1 {{ font-size: 28px; color: #f8fafc; margin-bottom: 8px; }}
.header .subtitle {{ color: #94a3b8; font-size: 14px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
          gap: 20px; margin-bottom: 24px; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 20px;
          border: 1px solid #334155; }}
.card h2 {{ font-size: 16px; color: #94a3b8; margin-bottom: 12px;
            text-transform: uppercase; letter-spacing: 1px; }}
.score-big {{ font-size: 56px; font-weight: 700; text-align: center; }}
.phase-badge {{ display: inline-block; padding: 6px 16px; border-radius: 20px;
                font-weight: 600; font-size: 14px; text-transform: uppercase;
                letter-spacing: 1px; }}
.metric {{ display: flex; justify-content: space-between; padding: 8px 0;
           border-bottom: 1px solid #334155; }}
.metric:last-child {{ border-bottom: none; }}
.metric-label {{ color: #94a3b8; }}
.metric-value {{ font-weight: 600; }}
.phase-timeline {{ position: relative; height: 30px; background: #0f172a;
                   border-radius: 6px; overflow: hidden; margin-top: 8px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; }}
th {{ color: #94a3b8; font-size: 12px; text-transform: uppercase; }}
svg {{ width: 100%; height: auto; }}
.chart-container {{ background: #0f172a; border-radius: 8px; padding: 12px; }}
</style>
</head>
<body>
<div class="header">
    <h1>[*] Swarm Consciousness Engine</h1>
    <div class="subtitle">Autonomous Collective Awareness Detection &mdash; {len(self.agents)} agents, {self._tick} ticks</div>
</div>

<div class="grid">
    <div class="card" style="text-align:center">
        <h2>Hive Mind Score</h2>
        <div class="score-big" style="color:{phase_color}">{current_score:.0f}</div>
        <div style="margin-top:8px">
            <span class="phase-badge" style="background:{phase_color}33;color:{phase_color}">{current_phase}</span>
        </div>
    </div>
    <div class="card">
        <h2>Component Scores</h2>
        <div class="metric"><span class="metric-label">Belief Alignment</span>
            <span class="metric-value">{current.belief_alignment if current else 0:.2f}</span></div>
        <div class="metric"><span class="metric-label">Attention Coherence</span>
            <span class="metric-value">{current.attention_coherence if current else 0:.2f}</span></div>
        <div class="metric"><span class="metric-label">Intentional Coherence</span>
            <span class="metric-value">{current.intentional_coherence if current else 0:.2f}</span></div>
        <div class="metric"><span class="metric-label">Information Flow</span>
            <span class="metric-value">{current.information_flow if current else 0:.2f}</span></div>
    </div>
    <div class="card">
        <h2>Report Summary</h2>
        <div class="metric"><span class="metric-label">Overall Score</span>
            <span class="metric-value">{report.overall_score:.1f}</span></div>
        <div class="metric"><span class="metric-label">Peak Phase</span>
            <span class="metric-value">{report.peak_phase}</span></div>
        <div class="metric"><span class="metric-label">Sync Events</span>
            <span class="metric-value">{report.synchronization_events}</span></div>
        <div class="metric"><span class="metric-label">Fragmentation Events</span>
            <span class="metric-value">{report.attention_fragmentation_events}</span></div>
        <div class="metric"><span class="metric-label">Contagions Detected</span>
            <span class="metric-value">{len(self._contagions)}</span></div>
    </div>
</div>

<div class="card" style="margin-bottom:20px">
    <h2>Coherence Timeline</h2>
    <div class="chart-container">
        <svg viewBox="0 0 {chart_w} {chart_h}" preserveAspectRatio="none">
            <rect width="{chart_w}" height="{chart_h}" fill="#0f172a"/>
            <!-- Grid lines -->
            <line x1="0" y1="{chart_h*0.25}" x2="{chart_w}" y2="{chart_h*0.25}" stroke="#334155" stroke-dasharray="4"/>
            <line x1="0" y1="{chart_h*0.5}" x2="{chart_w}" y2="{chart_h*0.5}" stroke="#334155" stroke-dasharray="4"/>
            <line x1="0" y1="{chart_h*0.75}" x2="{chart_w}" y2="{chart_h*0.75}" stroke="#334155" stroke-dasharray="4"/>
            <!-- Score line -->
            <polyline points="{polyline}" fill="none" stroke="{phase_color}" stroke-width="2.5"/>
        </svg>
    </div>
</div>

<div class="card" style="margin-bottom:20px">
    <h2>Phase Timeline</h2>
    <div class="phase-timeline">{phase_blocks}</div>
    <div style="display:flex;gap:12px;margin-top:8px;flex-wrap:wrap">
        {"".join(f'<span style="font-size:12px;color:{c}">● {p}</span>' for p, c in phase_colors.items())}
    </div>
</div>

<div class="grid">
    <div class="card">
        <h2>Belief Landscape</h2>
        <table><thead><tr><th>Topic</th><th>Agent Beliefs</th></tr></thead>
        <tbody>{belief_rows if belief_rows else "<tr><td colspan='2'>No beliefs recorded</td></tr>"}</tbody>
        </table>
    </div>
    <div class="card">
        <h2>Thought Contagions</h2>
        <table><thead><tr><th>Topic</th><th>Origin</th><th>Infected</th><th>Virality</th><th>Mutations</th></tr></thead>
        <tbody>{contagion_rows if contagion_rows else "<tr><td colspan='5'>No contagions detected</td></tr>"}</tbody>
        </table>
    </div>
</div>

<div class="card" style="margin-top:20px">
    <h2>Outlier Agents</h2>
    <p style="color:#94a3b8;font-size:14px">
    {', '.join(current.outlier_agents) if current and current.outlier_agents else 'No outliers detected — swarm is coherent'}
    </p>
</div>

</body>
</html>"""
        Path(path).write_text(html_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Demo Simulation
# ---------------------------------------------------------------------------


def run_demo(n_agents: int = 8, n_ticks: int = 60) -> SwarmConsciousnessEngine:
    """Run an interesting demo simulation showing phase transitions."""
    agents = [f"agent-{i}" for i in range(1, n_agents + 1)]
    topics = ["strategy", "resource_alloc", "threat_level", "expansion", "defense"]

    engine = SwarmConsciousnessEngine(agents=agents, topics=topics)
    random.seed(42)

    for tick in range(n_ticks):
        # Phase 1 (ticks 0-15): Chaotic — random beliefs and attention
        if tick < 15:
            for agent in agents:
                topic = random.choice(topics)
                engine.submit_belief(agent, topic, random.uniform(-1, 1), random.uniform(0.3, 0.9))
                engine.submit_attention(agent, random.choice(topics), random.uniform(0.3, 0.8))
                engine.submit_intent(agent, random.choice(["explore", "gather", "defend", "expand"]),
                                     random.choice(["move", "scan", "build", "fight"]))

        # Phase 2 (ticks 15-30): Convergence begins — agents start aligning on "strategy"
        elif tick < 30:
            consensus_val = 0.6 + random.uniform(-0.1, 0.1)
            for agent in agents:
                if random.random() < 0.6 + (tick - 15) * 0.02:
                    engine.submit_belief(agent, "strategy", consensus_val + random.uniform(-0.15, 0.15), 0.8)
                    engine.submit_attention(agent, "strategy", random.uniform(0.6, 0.9))
                    engine.submit_intent(agent, "optimize", random.choice(["refine", "test", "measure"]))
                else:
                    engine.submit_belief(agent, random.choice(topics), random.uniform(-0.5, 0.5), 0.5)
                    engine.submit_attention(agent, random.choice(topics), random.uniform(0.3, 0.6))
                    engine.submit_intent(agent, random.choice(["explore", "gather"]), "scan")

        # Phase 3 (ticks 30-45): High synchronization — most agents aligned
        elif tick < 45:
            for agent in agents:
                if random.random() < 0.85:
                    engine.submit_belief(agent, "strategy", 0.7 + random.uniform(-0.05, 0.05), 0.95)
                    engine.submit_belief(agent, "resource_alloc", 0.5 + random.uniform(-0.1, 0.1), 0.85)
                    engine.submit_attention(agent, "strategy", 0.9)
                    engine.submit_intent(agent, "optimize", "execute")
                else:
                    # Outlier agent
                    engine.submit_belief(agent, "strategy", -0.5, 0.7)
                    engine.submit_attention(agent, "defense", 0.8)
                    engine.submit_intent(agent, "defend", "fortify")

        # Phase 4 (ticks 45-60): Disruption and recovery
        else:
            disruption_strength = max(0, 1.0 - (tick - 45) * 0.07)
            for agent in agents:
                if random.random() < disruption_strength:
                    engine.submit_belief(agent, "threat_level", random.uniform(0.5, 1.0), 0.9)
                    engine.submit_attention(agent, "threat_level", 0.95)
                    engine.submit_intent(agent, "defend", "evade")
                else:
                    engine.submit_belief(agent, "strategy", 0.6 + random.uniform(-0.1, 0.1), 0.8)
                    engine.submit_attention(agent, "strategy", 0.7)
                    engine.submit_intent(agent, "optimize", "adapt")

        engine.tick()

    return engine


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm Consciousness Engine — collective awareness detection"
    )
    parser.add_argument("--agents", type=int, default=8, help="Number of agents (default: 8)")
    parser.add_argument("--ticks", type=int, default=60, help="Simulation ticks (default: 60)")
    parser.add_argument("--out", type=str, default=None, help="Export HTML report path")
    parser.add_argument("--json", type=str, default=None, help="Export JSON report path")
    args = parser.parse_args()

    print("[*] Swarm Consciousness Engine")
    print("=" * 50)
    print(f"Agents: {args.agents} | Ticks: {args.ticks}")
    print()

    engine = run_demo(n_agents=args.agents, n_ticks=args.ticks)
    report = engine.analyze()

    print(f"Overall Hive Mind Score: {report.overall_score:.1f}/100")
    print(f"Peak Phase: {report.peak_phase}")
    print(f"Synchronization Events: {report.synchronization_events}")
    print(f"Fragmentation Events: {report.attention_fragmentation_events}")
    print(f"Thought Contagions: {len(report.top_contagions)}")
    print()

    print("Phase History:")
    for tick, phase in report.phase_history:
        print(f"  Tick {tick:3d}: {phase}")
    print()

    if report.top_contagions:
        print("Top Thought Contagions:")
        for c in report.top_contagions[:3]:
            print(f"  [{c.belief_topic}] origin={c.origin_agent}, "
                  f"infected={len(c.infected_agents)}, virality={c.virality:.2f}")
    print()

    last = engine.snapshots[-1] if engine.snapshots else None
    if last:
        print("Final State:")
        print(f"  Belief Alignment:      {last.belief_alignment:.3f}")
        print(f"  Attention Coherence:   {last.attention_coherence:.3f}")
        print(f"  Intentional Coherence: {last.intentional_coherence:.3f}")
        print(f"  Information Flow:      {last.information_flow:.3f}")
        print(f"  Hive Mind Score:       {last.hive_mind_score:.1f}")
        print(f"  Phase:                 {last.phase}")
        if last.outlier_agents:
            print(f"  Outliers:              {', '.join(last.outlier_agents)}")

    if args.out:
        engine.export_html(args.out)
        print(f"\n[OK] HTML report exported to {args.out}")

    if args.json:
        engine.export_json(args.json)
        print(f"[OK] JSON report exported to {args.json}")


if __name__ == "__main__":
    main()
