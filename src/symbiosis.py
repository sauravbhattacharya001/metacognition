"""Swarm Symbiosis Engine — autonomous inter-agent symbiotic relationship detection.

Biologically-inspired by ecological symbiosis (mutualism, parasitism, commensalism),
this engine autonomously monitors inter-agent interactions, classifies relationship
types, detects exploitative patterns, and recommends mutualistic pairings.

Capabilities:

- **Interaction Tracker** — records inter-agent interactions with benefit/cost
  outcomes for each party, building a comprehensive interaction history.
- **Relationship Classifier** — analyzes rolling benefit averages to classify
  agent pairs into 5 ecological relationship types (mutualism, commensalism,
  parasitism, amensalism, competition) with confidence scoring.
- **Dependency Analyzer** — builds a dependency graph, classifies relationships
  as obligate (agent can't function alone) vs facultative (better together)
  based on solo-performance degradation analysis.
- **Parasitism Detector** — identifies exploitative agents draining resources,
  flags for quarantine with severity scoring and evidence chains.
- **Mutualism Optimizer** — identifies high-value mutualistic pairs and
  recommends strengthening strategies (co-scheduling, resource sharing).
- **Ecosystem Health Scorer** — composite health score 0-100 based on
  mutualism ratio, parasitism prevalence, dependency balance, diversity.
- **Interactive HTML Dashboard** — relationship network, parasite alerts,
  mutualism opportunities, health gauge, interaction timeline.

Usage (Python API)::

    from src.symbiosis import SymbiosisEngine

    engine = SymbiosisEngine()

    # Record interactions between agents
    engine.record_interaction("agent-1", "agent-2", benefit_a=0.8, benefit_b=0.7, context="task collab")
    engine.record_interaction("agent-1", "agent-2", benefit_a=0.9, benefit_b=0.6, context="resource share")
    engine.record_interaction("agent-3", "agent-4", benefit_a=0.9, benefit_b=-0.5, context="resource drain")

    # Analyze all relationships
    report = engine.analyze()
    print(report.ecosystem_health_score)   # 0-100
    print(report.relationships)            # classified pairs
    print(report.parasite_alerts)          # exploitation warnings
    print(report.mutualism_opportunities)  # strengthening suggestions

    engine.export_html("symbiosis_report.html")

CLI::

    python -m src.symbiosis                          # demo with simulated agents
    python -m src.symbiosis --agents 10              # number of agents
    python -m src.symbiosis --interactions 200       # interaction count
    python -m src.symbiosis --out report.html --json symbiosis.json
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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------

class RelationshipType(str, Enum):
    """Ecological relationship types between agent pairs."""
    MUTUALISM = "mutualism"          # +/+ both benefit
    COMMENSALISM = "commensalism"    # +/0 one benefits, other unaffected
    PARASITISM = "parasitism"        # +/- one benefits at other's expense
    AMENSALISM = "amensalism"        # 0/- one harmed, other unaffected
    COMPETITION = "competition"      # -/- both harmed


class DependencyType(str, Enum):
    """How dependent an agent is on its partner."""
    OBLIGATE = "obligate"        # can't function alone (>50% drop)
    FACULTATIVE = "facultative"  # better together (10-50% drop)
    NONE = "none"                # independent


@dataclass
class Interaction:
    """A single recorded interaction between two agents."""
    agent_a: str
    agent_b: str
    timestamp: float
    benefit_a: float  # -1.0 to 1.0
    benefit_b: float  # -1.0 to 1.0
    context: str = ""

    def __post_init__(self) -> None:
        self.benefit_a = max(-1.0, min(1.0, self.benefit_a))
        self.benefit_b = max(-1.0, min(1.0, self.benefit_b))


@dataclass
class SymbioticRelationship:
    """Classified relationship between two agents."""
    agent_a: str
    agent_b: str
    relationship_type: RelationshipType
    confidence: float  # 0-1
    interaction_count: int
    avg_benefit_a: float
    avg_benefit_b: float
    dependency_type_a: DependencyType = DependencyType.NONE
    dependency_type_b: DependencyType = DependencyType.NONE
    first_seen: float = 0.0
    last_seen: float = 0.0


@dataclass
class ParasiteAlert:
    """Alert for detected parasitic exploitation."""
    parasite_id: str
    host_id: str
    severity: float  # 0-1
    drain_rate: float
    interaction_count: int
    evidence_contexts: List[str] = field(default_factory=list)
    recommended_action: str = "quarantine"


@dataclass
class MutualismOpportunity:
    """Suggestion to strengthen a mutualistic pairing."""
    agent_a: str
    agent_b: str
    potential_benefit: float
    strategy: str
    current_strength: float = 0.0


@dataclass
class SymbiosisReport:
    """Full symbiosis analysis report."""
    relationships: List[SymbioticRelationship]
    parasite_alerts: List[ParasiteAlert]
    mutualism_opportunities: List[MutualismOpportunity]
    ecosystem_health_score: float
    dependency_graph: Dict[str, List[Dict[str, Any]]]
    relationship_distribution: Dict[str, int]
    insights: List[str]
    agent_count: int = 0
    interaction_count: int = 0


# ---------------------------------------------------------------------------
# Classification Thresholds
# ---------------------------------------------------------------------------

BENEFIT_THRESHOLD = 0.2      # above this = positive benefit
HARM_THRESHOLD = -0.2        # below this = negative (harmed)
NEUTRAL_BAND = 0.2           # within ±this of zero = unaffected
MIN_INTERACTIONS = 3         # minimum for classification
OBLIGATE_DROP = 0.5          # >50% solo performance drop
FACULTATIVE_DROP = 0.1       # 10-50% solo performance drop


# ---------------------------------------------------------------------------
# Symbiosis Engine
# ---------------------------------------------------------------------------

class SymbiosisEngine:
    """Autonomous inter-agent symbiotic relationship detection and management."""

    def __init__(
        self,
        benefit_threshold: float = BENEFIT_THRESHOLD,
        harm_threshold: float = HARM_THRESHOLD,
        min_interactions: int = MIN_INTERACTIONS,
    ) -> None:
        self.benefit_threshold = benefit_threshold
        self.harm_threshold = harm_threshold
        self.min_interactions = min_interactions
        self.interactions: List[Interaction] = []
        self._agent_solo_performance: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Interaction Recording
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        agent_a: str,
        agent_b: str,
        benefit_a: float,
        benefit_b: float,
        context: str = "",
        timestamp: Optional[float] = None,
    ) -> Interaction:
        """Record an interaction between two agents."""
        ts = timestamp if timestamp is not None else time.time()
        interaction = Interaction(
            agent_a=agent_a,
            agent_b=agent_b,
            timestamp=ts,
            benefit_a=benefit_a,
            benefit_b=benefit_b,
            context=context,
        )
        self.interactions.append(interaction)
        return interaction

    def record_solo_performance(self, agent_id: str, score: float) -> None:
        """Record an agent's solo performance (without partner interactions)."""
        self._agent_solo_performance[agent_id].append(score)

    def get_interactions_between(self, agent_a: str, agent_b: str) -> List[Interaction]:
        """Get all interactions between two specific agents (order-independent)."""
        results = []
        for ix in self.interactions:
            if (ix.agent_a == agent_a and ix.agent_b == agent_b) or \
               (ix.agent_a == agent_b and ix.agent_b == agent_a):
                results.append(ix)
        return results

    def get_agent_ids(self) -> Set[str]:
        """Get all unique agent IDs from interaction history."""
        agents: Set[str] = set()
        for ix in self.interactions:
            agents.add(ix.agent_a)
            agents.add(ix.agent_b)
        return agents

    # ------------------------------------------------------------------
    # Relationship Classification
    # ------------------------------------------------------------------

    def classify_relationship(self, agent_a: str, agent_b: str) -> Optional[SymbioticRelationship]:
        """Classify the relationship between two agents based on interaction history."""
        interactions = self.get_interactions_between(agent_a, agent_b)
        if len(interactions) < self.min_interactions:
            return None

        # Compute benefits from agent_a's perspective and agent_b's perspective
        benefits_a: List[float] = []
        benefits_b: List[float] = []
        for ix in interactions:
            if ix.agent_a == agent_a:
                benefits_a.append(ix.benefit_a)
                benefits_b.append(ix.benefit_b)
            else:
                benefits_a.append(ix.benefit_b)
                benefits_b.append(ix.benefit_a)

        avg_a = statistics.mean(benefits_a)
        avg_b = statistics.mean(benefits_b)

        # Classify based on averages
        rel_type = self._classify_from_averages(avg_a, avg_b)

        # Confidence based on interaction count and consistency
        consistency = 1.0 - (
            (statistics.stdev(benefits_a) if len(benefits_a) > 1 else 0.0) +
            (statistics.stdev(benefits_b) if len(benefits_b) > 1 else 0.0)
        ) / 2.0
        count_factor = min(1.0, len(interactions) / 10.0)
        confidence = max(0.0, min(1.0, consistency * count_factor))

        # Dependency analysis
        dep_a = self._assess_dependency(agent_a, avg_a)
        dep_b = self._assess_dependency(agent_b, avg_b)

        return SymbioticRelationship(
            agent_a=agent_a,
            agent_b=agent_b,
            relationship_type=rel_type,
            confidence=confidence,
            interaction_count=len(interactions),
            avg_benefit_a=round(avg_a, 4),
            avg_benefit_b=round(avg_b, 4),
            dependency_type_a=dep_a,
            dependency_type_b=dep_b,
            first_seen=interactions[0].timestamp,
            last_seen=interactions[-1].timestamp,
        )

    def _classify_from_averages(self, avg_a: float, avg_b: float) -> RelationshipType:
        """Classify relationship type from benefit averages."""
        a_positive = avg_a > self.benefit_threshold
        a_negative = avg_a < self.harm_threshold
        a_neutral = not a_positive and not a_negative

        b_positive = avg_b > self.benefit_threshold
        b_negative = avg_b < self.harm_threshold
        b_neutral = not b_positive and not b_negative

        if a_positive and b_positive:
            return RelationshipType.MUTUALISM
        elif a_positive and b_neutral:
            return RelationshipType.COMMENSALISM
        elif b_positive and a_neutral:
            return RelationshipType.COMMENSALISM
        elif a_positive and b_negative:
            return RelationshipType.PARASITISM
        elif b_positive and a_negative:
            return RelationshipType.PARASITISM
        elif a_neutral and b_negative:
            return RelationshipType.AMENSALISM
        elif b_neutral and a_negative:
            return RelationshipType.AMENSALISM
        elif a_negative and b_negative:
            return RelationshipType.COMPETITION
        else:
            # Both neutral — default to commensalism (benign)
            return RelationshipType.COMMENSALISM

    def _assess_dependency(self, agent_id: str, avg_benefit: float) -> DependencyType:
        """Assess how dependent an agent is on its partner interactions."""
        solo_scores = self._agent_solo_performance.get(agent_id, [])
        if not solo_scores or avg_benefit <= 0:
            return DependencyType.NONE

        avg_solo = statistics.mean(solo_scores)
        if avg_solo <= 0:
            return DependencyType.OBLIGATE if avg_benefit > 0.5 else DependencyType.FACULTATIVE

        # Estimate performance with partner vs solo
        # If benefit is high relative to solo performance, dependency exists
        relative_boost = avg_benefit / max(avg_solo, 0.01)
        if relative_boost > OBLIGATE_DROP:
            return DependencyType.OBLIGATE
        elif relative_boost > FACULTATIVE_DROP:
            return DependencyType.FACULTATIVE
        return DependencyType.NONE

    # ------------------------------------------------------------------
    # Parasitism Detection
    # ------------------------------------------------------------------

    def detect_parasites(self) -> List[ParasiteAlert]:
        """Detect agents exhibiting parasitic behavior."""
        alerts: List[ParasiteAlert] = []
        agents = self.get_agent_ids()
        pair_checked: Set[Tuple[str, str]] = set()

        for agent in agents:
            for other in agents:
                if agent == other:
                    continue
                pair = tuple(sorted([agent, other]))
                if pair in pair_checked:
                    continue
                pair_checked.add(pair)

                interactions = self.get_interactions_between(agent, other)
                if len(interactions) < self.min_interactions:
                    continue

                # Compute from agent's perspective
                benefits_self: List[float] = []
                benefits_other: List[float] = []
                contexts: List[str] = []
                for ix in interactions:
                    if ix.agent_a == agent:
                        benefits_self.append(ix.benefit_a)
                        benefits_other.append(ix.benefit_b)
                    else:
                        benefits_self.append(ix.benefit_b)
                        benefits_other.append(ix.benefit_a)
                    if ix.context:
                        contexts.append(ix.context)

                avg_self = statistics.mean(benefits_self)
                avg_other = statistics.mean(benefits_other)

                # Parasitism: one gains significantly, other loses
                if avg_self > self.benefit_threshold and avg_other < self.harm_threshold:
                    severity = min(1.0, (avg_self - avg_other) / 2.0)
                    drain_rate = abs(avg_other)
                    alerts.append(ParasiteAlert(
                        parasite_id=agent,
                        host_id=other,
                        severity=round(severity, 3),
                        drain_rate=round(drain_rate, 3),
                        interaction_count=len(interactions),
                        evidence_contexts=contexts[:5],
                        recommended_action="quarantine" if severity > 0.7 else "monitor",
                    ))
                elif avg_other > self.benefit_threshold and avg_self < self.harm_threshold:
                    severity = min(1.0, (avg_other - avg_self) / 2.0)
                    drain_rate = abs(avg_self)
                    alerts.append(ParasiteAlert(
                        parasite_id=other,
                        host_id=agent,
                        severity=round(severity, 3),
                        drain_rate=round(drain_rate, 3),
                        interaction_count=len(interactions),
                        evidence_contexts=contexts[:5],
                        recommended_action="quarantine" if severity > 0.7 else "monitor",
                    ))

        return alerts

    # ------------------------------------------------------------------
    # Mutualism Optimization
    # ------------------------------------------------------------------

    def find_mutualism_opportunities(self) -> List[MutualismOpportunity]:
        """Identify pairs with potential for stronger mutualistic relationships."""
        opportunities: List[MutualismOpportunity] = []
        agents = list(self.get_agent_ids())
        pair_checked: Set[Tuple[str, str]] = set()

        for i, agent_a in enumerate(agents):
            for agent_b in agents[i + 1:]:
                pair = tuple(sorted([agent_a, agent_b]))
                if pair in pair_checked:
                    continue
                pair_checked.add(pair)

                interactions = self.get_interactions_between(agent_a, agent_b)

                if len(interactions) >= self.min_interactions:
                    # Existing relationship — check if mutualistic and can be strengthened
                    benefits_a = []
                    benefits_b = []
                    for ix in interactions:
                        if ix.agent_a == agent_a:
                            benefits_a.append(ix.benefit_a)
                            benefits_b.append(ix.benefit_b)
                        else:
                            benefits_a.append(ix.benefit_b)
                            benefits_b.append(ix.benefit_a)

                    avg_a = statistics.mean(benefits_a)
                    avg_b = statistics.mean(benefits_b)

                    if avg_a > 0 and avg_b > 0:
                        # Already mutualistic — suggest strengthening
                        current = (avg_a + avg_b) / 2
                        potential = min(1.0, current * 1.5)
                        if potential > current + 0.1:
                            opportunities.append(MutualismOpportunity(
                                agent_a=agent_a,
                                agent_b=agent_b,
                                potential_benefit=round(potential, 3),
                                strategy="co-schedule: increase interaction frequency",
                                current_strength=round(current, 3),
                            ))
                elif 0 < len(interactions) < self.min_interactions:
                    # Few interactions — check if promising
                    benefits_a = []
                    benefits_b = []
                    for ix in interactions:
                        if ix.agent_a == agent_a:
                            benefits_a.append(ix.benefit_a)
                            benefits_b.append(ix.benefit_b)
                        else:
                            benefits_a.append(ix.benefit_b)
                            benefits_b.append(ix.benefit_a)
                    if benefits_a and benefits_b:
                        avg_a = statistics.mean(benefits_a)
                        avg_b = statistics.mean(benefits_b)
                        if avg_a > 0.1 and avg_b > 0.1:
                            opportunities.append(MutualismOpportunity(
                                agent_a=agent_a,
                                agent_b=agent_b,
                                potential_benefit=round((avg_a + avg_b) / 2, 3),
                                strategy="explore: increase interaction opportunities",
                                current_strength=round((avg_a + avg_b) / 2, 3),
                            ))

        # Sort by potential benefit descending
        opportunities.sort(key=lambda o: o.potential_benefit, reverse=True)
        return opportunities

    # ------------------------------------------------------------------
    # Dependency Graph
    # ------------------------------------------------------------------

    def build_dependency_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build a dependency graph showing agent interconnections."""
        graph: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        agents = list(self.get_agent_ids())

        for i, agent_a in enumerate(agents):
            for agent_b in agents[i + 1:]:
                rel = self.classify_relationship(agent_a, agent_b)
                if rel is None:
                    continue
                graph[agent_a].append({
                    "partner": agent_b,
                    "type": rel.relationship_type.value,
                    "dependency": rel.dependency_type_a.value,
                    "benefit": rel.avg_benefit_a,
                })
                graph[agent_b].append({
                    "partner": agent_a,
                    "type": rel.relationship_type.value,
                    "dependency": rel.dependency_type_b.value,
                    "benefit": rel.avg_benefit_b,
                })

        return dict(graph)

    # ------------------------------------------------------------------
    # Ecosystem Health Scoring
    # ------------------------------------------------------------------

    def compute_ecosystem_health(self, relationships: List[SymbioticRelationship]) -> float:
        """Compute ecosystem health score 0-100."""
        if not relationships:
            return 50.0  # neutral when no data

        # Distribution of relationship types
        type_counts: Dict[RelationshipType, int] = defaultdict(int)
        for rel in relationships:
            type_counts[rel.relationship_type] += 1

        total = len(relationships)

        # Factor 1: Mutualism ratio (higher = healthier) — weight 35%
        mutualism_ratio = type_counts.get(RelationshipType.MUTUALISM, 0) / total
        mutualism_score = mutualism_ratio * 100

        # Factor 2: Parasitism prevalence (lower = healthier) — weight 30%
        parasitism_ratio = type_counts.get(RelationshipType.PARASITISM, 0) / total
        parasitism_score = (1.0 - parasitism_ratio) * 100

        # Factor 3: Competition ratio (lower = healthier) — weight 15%
        competition_ratio = type_counts.get(RelationshipType.COMPETITION, 0) / total
        competition_score = (1.0 - competition_ratio) * 100

        # Factor 4: Relationship diversity (Shannon entropy) — weight 20%
        probs = [c / total for c in type_counts.values() if c > 0]
        max_entropy = math.log(len(RelationshipType)) if len(RelationshipType) > 1 else 1
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        diversity_score = (entropy / max_entropy) * 100 if max_entropy > 0 else 0

        health = (
            mutualism_score * 0.35 +
            parasitism_score * 0.30 +
            competition_score * 0.15 +
            diversity_score * 0.20
        )

        return round(max(0.0, min(100.0, health)), 1)

    # ------------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------------

    def analyze(self) -> SymbiosisReport:
        """Run full symbiosis analysis across all tracked agents."""
        agents = list(self.get_agent_ids())
        relationships: List[SymbioticRelationship] = []
        pair_checked: Set[Tuple[str, str]] = set()

        for i, agent_a in enumerate(agents):
            for agent_b in agents[i + 1:]:
                pair = tuple(sorted([agent_a, agent_b]))
                if pair in pair_checked:
                    continue
                pair_checked.add(pair)

                rel = self.classify_relationship(agent_a, agent_b)
                if rel is not None:
                    relationships.append(rel)

        parasite_alerts = self.detect_parasites()
        mutualism_opps = self.find_mutualism_opportunities()
        dependency_graph = self.build_dependency_graph()
        health_score = self.compute_ecosystem_health(relationships)

        # Distribution
        dist: Dict[str, int] = defaultdict(int)
        for rel in relationships:
            dist[rel.relationship_type.value] += 1

        # Generate insights
        insights = self._generate_insights(relationships, parasite_alerts, mutualism_opps, health_score)

        return SymbiosisReport(
            relationships=relationships,
            parasite_alerts=parasite_alerts,
            mutualism_opportunities=mutualism_opps,
            ecosystem_health_score=health_score,
            dependency_graph=dependency_graph,
            relationship_distribution=dict(dist),
            insights=insights,
            agent_count=len(agents),
            interaction_count=len(self.interactions),
        )

    def _generate_insights(
        self,
        relationships: List[SymbioticRelationship],
        alerts: List[ParasiteAlert],
        opportunities: List[MutualismOpportunity],
        health: float,
    ) -> List[str]:
        """Generate human-readable insights from the analysis."""
        insights: List[str] = []

        if not relationships:
            insights.append("No established relationships detected yet. More interactions needed.")
            return insights

        # Health assessment
        if health >= 80:
            insights.append(f"Ecosystem is thriving (health: {health}/100) — strong mutualistic bonds.")
        elif health >= 60:
            insights.append(f"Ecosystem is healthy (health: {health}/100) with room for improvement.")
        elif health >= 40:
            insights.append(f"Ecosystem needs attention (health: {health}/100) — imbalances detected.")
        else:
            insights.append(f"Ecosystem is stressed (health: {health}/100) — intervention recommended.")

        # Parasitism warnings
        if alerts:
            severe = [a for a in alerts if a.severity > 0.7]
            if severe:
                insights.append(
                    f"⚠️ {len(severe)} severe parasitism case(s) detected — quarantine recommended."
                )
            else:
                insights.append(f"{len(alerts)} mild parasitism case(s) detected — monitoring advised.")

        # Mutualism opportunities
        if opportunities:
            top = opportunities[0]
            insights.append(
                f"Top mutualism opportunity: {top.agent_a} ↔ {top.agent_b} "
                f"(potential: {top.potential_benefit:.2f})"
            )

        # Dependency warnings
        obligate_count = sum(
            1 for r in relationships
            if r.dependency_type_a == DependencyType.OBLIGATE or
               r.dependency_type_b == DependencyType.OBLIGATE
        )
        if obligate_count > 0:
            insights.append(
                f"⚠️ {obligate_count} obligate dependency detected — single point of failure risk."
            )

        return insights

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, report: SymbiosisReport) -> str:
        """Export report as JSON string."""
        data = {
            "ecosystem_health_score": report.ecosystem_health_score,
            "agent_count": report.agent_count,
            "interaction_count": report.interaction_count,
            "relationship_distribution": report.relationship_distribution,
            "relationships": [
                {
                    "agent_a": r.agent_a,
                    "agent_b": r.agent_b,
                    "type": r.relationship_type.value,
                    "confidence": r.confidence,
                    "interaction_count": r.interaction_count,
                    "avg_benefit_a": r.avg_benefit_a,
                    "avg_benefit_b": r.avg_benefit_b,
                    "dependency_a": r.dependency_type_a.value,
                    "dependency_b": r.dependency_type_b.value,
                }
                for r in report.relationships
            ],
            "parasite_alerts": [
                {
                    "parasite": a.parasite_id,
                    "host": a.host_id,
                    "severity": a.severity,
                    "drain_rate": a.drain_rate,
                    "action": a.recommended_action,
                }
                for a in report.parasite_alerts
            ],
            "mutualism_opportunities": [
                {
                    "agent_a": o.agent_a,
                    "agent_b": o.agent_b,
                    "potential": o.potential_benefit,
                    "strategy": o.strategy,
                }
                for o in report.mutualism_opportunities
            ],
            "insights": report.insights,
        }
        return json.dumps(data, indent=2)

    def export_html(self, path: str, report: Optional[SymbiosisReport] = None) -> str:
        """Export interactive HTML dashboard."""
        if report is None:
            report = self.analyze()

        rel_colors = {
            "mutualism": "#22c55e",
            "commensalism": "#3b82f6",
            "parasitism": "#ef4444",
            "amensalism": "#f59e0b",
            "competition": "#8b5cf6",
        }

        # Build relationship rows
        rel_rows = ""
        for r in report.relationships:
            color = rel_colors.get(r.relationship_type.value, "#666")
            rel_rows += f"""<tr>
                <td>{html_mod.escape(r.agent_a)}</td>
                <td>{html_mod.escape(r.agent_b)}</td>
                <td><span style="color:{color};font-weight:bold">{r.relationship_type.value}</span></td>
                <td>{r.confidence:.2f}</td>
                <td>{r.avg_benefit_a:+.3f}</td>
                <td>{r.avg_benefit_b:+.3f}</td>
                <td>{r.interaction_count}</td>
            </tr>"""

        # Build parasite alert cards
        alert_cards = ""
        for a in report.parasite_alerts:
            sev_color = "#ef4444" if a.severity > 0.7 else "#f59e0b"
            alert_cards += f"""<div style="border:2px solid {sev_color};border-radius:8px;padding:12px;margin:8px 0">
                <strong>🦠 {html_mod.escape(a.parasite_id)}</strong> → {html_mod.escape(a.host_id)}
                <br>Severity: <span style="color:{sev_color}">{a.severity:.1%}</span> | Drain: {a.drain_rate:.2f}/interaction
                <br>Action: <strong>{a.recommended_action}</strong>
            </div>"""

        if not alert_cards:
            alert_cards = "<p style='color:#22c55e'>✅ No parasitism detected</p>"

        # Mutualism opportunity cards
        opp_cards = ""
        for o in report.mutualism_opportunities[:5]:
            opp_cards += f"""<div style="border:1px solid #22c55e;border-radius:8px;padding:12px;margin:8px 0">
                <strong>{html_mod.escape(o.agent_a)}</strong> ↔ <strong>{html_mod.escape(o.agent_b)}</strong>
                <br>Potential: {o.potential_benefit:.2f} | Strategy: {html_mod.escape(o.strategy)}
            </div>"""

        if not opp_cards:
            opp_cards = "<p>No opportunities identified yet</p>"

        # Health gauge color
        h = report.ecosystem_health_score
        h_color = "#22c55e" if h >= 70 else "#f59e0b" if h >= 40 else "#ef4444"

        # Distribution bars
        dist_bars = ""
        max_count = max(report.relationship_distribution.values()) if report.relationship_distribution else 1
        for rtype, count in sorted(report.relationship_distribution.items(), key=lambda x: -x[1]):
            color = rel_colors.get(rtype, "#666")
            width = (count / max_count) * 100
            dist_bars += f"""<div style="margin:4px 0">
                <span style="display:inline-block;width:120px">{rtype}</span>
                <span style="display:inline-block;width:{width}%;background:{color};height:20px;border-radius:4px;vertical-align:middle"></span>
                <span style="margin-left:8px">{count}</span>
            </div>"""

        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Swarm Symbiosis Dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #0f172a; color: #e2e8f0; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #f1f5f9; border-bottom: 2px solid #334155; padding-bottom: 10px; }}
h2 {{ color: #94a3b8; margin-top: 30px; }}
.health-gauge {{ text-align: center; padding: 30px; background: #1e293b; border-radius: 12px; margin: 20px 0; }}
.health-score {{ font-size: 64px; font-weight: bold; color: {h_color}; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
.stat-card {{ background: #1e293b; padding: 16px; border-radius: 8px; text-align: center; }}
.stat-value {{ font-size: 28px; font-weight: bold; color: #f1f5f9; }}
.stat-label {{ color: #94a3b8; font-size: 14px; }}
table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; }}
th {{ background: #334155; padding: 12px; text-align: left; }}
td {{ padding: 10px 12px; border-top: 1px solid #334155; }}
.insights {{ background: #1e293b; padding: 16px; border-radius: 8px; margin: 20px 0; }}
.insight {{ padding: 8px 0; border-bottom: 1px solid #334155; }}
</style></head><body>
<div class="container">
<h1>🧬 Swarm Symbiosis Dashboard</h1>

<div class="health-gauge">
    <div class="health-score">{h}</div>
    <div style="color:#94a3b8">Ecosystem Health Score</div>
</div>

<div class="stats">
    <div class="stat-card"><div class="stat-value">{report.agent_count}</div><div class="stat-label">Agents</div></div>
    <div class="stat-card"><div class="stat-value">{report.interaction_count}</div><div class="stat-label">Interactions</div></div>
    <div class="stat-card"><div class="stat-value">{len(report.relationships)}</div><div class="stat-label">Relationships</div></div>
    <div class="stat-card"><div class="stat-value">{len(report.parasite_alerts)}</div><div class="stat-label">Parasite Alerts</div></div>
</div>

<h2>📊 Relationship Distribution</h2>
<div style="background:#1e293b;padding:16px;border-radius:8px">{dist_bars}</div>

<h2>🔗 Classified Relationships</h2>
<table>
<tr><th>Agent A</th><th>Agent B</th><th>Type</th><th>Confidence</th><th>Benefit A</th><th>Benefit B</th><th>Interactions</th></tr>
{rel_rows}
</table>

<h2>🦠 Parasitism Alerts</h2>
{alert_cards}

<h2>🤝 Mutualism Opportunities</h2>
{opp_cards}

<h2>💡 Insights</h2>
<div class="insights">
{"".join(f'<div class="insight">{html_mod.escape(i)}</div>' for i in report.insights)}
</div>

</div></body></html>"""

        Path(path).write_text(html_content, encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Demo / Simulation
# ---------------------------------------------------------------------------

def run_demo(num_agents: int = 8, num_interactions: int = 150) -> SymbiosisReport:
    """Run a demonstration with simulated agent interactions."""
    engine = SymbiosisEngine()
    agents = [f"agent-{i:02d}" for i in range(num_agents)]

    # Create diverse relationship patterns
    random.seed(42)
    base_time = 1700000000.0

    # Define some fixed relationship patterns for realism
    patterns: List[Tuple[int, int, float, float, str]] = []

    # Mutualistic pairs
    for _ in range(num_interactions // 4):
        a, b = random.sample(range(min(3, num_agents)), 2)
        patterns.append((a, b, random.uniform(0.4, 0.9), random.uniform(0.3, 0.8), "collaboration"))

    # Parasitic pairs
    for _ in range(num_interactions // 6):
        a = random.randint(3, min(4, num_agents - 1))
        b = random.randint(0, 2)
        patterns.append((a, b, random.uniform(0.5, 0.9), random.uniform(-0.8, -0.3), "resource drain"))

    # Competitive pairs
    for _ in range(num_interactions // 6):
        a, b = random.sample(range(4, min(6, num_agents)), 2) if num_agents > 5 else random.sample(range(num_agents), 2)
        patterns.append((a, b, random.uniform(-0.6, -0.2), random.uniform(-0.5, -0.2), "conflict"))

    # Commensal pairs
    for _ in range(num_interactions // 5):
        a = random.randint(0, num_agents - 1)
        b = random.randint(0, num_agents - 1)
        if a == b:
            b = (b + 1) % num_agents
        patterns.append((a, b, random.uniform(0.3, 0.7), random.uniform(-0.1, 0.1), "observation"))

    # Fill remaining with random
    while len(patterns) < num_interactions:
        a = random.randint(0, num_agents - 1)
        b = random.randint(0, num_agents - 1)
        if a == b:
            b = (b + 1) % num_agents
        patterns.append((a, b, random.uniform(-0.5, 0.8), random.uniform(-0.5, 0.8), "general"))

    for idx, (a_idx, b_idx, ben_a, ben_b, ctx) in enumerate(patterns[:num_interactions]):
        engine.record_interaction(
            agents[a_idx % num_agents],
            agents[b_idx % num_agents],
            benefit_a=ben_a,
            benefit_b=ben_b,
            context=ctx,
            timestamp=base_time + idx * 60,
        )

    # Record some solo performance for dependency analysis
    for agent in agents[:4]:
        for _ in range(5):
            engine.record_solo_performance(agent, random.uniform(0.2, 0.6))

    return engine.analyze()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Swarm Symbiosis Engine — inter-agent relationship analysis"
    )
    parser.add_argument("--agents", type=int, default=8, help="Number of agents to simulate")
    parser.add_argument("--interactions", type=int, default=150, help="Number of interactions")
    parser.add_argument("--out", type=str, default=None, help="Output HTML report path")
    parser.add_argument("--json", type=str, default=None, help="Output JSON report path")
    args = parser.parse_args()

    print("🧬 Swarm Symbiosis Engine")
    print("=" * 50)
    print(f"Simulating {args.agents} agents with {args.interactions} interactions...\n")

    SymbiosisEngine()
    # Run demo internally
    report = run_demo(num_agents=args.agents, num_interactions=args.interactions)

    print(f"Ecosystem Health Score: {report.ecosystem_health_score}/100")
    print(f"Agents: {report.agent_count} | Interactions: {report.interaction_count}")
    print(f"Relationships classified: {len(report.relationships)}")
    print(f"Parasite alerts: {len(report.parasite_alerts)}")
    print(f"Mutualism opportunities: {len(report.mutualism_opportunities)}")
    print()

    print("Relationship Distribution:")
    for rtype, count in sorted(report.relationship_distribution.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {rtype:14s} {bar} ({count})")
    print()

    if report.parasite_alerts:
        print("⚠️  Parasite Alerts:")
        for alert in report.parasite_alerts:
            print(f"  🦠 {alert.parasite_id} → {alert.host_id} "
                  f"(severity: {alert.severity:.0%}, action: {alert.recommended_action})")
        print()

    print("💡 Insights:")
    for insight in report.insights:
        print(f"  • {insight}")

    # Export
    if args.out:
        # Need to recreate engine for export
        eng2 = SymbiosisEngine()
        eng2.export_html(args.out, report)
        print(f"\n📄 HTML report: {args.out}")

    if args.json:
        eng2 = SymbiosisEngine()
        json_str = eng2.export_json(report)
        Path(args.json).write_text(json_str, encoding="utf-8")
        print(f"📋 JSON report: {args.json}")


if __name__ == "__main__":
    main()
