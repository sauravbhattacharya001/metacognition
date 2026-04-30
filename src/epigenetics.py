"""Swarm Epigenetics Engine — autonomous behavioral modification tracking.

Biologically inspired by epigenetics, where organisms modify gene expression
without changing DNA sequence.  This engine applies the concept to multi-agent
swarms: agents have behavioral "genes" (capabilities) that can be activated or
silenced by environmental signals, without modifying agent code.  Epigenetic
marks are inherited across agent generations with configurable fidelity.

Capabilities:

- **Epigenetic Marks** — 4 mark types (methylation/acetylation/phosphorylation/
  ubiquitination) that modulate gene expression strength up or down.
- **Gene Expression** — 12 default behavioral genes across 6 categories
  (cognitive, social, motor, sensory, metabolic, defensive) with dynamic
  expression levels computed from accumulated marks.
- **Environmental Signals** — 7 signal types (stress/abundance/threat/
  cooperation/isolation/novelty/competition) that trigger mark deposition.
- **Transgenerational Inheritance** — marks propagate to offspring with
  configurable noise and decay, enabling Lamarckian-style adaptation.
- **Mark Decay** — marks fade over time unless reinforced, modeling
  epigenetic reprogramming.
- **Fitness Landscape** — composite fitness from gene expression profiles,
  with penalties for silencing essential genes.
- **Health Scoring** — 0-100 composite measuring diversity, fitness,
  inheritance fidelity, and stress resilience.
- **Interactive HTML Dashboard** — Chart.js visualizations: expression
  heatmap, mark timeline, inheritance tree, fitness landscape, environment
  history.

Usage (Python API)::

    from src.epigenetics import EpigeneticsEngine, SignalType

    engine = EpigeneticsEngine(seed=42)
    for i in range(10):
        engine.register_agent(f"agent-{i}")

    # Emit environmental stress
    engine.emit_signal(SignalType.STRESS, intensity=0.8, duration=5,
                       target_genes=["resilience", "adaptability"])

    # Advance time
    for _ in range(20):
        engine.tick()

    # Reproduce with epigenetic inheritance
    engine.reproduce("agent-0", "child-0")

    # Analyze
    report = engine.analyze()
    print(report.health_score)
    print(report.insights)

    engine.export_html("epigenetics_report.html")

CLI::

    python -m src.epigenetics                         # default demo
    python -m src.epigenetics --agents 20 --generations 50
    python -m src.epigenetics --stress high           # high-stress environment
    python -m src.epigenetics --out report.html --json state.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Enums ────────────────────────────────────────────────────────────────


class MarkType(str, Enum):
    """Types of epigenetic marks."""
    METHYLATION = "methylation"          # silencing
    ACETYLATION = "acetylation"          # activation
    PHOSPHORYLATION = "phosphorylation"  # stress response
    UBIQUITINATION = "ubiquitination"    # degradation tag


class GeneCategory(str, Enum):
    """Functional categories of behavioral genes."""
    COGNITIVE = "cognitive"
    SOCIAL = "social"
    MOTOR = "motor"
    SENSORY = "sensory"
    METABOLIC = "metabolic"
    DEFENSIVE = "defensive"


class SignalType(str, Enum):
    """Environmental signal types."""
    STRESS = "stress"
    ABUNDANCE = "abundance"
    THREAT = "threat"
    COOPERATION = "cooperation"
    ISOLATION = "isolation"
    NOVELTY = "novelty"
    COMPETITION = "competition"


# ── Data Structures ──────────────────────────────────────────────────────


@dataclass
class EpigeneticMark:
    """A single epigenetic mark on a gene."""
    gene: str
    mark_type: MarkType
    strength: float  # 0.0 - 1.0
    source: str      # signal or event that caused it
    generation: int  # when applied
    heritable: bool = True
    decay_rate: float = 0.05
    age: int = 0     # ticks since applied


@dataclass
class Gene:
    """A behavioral gene with expression level."""
    name: str
    category: GeneCategory
    base_expression: float = 0.5   # baseline 0.0-1.0
    essential: bool = False        # silencing causes fitness penalty


@dataclass
class Epigenome:
    """Complete epigenetic state for one agent."""
    agent_id: str
    genes: Dict[str, Gene] = field(default_factory=dict)
    marks: List[EpigeneticMark] = field(default_factory=list)
    generation: int = 0
    parent_id: Optional[str] = None
    birth_tick: int = 0

    def get_expression(self, gene_name: str) -> float:
        """Compute effective expression of a gene considering all marks."""
        if gene_name not in self.genes:
            return 0.0
        gene = self.genes[gene_name]
        expr = gene.base_expression

        for mark in self.marks:
            if mark.gene != gene_name:
                continue
            if mark.mark_type == MarkType.METHYLATION:
                expr -= mark.strength * 0.6
            elif mark.mark_type == MarkType.ACETYLATION:
                expr += mark.strength * 0.5
            elif mark.mark_type == MarkType.PHOSPHORYLATION:
                expr += mark.strength * 0.3
            elif mark.mark_type == MarkType.UBIQUITINATION:
                expr -= mark.strength * 0.8

        return max(0.0, min(1.0, expr))

    def compute_fitness(self) -> float:
        """Compute overall fitness from expression profile."""
        if not self.genes:
            return 0.0
        total = 0.0
        penalty = 0.0
        for name, gene in self.genes.items():
            expr = self.get_expression(name)
            total += expr
            if gene.essential and expr < 0.2:
                penalty += (0.2 - expr) * 2.0
        avg_expr = total / len(self.genes)
        return max(0.0, min(1.0, avg_expr - penalty))

    def decay_marks(self) -> List[EpigeneticMark]:
        """Decay marks by one tick, remove expired ones. Returns removed marks."""
        remaining = []
        removed = []
        for mark in self.marks:
            mark.age += 1
            mark.strength -= mark.decay_rate
            if mark.strength <= 0.0:
                removed.append(mark)
            else:
                remaining.append(mark)
        self.marks = remaining
        return removed


@dataclass
class EnvironmentalSignal:
    """An active environmental signal affecting the swarm."""
    signal_type: SignalType
    intensity: float        # 0.0-1.0
    duration: int           # ticks remaining
    target_genes: List[str]
    mark_effect: MarkType
    started_tick: int = 0


@dataclass
class GenerationSnapshot:
    """Snapshot of epigenetic state at a point in time."""
    tick: int
    generation: int
    avg_fitness: float
    expression_profiles: Dict[str, Dict[str, float]]
    active_marks_count: int
    signal_count: int


@dataclass
class InheritanceEvent:
    """Record of epigenetic inheritance."""
    parent_id: str
    child_id: str
    tick: int
    marks_inherited: int
    marks_lost: int  # marks that failed to transfer
    fidelity: float  # fraction successfully inherited


@dataclass
class EpigeneticsReport:
    """Analysis report from the engine."""
    tick: int
    agent_count: int
    generation_count: int
    total_marks: int
    avg_fitness: float
    fitness_std: float
    expression_profiles: Dict[str, Dict[str, float]]
    silencing_patterns: Dict[str, float]  # gene -> fraction of agents with it silenced
    inheritance_fidelity: float
    stress_memory_score: float
    phenotypic_diversity: float
    fitness_landscape: Dict[str, float]  # agent -> fitness
    health_score: int  # 0-100
    health_breakdown: Dict[str, float]
    insights: List[str]
    mark_distribution: Dict[str, int]  # mark_type -> count


# ── Default Genome ───────────────────────────────────────────────────────

DEFAULT_GENOME: List[Tuple[str, GeneCategory, float, bool]] = [
    ("reasoning", GeneCategory.COGNITIVE, 0.6, True),
    ("planning", GeneCategory.COGNITIVE, 0.5, False),
    ("cooperation", GeneCategory.SOCIAL, 0.5, True),
    ("communication", GeneCategory.SOCIAL, 0.5, False),
    ("speed", GeneCategory.MOTOR, 0.5, False),
    ("precision", GeneCategory.MOTOR, 0.5, False),
    ("awareness", GeneCategory.SENSORY, 0.6, True),
    ("pattern_recognition", GeneCategory.SENSORY, 0.5, False),
    ("efficiency", GeneCategory.METABOLIC, 0.5, True),
    ("endurance", GeneCategory.METABOLIC, 0.5, False),
    ("resilience", GeneCategory.DEFENSIVE, 0.5, True),
    ("adaptability", GeneCategory.DEFENSIVE, 0.6, False),
]

# Signal → mark type mapping
SIGNAL_MARK_MAP: Dict[SignalType, MarkType] = {
    SignalType.STRESS: MarkType.PHOSPHORYLATION,
    SignalType.ABUNDANCE: MarkType.ACETYLATION,
    SignalType.THREAT: MarkType.PHOSPHORYLATION,
    SignalType.COOPERATION: MarkType.ACETYLATION,
    SignalType.ISOLATION: MarkType.METHYLATION,
    SignalType.NOVELTY: MarkType.ACETYLATION,
    SignalType.COMPETITION: MarkType.METHYLATION,
}

# Signal → default target genes when none specified
SIGNAL_DEFAULT_TARGETS: Dict[SignalType, List[str]] = {
    SignalType.STRESS: ["resilience", "adaptability", "endurance"],
    SignalType.ABUNDANCE: ["efficiency", "cooperation", "planning"],
    SignalType.THREAT: ["awareness", "speed", "resilience"],
    SignalType.COOPERATION: ["cooperation", "communication", "planning"],
    SignalType.ISOLATION: ["cooperation", "communication"],
    SignalType.NOVELTY: ["pattern_recognition", "adaptability", "reasoning"],
    SignalType.COMPETITION: ["cooperation", "communication", "efficiency"],
}


# ── Engine ───────────────────────────────────────────────────────────────


class EpigeneticsEngine:
    """Swarm Epigenetics Engine — tracks behavioral gene expression modification."""

    def __init__(
        self,
        inheritance_fidelity: float = 0.75,
        inheritance_noise: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.agents: Dict[str, Epigenome] = {}
        self.active_signals: List[EnvironmentalSignal] = []
        self.signal_history: List[EnvironmentalSignal] = []
        self.snapshots: List[GenerationSnapshot] = []
        self.inheritance_events: List[InheritanceEvent] = []
        self.tick_counter: int = 0
        self.generation_counter: int = 0
        self.inheritance_fidelity = inheritance_fidelity
        self.inheritance_noise = inheritance_noise
        self._rng = random.Random(seed)

    def register_agent(
        self,
        agent_id: str,
        genome: Optional[List[Tuple[str, GeneCategory, float, bool]]] = None,
        generation: int = 0,
    ) -> Epigenome:
        """Register a new agent with default or custom genome."""
        if genome is None:
            genome = DEFAULT_GENOME
        genes = {}
        for name, cat, base_expr, essential in genome:
            genes[name] = Gene(
                name=name,
                category=cat,
                base_expression=base_expr,
                essential=essential,
            )
        epigenome = Epigenome(
            agent_id=agent_id,
            genes=genes,
            generation=generation,
            birth_tick=self.tick_counter,
        )
        self.agents[agent_id] = epigenome
        return epigenome

    def emit_signal(
        self,
        signal_type: SignalType,
        intensity: float = 0.5,
        duration: int = 5,
        target_genes: Optional[List[str]] = None,
        mark_effect: Optional[MarkType] = None,
    ) -> EnvironmentalSignal:
        """Emit an environmental signal that affects agent epigenomes."""
        intensity = max(0.0, min(1.0, intensity))
        if target_genes is None:
            target_genes = SIGNAL_DEFAULT_TARGETS.get(signal_type, ["adaptability"])
        if mark_effect is None:
            mark_effect = SIGNAL_MARK_MAP.get(signal_type, MarkType.PHOSPHORYLATION)
        signal = EnvironmentalSignal(
            signal_type=signal_type,
            intensity=intensity,
            duration=duration,
            target_genes=target_genes,
            mark_effect=mark_effect,
            started_tick=self.tick_counter,
        )
        self.active_signals.append(signal)
        self.signal_history.append(signal)
        return signal

    def tick(self) -> None:
        """Advance simulation by one tick: apply signals, decay marks."""
        self.tick_counter += 1

        # Apply active signals to agents
        remaining_signals = []
        for signal in self.active_signals:
            signal.duration -= 1
            # Apply marks to all agents
            for epigenome in self.agents.values():
                for gene_name in signal.target_genes:
                    if gene_name in epigenome.genes:
                        # Probabilistic application based on intensity
                        if self._rng.random() < signal.intensity * 0.4:
                            mark = EpigeneticMark(
                                gene=gene_name,
                                mark_type=signal.mark_effect,
                                strength=signal.intensity * self._rng.uniform(0.3, 0.8),
                                source=signal.signal_type.value,
                                generation=epigenome.generation,
                                heritable=(signal.mark_effect != MarkType.UBIQUITINATION),
                                decay_rate=0.02 + self._rng.uniform(0, 0.04),
                            )
                            epigenome.marks.append(mark)
            if signal.duration > 0:
                remaining_signals.append(signal)
        self.active_signals = remaining_signals

        # Decay marks on all agents
        for epigenome in self.agents.values():
            epigenome.decay_marks()

        # Take periodic snapshots
        if self.tick_counter % 5 == 0:
            self._take_snapshot()

    def reproduce(
        self,
        parent_id: str,
        child_id: str,
        mutation_rate: float = 0.05,
    ) -> Optional[Epigenome]:
        """Create a child agent inheriting epigenetic marks from parent."""
        if parent_id not in self.agents:
            return None
        parent = self.agents[parent_id]
        self.generation_counter = max(self.generation_counter, parent.generation + 1)

        # Create child with same genome
        child = self.register_agent(
            child_id,
            genome=[(g.name, g.category, g.base_expression, g.essential)
                    for g in parent.genes.values()],
            generation=parent.generation + 1,
        )
        child.parent_id = parent_id

        # Inherit epigenetic marks with fidelity
        marks_inherited = 0
        marks_lost = 0
        for mark in parent.marks:
            if not mark.heritable:
                marks_lost += 1
                continue
            if self._rng.random() < self.inheritance_fidelity:
                # Inherit with noise
                new_strength = mark.strength + self._rng.gauss(0, self.inheritance_noise)
                new_strength = max(0.01, min(1.0, new_strength))
                inherited_mark = EpigeneticMark(
                    gene=mark.gene,
                    mark_type=mark.mark_type,
                    strength=new_strength,
                    source=f"inherited:{mark.source}",
                    generation=child.generation,
                    heritable=mark.heritable,
                    decay_rate=mark.decay_rate * (1 + mutation_rate),
                )
                child.marks.append(inherited_mark)
                marks_inherited += 1
            else:
                marks_lost += 1

        total = marks_inherited + marks_lost
        fidelity = marks_inherited / total if total > 0 else 1.0
        self.inheritance_events.append(InheritanceEvent(
            parent_id=parent_id,
            child_id=child_id,
            tick=self.tick_counter,
            marks_inherited=marks_inherited,
            marks_lost=marks_lost,
            fidelity=fidelity,
        ))
        return child

    def get_expression_profile(self, agent_id: str) -> Dict[str, float]:
        """Get current expression levels for all genes of an agent."""
        if agent_id not in self.agents:
            return {}
        epigenome = self.agents[agent_id]
        return {name: epigenome.get_expression(name) for name in epigenome.genes}

    def _take_snapshot(self) -> None:
        """Record a generation snapshot."""
        if not self.agents:
            return
        profiles = {}
        fitnesses = []
        for aid, epi in self.agents.items():
            profiles[aid] = {name: epi.get_expression(name) for name in epi.genes}
            fitnesses.append(epi.compute_fitness())
        self.snapshots.append(GenerationSnapshot(
            tick=self.tick_counter,
            generation=self.generation_counter,
            avg_fitness=statistics.mean(fitnesses) if fitnesses else 0.0,
            expression_profiles=profiles,
            active_marks_count=sum(len(e.marks) for e in self.agents.values()),
            signal_count=len(self.active_signals),
        ))

    def analyze(self) -> EpigeneticsReport:
        """Produce a comprehensive analysis report."""
        if not self.agents:
            return EpigeneticsReport(
                tick=self.tick_counter, agent_count=0, generation_count=0,
                total_marks=0, avg_fitness=0.0, fitness_std=0.0,
                expression_profiles={}, silencing_patterns={},
                inheritance_fidelity=0.0, stress_memory_score=0.0,
                phenotypic_diversity=0.0, fitness_landscape={},
                health_score=0, health_breakdown={}, insights=[],
                mark_distribution={},
            )

        # Expression profiles
        profiles: Dict[str, Dict[str, float]] = {}
        fitnesses: Dict[str, float] = {}
        for aid, epi in self.agents.items():
            profiles[aid] = {n: epi.get_expression(n) for n in epi.genes}
            fitnesses[aid] = epi.compute_fitness()

        fitness_values = list(fitnesses.values())
        avg_fitness = statistics.mean(fitness_values)
        fitness_std = statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0.0

        # Silencing patterns
        gene_names = list(next(iter(self.agents.values())).genes.keys())
        silencing: Dict[str, float] = {}
        for gn in gene_names:
            silenced_count = sum(
                1 for epi in self.agents.values()
                if epi.get_expression(gn) < 0.2
            )
            silencing[gn] = silenced_count / len(self.agents)

        # Mark distribution
        mark_dist: Dict[str, int] = defaultdict(int)
        total_marks = 0
        for epi in self.agents.values():
            for m in epi.marks:
                mark_dist[m.mark_type.value] += 1
                total_marks += 1

        # Inheritance fidelity
        if self.inheritance_events:
            avg_inh_fidelity = statistics.mean(
                e.fidelity for e in self.inheritance_events
            )
        else:
            avg_inh_fidelity = 1.0

        # Stress memory: fraction of inherited marks from stress
        stress_inherited = sum(
            1 for epi in self.agents.values()
            for m in epi.marks
            if "inherited" in m.source and "stress" in m.source
        )
        total_inherited = sum(
            1 for epi in self.agents.values()
            for m in epi.marks
            if "inherited" in m.source
        )
        stress_memory = stress_inherited / max(1, total_inherited)

        # Phenotypic diversity (Shannon entropy of expression vectors)
        diversity = self._compute_diversity(profiles)

        # Health scoring
        diversity_score = min(1.0, diversity / 2.0) * 25
        fitness_score = avg_fitness * 30
        fidelity_score = avg_inh_fidelity * 25
        balance_score = (1.0 - max(silencing.values(), default=0.0)) * 20
        health = int(diversity_score + fitness_score + fidelity_score + balance_score)
        health = max(0, min(100, health))

        # Insights
        insights = self._generate_insights(
            silencing, avg_fitness, fitness_std, diversity, stress_memory, avg_inh_fidelity
        )

        return EpigeneticsReport(
            tick=self.tick_counter,
            agent_count=len(self.agents),
            generation_count=self.generation_counter,
            total_marks=total_marks,
            avg_fitness=round(avg_fitness, 4),
            fitness_std=round(fitness_std, 4),
            expression_profiles=profiles,
            silencing_patterns=silencing,
            inheritance_fidelity=round(avg_inh_fidelity, 4),
            stress_memory_score=round(stress_memory, 4),
            phenotypic_diversity=round(diversity, 4),
            fitness_landscape=fitnesses,
            health_score=health,
            health_breakdown={
                "diversity": round(diversity_score, 2),
                "fitness": round(fitness_score, 2),
                "inheritance_fidelity": round(fidelity_score, 2),
                "expression_balance": round(balance_score, 2),
            },
            insights=insights,
            mark_distribution=dict(mark_dist),
        )

    def _compute_diversity(self, profiles: Dict[str, Dict[str, float]]) -> float:
        """Shannon entropy-based phenotypic diversity."""
        if len(profiles) < 2:
            return 0.0
        # Discretize expression into bins for entropy
        bins = 5
        gene_names = list(next(iter(profiles.values())).keys())
        total_entropy = 0.0
        for gn in gene_names:
            values = [p[gn] for p in profiles.values()]
            counts = [0] * bins
            for v in values:
                idx = min(int(v * bins), bins - 1)
                counts[idx] += 1
            n = len(values)
            entropy = 0.0
            for c in counts:
                if c > 0:
                    p = c / n
                    entropy -= p * math.log2(p)
            total_entropy += entropy
        return total_entropy / len(gene_names) if gene_names else 0.0

    def _generate_insights(
        self,
        silencing: Dict[str, float],
        avg_fitness: float,
        fitness_std: float,
        diversity: float,
        stress_memory: float,
        fidelity: float,
    ) -> List[str]:
        """Generate autonomous insights from analysis."""
        insights = []

        # Heavily silenced genes
        heavily_silenced = [g for g, v in silencing.items() if v > 0.5]
        if heavily_silenced:
            insights.append(
                f"Genes heavily silenced (>50% agents): {', '.join(heavily_silenced)}. "
                "Consider reducing silencing signals or these capabilities may atrophy."
            )

        # Low diversity
        if diversity < 0.5:
            insights.append(
                "Low phenotypic diversity detected. The swarm may be converging on a "
                "single behavioral phenotype, reducing adaptability."
            )

        # High fitness variance
        if fitness_std > 0.2:
            insights.append(
                f"High fitness variance ({fitness_std:.2f}) indicates unequal "
                "epigenetic burden — some agents are much fitter than others."
            )

        # Stress memory persistence
        if stress_memory > 0.3:
            insights.append(
                f"Strong transgenerational stress memory ({stress_memory:.0%}). "
                "Past stress is shaping offspring behavior — potentially adaptive "
                "or maladaptive depending on current environment."
            )

        # Low fitness
        if avg_fitness < 0.3:
            insights.append(
                f"Average fitness is critically low ({avg_fitness:.2f}). "
                "Excessive silencing or environmental pressure may be degrading "
                "the swarm's operational capacity."
            )

        # High fidelity
        if fidelity > 0.9 and self.inheritance_events:
            insights.append(
                "Very high inheritance fidelity (>90%). Epigenetic patterns are "
                "being faithfully transmitted — beneficial for stable environments, "
                "but may reduce adaptability to change."
            )

        # Low fidelity
        if fidelity < 0.4 and self.inheritance_events:
            insights.append(
                "Low inheritance fidelity (<40%). Most epigenetic marks are lost "
                "during reproduction — the swarm resets each generation."
            )

        if not insights:
            insights.append("Epigenetic state appears balanced. No critical patterns detected.")

        return insights

    def export_html(self, path: str) -> str:
        """Export interactive HTML dashboard."""
        report = self.analyze()
        gene_names = list(next(iter(self.agents.values())).genes.keys()) if self.agents else []
        agent_ids = sorted(self.agents.keys())

        # Expression heatmap data
        heatmap_data = []
        for aid in agent_ids:
            row = [report.expression_profiles.get(aid, {}).get(g, 0.0) for g in gene_names]
            heatmap_data.append(row)

        # Fitness data
        fitness_data = [report.fitness_landscape.get(aid, 0.0) for aid in agent_ids]

        # Snapshot timeline data
        timeline_ticks = [s.tick for s in self.snapshots]
        timeline_fitness = [s.avg_fitness for s in self.snapshots]
        timeline_marks = [s.active_marks_count for s in self.snapshots]

        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Swarm Epigenetics Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #0f1419; color: #e7e9ea; }}
.header {{ text-align: center; margin-bottom: 30px; }}
.header h1 {{ color: #1d9bf0; margin: 0; font-size: 2em; }}
.header .subtitle {{ color: #71767b; margin-top: 5px; }}
.tabs {{ display: flex; gap: 5px; margin-bottom: 20px; flex-wrap: wrap; }}
.tab {{ padding: 10px 20px; background: #1e2732; border: 1px solid #38444d; border-radius: 8px; cursor: pointer; color: #e7e9ea; }}
.tab.active {{ background: #1d9bf0; border-color: #1d9bf0; }}
.panel {{ display: none; background: #1e2732; border-radius: 12px; padding: 20px; }}
.panel.active {{ display: block; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
.metric {{ background: #273340; border-radius: 8px; padding: 15px; text-align: center; }}
.metric .value {{ font-size: 2em; color: #1d9bf0; font-weight: bold; }}
.metric .label {{ color: #71767b; font-size: 0.85em; margin-top: 5px; }}
.heatmap {{ overflow-x: auto; }}
.heatmap table {{ border-collapse: collapse; width: 100%; }}
.heatmap td, .heatmap th {{ padding: 8px; text-align: center; border: 1px solid #38444d; font-size: 0.8em; }}
.insight {{ padding: 10px 15px; background: #273340; border-left: 3px solid #1d9bf0; margin: 8px 0; border-radius: 4px; }}
canvas {{ max-height: 400px; }}
</style></head><body>
<div class="header">
<h1>🧬 Swarm Epigenetics Dashboard</h1>
<div class="subtitle">Tick {report.tick} | {report.agent_count} agents | Generation {report.generation_count}</div>
</div>

<div class="metrics">
<div class="metric"><div class="value">{report.health_score}</div><div class="label">Health Score</div></div>
<div class="metric"><div class="value">{report.avg_fitness:.2f}</div><div class="label">Avg Fitness</div></div>
<div class="metric"><div class="value">{report.total_marks}</div><div class="label">Active Marks</div></div>
<div class="metric"><div class="value">{report.phenotypic_diversity:.2f}</div><div class="label">Phenotypic Diversity</div></div>
<div class="metric"><div class="value">{report.inheritance_fidelity:.0%}</div><div class="label">Inheritance Fidelity</div></div>
</div>

<div class="tabs">
<div class="tab active" onclick="showTab(0)">Expression Heatmap</div>
<div class="tab" onclick="showTab(1)">Mark Timeline</div>
<div class="tab" onclick="showTab(2)">Fitness Landscape</div>
<div class="tab" onclick="showTab(3)">Environment History</div>
<div class="tab" onclick="showTab(4)">Insights</div>
</div>

<div class="panel active" id="panel-0">
<h3>Gene Expression Heatmap</h3>
<div class="heatmap"><table>
<tr><th>Agent</th>{''.join(f'<th>{html_mod.escape(g)}</th>' for g in gene_names)}</tr>
{''.join(f'<tr><td>{html_mod.escape(aid)}</td>' + ''.join(f'<td style="background:rgba(29,155,240,{v:.2f})">{v:.2f}</td>' for v in row) + '</tr>' for aid, row in zip(agent_ids, heatmap_data))}
</table></div>
</div>

<div class="panel" id="panel-1">
<h3>Marks & Fitness Over Time</h3>
<canvas id="timelineChart"></canvas>
</div>

<div class="panel" id="panel-2">
<h3>Agent Fitness Distribution</h3>
<canvas id="fitnessChart"></canvas>
</div>

<div class="panel" id="panel-3">
<h3>Environmental Signal History</h3>
<table style="width:100%;border-collapse:collapse">
<tr><th style="text-align:left;padding:8px;border-bottom:1px solid #38444d">Type</th><th>Intensity</th><th>Duration</th><th>Target Genes</th><th>Tick</th></tr>
{''.join(f'<tr><td style="padding:8px">{html_mod.escape(s.signal_type.value)}</td><td>{s.intensity:.2f}</td><td>{s.duration}</td><td>{html_mod.escape(", ".join(s.target_genes))}</td><td>{s.started_tick}</td></tr>' for s in self.signal_history[-20:])}
</table>
</div>

<div class="panel" id="panel-4">
<h3>Autonomous Insights</h3>
{''.join(f'<div class="insight">{html_mod.escape(i)}</div>' for i in report.insights)}
<h3>Mark Distribution</h3>
{''.join(f'<div class="insight"><strong>{html_mod.escape(k)}</strong>: {v} marks</div>' for k, v in report.mark_distribution.items())}
<h3>Silencing Patterns</h3>
{''.join(f'<div class="insight"><strong>{html_mod.escape(g)}</strong>: {v:.0%} agents silenced</div>' for g, v in sorted(report.silencing_patterns.items(), key=lambda x: -x[1]) if v > 0)}
</div>

<script>
function showTab(i) {{
  document.querySelectorAll('.tab').forEach((t,idx) => t.classList.toggle('active', idx===i));
  document.querySelectorAll('.panel').forEach((p,idx) => p.classList.toggle('active', idx===i));
}}

new Chart(document.getElementById('timelineChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(timeline_ticks)},
    datasets: [
      {{ label: 'Avg Fitness', data: {json.dumps(timeline_fitness)}, borderColor: '#1d9bf0', yAxisID: 'y' }},
      {{ label: 'Active Marks', data: {json.dumps(timeline_marks)}, borderColor: '#f97316', yAxisID: 'y1' }}
    ]
  }},
  options: {{
    scales: {{
      y: {{ type: 'linear', position: 'left', title: {{ display: true, text: 'Fitness' }} }},
      y1: {{ type: 'linear', position: 'right', title: {{ display: true, text: 'Marks' }}, grid: {{ drawOnChartArea: false }} }}
    }}
  }}
}});

new Chart(document.getElementById('fitnessChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(agent_ids)},
    datasets: [{{ label: 'Fitness', data: {json.dumps(fitness_data)}, backgroundColor: 'rgba(29,155,240,0.6)' }}]
  }},
  options: {{ scales: {{ y: {{ min: 0, max: 1 }} }} }}
}});
</script>
</body></html>"""

        Path(path).write_text(html_content, encoding="utf-8")
        return path

    def export_json(self, report: Optional[EpigeneticsReport] = None) -> str:
        """Export report as JSON string."""
        if report is None:
            report = self.analyze()
        return json.dumps(asdict(report), indent=2, default=str)


# ── Demo / Simulation ────────────────────────────────────────────────────


def run_demo(
    num_agents: int = 8,
    num_generations: int = 30,
    stress_level: str = "normal",
    seed: int = 42,
) -> Tuple[EpigeneticsEngine, EpigeneticsReport]:
    """Run a demonstration simulation."""
    engine = EpigeneticsEngine(seed=seed)

    # Register initial agents
    for i in range(num_agents):
        engine.register_agent(f"agent-{i:02d}")

    # Configure stress intensity
    stress_intensity = {"low": 0.3, "normal": 0.5, "high": 0.9}.get(stress_level, 0.5)

    # Simulate generations
    for gen in range(num_generations):
        # Emit environmental signals periodically
        rng = engine._rng
        if gen % 5 == 0:
            signal_type = rng.choice(list(SignalType))
            engine.emit_signal(signal_type, intensity=stress_intensity * rng.uniform(0.5, 1.0), duration=3)

        # Extra stress in high-stress mode
        if stress_level == "high" and gen % 3 == 0:
            engine.emit_signal(SignalType.STRESS, intensity=stress_intensity, duration=4)

        # Tick
        engine.tick()

        # Reproduce occasionally
        if gen % 7 == 0 and len(engine.agents) >= 2:
            parent_ids = list(engine.agents.keys())
            parent = rng.choice(parent_ids)
            child_id = f"child-{gen}-{rng.randint(0, 99):02d}"
            engine.reproduce(parent, child_id)

    report = engine.analyze()
    return engine, report


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Swarm Epigenetics Engine — behavioral gene expression tracking"
    )
    parser.add_argument("--agents", type=int, default=8, help="Number of initial agents")
    parser.add_argument("--generations", type=int, default=30, help="Simulation ticks")
    parser.add_argument("--stress", choices=["low", "normal", "high"], default="normal",
                        help="Environmental stress level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Output HTML report path")
    parser.add_argument("--json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    print("🧬 Swarm Epigenetics Engine")
    print("=" * 50)
    print(f"Agents: {args.agents} | Generations: {args.generations} | Stress: {args.stress}\n")

    engine, report = run_demo(
        num_agents=args.agents,
        num_generations=args.generations,
        stress_level=args.stress,
        seed=args.seed,
    )

    print(f"Health Score: {report.health_score}/100")
    print(f"Agents: {report.agent_count} | Generations: {report.generation_count}")
    print(f"Active Marks: {report.total_marks}")
    print(f"Avg Fitness: {report.avg_fitness:.3f} (σ={report.fitness_std:.3f})")
    print(f"Phenotypic Diversity: {report.phenotypic_diversity:.3f}")
    print(f"Inheritance Fidelity: {report.inheritance_fidelity:.0%}")
    print(f"Stress Memory: {report.stress_memory_score:.0%}")
    print()

    print("Mark Distribution:")
    for mt, count in sorted(report.mark_distribution.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 40)
        print(f"  {mt:20s} {bar} ({count})")
    print()

    print("Silencing Patterns (>10% agents):")
    for gene, frac in sorted(report.silencing_patterns.items(), key=lambda x: -x[1]):
        if frac > 0.1:
            print(f"  {gene:22s} {'▓' * int(frac * 20)} {frac:.0%}")
    print()

    print("💡 Insights:")
    for insight in report.insights:
        print(f"  • {insight}")

    if args.out:
        engine.export_html(args.out)
        print(f"\n📄 HTML report: {args.out}")

    if args.json:
        json_str = engine.export_json(report)
        Path(args.json).write_text(json_str, encoding="utf-8")
        print(f"📋 JSON report: {args.json}")


if __name__ == "__main__":
    main()
