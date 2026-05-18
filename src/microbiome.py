"""Swarm Microbiome Engine — autonomous commensal agent ecosystem management.

Biologically-inspired by the human microbiome (gut flora, skin microbiota,
oral communities).  Models commensal/beneficial "microbe" agents that provide
background services (nutrient processing, pathogen defense, signal modulation,
waste recycling, vitamin synthesis, immune training) to host agents.

Capabilities:

- **Colonization Engine** — microbe agents colonize host niches (gut, skin,
  oral, respiratory, urogenital, neural) with species-specific preferences
  and niche carrying capacities.
- **Metabolic Network Engine** — microbes produce metabolites (SCFAs, vitamins,
  neurotransmitters, bile acids) consumed by hosts.  Tracks production rates,
  cross-feeding between species, and metabolic pathway health.
- **Dysbiosis Detector** — monitors Shannon/Simpson diversity indices,
  pathogen-to-commensal ratio, keystone species loss, bloom events.
  Classifies: Eubiosis / Mild-Dysbiosis / Moderate-Dysbiosis / Severe-Dysbiosis / Critical.
- **Antibiotic Disruption Simulator** — models broad-spectrum and targeted
  antibiotic effects.  Broad kills indiscriminately (diversity crash),
  targeted kills specific species.  Tracks recovery trajectory.
- **Probiotic Intervention Engine** — introduces beneficial species, tracks
  engraftment success based on niche availability and competition.
- **Microbiome Health Scorer** — composite score 0-100 from diversity,
  pathogen ratio, metabolic output, niche balance, keystone presence.
  5 tiers: Thriving / Healthy / Stressed / Dysbiotic / Critical.
- **Insight Generator** — autonomous pattern detection: diversity trends,
  emerging pathogens, metabolic deficiencies, recovery progress.

Usage (Python API)::

    from src.microbiome import SwarmMicrobiomeEngine, MicrobeSpecies, Intervention

    engine = SwarmMicrobiomeEngine()
    engine.seed_species(MicrobeSpecies(name="Bacteroides", species_type="commensal",
        preferred_niches=["gut"], metabolites_produced=["butyrate"],
        antibiotic_susceptibility={"broad": 0.8, "metronidazole": 0.9}), "gut", 200.0)
    engine.tick(steps=50)
    report = engine.analyze()
    print(report.overall_health)
    engine.export_html("microbiome.html")

CLI::

    python -m src.microbiome                          # demo with defaults
    python -m src.microbiome --ticks 100              # simulation length
    python -m src.microbiome --scenario dysbiosis     # dysbiosis scenario
    python -m src.microbiome --scenario antibiotic    # antibiotic disruption
    python -m src.microbiome --scenario recovery      # recovery after antibiotic
    python -m src.microbiome --out report.html --json microbiome.json
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MicrobeSpecies:
    """A microbial species with ecological traits."""
    name: str
    species_type: str  # commensal, probiotic, pathogenic, opportunistic
    preferred_niches: List[str] = field(default_factory=list)
    metabolites_produced: List[str] = field(default_factory=list)
    antibiotic_susceptibility: Dict[str, float] = field(default_factory=dict)
    growth_rate: float = 0.08
    immune_evasion: float = 0.0  # 0-1, higher = harder for immune to control


@dataclass
class NicheState:
    """Current state of a body niche."""
    niche_name: str
    carrying_capacity: float
    populations: Dict[str, float] = field(default_factory=dict)  # species -> abundance
    diversity_index: float = 0.0
    simpson_index: float = 0.0


@dataclass
class Metabolite:
    """A metabolic product."""
    name: str
    producer_species: List[str] = field(default_factory=list)
    consumer_species: List[str] = field(default_factory=list)
    current_level: float = 0.0
    optimal_range: Tuple[float, float] = (0.0, 100.0)


@dataclass
class DysbiosisEvent:
    """A detected dysbiosis event."""
    tick: int
    niche: str
    event_type: str  # bloom, crash, invasion, keystone_loss
    severity: float  # 0-1
    details: str = ""


@dataclass
class Intervention:
    """An antibiotic/probiotic/prebiotic intervention."""
    intervention_type: str  # antibiotic, probiotic, prebiotic
    target_species: Optional[str] = None  # None = broad-spectrum
    spectrum: str = "broad"  # broad, narrow
    strength: float = 1.0
    species_to_introduce: Optional[MicrobeSpecies] = None  # for probiotic
    boost_species: Optional[List[str]] = None  # for prebiotic


@dataclass
class MicrobiomeSnapshot:
    """State at a single tick."""
    tick: int
    niche_states: Dict[str, Dict[str, float]]  # niche -> {species -> abundance}
    metabolite_levels: Dict[str, float]
    dysbiosis_events: List[Dict[str, Any]]
    health_score: float
    diversity_scores: Dict[str, float]  # niche -> Shannon
    dysbiosis_tier: str = "Eubiosis"
    total_population: float = 0.0


@dataclass
class MicrobiomeReport:
    """Full analysis report."""
    snapshots: List[MicrobiomeSnapshot] = field(default_factory=list)
    overall_health: float = 0.0
    tier: str = "Healthy"
    niche_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metabolic_health: float = 0.0
    intervention_history: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    peak_diversity: Dict[str, float] = field(default_factory=dict)
    total_dysbiosis_events: int = 0
    species_census: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NICHES: Dict[str, float] = {
    "gut": 1000.0,
    "skin": 500.0,
    "oral": 300.0,
    "respiratory": 200.0,
    "urogenital": 200.0,
    "neural": 100.0,
}

HEALTH_TIERS = [
    (80, "Thriving"),
    (60, "Healthy"),
    (40, "Stressed"),
    (20, "Dysbiotic"),
    (0, "Critical"),
]

DYSBIOSIS_TIERS = [
    (0.9, "Eubiosis"),
    (0.7, "Mild-Dysbiosis"),
    (0.5, "Moderate-Dysbiosis"),
    (0.3, "Severe-Dysbiosis"),
    (0.0, "Critical"),
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SwarmMicrobiomeEngine:
    """Autonomous commensal agent ecosystem management for multi-agent swarms."""

    def __init__(
        self,
        niches: Optional[Dict[str, float]] = None,
        num_host_agents: int = 5,
    ) -> None:
        self.num_host_agents = num_host_agents
        self._niche_configs = dict(niches or DEFAULT_NICHES)

        # State
        self._niches: Dict[str, NicheState] = {}
        for name, cap in self._niche_configs.items():
            self._niches[name] = NicheState(niche_name=name, carrying_capacity=cap)

        self._species_registry: Dict[str, MicrobeSpecies] = {}
        self._metabolites: Dict[str, Metabolite] = {}
        self._dysbiosis_events: List[DysbiosisEvent] = []
        self._interventions: List[Tuple[int, Intervention]] = []
        self._snapshots: List[MicrobiomeSnapshot] = []
        self._tick: int = 0
        self._active_antibiotics: List[Tuple[int, Intervention]] = []  # (expire_tick, intervention)
        self._keystone_species: Set[str] = set()

    @property
    def tick_count(self) -> int:
        return self._tick

    @property
    def niches(self) -> Dict[str, NicheState]:
        return dict(self._niches)

    @property
    def species(self) -> Dict[str, MicrobeSpecies]:
        return dict(self._species_registry)

    @property
    def metabolites(self) -> Dict[str, Metabolite]:
        return dict(self._metabolites)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def register_metabolite(self, name: str, producers: List[str],
                            consumers: Optional[List[str]] = None,
                            optimal_range: Tuple[float, float] = (0.0, 100.0)) -> Metabolite:
        """Register a metabolite in the network."""
        m = Metabolite(
            name=name,
            producer_species=list(producers),
            consumer_species=list(consumers or []),
            optimal_range=optimal_range,
        )
        self._metabolites[name] = m
        return m

    def mark_keystone(self, species_name: str) -> None:
        """Mark a species as keystone (critical for ecosystem stability)."""
        self._keystone_species.add(species_name)

    # ------------------------------------------------------------------
    # Colonization
    # ------------------------------------------------------------------

    def seed_species(self, species: MicrobeSpecies, niche: str,
                     initial_abundance: float = 100.0) -> None:
        """Introduce a species into a niche."""
        if niche not in self._niches:
            raise ValueError(f"Unknown niche: {niche}")
        self._species_registry[species.name] = species
        ns = self._niches[niche]
        ns.populations[species.name] = ns.populations.get(species.name, 0.0) + initial_abundance

    def get_abundance(self, species_name: str, niche: str) -> float:
        """Get current abundance of a species in a niche."""
        ns = self._niches.get(niche)
        if ns is None:
            return 0.0
        return ns.populations.get(species_name, 0.0)

    def get_total_population(self, niche: str) -> float:
        """Get total population in a niche."""
        ns = self._niches.get(niche)
        if ns is None:
            return 0.0
        return sum(ns.populations.values())

    # ------------------------------------------------------------------
    # Interventions
    # ------------------------------------------------------------------

    def apply_intervention(self, intervention: Intervention) -> None:
        """Apply an antibiotic, probiotic, or prebiotic intervention."""
        self._interventions.append((self._tick, intervention))

        if intervention.intervention_type == "antibiotic":
            self._apply_antibiotic(intervention)
        elif intervention.intervention_type == "probiotic":
            self._apply_probiotic(intervention)
        elif intervention.intervention_type == "prebiotic":
            self._apply_prebiotic(intervention)

    def _apply_antibiotic(self, intervention: Intervention) -> None:
        """Apply antibiotic effects across niches."""
        for ns in self._niches.values():
            species_to_remove: List[str] = []
            for sp_name, abundance in list(ns.populations.items()):
                sp = self._species_registry.get(sp_name)
                if sp is None:
                    continue

                if intervention.spectrum == "broad":
                    # Broad-spectrum: affects all based on susceptibility
                    suscept = sp.antibiotic_susceptibility.get("broad", 0.5)
                    kill_factor = suscept * intervention.strength
                    ns.populations[sp_name] = abundance * max(0.0, 1.0 - kill_factor)
                elif intervention.spectrum == "narrow" and intervention.target_species:
                    # Narrow: only target species
                    if sp_name == intervention.target_species:
                        suscept = max(sp.antibiotic_susceptibility.values()) if sp.antibiotic_susceptibility else 0.7
                        kill_factor = suscept * intervention.strength
                        ns.populations[sp_name] = abundance * max(0.0, 1.0 - kill_factor)

                if ns.populations[sp_name] < 0.5:
                    species_to_remove.append(sp_name)

            for sp_name in species_to_remove:
                del ns.populations[sp_name]

    def _apply_probiotic(self, intervention: Intervention) -> None:
        """Introduce probiotic species."""
        if intervention.species_to_introduce is None:
            return
        sp = intervention.species_to_introduce
        self._species_registry[sp.name] = sp
        # Try to engraft in preferred niches
        for niche_name in (sp.preferred_niches or list(self._niches.keys())[:1]):
            if niche_name not in self._niches:
                continue
            ns = self._niches[niche_name]
            total_pop = sum(ns.populations.values())
            # Engraftment success depends on available capacity
            available = max(0.0, ns.carrying_capacity - total_pop)
            engraftment = min(intervention.strength * 50, available * 0.5 + 10)
            ns.populations[sp.name] = ns.populations.get(sp.name, 0.0) + max(1.0, engraftment)

    def _apply_prebiotic(self, intervention: Intervention) -> None:
        """Boost targeted beneficial species growth."""
        if not intervention.boost_species:
            return
        for ns in self._niches.values():
            for sp_name in intervention.boost_species:
                if sp_name in ns.populations:
                    # Boost by 20-50% based on strength
                    boost = 1.0 + (0.2 * intervention.strength)
                    ns.populations[sp_name] *= boost

    # ------------------------------------------------------------------
    # Tick & Growth
    # ------------------------------------------------------------------

    def tick(self, steps: int = 1) -> MicrobiomeSnapshot:
        """Advance simulation by N steps. Returns final snapshot."""
        snapshot = None
        for _ in range(steps):
            snapshot = self._do_tick()
        return snapshot  # type: ignore[return-value]

    def _do_tick(self) -> MicrobiomeSnapshot:
        """Single tick: growth, metabolites, dysbiosis check."""
        self._tick += 1

        # Logistic growth for all species in all niches
        for ns in self._niches.values():
            total_pop = sum(ns.populations.values())
            new_pops: Dict[str, float] = {}
            for sp_name, abundance in ns.populations.items():
                sp = self._species_registry.get(sp_name)
                if sp is None:
                    new_pops[sp_name] = abundance
                    continue

                growth_rate = sp.growth_rate
                # Immune pressure on pathogens
                if sp.species_type == "pathogenic":
                    immune_effect = 0.02 * (1.0 - sp.immune_evasion)
                    growth_rate -= immune_effect

                # Logistic growth
                capacity_factor = 1.0 - (total_pop / ns.carrying_capacity)
                delta = abundance * growth_rate * capacity_factor
                # Random noise ±5%
                noise = random.uniform(-0.05, 0.05) * abundance
                new_abundance = abundance + delta + noise
                new_pops[sp_name] = max(0.0, new_abundance)

            # Remove extinct species
            ns.populations = {k: v for k, v in new_pops.items() if v >= 0.5}

            # Update diversity indices
            ns.diversity_index = self._shannon_diversity(ns)
            ns.simpson_index = self._simpson_diversity(ns)

        # Metabolite production
        self._update_metabolites()

        # Dysbiosis detection
        tick_events = self._detect_dysbiosis()

        # Build snapshot
        niche_states = {}
        diversity_scores = {}
        total_pop = 0.0
        for name, ns in self._niches.items():
            niche_states[name] = dict(ns.populations)
            diversity_scores[name] = ns.diversity_index
            total_pop += sum(ns.populations.values())

        metabolite_levels = {m.name: m.current_level for m in self._metabolites.values()}
        health = self._compute_health_score()
        self._classify_tier(health)

        snapshot = MicrobiomeSnapshot(
            tick=self._tick,
            niche_states=niche_states,
            metabolite_levels=metabolite_levels,
            dysbiosis_events=[
                {"tick": e.tick, "niche": e.niche, "type": e.event_type,
                 "severity": e.severity, "details": e.details}
                for e in tick_events
            ],
            health_score=health,
            diversity_scores=diversity_scores,
            dysbiosis_tier=self._classify_dysbiosis_tier(),
            total_population=total_pop,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def _update_metabolites(self) -> None:
        """Update metabolite levels based on producer abundance."""
        for m in self._metabolites.values():
            production = 0.0
            for sp_name in m.producer_species:
                # Sum abundance across all niches
                for ns in self._niches.values():
                    production += ns.populations.get(sp_name, 0.0) * 0.01

            consumption = 0.0
            for sp_name in m.consumer_species:
                for ns in self._niches.values():
                    consumption += ns.populations.get(sp_name, 0.0) * 0.005

            # Decay + production - consumption
            m.current_level = max(0.0, m.current_level * 0.95 + production - consumption)

    def _detect_dysbiosis(self) -> List[DysbiosisEvent]:
        """Detect dysbiosis events in all niches."""
        events: List[DysbiosisEvent] = []

        for ns in self._niches.values():
            total = sum(ns.populations.values())
            if total < 1.0:
                continue

            # Bloom detection: any species > 60% of niche
            for sp_name, abundance in ns.populations.items():
                ratio = abundance / total
                if ratio > 0.6:
                    sp = self._species_registry.get(sp_name)
                    sp_type = sp.species_type if sp else "unknown"
                    severity = min(1.0, (ratio - 0.6) * 2.5)
                    if sp_type == "pathogenic":
                        severity = min(1.0, severity + 0.3)
                    evt = DysbiosisEvent(
                        tick=self._tick, niche=ns.niche_name,
                        event_type="bloom", severity=severity,
                        details=f"{sp_name} bloom at {ratio:.1%} of {ns.niche_name}"
                    )
                    events.append(evt)
                    self._dysbiosis_events.append(evt)

            # Keystone loss detection
            for ks in self._keystone_species:
                if ks not in ns.populations and any(
                    ks in self._species_registry.get(ks, MicrobeSpecies(name=ks, species_type="unknown")).preferred_niches
                    for _ in [1]
                ):
                    # Check if keystone was recently present
                    was_present = False
                    for snap in self._snapshots[-5:]:
                        if ks in snap.niche_states.get(ns.niche_name, {}):
                            was_present = True
                            break
                    if was_present:
                        evt = DysbiosisEvent(
                            tick=self._tick, niche=ns.niche_name,
                            event_type="keystone_loss", severity=0.7,
                            details=f"Keystone species {ks} lost from {ns.niche_name}"
                        )
                        events.append(evt)
                        self._dysbiosis_events.append(evt)

            # Diversity crash: Shannon < 0.5 when we have >2 species potential
            if ns.diversity_index < 0.5 and len(ns.populations) > 1:
                if len(self._snapshots) >= 5:
                    prev_div = self._snapshots[-5].diversity_scores.get(ns.niche_name, 0.0)
                    if prev_div > 1.0 and ns.diversity_index < prev_div * 0.5:
                        evt = DysbiosisEvent(
                            tick=self._tick, niche=ns.niche_name,
                            event_type="crash", severity=0.6,
                            details=f"Diversity crash in {ns.niche_name}: {prev_div:.2f} → {ns.diversity_index:.2f}"
                        )
                        events.append(evt)
                        self._dysbiosis_events.append(evt)

            # Pathogen invasion: pathogenic > 30% of total
            pathogen_pop = sum(
                abd for sp_name, abd in ns.populations.items()
                if self._species_registry.get(sp_name, MicrobeSpecies(name=sp_name, species_type="unknown")).species_type == "pathogenic"
            )
            if total > 0 and pathogen_pop / total > 0.3:
                evt = DysbiosisEvent(
                    tick=self._tick, niche=ns.niche_name,
                    event_type="invasion", severity=min(1.0, pathogen_pop / total),
                    details=f"Pathogen invasion in {ns.niche_name}: {pathogen_pop/total:.1%}"
                )
                events.append(evt)
                self._dysbiosis_events.append(evt)

        return events

    # ------------------------------------------------------------------
    # Diversity Metrics
    # ------------------------------------------------------------------

    def _shannon_diversity(self, ns: NicheState) -> float:
        """Shannon diversity index H = -Σ(pi * ln(pi))."""
        total = sum(ns.populations.values())
        if total <= 0 or len(ns.populations) < 2:
            return 0.0
        entropy = 0.0
        for abundance in ns.populations.values():
            if abundance <= 0:
                continue
            p = abundance / total
            entropy -= p * math.log(p)
        return entropy

    def _simpson_diversity(self, ns: NicheState) -> float:
        """Simpson diversity index D = 1 - Σ(pi²)."""
        total = sum(ns.populations.values())
        if total <= 0 or len(ns.populations) < 2:
            return 0.0
        sum_p2 = sum((a / total) ** 2 for a in ns.populations.values() if a > 0)
        return 1.0 - sum_p2

    def compute_diversity(self, niche: str) -> float:
        """Public method: compute Shannon diversity for a niche."""
        ns = self._niches.get(niche)
        if ns is None:
            return 0.0
        return self._shannon_diversity(ns)

    # ------------------------------------------------------------------
    # Health Scoring
    # ------------------------------------------------------------------

    def _compute_health_score(self) -> float:
        """Composite health score 0-100."""
        if not self._niches:
            return 50.0

        scores: List[float] = []

        # 1. Diversity component (30%)
        diversity_scores = []
        for ns in self._niches.values():
            if sum(ns.populations.values()) < 1:
                continue
            max_possible = math.log(max(len(ns.populations), 2))
            norm = ns.diversity_index / max_possible if max_possible > 0 else 0
            diversity_scores.append(min(1.0, norm))
        avg_diversity = statistics.mean(diversity_scores) if diversity_scores else 0.0
        scores.append(avg_diversity * 30)

        # 2. Pathogen ratio component (25%) — lower is better
        pathogen_ratios = []
        for ns in self._niches.values():
            total = sum(ns.populations.values())
            if total < 1:
                continue
            pathogen_pop = sum(
                a for sp, a in ns.populations.items()
                if self._species_registry.get(sp, MicrobeSpecies(name=sp, species_type="unknown")).species_type == "pathogenic"
            )
            pathogen_ratios.append(pathogen_pop / total)
        avg_pathogen = statistics.mean(pathogen_ratios) if pathogen_ratios else 0.0
        scores.append((1.0 - min(1.0, avg_pathogen * 2)) * 25)

        # 3. Metabolic output component (20%)
        if self._metabolites:
            met_scores = []
            for m in self._metabolites.values():
                low, high = m.optimal_range
                if high <= low:
                    met_scores.append(0.5)
                elif m.current_level < low:
                    met_scores.append(max(0.0, m.current_level / low) if low > 0 else 0.0)
                elif m.current_level > high:
                    met_scores.append(max(0.0, 1.0 - (m.current_level - high) / high))
                else:
                    met_scores.append(1.0)
            avg_met = statistics.mean(met_scores)
            scores.append(avg_met * 20)
        else:
            scores.append(10.0)  # neutral if no metabolites tracked

        # 4. Niche balance component (15%) — populations not too empty or overloaded
        balance_scores = []
        for ns in self._niches.values():
            total = sum(ns.populations.values())
            ratio = total / ns.carrying_capacity if ns.carrying_capacity > 0 else 0
            # Optimal around 40-80% capacity
            if 0.4 <= ratio <= 0.8:
                balance_scores.append(1.0)
            elif ratio < 0.4:
                balance_scores.append(ratio / 0.4)
            else:
                balance_scores.append(max(0.0, 1.0 - (ratio - 0.8) * 2))
        avg_balance = statistics.mean(balance_scores) if balance_scores else 0.5
        scores.append(avg_balance * 15)

        # 5. Keystone presence component (10%)
        if self._keystone_species:
            present = 0
            for ks in self._keystone_species:
                for ns in self._niches.values():
                    if ks in ns.populations and ns.populations[ks] >= 1.0:
                        present += 1
                        break
            ks_score = present / len(self._keystone_species)
            scores.append(ks_score * 10)
        else:
            scores.append(5.0)

        return max(0.0, min(100.0, sum(scores)))

    def _classify_tier(self, score: float) -> str:
        """Classify health score into tier."""
        for threshold, label in HEALTH_TIERS:
            if score >= threshold:
                return label
        return "Critical"

    def _classify_dysbiosis_tier(self) -> str:
        """Classify current dysbiosis state."""
        # Use average diversity across niches
        diversities = []
        for ns in self._niches.values():
            total = sum(ns.populations.values())
            if total < 1:
                continue
            max_h = math.log(max(len(ns.populations), 2))
            norm = ns.diversity_index / max_h if max_h > 0 else 0
            diversities.append(norm)

        if not diversities:
            return "Eubiosis"

        avg = statistics.mean(diversities)
        for threshold, label in DYSBIOSIS_TIERS:
            if avg >= threshold:
                return label
        return "Critical"

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self) -> MicrobiomeReport:
        """Generate full analysis report."""
        report = MicrobiomeReport()
        report.snapshots = list(self._snapshots)

        if self._snapshots:
            report.overall_health = statistics.mean(s.health_score for s in self._snapshots)
            report.tier = self._classify_tier(report.overall_health)

        # Niche summaries
        for name, ns in self._niches.items():
            report.niche_summaries[name] = {
                "carrying_capacity": ns.carrying_capacity,
                "total_population": sum(ns.populations.values()),
                "species_count": len(ns.populations),
                "diversity": ns.diversity_index,
                "simpson": ns.simpson_index,
                "top_species": sorted(
                    ns.populations.items(), key=lambda x: x[1], reverse=True
                )[:5],
            }

        # Peak diversity per niche
        for name in self._niches:
            peak = 0.0
            for s in self._snapshots:
                d = s.diversity_scores.get(name, 0.0)
                if d > peak:
                    peak = d
            report.peak_diversity[name] = peak

        # Metabolic health
        if self._metabolites:
            in_range = 0
            for m in self._metabolites.values():
                if m.optimal_range[0] <= m.current_level <= m.optimal_range[1]:
                    in_range += 1
            report.metabolic_health = (in_range / len(self._metabolites)) * 100
        else:
            report.metabolic_health = 50.0

        # Intervention history
        report.intervention_history = [
            {"tick": t, "type": iv.intervention_type, "spectrum": iv.spectrum,
             "target": iv.target_species, "strength": iv.strength}
            for t, iv in self._interventions
        ]

        # Dysbiosis events
        report.total_dysbiosis_events = len(self._dysbiosis_events)

        # Species census (total across all niches)
        for ns in self._niches.values():
            for sp, abd in ns.populations.items():
                report.species_census[sp] = report.species_census.get(sp, 0.0) + abd

        # Insights
        report.insights = self._generate_insights(report)

        return report

    def _generate_insights(self, report: MicrobiomeReport) -> List[str]:
        """Generate autonomous insights."""
        insights: List[str] = []

        # Diversity trend
        if len(self._snapshots) >= 10:
            early = statistics.mean(
                statistics.mean(s.diversity_scores.values()) if s.diversity_scores else 0
                for s in self._snapshots[:5]
            )
            late = statistics.mean(
                statistics.mean(s.diversity_scores.values()) if s.diversity_scores else 0
                for s in self._snapshots[-5:]
            )
            if late < early * 0.7:
                insights.append(f"⚠️ Diversity declining: early avg {early:.2f} → late avg {late:.2f}")
            elif late > early * 1.3:
                insights.append(f"✅ Diversity improving: early avg {early:.2f} → late avg {late:.2f}")

        # Dominant pathogen warning
        for name, ns in self._niches.items():
            total = sum(ns.populations.values())
            if total < 1:
                continue
            for sp, abd in ns.populations.items():
                sp_obj = self._species_registry.get(sp)
                if sp_obj and sp_obj.species_type == "pathogenic" and abd / total > 0.2:
                    insights.append(f"🦠 Pathogen alert: {sp} at {abd/total:.0%} in {name}")

        # Metabolic deficiency
        for m in self._metabolites.values():
            if m.current_level < m.optimal_range[0]:
                insights.append(f"📉 Metabolic deficiency: {m.name} below optimal ({m.current_level:.1f} < {m.optimal_range[0]:.1f})")

        # Empty niches
        for name, ns in self._niches.items():
            if sum(ns.populations.values()) < 10:
                insights.append(f"🏜️ Near-empty niche: {name} (population < 10)")

        # Antibiotic recovery
        if self._interventions:
            antibiotics = [
                (t, iv) for t, iv in self._interventions
                if iv.intervention_type == "antibiotic"
            ]
            if antibiotics:
                last_abx_tick = antibiotics[-1][0]
                ticks_since = self._tick - last_abx_tick
                if ticks_since > 0 and ticks_since < 50:
                    insights.append(f"💊 Recovery phase: {ticks_since} ticks since last antibiotic")

        # Keystone status
        for ks in self._keystone_species:
            present_in = [
                n for n, ns in self._niches.items()
                if ks in ns.populations and ns.populations[ks] >= 1.0
            ]
            if not present_in:
                insights.append(f"⚠️ Keystone species {ks} absent from all niches")

        # Cross-niche correlation
        if len(self._niches) >= 2:
            niche_healths = {}
            for name, ns in self._niches.items():
                total = sum(ns.populations.values())
                niche_healths[name] = total / ns.carrying_capacity if ns.carrying_capacity > 0 else 0
            vals = list(niche_healths.values())
            if max(vals) - min(vals) > 0.5:
                best = max(niche_healths, key=niche_healths.get)  # type: ignore[arg-type]
                worst = min(niche_healths, key=niche_healths.get)  # type: ignore[arg-type]
                insights.append(f"🔗 Niche imbalance: {best} ({niche_healths[best]:.0%} capacity) vs {worst} ({niche_healths[worst]:.0%} capacity)")

        return insights

    # ------------------------------------------------------------------
    # HTML Export
    # ------------------------------------------------------------------

    def export_html(self, filepath: str) -> str:
        """Export interactive HTML dashboard."""
        report = self.analyze()
        html_content = self._render_html(report)
        Path(filepath).write_text(html_content, encoding="utf-8")
        return filepath

    def _render_html(self, report: MicrobiomeReport) -> str:
        """Render HTML dashboard."""
        # Niche rows
        niche_rows = ""
        for name, summary in report.niche_summaries.items():
            top = ", ".join(f"{s}({a:.0f})" for s, a in summary["top_species"][:3])
            niche_rows += (
                f'<tr><td>{html_mod.escape(name)}</td>'
                f'<td>{summary["total_population"]:.0f} / {summary["carrying_capacity"]:.0f}</td>'
                f'<td>{summary["species_count"]}</td>'
                f'<td>{summary["diversity"]:.2f}</td>'
                f'<td>{top}</td></tr>\n'
            )

        # Metabolite rows
        met_rows = ""
        for m in self._metabolites.values():
            low, high = m.optimal_range
            in_range = "✅" if low <= m.current_level <= high else "⚠️"
            met_rows += (
                f'<tr><td>{html_mod.escape(m.name)}</td>'
                f'<td>{m.current_level:.1f}</td>'
                f'<td>{low:.0f}–{high:.0f}</td>'
                f'<td>{in_range}</td></tr>\n'
            )

        # Dysbiosis event rows
        event_rows = ""
        for evt in self._dysbiosis_events[-20:]:
            event_rows += (
                f'<tr><td>{evt.tick}</td>'
                f'<td>{html_mod.escape(evt.niche)}</td>'
                f'<td>{html_mod.escape(evt.event_type)}</td>'
                f'<td>{evt.severity:.2f}</td>'
                f'<td>{html_mod.escape(evt.details)}</td></tr>\n'
            )

        # Intervention rows
        intv_rows = ""
        for t, iv in self._interventions:
            intv_rows += (
                f'<tr><td>{t}</td>'
                f'<td>{html_mod.escape(iv.intervention_type)}</td>'
                f'<td>{html_mod.escape(iv.spectrum)}</td>'
                f'<td>{html_mod.escape(iv.target_species or "all")}</td>'
                f'<td>{iv.strength:.1f}</td></tr>\n'
            )

        # Insight list
        insight_items = "".join(
            f'<li>{html_mod.escape(i)}</li>\n' for i in report.insights
        ) or '<li>No insights generated yet.</li>'

        health_color = '#76ff03' if report.overall_health > 60 else '#ffab40' if report.overall_health > 30 else '#ff5252'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Microbiome — Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0a0a0f; color: #e0e0e0; padding: 24px; }}
h1 {{ color: #00e5ff; margin-bottom: 8px; }}
h2 {{ color: #76ff03; margin: 24px 0 12px; font-size: 1.2em; }}
.subtitle {{ color: #888; margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }}
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
.tag-good {{ background: #1b5e20; color: #76ff03; }}
.tag-warn {{ background: #e65100; color: #ffab40; }}
.tag-bad {{ background: #b71c1c; color: #ff5252; }}
ul {{ margin: 12px 0; padding-left: 20px; }}
li {{ margin: 4px 0; line-height: 1.5; }}
.canvas {{ background: #111; border-radius: 8px; padding: 16px; margin-top: 16px; overflow-x: auto; }}
.ascii {{ font-family: monospace; font-size: 11px; white-space: pre; line-height: 1.4; color: #aaa; }}
</style>
</head>
<body>
<h1>🦠 Swarm Microbiome Dashboard</h1>
<p class="subtitle">Autonomous commensal ecosystem — {len(self._niches)} niches, {len(self._species_registry)} species, {self._tick} ticks</p>

<div class="grid">
  <div class="card">
    <h3>Microbiome Health</h3>
    <div class="metric" style="color:{health_color}">{report.overall_health:.1f}</div>
    <div class="bar"><div class="bar-fill" style="width:{report.overall_health}%;background:{health_color}"></div></div>
    <p style="color:#888;margin-top:4px">{report.tier}</p>
  </div>
  <div class="card">
    <h3>Dysbiosis State</h3>
    <div class="metric-sm">{self._classify_dysbiosis_tier()}</div>
    <p style="color:#888;margin-top:4px">{report.total_dysbiosis_events} events detected</p>
  </div>
  <div class="card">
    <h3>Total Species</h3>
    <div class="metric">{len(self._species_registry)}</div>
    <p style="color:#888;margin-top:4px">{len(report.species_census)} currently present</p>
  </div>
  <div class="card">
    <h3>Metabolic Health</h3>
    <div class="metric">{report.metabolic_health:.0f}%</div>
    <p style="color:#888;margin-top:4px">{len(self._metabolites)} metabolites tracked</p>
  </div>
</div>

<h2>Niche Populations</h2>
<div class="card">
<table>
<tr><th>Niche</th><th>Population / Capacity</th><th>Species</th><th>Shannon H</th><th>Top Species</th></tr>
{niche_rows}
</table>
</div>

<h2>Metabolites</h2>
<div class="card">
<table>
<tr><th>Metabolite</th><th>Level</th><th>Optimal Range</th><th>Status</th></tr>
{met_rows if met_rows else '<tr><td colspan="4">No metabolites registered.</td></tr>'}
</table>
</div>

<h2>Dysbiosis Events (last 20)</h2>
<div class="card">
<table>
<tr><th>Tick</th><th>Niche</th><th>Type</th><th>Severity</th><th>Details</th></tr>
{event_rows if event_rows else '<tr><td colspan="5">No dysbiosis events.</td></tr>'}
</table>
</div>

<h2>Interventions</h2>
<div class="card">
<table>
<tr><th>Tick</th><th>Type</th><th>Spectrum</th><th>Target</th><th>Strength</th></tr>
{intv_rows if intv_rows else '<tr><td colspan="5">No interventions applied.</td></tr>'}
</table>
</div>

<h2>Health Timeline</h2>
<div class="canvas">
<pre class="ascii">{self._render_health_timeline()}</pre>
</div>

<h2>🧠 Autonomous Insights</h2>
<div class="card">
<ul>
{insight_items}
</ul>
</div>

</body>
</html>"""

    def _render_health_timeline(self) -> str:
        """Render ASCII health score timeline."""
        if not self._snapshots:
            return "No snapshots yet."
        width = min(60, len(self._snapshots))
        samples = self._snapshots[-width:]
        height = 8
        lines = []
        lines.append("  Health Score (0-100)")
        for row in range(height, 0, -1):
            threshold = 100.0 * row / height
            line = f"  {threshold:5.0f}│"
            for s in samples:
                line += "█" if s.health_score >= threshold else " "
            lines.append(line)
        lines.append("       └" + "─" * len(samples))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON Export
    # ------------------------------------------------------------------

    def export_json(self, filepath: str) -> str:
        """Export analysis as JSON."""
        report = self.analyze()
        data = {
            "overall_health": report.overall_health,
            "tier": report.tier,
            "metabolic_health": report.metabolic_health,
            "total_dysbiosis_events": report.total_dysbiosis_events,
            "niche_summaries": {
                k: {**v, "top_species": [(s, a) for s, a in v["top_species"]]}
                for k, v in report.niche_summaries.items()
            },
            "species_census": report.species_census,
            "intervention_history": report.intervention_history,
            "insights": report.insights,
            "peak_diversity": report.peak_diversity,
            "snapshots": [
                {
                    "tick": s.tick,
                    "health_score": s.health_score,
                    "dysbiosis_tier": s.dysbiosis_tier,
                    "total_population": s.total_population,
                    "diversity_scores": s.diversity_scores,
                }
                for s in report.snapshots[-50:]  # last 50 to keep size reasonable
            ],
        }
        Path(filepath).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return filepath


# ---------------------------------------------------------------------------
# CLI Demo
# ---------------------------------------------------------------------------


def _create_default_species() -> List[Tuple[MicrobeSpecies, str, float]]:
    """Create a default set of species for demos."""
    return [
        # Gut commensals
        (MicrobeSpecies(
            name="Bacteroides", species_type="commensal",
            preferred_niches=["gut"], metabolites_produced=["butyrate", "propionate"],
            antibiotic_susceptibility={"broad": 0.7, "metronidazole": 0.9},
            growth_rate=0.08,
        ), "gut", 200.0),
        (MicrobeSpecies(
            name="Lactobacillus", species_type="commensal",
            preferred_niches=["gut", "oral", "urogenital"],
            metabolites_produced=["lactic_acid", "vitamin_k"],
            antibiotic_susceptibility={"broad": 0.6, "vancomycin": 0.3},
            growth_rate=0.07,
        ), "gut", 150.0),
        (MicrobeSpecies(
            name="Bifidobacterium", species_type="commensal",
            preferred_niches=["gut"], metabolites_produced=["acetate", "folate"],
            antibiotic_susceptibility={"broad": 0.8, "metronidazole": 0.4},
            growth_rate=0.06,
        ), "gut", 120.0),
        (MicrobeSpecies(
            name="Faecalibacterium", species_type="commensal",
            preferred_niches=["gut"], metabolites_produced=["butyrate"],
            antibiotic_susceptibility={"broad": 0.85},
            growth_rate=0.07,
        ), "gut", 100.0),
        # Skin commensals
        (MicrobeSpecies(
            name="S.epidermidis", species_type="commensal",
            preferred_niches=["skin"], metabolites_produced=["antimicrobial_peptides"],
            antibiotic_susceptibility={"broad": 0.5},
            growth_rate=0.06,
        ), "skin", 150.0),
        (MicrobeSpecies(
            name="Corynebacterium", species_type="commensal",
            preferred_niches=["skin", "respiratory"],
            metabolites_produced=["lipase"],
            antibiotic_susceptibility={"broad": 0.6},
            growth_rate=0.05,
        ), "skin", 100.0),
        # Oral
        (MicrobeSpecies(
            name="Streptococcus_mitis", species_type="commensal",
            preferred_niches=["oral"], metabolites_produced=["h2o2"],
            antibiotic_susceptibility={"broad": 0.7},
            growth_rate=0.08,
        ), "oral", 100.0),
        # Pathogens (low initial)
        (MicrobeSpecies(
            name="C.difficile", species_type="pathogenic",
            preferred_niches=["gut"], metabolites_produced=["toxin_a"],
            antibiotic_susceptibility={"broad": 0.2, "vancomycin": 0.9},
            growth_rate=0.12, immune_evasion=0.3,
        ), "gut", 5.0),
        (MicrobeSpecies(
            name="S.aureus", species_type="pathogenic",
            preferred_niches=["skin", "respiratory"],
            metabolites_produced=["enterotoxin"],
            antibiotic_susceptibility={"broad": 0.4, "methicillin": 0.1},
            growth_rate=0.1, immune_evasion=0.4,
        ), "skin", 10.0),
    ]


def _run_demo(args: argparse.Namespace) -> None:
    """Run an interactive demo simulation."""
    engine = SwarmMicrobiomeEngine(num_host_agents=5)

    # Register metabolites
    engine.register_metabolite("butyrate", ["Bacteroides", "Faecalibacterium"],
                               optimal_range=(5.0, 50.0))
    engine.register_metabolite("lactic_acid", ["Lactobacillus"],
                               optimal_range=(3.0, 30.0))
    engine.register_metabolite("vitamin_k", ["Lactobacillus", "Bacteroides"],
                               optimal_range=(2.0, 20.0))
    engine.register_metabolite("folate", ["Bifidobacterium"],
                               optimal_range=(1.0, 15.0))

    # Mark keystones
    engine.mark_keystone("Bacteroides")
    engine.mark_keystone("Lactobacillus")

    # Seed default species
    for sp, niche, abundance in _create_default_species():
        engine.seed_species(sp, niche, abundance)

    # Also seed some in secondary niches
    for sp, _, _ in _create_default_species():
        for niche in sp.preferred_niches[1:]:
            if niche in engine.niches:
                engine.seed_species(sp, niche, random.uniform(10, 50))

    print(f"🦠 Swarm Microbiome Simulation")
    print(f"   Niches: {len(engine.niches)} | Species: {len(engine.species)} | Ticks: {args.ticks}")
    print(f"   Scenario: {args.scenario}")
    print("─" * 60)

    scenario = args.scenario

    if scenario == "default":
        # Healthy natural fluctuations
        for t in range(args.ticks):
            snapshot = engine.tick()
            if snapshot.dysbiosis_events:
                for evt in snapshot.dysbiosis_events:
                    print(f"  ⚠️ Tick {t+1}: {evt['type']} in {evt['niche']} (sev={evt['severity']:.2f})")

    elif scenario == "dysbiosis":
        # First half normal, then pathogen invasion
        mid = args.ticks // 2
        for t in range(mid):
            engine.tick()
        # Introduce aggressive pathogen
        invader = MicrobeSpecies(
            name="Klebsiella_pneumoniae", species_type="pathogenic",
            preferred_niches=["gut", "respiratory"],
            metabolites_produced=["endotoxin"],
            antibiotic_susceptibility={"broad": 0.3},
            growth_rate=0.15, immune_evasion=0.6,
        )
        engine.seed_species(invader, "gut", 50.0)
        engine.seed_species(invader, "respiratory", 30.0)
        print(f"  🦠 Tick {mid}: Klebsiella_pneumoniae invasion!")
        for t in range(mid, args.ticks):
            snapshot = engine.tick()
            if snapshot.dysbiosis_events:
                for evt in snapshot.dysbiosis_events:
                    print(f"  ⚠️ Tick {t+1}: {evt['type']} in {evt['niche']}")

    elif scenario == "antibiotic":
        # Normal → antibiotic at 1/3 → observe aftermath
        abx_tick = args.ticks // 3
        for t in range(args.ticks):
            if t == abx_tick:
                print(f"  💊 Tick {t+1}: Broad-spectrum antibiotic applied!")
                engine.apply_intervention(Intervention(
                    intervention_type="antibiotic", spectrum="broad", strength=0.8
                ))
            snapshot = engine.tick()
            if snapshot.dysbiosis_events:
                for evt in snapshot.dysbiosis_events:
                    print(f"  ⚠️ Tick {t+1}: {evt['type']} in {evt['niche']}")

    elif scenario == "recovery":
        # Antibiotic at 1/4, probiotic at 1/2
        abx_tick = args.ticks // 4
        pro_tick = args.ticks // 2
        for t in range(args.ticks):
            if t == abx_tick:
                print(f"  💊 Tick {t+1}: Broad-spectrum antibiotic!")
                engine.apply_intervention(Intervention(
                    intervention_type="antibiotic", spectrum="broad", strength=0.8
                ))
            if t == pro_tick:
                print(f"  🧪 Tick {t+1}: Probiotic intervention!")
                probiotic = MicrobeSpecies(
                    name="Lactobacillus_rhamnosus", species_type="probiotic",
                    preferred_niches=["gut"], metabolites_produced=["lactic_acid"],
                    antibiotic_susceptibility={"broad": 0.5},
                    growth_rate=0.09,
                )
                engine.apply_intervention(Intervention(
                    intervention_type="probiotic", strength=1.0,
                    species_to_introduce=probiotic,
                ))
                engine.apply_intervention(Intervention(
                    intervention_type="prebiotic", strength=1.5,
                    boost_species=["Bacteroides", "Bifidobacterium"],
                ))
            snapshot = engine.tick()
            if snapshot.dysbiosis_events:
                for evt in snapshot.dysbiosis_events:
                    print(f"  ⚠️ Tick {t+1}: {evt['type']} in {evt['niche']}")
    else:
        print(f"  Unknown scenario: {scenario}, running default")
        for t in range(args.ticks):
            engine.tick()

    # Final report
    report = engine.analyze()
    print("\n" + "─" * 60)
    print(f"📊 Final Report")
    print(f"   Overall Health: {report.overall_health:.1f}/100 ({report.tier})")
    print(f"   Dysbiosis Events: {report.total_dysbiosis_events}")
    print(f"   Metabolic Health: {report.metabolic_health:.0f}%")
    print(f"\n   Niche Summary:")
    for name, summary in report.niche_summaries.items():
        pop = summary['total_population']
        cap = summary['carrying_capacity']
        bar = "█" * int((pop / cap) * 30) + "░" * (30 - int((pop / cap) * 30))
        print(f"     {name:15s} {bar} {pop:.0f}/{cap:.0f} ({summary['species_count']} spp, H={summary['diversity']:.2f})")

    if report.insights:
        print(f"\n   🧠 Insights:")
        for insight in report.insights:
            print(f"     {insight}")

    # Export
    if args.out:
        engine.export_html(args.out)
        print(f"\n   📄 HTML report: {args.out}")
    if args.json_out:
        engine.export_json(args.json_out)
        print(f"   📄 JSON export: {args.json_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm Microbiome Engine — autonomous commensal ecosystem management"
    )
    parser.add_argument("--ticks", type=int, default=80, help="Simulation ticks")
    parser.add_argument("--scenario", type=str, default="default",
                        choices=["default", "dysbiosis", "antibiotic", "recovery"],
                        help="Demo scenario")
    parser.add_argument("--out", type=str, default=None, help="HTML output path")
    parser.add_argument("--json", dest="json_out", type=str, default=None, help="JSON output path")
    args = parser.parse_args()
    _run_demo(args)


if __name__ == "__main__":
    main()
