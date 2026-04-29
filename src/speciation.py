"""Swarm Speciation Engine — autonomous niche discovery & agent specialization.

Over multi-task consensus rounds agents naturally *speciate* into specialist
niches based on performance patterns — mirroring ecological speciation.  The
engine detects emergent clusters, tracks speciation events (niche formation,
extinction, hybridization, adaptation, radiation), computes per-species health
metrics, and recommends optimal task routing.

Usage (CLI)::

    python -m src.speciation                             # defaults
    python -m src.speciation --agents 12 --rounds 60
    python -m src.speciation --byzantine 3 --task-types 6
    python -m src.speciation --export report.html --json state.json
"""
from __future__ import annotations

import argparse
import asyncio
import html as html_mod
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.metacognitive import MockAgent  # noqa: E402
from src.core.protocol import MBFTEngine  # noqa: E402
from src.core.state import RoundResult  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────

TASK_TYPES = [
    "analytical",
    "creative",
    "adversarial",
    "collaborative",
    "rapid",
    "strategic",
    "diagnostic",
    "exploratory",
]

SPECIES_PREFIXES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
    "Zeta", "Eta", "Theta", "Iota", "Kappa",
]
SPECIES_SUFFIXES = [
    "Specialists", "Generalists", "Adapters", "Pioneers",
    "Sentinels", "Navigators", "Catalysts", "Arbiters",
]

# ── Data Models ──────────────────────────────────────────────────────────


class TaskRecord(BaseModel):
    task_id: str = Field(default_factory=lambda: uuid4().hex[:10])
    task_type: str
    round_index: int
    leader_id: Optional[str] = None
    committed: bool = False
    committed_solution: Optional[str] = None
    voter_weights: Dict[str, float] = Field(default_factory=dict)
    slashed: List[str] = Field(default_factory=list)


class FitnessProfile(BaseModel):
    agent_id: str
    task_scores: Dict[str, float] = Field(default_factory=dict)  # task_type -> score 0-1
    leadership_rate: Dict[str, float] = Field(default_factory=dict)
    alignment_rate: Dict[str, float] = Field(default_factory=dict)
    calibration: Dict[str, float] = Field(default_factory=dict)
    overall: float = 0.0

    def vector(self, task_types: List[str]) -> List[float]:
        """Return fitness as a numeric vector over the given task types."""
        return [self.task_scores.get(t, 0.0) for t in task_types]


class SpeciesModel(BaseModel):
    species_id: str = Field(default_factory=lambda: uuid4().hex[:8])
    name: str = ""
    member_ids: List[str] = Field(default_factory=list)
    centroid: List[float] = Field(default_factory=list)
    population: int = 0
    avg_fitness: float = 0.0
    internal_diversity: float = 0.0
    niche_width: int = 0
    dominance_index: float = 0.0


class SpeciationEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex[:8])
    event_type: str  # formation, extinction, hybridization, adaptation, radiation
    timestamp: float = Field(default_factory=time.time)
    round_index: int = 0
    involved_species: List[str] = Field(default_factory=list)
    description: str = ""


class RoutingRecommendation(BaseModel):
    task_type: str
    recommended_species_id: str
    recommended_species_name: str
    recommended_agents: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""


class EcosystemSummary(BaseModel):
    num_species: int = 0
    num_agents: int = 0
    diversity_index: float = 0.0  # Shannon
    specialist_ratio: float = 0.0
    generalist_ratio: float = 0.0
    stability_score: float = 0.0
    total_rounds: int = 0
    total_events: int = 0


# ── K-Means (from scratch) ──────────────────────────────────────────────


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _centroid(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]


def _kmeans(
    vectors: List[List[float]],
    k: int,
    max_iter: int = 50,
    restarts: int = 5,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], List[List[float]], float]:
    """K-means clustering. Returns (labels, centroids, wcss)."""
    if not vectors or k <= 0:
        return [], [], 0.0
    rng = rng or random.Random()
    n = len(vectors)
    k = min(k, n)
    best_labels: List[int] = list(range(n))
    best_centroids: List[List[float]] = []
    best_wcss = float("inf")

    for _ in range(restarts):
        # random init
        indices = rng.sample(range(n), k)
        centroids = [list(vectors[i]) for i in indices]
        labels = [0] * n

        for _it in range(max_iter):
            # assign
            new_labels = []
            for v in vectors:
                dists = [_euclidean(v, c) for c in centroids]
                new_labels.append(dists.index(min(dists)))
            if new_labels == labels and _it > 0:
                break
            labels = new_labels

            # update centroids
            for ci in range(k):
                members = [vectors[j] for j in range(n) if labels[j] == ci]
                if members:
                    centroids[ci] = _centroid(members)

        wcss = sum(
            _euclidean(vectors[i], centroids[labels[i]]) ** 2
            for i in range(n)
        )
        if wcss < best_wcss:
            best_wcss = wcss
            best_labels = list(labels)
            best_centroids = [list(c) for c in centroids]

    return best_labels, best_centroids, best_wcss


def _elbow_k(
    vectors: List[List[float]],
    max_k: int = 8,
    rng: Optional[random.Random] = None,
) -> int:
    """Auto-detect k using the elbow method (20% marginal improvement threshold)."""
    if len(vectors) <= 2:
        return min(len(vectors), 2)
    max_k = min(max_k, len(vectors))
    wcss_values: List[float] = []
    for k in range(1, max_k + 1):
        _, _, wcss = _kmeans(vectors, k, rng=rng)
        wcss_values.append(wcss)

    if len(wcss_values) < 2:
        return 1

    # pick k where marginal improvement drops below 20%
    for i in range(1, len(wcss_values)):
        if wcss_values[i - 1] == 0:
            return i  # already perfect
        improvement = (wcss_values[i - 1] - wcss_values[i]) / wcss_values[i - 1]
        if improvement < 0.20:
            return max(i, 2)
    return max_k


# ── Speciation Engine ────────────────────────────────────────────────────


class SpeciationEngine:
    """Autonomous niche discovery and agent specialization tracker."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.task_records: List[TaskRecord] = []
        self.fitness_profiles: Dict[str, FitnessProfile] = {}
        self.species: List[SpeciesModel] = []
        self.previous_species: List[SpeciesModel] = []
        self.events: List[SpeciationEvent] = []
        self.active_task_types: List[str] = []
        self._round_counter = 0
        self._name_counter = 0

    def _generate_species_name(self) -> str:
        idx = self._name_counter
        self._name_counter += 1
        prefix = SPECIES_PREFIXES[idx % len(SPECIES_PREFIXES)]
        suffix = SPECIES_SUFFIXES[idx % len(SPECIES_SUFFIXES)]
        return f"{prefix} {suffix}"

    # ── Ingest ───────────────────────────────────────────────────────

    def record_round(self, task_type: str, result: RoundResult) -> TaskRecord:
        """Ingest one consensus round result tagged with a task type."""
        if task_type not in self.active_task_types:
            self.active_task_types.append(task_type)

        voter_weights = {v.voter_id: v.weight for v in result.votes}
        rec = TaskRecord(
            task_type=task_type,
            round_index=result.round_index,
            leader_id=result.leader_id,
            committed=result.committed,
            committed_solution=result.committed_solution,
            voter_weights=voter_weights,
            slashed=list(result.slashed),
        )
        self.task_records.append(rec)
        self._round_counter += 1
        return rec

    # ── Fitness Computation ──────────────────────────────────────────

    def update_fitness_profiles(self) -> Dict[str, FitnessProfile]:
        """Recompute fitness profiles from all recorded rounds."""
        # collect all agent ids
        agent_ids: set[str] = set()
        for rec in self.task_records:
            if rec.leader_id:
                agent_ids.add(rec.leader_id)
            agent_ids.update(rec.voter_weights.keys())
            agent_ids.update(rec.slashed)

        # per-agent per-task-type stats
        leadership_attempts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        leadership_successes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        vote_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        vote_aligned: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        slash_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        participation: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for rec in self.task_records:
            tt = rec.task_type
            if rec.leader_id:
                leadership_attempts[rec.leader_id][tt] += 1
                if rec.committed:
                    leadership_successes[rec.leader_id][tt] += 1
            for aid, w in rec.voter_weights.items():
                vote_counts[aid][tt] += 1
                participation[aid][tt] += 1
                if rec.committed and w > 0:
                    vote_aligned[aid][tt] += 1
                elif not rec.committed and w <= 0:
                    vote_aligned[aid][tt] += 1
            for aid in rec.slashed:
                slash_counts[aid][tt] += 1

        profiles: Dict[str, FitnessProfile] = {}
        for aid in agent_ids:
            lr: Dict[str, float] = {}
            ar: Dict[str, float] = {}
            cal: Dict[str, float] = {}
            ts: Dict[str, float] = {}

            for tt in self.active_task_types:
                la = leadership_attempts[aid].get(tt, 0)
                ls = leadership_successes[aid].get(tt, 0)
                lr[tt] = ls / la if la > 0 else 0.0

                vc = vote_counts[aid].get(tt, 0)
                va = vote_aligned[aid].get(tt, 0)
                ar[tt] = va / vc if vc > 0 else 0.0

                sc = slash_counts[aid].get(tt, 0)
                part = participation[aid].get(tt, 0)
                cal[tt] = 1.0 - (sc / part if part > 0 else 0.0)

                # composite task score: weighted average
                ts[tt] = 0.4 * lr.get(tt, 0) + 0.35 * ar.get(tt, 0) + 0.25 * cal.get(tt, 0)

            overall = sum(ts.values()) / len(ts) if ts else 0.0
            profiles[aid] = FitnessProfile(
                agent_id=aid,
                task_scores=ts,
                leadership_rate=lr,
                alignment_rate=ar,
                calibration=cal,
                overall=overall,
            )

        self.fitness_profiles = profiles
        return profiles

    # ── Species Detection ────────────────────────────────────────────

    def detect_species(self, k: Optional[int] = None) -> List[SpeciesModel]:
        """Cluster agents into species by fitness profile similarity."""
        if not self.fitness_profiles:
            return []

        self.previous_species = list(self.species)

        agent_ids = sorted(self.fitness_profiles.keys())
        if len(agent_ids) < 2:
            # single agent = single species
            fp = self.fitness_profiles[agent_ids[0]]
            sp = SpeciesModel(
                name=self._generate_species_name() if not self.species else self.species[0].name if self.species else self._generate_species_name(),
                member_ids=[agent_ids[0]],
                centroid=fp.vector(self.active_task_types),
                population=1,
                avg_fitness=fp.overall,
                internal_diversity=0.0,
                niche_width=sum(1 for s in fp.task_scores.values() if s > 0.3),
                dominance_index=1.0,
            )
            self.species = [sp]
            return self.species

        vectors = [self.fitness_profiles[a].vector(self.active_task_types) for a in agent_ids]

        if k is None:
            k = _elbow_k(vectors, max_k=min(8, len(agent_ids)), rng=self.rng)
        k = max(2, min(k, len(agent_ids)))

        labels, centroids, _ = _kmeans(vectors, k, rng=self.rng)

        # build species
        new_species: List[SpeciesModel] = []
        total_leadership = sum(
            sum(self.fitness_profiles[a].leadership_rate.values())
            for a in agent_ids
        )

        for ci in range(k):
            members = [agent_ids[j] for j in range(len(agent_ids)) if labels[j] == ci]
            if not members:
                continue

            member_vectors = [vectors[j] for j in range(len(agent_ids)) if labels[j] == ci]
            cent = centroids[ci]
            avg_fit = sum(self.fitness_profiles[m].overall for m in members) / len(members)

            # internal diversity = avg distance to centroid
            int_div = sum(_euclidean(v, cent) for v in member_vectors) / len(member_vectors) if member_vectors else 0.0

            # niche width = number of task dims where centroid > 0.3
            niche_w = sum(1 for c in cent if c > 0.3)

            # dominance = share of leadership
            sp_lead = sum(
                sum(self.fitness_profiles[m].leadership_rate.values())
                for m in members
            )
            dom = sp_lead / total_leadership if total_leadership > 0 else 0.0

            # try to match with previous species by centroid proximity
            name = self._match_previous_name(cent)

            new_species.append(SpeciesModel(
                name=name,
                member_ids=members,
                centroid=cent,
                population=len(members),
                avg_fitness=avg_fit,
                internal_diversity=int_div,
                niche_width=niche_w,
                dominance_index=dom,
            ))

        self.species = new_species
        return self.species

    def _match_previous_name(self, centroid: List[float]) -> str:
        """Try to reuse a previous species name if centroid is close."""
        best_dist = float("inf")
        best_name = ""
        for sp in self.previous_species:
            if sp.centroid:
                d = _euclidean(centroid, sp.centroid)
                if d < best_dist:
                    best_dist = d
                    best_name = sp.name
        if best_dist < 0.3 and best_name:
            return best_name
        return self._generate_species_name()

    # ── Speciation Events ────────────────────────────────────────────

    def detect_speciation_events(self) -> List[SpeciationEvent]:
        """Compare current species to previous to detect events."""
        new_events: List[SpeciationEvent] = []
        prev = {sp.name: sp for sp in self.previous_species}
        curr = {sp.name: sp for sp in self.species}

        prev_names = set(prev.keys())
        curr_names = set(curr.keys())

        # Formation: new names not in previous
        for name in curr_names - prev_names:
            sp = curr[name]
            new_events.append(SpeciationEvent(
                event_type="formation",
                round_index=self._round_counter,
                involved_species=[sp.species_id],
                description=f"New species '{name}' emerged with {sp.population} members",
            ))

        # Extinction: previous names not in current
        for name in prev_names - curr_names:
            sp = prev[name]
            new_events.append(SpeciationEvent(
                event_type="extinction",
                round_index=self._round_counter,
                involved_species=[sp.species_id],
                description=f"Species '{name}' went extinct (had {sp.population} members)",
            ))

        # Adaptation: same name but centroid shifted significantly
        for name in prev_names & curr_names:
            p = prev[name]
            c = curr[name]
            if p.centroid and c.centroid:
                drift = _euclidean(p.centroid, c.centroid)
                if drift > 0.15:
                    new_events.append(SpeciationEvent(
                        event_type="adaptation",
                        round_index=self._round_counter,
                        involved_species=[c.species_id],
                        description=f"Species '{name}' adapted (centroid drift {drift:.3f})",
                    ))

        # Hybridization: detect if two previous species merged into one current
        for cname, csp in curr.items():
            matching_prev = []
            for pname, psp in prev.items():
                overlap = len(set(csp.member_ids) & set(psp.member_ids))
                if overlap > 0 and pname != cname:
                    matching_prev.append(pname)
            if len(matching_prev) >= 2:
                new_events.append(SpeciationEvent(
                    event_type="hybridization",
                    round_index=self._round_counter,
                    involved_species=[csp.species_id],
                    description=f"Species '{cname}' formed by hybridization of {matching_prev}",
                ))

        # Radiation: detect if one previous species split into multiple current
        for pname, psp in prev.items():
            split_into = []
            for cname, csp in curr.items():
                overlap = len(set(psp.member_ids) & set(csp.member_ids))
                if overlap > 0:
                    split_into.append(cname)
            if len(split_into) >= 2 and pname in curr_names:
                new_events.append(SpeciationEvent(
                    event_type="radiation",
                    round_index=self._round_counter,
                    involved_species=[curr[pname].species_id],
                    description=f"Species '{pname}' radiated into {split_into}",
                ))

        self.events.extend(new_events)
        return new_events

    # ── Routing ──────────────────────────────────────────────────────

    def get_routing_recommendation(self, task_type: str) -> RoutingRecommendation:
        """Recommend which species to route a task to."""
        if not self.species or not self.active_task_types:
            return RoutingRecommendation(
                task_type=task_type,
                recommended_species_id="",
                recommended_species_name="none",
                confidence=0.0,
                reasoning="No species detected yet",
            )

        tt_idx = self.active_task_types.index(task_type) if task_type in self.active_task_types else -1

        best_sp: Optional[SpeciesModel] = None
        best_score = -1.0
        for sp in self.species:
            if tt_idx >= 0 and tt_idx < len(sp.centroid):
                score = sp.centroid[tt_idx]
            else:
                score = sp.avg_fitness
            if score > best_score:
                best_score = score
                best_sp = sp

        if best_sp is None:
            return RoutingRecommendation(
                task_type=task_type,
                recommended_species_id="",
                recommended_species_name="none",
                confidence=0.0,
                reasoning="No suitable species found",
            )

        return RoutingRecommendation(
            task_type=task_type,
            recommended_species_id=best_sp.species_id,
            recommended_species_name=best_sp.name,
            recommended_agents=list(best_sp.member_ids),
            confidence=min(best_score, 1.0),
            reasoning=f"Species '{best_sp.name}' has best fitness ({best_score:.3f}) for {task_type}",
        )

    # ── Health & Summary ─────────────────────────────────────────────

    def get_species_health(self) -> List[Dict[str, Any]]:
        """Health metrics for all species."""
        return [sp.model_dump() for sp in self.species]

    def get_ecosystem_summary(self) -> EcosystemSummary:
        """Overall ecosystem statistics."""
        n_agents = len(self.fitness_profiles)
        n_species = len(self.species)

        # Shannon diversity
        if n_species > 0 and n_agents > 0:
            proportions = [sp.population / n_agents for sp in self.species if sp.population > 0]
            shannon = -sum(p * math.log(p) for p in proportions if p > 0)
        else:
            shannon = 0.0

        # specialist vs generalist
        specialists = sum(1 for sp in self.species if sp.niche_width <= 2)
        generalists = sum(1 for sp in self.species if sp.niche_width > 2)
        spec_ratio = specialists / n_species if n_species else 0.0
        gen_ratio = generalists / n_species if n_species else 0.0

        # stability: fewer recent events = more stable
        recent = sum(1 for e in self.events if self._round_counter - e.round_index < 20)
        stability = max(0.0, 1.0 - recent * 0.1)

        return EcosystemSummary(
            num_species=n_species,
            num_agents=n_agents,
            diversity_index=round(shannon, 4),
            specialist_ratio=round(spec_ratio, 4),
            generalist_ratio=round(gen_ratio, 4),
            stability_score=round(stability, 4),
            total_rounds=self._round_counter,
            total_events=len(self.events),
        )

    # ── Export ────────────────────────────────────────────────────────

    def export_json(self) -> Dict[str, Any]:
        """Full state export."""
        return {
            "fitness_profiles": {k: v.model_dump() for k, v in self.fitness_profiles.items()},
            "species": [sp.model_dump() for sp in self.species],
            "events": [e.model_dump() for e in self.events],
            "ecosystem": self.get_ecosystem_summary().model_dump(),
            "task_types": self.active_task_types,
            "total_rounds": self._round_counter,
        }

    # ── HTML Report ──────────────────────────────────────────────────

    def render_html(self) -> str:
        """Interactive HTML dashboard."""
        eco = self.get_ecosystem_summary()

        # Species table rows
        species_rows = ""
        for sp in self.species:
            members_str = html_mod.escape(", ".join(sp.member_ids[:8]))
            if len(sp.member_ids) > 8:
                members_str += f" +{len(sp.member_ids) - 8} more"
            species_rows += f"""<tr>
                <td>{html_mod.escape(sp.name)}</td>
                <td>{sp.population}</td>
                <td>{sp.avg_fitness:.3f}</td>
                <td>{sp.internal_diversity:.3f}</td>
                <td>{sp.niche_width}</td>
                <td>{sp.dominance_index:.3f}</td>
                <td style="font-size:0.85em">{members_str}</td>
            </tr>"""

        # Fitness heatmap
        heatmap_html = ""
        if self.fitness_profiles and self.active_task_types:
            header = "".join(f"<th>{html_mod.escape(t)}</th>" for t in self.active_task_types)
            rows = ""
            for aid in sorted(self.fitness_profiles.keys()):
                fp = self.fitness_profiles[aid]
                cells = ""
                for tt in self.active_task_types:
                    s = fp.task_scores.get(tt, 0.0)
                    r = int(min(255, (1 - s) * 255))
                    g = int(min(255, s * 255))
                    cells += f'<td style="background:rgba({r},{g},80,0.6);text-align:center">{s:.2f}</td>'
                rows += f"<tr><td>{html_mod.escape(aid)}</td>{cells}</tr>"
            heatmap_html = f"""<table class="tbl">
                <tr><th>Agent</th>{header}</tr>
                {rows}
            </table>"""

        # Events timeline
        events_html = ""
        icons = {"formation": "🌱", "extinction": "💀", "hybridization": "🔀",
                 "adaptation": "🔄", "radiation": "💥"}
        for ev in reversed(self.events[-50:]):
            icon = icons.get(ev.event_type, "📌")
            events_html += f"""<div class="event">
                <span class="icon">{icon}</span>
                <span class="tag {ev.event_type}">{ev.event_type.upper()}</span>
                <span>Round {ev.round_index} — {html_mod.escape(ev.description)}</span>
            </div>"""

        # Routing panel
        routing_html = ""
        for tt in self.active_task_types:
            rec = self.get_routing_recommendation(tt)
            bar_w = int(rec.confidence * 100)
            routing_html += f"""<div class="route-item">
                <strong>{html_mod.escape(tt)}</strong> → {html_mod.escape(rec.recommended_species_name)}
                <div class="bar"><div class="fill" style="width:{bar_w}%"></div></div>
                <small>{rec.confidence:.2f} confidence</small>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Swarm Speciation Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;padding:20px}}
h1{{color:#e94560;margin-bottom:10px}}
h2{{color:#0f3460;background:#16213e;padding:10px 16px;border-radius:6px;margin:20px 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}}
.card{{background:#16213e;border-radius:8px;padding:16px;text-align:center}}
.card .val{{font-size:2em;color:#e94560;font-weight:700}}
.card .lbl{{color:#8899aa;font-size:0.85em}}
.tbl{{width:100%;border-collapse:collapse;margin:10px 0}}
.tbl th,.tbl td{{padding:8px 10px;border:1px solid #0f3460;text-align:left}}
.tbl th{{background:#0f3460;color:#e0e0e0}}
.tbl tr:nth-child(even){{background:rgba(15,52,96,0.3)}}
.event{{padding:8px 12px;margin:4px 0;background:#16213e;border-radius:4px;display:flex;align-items:center;gap:8px}}
.event .icon{{font-size:1.3em}}
.tag{{padding:2px 8px;border-radius:3px;font-size:0.75em;font-weight:700;text-transform:uppercase}}
.formation{{background:#27ae60;color:#fff}}
.extinction{{background:#e74c3c;color:#fff}}
.hybridization{{background:#9b59b6;color:#fff}}
.adaptation{{background:#f39c12;color:#fff}}
.radiation{{background:#3498db;color:#fff}}
.route-item{{background:#16213e;padding:12px;margin:6px 0;border-radius:6px}}
.bar{{background:#0f3460;height:8px;border-radius:4px;margin:6px 0}}
.fill{{background:#e94560;height:100%;border-radius:4px;transition:width 0.3s}}
</style></head><body>
<h1>🧬 Swarm Speciation Dashboard</h1>
<p>Autonomous niche discovery — {eco.total_rounds} rounds analyzed</p>

<div class="grid">
    <div class="card"><div class="val">{eco.num_species}</div><div class="lbl">Species</div></div>
    <div class="card"><div class="val">{eco.num_agents}</div><div class="lbl">Agents</div></div>
    <div class="card"><div class="val">{eco.diversity_index:.2f}</div><div class="lbl">Shannon Diversity</div></div>
    <div class="card"><div class="val">{eco.stability_score:.0%}</div><div class="lbl">Stability</div></div>
    <div class="card"><div class="val">{eco.specialist_ratio:.0%}</div><div class="lbl">Specialists</div></div>
    <div class="card"><div class="val">{eco.generalist_ratio:.0%}</div><div class="lbl">Generalists</div></div>
</div>

<h2>🗺️ Species Map</h2>
<table class="tbl">
    <tr><th>Species</th><th>Pop</th><th>Avg Fitness</th><th>Diversity</th><th>Niche Width</th><th>Dominance</th><th>Members</th></tr>
    {species_rows}
</table>

<h2>🔥 Fitness Heatmap</h2>
{heatmap_html if heatmap_html else '<p style="color:#667">No fitness data yet</p>'}

<h2>📜 Speciation Timeline</h2>
{events_html if events_html else '<p style="color:#667">No events recorded</p>'}

<h2>🎯 Task Routing</h2>
{routing_html if routing_html else '<p style="color:#667">No routing data</p>'}

<h2>📊 Ecosystem Summary</h2>
<pre style="background:#16213e;padding:16px;border-radius:6px;overflow-x:auto">{html_mod.escape(json.dumps(eco.model_dump(), indent=2))}</pre>

<footer style="text-align:center;color:#445;margin-top:30px;font-size:0.85em">
    Swarm Speciation Engine — metacognition framework
</footer>
</body></html>"""


# ── Simulation ───────────────────────────────────────────────────────────


async def run_simulation(
    n_agents: int = 10,
    n_byzantine: int = 2,
    n_rounds: int = 40,
    n_task_types: int = 5,
    seed: int = 42,
) -> SpeciationEngine:
    """Run a simulation with agents that have hidden task affinities."""
    rng = random.Random(seed)
    engine = SpeciationEngine(seed=seed)
    task_types = TASK_TYPES[:n_task_types]

    # Create agents with hidden affinities
    agents: List[MockAgent] = []
    agent_affinities: Dict[str, Dict[str, float]] = {}

    for i in range(n_agents):
        aid = f"agent-{i:02d}"
        is_byz = i < n_byzantine
        # each agent has 1-2 strong task types
        strong_types = rng.sample(task_types, k=min(rng.randint(1, 2), len(task_types)))
        affinities = {}
        for tt in task_types:
            if tt in strong_types:
                affinities[tt] = rng.uniform(0.7, 0.95)
            else:
                affinities[tt] = rng.uniform(0.2, 0.5)
        agent_affinities[aid] = affinities
        agents.append(MockAgent(
            agent_id=aid,
            answer="yes",
            confidence=0.8,
            byzantine=is_byz,
        ))

    # Run rounds
    detection_interval = max(5, n_rounds // 8)
    for r in range(n_rounds):
        tt = task_types[r % len(task_types)]

        # adjust agent confidence based on affinity for this task type
        for agent in agents:
            aff = agent_affinities[agent.id].get(tt, 0.5)
            agent.confidence = max(0.1, min(0.99, aff + rng.gauss(0, 0.05)))

        mbft = MBFTEngine(agents=agents, threshold=2.0, max_rounds=2)
        result = await mbft.run(f"task-{r}-{tt}")
        if result:
            engine.record_round(tt, result)

        # periodic detection
        if (r + 1) % detection_interval == 0 or r == n_rounds - 1:
            engine.update_fitness_profiles()
            engine.detect_species()
            engine.detect_speciation_events()

    return engine


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Swarm Speciation Engine")
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--byzantine", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--task-types", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export", type=str, default=None, help="HTML output path")
    parser.add_argument("--json", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    engine = asyncio.run(run_simulation(
        n_agents=args.agents,
        n_byzantine=args.byzantine,
        n_rounds=args.rounds,
        n_task_types=args.task_types,
        seed=args.seed,
    ))

    eco = engine.get_ecosystem_summary()
    print(f"\n🧬 Swarm Speciation Report")
    print(f"{'─' * 50}")
    print(f"Agents: {eco.num_agents}  |  Species: {eco.num_species}  |  Rounds: {eco.total_rounds}")
    print(f"Diversity: {eco.diversity_index:.3f}  |  Stability: {eco.stability_score:.0%}")
    print(f"Specialists: {eco.specialist_ratio:.0%}  |  Generalists: {eco.generalist_ratio:.0%}")
    print(f"Events: {eco.total_events}")

    print(f"\n📊 Species:")
    for sp in engine.species:
        print(f"  {sp.name}: {sp.population} agents, fitness={sp.avg_fitness:.3f}, "
              f"niche_width={sp.niche_width}, dominance={sp.dominance_index:.3f}")

    print(f"\n📜 Events:")
    for ev in engine.events:
        icons = {"formation": "🌱", "extinction": "💀", "hybridization": "🔀",
                 "adaptation": "🔄", "radiation": "💥"}
        print(f"  {icons.get(ev.event_type, '📌')} [{ev.event_type}] Round {ev.round_index}: {ev.description}")

    print(f"\n🎯 Routing Recommendations:")
    for tt in engine.active_task_types:
        rec = engine.get_routing_recommendation(tt)
        print(f"  {tt} → {rec.recommended_species_name} ({rec.confidence:.2f})")

    if args.export:
        Path(args.export).write_text(engine.render_html(), encoding="utf-8")
        print(f"\n📄 HTML report: {args.export}")

    if args.json:
        Path(args.json).write_text(json.dumps(engine.export_json(), indent=2, default=str), encoding="utf-8")
        print(f"📄 JSON export: {args.json}")


if __name__ == "__main__":
    main()
