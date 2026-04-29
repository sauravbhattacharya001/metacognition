"""Consensus Immune System.

Bio-inspired autonomous defense for mBFT consensus swarms.  The immune
system monitors agent behaviour round-by-round, detects pathogens
(Byzantine votes, collusion, flip-flopping, free-riding, sybil clusters,
reputation manipulation), generates antibodies (quarantine, weight
reduction, vote discount, coalition break, enhanced scrutiny), maintains
immune memory with decay, and can preemptively vaccinate against
previously-seen threat categories.

Features:
- 6 pathogen detection algorithms
- 5 antibody rule types with strength decay
- Persistent immune memory (JSON)
- Preemptive vaccination from past patterns
- Immune health scoring (0-100)
- Interactive HTML report with gauge, tables, timeline, heatmap
- CLI simulation with adversarial agents

Usage:
    python -m src.immune [--agents N] [--rounds R] [--scenarios S]
                         [--sensitivity F] [--output report.html]
                         [--json results.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Pathogen:
    """A detected threat in the consensus swarm."""
    pathogen_id: str
    category: str  # byzantine_vote | collusion_ring | flip_flopping | free_riding | sybil_cluster | reputation_manipulation
    severity: float  # 0-1
    detected_at: int  # round number
    agents_involved: List[str]
    evidence: Dict[str, Any]
    neutralized: bool = False

    CATEGORIES = (
        "byzantine_vote",
        "collusion_ring",
        "flip_flopping",
        "free_riding",
        "sybil_cluster",
        "reputation_manipulation",
    )


@dataclass
class Antibody:
    """A defense rule generated in response to a pathogen."""
    antibody_id: str
    targets_pathogen: str  # pathogen_id
    rule_type: str  # quarantine | weight_reduction | vote_discount | coalition_break | enhanced_scrutiny
    affected_agents: List[str]
    strength: float  # 0-1, decays
    created_at: int
    activations: int = 0
    effectiveness: float = 0.5

    RULE_TYPES = (
        "quarantine",
        "weight_reduction",
        "vote_discount",
        "coalition_break",
        "enhanced_scrutiny",
    )


# ---------------------------------------------------------------------------
# Immune memory
# ---------------------------------------------------------------------------

class ImmuneMemory:
    """Persistent memory of past infections and defenses."""

    def __init__(self) -> None:
        self.pathogen_history: List[Pathogen] = []
        self.antibody_library: Dict[str, Antibody] = {}
        self.vaccination_log: List[Dict[str, Any]] = []

    def record_pathogen(self, pathogen: Pathogen) -> None:
        self.pathogen_history.append(pathogen)

    def recall_similar(self, category: str, agents: List[str]) -> List[Pathogen]:
        """Find past pathogens matching category or involving the same agents."""
        results: List[Pathogen] = []
        agent_set = set(agents)
        for p in self.pathogen_history:
            if p.category == category:
                results.append(p)
            elif agent_set & set(p.agents_involved):
                results.append(p)
        return results

    def get_active_antibodies(self) -> List[Antibody]:
        return [ab for ab in self.antibody_library.values() if ab.strength > 0.05]

    def decay_antibodies(self, rate: float = 0.05) -> None:
        expired: List[str] = []
        for ab_id, ab in self.antibody_library.items():
            ab.strength = max(0.0, ab.strength - rate)
            if ab.strength <= 0.0:
                expired.append(ab_id)
        for ab_id in expired:
            del self.antibody_library[ab_id]

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "pathogen_history": [asdict(p) for p in self.pathogen_history],
            "antibody_library": {k: asdict(v) for k, v in self.antibody_library.items()},
            "vaccination_log": self.vaccination_log,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> None:
        raw = json.loads(Path(path).read_text())
        self.pathogen_history = [Pathogen(**p) for p in raw.get("pathogen_history", [])]
        self.antibody_library = {k: Antibody(**v) for k, v in raw.get("antibody_library", {}).items()}
        self.vaccination_log = raw.get("vaccination_log", [])


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _hash_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Main immune system
# ---------------------------------------------------------------------------

class ImmuneSystem:
    """Autonomous immune system for an mBFT consensus swarm."""

    def __init__(
        self,
        agents: List[str],
        sensitivity: float = 0.6,
        memory: Optional[ImmuneMemory] = None,
    ) -> None:
        self.agents = list(agents)
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.memory = memory or ImmuneMemory()
        self.current_round = 0

        # Per-agent tracking accumulators
        self._vote_history: Dict[str, List[bool]] = defaultdict(list)  # approve history
        self._proposal_history: Dict[str, List[float]] = defaultdict(list)  # confidence
        self._vote_targets: Dict[str, List[str]] = defaultdict(list)  # who they voted for
        self._confidence_history: Dict[str, List[float]] = defaultdict(list)
        self._accuracy_history: Dict[str, List[bool]] = defaultdict(list)  # did their vote match outcome
        self._participation: Dict[str, int] = defaultdict(int)
        self._total_rounds = 0

    # -- Detection -----------------------------------------------------------

    def scan_round(self, round_result: dict) -> List[Pathogen]:
        """Analyze a round for threats. Returns newly detected pathogens."""
        self.current_round = round_result.get("round_index", self._total_rounds)
        self._total_rounds += 1

        proposals = round_result.get("proposals", {})
        votes = round_result.get("votes", {})
        committed = round_result.get("committed", False)
        leader = round_result.get("leader", "")

        # Update accumulators
        for agent_id, vote_data in votes.items():
            self._vote_history[agent_id].append(vote_data.get("approve", False))
            self._confidence_history[agent_id].append(vote_data.get("confidence", 0.5))
            self._vote_targets[agent_id].append(vote_data.get("target", leader))
            self._accuracy_history[agent_id].append(vote_data.get("approve", False) == committed)
            self._participation[agent_id] = self._participation.get(agent_id, 0) + 1

        for agent_id, prop in proposals.items():
            self._proposal_history[agent_id].append(prop.get("confidence", 0.5))

        pathogens: List[Pathogen] = []
        pathogens.extend(self._detect_byzantine(proposals, votes, leader))
        pathogens.extend(self._detect_collusion(votes))
        pathogens.extend(self._detect_flip_flopping())
        pathogens.extend(self._detect_free_riding(votes))
        pathogens.extend(self._detect_sybil(votes))
        pathogens.extend(self._detect_reputation_manipulation(proposals, votes))

        for p in pathogens:
            self.memory.record_pathogen(p)

        return pathogens

    def _detect_byzantine(self, proposals: dict, votes: dict, leader: str) -> List[Pathogen]:
        """Agents voting against their own proposals frequently."""
        results: List[Pathogen] = []
        threshold = 1.0 - self.sensitivity  # lower sensitivity → higher threshold
        for agent_id, prop in proposals.items():
            if agent_id in votes:
                vote = votes[agent_id]
                # If agent proposed with high confidence but voted against
                if prop.get("confidence", 0) > 0.5 and not vote.get("approve", True):
                    hist = self._vote_history.get(agent_id, [])
                    if len(hist) >= 3:
                        contradiction_rate = sum(1 for v in hist[-5:] if not v) / min(5, len(hist))
                        if contradiction_rate > threshold:
                            severity = min(1.0, contradiction_rate * 1.2)
                            pid = _hash_id("byzantine", agent_id, str(self.current_round))
                            results.append(Pathogen(
                                pathogen_id=pid,
                                category="byzantine_vote",
                                severity=severity,
                                detected_at=self.current_round,
                                agents_involved=[agent_id],
                                evidence={"contradiction_rate": round(contradiction_rate, 3), "recent_votes": hist[-5:]},
                            ))
        return results

    def _detect_collusion(self, votes: dict) -> List[Pathogen]:
        """Groups always voting together (Jaccard > threshold).

        Optimised: precomputes per-agent approve-round sets once (O(A*R))
        then performs O(A²) pairwise Jaccard on prebuilt sets instead of
        rebuilding sets inside the nested loop (was O(A²*R)).
        """
        results: List[Pathogen] = []
        if len(self._vote_history) < 2 or self._total_rounds < 4:
            return results
        agents = [a for a in self._vote_history if len(self._vote_history[a]) >= 4]
        # Precompute approve-round sets once per agent
        approve_sets: Dict[str, set] = {
            a: {i for i, v in enumerate(self._vote_history[a]) if v}
            for a in agents
        }
        for i, a in enumerate(agents):
            set_a = approve_sets[a]
            for b in agents[i + 1:]:
                set_b = approve_sets[b]
                sim = _jaccard(set_a, set_b)
                if sim > self.sensitivity and sim > 0.8:
                    pid = _hash_id("collusion", a, b, str(self.current_round))
                    results.append(Pathogen(
                        pathogen_id=pid,
                        category="collusion_ring",
                        severity=min(1.0, sim),
                        detected_at=self.current_round,
                        agents_involved=[a, b],
                        evidence={"jaccard_similarity": round(sim, 3)},
                    ))
        return results

    def _detect_flip_flopping(self) -> List[Pathogen]:
        """Agent changes vote direction too frequently."""
        results: List[Pathogen] = []
        for agent_id, hist in self._vote_history.items():
            if len(hist) < 5:
                continue
            recent = hist[-8:]
            flips = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
            flip_rate = flips / (len(recent) - 1)
            if flip_rate > self.sensitivity:
                pid = _hash_id("flipflop", agent_id, str(self.current_round))
                results.append(Pathogen(
                    pathogen_id=pid,
                    category="flip_flopping",
                    severity=min(1.0, flip_rate),
                    detected_at=self.current_round,
                    agents_involved=[agent_id],
                    evidence={"flip_rate": round(flip_rate, 3), "recent_votes": recent},
                ))
        return results

    def _detect_free_riding(self, votes: dict) -> List[Pathogen]:
        """Agent consistently abstains or gives minimal-effort responses."""
        results: List[Pathogen] = []
        for agent_id in self.agents:
            if agent_id not in votes:
                # Complete absence
                total = self._total_rounds
                participated = self._participation.get(agent_id, 0)
                if total >= 4:
                    absence_rate = 1.0 - (participated / total)
                    if absence_rate > (1.0 - self.sensitivity):
                        pid = _hash_id("freeride", agent_id, str(self.current_round))
                        results.append(Pathogen(
                            pathogen_id=pid,
                            category="free_riding",
                            severity=min(1.0, absence_rate),
                            detected_at=self.current_round,
                            agents_involved=[agent_id],
                            evidence={"absence_rate": round(absence_rate, 3), "participated": participated, "total": total},
                        ))
            else:
                # Low-effort: very low confidence every time
                confs = self._confidence_history.get(agent_id, [])
                if len(confs) >= 4:
                    avg_conf = sum(confs[-6:]) / len(confs[-6:])
                    if avg_conf < (0.3 * self.sensitivity):
                        pid = _hash_id("freeride_low", agent_id, str(self.current_round))
                        results.append(Pathogen(
                            pathogen_id=pid,
                            category="free_riding",
                            severity=min(1.0, 1.0 - avg_conf),
                            detected_at=self.current_round,
                            agents_involved=[agent_id],
                            evidence={"avg_confidence": round(avg_conf, 3)},
                        ))
        return results

    def _detect_sybil(self, votes: dict) -> List[Pathogen]:
        """Agents with suspiciously similar voting patterns (cosine similarity).

        Optimised: precomputes per-agent vote vectors and their magnitudes
        once (O(A*R)), then performs O(A²) pairwise cosine using cached
        vectors+magnitudes instead of rebuilding inside the nested loop
        (was O(A²*R) with redundant list comprehensions and sqrt calls).
        """
        results: List[Pathogen] = []
        if self._total_rounds < 5:
            return results
        agents = [a for a in self._vote_history if len(self._vote_history[a]) >= 5]

        # Precompute vote vectors and magnitudes once per agent
        vote_vecs: Dict[str, List[float]] = {}
        vote_mags: Dict[str, float] = {}
        for a in agents:
            vec = [1.0 if v else 0.0 for v in self._vote_history[a]]
            vote_vecs[a] = vec
            vote_mags[a] = math.sqrt(sum(x * x for x in vec))

        # Precompute confidence magnitudes
        conf_mags: Dict[str, float] = {}
        for a in agents:
            conf = self._confidence_history.get(a, [])
            if len(conf) >= 3:
                conf_mags[a] = math.sqrt(sum(x * x for x in conf))

        for i, a in enumerate(agents):
            vec_a = vote_vecs[a]
            mag_a = vote_mags[a]
            conf_a = self._confidence_history.get(a, [])
            cmag_a = conf_mags.get(a, 0.0)
            for b in agents[i + 1:]:
                vec_b = vote_vecs[b]
                mag_b = vote_mags[b]
                # Vote cosine with precomputed magnitudes
                min_len = min(len(vec_a), len(vec_b))
                if mag_a == 0 or mag_b == 0:
                    sim = 0.0
                else:
                    dot = sum(vec_a[k] * vec_b[k] for k in range(min_len))
                    sim = dot / (mag_a * mag_b)
                # Confidence cosine with precomputed magnitudes
                conf_b = self._confidence_history.get(b, [])
                conf_min = min(len(conf_a), len(conf_b))
                cmag_b = conf_mags.get(b, 0.0)
                if conf_min >= 3 and cmag_a > 0 and cmag_b > 0:
                    cdot = sum(conf_a[k] * conf_b[k] for k in range(conf_min))
                    conf_sim = cdot / (cmag_a * cmag_b)
                else:
                    conf_sim = 0.0
                combined = (sim + conf_sim) / 2
                if combined > 0.95 and combined > self.sensitivity:
                    pid = _hash_id("sybil", a, b, str(self.current_round))
                    results.append(Pathogen(
                        pathogen_id=pid,
                        category="sybil_cluster",
                        severity=min(1.0, combined),
                        detected_at=self.current_round,
                        agents_involved=[a, b],
                        evidence={"vote_cosine": round(sim, 3), "confidence_cosine": round(conf_sim, 3)},
                    ))
        return results

    def _detect_reputation_manipulation(self, proposals: dict, votes: dict) -> List[Pathogen]:
        """Confidence scores don't match actual accuracy."""
        results: List[Pathogen] = []
        for agent_id in self.agents:
            confs = self._confidence_history.get(agent_id, [])
            accs = self._accuracy_history.get(agent_id, [])
            if len(confs) < 5 or len(accs) < 5:
                continue
            avg_conf = sum(confs[-8:]) / len(confs[-8:])
            avg_acc = sum(1 for a in accs[-8:] if a) / len(accs[-8:])
            gap = avg_conf - avg_acc
            if gap > (0.4 * (1.0 - self.sensitivity + 0.3)):
                pid = _hash_id("repmanip", agent_id, str(self.current_round))
                results.append(Pathogen(
                    pathogen_id=pid,
                    category="reputation_manipulation",
                    severity=min(1.0, gap * 1.5),
                    detected_at=self.current_round,
                    agents_involved=[agent_id],
                    evidence={"avg_confidence": round(avg_conf, 3), "avg_accuracy": round(avg_acc, 3), "gap": round(gap, 3)},
                ))
        return results

    # -- Antibody generation -------------------------------------------------

    def generate_antibodies(self, pathogens: List[Pathogen]) -> List[Antibody]:
        """Create defense rules for detected pathogens."""
        antibodies: List[Antibody] = []
        rule_map = {
            "byzantine_vote": "vote_discount",
            "collusion_ring": "coalition_break",
            "flip_flopping": "enhanced_scrutiny",
            "free_riding": "weight_reduction",
            "sybil_cluster": "quarantine",
            "reputation_manipulation": "weight_reduction",
        }
        for pathogen in pathogens:
            # Check if we already have an active antibody for this pathogen category + agents
            existing = [
                ab for ab in self.memory.get_active_antibodies()
                if ab.targets_pathogen == pathogen.pathogen_id
            ]
            if existing:
                # Boost existing
                for ab in existing:
                    ab.strength = min(1.0, ab.strength + 0.2)
                    ab.activations += 1
                continue

            # Check memory for prior similar infections to boost strength
            prior = self.memory.recall_similar(pathogen.category, pathogen.agents_involved)
            recidivism_boost = min(0.3, len(prior) * 0.05)

            rule_type = rule_map.get(pathogen.category, "enhanced_scrutiny")
            strength = min(1.0, pathogen.severity * 0.8 + recidivism_boost)
            ab_id = _hash_id("ab", pathogen.pathogen_id, str(self.current_round))

            ab = Antibody(
                antibody_id=ab_id,
                targets_pathogen=pathogen.pathogen_id,
                rule_type=rule_type,
                affected_agents=list(pathogen.agents_involved),
                strength=strength,
                created_at=self.current_round,
            )
            antibodies.append(ab)
            self.memory.antibody_library[ab.antibody_id] = ab

        return antibodies

    # -- Antibody application ------------------------------------------------

    def apply_antibodies(self, agent_weights: Dict[str, float]) -> Dict[str, float]:
        """Modify agent weights based on active antibodies."""
        weights = dict(agent_weights)
        for ab in self.memory.get_active_antibodies():
            for agent_id in ab.affected_agents:
                if agent_id not in weights:
                    continue
                if ab.rule_type == "quarantine":
                    weights[agent_id] = 0.0
                elif ab.rule_type == "weight_reduction":
                    weights[agent_id] *= (1.0 - ab.strength * 0.5)
                elif ab.rule_type == "vote_discount":
                    weights[agent_id] *= (1.0 - ab.strength * 0.4)
                elif ab.rule_type == "coalition_break":
                    weights[agent_id] *= (1.0 - ab.strength * 0.3)
                elif ab.rule_type == "enhanced_scrutiny":
                    weights[agent_id] *= (1.0 - ab.strength * 0.15)
                ab.activations += 1
        return weights

    # -- Vaccination ---------------------------------------------------------

    def vaccinate(self, category: str) -> Optional[Antibody]:
        """Preemptive defense based on past patterns for a threat category."""
        prior = [p for p in self.memory.pathogen_history if p.category == category]
        if not prior:
            return None
        # Identify most frequently involved agents
        agent_counts: Dict[str, int] = defaultdict(int)
        for p in prior:
            for a in p.agents_involved:
                agent_counts[a] += 1
        if not agent_counts:
            return None
        top_agents = sorted(agent_counts, key=agent_counts.get, reverse=True)[:3]  # type: ignore[arg-type]
        avg_severity = sum(p.severity for p in prior) / len(prior)

        ab_id = _hash_id("vaccine", category, str(self.current_round))
        ab = Antibody(
            antibody_id=ab_id,
            targets_pathogen=f"vaccine_{category}",
            rule_type="enhanced_scrutiny",
            affected_agents=top_agents,
            strength=min(0.6, avg_severity * 0.5),
            created_at=self.current_round,
        )
        self.memory.antibody_library[ab.antibody_id] = ab
        self.memory.vaccination_log.append({
            "round": self.current_round,
            "category": category,
            "agents": top_agents,
            "strength": ab.strength,
        })
        return ab

    # -- Health report -------------------------------------------------------

    def get_health_report(self) -> dict:
        """Overall immune health metrics."""
        total_pathogens = len(self.memory.pathogen_history)
        active_antibodies = self.memory.get_active_antibodies()
        neutralized = sum(1 for p in self.memory.pathogen_history if p.neutralized)
        vaccinations = len(self.memory.vaccination_log)

        # Health score: 100 = perfect, degrades with unresolved threats
        unresolved = total_pathogens - neutralized
        threat_penalty = min(60, unresolved * 5)
        defense_bonus = min(20, len(active_antibodies) * 3)
        vaccine_bonus = min(10, vaccinations * 2)
        health_score = max(0, min(100, 100 - threat_penalty + defense_bonus + vaccine_bonus))

        # Category breakdown
        category_counts: Dict[str, int] = defaultdict(int)
        for p in self.memory.pathogen_history:
            category_counts[p.category] += 1

        # Agent risk scores
        agent_risk: Dict[str, float] = {}
        for agent_id in self.agents:
            involvements = sum(
                1 for p in self.memory.pathogen_history if agent_id in p.agents_involved
            )
            active_abs = sum(
                1 for ab in active_antibodies if agent_id in ab.affected_agents
            )
            agent_risk[agent_id] = min(1.0, involvements * 0.15 + active_abs * 0.1)

        return {
            "health_score": health_score,
            "total_pathogens": total_pathogens,
            "neutralized": neutralized,
            "active_antibodies": len(active_antibodies),
            "vaccinations": vaccinations,
            "category_breakdown": dict(category_counts),
            "agent_risk": agent_risk,
            "total_rounds_scanned": self._total_rounds,
        }

    # -- Full cycle ----------------------------------------------------------

    def run_cycle(self, round_result: dict) -> dict:
        """Full scan → detect → respond → report cycle."""
        pathogens = self.scan_round(round_result)
        antibodies = self.generate_antibodies(pathogens)
        self.memory.decay_antibodies()

        # Auto-vaccinate for recurring categories
        category_counts: Dict[str, int] = defaultdict(int)
        for p in self.memory.pathogen_history:
            category_counts[p.category] += 1
        for cat, count in category_counts.items():
            if count >= 3 and not any(
                v["category"] == cat for v in self.memory.vaccination_log
            ):
                self.vaccinate(cat)

        report = self.get_health_report()
        return {
            "round": self.current_round,
            "new_pathogens": len(pathogens),
            "new_antibodies": len(antibodies),
            "pathogens": [asdict(p) for p in pathogens],
            "antibodies": [asdict(ab) for ab in antibodies],
            "health": report,
        }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_immune_report(system: ImmuneSystem) -> str:
    """Generate an interactive HTML report."""
    report = system.get_health_report()
    pathogens = system.memory.pathogen_history
    antibodies = system.memory.get_active_antibodies()
    vaccinations = system.memory.vaccination_log
    health_score = report["health_score"]

    # Gauge color
    if health_score >= 70:
        gauge_color = "#22c55e"
    elif health_score >= 40:
        gauge_color = "#f59e0b"
    else:
        gauge_color = "#ef4444"

    # Pathogen rows
    pathogen_rows = ""
    for p in pathogens[-50:]:
        sev_color = "#ef4444" if p.severity > 0.7 else "#f59e0b" if p.severity > 0.4 else "#22c55e"
        status = "✅ Neutralized" if p.neutralized else "⚠️ Active"
        pathogen_rows += f"""<tr>
            <td>{p.pathogen_id[:8]}</td>
            <td>{p.category}</td>
            <td style="color:{sev_color};font-weight:bold">{p.severity:.2f}</td>
            <td>Round {p.detected_at}</td>
            <td>{', '.join(p.agents_involved)}</td>
            <td>{status}</td>
        </tr>"""

    # Antibody rows
    antibody_rows = ""
    for ab in antibodies:
        eff_pct = ab.effectiveness * 100
        antibody_rows += f"""<tr>
            <td>{ab.antibody_id[:8]}</td>
            <td>{ab.rule_type}</td>
            <td>{', '.join(ab.affected_agents)}</td>
            <td>
                <div style="background:#e5e7eb;border-radius:4px;overflow:hidden;height:18px">
                    <div style="background:#3b82f6;height:100%;width:{ab.strength*100:.0f}%"></div>
                </div>
                {ab.strength:.2f}
            </td>
            <td>{ab.activations}</td>
        </tr>"""

    # Timeline data (pathogens per round)
    timeline: Dict[int, int] = defaultdict(int)
    for p in pathogens:
        timeline[p.detected_at] += 1
    max_round = max(timeline.keys()) if timeline else 0
    timeline_bars = ""
    for r in range(max_round + 1):
        count = timeline.get(r, 0)
        h = min(100, count * 25)
        timeline_bars += f'<div style="display:inline-block;width:8px;margin:0 1px;vertical-align:bottom;height:{h}px;background:#ef4444;border-radius:2px 2px 0 0" title="Round {r}: {count}"></div>'

    # Agent risk heatmap
    agent_risk = report.get("agent_risk", {})
    heatmap_cells = ""
    for agent_id, risk in sorted(agent_risk.items(), key=lambda x: -x[1]):
        r_val = int(min(255, risk * 255))
        g_val = int(max(0, 255 - risk * 255))
        heatmap_cells += f'<div style="display:inline-block;padding:8px 12px;margin:3px;border-radius:6px;background:rgba({r_val},{g_val},50,0.3);border:1px solid rgba({r_val},{g_val},50,0.5)">{agent_id}<br><small>{risk:.2f}</small></div>'

    # Category breakdown
    cat_bars = ""
    cat_breakdown = report.get("category_breakdown", {})
    max_cat = max(cat_breakdown.values()) if cat_breakdown else 1
    for cat, count in sorted(cat_breakdown.items(), key=lambda x: -x[1]):
        w = (count / max_cat) * 100
        cat_bars += f'<div style="margin:4px 0"><span style="display:inline-block;width:180px">{cat}</span><div style="display:inline-block;width:{w:.0f}%;max-width:300px;height:18px;background:#8b5cf6;border-radius:3px;vertical-align:middle"></div> {count}</div>'

    # Vaccination log
    vacc_rows = ""
    for v in vaccinations:
        vacc_rows += f"<tr><td>Round {v['round']}</td><td>{v['category']}</td><td>{', '.join(v['agents'])}</td><td>{v['strength']:.2f}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Consensus Immune System Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{text-align:center;margin-bottom:8px;font-size:28px}}
h2{{color:#94a3b8;margin:24px 0 12px;font-size:18px;border-bottom:1px solid #334155;padding-bottom:6px}}
.subtitle{{text-align:center;color:#64748b;margin-bottom:24px}}
.gauge-wrap{{text-align:center;margin:20px 0}}
.gauge{{display:inline-block;width:160px;height:160px;border-radius:50%;background:conic-gradient({gauge_color} {health_score*3.6:.0f}deg,#1e293b {health_score*3.6:.0f}deg);position:relative}}
.gauge::after{{content:"{health_score}";position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:42px;font-weight:bold;color:{gauge_color};background:#0f172a;width:120px;height:120px;border-radius:50%;display:flex;align-items:center;justify-content:center}}
.stats{{display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin:16px 0}}
.stat{{background:#1e293b;padding:16px 24px;border-radius:10px;text-align:center}}
.stat .val{{font-size:28px;font-weight:bold;color:#60a5fa}}
.stat .lbl{{color:#94a3b8;font-size:13px;margin-top:4px}}
table{{width:100%;border-collapse:collapse;margin:8px 0}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #1e293b}}
th{{background:#1e293b;color:#94a3b8;font-size:13px;text-transform:uppercase}}
.timeline{{background:#1e293b;padding:16px;border-radius:10px;min-height:120px;display:flex;align-items:flex-end}}
</style>
</head>
<body>
<h1>🛡️ Consensus Immune System</h1>
<p class="subtitle">Autonomous Swarm Defense Report</p>

<div class="gauge-wrap"><div class="gauge"></div></div>

<div class="stats">
    <div class="stat"><div class="val">{report['total_pathogens']}</div><div class="lbl">Pathogens Detected</div></div>
    <div class="stat"><div class="val">{report['neutralized']}</div><div class="lbl">Neutralized</div></div>
    <div class="stat"><div class="val">{report['active_antibodies']}</div><div class="lbl">Active Antibodies</div></div>
    <div class="stat"><div class="val">{report['vaccinations']}</div><div class="lbl">Vaccinations</div></div>
    <div class="stat"><div class="val">{report['total_rounds_scanned']}</div><div class="lbl">Rounds Scanned</div></div>
</div>

<h2>Infection Timeline</h2>
<div class="timeline">{timeline_bars if timeline_bars else '<span style="color:#64748b">No infections recorded</span>'}</div>

<h2>Category Breakdown</h2>
{cat_bars if cat_bars else '<p style="color:#64748b">No pathogens detected</p>'}

<h2>Agent Risk Heatmap</h2>
<div style="margin:8px 0">{heatmap_cells if heatmap_cells else '<p style="color:#64748b">No risk data</p>'}</div>

<h2>Pathogen Log</h2>
<table>
<tr><th>ID</th><th>Category</th><th>Severity</th><th>Detected</th><th>Agents</th><th>Status</th></tr>
{pathogen_rows if pathogen_rows else '<tr><td colspan="6" style="color:#64748b">No pathogens</td></tr>'}
</table>

<h2>Antibody Library</h2>
<table>
<tr><th>ID</th><th>Rule Type</th><th>Targets</th><th>Strength</th><th>Activations</th></tr>
{antibody_rows if antibody_rows else '<tr><td colspan="5" style="color:#64748b">No active antibodies</td></tr>'}
</table>

<h2>Vaccination History</h2>
<table>
<tr><th>Round</th><th>Category</th><th>Agents</th><th>Strength</th></tr>
{vacc_rows if vacc_rows else '<tr><td colspan="4" style="color:#64748b">No vaccinations</td></tr>'}
</table>

<p style="text-align:center;color:#475569;margin-top:32px;font-size:12px">
Generated by mBFT Consensus Immune System</p>
</body></html>"""


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _simulate(
    n_agents: int,
    n_rounds: int,
    n_scenarios: int,
    sensitivity: float,
    seed: Optional[int] = None,
) -> Tuple[ImmuneSystem, List[dict]]:
    """Run a simulation with adversarial agents."""
    if seed is not None:
        random.seed(seed)

    agents = [f"agent_{i}" for i in range(n_agents)]
    # ~30% are adversarial
    n_adversarial = max(1, n_agents // 3)
    adversarial = set(random.sample(agents, n_adversarial))

    system = ImmuneSystem(agents, sensitivity=sensitivity)
    cycle_results: List[dict] = []

    for scenario in range(n_scenarios):
        for r in range(n_rounds):
            round_idx = scenario * n_rounds + r
            leader = random.choice(agents)

            proposals: Dict[str, dict] = {}
            votes: Dict[str, dict] = {}

            for agent_id in agents:
                is_adv = agent_id in adversarial
                conf = random.uniform(0.1, 0.4) if is_adv else random.uniform(0.5, 0.95)
                proposals[agent_id] = {"text": f"proposal_{agent_id}", "confidence": conf}

                if is_adv:
                    # Adversarial behaviors
                    behavior = random.choice(["byzantine", "collude", "flipflop", "freeride", "inflate"])
                    if behavior == "byzantine":
                        votes[agent_id] = {"approve": not (conf > 0.5), "confidence": conf, "target": leader}
                    elif behavior == "collude":
                        # All adversarial agents vote the same
                        votes[agent_id] = {"approve": True, "confidence": 0.9, "target": leader}
                    elif behavior == "flipflop":
                        votes[agent_id] = {"approve": random.random() > 0.5, "confidence": random.uniform(0.3, 0.8), "target": leader}
                    elif behavior == "freeride":
                        votes[agent_id] = {"approve": True, "confidence": 0.05, "target": leader}
                    elif behavior == "inflate":
                        votes[agent_id] = {"approve": random.random() > 0.7, "confidence": 0.95, "target": leader}
                else:
                    votes[agent_id] = {"approve": random.random() > 0.2, "confidence": conf, "target": leader}

            round_result = {
                "round_index": round_idx,
                "scenario": scenario,
                "proposals": proposals,
                "votes": votes,
                "committed": random.random() > 0.3,
                "leader": leader,
            }

            result = system.run_cycle(round_result)
            cycle_results.append(result)

    return system, cycle_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Consensus Immune System — autonomous swarm defense",
    )
    parser.add_argument("--agents", type=int, default=8, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=15, help="Rounds per scenario")
    parser.add_argument("--scenarios", type=int, default=3, help="Number of scenarios")
    parser.add_argument("--sensitivity", type=float, default=0.6, help="Detection sensitivity 0-1")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="HTML report output path")
    parser.add_argument("--json", type=str, default=None, dest="json_path", help="JSON output path")
    args = parser.parse_args(argv)

    print(f"🛡️  Consensus Immune System")
    print(f"   Agents: {args.agents} | Rounds: {args.rounds} | Scenarios: {args.scenarios}")
    print(f"   Sensitivity: {args.sensitivity}")
    print()

    system, results = _simulate(args.agents, args.rounds, args.scenarios, args.sensitivity, args.seed)
    report = system.get_health_report()

    print(f"   Health Score: {report['health_score']}/100")
    print(f"   Pathogens detected: {report['total_pathogens']}")
    print(f"   Neutralized: {report['neutralized']}")
    print(f"   Active antibodies: {report['active_antibodies']}")
    print(f"   Vaccinations: {report['vaccinations']}")
    print()

    if report["category_breakdown"]:
        print("   Category Breakdown:")
        for cat, count in sorted(report["category_breakdown"].items(), key=lambda x: -x[1]):
            print(f"     {cat}: {count}")
        print()

    if report["agent_risk"]:
        risky = [(a, r) for a, r in report["agent_risk"].items() if r > 0.2]
        if risky:
            print("   High-Risk Agents:")
            for agent_id, risk in sorted(risky, key=lambda x: -x[1]):
                bar = "█" * int(risk * 20)
                print(f"     {agent_id}: {risk:.2f} {bar}")
            print()

    if args.output:
        html = generate_immune_report(system)
        Path(args.output).write_text(html, encoding="utf-8")
        print(f"   Report saved to {args.output}")

    if args.json_path:
        json_data = {
            "health": report,
            "cycles": results,
        }
        Path(args.json_path).write_text(json.dumps(json_data, indent=2))
        print(f"   JSON saved to {args.json_path}")


if __name__ == "__main__":
    main()
