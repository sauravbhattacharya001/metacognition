"""Consensus Diplomacy Engine — autonomous inter-agent diplomatic negotiation.

Analyzes multi-round mBFT consensus to model diplomatic relationships:
faction detection, treaty tracking, alliance scoring, diplomatic pressure
analysis, and autonomous negotiation recommendations.

Usage::

    python -m src.diplomacy --agents 8 --byzantine 2 --rounds 40 --tasks 15
    python -m src.diplomacy --agents 10 --rounds 60 --auto-negotiate --out report.html
    python -m src.diplomacy --help
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

@dataclass
class DiplomaticEvent:
    """A diplomatic event during consensus rounds."""
    round_idx: int
    event_type: str  # alliance_formed, treaty_broken, faction_merged, capitulation, diplomatic_isolation
    agents: List[str]
    detail: str

@dataclass
class Treaty:
    """An implicit treaty between two factions."""
    faction_a: int
    faction_b: int
    formed_round: int
    broken_round: Optional[int] = None
    strength: float = 0.0

    @property
    def active(self) -> bool:
        return self.broken_round is None

@dataclass
class Faction:
    """A group of agents that vote together."""
    faction_id: int
    members: List[str]
    avg_vote: float = 0.0
    power: float = 0.0
    color: str = "#888"

# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

def _cosine_sim(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    return dot / (mag_a * mag_b)

# ---------------------------------------------------------------------------
# Diplomacy Engine
# ---------------------------------------------------------------------------

FACTION_COLORS = [
    "#58a6ff", "#3fb950", "#f0883e", "#f85149", "#bc8cff",
    "#d2a8ff", "#79c0ff", "#56d364", "#e3b341", "#ff7b72",
]

class DiplomacyEngine:
    """Runs consensus simulations and analyzes diplomatic dynamics."""

    def __init__(self, n_agents: int = 8, n_byzantine: int = 2,
                 n_rounds: int = 40, n_tasks: int = 15,
                 auto_negotiate: bool = False, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.n_agents = n_agents
        self.n_byzantine = n_byzantine
        self.n_rounds = n_rounds
        self.n_tasks = n_tasks
        self.auto_negotiate = auto_negotiate

        self.agent_names = [f"agent_{i}" for i in range(n_agents)]
        self.byzantine_set = set(random.sample(self.agent_names, min(n_byzantine, n_agents)))

        # vote_matrix[task][round] = {agent: vote_weight}
        self.vote_matrix: List[List[Dict[str, float]]] = []
        # Per-agent vote vectors across all (task, round) pairs for similarity
        self.agent_vectors: Dict[str, List[float]] = defaultdict(list)

        self.factions: List[Faction] = []
        self.treaties: List[Treaty] = []
        self.events: List[DiplomaticEvent] = []
        self.alliance_matrix: Dict[Tuple[str, str], float] = {}
        self.pressure_scores: Dict[str, float] = {}
        self.recommendations: List[str] = []

    # -----------------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------------

    def _simulate_round(self, task_idx: int, round_idx: int) -> Dict[str, float]:
        """Simulate one consensus round, returning agent votes."""
        votes: Dict[str, float] = {}
        for agent in self.agent_names:
            if agent in self.byzantine_set:
                # Byzantine agents flip-flop or collude
                if random.random() < 0.3:
                    votes[agent] = -random.uniform(0.3, 1.0)
                elif random.random() < 0.5:
                    # Collude with other byzantines
                    votes[agent] = random.uniform(-0.8, -0.2)
                else:
                    votes[agent] = random.uniform(-0.5, 0.5)
            else:
                # Honest agents have personality-based voting
                base = random.gauss(0.6, 0.25)
                # Add slight faction tendency (based on agent index parity)
                idx = int(agent.split("_")[1])
                faction_bias = 0.1 * math.sin(idx * 1.5)
                votes[agent] = max(-1.0, min(1.0, base + faction_bias))
        return votes

    def run_simulation(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the full simulation and analysis."""
        if verbose:
            print(f"=== Consensus Diplomacy Engine ===")
            print(f"Agents: {self.n_agents} ({self.n_byzantine} Byzantine)")
            print(f"Tasks: {self.n_tasks}, Rounds/task: {self.n_rounds}")
            print()

        # Phase 1: Run consensus rounds
        if verbose:
            print("Phase 1: Running consensus simulation...")
        for t in range(self.n_tasks):
            task_rounds = []
            for r in range(self.n_rounds):
                votes = self._simulate_round(t, r)
                task_rounds.append(votes)
                for agent, v in votes.items():
                    self.agent_vectors[agent].append(v)
            self.vote_matrix.append(task_rounds)

        # Phase 2: Detect factions
        if verbose:
            print("Phase 2: Detecting factions...")
        self._detect_factions()

        # Phase 3: Compute alliance strengths
        if verbose:
            print("Phase 3: Computing alliance strengths...")
        self._compute_alliances()

        # Phase 4: Detect treaties
        if verbose:
            print("Phase 4: Tracking treaties...")
        self._detect_treaties()

        # Phase 5: Measure diplomatic pressure
        if verbose:
            print("Phase 5: Analyzing diplomatic pressure...")
        self._measure_pressure()

        # Phase 6: Generate diplomatic events
        if verbose:
            print("Phase 6: Building diplomatic timeline...")
        self._generate_events()

        # Phase 7: Autonomous negotiation
        if self.auto_negotiate:
            if verbose:
                print("Phase 7: Running autonomous diplomat...")
            self._auto_negotiate()

        summary = self._build_summary()

        if verbose:
            self._print_summary(summary)

        return summary

    # -----------------------------------------------------------------------
    # Analysis methods
    # -----------------------------------------------------------------------

    def _detect_factions(self) -> None:
        """Cluster agents by voting similarity using greedy agglomerative approach."""
        n = len(self.agent_names)
        sim_matrix = {}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.agent_names[i], self.agent_names[j]
                sim = _cosine_sim(self.agent_vectors[a], self.agent_vectors[b])
                sim_matrix[(a, b)] = sim
                sim_matrix[(b, a)] = sim

        # Greedy clustering: agents with >0.7 similarity join same faction
        assigned: Dict[str, int] = {}
        fid = 0
        for agent in self.agent_names:
            if agent in assigned:
                continue
            members = [agent]
            assigned[agent] = fid
            for other in self.agent_names:
                if other in assigned:
                    continue
                # Check similarity with all current members
                sims = [sim_matrix.get((agent, other), 0.0) for agent in members]
                if all(s > 0.5 for s in sims):
                    members.append(other)
                    assigned[other] = fid
            vec = self.agent_vectors[members[0]]
            avg_v = sum(sum(self.agent_vectors[m]) / len(self.agent_vectors[m]) for m in members) / len(members)
            power = sum(len(self.agent_vectors[m]) for m in members)
            self.factions.append(Faction(
                faction_id=fid,
                members=members,
                avg_vote=round(avg_v, 3),
                power=round(power / (len(self.agent_names) * len(self.agent_vectors[self.agent_names[0]])), 3),
                color=FACTION_COLORS[fid % len(FACTION_COLORS)],
            ))
            fid += 1

    def _compute_alliances(self) -> None:
        """Compute bilateral alliance strength for every agent pair."""
        for i, a in enumerate(self.agent_names):
            for j, b in enumerate(self.agent_names):
                if i >= j:
                    continue
                sim = _cosine_sim(self.agent_vectors[a], self.agent_vectors[b])
                # Scale to [-1, 1]: strongly correlated = allied, anti-correlated = hostile
                score = round(sim, 3)
                self.alliance_matrix[(a, b)] = score
                self.alliance_matrix[(b, a)] = score

    def _detect_treaties(self) -> None:
        """Detect implicit treaties between factions based on non-opposition."""
        if len(self.factions) < 2:
            return
        for i, fa in enumerate(self.factions):
            for j, fb in enumerate(self.factions):
                if i >= j:
                    continue
                # Measure cross-faction agreement per task
                agreements = 0
                for t in range(self.n_tasks):
                    fa_avg = sum(
                        sum(self.vote_matrix[t][r].get(m, 0) for r in range(self.n_rounds)) / self.n_rounds
                        for m in fa.members
                    ) / max(len(fa.members), 1)
                    fb_avg = sum(
                        sum(self.vote_matrix[t][r].get(m, 0) for r in range(self.n_rounds)) / self.n_rounds
                        for m in fb.members
                    ) / max(len(fb.members), 1)
                    if fa_avg * fb_avg > 0:  # Same sign = agreement
                        agreements += 1

                ratio = agreements / max(self.n_tasks, 1)
                if ratio > 0.6:
                    self.treaties.append(Treaty(
                        faction_a=fa.faction_id,
                        faction_b=fb.faction_id,
                        formed_round=0,
                        strength=round(ratio, 3),
                    ))
                    # Check for late breaking
                    late_agreements = 0
                    late_tasks = max(1, self.n_tasks // 3)
                    for t in range(self.n_tasks - late_tasks, self.n_tasks):
                        fa_avg = sum(
                            sum(self.vote_matrix[t][r].get(m, 0) for r in range(self.n_rounds)) / self.n_rounds
                            for m in fa.members
                        ) / max(len(fa.members), 1)
                        fb_avg = sum(
                            sum(self.vote_matrix[t][r].get(m, 0) for r in range(self.n_rounds)) / self.n_rounds
                            for m in fb.members
                        ) / max(len(fb.members), 1)
                        if fa_avg * fb_avg > 0:
                            late_agreements += 1
                    if late_agreements / late_tasks < 0.4:
                        self.treaties[-1].broken_round = self.n_tasks - late_tasks

    def _measure_pressure(self) -> None:
        """Measure how much each agent influences others' voting shifts."""
        for agent in self.agent_names:
            influence = 0.0
            vec = self.agent_vectors[agent]
            for other in self.agent_names:
                if other == agent:
                    continue
                ovec = self.agent_vectors[other]
                # Measure if other's votes drift toward agent's over time
                half = len(vec) // 2
                if half < 2:
                    continue
                early_diff = sum(abs(vec[k] - ovec[k]) for k in range(half)) / half
                late_diff = sum(abs(vec[k] - ovec[k]) for k in range(half, len(vec))) / (len(vec) - half)
                if early_diff > late_diff + 0.05:
                    influence += (early_diff - late_diff)
            self.pressure_scores[agent] = round(influence, 3)

    def _generate_events(self) -> None:
        """Generate diplomatic event timeline."""
        # Faction formation events
        for f in self.factions:
            self.events.append(DiplomaticEvent(
                round_idx=0,
                event_type="alliance_formed",
                agents=f.members,
                detail=f"Faction {f.faction_id} formed with {len(f.members)} members (avg vote: {f.avg_vote})",
            ))

        # Treaty events
        for tr in self.treaties:
            self.events.append(DiplomaticEvent(
                round_idx=tr.formed_round,
                event_type="alliance_formed",
                agents=[],
                detail=f"Treaty between Faction {tr.faction_a} and Faction {tr.faction_b} (strength: {tr.strength})",
            ))
            if tr.broken_round is not None:
                self.events.append(DiplomaticEvent(
                    round_idx=tr.broken_round,
                    event_type="treaty_broken",
                    agents=[],
                    detail=f"Treaty broken between Faction {tr.faction_a} and Faction {tr.faction_b}",
                ))

        # Detect capitulations (agent shifts significantly toward dominant faction)
        if self.factions:
            dominant = max(self.factions, key=lambda f: f.power)
            for agent in self.agent_names:
                if agent in dominant.members:
                    continue
                vec = self.agent_vectors[agent]
                dom_vec = []
                for m in dominant.members[:1]:
                    dom_vec = self.agent_vectors[m]
                if not dom_vec:
                    continue
                half = len(vec) // 2
                if half < 2:
                    continue
                early_sim = _cosine_sim(vec[:half], dom_vec[:half])
                late_sim = _cosine_sim(vec[half:], dom_vec[half:])
                if late_sim - early_sim > 0.15:
                    self.events.append(DiplomaticEvent(
                        round_idx=half,
                        event_type="capitulation",
                        agents=[agent],
                        detail=f"{agent} shifted toward Faction {dominant.faction_id} (sim: {early_sim:.2f}->{late_sim:.2f})",
                    ))

        # Detect isolated agents
        for agent in self.agent_names:
            max_sim = 0.0
            for other in self.agent_names:
                if other == agent:
                    continue
                s = self.alliance_matrix.get((agent, other), 0.0)
                max_sim = max(max_sim, s)
            if max_sim < 0.3:
                self.events.append(DiplomaticEvent(
                    round_idx=0,
                    event_type="diplomatic_isolation",
                    agents=[agent],
                    detail=f"{agent} is diplomatically isolated (max alliance: {max_sim:.2f})",
                ))

        self.events.sort(key=lambda e: e.round_idx)

    def _auto_negotiate(self) -> None:
        """Recommend optimal alliances for honest agents."""
        honest = [a for a in self.agent_names if a not in self.byzantine_set]
        if len(honest) < 2:
            self.recommendations.append("Too few honest agents for negotiation.")
            return

        # Find the strongest honest coalition
        honest_sims = []
        for i, a in enumerate(honest):
            for j, b in enumerate(honest):
                if i >= j:
                    continue
                s = self.alliance_matrix.get((a, b), self.alliance_matrix.get((b, a), 0.0))
                honest_sims.append((a, b, s))
        honest_sims.sort(key=lambda x: x[2], reverse=True)

        self.recommendations.append("=== Autonomous Diplomat Recommendations ===")
        self.recommendations.append(f"Honest agents: {', '.join(honest)}")
        self.recommendations.append(f"Byzantine agents: {', '.join(self.byzantine_set)}")
        self.recommendations.append("")

        # Strongest honest pairs
        self.recommendations.append("Strongest honest alliances to reinforce:")
        for a, b, s in honest_sims[:3]:
            self.recommendations.append(f"  * {a} <-> {b}: {s:+.3f}")

        # Weakest honest pairs (need strengthening)
        self.recommendations.append("")
        self.recommendations.append("Weakest honest relationships (strengthen these):")
        for a, b, s in honest_sims[-3:]:
            self.recommendations.append(f"  * {a} <-> {b}: {s:+.3f}")

        # Agents under Byzantine pressure
        self.recommendations.append("")
        self.recommendations.append("Agents under Byzantine influence pressure:")
        for agent in honest:
            byz_sims = [
                self.alliance_matrix.get((agent, b), self.alliance_matrix.get((b, agent), 0.0))
                for b in self.byzantine_set
            ]
            max_byz = max(byz_sims) if byz_sims else 0.0
            if max_byz > 0.4:
                self.recommendations.append(f"  ! {agent} has {max_byz:.2f} alignment with a Byzantine agent")

        # Optimal coalition strategy
        self.recommendations.append("")
        avg_honest_sim = sum(s for _, _, s in honest_sims) / max(len(honest_sims), 1)
        if avg_honest_sim > 0.6:
            self.recommendations.append(f"Coalition health: STRONG ({avg_honest_sim:.2f} avg similarity)")
            self.recommendations.append("Strategy: Maintain current alignment, isolate Byzantine agents")
        elif avg_honest_sim > 0.3:
            self.recommendations.append(f"Coalition health: MODERATE ({avg_honest_sim:.2f} avg similarity)")
            self.recommendations.append("Strategy: Strengthen weakest honest links, coordinate voting")
        else:
            self.recommendations.append(f"Coalition health: WEAK ({avg_honest_sim:.2f} avg similarity)")
            self.recommendations.append("Strategy: Urgent realignment needed - honest agents too fragmented")

    # -----------------------------------------------------------------------
    # Summary / output
    # -----------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        return {
            "config": {
                "agents": self.n_agents,
                "byzantine": self.n_byzantine,
                "rounds": self.n_rounds,
                "tasks": self.n_tasks,
                "auto_negotiate": self.auto_negotiate,
            },
            "factions": [
                {"id": f.faction_id, "members": f.members, "avg_vote": f.avg_vote,
                 "power": f.power, "color": f.color}
                for f in self.factions
            ],
            "treaties": [
                {"faction_a": t.faction_a, "faction_b": t.faction_b,
                 "formed": t.formed_round, "broken": t.broken_round,
                 "strength": t.strength, "active": t.active}
                for t in self.treaties
            ],
            "alliance_heatmap": {
                "agents": self.agent_names,
                "scores": {f"{a},{b}": v for (a, b), v in self.alliance_matrix.items()},
            },
            "events": [
                {"round": e.round_idx, "type": e.event_type,
                 "agents": e.agents, "detail": e.detail}
                for e in self.events
            ],
            "pressure": self.pressure_scores,
            "recommendations": self.recommendations,
            "byzantine_agents": list(self.byzantine_set),
        }

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print("DIPLOMATIC ANALYSIS RESULTS")
        print(f"{'='*60}")

        print(f"\nFactions detected: {len(self.factions)}")
        for f in self.factions:
            byz = [m for m in f.members if m in self.byzantine_set]
            print(f"  Faction {f.faction_id}: {', '.join(f.members)}")
            print(f"    Power: {f.power:.3f} | Avg vote: {f.avg_vote:+.3f}"
                  + (f" | Byzantine: {', '.join(byz)}" if byz else ""))

        print(f"\nTreaties: {len(self.treaties)}")
        for t in self.treaties:
            status = "ACTIVE" if t.active else f"BROKEN (round {t.broken_round})"
            print(f"  F{t.faction_a} <-> F{t.faction_b}: strength={t.strength:.3f} [{status}]")

        print(f"\nDiplomatic pressure (top 5):")
        top = sorted(self.pressure_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for agent, score in top:
            byz = " [BYZ]" if agent in self.byzantine_set else ""
            print(f"  {agent}: {score:.3f}{byz}")

        print(f"\nDiplomatic events: {len(self.events)}")
        for e in self.events[:10]:
            print(f"  [{e.round_idx:3d}] {e.event_type}: {e.detail}")
        if len(self.events) > 10:
            print(f"  ... and {len(self.events) - 10} more")

        if self.recommendations:
            print()
            for r in self.recommendations:
                print(r)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return html_mod.escape(str(s))

def _alliance_color(score: float) -> str:
    """Map alliance score [-1, 1] to color."""
    if score > 0.6:
        return "#3fb950"
    elif score > 0.3:
        return "#56d364"
    elif score > 0:
        return "#2ea04380"
    elif score > -0.3:
        return "#f8514930"
    elif score > -0.6:
        return "#f85149"
    else:
        return "#da3633"

def generate_html_report(summary: Dict[str, Any]) -> str:
    cfg = summary["config"]
    factions = summary["factions"]
    treaties = summary["treaties"]
    events = summary["events"]
    pressure = summary["pressure"]
    heatmap = summary["alliance_heatmap"]
    recommendations = summary["recommendations"]
    byzantine = set(summary.get("byzantine_agents", []))

    # Faction table rows
    faction_rows = ""
    for f in factions:
        byz_members = [m for m in f["members"] if m in byzantine]
        byz_note = f' <span style="color:#f85149">({", ".join(byz_members)} Byzantine)</span>' if byz_members else ""
        faction_rows += f"""<tr>
            <td><span style="display:inline-block;width:14px;height:14px;background:{_esc(f['color'])};border-radius:3px;vertical-align:middle"></span> Faction {f['id']}</td>
            <td>{_esc(', '.join(f['members']))}{byz_note}</td>
            <td>{f['avg_vote']:+.3f}</td>
            <td>{f['power']:.3f}</td>
        </tr>"""

    # Alliance heatmap
    agents = heatmap["agents"]
    scores = heatmap["scores"]
    heatmap_html = '<table style="border-collapse:collapse;font-size:12px"><tr><th></th>'
    for a in agents:
        label = a.split("_")[1]
        heatmap_html += f'<th style="padding:4px;writing-mode:vertical-rl;transform:rotate(180deg)">{_esc(label)}</th>'
    heatmap_html += "</tr>"
    for a in agents:
        label_a = a.split("_")[1]
        byz_mark = " 🔴" if a in byzantine else ""
        heatmap_html += f'<tr><td style="padding:4px;font-weight:bold">{_esc(label_a)}{byz_mark}</td>'
        for b in agents:
            if a == b:
                heatmap_html += '<td style="background:#30363d;padding:4px;text-align:center">—</td>'
            else:
                s = scores.get(f"{a},{b}", 0.0)
                bg = _alliance_color(s)
                heatmap_html += f'<td style="background:{bg};padding:4px;text-align:center;color:#fff;font-size:11px">{s:+.2f}</td>'
        heatmap_html += "</tr>"
    heatmap_html += "</table>"

    # Treaty table
    treaty_rows = ""
    for t in treaties:
        status = '<span style="color:#3fb950">●&nbsp;ACTIVE</span>' if t["active"] else f'<span style="color:#f85149">✕&nbsp;BROKEN (task {t["broken"]})</span>'
        treaty_rows += f"""<tr>
            <td>Faction {t['faction_a']} ↔ Faction {t['faction_b']}</td>
            <td>{t['strength']:.3f}</td>
            <td>{status}</td>
        </tr>"""
    if not treaty_rows:
        treaty_rows = '<tr><td colspan="3" style="text-align:center;color:#8b949e">No treaties detected</td></tr>'

    # Events timeline
    event_type_icons = {
        "alliance_formed": "🤝",
        "treaty_broken": "💔",
        "faction_merged": "🔗",
        "capitulation": "🏳️",
        "diplomatic_isolation": "🏝️",
    }
    events_html = ""
    for e in events[:30]:
        icon = event_type_icons.get(e["type"], "📌")
        etype_color = "#3fb950" if "formed" in e["type"] else "#f85149" if "broken" in e["type"] else "#e3b341" if e["type"] == "capitulation" else "#8b949e"
        events_html += f"""<div style="display:flex;gap:12px;padding:8px 0;border-bottom:1px solid #21262d">
            <div style="min-width:40px;color:#8b949e;font-size:13px">T{e['round']}</div>
            <div style="font-size:16px">{icon}</div>
            <div>
                <span style="color:{etype_color};font-weight:600;font-size:13px">{_esc(e['type'].replace('_', ' ').title())}</span><br>
                <span style="color:#c9d1d9;font-size:13px">{_esc(e['detail'])}</span>
            </div>
        </div>"""

    # Pressure bar chart (CSS)
    sorted_pressure = sorted(pressure.items(), key=lambda x: x[1], reverse=True)
    max_p = max((v for _, v in sorted_pressure), default=1.0) or 1.0
    pressure_html = ""
    for agent, score in sorted_pressure:
        pct = (score / max_p) * 100
        byz_mark = ' <span style="color:#f85149">[BYZ]</span>' if agent in byzantine else ""
        pressure_html += f"""<div style="display:flex;align-items:center;gap:8px;margin:4px 0">
            <div style="min-width:70px;font-size:13px">{_esc(agent)}{byz_mark}</div>
            <div style="flex:1;background:#21262d;border-radius:4px;height:18px">
                <div style="width:{pct:.0f}%;background:#58a6ff;height:100%;border-radius:4px"></div>
            </div>
            <div style="min-width:50px;text-align:right;font-size:13px">{score:.3f}</div>
        </div>"""

    # Faction power bars
    max_pow = max((f["power"] for f in factions), default=1.0) or 1.0
    power_html = ""
    for f in factions:
        pct = (f["power"] / max_pow) * 100
        power_html += f"""<div style="display:flex;align-items:center;gap:8px;margin:6px 0">
            <div style="min-width:80px;font-size:13px">Faction {f['id']}</div>
            <div style="flex:1;background:#21262d;border-radius:4px;height:22px">
                <div style="width:{pct:.0f}%;background:{f['color']};height:100%;border-radius:4px"></div>
            </div>
            <div style="min-width:50px;text-align:right;font-size:13px">{f['power']:.3f}</div>
        </div>"""

    # Recommendations
    rec_html = ""
    if recommendations:
        for r in recommendations:
            if r.startswith("==="):
                rec_html += f'<h3 style="color:#58a6ff;margin:16px 0 8px">{_esc(r.strip("= "))}</h3>'
            elif r.startswith("  *") or r.startswith("  !"):
                rec_html += f'<div style="padding:2px 0 2px 16px;font-size:13px;color:#c9d1d9">{_esc(r)}</div>'
            elif r.strip():
                rec_html += f'<div style="padding:4px 0;font-size:13px;color:#e3b341;font-weight:600">{_esc(r)}</div>'
    else:
        rec_html = '<p style="color:#8b949e">Run with --auto-negotiate for recommendations</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mBFT Consensus Diplomacy Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; background: #0d1117; color: #c9d1d9; padding: 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; font-size: 28px; margin-bottom: 8px; }}
  h2 {{ color: #c9d1d9; font-size: 20px; margin: 32px 0 12px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
  .subtitle {{ color: #8b949e; font-size: 14px; margin-bottom: 24px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 20px; flex: 1; min-width: 140px; }}
  .stat-value {{ font-size: 28px; font-weight: bold; color: #58a6ff; }}
  .stat-label {{ font-size: 13px; color: #8b949e; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; padding: 8px; border-bottom: 2px solid #30363d; color: #8b949e; font-size: 13px; }}
  td {{ padding: 8px; border-bottom: 1px solid #21262d; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
  <h1>🕊️ Consensus Diplomacy Engine</h1>
  <div class="subtitle">mBFT Inter-Agent Diplomatic Analysis | {cfg['agents']} agents, {cfg['byzantine']} Byzantine, {cfg['tasks']} tasks × {cfg['rounds']} rounds</div>

  <div class="stats">
    <div class="stat"><div class="stat-value">{len(factions)}</div><div class="stat-label">Factions</div></div>
    <div class="stat"><div class="stat-value">{len(treaties)}</div><div class="stat-label">Treaties</div></div>
    <div class="stat"><div class="stat-value">{sum(1 for t in treaties if t['active'])}</div><div class="stat-label">Active Treaties</div></div>
    <div class="stat"><div class="stat-value">{len(events)}</div><div class="stat-label">Diplomatic Events</div></div>
    <div class="stat"><div class="stat-value">{cfg['byzantine']}</div><div class="stat-label">Byzantine Agents</div></div>
  </div>

  <h2>🏛️ Factions</h2>
  <div class="card">
    <table>
      <tr><th>Faction</th><th>Members</th><th>Avg Vote</th><th>Power</th></tr>
      {faction_rows}
    </table>
  </div>

  <h2>⚡ Faction Power</h2>
  <div class="card">{power_html}</div>

  <h2>🗺️ Alliance Heatmap</h2>
  <div class="card" style="overflow-x:auto">{heatmap_html}</div>

  <h2>📜 Treaties</h2>
  <div class="card">
    <table>
      <tr><th>Parties</th><th>Strength</th><th>Status</th></tr>
      {treaty_rows}
    </table>
  </div>

  <h2>💪 Diplomatic Pressure</h2>
  <div class="card">{pressure_html}</div>

  <h2>📅 Diplomatic Events Timeline</h2>
  <div class="card">{events_html}</div>

  <h2>🤖 Autonomous Diplomat</h2>
  <div class="card">{rec_html}</div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Diplomacy Engine — autonomous inter-agent diplomatic negotiation"
    )
    parser.add_argument("--agents", type=int, default=8, help="Number of agents (default: 8)")
    parser.add_argument("--byzantine", type=int, default=2, help="Number of Byzantine agents (default: 2)")
    parser.add_argument("--rounds", type=int, default=40, help="Rounds per task (default: 40)")
    parser.add_argument("--tasks", type=int, default=15, help="Number of tasks (default: 15)")
    parser.add_argument("--auto-negotiate", action="store_true", help="Enable autonomous diplomat recommendations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Output file (html or json)")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal output")
    args = parser.parse_args()

    engine = DiplomacyEngine(
        n_agents=args.agents,
        n_byzantine=args.byzantine,
        n_rounds=args.rounds,
        n_tasks=args.tasks,
        auto_negotiate=args.auto_negotiate,
        seed=args.seed,
    )
    summary = engine.run_simulation(verbose=not args.quiet)

    if args.out:
        path = Path(args.out)
        if path.suffix == ".json":
            path.write_text(json.dumps(summary, indent=2))
        else:
            path.write_text(generate_html_report(summary))
        if not args.quiet:
            print(f"\nExported to {path}")


if __name__ == "__main__":
    main()
