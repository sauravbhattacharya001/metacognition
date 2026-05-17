"""Agent lifecycle advisor.

An *agentic* per-agent lifecycle planner for the mBFT consensus engine.
Where :class:`~src.swarm_health.SwarmHealthMonitor` answers
"is the swarm OK?" in aggregate, :class:`~src.disagreement_forensics.DisagreementForensics`
answers "why did *this round* fail?", and
:class:`~src.round_replay_advisor.RoundReplayAdvisor` answers "what could
have flipped that round?", this advisor answers a different question:

    *Which agents should I keep, watch, probe, demote, evict, or reinstate
    - and what is the projected swarm impact?*

It is a roster-level planner, not a round-level analyzer. Given the
engine's history + current reputation map, every agent gets a verdict on
the lifecycle ladder

    KEEP -> WATCH -> PROBE -> REINSTATE -> DEMOTE -> EVICT

a 0-100 ``lifecycle_risk`` score, structured ``reasons`` (e.g.
``CHRONIC_BLOCKER``, ``BAD_LEADER``, ``LOW_REPUTATION``,
``INACTIVE``, ``STAR_PERFORMER``), and a cheap projected commit-rate
delta if the verdict is removal-shaped. A cross-swarm playbook then
collapses individual verdicts into a small set of dedicated patterns
(``BYZANTINE_CLUSTER``, ``BAD_LEADER_BENCH``, ``CAPACITY_LOSS_RISK``,
``ECHO_CHAMBER``, ``STAR_PROMOTION``, ``INACTIVE_AGENT_SWEEP``,
``HEALTHY_FLEET``).

Design notes:
* zero new dependencies (``pydantic`` only, matching the rest of mBFT);
* deterministic and stateless - never mutates the engine, history, or
  reputation map;
* the advisor *recommends*; the operator (or a higher-level autopilot)
  *decides*.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from src.core.state import RoundResult, Vote


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


class AgentLifecycle(BaseModel):
    agent_id: str
    verdict: str  # KEEP | WATCH | PROBE | REINSTATE | DEMOTE | EVICT
    priority: str  # P0 | P1 | P2 | P3
    lifecycle_risk: float  # 0..100
    current_reputation: float
    times_led: int
    lead_success_rate: float
    votes_cast: int
    rejections_cast: int
    rejection_rate: float
    agreement_with_committed: int
    disagreement_with_committed: int
    calibration_score: float
    times_slashed: int
    rounds_inactive: int
    chronic_blocker_count: int
    reasons: List[str] = Field(default_factory=list)
    projected_commit_rate_delta: float = 0.0


class PlaybookItem(BaseModel):
    pattern: str
    priority: str
    targets: List[str] = Field(default_factory=list)
    reason: str
    expected_impact: str


class LifecycleAdvisorReport(BaseModel):
    generated_at: datetime
    rounds_analyzed: int
    agents: List[AgentLifecycle] = Field(default_factory=list)
    playbook: List[PlaybookItem] = Field(default_factory=list)
    overall_grade: str
    summary_headline: str

    # -- renderers ----------------------------------------------------------

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("Agent Lifecycle Advisor")
        lines.append("=" * 70)
        lines.append(f"generated_at:    {self.generated_at.isoformat()}")
        lines.append(f"rounds_analyzed: {self.rounds_analyzed}")
        lines.append(f"overall_grade:   {self.overall_grade}")
        lines.append(f"headline:        {self.summary_headline}")
        lines.append("")
        lines.append("-- Agents --")
        if not self.agents:
            lines.append("(no agents observed)")
        else:
            lines.append(
                f"{'agent':<14} {'verdict':<10} {'pri':<4} "
                f"{'risk':>5} {'rep':>6} {'led':>4} {'lead%':>6} "
                f"{'votes':>5} {'rej%':>5} reasons"
            )
            for a in self.agents:
                reasons = ",".join(a.reasons) if a.reasons else "-"
                lines.append(
                    f"{a.agent_id:<14} {a.verdict:<10} {a.priority:<4} "
                    f"{a.lifecycle_risk:>5.1f} {a.current_reputation:>6.3f} "
                    f"{a.times_led:>4d} {a.lead_success_rate*100:>5.1f}% "
                    f"{a.votes_cast:>5d} {a.rejection_rate*100:>4.1f}% "
                    f"{reasons}"
                )
        lines.append("")
        lines.append("-- Playbook --")
        if not self.playbook:
            lines.append("(none)")
        else:
            for item in self.playbook:
                tgt = ",".join(item.targets) if item.targets else "-"
                lines.append(f"[{item.priority}] {item.pattern} :: targets={tgt}")
                lines.append(f"    reason: {item.reason}")
                lines.append(f"    impact: {item.expected_impact}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# Agent Lifecycle Advisor")
        lines.append("")
        lines.append(f"- **generated_at:** `{self.generated_at.isoformat()}`")
        lines.append(f"- **rounds_analyzed:** {self.rounds_analyzed}")
        lines.append(f"- **overall_grade:** **{self.overall_grade}**")
        lines.append(f"- **headline:** {self.summary_headline}")
        lines.append("")
        lines.append("## Agents")
        lines.append("")
        if not self.agents:
            lines.append("_no agents observed_")
        else:
            lines.append(
                "| agent | verdict | priority | risk | rep | led | "
                "lead_success | votes | rej_rate | inactive | reasons |"
            )
            lines.append(
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
            )
            for a in self.agents:
                reasons = ", ".join(a.reasons) if a.reasons else "-"
                lines.append(
                    f"| `{a.agent_id}` | {a.verdict} | {a.priority} | "
                    f"{a.lifecycle_risk:.1f} | {a.current_reputation:.3f} | "
                    f"{a.times_led} | {a.lead_success_rate*100:.1f}% | "
                    f"{a.votes_cast} | {a.rejection_rate*100:.1f}% | "
                    f"{a.rounds_inactive} | {reasons} |"
                )
        lines.append("")
        lines.append("## Playbook")
        lines.append("")
        if not self.playbook:
            lines.append("_none_")
        else:
            for item in self.playbook:
                tgt = ", ".join(f"`{t}`" for t in item.targets) if item.targets else "_(swarm-wide)_"
                lines.append(f"### [{item.priority}] {item.pattern}")
                lines.append("")
                lines.append(f"- **targets:** {tgt}")
                lines.append(f"- **reason:** {item.reason}")
                lines.append(f"- **expected_impact:** {item.expected_impact}")
                lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            default=str,
            indent=2,
            sort_keys=True,
        )


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


_RISK_MULTIPLIER = {
    "cautious": 1.10,
    "balanced": 1.00,
    "aggressive": 0.90,
}


class AgentLifecycleAdvisor:
    """Per-agent lifecycle planner over an mBFT engine's history."""

    def __init__(
        self,
        eviction_threshold: float = 0.125,
        min_rounds_for_verdict: int = 3,
        watch_risk: float = 45.0,
    ) -> None:
        self.eviction_threshold = eviction_threshold
        self.min_rounds_for_verdict = min_rounds_for_verdict
        self.watch_risk = watch_risk

    # -- public entry -------------------------------------------------------

    def analyze(
        self,
        history: Sequence[RoundResult],
        reputation: Mapping[str, float],
        slash_factor: Optional[float] = None,
        risk_appetite: str = "balanced",
        now: Optional[Callable[[], datetime]] = None,
    ) -> LifecycleAdvisorReport:
        if risk_appetite not in _RISK_MULTIPLIER:
            raise ValueError(f"unknown risk_appetite: {risk_appetite!r}")
        clock = now or (lambda: datetime.now(timezone.utc))

        rounds_analyzed = len(history)
        agent_ids = set(reputation.keys())
        for r in history:
            agent_ids.add(r.leader_id)
            for v in r.votes:
                agent_ids.add(v.voter_id)

        profiles = self._build_profiles(history, reputation, agent_ids)

        agents: List[AgentLifecycle] = []
        for aid in sorted(agent_ids):
            p = profiles[aid]
            risk, reasons = self._score(p, rounds_analyzed, risk_appetite)
            verdict, priority, extra_reasons = self._classify(
                p, risk, rounds_analyzed
            )
            for er in extra_reasons:
                if er not in reasons:
                    reasons.append(er)
            delta = self._projected_delta(
                verdict, p, rounds_analyzed
            )
            agents.append(
                AgentLifecycle(
                    agent_id=aid,
                    verdict=verdict,
                    priority=priority,
                    lifecycle_risk=round(risk, 2),
                    current_reputation=round(p["reputation"], 4),
                    times_led=p["times_led"],
                    lead_success_rate=round(p["lead_success_rate"], 4),
                    votes_cast=p["votes_cast"],
                    rejections_cast=p["rejections_cast"],
                    rejection_rate=round(p["rejection_rate"], 4),
                    agreement_with_committed=p["agreement_with_committed"],
                    disagreement_with_committed=p["disagreement_with_committed"],
                    calibration_score=round(p["calibration_score"], 4),
                    times_slashed=p["times_slashed"],
                    rounds_inactive=p["rounds_inactive"],
                    chronic_blocker_count=p["chronic_blocker_count"],
                    reasons=reasons,
                    projected_commit_rate_delta=round(delta, 4),
                )
            )

        # Sort: lifecycle_risk desc, then agent_id asc
        agents.sort(key=lambda a: (-a.lifecycle_risk, a.agent_id))

        playbook = self._playbook(agents, profiles, rounds_analyzed)
        grade = self._grade(agents, playbook)
        headline = self._headline(agents, playbook, rounds_analyzed, grade)

        return LifecycleAdvisorReport(
            generated_at=clock(),
            rounds_analyzed=rounds_analyzed,
            agents=agents,
            playbook=playbook,
            overall_grade=grade,
            summary_headline=headline,
        )

    # -- profiling ----------------------------------------------------------

    def _build_profiles(
        self,
        history: Sequence[RoundResult],
        reputation: Mapping[str, float],
        agent_ids: set,
    ) -> Dict[str, dict]:
        profiles: Dict[str, dict] = {}
        for aid in agent_ids:
            profiles[aid] = {
                "reputation": reputation.get(aid, 1.0),
                "times_led": 0,
                "lead_committed": 0,
                "votes_cast": 0,
                "rejections_cast": 0,
                "agreement_with_committed": 0,
                "disagreement_with_committed": 0,
                "times_slashed": 0,
                "last_active_round": -1,
                "chronic_blocker_count": 0,
                "recent_activity": [],  # list of (round_idx, voted_positive_on_committed)
            }

        if not history:
            for aid, p in profiles.items():
                p["lead_success_rate"] = 0.0
                p["rejection_rate"] = 0.0
                relevant = p["agreement_with_committed"] + p["disagreement_with_committed"]
                p["calibration_score"] = 0.5 if relevant == 0 else (
                    p["agreement_with_committed"] / relevant
                )
                p["rounds_inactive"] = 0
            return profiles

        last_idx = max(r.round_index for r in history)

        for r in history:
            p_leader = profiles[r.leader_id]
            p_leader["times_led"] += 1
            p_leader["last_active_round"] = max(
                p_leader["last_active_round"], r.round_index
            )
            if r.committed:
                p_leader["lead_committed"] += 1

            for slashed_id in r.slashed:
                if slashed_id in profiles:
                    profiles[slashed_id]["times_slashed"] += 1

            # Identify chronic blocker contributions: voter rejected with
            # full reputation in a round that did NOT commit and aggregate
            # met threshold (i.e. veto was the blocker)
            unrefuted_blocker = (
                not r.committed
                and r.aggregate_weight >= r.threshold
            )

            for v in r.votes:
                p_v = profiles[v.voter_id]
                p_v["votes_cast"] += 1
                p_v["last_active_round"] = max(
                    p_v["last_active_round"], r.round_index
                )
                if v.is_rejection:
                    p_v["rejections_cast"] += 1
                if r.committed:
                    if v.weight > 0:
                        p_v["agreement_with_committed"] += 1
                        p_v["recent_activity"].append((r.round_index, True))
                    elif v.is_rejection:
                        p_v["disagreement_with_committed"] += 1
                        p_v["recent_activity"].append((r.round_index, False))
                if unrefuted_blocker and v.is_rejection:
                    p_v["chronic_blocker_count"] += 1

        for aid, p in profiles.items():
            p["lead_success_rate"] = (
                p["lead_committed"] / p["times_led"]
                if p["times_led"] > 0
                else 0.0
            )
            p["rejection_rate"] = (
                p["rejections_cast"] / p["votes_cast"]
                if p["votes_cast"] > 0
                else 0.0
            )
            relevant = (
                p["agreement_with_committed"] + p["disagreement_with_committed"]
            )
            p["calibration_score"] = (
                p["agreement_with_committed"] / relevant
                if relevant > 0
                else 0.5
            )
            if p["last_active_round"] < 0:
                p["rounds_inactive"] = len(history)
            else:
                p["rounds_inactive"] = max(0, last_idx - p["last_active_round"])

        return profiles

    # -- scoring ------------------------------------------------------------

    def _score(
        self,
        p: dict,
        rounds_analyzed: int,
        risk_appetite: str,
    ) -> tuple:
        components: List[float] = []
        reasons: List[str] = []

        # reputation pressure (always counts)
        rep_pressure = max(0.0, min(60.0, (1.0 - p["reputation"]) * 60.0))
        components.append(rep_pressure)
        if p["reputation"] < 0.5:
            reasons.append("LOW_REPUTATION")

        if p["chronic_blocker_count"] > 0:
            cb = min(75.0, p["chronic_blocker_count"] * 25.0)
            components.append(cb)
            reasons.append("CHRONIC_BLOCKER")

        if p["times_led"] >= 2:
            bad = (1.0 - p["lead_success_rate"]) * 50.0
            components.append(max(0.0, min(50.0, bad)))
            if p["lead_success_rate"] < 0.5:
                reasons.append("BAD_LEADER")

        relevant = (
            p["agreement_with_committed"] + p["disagreement_with_committed"]
        )
        if relevant >= 2:
            mis = (1.0 - p["calibration_score"]) * 35.0
            components.append(max(0.0, min(35.0, mis)))
            if p["calibration_score"] < 0.5:
                reasons.append("MISCALIBRATED")

        if p["rounds_inactive"] >= 3:
            inact = min(40.0, p["rounds_inactive"] * 8.0)
            components.append(inact)
            reasons.append("INACTIVE")

        if not components:
            return 0.0, reasons

        raw = sum(components) / len(components)
        raw *= _RISK_MULTIPLIER[risk_appetite]
        return max(0.0, min(100.0, raw)), reasons

    # -- classify -----------------------------------------------------------

    def _classify(
        self,
        p: dict,
        risk: float,
        rounds_analyzed: int,
    ) -> tuple:
        extra: List[str] = []
        observations = p["times_led"] + p["votes_cast"]

        # EVICT
        if (
            p["reputation"] < self.eviction_threshold
            and (p["chronic_blocker_count"] >= 2 or risk >= 75.0)
        ):
            return "EVICT", "P0", extra

        # DEMOTE
        if p["chronic_blocker_count"] >= 2 or (
            p["times_led"] >= 2 and p["lead_success_rate"] < 0.34
        ):
            return "DEMOTE", "P0", extra

        # REINSTATE (recovering previously-slashed agent)
        if (
            p["times_slashed"] >= 1
            and p["reputation"] >= 0.5
            and self._recovering(p)
        ):
            extra.append("REPUTATION_RECOVERING")
            return "REINSTATE", "P1", extra

        # WATCH
        if risk >= self.watch_risk:
            return "WATCH", "P1", extra

        # PROBE (insufficient observations)
        if observations < self.min_rounds_for_verdict:
            extra.append("INSUFFICIENT_DATA")
            return "PROBE", "P2", extra

        # KEEP, with STAR_PERFORMER flag if applicable
        if (
            p["times_led"] >= 2
            and p["lead_success_rate"] >= 0.75
            and p["calibration_score"] >= 0.75
        ):
            extra.append("STAR_PERFORMER")
        return "KEEP", "P3", extra

    @staticmethod
    def _recovering(p: dict) -> bool:
        # Check last 2 logged activity entries: must be agreements (True).
        recent = p["recent_activity"][-2:]
        if not recent:
            return False
        return all(flag for (_, flag) in recent)

    # -- projected delta ----------------------------------------------------

    def _projected_delta(
        self,
        verdict: str,
        p: dict,
        rounds_analyzed: int,
    ) -> float:
        if verdict not in ("EVICT", "DEMOTE"):
            return 0.0
        denom = max(1, rounds_analyzed)
        delta = p["chronic_blocker_count"] / denom
        if p["times_led"] >= 2 and p["lead_success_rate"] < 0.5:
            bad_leads = p["times_led"] - p["lead_committed"]
            delta += bad_leads / denom
        return min(0.5, delta)

    # -- playbook -----------------------------------------------------------

    def _playbook(
        self,
        agents: List[AgentLifecycle],
        profiles: Dict[str, dict],
        rounds_analyzed: int,
    ) -> List[PlaybookItem]:
        items: List[PlaybookItem] = []

        evict = [a for a in agents if a.verdict == "EVICT"]
        demote = [a for a in agents if a.verdict == "DEMOTE"]

        chronic = [a for a in agents if a.chronic_blocker_count >= 2]
        if len(chronic) >= 2:
            blocked_rounds = sum(a.chronic_blocker_count for a in chronic)
            items.append(
                PlaybookItem(
                    pattern="BYZANTINE_CLUSTER",
                    priority="P0",
                    targets=[a.agent_id for a in chronic],
                    reason=(
                        f"{len(chronic)} agents persistently veto committed "
                        f"rounds ({blocked_rounds} cumulative blocked votes)."
                    ),
                    expected_impact=(
                        "Removing or sandboxing this cluster should restore "
                        "commit throughput."
                    ),
                )
            )

        bad_leaders = [
            a for a in demote
            if a.times_led >= 2 and a.lead_success_rate < 0.34
        ]
        if len(bad_leaders) >= 2:
            items.append(
                PlaybookItem(
                    pattern="BAD_LEADER_BENCH",
                    priority="P0",
                    targets=[a.agent_id for a in bad_leaders],
                    reason=(
                        f"{len(bad_leaders)} agents have led >=2 rounds with "
                        "<34% commit rate."
                    ),
                    expected_impact=(
                        "Raise the leader-selection bar (confidence*reputation) "
                        "or rotate these agents out of leader eligibility."
                    ),
                )
            )

        swarm_size = len(agents)
        if swarm_size > 0 and len(evict) >= max(1, int(0.30 * swarm_size + 0.999)):
            items.append(
                PlaybookItem(
                    pattern="CAPACITY_LOSS_RISK",
                    priority="P1",
                    targets=[a.agent_id for a in evict],
                    reason=(
                        f"{len(evict)}/{swarm_size} agents are EVICT candidates "
                        "(>=30% of swarm)."
                    ),
                    expected_impact=(
                        "Provision replacements before evicting to avoid losing "
                        "quorum capacity."
                    ),
                )
            )

        # ECHO_CHAMBER: across non-slashed voters, overall rejection rate < 5%
        if rounds_analyzed >= 5:
            non_slashed_votes = 0
            non_slashed_rejs = 0
            for aid, p in profiles.items():
                if p["reputation"] >= 1.0 and p["votes_cast"] > 0:
                    non_slashed_votes += p["votes_cast"]
                    non_slashed_rejs += p["rejections_cast"]
            if non_slashed_votes >= 5:
                rate = non_slashed_rejs / non_slashed_votes
                if rate < 0.05:
                    items.append(
                        PlaybookItem(
                            pattern="ECHO_CHAMBER",
                            priority="P1",
                            targets=[],
                            reason=(
                                f"Non-slashed voter rejection rate is "
                                f"{rate*100:.1f}% over {non_slashed_votes} votes."
                            ),
                            expected_impact=(
                                "Inject adversarial voters or raise the threshold "
                                "to surface real disagreement."
                            ),
                        )
                    )

        stars = [a for a in agents if "STAR_PERFORMER" in a.reasons]
        if stars:
            items.append(
                PlaybookItem(
                    pattern="STAR_PROMOTION",
                    priority="P2",
                    targets=[a.agent_id for a in stars],
                    reason=(
                        f"{len(stars)} agent(s) lead reliably with high calibration."
                    ),
                    expected_impact=(
                        "Pin as fallback leader on next view-change."
                    ),
                )
            )

        inactives = [a for a in agents if a.rounds_inactive >= 5]
        if len(inactives) >= 2:
            items.append(
                PlaybookItem(
                    pattern="INACTIVE_AGENT_SWEEP",
                    priority="P2",
                    targets=[a.agent_id for a in inactives],
                    reason=(
                        f"{len(inactives)} agents inactive >=5 rounds."
                    ),
                    expected_impact=(
                        "Retire idle agents to free namespace and reduce "
                        "leader-election overhead."
                    ),
                )
            )

        if not items and rounds_analyzed >= 3:
            items.append(
                PlaybookItem(
                    pattern="HEALTHY_FLEET",
                    priority="P3",
                    targets=[],
                    reason=(
                        "No structural lifecycle issues detected across the swarm."
                    ),
                    expected_impact=(
                        "Maintain current configuration; revisit on next degraded run."
                    ),
                )
            )

        # P0-first ordering + dedupe by pattern
        seen = set()
        ordered: List[PlaybookItem] = []
        for item in sorted(items, key=lambda x: (_PRIORITY_RANK[x.priority], x.pattern)):
            if item.pattern in seen:
                continue
            seen.add(item.pattern)
            ordered.append(item)
        return ordered

    # -- grade & headline ---------------------------------------------------

    def _grade(
        self,
        agents: List[AgentLifecycle],
        playbook: List[PlaybookItem],
    ) -> str:
        if not agents:
            return "A"
        mean_risk = sum(a.lifecycle_risk for a in agents) / len(agents)
        p0_count = sum(1 for item in playbook if item.priority == "P0")
        evict_count = sum(1 for a in agents if a.verdict == "EVICT")

        if evict_count >= 1 or p0_count >= 2 or mean_risk >= 70.0:
            return "F"
        if p0_count >= 1 or mean_risk >= 55.0:
            return "D"
        if mean_risk >= 35.0:
            return "C"
        if mean_risk >= 18.0:
            return "B"
        return "A"

    def _headline(
        self,
        agents: List[AgentLifecycle],
        playbook: List[PlaybookItem],
        rounds_analyzed: int,
        grade: str,
    ) -> str:
        if rounds_analyzed == 0:
            return "No rounds observed yet - run the engine before planning lifecycle changes."
        bucket = defaultdict(int)
        for a in agents:
            bucket[a.verdict] += 1
        parts = [
            f"{bucket[v]} {v.lower()}"
            for v in ("EVICT", "DEMOTE", "WATCH", "REINSTATE", "PROBE", "KEEP")
            if bucket[v]
        ]
        verdict_str = ", ".join(parts) if parts else "no agents observed"
        top_pattern = playbook[0].pattern if playbook else "no_pattern"
        return (
            f"Grade {grade} over {rounds_analyzed} round(s): "
            f"{verdict_str}. Top playbook item: {top_pattern}."
        )


__all__ = [
    "AgentLifecycle",
    "PlaybookItem",
    "LifecycleAdvisorReport",
    "AgentLifecycleAdvisor",
]
