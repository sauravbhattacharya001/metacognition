"""Leader rotation advisor.

An *agentic* next-N-rounds leader rotation planner for the mBFT consensus
engine. Where :class:`~src.swarm_health.SwarmHealthMonitor` aggregates fleet
health, :class:`~src.agent_lifecycle_advisor.AgentLifecycleAdvisor` decides
keep/watch/evict, and
:class:`~src.voting_coalition_detector.VotingCoalitionDetector` looks for
factions, this advisor answers:

    *Who should lead the next N rounds - and in what order?*

For every known agent we score 0-100 ``lead_fitness`` over five signals
(lead-success-rate, calibration, reputation, recency pressure, and
penalties), assign a verdict on the rotation ladder
``LEAD_NOW -> LEAD_SOON -> STANDBY -> BENCH / SKIP / INSUFFICIENT_DATA``,
build a deterministic forced-leader queue of length ``horizon`` with a
per-slot expected commit probability + confidence band, and emit a small
cross-swarm playbook (``BREAK_LEADER_CAPTURE``,
``EMERGENCY_BACKUP_LEADER``, ``ROTATE_OUT_BLOCKER``,
``PROMOTE_RISING_STAR``, ``ELEVATE_CALIBRATED_FOLLOWER``,
``REBALANCE_LEADERSHIP_LOAD``, ``ADD_REDUNDANT_LEADER_CANDIDATES``,
``HEALTHY_ROTATION``).

Design notes:
* zero new dependencies (``pydantic`` only, matching the rest of mBFT);
* deterministic and read-only - never mutates engine/history/reputation;
* the advisor *recommends*; the operator (or autopilot) *decides*.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from src.core.state import RoundResult


_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AgentRotationProfile(BaseModel):
    agent_id: str
    verdict: str  # LEAD_NOW | LEAD_SOON | STANDBY | BENCH | SKIP | INSUFFICIENT_DATA
    lead_fitness: float  # 0..100
    reputation: float
    times_led: int
    lead_success_rate: float
    calibration_score: float
    votes_cast: int
    rejection_rate: float
    times_slashed: int
    chronic_blocker_count: int
    rounds_since_last_lead: int  # -1 if never led
    reasons: List[str] = Field(default_factory=list)


class LeaderSlot(BaseModel):
    slot_index: int  # 0-based
    agent_id: str
    expected_commit_prob: float  # 0..1
    confidence_band: str  # HIGH | MEDIUM | LOW
    reasons: List[str] = Field(default_factory=list)


class PlaybookItem(BaseModel):
    pattern: str
    priority: str  # P0..P3
    targets: List[str] = Field(default_factory=list)
    reason: str
    expected_impact: str


class LeaderRotationReport(BaseModel):
    generated_at: datetime
    rounds_analyzed: int
    horizon: int
    risk_appetite: str
    leader_capture_agent: Optional[str] = None
    agents: List[AgentRotationProfile] = Field(default_factory=list)
    rotation_queue: List[LeaderSlot] = Field(default_factory=list)
    playbook: List[PlaybookItem] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    overall_grade: str
    summary_headline: str

    # -- renderers ----------------------------------------------------------

    def to_text(self) -> str:
        out: List[str] = []
        out.append("=" * 70)
        out.append("Leader Rotation Advisor")
        out.append("=" * 70)
        out.append(f"generated_at:    {self.generated_at.isoformat()}")
        out.append(f"rounds_analyzed: {self.rounds_analyzed}")
        out.append(f"horizon:         {self.horizon}")
        out.append(f"risk_appetite:   {self.risk_appetite}")
        out.append(f"overall_grade:   {self.overall_grade}")
        out.append(f"headline:        {self.summary_headline}")
        out.append("")
        out.append("-- Rotation queue --")
        if not self.rotation_queue:
            out.append("(empty)")
        else:
            for s in self.rotation_queue:
                reasons = ",".join(s.reasons) if s.reasons else "-"
                out.append(
                    f"  slot {s.slot_index}: {s.agent_id:<14} "
                    f"prob={s.expected_commit_prob:.2f} "
                    f"band={s.confidence_band:<6} reasons={reasons}"
                )
        out.append("")
        out.append("-- Agents --")
        if not self.agents:
            out.append("(no agents observed)")
        else:
            for a in self.agents:
                reasons = ",".join(a.reasons) if a.reasons else "-"
                out.append(
                    f"  {a.agent_id:<14} {a.verdict:<18} "
                    f"fit={a.lead_fitness:>5.1f} rep={a.reputation:>5.3f} "
                    f"led={a.times_led} cal={a.calibration_score:.2f} "
                    f"reasons={reasons}"
                )
        out.append("")
        out.append("-- Playbook --")
        if not self.playbook:
            out.append("(none)")
        else:
            for item in self.playbook:
                tgt = ",".join(item.targets) if item.targets else "-"
                out.append(f"[{item.priority}] {item.pattern} :: targets={tgt}")
                out.append(f"    reason: {item.reason}")
                out.append(f"    impact: {item.expected_impact}")
        out.append("")
        out.append("-- Insights --")
        if not self.insights:
            out.append("(none)")
        else:
            for i in self.insights:
                out.append(f"  * {i}")
        return "\n".join(out)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# Leader Rotation Advisor")
        lines.append("")
        lines.append(f"- **generated_at:** `{self.generated_at.isoformat()}`")
        lines.append(f"- **rounds_analyzed:** {self.rounds_analyzed}")
        lines.append(f"- **horizon:** {self.horizon}")
        lines.append(f"- **risk_appetite:** {self.risk_appetite}")
        lines.append(f"- **overall_grade:** **{self.overall_grade}**")
        lines.append(f"- **headline:** {self.summary_headline}")
        lines.append("")
        lines.append("## Rotation queue")
        lines.append("")
        if not self.rotation_queue:
            lines.append("_empty_")
        else:
            lines.append("| slot | agent | fitness | prob | band | reasons |")
            lines.append("|---:|---|---:|---:|---|---|")
            for s in self.rotation_queue:
                reasons = ", ".join(s.reasons) if s.reasons else "-"
                # find matching fitness from agents
                fit = next(
                    (
                        a.lead_fitness
                        for a in self.agents
                        if a.agent_id == s.agent_id
                    ),
                    0.0,
                )
                lines.append(
                    f"| {s.slot_index} | `{s.agent_id}` | {fit:.1f} | "
                    f"{s.expected_commit_prob:.2f} | {s.confidence_band} | "
                    f"{reasons} |"
                )
        lines.append("")
        lines.append("## Agents")
        lines.append("")
        if not self.agents:
            lines.append("_no agents observed_")
        else:
            lines.append(
                "| agent | verdict | fitness | rep | led | lead_success | "
                "calibration | votes | slashed | reasons |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
            for a in self.agents:
                reasons = ", ".join(a.reasons) if a.reasons else "-"
                lines.append(
                    f"| `{a.agent_id}` | {a.verdict} | {a.lead_fitness:.1f} | "
                    f"{a.reputation:.3f} | {a.times_led} | "
                    f"{a.lead_success_rate*100:.1f}% | "
                    f"{a.calibration_score:.2f} | {a.votes_cast} | "
                    f"{a.times_slashed} | {reasons} |"
                )
        lines.append("")
        lines.append("## Playbook")
        lines.append("")
        if not self.playbook:
            lines.append("_none_")
        else:
            for item in self.playbook:
                tgt = (
                    ", ".join(f"`{t}`" for t in item.targets)
                    if item.targets
                    else "_(swarm-wide)_"
                )
                lines.append(f"### [{item.priority}] {item.pattern}")
                lines.append("")
                lines.append(f"- **targets:** {tgt}")
                lines.append(f"- **reason:** {item.reason}")
                lines.append(f"- **expected_impact:** {item.expected_impact}")
                lines.append("")
        lines.append("## Insights")
        lines.append("")
        if not self.insights:
            lines.append("_none_")
        else:
            for i in self.insights:
                lines.append(f"- {i}")
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


_RISK_KNOBS = {
    # (rep+cal multiplier, recency multiplier, slash mult, fresh penalty)
    "cautious": (1.15, 0.85, 1.0, 5.0),
    "balanced": (1.00, 1.00, 1.0, 0.0),
    "aggressive": (0.90, 1.25, 0.5, 0.0),
}


class LeaderRotationAdvisor:
    """Next-N forced-leader queue planner over an mBFT engine's history."""

    def __init__(
        self,
        horizon: int = 5,
        risk_appetite: str = "balanced",
        min_observations: int = 3,
        coalition_warning: bool = True,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be >= 1")
        if risk_appetite not in _RISK_KNOBS:
            raise ValueError(f"unknown risk_appetite: {risk_appetite!r}")
        self.horizon = horizon
        self.risk_appetite = risk_appetite
        self.min_observations = min_observations
        self.coalition_warning = coalition_warning
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    # -- public entry -------------------------------------------------------

    def recommend(
        self,
        history: Sequence[RoundResult],
        reputation: Mapping[str, float],
        agents: Sequence[str],
    ) -> LeaderRotationReport:
        history = list(history)
        rounds_analyzed = len(history)

        # Build the agent roster: union of supplied list, reputation keys,
        # and anyone observed in history. Read-only - we never mutate inputs.
        roster = set(agents) | set(reputation.keys())
        for r in history:
            roster.add(r.leader_id)
            for v in r.votes:
                roster.add(v.voter_id)
        roster_sorted = sorted(roster)

        profiles = self._build_profiles(history, reputation, roster_sorted)

        # Detect leader capture (consecutive + dominant) BEFORE scoring so
        # we can dampen the captured agent in the queue.
        capture_agent = self._detect_leader_capture(history)

        # Score every agent.
        agent_reports: List[AgentRotationProfile] = []
        for aid in roster_sorted:
            p = profiles[aid]
            fitness, reasons = self._score(p, rounds_analyzed)
            verdict, extra_reasons = self._classify(
                p, fitness, rounds_analyzed
            )
            for er in extra_reasons:
                if er not in reasons:
                    reasons.append(er)
            agent_reports.append(
                AgentRotationProfile(
                    agent_id=aid,
                    verdict=verdict,
                    lead_fitness=round(fitness, 2),
                    reputation=round(p["reputation"], 4),
                    times_led=p["times_led"],
                    lead_success_rate=round(p["lead_success_rate"], 4),
                    calibration_score=round(p["calibration_score"], 4),
                    votes_cast=p["votes_cast"],
                    rejection_rate=round(p["rejection_rate"], 4),
                    times_slashed=p["times_slashed"],
                    chronic_blocker_count=p["chronic_blocker_count"],
                    rounds_since_last_lead=p["rounds_since_last_lead"],
                    reasons=reasons,
                )
            )

        # Sort agents by fitness desc, then reputation desc, then id asc
        # (stable display ordering).
        agent_reports.sort(
            key=lambda a: (-a.lead_fitness, -a.reputation, a.agent_id)
        )

        # Build rotation queue.
        queue = self._build_queue(agent_reports, capture_agent)

        playbook = self._playbook(
            agent_reports, queue, capture_agent, rounds_analyzed
        )
        insights = self._insights(
            agent_reports, history, capture_agent, rounds_analyzed
        )
        grade = self._grade(agent_reports, queue, capture_agent)
        headline = self._headline(queue, capture_agent, grade)

        return LeaderRotationReport(
            generated_at=self._now_fn(),
            rounds_analyzed=rounds_analyzed,
            horizon=self.horizon,
            risk_appetite=self.risk_appetite,
            leader_capture_agent=capture_agent,
            agents=agent_reports,
            rotation_queue=queue,
            playbook=playbook,
            insights=insights,
            overall_grade=grade,
            summary_headline=headline,
        )

    # -- profiling ----------------------------------------------------------

    @staticmethod
    def _build_profiles(
        history: Sequence[RoundResult],
        reputation: Mapping[str, float],
        roster: Sequence[str],
    ) -> Dict[str, dict]:
        profiles: Dict[str, dict] = {
            aid: {
                "reputation": reputation.get(aid, 1.0),
                "times_led": 0,
                "lead_committed": 0,
                "votes_cast": 0,
                "rejections_cast": 0,
                "agreement_with_committed": 0,
                "disagreement_with_committed": 0,
                "times_slashed": 0,
                "last_lead_round": -1,
                "chronic_blocker_count": 0,
            }
            for aid in roster
        }

        last_round_idx = -1
        for r in history:
            if r.round_index > last_round_idx:
                last_round_idx = r.round_index

            p_leader = profiles.setdefault(
                r.leader_id,
                {
                    "reputation": reputation.get(r.leader_id, 1.0),
                    "times_led": 0,
                    "lead_committed": 0,
                    "votes_cast": 0,
                    "rejections_cast": 0,
                    "agreement_with_committed": 0,
                    "disagreement_with_committed": 0,
                    "times_slashed": 0,
                    "last_lead_round": -1,
                    "chronic_blocker_count": 0,
                },
            )
            p_leader["times_led"] += 1
            p_leader["last_lead_round"] = max(
                p_leader["last_lead_round"], r.round_index
            )
            if r.committed:
                p_leader["lead_committed"] += 1

            for slashed_id in r.slashed:
                if slashed_id in profiles:
                    profiles[slashed_id]["times_slashed"] += 1

            unrefuted_blocker = (
                not r.committed and r.aggregate_weight >= r.threshold
            )
            for v in r.votes:
                p_v = profiles.setdefault(
                    v.voter_id,
                    {
                        "reputation": reputation.get(v.voter_id, 1.0),
                        "times_led": 0,
                        "lead_committed": 0,
                        "votes_cast": 0,
                        "rejections_cast": 0,
                        "agreement_with_committed": 0,
                        "disagreement_with_committed": 0,
                        "times_slashed": 0,
                        "last_lead_round": -1,
                        "chronic_blocker_count": 0,
                    },
                )
                p_v["votes_cast"] += 1
                if v.is_rejection:
                    p_v["rejections_cast"] += 1
                if r.committed:
                    if v.weight > 0:
                        p_v["agreement_with_committed"] += 1
                    elif v.is_rejection:
                        p_v["disagreement_with_committed"] += 1
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
                p["agreement_with_committed"]
                + p["disagreement_with_committed"]
            )
            p["calibration_score"] = (
                p["agreement_with_committed"] / relevant
                if relevant > 0
                else 0.5
            )
            if p["last_lead_round"] < 0 or last_round_idx < 0:
                p["rounds_since_last_lead"] = -1
            else:
                p["rounds_since_last_lead"] = (
                    last_round_idx - p["last_lead_round"]
                )

        return profiles

    # -- scoring ------------------------------------------------------------

    def _score(
        self,
        profile: dict,
        rounds_analyzed: int,
    ) -> tuple[float, List[str]]:
        rep_cal_mult, recency_mult, slash_mult, fresh_penalty = _RISK_KNOBS[
            self.risk_appetite
        ]

        reasons: List[str] = []

        # 1. lead success component (50 default if unproven)
        if profile["times_led"] >= 2:
            lead_success_pts = profile["lead_success_rate"] * 100
            if profile["lead_success_rate"] >= 0.75:
                reasons.append("PROVEN_LEADER")
        else:
            lead_success_pts = 50.0
            if profile["times_led"] == 0:
                reasons.append("UNPROVEN")

        # 2. calibration component
        cal_pts = profile["calibration_score"] * 100
        if profile["calibration_score"] >= 0.85 and (
            profile["agreement_with_committed"]
            + profile["disagreement_with_committed"]
        ) >= 3:
            reasons.append("HIGH_CALIBRATION")
        elif profile["calibration_score"] < 0.3 and profile["votes_cast"] >= 5:
            reasons.append("LOW_CALIBRATION")

        # 3. reputation component
        rep_pts = max(0.0, min(profile["reputation"], 1.0)) * 100
        if profile["reputation"] >= 0.9:
            reasons.append("STRONG_REPUTATION")
        elif profile["reputation"] < 0.5:
            reasons.append("WEAK_REPUTATION")

        # 4. recency pressure (overdue for rotation)
        rsl = profile["rounds_since_last_lead"]
        if rsl < 0:
            # Never led - max recency pressure, but it doubles as UNPROVEN.
            recency = 1.0
        else:
            recency = min(1.0, rsl / 4.0)
        recency_pts = recency * 100
        if rsl == 0 and rounds_analyzed > 0:
            reasons.append("RECENT_LEADER")
        elif rsl >= 4:
            reasons.append("OVERDUE_FOR_ROTATION")

        # Weighted base (apply risk knob to rep+cal block).
        base = (
            0.30 * lead_success_pts
            + (0.25 * cal_pts + 0.20 * rep_pts) * rep_cal_mult / 1.0
            + 0.15 * recency_pts * recency_mult
            + 0.10 * (1.0 - min(1.0, profile["chronic_blocker_count"] / 3.0))
            * 100
        )

        # Risk re-normalisation: the rep+cal block is the only one that
        # changes weight. Subtract the extra mass injected so the base
        # doesn't blow past 100 under cautious/aggressive multipliers.
        # (0.25 + 0.20) * (rep_cal_mult - 1.0) * 100 worth of headroom max.
        # We don't divide-renormalize because we then clamp 0..100 below.

        slash_penalty = min(25.0, 5.0 * profile["times_slashed"]) * slash_mult
        if profile["times_slashed"] >= 2:
            reasons.append("SLASHED_REPEATEDLY")

        inactivity_penalty = 0.0
        if profile["votes_cast"] == 0 and profile["times_led"] == 0:
            inactivity_penalty = 30.0
            reasons.append("INACTIVE")

        chronic_penalty = 0.0
        if profile["chronic_blocker_count"] >= 2:
            chronic_penalty = 20.0
            reasons.append("CHRONIC_BLOCKER")

        # Cautious extra: penalize fresh blood
        fresh_pen = (
            fresh_penalty if profile["times_led"] == 0 and rounds_analyzed > 0 else 0.0
        )

        fitness = base - slash_penalty - inactivity_penalty - chronic_penalty - fresh_pen
        fitness = max(0.0, min(100.0, fitness))

        # Rising star tag
        if (
            profile["calibration_score"] >= 0.85
            and profile["times_led"] < 2
            and profile["votes_cast"] >= 3
        ):
            reasons.append("RISING_STAR")

        return fitness, reasons

    # -- classification -----------------------------------------------------

    def _classify(
        self,
        profile: dict,
        fitness: float,
        rounds_analyzed: int,
    ) -> tuple[str, List[str]]:
        observations = profile["times_led"] + profile["votes_cast"]
        extra: List[str] = []

        if rounds_analyzed < self.min_observations or observations == 0:
            return "INSUFFICIENT_DATA", extra
        if observations < self.min_observations and profile["votes_cast"] < self.min_observations:
            # Allow agents observed via voting even when leadership is sparse,
            # but still treat very-thin records as INSUFFICIENT_DATA.
            if observations < 2:
                return "INSUFFICIENT_DATA", extra

        if (
            profile["calibration_score"] < 0.3
            and profile["votes_cast"] >= 5
        ):
            return "SKIP", extra
        if fitness < 35 or profile["times_slashed"] >= 2:
            return "BENCH", extra
        if profile["chronic_blocker_count"] >= 2:
            # Chronic blocker still scored; verdict drops to BENCH regardless.
            return "BENCH", extra
        if fitness >= 70:
            return "LEAD_NOW", extra
        if fitness >= 50:
            return "LEAD_SOON", extra
        return "STANDBY", extra

    # -- leader capture -----------------------------------------------------

    @staticmethod
    def _detect_leader_capture(history: Sequence[RoundResult]) -> Optional[str]:
        if not history:
            return None
        leaders = [r.leader_id for r in history]
        # Rule 1: 3+ consecutive leads by the same agent in the tail
        tail = leaders[-5:] if len(leaders) >= 5 else leaders
        if len(tail) >= 3:
            run = 1
            for i in range(len(tail) - 1, 0, -1):
                if tail[i] == tail[i - 1]:
                    run += 1
                    if run >= 3:
                        return tail[i]
                else:
                    break
        # Rule 2: same agent leads >=60% of last 5 rounds
        if len(leaders) >= 5:
            last5 = leaders[-5:]
            counts = Counter(last5)
            top, ct = counts.most_common(1)[0]
            if ct / 5.0 >= 0.6:
                return top
        # Same agent leads everything in shorter histories (>=3 rounds)
        if len(leaders) >= 3:
            counts = Counter(leaders)
            top, ct = counts.most_common(1)[0]
            if ct / len(leaders) >= 0.75:
                return top
        return None

    # -- queue build --------------------------------------------------------

    def _build_queue(
        self,
        agents: Sequence[AgentRotationProfile],
        capture_agent: Optional[str],
    ) -> List[LeaderSlot]:
        # Working scores - mutable copy by id.
        scores = {a.agent_id: a.lead_fitness for a in agents}
        capture_lockout_until = (
            max(1, self.horizon // 2) if (capture_agent and self.coalition_warning) else 0
        )

        def eligibility_rank(a: AgentRotationProfile) -> int:
            # Lower = better. Viable agents share a bucket so the diversity
            # dampener can actually reorder picks within the queue;
            # INSUFFICIENT_DATA / BENCH / SKIP are explicit fallbacks.
            order = {
                "LEAD_NOW": 0,
                "LEAD_SOON": 0,
                "STANDBY": 0,
                "INSUFFICIENT_DATA": 1,
                "BENCH": 2,
                "SKIP": 3,
            }
            return order.get(a.verdict, 4)

        by_id = {a.agent_id: a for a in agents}
        queue: List[LeaderSlot] = []
        picked_history: List[str] = []  # ordered list of prior picks
        roster_size = len(agents)

        for slot in range(self.horizon):
            # Build candidate pool with current dampened scores.
            # Diversity model: anyone who appeared in the last
            # min(roster_size-1, 3) picks gets a recency penalty that
            # decays linearly. This guarantees distinct picks when the
            # roster has enough viable agents to support it.
            window = min(max(roster_size - 1, 0), 3)
            recent_window = picked_history[-window:] if window else []
            recency_penalty: Dict[str, float] = {}
            for offset, aid in enumerate(reversed(recent_window)):
                # offset=0 == most recently picked
                pen = 30.0 - 8.0 * offset
                if pen <= 0:
                    continue
                recency_penalty[aid] = max(recency_penalty.get(aid, 0.0), pen)

            candidates: List[tuple[AgentRotationProfile, float]] = []
            for a in agents:
                s = scores[a.agent_id] - recency_penalty.get(a.agent_id, 0.0)
                if (
                    capture_agent
                    and self.coalition_warning
                    and a.agent_id == capture_agent
                    and slot < capture_lockout_until
                ):
                    s -= 50.0
                candidates.append((a, s))

            # Sort by (verdict bucket, -score, -reputation, agent_id).
            candidates.sort(
                key=lambda t: (
                    eligibility_rank(t[0]),
                    -t[1],
                    -t[0].reputation,
                    t[0].agent_id,
                )
            )
            if not candidates:
                break
            chosen, chosen_score = candidates[0]

            # Expected commit prob: clamp 0..1 of dampened score / 100,
            # adjusted with min floor.
            prob = max(0.05, min(0.99, chosen_score / 100.0))
            if chosen.verdict in ("BENCH", "SKIP"):
                prob = min(prob, 0.30)
            elif chosen.verdict == "INSUFFICIENT_DATA":
                prob = min(prob, 0.50)

            band = (
                "HIGH"
                if chosen.lead_fitness >= 70 and chosen.times_led >= 2
                else "LOW"
                if chosen.lead_fitness < 50 or chosen.times_led == 0
                else "MEDIUM"
            )

            reasons: List[str] = []
            if picked_history and chosen.agent_id == picked_history[-1]:
                reasons.append("ROTATION_REUSE")
            if chosen.verdict in ("BENCH", "SKIP"):
                reasons.append("FALLBACK_PICK")
            if (
                capture_agent
                and chosen.agent_id == capture_agent
                and slot >= capture_lockout_until
            ):
                reasons.append("CAPTURE_LOCKOUT_LIFTED")
            # Propagate top reasons from agent profile
            for r in chosen.reasons[:3]:
                if r not in reasons:
                    reasons.append(r)

            queue.append(
                LeaderSlot(
                    slot_index=slot,
                    agent_id=chosen.agent_id,
                    expected_commit_prob=round(prob, 3),
                    confidence_band=band,
                    reasons=reasons,
                )
            )
            picked_history.append(chosen.agent_id)

        return queue

    # -- playbook -----------------------------------------------------------

    def _playbook(
        self,
        agents: Sequence[AgentRotationProfile],
        queue: Sequence[LeaderSlot],
        capture_agent: Optional[str],
        rounds_analyzed: int,
    ) -> List[PlaybookItem]:
        items: List[PlaybookItem] = []
        by_id = {a.agent_id: a for a in agents}

        eligible_count = sum(
            1 for a in agents if a.verdict in ("LEAD_NOW", "LEAD_SOON")
        )
        viable_count = sum(
            1
            for a in agents
            if a.verdict in ("LEAD_NOW", "LEAD_SOON", "STANDBY")
        )
        bench_or_skip = [
            a.agent_id for a in agents if a.verdict in ("BENCH", "SKIP")
        ]

        if capture_agent and self.coalition_warning:
            items.append(
                PlaybookItem(
                    pattern="BREAK_LEADER_CAPTURE",
                    priority="P0",
                    targets=[capture_agent],
                    reason=(
                        f"Agent {capture_agent} has dominated recent leader "
                        f"selection (consecutive or >=60% of last 5 rounds). "
                        f"Force-rotate to break the capture."
                    ),
                    expected_impact=(
                        "Restores leadership diversity and reduces single-"
                        "point-of-failure risk in finality."
                    ),
                )
            )

        # EMERGENCY when there is *no* LEAD_NOW/LEAD_SOON candidate at
        # all, or when the entire roster has been pushed below STANDBY.
        no_eligible_leaders = agents and eligible_count == 0
        all_grounded = agents and all(
            a.verdict in ("BENCH", "SKIP", "INSUFFICIENT_DATA") for a in agents
        )
        if no_eligible_leaders or all_grounded:
            items.append(
                PlaybookItem(
                    pattern="EMERGENCY_BACKUP_LEADER",
                    priority="P0",
                    targets=[a.agent_id for a in agents][:3],
                    reason=(
                        "All agents are BENCH/SKIP/INSUFFICIENT_DATA. "
                        "Provision a fresh, calibrated agent or relax slash "
                        "policy to recover leadership capacity."
                    ),
                    expected_impact=(
                        "Restores the swarm's ability to commit rounds at "
                        "all; without this, finality is at risk."
                    ),
                )
            )

        chronic_blockers = [
            a.agent_id for a in agents if a.chronic_blocker_count >= 2
        ]
        for cb in chronic_blockers:
            items.append(
                PlaybookItem(
                    pattern="ROTATE_OUT_BLOCKER",
                    priority="P1",
                    targets=[cb],
                    reason=(
                        f"Agent {cb} has vetoed {by_id[cb].chronic_blocker_count} "
                        f"otherwise-passing rounds. Demote out of leader pool."
                    ),
                    expected_impact=(
                        "Removes a known finality blocker from active "
                        "leadership consideration."
                    ),
                )
            )

        rising_stars = [
            a for a in agents if "RISING_STAR" in a.reasons
        ]
        top3_ids = {s.agent_id for s in queue[:3]}
        rising_not_in_top3 = [a.agent_id for a in rising_stars if a.agent_id not in top3_ids]
        for rs in rising_not_in_top3:
            items.append(
                PlaybookItem(
                    pattern="PROMOTE_RISING_STAR",
                    priority="P1",
                    targets=[rs],
                    reason=(
                        f"Agent {rs} has high calibration but minimal "
                        f"leadership history. Promote to next rotation to "
                        f"prove out."
                    ),
                    expected_impact=(
                        "Diversifies leadership bench and surfaces a likely "
                        "PROVEN_LEADER candidate."
                    ),
                )
            )

        # ELEVATE_CALIBRATED_FOLLOWER: best LEAD_NOW pick has never led but
        # has strong calibration.
        if queue:
            head = by_id.get(queue[0].agent_id)
            if (
                head is not None
                and head.times_led == 0
                and head.calibration_score >= 0.8
                and head.verdict == "LEAD_NOW"
            ):
                items.append(
                    PlaybookItem(
                        pattern="ELEVATE_CALIBRATED_FOLLOWER",
                        priority="P1",
                        targets=[head.agent_id],
                        reason=(
                            f"Top rotation pick `{head.agent_id}` has never "
                            f"led but is well-calibrated "
                            f"({head.calibration_score:.2f}). Promote with "
                            f"observation."
                        ),
                        expected_impact=(
                            "Validates a fresh leader cheaply and widens "
                            "the rotation bench."
                        ),
                    )
                )

        # REBALANCE_LEADERSHIP_LOAD if std-dev of times_led is high.
        if len(agents) >= 2:
            led_counts = [a.times_led for a in agents]
            mean = sum(led_counts) / len(led_counts)
            var = sum((c - mean) ** 2 for c in led_counts) / len(led_counts)
            sd = math.sqrt(var)
            if sd > 2.0:
                # Surface the top loaded leader as a target.
                heavies = sorted(
                    agents, key=lambda a: -a.times_led
                )[: max(1, len(agents) // 2)]
                items.append(
                    PlaybookItem(
                        pattern="REBALANCE_LEADERSHIP_LOAD",
                        priority="P2",
                        targets=[h.agent_id for h in heavies if h.times_led > mean],
                        reason=(
                            f"Leadership load is uneven (stddev={sd:.2f}). "
                            f"Redistribute toward under-utilized but viable "
                            f"candidates."
                        ),
                        expected_impact=(
                            "Smoother rotation reduces burnout-of-trust on "
                            "the dominant leader and surfaces hidden bench."
                        ),
                    )
                )

        # ADD_REDUNDANT_LEADER_CANDIDATES capacity risk
        if rounds_analyzed >= self.min_observations and eligible_count <= 2 and not capture_agent:
            items.append(
                PlaybookItem(
                    pattern="ADD_REDUNDANT_LEADER_CANDIDATES",
                    priority="P2",
                    targets=[],
                    reason=(
                        f"Only {eligible_count} agent(s) are LEAD_NOW/LEAD_SOON. "
                        f"Leadership bench is thin - one slash leaves the "
                        f"swarm without a strong leader."
                    ),
                    expected_impact=(
                        "Adds resilience to leadership capacity; reduces "
                        "single-point-of-failure on commit."
                    ),
                )
            )

        if not items and rounds_analyzed >= self.min_observations:
            items.append(
                PlaybookItem(
                    pattern="HEALTHY_ROTATION",
                    priority="P3",
                    targets=[],
                    reason=(
                        "Rotation queue is well-formed; multiple viable "
                        "leaders and no capture risk detected."
                    ),
                    expected_impact=(
                        "Maintain current rotation cadence; revisit on next "
                        "history window."
                    ),
                )
            )

        # Dedupe by (pattern, tuple(targets)), keep highest-priority instance.
        seen: Dict[tuple, PlaybookItem] = {}
        for it in items:
            key = (it.pattern, tuple(it.targets))
            prev = seen.get(key)
            if prev is None or _PRIORITY_RANK[it.priority] < _PRIORITY_RANK[prev.priority]:
                seen[key] = it
        out = sorted(
            seen.values(),
            key=lambda it: (_PRIORITY_RANK[it.priority], it.pattern, ",".join(it.targets)),
        )
        return out

    # -- insights -----------------------------------------------------------

    def _insights(
        self,
        agents: Sequence[AgentRotationProfile],
        history: Sequence[RoundResult],
        capture_agent: Optional[str],
        rounds_analyzed: int,
    ) -> List[str]:
        ins: List[str] = []
        if rounds_analyzed < self.min_observations:
            ins.append(
                f"INSUFFICIENT_HISTORY: only {rounds_analyzed} round(s) "
                f"analyzed; recommendations are tentative."
            )
        if capture_agent:
            ins.append(f"LEADER_CAPTURE_DETECTED: {capture_agent}")
        viable = sum(
            1
            for a in agents
            if a.verdict in ("LEAD_NOW", "LEAD_SOON", "STANDBY")
        )
        if viable <= 2:
            ins.append(
                f"LEADERSHIP_BENCH_THIN: only {viable} viable leader(s) "
                f"across roster."
            )
        if history:
            distinct = len({r.leader_id for r in history[-5:]})
            if distinct >= 3:
                ins.append(
                    f"DIVERSE_LEADERSHIP: {distinct} distinct leaders in "
                    f"the last {min(5, len(history))} round(s)."
                )
        if any("RISING_STAR" in a.reasons for a in agents):
            ins.append("RISING_STAR_AVAILABLE")
        return ins

    # -- grade --------------------------------------------------------------

    def _grade(
        self,
        agents: Sequence[AgentRotationProfile],
        queue: Sequence[LeaderSlot],
        capture_agent: Optional[str],
    ) -> str:
        if not queue:
            return "F"
        viable = sum(
            1 for a in agents if a.verdict in ("LEAD_NOW", "LEAD_SOON")
        )
        # F: no LEAD_NOW/LEAD_SOON anywhere on the roster, *or* all agents
        # are BENCH/SKIP/INSUFFICIENT_DATA.
        all_bad = all(
            a.verdict in ("BENCH", "SKIP", "INSUFFICIENT_DATA") for a in agents
        )
        if all_bad or viable == 0:
            return "F"
        if capture_agent and viable <= 1:
            return "F"

        top_picks = queue[: max(1, self.horizon)]
        top_ids = {s.agent_id for s in top_picks}
        top_fit = [a.lead_fitness for a in agents if a.agent_id in top_ids]
        mean_fit = sum(top_fit) / len(top_fit) if top_fit else 0.0

        if viable < 2:
            return "D"
        if mean_fit >= 80:
            return "A"
        if mean_fit >= 65:
            return "B"
        if mean_fit >= 50:
            return "C"
        if mean_fit >= 35:
            return "D"
        return "F"

    @staticmethod
    def _headline(
        queue: Sequence[LeaderSlot],
        capture_agent: Optional[str],
        grade: str,
    ) -> str:
        if not queue:
            return f"Rotation grade {grade}: no viable rotation could be built."
        head = queue[0]
        fit_pct = int(round(head.expected_commit_prob * 100))
        capture = (
            f"leader-capture risk on `{capture_agent}`"
            if capture_agent
            else "no leader-capture risk"
        )
        return (
            f"Rotation grade {grade}: {len(queue)}-slot queue led by "
            f"{head.agent_id} ({fit_pct}/100 expected commit); {capture}."
        )
