"""Voter turnout advisor.

11th agentic sibling alongside swarm_health, disagreement_forensics,
round_replay_advisor, agent_lifecycle_advisor, leader_rotation_advisor,
voting_coalition_detector, proposal_risk_scorer, threshold_tuning_advisor,
vote_dispersion_advisor, counter_proof_quality_advisor.

Where ``swarm_health`` answers "is the swarm OK in aggregate?", and
``agent_lifecycle_advisor`` answers "who do we keep / evict?", and
``vote_dispersion_advisor`` answers "how varied are the votes that were
cast?", this advisor answers a different question:

    *Who is showing up to vote, and when does low turnout put commits
    at risk?*

It is a participation/attendance auditor for the mBFT swarm. Given
``engine.history``, it infers the active roster (union of all agents
ever observed as voter or leader), then for each round computes
``turnout = n_voters / roster_size``, and for each agent computes
``absentee_rate`` over the window in which that agent was known to be
alive (eligible). It identifies:

* **chronic_absentees** - agents with absentee_rate >= 50% who were
  alive throughout
* **decaying_voters** - voted early, stopped showing up in the tail
* **emerged_voters** - new joiners appearing late in history (positive)
* **rounds_with_quorum_at_risk** - turnout fell below
  ``min_acceptable_turnout`` (default 0.6)
* **rounds_that_only_committed_because_absent_dissent** - the
  aggregate cleared threshold but, given each chronic absentee's
  historical mean vote weight, simulated turnout-restored aggregate
  would have either failed or had unrefuted rejection

It then emits a P0-first deduped playbook (RECRUIT_QUORUM /
REMOVE_CHRONIC_ABSENTEE / REINVITE_DECAYING_VOTER / LOWER_QUORUM_FLOOR
/ INVESTIGATE_TURNOUT_COLLAPSE / etc.), 0-100 portfolio
``turnout_score``, A-F grade, and structured insights.

Design notes:
* zero new dependencies (``pydantic`` only, matching the rest of mBFT);
* deterministic, stateless - never mutates the engine, history, or
  reputation map (history is deep-copied before analysis);
* the advisor *recommends*; the operator decides.
"""
from __future__ import annotations

import copy
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from pydantic import BaseModel, Field

from src.core.state import RoundResult


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AgentTurnout(BaseModel):
    """Per-agent participation snapshot."""

    agent_id: str
    rounds_eligible: int
    rounds_voted: int
    rounds_led: int
    attendance_rate: float  # 0..1
    absentee_rate: float  # 1 - attendance_rate (over eligible window)
    avg_vote_weight_when_present: float
    last_seen_round: Optional[int]
    first_seen_round: Optional[int]
    rounds_inactive_tail: int  # consecutive missed rounds at end of history
    status: str  # ACTIVE | CHRONIC_ABSENTEE | DECAYING | EMERGED | OCCASIONAL


class RoundTurnout(BaseModel):
    """Per-round turnout snapshot."""

    round_index: int
    leader_id: str
    n_voters: int
    roster_eligible: int
    turnout: float
    committed: bool
    aggregate_weight: float
    threshold: float
    absent_agents: List[str] = Field(default_factory=list)
    quorum_at_risk: bool  # turnout < min_acceptable_turnout
    only_committed_because_absent_dissent: bool
    reasons: List[str] = Field(default_factory=list)
    priority: str  # P0..P3


class PlaybookAction(BaseModel):
    id: str
    priority: str
    label: str
    reason: str
    owner: str
    blast_radius: int  # 1..5
    reversibility: str  # high|medium|low
    related_agents: List[str] = Field(default_factory=list)
    related_rounds: List[int] = Field(default_factory=list)
    suggested_value: Optional[float] = None


class TurnoutPortfolio(BaseModel):
    rounds_observed: int
    rounds_at_risk: int
    rounds_only_committed_because_absent_dissent: int
    roster_size: int
    chronic_absentee_count: int
    decaying_voter_count: int
    emerged_voter_count: int
    avg_turnout: float
    min_turnout: float
    max_turnout: float
    turnout_score: float  # 0..100, higher = healthier
    grade: str
    summary: str


class TurnoutReport(BaseModel):
    generated_at: datetime
    risk_appetite: str
    portfolio: TurnoutPortfolio
    rounds: List[RoundTurnout]
    agents: List[AgentTurnout]
    playbook: List[PlaybookAction]
    insights: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_APPETITE_THRESHOLD_SHIFT = {
    "cautious": +0.05,  # demand higher turnout to pass
    "balanced": 0.0,
    "aggressive": -0.05,
}


def _normalize_appetite(risk_appetite: str) -> str:
    if risk_appetite not in {"cautious", "balanced", "aggressive"}:
        raise ValueError("risk_appetite must be cautious|balanced|aggressive")
    return risk_appetite


def _collect_roster(history: Sequence[RoundResult]) -> Set[str]:
    roster: Set[str] = set()
    for rr in history:
        if rr.leader_id:
            roster.add(rr.leader_id)
        for v in rr.votes:
            roster.add(v.voter_id)
    return roster


def _per_agent_windows(
    history: Sequence[RoundResult], roster: Set[str]
) -> Dict[str, Dict]:
    """For each agent, determine first/last seen round and per-round presence."""
    info: Dict[str, Dict] = {
        a: {
            "first": None,
            "last": None,
            "voted_rounds": set(),
            "led_rounds": set(),
            "vote_weights": [],
        }
        for a in roster
    }
    for rr in history:
        idx = rr.round_index
        present_this_round: Set[str] = set()
        if rr.leader_id:
            info[rr.leader_id]["led_rounds"].add(idx)
            present_this_round.add(rr.leader_id)
        for v in rr.votes:
            info[v.voter_id]["voted_rounds"].add(idx)
            info[v.voter_id]["vote_weights"].append(float(v.weight))
            present_this_round.add(v.voter_id)
        for a in present_this_round:
            d = info[a]
            if d["first"] is None or idx < d["first"]:
                d["first"] = idx
            if d["last"] is None or idx > d["last"]:
                d["last"] = idx
    return info


def _classify_agent(
    agent_id: str,
    info: Dict,
    history_rounds: List[int],
    risk_appetite: str,
) -> AgentTurnout:
    first = info["first"]
    last = info["last"]
    voted = info["voted_rounds"]
    led = info["led_rounds"]
    weights = info["vote_weights"]
    avg_w = round(statistics.fmean(weights), 3) if weights else 0.0

    if first is None:
        # never seen at all (shouldn't normally happen given roster derivation)
        return AgentTurnout(
            agent_id=agent_id,
            rounds_eligible=0,
            rounds_voted=0,
            rounds_led=0,
            attendance_rate=0.0,
            absentee_rate=1.0,
            avg_vote_weight_when_present=0.0,
            last_seen_round=None,
            first_seen_round=None,
            rounds_inactive_tail=0,
            status="EMERGED",
        )

    eligible_rounds = [r for r in history_rounds if first <= r <= max(history_rounds)]
    # Treat agent as eligible from first_seen onwards (joined the swarm).
    rounds_eligible = len(eligible_rounds)
    present_count = sum(1 for r in eligible_rounds if r in voted or r in led)
    attendance = present_count / rounds_eligible if rounds_eligible else 0.0
    absentee = 1.0 - attendance

    # tail-inactive consecutive count from end of history
    tail_inactive = 0
    for r in reversed(history_rounds):
        if r < first:
            break
        if r in voted or r in led:
            break
        tail_inactive += 1

    # Status classification.
    n_history = len(history_rounds)
    last_idx = max(history_rounds) if history_rounds else 0
    earliest = min(history_rounds) if history_rounds else 0
    rounds_since_join = (last_idx - first + 1) if first is not None else 0

    status = "ACTIVE"
    if rounds_eligible >= 3 and absentee >= 0.5:
        status = "CHRONIC_ABSENTEE"
    elif rounds_eligible >= 4 and tail_inactive >= max(2, rounds_eligible // 3):
        status = "DECAYING"
    elif first > earliest and rounds_since_join <= max(2, n_history // 4):
        status = "EMERGED"
    elif rounds_eligible >= 2 and 0.2 <= absentee < 0.5:
        status = "OCCASIONAL"

    if risk_appetite == "cautious" and status == "OCCASIONAL" and absentee >= 0.3:
        status = "CHRONIC_ABSENTEE"
    if risk_appetite == "aggressive" and status == "OCCASIONAL":
        status = "ACTIVE"

    return AgentTurnout(
        agent_id=agent_id,
        rounds_eligible=rounds_eligible,
        rounds_voted=len(voted),
        rounds_led=len(led),
        attendance_rate=round(attendance, 3),
        absentee_rate=round(absentee, 3),
        avg_vote_weight_when_present=avg_w,
        last_seen_round=last,
        first_seen_round=first,
        rounds_inactive_tail=tail_inactive,
        status=status,
    )


def _classify_round(
    rr: RoundResult,
    roster_eligible_for_round: Set[str],
    agents_by_id: Dict[str, AgentTurnout],
    min_acceptable_turnout: float,
) -> RoundTurnout:
    voted_ids = {v.voter_id for v in rr.votes}
    if rr.leader_id:
        voted_ids.add(rr.leader_id)  # leader counts as present
    present = len(voted_ids & roster_eligible_for_round)
    eligible = max(1, len(roster_eligible_for_round))
    turnout = present / eligible
    absent = sorted(roster_eligible_for_round - voted_ids)

    quorum_at_risk = turnout < min_acceptable_turnout
    reasons: List[str] = []
    if quorum_at_risk:
        reasons.append("LOW_TURNOUT")
    if absent:
        # would absent dissenters have flipped a commit?
        sim_aggregate = rr.aggregate_weight
        had_phantom_rejection = False
        for a in absent:
            ah = agents_by_id.get(a)
            if not ah:
                continue
            w = ah.avg_vote_weight_when_present
            sim_aggregate += w
            if w < 0:
                had_phantom_rejection = True
        only_because_absent = (
            rr.committed
            and (
                sim_aggregate < rr.threshold
                or had_phantom_rejection
            )
        )
        if only_because_absent:
            reasons.append("PHANTOM_DISSENT_ABSENT")
    else:
        only_because_absent = False

    priority = "P3"
    if rr.committed and only_because_absent:
        priority = "P0"
    elif quorum_at_risk and not rr.committed:
        priority = "P1"
    elif quorum_at_risk:
        priority = "P2"
    elif turnout < 0.85:
        priority = "P3"

    return RoundTurnout(
        round_index=rr.round_index,
        leader_id=rr.leader_id,
        n_voters=len(rr.votes),
        roster_eligible=eligible,
        turnout=round(turnout, 3),
        committed=rr.committed,
        aggregate_weight=round(float(rr.aggregate_weight), 3),
        threshold=round(float(rr.threshold), 3),
        absent_agents=absent,
        quorum_at_risk=quorum_at_risk,
        only_committed_because_absent_dissent=only_because_absent,
        reasons=reasons,
        priority=priority,
    )


def _grade(
    rounds: List[RoundTurnout], turnout_score: float, chronic: int, n_roster: int
) -> str:
    if not rounds:
        return "A"
    p0_count = sum(1 for r in rounds if r.priority == "P0")
    risk_share = sum(1 for r in rounds if r.quorum_at_risk) / len(rounds)
    if p0_count >= 2 or turnout_score <= 30:
        return "F"
    if p0_count >= 1 or risk_share >= 0.4 or turnout_score <= 45:
        return "D"
    if risk_share >= 0.2 or turnout_score <= 60:
        return "C"
    if n_roster and chronic / n_roster >= 0.25:
        return "C"
    if turnout_score <= 80:
        return "B"
    return "A"


def _build_playbook(
    rounds: List[RoundTurnout],
    agents: List[AgentTurnout],
    risk_appetite: str,
    grade: str,
) -> List[PlaybookAction]:
    actions: List[PlaybookAction] = []

    chronic = [a for a in agents if a.status == "CHRONIC_ABSENTEE"]
    decaying = [a for a in agents if a.status == "DECAYING"]
    emerged = [a for a in agents if a.status == "EMERGED"]

    risk_rounds = [r for r in rounds if r.quorum_at_risk]
    phantom_rounds = [r for r in rounds if r.only_committed_because_absent_dissent]
    failed_low_turnout = [r for r in rounds if r.quorum_at_risk and not r.committed]

    # P0
    if phantom_rounds:
        actions.append(
            PlaybookAction(
                id="ROLLBACK_PHANTOM_COMMITS",
                priority="P0",
                label="Audit commits that only cleared because dissenters were absent",
                reason=(
                    f"{len(phantom_rounds)} committed round(s) would have failed or "
                    "had unrefuted rejection if chronic-absent agents had voted "
                    "their historical average weight"
                ),
                owner="governance",
                blast_radius=5,
                reversibility="medium",
                related_rounds=[r.round_index for r in phantom_rounds],
            )
        )
    if len(chronic) >= 2:
        actions.append(
            PlaybookAction(
                id="REMOVE_CHRONIC_ABSENTEE_CLUSTER",
                priority="P0",
                label="Evict or replace chronic-absentee voters",
                reason=(
                    f"{len(chronic)} voters have absentee_rate>=50%; they inflate "
                    "the denominator without contributing signal"
                ),
                owner="ops",
                blast_radius=4,
                reversibility="low",
                related_agents=[a.agent_id for a in chronic],
            )
        )
    elif len(chronic) == 1:
        actions.append(
            PlaybookAction(
                id="REMOVE_CHRONIC_ABSENTEE",
                priority="P1",
                label="Evict or replace chronic-absentee voter",
                reason=f"voter {chronic[0].agent_id} missed >=50% of eligible rounds",
                owner="ops",
                blast_radius=2,
                reversibility="low",
                related_agents=[chronic[0].agent_id],
            )
        )

    # P1
    if failed_low_turnout:
        actions.append(
            PlaybookAction(
                id="RECRUIT_QUORUM_BACKUPS",
                priority="P1",
                label="Recruit standby voters to guarantee minimum quorum",
                reason=(
                    f"{len(failed_low_turnout)} round(s) failed with turnout below "
                    "the acceptable floor - add 1-2 backup voters"
                ),
                owner="governance",
                blast_radius=3,
                reversibility="high",
            )
        )
    if decaying:
        actions.append(
            PlaybookAction(
                id="REINVITE_DECAYING_VOTERS",
                priority="P1",
                label="Re-invite or page voters who recently stopped showing up",
                reason=(
                    f"{len(decaying)} voter(s) participated early but missed the "
                    "tail of history; investigate liveness before evicting"
                ),
                owner="ops",
                blast_radius=2,
                reversibility="high",
                related_agents=[a.agent_id for a in decaying],
            )
        )

    # P2
    if risk_rounds and not failed_low_turnout:
        actions.append(
            PlaybookAction(
                id="INVESTIGATE_TURNOUT_COLLAPSE",
                priority="P2",
                label="Investigate why some rounds saw low turnout",
                reason=(
                    f"{len(risk_rounds)} round(s) ran with turnout below the "
                    "acceptable floor even though they ultimately committed"
                ),
                owner="ops",
                blast_radius=2,
                reversibility="high",
                related_rounds=[r.round_index for r in risk_rounds],
            )
        )
    if emerged:
        actions.append(
            PlaybookAction(
                id="ONBOARD_EMERGED_VOTERS",
                priority="P2",
                label="Onboard newly-joined voters and confirm calibration",
                reason=(
                    f"{len(emerged)} voter(s) joined late in the history window; "
                    "calibrate their proposals against committed solutions"
                ),
                owner="ops",
                blast_radius=1,
                reversibility="high",
                related_agents=[a.agent_id for a in emerged],
            )
        )

    if risk_appetite == "cautious" and grade in {"C", "D", "F"}:
        actions.append(
            PlaybookAction(
                id="SCHEDULE_QUORUM_AUDIT",
                priority="P2",
                label="Schedule a recurring quorum audit",
                reason="cautious risk appetite + degraded portfolio grade",
                owner="governance",
                blast_radius=1,
                reversibility="high",
            )
        )

    if not actions:
        actions.append(
            PlaybookAction(
                id="HEALTHY_PARTICIPATION",
                priority="P3",
                label="Quorum and participation are healthy - keep monitoring",
                reason="no chronic absentees, no quorum-at-risk rounds, no phantom commits",
                owner="ops",
                blast_radius=1,
                reversibility="high",
            )
        )

    # Aggressive trims P3 fallback + lone P2 when other actions exist.
    if risk_appetite == "aggressive" and len(actions) > 1:
        actions = [a for a in actions if a.priority != "P3"]
        has_p01 = any(a.priority in {"P0", "P1"} for a in actions)
        if has_p01:
            actions = [a for a in actions if a.priority != "P2" or a.id != "ONBOARD_EMERGED_VOTERS"]

    # P0-first deterministic sort.
    prio_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    actions.sort(key=lambda a: (prio_rank[a.priority], a.id))
    return actions


def _build_insights(
    rounds: List[RoundTurnout], agents: List[AgentTurnout], turnout_score: float
) -> List[str]:
    insights: List[str] = []
    if not rounds:
        insights.append("EMPTY_HISTORY")
        return insights

    chronic = sum(1 for a in agents if a.status == "CHRONIC_ABSENTEE")
    decaying = sum(1 for a in agents if a.status == "DECAYING")
    emerged = sum(1 for a in agents if a.status == "EMERGED")
    risk_rounds = sum(1 for r in rounds if r.quorum_at_risk)
    phantom = sum(1 for r in rounds if r.only_committed_because_absent_dissent)

    if phantom:
        insights.append("PHANTOM_DISSENT_COMMITS")
    if chronic >= 2:
        insights.append("CHRONIC_ABSENTEE_CLUSTER")
    elif chronic == 1:
        insights.append("CHRONIC_ABSENTEE_PRESENT")
    if decaying >= 2:
        insights.append("PARTICIPATION_DECAY_PATTERN")
    if emerged:
        insights.append("ROSTER_GROWTH")
    if risk_rounds and not phantom and chronic == 0:
        insights.append("OCCASIONAL_TURNOUT_DIPS")
    if risk_rounds / len(rounds) >= 0.5:
        insights.append("WIDESPREAD_TURNOUT_PROBLEMS")
    if turnout_score >= 90 and not insights:
        insights.append("HEALTHY_PARTICIPATION")
    if not insights:
        insights.append("MIXED_SIGNALS")
    return insights


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


class VoterTurnoutAdvisor:
    """Voter participation / absenteeism / quorum risk advisor."""

    def __init__(
        self,
        min_acceptable_turnout: float = 0.6,
        risk_appetite: str = "balanced",
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.risk_appetite = _normalize_appetite(risk_appetite)
        shift = _APPETITE_THRESHOLD_SHIFT[self.risk_appetite]
        self.min_acceptable_turnout = max(
            0.0, min(1.0, float(min_acceptable_turnout) + shift)
        )
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def analyze(
        self,
        engine_history: Iterable[RoundResult],
        reputation: Optional[Mapping[str, float]] = None,
    ) -> TurnoutReport:
        history = [copy.deepcopy(rr) for rr in engine_history]
        history.sort(key=lambda rr: rr.round_index)
        _ = dict(reputation) if reputation else {}

        roster = _collect_roster(history)
        history_rounds = [rr.round_index for rr in history]
        info = _per_agent_windows(history, roster)

        agents = [
            _classify_agent(a, info[a], history_rounds, self.risk_appetite)
            for a in sorted(roster)
        ]
        agents_by_id = {a.agent_id: a for a in agents}

        rounds: List[RoundTurnout] = []
        for rr in history:
            eligible_now = {
                a
                for a in roster
                if info[a]["first"] is not None and info[a]["first"] <= rr.round_index
            }
            rounds.append(
                _classify_round(
                    rr, eligible_now, agents_by_id, self.min_acceptable_turnout
                )
            )

        if rounds:
            avg_turnout = float(statistics.fmean(r.turnout for r in rounds))
            min_turnout = float(min(r.turnout for r in rounds))
            max_turnout = float(max(r.turnout for r in rounds))
        else:
            avg_turnout = min_turnout = max_turnout = 0.0

        # Turnout score 0..100
        # base = avg_turnout * 100, penalties for risk rounds + phantom commits
        risk_count = sum(1 for r in rounds if r.quorum_at_risk)
        phantom_count = sum(1 for r in rounds if r.only_committed_because_absent_dissent)
        chronic_count = sum(1 for a in agents if a.status == "CHRONIC_ABSENTEE")
        decaying_count = sum(1 for a in agents if a.status == "DECAYING")
        emerged_count = sum(1 for a in agents if a.status == "EMERGED")

        score = avg_turnout * 100.0
        score -= 5.0 * risk_count
        score -= 12.0 * phantom_count
        score -= 6.0 * chronic_count
        score -= 3.0 * decaying_count
        # appetite shift on score
        if self.risk_appetite == "cautious":
            score -= 5.0
        elif self.risk_appetite == "aggressive":
            score += 5.0
        score = max(0.0, min(100.0, score))

        grade = _grade(rounds, score, chronic_count, len(agents))
        summary = (
            f"VERDICT: grade={grade} rounds={len(rounds)} "
            f"roster={len(agents)} avg_turnout={avg_turnout:.2f} "
            f"chronic={chronic_count} at_risk={risk_count} "
            f"phantom={phantom_count} score={score:.1f}"
        )

        portfolio = TurnoutPortfolio(
            rounds_observed=len(rounds),
            rounds_at_risk=risk_count,
            rounds_only_committed_because_absent_dissent=phantom_count,
            roster_size=len(agents),
            chronic_absentee_count=chronic_count,
            decaying_voter_count=decaying_count,
            emerged_voter_count=emerged_count,
            avg_turnout=round(avg_turnout, 3),
            min_turnout=round(min_turnout, 3),
            max_turnout=round(max_turnout, 3),
            turnout_score=round(score, 2),
            grade=grade,
            summary=summary,
        )

        playbook = _build_playbook(rounds, agents, self.risk_appetite, grade)
        insights = _build_insights(rounds, agents, score)

        return TurnoutReport(
            generated_at=self.now_fn(),
            risk_appetite=self.risk_appetite,
            portfolio=portfolio,
            rounds=rounds,
            agents=agents,
            playbook=playbook,
            insights=insights,
        )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def to_text(report: TurnoutReport) -> str:
    p = report.portfolio
    lines: List[str] = [p.summary]
    lines.append(
        f"risk_appetite={report.risk_appetite}  "
        f"generated_at={report.generated_at.isoformat()}"
    )
    lines.append("")
    lines.append("Per-round turnout:")
    for r in report.rounds:
        lines.append(
            f"  round={r.round_index} leader={r.leader_id} "
            f"turnout={r.turnout} ({r.n_voters}/{r.roster_eligible}) "
            f"committed={r.committed} priority={r.priority} "
            f"absent={','.join(r.absent_agents) or '-'} "
            f"reasons={','.join(r.reasons) or '-'}"
        )
    lines.append("")
    lines.append("Per-agent attendance:")
    for a in report.agents:
        lines.append(
            f"  {a.agent_id}: status={a.status} attendance={a.attendance_rate} "
            f"voted={a.rounds_voted}/{a.rounds_eligible} led={a.rounds_led} "
            f"avg_w={a.avg_vote_weight_when_present} tail_inactive={a.rounds_inactive_tail}"
        )
    lines.append("")
    lines.append("Playbook:")
    for act in report.playbook:
        lines.append(f"  [{act.priority}] {act.id} owner={act.owner} - {act.label}")
        lines.append(f"      reason: {act.reason}")
    lines.append("")
    lines.append("Insights:")
    for ins in report.insights:
        lines.append(f"  - {ins}")
    return "\n".join(lines)


def to_markdown(report: TurnoutReport) -> str:
    p = report.portfolio
    lines: List[str] = ["# Voter Turnout Report", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| grade | {p.grade} |")
    lines.append(f"| rounds_observed | {p.rounds_observed} |")
    lines.append(f"| roster_size | {p.roster_size} |")
    lines.append(f"| avg_turnout | {p.avg_turnout} |")
    lines.append(f"| min_turnout | {p.min_turnout} |")
    lines.append(f"| max_turnout | {p.max_turnout} |")
    lines.append(f"| rounds_at_risk | {p.rounds_at_risk} |")
    lines.append(
        f"| rounds_only_committed_because_absent_dissent | "
        f"{p.rounds_only_committed_because_absent_dissent} |"
    )
    lines.append(f"| chronic_absentee_count | {p.chronic_absentee_count} |")
    lines.append(f"| decaying_voter_count | {p.decaying_voter_count} |")
    lines.append(f"| emerged_voter_count | {p.emerged_voter_count} |")
    lines.append(f"| turnout_score | {p.turnout_score} |")
    lines.append(f"| risk_appetite | {report.risk_appetite} |")
    lines.append("")
    lines.append("## Per-round turnout")
    lines.append("")
    lines.append(
        "| round | leader | turnout | n/eligible | committed | priority | "
        "phantom | reasons | absent |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in report.rounds:
        lines.append(
            f"| {r.round_index} | {r.leader_id} | {r.turnout} | "
            f"{r.n_voters}/{r.roster_eligible} | {r.committed} | {r.priority} | "
            f"{r.only_committed_because_absent_dissent} | "
            f"{','.join(r.reasons) or '-'} | {','.join(r.absent_agents) or '-'} |"
        )
    if not report.rounds:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    lines.append("")
    lines.append("## Per-agent attendance")
    lines.append("")
    lines.append(
        "| agent | status | attendance | voted/eligible | led | "
        "avg_w_present | tail_inactive | first | last |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for a in report.agents:
        lines.append(
            f"| {a.agent_id} | {a.status} | {a.attendance_rate} | "
            f"{a.rounds_voted}/{a.rounds_eligible} | {a.rounds_led} | "
            f"{a.avg_vote_weight_when_present} | {a.rounds_inactive_tail} | "
            f"{a.first_seen_round} | {a.last_seen_round} |"
        )
    if not report.agents:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    lines.append("")
    lines.append("## Playbook")
    lines.append("")
    lines.append("| priority | id | owner | blast | reversibility | label | reason |")
    lines.append("|---|---|---|---|---|---|---|")
    for act in report.playbook:
        lines.append(
            f"| {act.priority} | {act.id} | {act.owner} | {act.blast_radius} | "
            f"{act.reversibility} | {act.label} | {act.reason} |"
        )
    lines.append("")
    lines.append("## Insights")
    lines.append("")
    for ins in report.insights:
        lines.append(f"- {ins}")
    return "\n".join(lines)


def to_json(report: TurnoutReport) -> str:
    return json.dumps(
        report.model_dump(),
        sort_keys=True,
        indent=2,
        default=str,
    )
