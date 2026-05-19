"""Vote dispersion advisor.

9th agentic sibling alongside swarm_health, disagreement_forensics,
round_replay_advisor, agent_lifecycle_advisor, leader_rotation_advisor,
voting_coalition_detector, proposal_risk_scorer, threshold_tuning_advisor.

Detects vote-weight distribution pathologies across rounds:
* GROUPTHINK  - everyone agrees, no dissent, hidden Byzantine risk
* ECHO_LEADER - non-leader weights cluster around leader's positive signal
* POLARIZED   - bimodal weights (strong yes + strong no)
* FRAGMENTED  - high stddev with mixed signs but no clean clusters
* HEDGED      - everyone within +/-0.2, aggregate near threshold
* HEALTHY_DEBATE - clear majority but with at least one dissent and moderate stddev

Distinct from voting_coalition_detector (who groups together) - this
measures *how* the swarm votes (variance, polarity, hedging).

Pure stdlib + pydantic. Never mutates inputs. Deterministic.
"""
from __future__ import annotations

import copy
import json
import math
import statistics
from datetime import datetime
from typing import Callable, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from src.core.state import RoundResult


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RoundDispersion(BaseModel):
    round_index: int
    leader_id: str
    verdict: str
    dispersion_score: float
    n_voters: int
    stddev: float
    mean_weight: float
    rejections: int
    reasons: List[str] = Field(default_factory=list)
    priority: str  # P0..P3


class AgentContribution(BaseModel):
    agent_id: str
    votes_cast: int
    hedge_rate: float
    echo_rate: float
    dissent_rate: float
    independence_score: float
    verdict: str  # INDEPENDENT|CONFORMIST|HEDGER|CONTRARIAN|BALANCED


class PlaybookAction(BaseModel):
    id: str
    priority: str
    label: str
    reason: str
    owner: str
    blast_radius: int
    reversibility: str
    related_agents: List[str] = Field(default_factory=list)
    suggested_value: Optional[float] = None


class DispersionPortfolio(BaseModel):
    rounds_observed: int
    groupthink_round_count: int
    echo_leader_round_count: int
    polarized_round_count: int
    fragmented_round_count: int
    hedged_round_count: int
    healthy_round_count: int
    portfolio_dispersion_score: float
    grade: str
    summary: str


class DispersionReport(BaseModel):
    generated_at: datetime
    risk_appetite: str
    portfolio: DispersionPortfolio
    rounds: List[RoundDispersion]
    agents: List[AgentContribution]
    playbook: List[PlaybookAction]
    insights: List[str]


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _classify_round(
    rr: RoundResult,
    appetite: str,
) -> RoundDispersion:
    votes = list(rr.votes)
    n = len(votes)
    weights = [v.weight for v in votes]
    rejections = sum(1 for v in votes if v.weight < 0)
    positives = [w for w in weights if w > 0]
    mean_w = float(statistics.fmean(weights)) if weights else 0.0
    mean_pos = float(statistics.fmean(positives)) if positives else 0.0
    sd = _stddev(weights)
    reasons: List[str] = []
    verdict = "HEALTHY_DEBATE"

    big_pos = sum(1 for w in weights if w >= 0.6)
    big_neg = sum(1 for w in weights if w <= -0.4)
    near_zero = sum(1 for w in weights if abs(w) < 0.2)
    near_leader = (
        sum(1 for w in weights if w > 0 and abs(w - mean_pos) <= 0.10)
        if positives
        else 0
    )

    aggregate_close = abs(rr.aggregate_weight - rr.threshold) <= 0.5

    # Order matters: most specific / dangerous first.
    if rejections == 0 and n >= 4 and sd <= 0.05 and all(w >= 0.5 for w in weights):
        verdict = "GROUPTHINK"
        reasons.append("ZERO_DISSENT_LOW_VARIANCE")
    elif big_pos >= 2 and big_neg >= 2:
        verdict = "POLARIZED"
        reasons.append(f"BIMODAL_{big_pos}_POS_{big_neg}_NEG")
    elif n >= 4 and positives and near_leader / max(len(positives), 1) >= 0.75 and rejections == 0:
        verdict = "ECHO_LEADER"
        reasons.append("VOTES_CLUSTER_AROUND_LEADER_SIGNAL")
    elif near_zero >= max(2, int(0.6 * n)) and aggregate_close:
        verdict = "HEDGED"
        reasons.append("MOST_WEIGHTS_NEAR_ZERO")
    elif sd >= 0.45 and not (big_pos >= 2 and big_neg >= 2):
        verdict = "FRAGMENTED"
        reasons.append("HIGH_STDDEV_NO_CLEAR_CLUSTER")
    else:
        if rejections >= 1 and rr.aggregate_weight >= rr.threshold and 0.15 <= sd <= 0.40:
            verdict = "HEALTHY_DEBATE"
            reasons.append("DISSENT_PRESENT_MODERATE_VARIANCE")
        else:
            verdict = "HEALTHY_DEBATE"
            reasons.append("DEFAULT_NO_PATHOLOGY")

    # Score: start from a healthy baseline, penalize pathologies.
    base = 60.0
    if verdict == "GROUPTHINK":
        base = 10.0
    elif verdict == "ECHO_LEADER":
        base = 30.0
    elif verdict == "POLARIZED":
        base = 40.0
    elif verdict == "FRAGMENTED":
        base = 50.0
    elif verdict == "HEDGED":
        base = 45.0
    elif verdict == "HEALTHY_DEBATE":
        base = 85.0

    appetite_mult = 1.0
    if appetite == "cautious" and verdict in {"GROUPTHINK", "ECHO_LEADER"}:
        base = max(0.0, base - 5.0)  # cautious treats these as worse
        appetite_mult = 0.9  # penalize harder downstream too
    elif appetite == "aggressive" and verdict in {"GROUPTHINK", "ECHO_LEADER"}:
        base = min(100.0, base + 5.0)
        appetite_mult = 1.05

    score = max(0.0, min(100.0, base * appetite_mult))

    # Priority bucket per round
    if verdict == "GROUPTHINK":
        priority = "P0"
    elif verdict in {"ECHO_LEADER", "POLARIZED"}:
        priority = "P1"
    elif verdict in {"FRAGMENTED", "HEDGED"}:
        priority = "P2"
    else:
        priority = "P3"

    return RoundDispersion(
        round_index=rr.round_index,
        leader_id=rr.leader_id,
        verdict=verdict,
        dispersion_score=round(score, 2),
        n_voters=n,
        stddev=round(sd, 4),
        mean_weight=round(mean_w, 4),
        rejections=rejections,
        reasons=reasons,
        priority=priority,
    )


def _classify_agents(
    history: Sequence[RoundResult],
    appetite: str,
) -> List[AgentContribution]:
    # Per-round mean positive weight (for echo detection).
    round_mean_pos = {}
    for rr in history:
        pos = [v.weight for v in rr.votes if v.weight > 0]
        round_mean_pos[rr.round_index] = (
            float(statistics.fmean(pos)) if pos else 0.0
        )

    by_agent: dict[str, dict] = {}
    for rr in history:
        for v in rr.votes:
            d = by_agent.setdefault(
                v.voter_id,
                {"votes": 0, "hedge": 0, "echo": 0, "dissent": 0},
            )
            d["votes"] += 1
            if abs(v.weight) < 0.2:
                d["hedge"] += 1
            mp = round_mean_pos.get(rr.round_index, 0.0)
            if v.weight > 0 and mp > 0 and abs(v.weight - mp) <= 0.10:
                d["echo"] += 1
            if v.weight < 0:
                d["dissent"] += 1

    out: List[AgentContribution] = []
    for aid, d in by_agent.items():
        n = max(d["votes"], 1)
        hedge_rate = d["hedge"] / n
        echo_rate = d["echo"] / n
        dissent_rate = d["dissent"] / n
        ind = 100.0 - (50.0 * hedge_rate + 50.0 * echo_rate)
        if appetite == "cautious":
            ind *= 0.95
        elif appetite == "aggressive":
            ind = min(100.0, ind * 1.05)
        ind = max(0.0, min(100.0, ind))

        if ind >= 70 and dissent_rate >= 0.10:
            verdict = "INDEPENDENT"
        elif echo_rate >= 0.7:
            verdict = "CONFORMIST"
        elif hedge_rate >= 0.6:
            verdict = "HEDGER"
        elif dissent_rate >= 0.5:
            verdict = "CONTRARIAN"
        else:
            verdict = "BALANCED"

        out.append(
            AgentContribution(
                agent_id=aid,
                votes_cast=d["votes"],
                hedge_rate=round(hedge_rate, 4),
                echo_rate=round(echo_rate, 4),
                dissent_rate=round(dissent_rate, 4),
                independence_score=round(ind, 2),
                verdict=verdict,
            )
        )

    out.sort(key=lambda a: (-a.independence_score, a.agent_id))
    return out


def _grade(
    rounds_observed: int,
    groupthink: int,
    mean_score: float,
) -> str:
    if rounds_observed == 0:
        return "A"
    gt_pct = groupthink / rounds_observed
    if gt_pct >= 0.30 or mean_score <= 25:
        return "F"
    if gt_pct >= 0.20 or mean_score <= 40:
        return "D"
    if gt_pct >= 0.10 or mean_score <= 55:
        return "C"
    if mean_score <= 75:
        return "B"
    return "A"


def _build_playbook(
    rounds: Sequence[RoundDispersion],
    agents: Sequence[AgentContribution],
    threshold: float,
    appetite: str,
    grade: str,
) -> List[PlaybookAction]:
    groupthink = sum(1 for r in rounds if r.verdict == "GROUPTHINK")
    echo_leader = sum(1 for r in rounds if r.verdict == "ECHO_LEADER")
    polarized = sum(1 for r in rounds if r.verdict == "POLARIZED")
    fragmented = sum(1 for r in rounds if r.verdict == "FRAGMENTED")
    hedged = sum(1 for r in rounds if r.verdict == "HEDGED")
    n_rounds = len(rounds)

    independents = [a for a in agents if a.verdict == "INDEPENDENT"]
    conforming = [a for a in agents if a.verdict in {"CONFORMIST", "HEDGER"}]
    monoculture = (
        len(agents) > 0 and len(conforming) / len(agents) >= 0.7
    )

    actions: List[PlaybookAction] = []

    if groupthink >= 2:
        actions.append(
            PlaybookAction(
                id="INJECT_DEVILS_ADVOCATE",
                priority="P0",
                label="Inject devil's advocate agent",
                reason=f"{groupthink} rounds showed zero-dissent groupthink",
                owner="governance",
                blast_radius=4,
                reversibility="medium",
            )
        )
    if n_rounds > 0 and groupthink / n_rounds >= 0.30:
        actions.append(
            PlaybookAction(
                id="INVESTIGATE_HIDDEN_BYZANTINE",
                priority="P0",
                label="Investigate hidden Byzantine risk",
                reason=(
                    f"{groupthink}/{n_rounds} rounds were groupthink "
                    "(>=30%); silent dissent risk"
                ),
                owner="security",
                blast_radius=3,
                reversibility="low",
            )
        )
    if echo_leader >= 2:
        actions.append(
            PlaybookAction(
                id="ROTATE_LEADERSHIP_FOR_DIVERSITY",
                priority="P1",
                label="Rotate leadership for vote diversity",
                reason=f"{echo_leader} rounds showed echo-leader clustering",
                owner="ops",
                blast_radius=3,
                reversibility="high",
            )
        )
    if polarized >= 2:
        actions.append(
            PlaybookAction(
                id="MEDIATE_POLARIZATION",
                priority="P1",
                label="Mediate polarized votes",
                reason=f"{polarized} rounds showed bimodal voting",
                owner="governance",
                blast_radius=3,
                reversibility="medium",
            )
        )
    if monoculture:
        actions.append(
            PlaybookAction(
                id="RECRUIT_INDEPENDENT_VOTERS",
                priority="P1",
                label="Recruit independent voters",
                reason=(
                    f"{len(conforming)}/{len(agents)} agents are "
                    "conformist or hedging (>=70%)"
                ),
                owner="governance",
                blast_radius=4,
                reversibility="medium",
            )
        )
    if independents:
        actions.append(
            PlaybookAction(
                id="REWARD_INDEPENDENT_VOTERS",
                priority="P2",
                label="Reward independent voters",
                reason=f"{len(independents)} agents show INDEPENDENT verdict",
                owner="governance",
                blast_radius=1,
                reversibility="high",
                related_agents=[a.agent_id for a in independents[:5]],
            )
        )
    if fragmented >= 3:
        actions.append(
            PlaybookAction(
                id="RAISE_PROOF_STANDARDS",
                priority="P2",
                label="Raise proof standards",
                reason=f"{fragmented} fragmented rounds suggest weak proofs",
                owner="governance",
                blast_radius=2,
                reversibility="medium",
            )
        )
    if hedged >= 3:
        actions.append(
            PlaybookAction(
                id="REVIEW_THRESHOLD_TUNING",
                priority="P2",
                label="Review threshold tuning (likely too high)",
                reason=f"{hedged} hedged rounds near threshold",
                owner="governance",
                blast_radius=2,
                reversibility="high",
                suggested_value=round(threshold * 0.9, 4),
            )
        )

    # Appetite knobs
    if appetite == "cautious" and grade in {"C", "D", "F"}:
        actions.append(
            PlaybookAction(
                id="SCHEDULE_DISPERSION_AUDIT",
                priority="P2",
                label="Schedule dispersion audit",
                reason=f"grade={grade} under cautious appetite warrants follow-up",
                owner="governance",
                blast_radius=1,
                reversibility="high",
            )
        )

    if not actions:
        actions.append(
            PlaybookAction(
                id="HEALTHY_DELIBERATION",
                priority="P3",
                label="Healthy deliberation - maintain observability",
                reason="No dispersion pathologies detected",
                owner="governance",
                blast_radius=1,
                reversibility="high",
            )
        )

    # Aggressive trimming
    if appetite == "aggressive":
        has_p0_or_p1 = any(a.priority in {"P0", "P1"} for a in actions)
        if has_p0_or_p1:
            # Drop P3 fallback
            actions = [a for a in actions if a.priority != "P3"]
            # Drop lone P2 (keep if multiple)
            p2 = [a for a in actions if a.priority == "P2"]
            if len(p2) == 1:
                actions = [a for a in actions if a.priority != "P2"]

    # Dedupe by id, deterministic sort
    seen = set()
    deduped: List[PlaybookAction] = []
    for a in actions:
        if a.id in seen:
            continue
        seen.add(a.id)
        deduped.append(a)
    deduped.sort(key=lambda a: (_PRIORITY_ORDER.get(a.priority, 9), a.id))
    return deduped


def _build_insights(
    rounds: Sequence[RoundDispersion],
    agents: Sequence[AgentContribution],
) -> List[str]:
    insights: List[str] = []
    if not rounds:
        insights.append("EMPTY_HISTORY")
        return insights
    if len(rounds) < 3:
        insights.append("INSUFFICIENT_DATA")

    counts = {
        v: sum(1 for r in rounds if r.verdict == v)
        for v in (
            "GROUPTHINK",
            "ECHO_LEADER",
            "POLARIZED",
            "FRAGMENTED",
            "HEDGED",
            "HEALTHY_DEBATE",
        )
    }
    if counts["GROUPTHINK"] >= 2:
        insights.append("GROUPTHINK_PATTERN")
    if counts["ECHO_LEADER"] >= 2:
        insights.append("ECHO_LEADER_PATTERN")
    if counts["POLARIZED"] >= 2:
        insights.append("POLARIZATION_PATTERN")
    if counts["FRAGMENTED"] >= 3:
        insights.append("FRAGMENTATION_PATTERN")
    if counts["HEDGED"] >= 3:
        insights.append("HEDGING_PATTERN")

    conforming = sum(1 for a in agents if a.verdict in {"CONFORMIST", "HEDGER"})
    if agents and conforming / len(agents) >= 0.7:
        insights.append("MONOCULTURE_VOTING")
    inds = sum(1 for a in agents if a.verdict == "INDEPENDENT")
    if inds >= 2:
        insights.append("INDEPENDENT_VOICES_PRESENT")

    if not insights:
        insights.append("HEALTHY_DELIBERATION")
    return insights


class VoteDispersionAdvisor:
    """Vote weight distribution / dispersion / groupthink advisor."""

    def __init__(
        self,
        engine_threshold: float = 3.0,
        risk_appetite: str = "balanced",
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if risk_appetite not in {"cautious", "balanced", "aggressive"}:
            raise ValueError("risk_appetite must be cautious|balanced|aggressive")
        self.engine_threshold = float(engine_threshold)
        self.risk_appetite = risk_appetite
        self.now_fn = now_fn or datetime.utcnow

    def analyze(
        self,
        engine_history: Iterable[RoundResult],
        reputation: Optional[Mapping[str, float]] = None,
    ) -> DispersionReport:
        history = [copy.deepcopy(rr) for rr in engine_history]
        _ = dict(reputation) if reputation else {}

        rounds = [_classify_round(rr, self.risk_appetite) for rr in history]
        rounds.sort(key=lambda r: r.round_index)
        agents = _classify_agents(history, self.risk_appetite)

        counts = {
            "GROUPTHINK": 0,
            "ECHO_LEADER": 0,
            "POLARIZED": 0,
            "FRAGMENTED": 0,
            "HEDGED": 0,
            "HEALTHY_DEBATE": 0,
        }
        for r in rounds:
            counts[r.verdict] = counts.get(r.verdict, 0) + 1

        mean_score = (
            float(statistics.fmean(r.dispersion_score for r in rounds))
            if rounds
            else 0.0
        )
        grade = _grade(len(rounds), counts["GROUPTHINK"], mean_score)
        summary = (
            f"VERDICT: grade={grade} rounds={len(rounds)} "
            f"groupthink={counts['GROUPTHINK']} echo={counts['ECHO_LEADER']} "
            f"polarized={counts['POLARIZED']} mean_score={mean_score:.1f}"
        )

        portfolio = DispersionPortfolio(
            rounds_observed=len(rounds),
            groupthink_round_count=counts["GROUPTHINK"],
            echo_leader_round_count=counts["ECHO_LEADER"],
            polarized_round_count=counts["POLARIZED"],
            fragmented_round_count=counts["FRAGMENTED"],
            hedged_round_count=counts["HEDGED"],
            healthy_round_count=counts["HEALTHY_DEBATE"],
            portfolio_dispersion_score=round(mean_score, 2),
            grade=grade,
            summary=summary,
        )

        playbook = _build_playbook(
            rounds, agents, self.engine_threshold, self.risk_appetite, grade
        )
        insights = _build_insights(rounds, agents)

        return DispersionReport(
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


def to_text(report: DispersionReport) -> str:
    lines: List[str] = []
    p = report.portfolio
    lines.append(p.summary)
    lines.append(f"risk_appetite={report.risk_appetite}  generated_at={report.generated_at.isoformat()}")
    lines.append("")
    lines.append("Per-round verdicts:")
    for r in report.rounds:
        lines.append(
            f"  round={r.round_index} leader={r.leader_id} verdict={r.verdict} "
            f"score={r.dispersion_score} sd={r.stddev} n={r.n_voters} "
            f"rej={r.rejections} priority={r.priority}"
        )
    lines.append("")
    lines.append("Per-agent contributions:")
    for a in report.agents:
        lines.append(
            f"  {a.agent_id}: verdict={a.verdict} indep={a.independence_score} "
            f"hedge={a.hedge_rate} echo={a.echo_rate} dissent={a.dissent_rate} "
            f"n={a.votes_cast}"
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


def to_markdown(report: DispersionReport) -> str:
    p = report.portfolio
    lines: List[str] = []
    lines.append(f"# Vote Dispersion Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| grade | {p.grade} |")
    lines.append(f"| rounds_observed | {p.rounds_observed} |")
    lines.append(f"| portfolio_dispersion_score | {p.portfolio_dispersion_score} |")
    lines.append(f"| groupthink_rounds | {p.groupthink_round_count} |")
    lines.append(f"| echo_leader_rounds | {p.echo_leader_round_count} |")
    lines.append(f"| polarized_rounds | {p.polarized_round_count} |")
    lines.append(f"| fragmented_rounds | {p.fragmented_round_count} |")
    lines.append(f"| hedged_rounds | {p.hedged_round_count} |")
    lines.append(f"| healthy_rounds | {p.healthy_round_count} |")
    lines.append(f"| risk_appetite | {report.risk_appetite} |")
    lines.append("")
    lines.append("## Per-round verdicts")
    lines.append("")
    lines.append("| round | leader | verdict | priority | score | stddev | n | rejections |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in report.rounds:
        lines.append(
            f"| {r.round_index} | {r.leader_id} | {r.verdict} | {r.priority} | "
            f"{r.dispersion_score} | {r.stddev} | {r.n_voters} | {r.rejections} |"
        )
    if not report.rounds:
        lines.append("| - | - | - | - | - | - | - | - |")
    lines.append("")
    lines.append("## Per-agent contributions")
    lines.append("")
    lines.append("| agent | verdict | independence | hedge_rate | echo_rate | dissent_rate | votes |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in report.agents:
        lines.append(
            f"| {a.agent_id} | {a.verdict} | {a.independence_score} | "
            f"{a.hedge_rate} | {a.echo_rate} | {a.dissent_rate} | {a.votes_cast} |"
        )
    if not report.agents:
        lines.append("| - | - | - | - | - | - | - |")
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


def to_json(report: DispersionReport) -> str:
    return json.dumps(
        report.model_dump(),
        sort_keys=True,
        indent=2,
        default=str,
    )
