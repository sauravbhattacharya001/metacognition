"""Swarm health monitor.

An *agentic* observability layer for the mBFT engine. Given a sequence of
``RoundResult`` objects (typically ``engine.history``) plus the current
reputation map, the monitor proactively surfaces:

* per-agent calibration scores (does their stated confidence track how
  often the swarm actually committed when they led / voted with the
  majority?),
* reputation drift and slash velocity,
* persistent dissenters / suspected Byzantine agents,
* round-level health signals (how often the swarm fails to commit, how
  close aggregate weights run to the threshold), and
* concrete, conservative *recommendations* the operator can apply
  (threshold up/down, slash factor up/down, agents to investigate).

The monitor never mutates the engine. It only reads history and emits
``SwarmHealthReport`` objects which can be rendered as text / markdown /
JSON / CSV for dashboards, daily digests, or CI gates.

Design notes:
* zero external dependencies beyond what mBFT already uses (``pydantic``),
* deterministic given the same inputs, so it is easy to test,
* recommendations are always rate-limited and bounded — the monitor
  *suggests*, the human (or a higher-level autopilot) *decides*.
"""
from __future__ import annotations

import json
from typing import Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from src.core.state import RoundResult


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AgentHealth(BaseModel):
    """Per-agent health snapshot."""

    agent_id: str
    reputation: float
    leader_rounds: int = 0
    leader_commits: int = 0
    times_slashed: int = 0
    votes_cast: int = 0
    rejections_cast: int = 0
    agreed_with_commit: int = 0
    disagreed_with_commit: int = 0
    avg_vote_weight: float = 0.0
    calibration_score: float = 0.0  # 1.0 = perfectly calibrated
    status: str = "ok"  # one of: ok, watch, suspect, slashed_out

    @property
    def leader_success_rate(self) -> float:
        if self.leader_rounds == 0:
            return 0.0
        return self.leader_commits / self.leader_rounds


class Recommendation(BaseModel):
    """A single actionable suggestion."""

    kind: str  # threshold|slash_factor|investigate|swarm_size|none
    severity: str  # info|warn|critical
    message: str
    suggested_value: Optional[float] = None
    target_agent: Optional[str] = None


class SwarmHealthReport(BaseModel):
    """Aggregate report produced by :class:`SwarmHealthMonitor`."""

    rounds_observed: int
    rounds_committed: int
    commit_rate: float
    avg_aggregate_weight: float
    avg_margin_to_threshold: float  # positive = comfortably above
    threshold: float
    leader_diversity: float  # 0..1 — fraction of unique leaders / total rounds
    agents: List[AgentHealth] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)

    # -- exporters ----------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# Swarm Health Report")
        lines.append("")
        lines.append(f"- Rounds observed: **{self.rounds_observed}**")
        lines.append(
            f"- Commit rate: **{self.commit_rate:.0%}** "
            f"({self.rounds_committed}/{self.rounds_observed})"
        )
        lines.append(
            f"- Threshold ?: **{self.threshold:.3f}** "
            f"(avg aggregate {self.avg_aggregate_weight:.3f}, "
            f"avg margin {self.avg_margin_to_threshold:+.3f})"
        )
        lines.append(f"- Leader diversity: **{self.leader_diversity:.2f}**")
        lines.append("")
        lines.append("## Agents")
        lines.append("")
        lines.append(
            "| agent | rep | status | lead | commit% | "
            "rejects | calibration |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        for a in self.agents:
            lines.append(
                f"| {a.agent_id} | {a.reputation:.3f} | {a.status} | "
                f"{a.leader_rounds} | {a.leader_success_rate:.0%} | "
                f"{a.rejections_cast} | {a.calibration_score:.2f} |"
            )
        lines.append("")
        lines.append("## Recommendations")
        lines.append("")
        if not self.recommendations:
            lines.append("_None — swarm looks healthy._")
        else:
            for r in self.recommendations:
                tag = f"[{r.severity.upper()}] {r.kind}"
                target = f" (agent={r.target_agent})" if r.target_agent else ""
                suggested = (
                    f" → suggested {r.suggested_value:.3f}"
                    if r.suggested_value is not None
                    else ""
                )
                lines.append(f"- **{tag}**{target}: {r.message}{suggested}")
        lines.append("")
        return "\n".join(lines)

    def to_text(self) -> str:
        out: List[str] = []
        out.append("=" * 60)
        out.append("SWARM HEALTH REPORT")
        out.append("=" * 60)
        out.append(
            f"rounds={self.rounds_observed}  "
            f"commits={self.rounds_committed}  "
            f"rate={self.commit_rate:.0%}"
        )
        out.append(
            f"?={self.threshold:.3f}  "
            f"avg_aggregate={self.avg_aggregate_weight:.3f}  "
            f"avg_margin={self.avg_margin_to_threshold:+.3f}"
        )
        out.append(f"leader_diversity={self.leader_diversity:.2f}")
        out.append("")
        out.append("agents:")
        for a in self.agents:
            out.append(
                f"  {a.agent_id:>6}  rep={a.reputation:.3f}  "
                f"status={a.status:<8}  "
                f"lead={a.leader_rounds}({a.leader_success_rate:.0%})  "
                f"rej={a.rejections_cast}  "
                f"cal={a.calibration_score:.2f}"
            )
        out.append("")
        out.append("recommendations:")
        if not self.recommendations:
            out.append("  (none — swarm healthy)")
        else:
            for r in self.recommendations:
                out.append(
                    f"  [{r.severity}] {r.kind}: {r.message}"
                    + (
                        f"  -> {r.suggested_value:.3f}"
                        if r.suggested_value is not None
                        else ""
                    )
                )
        out.append("=" * 60)
        return "\n".join(out)

    def to_csv(self) -> str:
        header = (
            "agent_id,reputation,status,leader_rounds,leader_commits,"
            "leader_success_rate,times_slashed,votes_cast,rejections_cast,"
            "agreed_with_commit,disagreed_with_commit,avg_vote_weight,"
            "calibration_score"
        )
        rows = [header]
        for a in self.agents:
            rows.append(
                ",".join(
                    str(x)
                    for x in [
                        a.agent_id,
                        f"{a.reputation:.6f}",
                        a.status,
                        a.leader_rounds,
                        a.leader_commits,
                        f"{a.leader_success_rate:.6f}",
                        a.times_slashed,
                        a.votes_cast,
                        a.rejections_cast,
                        a.agreed_with_commit,
                        a.disagreed_with_commit,
                        f"{a.avg_vote_weight:.6f}",
                        f"{a.calibration_score:.6f}",
                    ]
                )
            )
        return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class SwarmHealthMonitor:
    """Analyse mBFT round history and produce a health report.

    Typical usage::

        engine = MBFTEngine(agents=..., threshold=1.5)
        await engine.run(task)
        report = SwarmHealthMonitor().analyze(
            history=engine.history,
            reputation=engine.reputation,
            threshold=engine.threshold,
            agent_ids=[a.id for a in engine.agents],
        )
        print(report.to_markdown())

    The monitor is intentionally side-effect-free: it never mutates the
    engine, agents, or reputation map.
    """

    def __init__(
        self,
        *,
        suspect_rejection_rate: float = 0.6,
        slashed_reputation_floor: float = 0.25,
        low_margin_ratio: float = 0.05,
        high_margin_ratio: float = 0.5,
        low_commit_rate: float = 0.5,
        high_commit_rate: float = 0.95,
        slash_factor_floor: float = 0.05,
        slash_factor_ceiling: float = 0.95,
    ) -> None:
        self.suspect_rejection_rate = suspect_rejection_rate
        self.slashed_reputation_floor = slashed_reputation_floor
        self.low_margin_ratio = low_margin_ratio
        self.high_margin_ratio = high_margin_ratio
        self.low_commit_rate = low_commit_rate
        self.high_commit_rate = high_commit_rate
        self.slash_factor_floor = slash_factor_floor
        self.slash_factor_ceiling = slash_factor_ceiling

    # -- public API ---------------------------------------------------------

    def analyze(
        self,
        *,
        history: Sequence[RoundResult],
        reputation: Mapping[str, float],
        threshold: float,
        agent_ids: Optional[Iterable[str]] = None,
        slash_factor: Optional[float] = None,
    ) -> SwarmHealthReport:
        """Build a :class:`SwarmHealthReport` from a slice of history."""
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        ids = list(agent_ids) if agent_ids is not None else list(reputation)
        # Make sure every agent we have reputation for shows up.
        for aid in reputation:
            if aid not in ids:
                ids.append(aid)

        agents = {
            aid: AgentHealth(agent_id=aid, reputation=float(reputation.get(aid, 1.0)))
            for aid in ids
        }

        rounds_observed = len(history)
        rounds_committed = sum(1 for r in history if r.committed)
        commit_rate = (
            rounds_committed / rounds_observed if rounds_observed else 0.0
        )

        agg_total = 0.0
        margin_total = 0.0
        leaders_seen: List[str] = []

        for rnd in history:
            agg_total += rnd.aggregate_weight
            margin_total += rnd.aggregate_weight - rnd.threshold
            leaders_seen.append(rnd.leader_id)

            leader = agents.get(rnd.leader_id)
            if leader is None:
                # Unknown leader (shouldn't happen, but be defensive).
                leader = AgentHealth(agent_id=rnd.leader_id, reputation=1.0)
                agents[rnd.leader_id] = leader
            leader.leader_rounds += 1
            if rnd.committed:
                leader.leader_commits += 1
            if rnd.leader_id in rnd.slashed:
                leader.times_slashed += 1

            for v in rnd.votes:
                voter = agents.get(v.voter_id)
                if voter is None:
                    voter = AgentHealth(agent_id=v.voter_id, reputation=1.0)
                    agents[v.voter_id] = voter
                voter.votes_cast += 1
                voter.avg_vote_weight += v.weight  # running sum, normalised below
                if v.is_rejection:
                    voter.rejections_cast += 1
                if rnd.committed:
                    if v.is_rejection:
                        voter.disagreed_with_commit += 1
                    else:
                        voter.agreed_with_commit += 1

        avg_aggregate = agg_total / rounds_observed if rounds_observed else 0.0
        avg_margin = margin_total / rounds_observed if rounds_observed else 0.0
        leader_diversity = (
            len(set(leaders_seen)) / rounds_observed if rounds_observed else 0.0
        )

        # Finalise per-agent derived metrics.
        for a in agents.values():
            if a.votes_cast > 0:
                a.avg_vote_weight = a.avg_vote_weight / a.votes_cast
            a.calibration_score = self._calibration(a)
            a.status = self._classify(a)

        ordered = sorted(agents.values(), key=lambda a: a.agent_id)

        report = SwarmHealthReport(
            rounds_observed=rounds_observed,
            rounds_committed=rounds_committed,
            commit_rate=commit_rate,
            avg_aggregate_weight=avg_aggregate,
            avg_margin_to_threshold=avg_margin,
            threshold=threshold,
            leader_diversity=leader_diversity,
            agents=ordered,
        )
        report.recommendations = self._recommend(report, slash_factor)
        return report

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _calibration(a: AgentHealth) -> float:
        """Heuristic in [0, 1].

        Rewards voters whose accept/reject calls line up with what the
        swarm ultimately committed; punishes persistent disagreement
        with committed rounds.
        """
        relevant = a.agreed_with_commit + a.disagreed_with_commit
        if relevant == 0:
            # No commits to compare against — neutral but not zero, so
            # untested agents don't get flagged immediately.
            return 0.5
        return a.agreed_with_commit / relevant

    def _classify(self, a: AgentHealth) -> str:
        if a.reputation < self.slashed_reputation_floor:
            return "slashed_out"
        rejection_rate = (
            a.rejections_cast / a.votes_cast if a.votes_cast else 0.0
        )
        if (
            rejection_rate >= self.suspect_rejection_rate
            and a.calibration_score < 0.4
        ):
            return "suspect"
        if a.times_slashed >= 2 or a.calibration_score < 0.4:
            return "watch"
        return "ok"

    def _recommend(
        self,
        report: SwarmHealthReport,
        slash_factor: Optional[float],
    ) -> List[Recommendation]:
        recs: List[Recommendation] = []

        if report.rounds_observed == 0:
            recs.append(
                Recommendation(
                    kind="none",
                    severity="info",
                    message="No rounds observed yet — run the engine before tuning.",
                )
            )
            return recs

        # Threshold tuning based on how often we commit and the margin.
        margin_ratio = (
            report.avg_margin_to_threshold / report.threshold
            if report.threshold
            else 0.0
        )
        if (
            report.commit_rate >= self.high_commit_rate
            and margin_ratio >= self.high_margin_ratio
        ):
            suggested = report.threshold * (1.0 + min(margin_ratio, 0.5) / 2.0)
            recs.append(
                Recommendation(
                    kind="threshold",
                    severity="info",
                    message=(
                        "Commits are passing with a large margin — "
                        "raising the threshold would tighten consensus."
                    ),
                    suggested_value=round(suggested, 4),
                )
            )
        elif report.commit_rate <= self.low_commit_rate:
            shrink = max(0.5, 1.0 - max(0.0, -margin_ratio) / 2.0)
            suggested = report.threshold * shrink
            recs.append(
                Recommendation(
                    kind="threshold",
                    severity="warn",
                    message=(
                        "Swarm rarely commits — consider lowering the "
                        "threshold or adding more aligned agents."
                    ),
                    suggested_value=round(suggested, 4),
                )
            )
        elif abs(margin_ratio) < self.low_margin_ratio:
            recs.append(
                Recommendation(
                    kind="threshold",
                    severity="warn",
                    message=(
                        "Aggregate weights hover at the threshold — "
                        "results may flip on small perturbations."
                    ),
                )
            )

        # Slash factor tuning (only when caller passed the current value).
        if slash_factor is not None:
            total_slashed = sum(a.times_slashed for a in report.agents)
            if (
                total_slashed >= max(2, report.rounds_observed // 2)
                and slash_factor > self.slash_factor_floor
            ):
                recs.append(
                    Recommendation(
                        kind="slash_factor",
                        severity="warn",
                        message=(
                            "Leaders are being slashed frequently — a "
                            "softer slash_factor will let recovered "
                            "agents lead again."
                        ),
                        suggested_value=round(
                            max(self.slash_factor_floor, slash_factor * 0.5),
                            4,
                        ),
                    )
                )
            elif (
                total_slashed == 0
                and report.commit_rate < self.high_commit_rate
                and slash_factor < self.slash_factor_ceiling
            ):
                recs.append(
                    Recommendation(
                        kind="slash_factor",
                        severity="info",
                        message=(
                            "No slashes recorded — accountability is weak. "
                            "A stricter slash_factor will speed up "
                            "view-changes after bad proposals."
                        ),
                        suggested_value=round(
                            min(self.slash_factor_ceiling, slash_factor * 1.5),
                            4,
                        ),
                    )
                )

        # Per-agent flags.
        for a in report.agents:
            if a.status == "suspect":
                recs.append(
                    Recommendation(
                        kind="investigate",
                        severity="critical",
                        message=(
                            f"Agent {a.agent_id} disagrees with the swarm "
                            f"on most commits "
                            f"(calibration={a.calibration_score:.2f}, "
                            f"rejections={a.rejections_cast}/{a.votes_cast})."
                        ),
                        target_agent=a.agent_id,
                    )
                )
            elif a.status == "slashed_out":
                recs.append(
                    Recommendation(
                        kind="investigate",
                        severity="warn",
                        message=(
                            f"Agent {a.agent_id} reputation has collapsed "
                            f"({a.reputation:.3f}). Consider quarantining."
                        ),
                        target_agent=a.agent_id,
                    )
                )

        # Swarm size sanity check.
        if len(report.agents) < 4 and report.commit_rate < self.high_commit_rate:
            recs.append(
                Recommendation(
                    kind="swarm_size",
                    severity="info",
                    message=(
                        "Small swarms can't out-vote a single Byzantine "
                        "agent. Consider adding more agents."
                    ),
                    suggested_value=4,
                )
            )

        return recs


__all__ = [
    "AgentHealth",
    "Recommendation",
    "SwarmHealthReport",
    "SwarmHealthMonitor",
]
