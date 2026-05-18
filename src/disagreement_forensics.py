"""Disagreement forensics.

An *agentic* post-mortem layer for the mBFT engine. Where
:class:`~src.swarm_health.SwarmHealthMonitor` answers "is my swarm OK?"
in aggregate, ``DisagreementForensics`` zooms in on a single question:

    *Why did this round fail to commit, and what should I do about it?*

For every non-committed :class:`~src.core.state.RoundResult`, the analyzer:

* classifies the **blocker** as one of
  ``UNREFUTED_REJECTION`` (a non-slashed dissenter vetoed) or
  ``BELOW_THRESHOLD`` (aggregate weight fell short),
* attributes responsibility to the leader, the strongest dissenter, or
  the silent-majority underweight cohort,
* extracts the **counter-proof chain** (whoever rejected with a written
  rebuttal),
* identifies the **alternative leader** the engine would promote on the
  next view-change, and
* emits a per-round, severity-tagged remediation suggestion with a
  human-readable reason.

Cross-round patterns are then surfaced as a small playbook:
``CHRONIC_BLOCKER`` (one voter vetoes round after round),
``CALIBRATION_COLLAPSE`` (leaders consistently overconfident relative to
aggregate-realized weight), and ``THRESHOLD_TOO_HIGH`` (most failures
miss the bar by a small margin while no rejections fire).

Design notes:
* zero new dependencies (``pydantic`` only, matching the rest of mBFT);
* deterministic and stateless - feed it ``engine.history`` and you get the
  same report every time;
* never mutates the engine;
* the analyzer *recommends*; the operator (or a higher-level autopilot)
  *decides*.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field

from src.core.state import RoundResult, Vote


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CounterProofEntry(BaseModel):
    voter_id: str
    weight: float
    counter_proof: str


class RoundForensics(BaseModel):
    round_index: int
    leader_id: str
    aggregate_weight: float
    threshold: float
    margin: float  # aggregate - threshold (negative = below)
    blocker: str  # COMMITTED / UNREFUTED_REJECTION / BELOW_THRESHOLD
    severity: str  # info / low / medium / high / critical
    primary_blame: Optional[str]  # agent_id most responsible (or None if COMMITTED)
    blame_reason: str
    counter_proofs: List[CounterProofEntry] = Field(default_factory=list)
    rejector_ids: List[str] = Field(default_factory=list)
    next_view_leader: Optional[str]  # who the engine would promote next
    recommendation: str
    recommendation_priority: str  # P0 / P1 / P2 / none


class PatternFinding(BaseModel):
    code: str  # CHRONIC_BLOCKER / CALIBRATION_COLLAPSE / THRESHOLD_TOO_HIGH / ...
    priority: str  # P0 / P1 / P2
    headline: str
    detail: str
    suspects: List[str] = Field(default_factory=list)


class ForensicsReport(BaseModel):
    total_rounds: int
    committed_rounds: int
    failed_rounds: int
    rounds: List[RoundForensics] = Field(default_factory=list)
    patterns: List[PatternFinding] = Field(default_factory=list)
    headline: str

    # -- rendering ----------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# Disagreement Forensics")
        lines.append("")
        lines.append(f"**{self.headline}**")
        lines.append("")
        lines.append(
            f"- Rounds analyzed: **{self.total_rounds}** "
            f"(committed: {self.committed_rounds}, failed: {self.failed_rounds})"
        )
        lines.append("")

        if self.patterns:
            lines.append("## Cross-round patterns")
            lines.append("")
            for p in self.patterns:
                susp = f" (suspects: {', '.join(p.suspects)})" if p.suspects else ""
                lines.append(f"- **[{p.priority}] {p.code}** — {p.headline}{susp}")
                lines.append(f"  - {p.detail}")
            lines.append("")

        lines.append("## Per-round verdicts")
        lines.append("")
        for r in self.rounds:
            tag = (
                "[COMMITTED]" if r.blocker == "COMMITTED" else f"[{r.blocker}]"
            )
            lines.append(
                f"### Round {r.round_index} — leader `{r.leader_id}` — {tag}"
            )
            lines.append("")
            lines.append(
                f"- weight: **{r.aggregate_weight:.3f}** vs threshold "
                f"**{r.threshold:.3f}** (margin {r.margin:+.3f})"
            )
            lines.append(f"- severity: `{r.severity}`")
            if r.primary_blame:
                lines.append(f"- primary attribution: `{r.primary_blame}` — {r.blame_reason}")
            else:
                lines.append(f"- attribution: — {r.blame_reason}")
            if r.next_view_leader and r.blocker != "COMMITTED":
                lines.append(f"- next-view leader would be: `{r.next_view_leader}`")
            if r.counter_proofs:
                lines.append("- counter-proofs:")
                for cp in r.counter_proofs:
                    lines.append(
                        f"    - `{cp.voter_id}` (w={cp.weight:+.2f}): {cp.counter_proof}"
                    )
            if r.recommendation_priority != "none":
                lines.append(
                    f"- recommendation **[{r.recommendation_priority}]**: {r.recommendation}"
                )
            else:
                lines.append(f"- recommendation: {r.recommendation}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append(f"Disagreement Forensics — {self.headline}")
        lines.append(
            f"  rounds: {self.total_rounds} "
            f"(committed: {self.committed_rounds}, failed: {self.failed_rounds})"
        )
        if self.patterns:
            lines.append("  patterns:")
            for p in self.patterns:
                lines.append(f"    [{p.priority}] {p.code}: {p.headline}")
                if p.suspects:
                    lines.append(f"        suspects: {', '.join(p.suspects)}")
        lines.append("  rounds:")
        for r in self.rounds:
            verdict = "COMMIT" if r.blocker == "COMMITTED" else r.blocker
            lines.append(
                f"    r{r.round_index} leader={r.leader_id} {verdict} "
                f"w={r.aggregate_weight:.2f}/{r.threshold:.2f} "
                f"margin={r.margin:+.2f} sev={r.severity}"
            )
            if r.primary_blame:
                lines.append(
                    f"        blame={r.primary_blame} — {r.blame_reason}"
                )
            if r.recommendation_priority != "none":
                lines.append(
                    f"        [{r.recommendation_priority}] {r.recommendation}"
                )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class DisagreementForensics:
    """Per-round root-cause analyzer for non-committed mBFT rounds.

    Parameters
    ----------
    chronic_blocker_min_rounds:
        Minimum number of failed rounds an agent must veto before
        ``CHRONIC_BLOCKER`` fires.
    calibration_collapse_overconfidence:
        Fraction (0..1) of failed rounds in which the leader's *own*
        contribution (estimated by aggregate weight under the threshold)
        is needed for ``CALIBRATION_COLLAPSE`` to fire.
    threshold_too_high_close_margin:
        A round whose ``|margin| <= close_margin`` is considered a
        "near miss". If most failures are near misses *and* none of them
        carry an unrefuted rejection, ``THRESHOLD_TOO_HIGH`` fires.
    """

    def __init__(
        self,
        chronic_blocker_min_rounds: int = 2,
        calibration_collapse_overconfidence: float = 0.5,
        threshold_too_high_close_margin: float = 0.25,
    ) -> None:
        if chronic_blocker_min_rounds < 1:
            raise ValueError("chronic_blocker_min_rounds must be >= 1")
        if not 0.0 < calibration_collapse_overconfidence <= 1.0:
            raise ValueError(
                "calibration_collapse_overconfidence must be in (0, 1]"
            )
        if threshold_too_high_close_margin < 0:
            raise ValueError("threshold_too_high_close_margin must be >= 0")
        self.chronic_blocker_min_rounds = chronic_blocker_min_rounds
        self.calibration_collapse_overconfidence = (
            calibration_collapse_overconfidence
        )
        self.threshold_too_high_close_margin = threshold_too_high_close_margin

    # -- public API ---------------------------------------------------------

    def analyze(self, history: Sequence[RoundResult]) -> ForensicsReport:
        history = list(history)
        rounds: List[RoundForensics] = [self._analyze_round(r) for r in history]
        patterns = self._derive_patterns(history, rounds)

        total = len(history)
        committed = sum(1 for r in history if r.committed)
        failed = total - committed

        if total == 0:
            headline = "no rounds recorded"
        elif failed == 0:
            headline = f"all {total} rounds committed cleanly"
        elif committed == 0:
            headline = (
                f"{failed} round(s) failed; swarm never reached consensus"
            )
        else:
            headline = (
                f"{failed}/{total} round(s) failed to commit"
            )

        return ForensicsReport(
            total_rounds=total,
            committed_rounds=committed,
            failed_rounds=failed,
            rounds=rounds,
            patterns=patterns,
            headline=headline,
        )

    # -- per-round ----------------------------------------------------------

    def _analyze_round(self, r: RoundResult) -> RoundForensics:
        margin = r.aggregate_weight - r.threshold
        rejections = [v for v in r.votes if v.is_rejection]
        rejector_ids = [v.voter_id for v in rejections]
        counter_proofs = [
            CounterProofEntry(
                voter_id=v.voter_id,
                weight=v.weight,
                counter_proof=v.counter_proof or "(no written proof)",
            )
            for v in rejections
        ]
        next_view = self._pick_counter_leader(rejections)

        if r.committed:
            return RoundForensics(
                round_index=r.round_index,
                leader_id=r.leader_id,
                aggregate_weight=r.aggregate_weight,
                threshold=r.threshold,
                margin=margin,
                blocker="COMMITTED",
                severity="info",
                primary_blame=None,
                blame_reason=(
                    f"clean commit; margin +{margin:.3f} above threshold"
                ),
                counter_proofs=counter_proofs,
                rejector_ids=rejector_ids,
                next_view_leader=None,
                recommendation="no action needed",
                recommendation_priority="none",
            )

        # Failed round - classify blocker.
        if rejections and r.aggregate_weight >= r.threshold:
            # Hit the bar but at least one rejection vetoed.
            blocker = "UNREFUTED_REJECTION"
            strongest = self._strongest_rejection(rejections)
            severity = "high" if abs(strongest.weight) >= 0.7 else "medium"
            blame = strongest.voter_id
            blame_reason = (
                f"weight cleared threshold (+{margin:.3f}) but `{blame}` "
                f"issued an unrefuted veto (w={strongest.weight:+.2f})"
            )
            if strongest.counter_proof:
                recommendation = (
                    f"address `{blame}`'s counter-proof in next round; "
                    f"if rebutted, expect commit"
                )
                priority = "P1"
            else:
                recommendation = (
                    f"`{blame}` rejected without a written counter-proof "
                    f"— consider tightening verify_proposal contract"
                )
                priority = "P0"
        else:
            # Aggregate under threshold (and possibly rejections too).
            blocker = "BELOW_THRESHOLD"
            shortfall = -margin  # positive
            if shortfall <= self.threshold_too_high_close_margin:
                severity = "low"
            elif shortfall <= 2 * self.threshold_too_high_close_margin:
                severity = "medium"
            else:
                severity = "high"
            if rejections:
                strongest = self._strongest_rejection(rejections)
                blame = strongest.voter_id
                blame_reason = (
                    f"weight short by {shortfall:.3f}; strongest dissenter "
                    f"`{blame}` likely tipped the balance"
                )
            else:
                blame = r.leader_id
                blame_reason = (
                    f"weight short by {shortfall:.3f} with no rejections — "
                    f"leader `{blame}` (or followers) under-weighted"
                )
            if shortfall <= self.threshold_too_high_close_margin and not rejections:
                recommendation = (
                    f"near-miss by {shortfall:.3f}; consider lowering "
                    f"threshold or recruiting one more agent"
                )
                priority = "P2"
            elif rejections:
                recommendation = (
                    f"escalate `{next_view}` to next-round leader and "
                    f"address the counter-proof"
                ) if next_view else (
                    f"address dissent before re-running"
                )
                priority = "P1"
            else:
                recommendation = (
                    f"swarm confidence collapsed — recalibrate agents or "
                    f"lower threshold"
                )
                priority = "P0"

        return RoundForensics(
            round_index=r.round_index,
            leader_id=r.leader_id,
            aggregate_weight=r.aggregate_weight,
            threshold=r.threshold,
            margin=margin,
            blocker=blocker,
            severity=severity,
            primary_blame=blame,
            blame_reason=blame_reason,
            counter_proofs=counter_proofs,
            rejector_ids=rejector_ids,
            next_view_leader=next_view,
            recommendation=recommendation,
            recommendation_priority=priority,
        )

    @staticmethod
    def _strongest_rejection(rejections: List[Vote]) -> Vote:
        # Most negative weight = strongest dissent.
        return min(rejections, key=lambda v: v.weight)

    @staticmethod
    def _pick_counter_leader(rejections: List[Vote]) -> Optional[str]:
        if not rejections:
            return None
        return DisagreementForensics._strongest_rejection(rejections).voter_id

    # -- cross-round patterns ----------------------------------------------

    def _derive_patterns(
        self,
        history: Sequence[RoundResult],
        rounds: Sequence[RoundForensics],
    ) -> List[PatternFinding]:
        patterns: List[PatternFinding] = []
        failed = [r for r in history if not r.committed]

        if not failed:
            return patterns

        # CHRONIC_BLOCKER: same voter rejects across many failed rounds.
        rejection_counts: Counter[str] = Counter()
        for r in failed:
            for v in r.votes:
                if v.is_rejection:
                    rejection_counts[v.voter_id] += 1
        chronic = [
            aid
            for aid, n in rejection_counts.items()
            if n >= self.chronic_blocker_min_rounds
        ]
        if chronic:
            patterns.append(
                PatternFinding(
                    code="CHRONIC_BLOCKER",
                    priority="P0",
                    headline=(
                        f"{len(chronic)} agent(s) vetoed in >="
                        f"{self.chronic_blocker_min_rounds} failed rounds"
                    ),
                    detail=(
                        "Investigate whether these agents are calibrated "
                        "dissenters (keep) or noisy contrarians (slash / "
                        "quarantine). Pin one as next-round leader if their "
                        "counter-proofs are sound."
                    ),
                    suspects=sorted(chronic),
                )
            )

        # CALIBRATION_COLLAPSE: many BELOW_THRESHOLD failures with no
        # rejections - leaders proposed but the swarm under-weighted.
        below_no_rej = [
            r
            for r, fr in zip(history, rounds)
            if fr.blocker == "BELOW_THRESHOLD" and not any(v.is_rejection for v in r.votes)
        ]
        if (
            len(failed) > 0
            and len(below_no_rej) / len(failed)
            >= self.calibration_collapse_overconfidence
        ):
            leader_blame: Counter[str] = Counter(r.leader_id for r in below_no_rej)
            patterns.append(
                PatternFinding(
                    code="CALIBRATION_COLLAPSE",
                    priority="P1",
                    headline=(
                        f"{len(below_no_rej)} failed round(s) had no "
                        f"rejections - confidence too low to commit"
                    ),
                    detail=(
                        "Followers neither rejected nor weighted in strongly. "
                        "Recalibrate agents, recruit additional voters, or "
                        "lower the consensus threshold."
                    ),
                    suspects=[aid for aid, _ in leader_blame.most_common(3)],
                )
            )

        # THRESHOLD_TOO_HIGH: most failures are near-misses and no rejections.
        close = [
            r
            for r, fr in zip(history, rounds)
            if fr.blocker == "BELOW_THRESHOLD"
            and abs(fr.margin) <= self.threshold_too_high_close_margin
        ]
        any_rejection_in_failed = any(
            any(v.is_rejection for v in r.votes) for r in failed
        )
        if (
            len(failed) > 0
            and len(close) / len(failed) >= 0.5
            and not any_rejection_in_failed
        ):
            patterns.append(
                PatternFinding(
                    code="THRESHOLD_TOO_HIGH",
                    priority="P2",
                    headline=(
                        f"{len(close)}/{len(failed)} failures missed the "
                        f"threshold by <= "
                        f"{self.threshold_too_high_close_margin}"
                    ),
                    detail=(
                        "Consider lowering the consensus threshold or "
                        "adding one more high-confidence agent. The swarm "
                        "is not contesting — it just cannot quite muster "
                        "enough weight."
                    ),
                )
            )

        return patterns
