"""Threshold & slash-factor auto-tuning advisor.

Sibling to ``swarm_health``, ``disagreement_forensics``, ``round_replay_advisor``,
``agent_lifecycle_advisor``, ``voting_coalition_detector``, ``leader_rotation_advisor``,
and ``proposal_risk_scorer``. While those advisors classify the *current* state
of the swarm, this one recommends concrete hyperparameter changes to the engine
itself: ``threshold`` and ``slash_factor``.

Inputs
------
* ``history`` — ``Sequence[RoundResult]`` from ``engine.history``.
* ``reputation`` — current reputation map (``engine.reputation``).
* ``threshold`` — current commit threshold (``engine.threshold``).
* ``slash_factor`` — current slash factor (``engine.slash_factor``).
* ``risk_appetite`` — ``cautious`` / ``balanced`` / ``aggressive``.

The advisor never mutates the engine. It only reads history and emits a
``ThresholdTuningReport`` with structured findings, a recommended
``(threshold, slash_factor)`` pair, a P0-first deduped playbook, insights,
A-F grade, and text / markdown / JSON renderers.

Design constraints
------------------
* deterministic for a fixed input + injectable ``now_fn``,
* zero new dependencies (stdlib + pydantic only),
* never mutates ``history``, ``reputation``, or any engine state,
* recommended values are bounded around the current value (no jumps >50%
  per call) so an autopilot applying suggestions in a loop will converge
  smoothly rather than oscillate.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from typing import Callable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from src.core.state import RoundResult


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TuningFinding(BaseModel):
    code: str
    severity: int  # 0..100
    priority: str  # P0 / P1 / P2 / P3
    headline: str
    reason: str
    evidence: List[str] = Field(default_factory=list)


class TuningAction(BaseModel):
    id: str
    priority: str
    label: str
    reason: str
    owner: str  # operator / governance / oncall
    blast_radius: int  # 1..5
    reversibility: str  # low / medium / high
    suggested_value: Optional[float] = None


class TuningRecommendation(BaseModel):
    current_threshold: float
    recommended_threshold: float
    current_slash_factor: float
    recommended_slash_factor: float
    threshold_delta: float
    slash_factor_delta: float
    confidence: float  # 0..1, scales with history size


class ThresholdTuningReport(BaseModel):
    generated_at: datetime
    rounds_observed: int
    risk_appetite: str
    headline: str
    grade: str
    overall_risk_score: float  # 0..100; high = mistuned
    findings: List[TuningFinding] = Field(default_factory=list)
    recommendation: TuningRecommendation
    playbook: List[TuningAction] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------
    def to_text(self) -> str:
        lines = [
            self.headline,
            f"Rounds observed: {self.rounds_observed}",
            f"Risk appetite: {self.risk_appetite}",
            f"Overall mistuning risk: {self.overall_risk_score:.1f}/100  (grade {self.grade})",
            "",
            "Recommendation:",
            f"  threshold:     {self.recommendation.current_threshold:.3f}"
            f" -> {self.recommendation.recommended_threshold:.3f}"
            f"  (delta {self.recommendation.threshold_delta:+.3f})",
            f"  slash_factor:  {self.recommendation.current_slash_factor:.3f}"
            f" -> {self.recommendation.recommended_slash_factor:.3f}"
            f"  (delta {self.recommendation.slash_factor_delta:+.3f})",
            f"  confidence:    {self.recommendation.confidence:.2f}",
        ]
        if self.findings:
            lines.append("")
            lines.append("Findings:")
            for f in self.findings:
                lines.append(f"  [{f.priority}] {f.code} (sev {f.severity}): {f.headline}")
                lines.append(f"      {f.reason}")
        if self.playbook:
            lines.append("")
            lines.append("Playbook:")
            for a in self.playbook:
                val = "" if a.suggested_value is None else f" -> {a.suggested_value:.3f}"
                lines.append(
                    f"  [{a.priority}] {a.id}: {a.label} ({a.owner}, blast={a.blast_radius}){val}"
                )
                lines.append(f"      {a.reason}")
        if self.insights:
            lines.append("")
            lines.append("Insights:")
            for i in self.insights:
                lines.append(f"  - {i}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            f"# Threshold Tuning Report",
            "",
            f"**{self.headline}**",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Rounds observed | {self.rounds_observed} |",
            f"| Risk appetite | {self.risk_appetite} |",
            f"| Overall mistuning risk | {self.overall_risk_score:.1f}/100 |",
            f"| Grade | {self.grade} |",
            "",
            "## Recommendation",
            "",
            "| Parameter | Current | Recommended | Delta | Confidence |",
            "|---|---|---|---|---|",
            (
                f"| threshold | {self.recommendation.current_threshold:.3f} "
                f"| {self.recommendation.recommended_threshold:.3f} "
                f"| {self.recommendation.threshold_delta:+.3f} "
                f"| {self.recommendation.confidence:.2f} |"
            ),
            (
                f"| slash_factor | {self.recommendation.current_slash_factor:.3f} "
                f"| {self.recommendation.recommended_slash_factor:.3f} "
                f"| {self.recommendation.slash_factor_delta:+.3f} "
                f"| {self.recommendation.confidence:.2f} |"
            ),
        ]
        if self.findings:
            lines += [
                "",
                "## Findings",
                "",
                "| Priority | Code | Severity | Headline | Reason |",
                "|---|---|---|---|---|",
            ]
            for f in self.findings:
                lines.append(
                    f"| {f.priority} | {f.code} | {f.severity} | {f.headline} | {f.reason} |"
                )
        if self.playbook:
            lines += [
                "",
                "## Playbook",
                "",
                "| Priority | Id | Label | Owner | Blast | Reversibility | Suggested |",
                "|---|---|---|---|---|---|---|",
            ]
            for a in self.playbook:
                val = "" if a.suggested_value is None else f"{a.suggested_value:.3f}"
                lines.append(
                    f"| {a.priority} | {a.id} | {a.label} | {a.owner} | "
                    f"{a.blast_radius} | {a.reversibility} | {val} |"
                )
        lines += ["", "## Insights", ""]
        if self.insights:
            for i in self.insights:
                lines.append(f"- {i}")
        else:
            lines.append("- (none)")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            indent=2,
            default=str,
        )


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


_RISK_APPETITES = ("cautious", "balanced", "aggressive")


from .stats_utils import clamp as _clamp


def _priority_for(severity: int) -> str:
    if severity >= 75:
        return "P0"
    if severity >= 55:
        return "P1"
    if severity >= 30:
        return "P2"
    return "P3"


class ThresholdTuningAdvisor:
    """Recommend ``(threshold, slash_factor)`` adjustments from history.

    The advisor never mutates the engine. It accepts plain :class:`RoundResult`
    objects so it can be unit-tested without spinning up a full swarm.
    """

    def __init__(self, *, now_fn: Optional[Callable[[], datetime]] = None) -> None:
        self._now_fn = now_fn or datetime.utcnow

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        history: Sequence[RoundResult],
        *,
        threshold: float,
        slash_factor: float,
        reputation: Optional[Mapping[str, float]] = None,
        risk_appetite: str = "balanced",
        close_margin: float = 0.5,
    ) -> ThresholdTuningReport:
        if risk_appetite not in _RISK_APPETITES:
            raise ValueError(
                f"risk_appetite must be one of {_RISK_APPETITES!r}, got {risk_appetite!r}"
            )

        rounds = list(history)
        n = len(rounds)
        reputation = dict(reputation or {})

        appetite_mult = {"cautious": 1.15, "balanced": 1.0, "aggressive": 0.85}[risk_appetite]

        findings: List[TuningFinding] = []
        insights: List[str] = []

        # ---- aggregate stats ------------------------------------------------
        if n == 0:
            recommendation = TuningRecommendation(
                current_threshold=threshold,
                recommended_threshold=threshold,
                current_slash_factor=slash_factor,
                recommended_slash_factor=slash_factor,
                threshold_delta=0.0,
                slash_factor_delta=0.0,
                confidence=0.0,
            )
            insights.append("INSUFFICIENT_HISTORY: no rounds observed; keep current settings")
            playbook = [
                TuningAction(
                    id="OBSERVE_MORE_ROUNDS",
                    priority="P3",
                    label="Observe more rounds before tuning",
                    reason="History is empty; tuning recommendations require >=3 rounds.",
                    owner="operator",
                    blast_radius=1,
                    reversibility="high",
                )
            ]
            return ThresholdTuningReport(
                generated_at=self._now_fn(),
                rounds_observed=0,
                risk_appetite=risk_appetite,
                headline="VERDICT: grade=A no history; recommendation unchanged",
                grade="A",
                overall_risk_score=0.0,
                findings=findings,
                recommendation=recommendation,
                playbook=playbook,
                insights=insights,
            )

        committed = [r for r in rounds if r.committed]
        failed = [r for r in rounds if not r.committed]
        commit_rate = len(committed) / n

        aggregates = [r.aggregate_weight for r in rounds]
        median_agg = statistics.median(aggregates)

        # close margin = |aggregate - threshold| / threshold
        close_failures = [
            r for r in failed if 0 < (threshold - r.aggregate_weight) <= close_margin
        ]
        narrow_passes = [
            r for r in committed if 0 < (r.aggregate_weight - threshold) <= close_margin
        ]

        # rejections behaviour
        total_rejections = sum(
            1 for r in rounds for v in r.votes if v.is_rejection
        )
        unrefuted_rejection_failures = []
        for r in failed:
            has_rejection = any(v.is_rejection for v in r.votes)
            if has_rejection and r.aggregate_weight >= threshold:
                unrefuted_rejection_failures.append(r)

        # slashing rate
        rounds_with_slashes = [r for r in rounds if r.slashed]
        slash_rate = len(rounds_with_slashes) / n

        # slashed_out reputation share — flat-line indicator
        slashed_out_ids = {
            aid for aid, rep in reputation.items() if rep <= 0.0
        }

        # ---- findings -------------------------------------------------------
        # 1. low commit rate
        if n >= 3 and commit_rate < 0.4:
            sev = int(round((0.6 - commit_rate) * 120 * appetite_mult))
            sev = _clamp(sev, 30, 90)
            findings.append(
                TuningFinding(
                    code="LOW_COMMIT_RATE",
                    severity=int(sev),
                    priority=_priority_for(int(sev)),
                    headline=f"Commit rate {commit_rate:.0%} below 40%",
                    reason=(
                        "The swarm rarely commits; either the threshold is too high "
                        "relative to typical aggregates or proposals are systematically weak."
                    ),
                    evidence=[f"committed={len(committed)}/{n}"],
                )
            )

        # 2. too many close failures (threshold likely too high)
        if n >= 3 and len(close_failures) / n >= 0.30:
            sev = int(round((len(close_failures) / n) * 110 * appetite_mult))
            sev = _clamp(sev, 35, 85)
            findings.append(
                TuningFinding(
                    code="THRESHOLD_LIKELY_TOO_HIGH",
                    severity=int(sev),
                    priority=_priority_for(int(sev)),
                    headline=(
                        f"{len(close_failures)} of {n} rounds failed within {close_margin} of threshold"
                    ),
                    reason=(
                        "Many failures cluster just below the threshold. Lowering it (or accepting "
                        "a slightly wider close-margin band) would convert these into commits."
                    ),
                    evidence=[f"close_failures={len(close_failures)}", f"threshold={threshold:.3f}"],
                )
            )

        # 3. too many narrow passes with active rejection — threshold likely too low
        risky_passes = [r for r in narrow_passes if any(v.is_rejection for v in r.votes)]
        if n >= 3 and len(risky_passes) / n >= 0.25:
            sev = int(round((len(risky_passes) / n) * 110 * appetite_mult))
            sev = _clamp(sev, 30, 80)
            findings.append(
                TuningFinding(
                    code="THRESHOLD_LIKELY_TOO_LOW",
                    severity=int(sev),
                    priority=_priority_for(int(sev)),
                    headline=(
                        f"{len(risky_passes)} commits passed within {close_margin} of threshold "
                        f"despite active rejections"
                    ),
                    reason=(
                        "Several rounds committed by a hair while at least one voter rejected. "
                        "Raising the threshold buys safety margin without much commit-rate cost."
                    ),
                    evidence=[f"risky_passes={len(risky_passes)}"],
                )
            )

        # 4. unrefuted rejection failures — threshold OK, but rejection power vs slashing mismatched
        if unrefuted_rejection_failures:
            sev = int(round(min(80, 40 + 10 * len(unrefuted_rejection_failures)) * appetite_mult))
            sev = _clamp(sev, 40, 90)
            findings.append(
                TuningFinding(
                    code="UNREFUTED_REJECTION_VETOES",
                    severity=int(sev),
                    priority=_priority_for(int(sev)),
                    headline=(
                        f"{len(unrefuted_rejection_failures)} rounds had aggregate>=threshold but "
                        f"a non-slashed rejection blocked commit"
                    ),
                    reason=(
                        "Rejecter weight is decisive but not being slashed often enough. "
                        "Raising slash_factor will discipline bad-faith vetoes."
                    ),
                    evidence=[
                        f"unrefuted_rejection_failures={len(unrefuted_rejection_failures)}",
                        f"slash_factor={slash_factor:.3f}",
                    ],
                )
            )

        # 5. slash rate runaway — slashing too aggressive
        if n >= 3 and slash_rate >= 0.5 and len(slashed_out_ids) >= 2:
            sev = int(round(40 + 10 * len(slashed_out_ids)) * appetite_mult)
            sev = _clamp(sev, 35, 80)
            findings.append(
                TuningFinding(
                    code="SLASH_RUNAWAY",
                    severity=int(sev),
                    priority=_priority_for(int(sev)),
                    headline=(
                        f"{len(slashed_out_ids)} agents have reputation<=0 and slashes occur in "
                        f"{slash_rate:.0%} of rounds"
                    ),
                    reason=(
                        "Slashing is decimating the roster. Lower slash_factor to preserve "
                        "diversity, or audit upstream proposal quality."
                    ),
                    evidence=[
                        f"slashed_out={sorted(slashed_out_ids)}",
                        f"slash_factor={slash_factor:.3f}",
                    ],
                )
            )

        # 6. no rejections at all — possible echo chamber relative to slash setting
        if n >= 5 and total_rejections == 0:
            sev = int(round(30 * appetite_mult))
            findings.append(
                TuningFinding(
                    code="NO_REJECTIONS_OBSERVED",
                    severity=int(sev),
                    priority=_priority_for(int(sev)),
                    headline="No rejections in observed history",
                    reason=(
                        "The swarm never dissents. Either slash_factor is intimidating voters or "
                        "the proposal pool is uniformly easy. Consider lowering slash_factor."
                    ),
                    evidence=[f"slash_factor={slash_factor:.3f}"],
                )
            )

        # 7. healthy commit rate and no close calls — no change needed
        healthy = (
            0.55 <= commit_rate <= 0.85
            and len(close_failures) / max(n, 1) <= 0.15
            and len(risky_passes) / max(n, 1) <= 0.15
            and not unrefuted_rejection_failures
            and not (slash_rate >= 0.5 and len(slashed_out_ids) >= 2)
        )
        if healthy and n >= 5:
            findings.append(
                TuningFinding(
                    code="HEALTHY_TUNING",
                    severity=10,
                    priority="P3",
                    headline="Current threshold and slash_factor look well-tuned",
                    reason=(
                        "Commit rate is in the healthy band, close-call density is low, and "
                        "no veto pathology is evident."
                    ),
                )
            )

        # ---- recommendation -------------------------------------------------
        # Bounds: never move more than 50% per call. floors/ceilings to keep
        # the engine usable.
        rec_thr = threshold
        rec_slash = slash_factor

        # threshold adjustment
        thr_finding_codes = {f.code for f in findings}
        if "THRESHOLD_LIKELY_TOO_HIGH" in thr_finding_codes:
            # target = median aggregate + small safety
            target = max(0.1, min(threshold, median_agg + 0.1))
            rec_thr = (threshold + target) / 2.0
        if "THRESHOLD_LIKELY_TOO_LOW" in thr_finding_codes:
            target = threshold + 0.4
            rec_thr = (threshold + target) / 2.0
        if "LOW_COMMIT_RATE" in thr_finding_codes and "THRESHOLD_LIKELY_TOO_LOW" not in thr_finding_codes:
            # only nudge down if not already nudging up
            rec_thr = min(rec_thr, threshold * 0.85)

        # slash_factor adjustment
        if "UNREFUTED_REJECTION_VETOES" in thr_finding_codes:
            rec_slash = min(1.0, slash_factor + 0.20)
        if "SLASH_RUNAWAY" in thr_finding_codes:
            rec_slash = max(0.05, slash_factor - 0.20)
        if "NO_REJECTIONS_OBSERVED" in thr_finding_codes and "UNREFUTED_REJECTION_VETOES" not in thr_finding_codes:
            rec_slash = max(0.05, slash_factor - 0.10)

        # Bound per-call delta to +/-50% of current to converge smoothly.
        thr_lo = max(0.05, threshold * 0.5)
        thr_hi = threshold * 1.5 if threshold > 0 else threshold + 1.0
        rec_thr = _clamp(rec_thr, thr_lo, thr_hi)

        slash_lo = max(0.05, slash_factor * 0.5)
        slash_hi = min(1.0, slash_factor * 1.5) if slash_factor > 0 else 0.5
        rec_slash = _clamp(rec_slash, slash_lo, slash_hi)

        # Appetite final shrink/expand: aggressive lets us move more, cautious less.
        # We do this by interpolating against the current value.
        shrink_mult = {"cautious": 0.6, "balanced": 1.0, "aggressive": 1.25}[risk_appetite]
        rec_thr = threshold + (rec_thr - threshold) * shrink_mult
        rec_slash = slash_factor + (rec_slash - slash_factor) * shrink_mult

        confidence = _clamp(min(1.0, n / 20.0), 0.0, 1.0)

        recommendation = TuningRecommendation(
            current_threshold=threshold,
            recommended_threshold=round(rec_thr, 4),
            current_slash_factor=slash_factor,
            recommended_slash_factor=round(rec_slash, 4),
            threshold_delta=round(rec_thr - threshold, 4),
            slash_factor_delta=round(rec_slash - slash_factor, 4),
            confidence=round(confidence, 3),
        )

        # ---- playbook -------------------------------------------------------
        playbook: List[TuningAction] = []
        seen_ids: set = set()

        def add(action: TuningAction) -> None:
            if action.id in seen_ids:
                return
            seen_ids.add(action.id)
            playbook.append(action)

        if "THRESHOLD_LIKELY_TOO_HIGH" in thr_finding_codes:
            add(
                TuningAction(
                    id="LOWER_THRESHOLD",
                    priority="P1",
                    label="Lower commit threshold",
                    reason=(
                        "Many failures clustered just below the current threshold. "
                        f"Try {recommendation.recommended_threshold:.3f}."
                    ),
                    owner="governance",
                    blast_radius=3,
                    reversibility="medium",
                    suggested_value=recommendation.recommended_threshold,
                )
            )
        if "THRESHOLD_LIKELY_TOO_LOW" in thr_finding_codes:
            add(
                TuningAction(
                    id="RAISE_THRESHOLD",
                    priority="P1",
                    label="Raise commit threshold",
                    reason=(
                        "Multiple commits squeaked through despite active rejections. "
                        f"Raise to {recommendation.recommended_threshold:.3f} to add safety margin."
                    ),
                    owner="governance",
                    blast_radius=3,
                    reversibility="medium",
                    suggested_value=recommendation.recommended_threshold,
                )
            )
        if "UNREFUTED_REJECTION_VETOES" in thr_finding_codes:
            add(
                TuningAction(
                    id="RAISE_SLASH_FACTOR",
                    priority="P0",
                    label="Raise slash_factor to discipline veto power",
                    reason=(
                        "Non-slashed rejections are blocking rounds that otherwise meet threshold. "
                        f"Raise slash_factor to {recommendation.recommended_slash_factor:.3f}."
                    ),
                    owner="governance",
                    blast_radius=4,
                    reversibility="medium",
                    suggested_value=recommendation.recommended_slash_factor,
                )
            )
        if "SLASH_RUNAWAY" in thr_finding_codes:
            add(
                TuningAction(
                    id="LOWER_SLASH_FACTOR",
                    priority="P0",
                    label="Lower slash_factor to preserve roster",
                    reason=(
                        "Too many agents have been flat-lined. Lower slash_factor to "
                        f"{recommendation.recommended_slash_factor:.3f} and audit proposal quality."
                    ),
                    owner="governance",
                    blast_radius=4,
                    reversibility="low",
                    suggested_value=recommendation.recommended_slash_factor,
                )
            )
        if "LOW_COMMIT_RATE" in thr_finding_codes:
            add(
                TuningAction(
                    id="DIAGNOSE_LOW_COMMIT_RATE",
                    priority="P1",
                    label="Diagnose low commit rate",
                    reason=(
                        "Commit rate is below 40%. Inspect proposal quality and leader rotation "
                        "alongside any threshold change."
                    ),
                    owner="operator",
                    blast_radius=2,
                    reversibility="high",
                )
            )
        if "NO_REJECTIONS_OBSERVED" in thr_finding_codes:
            add(
                TuningAction(
                    id="ENCOURAGE_DISSENT",
                    priority="P2",
                    label="Investigate absence of dissent",
                    reason=(
                        "Zero rejections across observed history may signal echo-chamber "
                        "behaviour or a too-high slash penalty deterring counter-proofs."
                    ),
                    owner="operator",
                    blast_radius=2,
                    reversibility="high",
                )
            )

        # Cross-cutting: cautious always adds an audit when grade is C/D/F.
        # We compute grade first.
        if findings:
            top_sev = max(f.severity for f in findings)
            rest_sum = sum(f.severity for f in findings) - top_sev
            overall_risk_score = _clamp(top_sev + 0.4 * min(rest_sum, 60), 0.0, 100.0)
        else:
            overall_risk_score = 0.0

        if overall_risk_score >= 75:
            grade = "F"
        elif overall_risk_score >= 55:
            grade = "D"
        elif overall_risk_score >= 35:
            grade = "C"
        elif overall_risk_score >= 18:
            grade = "B"
        else:
            grade = "A"

        # Forced F if both threshold-too-high and slash-runaway present (system collapse).
        if "SLASH_RUNAWAY" in thr_finding_codes and "LOW_COMMIT_RATE" in thr_finding_codes:
            grade = "F"

        if risk_appetite == "cautious" and grade in {"C", "D", "F"}:
            add(
                TuningAction(
                    id="SCHEDULE_TUNING_REVIEW",
                    priority="P2",
                    label="Schedule a follow-up tuning review",
                    reason="Cautious risk appetite: revisit after applying changes.",
                    owner="operator",
                    blast_radius=1,
                    reversibility="high",
                )
            )

        if not playbook:
            add(
                TuningAction(
                    id="HOLD_CURRENT_TUNING",
                    priority="P3",
                    label="Hold current threshold and slash_factor",
                    reason="No tuning pathology detected; current settings look healthy.",
                    owner="operator",
                    blast_radius=1,
                    reversibility="high",
                )
            )

        # Aggressive: trim P3 fallback if any P0/P1 present.
        if risk_appetite == "aggressive":
            has_high = any(a.priority in {"P0", "P1"} for a in playbook)
            if has_high:
                playbook = [a for a in playbook if a.priority != "P3"]

        # Sort P0 first, then by id for determinism
        priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        playbook.sort(key=lambda a: (priority_order.get(a.priority, 9), a.id))

        # ---- insights -------------------------------------------------------
        if n < 3:
            insights.append("THIN_HISTORY: <3 rounds observed; treat recommendation as advisory only")
        if commit_rate >= 0.95:
            insights.append("HIGH_COMMIT_RATE: swarm commits nearly every round; tuning may be too lax")
        if commit_rate <= 0.2:
            insights.append("VERY_LOW_COMMIT_RATE: swarm is effectively stalled")
        if len(slashed_out_ids) >= 1:
            insights.append(f"SLASHED_OUT_AGENTS: {sorted(slashed_out_ids)}")
        if total_rejections > 0 and n >= 3:
            insights.append(
                f"REJECTION_DENSITY: {total_rejections} rejection votes across {n} rounds"
            )
        if "THRESHOLD_LIKELY_TOO_HIGH" in thr_finding_codes and "THRESHOLD_LIKELY_TOO_LOW" in thr_finding_codes:
            insights.append(
                "CONFLICTING_SIGNALS: both too-high and too-low evidence present; "
                "investigate proposal quality before adjusting"
            )
        if recommendation.threshold_delta == 0.0 and recommendation.slash_factor_delta == 0.0:
            insights.append("NO_CHANGE_RECOMMENDED")
        if not insights:
            insights.append("STABLE_BASELINE: tuning signals are quiet")

        headline = (
            f"VERDICT: grade={grade} rounds={n} commit_rate={commit_rate:.0%} "
            f"thr={threshold:.3f}->{recommendation.recommended_threshold:.3f} "
            f"slash={slash_factor:.3f}->{recommendation.recommended_slash_factor:.3f}"
        )

        return ThresholdTuningReport(
            generated_at=self._now_fn(),
            rounds_observed=n,
            risk_appetite=risk_appetite,
            headline=headline,
            grade=grade,
            overall_risk_score=round(overall_risk_score, 1),
            findings=findings,
            recommendation=recommendation,
            playbook=playbook,
            insights=insights,
        )


__all__ = [
    "ThresholdTuningAdvisor",
    "ThresholdTuningReport",
    "TuningRecommendation",
    "TuningFinding",
    "TuningAction",
]
