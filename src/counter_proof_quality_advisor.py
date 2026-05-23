"""Counter-proof quality advisor.

10th agentic sibling to ``swarm_health``, ``disagreement_forensics``,
``round_replay_advisor``, ``agent_lifecycle_advisor``,
``voting_coalition_detector``, ``leader_rotation_advisor``,
``proposal_risk_scorer``, ``threshold_tuning_advisor``, and
``vote_dispersion_advisor``.

Where its siblings analyze *who* rejected and *whether* it mattered, this
advisor scrutinises *how well-argued* the rejection itself was. mBFT's
defeasibility hinges on counter-proofs (``Vote.counter_proof``) being
substantive — a bare "no" with no justification undermines the whole
veto mechanism. This module reads ``engine.history`` and emits per-rejection
verdicts plus per-voter counter-proof quality scores, plus a cross-swarm
P0-first playbook for fixing the lazy/abusive rejection patterns it finds.

Detectors (per rejection vote):

* ``MISSING_COUNTER_PROOF`` — ``counter_proof`` is ``None`` or whitespace
  (P0; the rejecter is exploiting the veto mechanism without justification).
* ``VAGUE_COUNTER_PROOF`` — fewer than ``min_chars`` chars or fewer than
  ``min_tokens`` distinct content tokens (P1).
* ``TEMPLATE_REPETITION`` — same normalized counter-proof reused by the
  same voter in >= ``template_threshold`` rounds (P1).
* ``LOW_INFORMATION`` — high stop-word ratio and no negation/contradiction
  markers (P2).
* ``GENERIC_PHRASE`` — text matches a small allow-list of low-signal
  rejection phrases ("wrong", "disagree", "bad proof", etc.) (P2).
* ``CONTRADICTORY_REJECTER`` — voter previously voted positively on a
  proposal with the same ``committed_solution`` text (or its solution text
  if not committed) (P2).
* ``HIGH_QUALITY`` — informative; long, unique, contains contradiction
  marker (P3, positive signal).

Per-voter aggregates:

* ``rejections_cast`` / ``rejections_with_proof`` / ``avg_proof_length``
* ``quality_score`` 0..100, modulated by risk_appetite
* ``status`` ``ok`` / ``watch`` / ``coach`` / ``slash_candidate``

A-F grade, P0-first deduped playbook, insights, deterministic with
injectable ``now_fn``, never mutates ``history`` or ``reputation``.
Renderers: ``to_text`` / ``to_markdown`` (tables) / ``to_json``
(``sort_keys=True, indent=2, default=str`` byte-stable).
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from src.core.state import RoundResult, Vote
from src.stats_utils import clamp as _clamp


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CounterProofFinding(BaseModel):
    round_index: int
    voter_id: str
    weight: float
    code: str
    severity: int  # 0..100
    priority: str  # P0 / P1 / P2 / P3
    reason: str
    excerpt: Optional[str] = None


class VoterQuality(BaseModel):
    voter_id: str
    rejections_cast: int = 0
    rejections_with_proof: int = 0
    missing_proof_count: int = 0
    vague_proof_count: int = 0
    template_repetition_count: int = 0
    contradictory_count: int = 0
    high_quality_count: int = 0
    avg_proof_length: float = 0.0
    quality_score: float = 0.0  # 0..100, higher is better
    status: str = "ok"  # ok | watch | coach | slash_candidate


class CPAdvisorAction(BaseModel):
    id: str
    priority: str
    label: str
    reason: str
    owner: str
    blast_radius: int  # 1..5
    reversibility: str  # low / medium / high
    target_voters: List[str] = Field(default_factory=list)
    suggested_value: Optional[float] = None


class CounterProofQualityReport(BaseModel):
    generated_at: datetime
    rounds_observed: int
    rejections_observed: int
    risk_appetite: str
    headline: str
    grade: str  # A..F
    portfolio_quality_score: float  # 0..100 (higher = better)
    overall_risk_score: float  # 0..100 (higher = worse)
    findings: List[CounterProofFinding] = Field(default_factory=list)
    voters: List[VoterQuality] = Field(default_factory=list)
    playbook: List[CPAdvisorAction] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------
    def to_text(self) -> str:
        lines: List[str] = [
            self.headline,
            f"Rounds observed: {self.rounds_observed}",
            f"Rejections observed: {self.rejections_observed}",
            f"Risk appetite: {self.risk_appetite}",
            (
                f"Portfolio quality: {self.portfolio_quality_score:.1f}/100  "
                f"(risk {self.overall_risk_score:.1f}/100, grade {self.grade})"
            ),
        ]
        if self.voters:
            lines.append("")
            lines.append("Voters:")
            for v in self.voters:
                lines.append(
                    f"  {v.voter_id}: rejections={v.rejections_cast} "
                    f"w/proof={v.rejections_with_proof} missing={v.missing_proof_count} "
                    f"vague={v.vague_proof_count} template={v.template_repetition_count} "
                    f"hq={v.high_quality_count} score={v.quality_score:.1f} "
                    f"[{v.status}]"
                )
        if self.findings:
            lines.append("")
            lines.append("Findings:")
            for f in self.findings:
                exc = "" if not f.excerpt else f"  -- \"{f.excerpt}\""
                lines.append(
                    f"  [r{f.round_index} {f.priority}] {f.voter_id} {f.code} "
                    f"(sev {f.severity}): {f.reason}{exc}"
                )
        if self.playbook:
            lines.append("")
            lines.append("Playbook:")
            for a in self.playbook:
                tgt = "" if not a.target_voters else f" -> {','.join(a.target_voters)}"
                lines.append(
                    f"  [{a.priority}] {a.id}: {a.label} ({a.owner}, blast={a.blast_radius}){tgt}"
                )
                lines.append(f"      {a.reason}")
        lines.append("")
        lines.append("Insights:")
        if self.insights:
            for i in self.insights:
                lines.append(f"  - {i}")
        else:
            lines.append("  - (none)")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines: List[str] = [
            "# Counter-Proof Quality Report",
            "",
            f"**{self.headline}**",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Rounds observed | {self.rounds_observed} |",
            f"| Rejections observed | {self.rejections_observed} |",
            f"| Risk appetite | {self.risk_appetite} |",
            f"| Portfolio quality | {self.portfolio_quality_score:.1f}/100 |",
            f"| Overall risk | {self.overall_risk_score:.1f}/100 |",
            f"| Grade | {self.grade} |",
        ]
        lines += [
            "",
            "## Voters",
            "",
            "| Voter | Rejections | w/Proof | Missing | Vague | Template | HQ | Score | Status |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        if self.voters:
            for v in self.voters:
                lines.append(
                    f"| {v.voter_id} | {v.rejections_cast} | {v.rejections_with_proof} "
                    f"| {v.missing_proof_count} | {v.vague_proof_count} "
                    f"| {v.template_repetition_count} | {v.high_quality_count} "
                    f"| {v.quality_score:.1f} | {v.status} |"
                )
        else:
            lines.append("| _none_ | | | | | | | | |")
        lines += [
            "",
            "## Findings",
            "",
            "| Round | Voter | Code | Severity | Priority | Reason |",
            "|---|---|---|---|---|---|",
        ]
        if self.findings:
            for f in self.findings:
                reason = f.reason.replace("|", "\\|")
                lines.append(
                    f"| {f.round_index} | {f.voter_id} | {f.code} | {f.severity} "
                    f"| {f.priority} | {reason} |"
                )
        else:
            lines.append("| _none_ | | | | | |")
        lines += [
            "",
            "## Playbook",
            "",
            "| Priority | Id | Label | Owner | Blast | Reversibility | Targets |",
            "|---|---|---|---|---|---|---|",
        ]
        if self.playbook:
            for a in self.playbook:
                tgt = ",".join(a.target_voters) if a.target_voters else ""
                lines.append(
                    f"| {a.priority} | {a.id} | {a.label} | {a.owner} | "
                    f"{a.blast_radius} | {a.reversibility} | {tgt} |"
                )
        else:
            lines.append("| _none_ | | | | | | |")
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

# Small allow-list of low-signal rejection phrases. Lowercased, exact-after-normalize.
_GENERIC_PHRASES: Set[str] = {
    "no",
    "nope",
    "wrong",
    "disagree",
    "bad proof",
    "bad",
    "incorrect",
    "not valid",
    "invalid",
    "rejected",
    "reject",
    "does not work",
    "doesnt work",
    "doesn't work",
    "nah",
    "i disagree",
    "false",
}

# Tokens that signal an actual argument (negation / contradiction / reference).
_CONTRADICTION_TOKENS: Set[str] = {
    "because",
    "since",
    "however",
    "but",
    "contradicts",
    "violates",
    "fails",
    "counter",
    "counterexample",
    "axiom",
    "theorem",
    "lemma",
    "proof",
    "step",
    "line",
    "section",
    "case",
    "instead",
    "rather",
    "wrong because",
    "invalid because",
}

_STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "is",
    "it",
    "this",
    "that",
    "be",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "are",
    "was",
    "were",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "its",
    "his",
    "her",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS]


def _has_contradiction_marker(text: str) -> bool:
    norm = " " + _normalize(text) + " "
    return any(f" {tok} " in norm for tok in _CONTRADICTION_TOKENS)


def _priority_for(severity: int) -> str:
    if severity >= 75:
        return "P0"
    if severity >= 55:
        return "P1"
    if severity >= 30:
        return "P2"
    return "P3"


def _excerpt(text: Optional[str], limit: int = 60) -> Optional[str]:
    if not text:
        return None
    norm = _normalize(text)
    if len(norm) <= limit:
        return norm
    return norm[: limit - 3] + "..."


class CounterProofQualityAdvisor:
    """Audit counter_proof quality on rejection votes across history.

    The advisor never mutates the engine. It reads ``RoundResult`` objects
    and an optional ``reputation`` map and emits a
    :class:`CounterProofQualityReport`.
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
        reputation: Optional[Mapping[str, float]] = None,
        risk_appetite: str = "balanced",
        min_chars: int = 20,
        min_tokens: int = 4,
        template_threshold: int = 2,
    ) -> CounterProofQualityReport:
        if risk_appetite not in _RISK_APPETITES:
            raise ValueError(
                f"risk_appetite must be one of {_RISK_APPETITES!r}, got {risk_appetite!r}"
            )

        appetite_mult = {"cautious": 1.15, "balanced": 1.0, "aggressive": 0.85}[risk_appetite]

        rounds = list(history)
        n = len(rounds)
        rep_map = dict(reputation or {})

        # Collect rejections (voter, round_idx, weight, cp_text)
        rejections: List[Tuple[int, str, float, Optional[str]]] = []
        for r in rounds:
            for v in r.votes:
                if v.is_rejection:
                    rejections.append((r.round_index, v.voter_id, v.weight, v.counter_proof))

        # Build per-voter normalized-cp counter for template detection
        voter_cp_norms: Dict[str, Counter] = defaultdict(Counter)
        for _, voter, _, cp in rejections:
            norm = _normalize(cp)
            if norm:
                voter_cp_norms[voter][norm] += 1

        # Track per-voter positive votes scoped to (leader_id, committed_solution-or-leader),
        # to detect contradictory rejecter (voter rejects a round whose leader
        # they had previously supported on a committed proposal).
        prior_positive_leaders: Dict[str, Set[str]] = defaultdict(set)

        # Walk in order to build prior-positive history
        findings: List[CounterProofFinding] = []
        voter_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {
                "rejections_cast": 0,
                "rejections_with_proof": 0,
                "missing_proof_count": 0,
                "vague_proof_count": 0,
                "template_repetition_count": 0,
                "contradictory_count": 0,
                "high_quality_count": 0,
                "low_information_count": 0,
                "generic_phrase_count": 0,
                "_proof_length_sum": 0,
            }
        )

        for r in rounds:
            # Update prior_positive_solutions with positive votes from this round
            # *before* we process rejections of subsequent rounds. But we need
            # positives that happened in *strictly prior* rounds for the contradiction
            # check, so we update at the end of the loop body.
            # (positives are recorded after the rejection loop below)
            for v in r.votes:
                if v.is_rejection:
                    stats = voter_stats[v.voter_id]
                    stats["rejections_cast"] += 1
                    cp = v.counter_proof
                    norm = _normalize(cp)
                    stats["_proof_length_sum"] += len(norm)
                    if not norm:
                        stats["missing_proof_count"] += 1
                        sev = int(round(80 * appetite_mult))
                        sev = _clamp(sev, 60, 95)
                        findings.append(
                            CounterProofFinding(
                                round_index=r.round_index,
                                voter_id=v.voter_id,
                                weight=v.weight,
                                code="MISSING_COUNTER_PROOF",
                                severity=int(sev),
                                priority=_priority_for(int(sev)),
                                reason=(
                                    "Rejection has no counter_proof; the defeasibility "
                                    "argument is unsupported."
                                ),
                                excerpt=None,
                            )
                        )
                        continue  # subsequent text-based detectors are N/A
                    stats["rejections_with_proof"] += 1

                    content = _content_tokens(norm)
                    # 1. vague (too short or too few content tokens)
                    is_vague = len(norm) < min_chars or len(set(content)) < min_tokens
                    if is_vague:
                        stats["vague_proof_count"] += 1
                        sev = int(round(60 * appetite_mult))
                        sev = _clamp(sev, 45, 80)
                        findings.append(
                            CounterProofFinding(
                                round_index=r.round_index,
                                voter_id=v.voter_id,
                                weight=v.weight,
                                code="VAGUE_COUNTER_PROOF",
                                severity=int(sev),
                                priority=_priority_for(int(sev)),
                                reason=(
                                    f"Counter-proof is short ({len(norm)} chars, "
                                    f"{len(set(content))} unique content tokens); "
                                    "expand the argument."
                                ),
                                excerpt=_excerpt(cp),
                            )
                        )

                    # 2. template repetition
                    if voter_cp_norms[v.voter_id][norm] >= template_threshold:
                        stats["template_repetition_count"] += 1
                        sev = int(round(55 * appetite_mult))
                        sev = _clamp(sev, 40, 75)
                        findings.append(
                            CounterProofFinding(
                                round_index=r.round_index,
                                voter_id=v.voter_id,
                                weight=v.weight,
                                code="TEMPLATE_REPETITION",
                                severity=int(sev),
                                priority=_priority_for(int(sev)),
                                reason=(
                                    f"Counter-proof reused {voter_cp_norms[v.voter_id][norm]}x "
                                    "verbatim; likely templated rejection."
                                ),
                                excerpt=_excerpt(cp),
                            )
                        )

                    # 3. generic phrase
                    if norm in _GENERIC_PHRASES:
                        stats["generic_phrase_count"] += 1
                        sev = int(round(45 * appetite_mult))
                        sev = _clamp(sev, 30, 65)
                        findings.append(
                            CounterProofFinding(
                                round_index=r.round_index,
                                voter_id=v.voter_id,
                                weight=v.weight,
                                code="GENERIC_PHRASE",
                                severity=int(sev),
                                priority=_priority_for(int(sev)),
                                reason=(
                                    "Counter-proof matches a generic low-signal "
                                    "rejection phrase."
                                ),
                                excerpt=_excerpt(cp),
                            )
                        )

                    # 4. low information (high stopword ratio + no contradiction marker)
                    if len(content) >= 3 and not _has_contradiction_marker(norm):
                        total = max(1, len(_tokens(norm)))
                        stop_ratio = 1.0 - (len(content) / total)
                        if stop_ratio >= 0.7:
                            stats["low_information_count"] += 1
                            sev = int(round(35 * appetite_mult))
                            sev = _clamp(sev, 25, 55)
                            findings.append(
                                CounterProofFinding(
                                    round_index=r.round_index,
                                    voter_id=v.voter_id,
                                    weight=v.weight,
                                    code="LOW_INFORMATION",
                                    severity=int(sev),
                                    priority=_priority_for(int(sev)),
                                    reason=(
                                        "Counter-proof is mostly stopwords and lacks a "
                                        "contradiction marker (because/violates/fails/etc.)."
                                    ),
                                    excerpt=_excerpt(cp),
                                )
                            )

                    # 5. contradictory rejecter
                    if r.leader_id in prior_positive_leaders[v.voter_id]:
                        stats["contradictory_count"] += 1
                        sev = int(round(55 * appetite_mult))
                        sev = _clamp(sev, 40, 75)
                        findings.append(
                            CounterProofFinding(
                                round_index=r.round_index,
                                voter_id=v.voter_id,
                                weight=v.weight,
                                code="CONTRADICTORY_REJECTER",
                                severity=int(sev),
                                priority=_priority_for(int(sev)),
                                reason=(
                                    "Voter previously voted positively on a committed "
                                    "proposal from the same leader; rejection contradicts "
                                    "earlier stance."
                                ),
                                excerpt=_excerpt(cp),
                            )
                        )

                    # 6. high quality (only when no other negative finding fired)
                    is_hq = (
                        not is_vague
                        and voter_cp_norms[v.voter_id][norm] == 1
                        and norm not in _GENERIC_PHRASES
                        and _has_contradiction_marker(norm)
                        and len(norm) >= max(min_chars * 2, 40)
                    )
                    if is_hq:
                        stats["high_quality_count"] += 1
                        sev = 5
                        findings.append(
                            CounterProofFinding(
                                round_index=r.round_index,
                                voter_id=v.voter_id,
                                weight=v.weight,
                                code="HIGH_QUALITY",
                                severity=sev,
                                priority="P3",
                                reason=(
                                    "Detailed, unique counter-proof with a "
                                    "contradiction marker."
                                ),
                                excerpt=_excerpt(cp),
                            )
                        )

            # After processing rejections in this round, record any positives.
            # We only count positives on *committed* rounds (otherwise the
            # voter's "support" was for a proposal the swarm itself rejected,
            # which isn't a contradiction worth flagging).
            if r.committed:
                for v in r.votes:
                    if not v.is_rejection and v.weight > 0:
                        prior_positive_leaders[v.voter_id].add(r.leader_id)

        # --- voters -------------------------------------------------------
        voters: List[VoterQuality] = []
        for voter_id in sorted(voter_stats.keys()):
            s = voter_stats[voter_id]
            rejections_cast = s["rejections_cast"]
            avg_len = (
                s["_proof_length_sum"] / rejections_cast if rejections_cast else 0.0
            )
            # quality_score: start at 100, deduct per-issue
            deduction = (
                25 * s["missing_proof_count"]
                + 12 * s["vague_proof_count"]
                + 10 * s["template_repetition_count"]
                + 10 * s["contradictory_count"]
                + 8 * s["generic_phrase_count"]
                + 6 * s["low_information_count"]
            )
            bonus = 5 * s["high_quality_count"]
            base = 100.0 - (deduction / max(1, rejections_cast))
            score = _clamp(base + bonus / max(1, rejections_cast), 0.0, 100.0)
            # appetite shift: cautious is harsher, aggressive lenient
            shift = {"cautious": -5.0, "balanced": 0.0, "aggressive": +5.0}[risk_appetite]
            score = _clamp(score + shift, 0.0, 100.0)

            status = "ok"
            missing_ratio = s["missing_proof_count"] / max(1, rejections_cast)
            if missing_ratio >= 0.5 and rejections_cast >= 2:
                status = "slash_candidate"
            elif score < 40:
                status = "coach"
            elif score < 65:
                status = "watch"

            voters.append(
                VoterQuality(
                    voter_id=voter_id,
                    rejections_cast=rejections_cast,
                    rejections_with_proof=s["rejections_with_proof"],
                    missing_proof_count=s["missing_proof_count"],
                    vague_proof_count=s["vague_proof_count"],
                    template_repetition_count=s["template_repetition_count"],
                    contradictory_count=s["contradictory_count"],
                    high_quality_count=s["high_quality_count"],
                    avg_proof_length=round(avg_len, 2),
                    quality_score=round(score, 1),
                    status=status,
                )
            )

        # --- portfolio + grade -------------------------------------------
        if voters:
            portfolio_quality = statistics.mean(v.quality_score for v in voters)
        else:
            portfolio_quality = 100.0  # no rejections == nothing to grade negatively
        overall_risk = _clamp(100.0 - portfolio_quality, 0.0, 100.0)

        # Forced grade gates
        slash_candidates = [v for v in voters if v.status == "slash_candidate"]
        if slash_candidates:
            grade = "F"
        elif overall_risk >= 60:
            grade = "D"
        elif overall_risk >= 40:
            grade = "C"
        elif overall_risk >= 20:
            grade = "B"
        else:
            grade = "A"

        # --- playbook ----------------------------------------------------
        playbook: List[CPAdvisorAction] = []
        seen: Set[str] = set()

        def add(action: CPAdvisorAction) -> None:
            if action.id in seen:
                return
            seen.add(action.id)
            playbook.append(action)

        if slash_candidates:
            add(
                CPAdvisorAction(
                    id="SLASH_NO_PROOF_REJECTERS",
                    priority="P0",
                    label="Slash voters rejecting without counter-proof",
                    reason=(
                        "These voters reject >=50% of the time with no counter_proof, "
                        "exploiting the veto mechanism without justification."
                    ),
                    owner="governance",
                    blast_radius=5,
                    reversibility="low",
                    target_voters=[v.voter_id for v in slash_candidates],
                )
            )

        vague_voters = [v for v in voters if v.vague_proof_count >= 2]
        if vague_voters:
            add(
                CPAdvisorAction(
                    id="COACH_VAGUE_REJECTERS",
                    priority="P1",
                    label="Coach voters to expand vague counter-proofs",
                    reason=(
                        "These voters routinely file short, low-content counter-proofs."
                    ),
                    owner="operator",
                    blast_radius=2,
                    reversibility="high",
                    target_voters=[v.voter_id for v in vague_voters],
                )
            )

        template_voters = [v for v in voters if v.template_repetition_count >= 1]
        if len(template_voters) >= 1 and any(
            v.template_repetition_count >= 2 for v in template_voters
        ):
            add(
                CPAdvisorAction(
                    id="DEDUPE_TEMPLATE_REJECTERS",
                    priority="P1",
                    label="Investigate templated rejection patterns",
                    reason=(
                        "One or more voters reuse the same counter-proof verbatim "
                        "across multiple rounds; likely lazy or scripted rejection."
                    ),
                    owner="operator",
                    blast_radius=2,
                    reversibility="high",
                    target_voters=[v.voter_id for v in template_voters],
                )
            )

        contradictory_voters = [v for v in voters if v.contradictory_count >= 1]
        if contradictory_voters:
            add(
                CPAdvisorAction(
                    id="AUDIT_CONTRADICTORY_REJECTERS",
                    priority="P1",
                    label="Audit voters with contradictory rejection history",
                    reason=(
                        "These voters rejected a solution they had previously "
                        "supported; check for strategic flip-flopping."
                    ),
                    owner="governance",
                    blast_radius=3,
                    reversibility="medium",
                    target_voters=[v.voter_id for v in contradictory_voters],
                )
            )

        generic_voters = [v for v in voters if any(
            f.voter_id == v.voter_id and f.code == "GENERIC_PHRASE" for f in findings
        )]
        if len(generic_voters) >= 2:
            add(
                CPAdvisorAction(
                    id="STANDARDIZE_COUNTER_PROOF_FORMAT",
                    priority="P2",
                    label="Standardize counter-proof format guidance",
                    reason=(
                        "Multiple voters file generic rejection phrases; publish a "
                        "minimal counter-proof template (e.g. claim + violated axiom + "
                        "counterexample)."
                    ),
                    owner="governance",
                    blast_radius=2,
                    reversibility="high",
                    target_voters=[v.voter_id for v in generic_voters],
                )
            )

        low_info_findings = [f for f in findings if f.code == "LOW_INFORMATION"]
        if len(low_info_findings) >= 3:
            add(
                CPAdvisorAction(
                    id="REQUIRE_CONTRADICTION_MARKERS",
                    priority="P2",
                    label="Require contradiction markers in counter-proofs",
                    reason=(
                        ">=3 rejections lack any contradiction marker; consider "
                        "rejecting counter-proofs that don't reference why."
                    ),
                    owner="governance",
                    blast_radius=3,
                    reversibility="medium",
                )
            )

        hq_voters = [v for v in voters if v.high_quality_count >= 1]
        if hq_voters:
            add(
                CPAdvisorAction(
                    id="REWARD_HIGH_QUALITY_REJECTERS",
                    priority="P2",
                    label="Reward voters with high-quality counter-proofs",
                    reason=(
                        "These voters consistently file detailed, well-argued "
                        "counter-proofs; consider boosting their reputation weight."
                    ),
                    owner="governance",
                    blast_radius=2,
                    reversibility="high",
                    target_voters=[v.voter_id for v in hq_voters],
                )
            )

        if risk_appetite == "cautious" and grade in {"C", "D", "F"}:
            add(
                CPAdvisorAction(
                    id="SCHEDULE_REJECTION_AUDIT",
                    priority="P2",
                    label="Schedule a follow-up rejection-quality audit",
                    reason="Cautious risk appetite: re-evaluate after applying changes.",
                    owner="operator",
                    blast_radius=1,
                    reversibility="high",
                )
            )

        if not playbook:
            add(
                CPAdvisorAction(
                    id="HEALTHY_REJECTION_DISCIPLINE",
                    priority="P3",
                    label="Maintain current rejection-quality discipline",
                    reason="No rejection-quality pathology detected.",
                    owner="operator",
                    blast_radius=1,
                    reversibility="high",
                )
            )

        # Aggressive trims P3 + lone P2 when any P0/P1 present
        if risk_appetite == "aggressive":
            has_high = any(a.priority in {"P0", "P1"} for a in playbook)
            if has_high:
                playbook = [a for a in playbook if a.priority != "P3"]

        # Sort P0 first, then by id
        priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        playbook.sort(key=lambda a: (priority_order.get(a.priority, 9), a.id))

        # --- insights ----------------------------------------------------
        insights: List[str] = []
        if not rejections:
            insights.append(
                "NO_REJECTIONS_OBSERVED: history contains no rejection votes; "
                "counter-proof quality cannot be assessed"
            )
        total_missing = sum(v.missing_proof_count for v in voters)
        total_rej = len(rejections)
        if total_rej and total_missing / total_rej >= 0.3:
            insights.append(
                f"PROOFLESS_REJECTION_CULTURE: {total_missing}/{total_rej} rejections lack counter-proof"
            )
        if any(v.template_repetition_count >= 2 for v in voters):
            insights.append("TEMPLATED_REJECTIONS_DETECTED")
        if slash_candidates:
            insights.append(
                "SLASH_CANDIDATE_PRESENT: at least one voter rejects without proof >=50% of the time"
            )
        if hq_voters and not slash_candidates:
            insights.append("HIGH_QUALITY_REJECTERS_PRESENT")
        if n >= 3 and total_rej == 0:
            insights.append("ECHO_CHAMBER_RISK: zero rejections across >=3 rounds")
        if portfolio_quality >= 85 and total_rej > 0:
            insights.append("HEALTHY_COUNTER_PROOF_DISCIPLINE")
        if not insights:
            insights.append("MIXED_SIGNALS")

        # Sort findings by (round_index, priority, voter_id) for determinism
        findings.sort(
            key=lambda f: (
                f.round_index,
                priority_order.get(f.priority, 9),
                f.voter_id,
                f.code,
            )
        )

        headline = (
            f"VERDICT: grade={grade} rejections={total_rej} voters={len(voters)} "
            f"quality={portfolio_quality:.1f}/100"
        )

        return CounterProofQualityReport(
            generated_at=self._now_fn(),
            rounds_observed=n,
            rejections_observed=total_rej,
            risk_appetite=risk_appetite,
            headline=headline,
            grade=grade,
            portfolio_quality_score=round(portfolio_quality, 1),
            overall_risk_score=round(overall_risk, 1),
            findings=findings,
            voters=voters,
            playbook=playbook,
            insights=insights,
        )


__all__ = [
    "CounterProofQualityAdvisor",
    "CounterProofQualityReport",
    "CounterProofFinding",
    "VoterQuality",
    "CPAdvisorAction",
]
