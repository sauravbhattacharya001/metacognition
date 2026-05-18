"""Proposal risk scorer.

A pre-submission **agentic** advisor that sits alongside the
``SwarmHealthMonitor``, ``DisagreementForensics``, ``RoundReplayAdvisor``,
``AgentLifecycleAdvisor``, ``VotingCoalitionDetector`` and
``LeaderRotationAdvisor`` family of mBFT sibling advisors.

Where those tools operate *after* a round has run, this advisor operates
*before* the leader broadcasts. Given the draft :class:`Proposal`, the
leader's reputation, the recent ``engine.history`` and (optionally) the
current voter roster, it predicts whether the proposal is likely to
commit and emits an actionable, prioritised risk report so the leader can
revise (or abandon) the proposal *before* burning a round.

Design notes:
* Pure stdlib + pydantic, never mutates inputs.
* Deterministic given an injectable ``now_fn``.
* Renderers ``to_text`` / ``to_markdown`` / ``to_json`` (byte-stable JSON
  via ``sort_keys=True, indent=2, default=str``).
"""
from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.core.state import Proposal, RoundResult, Vote

__all__ = [
    "RiskFactor",
    "PredictedVoter",
    "PlaybookAction",
    "ProposalRiskReport",
    "ProposalRiskScorer",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

_DIM_WEIGHTS: Dict[str, float] = {
    "proof_quality": 0.20,
    "confidence_calibration": 0.15,
    "leader_reputation_pressure": 0.10,
    "solution_uniqueness": 0.10,
    "predicted_aggregate_shortfall": 0.25,
    "unrefuted_rejection_risk": 0.15,
    "format_compliance": 0.05,
}

_APPETITE_MULT: Dict[str, float] = {
    "cautious": 1.10,
    "balanced": 1.00,
    "aggressive": 0.85,
}

_PLACEHOLDER_RE = re.compile(
    r"\{\{[^}]+\}\}|<INSERT[^>]*>|<TODO[^>]*>|\bTODO\b|\bXXX\b|\bTBD\b",
    re.IGNORECASE,
)

_PROOF_MARKERS_RE = re.compile(
    r"\b(because|therefore|since|hence|thus|qed|lemma|step\s*\d+|proof\s*:|"
    r"we\s+have|by\s+induction|by\s+contradiction)\b",
    re.IGNORECASE,
)

_CITATION_RE = re.compile(
    r"\[\d+\]|\(\d{4}\)|doi\s*:\s*\S+|https?://\S+",
    re.IGNORECASE,
)

_COUNTER_ANTICIPATION_RE = re.compile(
    r"\b(counter[- ]?proof|anticipat\w*|rebuttal|preempt\w*|"
    r"refut\w*|address\w*\s+objection|edge\s*case)\b",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class RiskFactor(BaseModel):
    code: str
    dimension: str
    severity: float = Field(ge=0.0, le=100.0)
    weight: float
    reason: str


class PredictedVoter(BaseModel):
    voter_id: str
    rep: float
    predicted_weight: float
    prior_rejections_of_leader: int
    contribution_band: str  # 'positive' | 'fence' | 'rejection'


class PlaybookAction(BaseModel):
    id: str
    priority: str
    label: str
    owner: str
    blast_radius: int = Field(ge=1, le=5)
    reversibility: str
    reason: str
    expected_impact: str
    targets: List[str] = Field(default_factory=list)


class ProposalRiskReport(BaseModel):
    proposal_id: str
    leader_id: str
    verdict: str
    grade: str
    overall_risk_score: float
    predicted_commit_probability: float
    predicted_aggregate: float
    threshold: float
    risk_appetite: str
    factors: List[RiskFactor] = Field(default_factory=list)
    predicted_voters: List[PredictedVoter] = Field(default_factory=list)
    playbook: List[PlaybookAction] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    summary: str
    generated_at: str

    # ---- renderers ----
    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            indent=2,
            default=str,
        )

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("ProposalRiskScorer report")
        lines.append("=" * 30)
        lines.append(self.summary)
        lines.append(
            f"risk={self.overall_risk_score:.1f}/100  "
            f"p(commit)={self.predicted_commit_probability:.2f}  "
            f"predicted_agg={self.predicted_aggregate:.3f} / "
            f"threshold={self.threshold:.3f}"
        )
        lines.append(
            f"verdict={self.verdict}  grade={self.grade}  appetite={self.risk_appetite}"
        )
        lines.append(f"generated_at={self.generated_at}")
        if self.factors:
            lines.append("")
            lines.append("Risk factors:")
            for f in self.factors:
                lines.append(
                    f"  [{f.code}] dim={f.dimension} sev={f.severity:.0f} "
                    f"w={f.weight:.2f} - {f.reason}"
                )
        if self.predicted_voters:
            lines.append("")
            lines.append("Predicted voters (top 10):")
            for v in self.predicted_voters[:10]:
                lines.append(
                    f"  {v.voter_id} rep={v.rep:.2f} "
                    f"w={v.predicted_weight:+.2f} "
                    f"band={v.contribution_band} "
                    f"prior_rej={v.prior_rejections_of_leader}"
                )
        if self.playbook:
            lines.append("")
            lines.append("Playbook:")
            for a in self.playbook:
                tgt = f" -> {','.join(a.targets)}" if a.targets else ""
                lines.append(
                    f"  [{a.priority}] {a.id}{tgt} (owner={a.owner}, "
                    f"blast={a.blast_radius}, rev={a.reversibility}): {a.label}"
                )
        if self.insights:
            lines.append("")
            lines.append("Insights:")
            for ins in self.insights:
                lines.append(f"  - {ins}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# ProposalRiskScorer report")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(self.summary)
        lines.append("")
        lines.append(
            f"- **proposal_id**: `{self.proposal_id}`\n"
            f"- **leader_id**: `{self.leader_id}`\n"
            f"- **verdict**: `{self.verdict}`\n"
            f"- **grade**: `{self.grade}`\n"
            f"- **overall_risk_score**: {self.overall_risk_score:.1f}/100\n"
            f"- **predicted_commit_probability**: {self.predicted_commit_probability:.2f}\n"
            f"- **predicted_aggregate**: {self.predicted_aggregate:.3f} / "
            f"threshold {self.threshold:.3f}\n"
            f"- **risk_appetite**: `{self.risk_appetite}`\n"
            f"- **generated_at**: `{self.generated_at}`"
        )
        if self.factors:
            lines.append("")
            lines.append("## Risk factors")
            lines.append("")
            lines.append("| Code | Dimension | Severity | Weight | Reason |")
            lines.append("|---|---|---|---|---|")
            for f in self.factors:
                lines.append(
                    f"| `{f.code}` | {f.dimension} | {f.severity:.0f} | "
                    f"{f.weight:.2f} | {f.reason} |"
                )
        if self.predicted_voters:
            lines.append("")
            lines.append("## Predicted voters (top 10)")
            lines.append("")
            lines.append(
                "| Voter | Rep | Predicted weight | Band | Prior rejections of leader |"
            )
            lines.append("|---|---|---|---|---|")
            for v in self.predicted_voters[:10]:
                lines.append(
                    f"| `{v.voter_id}` | {v.rep:.2f} | "
                    f"{v.predicted_weight:+.2f} | {v.contribution_band} | "
                    f"{v.prior_rejections_of_leader} |"
                )
        if self.playbook:
            lines.append("")
            lines.append("## Playbook")
            lines.append("")
            lines.append(
                "| Priority | Action | Owner | Blast | Reversibility | Label | Reason |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for a in self.playbook:
                lines.append(
                    f"| {a.priority} | `{a.id}` | {a.owner} | "
                    f"{a.blast_radius} | {a.reversibility} | {a.label} | {a.reason} |"
                )
        if self.insights:
            lines.append("")
            lines.append("## Insights")
            lines.append("")
            for ins in self.insights:
                lines.append(f"- {ins}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


class ProposalRiskScorer:
    """Pre-submission proposal risk scorer."""

    def __init__(
        self,
        *,
        threshold: float = 3.0,
        min_obs: int = 2,
        risk_appetite: str = "balanced",
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if risk_appetite not in _APPETITE_MULT:
            raise ValueError(
                f"risk_appetite must be one of {sorted(_APPETITE_MULT)}"
            )
        self.threshold = float(threshold)
        self.min_obs = int(min_obs)
        self.risk_appetite = risk_appetite
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    # ----- public entry point -----
    def score(
        self,
        proposal: Proposal,
        *,
        leader_reputation: float = 0.5,
        history: Sequence[RoundResult] = (),
        roster: Optional[Mapping[str, float]] = None,
    ) -> ProposalRiskReport:
        # defensive: never mutate caller inputs
        proposal_snapshot = copy.deepcopy(proposal)
        history_snapshot = [copy.deepcopy(r) for r in (history or [])]
        roster_snapshot: Dict[str, float] = dict(roster or {})

        # ensure leader present in roster shape only (do not mutate caller's dict)
        leader_id = proposal_snapshot.agent_id

        factors: List[RiskFactor] = []
        insights: List[str] = []

        # ----- dimension 1: proof_quality -----
        proof_sev, proof_reason = self._score_proof(proposal_snapshot)
        factors.append(
            RiskFactor(
                code=("MISSING_PROOF" if proof_sev >= 90 else "PROOF_QUALITY"),
                dimension="proof_quality",
                severity=proof_sev,
                weight=_DIM_WEIGHTS["proof_quality"],
                reason=proof_reason,
            )
        )
        if proof_sev >= 60:
            insights.append("PROOF_LIKELY_INSUFFICIENT")

        # ----- dimension 2: confidence_calibration -----
        cal_sev, cal_code, cal_reason, cal_flags = self._score_calibration(
            proposal_snapshot, leader_id, leader_reputation, history_snapshot
        )
        factors.append(
            RiskFactor(
                code=cal_code,
                dimension="confidence_calibration",
                severity=cal_sev,
                weight=_DIM_WEIGHTS["confidence_calibration"],
                reason=cal_reason,
            )
        )
        if "OVERCONFIDENT" in cal_flags:
            insights.append("OVERCONFIDENT_LEADER")
        if "UNDERCONFIDENT" in cal_flags:
            insights.append("UNDERCONFIDENT_LEADER")

        # ----- dimension 3: leader_reputation_pressure -----
        leader_sev, leader_reason = self._score_leader_rep(
            leader_id, leader_reputation, history_snapshot
        )
        factors.append(
            RiskFactor(
                code="LEADER_REP_PRESSURE",
                dimension="leader_reputation_pressure",
                severity=leader_sev,
                weight=_DIM_WEIGHTS["leader_reputation_pressure"],
                reason=leader_reason,
            )
        )

        # ----- dimension 4: solution_uniqueness -----
        uniq_sev, uniq_code, uniq_reason, uniq_flags = self._score_uniqueness(
            proposal_snapshot, history_snapshot
        )
        factors.append(
            RiskFactor(
                code=uniq_code,
                dimension="solution_uniqueness",
                severity=uniq_sev,
                weight=_DIM_WEIGHTS["solution_uniqueness"],
                reason=uniq_reason,
            )
        )
        if "STALE_REJECTED_REPLAY" in uniq_flags:
            insights.append("STALE_REJECTED_REPLAY")
        if "REDUNDANT_PROPOSAL" in uniq_flags:
            insights.append("REDUNDANT_PROPOSAL")

        # ----- dimension 5: predicted_aggregate_shortfall + predicted voters -----
        predicted_voters, predicted_agg = self._predict_voters(
            proposal_snapshot, leader_id, history_snapshot, roster_snapshot
        )
        shortfall_pct = 0.0
        if predicted_agg < self.threshold:
            shortfall_pct = max(
                0.0,
                (self.threshold - predicted_agg) / max(self.threshold, 1e-6),
            ) * 100.0
        shortfall_sev = _clamp(shortfall_pct)
        shortfall_reason = (
            f"predicted_agg={predicted_agg:.3f} vs threshold={self.threshold:.3f} "
            f"(shortfall {shortfall_pct:.0f}%)"
        )
        factors.append(
            RiskFactor(
                code=(
                    "LIKELY_BELOW_THRESHOLD"
                    if shortfall_pct > 0
                    else "PREDICTED_OK"
                ),
                dimension="predicted_aggregate_shortfall",
                severity=shortfall_sev,
                weight=_DIM_WEIGHTS["predicted_aggregate_shortfall"],
                reason=shortfall_reason,
            )
        )
        if shortfall_pct > 0:
            insights.append("LIKELY_BELOW_THRESHOLD")

        # ----- dimension 6: unrefuted_rejection_risk -----
        unref_sev, unref_reason, chronic_count = self._score_unrefuted(
            proposal_snapshot, leader_id, history_snapshot, predicted_voters
        )
        factors.append(
            RiskFactor(
                code=(
                    "UNREFUTED_REJECTION_RISK"
                    if unref_sev > 0
                    else "NO_KNOWN_REJECTORS"
                ),
                dimension="unrefuted_rejection_risk",
                severity=unref_sev,
                weight=_DIM_WEIGHTS["unrefuted_rejection_risk"],
                reason=unref_reason,
            )
        )
        if chronic_count >= 2:
            insights.append("CHRONIC_BLOCKER_AUDIENCE")

        # ----- dimension 7: format_compliance -----
        fmt_sev, fmt_code, fmt_reason, fmt_flags = self._score_format(
            proposal_snapshot
        )
        factors.append(
            RiskFactor(
                code=fmt_code,
                dimension="format_compliance",
                severity=fmt_sev,
                weight=_DIM_WEIGHTS["format_compliance"],
                reason=fmt_reason,
            )
        )

        # ----- meta insights -----
        if len(history_snapshot) < 2:
            insights.append("EMPTY_HISTORY")
        if len(roster_snapshot) < 3:
            insights.append("THIN_ROSTER")

        # ----- aggregate risk score -----
        weighted = sum(f.severity * f.weight for f in factors)
        weight_sum = sum(f.weight for f in factors)
        base_score = weighted / weight_sum if weight_sum else 0.0
        appetite_mult = _APPETITE_MULT[self.risk_appetite]
        overall = _clamp(base_score * appetite_mult)

        # ----- verdict -----
        force_block = (
            "PLACEHOLDER" in fmt_flags
            or "STALE_REJECTED_REPLAY" in uniq_flags
            or overall >= 80
        )
        if force_block:
            verdict = "BLOCK_SUBMISSION"
        elif overall >= 60:
            verdict = "HIGH_RISK"
        elif overall >= 40:
            verdict = "ELEVATED"
        elif overall >= 20:
            verdict = "LOW"
        else:
            verdict = "SAFE"

        # ----- predicted commit probability (logistic) -----
        try:
            p_commit = 1.0 / (1.0 + math.exp((overall - 50.0) / 15.0))
        except OverflowError:
            p_commit = 0.02 if overall > 50 else 0.98
        p_commit = max(0.02, min(0.98, p_commit))

        # ----- grade -----
        if verdict == "BLOCK_SUBMISSION":
            grade = "F"
        elif overall < 20:
            grade = "A"
        elif overall < 40:
            grade = "B"
        elif overall < 60:
            grade = "C"
        elif overall < 80:
            grade = "D"
        else:
            grade = "F"

        # ----- playbook -----
        playbook = self._build_playbook(
            proof_sev=proof_sev,
            cal_flags=cal_flags,
            leader_rep=leader_reputation,
            uniq_flags=uniq_flags,
            shortfall_pct=shortfall_pct,
            unref_sev=unref_sev,
            fmt_flags=fmt_flags,
            verdict=verdict,
            grade=grade,
            predicted_voters=predicted_voters,
        )

        # ----- summary headline -----
        summary = (
            f"VERDICT {verdict}: risk={overall:.0f}/100 grade={grade} "
            f"p(commit)={p_commit:.2f} ({len(playbook)} actions)"
        )

        return ProposalRiskReport(
            proposal_id=proposal_snapshot.proposal_id,
            leader_id=leader_id,
            verdict=verdict,
            grade=grade,
            overall_risk_score=round(overall, 3),
            predicted_commit_probability=round(p_commit, 4),
            predicted_aggregate=round(predicted_agg, 4),
            threshold=self.threshold,
            risk_appetite=self.risk_appetite,
            factors=factors,
            predicted_voters=predicted_voters,
            playbook=playbook,
            insights=insights,
            summary=summary,
            generated_at=self._now_fn().isoformat(),
        )

    # ----------------- helpers -----------------

    def _score_proof(self, p: Proposal) -> Tuple[float, str]:
        proof = (p.proof or "").strip()
        if not proof:
            return 95.0, "missing proof body"
        n = len(proof)
        sev = 0.0
        notes: List[str] = []
        if n < 20:
            sev += 60
            notes.append(f"very short proof ({n} chars)")
        elif n < 60:
            sev += 35
            notes.append(f"short proof ({n} chars)")
        markers = len(_PROOF_MARKERS_RE.findall(proof))
        if markers == 0:
            sev += 25
            notes.append("no reasoning markers (because/therefore/...)")
        elif markers == 1:
            sev += 8
            notes.append("only one reasoning marker")
        if not _CITATION_RE.search(proof):
            sev += 10
            notes.append("no citation-like tokens")
        sev = _clamp(sev)
        reason = "; ".join(notes) if notes else "proof appears substantive"
        return sev, reason

    def _leader_history(
        self, leader_id: str, history: Sequence[RoundResult]
    ) -> Tuple[int, int]:
        led = 0
        committed = 0
        for r in history:
            if r.leader_id == leader_id:
                led += 1
                if r.committed:
                    committed += 1
        return led, committed

    def _score_calibration(
        self,
        p: Proposal,
        leader_id: str,
        leader_rep: float,
        history: Sequence[RoundResult],
    ) -> Tuple[float, str, str, List[str]]:
        led, committed = self._leader_history(leader_id, history)
        flags: List[str] = []
        if led < self.min_obs:
            sev = 20.0 + 30.0 * max(0.0, p.confidence - 0.8)
            return (
                _clamp(sev),
                "CALIBRATION_INSUFFICIENT_DATA",
                f"only {led} prior leads; cannot calibrate confidence={p.confidence:.2f}",
                flags,
            )
        success_rate = committed / led if led else 0.0
        gap = p.confidence - success_rate
        if gap >= 0.30:
            sev = _clamp(40.0 + 100.0 * (gap - 0.30))
            flags.append("OVERCONFIDENT")
            return (
                sev,
                "OVERCONFIDENT",
                f"confidence={p.confidence:.2f} vs historical success={success_rate:.2f}",
                flags,
            )
        if gap <= -0.30:
            sev = _clamp(15.0 + 40.0 * (abs(gap) - 0.30))
            flags.append("UNDERCONFIDENT")
            return (
                sev,
                "UNDERCONFIDENT",
                f"confidence={p.confidence:.2f} below historical success={success_rate:.2f}",
                flags,
            )
        return (
            _clamp(15.0 + 30.0 * abs(gap)),
            "CALIBRATED",
            f"confidence={p.confidence:.2f} ~ historical success={success_rate:.2f}",
            flags,
        )

    def _score_leader_rep(
        self,
        leader_id: str,
        leader_rep: float,
        history: Sequence[RoundResult],
    ) -> Tuple[float, str]:
        sev = _clamp((1.0 - max(0.0, min(1.0, leader_rep))) * 100.0)
        # chronic-blocker pressure: did this leader appear as rejector?
        rej_count = 0
        for r in history:
            for v in r.votes:
                if v.voter_id == leader_id and v.weight < 0:
                    rej_count += 1
        if rej_count >= 2:
            sev = _clamp(sev + 15.0)
            return sev, (
                f"leader_rep={leader_rep:.2f}; leader has {rej_count} prior "
                f"rejections of other proposals (chronic-blocker hint)"
            )
        return sev, f"leader_rep={leader_rep:.2f}"

    def _score_uniqueness(
        self,
        p: Proposal,
        history: Sequence[RoundResult],
    ) -> Tuple[float, str, str, List[str]]:
        flags: List[str] = []
        sol = (p.solution or "").strip()
        if not sol:
            return 50.0, "EMPTY_SOLUTION", "solution is empty", flags
        sol_tokens = _tokens(sol)
        recent = list(history)[-8:]
        committed_sols: List[str] = []
        rejected_sols: List[str] = []
        for r in recent:
            if r.committed and r.committed_solution:
                committed_sols.append(r.committed_solution)
            elif not r.committed:
                # treat the leader's known proposed solution: we don't have it
                # directly, but votes' counter_proof captures the rejection. We
                # use the most recent committed_solution slot as a proxy when
                # available. For non-committed rounds we fall back to scanning
                # counter_proofs as 'rejected ideas'.
                for v in r.votes:
                    if v.weight < 0 and v.counter_proof:
                        rejected_sols.append(v.counter_proof)

        # match against committed (REDUNDANT)
        for cs in committed_sols:
            j = _jaccard(sol_tokens, _tokens(cs))
            if j >= 0.85 or sol.strip() == cs.strip():
                flags.append("REDUNDANT_PROPOSAL")
                return (
                    55.0,
                    "REDUNDANT",
                    f"solution closely matches a recently committed solution "
                    f"(jaccard={j:.2f})",
                    flags,
                )
        # match against rejected (STALE_REJECTED_REPLAY)
        for rs in rejected_sols:
            j = _jaccard(sol_tokens, _tokens(rs))
            if j >= 0.85 or sol.strip() == rs.strip():
                flags.append("STALE_REJECTED_REPLAY")
                return (
                    90.0,
                    "STALE_REJECTED_REPLAY",
                    f"solution closely matches a recently rejected idea "
                    f"(jaccard={j:.2f})",
                    flags,
                )
        return 10.0, "NOVEL_PROPOSAL", "solution appears novel vs recent history", flags

    def _predict_voters(
        self,
        p: Proposal,
        leader_id: str,
        history: Sequence[RoundResult],
        roster: Mapping[str, float],
    ) -> Tuple[List[PredictedVoter], float]:
        # voter prior alignment with this leader, derived from history
        votes_with_leader: Dict[str, List[float]] = {}
        for r in history:
            if r.leader_id != leader_id:
                continue
            for v in r.votes:
                votes_with_leader.setdefault(v.voter_id, []).append(v.weight)

        predicted: List[PredictedVoter] = []
        agg = 0.0
        # also predict for any voter we've seen historically even if not in roster
        all_voter_ids = set(roster.keys()) | set(votes_with_leader.keys())
        all_voter_ids.discard(leader_id)
        for vid in sorted(all_voter_ids):
            rep = float(roster.get(vid, 0.5))
            history_w = votes_with_leader.get(vid, [])
            prior_rej = sum(1 for w in history_w if w < 0)
            if history_w:
                avg = sum(history_w) / len(history_w)
                # blend in the proposal confidence as a mild positive nudge
                est = 0.7 * avg + 0.3 * (p.confidence - 0.5) * 2 * rep
            else:
                # unknown alignment - default slightly positive scaled by rep
                est = 0.2 * rep + 0.3 * (p.confidence - 0.5)
            # blend with reputation magnitude
            est = max(-1.0, min(1.0, est))
            predicted_weight = est * rep
            if predicted_weight > 0.1:
                band = "positive"
            elif predicted_weight < -0.1:
                band = "rejection"
            else:
                band = "fence"
            predicted.append(
                PredictedVoter(
                    voter_id=vid,
                    rep=rep,
                    predicted_weight=round(predicted_weight, 4),
                    prior_rejections_of_leader=prior_rej,
                    contribution_band=band,
                )
            )
            agg += predicted_weight
        predicted.sort(key=lambda x: (-x.predicted_weight, x.voter_id))
        return predicted, agg

    def _score_unrefuted(
        self,
        p: Proposal,
        leader_id: str,
        history: Sequence[RoundResult],
        predicted_voters: Sequence[PredictedVoter],
    ) -> Tuple[float, str, int]:
        chronic = [v for v in predicted_voters if v.prior_rejections_of_leader >= 2]
        chronic_count = len(chronic)
        if chronic_count == 0:
            return 0.0, "no chronic rejectors of this leader present", 0
        has_counter_anticipation = bool(
            _COUNTER_ANTICIPATION_RE.search(p.proof or "")
        )
        sev = _clamp(30.0 + 15.0 * chronic_count, 0.0, 90.0)
        if has_counter_anticipation:
            sev = _clamp(sev - 25.0)
            reason = (
                f"{chronic_count} chronic rejector(s) of this leader; "
                f"proof shows counter-anticipation markers (sev reduced)"
            )
        else:
            reason = (
                f"{chronic_count} chronic rejector(s) of this leader and "
                f"no counter-anticipation markers in proof"
            )
        return sev, reason, chronic_count

    def _score_format(
        self, p: Proposal
    ) -> Tuple[float, str, str, List[str]]:
        flags: List[str] = []
        sol = p.solution or ""
        if _PLACEHOLDER_RE.search(sol) or _PLACEHOLDER_RE.search(p.proof or ""):
            flags.append("PLACEHOLDER")
            return (
                95.0,
                "PLACEHOLDER",
                "unresolved placeholder token in solution or proof",
                flags,
            )
        if not sol.strip():
            return 80.0, "EMPTY_SOLUTION", "solution is empty", flags
        if p.confidence in (0.0, 1.0):
            return (
                40.0,
                "SUSPICIOUS_CONFIDENCE_EDGE",
                f"confidence is exactly {p.confidence:.2f}; treat as suspicious",
                flags,
            )
        return 5.0, "FORMAT_OK", "format checks pass", flags

    # ----------------- playbook -----------------

    def _build_playbook(
        self,
        *,
        proof_sev: float,
        cal_flags: List[str],
        leader_rep: float,
        uniq_flags: List[str],
        shortfall_pct: float,
        unref_sev: float,
        fmt_flags: List[str],
        verdict: str,
        grade: str,
        predicted_voters: Sequence[PredictedVoter],
    ) -> List[PlaybookAction]:
        items: List[PlaybookAction] = []

        def add(
            id_: str,
            priority: str,
            label: str,
            owner: str,
            blast: int,
            rev: str,
            reason: str,
            impact: str,
            targets: Optional[List[str]] = None,
        ) -> None:
            items.append(
                PlaybookAction(
                    id=id_,
                    priority=priority,
                    label=label,
                    owner=owner,
                    blast_radius=blast,
                    reversibility=rev,
                    reason=reason,
                    expected_impact=impact,
                    targets=list(targets or []),
                )
            )

        # P0s
        if "PLACEHOLDER" in fmt_flags:
            add(
                "REMOVE_PLACEHOLDER",
                "P0",
                "Resolve placeholder tokens before submission",
                "leader",
                2,
                "high",
                "unresolved placeholder token detected",
                "blocks submission until cleared",
            )
        if proof_sev >= 70:
            add(
                "REWRITE_PROOF",
                "P0",
                "Rewrite the proof: add reasoning markers, citations, length",
                "proof_engineer",
                3,
                "high",
                f"proof_quality severity={proof_sev:.0f}",
                "moves proof_quality factor into safe band",
            )
        if "STALE_REJECTED_REPLAY" in uniq_flags:
            add(
                "ABANDON_STALE_REPLAY",
                "P0",
                "Do not resubmit a previously rejected idea unchanged",
                "leader",
                3,
                "high",
                "solution closely matches a recently rejected idea",
                "prevents wasted round + reputation hit",
            )
        if "OVERCONFIDENT" in cal_flags and leader_rep < 0.4:
            add(
                "LOWER_CONFIDENCE",
                "P0",
                "Lower stated confidence to match historical lead_success",
                "leader",
                2,
                "high",
                "OVERCONFIDENT + low leader reputation",
                "reduces calibration risk + future slashing pressure",
            )

        # P1s
        if unref_sev >= 50:
            add(
                "ADD_COUNTER_PROOF",
                "P1",
                "Pre-emptively address known dissenters' objections in proof",
                "proof_engineer",
                2,
                "high",
                f"unrefuted_rejection_risk severity={unref_sev:.0f}",
                "reduces unrefuted_rejection_risk by ~30 points",
            )
        rejectors = [
            v.voter_id
            for v in predicted_voters
            if v.contribution_band == "rejection"
        ][:2]
        if rejectors:
            add(
                "PRE_NEGOTIATE_WITH",
                "P1",
                "Reach out to likely rejectors before broadcast",
                "leader",
                2,
                "high",
                f"predicted rejection from {','.join(rejectors)}",
                "may convert rejections to fence/positive",
                targets=rejectors,
            )
        if shortfall_pct >= 30:
            add(
                "SHRINK_SCOPE",
                "P1",
                "Narrow the proposal scope to reach quorum threshold",
                "leader",
                3,
                "high",
                f"predicted aggregate shortfall {shortfall_pct:.0f}%",
                "smaller scope improves predicted_aggregate",
            )

        # P2s
        if verdict in ("HIGH_RISK", "ELEVATED"):
            add(
                "REQUEST_PEER_REVIEW",
                "P2",
                "Get a peer to dry-run-vote the proposal before broadcast",
                "governance",
                1,
                "high",
                f"verdict={verdict}",
                "catches issues the scorer missed",
            )
        if verdict == "BLOCK_SUBMISSION":
            add(
                "LOG_FOR_AUDIT",
                "P2",
                "Log the blocked proposal for audit trail",
                "platform",
                1,
                "high",
                "BLOCK_SUBMISSION verdict requires audit entry",
                "compliance",
            )
        if self.risk_appetite == "cautious" and grade in ("C", "D", "F"):
            add(
                "SECOND_REVIEWER",
                "P2",
                "Solicit a second independent reviewer",
                "governance",
                1,
                "high",
                f"cautious appetite + grade {grade}",
                "lowers tail risk",
            )

        # P3 fallback
        if not items and grade in ("A", "B"):
            add(
                "PROPOSAL_READY",
                "P3",
                "Proposal looks safe to broadcast",
                "leader",
                1,
                "high",
                f"verdict={verdict}, grade={grade}",
                "no action required",
            )

        # dedupe by id (keep first - highest priority earliest)
        seen: set = set()
        deduped: List[PlaybookAction] = []
        for a in items:
            if a.id in seen:
                continue
            seen.add(a.id)
            deduped.append(a)
        deduped.sort(key=lambda a: (_PRIORITY_RANK[a.priority], a.id))

        # aggressive trims lone P2 + all P3 (when other priorities exist)
        if self.risk_appetite == "aggressive":
            has_higher = any(a.priority in ("P0", "P1") for a in deduped)
            if has_higher:
                deduped = [a for a in deduped if a.priority not in ("P3",)]
                # trim lone P2s
                p2s = [a for a in deduped if a.priority == "P2"]
                if len(p2s) == 1:
                    deduped = [a for a in deduped if a.priority != "P2"]
            else:
                # no P0/P1 -- still drop P3 fallback
                deduped = [a for a in deduped if a.priority != "P3"]

        return deduped
