"""Round replay advisor.

An *agentic* counterfactual "what-if" engine for the mBFT consensus
protocol. It sits alongside :class:`~src.swarm_health.SwarmHealthMonitor`
(aggregate health view) and :class:`~src.disagreement_forensics.DisagreementForensics`
(post-mortem root-cause), and answers a different question:

    *Given this round failed, what is the cheapest, safest knob I could
    have turned to make it commit?*

For every non-committed :class:`~src.core.state.RoundResult`, the advisor
replays the round under a small catalogue of interventions
(threshold lowering, swapping the leader to the strongest dissenter,
demoting a slashed agent's rejection, ignoring a persistent dissenter,
adding a redundant agent) and projects whether each intervention would
have flipped the outcome to ``committed``. Each intervention is
priced by ``cost_band`` and ``reversibility``, ranked into
``P0/P1/P2/P3`` and the cheapest commit-flipper is surfaced as
``recommended``.

Cross-round patterns are then collapsed into a small ``playbook``:
``REPEAT_BLOCKER_VOTER`` (one voter unsticks >=2 rounds when ignored),
``SYSTEMATIC_THRESHOLD_TOO_HIGH`` (>=50% of failures unstick by lowering
the threshold to the realised aggregate), ``SLASH_RECOVERY_NEEDED``
(>=2 rounds where dropping a slashed agent's rejection commits), and
``LEADER_BENCH_THIN`` (swapping leader never closes the margin).

Design notes:
* zero new dependencies (``pydantic`` only),
* deterministic and stateless - never mutates the engine, history, or
  reputation map,
* the advisor *recommends*; the operator (or autopilot) *decides*.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.core.state import RoundResult, Vote


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


class InterventionOutcome(BaseModel):
    intervention: str
    description: str
    target: Optional[str] = None
    projected_aggregate: float
    projected_committed: bool
    projected_unrefuted_rejection: bool
    delta_p_commit: float
    cost_band: str  # 'low' | 'medium' | 'high'
    reversibility: str  # 'reversible' | 'one_way'
    priority: str  # 'P0' | 'P1' | 'P2' | 'P3'
    notes: List[str] = Field(default_factory=list)


class RoundReplay(BaseModel):
    round_index: int
    original_committed: bool
    original_aggregate: float
    original_threshold: float
    original_blocker: str  # NONE | UNREFUTED_REJECTION | BELOW_THRESHOLD
    top_interventions: List[InterventionOutcome] = Field(default_factory=list)
    recommended: Optional[InterventionOutcome] = None


class PlaybookItem(BaseModel):
    pattern: str
    priority: str
    detail: str
    target: Optional[str] = None
    suggested_value: Optional[float] = None


class ReplayReport(BaseModel):
    rounds_replayed: int
    rounds_flippable: int
    rounds_unsalvageable: int
    per_round: List[RoundReplay] = Field(default_factory=list)
    playbook: List[PlaybookItem] = Field(default_factory=list)
    overall_grade: str
    generated_at: str

    # ---------------- renderers ----------------
    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("RoundReplayAdvisor report")
        lines.append("=" * 30)
        lines.append(
            f"rounds_replayed={self.rounds_replayed} "
            f"flippable={self.rounds_flippable} "
            f"unsalvageable={self.rounds_unsalvageable} "
            f"grade={self.overall_grade}"
        )
        lines.append(f"generated_at={self.generated_at}")
        for r in self.per_round:
            lines.append("")
            lines.append(
                f"round {r.round_index}: blocker={r.original_blocker} "
                f"aggregate={r.original_aggregate:.3f} "
                f"threshold={r.original_threshold:.3f}"
            )
            if r.recommended is not None:
                lines.append(
                    f"  -> RECOMMENDED [{r.recommended.priority}] "
                    f"{r.recommended.intervention} "
                    f"(cost={r.recommended.cost_band}, "
                    f"reversibility={r.recommended.reversibility})"
                )
                lines.append(f"     {r.recommended.description}")
            for iv in r.top_interventions:
                marker = " *" if (r.recommended is not None and iv.intervention == r.recommended.intervention) else "  "
                lines.append(
                    f"  {marker}[{iv.priority}] {iv.intervention} "
                    f"-> agg={iv.projected_aggregate:.3f} "
                    f"committed={iv.projected_committed} "
                    f"delta_p={iv.delta_p_commit:+.2f} "
                    f"cost={iv.cost_band}"
                )
        if self.playbook:
            lines.append("")
            lines.append("Cross-round playbook:")
            for item in self.playbook:
                target_part = f" target={item.target}" if item.target else ""
                value_part = (
                    f" value={item.suggested_value:.3f}"
                    if item.suggested_value is not None
                    else ""
                )
                lines.append(
                    f"  [{item.priority}] {item.pattern}{target_part}{value_part}: {item.detail}"
                )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# RoundReplayAdvisor report")
        lines.append("")
        lines.append(
            f"- **rounds_replayed**: {self.rounds_replayed}\n"
            f"- **rounds_flippable**: {self.rounds_flippable}\n"
            f"- **rounds_unsalvageable**: {self.rounds_unsalvageable}\n"
            f"- **overall_grade**: `{self.overall_grade}`\n"
            f"- **generated_at**: `{self.generated_at}`"
        )
        for r in self.per_round:
            lines.append("")
            lines.append(
                f"## Round {r.round_index} - blocker `{r.original_blocker}`"
            )
            lines.append(
                f"aggregate={r.original_aggregate:.3f}, "
                f"threshold={r.original_threshold:.3f}"
            )
            if r.recommended is not None:
                lines.append("")
                lines.append(
                    f"**Recommended:** `[{r.recommended.priority}] "
                    f"{r.recommended.intervention}` "
                    f"(cost={r.recommended.cost_band}, "
                    f"reversibility={r.recommended.reversibility}) - "
                    f"{r.recommended.description}"
                )
            if r.top_interventions:
                lines.append("")
                lines.append("| Priority | Intervention | Projected agg | Committed | delta_p | Cost |")
                lines.append("|---|---|---|---|---|---|")
                for iv in r.top_interventions:
                    lines.append(
                        f"| {iv.priority} | {iv.intervention} | "
                        f"{iv.projected_aggregate:.3f} | "
                        f"{iv.projected_committed} | "
                        f"{iv.delta_p_commit:+.2f} | {iv.cost_band} |"
                    )
        if self.playbook:
            lines.append("")
            lines.append("## Cross-round playbook")
            for item in self.playbook:
                target_part = f" (target=`{item.target}`)" if item.target else ""
                value_part = (
                    f" suggested={item.suggested_value:.3f}"
                    if item.suggested_value is not None
                    else ""
                )
                lines.append(
                    f"- **[{item.priority}] {item.pattern}**{target_part}{value_part}: {item.detail}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


class RoundReplayAdvisor:
    """Counterfactual replay engine for failed mBFT rounds."""

    def __init__(
        self,
        close_margin: float = 0.5,
        max_interventions_per_round: int = 6,
    ) -> None:
        if close_margin <= 0.0:
            raise ValueError("close_margin must be positive.")
        if max_interventions_per_round <= 0:
            raise ValueError("max_interventions_per_round must be positive.")
        self.close_margin = close_margin
        self.max_interventions_per_round = max_interventions_per_round

    # ----- public entry point -----
    def analyze(
        self,
        history: Sequence[RoundResult],
        reputation: Mapping[str, float],
        slash_factor: float = 0.5,
        now: Optional[Callable[[], datetime]] = None,
    ) -> ReplayReport:
        now_fn = now or (lambda: datetime.now(timezone.utc))

        per_round: List[RoundReplay] = []
        flippable = 0
        unsalvageable = 0

        for rr in history:
            if rr.committed:
                continue  # only failed rounds replay
            replay = self._replay_round(rr, reputation)
            per_round.append(replay)
            if replay.recommended is not None and replay.recommended.projected_committed:
                flippable += 1
            else:
                # check if ANY intervention flipped
                if any(iv.projected_committed for iv in replay.top_interventions):
                    flippable += 1
                else:
                    unsalvageable += 1

        playbook = self._cross_round_playbook(per_round)
        grade = self._grade(rounds_replayed=len(per_round), flippable=flippable)

        return ReplayReport(
            rounds_replayed=len(per_round),
            rounds_flippable=flippable,
            rounds_unsalvageable=unsalvageable,
            per_round=per_round,
            playbook=playbook,
            overall_grade=grade,
            generated_at=now_fn().isoformat(),
        )

    # ----- per-round replay -----
    def _replay_round(
        self,
        rr: RoundResult,
        reputation: Mapping[str, float],
    ) -> RoundReplay:
        blocker = self._classify_blocker(rr, reputation)
        candidates: List[InterventionOutcome] = []

        # 1) LOWER_THRESHOLD_TO_AGGREGATE - only if no unrefuted rejection
        unrefuted_now = self._has_unrefuted_rejection(rr.votes, reputation)
        if not unrefuted_now and rr.aggregate_weight < rr.threshold:
            shortfall = rr.threshold - rr.aggregate_weight
            committed = True  # by construction
            delta = 1.0 if committed else 0.0
            candidates.append(
                InterventionOutcome(
                    intervention="LOWER_THRESHOLD_TO_AGGREGATE",
                    description=(
                        f"Lower threshold from {rr.threshold:.3f} to "
                        f"{rr.aggregate_weight:.3f} (shortfall {shortfall:.3f})."
                    ),
                    target=None,
                    projected_aggregate=rr.aggregate_weight,
                    projected_committed=committed,
                    projected_unrefuted_rejection=False,
                    delta_p_commit=delta,
                    cost_band="high",  # changes engine-wide policy
                    reversibility="reversible",
                    priority="P1" if committed else "P3",
                    notes=[
                        f"shortfall_closed={shortfall:.3f}",
                        "policy_change=threshold",
                    ],
                )
            )

        # 2) DEMOTE_SLASHED_REJECTION - flips unrefuted rejection if rejecter rep<1.0
        if unrefuted_now:
            slashed_rejecters = [
                v for v in rr.votes
                if v.is_rejection and self._rep(v.voter_id, reputation) < 1.0
            ]
            if slashed_rejecters:
                # remove all of them in aggregate
                proj_agg, proj_unref = self._recompute(
                    rr=rr,
                    reputation=reputation,
                    drop_voters={v.voter_id for v in slashed_rejecters},
                )
                committed = (proj_agg >= rr.threshold) and not proj_unref
                delta = 1.0 if committed else self._partial_delta(rr, proj_agg)
                candidates.append(
                    InterventionOutcome(
                        intervention="DEMOTE_SLASHED_REJECTION",
                        description=(
                            f"Ignore rejections from already-slashed voters "
                            f"({', '.join(sorted(v.voter_id for v in slashed_rejecters))})."
                        ),
                        target=None,
                        projected_aggregate=proj_agg,
                        projected_committed=committed,
                        projected_unrefuted_rejection=proj_unref,
                        delta_p_commit=delta,
                        cost_band="low",
                        reversibility="reversible",
                        priority=self._priority(committed, delta, cost="low"),
                        notes=[f"voters_demoted={len(slashed_rejecters)}"],
                    )
                )

        # 3) IGNORE_VOTER(voter_id) - one per dissenter
        for v in rr.votes:
            if not v.is_rejection:
                continue
            proj_agg, proj_unref = self._recompute(
                rr=rr,
                reputation=reputation,
                drop_voters={v.voter_id},
            )
            committed = (proj_agg >= rr.threshold) and not proj_unref
            delta = 1.0 if committed else self._partial_delta(rr, proj_agg)
            # IGNORE single voter at rep=1 is medium cost; if rep<1 it's low (already slashed)
            rep_v = self._rep(v.voter_id, reputation)
            cost = "low" if rep_v < 1.0 else "medium"
            candidates.append(
                InterventionOutcome(
                    intervention=f"IGNORE_VOTER({v.voter_id})",
                    description=(
                        f"Suppress dissenter '{v.voter_id}' (rep={rep_v:.2f}, "
                        f"weight={v.weight:+.2f}) for this round."
                    ),
                    target=v.voter_id,
                    projected_aggregate=proj_agg,
                    projected_committed=committed,
                    projected_unrefuted_rejection=proj_unref,
                    delta_p_commit=delta,
                    cost_band=cost,
                    reversibility="reversible",
                    priority=self._priority(committed, delta, cost=cost),
                    notes=[f"voter_rep={rep_v:.2f}"],
                )
            )

        # 4) SWAP_LEADER_TO_STRONGEST_DISSENTER
        rejecters = [v for v in rr.votes if v.is_rejection]
        if rejecters:
            strongest = max(rejecters, key=lambda v: abs(v.weight))
            # Heuristic: new leader contributes |weight|*rep as confidence*rep.
            new_leader_contrib = abs(strongest.weight) * self._rep(
                strongest.voter_id, reputation
            )
            # Remove the original leader's vote contribution from sum (if any),
            # remove the new leader's negative vote, and add new leader confidence.
            # We don't have the original leader's contribution split, so reconstruct.
            # original: aggregate = leader_conf*leader_rep + sum(vote.weight*rep)
            # we approximate leader_conf*leader_rep as (aggregate - sum_votes).
            sum_votes = sum(
                v.weight * self._rep(v.voter_id, reputation) for v in rr.votes
            )
            original_leader_contrib = rr.aggregate_weight - sum_votes
            # New aggregate: remove original leader contrib, remove strongest's
            # vote, add new leader contrib.
            proj_agg = (
                rr.aggregate_weight
                - original_leader_contrib
                - (strongest.weight * self._rep(strongest.voter_id, reputation))
                + new_leader_contrib
            )
            # Unrefuted-rejection check: drop strongest's vote
            _, proj_unref = self._recompute(
                rr=rr,
                reputation=reputation,
                drop_voters={strongest.voter_id},
            )
            committed = (proj_agg >= rr.threshold) and not proj_unref
            delta = 1.0 if committed else self._partial_delta(rr, proj_agg)
            candidates.append(
                InterventionOutcome(
                    intervention="SWAP_LEADER_TO_STRONGEST_DISSENTER",
                    description=(
                        f"Promote dissenter '{strongest.voter_id}' to leader "
                        f"(would happen on next view-change anyway)."
                    ),
                    target=strongest.voter_id,
                    projected_aggregate=proj_agg,
                    projected_committed=committed,
                    projected_unrefuted_rejection=proj_unref,
                    delta_p_commit=delta,
                    cost_band="medium",
                    reversibility="reversible",
                    priority=self._priority(committed, delta, cost="medium"),
                    notes=["requires_next_view=true"],
                )
            )

        # 5) ADD_REDUNDANT_AGENT - BELOW_THRESHOLD, no unrefuted rejection, within margin
        if (
            not unrefuted_now
            and rr.aggregate_weight < rr.threshold
            and (rr.threshold - rr.aggregate_weight) <= self.close_margin
        ):
            # hypothetical: confidence=0.7, weight=+0.7, rep=1.0
            extra = 0.7 * 1.0 + 0.7 * 1.0
            proj_agg = rr.aggregate_weight + extra
            committed = proj_agg >= rr.threshold
            delta = 1.0 if committed else self._partial_delta(rr, proj_agg)
            candidates.append(
                InterventionOutcome(
                    intervention="ADD_REDUNDANT_AGENT",
                    description=(
                        f"Add a hypothetical agent (rep=1.0, confidence=0.7, "
                        f"vote=+0.7) to close the {rr.threshold - rr.aggregate_weight:.3f} "
                        f"margin."
                    ),
                    target=None,
                    projected_aggregate=proj_agg,
                    projected_committed=committed,
                    projected_unrefuted_rejection=False,
                    delta_p_commit=delta,
                    cost_band="high",  # operational cost: new agent
                    reversibility="reversible",
                    priority=self._priority(committed, delta, cost="high"),
                    notes=["hypothetical_agent=true"],
                )
            )

        # Sort: priority asc (P0 first), then delta_p_commit desc, then cost (low<med<high)
        cost_rank = {"low": 0, "medium": 1, "high": 2}
        candidates.sort(
            key=lambda c: (
                _PRIORITY_RANK.get(c.priority, 9),
                -c.delta_p_commit,
                cost_rank.get(c.cost_band, 9),
                c.intervention,
            )
        )
        top = candidates[: self.max_interventions_per_round]

        # recommended: cheapest P0 if any (cost_band rank), else best P1
        recommended: Optional[InterventionOutcome] = None
        p0s = [c for c in top if c.priority == "P0"]
        if p0s:
            recommended = min(
                p0s, key=lambda c: (cost_rank.get(c.cost_band, 9), c.intervention)
            )
        else:
            p1s = [c for c in top if c.priority == "P1" and c.projected_committed]
            if p1s:
                recommended = min(
                    p1s, key=lambda c: (cost_rank.get(c.cost_band, 9), c.intervention)
                )

        return RoundReplay(
            round_index=rr.round_index,
            original_committed=rr.committed,
            original_aggregate=rr.aggregate_weight,
            original_threshold=rr.threshold,
            original_blocker=blocker,
            top_interventions=top,
            recommended=recommended,
        )

    # ----- helpers -----
    @staticmethod
    def _rep(voter_id: str, reputation: Mapping[str, float]) -> float:
        return float(reputation.get(voter_id, 1.0))

    def _has_unrefuted_rejection(
        self,
        votes: Sequence[Vote],
        reputation: Mapping[str, float],
    ) -> bool:
        return any(v.is_rejection and self._rep(v.voter_id, reputation) >= 1.0 for v in votes)

    def _recompute(
        self,
        rr: RoundResult,
        reputation: Mapping[str, float],
        drop_voters: set,
    ) -> Tuple[float, bool]:
        """Recompute aggregate and unrefuted-rejection flag when some voters
        are dropped. Holds leader contribution constant (approximated as
        original_aggregate - sum_votes).
        """
        sum_votes_all = sum(
            v.weight * self._rep(v.voter_id, reputation) for v in rr.votes
        )
        leader_contrib = rr.aggregate_weight - sum_votes_all
        sum_votes_kept = sum(
            v.weight * self._rep(v.voter_id, reputation)
            for v in rr.votes
            if v.voter_id not in drop_voters
        )
        proj_agg = leader_contrib + sum_votes_kept
        proj_unref = any(
            v.is_rejection
            and self._rep(v.voter_id, reputation) >= 1.0
            and v.voter_id not in drop_voters
            for v in rr.votes
        )
        return proj_agg, proj_unref

    def _classify_blocker(
        self, rr: RoundResult, reputation: Mapping[str, float]
    ) -> str:
        if rr.committed:
            return "NONE"
        if self._has_unrefuted_rejection(rr.votes, reputation):
            return "UNREFUTED_REJECTION"
        return "BELOW_THRESHOLD"

    def _partial_delta(self, rr: RoundResult, proj_agg: float) -> float:
        """0.5 if intervention closed >=50% of shortfall, 0 if neutral,
        negative if it widened the gap."""
        shortfall = rr.threshold - rr.aggregate_weight
        if shortfall <= 0:
            return 0.0
        closed = proj_agg - rr.aggregate_weight
        ratio = closed / shortfall
        if ratio >= 0.5:
            return 0.5
        if ratio <= -0.25:
            return -0.25
        return 0.0

    def _priority(self, committed: bool, delta: float, cost: str) -> str:
        if committed and cost == "low":
            return "P0"
        if committed and cost == "medium":
            return "P0"
        if committed:  # high cost
            return "P1"
        if delta >= 0.5:
            return "P2"
        if delta < 0:
            return "P3"
        return "P3"

    # ----- cross-round patterns -----
    def _cross_round_playbook(
        self, per_round: List[RoundReplay]
    ) -> List[PlaybookItem]:
        items: List[PlaybookItem] = []
        if not per_round:
            return items

        n = len(per_round)

        # REPEAT_BLOCKER_VOTER: same voter id flips >=2 rounds via IGNORE_VOTER
        voter_flips: Counter = Counter()
        for r in per_round:
            for iv in r.top_interventions:
                if (
                    iv.intervention.startswith("IGNORE_VOTER(")
                    and iv.projected_committed
                    and iv.target is not None
                ):
                    voter_flips[iv.target] += 1
        for voter, count in sorted(voter_flips.items()):
            if count >= 2:
                items.append(
                    PlaybookItem(
                        pattern="REPEAT_BLOCKER_VOTER",
                        priority="P0",
                        detail=(
                            f"Voter '{voter}' unblocks {count} failed rounds when "
                            f"ignored - investigate or suspend."
                        ),
                        target=voter,
                    )
                )

        # SYSTEMATIC_THRESHOLD_TOO_HIGH: >=50% of failures flip via LOWER_THRESHOLD
        lt_flips = [
            r for r in per_round
            if any(
                iv.intervention == "LOWER_THRESHOLD_TO_AGGREGATE"
                and iv.projected_committed
                for iv in r.top_interventions
            )
        ]
        if n >= 2 and len(lt_flips) / n >= 0.5:
            aggs = sorted(
                next(
                    iv.projected_aggregate
                    for iv in r.top_interventions
                    if iv.intervention == "LOWER_THRESHOLD_TO_AGGREGATE"
                )
                for r in lt_flips
            )
            median = aggs[len(aggs) // 2]
            items.append(
                PlaybookItem(
                    pattern="SYSTEMATIC_THRESHOLD_TOO_HIGH",
                    priority="P1",
                    detail=(
                        f"{len(lt_flips)}/{n} failures would commit if threshold "
                        f"were lowered. Consider calibrating threshold downward."
                    ),
                    suggested_value=median,
                )
            )

        # SLASH_RECOVERY_NEEDED: >=2 rounds DEMOTE_SLASHED_REJECTION flipped
        slash_flips = sum(
            1
            for r in per_round
            for iv in r.top_interventions
            if iv.intervention == "DEMOTE_SLASHED_REJECTION"
            and iv.projected_committed
        )
        if slash_flips >= 2:
            items.append(
                PlaybookItem(
                    pattern="SLASH_RECOVERY_NEEDED",
                    priority="P1",
                    detail=(
                        f"{slash_flips} rounds blocked by rejections from already-"
                        f"slashed voters. Consider shortening slash horizon or "
                        f"auto-demoting slashed-voter rejections."
                    ),
                )
            )

        # LEADER_BENCH_THIN: swap_leader never closes margin (delta<=0 in every appearance) and appears >=2
        swap_appearances = [
            iv
            for r in per_round
            for iv in r.top_interventions
            if iv.intervention == "SWAP_LEADER_TO_STRONGEST_DISSENTER"
        ]
        if len(swap_appearances) >= 2 and all(
            iv.delta_p_commit <= 0 for iv in swap_appearances
        ):
            items.append(
                PlaybookItem(
                    pattern="LEADER_BENCH_THIN",
                    priority="P2",
                    detail=(
                        "Swapping leader to the strongest dissenter never improves "
                        "the projected margin - the bench is thin. Recruit more "
                        "diverse, well-calibrated agents."
                    ),
                )
            )

        # Stable, deterministic ordering: priority asc, then pattern, then target
        items.sort(
            key=lambda i: (
                _PRIORITY_RANK.get(i.priority, 9),
                i.pattern,
                i.target or "",
            )
        )
        return items

    def _grade(self, rounds_replayed: int, flippable: int) -> str:
        if rounds_replayed == 0:
            return "A"
        ratio = flippable / rounds_replayed
        if ratio >= 0.9:
            return "A"
        if ratio >= 0.75:
            return "B"
        if ratio >= 0.5:
            return "C"
        if ratio >= 0.25:
            return "D"
        return "F"
