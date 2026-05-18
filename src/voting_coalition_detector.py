"""Voting coalition detector.

An *agentic* coalition / voting-bloc analyzer for the mBFT consensus
engine. Where :class:`~src.swarm_health.SwarmHealthMonitor` answers
"is the swarm OK?" in aggregate,
:class:`~src.disagreement_forensics.DisagreementForensics` answers
"why did *this round* fail?",
:class:`~src.round_replay_advisor.RoundReplayAdvisor` answers
"what would have flipped that round?", and
:class:`~src.agent_lifecycle_advisor.AgentLifecycleAdvisor` answers
"who should I keep, demote, or evict?",
this advisor answers a different, fleet-correlation question:

    *Which agents move as a pack? Are those packs healthy "natural
    affinity" between calibrated peers, or are they echo chambers,
    blocking blocs, or a dominant faction that can rubber-stamp
    anything they want?*

For every pair of agents, we compute a Jaccard-style ``agreement_score``
over the rounds where both voted on the same proposal. Pairs whose
agreement clears ``cohesion_threshold`` get bonded; bonded pairs are
then merged with a simple union-find into coalitions. Each coalition
gets:

* a ``cohesion`` value (mean pairwise agreement among members),
* a ``commit_alignment`` (rounds they collectively backed the
  committed solution / rounds they collectively voted),
* a ``rejection_alignment`` (rounds they jointly rejected),
* a ``control_fraction`` (combined positive weight / total aggregate
  weight, averaged over rounds they co-voted in),
* a ``leader_capture_rate`` (rounds where a coalition member was the
  leader),
* a verdict on the ladder
  ``BENIGN_AFFINITY -> ECHO_CHAMBER -> KINGMAKER -> BLOCKING_BLOC ->
  DOMINANT_FACTION``,
* a P0-P3 priority bucket and structured ``reasons``.

A cross-swarm playbook then collapses individual coalition verdicts
into a small set of deduped patterns
(``ROTATE_LEADERS``, ``DIVERSIFY_SWARM``, ``RAISE_THRESHOLD``,
``INVESTIGATE_BLOCKING``, ``SPLIT_DOMINANT_FACTION``,
``AUDIT_KINGMAKER``, ``HEALTHY_FLEET``), each with owner +
blast_radius + reversibility.

Design notes:
* zero new dependencies (``pydantic`` only, matching the rest of mBFT);
* deterministic and stateless - never mutates the engine, history, or
  reputation map;
* never mutates inputs; ``risk_appetite`` modulates thresholds only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.core.state import RoundResult, Vote


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


class Coalition(BaseModel):
    coalition_id: str
    members: List[str]
    member_count: int
    cohesion: float  # mean pairwise Jaccard agreement (0..1)
    rounds_active: int  # rounds where >=2 members co-voted on same proposal
    commit_alignment: float  # 0..1, how often joint vote matched committed
    rejection_alignment: float  # 0..1, fraction of joint actions that were rejections
    control_fraction: float  # 0..1, mean (sum positive member weight / aggregate)
    leader_capture_rate: float  # 0..1, rounds with a member as leader
    avg_member_reputation: float
    verdict: str  # BENIGN_AFFINITY | ECHO_CHAMBER | KINGMAKER | BLOCKING_BLOC | DOMINANT_FACTION
    priority: str  # P0|P1|P2|P3
    risk_score: float  # 0..100
    reasons: List[str] = Field(default_factory=list)


class PlaybookItem(BaseModel):
    pattern: str
    priority: str
    label: str
    targets: List[str] = Field(default_factory=list)
    coalition_ids: List[str] = Field(default_factory=list)
    reason: str
    owner: str
    blast_radius: int  # 1..5
    reversibility: str  # low|medium|high
    suggested_value: Optional[float] = None


class CoalitionReport(BaseModel):
    generated_at: str
    rounds_observed: int
    rounds_committed: int
    commit_rate: float
    threshold: float
    agent_count: int
    cohesion_threshold: float
    risk_appetite: str
    coalitions: List[Coalition] = Field(default_factory=list)
    playbook: List[PlaybookItem] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    overall_grade: str = "A"  # A..F
    summary: str = ""

    # -- exporters ----------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent, sort_keys=True, default=str)

    def to_text(self) -> str:
        out: List[str] = []
        out.append("=" * 60)
        out.append("VOTING COALITION REPORT")
        out.append("=" * 60)
        out.append(self.summary)
        out.append(
            f"rounds={self.rounds_observed} commits={self.rounds_committed} "
            f"rate={self.commit_rate:.0%} grade={self.overall_grade}"
        )
        out.append(
            f"agents={self.agent_count} cohesion>={self.cohesion_threshold:.2f} "
            f"risk={self.risk_appetite}"
        )
        out.append("")
        out.append("coalitions:")
        if not self.coalitions:
            out.append("  (none detected)")
        else:
            for c in self.coalitions:
                out.append(
                    f"  [{c.priority}] {c.coalition_id} {c.verdict} "
                    f"members={','.join(c.members)} "
                    f"cohesion={c.cohesion:.2f} control={c.control_fraction:.2f} "
                    f"leader_capture={c.leader_capture_rate:.0%} "
                    f"commit_align={c.commit_alignment:.0%} risk={c.risk_score:.0f}"
                )
                if c.reasons:
                    out.append(f"      reasons: {', '.join(c.reasons)}")
        out.append("")
        out.append("playbook:")
        if not self.playbook:
            out.append("  (no actions)")
        else:
            for p in self.playbook:
                out.append(
                    f"  [{p.priority}] {p.pattern} owner={p.owner} "
                    f"blast={p.blast_radius} rev={p.reversibility}"
                )
                out.append(f"      {p.label}: {p.reason}")
                if p.targets:
                    out.append(f"      targets: {', '.join(p.targets)}")
        out.append("")
        out.append("insights:")
        if not self.insights:
            out.append("  (none)")
        else:
            for s in self.insights:
                out.append(f"  - {s}")
        out.append("=" * 60)
        return "\n".join(out)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# Voting Coalition Report")
        lines.append("")
        lines.append(f"_{self.summary}_")
        lines.append("")
        lines.append(
            f"- Rounds observed: **{self.rounds_observed}** "
            f"(committed {self.rounds_committed}, rate {self.commit_rate:.0%})"
        )
        lines.append(f"- Agents tracked: **{self.agent_count}**")
        lines.append(
            f"- Cohesion threshold: **{self.cohesion_threshold:.2f}**, "
            f"risk_appetite: **{self.risk_appetite}**"
        )
        lines.append(f"- Overall grade: **{self.overall_grade}**")
        lines.append("")
        lines.append("## Coalitions")
        lines.append("")
        if not self.coalitions:
            lines.append("_None detected — no significant voting blocs._")
        else:
            lines.append(
                "| id | verdict | priority | members | cohesion | control | "
                "leader capture | commit align | risk |"
            )
            lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
            for c in self.coalitions:
                lines.append(
                    f"| {c.coalition_id} | {c.verdict} | {c.priority} | "
                    f"{', '.join(c.members)} | {c.cohesion:.2f} | "
                    f"{c.control_fraction:.2f} | "
                    f"{c.leader_capture_rate:.0%} | "
                    f"{c.commit_alignment:.0%} | {c.risk_score:.0f} |"
                )
            lines.append("")
            for c in self.coalitions:
                if c.reasons:
                    lines.append(
                        f"- **{c.coalition_id}** reasons: "
                        f"{', '.join(c.reasons)}"
                    )
            lines.append("")
        lines.append("## Playbook")
        lines.append("")
        if not self.playbook:
            lines.append("_No actions recommended._")
        else:
            for p in self.playbook:
                lines.append(
                    f"- **[{p.priority}] {p.pattern}** "
                    f"(owner={p.owner}, blast={p.blast_radius}, "
                    f"reversibility={p.reversibility}) — {p.label}: "
                    f"{p.reason}"
                )
                if p.targets:
                    lines.append(f"    - targets: {', '.join(p.targets)}")
        lines.append("")
        lines.append("## Insights")
        lines.append("")
        if not self.insights:
            lines.append("_None — coalition structure looks unremarkable._")
        else:
            for s in self.insights:
                lines.append(f"- {s}")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


_VERDICT_RISK = {
    "BENIGN_AFFINITY": 10.0,
    "ECHO_CHAMBER": 45.0,
    "KINGMAKER": 60.0,
    "BLOCKING_BLOC": 75.0,
    "DOMINANT_FACTION": 90.0,
}


class VotingCoalitionDetector:
    """Detect persistent voting blocs across a slice of round history.

    Typical usage::

        report = VotingCoalitionDetector().analyze(
            history=engine.history,
            reputation=engine.reputation,
            threshold=engine.threshold,
            agent_ids=[a.id for a in engine.agents],
            risk_appetite="balanced",
        )
        print(report.to_markdown())
    """

    def __init__(
        self,
        *,
        cohesion_threshold: float = 0.85,
        min_co_votes: int = 3,
        dominant_control_fraction: float = 0.6,
        kingmaker_leader_rate: float = 0.5,
        blocking_rejection_alignment: float = 0.5,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if not 0.0 < cohesion_threshold <= 1.0:
            raise ValueError("cohesion_threshold must be in (0, 1]")
        if min_co_votes < 1:
            raise ValueError("min_co_votes must be >= 1")
        self.cohesion_threshold = cohesion_threshold
        self.min_co_votes = min_co_votes
        self.dominant_control_fraction = dominant_control_fraction
        self.kingmaker_leader_rate = kingmaker_leader_rate
        self.blocking_rejection_alignment = blocking_rejection_alignment
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    # -- public API ---------------------------------------------------------

    def analyze(
        self,
        *,
        history: Sequence[RoundResult],
        reputation: Optional[Mapping[str, float]] = None,
        threshold: float = 0.0,
        agent_ids: Optional[Iterable[str]] = None,
        risk_appetite: str = "balanced",
    ) -> CoalitionReport:
        if risk_appetite not in ("cautious", "balanced", "aggressive"):
            raise ValueError(
                "risk_appetite must be 'cautious', 'balanced', or 'aggressive'"
            )
        reputation = dict(reputation or {})
        ids: List[str] = list(agent_ids) if agent_ids is not None else []
        # Augment with anyone seen in history.
        for r in history:
            if r.leader_id not in ids:
                ids.append(r.leader_id)
            for v in r.votes:
                if v.voter_id not in ids:
                    ids.append(v.voter_id)
        for aid in reputation:
            if aid not in ids:
                ids.append(aid)
        ids = sorted(set(ids))

        # Apply appetite to thresholds (cautious flags more, aggressive flags fewer).
        cohesion_t, control_t, leader_t, rej_align_t = self._adjusted_thresholds(
            risk_appetite
        )

        # ---- Build per-round voter snapshots ---------------------------
        # For each round, record map voter -> sign (+1 / -1 / 0) of vote weight.
        # Sign over the *leader proposal* — votes target the leader proposal.
        round_records: List[Dict[str, object]] = []
        for r in history:
            sign_map: Dict[str, int] = {}
            weight_map: Dict[str, float] = {}
            for v in r.votes:
                sign_map[v.voter_id] = (
                    1 if v.weight > 0 else (-1 if v.weight < 0 else 0)
                )
                weight_map[v.voter_id] = float(v.weight)
            round_records.append(
                {
                    "signs": sign_map,
                    "weights": weight_map,
                    "leader_id": r.leader_id,
                    "aggregate_weight": float(r.aggregate_weight),
                    "committed": bool(r.committed),
                }
            )

        # ---- Pairwise agreement ---------------------------------------
        # agreement[(a,b)] = (matching_signs, co_votes)
        pair_match: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_total: Dict[Tuple[str, str], int] = defaultdict(int)

        for rec in round_records:
            sign_map: Dict[str, int] = rec["signs"]  # type: ignore[assignment]
            voters = sorted(sign_map)
            for i in range(len(voters)):
                for j in range(i + 1, len(voters)):
                    a, b = voters[i], voters[j]
                    pair_total[(a, b)] += 1
                    if sign_map[a] == sign_map[b] and sign_map[a] != 0:
                        pair_match[(a, b)] += 1

        # ---- Union-find on bonded pairs -------------------------------
        parent: Dict[str, str] = {aid: aid for aid in ids}

        def _find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: str, b: str) -> None:
            ra, rb = _find(a), _find(b)
            if ra == rb:
                return
            # Stable union: smaller id becomes root for determinism.
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        bonded_pairs: List[Tuple[str, str, float]] = []
        for (a, b), total in pair_total.items():
            if total < self.min_co_votes:
                continue
            agreement = pair_match[(a, b)] / total
            if agreement >= cohesion_t:
                bonded_pairs.append((a, b, agreement))
                _union(a, b)

        # ---- Group bonded members into coalitions ---------------------
        clusters: Dict[str, List[str]] = defaultdict(list)
        bonded_members = {a for a, _, _ in bonded_pairs} | {
            b for _, b, _ in bonded_pairs
        }
        for aid in bonded_members:
            clusters[_find(aid)].append(aid)

        # ---- Build per-coalition stats --------------------------------
        coalitions: List[Coalition] = []
        for idx, (root, members) in enumerate(
            sorted(clusters.items(), key=lambda kv: kv[0])
        ):
            members = sorted(set(members))
            if len(members) < 2:
                continue
            # Mean pairwise agreement among members.
            agreements: List[float] = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    key = (members[i], members[j])
                    total = pair_total.get(key, 0)
                    if total >= self.min_co_votes:
                        agreements.append(pair_match[key] / total)
            cohesion = sum(agreements) / len(agreements) if agreements else 0.0

            # Per-round behaviour.
            rounds_active = 0
            joint_actions = 0
            joint_rejections = 0
            joint_commit_aligned = 0
            joint_commit_relevant = 0
            control_samples: List[float] = []
            leader_hits = 0
            member_set = set(members)
            for rec in round_records:
                sign_map: Dict[str, int] = rec["signs"]  # type: ignore[assignment]
                weight_map: Dict[str, float] = rec["weights"]  # type: ignore[assignment]
                present = [m for m in members if m in sign_map]
                if len(present) < 2:
                    continue
                # Joint action requires agreement on direction.
                signs = {sign_map[m] for m in present}
                if len(signs) > 1:
                    # Not acting as a block this round.
                    rounds_active += 1
                    continue
                only_sign = next(iter(signs))
                if only_sign == 0:
                    rounds_active += 1
                    continue
                rounds_active += 1
                joint_actions += 1
                if only_sign < 0:
                    joint_rejections += 1
                if rec["committed"]:
                    joint_commit_relevant += 1
                    if only_sign > 0:
                        joint_commit_aligned += 1
                # Control fraction (positive contribution this round).
                agg = rec["aggregate_weight"]  # type: ignore[assignment]
                if agg and only_sign > 0:
                    pos_sum = sum(
                        max(0.0, weight_map[m]) for m in present
                    )
                    if pos_sum > 0:
                        control_samples.append(min(1.0, pos_sum / abs(agg)))
                if rec["leader_id"] in member_set:
                    leader_hits += 1
            commit_alignment = (
                joint_commit_aligned / joint_commit_relevant
                if joint_commit_relevant
                else 0.0
            )
            rejection_alignment = (
                joint_rejections / joint_actions if joint_actions else 0.0
            )
            control_fraction = (
                sum(control_samples) / len(control_samples)
                if control_samples
                else 0.0
            )
            leader_capture_rate = (
                leader_hits / len(history) if history else 0.0
            )
            avg_rep = (
                sum(reputation.get(m, 1.0) for m in members) / len(members)
            )

            verdict, reasons = self._classify(
                cohesion=cohesion,
                control_fraction=control_fraction,
                leader_capture_rate=leader_capture_rate,
                rejection_alignment=rejection_alignment,
                commit_alignment=commit_alignment,
                joint_actions=joint_actions,
                control_t=control_t,
                leader_t=leader_t,
                rej_align_t=rej_align_t,
                appetite=risk_appetite,
            )
            risk_score = self._risk_score(verdict, control_fraction, cohesion)
            priority = self._priority(verdict, risk_appetite)

            coalitions.append(
                Coalition(
                    coalition_id=f"C{idx + 1}",
                    members=members,
                    member_count=len(members),
                    cohesion=round(cohesion, 4),
                    rounds_active=rounds_active,
                    commit_alignment=round(commit_alignment, 4),
                    rejection_alignment=round(rejection_alignment, 4),
                    control_fraction=round(control_fraction, 4),
                    leader_capture_rate=round(leader_capture_rate, 4),
                    avg_member_reputation=round(avg_rep, 4),
                    verdict=verdict,
                    priority=priority,
                    risk_score=round(risk_score, 2),
                    reasons=reasons,
                )
            )

        # Sort by priority then by risk desc then by id.
        coalitions.sort(
            key=lambda c: (_PRIORITY_RANK[c.priority], -c.risk_score, c.coalition_id)
        )

        rounds_observed = len(history)
        rounds_committed = sum(1 for r in history if r.committed)
        commit_rate = (
            rounds_committed / rounds_observed if rounds_observed else 0.0
        )

        report = CoalitionReport(
            generated_at=self._now_fn().isoformat(),
            rounds_observed=rounds_observed,
            rounds_committed=rounds_committed,
            commit_rate=round(commit_rate, 4),
            threshold=threshold,
            agent_count=len(ids),
            cohesion_threshold=cohesion_t,
            risk_appetite=risk_appetite,
            coalitions=coalitions,
        )
        report.playbook = self._build_playbook(report, risk_appetite)
        report.insights = self._insights(report)
        report.overall_grade = self._grade(report)
        report.summary = self._summary(report)
        return report

    # -- helpers ------------------------------------------------------------

    def _adjusted_thresholds(
        self, appetite: str
    ) -> Tuple[float, float, float, float]:
        cohesion_t = self.cohesion_threshold
        control_t = self.dominant_control_fraction
        leader_t = self.kingmaker_leader_rate
        rej_t = self.blocking_rejection_alignment
        if appetite == "cautious":
            cohesion_t = max(0.5, cohesion_t - 0.10)
            control_t = max(0.30, control_t - 0.10)
            leader_t = max(0.20, leader_t - 0.10)
            rej_t = max(0.20, rej_t - 0.10)
        elif appetite == "aggressive":
            cohesion_t = min(0.99, cohesion_t + 0.05)
            control_t = min(0.95, control_t + 0.10)
            leader_t = min(0.95, leader_t + 0.10)
            rej_t = min(0.95, rej_t + 0.10)
        return cohesion_t, control_t, leader_t, rej_t

    def _classify(
        self,
        *,
        cohesion: float,
        control_fraction: float,
        leader_capture_rate: float,
        rejection_alignment: float,
        commit_alignment: float,
        joint_actions: int,
        control_t: float,
        leader_t: float,
        rej_align_t: float,
        appetite: str,
    ) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        reasons.append(f"COHESION_{int(cohesion * 100)}")
        # Highest match wins (ladder, most severe first).
        if (
            control_fraction >= control_t
            and leader_capture_rate >= leader_t
        ):
            reasons.append("DOMINANT_CONTROL")
            reasons.append("LEADER_CAPTURE")
            return "DOMINANT_FACTION", reasons
        if (
            rejection_alignment >= rej_align_t
            and joint_actions >= self.min_co_votes
        ):
            reasons.append("PERSISTENT_REJECTIONS")
            return "BLOCKING_BLOC", reasons
        if leader_capture_rate >= leader_t:
            reasons.append("LEADER_CAPTURE")
            return "KINGMAKER", reasons
        # Echo chamber: high agreement but low commit alignment, not dominating.
        if (
            joint_actions >= self.min_co_votes
            and commit_alignment < 0.5
        ):
            reasons.append("LOW_COMMIT_ALIGNMENT")
            return "ECHO_CHAMBER", reasons
        if (
            joint_actions >= self.min_co_votes
            and cohesion >= 0.95
            and control_fraction < control_t
        ):
            reasons.append("HIGH_COHESION_LOW_CONTROL")
            return "ECHO_CHAMBER", reasons
        reasons.append("AFFINITY_ONLY")
        return "BENIGN_AFFINITY", reasons

    @staticmethod
    def _risk_score(verdict: str, control_fraction: float, cohesion: float) -> float:
        base = _VERDICT_RISK[verdict]
        # Nudge by control & cohesion (small additive bumps).
        return max(0.0, min(100.0, base + 10.0 * control_fraction + 5.0 * cohesion))

    @staticmethod
    def _priority(verdict: str, appetite: str) -> str:
        if verdict in ("DOMINANT_FACTION", "BLOCKING_BLOC"):
            return "P0"
        if verdict == "KINGMAKER":
            return "P1"
        if verdict == "ECHO_CHAMBER":
            # Cautious treats echo chambers as more urgent.
            return "P1" if appetite == "cautious" else "P2"
        return "P3"

    def _build_playbook(
        self, report: CoalitionReport, appetite: str
    ) -> List[PlaybookItem]:
        items: List[PlaybookItem] = []
        by_verdict: Dict[str, List[Coalition]] = defaultdict(list)
        for c in report.coalitions:
            by_verdict[c.verdict].append(c)

        if by_verdict["DOMINANT_FACTION"]:
            cs = by_verdict["DOMINANT_FACTION"]
            items.append(
                PlaybookItem(
                    pattern="SPLIT_DOMINANT_FACTION",
                    priority="P0",
                    label="Split dominant faction",
                    targets=sorted({m for c in cs for m in c.members}),
                    coalition_ids=[c.coalition_id for c in cs],
                    reason=(
                        f"{len(cs)} coalition(s) jointly hold >= "
                        f"{int(self.dominant_control_fraction * 100)}% of vote "
                        "weight while leading regularly. Add independent "
                        "agents or rotate leadership outside the bloc."
                    ),
                    owner="governance",
                    blast_radius=5,
                    reversibility="medium",
                )
            )
        if by_verdict["BLOCKING_BLOC"]:
            cs = by_verdict["BLOCKING_BLOC"]
            items.append(
                PlaybookItem(
                    pattern="INVESTIGATE_BLOCKING",
                    priority="P0",
                    label="Investigate persistent blocking bloc",
                    targets=sorted({m for c in cs for m in c.members}),
                    coalition_ids=[c.coalition_id for c in cs],
                    reason=(
                        "Coalition rejects together on most joint actions. "
                        "Audit for shared upstream signal, then consider "
                        "slashing or quarantining."
                    ),
                    owner="security",
                    blast_radius=4,
                    reversibility="medium",
                )
            )
        if by_verdict["KINGMAKER"]:
            cs = by_verdict["KINGMAKER"]
            items.append(
                PlaybookItem(
                    pattern="ROTATE_LEADERS",
                    priority="P1",
                    label="Rotate leadership outside coalition",
                    targets=sorted({m for c in cs for m in c.members}),
                    coalition_ids=[c.coalition_id for c in cs],
                    reason=(
                        "Coalition members lead a disproportionate share of "
                        "rounds. Enforce round-robin or weighted-random "
                        "leader selection to break capture."
                    ),
                    owner="ops",
                    blast_radius=3,
                    reversibility="high",
                )
            )
            items.append(
                PlaybookItem(
                    pattern="AUDIT_KINGMAKER",
                    priority="P2",
                    label="Audit kingmaker calibration",
                    targets=sorted({m for c in cs for m in c.members}),
                    coalition_ids=[c.coalition_id for c in cs],
                    reason=(
                        "Verify the kingmaker coalition's commit_alignment "
                        "with ground truth (if available); if low, treat as "
                        "ECHO_CHAMBER next pass."
                    ),
                    owner="research",
                    blast_radius=2,
                    reversibility="high",
                )
            )
        if by_verdict["ECHO_CHAMBER"]:
            cs = by_verdict["ECHO_CHAMBER"]
            items.append(
                PlaybookItem(
                    pattern="DIVERSIFY_SWARM",
                    priority="P1" if appetite == "cautious" else "P2",
                    label="Diversify swarm composition",
                    targets=sorted({m for c in cs for m in c.members}),
                    coalition_ids=[c.coalition_id for c in cs],
                    reason=(
                        "Members vote together far above chance but commit "
                        "alignment is weak. Add agents with different priors, "
                        "models, or training data to reduce correlation."
                    ),
                    owner="research",
                    blast_radius=3,
                    reversibility="high",
                )
            )
        # Cross-cutting: if combined coalitions cover > half the swarm AND
        # threshold seems easy to clear, suggest raising it.
        if report.coalitions and report.threshold > 0:
            covered = {m for c in report.coalitions for m in c.members}
            coverage = len(covered) / report.agent_count if report.agent_count else 0
            if coverage >= 0.5 and report.commit_rate >= 0.9:
                suggested = round(report.threshold * 1.15, 4)
                items.append(
                    PlaybookItem(
                        pattern="RAISE_THRESHOLD",
                        priority="P1",
                        label="Tighten consensus threshold",
                        targets=[],
                        coalition_ids=[c.coalition_id for c in report.coalitions],
                        reason=(
                            f"Coalitions cover {int(coverage * 100)}% of the "
                            "swarm and commits pass easily — tighter "
                            "consensus reduces rubber-stamp risk."
                        ),
                        owner="governance",
                        blast_radius=3,
                        reversibility="high",
                        suggested_value=suggested,
                    )
                )

        if not items and report.rounds_observed >= 3:
            items.append(
                PlaybookItem(
                    pattern="HEALTHY_FLEET",
                    priority="P3",
                    label="No coalition action required",
                    targets=[],
                    coalition_ids=[],
                    reason=(
                        "No coordinated voting blocs above cohesion threshold; "
                        "swarm independence looks healthy."
                    ),
                    owner="ops",
                    blast_radius=1,
                    reversibility="high",
                )
            )

        # Aggressive trims P2/P3 items.
        if appetite == "aggressive":
            items = [
                p for p in items if p.priority in ("P0", "P1")
            ] or items[:1]
        # Dedup by pattern (keep first occurrence by priority).
        seen = set()
        deduped: List[PlaybookItem] = []
        for p in sorted(items, key=lambda x: (_PRIORITY_RANK[x.priority], x.pattern)):
            if p.pattern in seen:
                continue
            seen.add(p.pattern)
            deduped.append(p)
        return deduped

    def _insights(self, report: CoalitionReport) -> List[str]:
        out: List[str] = []
        if not report.coalitions:
            return out
        verdicts = [c.verdict for c in report.coalitions]
        if "DOMINANT_FACTION" in verdicts:
            out.append(
                "DOMINANT_FACTION_PRESENT: a single bloc can rubber-stamp "
                "commits."
            )
        if verdicts.count("BLOCKING_BLOC") >= 1:
            out.append(
                "BLOCKING_BLOC_PRESENT: chronic joint rejections detected."
            )
        if verdicts.count("ECHO_CHAMBER") >= 2:
            out.append(
                f"MULTIPLE_ECHO_CHAMBERS: {verdicts.count('ECHO_CHAMBER')} "
                "blocs with low commit alignment."
            )
        if any(c.leader_capture_rate >= 0.5 for c in report.coalitions):
            out.append(
                "LEADER_CAPTURE: at least one coalition leads >=50% of rounds."
            )
        covered = {m for c in report.coalitions for m in c.members}
        if report.agent_count and len(covered) / report.agent_count >= 0.66:
            out.append(
                f"BROAD_BLOC_COVERAGE: coalitions cover "
                f"{int(100 * len(covered) / report.agent_count)}% of agents."
            )
        if not out:
            out.append("LOW_RISK_COALITION_STRUCTURE: blocs look benign.")
        return out

    @staticmethod
    def _grade(report: CoalitionReport) -> str:
        if not report.coalitions:
            return "A"
        worst = max(c.risk_score for c in report.coalitions)
        verdicts = [c.verdict for c in report.coalitions]
        if "DOMINANT_FACTION" in verdicts or worst >= 85:
            return "F"
        if "BLOCKING_BLOC" in verdicts or worst >= 70:
            return "D"
        if "KINGMAKER" in verdicts or worst >= 55:
            return "C"
        if "ECHO_CHAMBER" in verdicts or worst >= 35:
            return "B"
        return "A"

    @staticmethod
    def _summary(report: CoalitionReport) -> str:
        if not report.coalitions:
            return (
                f"No coalitions detected across {report.rounds_observed} "
                "rounds — agents vote independently."
            )
        counts = defaultdict(int)
        for c in report.coalitions:
            counts[c.verdict] += 1
        parts = [f"{n} {v}" for v, n in sorted(counts.items())]
        return (
            f"{len(report.coalitions)} coalition(s) detected "
            f"({', '.join(parts)}); grade {report._grade_or_default()}."
        )

    def _grade_or_default(self) -> str:  # pragma: no cover - internal helper
        return getattr(self, "overall_grade", "A")


# Patch instance helper so the summary above can reach the grade safely.
def _coalition_grade_or_default(self: CoalitionReport) -> str:
    return self.overall_grade or "A"


CoalitionReport._grade_or_default = _coalition_grade_or_default  # type: ignore[attr-defined]


__all__ = [
    "Coalition",
    "PlaybookItem",
    "CoalitionReport",
    "VotingCoalitionDetector",
]
