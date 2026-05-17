"""Consensus forecaster.

A *proactive* counterpart to :mod:`src.swarm_health`. Where the health
monitor looks backwards at what already happened, the forecaster looks
*forward*: given an engine's history, reputation map, and current config,
it predicts what is likely to happen in the **next** mBFT round before
the round is actually executed:

* the most likely leader and a short ranked list of contenders,
* a point estimate (plus a low/high range) for the next round's
  aggregate weight ``Σ V_i``,
* the estimated probability of commit,
* the agents most likely to cast rejections,
* a small playbook of P0/P1/P2 pre-round interventions the operator
  can apply (lower θ slightly, pause a suspect, swap in a fresh agent,
  raise the slash_factor temporarily, etc.) — each tagged with an
  expected effect on the predicted commit probability.

Why this matters: the existing monitor tells you "your swarm has been
unhealthy for the last 8 rounds." The forecaster tells you "round 9 is
also going to fail because a3 will reject a1's proposal again — here's
how to break the loop *before* you pay another slash." That is the kind
of agency the project is trending toward: detection → recommendation →
goal-oriented suggestion.

Determinism: given the same history+reputation+config, the forecaster
returns identical output. No randomness, no network.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.core.state import RoundResult, Vote


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AgentForecast(BaseModel):
    """Per-agent prediction for the upcoming round."""

    agent_id: str
    reputation: float
    avg_proposal_confidence: float  # inferred from prior leadership outcomes
    avg_vote_weight: float
    rejection_rate: float
    leader_score: float  # ~ confidence * reputation; higher == more likely leader
    p_leader: float  # softmax over leader_score, in [0,1]
    p_will_reject: float  # in [0,1], based on past behavior


class Intervention(BaseModel):
    """Concrete pre-round adjustment the operator can apply."""

    priority: str  # P0|P1|P2|P3
    kind: str  # threshold|pause_agent|swap_agent|slash_factor|add_agents|none
    message: str
    suggested_value: Optional[float] = None
    target_agent: Optional[str] = None
    expected_p_commit_delta: float = 0.0  # approximate, in [-1, 1]


class ConsensusForecast(BaseModel):
    """The forecaster's output."""

    rounds_observed: int
    threshold: float
    slash_factor: float
    predicted_leader_id: Optional[str]
    predicted_aggregate_weight: float
    aggregate_weight_low: float
    aggregate_weight_high: float
    p_commit: float
    p_unrefuted_rejection: float
    leaders_ranked: List[AgentForecast] = Field(default_factory=list)
    likely_rejectors: List[AgentForecast] = Field(default_factory=list)
    interventions: List[Intervention] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    # -- exporters ----------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# Consensus Forecast — Next Round")
        lines.append("")
        lines.append(f"- Rounds of history: **{self.rounds_observed}**")
        lines.append(
            f"- Threshold θ: **{self.threshold:.3f}**  "
            f"slash_factor: **{self.slash_factor:.3f}**"
        )
        leader = self.predicted_leader_id or "—"
        lines.append(f"- Predicted leader: **{leader}**")
        lines.append(
            f"- Predicted aggregate weight: **{self.predicted_aggregate_weight:.3f}** "
            f"(range {self.aggregate_weight_low:.3f}…{self.aggregate_weight_high:.3f})"
        )
        lines.append(
            f"- Commit probability: **{self.p_commit:.0%}**  "
            f"(P[unrefuted rejection] = {self.p_unrefuted_rejection:.0%})"
        )
        lines.append("")
        lines.append("## Leader Candidates")
        lines.append("")
        lines.append("| agent | rep | conf~ | score | p(lead) | p(reject) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for a in self.leaders_ranked:
            lines.append(
                f"| {a.agent_id} | {a.reputation:.3f} | "
                f"{a.avg_proposal_confidence:.2f} | "
                f"{a.leader_score:.3f} | {a.p_leader:.0%} | "
                f"{a.p_will_reject:.0%} |"
            )
        lines.append("")
        if self.likely_rejectors:
            lines.append("## Likely Rejectors")
            lines.append("")
            for a in self.likely_rejectors:
                lines.append(
                    f"- **{a.agent_id}** — p(reject) = {a.p_will_reject:.0%}  "
                    f"(reputation {a.reputation:.3f})"
                )
            lines.append("")
        lines.append("## Interventions")
        lines.append("")
        if not self.interventions:
            lines.append("_None — round is on track to commit._")
        else:
            for iv in self.interventions:
                tag = f"[{iv.priority}] {iv.kind}"
                tgt = f" (agent={iv.target_agent})" if iv.target_agent else ""
                sug = (
                    f" → suggested {iv.suggested_value:.3f}"
                    if iv.suggested_value is not None
                    else ""
                )
                delta = (
                    f"  Δp(commit)≈{iv.expected_p_commit_delta:+.0%}"
                    if iv.expected_p_commit_delta
                    else ""
                )
                lines.append(f"- **{tag}**{tgt}: {iv.message}{sug}{delta}")
        if self.notes:
            lines.append("")
            lines.append("## Notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
        lines.append("")
        return "\n".join(lines)

    def to_text(self) -> str:
        out: List[str] = []
        out.append("=" * 60)
        out.append("CONSENSUS FORECAST — NEXT ROUND")
        out.append("=" * 60)
        out.append(
            f"rounds_observed={self.rounds_observed}  "
            f"θ={self.threshold:.3f}  slash_factor={self.slash_factor:.3f}"
        )
        out.append(f"predicted_leader={self.predicted_leader_id or '—'}")
        out.append(
            f"predicted_aggregate={self.predicted_aggregate_weight:.3f} "
            f"[{self.aggregate_weight_low:.3f}..{self.aggregate_weight_high:.3f}]"
        )
        out.append(
            f"p_commit={self.p_commit:.0%}  "
            f"p_unrefuted_rejection={self.p_unrefuted_rejection:.0%}"
        )
        out.append("")
        out.append("leader candidates:")
        for a in self.leaders_ranked:
            out.append(
                f"  {a.agent_id:>6}  rep={a.reputation:.3f}  "
                f"conf~={a.avg_proposal_confidence:.2f}  "
                f"score={a.leader_score:.3f}  "
                f"p_lead={a.p_leader:.0%}  "
                f"p_reject={a.p_will_reject:.0%}"
            )
        if self.likely_rejectors:
            out.append("")
            out.append("likely rejectors:")
            for a in self.likely_rejectors:
                out.append(
                    f"  {a.agent_id:>6}  p_reject={a.p_will_reject:.0%}  "
                    f"rep={a.reputation:.3f}"
                )
        out.append("")
        out.append("interventions:")
        if not self.interventions:
            out.append("  (none — round is on track to commit)")
        else:
            for iv in self.interventions:
                line = f"  [{iv.priority}] {iv.kind}: {iv.message}"
                if iv.suggested_value is not None:
                    line += f" -> {iv.suggested_value:.3f}"
                if iv.expected_p_commit_delta:
                    line += f"  Δp_commit~={iv.expected_p_commit_delta:+.0%}"
                out.append(line)
        if self.notes:
            out.append("")
            out.append("notes:")
            for n in self.notes:
                out.append(f"  - {n}")
        out.append("=" * 60)
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


class ConsensusForecaster:
    """Predict the next mBFT round's outcome from prior history.

    Typical usage::

        engine = MBFTEngine(agents=..., threshold=1.5, slash_factor=0.5)
        await engine.run(task)
        forecast = ConsensusForecaster().forecast(
            history=engine.history,
            reputation=engine.reputation,
            threshold=engine.threshold,
            slash_factor=engine.slash_factor,
            agent_ids=[a.id for a in engine.agents],
        )
        print(forecast.to_markdown())

    The forecaster never mutates anything; it only reads history.
    """

    def __init__(
        self,
        *,
        recent_window: int = 5,
        confidence_prior: float = 0.6,
        rejection_smoothing: float = 1.0,
        commit_target: float = 0.7,
        unrefuted_veto_rep_threshold: float = 1.0,
    ) -> None:
        self.recent_window = max(1, int(recent_window))
        self.confidence_prior = float(confidence_prior)
        self.rejection_smoothing = float(rejection_smoothing)
        self.commit_target = float(commit_target)
        self.unrefuted_veto_rep_threshold = float(unrefuted_veto_rep_threshold)

    # -- public API ---------------------------------------------------------

    def forecast(
        self,
        *,
        history: Sequence[RoundResult],
        reputation: Mapping[str, float],
        threshold: float,
        slash_factor: float,
        agent_ids: Optional[Iterable[str]] = None,
    ) -> ConsensusForecast:
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        ids: List[str] = list(agent_ids) if agent_ids is not None else []
        for aid in reputation:
            if aid not in ids:
                ids.append(aid)
        # also pick up anyone who appears in history
        for rnd in history:
            if rnd.leader_id not in ids:
                ids.append(rnd.leader_id)
            for v in rnd.votes:
                if v.voter_id not in ids:
                    ids.append(v.voter_id)

        recent = list(history)[-self.recent_window :] if history else []

        profiles = {
            aid: self._profile(aid, history, recent, reputation) for aid in ids
        }

        # --- leader prediction (softmax over confidence * reputation) ------
        scores = {aid: p.leader_score for aid, p in profiles.items()}
        p_leader = _softmax(scores)
        for aid, prob in p_leader.items():
            profiles[aid].p_leader = prob

        if scores:
            predicted_leader_id = max(scores, key=scores.get)
        else:
            predicted_leader_id = None

        # --- aggregate weight prediction (point + range) -------------------
        if predicted_leader_id is None:
            point = low = high = 0.0
            p_commit = 0.0
            p_unrefuted = 0.0
        else:
            point, low, high = self._predict_aggregate(
                predicted_leader_id, profiles
            )
            # logistic mapping of margin -> p(commit)
            margin = point - threshold
            scale = max(0.25, threshold * 0.25)
            p_pass = _logistic(margin / scale)
            p_unrefuted = self._p_unrefuted_rejection(
                profiles, predicted_leader_id, reputation
            )
            p_commit = max(0.0, p_pass * (1.0 - p_unrefuted))

        # --- ranked leader candidates / likely rejectors -------------------
        leaders_ranked = sorted(
            profiles.values(),
            key=lambda a: (-a.leader_score, a.agent_id),
        )
        likely_rejectors = [
            a
            for a in sorted(
                profiles.values(),
                key=lambda a: (-a.p_will_reject, a.agent_id),
            )
            if a.p_will_reject >= 0.4 and a.agent_id != predicted_leader_id
        ]

        notes = self._collect_notes(history, profiles)
        interventions = self._recommend(
            profiles=profiles,
            predicted_leader_id=predicted_leader_id,
            p_commit=p_commit,
            p_unrefuted=p_unrefuted,
            point=point,
            threshold=threshold,
            slash_factor=slash_factor,
        )

        return ConsensusForecast(
            rounds_observed=len(history),
            threshold=threshold,
            slash_factor=slash_factor,
            predicted_leader_id=predicted_leader_id,
            predicted_aggregate_weight=point,
            aggregate_weight_low=low,
            aggregate_weight_high=high,
            p_commit=p_commit,
            p_unrefuted_rejection=p_unrefuted,
            leaders_ranked=leaders_ranked,
            likely_rejectors=likely_rejectors,
            interventions=interventions,
            notes=notes,
        )

    # -- helpers ------------------------------------------------------------

    def _profile(
        self,
        aid: str,
        history: Sequence[RoundResult],
        recent: Sequence[RoundResult],
        reputation: Mapping[str, float],
    ) -> AgentForecast:
        rep = float(reputation.get(aid, 1.0))

        # Infer "typical" confidence-when-leading from past leadership.
        led_rounds = [r for r in history if r.leader_id == aid]
        if led_rounds:
            # leader's own contribution to aggregate was confidence * rep_at_time.
            # We only know current rep; treat that as a lower-bound proxy.
            # Use aggregate normalised by (rep) as a coarse confidence estimate.
            est = []
            for r in led_rounds:
                # subtract other votes to recover leader's contribution
                others = sum(v.weight for v in r.votes)
                # leader_part = aggregate - sum(weight_i * rep_i); approximate
                # rep_i with 1.0 since we don't have historical rep timeline.
                approx_leader = r.aggregate_weight - others
                # leader contribution = confidence * reputation_at_time
                # divide by max(rep,1e-3) for a confidence proxy in [0,1]
                if rep > 1e-3:
                    est.append(
                        max(0.0, min(1.0, approx_leader / max(rep, 1e-3)))
                    )
            avg_conf = sum(est) / len(est) if est else self.confidence_prior
        else:
            avg_conf = self.confidence_prior

        # Vote weight history (as a follower).
        vote_weights: List[float] = []
        rejections = 0
        votes_cast = 0
        for r in history:
            for v in r.votes:
                if v.voter_id == aid:
                    vote_weights.append(v.weight)
                    votes_cast += 1
                    if v.is_rejection:
                        rejections += 1

        avg_vw = sum(vote_weights) / len(vote_weights) if vote_weights else 0.5
        # Laplace-smoothed rejection rate.
        rej_rate = (rejections + self.rejection_smoothing * 0.1) / (
            votes_cast + self.rejection_smoothing
        )
        rej_rate = max(0.0, min(1.0, rej_rate))

        # Recency bias for rejection: if recent rounds show more rejections,
        # lean on that.
        recent_rej = 0
        recent_votes = 0
        for r in recent:
            for v in r.votes:
                if v.voter_id == aid:
                    recent_votes += 1
                    if v.is_rejection:
                        recent_rej += 1
        if recent_votes > 0:
            recent_rate = recent_rej / recent_votes
            rej_rate = 0.5 * rej_rate + 0.5 * recent_rate

        leader_score = avg_conf * rep

        return AgentForecast(
            agent_id=aid,
            reputation=rep,
            avg_proposal_confidence=avg_conf,
            avg_vote_weight=avg_vw,
            rejection_rate=rej_rate,
            leader_score=leader_score,
            p_leader=0.0,  # filled in by softmax
            p_will_reject=rej_rate,
        )

    def _predict_aggregate(
        self,
        leader_id: str,
        profiles: Mapping[str, AgentForecast],
    ) -> Tuple[float, float, float]:
        leader = profiles[leader_id]
        leader_part = leader.avg_proposal_confidence * leader.reputation
        follower_part = 0.0
        spread = 0.0
        for aid, p in profiles.items():
            if aid == leader_id:
                continue
            # Expected effective vote = avg_vote_weight * reputation
            expected = p.avg_vote_weight * p.reputation
            follower_part += expected
            # variance proxy: range between (rejection -> -|w|*rep) and
            # (agreement -> +|w|*rep)
            spread += abs(p.avg_vote_weight) * p.reputation
        point = leader_part + follower_part
        low = leader_part + follower_part - 0.5 * spread
        high = leader_part + follower_part + 0.5 * spread
        return point, low, high

    def _p_unrefuted_rejection(
        self,
        profiles: Mapping[str, AgentForecast],
        leader_id: str,
        reputation: Mapping[str, float],
    ) -> float:
        # P(at least one full-reputation agent rejects) = 1 - Π(1 - p_reject_i)
        prod_not = 1.0
        any_eligible = False
        for aid, p in profiles.items():
            if aid == leader_id:
                continue
            rep = float(reputation.get(aid, 1.0))
            if rep >= self.unrefuted_veto_rep_threshold:
                any_eligible = True
                prod_not *= 1.0 - p.p_will_reject
        if not any_eligible:
            return 0.0
        return max(0.0, min(1.0, 1.0 - prod_not))

    def _collect_notes(
        self,
        history: Sequence[RoundResult],
        profiles: Mapping[str, AgentForecast],
    ) -> List[str]:
        notes: List[str] = []
        if not history:
            notes.append(
                "No prior rounds — predictions use neutral priors only."
            )
            return notes
        # streak of failures?
        tail_fail = 0
        for r in reversed(history):
            if r.committed:
                break
            tail_fail += 1
        if tail_fail >= 2:
            notes.append(
                f"Swarm has failed to commit for the last {tail_fail} rounds."
            )
        # collapsed reputations?
        collapsed = [aid for aid, p in profiles.items() if p.reputation < 0.25]
        if collapsed:
            notes.append(
                "Agents with collapsed reputation (rep<0.25): "
                + ", ".join(sorted(collapsed))
            )
        return notes

    def _recommend(
        self,
        *,
        profiles: Mapping[str, AgentForecast],
        predicted_leader_id: Optional[str],
        p_commit: float,
        p_unrefuted: float,
        point: float,
        threshold: float,
        slash_factor: float,
    ) -> List[Intervention]:
        recs: List[Intervention] = []
        if predicted_leader_id is None:
            recs.append(
                Intervention(
                    priority="P3",
                    kind="none",
                    message="No agents available — populate the swarm first.",
                )
            )
            return recs

        # On-track case: nothing critical to do.
        if p_commit >= self.commit_target and p_unrefuted < 0.25:
            recs.append(
                Intervention(
                    priority="P3",
                    kind="none",
                    message=(
                        "Forecast looks good — proceed with the round "
                        "as configured."
                    ),
                )
            )
            return recs

        # P0: lone full-reputation veto from a chronic rejector.
        chronic_vetoers = [
            p
            for aid, p in profiles.items()
            if aid != predicted_leader_id
            and p.reputation >= self.unrefuted_veto_rep_threshold
            and p.p_will_reject >= 0.6
        ]
        for v in sorted(chronic_vetoers, key=lambda x: -x.p_will_reject)[:2]:
            recs.append(
                Intervention(
                    priority="P0",
                    kind="pause_agent",
                    target_agent=v.agent_id,
                    message=(
                        f"Agent {v.agent_id} has rejected "
                        f"{v.rejection_rate:.0%} of recent rounds with "
                        "full veto power — pausing it for one round will "
                        "remove the unrefuted-rejection blocker."
                    ),
                    expected_p_commit_delta=min(0.5, v.p_will_reject * 0.6),
                )
            )

        # P1: threshold tuning when aggregate is close.
        margin = point - threshold
        scale = max(0.25, threshold * 0.25)
        if margin < 0 and margin > -scale and p_unrefuted < 0.5:
            suggested = max(0.1, threshold + margin * 0.8)
            recs.append(
                Intervention(
                    priority="P1",
                    kind="threshold",
                    message=(
                        "Predicted aggregate is just below θ — lowering "
                        "the threshold slightly would let this round "
                        "commit without changing the agent mix."
                    ),
                    suggested_value=round(suggested, 4),
                    expected_p_commit_delta=0.25,
                )
            )
        elif margin >= scale and p_commit < self.commit_target:
            # Margin is fine but rejection probability is dragging us down.
            recs.append(
                Intervention(
                    priority="P1",
                    kind="slash_factor",
                    message=(
                        "Aggregate is healthy but commit prob is low due "
                        "to rejection risk — raising slash_factor will "
                        "deter speculative rejections."
                    ),
                    suggested_value=round(min(0.95, slash_factor * 1.5), 4),
                    expected_p_commit_delta=0.1,
                )
            )

        # P1: add fresh agents if the active set is small.
        active = [p for p in profiles.values() if p.reputation >= 0.25]
        if len(active) < 4:
            recs.append(
                Intervention(
                    priority="P1",
                    kind="add_agents",
                    message=(
                        "Only "
                        f"{len(active)} agents still carry meaningful "
                        "reputation — adding fresh agents will dilute "
                        "single-point vetoes and raise commit probability."
                    ),
                    suggested_value=4,
                    expected_p_commit_delta=0.15,
                )
            )

        # P2: swap leader if the predicted leader has been slashed often.
        leader = profiles[predicted_leader_id]
        if leader.reputation < 0.5:
            # find best alternative
            alts = [
                (aid, p.leader_score)
                for aid, p in profiles.items()
                if aid != predicted_leader_id and p.reputation >= 0.5
            ]
            if alts:
                alt_id, _ = max(alts, key=lambda x: x[1])
                recs.append(
                    Intervention(
                        priority="P2",
                        kind="swap_agent",
                        target_agent=alt_id,
                        message=(
                            f"Predicted leader {predicted_leader_id} has a "
                            f"weakened reputation ({leader.reputation:.2f}); "
                            f"force-electing {alt_id} for one round may "
                            "stabilise the swarm."
                        ),
                        expected_p_commit_delta=0.15,
                    )
                )

        if not recs:
            recs.append(
                Intervention(
                    priority="P2",
                    kind="none",
                    message=(
                        "No high-confidence intervention found — "
                        "consider re-running with fresh proposals."
                    ),
                )
            )
        return recs


# ---------------------------------------------------------------------------
# Small numerical helpers (kept private to avoid a numpy dep)
# ---------------------------------------------------------------------------


def _softmax(scores: Mapping[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    # numerically stable softmax with a mild temperature so the top
    # candidate doesn't completely dominate.
    temperature = 0.5
    items = list(scores.items())
    max_s = max(s for _, s in items)
    exps = [(k, math.exp((s - max_s) / temperature)) for k, s in items]
    z = sum(e for _, e in exps) or 1.0
    return {k: e / z for k, e in exps}


def _logistic(x: float) -> float:
    # standard logistic, clipped to avoid overflow warnings.
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


__all__ = [
    "AgentForecast",
    "Intervention",
    "ConsensusForecast",
    "ConsensusForecaster",
]
