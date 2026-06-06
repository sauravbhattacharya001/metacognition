"""Consensus latency profiler.

Agentic advisor that analyses mBFT engine history for round-progression
bottlenecks - how quickly consensus resolves, what patterns cause stalls,
and concrete interventions to improve commit velocity.

Distinct from:
* swarm_health - per-agent calibration/reputation scoring
* vote_dispersion - how the swarm votes (variance, polarity)
* threshold_tuning_advisor - whether theta itself is well-calibrated
* disagreement_forensics - root-cause on specific disagreements

This profiler answers: "Is consensus resolving quickly enough, and if not, why?"

Detects 8 pathologies:
1. SLOW_CONVERGENCE   - engine uses >= max_rounds/2 to commit
2. SERIAL_SLASH_CASCADE - same leader slashed then retry still fails
3. LEADER_MONOPOLY    - single agent leads >= 60% of committed rounds
4. REVOLVING_DOOR     - leader changes every round without commit
5. NEAR_MISS_STALL    - aggregate within 5% of threshold but not committed
6. VETO_BOTTLENECK    - single rejector blocks multiple rounds
7. STALE_CONSENSUS    - commit rate declining over trailing window
8. INSTANT_COMMIT     - commit on round 0 every time (maybe threshold too low)

Pure stdlib + pydantic. Never mutates inputs. Deterministic.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from src.core.state import RoundResult


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LatencyFinding(BaseModel):
    """A single detected latency pathology."""

    mode: str
    severity: int = Field(ge=0, le=100)
    priority: str
    reason: str
    evidence: Dict[str, object] = Field(default_factory=dict)


class LatencyAction(BaseModel):
    """Concrete intervention to improve consensus velocity."""

    id: str
    priority: str
    label: str
    reason: str
    owner: str
    blast_radius: int = Field(ge=1, le=5)
    reversibility: str
    related_findings: List[str] = Field(default_factory=list)
    suggested_value: Optional[str] = None


class LatencyReport(BaseModel):
    """Full profiler output."""

    total_rounds: int
    committed_rounds: int
    commit_rate: float
    mean_rounds_to_commit: float
    max_rounds_to_commit: int
    total_slashes: int
    unique_leaders: int
    dominant_leader_share: float

    latency_score: int = Field(ge=0, le=100)
    grade: str
    verdict: str
    headline: str

    findings: List[LatencyFinding] = Field(default_factory=list)
    playbook: List[LatencyAction] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    generated_at: str = ""


_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
_RISK_MULT = {"cautious": 1.15, "balanced": 1.0, "aggressive": 0.85}


class ConsensusLatencyProfiler:
    """Analyse mBFT round histories for convergence speed pathologies."""

    def __init__(
        self,
        risk_appetite: str = "balanced",
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if risk_appetite not in _RISK_MULT:
            raise ValueError(f"risk_appetite must be one of {list(_RISK_MULT)}, got {risk_appetite!r}")
        self._appetite = risk_appetite
        self._mult = _RISK_MULT[risk_appetite]
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def analyze(self, history: Sequence[RoundResult], max_rounds: int = 4) -> LatencyReport:
        """Produce a latency report from engine history."""
        history = list(copy.deepcopy(history))
        runs = self._segment_runs(history)

        total_rounds = len(history)
        committed_rounds = sum(1 for r in history if r.committed)
        commit_rate = committed_rounds / max(total_rounds, 1)

        rounds_to_commit_list: List[int] = []
        for run in runs:
            for i, r in enumerate(run):
                if r.committed:
                    rounds_to_commit_list.append(i + 1)
                    break

        mean_rtc = (
            sum(rounds_to_commit_list) / len(rounds_to_commit_list)
            if rounds_to_commit_list
            else float(max_rounds)
        )
        max_rtc = max(rounds_to_commit_list) if rounds_to_commit_list else max_rounds

        total_slashes = sum(len(r.slashed) for r in history)

        leader_counts: Counter = Counter()
        committed_leader_counts: Counter = Counter()
        for r in history:
            leader_counts[r.leader_id] += 1
            if r.committed:
                committed_leader_counts[r.leader_id] += 1

        unique_leaders = len(leader_counts)
        dominant_leader = committed_leader_counts.most_common(1)
        dominant_share = (
            dominant_leader[0][1] / max(committed_rounds, 1)
            if dominant_leader else 0.0
        )

        findings: List[LatencyFinding] = []
        findings.extend(self._detect_slow_convergence(runs, max_rounds))
        findings.extend(self._detect_serial_slash_cascade(runs))
        findings.extend(self._detect_leader_monopoly(committed_leader_counts, committed_rounds))
        findings.extend(self._detect_revolving_door(runs))
        findings.extend(self._detect_near_miss_stall(history))
        findings.extend(self._detect_veto_bottleneck(history))
        findings.extend(self._detect_stale_consensus(runs))
        findings.extend(self._detect_instant_commit(runs))

        findings.sort(key=lambda f: (_PRIORITY_ORDER.get(f.priority, 9), -f.severity, f.mode))

        latency_score = self._compute_score(findings)
        grade = self._grade(latency_score, findings)
        verdict = self._verdict(grade)
        playbook = self._build_playbook(findings)
        insights = self._build_insights(findings, commit_rate, mean_rtc, total_slashes, total_rounds, runs)

        headline = (
            f"VERDICT: grade={grade} commit_rate={commit_rate:.0%} "
            f"mean_rounds_to_commit={mean_rtc:.1f} "
            f"P0={sum(1 for f in findings if f.priority == 'P0')} "
            f"P1={sum(1 for f in findings if f.priority == 'P1')}"
        )

        return LatencyReport(
            total_rounds=total_rounds,
            committed_rounds=committed_rounds,
            commit_rate=round(commit_rate, 4),
            mean_rounds_to_commit=round(mean_rtc, 2),
            max_rounds_to_commit=max_rtc,
            total_slashes=total_slashes,
            unique_leaders=unique_leaders,
            dominant_leader_share=round(dominant_share, 4),
            latency_score=latency_score,
            grade=grade,
            verdict=verdict,
            headline=headline,
            findings=findings,
            playbook=playbook,
            insights=insights,
            generated_at=self._now_fn().isoformat(),
        )

    # --- Renderers ---

    def to_text(self, report: LatencyReport) -> str:
        lines = [report.headline, "", "--- Summary ---"]
        lines.append(f"  Total rounds: {report.total_rounds}")
        lines.append(f"  Committed: {report.committed_rounds}")
        lines.append(f"  Commit rate: {report.commit_rate:.0%}")
        lines.append(f"  Mean rounds to commit: {report.mean_rounds_to_commit:.1f}")
        lines.append(f"  Total slashes: {report.total_slashes}")
        lines.append(f"  Score: {report.latency_score}/100")
        lines.append(f"  Grade: {report.grade}")
        lines.append("")
        lines.append("--- Findings ---")
        for f in report.findings:
            lines.append(f"  [{f.priority}] {f.mode} (sev={f.severity}): {f.reason}")
        lines.append("")
        lines.append("--- Playbook ---")
        for a in report.playbook:
            lines.append(f"  [{a.priority}] {a.label}: {a.reason}")
        lines.append("")
        lines.append("--- Insights ---")
        for i in report.insights:
            lines.append(f"  - {i}")
        return "\n".join(lines)

    def to_markdown(self, report: LatencyReport) -> str:
        lines = ["## Summary", "", "| Metric | Value |", "|--------|-------|"]
        lines.append(f"| Total rounds | {report.total_rounds} |")
        lines.append(f"| Committed | {report.committed_rounds} |")
        lines.append(f"| Commit rate | {report.commit_rate:.0%} |")
        lines.append(f"| Mean rounds to commit | {report.mean_rounds_to_commit:.1f} |")
        lines.append(f"| Max rounds to commit | {report.max_rounds_to_commit} |")
        lines.append(f"| Total slashes | {report.total_slashes} |")
        lines.append(f"| Dominant leader share | {report.dominant_leader_share:.0%} |")
        lines.append(f"| Score | {report.latency_score}/100 |")
        lines.append(f"| Grade | {report.grade} |")
        lines.append(f"| Verdict | {report.verdict} |")
        lines.append("")
        lines.append("## Findings")
        lines.append("")
        if report.findings:
            lines.append("| Priority | Mode | Severity | Reason |")
            lines.append("|----------|------|----------|--------|")
            for f in report.findings:
                lines.append(f"| {f.priority} | {f.mode} | {f.severity} | {f.reason.replace('|', '\\|')} |")
        else:
            lines.append("No findings.")
        lines.append("")
        lines.append("## Playbook")
        lines.append("")
        if report.playbook:
            lines.append("| Priority | Action | Owner | Blast | Reversibility | Reason |")
            lines.append("|----------|--------|-------|-------|---------------|--------|")
            for a in report.playbook:
                lines.append(f"| {a.priority} | {a.label} | {a.owner} | {a.blast_radius} | {a.reversibility} | {a.reason.replace('|', '\\|')} |")
        else:
            lines.append("No actions needed.")
        lines.append("")
        lines.append("## Insights")
        lines.append("")
        for i in report.insights:
            lines.append(f"- {i}")
        return "\n".join(lines)

    def to_json(self, report: LatencyReport) -> str:
        return json.dumps(report.model_dump(), sort_keys=True, indent=2, default=str)

    # --- Segmentation ---

    def _segment_runs(self, history: List[RoundResult]) -> List[List[RoundResult]]:
        if not history:
            return []
        runs: List[List[RoundResult]] = []
        current: List[RoundResult] = []
        for r in history:
            if r.round_index == 0 and current:
                runs.append(current)
                current = []
            current.append(r)
            if r.committed:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        return runs

    # --- Detectors ---

    def _detect_slow_convergence(self, runs: List[List[RoundResult]], max_rounds: int) -> List[LatencyFinding]:
        threshold_rounds = max(max_rounds // 2, 2)
        slow_runs = [run for run in runs if any(r.committed for r in run) and len(run) >= threshold_rounds]
        if not slow_runs:
            return []
        share = len(slow_runs) / max(len(runs), 1)
        sev = min(100, max(0, int(min(90, int(40 + 50 * share)) * self._mult)))
        priority = "P0" if sev >= 70 else "P1" if sev >= 50 else "P2"
        return [LatencyFinding(
            mode="SLOW_CONVERGENCE", severity=sev, priority=priority,
            reason=f"{len(slow_runs)}/{len(runs)} runs needed >={threshold_rounds} rounds to commit",
            evidence={"slow_run_count": len(slow_runs), "total_runs": len(runs), "threshold_rounds": threshold_rounds},
        )]

    def _detect_serial_slash_cascade(self, runs: List[List[RoundResult]]) -> List[LatencyFinding]:
        cascades = 0
        for run in runs:
            prev_slashed: set = set()
            for r in run:
                current_slashed = set(r.slashed)
                if prev_slashed & current_slashed:
                    cascades += 1
                prev_slashed = current_slashed
        if cascades == 0:
            return []
        sev = min(100, max(0, int(min(85, 50 + 10 * cascades) * self._mult)))
        priority = "P0" if sev >= 70 else "P1"
        return [LatencyFinding(
            mode="SERIAL_SLASH_CASCADE", severity=sev, priority=priority,
            reason=f"{cascades} consecutive slash cascades detected",
            evidence={"cascade_count": cascades},
        )]

    def _detect_leader_monopoly(self, committed_leader_counts: Counter, committed_rounds: int) -> List[LatencyFinding]:
        if committed_rounds < 3:
            return []
        top = committed_leader_counts.most_common(1)
        if not top:
            return []
        leader_id, count = top[0]
        share = count / committed_rounds
        if share < 0.60:
            return []
        sev = min(100, max(0, int(min(70, 30 + 60 * (share - 0.6) / 0.4) * self._mult)))
        priority = "P1" if sev >= 50 else "P2"
        return [LatencyFinding(
            mode="LEADER_MONOPOLY", severity=sev, priority=priority,
            reason=f"Agent '{leader_id}' leads {share:.0%} of committed rounds ({count}/{committed_rounds})",
            evidence={"leader_id": leader_id, "share": round(share, 3)},
        )]

    def _detect_revolving_door(self, runs: List[List[RoundResult]]) -> List[LatencyFinding]:
        revolving = 0
        for run in runs:
            if len(run) < 3:
                continue
            leaders = [r.leader_id for r in run]
            if len(set(leaders)) == len(leaders) and not any(r.committed for r in run):
                revolving += 1
        if revolving == 0:
            return []
        sev = min(100, max(0, int(min(75, 40 + 15 * revolving) * self._mult)))
        priority = "P1" if sev >= 50 else "P2"
        return [LatencyFinding(
            mode="REVOLVING_DOOR", severity=sev, priority=priority,
            reason=f"{revolving} runs have a different leader every round with no commit",
            evidence={"revolving_runs": revolving},
        )]

    def _detect_near_miss_stall(self, history: List[RoundResult]) -> List[LatencyFinding]:
        near_misses = 0
        for r in history:
            if r.committed or r.threshold <= 0:
                continue
            margin = (r.threshold - r.aggregate_weight) / r.threshold
            if 0 < margin <= 0.05:
                near_misses += 1
        if near_misses == 0:
            return []
        sev = min(100, max(0, int(min(65, 35 + 10 * near_misses) * self._mult)))
        priority = "P1" if sev >= 50 else "P2"
        return [LatencyFinding(
            mode="NEAR_MISS_STALL", severity=sev, priority=priority,
            reason=f"{near_misses} rounds fell within 5% of threshold without committing",
            evidence={"near_miss_count": near_misses},
        )]

    def _detect_veto_bottleneck(self, history: List[RoundResult]) -> List[LatencyFinding]:
        rejector_blocks: Counter = Counter()
        for r in history:
            if r.committed:
                continue
            for v in r.votes:
                if v.is_rejection:
                    rejector_blocks[v.voter_id] += 1
        chronic = [(vid, cnt) for vid, cnt in rejector_blocks.items() if cnt >= 2]
        if not chronic:
            return []
        worst_id, worst_cnt = max(chronic, key=lambda x: x[1])
        sev = min(100, max(0, int(min(80, 40 + 12 * worst_cnt) * self._mult)))
        priority = "P0" if sev >= 70 else "P1"
        return [LatencyFinding(
            mode="VETO_BOTTLENECK", severity=sev, priority=priority,
            reason=f"Agent '{worst_id}' vetoed {worst_cnt} non-committed rounds",
            evidence={"blocker_id": worst_id, "veto_count": worst_cnt, "chronic_vetters": len(chronic)},
        )]

    def _detect_stale_consensus(self, runs: List[List[RoundResult]]) -> List[LatencyFinding]:
        if len(runs) < 6:
            return []
        mid = len(runs) // 2
        rate_first = sum(1 for run in runs[:mid] if any(r.committed for r in run)) / max(mid, 1)
        rate_second = sum(1 for run in runs[mid:] if any(r.committed for r in run)) / max(len(runs) - mid, 1)
        decline = rate_first - rate_second
        if decline < 0.20:
            return []
        sev = min(100, max(0, int(min(75, 40 + 70 * decline) * self._mult)))
        priority = "P1" if sev >= 50 else "P2"
        return [LatencyFinding(
            mode="STALE_CONSENSUS", severity=sev, priority=priority,
            reason=f"Commit rate declined from {rate_first:.0%} to {rate_second:.0%}",
            evidence={"rate_first_half": round(rate_first, 3), "rate_second_half": round(rate_second, 3)},
        )]

    def _detect_instant_commit(self, runs: List[List[RoundResult]]) -> List[LatencyFinding]:
        if len(runs) < 3:
            return []
        instant = sum(1 for run in runs if len(run) == 1 and run[0].committed)
        share = instant / len(runs)
        if share < 0.90:
            return []
        sev = min(100, max(0, int(min(50, 25 + 25 * share) * self._mult)))
        priority = "P2"
        return [LatencyFinding(
            mode="INSTANT_COMMIT", severity=sev, priority=priority,
            reason=f"{instant}/{len(runs)} runs commit on the first round",
            evidence={"instant_count": instant, "share": round(share, 3)},
        )]

    # --- Scoring ---

    def _compute_score(self, findings: List[LatencyFinding]) -> int:
        if not findings:
            return 100
        weights = {"P0": 25, "P1": 15, "P2": 8, "P3": 3}
        penalty = sum(f.severity * weights.get(f.priority, 3) / 100 for f in findings)
        return max(0, min(100, int(100 - penalty)))

    def _grade(self, score: int, findings: List[LatencyFinding]) -> str:
        p0 = sum(1 for f in findings if f.priority == "P0")
        if p0 >= 2 or score < 25:
            return "F"
        if p0 >= 1 or score < 40:
            return "D"
        if score < 55:
            return "C"
        if score < 70:
            return "B"
        return "A"

    def _verdict(self, grade: str) -> str:
        return {"A": "FAST_CONSENSUS", "B": "ACCEPTABLE", "C": "SLUGGISH", "D": "SLOW", "F": "STALLED"}.get(grade, "UNKNOWN")

    # --- Playbook ---

    def _build_playbook(self, findings: List[LatencyFinding]) -> List[LatencyAction]:
        actions: List[LatencyAction] = []
        modes = {f.mode for f in findings}

        if "VETO_BOTTLENECK" in modes:
            f = next(x for x in findings if x.mode == "VETO_BOTTLENECK")
            actions.append(LatencyAction(
                id="INVESTIGATE_CHRONIC_VETTOR", priority="P0",
                label="Investigate chronic vettor",
                reason=f"Agent {f.evidence.get('blocker_id', '?')} is blocking consensus",
                owner="swarm_operator", blast_radius=3, reversibility="high",
                related_findings=["VETO_BOTTLENECK"],
            ))

        if "SERIAL_SLASH_CASCADE" in modes:
            actions.append(LatencyAction(
                id="BREAK_SLASH_CASCADE", priority="P0",
                label="Break slash cascade",
                reason="Same agent repeatedly slashed and retried wastes rounds",
                owner="swarm_operator", blast_radius=4, reversibility="medium",
                related_findings=["SERIAL_SLASH_CASCADE"],
            ))

        if "SLOW_CONVERGENCE" in modes:
            f = next(x for x in findings if x.mode == "SLOW_CONVERGENCE")
            p = "P0" if f.severity >= 70 else "P1"
            actions.append(LatencyAction(
                id="LOWER_THRESHOLD_OR_ADD_AGENTS", priority=p,
                label="Lower threshold or add aligned agents",
                reason="Runs take too many rounds to converge",
                owner="protocol_tuner", blast_radius=4, reversibility="high",
                related_findings=["SLOW_CONVERGENCE"],
                suggested_value="theta -= 0.1 or add 1 aligned agent",
            ))

        if "REVOLVING_DOOR" in modes:
            actions.append(LatencyAction(
                id="STABILISE_LEADER_ELECTION", priority="P1",
                label="Stabilise leader election",
                reason="Leadership rotates every round without resolution",
                owner="protocol_tuner", blast_radius=3, reversibility="high",
                related_findings=["REVOLVING_DOOR"],
            ))

        if "NEAR_MISS_STALL" in modes:
            actions.append(LatencyAction(
                id="NUDGE_THRESHOLD_DOWN", priority="P1",
                label="Nudge threshold down slightly",
                reason="Consensus repeatedly misses by less than 5%",
                owner="protocol_tuner", blast_radius=2, reversibility="high",
                related_findings=["NEAR_MISS_STALL"],
                suggested_value="theta -= 0.05",
            ))

        if "STALE_CONSENSUS" in modes:
            actions.append(LatencyAction(
                id="DIAGNOSE_DEGRADATION", priority="P1",
                label="Diagnose consensus degradation",
                reason="Commit rate is declining over time",
                owner="swarm_operator", blast_radius=3, reversibility="high",
                related_findings=["STALE_CONSENSUS"],
            ))

        if "LEADER_MONOPOLY" in modes:
            actions.append(LatencyAction(
                id="DIVERSIFY_LEADERSHIP", priority="P2",
                label="Diversify leader election",
                reason="Single agent dominates committed leadership",
                owner="protocol_tuner", blast_radius=2, reversibility="high",
                related_findings=["LEADER_MONOPOLY"],
            ))

        if "INSTANT_COMMIT" in modes:
            actions.append(LatencyAction(
                id="RAISE_THRESHOLD", priority="P2",
                label="Raise threshold for meaningful verification",
                reason="Everything commits instantly - verification may be rubber-stamping",
                owner="protocol_tuner", blast_radius=3, reversibility="high",
                related_findings=["INSTANT_COMMIT"],
                suggested_value="theta += 0.2",
            ))

        # Cautious adds audit
        if self._appetite == "cautious":
            grade = self._grade(self._compute_score(findings), findings)
            if grade in ("C", "D", "F"):
                actions.append(LatencyAction(
                    id="SCHEDULE_LATENCY_AUDIT", priority="P2",
                    label="Schedule latency audit",
                    reason="Cautious appetite with poor grade warrants review",
                    owner="swarm_operator", blast_radius=1, reversibility="high",
                    related_findings=[f.mode for f in findings[:3]],
                ))

        # Fallback
        if not actions:
            actions.append(LatencyAction(
                id="CONSENSUS_HEALTHY", priority="P3",
                label="No latency intervention needed",
                reason="Consensus is resolving at healthy velocity",
                owner="swarm_operator", blast_radius=1, reversibility="high",
                related_findings=[],
            ))

        # Aggressive trims
        if self._appetite == "aggressive":
            has_p0_p1 = any(a.priority in ("P0", "P1") for a in actions)
            if has_p0_p1:
                actions = [a for a in actions if a.priority != "P3"]
                p2_only = [a for a in actions if a.priority == "P2"]
                if len(p2_only) == 1:
                    actions = [a for a in actions if a.priority != "P2"]

        # Sort playbook
        actions.sort(key=lambda a: (_PRIORITY_ORDER.get(a.priority, 9), a.id))
        return actions

    # --- Insights ---

    def _build_insights(self, findings, commit_rate, mean_rtc, total_slashes, total_rounds, runs):
        insights: List[str] = []
        modes = {f.mode for f in findings}

        if "VETO_BOTTLENECK" in modes:
            insights.append("CHRONIC_VETTOR_PRESENT")
        if "SERIAL_SLASH_CASCADE" in modes:
            insights.append("SLASH_CASCADE_PATTERN")
        if "REVOLVING_DOOR" in modes:
            insights.append("LEADERSHIP_INSTABILITY")
        if "STALE_CONSENSUS" in modes:
            insights.append("CONSENSUS_DEGRADING")
        if "INSTANT_COMMIT" in modes:
            insights.append("RUBBER_STAMP_RISK")
        if commit_rate >= 0.90 and not findings:
            insights.append("HEALTHY_CONSENSUS_VELOCITY")
        if total_slashes > total_rounds:
            insights.append("HIGH_SLASH_RATE")
        if mean_rtc <= 1.2 and commit_rate >= 0.80 and "INSTANT_COMMIT" not in modes:
            insights.append("FAST_CONVERGENCE")
        if not insights:
            insights.append("NO_NOTABLE_SIGNALS")

        return sorted(insights)
