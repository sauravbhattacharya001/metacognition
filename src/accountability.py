"""Consensus Accountability Ledger -- tamper-evident audit trail for mBFT.

Maintains a hash-chained immutable record of every consensus round, enabling
third-party verification, dispute resolution, and autonomous anomaly detection
across the consensus history.

Each ledger entry contains a SHA-256 hash linking to the previous entry,
creating a blockchain-like integrity guarantee.  The auto-audit engine
detects suspicious patterns: reputation gaming, vote flipping, leader
monopolies, confidence manipulation, and collusion clusters.

Usage::

    python -m src.accountability                        # default 15-round demo
    python -m src.accountability --rounds 25 --agents 8
    python -m src.accountability --inject-tamper         # demo tamper detection
    python -m src.accountability --audit-only ledger.json  # audit existing ledger
    python -m src.accountability --export report.html
    python -m src.accountability --export ledger.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import html as html_mod
import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine
from src.core.state import RoundResult, Vote

# -- Ledger Entry ---------------------------------------------------------


@dataclass
class LedgerEntry:
    """Single immutable record in the accountability ledger."""

    index: int
    timestamp: str
    round_result: Dict[str, Any]
    prev_hash: str
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """SHA-256 of the canonical entry content."""
        payload = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "round_result": self.round_result,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def __post_init__(self) -> None:
        if not self.entry_hash:
            self.entry_hash = self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "round_result": self.round_result,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


# -- Accountability Ledger ------------------------------------------------


class AccountabilityLedger:
    """Hash-chained audit trail for consensus rounds."""

    GENESIS_HASH = "0" * 64

    def __init__(self) -> None:
        self.entries: List[LedgerEntry] = []

    def append(self, round_result: RoundResult) -> LedgerEntry:
        """Add a round result to the ledger with hash-chain linking."""
        prev_hash = (
            self.entries[-1].entry_hash if self.entries else self.GENESIS_HASH
        )
        rr_dict = {
            "round_index": round_result.round_index,
            "leader_id": round_result.leader_id,
            "committed": round_result.committed,
            "committed_solution": round_result.committed_solution,
            "aggregate_weight": round_result.aggregate_weight,
            "threshold": round_result.threshold,
            "votes": [
                {
                    "voter_id": v.voter_id,
                    "weight": v.weight,
                    "is_rejection": v.is_rejection,
                    "has_counter_proof": v.counter_proof is not None,
                }
                for v in round_result.votes
            ],
            "slashed": round_result.slashed,
        }
        entry = LedgerEntry(
            index=len(self.entries),
            timestamp=datetime.now(timezone.utc).isoformat(),
            round_result=rr_dict,
            prev_hash=prev_hash,
        )
        self.entries.append(entry)
        return entry

    def verify_integrity(self) -> List[Dict[str, Any]]:
        """Verify the full hash chain. Returns list of violations."""
        violations: List[Dict[str, Any]] = []
        for i, entry in enumerate(self.entries):
            # Check prev_hash link
            expected_prev = (
                self.entries[i - 1].entry_hash
                if i > 0
                else self.GENESIS_HASH
            )
            if entry.prev_hash != expected_prev:
                violations.append(
                    {
                        "type": "broken_chain",
                        "index": i,
                        "expected_prev": expected_prev,
                        "actual_prev": entry.prev_hash,
                        "severity": "CRITICAL",
                    }
                )
            # Check self-hash
            recomputed = entry.compute_hash()
            if entry.entry_hash != recomputed:
                violations.append(
                    {
                        "type": "tampered_entry",
                        "index": i,
                        "stored_hash": entry.entry_hash,
                        "recomputed_hash": recomputed,
                        "severity": "CRITICAL",
                    }
                )
        return violations

    def export_json(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_json(cls, data: List[Dict[str, Any]]) -> "AccountabilityLedger":
        """Deserialise a previously-exported ledger.

        Validates the incoming structure defensively so that a malformed or
        attacker-crafted JSON file cannot crash the audit pipeline or inject
        unexpected types into downstream analysis.
        """
        if not isinstance(data, list):
            raise ValueError("Ledger JSON must be a list of entry objects.")
        ledger = cls()
        required_keys = {"index", "timestamp", "round_result", "prev_hash", "entry_hash"}
        for i, d in enumerate(data):
            if not isinstance(d, dict):
                raise ValueError(f"Ledger entry {i} is not an object.")
            missing = required_keys - d.keys()
            if missing:
                raise ValueError(f"Ledger entry {i} missing keys: {missing}")
            if not isinstance(d["index"], int):
                raise ValueError(f"Ledger entry {i}: 'index' must be an integer.")
            if not isinstance(d["round_result"], dict):
                raise ValueError(f"Ledger entry {i}: 'round_result' must be an object.")
            entry = LedgerEntry(
                index=d["index"],
                timestamp=str(d["timestamp"]),
                round_result=d["round_result"],
                prev_hash=str(d["prev_hash"]),
                entry_hash=str(d["entry_hash"]),
            )
            ledger.entries.append(entry)
        return ledger


# -- Anomaly Findings -----------------------------------------------------

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}


@dataclass
class AuditFinding:
    category: str
    severity: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence,
        }


# -- Auto-Audit Engine ---------------------------------------------------


class AuditEngine:
    """Autonomous anomaly detection across ledger history."""

    def __init__(self, ledger: AccountabilityLedger) -> None:
        self.ledger = ledger
        self.findings: List[AuditFinding] = []

    def run_full_audit(self) -> List[AuditFinding]:
        self.findings = []
        self._check_integrity()
        self._check_leader_monopoly()
        self._check_vote_flipping()
        self._check_confidence_manipulation()
        self._check_slash_patterns()
        self._check_collusion_clusters()
        self._check_commit_rate()
        self.findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 99))
        return self.findings

    def _check_integrity(self) -> None:
        violations = self.ledger.verify_integrity()
        for v in violations:
            self.findings.append(
                AuditFinding(
                    category="Integrity",
                    severity="CRITICAL",
                    description=f"Hash chain violation at entry {v['index']}: {v['type']}",
                    evidence=v,
                )
            )
        if not violations:
            self.findings.append(
                AuditFinding(
                    category="Integrity",
                    severity="INFO",
                    description=f"Hash chain intact across {len(self.ledger.entries)} entries",
                )
            )

    def _check_leader_monopoly(self) -> None:
        if not self.ledger.entries:
            return
        leader_counts: Dict[str, int] = {}
        for e in self.ledger.entries:
            lid = e.round_result["leader_id"]
            leader_counts[lid] = leader_counts.get(lid, 0) + 1
        total = len(self.ledger.entries)
        for agent, count in leader_counts.items():
            ratio = count / total
            if ratio > 0.5:
                self.findings.append(
                    AuditFinding(
                        category="Leader Monopoly",
                        severity="HIGH",
                        description=f"Agent {agent} led {ratio:.0%} of rounds ({count}/{total})",
                        evidence={"agent": agent, "count": count, "ratio": ratio},
                    )
                )
            elif ratio > 0.35:
                self.findings.append(
                    AuditFinding(
                        category="Leader Dominance",
                        severity="MEDIUM",
                        description=f"Agent {agent} led {ratio:.0%} of rounds ({count}/{total})",
                        evidence={"agent": agent, "count": count, "ratio": ratio},
                    )
                )

    def _check_vote_flipping(self) -> None:
        """Detect agents who frequently change vote direction."""
        if len(self.ledger.entries) < 3:
            return
        voter_history: Dict[str, List[float]] = {}
        for e in self.ledger.entries:
            for v in e.round_result.get("votes", []):
                vid = v["voter_id"]
                voter_history.setdefault(vid, []).append(v["weight"])
        for agent, weights in voter_history.items():
            if len(weights) < 3:
                continue
            flips = sum(
                1
                for i in range(1, len(weights))
                if (weights[i] > 0) != (weights[i - 1] > 0)
            )
            flip_rate = flips / (len(weights) - 1)
            if flip_rate > 0.6:
                self.findings.append(
                    AuditFinding(
                        category="Vote Flipping",
                        severity="HIGH",
                        description=f"Agent {agent} flipped vote direction {flip_rate:.0%} of the time ({flips} flips in {len(weights)} votes)",
                        evidence={
                            "agent": agent,
                            "flip_rate": flip_rate,
                            "flips": flips,
                            "total_votes": len(weights),
                        },
                    )
                )
            elif flip_rate > 0.4:
                self.findings.append(
                    AuditFinding(
                        category="Vote Instability",
                        severity="MEDIUM",
                        description=f"Agent {agent} shows moderate vote instability ({flip_rate:.0%})",
                        evidence={"agent": agent, "flip_rate": flip_rate},
                    )
                )

    def _check_confidence_manipulation(self) -> None:
        """Detect suspiciously uniform or extreme confidence patterns."""
        if not self.ledger.entries:
            return
        voter_weights: Dict[str, List[float]] = {}
        for e in self.ledger.entries:
            for v in e.round_result.get("votes", []):
                vid = v["voter_id"]
                voter_weights.setdefault(vid, []).append(abs(v["weight"]))
        for agent, weights in voter_weights.items():
            if len(weights) < 3:
                continue
            mean_w = sum(weights) / len(weights)
            variance = sum((w - mean_w) ** 2 for w in weights) / len(weights)
            stddev = math.sqrt(variance)
            # Suspiciously uniform (always exact same confidence)
            if stddev < 0.01 and len(weights) >= 5:
                self.findings.append(
                    AuditFinding(
                        category="Confidence Manipulation",
                        severity="MEDIUM",
                        description=f"Agent {agent} has suspiciously uniform confidence (σ={stddev:.4f}) across {len(weights)} votes",
                        evidence={
                            "agent": agent,
                            "mean": mean_w,
                            "stddev": stddev,
                            "votes": len(weights),
                        },
                    )
                )
            # Always extreme (always >0.95)
            extreme_count = sum(1 for w in weights if w > 0.95)
            if extreme_count / len(weights) > 0.8 and len(weights) >= 5:
                self.findings.append(
                    AuditFinding(
                        category="Overconfidence",
                        severity="LOW",
                        description=f"Agent {agent} votes with extreme confidence {extreme_count}/{len(weights)} times",
                        evidence={
                            "agent": agent,
                            "extreme_ratio": extreme_count / len(weights),
                        },
                    )
                )

    def _check_slash_patterns(self) -> None:
        """Detect agents that are repeatedly slashed (potential bad actors)."""
        slash_counts: Dict[str, int] = {}
        for e in self.ledger.entries:
            for s in e.round_result.get("slashed", []):
                slash_counts[s] = slash_counts.get(s, 0) + 1
        total = len(self.ledger.entries)
        for agent, count in slash_counts.items():
            ratio = count / total if total > 0 else 0
            if ratio > 0.3:
                self.findings.append(
                    AuditFinding(
                        category="Repeat Offender",
                        severity="HIGH",
                        description=f"Agent {agent} slashed in {ratio:.0%} of rounds ({count}/{total})",
                        evidence={"agent": agent, "count": count, "ratio": ratio},
                    )
                )

    def _check_collusion_clusters(self) -> None:
        """Detect groups of agents that always vote the same way."""
        if len(self.ledger.entries) < 5:
            return
        agents = set()
        vote_vectors: Dict[str, List[int]] = {}
        for e in self.ledger.entries:
            for v in e.round_result.get("votes", []):
                vid = v["voter_id"]
                agents.add(vid)
                vote_vectors.setdefault(vid, []).append(
                    1 if v["weight"] > 0 else -1
                )
        # Pad to equal length
        max_len = max((len(v) for v in vote_vectors.values()), default=0)
        for vid in vote_vectors:
            while len(vote_vectors[vid]) < max_len:
                vote_vectors[vid].append(0)
        # Check pairwise agreement
        agent_list = sorted(vote_vectors.keys())
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a, b = agent_list[i], agent_list[j]
                va, vb = vote_vectors[a], vote_vectors[b]
                agreements = sum(
                    1 for x, y in zip(va, vb) if x == y and x != 0
                )
                comparisons = sum(
                    1 for x, y in zip(va, vb) if x != 0 and y != 0
                )
                if comparisons >= 5:
                    agreement_rate = agreements / comparisons
                    if agreement_rate > 0.9:
                        self.findings.append(
                            AuditFinding(
                                category="Potential Collusion",
                                severity="MEDIUM",
                                description=f"Agents {a} and {b} agree {agreement_rate:.0%} ({agreements}/{comparisons} rounds)",
                                evidence={
                                    "agents": [a, b],
                                    "agreement_rate": agreement_rate,
                                    "comparisons": comparisons,
                                },
                            )
                        )

    def _check_commit_rate(self) -> None:
        """Flag unusually high or low commit rates."""
        if not self.ledger.entries:
            return
        commits = sum(
            1 for e in self.ledger.entries if e.round_result["committed"]
        )
        total = len(self.ledger.entries)
        rate = commits / total
        if rate < 0.2:
            self.findings.append(
                AuditFinding(
                    category="Low Commit Rate",
                    severity="HIGH",
                    description=f"Only {rate:.0%} of rounds committed ({commits}/{total}) -- possible systemic disagreement",
                    evidence={"commit_rate": rate, "commits": commits},
                )
            )
        elif rate > 0.95 and total >= 5:
            self.findings.append(
                AuditFinding(
                    category="Rubber-Stamp Warning",
                    severity="MEDIUM",
                    description=f"{rate:.0%} commit rate ({commits}/{total}) -- consensus may lack genuine deliberation",
                    evidence={"commit_rate": rate, "commits": commits},
                )
            )
        else:
            self.findings.append(
                AuditFinding(
                    category="Commit Rate",
                    severity="INFO",
                    description=f"Commit rate: {rate:.0%} ({commits}/{total})",
                    evidence={"commit_rate": rate},
                )
            )

    def summary(self) -> Dict[str, Any]:
        by_severity: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        for f in self.findings:
            by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
            by_category[f.category] = by_category.get(f.category, 0) + 1
        critical_or_high = by_severity.get("CRITICAL", 0) + by_severity.get(
            "HIGH", 0
        )
        if critical_or_high == 0:
            verdict = "CLEAN"
        elif critical_or_high <= 2:
            verdict = "CAUTION"
        else:
            verdict = "ALERT"
        return {
            "verdict": verdict,
            "total_findings": len(self.findings),
            "by_severity": by_severity,
            "by_category": by_category,
        }


# -- Simulation Runner ----------------------------------------------------

TASKS = [
    "What is 2 + 2?",
    "Is P = NP?",
    "What colour is the sky?",
    "Compute the integral of x^2.",
    "Name the largest planet.",
    "What is the capital of France?",
    "Simplify sqrt(144).",
    "Define entropy.",
    "Is infinity a number?",
    "What is Occam's razor?",
    "Name the fastest sorting algorithm.",
    "What is the Halting Problem?",
    "Define a Nash Equilibrium.",
    "Explain quantum entanglement.",
    "What is Gödel's incompleteness theorem?",
]


def _build_agents(
    n_agents: int, n_byzantine: int
) -> List[MockAgent]:
    answers = ["A", "B", "C"]
    agents = []
    for i in range(n_agents):
        is_byz = i < n_byzantine
        conf = random.uniform(0.3, 0.9) if not is_byz else random.uniform(0.5, 1.0)
        ans = random.choice(answers)
        agents.append(
            MockAgent(
                agent_id=f"agent_{i}",
                answer=ans,
                confidence=round(conf, 3),
                byzantine=is_byz,
                accept_set={ans} if not is_byz else set(answers),
            )
        )
    return agents


async def run_simulation(
    n_rounds: int = 15,
    n_agents: int = 6,
    n_byzantine: int = 1,
    threshold: float = 2.0,
    inject_tamper: bool = False,
) -> Tuple[AccountabilityLedger, AuditEngine]:
    """Run consensus rounds and record everything to the ledger."""
    ledger = AccountabilityLedger()

    for round_num in range(n_rounds):
        agents = _build_agents(n_agents, n_byzantine)
        engine = MBFTEngine(agents, threshold=threshold, max_rounds=1)
        task = TASKS[round_num % len(TASKS)]
        result = await engine.run(task)
        if result:
            result.round_index = round_num
            ledger.append(result)

    # Tamper injection for demo
    if inject_tamper and len(ledger.entries) >= 3:
        target = ledger.entries[len(ledger.entries) // 2]
        target.round_result["leader_id"] = "TAMPERED_AGENT"
        # Hash is now stale -- integrity check will catch this

    audit = AuditEngine(ledger)
    audit.run_full_audit()
    return ledger, audit


# -- HTML Report ----------------------------------------------------------


def _generate_html(
    ledger: AccountabilityLedger, audit: AuditEngine
) -> str:
    summary = audit.summary()
    verdict_colors = {"CLEAN": "#22c55e", "CAUTION": "#f59e0b", "ALERT": "#ef4444"}
    severity_colors = {
        "CRITICAL": "#ef4444",
        "HIGH": "#f97316",
        "MEDIUM": "#f59e0b",
        "LOW": "#3b82f6",
        "INFO": "#6b7280",
    }

    # Build chain visualization data
    chain_html = ""
    for e in ledger.entries:
        rr = e.round_result
        committed_cls = "committed" if rr["committed"] else "rejected"
        short_hash = e.entry_hash[:12]
        prev_short = e.prev_hash[:12]
        chain_html += f"""
        <div class="block {committed_cls}">
            <div class="block-idx">#{e.index}</div>
            <div class="block-hash" title="{html_mod.escape(e.entry_hash)}">{html_mod.escape(short_hash)}...</div>
            <div class="block-link">← {html_mod.escape(prev_short)}...</div>
            <div class="block-leader">Leader: {html_mod.escape(rr['leader_id'])}</div>
            <div class="block-status">{'✓ COMMIT' if rr['committed'] else '✗ REJECT'}</div>
            <div class="block-weight">Agg: {rr['aggregate_weight']:.2f} / {rr['threshold']}</div>
        </div>"""

    # Build findings HTML
    findings_html = ""
    for f in audit.findings:
        color = severity_colors.get(f.severity, "#6b7280")
        findings_html += f"""
        <div class="finding">
            <span class="severity-badge" style="background:{color}">{html_mod.escape(f.severity)}</span>
            <span class="finding-cat">{html_mod.escape(f.category)}</span>
            <span class="finding-desc">{html_mod.escape(f.description)}</span>
        </div>"""

    # Leader distribution for chart
    leader_counts: Dict[str, int] = {}
    for e in ledger.entries:
        lid = e.round_result["leader_id"]
        leader_counts[lid] = leader_counts.get(lid, 0) + 1

    # Vote weight timeline
    round_aggs = [
        {"round": e.index, "agg": e.round_result["aggregate_weight"], "committed": e.round_result["committed"]}
        for e in ledger.entries
    ]

    verdict_color = verdict_colors.get(summary["verdict"], "#6b7280")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mBFT Accountability Ledger -- Audit Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;padding:2rem}}
h1{{text-align:center;font-size:1.8rem;margin-bottom:.3rem;color:#f8fafc}}
.subtitle{{text-align:center;color:#94a3b8;margin-bottom:2rem;font-size:.95rem}}
.verdict-bar{{text-align:center;padding:1rem;border-radius:12px;margin-bottom:2rem;font-size:1.3rem;font-weight:700;
  background:{verdict_color}22;border:2px solid {verdict_color};color:{verdict_color}}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:2rem}}
.stat-card{{background:#1e293b;border-radius:10px;padding:1.2rem;text-align:center}}
.stat-val{{font-size:2rem;font-weight:700;color:#f8fafc}}
.stat-label{{color:#94a3b8;font-size:.85rem;margin-top:.3rem}}
h2{{font-size:1.2rem;color:#f8fafc;margin:1.5rem 0 1rem;padding-bottom:.5rem;border-bottom:1px solid #334155}}
.chain-container{{display:flex;gap:.5rem;overflow-x:auto;padding:1rem 0;margin-bottom:1rem}}
.block{{min-width:140px;padding:.8rem;border-radius:8px;font-size:.75rem;flex-shrink:0}}
.block.committed{{background:#166534;border:1px solid #22c55e}}
.block.rejected{{background:#7f1d1d;border:1px solid #ef4444}}
.block-idx{{font-weight:700;font-size:.9rem;margin-bottom:.3rem}}
.block-hash{{color:#67e8f9;font-family:monospace;font-size:.7rem}}
.block-link{{color:#94a3b8;font-family:monospace;font-size:.65rem;margin-bottom:.3rem}}
.block-leader{{color:#fbbf24}}
.block-status{{font-weight:700;margin-top:.3rem}}
.block-weight{{color:#94a3b8;font-size:.7rem}}
.finding{{display:flex;align-items:center;gap:.8rem;padding:.7rem 1rem;background:#1e293b;border-radius:8px;margin-bottom:.5rem}}
.severity-badge{{color:#fff;padding:.2rem .6rem;border-radius:4px;font-size:.75rem;font-weight:700;min-width:70px;text-align:center}}
.finding-cat{{color:#fbbf24;font-weight:600;min-width:160px}}
.finding-desc{{color:#cbd5e1;flex:1}}
canvas{{background:#1e293b;border-radius:10px;width:100%;margin-bottom:1rem}}
.section{{background:#1e293b;border-radius:10px;padding:1.5rem;margin-bottom:1.5rem}}
.recommendations{{list-style:none;padding:0}}
.recommendations li{{padding:.6rem 0;border-bottom:1px solid #334155;color:#cbd5e1}}
.recommendations li:last-child{{border:none}}
.recommendations li::before{{content:' ';margin-right:.3rem}}
.footer{{text-align:center;color:#475569;font-size:.8rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #1e293b}}
</style>
</head>
<body>
<h1>⛓️ mBFT Accountability Ledger</h1>
<p class="subtitle">Tamper-Evident Consensus Audit Trail -- {len(ledger.entries)} entries recorded</p>

<div class="verdict-bar">AUDIT VERDICT: {summary['verdict']} -- {summary['total_findings']} findings</div>

<div class="stats">
  <div class="stat-card"><div class="stat-val">{len(ledger.entries)}</div><div class="stat-label">Ledger Entries</div></div>
  <div class="stat-card"><div class="stat-val">{summary['by_severity'].get('CRITICAL', 0)}</div><div class="stat-label">Critical Findings</div></div>
  <div class="stat-card"><div class="stat-val">{summary['by_severity'].get('HIGH', 0)}</div><div class="stat-label">High Findings</div></div>
  <div class="stat-card"><div class="stat-val">{sum(1 for e in ledger.entries if e.round_result['committed'])}/{len(ledger.entries)}</div><div class="stat-label">Commit Rate</div></div>
</div>

<h2>🔗 Hash Chain</h2>
<div class="chain-container">{chain_html}</div>

<h2>📊 Aggregate Weight Timeline</h2>
<canvas id="timelineChart" height="200"></canvas>

<h2>👑 Leader Distribution</h2>
<canvas id="leaderChart" height="200"></canvas>

<h2>🔍 Audit Findings</h2>
<div class="section">{findings_html if findings_html else '<p style="color:#94a3b8">No findings -- ledger is clean.</p>'}</div>

<h2> Recommendations</h2>
<div class="section">
<ul class="recommendations" id="recommendations"></ul>
</div>

<div class="footer">
  mBFT Accountability Ledger -- generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br>
  Consensus Accountability &amp; Tamper-Evident Audit Trail
</div>

<script>
const roundData = {json.dumps(round_aggs)};
const leaderData = {json.dumps(leader_counts)};
const findings = {json.dumps([f.to_dict() for f in audit.findings])};

// Timeline chart
(function() {{
  const canvas = document.getElementById('timelineChart');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = 400;
  ctx.scale(2, 2);
  const W = canvas.offsetWidth, H = 200;
  const pad = {{t:30, b:40, l:50, r:20}};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  if (roundData.length === 0) return;
  const maxAgg = Math.max(...roundData.map(d => d.agg), roundData[0].agg * 1.2 || 5);

  // Grid
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = pad.t + (ph * i / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    ctx.fillStyle = '#94a3b8'; ctx.font = '10px monospace';
    ctx.fillText((maxAgg * (1 - i/4)).toFixed(1), 5, y + 4);
  }}

  // Threshold line
  const threshY = pad.t + ph * (1 - (roundData[0]?.agg ? 2.0 / maxAgg : 0.5));
  ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(pad.l, threshY); ctx.lineTo(pad.l+pw, threshY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#f59e0b'; ctx.font = '10px sans-serif';
  ctx.fillText('threshold', pad.l + pw - 50, threshY - 5);

  // Line
  ctx.beginPath(); ctx.strokeStyle = '#67e8f9'; ctx.lineWidth = 2;
  roundData.forEach((d, i) => {{
    const x = pad.l + (i / Math.max(roundData.length - 1, 1)) * pw;
    const y = pad.t + ph * (1 - d.agg / maxAgg);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();

  // Points
  roundData.forEach((d, i) => {{
    const x = pad.l + (i / Math.max(roundData.length - 1, 1)) * pw;
    const y = pad.t + ph * (1 - d.agg / maxAgg);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = d.committed ? '#22c55e' : '#ef4444'; ctx.fill();
  }});

  ctx.fillStyle = '#94a3b8'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
  roundData.forEach((d, i) => {{
    if (i % Math.max(1, Math.floor(roundData.length / 10)) === 0) {{
      const x = pad.l + (i / Math.max(roundData.length - 1, 1)) * pw;
      ctx.fillText('#' + d.round, x, H - 10);
    }}
  }});
}})();

// Leader pie chart
(function() {{
  const canvas = document.getElementById('leaderChart');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = 400;
  ctx.scale(2, 2);
  const W = canvas.offsetWidth, H = 200;
  const agents = Object.keys(leaderData);
  const total = Object.values(leaderData).reduce((a,b) => a+b, 0);
  if (total === 0) return;
  const colors = ['#22c55e','#3b82f6','#f59e0b','#ef4444','#8b5cf6','#ec4899','#14b8a6','#f97316','#6366f1','#06b6d4'];
  const cx = W * 0.35, cy = H * 0.5, r = Math.min(W * 0.25, H * 0.4);
  let angle = -Math.PI / 2;
  agents.forEach((a, i) => {{
    const slice = (leaderData[a] / total) * Math.PI * 2;
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, angle, angle + slice);
    ctx.fillStyle = colors[i % colors.length]; ctx.fill();
    angle += slice;
  }});
  // Legend
  let ly = 20;
  agents.forEach((a, i) => {{
    ctx.fillStyle = colors[i % colors.length];
    ctx.fillRect(W * 0.7, ly, 12, 12);
    ctx.fillStyle = '#e2e8f0'; ctx.font = '11px sans-serif';
    ctx.fillText(a + ' (' + leaderData[a] + ')', W * 0.7 + 18, ly + 10);
    ly += 20;
  }});
}})();

// Recommendations
(function() {{
  const el = document.getElementById('recommendations');
  const recs = [];
  const cats = findings.map(f => f.category);
  if (cats.includes('Integrity') && findings.some(f => f.category === 'Integrity' && f.severity === 'CRITICAL'))
    recs.push('URGENT: Hash chain tampered -- investigate entries flagged with integrity violations immediately');
  if (cats.includes('Leader Monopoly'))
    recs.push('Implement leader rotation or reputation decay to prevent single-agent dominance');
  if (cats.includes('Vote Flipping'))
    recs.push('Review flip agents for Byzantine behavior or miscalibration -- consider probation');
  if (cats.includes('Potential Collusion'))
    recs.push('Monitor correlated voting pairs -- introduce vote privacy or randomized ordering');
  if (cats.includes('Repeat Offender'))
    recs.push('Repeatedly slashed agents should face escalating penalties or temporary exclusion');
  if (cats.includes('Low Commit Rate'))
    recs.push('Low commit rate suggests threshold may be too high or agents are too divergent -- tune parameters');
  if (cats.includes('Rubber-Stamp Warning'))
    recs.push('Near-100% commit rate may indicate groupthink -- introduce adversarial agents for genuine deliberation');
  if (cats.includes('Overconfidence'))
    recs.push('Overconfident agents may need calibration training to improve epistemic humility');
  if (recs.length === 0)
    recs.push('Ledger looks healthy -- continue monitoring with regular audits');
  recs.forEach(r => {{
    const li = document.createElement('li');
    li.textContent = r;
    el.appendChild(li);
  }});
}})();
</script>
</body>
</html>"""


# -- CLI Entry Point ------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mBFT Consensus Accountability Ledger"
    )
    p.add_argument("--rounds", type=int, default=15, help="Number of consensus rounds")
    p.add_argument("--agents", type=int, default=6, help="Number of agents")
    p.add_argument("--byzantine", type=int, default=1, help="Number of Byzantine agents")
    p.add_argument("--threshold", type=float, default=2.0, help="Commit threshold")
    p.add_argument("--inject-tamper", action="store_true", help="Inject a tamper for demo")
    p.add_argument(
        "--audit-only",
        type=str,
        default=None,
        help="Path to existing ledger JSON to audit (skip simulation)",
    )
    p.add_argument("--export", type=str, default=None, help="Export path (.html or .json)")
    return p.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> None:
    if args.audit_only:
        path = Path(args.audit_only).resolve()
        if not path.is_file():
            print(f"Error: ledger file not found: {path}", file=sys.stderr)
            sys.exit(1)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"Error: invalid JSON in {path.name}: {exc}", file=sys.stderr)
            sys.exit(1)
        try:
            ledger = AccountabilityLedger.from_json(data)
        except (ValueError, TypeError) as exc:
            print(f"Error: malformed ledger structure: {exc}", file=sys.stderr)
            sys.exit(1)
        audit = AuditEngine(ledger)
        audit.run_full_audit()
        print(f"Audited {len(ledger.entries)} entries from {path.name}")
    else:
        print(
            f"Running {args.rounds}-round simulation with {args.agents} agents "
            f"({args.byzantine} Byzantine)..."
        )
        ledger, audit = await run_simulation(
            n_rounds=args.rounds,
            n_agents=args.agents,
            n_byzantine=args.byzantine,
            threshold=args.threshold,
            inject_tamper=args.inject_tamper,
        )
        print(f"Recorded {len(ledger.entries)} ledger entries.")

    # Print findings
    summary = audit.summary()
    print(f"\n{'='*60}")
    print(f"  AUDIT VERDICT: {summary['verdict']}")
    print(f"  Total findings: {summary['total_findings']}")
    print(f"{'='*60}")
    for f in audit.findings:
        icon = {"CRITICAL": "[!]", "HIGH": "[H]", "MEDIUM": "[M]", "LOW": "[L]", "INFO": "[i]"}.get(
            f.severity, "[?]"
        )
        print(f"  {icon} [{f.severity}] {f.category}: {f.description}")

    # Recommendations
    print(f"\n{'-'*60}")
    print("   RECOMMENDATIONS")
    print(f"{'-'*60}")
    cats = {f.category for f in audit.findings}
    recs = []
    if any(f.severity == "CRITICAL" and f.category == "Integrity" for f in audit.findings):
        recs.append("URGENT: Hash chain tampered -- investigate immediately")
    if "Leader Monopoly" in cats:
        recs.append("Implement leader rotation to prevent dominance")
    if "Vote Flipping" in cats:
        recs.append("Review unstable voters for miscalibration")
    if "Potential Collusion" in cats:
        recs.append("Monitor correlated voters -- consider vote privacy")
    if "Repeat Offender" in cats:
        recs.append("Escalate penalties for repeatedly slashed agents")
    if not recs:
        recs.append("Ledger healthy -- continue regular auditing")
    for r in recs:
        print(f"  * {r}")

    # Export
    if args.export:
        path = Path(args.export)
        if path.suffix == ".json":
            path.write_text(json.dumps(ledger.export_json(), indent=2))
            print(f"\nExported ledger JSON -> {path}")
        else:
            html = _generate_html(ledger, audit)
            path.write_text(html, encoding="utf-8")
            print(f"\nExported HTML report -> {path}")


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
