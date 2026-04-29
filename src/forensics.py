"""Consensus Forensics Analyzer — autonomous post-run forensic investigation.

Analyzes MBFTEngine history to detect voting blocs, collusion patterns,
influence asymmetries, and reputation manipulation. Produces an interactive
HTML report with charts, findings, and proactive recommendations.

Usage::

    python -m src.forensics --demo
    python -m src.forensics --demo --agents 8 --rounds 6 --byzantine 2
    python -m src.forensics --demo -o report.html
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Allow running as a module from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine
from src.core.state import RoundResult


# ── Data structures ────────────────────────────────────────────────────

@dataclass
class AgentProfile:
    agent_id: str
    total_votes: int = 0
    acceptances: int = 0
    rejections: int = 0
    times_leader: int = 0
    times_slashed: int = 0
    avg_confidence: float = 0.0
    influence_score: float = 0.0
    bloc_id: Optional[int] = None


@dataclass
class VotingBloc:
    bloc_id: int
    members: List[str] = field(default_factory=list)
    agreement_rate: float = 0.0
    avg_confidence: float = 0.0
    label: str = ""


@dataclass
class Finding:
    severity: str  # "critical", "warning", "info"
    category: str
    title: str
    detail: str
    agents_involved: List[str] = field(default_factory=list)


@dataclass
class ForensicReport:
    agent_profiles: Dict[str, AgentProfile] = field(default_factory=dict)
    blocs: List[VotingBloc] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    round_summaries: List[dict] = field(default_factory=list)
    agreement_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    influence_timeline: Dict[str, List[float]] = field(default_factory=dict)


# ── Analyzer ───────────────────────────────────────────────────────────

class ForensicsAnalyzer:
    """Performs forensic analysis on MBFTEngine run history."""

    def __init__(self, history: List[RoundResult], reputation: Dict[str, float]) -> None:
        self.history = history
        self.reputation = reputation
        self.report = ForensicReport()

    def analyze(self) -> ForensicReport:
        """Run full forensic analysis pipeline."""
        self._build_agent_profiles()
        self._compute_agreement_matrix()
        self._detect_voting_blocs()
        self._compute_influence_timeline()
        self._detect_collusion()
        self._detect_reputation_manipulation()
        self._detect_confidence_gaming()
        self._assess_consensus_health()
        self._generate_recommendations()
        self._build_round_summaries()
        return self.report

    def _build_agent_profiles(self) -> None:
        profiles: Dict[str, AgentProfile] = {}
        confidences: Dict[str, List[float]] = defaultdict(list)

        for rr in self.history:
            # Leader
            if rr.leader_id not in profiles:
                profiles[rr.leader_id] = AgentProfile(agent_id=rr.leader_id)
            profiles[rr.leader_id].times_leader += 1

            for slashed_id in rr.slashed:
                if slashed_id not in profiles:
                    profiles[slashed_id] = AgentProfile(agent_id=slashed_id)
                profiles[slashed_id].times_slashed += 1

            for v in rr.votes:
                if v.voter_id not in profiles:
                    profiles[v.voter_id] = AgentProfile(agent_id=v.voter_id)
                p = profiles[v.voter_id]
                p.total_votes += 1
                if v.is_rejection:
                    p.rejections += 1
                else:
                    p.acceptances += 1
                confidences[v.voter_id].append(abs(v.weight))

        for aid, confs in confidences.items():
            profiles[aid].avg_confidence = statistics.mean(confs) if confs else 0.0

        # Influence = reputation * avg_confidence * acceptance_rate
        for aid, p in profiles.items():
            rep = self.reputation.get(aid, 1.0)
            acc_rate = p.acceptances / max(p.total_votes, 1)
            p.influence_score = rep * p.avg_confidence * (0.5 + 0.5 * acc_rate)

        self.report.agent_profiles = profiles

    def _compute_agreement_matrix(self) -> None:
        """Pairwise agreement rate between all voting agents.

        Uses a single O(R × V²) pass over rounds instead of the previous
        O(A² × R) triple-nested loop.  For each round we iterate voter
        pairs once and accumulate agree/total counts directly, which also
        improves cache locality when A (total agents) is large.
        """
        # Accumulate pairwise agree/total in a single pass over rounds.
        # Keys are (min_id, max_id) to avoid storing both directions.
        pair_agree: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_total: Dict[Tuple[str, str], int] = defaultdict(int)

        for rr in self.history:
            # Build per-round voter list with decisions
            voters: List[Tuple[str, bool]] = [
                (v.voter_id, not v.is_rejection) for v in rr.votes
            ]
            n = len(voters)
            for i in range(n):
                a_id, a_dec = voters[i]
                for j in range(i + 1, n):
                    b_id, b_dec = voters[j]
                    key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
                    pair_total[key] += 1
                    if a_dec == b_dec:
                        pair_agree[key] += 1

        # Build symmetric matrix from accumulated counts
        agents = sorted(self.report.agent_profiles.keys())
        matrix: Dict[str, Dict[str, float]] = {}
        for a in agents:
            matrix[a] = {a: 1.0}

        for (a, b), total in pair_total.items():
            rate = pair_agree.get((a, b), 0) / total
            matrix.setdefault(a, {})[b] = rate
            matrix.setdefault(b, {})[a] = rate

        # Fill missing pairs (agents that never co-voted) with 0
        for a in agents:
            row = matrix[a]
            for b in agents:
                if b not in row:
                    row[b] = 0.0

        self.report.agreement_matrix = matrix

    def _detect_voting_blocs(self) -> None:
        """Simple greedy clustering based on agreement matrix."""
        matrix = self.report.agreement_matrix
        agents = sorted(matrix.keys())
        assigned: Dict[str, int] = {}
        blocs: List[VotingBloc] = []
        bloc_id = 0

        for a in agents:
            if a in assigned:
                continue
            # Start a new bloc
            members = [a]
            assigned[a] = bloc_id
            for b in agents:
                if b in assigned:
                    continue
                # Join bloc if agreement with all existing members > 0.7
                if all(matrix.get(m, {}).get(b, 0) >= 0.7 for m in members):
                    members.append(b)
                    assigned[b] = bloc_id

            # Compute bloc stats
            avg_agree = 0.0
            pairs = 0
            for i, m1 in enumerate(members):
                for m2 in members[i + 1:]:
                    avg_agree += matrix.get(m1, {}).get(m2, 0)
                    pairs += 1
            avg_agree = avg_agree / max(pairs, 1)

            avg_conf = statistics.mean(
                self.report.agent_profiles[m].avg_confidence
                for m in members
                if m in self.report.agent_profiles
            ) if members else 0.0

            label = "Byzantine" if all(
                self.report.agent_profiles.get(m, AgentProfile(agent_id=m)).times_slashed > 0
                for m in members
            ) else f"Bloc-{bloc_id}"

            bloc = VotingBloc(
                bloc_id=bloc_id,
                members=members,
                agreement_rate=avg_agree,
                avg_confidence=avg_conf,
                label=label,
            )
            blocs.append(bloc)
            for m in members:
                if m in self.report.agent_profiles:
                    self.report.agent_profiles[m].bloc_id = bloc_id
            bloc_id += 1

        self.report.blocs = blocs

    def _compute_influence_timeline(self) -> None:
        """Track cumulative influence per agent across rounds."""
        timeline: Dict[str, List[float]] = defaultdict(list)
        cumulative: Dict[str, float] = defaultdict(float)

        for rr in self.history:
            for v in rr.votes:
                cumulative[v.voter_id] += abs(v.weight)
            for aid in cumulative:
                timeline[aid].append(cumulative[aid])

        self.report.influence_timeline = dict(timeline)

    def _detect_collusion(self) -> None:
        """Flag agent pairs with suspiciously high agreement + coordinated rejections."""
        matrix = self.report.agreement_matrix
        agents = sorted(matrix.keys())

        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                agreement = matrix.get(a, {}).get(b, 0)
                if agreement >= 0.95 and len(self.history) >= 3:
                    # Check if they both reject often
                    pa = self.report.agent_profiles.get(a)
                    pb = self.report.agent_profiles.get(b)
                    if pa and pb and pa.rejections > 0 and pb.rejections > 0:
                        self.report.findings.append(Finding(
                            severity="warning",
                            category="collusion",
                            title=f"Potential collusion: {a} ↔ {b}",
                            detail=f"Agreement rate {agreement:.0%} with coordinated rejections "
                                   f"({pa.rejections} + {pb.rejections}). May indicate coordinated Byzantine behavior.",
                            agents_involved=[a, b],
                        ))

    def _detect_reputation_manipulation(self) -> None:
        """Flag agents whose reputation dropped disproportionately."""
        for aid, rep in self.reputation.items():
            p = self.report.agent_profiles.get(aid)
            if p and rep < 0.5:
                if p.times_leader > 0 and p.times_slashed >= p.times_leader:
                    self.report.findings.append(Finding(
                        severity="critical",
                        category="reputation",
                        title=f"Severe reputation loss: {aid}",
                        detail=f"Reputation dropped to {rep:.3f}. "
                               f"Slashed {p.times_slashed}x in {p.times_leader} leadership attempts. "
                               f"Agent may be consistently proposing incorrect solutions.",
                        agents_involved=[aid],
                    ))

    def _detect_confidence_gaming(self) -> None:
        """Flag agents with suspiciously high or volatile confidence."""
        for aid, p in self.report.agent_profiles.items():
            if p.avg_confidence > 0.95 and p.rejections > p.acceptances:
                self.report.findings.append(Finding(
                    severity="warning",
                    category="confidence",
                    title=f"Confidence gaming suspected: {aid}",
                    detail=f"Avg confidence {p.avg_confidence:.2f} but rejection rate "
                           f"{p.rejections}/{p.total_votes}. High confidence + frequent rejection "
                           f"may indicate strategic confidence inflation.",
                    agents_involved=[aid],
                ))

    def _assess_consensus_health(self) -> None:
        """Overall consensus health checks."""
        committed_count = sum(1 for rr in self.history if rr.committed)
        total = len(self.history)

        if total > 0:
            commit_rate = committed_count / total
            if commit_rate < 0.3:
                self.report.findings.append(Finding(
                    severity="critical",
                    category="health",
                    title="Low consensus commit rate",
                    detail=f"Only {committed_count}/{total} rounds committed ({commit_rate:.0%}). "
                           f"The swarm may have too many conflicting agents or the threshold is too high.",
                ))
            elif commit_rate < 0.6:
                self.report.findings.append(Finding(
                    severity="warning",
                    category="health",
                    title="Moderate consensus commit rate",
                    detail=f"{committed_count}/{total} rounds committed ({commit_rate:.0%}). "
                           f"Consider adjusting threshold or investigating dissenting agents.",
                ))

        # Check for single-agent dominance
        for aid, p in self.report.agent_profiles.items():
            if total > 0 and p.times_leader / max(total, 1) > 0.7:
                self.report.findings.append(Finding(
                    severity="warning",
                    category="dominance",
                    title=f"Leader monopoly: {aid}",
                    detail=f"Agent led {p.times_leader}/{total} rounds ({p.times_leader/total:.0%}). "
                           f"Lack of leader diversity may indicate reputation imbalance.",
                    agents_involved=[aid],
                ))

    def _generate_recommendations(self) -> None:
        recs = []
        severities = [f.severity for f in self.report.findings]

        if "critical" in severities:
            recs.append("🔴 Critical issues detected — review agent composition before next run")

        # Bloc-based recs
        large_blocs = [b for b in self.report.blocs if len(b.members) >= 3]
        if large_blocs:
            recs.append(f"⚠️ {len(large_blocs)} large voting bloc(s) detected — consider diversifying agent pool")

        collusion_findings = [f for f in self.report.findings if f.category == "collusion"]
        if collusion_findings:
            recs.append("🔍 Investigate potential collusion — consider rotating agent assignments")

        rep_issues = [f for f in self.report.findings if f.category == "reputation"]
        if rep_issues:
            recs.append("🛡️ Agents with severe reputation loss should be retrained or replaced")

        if not self.report.findings:
            recs.append("✅ No anomalies detected — consensus swarm appears healthy")

        recs.append("📊 Run periodic forensic analysis to track swarm health over time")
        self.report.recommendations = recs

    def _build_round_summaries(self) -> None:
        for rr in self.history:
            accept_count = sum(1 for v in rr.votes if not v.is_rejection)
            reject_count = sum(1 for v in rr.votes if v.is_rejection)
            self.report.round_summaries.append({
                "round": rr.round_index,
                "leader": rr.leader_id,
                "committed": rr.committed,
                "aggregate": round(rr.aggregate_weight, 3),
                "threshold": rr.threshold,
                "accepts": accept_count,
                "rejects": reject_count,
                "slashed": rr.slashed,
            })


# ── HTML Report Generator ─────────────────────────────────────────────

def generate_html_report(report: ForensicReport) -> str:
    """Generate a self-contained interactive HTML forensic report."""

    # Prepare data for charts
    agents_data = []
    for aid, p in sorted(report.agent_profiles.items()):
        agents_data.append({
            "id": aid,
            "influence": round(p.influence_score, 3),
            "confidence": round(p.avg_confidence, 3),
            "acceptRate": round(p.acceptances / max(p.total_votes, 1), 3),
            "timesLeader": p.times_leader,
            "timesSlashed": p.times_slashed,
            "bloc": p.bloc_id,
        })

    blocs_data = []
    for b in report.blocs:
        blocs_data.append({
            "id": b.bloc_id,
            "label": b.label,
            "members": b.members,
            "agreementRate": round(b.agreement_rate, 3),
            "avgConfidence": round(b.avg_confidence, 3),
        })

    findings_data = []
    for f in report.findings:
        findings_data.append({
            "severity": f.severity,
            "category": f.category,
            "title": f.title,
            "detail": f.detail,
            "agents": f.agents_involved,
        })

    matrix_agents = sorted(report.agreement_matrix.keys())
    matrix_values = []
    for a in matrix_agents:
        row = []
        for b in matrix_agents:
            row.append(round(report.agreement_matrix.get(a, {}).get(b, 0), 2))
        matrix_values.append(row)

    timeline_data = {}
    for aid, vals in report.influence_timeline.items():
        timeline_data[aid] = [round(v, 3) for v in vals]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mBFT Consensus Forensics Report</title>
<style>
  :root {{ --bg: #0d1117; --surface: #161b22; --border: #30363d; --text: #c9d1d9;
           --accent: #58a6ff; --green: #3fb950; --yellow: #d29922; --red: #f85149;
           --purple: #bc8cff; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         padding:24px; line-height:1.6; }}
  h1 {{ color:var(--accent); font-size:28px; margin-bottom:4px; }}
  h2 {{ color:var(--accent); font-size:20px; margin:32px 0 16px; border-bottom:1px solid var(--border); padding-bottom:8px; }}
  .subtitle {{ color:#8b949e; margin-bottom:24px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(280px,1fr)); gap:16px; margin-bottom:24px; }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px; }}
  .card h3 {{ font-size:14px; color:#8b949e; text-transform:uppercase; margin-bottom:8px; }}
  .card .value {{ font-size:32px; font-weight:700; }}
  .severity-critical {{ color:var(--red); }}
  .severity-warning {{ color:var(--yellow); }}
  .severity-info {{ color:var(--accent); }}
  .finding {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:12px;
             border-left:4px solid var(--border); }}
  .finding.critical {{ border-left-color:var(--red); }}
  .finding.warning {{ border-left-color:var(--yellow); }}
  .finding.info {{ border-left-color:var(--accent); }}
  .finding-title {{ font-weight:600; margin-bottom:4px; }}
  .finding-detail {{ color:#8b949e; font-size:14px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600; margin-right:4px; }}
  .badge-critical {{ background:rgba(248,81,73,0.2); color:var(--red); }}
  .badge-warning {{ background:rgba(210,153,34,0.2); color:var(--yellow); }}
  .badge-info {{ background:rgba(88,166,255,0.2); color:var(--accent); }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; }}
  th, td {{ padding:10px 12px; text-align:left; border-bottom:1px solid var(--border); font-size:14px; }}
  th {{ color:#8b949e; font-weight:600; background:var(--surface); }}
  tr:hover {{ background:rgba(88,166,255,0.05); }}
  .bar {{ height:8px; border-radius:4px; background:var(--accent); display:inline-block; vertical-align:middle; }}
  .matrix {{ display:grid; gap:2px; margin:16px 0; }}
  .matrix-cell {{ width:100%; aspect-ratio:1; border-radius:4px; display:flex; align-items:center; justify-content:center;
                  font-size:11px; font-weight:600; cursor:default; }}
  .rec {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px 16px;
          margin-bottom:8px; font-size:14px; }}
  canvas {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; }}
  .chart-container {{ position:relative; margin:16px 0; }}
  .tabs {{ display:flex; gap:4px; margin-bottom:16px; }}
  .tab {{ padding:8px 16px; background:var(--surface); border:1px solid var(--border); border-radius:6px;
          cursor:pointer; font-size:14px; color:var(--text); }}
  .tab.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
  .tab-content {{ display:none; }}
  .tab-content.active {{ display:block; }}
</style>
</head>
<body>
<h1>🔬 mBFT Consensus Forensics Report</h1>
<p class="subtitle">Autonomous post-run forensic analysis of consensus behavior</p>

<div class="grid">
  <div class="card"><h3>Rounds Analyzed</h3><div class="value" id="stat-rounds">-</div></div>
  <div class="card"><h3>Agents Profiled</h3><div class="value" id="stat-agents">-</div></div>
  <div class="card"><h3>Voting Blocs</h3><div class="value" id="stat-blocs">-</div></div>
  <div class="card"><h3>Findings</h3><div class="value" id="stat-findings">-</div></div>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('findings')">Findings</div>
  <div class="tab" onclick="switchTab('agents')">Agent Profiles</div>
  <div class="tab" onclick="switchTab('blocs')">Voting Blocs</div>
  <div class="tab" onclick="switchTab('matrix')">Agreement Matrix</div>
  <div class="tab" onclick="switchTab('timeline')">Influence Timeline</div>
  <div class="tab" onclick="switchTab('rounds')">Round Log</div>
  <div class="tab" onclick="switchTab('recs')">Recommendations</div>
</div>

<div id="tab-findings" class="tab-content active"></div>
<div id="tab-agents" class="tab-content"></div>
<div id="tab-blocs" class="tab-content"></div>
<div id="tab-matrix" class="tab-content"></div>
<div id="tab-timeline" class="tab-content">
  <h2>Influence Timeline</h2>
  <div class="chart-container"><canvas id="timelineCanvas" width="800" height="400"></canvas></div>
</div>
<div id="tab-rounds" class="tab-content"></div>
<div id="tab-recs" class="tab-content"></div>

<script>
const DATA = {{
  agents: {json.dumps(agents_data)},
  blocs: {json.dumps(blocs_data)},
  findings: {json.dumps(findings_data)},
  rounds: {json.dumps(report.round_summaries)},
  recommendations: {json.dumps(report.recommendations)},
  matrixAgents: {json.dumps(matrix_agents)},
  matrixValues: {json.dumps(matrix_values)},
  timeline: {json.dumps(timeline_data)},
}};

// Stats
document.getElementById('stat-rounds').textContent = DATA.rounds.length;
document.getElementById('stat-agents').textContent = DATA.agents.length;
document.getElementById('stat-blocs').textContent = DATA.blocs.length;
const fc = DATA.findings.filter(f=>f.severity==='critical').length;
const fw = DATA.findings.filter(f=>f.severity==='warning').length;
const fi = DATA.findings.filter(f=>f.severity==='info').length;
let fhtml = '';
if(fc) fhtml += `<span class="severity-critical">${{fc}}🔴</span> `;
if(fw) fhtml += `<span class="severity-warning">${{fw}}🟡</span> `;
if(fi) fhtml += `<span class="severity-info">${{fi}}🔵</span>`;
document.getElementById('stat-findings').innerHTML = fhtml || '0';

// Tabs
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active', t.textContent.toLowerCase().includes(name.substring(0,4))));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if(name==='timeline') drawTimeline();
  if(name==='matrix') drawMatrix();
}}

// Findings tab
(function() {{
  const el = document.getElementById('tab-findings');
  let html = '<h2>Forensic Findings</h2>';
  if(!DATA.findings.length) html += '<p style="color:#8b949e">No anomalies detected.</p>';
  DATA.findings.forEach(f => {{
    html += `<div class="finding ${{f.severity}}">
      <span class="badge badge-${{f.severity}}">${{f.severity.toUpperCase()}}</span>
      <span class="badge badge-info">${{f.category}}</span>
      <div class="finding-title" style="margin-top:8px">${{f.title}}</div>
      <div class="finding-detail">${{f.detail}}</div>
    </div>`;
  }});
  el.innerHTML = html;
}})();

// Agents tab
(function() {{
  const el = document.getElementById('tab-agents');
  let html = '<h2>Agent Profiles</h2><table><tr><th>Agent</th><th>Influence</th><th>Confidence</th><th>Accept Rate</th><th>Leader</th><th>Slashed</th><th>Bloc</th></tr>';
  DATA.agents.forEach(a => {{
    const maxInf = Math.max(...DATA.agents.map(x=>x.influence));
    const barW = maxInf > 0 ? (a.influence/maxInf*100) : 0;
    html += `<tr>
      <td><strong>${{a.id}}</strong></td>
      <td><span class="bar" style="width:${{barW}}px"></span> ${{a.influence.toFixed(3)}}</td>
      <td>${{(a.confidence*100).toFixed(1)}}%</td>
      <td>${{(a.acceptRate*100).toFixed(0)}}%</td>
      <td>${{a.timesLeader}}</td>
      <td style="color:${{a.timesSlashed>0?'var(--red)':'inherit'}}">${{a.timesSlashed}}</td>
      <td>${{a.bloc !== null ? 'Bloc-'+a.bloc : '-'}}</td>
    </tr>`;
  }});
  html += '</table>';
  el.innerHTML = html;
}})();

// Blocs tab
(function() {{
  const el = document.getElementById('tab-blocs');
  let html = '<h2>Voting Blocs</h2>';
  DATA.blocs.forEach(b => {{
    const color = b.label === 'Byzantine' ? 'var(--red)' : 'var(--accent)';
    html += `<div class="card" style="margin-bottom:12px;border-left:4px solid ${{color}}">
      <h3 style="color:${{color}}">${{b.label}} (${{b.members.length}} members)</h3>
      <p>Members: ${{b.members.join(', ')}}</p>
      <p>Internal agreement: ${{(b.agreementRate*100).toFixed(0)}}% · Avg confidence: ${{(b.avgConfidence*100).toFixed(0)}}%</p>
    </div>`;
  }});
  el.innerHTML = html;
}})();

// Matrix tab
function drawMatrix() {{
  const el = document.getElementById('tab-matrix');
  const n = DATA.matrixAgents.length;
  if(!n) {{ el.innerHTML = '<h2>Agreement Matrix</h2><p>No data.</p>'; return; }}
  let html = '<h2>Agreement Matrix</h2><p style="color:#8b949e;font-size:14px">Pairwise voting agreement between agents. Green = high agreement, red = disagreement.</p>';
  html += `<div class="matrix" style="grid-template-columns:80px repeat(${{n}},1fr);max-width:${{n*60+80}}px">`;
  html += '<div></div>';
  DATA.matrixAgents.forEach(a => html += `<div style="font-size:11px;text-align:center;color:#8b949e">${{a}}</div>`);
  DATA.matrixValues.forEach((row, i) => {{
    html += `<div style="font-size:11px;color:#8b949e;display:flex;align-items:center">${{DATA.matrixAgents[i]}}</div>`;
    row.forEach(v => {{
      const r = Math.round(255*(1-v)), g = Math.round(180*v+40), b2 = Math.round(60);
      html += `<div class="matrix-cell" style="background:rgba(${{r}},${{g}},${{b2}},0.6)" title="${{v.toFixed(2)}}">${{v.toFixed(2)}}</div>`;
    }});
  }});
  html += '</div>';
  el.innerHTML = html;
}}

// Timeline chart
function drawTimeline() {{
  const canvas = document.getElementById('timelineCanvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const pad = {{t:40,r:20,b:40,l:60}};
  const agents = Object.keys(DATA.timeline);
  if(!agents.length) return;
  const maxLen = Math.max(...agents.map(a=>DATA.timeline[a].length));
  const maxVal = Math.max(...agents.flatMap(a=>DATA.timeline[a]),1);
  const colors = ['#58a6ff','#3fb950','#d29922','#f85149','#bc8cff','#f0883e','#79c0ff','#56d364'];

  // Grid
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  for(let i=0;i<=4;i++) {{
    const y = pad.t + (H-pad.t-pad.b)*i/4;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke();
    ctx.fillStyle='#8b949e'; ctx.font='12px sans-serif'; ctx.textAlign='right';
    ctx.fillText((maxVal*(1-i/4)).toFixed(1), pad.l-8, y+4);
  }}

  // Lines
  agents.forEach((aid, ai) => {{
    const vals = DATA.timeline[aid];
    ctx.strokeStyle = colors[ai % colors.length];
    ctx.lineWidth = 2; ctx.beginPath();
    vals.forEach((v, vi) => {{
      const x = pad.l + (W-pad.l-pad.r)*vi/(maxLen-1||1);
      const y = pad.t + (H-pad.t-pad.b)*(1-v/maxVal);
      vi===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }});
    ctx.stroke();
    // Label
    const lastV = vals[vals.length-1];
    const lx = pad.l + (W-pad.l-pad.r);
    const ly = pad.t + (H-pad.t-pad.b)*(1-lastV/maxVal);
    ctx.fillStyle = colors[ai % colors.length];
    ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(aid, lx+4, ly+4);
  }});

  // Title
  ctx.fillStyle = '#c9d1d9'; ctx.font = 'bold 14px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('Cumulative Influence Over Rounds', W/2, 20);
  ctx.fillStyle = '#8b949e'; ctx.font = '12px sans-serif';
  ctx.fillText('Round', W/2, H-8);
}}

// Rounds tab
(function() {{
  const el = document.getElementById('tab-rounds');
  let html = '<h2>Round Log</h2><table><tr><th>Round</th><th>Leader</th><th>Result</th><th>Aggregate</th><th>Threshold</th><th>Accepts</th><th>Rejects</th><th>Slashed</th></tr>';
  DATA.rounds.forEach(r => {{
    const result = r.committed ? '<span style="color:var(--green)">✅ Committed</span>' : '<span style="color:var(--red)">❌ Failed</span>';
    html += `<tr><td>${{r.round}}</td><td>${{r.leader}}</td><td>${{result}}</td><td>${{r.aggregate}}</td><td>${{r.threshold}}</td><td>${{r.accepts}}</td><td>${{r.rejects}}</td><td>${{r.slashed.join(', ')||'-'}}</td></tr>`;
  }});
  html += '</table>';
  el.innerHTML = html;
}})();

// Recs tab
(function() {{
  const el = document.getElementById('tab-recs');
  let html = '<h2>Proactive Recommendations</h2>';
  DATA.recommendations.forEach(r => html += `<div class="rec">${{r}}</div>`);
  el.innerHTML = html;
}})();
</script>
</body>
</html>"""


# ── CLI Entry Point ────────────────────────────────────────────────────

async def run_demo(n_agents: int = 6, n_rounds: int = 4, n_byzantine: int = 1) -> ForensicReport:
    """Run a demo consensus and analyze forensically."""
    agents = []
    answers = ["42", "42", "42", "41", "43", "42", "42", "42"]
    for i in range(n_agents):
        is_byz = i >= n_agents - n_byzantine
        agents.append(MockAgent(
            agent_id=f"agent-{i}",
            answer=answers[i % len(answers)] if not is_byz else "999",
            confidence=round(random.uniform(0.5, 0.95), 2),
            byzantine=is_byz,
        ))

    engine = MBFTEngine(agents=agents, threshold=1.5, max_rounds=n_rounds)
    await engine.run("What is the answer to life?")

    analyzer = ForensicsAnalyzer(engine.history, engine.reputation)
    return analyzer.analyze()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Forensics Analyzer — autonomous post-run forensic investigation"
    )
    parser.add_argument("--demo", action="store_true", help="Run demo consensus and analyze")
    parser.add_argument("--agents", type=int, default=6, help="Number of agents in demo (default: 6)")
    parser.add_argument("--rounds", type=int, default=4, help="Max rounds in demo (default: 4)")
    parser.add_argument("--byzantine", type=int, default=1, help="Number of Byzantine agents (default: 1)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output HTML report path")
    args = parser.parse_args()

    if not args.demo:
        print("Usage: python -m src.forensics --demo [--agents N] [--rounds N] [--byzantine N] [-o report.html]")
        print("\nRun --demo to execute a sample consensus and perform forensic analysis.")
        return

    report = asyncio.run(run_demo(args.agents, args.rounds, args.byzantine))

    # Print summary
    print("=" * 60)
    print("🔬 CONSENSUS FORENSICS REPORT")
    print("=" * 60)

    print(f"\n📊 Rounds analyzed: {len(report.round_summaries)}")
    print(f"👥 Agents profiled: {len(report.agent_profiles)}")
    print(f"🏛️  Voting blocs: {len(report.blocs)}")
    print(f"⚠️  Findings: {len(report.findings)}")

    print("\n── Agent Profiles ──")
    for aid, p in sorted(report.agent_profiles.items()):
        bar = "█" * int(p.influence_score * 20)
        print(f"  {aid:12s}  influence={p.influence_score:.3f} {bar}  "
              f"conf={p.avg_confidence:.2f}  leader={p.times_leader}  slash={p.times_slashed}")

    if report.blocs:
        print("\n── Voting Blocs ──")
        for b in report.blocs:
            print(f"  {b.label}: {', '.join(b.members)}  (agreement={b.agreement_rate:.0%})")

    if report.findings:
        print("\n── Findings ──")
        for f in report.findings:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(f.severity, "⚪")
            print(f"  {icon} [{f.category}] {f.title}")
            print(f"     {f.detail}")

    print("\n── Recommendations ──")
    for r in report.recommendations:
        print(f"  {r}")

    # HTML report
    output = args.output or "forensics_report.html"
    html = generate_html_report(report)
    Path(output).write_text(html, encoding="utf-8")
    print(f"\n📄 Interactive report saved to: {output}")


if __name__ == "__main__":
    main()
