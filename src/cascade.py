"""Consensus Information Cascade Detector — detect herding behavior in mBFT.

Information cascades occur when agents abandon their private signals (their
own proposals) and blindly follow the leader/majority.  This is a well-known
failure mode in group decision-making (Banerjee 1992, Bikhchandani et al. 1992)
that undermines the epistemic diversity mBFT relies on.

Detection channels:
1. **Proposal Diversity Collapse** — unique-solution ratio drops across rounds
2. **Confidence Herding** — vote confidences cluster near the leader's value
3. **Flip-Flop Detection** — agents switch from rejection to acceptance without
   new evidence (counter-proof present → absent between rounds)
4. **Echo Chamber Index** — fraction of agents whose votes mirror the leader
   without generating counter-proofs
5. **Cascade Velocity** — how quickly the fleet converges to unanimity
6. **Private Signal Abandonment** — agents whose proposals differ from the
   leader but who vote to accept anyway (ignoring their own evidence)

Generates interactive HTML reports with charts and proactive recommendations.

Usage:
    python -m src.cascade [--runs N] [--agents N] [--threshold F]
                          [--cascade-agents N] [--auto-monitor]
                          [--interval N] [--output FILE]
"""
from __future__ import annotations

import asyncio
import argparse
import html as html_mod
import io
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.state import Proposal, RoundResult


# ── Data Structures ──────────────────────────────────────────────


@dataclass
class CascadeSignal:
    """A single detected cascade indicator."""
    name: str
    severity: str          # info / warning / critical
    score: float           # 0.0 – 1.0
    description: str
    recommendation: str


@dataclass
class AgentCascadeProfile:
    """Per-agent cascade behaviour summary."""
    agent_id: str
    flip_flop_count: int = 0
    echo_count: int = 0
    signal_abandonment_count: int = 0
    avg_confidence_deviation: float = 0.0
    cascade_susceptibility: float = 0.0   # composite 0-1


@dataclass
class CascadeReport:
    """Full cascade analysis for a batch of consensus runs."""
    signals: List[CascadeSignal] = field(default_factory=list)
    agent_profiles: List[AgentCascadeProfile] = field(default_factory=list)
    diversity_timeline: List[float] = field(default_factory=list)
    confidence_spread_timeline: List[float] = field(default_factory=list)
    echo_index_timeline: List[float] = field(default_factory=list)
    cascade_velocity: float = 0.0
    overall_cascade_risk: float = 0.0
    total_rounds: int = 0
    total_runs: int = 0


# ── Analysis Functions ───────────────────────────────────────────


def _diversity_ratio(proposals: List[Proposal]) -> float:
    """Fraction of unique solutions among proposals."""
    if not proposals:
        return 1.0
    unique = len({p.solution for p in proposals})
    return unique / len(proposals)


def _confidence_spread(votes_weights: List[float]) -> float:
    """Standard deviation of vote weights (measures clustering)."""
    if len(votes_weights) < 2:
        return 0.0
    return statistics.stdev(votes_weights)


def _echo_index(votes_weights: List[float]) -> float:
    """Fraction of votes that are positive (echo the leader)."""
    if not votes_weights:
        return 0.0
    positives = sum(1 for w in votes_weights if w > 0)
    return positives / len(votes_weights)


def _detect_flip_flops(
    history: List[RoundResult],
) -> Dict[str, int]:
    """Count per-agent sign flips between consecutive rounds."""
    flips: Dict[str, int] = defaultdict(int)
    prev_signs: Dict[str, int] = {}
    for result in history:
        curr_signs: Dict[str, int] = {}
        for v in result.votes:
            sign = 1 if v.weight >= 0 else -1
            curr_signs[v.voter_id] = sign
            if v.voter_id in prev_signs and prev_signs[v.voter_id] != sign:
                # Flip from reject → accept without counter-proof = cascade
                if sign == 1 and v.counter_proof is None:
                    flips[v.voter_id] += 1
        prev_signs = curr_signs
    return dict(flips)


def _signal_abandonment(
    proposals: List[Proposal],
    leader_id: str,
    votes: list,
) -> List[str]:
    """Find agents whose proposal differed from leader but voted to accept."""
    leader_solution = None
    for p in proposals:
        if p.agent_id == leader_id:
            leader_solution = p.solution
            break
    if leader_solution is None:
        return []

    dissenters = {
        p.agent_id for p in proposals
        if p.agent_id != leader_id and p.solution != leader_solution
    }
    abandoners = []
    for v in votes:
        if v.voter_id in dissenters and v.weight > 0:
            abandoners.append(v.voter_id)
    return abandoners


def _cascade_velocity(echo_timeline: List[float]) -> float:
    """Rate of convergence toward unanimity (slope of echo index)."""
    if len(echo_timeline) < 2:
        return 0.0
    n = len(echo_timeline)
    x_mean = (n - 1) / 2
    y_mean = statistics.mean(echo_timeline)
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(echo_timeline))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def analyze_cascade(
    all_histories: List[List[RoundResult]],
    all_proposals: Optional[List[List[List[Proposal]]]] = None,
) -> CascadeReport:
    """Run full cascade analysis across multiple consensus runs.

    Parameters
    ----------
    all_histories : list of list of RoundResult
        Each inner list is one run's round history.
    all_proposals : optional list of list of list of Proposal
        Per-run, per-round proposal lists.  When ``None`` the analysis
        skips proposal-dependent checks (diversity, signal abandonment).
    """
    report = CascadeReport()
    agent_flips: Dict[str, int] = defaultdict(int)
    agent_echoes: Dict[str, int] = defaultdict(int)
    agent_abandon: Dict[str, int] = defaultdict(int)
    agent_conf_devs: Dict[str, List[float]] = defaultdict(list)
    agent_rounds: Dict[str, int] = defaultdict(int)

    for run_idx, history in enumerate(all_histories):
        proposals_per_round = (
            all_proposals[run_idx] if all_proposals and run_idx < len(all_proposals) else None
        )

        # Flip-flop detection
        flips = _detect_flip_flops(history)
        for aid, cnt in flips.items():
            agent_flips[aid] += cnt

        for r_idx, result in enumerate(history):
            report.total_rounds += 1
            weights = [v.weight for v in result.votes]

            # Echo index
            ei = _echo_index(weights)
            report.echo_index_timeline.append(ei)
            for v in result.votes:
                agent_rounds[v.voter_id] += 1
                if v.weight > 0:
                    agent_echoes[v.voter_id] += 1

            # Confidence spread
            abs_weights = [abs(w) for w in weights]
            spread = _confidence_spread(abs_weights)
            report.confidence_spread_timeline.append(spread)

            # Leader confidence deviation
            leader_conf = None
            for v in result.votes:
                if v.voter_id == result.leader_id:
                    leader_conf = abs(v.weight)
                    break
            if leader_conf is not None:
                for v in result.votes:
                    if v.voter_id != result.leader_id:
                        dev = abs(abs(v.weight) - leader_conf)
                        agent_conf_devs[v.voter_id].append(dev)

            # Diversity (if proposals available)
            if proposals_per_round and r_idx < len(proposals_per_round):
                props = proposals_per_round[r_idx]
                dr = _diversity_ratio(props)
                report.diversity_timeline.append(dr)

                # Signal abandonment
                abandoners = _signal_abandonment(
                    props, result.leader_id, result.votes
                )
                for aid in abandoners:
                    agent_abandon[aid] += 1

    report.total_runs = len(all_histories)

    # Cascade velocity
    report.cascade_velocity = _cascade_velocity(report.echo_index_timeline)

    # Build agent profiles
    all_agents = set(agent_rounds.keys())
    for aid in sorted(all_agents):
        rounds = max(agent_rounds.get(aid, 1), 1)
        devs = agent_conf_devs.get(aid, [])
        avg_dev = statistics.mean(devs) if devs else 0.5

        profile = AgentCascadeProfile(
            agent_id=aid,
            flip_flop_count=agent_flips.get(aid, 0),
            echo_count=agent_echoes.get(aid, 0),
            signal_abandonment_count=agent_abandon.get(aid, 0),
            avg_confidence_deviation=avg_dev,
        )
        # Composite susceptibility: weighted sum of normalized indicators
        echo_rate = profile.echo_count / rounds
        abandon_rate = profile.signal_abandonment_count / max(rounds, 1)
        flip_rate = profile.flip_flop_count / max(rounds - 1, 1)
        conformity = 1.0 - min(avg_dev, 1.0)  # low deviation = high conformity
        profile.cascade_susceptibility = min(1.0, (
            0.30 * echo_rate +
            0.30 * abandon_rate +
            0.20 * flip_rate +
            0.20 * conformity
        ))
        report.agent_profiles.append(profile)

    # Generate signals
    report.signals = _generate_signals(report)

    # Overall risk
    if report.signals:
        max_sev = {"info": 0.3, "warning": 0.6, "critical": 1.0}
        risk_scores = [s.score * max_sev.get(s.severity, 0.5) for s in report.signals]
        report.overall_cascade_risk = min(1.0, statistics.mean(risk_scores) * 1.5)
    else:
        report.overall_cascade_risk = 0.0

    return report


def _generate_signals(report: CascadeReport) -> List[CascadeSignal]:
    """Generate cascade warning signals from analysis data."""
    signals: List[CascadeSignal] = []

    # 1. Proposal diversity collapse
    if report.diversity_timeline:
        avg_div = statistics.mean(report.diversity_timeline)
        if avg_div < 0.3:
            signals.append(CascadeSignal(
                name="Diversity Collapse",
                severity="critical",
                score=1.0 - avg_div,
                description=(
                    f"Average proposal diversity is {avg_div:.2f} — agents are "
                    f"converging on identical solutions prematurely."
                ),
                recommendation=(
                    "Introduce diversity incentives: reward novel proposals, "
                    "add noise to agent prompts, or rotate leader exclusions."
                ),
            ))
        elif avg_div < 0.6:
            signals.append(CascadeSignal(
                name="Diversity Decline",
                severity="warning",
                score=1.0 - avg_div,
                description=(
                    f"Proposal diversity at {avg_div:.2f} is below healthy levels."
                ),
                recommendation=(
                    "Monitor for further decline. Consider adding independent "
                    "proposal generation constraints."
                ),
            ))

    # 2. Confidence herding
    if report.confidence_spread_timeline:
        avg_spread = statistics.mean(report.confidence_spread_timeline)
        if avg_spread < 0.05:
            signals.append(CascadeSignal(
                name="Confidence Herding",
                severity="critical",
                score=max(0.0, 1.0 - avg_spread * 20),
                description=(
                    f"Vote confidence spread is extremely low ({avg_spread:.3f}). "
                    f"Agents are anchoring to the leader's confidence."
                ),
                recommendation=(
                    "Add confidence calibration noise, use blind voting, or "
                    "hide the leader's confidence until after vote submission."
                ),
            ))
        elif avg_spread < 0.15:
            signals.append(CascadeSignal(
                name="Confidence Clustering",
                severity="warning",
                score=max(0.0, 1.0 - avg_spread * 7),
                description=(
                    f"Vote confidence spread ({avg_spread:.3f}) suggests mild herding."
                ),
                recommendation=(
                    "Consider sequential revelation: agents submit confidence "
                    "before seeing others' values."
                ),
            ))

    # 3. Echo chamber
    if report.echo_index_timeline:
        avg_echo = statistics.mean(report.echo_index_timeline)
        if avg_echo > 0.9:
            signals.append(CascadeSignal(
                name="Echo Chamber",
                severity="critical",
                score=avg_echo,
                description=(
                    f"Echo index is {avg_echo:.2f} — {avg_echo*100:.0f}% of votes "
                    f"agree with the leader. Dissent is virtually absent."
                ),
                recommendation=(
                    "Mandate devil's advocate roles, require at least one "
                    "rejection per round, or introduce adversarial agents."
                ),
            ))
        elif avg_echo > 0.75:
            signals.append(CascadeSignal(
                name="High Agreement",
                severity="warning",
                score=avg_echo,
                description=(
                    f"Echo index at {avg_echo:.2f} may indicate social conformity."
                ),
                recommendation=(
                    "Add anonymous voting or require written justifications "
                    "for agreement votes."
                ),
            ))

    # 4. Cascade velocity
    if abs(report.cascade_velocity) > 0.1:
        sev = "critical" if report.cascade_velocity > 0.2 else "warning"
        signals.append(CascadeSignal(
            name="Rapid Convergence",
            severity=sev,
            score=min(1.0, abs(report.cascade_velocity) * 3),
            description=(
                f"Cascade velocity is {report.cascade_velocity:.3f} — "
                f"the fleet is converging toward unanimity {'rapidly' if sev == 'critical' else 'steadily'}."
            ),
            recommendation=(
                "Slow down consensus: add deliberation rounds, increase "
                "required quorum, or add cooling-off periods between rounds."
            ),
        ))

    # 5. Individual susceptibility
    susceptible = [p for p in report.agent_profiles if p.cascade_susceptibility > 0.7]
    if susceptible:
        names = ", ".join(p.agent_id for p in susceptible[:5])
        sev = "critical" if len(susceptible) > len(report.agent_profiles) / 2 else "warning"
        signals.append(CascadeSignal(
            name="Susceptible Agents",
            severity=sev,
            score=statistics.mean([p.cascade_susceptibility for p in susceptible]),
            description=(
                f"{len(susceptible)} agent(s) show high cascade susceptibility: {names}"
            ),
            recommendation=(
                "Retrain or replace susceptible agents. Consider assigning "
                "them contrarian roles to break cascade patterns."
            ),
        ))

    # 6. Signal abandonment
    total_abandon = sum(p.signal_abandonment_count for p in report.agent_profiles)
    if total_abandon > 0 and report.total_rounds > 0:
        rate = total_abandon / report.total_rounds
        if rate > 0.5:
            signals.append(CascadeSignal(
                name="Private Signal Abandonment",
                severity="critical",
                score=min(1.0, rate),
                description=(
                    f"Agents are abandoning their own proposals {rate:.1f}x per round "
                    f"on average — they propose one thing but vote for another."
                ),
                recommendation=(
                    "Strengthen commitment to private signals: penalize "
                    "vote-proposal inconsistency, or weight votes by "
                    "proposal-similarity."
                ),
            ))
        elif rate > 0.2:
            signals.append(CascadeSignal(
                name="Signal Weakening",
                severity="warning",
                score=min(1.0, rate * 2),
                description=(
                    f"Signal abandonment rate is {rate:.2f} per round."
                ),
                recommendation=(
                    "Track proposal-vote consistency and surface it in "
                    "agent reputation scores."
                ),
            ))

    if not signals:
        signals.append(CascadeSignal(
            name="No Cascade Detected",
            severity="info",
            score=0.0,
            description="No information cascade patterns detected. Fleet appears healthy.",
            recommendation="Continue monitoring. Cascade risk is currently low.",
        ))

    return signals


# ── Simulation ───────────────────────────────────────────────────


async def _simulate(
    num_runs: int,
    num_agents: int,
    threshold: float,
    num_cascade_agents: int,
) -> CascadeReport:
    """Run mBFT simulations and analyze for cascade patterns."""
    from src.agents.metacognitive import MockAgent
    from src.core.protocol import MBFTEngine

    all_histories: List[List[RoundResult]] = []
    all_proposals: List[List[List[Proposal]]] = []

    for run_i in range(num_runs):
        agents = []
        # Normal agents: each has their own answer
        for i in range(num_agents - num_cascade_agents):
            agents.append(MockAgent(
                agent_id=f"independent-{i}",
                answer=f"solution-{i}",
                confidence=0.5 + (i % 5) * 0.1,
            ))
        # Cascade-prone agents: copy the first agent's answer (herding)
        leader_answer = agents[0].answer if agents else "solution-0"
        for i in range(num_cascade_agents):
            agents.append(MockAgent(
                agent_id=f"herder-{i}",
                answer=leader_answer,
                confidence=agents[0].confidence if agents else 0.7,
                accept_set={leader_answer},
            ))

        engine = MBFTEngine(agents, threshold=threshold, max_rounds=4)
        await engine.run(f"task-{run_i}")

        all_histories.append(engine.history)

        # Collect proposals per round
        run_proposals: List[List[Proposal]] = []
        for _ in engine.history:
            props = []
            for a in agents:
                p = await a.generate_proposal(f"task-{run_i}")
                props.append(p)
            run_proposals.append(props)
        all_proposals.append(run_proposals)

    return analyze_cascade(all_histories, all_proposals)


# ── HTML Report ──────────────────────────────────────────────────


def _render_html(report: CascadeReport) -> str:
    """Generate interactive HTML dashboard."""
    severity_colors = {
        "info": "#3b82f6",
        "warning": "#f59e0b",
        "critical": "#ef4444",
    }
    risk_color = (
        "#ef4444" if report.overall_cascade_risk > 0.6
        else "#f59e0b" if report.overall_cascade_risk > 0.3
        else "#22c55e"
    )
    risk_label = (
        "HIGH" if report.overall_cascade_risk > 0.6
        else "MODERATE" if report.overall_cascade_risk > 0.3
        else "LOW"
    )

    signals_html = ""
    for s in report.signals:
        c = severity_colors.get(s.severity, "#6b7280")
        signals_html += f"""
        <div style="background:{c}15;border-left:4px solid {c};padding:12px 16px;
                    margin:8px 0;border-radius:0 8px 8px 0;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong style="color:{c};">{html_mod.escape(s.name)}</strong>
            <span style="background:{c};color:white;padding:2px 8px;border-radius:10px;
                         font-size:0.75em;text-transform:uppercase;">{s.severity}</span>
          </div>
          <div style="margin:6px 0;font-size:0.9em;color:#374151;">
            {html_mod.escape(s.description)}
          </div>
          <div style="font-size:0.85em;color:#6b7280;">
            💡 {html_mod.escape(s.recommendation)}
          </div>
          <div style="margin-top:4px;">
            <div style="background:#e5e7eb;border-radius:4px;height:6px;width:100%;">
              <div style="background:{c};height:6px;border-radius:4px;
                          width:{s.score*100:.0f}%;"></div>
            </div>
          </div>
        </div>"""

    # Agent table
    agent_rows = ""
    for p in sorted(report.agent_profiles, key=lambda x: -x.cascade_susceptibility):
        sc = p.cascade_susceptibility
        bar_color = "#ef4444" if sc > 0.7 else "#f59e0b" if sc > 0.4 else "#22c55e"
        agent_rows += f"""
        <tr>
          <td style="padding:8px;font-weight:600;">{html_mod.escape(p.agent_id)}</td>
          <td style="padding:8px;text-align:center;">{p.flip_flop_count}</td>
          <td style="padding:8px;text-align:center;">{p.echo_count}</td>
          <td style="padding:8px;text-align:center;">{p.signal_abandonment_count}</td>
          <td style="padding:8px;text-align:center;">{p.avg_confidence_deviation:.3f}</td>
          <td style="padding:8px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <div style="flex:1;background:#e5e7eb;border-radius:4px;height:8px;">
                <div style="background:{bar_color};height:8px;border-radius:4px;
                            width:{sc*100:.0f}%;"></div>
              </div>
              <span style="font-weight:600;color:{bar_color};">{sc:.2f}</span>
            </div>
          </td>
        </tr>"""

    # Canvas charts (echo + diversity + confidence spread)
    echo_json = json.dumps(report.echo_index_timeline[:100])
    div_json = json.dumps(report.diversity_timeline[:100])
    spread_json = json.dumps(report.confidence_spread_timeline[:100])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Information Cascade Detector — mBFT</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        background:#0f172a;color:#e2e8f0;padding:20px;}}
  .container{{max-width:1200px;margin:0 auto;}}
  h1{{font-size:1.8em;margin-bottom:4px;}}
  .subtitle{{color:#94a3b8;margin-bottom:20px;}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0;}}
  .card{{background:#1e293b;border-radius:12px;padding:16px;}}
  .card .label{{font-size:0.8em;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;}}
  .card .value{{font-size:1.8em;font-weight:700;margin-top:4px;}}
  .section{{background:#1e293b;border-radius:12px;padding:20px;margin:16px 0;}}
  .section h2{{font-size:1.2em;margin-bottom:12px;border-bottom:1px solid #334155;padding-bottom:8px;}}
  table{{width:100%;border-collapse:collapse;}}
  th{{padding:10px 8px;text-align:left;border-bottom:2px solid #334155;color:#94a3b8;
      font-size:0.8em;text-transform:uppercase;letter-spacing:0.5px;}}
  td{{border-bottom:1px solid #1e293b;}}
  tr:hover td{{background:#334155;}}
  canvas{{width:100%;height:200px;border-radius:8px;background:#0f172a;margin:8px 0;}}
  .risk-badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;
               font-size:0.9em;color:white;background:{risk_color};}}
</style>
</head>
<body>
<div class="container">
  <h1>🌊 Information Cascade Detector</h1>
  <p class="subtitle">mBFT consensus herding analysis — {report.total_runs} runs, {report.total_rounds} rounds</p>

  <div class="grid">
    <div class="card">
      <div class="label">Cascade Risk</div>
      <div class="value" style="color:{risk_color};">{report.overall_cascade_risk:.0%}</div>
      <span class="risk-badge">{risk_label}</span>
    </div>
    <div class="card">
      <div class="label">Cascade Velocity</div>
      <div class="value">{report.cascade_velocity:+.3f}</div>
      <div style="color:#94a3b8;font-size:0.8em;">convergence rate per round</div>
    </div>
    <div class="card">
      <div class="label">Avg Echo Index</div>
      <div class="value">{statistics.mean(report.echo_index_timeline) if report.echo_index_timeline else 0:.2f}</div>
      <div style="color:#94a3b8;font-size:0.8em;">leader agreement ratio</div>
    </div>
    <div class="card">
      <div class="label">Signals Detected</div>
      <div class="value">{len(report.signals)}</div>
      <div style="color:#94a3b8;font-size:0.8em;">
        {sum(1 for s in report.signals if s.severity=='critical')} critical,
        {sum(1 for s in report.signals if s.severity=='warning')} warning
      </div>
    </div>
  </div>

  <div class="section">
    <h2>🚨 Cascade Signals</h2>
    {signals_html}
  </div>

  <div class="section">
    <h2>📈 Echo Index Timeline</h2>
    <canvas id="echoChart"></canvas>
  </div>

  <div class="section">
    <h2>📊 Confidence Spread Timeline</h2>
    <canvas id="spreadChart"></canvas>
  </div>

  {"<div class='section'><h2>🎨 Proposal Diversity Timeline</h2><canvas id='divChart'></canvas></div>" if report.diversity_timeline else ""}

  <div class="section">
    <h2>🧬 Agent Cascade Profiles</h2>
    <table>
      <thead>
        <tr>
          <th>Agent</th><th>Flip-Flops</th><th>Echoes</th>
          <th>Signal Abandon</th><th>Avg Conf Dev</th><th>Susceptibility</th>
        </tr>
      </thead>
      <tbody>{agent_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>💡 Proactive Recommendations</h2>
    <ul style="list-style:none;padding:0;">
      <li style="padding:8px 0;border-bottom:1px solid #334155;">
        🛡️ <strong>Blind Voting Protocol</strong> — Hide leader confidence and identity
        until after votes are submitted to prevent anchoring.
      </li>
      <li style="padding:8px 0;border-bottom:1px solid #334155;">
        🎭 <strong>Devil's Advocate Rotation</strong> — Mandate at least one agent per
        round to argue against the leader's proposal.
      </li>
      <li style="padding:8px 0;border-bottom:1px solid #334155;">
        📊 <strong>Consistency Scoring</strong> — Track proposal-vote alignment in
        reputation scores to penalize signal abandonment.
      </li>
      <li style="padding:8px 0;border-bottom:1px solid #334155;">
        ⏱️ <strong>Deliberation Cooling</strong> — Add mandatory wait periods between
        rounds to prevent snap cascade convergence.
      </li>
      <li style="padding:8px 0;">
        🔀 <strong>Diversity Incentives</strong> — Bonus reputation for unique proposals
        that still pass verification.
      </li>
    </ul>
  </div>
</div>

<script>
function drawTimeline(canvasId, data, color, label) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas || !data.length) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = {{top:20,right:20,bottom:30,left:50}};
  const pW = W - pad.left - pad.right;
  const pH = H - pad.top - pad.bottom;
  const maxV = Math.max(...data, 1);
  const minV = Math.min(...data, 0);
  const range = maxV - minV || 1;

  // Grid
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = pad.top + (pH * i / 4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    ctx.fillStyle = '#94a3b8'; ctx.font = '11px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText((maxV - (range * i / 4)).toFixed(2), pad.left - 6, y + 4);
  }}

  // Line
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
  data.forEach((v, i) => {{
    const x = pad.left + (i / Math.max(data.length - 1, 1)) * pW;
    const y = pad.top + pH - ((v - minV) / range) * pH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();

  // Fill
  const lastX = pad.left + pW;
  ctx.lineTo(lastX, pad.top + pH);
  ctx.lineTo(pad.left, pad.top + pH);
  ctx.closePath();
  ctx.fillStyle = color + '20';
  ctx.fill();

  // Label
  ctx.fillStyle = '#e2e8f0'; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText(label, pad.left, pad.top - 6);
}}

drawTimeline('echoChart', {echo_json}, '#f59e0b', 'Echo Index (leader agreement)');
drawTimeline('spreadChart', {spread_json}, '#3b82f6', 'Confidence Spread (std dev)');
{"drawTimeline('divChart', " + div_json + ", '#22c55e', 'Proposal Diversity Ratio');" if report.diversity_timeline else ""}
</script>
</body>
</html>"""


# ── CLI Entry Point ──────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Consensus Information Cascade Detector",
    )
    p.add_argument("--runs", type=int, default=10, help="Number of simulation runs")
    p.add_argument("--agents", type=int, default=6, help="Total agents per run")
    p.add_argument("--threshold", type=float, default=0.6, help="Commit threshold")
    p.add_argument(
        "--cascade-agents", type=int, default=3,
        help="Number of cascade-prone (herder) agents",
    )
    p.add_argument("--auto-monitor", action="store_true", help="Continuous monitoring loop")
    p.add_argument("--interval", type=int, default=30, help="Monitor interval (seconds)")
    p.add_argument("--output", type=str, default=None, help="Output HTML file path")
    p.add_argument("--json", action="store_true", help="Output JSON instead of HTML")
    return p


def _report_to_dict(report: CascadeReport) -> dict:
    """Serialize report to JSON-safe dict."""
    return {
        "overall_cascade_risk": round(report.overall_cascade_risk, 4),
        "cascade_velocity": round(report.cascade_velocity, 4),
        "total_runs": report.total_runs,
        "total_rounds": report.total_rounds,
        "signals": [
            {
                "name": s.name,
                "severity": s.severity,
                "score": round(s.score, 4),
                "description": s.description,
                "recommendation": s.recommendation,
            }
            for s in report.signals
        ],
        "agent_profiles": [
            {
                "agent_id": p.agent_id,
                "flip_flop_count": p.flip_flop_count,
                "echo_count": p.echo_count,
                "signal_abandonment_count": p.signal_abandonment_count,
                "avg_confidence_deviation": round(p.avg_confidence_deviation, 4),
                "cascade_susceptibility": round(p.cascade_susceptibility, 4),
            }
            for p in report.agent_profiles
        ],
        "echo_index_timeline": [round(v, 4) for v in report.echo_index_timeline],
        "diversity_timeline": [round(v, 4) for v in report.diversity_timeline],
        "confidence_spread_timeline": [round(v, 4) for v in report.confidence_spread_timeline],
    }


async def _main() -> None:
    args = _build_parser().parse_args()

    if args.cascade_agents >= args.agents:
        print("Error: --cascade-agents must be less than --agents", file=sys.stderr)
        sys.exit(1)

    async def _run_once() -> CascadeReport:
        return await _simulate(
            num_runs=args.runs,
            num_agents=args.agents,
            threshold=args.threshold,
            num_cascade_agents=args.cascade_agents,
        )

    if args.auto_monitor:
        print(f"[cascade] Auto-monitor mode — interval {args.interval}s")
        iteration = 0
        while True:
            iteration += 1
            report = await _run_once()
            risk_icon = "🔴" if report.overall_cascade_risk > 0.6 else "🟡" if report.overall_cascade_risk > 0.3 else "🟢"
            print(
                f"[cascade] #{iteration} {risk_icon} risk={report.overall_cascade_risk:.2f} "
                f"velocity={report.cascade_velocity:+.3f} "
                f"signals={len(report.signals)} "
                f"echo={statistics.mean(report.echo_index_timeline) if report.echo_index_timeline else 0:.2f}"
            )
            for s in report.signals:
                if s.severity in ("warning", "critical"):
                    print(f"  ⚠️  {s.name}: {s.description}")
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(_render_html(report))
            await asyncio.sleep(args.interval)
    else:
        report = await _run_once()

        if args.json:
            print(json.dumps(_report_to_dict(report), indent=2))
        elif args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(_render_html(report))
            print(f"[cascade] Report written to {args.output}")
            risk_icon = "🔴" if report.overall_cascade_risk > 0.6 else "🟡" if report.overall_cascade_risk > 0.3 else "🟢"
            print(
                f"[cascade] {risk_icon} Overall cascade risk: {report.overall_cascade_risk:.0%}"
            )
        else:
            # Print text summary
            risk_icon = "🔴" if report.overall_cascade_risk > 0.6 else "🟡" if report.overall_cascade_risk > 0.3 else "🟢"
            print(f"\n{'='*60}")
            print(f"  🌊 Information Cascade Detector — mBFT")
            print(f"{'='*60}")
            print(f"  Runs: {report.total_runs}  |  Rounds: {report.total_rounds}")
            print(f"  {risk_icon} Overall Cascade Risk: {report.overall_cascade_risk:.0%}")
            print(f"  Cascade Velocity: {report.cascade_velocity:+.3f}")
            if report.echo_index_timeline:
                print(f"  Avg Echo Index: {statistics.mean(report.echo_index_timeline):.2f}")
            print(f"\n  Signals:")
            for s in report.signals:
                icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(s.severity, "•")
                print(f"    {icon} [{s.severity.upper()}] {s.name} ({s.score:.2f})")
                print(f"       {s.description}")
                print(f"       💡 {s.recommendation}")
            print(f"\n  Agent Susceptibility:")
            for p in sorted(report.agent_profiles, key=lambda x: -x.cascade_susceptibility):
                bar = "█" * int(p.cascade_susceptibility * 20) + "░" * (20 - int(p.cascade_susceptibility * 20))
                print(f"    {p.agent_id:20s} [{bar}] {p.cascade_susceptibility:.2f}")
            print(f"{'='*60}\n")


def main() -> None:
    # Ensure stdout handles Unicode (Windows cp1252 workaround)
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace('-', '') != 'utf8':
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace',
        )
    asyncio.run(_main())


if __name__ == "__main__":
    main()
