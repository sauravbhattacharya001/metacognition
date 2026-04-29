"""Consensus Governance Engine — autonomous self-governance via meta-consensus.

Agents propose and vote on protocol parameter changes (threshold, slash_factor,
max_rounds) using the mBFT protocol itself. The system evaluates proposals
through multi-round deliberation, tracks governance history, and generates
interactive HTML reports showing amendment timelines, faction voting patterns,
and parameter evolution.

Usage::

    python -m src.governance                     # interactive demo
    python -m src.governance --agents 7 --amendments 5
    python -m src.governance --export report.html
    python -m src.governance --export report.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass
class Amendment:
    """A proposed change to a protocol parameter."""
    amendment_id: int
    proposer: str
    parameter: str
    current_value: float
    proposed_value: float
    rationale: str
    round_results: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"  # pending | ratified | rejected
    vote_tally: float = 0.0
    threshold_at_vote: float = 0.0


@dataclass
class GovernanceState:
    """Mutable protocol parameters managed by governance."""
    threshold: float = 2.0
    slash_factor: float = 0.5
    max_rounds: int = 4

    def as_dict(self) -> Dict[str, float]:
        return {
            "threshold": self.threshold,
            "slash_factor": self.slash_factor,
            "max_rounds": float(self.max_rounds),
        }

    def apply(self, parameter: str, value: float) -> None:
        if parameter == "threshold":
            self.threshold = max(0.1, value)
        elif parameter == "slash_factor":
            self.slash_factor = max(0.05, min(0.95, value))
        elif parameter == "max_rounds":
            self.max_rounds = max(1, int(value))


# ---------------------------------------------------------------------------
# Governance agent with ideological bias
# ---------------------------------------------------------------------------

IDEOLOGIES = [
    ("conservative", "Prefers stability; votes against large changes"),
    ("progressive", "Favors experimentation; votes for bold changes"),
    ("moderate", "Evaluates on merit; slight bias toward small changes"),
    ("hawk", "Wants stricter thresholds and harsher slashing"),
    ("dove", "Wants lower thresholds and gentler slashing"),
]


@dataclass
class GovernanceAgent:
    agent_id: str
    ideology: str
    ideology_desc: str
    reputation: float = 1.0
    vote_history: List[Dict[str, Any]] = field(default_factory=list)

    def evaluate_amendment(self, amendment: Amendment, state: GovernanceState) -> Tuple[float, str]:
        """Return (vote_weight in [-1, 1], reasoning)."""
        delta = amendment.proposed_value - amendment.current_value
        magnitude = abs(delta) / max(abs(amendment.current_value), 0.01)

        # Base evaluation: is the change reasonable?
        base = 0.0
        reasoning_parts = []

        # Parameter-specific logic
        if amendment.parameter == "threshold":
            if amendment.proposed_value < 0.5:
                base -= 0.4
                reasoning_parts.append("threshold too low for safety")
            elif amendment.proposed_value > len(state.as_dict()) * 3:
                base -= 0.3
                reasoning_parts.append("threshold unreachably high")
            else:
                base += 0.2
                reasoning_parts.append("threshold in reasonable range")
        elif amendment.parameter == "slash_factor":
            if amendment.proposed_value < 0.1:
                base -= 0.3
                reasoning_parts.append("slash too aggressive")
            elif amendment.proposed_value > 0.9:
                base -= 0.2
                reasoning_parts.append("slash too lenient")
            else:
                base += 0.2
                reasoning_parts.append("slash factor balanced")
        elif amendment.parameter == "max_rounds":
            if amendment.proposed_value < 2:
                base -= 0.4
                reasoning_parts.append("too few rounds for deliberation")
            elif amendment.proposed_value > 10:
                base -= 0.2
                reasoning_parts.append("excessive rounds slow consensus")
            else:
                base += 0.2
                reasoning_parts.append("round count reasonable")

        # Ideological bias
        if self.ideology == "conservative":
            bias = -0.3 * magnitude
            reasoning_parts.append(f"conservative caution (Δ={magnitude:.1%})")
        elif self.ideology == "progressive":
            bias = 0.3 * min(magnitude, 1.0)
            reasoning_parts.append("progressive openness to change")
        elif self.ideology == "moderate":
            bias = -0.1 * magnitude if magnitude > 0.5 else 0.1
            reasoning_parts.append("moderate merit evaluation")
        elif self.ideology == "hawk":
            if amendment.parameter == "threshold" and delta > 0:
                bias = 0.3
                reasoning_parts.append("hawk favors stricter threshold")
            elif amendment.parameter == "slash_factor" and delta < 0:
                bias = 0.3
                reasoning_parts.append("hawk favors harsher slashing")
            else:
                bias = -0.1
                reasoning_parts.append("hawk skeptical of relaxation")
        elif self.ideology == "dove":
            if amendment.parameter == "threshold" and delta < 0:
                bias = 0.3
                reasoning_parts.append("dove favors lower threshold")
            elif amendment.parameter == "slash_factor" and delta > 0:
                bias = 0.3
                reasoning_parts.append("dove favors gentler slashing")
            else:
                bias = -0.1
                reasoning_parts.append("dove skeptical of strictness")
        else:
            bias = 0.0

        # Reputation modulates conviction
        noise = random.gauss(0, 0.1)
        weight = max(-1.0, min(1.0, (base + bias + noise) * self.reputation))

        reasoning = "; ".join(reasoning_parts)
        self.vote_history.append({
            "amendment_id": amendment.amendment_id,
            "weight": weight,
            "reasoning": reasoning,
        })
        return weight, reasoning


# ---------------------------------------------------------------------------
# Governance Engine
# ---------------------------------------------------------------------------


class GovernanceEngine:
    """Runs governance rounds where agents deliberate on parameter changes."""

    def __init__(self, n_agents: int = 5, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
        self.state = GovernanceState()
        self.agents = self._create_agents(n_agents)
        self.amendments: List[Amendment] = []
        self.parameter_history: List[Dict[str, Any]] = [
            {"step": 0, "event": "genesis", **self.state.as_dict()}
        ]

    def _create_agents(self, n: int) -> List[GovernanceAgent]:
        agents = []
        for i in range(n):
            ideology, desc = IDEOLOGIES[i % len(IDEOLOGIES)]
            agents.append(GovernanceAgent(
                agent_id=f"gov-{i:02d}",
                ideology=ideology,
                ideology_desc=desc,
            ))
        return agents

    def propose_amendment(self, proposer: Optional[GovernanceAgent] = None) -> Amendment:
        """Generate a random but contextually sensible amendment."""
        if proposer is None:
            proposer = random.choice(self.agents)

        param = random.choice(["threshold", "slash_factor", "max_rounds"])
        current = getattr(self.state, param)
        current_f = float(current)

        # Generate proposed value based on ideology
        if proposer.ideology == "hawk":
            if param == "threshold":
                proposed = current_f * random.uniform(1.05, 1.4)
            elif param == "slash_factor":
                proposed = current_f * random.uniform(0.6, 0.95)
            else:
                proposed = current_f + random.choice([-1, 0, 1])
        elif proposer.ideology == "dove":
            if param == "threshold":
                proposed = current_f * random.uniform(0.7, 0.95)
            elif param == "slash_factor":
                proposed = current_f * random.uniform(1.05, 1.4)
            else:
                proposed = current_f + random.choice([0, 1, 2])
        elif proposer.ideology == "progressive":
            proposed = current_f * random.uniform(0.5, 1.5)
        elif proposer.ideology == "conservative":
            proposed = current_f * random.uniform(0.9, 1.1)
        else:
            proposed = current_f * random.uniform(0.7, 1.3)

        if param == "max_rounds":
            proposed = max(1, round(proposed))
        else:
            proposed = round(proposed, 3)

        rationales = {
            "threshold": f"Adjust consensus threshold from {current_f} to {proposed} for {'tighter' if proposed > current_f else 'more flexible'} consensus",
            "slash_factor": f"Change slash factor from {current_f} to {proposed} to {'punish' if proposed < current_f else 'rehabilitate'} faulty leaders {'more' if proposed < current_f else 'less'} aggressively",
            "max_rounds": f"Set max rounds to {int(proposed)} from {int(current_f)} for {'deeper' if proposed > current_f else 'faster'} deliberation",
        }

        amendment = Amendment(
            amendment_id=len(self.amendments) + 1,
            proposer=proposer.agent_id,
            parameter=param,
            current_value=current_f,
            proposed_value=proposed,
            rationale=rationales[param],
        )
        return amendment

    def deliberate(self, amendment: Amendment, verbose: bool = False) -> Amendment:
        """Run a governance vote on the amendment."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Amendment #{amendment.amendment_id}: {amendment.rationale}")
            print(f"  Parameter: {amendment.parameter}")
            print(f"  Current → Proposed: {amendment.current_value} → {amendment.proposed_value}")
            print(f"  Proposed by: {amendment.proposer}")
            print(f"{'='*60}")

        votes: List[Dict[str, Any]] = []
        total_weight = 0.0

        for agent in self.agents:
            weight, reasoning = agent.evaluate_amendment(amendment, self.state)
            effective = weight * agent.reputation
            total_weight += effective
            vote_record = {
                "agent_id": agent.agent_id,
                "ideology": agent.ideology,
                "weight": round(weight, 3),
                "effective_weight": round(effective, 3),
                "reputation": round(agent.reputation, 3),
                "reasoning": reasoning,
            }
            votes.append(vote_record)
            if verbose:
                symbol = "✓" if weight > 0 else "✗" if weight < 0 else "—"
                print(f"  {symbol} {agent.agent_id} ({agent.ideology}): "
                      f"w={weight:+.3f} eff={effective:+.3f} | {reasoning}")

        amendment.round_results = votes
        amendment.vote_tally = round(total_weight, 3)
        amendment.threshold_at_vote = self.state.threshold

        # Ratification: tally must exceed a governance quorum (half of agent count)
        quorum = len(self.agents) * 0.3
        ratified = total_weight >= quorum

        if ratified:
            amendment.status = "ratified"
            self.state.apply(amendment.parameter, amendment.proposed_value)
            self.parameter_history.append({
                "step": len(self.amendments) + 1,
                "event": f"amendment-{amendment.amendment_id}",
                **self.state.as_dict(),
            })
            if verbose:
                print(f"\n  ✅ RATIFIED (tally={total_weight:.3f} ≥ quorum={quorum:.1f})")
                print(f"  New {amendment.parameter} = {getattr(self.state, amendment.parameter)}")
        else:
            amendment.status = "rejected"
            if verbose:
                print(f"\n  ❌ REJECTED (tally={total_weight:.3f} < quorum={quorum:.1f})")

        self.amendments.append(amendment)
        return amendment

    def run_session(self, n_amendments: int = 5, verbose: bool = False) -> Dict[str, Any]:
        """Run a full governance session with multiple amendments."""
        if verbose:
            print("\n" + "╔" + "═"*58 + "╗")
            print("║  mBFT Consensus Governance Engine                        ║")
            print("╚" + "═"*58 + "╝")
            print(f"\nAgents: {len(self.agents)}")
            for a in self.agents:
                print(f"  • {a.agent_id}: {a.ideology} — {a.ideology_desc}")
            print(f"\nInitial parameters: {self.state.as_dict()}")

        for _ in range(n_amendments):
            amendment = self.propose_amendment()
            self.deliberate(amendment, verbose=verbose)

        # Compute faction analysis
        factions = self._analyze_factions()

        summary = {
            "session_time": datetime.now(timezone.utc).isoformat(),
            "n_agents": len(self.agents),
            "n_amendments": len(self.amendments),
            "ratified": sum(1 for a in self.amendments if a.status == "ratified"),
            "rejected": sum(1 for a in self.amendments if a.status == "rejected"),
            "final_parameters": self.state.as_dict(),
            "parameter_history": self.parameter_history,
            "factions": factions,
            "amendments": [self._amendment_dict(a) for a in self.amendments],
            "agents": [self._agent_dict(a) for a in self.agents],
        }

        if verbose:
            print(f"\n{'═'*60}")
            print("SESSION SUMMARY")
            print(f"  Ratified: {summary['ratified']}/{summary['n_amendments']}")
            print(f"  Final parameters: {summary['final_parameters']}")
            print(f"\n  Faction Analysis:")
            for f in factions:
                print(f"    {f['ideology']}: avg_vote={f['avg_vote']:+.3f} "
                      f"influence={f['influence']:.3f}")

        return summary

    def _analyze_factions(self) -> List[Dict[str, Any]]:
        """Analyze voting patterns by ideology."""
        ideology_votes: Dict[str, List[float]] = {}
        for agent in self.agents:
            if agent.ideology not in ideology_votes:
                ideology_votes[agent.ideology] = []
            for vh in agent.vote_history:
                ideology_votes[agent.ideology].append(vh["weight"])

        factions = []
        for ideology, votes in ideology_votes.items():
            avg = sum(votes) / len(votes) if votes else 0.0
            # Influence = consistency * magnitude
            if len(votes) > 1:
                mean = avg
                variance = sum((v - mean) ** 2 for v in votes) / len(votes)
                consistency = 1.0 / (1.0 + math.sqrt(variance))
            else:
                consistency = 1.0
            influence = abs(avg) * consistency
            factions.append({
                "ideology": ideology,
                "avg_vote": round(avg, 3),
                "consistency": round(consistency, 3),
                "influence": round(influence, 3),
                "n_votes": len(votes),
            })
        return sorted(factions, key=lambda f: -f["influence"])

    def _amendment_dict(self, a: Amendment) -> Dict[str, Any]:
        return {
            "id": a.amendment_id,
            "proposer": a.proposer,
            "parameter": a.parameter,
            "current": a.current_value,
            "proposed": a.proposed_value,
            "rationale": a.rationale,
            "status": a.status,
            "tally": a.vote_tally,
            "votes": a.round_results,
        }

    def _agent_dict(self, a: GovernanceAgent) -> Dict[str, Any]:
        return {
            "id": a.agent_id,
            "ideology": a.ideology,
            "reputation": round(a.reputation, 3),
            "n_votes": len(a.vote_history),
            "avg_vote": round(
                sum(v["weight"] for v in a.vote_history) / len(a.vote_history), 3
            ) if a.vote_history else 0.0,
        }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def generate_html_report(summary: Dict[str, Any]) -> str:
    """Generate an interactive HTML governance report."""
    amendments_json = json.dumps(summary["amendments"], indent=2)
    history_json = json.dumps(summary["parameter_history"], indent=2)
    factions_json = json.dumps(summary["factions"], indent=2)
    agents_json = json.dumps(summary["agents"], indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mBFT Governance Report</title>
<style>
  :root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9;
           --accent: #58a6ff; --green: #3fb950; --red: #f85149; --yellow: #d29922; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont,
         'Segoe UI', sans-serif; padding: 2rem; }}
  h1 {{ color: var(--accent); margin-bottom: 0.5rem; }}
  h2 {{ color: var(--accent); margin: 1.5rem 0 0.75rem; font-size: 1.2rem; }}
  .subtitle {{ color: #8b949e; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }}
  .stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center; }}
  .stat .value {{ font-size: 2rem; font-weight: bold; color: var(--accent); }}
  .stat .label {{ color: #8b949e; font-size: 0.85rem; margin-top: 0.25rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
           padding: 1rem; margin: 0.5rem 0; }}
  .ratified {{ border-left: 3px solid var(--green); }}
  .rejected {{ border-left: 3px solid var(--red); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
  .badge-ratified {{ background: rgba(63,185,80,0.2); color: var(--green); }}
  .badge-rejected {{ background: rgba(248,81,73,0.2); color: var(--red); }}
  .vote-bar {{ height: 8px; border-radius: 4px; background: var(--border); margin: 4px 0; position: relative; overflow: hidden; }}
  .vote-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .vote-pos {{ background: var(--green); }}
  .vote-neg {{ background: var(--red); }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ padding: 0.5rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--accent); font-size: 0.85rem; }}
  canvas {{ width: 100%; height: 250px; margin: 1rem 0; }}
  .timeline {{ position: relative; padding-left: 2rem; }}
  .timeline::before {{ content: ''; position: absolute; left: 0.75rem; top: 0; bottom: 0;
                       width: 2px; background: var(--border); }}
  .timeline-item {{ position: relative; margin: 1rem 0; }}
  .timeline-item::before {{ content: ''; position: absolute; left: -1.6rem; top: 0.5rem;
                            width: 10px; height: 10px; border-radius: 50%; border: 2px solid var(--accent);
                            background: var(--bg); }}
  .timeline-item.ratified::before {{ background: var(--green); border-color: var(--green); }}
  .timeline-item.rejected::before {{ background: var(--red); border-color: var(--red); }}
  .tabs {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); margin: 1.5rem 0 1rem; }}
  .tab {{ padding: 0.5rem 1rem; cursor: pointer; border-bottom: 2px solid transparent; color: #8b949e; }}
  .tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
</style>
</head>
<body>
<h1>🏛️ mBFT Governance Report</h1>
<p class="subtitle">Consensus self-governance session — {summary['session_time'][:10]}</p>

<div class="grid">
  <div class="stat"><div class="value">{summary['n_agents']}</div><div class="label">Agents</div></div>
  <div class="stat"><div class="value">{summary['n_amendments']}</div><div class="label">Amendments</div></div>
  <div class="stat"><div class="value" style="color:var(--green)">{summary['ratified']}</div><div class="label">Ratified</div></div>
  <div class="stat"><div class="value" style="color:var(--red)">{summary['rejected']}</div><div class="label">Rejected</div></div>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('amendments')">Amendments</div>
  <div class="tab" onclick="switchTab('factions')">Factions</div>
  <div class="tab" onclick="switchTab('evolution')">Parameter Evolution</div>
  <div class="tab" onclick="switchTab('agents')">Agents</div>
</div>

<div id="amendments" class="tab-content active">
  <h2>Amendment Timeline</h2>
  <div class="timeline" id="timeline"></div>
</div>

<div id="factions" class="tab-content">
  <h2>Faction Analysis</h2>
  <table>
    <tr><th>Ideology</th><th>Avg Vote</th><th>Consistency</th><th>Influence</th><th>Votes Cast</th></tr>
  </table>
  <div id="faction-table"></div>
  <canvas id="factionChart"></canvas>
</div>

<div id="evolution" class="tab-content">
  <h2>Parameter Evolution</h2>
  <canvas id="paramChart"></canvas>
  <div id="param-table"></div>
</div>

<div id="agents" class="tab-content">
  <h2>Agent Roster</h2>
  <div id="agent-cards"></div>
</div>

<script>
const amendments = {amendments_json};
const history = {history_json};
const factions = {factions_json};
const agents = {agents_json};

function switchTab(id) {{
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
  if (id === 'evolution') drawParamChart();
  if (id === 'factions') drawFactionChart();
}}

// Timeline
const timeline = document.getElementById('timeline');
amendments.forEach(a => {{
  const isR = a.status === 'ratified';
  const div = document.createElement('div');
  div.className = 'timeline-item ' + a.status;
  const votesHtml = a.votes.map(v => {{
    const pct = ((v.weight + 1) / 2) * 100;
    const cls = v.weight >= 0 ? 'vote-pos' : 'vote-neg';
    return `<div style="display:flex;align-items:center;gap:8px;font-size:0.8rem;margin:2px 0">
      <span style="width:60px">${{v.agent_id}}</span>
      <div class="vote-bar" style="flex:1"><div class="vote-fill ${{cls}}" style="width:${{pct}}%"></div></div>
      <span style="width:50px;text-align:right">${{v.weight > 0 ? '+' : ''}}${{v.weight.toFixed(3)}}</span>
    </div>`;
  }}).join('');
  div.innerHTML = `<div class="card ${{a.status}}">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <strong>Amendment #${{a.id}}: ${{a.parameter}}</strong>
      <span class="badge badge-${{a.status}}">${{a.status.toUpperCase()}}</span>
    </div>
    <p style="color:#8b949e;margin:0.25rem 0;font-size:0.9rem">${{a.rationale}}</p>
    <p style="font-size:0.85rem">Tally: <strong>${{a.tally > 0 ? '+' : ''}}${{a.tally.toFixed(3)}}</strong> | Proposed by: ${{a.proposer}}</p>
    <div style="margin-top:0.5rem">${{votesHtml}}</div>
  </div>`;
  timeline.appendChild(div);
}});

// Faction table
const ftable = document.getElementById('faction-table');
factions.forEach(f => {{
  const row = document.createElement('div');
  row.className = 'card';
  row.innerHTML = `<div style="display:flex;justify-content:space-between">
    <strong>${{f.ideology}}</strong>
    <span>Avg: ${{f.avg_vote > 0 ? '+' : ''}}${{f.avg_vote.toFixed(3)}} | Consistency: ${{f.consistency.toFixed(3)}} | Influence: ${{f.influence.toFixed(3)}}</span>
  </div>`;
  ftable.appendChild(row);
}});

// Agent cards
const acards = document.getElementById('agent-cards');
agents.forEach(a => {{
  const div = document.createElement('div');
  div.className = 'card';
  div.innerHTML = `<strong>${{a.id}}</strong> <span class="badge" style="background:rgba(88,166,255,0.2);color:var(--accent)">${{a.ideology}}</span>
    <p style="font-size:0.85rem;margin-top:0.25rem">Reputation: ${{a.reputation.toFixed(3)}} | Votes: ${{a.n_votes}} | Avg: ${{a.avg_vote > 0 ? '+' : ''}}${{a.avg_vote.toFixed(3)}}</p>`;
  acards.appendChild(div);
}});

// Canvas charts
function drawParamChart() {{
  const canvas = document.getElementById('paramChart');
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth; canvas.height = 250;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const params = ['threshold', 'slash_factor', 'max_rounds'];
  const colors = ['#58a6ff', '#3fb950', '#d29922'];
  const maxVals = params.map(p => Math.max(...history.map(h => h[p])) * 1.2 || 1);

  params.forEach((param, pi) => {{
    ctx.strokeStyle = colors[pi];
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((h, i) => {{
      const x = (i / Math.max(history.length - 1, 1)) * (canvas.width - 60) + 30;
      const y = canvas.height - 30 - ((h[param] / maxVals[pi]) * (canvas.height - 60));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
    // Label
    ctx.fillStyle = colors[pi];
    ctx.font = '12px sans-serif';
    ctx.fillText(param, canvas.width - 100, 20 + pi * 18);
  }});
}}

function drawFactionChart() {{
  const canvas = document.getElementById('factionChart');
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth; canvas.height = 250;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const barW = (canvas.width - 60) / factions.length;
  const maxInf = Math.max(...factions.map(f => f.influence)) * 1.3 || 1;

  factions.forEach((f, i) => {{
    const x = 30 + i * barW;
    const h = (f.influence / maxInf) * (canvas.height - 60);
    ctx.fillStyle = f.avg_vote >= 0 ? '#3fb950' : '#f85149';
    ctx.fillRect(x + 10, canvas.height - 30 - h, barW - 20, h);
    ctx.fillStyle = '#c9d1d9';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(f.ideology, x + barW / 2, canvas.height - 12);
  }});
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Governance Engine — autonomous self-governance via meta-consensus"
    )
    parser.add_argument("--agents", type=int, default=5, help="Number of governance agents (default: 5)")
    parser.add_argument("--amendments", type=int, default=5, help="Number of amendments to propose (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--export", type=str, default=None, help="Export report (html or json)")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal output")
    args = parser.parse_args()

    engine = GovernanceEngine(n_agents=args.agents, seed=args.seed)
    summary = engine.run_session(n_amendments=args.amendments, verbose=not args.quiet)

    if args.export:
        path = Path(args.export)
        if path.suffix == ".json":
            path.write_text(json.dumps(summary, indent=2))
        else:
            path.write_text(generate_html_report(summary))
        if not args.quiet:
            print(f"\nExported to {path}")


if __name__ == "__main__":
    main()
