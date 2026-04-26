"""Autonomous Consensus Red Team — adversarial attack campaign testing for mBFT.

Runs 8 attack strategies across varying Byzantine ratios to systematically
probe protocol resilience.  Produces an interactive HTML report with heatmap,
vulnerability cards, and hardening recommendations.

Usage::

    python -m src.redteam --autopilot --output redteam_report.html
    python -m src.redteam --attack sybil_swarm --ratio 0.33
    python -m src.redteam --json results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.agents.base import BaseAgent
from src.core.protocol import MBFTEngine
from src.core.state import Proposal, Vote


# ---------------------------------------------------------------------------
# Attack strategies
# ---------------------------------------------------------------------------

class AttackStrategy(str, Enum):
    CONFIDENCE_INFLATION = "confidence_inflation"
    COORDINATED_REJECTION = "coordinated_rejection"
    REPUTATION_FARMING = "reputation_farming"
    SYBIL_SWARM = "sybil_swarm"
    FLIP_FLOP = "flip_flop"
    STEALTH_POISON = "stealth_poison"
    TARGETED_SLASH = "targeted_slash"
    ENTROPY_MAXIMIZER = "entropy_maximizer"


ATTACK_DESCRIPTIONS: Dict[AttackStrategy, str] = {
    AttackStrategy.CONFIDENCE_INFLATION: "Byzantine agents always propose with confidence=1.0 regardless of solution quality",
    AttackStrategy.COORDINATED_REJECTION: "Byzantine agents always vote -1.0 to block consensus",
    AttackStrategy.REPUTATION_FARMING: "Byzantines cooperate initially to build reputation, then defect",
    AttackStrategy.SYBIL_SWARM: "Flood the network with many low-quality Byzantine agents",
    AttackStrategy.FLIP_FLOP: "Byzantines alternate between cooperative and adversarial behavior",
    AttackStrategy.STEALTH_POISON: "Byzantines propose subtly wrong solutions with high confidence",
    AttackStrategy.TARGETED_SLASH: "Byzantines coordinate rejections against the highest-rep honest agent",
    AttackStrategy.ENTROPY_MAXIMIZER: "Byzantines randomize all behaviors to maximize protocol confusion",
}


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class HonestAgent(BaseAgent):
    """Always-correct cooperative agent."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)

    async def generate_proposal(self, task: str) -> Proposal:
        return Proposal(
            agent_id=self.id,
            solution=f"honest_solution_{self.id}",
            proof=f"valid_proof_{self.id}",
            confidence=0.7,
        )

    async def verify_proposal(self, leader_proposal: Proposal) -> Vote:
        # Honest agents approve honest proposals, reject poisoned ones
        is_honest = leader_proposal.solution.startswith("honest_solution")
        weight = 0.8 if is_honest else -0.6
        return Vote(
            voter_id=self.id,
            target_proposal_id=leader_proposal.proposal_id,
            weight=weight,
            counter_proof=None if is_honest else "detected_poison",
        )


class RedTeamAgent(BaseAgent):
    """Configurable adversarial agent."""

    def __init__(
        self,
        agent_id: str,
        strategy: AttackStrategy,
        *,
        farm_rounds: int = 3,
        target_id: Optional[str] = None,
    ) -> None:
        super().__init__(agent_id)
        self.strategy = strategy
        self.farm_rounds = farm_rounds
        self.target_id = target_id
        self._round = 0
        self._cooperative_phase = True

    async def generate_proposal(self, task: str) -> Proposal:
        self._round += 1

        if self.strategy == AttackStrategy.CONFIDENCE_INFLATION:
            return Proposal(agent_id=self.id, solution=f"byzantine_{self.id}", proof="fake", confidence=1.0)

        if self.strategy == AttackStrategy.REPUTATION_FARMING:
            if self._round <= self.farm_rounds:
                return Proposal(agent_id=self.id, solution=f"honest_solution_{self.id}", proof="valid", confidence=0.7)
            return Proposal(agent_id=self.id, solution=f"poisoned_{self.id}", proof="fake", confidence=0.95)

        if self.strategy == AttackStrategy.STEALTH_POISON:
            return Proposal(agent_id=self.id, solution=f"subtly_wrong_{self.id}", proof="plausible", confidence=0.85)

        if self.strategy == AttackStrategy.FLIP_FLOP:
            if self._round % 2 == 0:
                return Proposal(agent_id=self.id, solution=f"honest_solution_{self.id}", proof="valid", confidence=0.7)
            return Proposal(agent_id=self.id, solution=f"byzantine_{self.id}", proof="fake", confidence=0.9)

        if self.strategy == AttackStrategy.ENTROPY_MAXIMIZER:
            conf = random.uniform(0.0, 1.0)
            sol = random.choice([f"honest_solution_{self.id}", f"chaos_{self.id}", f"random_{random.randint(0,999)}"])
            return Proposal(agent_id=self.id, solution=sol, proof="random", confidence=conf)

        if self.strategy == AttackStrategy.SYBIL_SWARM:
            return Proposal(agent_id=self.id, solution=f"sybil_{self.id}", proof="weak", confidence=0.5)

        # Default for coordinated_rejection, targeted_slash — still propose honestly-ish
        return Proposal(agent_id=self.id, solution=f"byzantine_{self.id}", proof="fake", confidence=0.6)

    async def verify_proposal(self, leader_proposal: Proposal) -> Vote:
        if self.strategy == AttackStrategy.COORDINATED_REJECTION:
            return Vote(voter_id=self.id, target_proposal_id=leader_proposal.proposal_id, weight=-1.0, counter_proof="coordinated_block")

        if self.strategy == AttackStrategy.REPUTATION_FARMING:
            if self._round <= self.farm_rounds:
                return Vote(voter_id=self.id, target_proposal_id=leader_proposal.proposal_id, weight=0.8)
            return Vote(voter_id=self.id, target_proposal_id=leader_proposal.proposal_id, weight=-0.9, counter_proof="farmed_defection")

        if self.strategy == AttackStrategy.TARGETED_SLASH:
            if self.target_id and leader_proposal.agent_id == self.target_id:
                return Vote(voter_id=self.id, target_proposal_id=leader_proposal.proposal_id, weight=-1.0, counter_proof="targeted_rejection")
            return Vote(voter_id=self.id, target_proposal_id=leader_proposal.proposal_id, weight=0.5)

        if self.strategy == AttackStrategy.FLIP_FLOP:
            w = 0.7 if self._round % 2 == 0 else -0.8
            return Vote(voter_id=self.id, target_proposal_id=leader_proposal.proposal_id, weight=w)

        if self.strategy == AttackStrategy.ENTROPY_MAXIMIZER:
            w = random.uniform(-1.0, 1.0)
            return Vote(voter_id=self.id, target_proposal_id=leader_proposal.proposal_id, weight=w)

        if self.strategy == AttackStrategy.STEALTH_POISON:
            # Approve everything to avoid suspicion
            return Vote(voter_id=self.id, target_proposal_id=leader_proposal.proposal_id, weight=0.75)

        # confidence_inflation, sybil_swarm — approve own kind
        is_byzantine = not leader_proposal.solution.startswith("honest_solution")
        w = 0.9 if is_byzantine else 0.3
        return Vote(voter_id=self.id, target_proposal_id=leader_proposal.proposal_id, weight=w)


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    committed: bool
    rounds_used: int
    max_rounds: int
    leader_was_honest: bool
    false_commit: bool  # Byzantine solution committed
    reputation_accuracy: float  # honest agents have higher avg rep than byzantines


@dataclass
class ConfigResult:
    attack: AttackStrategy
    ratio: float
    total_agents: int
    byzantine_count: int
    trials: List[TrialResult]

    @property
    def commit_rate(self) -> float:
        return sum(1 for t in self.trials if t.committed) / max(len(self.trials), 1)

    @property
    def false_commit_rate(self) -> float:
        return sum(1 for t in self.trials if t.false_commit) / max(len(self.trials), 1)

    @property
    def avg_rounds(self) -> float:
        return statistics.mean(t.rounds_used for t in self.trials)

    @property
    def avg_rep_accuracy(self) -> float:
        return statistics.mean(t.reputation_accuracy for t in self.trials)

    @property
    def resilience_score(self) -> float:
        safe = 1.0 - self.false_commit_rate
        efficiency = 1.0 - (self.avg_rounds / max(self.trials[0].max_rounds, 1)) if self.trials else 0.5
        rep = self.avg_rep_accuracy
        return max(0.0, min(100.0, 100.0 * safe * (0.4 + 0.3 * efficiency + 0.3 * rep)))


@dataclass
class CampaignResult:
    configs: List[ConfigResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if not self.configs:
            return 0.0
        return statistics.mean(c.resilience_score for c in self.configs)

    @property
    def grade(self) -> str:
        s = self.overall_score
        if s >= 90:
            return "A"
        if s >= 80:
            return "B"
        if s >= 70:
            return "C"
        if s >= 55:
            return "D"
        return "F"

    def to_dict(self) -> Dict[str, Any]:
        rows = []
        for c in self.configs:
            rows.append({
                "attack": c.attack.value,
                "ratio": c.ratio,
                "total_agents": c.total_agents,
                "byzantine_count": c.byzantine_count,
                "commit_rate": round(c.commit_rate, 3),
                "false_commit_rate": round(c.false_commit_rate, 3),
                "avg_rounds": round(c.avg_rounds, 2),
                "avg_rep_accuracy": round(c.avg_rep_accuracy, 3),
                "resilience_score": round(c.resilience_score, 1),
            })
        return {
            "overall_score": round(self.overall_score, 1),
            "grade": self.grade,
            "results": rows,
        }


# ---------------------------------------------------------------------------
# Campaign runner
# ---------------------------------------------------------------------------

class RedTeamCampaign:
    DEFAULT_RATIOS = [0.1, 0.2, 0.33, 0.4, 0.5]

    def __init__(
        self,
        total_agents: int = 10,
        threshold: float = 3.0,
        max_rounds: int = 4,
        trials: int = 5,
    ) -> None:
        self.total_agents = total_agents
        self.threshold = threshold
        self.max_rounds = max_rounds
        self.trials = trials

    def _build_agents(
        self, attack: AttackStrategy, ratio: float
    ) -> tuple[List[BaseAgent], List[str], List[str]]:
        n_byz = max(1, int(self.total_agents * ratio))
        n_honest = self.total_agents - n_byz
        honest_ids = [f"honest_{i}" for i in range(n_honest)]
        byz_ids = [f"byz_{i}" for i in range(n_byz)]

        agents: List[BaseAgent] = [HonestAgent(hid) for hid in honest_ids]

        target = honest_ids[0] if honest_ids else None
        for bid in byz_ids:
            agents.append(RedTeamAgent(bid, attack, target_id=target))

        random.shuffle(agents)
        return agents, honest_ids, byz_ids

    async def run_trial(
        self, attack: AttackStrategy, ratio: float
    ) -> TrialResult:
        agents, honest_ids, byz_ids = self._build_agents(attack, ratio)
        engine = MBFTEngine(
            agents=agents,
            threshold=self.threshold,
            max_rounds=self.max_rounds,
        )
        result = await engine.run("red_team_test_task")

        committed = result is not None and result.committed
        rounds_used = len(engine.history)
        leader_honest = result.leader_id in honest_ids if result else False

        false_commit = False
        if committed and result and result.committed_solution:
            false_commit = not result.committed_solution.startswith("honest_solution")

        # Reputation accuracy: honest avg > byzantine avg
        rep = engine.reputation
        honest_rep = statistics.mean(rep.get(h, 0) for h in honest_ids) if honest_ids else 0
        byz_rep = statistics.mean(rep.get(b, 0) for b in byz_ids) if byz_ids else 0
        rep_accuracy = 1.0 if honest_rep > byz_rep else (0.5 if honest_rep == byz_rep else 0.0)

        return TrialResult(
            committed=committed,
            rounds_used=rounds_used,
            max_rounds=self.max_rounds,
            leader_was_honest=leader_honest,
            false_commit=false_commit,
            reputation_accuracy=rep_accuracy,
        )

    async def run_config(
        self, attack: AttackStrategy, ratio: float
    ) -> ConfigResult:
        trials = []
        for _ in range(self.trials):
            # Reset random seed per-trial would reduce variance, but we want natural variance
            t = await self.run_trial(attack, ratio)
            trials.append(t)
        n_byz = max(1, int(self.total_agents * ratio))
        return ConfigResult(
            attack=attack,
            ratio=ratio,
            total_agents=self.total_agents,
            byzantine_count=n_byz,
            trials=trials,
        )

    async def run_campaign(
        self,
        attacks: Optional[List[AttackStrategy]] = None,
        ratios: Optional[List[float]] = None,
    ) -> CampaignResult:
        attacks = attacks or list(AttackStrategy)
        ratios = ratios or self.DEFAULT_RATIOS
        campaign = CampaignResult()

        total = len(attacks) * len(ratios)
        done = 0
        for attack in attacks:
            for ratio in ratios:
                cfg = await self.run_config(attack, ratio)
                campaign.configs.append(cfg)
                done += 1
                pct = done * 100 // total
                print(f"  [{pct:3d}%] {attack.value} @ {ratio:.0%} → resilience {cfg.resilience_score:.1f}")

        return campaign


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _severity(score: float) -> str:
    if score >= 80:
        return "low"
    if score >= 60:
        return "medium"
    if score >= 40:
        return "high"
    return "critical"


def _color(score: float) -> str:
    if score >= 80:
        return "#22c55e"
    if score >= 60:
        return "#eab308"
    if score >= 40:
        return "#f97316"
    return "#ef4444"


def _grade_color(grade: str) -> str:
    return {"A": "#22c55e", "B": "#86efac", "C": "#eab308", "D": "#f97316", "F": "#ef4444"}.get(grade, "#888")


def generate_html_report(campaign: CampaignResult) -> str:
    attacks = list(AttackStrategy)
    ratios = sorted(set(c.ratio for c in campaign.configs))

    # Build heatmap data
    heatmap: Dict[str, Dict[float, float]] = {}
    for c in campaign.configs:
        heatmap.setdefault(c.attack.value, {})[c.ratio] = c.resilience_score

    # Per-attack summaries
    attack_summaries = {}
    for a in attacks:
        cfgs = [c for c in campaign.configs if c.attack == a]
        if cfgs:
            avg = statistics.mean(c.resilience_score for c in cfgs)
            worst = min(cfgs, key=lambda c: c.resilience_score)
            attack_summaries[a] = {
                "avg": avg,
                "worst_ratio": worst.ratio,
                "worst_score": worst.resilience_score,
                "false_commits": sum(c.false_commit_rate > 0 for c in cfgs),
                "severity": _severity(avg),
            }

    # Recommendations
    recommendations = []
    for a, s in attack_summaries.items():
        if s["avg"] < 60:
            recommendations.append(f"🔴 <strong>{a.value}</strong>: avg resilience {s['avg']:.0f}/100 — protocol vulnerable, consider adding {_recommendation(a)}")
        elif s["avg"] < 80:
            recommendations.append(f"🟡 <strong>{a.value}</strong>: avg resilience {s['avg']:.0f}/100 — moderate risk, recommend {_recommendation(a)}")

    if not recommendations:
        recommendations.append("✅ Protocol shows strong resilience across all tested attack vectors")

    # Heatmap rows
    heatmap_rows = ""
    for a in attacks:
        cells = ""
        for r in ratios:
            score = heatmap.get(a.value, {}).get(r, 0)
            bg = _color(score)
            cells += f'<td style="background:{bg};color:#000;font-weight:bold;text-align:center">{score:.0f}</td>'
        heatmap_rows += f"<tr><td style='text-align:left;padding:8px;font-weight:600'>{a.value}</td>{cells}</tr>\n"

    # Attack cards
    cards_html = ""
    for a in attacks:
        s = attack_summaries.get(a)
        if not s:
            continue
        sev = s["severity"]
        sev_color = {"low": "#22c55e", "medium": "#eab308", "high": "#f97316", "critical": "#ef4444"}[sev]
        cards_html += f"""
        <div class="card">
            <div class="card-header">
                <span class="attack-name">{a.value}</span>
                <span class="severity" style="background:{sev_color}">{sev.upper()}</span>
            </div>
            <p class="desc">{ATTACK_DESCRIPTIONS[a]}</p>
            <div class="metrics">
                <div>Avg Resilience: <strong>{s['avg']:.1f}</strong></div>
                <div>Worst @ {s['worst_ratio']:.0%}: <strong>{s['worst_score']:.1f}</strong></div>
                <div>False Commits: <strong>{s['false_commits']}/{len(ratios)}</strong> configs</div>
            </div>
        </div>"""

    recs_html = "\n".join(f"<li>{r}</li>" for r in recommendations)
    ratio_headers = "".join(f"<th>{r:.0%}</th>" for r in ratios)

    grade = campaign.grade
    gc = _grade_color(grade)

    # Pre-compute JSON data for JS chart (avoid f-string brace conflicts)
    chart_ratios_json = json.dumps(ratios)
    chart_data_json = json.dumps({a.value: [heatmap.get(a.value, {}).get(r, 0) for r in ratios] for a in attacks})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>mBFT Red Team Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.6}}
h1{{font-size:2em;margin-bottom:4px}} h2{{font-size:1.4em;margin:32px 0 16px;color:#94a3b8}}
.grade-badge{{display:inline-block;font-size:3em;font-weight:900;color:{gc};border:4px solid {gc};border-radius:16px;padding:8px 24px;margin:16px 0}}
.overall{{font-size:1.2em;color:#94a3b8}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{padding:10px 12px;border:1px solid #334155}}
th{{background:#1e293b;color:#94a3b8;font-weight:600}}
.card{{background:#1e293b;border-radius:12px;padding:20px;margin:12px 0}}
.card-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
.attack-name{{font-weight:700;font-size:1.1em}}
.severity{{padding:4px 12px;border-radius:20px;font-size:0.8em;font-weight:700;color:#000}}
.desc{{color:#94a3b8;font-size:0.9em;margin-bottom:12px}}
.metrics{{display:flex;gap:24px;flex-wrap:wrap}}
.metrics div{{font-size:0.95em}}
ul{{padding-left:24px}} li{{margin:6px 0}}
.footer{{margin-top:40px;color:#475569;font-size:0.85em;text-align:center}}
canvas{{max-width:100%;margin:16px 0}}
</style>
</head>
<body>
<h1>🔴 mBFT Consensus Red Team Report</h1>
<p class="overall">Autonomous adversarial assessment of the mBFT protocol</p>

<div class="grade-badge">{grade}</div>
<p class="overall">Overall Resilience Score: <strong>{campaign.overall_score:.1f}/100</strong></p>

<h2>Resilience Heatmap (Attacks × Byzantine Ratios)</h2>
<table>
<tr><th>Attack Strategy</th>{ratio_headers}</tr>
{heatmap_rows}
</table>

<h2>Resilience Curve</h2>
<canvas id="chart" width="800" height="400"></canvas>
<script>
(function(){{
const canvas=document.getElementById('chart'),ctx=canvas.getContext('2d');
const W=canvas.width,H=canvas.height,pad={{t:30,r:20,b:50,l:60}};
const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
const ratios={chart_ratios_json};
const data={chart_data_json};
const colors=['#ef4444','#f97316','#eab308','#22c55e','#3b82f6','#8b5cf6','#ec4899','#14b8a6'];
ctx.fillStyle='#1e293b';ctx.fillRect(0,0,W,H);
// axes
ctx.strokeStyle='#334155';ctx.lineWidth=1;
ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,H-pad.b);ctx.lineTo(W-pad.r,H-pad.b);ctx.stroke();
// labels
ctx.fillStyle='#94a3b8';ctx.font='12px sans-serif';ctx.textAlign='center';
ratios.forEach((r,i)=>{{const x=pad.l+i*cw/(ratios.length-1||1);ctx.fillText((r*100).toFixed(0)+'%',x,H-pad.b+20)}});
ctx.textAlign='right';
for(let v=0;v<=100;v+=20){{const y=H-pad.b-v*ch/100;ctx.fillText(v,pad.l-8,y+4);ctx.beginPath();ctx.strokeStyle='#1e3a5f';ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke()}}
// lines
const keys=Object.keys(data);
keys.forEach((k,ki)=>{{
ctx.strokeStyle=colors[ki%colors.length];ctx.lineWidth=2;ctx.beginPath();
data[k].forEach((v,i)=>{{const x=pad.l+i*cw/(ratios.length-1||1),y=H-pad.b-v*ch/100;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)}});
ctx.stroke();
// label
const lastY=H-pad.b-data[k][data[k].length-1]*ch/100;
ctx.fillStyle=colors[ki%colors.length];ctx.textAlign='left';ctx.font='10px sans-serif';
ctx.fillText(k.replace(/_/g,' ').slice(0,15),W-pad.r+4,lastY+3);
}});
}})();
</script>

<h2>Attack Analysis</h2>
{cards_html}

<h2>Vulnerability Summary &amp; Recommendations</h2>
<ul>{recs_html}</ul>

<div class="footer">
Generated by mBFT Red Team &middot; {len(campaign.configs)} configurations tested &middot; {sum(len(c.trials) for c in campaign.configs)} total trials
</div>
</body>
</html>"""
    return html


def _recommendation(attack: AttackStrategy) -> str:
    recs = {
        AttackStrategy.CONFIDENCE_INFLATION: "confidence decay or calibration checks on repeated high-confidence proposals",
        AttackStrategy.COORDINATED_REJECTION: "quorum-based rejection thresholds to prevent minority blocking",
        AttackStrategy.REPUTATION_FARMING: "longer probation periods and reputation-weighted voting delays",
        AttackStrategy.SYBIL_SWARM: "proof-of-work or stake requirements for agent registration",
        AttackStrategy.FLIP_FLOP: "behavioral consistency scoring and pattern detection",
        AttackStrategy.STEALTH_POISON: "cross-validation of solutions by multiple independent verifiers",
        AttackStrategy.TARGETED_SLASH: "distributed leader protection and rotation randomization",
        AttackStrategy.ENTROPY_MAXIMIZER: "entropy-based anomaly detection to flag chaotic agents",
    }
    return recs.get(attack, "further investigation")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import io, os
    if os.name == 'nt':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Red Team — adversarial protocol testing"
    )
    parser.add_argument("--agents", type=int, default=10, help="Total agents")
    parser.add_argument("--threshold", type=float, default=3.0, help="Commit threshold")
    parser.add_argument("--max-rounds", type=int, default=4, help="Max rounds per trial")
    parser.add_argument("--trials", type=int, default=5, help="Trials per configuration")
    parser.add_argument("--attack", type=str, default="all", help="Attack name or 'all'")
    parser.add_argument("--ratio", type=float, default=None, help="Single Byzantine ratio")
    parser.add_argument("--autopilot", action="store_true", help="Run full campaign")
    parser.add_argument("--output", type=str, default="redteam_report.html", help="HTML output path")
    parser.add_argument("--json", type=str, default=None, help="JSON export path")
    args = parser.parse_args()

    campaign = RedTeamCampaign(
        total_agents=args.agents,
        threshold=args.threshold,
        max_rounds=args.max_rounds,
        trials=args.trials,
    )

    if args.autopilot or args.attack == "all":
        attacks = list(AttackStrategy)
    else:
        try:
            attacks = [AttackStrategy(args.attack)]
        except ValueError:
            print(f"Unknown attack: {args.attack}")
            print(f"Available: {', '.join(a.value for a in AttackStrategy)}")
            sys.exit(1)

    ratios = [args.ratio] if args.ratio is not None else None

    print("🔴 mBFT Red Team — Autonomous Adversarial Campaign")
    print(f"   Agents: {args.agents} | Threshold: {args.threshold} | Rounds: {args.max_rounds} | Trials: {args.trials}")
    print(f"   Attacks: {len(attacks)} | Ratios: {len(ratios) if ratios else 5}")
    print()

    result = asyncio.run(campaign.run_campaign(attacks=attacks, ratios=ratios))

    print(f"\n{'='*60}")
    print(f"  OVERALL GRADE: {result.grade} ({result.overall_score:.1f}/100)")
    print(f"{'='*60}")

    # Vulnerabilities
    for c in result.configs:
        if c.resilience_score < 60:
            print(f"  ⚠️  {c.attack.value} @ {c.ratio:.0%}: resilience {c.resilience_score:.1f} (false commits: {c.false_commit_rate:.0%})")

    html = generate_html_report(result)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📄 HTML report: {args.output}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"📊 JSON export: {args.json}")


if __name__ == "__main__":
    main()
