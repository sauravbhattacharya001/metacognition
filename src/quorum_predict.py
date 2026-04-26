"""Consensus Quorum Predictor — autonomous pre-round outcome forecasting.

Analyses historical round results to predict whether upcoming consensus rounds
will reach quorum *before* they execute.  Uses agent reputation trajectories,
voting pattern statistics, and proposal-confidence signals to estimate commit
probability and recommend optimal agent subsets for reliable quorum.

Usage (CLI)::

    python -m src.quorum_predict [OPTIONS]

Options:
    --agents N          Number of agents (default: 7)
    --rounds N          Historical rounds to simulate (default: 40)
    --byzantine N       Byzantine agent count (default: 1)
    --threshold FLOAT   mBFT commit threshold (default: 2.0)
    --forecast N        Future rounds to forecast (default: 10)
    --auto-select       Enable autonomous optimal-subset selection
    --html FILE         Export interactive HTML report
    --json FILE         Export raw prediction data as JSON

The predictor builds an internal model from observed round history, then for
each forecast round:

1. **Vote Tendency Model** — per-agent mean/stdev of historical vote weights
2. **Reputation Trajectory** — linear extrapolation of reputation over time
3. **Commit Probability** — Monte Carlo sampling of predicted vote distributions
4. **Risk Agents** — agents whose predicted contributions hurt quorum odds
5. **Optimal Subset** — greedy selection of agents maximizing commit probability
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── project imports ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.metacognitive import MockAgent  # noqa: E402
from src.core.protocol import MBFTEngine  # noqa: E402
from src.core.state import RoundResult  # noqa: E402


# ── data models ────────────────────────────────────────────────────────


@dataclass
class AgentProfile:
    """Statistical profile built from historical voting."""

    agent_id: str
    vote_mean: float = 0.0
    vote_std: float = 0.0
    rejection_rate: float = 0.0
    reputation_slope: float = 0.0  # positive = improving
    reputation_current: float = 1.0
    rounds_participated: int = 0


@dataclass
class RoundForecast:
    """Prediction for a single future round."""

    round_index: int
    commit_probability: float
    predicted_aggregate: float
    threshold: float
    risk_agents: List[str] = field(default_factory=list)
    optimal_subset: List[str] = field(default_factory=list)
    optimal_commit_prob: float = 0.0


@dataclass
class PredictionReport:
    """Full prediction output."""

    agent_profiles: List[AgentProfile]
    forecasts: List[RoundForecast]
    overall_health: str  # "healthy" | "at-risk" | "critical"
    recommendations: List[str] = field(default_factory=list)


# ── predictor engine ───────────────────────────────────────────────────

MC_SAMPLES = 2000


class QuorumPredictor:
    """Builds a vote-tendency model and forecasts quorum outcomes."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.profiles: Dict[str, AgentProfile] = {}

    # ── model building ─────────────────────────────────────────────

    def fit(self, history: List[RoundResult], reputations: Dict[str, float]) -> None:
        """Learn per-agent voting statistics from round history."""
        votes_by_agent: Dict[str, List[float]] = {}
        rejections_by_agent: Dict[str, int] = {}
        participation: Dict[str, int] = {}

        for rr in history:
            for v in rr.votes:
                votes_by_agent.setdefault(v.voter_id, []).append(v.weight)
                rejections_by_agent[v.voter_id] = rejections_by_agent.get(v.voter_id, 0) + (
                    1 if v.is_rejection else 0
                )
                participation[v.voter_id] = participation.get(v.voter_id, 0) + 1

        # Reputation trajectory: split history into halves
        mid = max(1, len(history) // 2)
        rep_early: Dict[str, List[float]] = {}
        rep_late: Dict[str, List[float]] = {}
        for i, rr in enumerate(history):
            bucket = rep_early if i < mid else rep_late
            for v in rr.votes:
                bucket.setdefault(v.voter_id, []).append(v.weight)

        for aid, weights in votes_by_agent.items():
            n = len(weights)
            mean = sum(weights) / n
            std = math.sqrt(sum((w - mean) ** 2 for w in weights) / max(n - 1, 1))
            rej_rate = rejections_by_agent.get(aid, 0) / n

            early_mean = _safe_mean(rep_early.get(aid, [0.0]))
            late_mean = _safe_mean(rep_late.get(aid, [0.0]))
            slope = late_mean - early_mean

            self.profiles[aid] = AgentProfile(
                agent_id=aid,
                vote_mean=mean,
                vote_std=std,
                rejection_rate=rej_rate,
                reputation_slope=slope,
                reputation_current=reputations.get(aid, 1.0),
                rounds_participated=participation.get(aid, 0),
            )

    # ── forecasting ────────────────────────────────────────────────

    def forecast(
        self, n_rounds: int, auto_select: bool = False
    ) -> PredictionReport:
        agents = list(self.profiles.values())
        forecasts: List[RoundForecast] = []

        for ri in range(n_rounds):
            prob, agg = self._mc_commit_prob([a.agent_id for a in agents])
            risk = [
                a.agent_id
                for a in agents
                if a.vote_mean < 0 or a.rejection_rate > 0.4 or a.reputation_slope < -0.1
            ]

            opt_subset: List[str] = []
            opt_prob = prob
            if auto_select and len(agents) > 2:
                opt_subset, opt_prob = self._greedy_subset(agents)

            forecasts.append(
                RoundForecast(
                    round_index=ri,
                    commit_probability=round(prob, 4),
                    predicted_aggregate=round(agg, 4),
                    threshold=self.threshold,
                    risk_agents=risk,
                    optimal_subset=opt_subset,
                    optimal_commit_prob=round(opt_prob, 4),
                )
            )

        avg_prob = _safe_mean([f.commit_probability for f in forecasts])
        if avg_prob >= 0.8:
            health = "healthy"
        elif avg_prob >= 0.5:
            health = "at-risk"
        else:
            health = "critical"

        recs = self._generate_recommendations(agents, forecasts, health)

        return PredictionReport(
            agent_profiles=agents,
            forecasts=forecasts,
            overall_health=health,
            recommendations=recs,
        )

    def _mc_commit_prob(self, agent_ids: List[str]) -> Tuple[float, float]:
        """Monte Carlo estimate of commit probability."""
        commits = 0
        agg_sum = 0.0
        for _ in range(MC_SAMPLES):
            total = 0.0
            min_vote = float("inf")
            for aid in agent_ids:
                p = self.profiles[aid]
                vote = random.gauss(p.vote_mean, max(p.vote_std, 0.05))
                vote = max(-1.0, min(1.0, vote))
                total += vote
                min_vote = min(min_vote, vote)
            agg_sum += total
            if total >= self.threshold and min_vote >= 0:
                commits += 1
        return commits / MC_SAMPLES, agg_sum / MC_SAMPLES

    def _greedy_subset(
        self, agents: List[AgentProfile]
    ) -> Tuple[List[str], float]:
        """Greedy forward selection of optimal agent subset."""
        # Sort by vote_mean descending
        ranked = sorted(agents, key=lambda a: a.vote_mean, reverse=True)
        best_subset: List[str] = []
        best_prob = 0.0

        current: List[str] = []
        for a in ranked:
            current.append(a.agent_id)
            if len(current) < 2:
                continue
            prob, _ = self._mc_commit_prob(current)
            if prob > best_prob:
                best_prob = prob
                best_subset = list(current)

        return best_subset, best_prob

    def _generate_recommendations(
        self,
        agents: List[AgentProfile],
        forecasts: List[RoundForecast],
        health: str,
    ) -> List[str]:
        recs: List[str] = []
        if health == "critical":
            recs.append(
                "⚠️ CRITICAL: Average commit probability below 50%. "
                "Consider removing Byzantine agents or lowering threshold."
            )
        elif health == "at-risk":
            recs.append(
                "⚡ AT-RISK: Quorum is achievable but unreliable. "
                "Monitor risk agents closely."
            )

        # Identify worst agents
        worst = sorted(agents, key=lambda a: a.vote_mean)[:2]
        for w in worst:
            if w.vote_mean < 0:
                recs.append(
                    f"🚨 Agent '{w.agent_id}' has negative vote mean "
                    f"({w.vote_mean:.3f}). Likely Byzantine — consider exclusion."
                )
            elif w.rejection_rate > 0.3:
                recs.append(
                    f"⚠️ Agent '{w.agent_id}' rejects {w.rejection_rate:.0%} of proposals. "
                    f"Review calibration or lower its weight."
                )

        # Reputation trajectory warnings
        declining = [a for a in agents if a.reputation_slope < -0.15]
        for d in declining:
            recs.append(
                f"📉 Agent '{d.agent_id}' reputation declining "
                f"(slope={d.reputation_slope:+.3f}). May degrade future quorum."
            )

        # Optimal subset recommendation
        with_opt = [f for f in forecasts if f.optimal_subset]
        if with_opt:
            avg_gain = _safe_mean(
                [f.optimal_commit_prob - f.commit_probability for f in with_opt]
            )
            if avg_gain > 0.05:
                recs.append(
                    f"🎯 Optimal agent subsets improve commit probability by "
                    f"+{avg_gain:.1%} on average. Enable auto-select for production."
                )

        if not recs:
            recs.append("✅ Consensus health looks good. No immediate action needed.")

        return recs


# ── HTML report ────────────────────────────────────────────────────────


def _render_html(report: PredictionReport) -> str:
    profiles_json = json.dumps(
        [
            {
                "id": p.agent_id,
                "voteMean": round(p.vote_mean, 4),
                "voteStd": round(p.vote_std, 4),
                "rejRate": round(p.rejection_rate, 4),
                "repSlope": round(p.reputation_slope, 4),
                "repCurrent": round(p.reputation_current, 4),
            }
            for p in report.agent_profiles
        ]
    )
    forecasts_json = json.dumps(
        [
            {
                "round": f.round_index,
                "commitProb": f.commit_probability,
                "aggregate": f.predicted_aggregate,
                "threshold": f.threshold,
                "riskAgents": f.risk_agents,
                "optSubset": f.optimal_subset,
                "optProb": f.optimal_commit_prob,
            }
            for f in report.forecasts
        ]
    )
    recs_json = json.dumps(report.recommendations)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mBFT Quorum Predictor</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0a0e1a;color:#e0e6f0;padding:20px}}
h1{{text-align:center;font-size:1.8em;margin:20px 0 5px;color:#60a5fa}}
.subtitle{{text-align:center;color:#94a3b8;margin-bottom:25px;font-size:.95em}}
.health-badge{{display:inline-block;padding:4px 14px;border-radius:12px;font-weight:700;font-size:.85em}}
.healthy{{background:#16a34a33;color:#4ade80;border:1px solid #16a34a}}
.at-risk{{background:#d9770633;color:#fb923c;border:1px solid #d97706}}
.critical{{background:#dc262633;color:#f87171;border:1px solid #dc2626}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;max-width:1200px;margin:0 auto}}
.card{{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:18px}}
.card h2{{font-size:1.1em;color:#93c5fd;margin-bottom:12px}}
canvas{{width:100%;height:250px}}
table{{width:100%;border-collapse:collapse;font-size:.85em}}
th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid #1e293b}}
th{{color:#93c5fd;font-weight:600}}
.risk{{color:#f87171}}
.safe{{color:#4ade80}}
.rec{{background:#1e1b4b;border-left:3px solid #818cf8;padding:10px 14px;margin:6px 0;border-radius:0 8px 8px 0;font-size:.9em}}
.full{{grid-column:1/-1}}
@media(max-width:768px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>🔮 Consensus Quorum Predictor</h1>
<p class="subtitle">Autonomous pre-round outcome forecasting for mBFT consensus
  &nbsp;|&nbsp; <span class="health-badge {report.overall_health}">{report.overall_health.upper()}</span></p>
<div class="grid">
  <div class="card">
    <h2>📊 Commit Probability Forecast</h2>
    <canvas id="probChart"></canvas>
  </div>
  <div class="card">
    <h2>📈 Predicted Aggregate Weight</h2>
    <canvas id="aggChart"></canvas>
  </div>
  <div class="card">
    <h2>🧬 Agent Voting Profiles</h2>
    <div style="overflow-x:auto"><table>
      <tr><th>Agent</th><th>Vote μ</th><th>Vote σ</th><th>Rej%</th><th>Rep Slope</th><th>Risk</th></tr>
      <tbody id="profileTable"></tbody>
    </table></div>
  </div>
  <div class="card">
    <h2>🎯 Optimal Subset Analysis</h2>
    <canvas id="subsetChart"></canvas>
  </div>
  <div class="card full">
    <h2>💡 Proactive Recommendations</h2>
    <div id="recs"></div>
  </div>
</div>
<script>
const profiles={profiles_json};
const forecasts={forecasts_json};
const recs={recs_json};

// Mini canvas chart helper
function lineChart(id, labels, datasets) {{
  const c=document.getElementById(id),ctx=c.getContext('2d');
  c.width=c.offsetWidth*2;c.height=c.offsetHeight*2;
  ctx.scale(2,2);
  const W=c.offsetWidth,H=c.offsetHeight,pad=40;
  const allVals=datasets.flatMap(d=>d.data);
  const mn=Math.min(...allVals),mx=Math.max(...allVals);
  const range=mx-mn||1;
  function x(i){{return pad+(W-pad*2)*i/(labels.length-1||1)}}
  function y(v){{return H-pad-(H-pad*2)*(v-mn)/range}}
  // grid
  ctx.strokeStyle='#1e293b';ctx.lineWidth=.5;
  for(let i=0;i<=4;i++){{const yy=pad+(H-pad*2)*i/4;ctx.beginPath();ctx.moveTo(pad,yy);ctx.lineTo(W-10,yy);ctx.stroke();
    ctx.fillStyle='#64748b';ctx.font='10px system-ui';ctx.fillText((mx-range*i/4).toFixed(2),2,yy+3);}}
  // axes labels
  labels.forEach((l,i)=>{{if(i%Math.ceil(labels.length/10)===0){{ctx.fillStyle='#64748b';ctx.font='10px system-ui';ctx.fillText(l,x(i)-5,H-5);}}}});
  // lines
  const colors=['#60a5fa','#f87171','#4ade80','#fbbf24'];
  datasets.forEach((ds,di)=>{{ctx.strokeStyle=colors[di%colors.length];ctx.lineWidth=2;ctx.beginPath();
    ds.data.forEach((v,i)=>{{i===0?ctx.moveTo(x(i),y(v)):ctx.lineTo(x(i),y(v))}});ctx.stroke();
    // dots
    ctx.fillStyle=colors[di%colors.length];
    ds.data.forEach((v,i)=>{{ctx.beginPath();ctx.arc(x(i),y(v),3,0,Math.PI*2);ctx.fill();}});
  }});
}}

const labels=forecasts.map(f=>'R'+f.round);
lineChart('probChart',labels,[{{data:forecasts.map(f=>f.commitProb)}}]);
lineChart('aggChart',labels,[
  {{data:forecasts.map(f=>f.aggregate)}},
  {{data:forecasts.map(f=>f.threshold)}}
]);
if(forecasts.some(f=>f.optProb>0)){{
  lineChart('subsetChart',labels,[
    {{data:forecasts.map(f=>f.commitProb)}},
    {{data:forecasts.map(f=>f.optProb)}}
  ]);
}} else {{
  document.getElementById('subsetChart').parentElement.querySelector('h2').textContent+=' (enable --auto-select)';
}}

// profile table
const tb=document.getElementById('profileTable');
profiles.forEach(p=>{{
  const risk=p.voteMean<0||p.rejRate>0.4||p.repSlope<-0.1;
  tb.innerHTML+=`<tr><td>${{p.id}}</td><td>${{p.voteMean.toFixed(3)}}</td><td>${{p.voteStd.toFixed(3)}}</td><td>${{(p.rejRate*100).toFixed(1)}}%</td><td>${{p.repSlope>=0?'+':''}}${{p.repSlope.toFixed(3)}}</td><td class="${{risk?'risk':'safe'}}">${{risk?'⚠️ Risk':'✅ OK'}}</td></tr>`;
}});

// recs
const rd=document.getElementById('recs');
recs.forEach(r=>rd.innerHTML+=`<div class="rec">${{r}}</div>`);
</script>
</body>
</html>"""


# ── CLI ────────────────────────────────────────────────────────────────


def _safe_mean(vals: List[float]) -> float:
    return sum(vals) / max(len(vals), 1)


async def _run_simulation(
    n_agents: int,
    n_byzantine: int,
    n_rounds: int,
    threshold: float,
) -> Tuple[List[RoundResult], Dict[str, float], List["BaseAgent"]]:
    """Run a historical simulation to produce training data for the predictor."""
    agents = []
    answers = ["solution-A", "solution-B", "solution-C"]
    for i in range(n_agents):
        is_byz = i < n_byzantine
        conf = round(random.uniform(0.3, 0.5), 2) if is_byz else round(random.uniform(0.6, 0.95), 2)
        answer = random.choice(answers)
        a = MockAgent(
            agent_id=f"agent-{i}",
            answer=answer,
            confidence=conf,
            byzantine=is_byz,
            accept_set={answer, answers[0]},  # partial agreement
        )
        agents.append(a)

    engine = MBFTEngine(agents, threshold=threshold, max_rounds=n_rounds)
    history: List[RoundResult] = []
    for _ in range(n_rounds):
        result = await engine.run(f"consensus-task-{random.randint(0, 9999)}")
        if result:
            history.append(result)
        # Reset engine for next independent round
        engine.history = []

    return history, dict(engine._reputation), agents


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Consensus Quorum Predictor"
    )
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--byzantine", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--forecast", type=int, default=10)
    parser.add_argument("--auto-select", action="store_true")
    parser.add_argument("--html", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    print("🔮 mBFT Quorum Predictor")
    print("=" * 50)
    print(f"Agents: {args.agents} ({args.byzantine} Byzantine)")
    print(f"Historical rounds: {args.rounds}")
    print(f"Forecast rounds: {args.forecast}")
    print(f"Threshold: {args.threshold}")
    print()

    print("📡 Running historical simulation...")
    history, reps, agents = await _run_simulation(
        args.agents, args.byzantine, args.rounds, args.threshold
    )
    print(f"   Collected {len(history)} round results")
    committed = sum(1 for r in history if r.committed)
    print(f"   Historical commit rate: {committed}/{len(history)} ({committed/max(len(history),1):.0%})")
    print()

    print("🧠 Building vote-tendency model...")
    predictor = QuorumPredictor(threshold=args.threshold)
    predictor.fit(history, reps)
    print(f"   Profiled {len(predictor.profiles)} agents")
    print()

    print("🔮 Forecasting future rounds...")
    report = predictor.forecast(args.forecast, auto_select=args.auto_select)
    print()

    # Display results
    print(f"🏥 Overall Health: {report.overall_health.upper()}")
    print()

    print("📊 Agent Profiles:")
    print(f"  {'Agent':<12} {'Vote μ':>8} {'Vote σ':>8} {'Rej%':>8} {'Trend':>8} {'Status':>8}")
    print("  " + "-" * 60)
    for p in report.agent_profiles:
        risk = p.vote_mean < 0 or p.rejection_rate > 0.4 or p.reputation_slope < -0.1
        status = "⚠️ RISK" if risk else "✅ OK"
        print(
            f"  {p.agent_id:<12} {p.vote_mean:>8.3f} {p.vote_std:>8.3f} "
            f"{p.rejection_rate:>7.1%} {p.reputation_slope:>+8.3f} {status:>8}"
        )
    print()

    print("🔮 Round Forecasts:")
    print(f"  {'Round':>6} {'P(commit)':>10} {'Agg Weight':>11} {'Threshold':>10} {'Risk Agents'}")
    print("  " + "-" * 60)
    for f in report.forecasts:
        risk_str = ", ".join(f.risk_agents) if f.risk_agents else "none"
        print(
            f"  {f.round_index:>6} {f.commit_probability:>10.1%} "
            f"{f.predicted_aggregate:>11.3f} {f.threshold:>10.1f}   {risk_str}"
        )

    if args.auto_select:
        print()
        print("🎯 Optimal Subsets:")
        for f in report.forecasts:
            if f.optimal_subset:
                print(
                    f"  Round {f.round_index}: {', '.join(f.optimal_subset)} "
                    f"→ {f.optimal_commit_prob:.1%}"
                )
    print()

    print("💡 Recommendations:")
    for r in report.recommendations:
        print(f"  {r}")

    if args.html:
        html = _render_html(report)
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"\n📄 HTML report: {args.html}")

    if args.json:
        data = {
            "health": report.overall_health,
            "profiles": [
                {
                    "id": p.agent_id,
                    "vote_mean": p.vote_mean,
                    "vote_std": p.vote_std,
                    "rejection_rate": p.rejection_rate,
                    "reputation_slope": p.reputation_slope,
                }
                for p in report.agent_profiles
            ],
            "forecasts": [
                {
                    "round": f.round_index,
                    "commit_probability": f.commit_probability,
                    "predicted_aggregate": f.predicted_aggregate,
                    "risk_agents": f.risk_agents,
                    "optimal_subset": f.optimal_subset,
                    "optimal_commit_prob": f.optimal_commit_prob,
                }
                for f in report.forecasts
            ],
            "recommendations": report.recommendations,
        }
        with open(args.json, "w", encoding="utf-8") as fj:
            json.dump(data, fj, indent=2)
        print(f"📄 JSON export: {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
