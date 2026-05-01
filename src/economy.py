"""Consensus Economy Simulator — resource economics for mBFT consensus.

Introduces an economic layer where agents have finite budgets they must
strategically allocate across consensus rounds.  Investment in proposals
affects voting weight; successful commits yield returns while failed rounds
cost resources.  Market dynamics (inflation, taxation, subsidies) evolve
the economy over time, and autonomous fiscal policies adapt to maintain
system health.

Usage::

    python -m src.economy                          # default 12-round simulation
    python -m src.economy --rounds 20 --agents 7
    python -m src.economy --strategy-mix diverse    # mix of 6 strategies
    python -m src.economy --autopilot               # autonomous fiscal policy
    python -m src.economy --export report.html
    python -m src.economy --export results.json
"""
from __future__ import annotations

import argparse
import asyncio
import html as html_mod
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agents.metacognitive import MockAgent
from src.core.protocol import MBFTEngine

# ── Economic agent strategies ────────────────────────────────────────────

STRATEGIES = {
    "conservative": "Low-risk: invest small amounts, preserve capital",
    "aggressive": "High-risk: invest heavily, seek large returns",
    "contrarian": "Invest against the majority for contrarian payoffs",
    "adaptive": "Adjust investment based on recent success/failure",
    "momentum": "Follow the trend — invest more when winning",
    "balanced": "Split investment evenly across opportunities",
}

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
]


@dataclass
class EconAgent:
    """Agent with economic state."""
    agent_id: str
    strategy: str
    budget: float = 100.0
    total_invested: float = 0.0
    total_returns: float = 0.0
    wins: int = 0
    losses: int = 0
    bankruptcies: int = 0
    confidence_base: float = 0.7
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def roi(self) -> float:
        return (self.total_returns / self.total_invested - 1.0) if self.total_invested > 0 else 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def decide_investment(self, round_idx: int) -> float:
        """Choose how much budget to invest this round.

        Returns 0 when the agent has no positive budget, preventing
        negative-balance agents (e.g. after inflation erosion) from
        placing phantom investments.
        """
        if self.budget <= 0:
            self.budget = 0.0  # clamp to zero — no debt accumulation
            return 0.0

        if self.strategy == "conservative":
            return min(self.budget * 0.15, self.budget)
        elif self.strategy == "aggressive":
            return min(self.budget * 0.5, self.budget)
        elif self.strategy == "contrarian":
            # Invest more after losses (contrarian bet)
            loss_streak = 0
            for h in reversed(self.history):
                if not h.get("won"):
                    loss_streak += 1
                else:
                    break
            multiplier = min(0.2 + loss_streak * 0.1, 0.6)
            return min(self.budget * multiplier, self.budget)
        elif self.strategy == "adaptive":
            # Recent performance dictates investment
            recent = self.history[-5:]
            if not recent:
                return min(self.budget * 0.25, self.budget)
            recent_wr = sum(1 for h in recent if h.get("won")) / len(recent)
            frac = 0.15 + recent_wr * 0.35
            return min(self.budget * frac, self.budget)
        elif self.strategy == "momentum":
            # Increase if on winning streak
            streak = 0
            for h in reversed(self.history):
                if h.get("won"):
                    streak += 1
                else:
                    break
            frac = min(0.2 + streak * 0.08, 0.55)
            return min(self.budget * frac, self.budget)
        else:  # balanced
            return min(self.budget * 0.25, self.budget)


@dataclass
class MarketState:
    """Global economic state."""
    inflation_rate: float = 0.02
    tax_rate: float = 0.05
    subsidy_pool: float = 0.0
    total_gdp: float = 0.0
    gini_coefficient: float = 0.0
    round_idx: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FiscalPolicy:
    """Autonomous fiscal policy decisions."""
    action: str
    reason: str
    parameters: Dict[str, float] = field(default_factory=dict)


# ── Core simulation ──────────────────────────────────────────────────────

def _compute_gini(values: List[float]) -> float:
    """Compute Gini coefficient for wealth distribution."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    cumulative = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_v):
        cumulative += v
        weighted_sum += (2 * (i + 1) - n - 1) * v
    return weighted_sum / (n * total)


def _autonomous_fiscal_policy(market: MarketState, agents: List[EconAgent]) -> Optional[FiscalPolicy]:
    """Determine fiscal intervention based on economic indicators."""
    budgets = [a.budget for a in agents]
    gini = _compute_gini(budgets)
    avg_budget = sum(budgets) / len(budgets) if budgets else 0
    bankrupt_count = sum(1 for a in agents if a.budget <= 0)
    bankrupt_ratio = bankrupt_count / len(agents) if agents else 0

    # Inequality too high → redistribute
    if gini > 0.5:
        return FiscalPolicy(
            action="redistribute",
            reason=f"Gini coefficient {gini:.3f} exceeds 0.5 — wealth too concentrated",
            parameters={"tax_increase": 0.03, "subsidy_amount": avg_budget * 0.1},
        )

    # Too many bankruptcies → bailout
    if bankrupt_ratio > 0.3:
        return FiscalPolicy(
            action="bailout",
            reason=f"{bankrupt_count}/{len(agents)} agents bankrupt — systemic risk",
            parameters={"bailout_amount": 20.0},
        )

    # Economy stagnating (low average budget) → stimulus
    if avg_budget < 30 and market.round_idx > 3:
        return FiscalPolicy(
            action="stimulus",
            reason=f"Average budget {avg_budget:.1f} critically low — economic stimulus needed",
            parameters={"stimulus_amount": 15.0},
        )

    # High inflation → tighten
    if market.inflation_rate > 0.08:
        return FiscalPolicy(
            action="tighten",
            reason=f"Inflation {market.inflation_rate:.1%} too high — tightening monetary policy",
            parameters={"inflation_reduction": 0.02},
        )

    return None


def _apply_fiscal_policy(policy: FiscalPolicy, market: MarketState, agents: List[EconAgent]) -> None:
    """Execute a fiscal policy decision.

    All spending is bounded by the available ``subsidy_pool`` to prevent
    unbounded money creation.  When the pool is exhausted mid-disbursement
    remaining agents receive nothing — a deliberate scarcity constraint.
    """
    if policy.action == "redistribute":
        market.tax_rate = min(market.tax_rate + policy.parameters.get("tax_increase", 0), 0.25)
        subsidy = policy.parameters.get("subsidy_amount", 10)
        poorest = sorted(agents, key=lambda a: a.budget)[:max(1, len(agents) // 3)]
        for a in poorest:
            disbursement = min(subsidy, market.subsidy_pool)
            if disbursement <= 0:
                break
            a.budget += disbursement
            market.subsidy_pool -= disbursement

    elif policy.action == "bailout":
        amount = policy.parameters.get("bailout_amount", 20)
        for a in agents:
            if a.budget <= 0:
                disbursement = min(amount, market.subsidy_pool)
                if disbursement <= 0:
                    break
                a.budget = disbursement
                a.bankruptcies += 1
                market.subsidy_pool -= disbursement

    elif policy.action == "stimulus":
        amount = policy.parameters.get("stimulus_amount", 15)
        for a in agents:
            disbursement = min(amount, market.subsidy_pool)
            if disbursement <= 0:
                break
            a.budget += disbursement
            market.subsidy_pool -= disbursement

    elif policy.action == "tighten":
        reduction = policy.parameters.get("inflation_reduction", 0.02)
        market.inflation_rate = max(market.inflation_rate - reduction, 0.0)


async def run_economy(
    num_agents: int = 5,
    num_rounds: int = 12,
    strategy_mix: str = "diverse",
    autopilot: bool = False,
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """Run the full economic simulation."""
    # Create agents with strategies
    strategies_list = list(STRATEGIES.keys())
    agents: List[EconAgent] = []
    for i in range(num_agents):
        if strategy_mix == "diverse":
            strat = strategies_list[i % len(strategies_list)]
        elif strategy_mix in STRATEGIES:
            strat = strategy_mix
        else:
            strat = random.choice(strategies_list)
        agents.append(EconAgent(
            agent_id=f"econ-{i:02d}",
            strategy=strat,
            budget=100.0,
            confidence_base=random.uniform(0.55, 0.9),
        ))

    market = MarketState()
    rounds_data: List[Dict[str, Any]] = []
    fiscal_log: List[Dict[str, Any]] = []

    for r in range(num_rounds):
        market.round_idx = r
        task = TASKS[r % len(TASKS)]

        # Agents decide investments
        investments: Dict[str, float] = {}
        for a in agents:
            inv = a.decide_investment(r)
            investments[a.agent_id] = inv
            a.budget -= inv
            a.total_invested += inv

        # Build mock agents with investment-weighted confidence
        total_inv = sum(investments.values()) or 1.0
        mock_agents = []
        for a in agents:
            inv_weight = investments[a.agent_id] / total_inv
            conf = min(a.confidence_base * (0.5 + inv_weight * 2.0), 0.99)
            # Introduce occasional disagreement
            answer = "consensus-answer" if random.random() < 0.7 else f"alt-{a.agent_id}"
            mock_agents.append(MockAgent(
                agent_id=a.agent_id,
                answer=answer,
                confidence=max(0.01, conf),
            ))

        # Run consensus
        engine = MBFTEngine(mock_agents, threshold=threshold, max_rounds=2)
        result = await engine.run(task)
        committed = result.committed if result else False

        # Distribute returns
        if committed:
            return_pool = sum(investments.values()) * 1.3  # 30% return on success
            for a in agents:
                share = investments[a.agent_id] / total_inv if total_inv > 0 else 0
                ret = return_pool * share
                tax = ret * market.tax_rate
                net = ret - tax
                a.budget += net
                a.total_returns += net
                a.wins += 1
                market.subsidy_pool += tax
                a.history.append({"round": r, "won": True, "invested": investments[a.agent_id], "return": net})
        else:
            # Failed round — partial loss (refund half the investment)
            for a in agents:
                invested = investments[a.agent_id]
                refund = invested * 0.5  # get half back
                loss = invested - refund  # actual capital lost
                a.budget += refund
                a.total_returns -= loss  # track net loss in returns
                a.losses += 1
                a.history.append({"round": r, "won": False, "invested": invested, "return": -loss})

        # Apply inflation
        for a in agents:
            a.budget *= (1.0 - market.inflation_rate)

        # Compute market stats
        budgets = [a.budget for a in agents]
        market.gini_coefficient = _compute_gini(budgets)
        market.total_gdp = sum(budgets)

        # Autonomous fiscal policy
        policy_applied = None
        if autopilot:
            policy = _autonomous_fiscal_policy(market, agents)
            if policy:
                _apply_fiscal_policy(policy, market, agents)
                policy_applied = {"action": policy.action, "reason": policy.reason, "params": policy.parameters}
                fiscal_log.append({"round": r, **policy_applied})

        rounds_data.append({
            "round": r,
            "task": task,
            "committed": committed,
            "total_invested": sum(investments.values()),
            "gdp": market.total_gdp,
            "gini": market.gini_coefficient,
            "inflation": market.inflation_rate,
            "tax_rate": market.tax_rate,
            "bankrupt_count": sum(1 for a in agents if a.budget <= 0),
            "fiscal_policy": policy_applied,
            "investments": {aid: round(v, 2) for aid, v in investments.items()},
        })

        market.history.append({
            "round": r, "gdp": market.total_gdp,
            "gini": market.gini_coefficient, "inflation": market.inflation_rate,
        })

    # Final rankings
    rankings = sorted(agents, key=lambda a: a.budget, reverse=True)
    agent_summaries = []
    for rank, a in enumerate(rankings, 1):
        agent_summaries.append({
            "rank": rank,
            "agent_id": a.agent_id,
            "strategy": a.strategy,
            "final_budget": round(a.budget, 2),
            "roi": round(a.roi * 100, 1),
            "win_rate": round(a.win_rate * 100, 1),
            "total_invested": round(a.total_invested, 2),
            "total_returns": round(a.total_returns, 2),
            "bankruptcies": a.bankruptcies,
        })

    return {
        "config": {
            "num_agents": num_agents,
            "num_rounds": num_rounds,
            "strategy_mix": strategy_mix,
            "autopilot": autopilot,
            "threshold": threshold,
        },
        "rankings": agent_summaries,
        "rounds": rounds_data,
        "fiscal_log": fiscal_log,
        "market_final": {
            "gdp": round(market.total_gdp, 2),
            "gini": round(market.gini_coefficient, 3),
            "inflation": round(market.inflation_rate, 4),
            "tax_rate": round(market.tax_rate, 4),
            "subsidy_pool": round(market.subsidy_pool, 2),
        },
    }


# ── HTML report ──────────────────────────────────────────────────────────

def _generate_html(data: Dict[str, Any]) -> str:
    rankings = data["rankings"]
    rounds = data["rounds"]
    fiscal = data["fiscal_log"]
    market = data["market_final"]
    cfg = data["config"]

    # GDP sparkline data
    gdp_values = [r["gdp"] for r in rounds]
    gini_values = [r["gini"] for r in rounds]
    max_gdp = max(gdp_values) if gdp_values else 1

    def bar(val: float, mx: float, color: str = "#4fc3f7") -> str:
        pct = (val / mx * 100) if mx else 0
        return f'<div style="background:{color};height:18px;width:{pct:.1f}%;border-radius:3px"></div>'

    rows_rank = ""
    for r in rankings:
        color = "#4caf50" if r["roi"] > 0 else "#f44336" if r["roi"] < -20 else "#ff9800"
        rows_rank += f"""<tr>
            <td>#{r['rank']}</td><td>{html_mod.escape(r['agent_id'])}</td><td><b>{html_mod.escape(r['strategy'])}</b></td>
            <td style="color:{color}">${r['final_budget']:.0f}</td>
            <td>{r['roi']:.1f}%</td><td>{r['win_rate']:.0f}%</td>
            <td>{r['total_invested']:.0f}</td><td>{r['bankruptcies']}</td></tr>"""

    rows_round = ""
    for rd in rounds:
        status = "✅" if rd["committed"] else "❌"
        policy = rd["fiscal_policy"]["action"] if rd["fiscal_policy"] else "—"
        rows_round += f"""<tr>
            <td>{rd['round']}</td><td>{status}</td><td>${rd['total_invested']:.0f}</td>
            <td>${rd['gdp']:.0f}</td><td>{rd['gini']:.3f}</td>
            <td>{rd['bankrupt_count']}</td><td>{html_mod.escape(policy)}</td></tr>"""

    fiscal_rows = ""
    for f in fiscal:
        fiscal_rows += f"<tr><td>{f['round']}</td><td><b>{html_mod.escape(f['action'])}</b></td><td>{html_mod.escape(f['reason'])}</td></tr>"

    # Canvas GDP chart
    gdp_points = ""
    if gdp_values:
        max_g = max(gdp_values) or 1
        for i, v in enumerate(gdp_values):
            x = 50 + i * (700 / max(len(gdp_values) - 1, 1))
            y = 180 - (v / max_g) * 160
            gdp_points += f"{x},{y} "

    gini_points = ""
    if gini_values:
        for i, v in enumerate(gini_values):
            x = 50 + i * (700 / max(len(gini_values) - 1, 1))
            y = 180 - v * 160
            gini_points += f"{x},{y} "

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Consensus Economy Report</title>
<style>
:root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9; --accent: #58a6ff; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:var(--bg); color:var(--text); padding:24px; }}
h1 {{ color:#fff; margin-bottom:8px; }} h2 {{ color:var(--accent); margin:24px 0 12px; }}
.subtitle {{ color:#8b949e; margin-bottom:24px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:24px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; text-align:center; }}
.card .value {{ font-size:1.8em; font-weight:bold; color:#fff; }}
.card .label {{ color:#8b949e; font-size:0.85em; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:8px; overflow:hidden; margin-bottom:20px; }}
th {{ background:#21262d; color:var(--accent); padding:10px; text-align:left; font-size:0.85em; }}
td {{ padding:8px 10px; border-top:1px solid var(--border); font-size:0.9em; }}
tr:hover {{ background:#1c2128; }}
svg {{ background:var(--card); border-radius:8px; border:1px solid var(--border); }}
.tag {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:0.75em; }}
</style></head><body>
<h1>💰 Consensus Economy Simulator</h1>
<p class="subtitle">{cfg['num_agents']} agents · {cfg['num_rounds']} rounds · strategy: {cfg['strategy_mix']} · autopilot: {'ON' if cfg['autopilot'] else 'OFF'}</p>

<div class="cards">
  <div class="card"><div class="value">${market['gdp']:.0f}</div><div class="label">Final GDP</div></div>
  <div class="card"><div class="value">{market['gini']:.3f}</div><div class="label">Gini Index</div></div>
  <div class="card"><div class="value">{market['inflation']:.1%}</div><div class="label">Inflation</div></div>
  <div class="card"><div class="value">{market['tax_rate']:.1%}</div><div class="label">Tax Rate</div></div>
  <div class="card"><div class="value">{len(fiscal)}</div><div class="label">Fiscal Actions</div></div>
  <div class="card"><div class="value">${market['subsidy_pool']:.0f}</div><div class="label">Subsidy Pool</div></div>
</div>

<h2>📊 Economic Trends</h2>
<svg viewBox="0 0 800 200" width="100%" style="max-width:800px;margin-bottom:20px">
  <text x="10" y="15" fill="#8b949e" font-size="11">GDP &amp; Gini over rounds</text>
  <line x1="50" y1="180" x2="750" y2="180" stroke="#30363d" />
  <polyline points="{gdp_points}" fill="none" stroke="#4fc3f7" stroke-width="2"/>
  <polyline points="{gini_points}" fill="none" stroke="#f48fb1" stroke-width="2" stroke-dasharray="4"/>
  <text x="760" y="100" fill="#4fc3f7" font-size="10">GDP</text>
  <text x="760" y="115" fill="#f48fb1" font-size="10">Gini</text>
</svg>

<h2>🏆 Agent Rankings</h2>
<table>
<tr><th>#</th><th>Agent</th><th>Strategy</th><th>Budget</th><th>ROI</th><th>Win%</th><th>Invested</th><th>Bankrupt</th></tr>
{rows_rank}
</table>

<h2>📈 Round-by-Round</h2>
<table>
<tr><th>Round</th><th>Commit</th><th>Invested</th><th>GDP</th><th>Gini</th><th>Bankrupt</th><th>Fiscal</th></tr>
{rows_round}
</table>

{"<h2>🏛️ Fiscal Policy Log</h2><table><tr><th>Round</th><th>Action</th><th>Reason</th></tr>" + fiscal_rows + "</table>" if fiscal else ""}

<p style="color:#8b949e;margin-top:32px;text-align:center">Consensus Economy Simulator · mBFT Metacognition Framework</p>
</body></html>"""


# ── CLI ──────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Consensus Economy Simulator")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=12, help="Simulation rounds")
    parser.add_argument("--strategy-mix", default="diverse",
                        choices=["diverse", *STRATEGIES.keys()], help="Agent strategy composition")
    parser.add_argument("--autopilot", action="store_true", help="Enable autonomous fiscal policy")
    parser.add_argument("--threshold", type=float, default=0.6, help="Consensus threshold")
    parser.add_argument("--export", type=str, default=None, help="Export to .html or .json")
    args = parser.parse_args()

    data = asyncio.run(run_economy(
        num_agents=args.agents,
        num_rounds=args.rounds,
        strategy_mix=args.strategy_mix,
        autopilot=args.autopilot,
        threshold=args.threshold,
    ))

    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if args.export:
        path = Path(args.export)
        if path.suffix == ".html":
            path.write_text(_generate_html(data), encoding="utf-8")
            print(f"[OK] HTML report -> {path}")
        elif path.suffix == ".json":
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"[OK] JSON export -> {path}")
        else:
            print(f"[ERROR] Unsupported format: {path.suffix}")
            sys.exit(1)
    else:
        # Terminal summary
        cfg = data["config"]
        print(f"\nConsensus Economy Simulator")
        print(f"{'-' * 50}")
        print(f"Agents: {cfg['num_agents']}  Rounds: {cfg['num_rounds']}  Strategy: {cfg['strategy_mix']}  Autopilot: {'ON' if cfg['autopilot'] else 'OFF'}")
        print()

        print("Rankings:")
        print(f"{'#':<4} {'Agent':<12} {'Strategy':<14} {'Budget':>8} {'ROI':>8} {'Win%':>6}")
        print("-" * 56)
        for r in data["rankings"]:
            print(f"#{r['rank']:<3} {r['agent_id']:<12} {r['strategy']:<14} ${r['final_budget']:>6.0f} {r['roi']:>6.1f}% {r['win_rate']:>5.0f}%")

        print(f"\nMarket: GDP=${data['market_final']['gdp']:.0f}  Gini={data['market_final']['gini']:.3f}  Inflation={data['market_final']['inflation']:.1%}")

        if data["fiscal_log"]:
            print(f"\nFiscal Actions: {len(data['fiscal_log'])}")
            for f in data["fiscal_log"]:
                print(f"  Round {f['round']}: {f['action']} - {f['reason']}")



if __name__ == "__main__":
    _cli()
