"""Prediction Market Engine — swarm intelligence via belief trading.

Agents create markets on questions, trade belief-shares using a Logarithmic
Market Scoring Rule (LMSR), and market prices aggregate the swarm's
distributed intelligence.  Honest agents trade toward their calibrated
beliefs while Byzantine agents trade randomly, producing a natural signal
that separates reliable agents from unreliable ones via profitability.

Usage (CLI)::

    python -m src.prediction_market                    # default simulation
    python -m src.prediction_market --agents 10 --questions 8
    python -m src.prediction_market --byzantine 3 --rounds 20
    python -m src.prediction_market --liquidity 200 --balance 2000
    python -m src.prediction_market --export report.html
    python -m src.prediction_market --json results.json
"""
from __future__ import annotations

import argparse
import asyncio
import html as html_mod
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.metacognitive import MockAgent  # noqa: E402

# ── Data Models ──────────────────────────────────────────────────────────


class Market(BaseModel):
    market_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    question: str
    creator_id: str
    created_at: float = Field(default_factory=time.time)
    resolution_time: Optional[float] = None
    resolved: bool = False
    outcome: Optional[bool] = None
    liquidity_param: float = 100.0
    shares_yes: float = 0.0  # outstanding yes-shares in LMSR pool
    shares_no: float = 0.0   # outstanding no-shares in LMSR pool


class Position(BaseModel):
    agent_id: str
    market_id: str
    shares_yes: float = 0.0
    shares_no: float = 0.0
    total_invested: float = 0.0


class Trade(BaseModel):
    trade_id: str = Field(default_factory=lambda: uuid4().hex[:10])
    agent_id: str
    market_id: str
    direction: str  # "yes" or "no"
    shares: float
    cost: float
    timestamp: float = Field(default_factory=time.time)


class MarketSnapshot(BaseModel):
    market_id: str
    question: str
    price_yes: float
    price_no: float
    volume: float
    num_traders: int
    resolved: bool
    outcome: Optional[bool] = None


class AgentPortfolio(BaseModel):
    agent_id: str
    balance: float
    markets_traded: int
    total_profit: float
    accuracy: float


class MarketVerdict(BaseModel):
    market_id: str
    resolution_method: str  # "consensus" or "manual"
    confidence: float
    evidence_summary: str


# ── LMSR helpers ─────────────────────────────────────────────────────────


def lmsr_cost(q_yes: float, q_no: float, b: float) -> float:
    """LMSR cost function: C = b * ln(exp(q_yes/b) + exp(q_no/b))."""
    # Numerically stable version using log-sum-exp trick
    m = max(q_yes / b, q_no / b)
    return b * (m + math.log(math.exp(q_yes / b - m) + math.exp(q_no / b - m)))


def lmsr_price_yes(q_yes: float, q_no: float, b: float) -> float:
    """Instantaneous price of a YES share = exp(q_yes/b) / (exp(q_yes/b) + exp(q_no/b))."""
    diff = (q_yes - q_no) / b
    # sigmoid
    if diff > 500:
        return 1.0
    if diff < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-diff))


# ── Engine ───────────────────────────────────────────────────────────────


SAMPLE_QUESTIONS = [
    "Will the consensus commit on the first round?",
    "Will Byzantine agents exceed 30% of the swarm?",
    "Will average agent confidence be above 0.7?",
    "Will the swarm reach quorum within 3 rounds?",
    "Will reputation variance increase this epoch?",
    "Will the top agent maintain leadership for 5+ rounds?",
    "Will network latency cause a missed commit window?",
    "Will the economy enter deflation this cycle?",
    "Will a new coalition form among honest agents?",
    "Will the swarm detect the implanted Byzantine agent?",
]


class PredictionMarketEngine:
    """LMSR-based prediction market for multi-agent belief aggregation."""

    def __init__(
        self,
        agents: List[MockAgent],
        liquidity: float = 100.0,
        initial_balance: float = 1000.0,
    ) -> None:
        if not agents:
            raise ValueError("Need at least one agent.")
        if liquidity <= 0:
            raise ValueError("Liquidity parameter must be positive.")
        self.agents = {a.id: a for a in agents}
        self.liquidity = liquidity
        self.initial_balance = initial_balance
        self.balances: Dict[str, float] = {a.id: initial_balance for a in agents}
        self.markets: Dict[str, Market] = {}
        self.positions: Dict[str, Dict[str, Position]] = {}  # market_id -> agent_id -> Position
        self.trades: List[Trade] = []
        self.verdicts: Dict[str, MarketVerdict] = {}
        # Incremental indexes — avoid O(T) full-scan in queries
        self._market_volume: Dict[str, float] = {}        # market_id -> cumulative volume
        self._market_traders: Dict[str, set] = {}          # market_id -> set of agent_ids
        self._agent_markets: Dict[str, set] = {}           # agent_id -> set of market_ids traded

    # ── Market operations ────────────────────────────────────────────

    def create_market(
        self,
        creator_id: str,
        question: str,
        resolution_time: Optional[float] = None,
        liquidity_param: Optional[float] = None,
    ) -> Market:
        b = liquidity_param if liquidity_param is not None else self.liquidity
        m = Market(
            question=question,
            creator_id=creator_id,
            resolution_time=resolution_time,
            liquidity_param=b,
        )
        self.markets[m.market_id] = m
        self.positions[m.market_id] = {}
        return m

    def get_price(self, market_id: str) -> Tuple[float, float]:
        m = self.markets[market_id]
        p_yes = lmsr_price_yes(m.shares_yes, m.shares_no, m.liquidity_param)
        return (p_yes, 1.0 - p_yes)

    def cost_of_trade(self, market_id: str, direction: str, shares: float) -> float:
        """Return the cost to buy *shares* in *direction* under LMSR."""
        if shares <= 0:
            raise ValueError("Shares must be positive.")
        m = self.markets[market_id]
        old_cost = lmsr_cost(m.shares_yes, m.shares_no, m.liquidity_param)
        if direction == "yes":
            new_cost = lmsr_cost(m.shares_yes + shares, m.shares_no, m.liquidity_param)
        elif direction == "no":
            new_cost = lmsr_cost(m.shares_yes, m.shares_no + shares, m.liquidity_param)
        else:
            raise ValueError(f"Direction must be 'yes' or 'no', got {direction!r}")
        return new_cost - old_cost

    def trade(self, agent_id: str, market_id: str, direction: str, shares: float) -> Trade:
        if market_id not in self.markets:
            raise KeyError(f"Market {market_id} not found.")
        m = self.markets[market_id]
        if m.resolved:
            raise RuntimeError(f"Market {market_id} already resolved.")
        if shares <= 0:
            raise ValueError("Shares must be positive.")

        cost = self.cost_of_trade(market_id, direction, shares)
        if cost > self.balances.get(agent_id, 0):
            raise ValueError(
                f"Agent {agent_id} has insufficient balance "
                f"({self.balances.get(agent_id, 0):.2f} < {cost:.2f})."
            )

        # Execute trade
        self.balances[agent_id] -= cost
        if direction == "yes":
            m.shares_yes += shares
        else:
            m.shares_no += shares

        # Update position
        if agent_id not in self.positions[market_id]:
            self.positions[market_id][agent_id] = Position(
                agent_id=agent_id, market_id=market_id
            )
        pos = self.positions[market_id][agent_id]
        if direction == "yes":
            pos.shares_yes += shares
        else:
            pos.shares_no += shares
        pos.total_invested += cost

        t = Trade(
            agent_id=agent_id,
            market_id=market_id,
            direction=direction,
            shares=shares,
            cost=cost,
        )
        self.trades.append(t)

        # Update incremental indexes
        if market_id not in self._market_volume:
            self._market_volume[market_id] = 0.0
            self._market_traders[market_id] = set()
        self._market_volume[market_id] += cost
        self._market_traders[market_id].add(agent_id)
        if agent_id not in self._agent_markets:
            self._agent_markets[agent_id] = set()
        self._agent_markets[agent_id].add(market_id)

        return t

    def resolve_market(self, market_id: str, outcome: bool) -> None:
        m = self.markets[market_id]
        if m.resolved:
            raise RuntimeError(f"Market {market_id} already resolved.")
        m.resolved = True
        m.outcome = outcome

        # Pay out: each YES share pays 1.0 if outcome=True, each NO share pays 1.0 if False
        for pos in self.positions.get(market_id, {}).values():
            payout = pos.shares_yes if outcome else pos.shares_no
            self.balances[pos.agent_id] += payout

        self.verdicts[market_id] = MarketVerdict(
            market_id=market_id,
            resolution_method="manual",
            confidence=1.0,
            evidence_summary=f"Manually resolved as {'YES' if outcome else 'NO'}.",
        )

    def auto_resolve(self, market_id: str) -> MarketVerdict:
        """Resolve via weighted agent consensus — agents with higher confidence
        get more weight.  The majority-weighted vote determines the outcome."""
        m = self.markets[market_id]
        if m.resolved:
            raise RuntimeError(f"Market {market_id} already resolved.")

        yes_weight = 0.0
        no_weight = 0.0
        for agent in self.agents.values():
            w = agent.confidence
            # Non-byzantine agents vote based on market price signal
            p_yes, _ = self.get_price(market_id)
            if agent.byzantine:
                # Byzantine agents vote randomly
                if random.random() < 0.5:
                    yes_weight += w
                else:
                    no_weight += w
            else:
                if p_yes >= 0.5:
                    yes_weight += w
                else:
                    no_weight += w

        total = yes_weight + no_weight
        outcome = yes_weight >= no_weight
        confidence = max(yes_weight, no_weight) / total if total > 0 else 0.5

        m.resolved = True
        m.outcome = outcome

        # Pay out
        for pos in self.positions.get(market_id, {}).values():
            payout = pos.shares_yes if outcome else pos.shares_no
            self.balances[pos.agent_id] += payout

        verdict = MarketVerdict(
            market_id=market_id,
            resolution_method="consensus",
            confidence=confidence,
            evidence_summary=(
                f"Consensus vote: YES={yes_weight:.2f} vs NO={no_weight:.2f}. "
                f"Resolved {'YES' if outcome else 'NO'} with {confidence:.0%} confidence."
            ),
        )
        self.verdicts[market_id] = verdict
        return verdict

    # ── Queries ──────────────────────────────────────────────────────

    def get_market_snapshot(self, market_id: str) -> MarketSnapshot:
        m = self.markets[market_id]
        p_yes, p_no = self.get_price(market_id)
        volume = self._market_volume.get(market_id, 0.0)
        num_traders = len(self._market_traders.get(market_id, ()))
        return MarketSnapshot(
            market_id=m.market_id,
            question=m.question,
            price_yes=round(p_yes, 4),
            price_no=round(p_no, 4),
            volume=round(volume, 2),
            num_traders=num_traders,
            resolved=m.resolved,
            outcome=m.outcome,
        )

    def get_portfolio(self, agent_id: str) -> AgentPortfolio:
        markets_traded = self._agent_markets.get(agent_id, set())

        total_invested = 0.0
        total_payout = 0.0
        correct = 0
        resolved_count = 0
        for mid in markets_traded:
            m = self.markets[mid]
            pos = self.positions.get(mid, {}).get(agent_id)
            if pos:
                total_invested += pos.total_invested
            if m.resolved and pos:
                resolved_count += 1
                payout = pos.shares_yes if m.outcome else pos.shares_no
                total_payout += payout
                # Agent was "correct" if they held more shares on the winning side
                winning = pos.shares_yes if m.outcome else pos.shares_no
                losing = pos.shares_no if m.outcome else pos.shares_yes
                if winning > losing:
                    correct += 1

        return AgentPortfolio(
            agent_id=agent_id,
            balance=round(self.balances.get(agent_id, 0), 2),
            markets_traded=len(markets_traded),
            total_profit=round(total_payout - total_invested, 2),
            accuracy=round(correct / resolved_count, 4) if resolved_count > 0 else 0.0,
        )

    def get_leaderboard(self) -> List[AgentPortfolio]:
        portfolios = [self.get_portfolio(aid) for aid in self.agents]
        return sorted(portfolios, key=lambda p: p.total_profit, reverse=True)

    # ── Simulation ───────────────────────────────────────────────────

    async def run_simulation(
        self,
        questions: Optional[List[str]] = None,
        rounds: int = 10,
    ) -> List[MarketSnapshot]:
        if questions is None:
            questions = random.sample(SAMPLE_QUESTIONS, min(5, len(SAMPLE_QUESTIONS)))

        # Create markets
        agent_ids = list(self.agents.keys())
        for q in questions:
            creator = random.choice(agent_ids)
            self.create_market(creator, q)

        market_ids = list(self.markets.keys())

        # Trading rounds
        for _round in range(rounds):
            random.shuffle(agent_ids)
            for aid in agent_ids:
                agent = self.agents[aid]
                # Each agent picks 1-3 markets to trade in this round
                targets = random.sample(market_ids, min(random.randint(1, 3), len(market_ids)))
                for mid in targets:
                    m = self.markets[mid]
                    if m.resolved:
                        continue
                    try:
                        shares = random.uniform(1.0, 10.0)
                        if agent.byzantine:
                            direction = random.choice(["yes", "no"])
                        else:
                            # Honest agent: trade based on confidence
                            # High confidence → YES, low → NO (simplified)
                            p_yes, _ = self.get_price(mid)
                            # Agent thinks truth is ~ their confidence
                            believed_prob = agent.confidence
                            if believed_prob > p_yes + 0.05:
                                direction = "yes"
                            elif believed_prob < p_yes - 0.05:
                                direction = "no"
                            else:
                                continue  # price ~= belief, skip
                        self.trade(aid, mid, direction, shares)
                    except (ValueError, RuntimeError):
                        pass  # insufficient balance or resolved

        # Auto-resolve all markets
        for mid in market_ids:
            if not self.markets[mid].resolved:
                self.auto_resolve(mid)

        return [self.get_market_snapshot(mid) for mid in market_ids]

    # ── Export ────────────────────────────────────────────────────────

    def export_json(self, path: str) -> None:
        data = {
            "markets": [self.get_market_snapshot(mid).model_dump() for mid in self.markets],
            "leaderboard": [p.model_dump() for p in self.get_leaderboard()],
            "trades": [t.model_dump() for t in self.trades],
            "verdicts": {k: v.model_dump() for k, v in self.verdicts.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def export_html(self, path: str) -> None:
        snapshots = [self.get_market_snapshot(mid) for mid in self.markets]
        leaderboard = self.get_leaderboard()

        # Build market rows
        market_rows = ""
        for s in snapshots:
            outcome_str = ""
            if s.resolved:
                outcome_str = f'<span class="badge {"yes" if s.outcome else "no"}">{"YES" if s.outcome else "NO"}</span>'
            else:
                outcome_str = '<span class="badge open">OPEN</span>'
            market_rows += f"""<tr>
                <td>{html_mod.escape(s.question)}</td>
                <td class="num">{s.price_yes:.1%}</td>
                <td class="num">{s.price_no:.1%}</td>
                <td class="num">${s.volume:.0f}</td>
                <td class="num">{s.num_traders}</td>
                <td>{outcome_str}</td>
            </tr>\n"""

        # Build leaderboard rows
        lb_rows = ""
        for i, p in enumerate(leaderboard, 1):
            agent = self.agents.get(p.agent_id)
            agent_type = "🤖 Byzantine" if agent and agent.byzantine else "✅ Honest"
            profit_class = "profit" if p.total_profit >= 0 else "loss"
            lb_rows += f"""<tr>
                <td class="num">{i}</td>
                <td>{html_mod.escape(p.agent_id)}</td>
                <td>{agent_type}</td>
                <td class="num">${p.balance:.0f}</td>
                <td class="num {profit_class}">${p.total_profit:+.0f}</td>
                <td class="num">{p.accuracy:.0%}</td>
                <td class="num">{p.markets_traded}</td>
            </tr>\n"""

        # Build trade log (last 50)
        recent_trades = self.trades[-50:]
        trade_rows = ""
        for t in reversed(recent_trades):
            dir_class = "yes" if t.direction == "yes" else "no"
            trade_rows += f"""<tr>
                <td>{html_mod.escape(t.agent_id)}</td>
                <td>{html_mod.escape(t.market_id)}</td>
                <td><span class="badge {dir_class}">{t.direction.upper()}</span></td>
                <td class="num">{t.shares:.1f}</td>
                <td class="num">${t.cost:.2f}</td>
            </tr>\n"""

        # Verdict rows
        verdict_rows = ""
        for mid, v in self.verdicts.items():
            q = self.markets[mid].question if mid in self.markets else mid
            verdict_rows += f"""<tr>
                <td>{html_mod.escape(q)}</td>
                <td>{html_mod.escape(v.resolution_method)}</td>
                <td class="num">{v.confidence:.0%}</td>
                <td>{html_mod.escape(v.evidence_summary)}</td>
            </tr>\n"""

        page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>mBFT Prediction Market Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:2rem}}
h1{{font-size:1.8rem;margin-bottom:.5rem;color:#38bdf8}}
h2{{font-size:1.3rem;margin:2rem 0 1rem;color:#7dd3fc;border-bottom:1px solid #1e293b;padding-bottom:.5rem}}
.subtitle{{color:#94a3b8;margin-bottom:2rem}}
table{{width:100%;border-collapse:collapse;margin-bottom:1.5rem}}
th{{text-align:left;padding:.6rem .8rem;background:#1e293b;color:#94a3b8;font-size:.85rem;text-transform:uppercase;letter-spacing:.05em}}
td{{padding:.5rem .8rem;border-bottom:1px solid #1e293b}}
tr:hover{{background:#1e293b80}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.badge{{padding:.15rem .5rem;border-radius:.25rem;font-size:.8rem;font-weight:600}}
.badge.yes{{background:#166534;color:#4ade80}}
.badge.no{{background:#7f1d1d;color:#f87171}}
.badge.open{{background:#1e3a5f;color:#38bdf8}}
.profit{{color:#4ade80}}
.loss{{color:#f87171}}
.stats{{display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:2rem}}
.stat-card{{background:#1e293b;padding:1rem 1.5rem;border-radius:.5rem;min-width:140px}}
.stat-val{{font-size:1.5rem;font-weight:700;color:#38bdf8}}
.stat-label{{font-size:.8rem;color:#94a3b8;margin-top:.25rem}}
</style></head><body>
<h1>🔮 mBFT Prediction Market</h1>
<p class="subtitle">Swarm intelligence aggregated through belief trading</p>

<div class="stats">
  <div class="stat-card"><div class="stat-val">{len(self.markets)}</div><div class="stat-label">Markets</div></div>
  <div class="stat-card"><div class="stat-val">{len(self.agents)}</div><div class="stat-label">Agents</div></div>
  <div class="stat-card"><div class="stat-val">{len(self.trades)}</div><div class="stat-label">Trades</div></div>
  <div class="stat-card"><div class="stat-val">{sum(1 for m in self.markets.values() if m.resolved)}</div><div class="stat-label">Resolved</div></div>
</div>

<h2>📊 Markets</h2>
<table><thead><tr><th>Question</th><th>P(YES)</th><th>P(NO)</th><th>Volume</th><th>Traders</th><th>Status</th></tr></thead>
<tbody>{market_rows}</tbody></table>

<h2>🏆 Leaderboard</h2>
<table><thead><tr><th>#</th><th>Agent</th><th>Type</th><th>Balance</th><th>Profit</th><th>Accuracy</th><th>Markets</th></tr></thead>
<tbody>{lb_rows}</tbody></table>

<h2>🔍 Resolution Verdicts</h2>
<table><thead><tr><th>Question</th><th>Method</th><th>Confidence</th><th>Evidence</th></tr></thead>
<tbody>{verdict_rows}</tbody></table>

<h2>📜 Recent Trades</h2>
<table><thead><tr><th>Agent</th><th>Market</th><th>Direction</th><th>Shares</th><th>Cost</th></tr></thead>
<tbody>{trade_rows}</tbody></table>

<script>
// Simple sort on click
document.querySelectorAll('th').forEach(th => {{
  th.style.cursor = 'pointer';
  th.addEventListener('click', () => {{
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const idx = Array.from(th.parentNode.children).indexOf(th);
    const asc = th.dataset.sort !== 'asc';
    th.dataset.sort = asc ? 'asc' : 'desc';
    rows.sort((a, b) => {{
      const av = a.children[idx]?.textContent.trim() || '';
      const bv = b.children[idx]?.textContent.trim() || '';
      const an = parseFloat(av.replace(/[$%+,]/g, ''));
      const bn = parseFloat(bv.replace(/[$%+,]/g, ''));
      if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
      return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>
</body></html>"""
        Path(path).write_text(page, encoding="utf-8")

    # ── Pretty-print ─────────────────────────────────────────────────

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("  🔮  mBFT PREDICTION MARKET RESULTS")
        print("=" * 70)

        print(f"\n  Markets: {len(self.markets)}  |  Agents: {len(self.agents)}  |  Trades: {len(self.trades)}")

        print("\n  MARKETS")
        print("  " + "-" * 66)
        for mid in self.markets:
            s = self.get_market_snapshot(mid)
            status = "OPEN"
            if s.resolved:
                status = "YES ✓" if s.outcome else "NO ✗"
            print(f"  {s.question[:50]:<50s}  P(Y)={s.price_yes:.0%}  [{status}]")

        print("\n  LEADERBOARD")
        print("  " + "-" * 66)
        print(f"  {'#':<3} {'Agent':<14} {'Type':<12} {'Balance':>8} {'Profit':>9} {'Acc':>6}")
        for i, p in enumerate(self.get_leaderboard(), 1):
            agent = self.agents.get(p.agent_id)
            atype = "Byzantine" if agent and agent.byzantine else "Honest"
            print(f"  {i:<3} {p.agent_id:<14} {atype:<12} ${p.balance:>7.0f} {p.total_profit:>+8.0f} {p.accuracy:>5.0%}")

        if self.verdicts:
            print("\n  VERDICTS")
            print("  " + "-" * 66)
            for mid, v in self.verdicts.items():
                q = self.markets[mid].question[:40] if mid in self.markets else mid
                print(f"  {q:<40s}  [{v.resolution_method}] {v.confidence:.0%}")
        print()


# ── CLI ──────────────────────────────────────────────────────────────────


def _build_agents(n_agents: int, n_byzantine: int) -> List[MockAgent]:
    agents: List[MockAgent] = []
    for i in range(n_agents):
        is_byz = i < n_byzantine
        agents.append(
            MockAgent(
                agent_id=f"agent-{i}",
                answer="yes" if not is_byz else random.choice(["yes", "no"]),
                confidence=random.uniform(0.3, 0.95),
                byzantine=is_byz,
            )
        )
    return agents


async def _main(args: argparse.Namespace) -> None:
    random.seed(42)
    agents = _build_agents(args.agents, args.byzantine)
    engine = PredictionMarketEngine(
        agents=agents,
        liquidity=args.liquidity,
        initial_balance=args.balance,
    )

    questions = random.sample(SAMPLE_QUESTIONS, min(args.questions, len(SAMPLE_QUESTIONS)))
    await engine.run_simulation(questions=questions, rounds=args.rounds)

    engine.print_summary()

    if args.export:
        engine.export_html(args.export)
        print(f"  HTML report → {args.export}")
    if args.json:
        engine.export_json(args.json)
        print(f"  JSON data   → {args.json}")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="mBFT Prediction Market Engine — swarm belief trading"
    )
    parser.add_argument("--agents", type=int, default=7, help="Number of agents")
    parser.add_argument("--byzantine", type=int, default=1, help="Byzantine agent count")
    parser.add_argument("--questions", type=int, default=5, help="Number of markets")
    parser.add_argument("--rounds", type=int, default=10, help="Trading rounds")
    parser.add_argument("--liquidity", type=float, default=100.0, help="LMSR liquidity parameter")
    parser.add_argument("--balance", type=float, default=1000.0, help="Initial agent balance")
    parser.add_argument("--export", type=str, default=None, help="Export HTML report")
    parser.add_argument("--json", type=str, default=None, help="Export JSON data")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    cli()
