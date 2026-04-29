"""Tests for Prediction Market Engine."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from src.prediction_market import (
    AgentPortfolio,
    Market,
    MarketSnapshot,
    MarketVerdict,
    Position,
    PredictionMarketEngine,
    Trade,
    lmsr_cost,
    lmsr_price_yes,
)
from src.agents.metacognitive import MockAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agents(n: int = 5, n_byz: int = 1) -> list[MockAgent]:
    agents = []
    for i in range(n):
        agents.append(
            MockAgent(
                agent_id=f"a{i}",
                answer="yes",
                confidence=0.7 + i * 0.05,
                byzantine=i < n_byz,
            )
        )
    return agents


def _engine(n: int = 5, n_byz: int = 1, **kw) -> PredictionMarketEngine:
    return PredictionMarketEngine(_agents(n, n_byz), **kw)


# ---------------------------------------------------------------------------
# LMSR math
# ---------------------------------------------------------------------------

class TestLMSR:
    def test_prices_sum_to_one(self):
        for q_y in [0, 10, 50, -20]:
            for q_n in [0, 5, 50, -10]:
                p = lmsr_price_yes(q_y, q_n, 100.0)
                assert abs(p + (1.0 - p) - 1.0) < 1e-10

    def test_equal_shares_gives_half(self):
        p = lmsr_price_yes(0, 0, 100.0)
        assert abs(p - 0.5) < 1e-10

    def test_more_yes_raises_price(self):
        p1 = lmsr_price_yes(0, 0, 100.0)
        p2 = lmsr_price_yes(50, 0, 100.0)
        assert p2 > p1

    def test_cost_positive_for_buy(self):
        c1 = lmsr_cost(0, 0, 100.0)
        c2 = lmsr_cost(10, 0, 100.0)
        assert c2 > c1

    def test_cost_symmetric(self):
        c_yes = lmsr_cost(10, 0, 100.0) - lmsr_cost(0, 0, 100.0)
        c_no = lmsr_cost(0, 10, 100.0) - lmsr_cost(0, 0, 100.0)
        assert abs(c_yes - c_no) < 1e-10

    def test_extreme_values_clamp(self):
        # Very high q_yes should give price near 1
        p = lmsr_price_yes(100000, 0, 100.0)
        assert p > 0.999

    def test_extreme_negative_clamp(self):
        p = lmsr_price_yes(0, 100000, 100.0)
        assert p < 0.001


# ---------------------------------------------------------------------------
# Engine init
# ---------------------------------------------------------------------------

class TestEngineInit:
    def test_creates_with_agents(self):
        e = _engine(3, 0)
        assert len(e.agents) == 3

    def test_initial_balances(self):
        e = _engine(3, 0, initial_balance=500.0)
        for b in e.balances.values():
            assert b == 500.0

    def test_empty_agents_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            PredictionMarketEngine([])

    def test_zero_liquidity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PredictionMarketEngine(_agents(2, 0), liquidity=0)


# ---------------------------------------------------------------------------
# Market creation
# ---------------------------------------------------------------------------

class TestMarketCreation:
    def test_create_market(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Will it rain?")
        assert m.question == "Will it rain?"
        assert m.creator_id == "a0"
        assert not m.resolved

    def test_market_in_engine(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        assert m.market_id in e.markets

    def test_initial_price_is_fifty_fifty(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        p_yes, p_no = e.get_price(m.market_id)
        assert abs(p_yes - 0.5) < 1e-6
        assert abs(p_no - 0.5) < 1e-6

    def test_custom_liquidity(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?", liquidity_param=50.0)
        assert m.liquidity_param == 50.0


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------

class TestTrading:
    def test_basic_trade(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        t = e.trade("a0", m.market_id, "yes", 5.0)
        assert t.direction == "yes"
        assert t.shares == 5.0
        assert t.cost > 0

    def test_balance_decreases(self):
        e = _engine(3, 0, initial_balance=1000.0)
        m = e.create_market("a0", "Q?")
        before = e.balances["a0"]
        t = e.trade("a0", m.market_id, "yes", 5.0)
        assert e.balances["a0"] == pytest.approx(before - t.cost)

    def test_position_updated(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.trade("a0", m.market_id, "yes", 5.0)
        pos = e.positions[m.market_id]["a0"]
        assert pos.shares_yes == 5.0
        assert pos.shares_no == 0.0

    def test_price_moves_after_yes_trade(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        p_before, _ = e.get_price(m.market_id)
        e.trade("a0", m.market_id, "yes", 20.0)
        p_after, _ = e.get_price(m.market_id)
        assert p_after > p_before

    def test_price_moves_after_no_trade(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        _, p_before = e.get_price(m.market_id)
        e.trade("a0", m.market_id, "no", 20.0)
        _, p_after = e.get_price(m.market_id)
        assert p_after > p_before

    def test_insufficient_balance(self):
        e = _engine(3, 0, initial_balance=1.0)
        m = e.create_market("a0", "Q?")
        with pytest.raises(ValueError, match="insufficient balance"):
            e.trade("a0", m.market_id, "yes", 1000.0)

    def test_zero_shares_raises(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        with pytest.raises(ValueError, match="positive"):
            e.trade("a0", m.market_id, "yes", 0)

    def test_negative_shares_raises(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        with pytest.raises(ValueError, match="positive"):
            e.trade("a0", m.market_id, "yes", -5)

    def test_invalid_direction_raises(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        with pytest.raises(ValueError, match="Direction"):
            e.cost_of_trade(m.market_id, "maybe", 5.0)

    def test_trade_on_resolved_raises(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.resolve_market(m.market_id, True)
        with pytest.raises(RuntimeError, match="resolved"):
            e.trade("a0", m.market_id, "yes", 5.0)

    def test_unknown_market_raises(self):
        e = _engine(3, 0)
        with pytest.raises(KeyError):
            e.trade("a0", "nonexistent", "yes", 5.0)

    def test_multiple_trades_accumulate(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.trade("a0", m.market_id, "yes", 5.0)
        e.trade("a0", m.market_id, "no", 3.0)
        pos = e.positions[m.market_id]["a0"]
        assert pos.shares_yes == 5.0
        assert pos.shares_no == 3.0

    def test_cost_of_trade_matches_actual(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        expected = e.cost_of_trade(m.market_id, "yes", 10.0)
        before = e.balances["a0"]
        e.trade("a0", m.market_id, "yes", 10.0)
        actual = before - e.balances["a0"]
        assert abs(expected - actual) < 1e-10


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestResolution:
    def test_manual_resolve_yes(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.trade("a0", m.market_id, "yes", 10.0)
        bal_before = e.balances["a0"]
        e.resolve_market(m.market_id, True)
        assert e.markets[m.market_id].resolved
        assert e.markets[m.market_id].outcome is True
        assert e.balances["a0"] > bal_before  # got payout

    def test_manual_resolve_no(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.trade("a0", m.market_id, "no", 10.0)
        bal_before = e.balances["a0"]
        e.resolve_market(m.market_id, False)
        assert e.balances["a0"] > bal_before

    def test_double_resolve_raises(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.resolve_market(m.market_id, True)
        with pytest.raises(RuntimeError, match="resolved"):
            e.resolve_market(m.market_id, False)

    def test_auto_resolve(self):
        e = _engine(5, 0)
        m = e.create_market("a0", "Q?")
        # Push price high so consensus votes YES
        e.trade("a0", m.market_id, "yes", 50.0)
        v = e.auto_resolve(m.market_id)
        assert v.resolution_method == "consensus"
        assert v.confidence > 0
        assert e.markets[m.market_id].resolved

    def test_auto_resolve_already_resolved(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.resolve_market(m.market_id, True)
        with pytest.raises(RuntimeError, match="resolved"):
            e.auto_resolve(m.market_id)

    def test_verdict_recorded(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.resolve_market(m.market_id, True)
        assert m.market_id in e.verdicts
        assert e.verdicts[m.market_id].resolution_method == "manual"


# ---------------------------------------------------------------------------
# Portfolio & leaderboard
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_portfolio_initial(self):
        e = _engine(3, 0, initial_balance=1000.0)
        p = e.get_portfolio("a0")
        assert p.balance == 1000.0
        assert p.markets_traded == 0
        assert p.total_profit == 0.0

    def test_portfolio_after_trade(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.trade("a0", m.market_id, "yes", 5.0)
        p = e.get_portfolio("a0")
        assert p.markets_traded == 1

    def test_leaderboard_sorted(self):
        e = _engine(5, 1)
        m = e.create_market("a0", "Q?")
        e.trade("a1", m.market_id, "yes", 10.0)
        e.trade("a2", m.market_id, "no", 10.0)
        e.resolve_market(m.market_id, True)
        lb = e.get_leaderboard()
        profits = [p.total_profit for p in lb]
        assert profits == sorted(profits, reverse=True)

    def test_leaderboard_contains_all_agents(self):
        e = _engine(4, 0)
        lb = e.get_leaderboard()
        assert len(lb) == 4


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_fields(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Will it work?")
        s = e.get_market_snapshot(m.market_id)
        assert s.question == "Will it work?"
        assert not s.resolved
        assert abs(s.price_yes - 0.5) < 0.01

    def test_snapshot_volume(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.trade("a0", m.market_id, "yes", 10.0)
        s = e.get_market_snapshot(m.market_id)
        assert s.volume > 0
        assert s.num_traders == 1


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TestSimulation:
    @pytest.mark.asyncio
    async def test_simulation_runs(self):
        e = _engine(5, 1)
        results = await e.run_simulation(
            questions=["Q1?", "Q2?", "Q3?"], rounds=5
        )
        assert len(results) == 3
        for s in results:
            assert s.resolved

    @pytest.mark.asyncio
    async def test_simulation_default_questions(self):
        e = _engine(5, 1)
        results = await e.run_simulation(rounds=3)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_simulation_trades_generated(self):
        e = _engine(5, 1)
        await e.run_simulation(questions=["Q?"], rounds=5)
        assert len(e.trades) > 0


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_json_export(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.trade("a0", m.market_id, "yes", 5.0)
        e.resolve_market(m.market_id, True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        e.export_json(path)
        data = json.loads(Path(path).read_text())
        assert "markets" in data
        assert "leaderboard" in data
        assert "trades" in data
        Path(path).unlink()

    def test_html_export(self):
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.trade("a0", m.market_id, "yes", 5.0)
        e.resolve_market(m.market_id, True)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        e.export_html(path)
        content = Path(path).read_text(encoding="utf-8")
        assert "Prediction Market" in content
        assert "Leaderboard" in content
        Path(path).unlink()

    def test_json_structure(self):
        e = _engine(3, 1)
        m = e.create_market("a0", "Q?")
        e.trade("a0", m.market_id, "yes", 5.0)
        e.resolve_market(m.market_id, True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        e.export_json(path)
        data = json.loads(Path(path).read_text())
        assert len(data["markets"]) == 1
        assert data["markets"][0]["resolved"] is True
        assert len(data["leaderboard"]) == 3
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_trades_resolve(self):
        """Resolving a market with no trades should work."""
        e = _engine(3, 0)
        m = e.create_market("a0", "Q?")
        e.resolve_market(m.market_id, True)
        assert e.markets[m.market_id].resolved

    def test_many_small_trades(self):
        e = _engine(3, 0, initial_balance=10000.0)
        m = e.create_market("a0", "Q?")
        for _ in range(50):
            e.trade("a0", m.market_id, "yes", 0.1)
        pos = e.positions[m.market_id]["a0"]
        assert abs(pos.shares_yes - 5.0) < 1e-6

    def test_multiple_agents_trade(self):
        e = _engine(5, 0)
        m = e.create_market("a0", "Q?")
        for i in range(5):
            e.trade(f"a{i}", m.market_id, "yes" if i % 2 == 0 else "no", 5.0)
        assert len(e.positions[m.market_id]) == 5
