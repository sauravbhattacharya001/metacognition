"""Tests for src.economy (Consensus Economy Simulator).

Covers:
- EconAgent: roi, win_rate, decide_investment for all 6 strategies + edge cases
- MarketState defaults
- _compute_gini: equality, max inequality, single-agent
- _autonomous_fiscal_policy: redistribute, bailout, stimulus, tighten, no-op
- _apply_fiscal_policy: subsidy_pool bounding, redistribute targets poorest
- run_economy: end-to-end shape, deterministic ranking ordering
- _generate_html: contains key sections
"""
from __future__ import annotations

import asyncio
import random

import pytest

from src.economy import (
    EconAgent,
    FiscalPolicy,
    MarketState,
    STRATEGIES,
    _apply_fiscal_policy,
    _autonomous_fiscal_policy,
    _compute_gini,
    _generate_html,
    run_economy,
)


# ---------------------------------------------------------------------------
# EconAgent
# ---------------------------------------------------------------------------

class TestEconAgentProperties:
    def test_roi_with_no_investment_is_zero(self):
        a = EconAgent("a", "balanced", budget=100.0)
        assert a.roi == 0.0

    def test_roi_positive_when_returns_exceed_investment(self):
        a = EconAgent("a", "balanced")
        a.total_invested = 100.0
        a.total_returns = 150.0
        assert a.roi == pytest.approx(0.5)

    def test_roi_negative_when_returns_below_investment(self):
        a = EconAgent("a", "balanced")
        a.total_invested = 100.0
        a.total_returns = 40.0
        assert a.roi == pytest.approx(-0.6)

    def test_win_rate_no_rounds_is_zero(self):
        a = EconAgent("a", "balanced")
        assert a.win_rate == 0.0

    def test_win_rate_computed_from_wins_and_losses(self):
        a = EconAgent("a", "balanced")
        a.wins = 7
        a.losses = 3
        assert a.win_rate == pytest.approx(0.7)


class TestDecideInvestment:
    def test_zero_budget_returns_zero(self):
        a = EconAgent("a", "aggressive", budget=0.0)
        assert a.decide_investment(0) == 0.0

    def test_negative_budget_clamps_and_returns_zero(self):
        a = EconAgent("a", "aggressive", budget=-50.0)
        assert a.decide_investment(0) == 0.0
        assert a.budget == 0.0  # clamped

    def test_conservative_invests_15_percent(self):
        a = EconAgent("a", "conservative", budget=100.0)
        assert a.decide_investment(0) == pytest.approx(15.0)

    def test_aggressive_invests_50_percent(self):
        a = EconAgent("a", "aggressive", budget=100.0)
        assert a.decide_investment(0) == pytest.approx(50.0)

    def test_balanced_invests_25_percent(self):
        a = EconAgent("a", "balanced", budget=100.0)
        assert a.decide_investment(0) == pytest.approx(25.0)

    def test_unknown_strategy_falls_back_to_balanced(self):
        a = EconAgent("a", "made-up", budget=100.0)
        # The else branch returns 25% just like balanced.
        assert a.decide_investment(0) == pytest.approx(25.0)

    def test_contrarian_scales_with_loss_streak(self):
        a = EconAgent("a", "contrarian", budget=100.0)
        # No history -> base 20%
        base = a.decide_investment(0)
        assert base == pytest.approx(20.0)
        # 3-loss streak -> 20 + 30 = 50%
        a.history = [{"won": False}] * 3
        assert a.decide_investment(0) == pytest.approx(50.0)
        # Very long streak still caps at 60%
        a.history = [{"won": False}] * 20
        assert a.decide_investment(0) == pytest.approx(60.0)

    def test_contrarian_resets_streak_on_win(self):
        a = EconAgent("a", "contrarian", budget=100.0)
        a.history = [{"won": True}, {"won": False}, {"won": False}]
        # Reversed: F, F, T -> streak counts 2 then stops -> 20 + 20 = 40
        assert a.decide_investment(0) == pytest.approx(40.0)

    def test_adaptive_no_history_uses_25_percent(self):
        a = EconAgent("a", "adaptive", budget=100.0)
        assert a.decide_investment(0) == pytest.approx(25.0)

    def test_adaptive_all_wins_uses_50_percent(self):
        a = EconAgent("a", "adaptive", budget=100.0)
        a.history = [{"won": True}] * 5
        # 0.15 + 1.0*0.35 = 0.50
        assert a.decide_investment(0) == pytest.approx(50.0)

    def test_adaptive_all_losses_uses_15_percent(self):
        a = EconAgent("a", "adaptive", budget=100.0)
        a.history = [{"won": False}] * 5
        assert a.decide_investment(0) == pytest.approx(15.0)

    def test_momentum_scales_with_win_streak(self):
        a = EconAgent("a", "momentum", budget=100.0)
        assert a.decide_investment(0) == pytest.approx(20.0)
        a.history = [{"won": True}, {"won": True}]
        # 0.20 + 2*0.08 = 0.36
        assert a.decide_investment(0) == pytest.approx(36.0)
        a.history = [{"won": True}] * 50
        # cap at 0.55
        assert a.decide_investment(0) == pytest.approx(55.0)


# ---------------------------------------------------------------------------
# MarketState
# ---------------------------------------------------------------------------

class TestMarketState:
    def test_default_values(self):
        m = MarketState()
        assert m.inflation_rate == pytest.approx(0.02)
        assert m.tax_rate == pytest.approx(0.05)
        assert m.subsidy_pool == 0.0
        assert m.total_gdp == 0.0
        assert m.round_idx == 0
        assert m.history == []


# ---------------------------------------------------------------------------
# Gini
# ---------------------------------------------------------------------------

class TestGini:
    def test_perfect_equality_is_zero(self):
        assert _compute_gini([10.0, 10.0, 10.0, 10.0]) == pytest.approx(0.0)

    def test_max_inequality_approaches_one_minus_one_over_n(self):
        # Standard discrete Gini of [0,...,0,x] over n samples = (n-1)/n.
        g = _compute_gini([0.0, 0.0, 0.0, 100.0])
        assert g == pytest.approx(0.75, abs=1e-6)

    def test_single_value(self):
        # One sample → no inequality.
        assert _compute_gini([42.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fiscal policy
# ---------------------------------------------------------------------------

def _agents_with_budgets(budgets):
    return [EconAgent(f"a{i}", "balanced", budget=b)
            for i, b in enumerate(budgets)]


class TestAutonomousFiscalPolicy:
    def test_no_action_when_economy_healthy(self):
        market = MarketState()
        agents = _agents_with_budgets([100, 105, 95, 100])
        assert _autonomous_fiscal_policy(market, agents) is None

    def test_redistributes_on_high_gini(self):
        market = MarketState()
        agents = _agents_with_budgets([0, 0, 0, 0, 1000])
        policy = _autonomous_fiscal_policy(market, agents)
        assert policy is not None
        assert policy.action == "redistribute"
        assert "Gini" in policy.reason
        assert "tax_increase" in policy.parameters

    def test_bailout_on_high_bankruptcy_ratio(self):
        market = MarketState()
        # Gini of (k zeros, n-k equal positives) = k/n. With 4 bankrupt out
        # of 10, gini = 0.4 (below the 0.5 redistribute threshold) while
        # bankrupt_ratio = 0.4 > 0.3 triggers bailout.
        agents = _agents_with_budgets([100, 100, 100, 100, 100, 100, 0, 0, 0, 0])
        policy = _autonomous_fiscal_policy(market, agents)
        assert policy is not None
        assert policy.action == "bailout"

    def test_stimulus_when_avg_low_after_round_3(self):
        market = MarketState(round_idx=5)
        agents = _agents_with_budgets([20, 22, 18, 21])
        policy = _autonomous_fiscal_policy(market, agents)
        assert policy is not None
        assert policy.action == "stimulus"

    def test_no_stimulus_before_round_3(self):
        market = MarketState(round_idx=2)
        agents = _agents_with_budgets([20, 22, 18, 21])
        # Equal-ish budgets, no bankruptcies, low avg but too early -> None
        assert _autonomous_fiscal_policy(market, agents) is None

    def test_tighten_on_high_inflation(self):
        market = MarketState(inflation_rate=0.10, round_idx=5)
        # Budgets well above stimulus threshold and equal so no other policy fires.
        agents = _agents_with_budgets([100, 100, 100, 100])
        policy = _autonomous_fiscal_policy(market, agents)
        assert policy is not None
        assert policy.action == "tighten"


class TestApplyFiscalPolicy:
    def test_redistribute_caps_tax_at_25_percent(self):
        market = MarketState(tax_rate=0.24, subsidy_pool=100.0)
        agents = _agents_with_budgets([10, 20, 30, 40])
        policy = FiscalPolicy(
            action="redistribute", reason="test",
            parameters={"tax_increase": 0.10, "subsidy_amount": 5.0},
        )
        _apply_fiscal_policy(policy, market, agents)
        assert market.tax_rate == pytest.approx(0.25)

    def test_redistribute_targets_poorest_third(self):
        market = MarketState(subsidy_pool=1000.0, tax_rate=0.05)
        agents = _agents_with_budgets([10, 20, 30, 40, 50, 60])
        before = [a.budget for a in agents]
        policy = FiscalPolicy(
            action="redistribute", reason="test",
            parameters={"tax_increase": 0.01, "subsidy_amount": 100.0},
        )
        _apply_fiscal_policy(policy, market, agents)
        # Sorted by budget asc, poorest two (n//3 = 2) get subsidy
        assert agents[0].budget == pytest.approx(before[0] + 100.0)
        assert agents[1].budget == pytest.approx(before[1] + 100.0)
        # Richer agents unchanged
        for i in range(2, 6):
            assert agents[i].budget == pytest.approx(before[i])

    def test_redistribute_bounded_by_subsidy_pool(self):
        market = MarketState(subsidy_pool=50.0, tax_rate=0.05)
        agents = _agents_with_budgets([10, 20, 30])
        policy = FiscalPolicy(
            action="redistribute", reason="t",
            parameters={"tax_increase": 0.0, "subsidy_amount": 100.0},
        )
        _apply_fiscal_policy(policy, market, agents)
        # Pool only had 50: poorest gets 50, pool drains, second poorest gets 0.
        assert market.subsidy_pool == pytest.approx(0.0)
        assert agents[0].budget == pytest.approx(60.0)  # 10 + 50
        # Second poorest got nothing - pool exhausted
        assert agents[1].budget == pytest.approx(20.0)

    def test_bailout_only_helps_bankrupt_agents(self):
        market = MarketState(subsidy_pool=100.0)
        agents = _agents_with_budgets([50.0, 0.0, 0.0, 30.0])
        policy = FiscalPolicy(action="bailout", reason="t",
                               parameters={"bailout_amount": 20.0})
        _apply_fiscal_policy(policy, market, agents)
        assert agents[0].budget == pytest.approx(50.0)  # unchanged
        assert agents[1].budget == pytest.approx(20.0)
        assert agents[1].bankruptcies == 1
        assert agents[2].budget == pytest.approx(20.0)
        assert agents[2].bankruptcies == 1
        assert agents[3].budget == pytest.approx(30.0)  # unchanged
        assert market.subsidy_pool == pytest.approx(60.0)

    def test_stimulus_distributes_to_all(self):
        market = MarketState(subsidy_pool=100.0)
        agents = _agents_with_budgets([10.0, 20.0, 30.0])
        policy = FiscalPolicy(action="stimulus", reason="t",
                               parameters={"stimulus_amount": 10.0})
        _apply_fiscal_policy(policy, market, agents)
        assert [a.budget for a in agents] == pytest.approx([20.0, 30.0, 40.0])
        assert market.subsidy_pool == pytest.approx(70.0)

    def test_tighten_reduces_inflation_but_not_below_zero(self):
        market = MarketState(inflation_rate=0.01)
        agents = _agents_with_budgets([10.0])
        policy = FiscalPolicy(action="tighten", reason="t",
                               parameters={"inflation_reduction": 0.05})
        _apply_fiscal_policy(policy, market, agents)
        assert market.inflation_rate == pytest.approx(0.0)

    def test_unknown_action_is_noop(self):
        market = MarketState(subsidy_pool=100.0, tax_rate=0.05,
                              inflation_rate=0.02)
        agents = _agents_with_budgets([10.0])
        policy = FiscalPolicy(action="mystery", reason="t")
        _apply_fiscal_policy(policy, market, agents)
        assert market.subsidy_pool == 100.0
        assert market.tax_rate == 0.05
        assert market.inflation_rate == 0.02
        assert agents[0].budget == 10.0


# ---------------------------------------------------------------------------
# Integration: run_economy
# ---------------------------------------------------------------------------

class TestRunEconomy:
    def test_smoke_shape(self):
        random.seed(123)
        data = asyncio.run(run_economy(
            num_agents=3, num_rounds=2, strategy_mix="balanced"))
        assert set(data.keys()) == {
            "config", "rankings", "rounds", "fiscal_log", "market_final"
        }
        assert data["config"]["num_agents"] == 3
        assert data["config"]["num_rounds"] == 2
        assert len(data["rankings"]) == 3
        assert len(data["rounds"]) == 2

    def test_rankings_sorted_by_budget_descending(self):
        random.seed(7)
        data = asyncio.run(run_economy(
            num_agents=4, num_rounds=3, strategy_mix="diverse"))
        budgets = [r["final_budget"] for r in data["rankings"]]
        assert budgets == sorted(budgets, reverse=True)
        # Ranks are 1..N and contiguous.
        assert [r["rank"] for r in data["rankings"]] == [1, 2, 3, 4]

    def test_diverse_assigns_known_strategies(self):
        random.seed(0)
        data = asyncio.run(run_economy(
            num_agents=6, num_rounds=1, strategy_mix="diverse"))
        for r in data["rankings"]:
            assert r["strategy"] in STRATEGIES

    def test_fixed_strategy_assigns_that_strategy_to_all(self):
        random.seed(0)
        data = asyncio.run(run_economy(
            num_agents=3, num_rounds=1, strategy_mix="conservative"))
        for r in data["rankings"]:
            assert r["strategy"] == "conservative"

    def test_autopilot_can_log_fiscal_actions(self):
        random.seed(42)
        data = asyncio.run(run_economy(
            num_agents=4, num_rounds=8, strategy_mix="diverse",
            autopilot=True))
        # We can't guarantee a policy fires every time, but the log must
        # exist and contain only well-formed entries when populated.
        assert isinstance(data["fiscal_log"], list)
        for entry in data["fiscal_log"]:
            assert "round" in entry and "action" in entry and "reason" in entry


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

class TestGenerateHtml:
    def test_contains_core_sections(self):
        random.seed(1)
        data = asyncio.run(run_economy(
            num_agents=3, num_rounds=2, strategy_mix="balanced"))
        html = _generate_html(data)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "Consensus Economy Simulator" in html
        assert "Agent Rankings" in html
        assert "Round-by-Round" in html

    def test_escapes_strategy_names(self):
        # Force a malicious-looking strategy via direct construction; we
        # smoke-test that the helper does not raise and writes the value
        # safely through html.escape().
        data = {
            "config": {"num_agents": 1, "num_rounds": 0,
                        "strategy_mix": "<script>", "autopilot": False,
                        "threshold": 0.5},
            "rankings": [{
                "rank": 1, "agent_id": "<x>", "strategy": "<b>",
                "final_budget": 1.0, "roi": 0.0, "win_rate": 0.0,
                "total_invested": 0.0, "total_returns": 0.0,
                "bankruptcies": 0,
            }],
            "rounds": [],
            "fiscal_log": [],
            "market_final": {"gdp": 0.0, "gini": 0.0, "inflation": 0.0,
                              "tax_rate": 0.0, "subsidy_pool": 0.0},
        }
        out = _generate_html(data)
        # Raw tags must NOT appear; escaped form must.
        assert "<x>" not in out
        assert "&lt;x&gt;" in out
        assert "&lt;b&gt;" in out
