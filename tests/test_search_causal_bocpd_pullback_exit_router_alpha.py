from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from training.search_causal_bocpd_pullback_exit_router_alpha import (
    ROUTER_SPEC,
    Config,
    _frozen_execution_config,
    fit_state_actions,
    trade_utility,
)


class _CounterfactualEngine:
    def trade_at(
        self, signal: int, side: int, hold: int, take_bps: int, stop_bps: int
    ) -> SimpleNamespace:
        del side, hold, stop_bps
        if signal < 5:
            price_factor = 1.03 if take_bps == 400 else 1.01
        else:
            price_factor = 1.025 if take_bps == 400 else 1.005
        return SimpleNamespace(
            price_factor=price_factor,
            funding_factor=1.0,
            funding_debit_factor=1.0,
            adverse_price_factor=0.99,
        )


def _base_trade(signal: int) -> SimpleNamespace:
    return SimpleNamespace(signal_position=signal)


def test_fit_state_actions_requires_minimum_support_before_routing_tp4() -> None:
    states = np.asarray([1] * 5 + [2] * 4, dtype=np.int16)
    capitulation = np.zeros(len(states), dtype=bool)
    trades = [_base_trade(signal) for signal in range(len(states))]

    actions, quality = fit_state_actions(
        _CounterfactualEngine(),
        trades,
        states,
        capitulation,
        risk_lambda=0.0,
        minimum_state_trades=5,
    )

    assert actions[1] == "tp4"
    assert actions[2] == "tp12"
    assert quality["1"]["n"] == 5
    assert quality["2"]["n"] == 4


def test_trade_utility_penalizes_strict_adverse_excursion() -> None:
    trade = SimpleNamespace(
        price_factor=1.02,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        adverse_price_factor=0.90,
    )

    plain = trade_utility(trade, 0.0)
    penalized = trade_utility(trade, 0.25)

    assert penalized < plain


def test_router_grid_discloses_exactly_eight_cells() -> None:
    cells = (
        len(ROUTER_SPEC["hazard_hours"])
        * len(ROUTER_SPEC["primary_quantiles"])
        * len(ROUTER_SPEC["risk_lambdas"])
    )

    assert cells == ROUTER_SPEC["grid_cells"] == 8


def test_frozen_execution_config_is_path_independent() -> None:
    relative = Config()
    absolute = Config(
        input_csv="/tmp/copied/market.csv.gz",
        funding_csv="/tmp/copied/funding.csv.gz",
        premium_csv="/tmp/copied/premium.csv.gz",
    )

    assert _frozen_execution_config(relative) == _frozen_execution_config(absolute)
    assert "input_csv" not in _frozen_execution_config(relative)
