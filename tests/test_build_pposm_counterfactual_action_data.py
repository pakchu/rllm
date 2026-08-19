from types import SimpleNamespace

import pandas as pd
import pytest

from training import build_pposm_counterfactual_action_data as builder
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine


def _manifest() -> dict:
    return {
        "spec": {
            "side": 1,
            "hold_bars": 2,
            "stop_bps": 1_000_000,
            "capitulation_take_bps": 400,
            "normal_take_bps": 1_200,
        },
        "state_thresholds": {
            "htf_1w_return_1_q50": -1.0,
            "rex_576_range_width_pct_q50": 2.0,
            "quote_vol_z_1d_q20": -1.0,
            "premium_index_change_q67": 3.0,
            "rex_576_range_pos_q67": 0.8,
        },
    }


def _features() -> dict[str, float]:
    return {
        "htf_1w_return_1": -2.0,
        "rex_576_range_width_pct": 3.0,
        "quote_vol_z_1d": 0.0,
        "rex_576_range_pos": 0.9,
        "bb_z": 0.2,
        "premium_index_change": 4.0,
        "htf_3d_return_1": 0.1,
    }


def _engine() -> tuple[ExecutionEngine, SimpleNamespace]:
    market = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=4, freq="5min"),
            "open": [100.0, 100.0, 101.0, 102.0],
            "high": [100.0, 100.0, 101.0, 102.0],
            "low": [100.0, 100.0, 101.0, 102.0],
        }
    )
    funding = pd.DataFrame(columns=["date", "funding_rate"])
    cfg = SimpleNamespace(leverage=0.5, fee_rate=0.0005, slippage_rate=0.0001)
    return ExecutionEngine(market, funding, cfg), cfg


def test_action_utilities_are_exact_base_cost_net_returns():
    engine, cfg = _engine()
    trades = builder.counterfactual_trades(engine, 0, _manifest())
    utilities = builder.action_utilities(trades, cfg)
    one_side = 1.0 - 0.5 * (0.0005 + 0.0001)
    expected = one_side * (1.0 + 0.5 * 0.02) * one_side - 1.0
    assert utilities["SKIP"] == 0.0
    assert utilities["TP4"] == pytest.approx(expected)
    assert utilities["TP12"] == pytest.approx(expected)


def test_prompt_has_symbolic_and_raw_signal_state_but_no_offline_values():
    features = _features()
    predicates = builder.causal_predicates(features, _manifest()["state_thresholds"])
    prompt = builder.causal_prompt(features, predicates)
    assert "causal_predicates" in prompt
    assert "signal_time_state" in prompt
    for forbidden in (
        "action_utilities",
        "net_return",
        "entry_position",
        "exit_position",
        "offline_exact_base_cost_execution",
    ):
        assert forbidden not in prompt


def test_best_action_uses_highest_utility():
    assert builder.best_action({"SKIP": 0.0, "TP4": 0.01, "TP12": 0.03}) == "TP12"
    assert builder.best_action({"SKIP": 0.0, "TP4": 0.02, "TP12": -0.01}) == "TP4"
    assert builder.best_action({"SKIP": 0.0, "TP4": -0.01, "TP12": -0.02}) == "SKIP"


def test_best_action_ties_follow_skip_tp4_tp12_priority():
    assert builder.best_action({"SKIP": 0.0, "TP4": 0.0, "TP12": 0.0}) == "SKIP"
    assert builder.best_action({"SKIP": -1.0, "TP4": 2.0, "TP12": 2.0}) == "TP4"


def test_row_bytes_are_deterministic_and_keep_identity_and_positions(monkeypatch):
    engine, cfg = _engine()
    market = engine.market
    state = pd.DataFrame([_features()] * len(market))
    monkeypatch.setattr(builder.numeric, "_signal_features", lambda _state, _signal: _features())
    kwargs = {
        "split": "train",
        "window": "pre_2024",
        "signal_position": 0,
        "market": market,
        "state": state,
        "manifest": _manifest(),
        "strategy_cfg": cfg,
        "engine": engine,
    }
    first = builder.build_row(**kwargs)
    second = builder.build_row(**kwargs)
    assert builder._jsonl_bytes([first]) == builder._jsonl_bytes([second])
    assert first["metadata"]["identity"] == "pposm-counterfactual-action|pre_2024|0"
    assert first["metadata"]["executable_positions"] == {
        "TP4": {"entry_position": 1, "exit_position": 3},
        "TP12": {"entry_position": 1, "exit_position": 3},
    }
    assert first["target"] == builder.best_action(first["metadata"]["action_utilities"])
