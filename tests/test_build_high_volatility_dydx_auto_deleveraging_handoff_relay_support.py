import json

import numpy as np
import pandas as pd

from training import build_high_volatility_dydx_auto_deleveraging_handoff_relay_support as support


def test_parse_trade_accepts_frozen_risk_engine_types() -> None:
    row = support.parse_trade({
        "id": "abc",
        "side": "SELL",
        "size": "2",
        "price": "100",
        "type": "DELEVERAGED",
        "createdAt": "2025-01-01T00:00:00.000Z",
        "createdAtHeight": "42",
    })
    assert row["notional"] == 200.0
    assert row["type"] == "DELEVERAGED"
    assert row["created_at_height"] == 42


def test_strict_prior_midrank_includes_zero_history_and_excludes_current() -> None:
    ranked = support.strict_prior_midrank(
        pd.Series([0.0, 0.0, 10.0, 20.0]), lookback=3, minimum=2
    )
    assert np.isnan(ranked.iloc[0])
    assert np.isnan(ranked.iloc[1])
    assert ranked.iloc[2] == 1.0
    assert ranked.iloc[3] == 1.0


def test_deleveraging_state_switches_fade_to_follow_and_reserves_equal_exit() -> None:
    times = pd.date_range("2024-01-01T00:00:00Z", periods=12, freq="1h")
    panel = pd.DataFrame({
        "decision_time": times,
        "source_valid": True,
        "forced_buy_notional": [0.0, 100.0] * 6,
        "forced_sell_notional": 0.0,
        "forced_notional": [0.0, 100.0] * 6,
        "forced_flow": [0.0, 100.0] * 6,
        "deleveraged_notional": [0.0, 25.0] + [0.0, 0.0] * 5,
        "deleveraged_state": [False, True] + [False, False] * 5,
        "forced_notional_rank": [0.0, 0.9] * 6,
        "realized_variation": 1.0,
        "realized_variation_rank": 0.9,
    })
    clock = support.candidate_clock(panel)
    assert list(clock.decision_time) == [times[1], times[9]]
    assert list(clock.side) == [1, -1]


def test_openapi_schema_contract_is_exact() -> None:
    prereg = json.loads(support.prereg.DEFAULT_OUTPUT.read_text())
    assert prereg["source_contract"]["accepted_types"] == [
        "LIMIT", "LIQUIDATED", "DELEVERAGED", "TWAP_SUBORDER"
    ]
    assert prereg["policy"]["history_observations"] == 720
