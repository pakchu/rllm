from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import training.search_rex_short_disagreement_rebound_alpha as module


def _preregistration() -> dict:
    return json.loads(module.PREREGISTRATION.read_text(encoding="utf-8"))


def _market(rows: int = 260) -> pd.DataFrame:
    close = np.full(rows, 100.0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 10.0),
            "quote_asset_volume": np.full(rows, 1_000.0),
            "taker_buy_base": np.full(rows, 5.0),
            "taker_buy_quote": np.full(rows, 500.0),
            "open_interest": np.arange(rows, dtype=float) + 1_000.0,
        }
    )


def _feature_stub(rows: int = 260) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rex_strength": np.zeros(rows),
            "rex_direction": np.zeros(rows),
            "price_ret_4h": np.zeros(rows),
            "oi_ret_4h": np.zeros(rows),
            "oi_minus_price_4h": np.zeros(rows),
            "flow_1h": np.zeros(rows),
            "flow_15m": np.zeros(rows),
            "observed_oi": np.full(rows, 1_000.0),
            "oi_available": np.ones(rows, dtype=bool),
            "atr_1h": np.ones(rows),
        }
    )


def test_candidate_grid_matches_frozen_48_cells() -> None:
    rows = module.candidate_grid(_preregistration())
    assert len(rows) == 48
    assert len({row["name"] for row in rows}) == 48


def test_open_interest_is_delayed_one_bar_and_missing_fails_closed() -> None:
    market = _market()
    market.loc[10, "open_interest"] = np.nan
    features = module.build_features(market)

    assert np.isnan(features.loc[0, "observed_oi"])
    assert features.loc[1, "observed_oi"] == market.loc[0, "open_interest"]
    assert np.isnan(features.loc[11, "observed_oi"])
    assert not bool(features.loc[11, "oi_available"])


def test_future_mutation_cannot_change_earlier_features() -> None:
    market = _market()
    before = module.build_features(market)
    mutated = market.copy()
    mutated.loc[
        220:,
        [
            "high",
            "low",
            "close",
            "volume",
            "taker_buy_quote",
            "open_interest",
        ],
    ] *= 7
    after = module.build_features(mutated)

    pd.testing.assert_frame_equal(before.iloc[:220], after.iloc[:220])


def test_raw_clock_uses_strict_threshold_and_short_direction() -> None:
    features = _feature_stub()
    sampled = module._sampled_positions(len(features), 24)
    features.loc[sampled[:3], "rex_strength"] = [1.0, 1.1, 1.2]
    features.loc[sampled[:3], "rex_direction"] = [-1.0, 1.0, -1.0]

    anchors = module.raw_short_anchors(
        features, strength_threshold=1.0, stride_bars=24
    )

    assert anchors.tolist() == [int(sampled[2])]


def test_confirmation_requires_reclaim_flow_oi_atr_and_range_recovery() -> None:
    prereg = _preregistration()
    market = _market()
    features = _feature_stub()
    anchor = int(module._sampled_positions(len(features), 24)[0])
    confirmation = anchor + 6
    features.loc[anchor, ["rex_strength", "rex_direction"]] = [2.0, -1.0]
    features.loc[anchor, "oi_minus_price_4h"] = 2.0
    features.loc[anchor, "flow_1h"] = -0.5
    features.loc[confirmation, "flow_15m"] = 0.5
    features.loc[anchor, "observed_oi"] = 1_000.0
    features.loc[confirmation, "observed_oi"] = 1_001.0
    market.loc[anchor, "close"] = 100.0
    market.loc[anchor + 1 : confirmation, "low"] = 99.5
    market.loc[anchor + 1 : confirmation, "high"] = 101.0
    market.loc[confirmation, "close"] = 100.8
    thresholds = {
        "rex_strength": 1.0,
        "oi_minus_price": {"0.5": 1.0, "0.75": 1.5},
    }
    candidate = next(
        row
        for row in module.candidate_grid(prereg)
        if row["oi_minus_price_anchor_quantile"] == 0.75
        and row["confirmation_wait_bars"] == 6
        and row["minimum_oi_retention"] == 0.0
        and row["minimum_post_anchor_range_position"] == 0.5
        and row["hold_bars"] == 36
    )

    positions = module.signal_positions(
        market, features, thresholds, candidate, prereg
    )
    assert positions.tolist() == [confirmation]

    broken = market.copy()
    broken.loc[confirmation, "close"] = 99.0
    assert (
        module.signal_positions(broken, features, thresholds, candidate, prereg).size
        == 0
    )


def test_simulator_counts_idle_calendar_and_contains_exit() -> None:
    market = _market(1_000)
    market.loc[11:20, "open"] = np.linspace(100.0, 110.0, 10)
    market.loc[11:20, "high"] = market.loc[11:20, "open"] + 1.0
    market.loc[11:20, "low"] = market.loc[11:20, "open"] - 1.0
    result = module.simulate(
        market,
        [10, 998],
        start="2023-01-01",
        end="2024-01-01",
        hold_bars=6,
        leverage=0.5,
        cost_rate=0.0006,
    )

    assert result["trades"] == 1
    assert result["absolute_return_pct"] != result["cagr_pct"]
    assert result["strict_mdd_pct"] > 0.0


def test_module_does_not_reference_future_labelled_rex_jsonl() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "rex_event_reasoning_policy_sft_20260712.jsonl" not in source
