from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import training.search_oi_failed_breakdown_recoil_alpha as module


def _preregistration() -> dict:
    return json.loads(
        module.PREREGISTRATION.read_text(encoding="utf-8")
    )


def _market(rows: int = 90) -> pd.DataFrame:
    close = np.full(rows, 100.0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "quote_asset_volume": np.full(rows, 1_000.0),
            "taker_buy_quote": np.full(rows, 500.0),
            "open_interest": np.arange(rows, dtype=float) + 1_000.0,
        }
    )


def test_candidate_grid_matches_preregistered_96_cells() -> None:
    rows = module.candidate_grid(_preregistration())
    assert len(rows) == 96
    assert len({row["name"] for row in rows}) == 96


def test_open_interest_is_delayed_one_bar_and_missing_fails_closed() -> None:
    market = _market()
    market.loc[10, "open_interest"] = np.nan
    features = module.build_features(market)

    assert np.isnan(features.loc[0, "observed_oi"])
    assert features.loc[1, "observed_oi"] == market.loc[0, "open_interest"]
    assert np.isnan(features.loc[11, "observed_oi"])
    assert not bool(features.loc[11, "oi_available"])


def test_mutating_future_rows_cannot_change_earlier_features() -> None:
    market = _market()
    before = module.build_features(market)
    mutated = market.copy()
    mutated.loc[70:, ["high", "low", "close", "open_interest", "taker_buy_quote"]] *= 7
    after = module.build_features(mutated)

    pd.testing.assert_frame_equal(before.iloc[:70], after.iloc[:70])


def test_confirmation_requires_reclaim_flow_oi_atr_and_range_position() -> None:
    prereg = _preregistration()
    market = _market(80)
    features = pd.DataFrame(
        {
            "price_ret_4h": np.zeros(80),
            "oi_minus_price_4h": np.zeros(80),
            "flow_1h": np.zeros(80),
            "flow_15m": np.zeros(80),
            "range_position_4h": np.ones(80),
            "atr_1h": np.ones(80),
            "observed_oi": np.full(80, 1_000.0),
            "oi_available": np.ones(80, dtype=bool),
        }
    )
    anchor = 48
    wait = 6
    confirmation = anchor + wait
    features.loc[anchor, "price_ret_4h"] = -2.0
    features.loc[anchor, "oi_minus_price_4h"] = 2.0
    features.loc[anchor, "flow_1h"] = -1.0
    features.loc[anchor, "range_position_4h"] = 0.0
    features.loc[confirmation, "flow_15m"] = 1.0
    market.loc[anchor, "close"] = 100.0
    market.loc[anchor + 1 : confirmation, "low"] = 99.0
    market.loc[anchor + 1 : confirmation, "high"] = 101.0
    market.loc[confirmation, "close"] = 100.8
    thresholds = {
        "price_return": {"0.2": -1.0},
        "oi_minus_price": {"0.75": 1.0, "0.9": 1.5},
        "taker_sell": {"0.2": -0.5, "0.35": -0.25},
        "range_position": {"0.25": 0.25},
    }
    candidate = next(
        row
        for row in module.candidate_grid(prereg)
        if row["oi_minus_price_quantile"] == 0.75
        and row["taker_sell_quantile"] == 0.2
        and row["confirmation_wait_bars"] == wait
        and row["minimum_oi_retention"] == 0.0
        and row["minimum_post_anchor_range_position"] == 0.65
        and row["hold_bars"] == 36
    )

    positions = module.signal_positions(
        market, features, thresholds, candidate, prereg
    )
    assert positions.tolist() == [confirmation]

    broken = market.copy()
    broken.loc[confirmation, "close"] = 99.5
    assert (
        module.signal_positions(broken, features, thresholds, candidate, prereg).size
        == 0
    )


def test_simulator_counts_idle_calendar_time_and_split_contains_exit() -> None:
    market = _market(600)
    market.loc[11:20, "open"] = np.linspace(100.0, 110.0, 10)
    market.loc[11:20, "high"] = market.loc[11:20, "open"] + 1.0
    market.loc[11:20, "low"] = market.loc[11:20, "open"] - 1.0
    result = module.simulate(
        market,
        [10, 598],
        start="2023-01-01",
        end="2024-01-01",
        hold_bars=6,
        leverage=0.5,
        cost_rate=0.0006,
    )

    assert result["trades"] == 1
    assert result["absolute_return_pct"] != result["cagr_pct"]
    assert result["strict_mdd_pct"] > 0.0


def test_pre2024_manifest_rejects_mutation(tmp_path) -> None:
    prereg = _preregistration()
    cfg = module.Config(
        preregistration=str(module.PREREGISTRATION),
        pre2024_output=str(tmp_path / "pre.json"),
    )
    payload = {
        "phase": "pre2024_selection",
        "test2024_opened": False,
        "decision": "open_test2024",
        "shortlist": [{"candidate": {"name": "x"}, "stats": {}}],
        "preregistration_sha256": module._sha256(module.PREREGISTRATION),
        "source_prefix": {"frame_hash": "a"},
        "thresholds": {"x": 1.0},
    }
    freeze = {
        "preregistration_sha256": payload["preregistration_sha256"],
        "source_prefix": payload["source_prefix"],
        "thresholds": payload["thresholds"],
        "shortlist": payload["shortlist"],
    }
    payload["freeze_hash"] = module._json_hash(freeze)
    module._validate_pre2024_manifest(cfg, prereg, payload)

    payload["thresholds"]["x"] = 2.0
    with pytest.raises(RuntimeError, match="freeze hash"):
        module._validate_pre2024_manifest(cfg, prereg, payload)
