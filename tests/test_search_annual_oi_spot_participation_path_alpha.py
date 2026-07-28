from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import training.search_annual_oi_spot_participation_path_alpha as module


def _preregistration() -> dict:
    return json.loads(module.PREREGISTRATION.read_text(encoding="utf-8"))


def _market(rows: int = 9_000) -> pd.DataFrame:
    date = pd.date_range("2022-01-01", periods=rows, freq="5min")
    trend = np.linspace(100.0, 120.0, rows)
    return pd.DataFrame(
        {
            "date": date,
            "open": trend,
            "high": trend + 1.0,
            "low": trend - 1.0,
            "close": trend,
            "quote_asset_volume": np.linspace(1_000.0, 2_000.0, rows),
            "taker_buy_quote": np.linspace(450.0, 1_100.0, rows),
            "open_interest": np.linspace(10_000.0, 12_000.0, rows),
            "open_interest_value": np.linspace(20_000.0, 24_000.0, rows),
            "spot_close": trend * 1.001,
            "spot_volume": np.linspace(10.0, 20.0, rows),
            "spot_rows": np.full(rows, 5),
        }
    )


def test_frozen_grid_has_24_policy_cells() -> None:
    prereg = _preregistration()
    grid = prereg["frozen_grid"]
    count = (
        len(module.model_specs(prereg))
        * len(grid["score_quantile"])
        * len(grid["allowed_side"])
    )
    assert count == grid["model_policy_cells"] == 24


def test_feature_contract_has_exact_35_non_rank7_columns() -> None:
    assert len(module.FEATURE_COLUMNS) == 35
    forbidden = {
        "dxy_momentum",
        "usdkrw_zscore",
        "kimchi_premium_change",
        "funding_zscore",
        "premium_index_zscore",
        "nested_high_work_ratio",
        "braid_recent_24h_side",
    }
    assert forbidden.isdisjoint(module.FEATURE_COLUMNS)


def test_open_interest_is_delayed_and_missing_fails_closed() -> None:
    market = _market()
    market.loc[4_000, "open_interest"] = np.nan
    _, valid = module.build_features(market)

    assert not bool(valid[4_001])
    clean = _market()
    features_clean, valid_clean = module.build_features(clean)
    mutated = clean.copy()
    mutated.loc[8_500:, ["open_interest", "spot_close", "spot_volume"]] *= 7
    features_mutated, valid_mutated = module.build_features(mutated)
    pd.testing.assert_frame_equal(
        features_clean.iloc[:8_500], features_mutated.iloc[:8_500]
    )
    assert np.array_equal(valid_clean[:8_500], valid_mutated[:8_500])


def test_decision_clock_is_completed_hour_bar_only() -> None:
    market = _market()
    _, valid = module.build_features(market)
    decisions = module.decision_mask(market, valid)
    dates = pd.to_datetime(market.loc[decisions, "date"])
    assert len(dates) > 0
    assert dates.dt.minute.eq(0).all()


def test_policy_masks_respect_side_and_positive_threshold() -> None:
    fitted = {
        "fit_predictions": np.asarray(
            [[0.1, -0.2], [0.2, -0.1], [0.3, 0.05], [0.4, 0.1]]
        ),
        "predict_positions": np.asarray([1, 2, 3, 4]),
        "predictions": np.asarray(
            [[0.5, 0.1], [0.1, 0.6], [-0.2, -0.1], [0.7, 0.2]]
        ),
    }
    long_active, short_active, meta = module.policy_masks(
        8, fitted, quantile=0.8, allowed_side="both"
    )
    assert long_active.tolist() == [False, True, False, False, True, False, False, False]
    assert short_active.tolist() == [False, False, True, False, False, False, False, False]
    assert meta["long_signals"] == 2
    assert meta["short_signals"] == 1

    long_only, short_only, _ = module.policy_masks(
        8, fitted, quantile=0.8, allowed_side="long"
    )
    assert long_only.sum() == 2
    assert not short_only.any()


def test_fold_fit_mask_purges_targets_reaching_cutoff() -> None:
    dates = pd.Series(pd.date_range("2022-12-31 23:00", periods=30, freq="5min"))
    positions = np.asarray([0, 6, 12, 18], dtype=np.int64)
    exits = np.asarray([5, 11, 20, 29], dtype=np.int64)
    utilities = np.ones((4, 2), dtype=float)
    mask = module.fold_fit_mask(
        dates,
        positions,
        exits,
        utilities,
        fit_end="2023-01-01",
    )
    assert mask.tolist() == [True, True, False, False]


def test_source_does_not_reference_future_labelled_rex_jsonl() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "rex_event_reasoning_policy_sft_20260712.jsonl" not in source
