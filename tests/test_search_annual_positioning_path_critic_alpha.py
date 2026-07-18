from __future__ import annotations

import numpy as np
import pandas as pd

from training import search_annual_positioning_path_critic_alpha as alpha


def test_eligible_training_mask_requires_exit_before_cutoff() -> None:
    dates = np.asarray(pd.date_range("2021-12-31 23:40", periods=4, freq="5min"), dtype="datetime64[ns]")
    exits = np.asarray([98, 99, 100, 101])
    mask = alpha.eligible_training_mask(
        dates,
        exits,
        cutoff_date="2022-01-01",
        cutoff_position=100,
    )
    assert mask.tolist() == [True, True, False, False]


def test_expected_policy_is_single_frozen_policy() -> None:
    assert alpha.EXPECTED_POLICY == {
        "model": "annual_h288_mean",
        "hold_bars": 288,
        "score_quantile": 0.80,
        "side": "both",
        "execution_stride_bars": 12,
    }


def test_schedule_enters_next_open_and_requires_split_contained_exit() -> None:
    dates = pd.date_range("2022-01-01", periods=12, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.arange(100.0, 112.0),
            "high": np.arange(101.0, 113.0),
            "low": np.arange(99.0, 111.0),
        }
    )
    funding = pd.DataFrame({"date": pd.to_datetime([]), "funding_rate": pd.Series(dtype=float)})
    cfg = alpha.Config("m", "x", "f", "o", "z", leverage=0.5)
    engine = alpha.ExecutionEngine(market, funding, alpha.execution_config(cfg))
    long_active = np.zeros(len(market), dtype=bool)
    short_active = np.zeros(len(market), dtype=bool)
    long_active[2] = True
    original = alpha.WINDOWS["development_2022"]
    alpha.WINDOWS["development_2022"] = (str(dates[0]), str(dates[8]))
    try:
        trades = alpha.schedule_window(
            engine,
            long_active,
            short_active,
            window="development_2022",
            hold_bars=3,
            stride_bars=1,
        )
    finally:
        alpha.WINDOWS["development_2022"] = original
    assert len(trades) == 1
    assert trades[0].signal_position == 2
    assert trades[0].entry_position == 3
    assert trades[0].exit_position == 6


def test_canonical_hash_ignores_mapping_order() -> None:
    assert alpha.canonical_hash({"a": 1, "b": 2}) == alpha.canonical_hash({"b": 2, "a": 1})


def test_build_causal_features_masks_unavailable_external_values(monkeypatch) -> None:
    market = pd.DataFrame(
        {
            "dxy_available": [1, 0],
            "kimchi_available": [0, 1],
            "usdkrw_available": [1, 0],
        }
    )
    raw = pd.DataFrame(
        {
            "dxy_zscore": [1.0, 2.0],
            "dxy_momentum": [1.0, 2.0],
            "kimchi_premium_zscore": [3.0, 4.0],
            "kimchi_premium_change": [3.0, 4.0],
            "usdkrw_zscore": [5.0, 6.0],
            "usdkrw_momentum": [5.0, 6.0],
        }
    )
    monkeypatch.setattr(alpha, "build_model_features", lambda _: raw.copy())
    features = alpha.build_causal_features(market)
    assert np.isnan(features.loc[1, "dxy_zscore"])
    assert np.isnan(features.loc[0, "kimchi_premium_change"])
    assert np.isnan(features.loc[1, "usdkrw_momentum"])
    assert features["dxy_available"].tolist() == [1.0, 0.0]
