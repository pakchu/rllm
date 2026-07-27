from __future__ import annotations

import numpy as np
import pandas as pd

from training.backtest_all_alpha_month import (
    FAMILIES,
    SOURCE_PATHS,
    STATUSES,
    _cap_frames_asof,
    _duplicate_groups,
    _dynamic_exit_arrays,
    _inventory,
    _research_offset,
    _signal_digest,
)

EXPECTED_ALPHAS = {
    "pb30_base",
    "pb30_addon",
    "nonpb30_taker",
    "oi_divergence_pullback",
    "oi_divergence_highfreq",
    "oi_divergence_highfreq_selector",
    "oi_upbit_ratio288_low",
    "oi_alt_ratio72_dynamic_exit",
    "short_kimchi3d",
    "short_premium_panic",
    "new_long_minimal_funding_premium",
    "funding_premium_lr_impact_central",
    "calendar_oi_funding_friday_asia_long",
    "kalman_funding_premium_long",
    "bocpd_funding_premium_long",
    "semimarkov_funding_premium_long",
    "rex_htf_range_veto",
    "legacy_rex_dual_regime_auto",
    "legacy_rex_dual_regime_short",
    "fresh_kimchi_fx",
    "frozen_annual_rank7",
    "rex_taker_low_range_position",
    "cand_rex_veto_7",
    "markov_transition_long",
}


def _market(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2026-07-01", periods=rows, freq="5min")
    open_price = np.linspace(100.0, 104.0, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": open_price * 1.002,
            "low": open_price * 0.998,
            "close": open_price,
        }
    )


def test_atomic_registry_is_complete_and_accounted_for() -> None:
    assert set(SOURCE_PATHS) == EXPECTED_ALPHAS
    assert set(FAMILIES) == EXPECTED_ALPHAS
    assert set(STATUSES) == EXPECTED_ALPHAS

    inventory = _inventory()
    assert inventory["scored_atomic_alphas"] == 24
    assert inventory["unaccounted_atomic_files"] == []
    assert len(inventory["research_pool_files"]) == 6
    assert (
        "configs/live/rex_llm_binance_testnet_bear_pilot.json"
        in inventory["runtime_configs_excluded"]
    )


def test_missing_legacy_offset_uses_frozen_research_clock() -> None:
    assert _research_offset(12) == 11
    assert _research_offset(6) == 5
    assert _research_offset(24, 35) == 11


def test_cached_frames_are_hard_capped_at_asof() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-07-27 14:50:00Z",
                    "2026-07-27 14:55:00Z",
                    "2026-07-27 15:00:00Z",
                ]
            )
        }
    )
    features = pd.DataFrame({"value": [1, 2, 3]})
    funding = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-07-27 08:00:00Z",
                    "2026-07-27 16:00:00Z",
                ]
            ),
            "funding_rate": [0.1, 0.2],
        }
    )

    capped_market, capped_features, capped_funding, diagnostic = (
        _cap_frames_asof(
            market,
            features,
            funding,
            asof="2026-07-27T15:03:00Z",
        )
    )

    assert len(capped_market) == 2
    assert capped_features["value"].tolist() == [1, 2]
    assert len(capped_funding) == 1
    assert diagnostic["market_rows_discarded_after_asof"] == 1
    assert diagnostic["funding_rows_discarded_after_asof"] == 1


def test_signal_digest_preserves_direction() -> None:
    digests = {
        _signal_digest(np.asarray([value, 0], dtype=np.int8))
        for value in (-1, 0, 1)
    }
    assert len(digests) == 3


def test_duplicate_detection_does_not_merge_flat_long_and_short() -> None:
    signals = {
        "flat": np.asarray([0, 0], dtype=np.int8),
        "long": np.asarray([1, 0], dtype=np.int8),
        "long_alias": np.asarray([1, 0], dtype=np.int8),
        "short": np.asarray([-1, 0], dtype=np.int8),
    }
    arrays = {
        name: {
            "R": np.zeros(2),
            "L": np.zeros(2),
            "H": np.zeros(2),
        }
        for name in signals
    }
    groups = _duplicate_groups(
        signals,
        arrays,
        np.asarray([True, True]),
    )
    assert groups["exact_signal"] == [["long", "long_alias"]]


def test_dynamic_exit_is_observed_then_executed_at_next_open() -> None:
    market = _market()
    features = pd.DataFrame({"exit_feature": np.zeros(len(market))})
    signal = np.zeros(len(market), dtype=np.int8)
    signal[10] = 1
    features.loc[13, "exit_feature"] = 1.0

    result = _dynamic_exit_arrays(
        market,
        features,
        signal,
        name="dynamic",
        hold_bars=20,
        dynamic={
            "min_bars": 3,
            "gates": [
                {
                    "feature": "exit_feature",
                    "op": ">=",
                    "threshold": 1.0,
                }
            ],
        },
        start=pd.Timestamp("2026-07-01"),
        end=pd.Timestamp("2026-07-02"),
    )

    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["signal_date"] == "2026-07-01 00:50:00"
    assert trade["entry_date"] == "2026-07-01 00:55:00"
    assert trade["exit_date"] == "2026-07-01 01:10:00"
    assert trade["source"] == "dynamic_exit"
