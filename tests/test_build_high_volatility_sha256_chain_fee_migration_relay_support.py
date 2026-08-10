import numpy as np
import pandas as pd

from training import build_high_volatility_sha256_chain_fee_migration_relay_support as support


def test_parse_source_row_enforces_completion_lag() -> None:
    row = support.parse_source_row({
        "asset": "bch",
        "time": "2023-01-01T00:00:00.000000000Z",
        "FeeTotNtv": "1.5",
        "IssTotNtv": "900",
        "AssetEODCompletionTime": "1672617600",
    })
    assert row["asset"] == "bch"
    assert row["fee_total"] == 1.5
    assert row["issuance"] == 900.0
    assert row["available_at"] == pd.Timestamp("2023-01-02T00:00:00Z")


def test_strict_prior_midrank_excludes_current() -> None:
    ranked = support.strict_prior_midrank(
        pd.Series([1.0, 2.0, 3.0, 4.0]), lookback=3, minimum=2
    )
    assert np.isnan(ranked.iloc[0])
    assert np.isnan(ranked.iloc[1])
    assert ranked.iloc[2] == 1.0
    assert ranked.iloc[3] == 1.0


def test_reservation_keeps_equal_exit_entry_and_uses_migration_side() -> None:
    observations = pd.date_range("2024-01-01T00:00:00Z", periods=6, freq="1d")
    decisions = observations + pd.Timedelta(days=1, hours=3)
    panel = pd.DataFrame({
        "observation_time": observations,
        "btc_available_at": decisions,
        "bch_available_at": decisions,
        "feature_available_time": decisions,
        "decision_time": decisions,
        "btc_fee_total": 10.0,
        "btc_issuance": 900.0,
        "bch_fee_total": 1.0,
        "bch_issuance": 900.0,
        "btc_fee_pressure": 0.01,
        "bch_fee_pressure": 0.001,
        "relative_fee_pressure": 1.0,
        "fee_migration": [1.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        "btc_fee_pressure_change": [1.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        "source_valid": True,
        "realized_variation": 1.0,
        "absolute_migration_rank": [0.9, 0.0, 0.9, 0.0, 0.9, 0.0],
        "realized_variation_rank": 0.9,
    })
    clock = support.candidate_clock(panel)
    assert list(clock.observation_time) == [observations[0], observations[2], observations[4]]
    assert list(clock.side) == [1, -1, 1]
