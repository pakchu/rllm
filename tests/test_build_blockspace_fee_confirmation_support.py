from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from training import build_blockspace_fee_confirmation_support as support
from training import preregister_blockspace_fee_confirmation as prereg


def synthetic_blockspace(rows: int = 240) -> pd.DataFrame:
    observation = pd.date_range("2021-01-01", periods=rows, freq="D")
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "observation_date": observation,
            "available_at": observation + pd.Timedelta(days=1, hours=4),
            "FeeTotNtv": 20.0 + np.sin(phase / 7.0),
            "IssTotNtv": 900.0 + np.sin(phase / 30.0),
            "BlkCnt": 144.0 + np.sin(phase / 5.0),
            "TxCnt": 250_000.0 + phase * 10.0,
        }
    )


def test_build_features_is_price_independent_and_stale_rows_cannot_signal() -> None:
    frame = synthetic_blockspace()
    frame.loc[150, "FeeTotNtv"] *= 10.0
    frame.loc[150, "available_at"] = frame.loc[150, "observation_date"] + pd.Timedelta(days=4)
    features = support.build_features(frame, prereg.Policy())
    assert not any("price" in column.lower() for column in features.columns)
    assert features.loc[150, "source_lag_days"] == 4.0
    assert not bool(features.loc[150, "eligible"])


def test_zero_fee_is_ineligible_without_infinite_feature() -> None:
    frame = synthetic_blockspace()
    frame.loc[150, "FeeTotNtv"] = 0.0
    features = support.build_features(frame, prereg.Policy())
    assert np.isnan(features.loc[150, "fee_share"])
    assert not bool(features.loc[150, "eligible"])


def test_event_is_false_to_true_transition_only() -> None:
    features = support.build_features(synthetic_blockspace(), prereg.Policy())
    features["eligible"] = False
    features.loc[130:132, "eligible"] = True
    features["event"] = features["eligible"] & ~features["eligible"].shift(1, fill_value=False)
    assert features.index[features["event"]].tolist() == [130]


def test_schedule_is_delayed_and_nonoverlapping() -> None:
    policy = prereg.Policy()
    features = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["2021-03-01", "2021-03-02", "2021-03-06"]),
            "available_at": pd.to_datetime(
                ["2021-03-02 04:02", "2021-03-03 04:02", "2021-03-07 04:02"]
            ),
            "event": [True, True, True],
            "fee_share": [-3.0] * 3,
            "transaction_density": [7.0] * 3,
            "fee_share_z": [2.0] * 3,
            "transaction_density_z": [1.0] * 3,
            "composite": [2.5] * 3,
            "source_lag_days": [1.2] * 3,
            "fee_reference_count": [180] * 3,
            "density_reference_count": [180] * 3,
        }
    )
    clock = support.schedule_clock(features, policy)
    assert len(clock) == 2
    assert clock.iloc[0]["earliest_tradable_open"] == pd.Timestamp("2021-03-02 04:05")
    assert clock.iloc[0]["entry_date"] == pd.Timestamp("2021-03-02 04:10")
    assert clock.iloc[1]["entry_date"] >= clock.iloc[0]["exit_date"]


def test_load_preregistration_rejects_opened_outcomes(tmp_path) -> None:
    payload = prereg.build_manifest()
    payload["outcomes_opened"] = True
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    payload["manifest_hash"] = prereg.canonical_hash(core)
    path = tmp_path / "opened.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="cannot open outcomes"):
        support.load_preregistration(path)


def test_support_summary_fails_closed_on_small_clock() -> None:
    clock = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(
                ["2021-03-01", "2021-04-01", "2022-01-01", "2023-02-01"]
            )
        }
    )
    summary = support.support_summary(clock, prereg.build_manifest())
    assert summary["counts"]["train_2021"] == 2
    assert summary["counts"]["train_2022"] == 1
    assert summary["counts"]["selection_2023"] == 1
    assert summary["passed"] is False
