from __future__ import annotations

import numpy as np
import pandas as pd

from training import build_network_weak_signal_ensemble_support as support
from training import preregister_network_weak_signal_ensemble as prereg


def synthetic_sources(rows: int = 500) -> pd.DataFrame:
    observation = pd.date_range("2019-01-01", periods=rows, freq="D")
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "observation_date": observation,
            "available_at": observation + pd.Timedelta(days=1, hours=4),
            "FeeTotNtv": 20.0 + 2.0 * np.sin(phase / 11.0),
            "IssTotNtv": 900.0 + np.sin(phase / 30.0),
            "BlkCnt": 144.0 + 2.0 * np.sin(phase / 5.0),
            "TxCnt": 250_000.0 + phase * 20.0,
            "AdrActCnt": 500_000.0 + phase * 30.0 + 1_000.0 * np.sin(phase / 13.0),
            "TxTfrCnt": 600_000.0 + phase * 25.0 + 1_200.0 * np.cos(phase / 17.0),
        }
    )


def test_daily_features_are_finite_after_reference_and_price_free() -> None:
    features = support.build_daily_features(synthetic_sources(), prereg.Policy())
    assert not any("price" in column.lower() for column in features.columns)
    tail = features.iloc[-50:][list(prereg.FEATURE_COLUMNS)].to_numpy(float)
    assert np.isfinite(tail).all()
    assert np.max(np.abs(tail)) <= prereg.Policy().feature_clip


def test_feature_clock_uses_previous_completed_day() -> None:
    daily = support.build_daily_features(synthetic_sources(1000), prereg.Policy())
    clock = support.build_feature_clock(daily, prereg.Policy())
    row = clock.iloc[-1]
    assert pd.Timestamp(row["source_observation_date"]) < pd.Timestamp(row["decision_date"]).floor("D")
    assert pd.Timestamp(row["entry_date"]) == pd.Timestamp(row["decision_date"]) + pd.Timedelta(minutes=5)
    assert pd.Timestamp(row["exit_date"]) == pd.Timestamp(row["entry_date"]) + pd.Timedelta(days=7)


def test_prediction_requires_feature_availability() -> None:
    policy = prereg.Policy()
    daily = support.build_daily_features(synthetic_sources(1000), policy)
    clock = support.build_feature_clock(daily, policy)
    eligible = clock.loc[clock["prediction_eligible"]]
    assert (
        pd.to_datetime(eligible["feature_available_at"])
        <= pd.to_datetime(eligible["decision_date"])
    ).all()
    assert eligible["observation_age_days"].le(policy.maximum_observation_age_days).all()


def test_support_summary_counts_only_prediction_rows() -> None:
    policy = prereg.Policy()
    dates = pd.date_range("2021-03-01 12:00", periods=100, freq="W-MON")
    clock = pd.DataFrame(
        {
            "decision_date": dates,
            "entry_date": dates + pd.Timedelta(minutes=5),
            "exit_date": dates + pd.Timedelta(days=7, minutes=5),
            "feature_available_at": dates - pd.Timedelta(hours=2),
            "all_features_finite": True,
            "prediction_eligible": True,
        }
    )
    # Add enough pre-prediction training examples to satisfy the model history check.
    history_dates = pd.date_range("2020-01-06 12:00", periods=60, freq="W-MON")
    history = pd.DataFrame(
        {
            "decision_date": history_dates,
            "entry_date": history_dates + pd.Timedelta(minutes=5),
            "exit_date": history_dates + pd.Timedelta(days=7, minutes=5),
            "feature_available_at": pd.Timestamp("2021-02-26"),
            "all_features_finite": True,
            "prediction_eligible": False,
        }
    )
    summary = support.support_summary(pd.concat([history, clock], ignore_index=True), prereg.build_manifest())
    assert summary["candidate_counts"]["train_2021_2022"] == 96
    assert summary["candidate_counts"]["selection_2023"] == 4
    assert summary["initial_fully_available_training_samples"] >= policy.minimum_train_samples
    assert summary["passed"] is False
