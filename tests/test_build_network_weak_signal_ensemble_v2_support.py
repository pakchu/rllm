from __future__ import annotations

import pandas as pd

from training import build_network_weak_signal_ensemble_v2_support as support
from training import preregister_network_weak_signal_ensemble_v2 as prereg


def test_support_summary_uses_frozen_v2_train_start() -> None:
    policy = prereg.Policy()
    history_dates = pd.date_range("2020-05-25 12:00", periods=54, freq="W-MON")
    prediction_dates = pd.date_range(policy.prediction_start + " 12:00", periods=82, freq="W-MON")
    history = pd.DataFrame(
        {
            "decision_date": history_dates,
            "entry_date": history_dates + pd.Timedelta(minutes=5),
            "exit_date": history_dates + pd.Timedelta(days=7, minutes=5),
            "feature_available_at": history_dates - pd.Timedelta(hours=2),
            "all_features_finite": True,
            "prediction_eligible": False,
        }
    )
    prediction = pd.DataFrame(
        {
            "decision_date": prediction_dates,
            "entry_date": prediction_dates + pd.Timedelta(minutes=5),
            "exit_date": prediction_dates + pd.Timedelta(days=7, minutes=5),
            "feature_available_at": prediction_dates - pd.Timedelta(hours=2),
            "all_features_finite": True,
            "prediction_eligible": True,
        }
    )
    summary = support.support_summary(
        pd.concat([history, prediction], ignore_index=True), prereg.build_manifest()
    )
    assert summary["candidate_counts"]["train_2021"] == 30
    assert summary["candidate_counts"]["train_2021_2022"] == 82
    assert summary["candidate_counts"]["selection_2023"] == 0
    assert summary["initial_fully_available_training_samples"] == 53
    assert summary["passed"] is False


def test_support_summary_passes_complete_synthetic_clock() -> None:
    policy = prereg.Policy()
    history_dates = pd.date_range("2020-05-25 12:00", periods=54, freq="W-MON")
    prediction_dates = pd.date_range(policy.prediction_start + " 12:00", "2023-12-18 12:00", freq="W-MON")
    all_dates = history_dates.append(prediction_dates)
    clock = pd.DataFrame(
        {
            "decision_date": all_dates,
            "entry_date": all_dates + pd.Timedelta(minutes=5),
            "exit_date": all_dates + pd.Timedelta(days=7, minutes=5),
            "feature_available_at": all_dates - pd.Timedelta(hours=2),
            "all_features_finite": True,
            "prediction_eligible": [False] * len(history_dates) + [True] * len(prediction_dates),
        }
    )
    summary = support.support_summary(clock, prereg.build_manifest())
    assert summary["initial_fully_available_training_samples"] == 53
    assert summary["candidate_counts"] == {
        "train_2021_2022": 82,
        "train_2021": 30,
        "train_2022": 52,
        "selection_2023": 51,
        "selection_2023_h1": 26,
        "selection_2023_h2": 25,
    }
    assert summary["passed"] is True
