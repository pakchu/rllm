from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from training import evaluate_network_weak_signal_ensemble_v2 as evaluate
from training import preregister_network_weak_signal_ensemble_v2 as prereg


@pytest.fixture(autouse=True)
def bypass_preoutcome_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate, "verify_evaluation_freeze", lambda: {})


def synthetic_labelled(*, constant_target: float | None = None) -> pd.DataFrame:
    dates = pd.date_range("2020-04-06 12:00", periods=70, freq="W-MON")
    phase = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame(
        {
            "decision_date": dates,
            "entry_date": dates + pd.Timedelta(minutes=5),
            "exit_date": dates + pd.Timedelta(days=7, minutes=5),
            "feature_available_at": dates - pd.Timedelta(hours=4),
            "all_features_finite": True,
            "prediction_eligible": False,
            "decision_position": np.arange(len(dates)) * 2016,
            "entry_position": np.arange(len(dates)) * 2016 + 1,
            "exit_position": np.arange(len(dates)) * 2016 + 2017,
        }
    )
    for offset, column in enumerate(prereg.FEATURE_COLUMNS, start=1):
        frame[column] = np.sin(phase / (4.0 + offset)) + phase * 0.001 * offset
    frame["target_log_return"] = (
        constant_target
        if constant_target is not None
        else 0.002 * np.sin(phase / 3.0) - 0.001 * np.cos(phase / 7.0)
    )
    frame.loc[60:, "prediction_eligible"] = True
    return frame


def test_pre2024_parser_stops_before_sealed_row(tmp_path) -> None:
    source = tmp_path / "market.csv.gz"
    with gzip.open(source, "wt") as handle:
        handle.write("date,open,high,low,close,unused\n")
        handle.write("2023-12-31 23:55:00,100,102,99,101,7\n")
        handle.write("2024-01-01 00:00:00,999,999,999,999,9\n")
        handle.write("2024-01-01 00:05:00,888,888,888,888,9\n")
    market = evaluate._parse_pre2024_market(source)
    assert len(market) == 1
    assert market.iloc[0]["open"] == 100.0
    assert not market.eq(999).any().any()


def test_support_verification_drops_ineligible_clock_rows_crossing_2024() -> None:
    clock, _, support = evaluate.verify_support_and_clock()
    assert clock["exit_date"].max() < evaluate.SELECTION_END
    assert int(clock["prediction_eligible"].sum()) == support["feature_clock"][
        "prediction_eligible_rows"
    ]


def test_attach_labels_uses_scheduled_entry_and_exit_opens() -> None:
    dates = pd.date_range("2021-01-04 12:00", periods=5, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 110.0, 120.0, 132.0, 140.0],
            "high": [101.0, 111.0, 121.0, 133.0, 141.0],
            "low": [99.0, 109.0, 119.0, 131.0, 139.0],
            "close": [100.0, 110.0, 120.0, 132.0, 140.0],
        }
    )
    clock = pd.DataFrame(
        {
            "decision_date": [dates[0]],
            "entry_date": [dates[1]],
            "exit_date": [dates[3]],
        }
    )
    labelled = evaluate.attach_return_labels(clock, market)
    assert labelled.iloc[0]["target_log_return"] == pytest.approx(np.log(132.0 / 110.0))
    assert labelled.iloc[0]["decision_position"] == 0
    assert labelled.iloc[0]["entry_position"] == 1
    assert labelled.iloc[0]["exit_position"] == 3


def test_online_fit_uses_only_labels_exited_before_decision() -> None:
    frame = synthetic_labelled()
    predictions = evaluate.online_forecasts(
        frame,
        feature_columns=prereg.FEATURE_COLUMNS,
        cfg=evaluate.EvaluationConfig(cluster_permutations=10),
    )
    assert len(predictions) == 10
    assert predictions["train_count"].min() >= prereg.Policy().minimum_train_samples
    assert (predictions["latest_train_exit"] <= predictions["decision_date"]).all()
    # The immediately preceding seven-day label exits five minutes after the
    # new decision and therefore must not enter that refit.
    first_prediction_index = 60
    assert predictions.iloc[0]["latest_train_exit"] == frame.iloc[
        first_prediction_index - 2
    ]["exit_date"]


def test_target_mean_is_not_added_back_as_long_drift() -> None:
    frame = synthetic_labelled(constant_target=0.01)
    predictions = evaluate.online_forecasts(
        frame,
        feature_columns=prereg.FEATURE_COLUMNS,
        cfg=evaluate.EvaluationConfig(cluster_permutations=10),
        abstain=False,
    )
    assert np.allclose(predictions["forecast"].to_numpy(float), 0.0, atol=1e-14)


def test_permutation_never_moves_rows_across_years() -> None:
    values = np.arange(18, dtype=float).reshape(6, 3)
    years = np.array([2021, 2021, 2021, 2022, 2022, 2022])
    permuted = evaluate._permute_training_rows_within_year(values, years, seed=7)
    assert {tuple(row) for row in permuted[:3]} == {tuple(row) for row in values[:3]}
    assert {tuple(row) for row in permuted[3:]} == {tuple(row) for row in values[3:]}


def test_delayed_control_moves_entry_only_and_preserves_nonoverlap() -> None:
    frame = synthetic_labelled()
    schedules, _ = evaluate.build_schedules(
        frame, evaluate.EvaluationConfig(cluster_permutations=10)
    )
    primary = schedules["primary"]
    delayed = schedules["one_bar_delayed_entry"]
    assert delayed["entry_position"].tolist() == (primary["entry_position"] + 1).tolist()
    assert delayed["entry_date"].tolist() == (
        primary["entry_date"] + pd.Timedelta(minutes=5)
    ).tolist()
    assert delayed["exit_position"].tolist() == primary["exit_position"].tolist()
    assert delayed["exit_date"].tolist() == primary["exit_date"].tolist()


def test_strict_mdd_uses_favorable_then_adverse_held_ohlc() -> None:
    dates = pd.date_range("2022-01-03 12:00", periods=5, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 105.0, 110.0, 111.0],
            "high": [100.0, 120.0, 108.0, 111.0, 112.0],
            "low": [100.0, 90.0, 100.0, 109.0, 110.0],
            "close": [100.0, 105.0, 106.0, 110.0, 111.0],
        }
    )
    funding = pd.DataFrame(
        {
            "funding_time_ms": np.array([], dtype=np.int64),
            "funding_time": pd.to_datetime([]),
            "funding_rate": np.array([], dtype=float),
        }
    )
    schedule = pd.DataFrame(
        {
            "decision_position": [0],
            "entry_position": [1],
            "exit_position": [3],
            "decision_date": [dates[0]],
            "entry_date": [dates[1]],
            "exit_date": [dates[3]],
            "side": [1],
        }
    )
    result = evaluate.simulate_schedule(
        market,
        funding,
        schedule,
        start="2022-01-03 12:00",
        end="2022-01-04 12:00",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(cluster_permutations=10),
        compute_cluster=False,
    )
    assert result["absolute_return_pct"] == pytest.approx(5.0)
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.95 / 1.10) * 100.0)
    assert result["wall_clock_years"] == pytest.approx(1.0 / 365.25)


def test_strict_mdd_separates_funding_credit_and_debit_conservatively() -> None:
    dates = pd.date_range("2022-01-03 12:00", periods=5, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 105.0, 110.0, 111.0],
            "high": [100.0, 120.0, 108.0, 111.0, 112.0],
            "low": [100.0, 90.0, 100.0, 109.0, 110.0],
            "close": [100.0, 105.0, 106.0, 110.0, 111.0],
        }
    )
    funding_dates = pd.Series([dates[1], dates[2]])
    funding = pd.DataFrame(
        {
            "funding_time_ms": (
                funding_dates.astype("int64") // 1_000_000
            ).to_numpy(np.int64),
            "funding_time": funding_dates,
            # At 0.5x, the long receives +1% then pays -1%; net funding is zero.
            "funding_rate": [-0.02, 0.02],
        }
    )
    schedule = pd.DataFrame(
        {
            "decision_position": [0],
            "entry_position": [1],
            "exit_position": [3],
            "decision_date": [dates[0]],
            "entry_date": [dates[1]],
            "exit_date": [dates[3]],
            "side": [1],
        }
    )
    result = evaluate.simulate_schedule(
        market,
        funding,
        schedule,
        start="2022-01-03 12:00",
        end="2022-01-04 12:00",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(cluster_permutations=10),
        compute_cluster=False,
    )
    assert result["absolute_return_pct"] == pytest.approx(5.0)
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.94 / 1.11) * 100.0)
