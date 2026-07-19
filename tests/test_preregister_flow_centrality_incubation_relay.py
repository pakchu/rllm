from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from training import preregister_flow_centrality_incubation_relay as fcir


def _flow(rows: int = 40) -> pd.DataFrame:
    index = pd.date_range("2023-01-01 01:00", periods=rows, freq="1h")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.normal(0.0, 0.1, size=(rows, len(fcir.SYMBOLS))),
        index=index,
        columns=cast(Any, list(fcir.SYMBOLS)),
    )


def test_prior_directed_weights_exclude_current_hour() -> None:
    original = _flow()
    changed = original.copy()
    changed.iloc[-1] = np.arange(len(fcir.SYMBOLS)) * 1_000.0
    first, first_effective = fcir.prior_directed_weights(
        original, window=20, minimum=15
    )
    second, second_effective = fcir.prior_directed_weights(
        changed, window=20, minimum=15
    )
    pd.testing.assert_series_equal(first.iloc[-1], second.iloc[-1])
    assert first_effective.iloc[-1] == pytest.approx(second_effective.iloc[-1])


def test_prior_quantile_excludes_current_value() -> None:
    values = pd.Series([1.0, 2.0, 100.0])
    threshold = fcir.prior_quantile(values, quantile=0.5, window=2, minimum=2)
    assert pd.isna(threshold.iloc[1])
    assert threshold.iloc[2] == pytest.approx(1.5)


def test_source_prefix_reader_stops_before_sealed_boundary(tmp_path: Path) -> None:
    path = tmp_path / "source.csv.gz"
    rows = []
    for timestamp in ("2023-12-31 23:00:00", "2024-01-01 00:00:00"):
        for symbol in fcir.SYMBOLS:
            row = {column: "0" for column in fcir.SOURCE_COLUMNS}
            row.update(
                {
                    "source_hour_open_utc": str(pd.Timestamp(timestamp) - pd.Timedelta(hours=1)),
                    "feature_available_time_utc": timestamp,
                    "symbol": symbol,
                    "taker_flow_fraction": "0.1",
                    "source_complete": "True",
                    "feature_valid": "True",
                    "feature_invalid_reason": "ok",
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows, columns=cast(Any, list(fcir.SOURCE_COLUMNS)))
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    prefix = fcir.load_source_prefix(
        path,
        end_exclusive=cast(pd.Timestamp, pd.Timestamp("2024-01-01")),
    )
    assert len(prefix) == len(fcir.SYMBOLS)
    assert prefix["feature_available_time_utc"].max() == pd.Timestamp(
        "2023-12-31 23:00:00"
    )


def test_support_selector_uses_strength_not_count_and_rejects_outcomes() -> None:
    weak = {
        "central_flow_quantile": 0.70,
        "minimum_effective_names": 2.2,
        "train_support": {"events": 500},
        "checks": {"support": True},
        "passes": True,
    }
    strong = {
        "central_flow_quantile": 0.75,
        "minimum_effective_names": 3.0,
        "train_support": {"events": 60},
        "checks": {"support": True},
        "passes": True,
    }
    assert fcir.select_support_cell([weak, strong]) is strong
    contaminated = {**strong, "train_support": {"events": 60, "cagr": 1.0}}
    with pytest.raises(ValueError, match="forbidden outcome"):
        fcir.select_support_cell([contaminated])


def test_train_support_gate_rejects_a_missing_quarter() -> None:
    summary = {
        "events": 80,
        "side_share_min": 0.5,
        "subwindows": {"train_h1": 40, "train_h2": 40},
        "quarter_counts": {"2023Q1": 20, "2023Q2": 20, "2023Q3": 40},
        "maximum_month_share": 0.2,
    }
    checks = fcir.train_support_checks(summary, fcir.Config())
    assert checks["train_quarter_counts"] is False


def test_schedule_delays_entry_holds_twelve_hours_and_skips_overlap() -> None:
    index = pd.to_datetime(
        ["2023-03-01 01:00", "2023-03-01 02:00", "2023-03-01 14:00"]
    )
    features = pd.DataFrame(
        {
            "central_flow": [0.2, -0.3, -0.4],
            "equal_weight_flow": [0.01, 0.01, -0.01],
            "central_abs_q75": [0.1, 0.1, 0.1],
            "crowd_quiet_threshold": [0.05, 0.05, 0.05],
            "effective_names": [3.1, 3.2, 3.3],
            **{
                f"weight_{symbol.lower()}": [1 / 6] * 3
                for symbol in fcir.SYMBOLS
            },
        },
        index=index,
    )
    state = pd.Series([True, False, True], index=index)
    events = fcir.schedule_events(
        features,
        state,
        central_quantile=0.75,
        cfg=fcir.Config(),
    )
    assert len(events) == 2
    assert events["entry_time"].tolist() == [
        pd.Timestamp("2023-03-01 01:05"),
        pd.Timestamp("2023-03-01 14:05"),
    ]
    assert bool(
        cast(pd.Series, events["exit_time"])
        .eq(events["entry_time"] + pd.Timedelta(hours=12))
        .all()
    )


def test_novelty_metric_is_symmetric_and_exact() -> None:
    new = pd.DatetimeIndex(
        pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-05"])
    )
    prior = pd.DatetimeIndex(
        pd.to_datetime(
            ["2023-01-01", "2023-01-03 05:00", "2023-01-05"], format="mixed"
        )
    )
    result = fcir.novelty_metrics(
        new, prior, tolerance=cast(pd.Timedelta, pd.Timedelta(hours=6))
    )
    assert result["exact_jaccard"] == pytest.approx(1 / 2)
    assert result["new_near_prior_share"] == pytest.approx(1.0)
    assert result["prior_near_new_share"] == pytest.approx(1.0)
    assert result["max_bidirectional_near_share"] == pytest.approx(1.0)


def test_protocol_keeps_btc_outcomes_sealed() -> None:
    payload = fcir.protocol()
    boundary = payload["evidence_boundary"]
    assert boundary["post_entry_outcomes_opened"] is False
    assert "BTC OHLC or funding" in boundary["forbidden"]
    assert payload["signal"]["side"].startswith("sign(central_flow)")
    assert payload["selection"]["outcomes_used"] is False
    assert payload["signal"]["threshold_history"] == (
        "rolling 2160-hour window, activates after 720 prior valid observations, "
        "current t excluded"
    )
    assert payload["outcome_gate"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["outcome_gate"]["strict_mdd_max_pct"] == 15.0
    assert payload["outcome_gate"]["minimum_trades"] == {
        "train": 60,
        "test": 80,
        "eval": 55,
        "final": 30,
    }
