from __future__ import annotations

import pandas as pd

from training import build_coinm_next_maturity_shock_relay_support as s


def test_causal_quantile_excludes_current_and_overlapping_paths() -> None:
    values = pd.Series(range(12), dtype=float)
    pair = pd.Series(["a"] * 12)
    threshold = s.causal_quantile(values, pair, 0.5, shift=3, window=4, min_periods=4)
    assert threshold.iloc[:6].isna().all()
    assert threshold.iloc[6] == 1.5
    values.iloc[6] = 10_000.0
    replay = s.causal_quantile(values, pair, 0.5, shift=3, window=4, min_periods=4)
    assert replay.iloc[6] == threshold.iloc[6]


def test_support_build_never_loads_execution_outcomes(tmp_path) -> None:
    report = s.build(tmp_path / "support.json", tmp_path / "clock.csv.gz")
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["source"]["execution_btcusdt_rows_loaded"] == 0
    assert report["source"]["funding_rows_loaded"] == 0
    assert report["support"]["fit"]["events"] == 93
    assert report["support"]["test_support"]["events"] == 65


def test_primary_clock_is_causal_and_nonoverlapping(tmp_path) -> None:
    report = s.build(tmp_path / "support.json", tmp_path / "clock.csv.gz")
    clock = pd.read_csv(report["clock"]["path"])
    primary = clock.loc[clock["control"].eq("primary")].copy()
    for column in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
        primary[column] = pd.to_datetime(primary[column], utc=True)
    assert (
        primary["feature_available_time"]
        .eq(primary["signal_time"] + pd.Timedelta(minutes=5))
        .all()
    )
    assert (
        primary["entry_time"]
        .eq(primary["signal_time"] + pd.Timedelta(minutes=10))
        .all()
    )
    assert primary["exit_time"].eq(primary["entry_time"] + pd.Timedelta(hours=3)).all()
    assert (
        primary["entry_time"].iloc[1:].reset_index(drop=True)
        >= primary["exit_time"].iloc[:-1].reset_index(drop=True)
    ).all()
    assert (
        primary["side"]
        .eq(primary["next_flow"].apply(lambda x: 1 if x > 0 else -1))
        .all()
    )
