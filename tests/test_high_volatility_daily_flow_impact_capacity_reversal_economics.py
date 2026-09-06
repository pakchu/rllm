import gzip

import pandas as pd

from training import evaluate_high_volatility_daily_flow_impact_capacity_reversal_economics as e


def test_empty_diagnostic_clock_is_valid(tmp_path):
    path = tmp_path / "empty.csv.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("entry_time,exit_time,side\n")
    result = e.load_clock_allow_empty(
        path,
        "train",
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    )
    assert result.empty
    assert result.columns.tolist() == ["entry_time", "exit_time", "side"]
    assert str(result["entry_time"].dtype) == "datetime64[ns, UTC]"


def test_frozen_controls_and_strict_costs():
    assert e.CONTROLS == (
        "no_impact_tail", "no_variation_gate", "negative_beta_state",
        "one_day_stale_capacity", "direction_flip", "forced_long",
    )
    assert e.LEVERAGE == 0.5
    assert e.BASE_COST == 0.0006
    assert e.STRESS_COST == 0.0010
    assert "load_clock_allow_empty" in open(e.__file__).read()
