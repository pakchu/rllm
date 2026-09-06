import gzip

import pandas as pd

from training import evaluate_high_volatility_oi_price_coactivity_sponsorship_relay_economics as economics


def test_frozen_accounting_stage_order_and_controls():
    assert economics.LEVERAGE == 0.5
    assert economics.BASE_COST == 0.0006
    assert economics.STRESS_COST == 0.001
    assert tuple(economics.STAGES) == ("train", "test", "eval", "final")
    assert economics.PREDECESSOR == {"test": "train", "eval": "test", "final": "eval"}
    assert economics.CONTROLS == (
        "no_coactivity_gate",
        "no_gross_oi_activity_gate",
        "no_variation_gate",
        "one_block_stale_features",
        "direction_flip",
        "forced_long",
    )


def test_public_metrics_removes_trade_rows_only():
    assert economics.public_metrics({"x": 1, "trade_rows": [1]}) == {"x": 1}


def test_empty_diagnostic_clock_is_valid_before_outcomes(tmp_path):
    path = tmp_path / "empty.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("candidate,control,split,entry_time,exit_time,side\n")
    clock = economics.load_clock_allow_empty(
        path,
        "train",
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    )
    assert clock.empty
    assert list(clock) == ["entry_time", "exit_time", "side"]
