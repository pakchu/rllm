import gzip

import pandas as pd

from training import evaluate_high_volatility_sample_entropy_collapse_continuation_economics as economics


def test_empty_diagnostic_clock_is_valid(tmp_path) -> None:
    path = tmp_path / "empty.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("entry_time,exit_time,side\n")
    clock = economics.load_clock_allow_empty(
        path, "train", pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")
    )
    assert clock.empty


def test_frozen_accounting_and_controls() -> None:
    assert economics.CONTROLS == (
        "no_entropy_gate", "no_variation_gate", "sign_entropy_low_tail",
        "one_decision_stale_entropy", "direction_flip", "forced_long",
    )
    assert economics.LEVERAGE == 0.5
    assert economics.BASE_COST == 0.0006
    assert economics.STRESS_COST == 0.001
    assert "load_clock_allow_empty" in open(economics.__file__).read()
