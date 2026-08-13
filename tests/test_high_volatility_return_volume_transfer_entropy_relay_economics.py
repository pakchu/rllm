import gzip

import pandas as pd

from training import evaluate_high_volatility_return_volume_transfer_entropy_relay_economics as economics


def test_empty_clock_is_valid(tmp_path) -> None:
    path = tmp_path / "empty.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("entry_time,exit_time,side\n")
    loaded = economics.load_clock_allow_empty(
        path,
        "train",
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    )
    assert loaded.empty
    assert list(loaded.columns) == ["entry_time", "exit_time", "side"]


def test_contract() -> None:
    assert economics.CONTROLS == (
        "no_information_tail",
        "no_variation_gate",
        "unconditional_transition_lift",
        "contemporaneous_conditioned_information",
        "one_decision_stale_information",
        "direction_flip",
        "same_clock_forced_long",
    )
    assert economics.LEVERAGE == 0.5
    assert economics.BASE_COST == 0.0006
    assert economics.STRESS_COST == 0.001
    assert "load_clock_allow_empty" in open(economics.__file__).read()
