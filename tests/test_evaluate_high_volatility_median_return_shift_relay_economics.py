from pathlib import Path

import pandas as pd

from training import evaluate_high_volatility_median_return_shift_relay_economics as e


def test_evaluator_is_bound_and_empty_safe():
    assert e.POLICY_ID == "HVMRSR-8"
    assert e.sha256(e.PREREG) == e.PREREG_SHA
    assert e.sha256(e.SUPPORT) == e.SUPPORT_SHA
    assert e.sha256(e.NOVELTY) == e.NOVELTY_SHA
    assert e.sha256(e.CLOCK) == e.CLOCK_SHA
    assert e.CONTROLS == (
        "no_shift_tail",
        "no_variation_gate",
        "mean_return_shift",
        "one_boundary_stale_geometry",
        "direction_flip",
    )
    source = Path(e.__file__).read_text()
    assert source.index("def load_clock_allow_empty") < source.index("def evaluate_primary")


def test_empty(tmp_path):
    path = tmp_path / "empty.csv.gz"
    pd.DataFrame(columns=["entry_time", "exit_time", "side"]).to_csv(
        path, index=False, compression="gzip"
    )
    clock = e.load_clock_allow_empty(
        path,
        "train",
        pd.Timestamp("2023-07-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
    )
    assert clock.empty
