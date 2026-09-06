from pathlib import Path

import pandas as pd

from training import evaluate_high_volatility_wilder_adx_trend_relay_economics as economics


def test_evaluator_is_bound_and_empty_safe() -> None:
    assert economics.POLICY_ID == "HVWADX-24"
    assert economics.sha256(economics.PREREG) == economics.PREREG_SHA
    assert economics.sha256(economics.SUPPORT) == economics.SUPPORT_SHA
    assert economics.sha256(economics.NOVELTY) == economics.NOVELTY_SHA
    assert economics.sha256(economics.CLOCK) == economics.CLOCK_SHA
    assert economics.CONTROLS == (
        "no_adx_gate", "no_variation_gate", "one_bar_stale_state", "direction_flip"
    )
    source = Path(economics.__file__).read_text()
    assert source.index("def load_clock_allow_empty") < source.index("def evaluate_primary")


def test_empty_clock_is_valid(tmp_path) -> None:
    path = tmp_path / "empty.csv.gz"
    pd.DataFrame(columns=["entry_time", "exit_time", "side"]).to_csv(
        path, index=False, compression="gzip"
    )
    clock = economics.load_clock_allow_empty(
        path, "train", pd.Timestamp("2023-07-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC")
    )
    assert clock.empty
