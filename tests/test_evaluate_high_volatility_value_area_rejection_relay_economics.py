from pathlib import Path

import pandas as pd

from training import evaluate_high_volatility_value_area_rejection_relay_economics as economics


def test_bound_empty_and_strict() -> None:
    assert economics.POLICY_ID == "HVVAR-8"
    assert economics.sha256(economics.PREREG) == economics.PREREG_SHA
    assert economics.sha256(economics.SUPPORT) == economics.SUPPORT_SHA
    assert economics.sha256(economics.NOVELTY) == economics.NOVELTY_SHA
    assert economics.sha256(economics.CLOCK) == economics.CLOCK_SHA
    assert economics.CONTROLS == (
        "no_variation_gate", "no_onset", "direction_flip", "value_area_68pct", "forced_long"
    )
    assert economics.LEVERAGE == 0.5
    assert economics.BASE_COST == 0.0006
    assert economics.STRESS_COST == 0.001
    source = Path(economics.__file__).read_text()
    assert source.index("def load_clock_allow_empty") < source.index("def evaluate_primary")


def test_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv.gz"
    pd.DataFrame(columns=["entry_time", "exit_time", "side"]).to_csv(
        path, index=False, compression="gzip"
    )
    loaded = economics.load_clock_allow_empty(
        path,
        "train",
        pd.Timestamp("2023-07-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
    )
    assert loaded.empty
