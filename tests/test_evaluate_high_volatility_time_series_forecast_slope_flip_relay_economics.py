from pathlib import Path
import pandas as pd
from training import evaluate_high_volatility_time_series_forecast_slope_flip_relay_economics as e


def test_bound_and_empty_safe():
    assert e.POLICY_ID == "HVTSF3-24"
    assert e.sha256(e.PREREG) == e.PREREG_SHA
    assert e.sha256(e.SUPPORT) == e.SUPPORT_SHA
    assert e.sha256(e.NOVELTY) == e.NOVELTY_SHA
    assert e.sha256(e.CLOCK) == e.CLOCK_SHA
    assert e.CONTROLS == ("no_variation_gate", "forecast_level_side", "current_regression_slope_flip", "one_bar_stale_flip", "direction_flip", "forced_long")
    source = Path(e.__file__).read_text()
    assert source.index("def load_clock_allow_empty") < source.index("def evaluate_primary")


def test_empty_clock(tmp_path):
    path = tmp_path / "empty.csv.gz"
    pd.DataFrame(columns=["entry_time", "exit_time", "side"]).to_csv(path, index=False, compression="gzip")
    loaded = e.load_clock_allow_empty(path, "train", pd.Timestamp("2023-07-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
    assert loaded.empty
