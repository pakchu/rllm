from pathlib import Path

import pandas as pd

from training import evaluate_high_volatility_ethereum_flaw_pressure_relay_economics as evaluator


def test_hvefpr_evaluator_is_bound_and_empty_safe():
    assert evaluator.POLICY_ID == "HVEFPR-24"
    assert evaluator.sha256(evaluator.PREREG) == evaluator.PREREG_SHA
    assert evaluator.sha256(evaluator.SUPPORT) == evaluator.SUPPORT_SHA
    assert evaluator.sha256(evaluator.NOVELTY) == evaluator.NOVELTY_SHA
    assert evaluator.sha256(evaluator.CLOCK) == evaluator.CLOCK_SHA
    assert evaluator.CONTROLS == (
        "no_btc_volatility_gate",
        "pressure_direction_flip",
        "one_day_stale_pressure",
        "raw_day_over_day_pressure",
        "same_clock_forced_long",
    )
    source = Path(evaluator.__file__).read_text()
    assert source.index("def load_clock_allow_empty") < source.index("def evaluate_primary")


def test_empty_clock_is_valid(tmp_path):
    path = tmp_path / "empty.csv.gz"
    pd.DataFrame(columns=["entry_time", "exit_time", "side"]).to_csv(path, index=False, compression="gzip")
    result = evaluator.load_clock_allow_empty(path, "train", pd.Timestamp("2023-07-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
    assert result.empty
    assert list(result.columns) == ["entry_time", "exit_time", "side"]


def test_registered_accounting_constants_are_frozen():
    assert evaluator.LEVERAGE == 0.5
    assert evaluator.BASE_COST == 0.0006
    assert evaluator.STRESS_COST == 0.0010
    assert list(evaluator.STAGES) == ["train", "test", "eval", "final"]
