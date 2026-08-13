import json
from pathlib import Path

import pandas as pd

from training import evaluate_high_volatility_exchange_reserve_pressure_relay_economics as evaluator


def test_frozen_bindings_controls_and_loader_order() -> None:
    assert evaluator.POLICY_ID == "HVEXRP-24"
    assert evaluator.sha256(evaluator.PREREG) == evaluator.PREREG_SHA
    assert evaluator.sha256(evaluator.SUPPORT) == evaluator.SUPPORT_SHA
    assert evaluator.sha256(evaluator.NOVELTY) == evaluator.NOVELTY_SHA
    assert evaluator.sha256(evaluator.CLOCK) == evaluator.CLOCK_SHA
    assert evaluator.FREEZE == Path(
        "results/high_volatility_exchange_reserve_pressure_relay_economic_evaluator_freeze_2026-08-13.json"
    )
    assert evaluator.CONTROLS == (
        "no_btc_variation_gate",
        "exchange_reserve_direction_flip",
        "one_day_stale_reserve_change",
        "exchange_reserve_level_rank",
        "same_clock_forced_long",
    )
    source = Path(evaluator.__file__).read_text()
    assert source.index("def load_clock_allow_empty") < source.index("def evaluate_primary")
    freeze = json.loads(evaluator.FREEZE.read_text(encoding="utf-8"))
    core = {key: value for key, value in freeze.items() if key != "manifest_hash"}
    assert freeze["manifest_hash"] == evaluator.canonical_hash(core)
    assert freeze["evaluator"]["sha256"] == evaluator.sha256(Path(evaluator.__file__))
    assert freeze["outcomes_opened"] is False
    assert freeze["empty_diagnostic_controls_handled_before_outcomes"] is True


def test_empty_clock_is_valid_without_opening_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv.gz"
    pd.DataFrame(columns=["entry_time", "exit_time", "side"]).to_csv(
        path, index=False, compression="gzip"
    )

    clock = evaluator.load_clock_allow_empty(
        path,
        "train",
        pd.Timestamp("2023-07-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
    )

    assert clock.empty
    assert list(clock.columns) == ["entry_time", "exit_time", "side"]
    assert str(clock["entry_time"].dtype) == "datetime64[ns, UTC]"
    assert str(clock["exit_time"].dtype) == "datetime64[ns, UTC]"
