from pathlib import Path

import pandas as pd

from training import evaluate_high_volatility_crypto_market_mode_ignition_relay_economics as economics


def test_evaluator_is_frozen_to_policy_predecessors_and_controls():
    assert economics.POLICY_ID == "HVCMMI-8"
    assert economics.CONTROLS == (
        "no_btc_variation_gate", "no_mode_onset", "equal_weight_final_hour",
        "one_block_stale_geometry", "direction_flip", "forced_long",
    )
    assert economics.sha256(economics.PREREG) == economics.PREREG_SHA
    assert economics.sha256(economics.SUPPORT) == economics.SUPPORT_SHA
    assert economics.sha256(economics.NOVELTY) == economics.NOVELTY_SHA
    assert economics.sha256(economics.CLOCK) == economics.CLOCK_SHA


def test_load_clock_allow_empty_precedes_outcome_execution(tmp_path):
    path = tmp_path / "empty.csv.gz"
    pd.DataFrame(columns=["entry_time", "exit_time", "side"]).to_csv(path, index=False, compression="gzip")
    frame = economics.load_clock_allow_empty(
        path, "train", pd.Timestamp("2023-07-01T00:00Z"), pd.Timestamp("2024-01-01T00:00Z")
    )
    assert frame.empty
    assert str(frame.entry_time.dtype) == "datetime64[ns, UTC]"
    source = Path(economics.__file__).read_text()
    assert source.index("def load_clock_allow_empty") < source.index("def evaluate_primary")
