from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training.export_causal_residual_expert_switcher_2026_sources import (
    CONFIRMATION_START,
    DEFAULT_MANIFEST,
    END,
    EXPECTED_PROTOCOL_HASH,
    SEED_COLUMNS,
    START,
    SYMBOLS,
    sha256_file,
)


def test_cres_2026_source_artifact_is_exact_and_outcome_blind() -> None:
    manifest = json.loads(Path(DEFAULT_MANIFEST).read_text())
    assert manifest["preregistration_protocol_hash"] == EXPECTED_PROTOCOL_HASH
    assert manifest["post_entry_2026_strategy_returns_calculated"] is False
    assert manifest["physical_prefix"] == {
        "start": str(START),
        "end_exclusive": str(END),
    }
    assert len(manifest["records"]) == len(SYMBOLS)
    for record in manifest["records"]:
        assert record["market_rows"] == 157_248
        assert record["confirmation_market_rows"] == 52_128
        assert sha256_file(Path(record["output_market"])) == record["output_market_sha256"]
        assert sha256_file(Path(record["output_funding"])) == record["output_funding_sha256"]

    seed_record = manifest["historical_training_seed"]
    assert seed_record["contains_2026_rows"] is False
    assert seed_record["columns"] == list(SEED_COLUMNS)
    assert sha256_file(Path(seed_record["path"])) == seed_record["sha256"]
    seed = pd.read_csv(seed_record["path"], parse_dates=["signal_time", "entry_time", "exit_time"])
    assert len(seed) == seed_record["rows"] == 323
    assert (seed["exit_time"] < CONFIRMATION_START).all()
