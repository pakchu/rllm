from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import evaluate_coinm_next_maturity_shock_relay as evaluator


FREEZE = Path(
    "results/coinm_next_maturity_shock_relay_evaluator_freeze_2026-07-19.json"
)


def test_canonical_freeze_is_outcome_blind_and_reproducible() -> None:
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == (
        "8c8957f1604be8960ff2b3bf6df4fc32105138fc00422326ca5a931a989f0868"
    )
    stored = json.loads(FREEZE.read_text(encoding="utf-8"))
    verified = evaluator.verify_evaluator_freeze(FREEZE)
    assert verified == stored
    assert stored["manifest_hash"] == (
        "1004cc198dde1f8c20010f4b8dae0242c863d23afdc0528920557eb6324f67c5"
    )
    assert stored["evaluator_source_sha256"] == (
        "8638aad9eaa1e8a2d6a89fadf98d7a893e8e83907c58bc044ba0a7191d5ab95e"
    )
    assert stored["opened_windows"] == []
    assert stored["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert stored["funding_rows_parsed_during_freeze"] == 0
    assert stored["simulation_run_during_freeze"] == 0
    assert len(stored["physical_market_files"]["train"]) == 29
    assert len(stored["physical_market_files"]["test"]) == 12
    assert (
        stored["physical_funding_row_windows"]["train"]["end_row_exclusive"]
        == stored["physical_funding_row_windows"]["test"]["start_row_inclusive"]
        == 3288
    )
