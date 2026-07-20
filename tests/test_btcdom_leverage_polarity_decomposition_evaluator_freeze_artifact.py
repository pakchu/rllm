from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import evaluate_btcdom_leverage_polarity_decomposition as evaluator


FREEZE = Path(
    "results/btcdom_leverage_polarity_decomposition_evaluator_freeze_2026-07-20.json"
)
CLOCKS = Path(
    "data/btcdom_leverage_polarity_decomposition_evaluation_clocks_2022_2023.csv.gz"
)
FREEZE_SHA256 = "e96d299daa27ca598ce7ebe14a3f42eacc5a677703356c8ab193d8da1b8a371d"
CLOCK_SHA256 = "38ccc18df700d24462d0cae91e34733856ed053dc400c584a3eedaf3f9ed60f1"
MANIFEST_HASH = "c55b8b23f1ec45821ca670fab6f5811826b37d839c01d1e44d2a0ca1b30a31fd"
EVALUATOR_SHA256 = "748fae0511cfe3f3eca48f43627bc1a6b728253c225ee1fc6e2aba14f390b17c"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dlpd_evaluator_freeze_is_hash_locked_and_outcome_unopened() -> None:
    assert _sha256(FREEZE) == FREEZE_SHA256
    assert _sha256(CLOCKS) == CLOCK_SHA256
    assert _sha256(evaluator.EVALUATOR_SOURCE) == EVALUATOR_SHA256

    report = json.loads(FREEZE.read_text())
    assert report["manifest_hash"] == MANIFEST_HASH
    assert report["evaluator_source_sha256"] == EVALUATOR_SHA256
    assert report["evaluation_clock_sha256"] == CLOCK_SHA256
    assert report["phase_scope"] == ["train", "test"]
    assert report["opened_windows"] == []
    assert report["sealed_windows"] == ["train", "test", "eval", "final"]
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["execution_outcome_data_bytes_hashed_during_freeze"] is False
    assert report["simulation_run_during_freeze"] is False
    assert report["mutable_parameters"] == []
    assert report["execution_source_specs"]["train"]["exit_boundary_required"] is False
    assert report["execution_source_specs"]["test"]["exit_boundary_required"] is False


def test_dlpd_evaluator_freeze_records_exact_control_counts() -> None:
    report = evaluator.verify_evaluator_freeze(FREEZE)
    records = report["schedule_records"]
    expected = {
        "primary": {"train": 237, "test": 184},
        "btc_only_tail": {"train": 566, "test": 499},
        "dom_only_mirror": {"train": 530, "test": 488},
        "same_sign": {"train": 240, "test": 222},
        "stale_btc_1h": {"train": 233, "test": 188},
        "stale_dom_1h": {"train": 242, "test": 203},
        "direction_flip": {"train": 237, "test": 184},
        "deterministic_random_side": {"train": 237, "test": 184},
        "extra_latency_1h": {"train": 237, "test": 184},
    }
    assert {name: item["stage_counts"] for name, item in records.items()} == expected
    assert sum(item["events"] for item in records.values()) == 5_095
