from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import evaluate_stablecoin_quote_flow_diffusion as evaluator


FREEZE_SHA256 = "84b318982d58b4805805e9907970c95b9cc2fb1b7c330f263b90ee6f3b4bc47c"
EVALUATOR_SHA256 = "0ea59a107f05777ba91ab1c8fc5900e724ba48ec6ce647a42c34c34422222e3b"
MANIFEST_HASH = "f7913f23ef090658e11641e50f83581e8d3066318f74ac12c21c9e0da7d36903"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_sqfd_evaluator_freeze_bytes_and_manifest_are_locked() -> None:
    assert _sha256(evaluator.EVALUATOR_SOURCE) == EVALUATOR_SHA256
    assert _sha256(evaluator.EVALUATOR_FREEZE) == FREEZE_SHA256
    payload = json.loads(evaluator.EVALUATOR_FREEZE.read_text(encoding="utf-8"))

    assert payload == evaluator.verify_evaluator_freeze()
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["evaluator_source_sha256"] == EVALUATOR_SHA256


def test_sqfd_evaluator_freeze_opened_no_outcomes_and_sealed_every_stage() -> None:
    payload = evaluator.verify_evaluator_freeze()

    assert payload["opened_windows"] == []
    assert payload["sealed_windows"] == ["train", "test", "eval", "final"]
    assert payload["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_parsed_during_freeze"] == 0
    assert payload["execution_data_bytes_hashed_during_freeze"] is False
    assert payload["simulation_run_during_freeze"] is False
    assert payload["mutable_parameters"] == []


def test_sqfd_evaluator_freezes_all_controls_and_primary_stage_counts() -> None:
    payload = evaluator.verify_evaluator_freeze()

    assert payload["mechanism_controls"] == [
        "no_alt_breadth",
        "no_usdt_lag",
        "no_participation",
        "usdt_only",
        "direction_flip",
        "extra_latency_1h",
        "deterministic_random_side",
    ]
    assert payload["schedule_records"]["primary"]["stage_counts"] == {
        "train": 55,
        "test": 185,
        "eval": 217,
        "final": 93,
    }
    assert payload["train_execution_source"]["exit_boundary_required"] is False
    assert all(
        spec["exit_boundary_required"] is False
        for spec in payload["sealed_future_source_specs"].values()
    )
