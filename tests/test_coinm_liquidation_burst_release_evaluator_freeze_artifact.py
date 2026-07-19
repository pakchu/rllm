from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd

from training import evaluate_coinm_liquidation_burst_release as evaluator


FREEZE = Path(
    "results/coinm_liquidation_burst_release_evaluator_freeze_2026-07-19.json"
)
EXPECTED_FREEZE_SHA256 = (
    "44f5090cf097eb6d13064b93ad4af87a98dc23460f9624b73d06d0fcf9a9695e"
)
EXPECTED_SPLITS = {
    "train": (
        "3fee6f32fc3fe630a48d10b5b218303675495149f7367ee762c9806a2ac0b7b5",
        40,
    ),
    "test": (
        "54edbc396f6d37e5e6c07abf971abe977d9c93aac5f7ab6ec6f0248ddef0682e",
        128,
    ),
    "eval": (
        "cc72e2b5adaa25eb10e95e782a502a80d49bd7b6e934ab4fdfdfb237321f6530",
        109,
    ),
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_committed_clbr_evaluator_freeze_is_exact_and_outcome_sealed() -> None:
    assert _sha256(FREEZE) == EXPECTED_FREEZE_SHA256
    payload = json.loads(FREEZE.read_text())
    assert payload == evaluator.verify_evaluator_freeze()
    assert payload["freeze_hash"] == (
        "a9b6bbc58b00fcce6c1e6188e14d77c03ed36b85c932c595a28ca6be4ec089b7"
    )
    assert payload["evaluation_source_sha256"] == (
        "68c9598ab0cf7d4ee4036cf859e6ae446043cc8320feb7d21424edda3780b08b"
    )
    assert payload["opened_windows"] == []
    assert payload["sealed_windows"] == ["train", "test", "eval"]
    assert payload["mutable_parameters"] == []
    assert payload["candidate_returns_computed_before_freeze"] is False
    assert payload["simulation_run"] is False
    assert payload["config"]["base_cost_rate_per_side"] == 0.0006
    assert payload["config"]["stress_cost_rate_per_side"] == 0.0012
    assert payload["bootstrap_contract"] == {
        "method": "circular stationary block bootstrap of net trade returns under centered null",
        "mean_block_trades": 8,
        "resamples": 10_000,
        "seed": 20_260_719,
        "p_value": "(1 + count(centered_bootstrap_mean >= observed_mean)) / (B + 1)",
    }

    for stage, (expected_hash, expected_rows) in EXPECTED_SPLITS.items():
        metadata = payload["split_clocks"][stage]
        path = Path(metadata["path"])
        assert path == evaluator._split_clock_path(evaluator.EvaluationConfig(), stage)
        assert _sha256(path) == expected_hash == metadata["sha256"]
        clocks = pd.read_csv(
            path,
            compression="gzip",
            parse_dates=["entry_time", "planned_exit_time"],
        )
        assert len(clocks) == expected_rows == metadata["rows"]
        assert cast(pd.Series, clocks["split"]).eq(stage).all()
        assert cast(pd.Series, clocks["candidate"]).eq("CLBR-24").all()
        assert cast(pd.Series, clocks["entry_time"]).is_monotonic_increasing
        assert cast(pd.Series, clocks["planned_exit_time"]).sub(
            clocks["entry_time"]
        ).eq(pd.Timedelta(hours=2)).all()

    for stage in evaluator.STAGES:
        assert not evaluator.RESULT_PATHS[stage].exists()
