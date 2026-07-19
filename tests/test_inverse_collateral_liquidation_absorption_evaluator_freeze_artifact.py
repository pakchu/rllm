from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import evaluate_inverse_collateral_liquidation_absorption as evaluator


FREEZE = Path(
    "results/inverse_collateral_liquidation_absorption_evaluator_freeze_2026-07-19.json"
)
FREEZE_SHA256 = "f321370b85dc85720eb32b4a0dd88431c212486e9710df120ce3d79b6d62d819"
FREEZE_HASH = "980972548b9c9236c037984376930ae9e9e699c186555236d5e7da5f18d27413"
EVALUATOR_SHA256 = "839e14206c0c5d64a9a8372427da09e403ff799aa1bda40469254353ab3d506f"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_frozen_icla_evaluator_artifact_is_source_only_and_reproducible() -> None:
    assert _sha256(FREEZE) == FREEZE_SHA256
    freeze = json.loads(FREEZE.read_text())

    assert freeze["freeze_hash"] == FREEZE_HASH
    assert freeze["evaluation_source_sha256"] == EVALUATOR_SHA256
    assert _sha256(freeze["evaluation_source"]) == EVALUATOR_SHA256
    assert freeze["opened_windows"] == []
    assert freeze["sealed_windows"] == ["train", "test", "eval"]
    assert freeze["candidate_returns_computed_before_freeze"] is False
    assert freeze["simulation_run"] is False
    assert freeze["mutable_parameters"] == []

    preconditions = freeze["source_only_preconditions"]
    assert preconditions["support_passes"] is True
    assert preconditions["clbr_alias_rejected"] is True
    assert preconditions["clbr_entry_jaccard"] == 0.007662835249042145
    assert preconditions["maximum_clbr_entry_jaccard"] == 0.10

    assert freeze["config"]["base_cost_rate_per_side"] == 0.0006
    assert freeze["config"]["stress_cost_rate_per_side"] == 0.0010
    assert freeze["config"]["bootstrap_mean_block_trades"] == 4
    assert freeze["config"]["bootstrap_resamples"] == 50_000
    assert freeze["promotion_gates"] == evaluator.promotion_gate_contract()

    assert {
        stage: metadata["rows"] for stage, metadata in freeze["split_clocks"].items()
    } == {"train": 30, "test": 111, "eval": 108}
    expected_control_rows = {
        "train": {
            "direction_flip": 30,
            "liquidation_only_fade": 134,
            "delayed_5m": 30,
            "random_clocks": 30,
        },
        "test": {
            "direction_flip": 111,
            "liquidation_only_fade": 394,
            "delayed_5m": 111,
            "random_clocks": 111,
        },
        "eval": {
            "direction_flip": 108,
            "liquidation_only_fade": 337,
            "delayed_5m": 108,
            "random_clocks": 108,
        },
    }
    for stage, controls in freeze["control_clocks"].items():
        assert {name: metadata["rows"] for name, metadata in controls.items()} == (
            expected_control_rows[stage]
        )

    for metadata in freeze["split_clocks"].values():
        assert _sha256(metadata["path"]) == metadata["sha256"]
    for controls in freeze["control_clocks"].values():
        for metadata in controls.values():
            assert _sha256(metadata["path"]) == metadata["sha256"]

    reproduced = evaluator.verify_evaluator_freeze()
    assert reproduced["freeze_hash"] == FREEZE_HASH
