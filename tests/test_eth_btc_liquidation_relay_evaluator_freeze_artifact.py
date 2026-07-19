from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import evaluate_eth_btc_liquidation_relay as evaluator


FREEZE = Path("results/eth_btc_liquidation_relay_evaluator_freeze_2026-07-19.json")
FREEZE_SHA256 = "caeefdbf2bed2ab403d185863971ae978ba95497b55864d9b1b44bf1949deb59"
FREEZE_HASH = "531baed0b17b067fd402c1293b06dc67d2bf04a9d3b4b1f5b712a5e312de9707"
EVALUATOR_SHA256 = "514b7560b6290684fb6f79e1fa8b6f21dcf3ab5b19b8e01ee024f67db71831d2"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_frozen_eblr_evaluator_is_source_only_and_reproducible() -> None:
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
    assert preconditions["clbr_overlap_rejected"] is True
    assert preconditions["icla_overlap_rejected"] is True
    assert preconditions["clbr_entry_jaccard"] == 0.0
    assert preconditions["icla_entry_jaccard"] == 0.0
    assert preconditions["maximum_entry_jaccard"] == 0.10

    assert freeze["config"]["leverage"] == 1.0
    assert freeze["config"]["base_cost_rate_per_side"] == 0.0006
    assert freeze["config"]["stress_cost_rate_per_side"] == 0.0010
    assert freeze["config"]["bootstrap_mean_block_trades"] == 4
    assert freeze["config"]["bootstrap_resamples"] == 50_000
    assert freeze["promotion_gates"] == evaluator.promotion_gate_contract()
    assert {
        stage: metadata["rows"]
        for stage, metadata in freeze["split_clocks"].items()
    } == {"train": 21, "test": 132, "eval": 113}

    expected_control_rows = {
        "train": {
            "direction_flip": 21,
            "btc_only_direct_shock": 77,
            "no_btc_quiet_gate": 51,
            "delayed_5m": 21,
            "future_eth_placebo": 21,
            "random_clocks": 21,
        },
        "test": {
            "direction_flip": 132,
            "btc_only_direct_shock": 344,
            "no_btc_quiet_gate": 270,
            "delayed_5m": 132,
            "future_eth_placebo": 132,
            "random_clocks": 132,
        },
        "eval": {
            "direction_flip": 113,
            "btc_only_direct_shock": 274,
            "no_btc_quiet_gate": 222,
            "delayed_5m": 113,
            "future_eth_placebo": 113,
            "random_clocks": 113,
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
