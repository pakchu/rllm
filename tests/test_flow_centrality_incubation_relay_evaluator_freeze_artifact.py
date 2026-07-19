from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from training import evaluate_flow_centrality_incubation_relay as evaluator


FREEZE = Path(
    "results/flow_centrality_incubation_relay_evaluator_freeze_2026-07-19.json"
)
CONTROLS = Path("data/flow_centrality_incubation_relay_control_clocks_2023_2026.csv.gz")
EVALUATOR_SHA256 = "036b22442a2080e7ea5ffe914c605a9b1b1a55b128a315a2f2f05be7b37a736d"
FREEZE_SHA256 = "05a08704b99086148752fee19d291b36ff412e9c53af72b3b6f0cf4c178ae204"
CONTROL_SHA256 = "3ffd7cd3b55aa81d3dbe4f46ec611c0209c947fcfac1c02cc9afc3274cc04711"
FREEZE_MANIFEST_HASH = (
    "db449785021045afda156a1e1772c11b2b8bcdaf19db9714c3424f0e4e2e88d9"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fcir_evaluator_freeze_is_hash_locked_and_outcome_unopened() -> None:
    assert _sha256(evaluator.EVALUATOR_SOURCE) == EVALUATOR_SHA256
    assert _sha256(FREEZE) == FREEZE_SHA256
    assert _sha256(CONTROLS) == CONTROL_SHA256

    report = cast(dict[str, Any], json.loads(FREEZE.read_text()))
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == FREEZE_MANIFEST_HASH
    assert report["manifest_hash"] == evaluator._canonical_hash(core)
    assert report["support_commit"] == evaluator.SUPPORT_COMMIT
    assert report["evaluator_source_sha256"] == EVALUATOR_SHA256
    assert report["control_clock_sha256"] == CONTROL_SHA256
    assert report["opened_windows"] == []
    assert report["sealed_windows"] == list(evaluator.STAGE_ORDER)
    assert report["mutable_parameters"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["execution_outcome_data_bytes_hashed_during_freeze"] is False
    assert report["simulation_run_during_freeze"] is False
    assert report["evaluation_config"] == {
        "leverage": 0.5,
        "base_cost_notional_per_side": 0.0006,
        "stress_cost_notional_per_side": 0.001,
        "hold_bars": 144,
        "exact_cluster_max": 20,
        "cluster_draws": 20_000,
        "cluster_seed": 20_260_719,
    }


def test_fcir_frozen_control_family_and_stage_counts_are_exact() -> None:
    report = cast(dict[str, Any], json.loads(FREEZE.read_text()))
    records = cast(dict[str, dict[str, Any]], report["schedule_records"])

    assert set(records) == set(evaluator.ALL_CONTROLS)
    for record in records.values():
        assert record["events"] == 247
        assert record["stage_counts"] == {
            "train": 62,
            "test": 90,
            "eval": 61,
            "final": 34,
        }
    assert records["primary"]["schedule_hash"] == (
        "b233abb0772afa0a78f638fe412f8e644f37b83a250f79c3514d1c22d6725239"
    )
    assert records["direction_flip"]["schedule_hash"] == (
        "c265702f8b82e45b7e251912f30621adc8d33e03571470defd1910acc75b22ce"
    )
    assert records["equal_weight_side"]["schedule_hash"] == (
        "6ceb11c22d9156e0575e89ef807d14ce7a6c4f712533775ea4ea7941fad5598f"
    )
    assert records["stale_network_24h"]["schedule_hash"] == (
        "c0883c711e586c23422580425cae12e9a4fc92ef66e842f19590e96848e9183c"
    )
    assert records["symbol_permuted_network"]["schedule_hash"] == (
        "5f2037a644fe249b3830bc63e44604a51425f73277c2982ed6cfc2bde1ffaf83"
    )
    assert records["deterministic_random_side"]["schedule_hash"] == (
        "719a2c5f0420b1ee251486bfdc9dc2501829d092a711b435b24077871273fbaa"
    )
    assert records["extra_latency_1h"]["schedule_hash"] == (
        "bd1c232bb8e073e17602eca784a3ee02c0114b16268fe22544c0f08e878a7e55"
    )


def test_fcir_freeze_verifier_recomputes_source_only_control_semantics() -> None:
    report = evaluator.verify_evaluator_freeze(FREEZE)

    assert report["manifest_hash"] == FREEZE_MANIFEST_HASH
