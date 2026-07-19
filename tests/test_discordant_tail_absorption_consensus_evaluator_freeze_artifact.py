from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from training import evaluate_discordant_tail_absorption_consensus as evaluator


FREEZE = Path(
    "results/discordant_tail_absorption_consensus_evaluator_freeze_2026-07-19.json"
)
CONTROLS = Path(
    "data/discordant_tail_absorption_consensus_control_clocks_2023_2026.csv.gz"
)
EVALUATOR_SHA256 = "f2d87eb64f40c4c0e55cf1f670193a803b3268172eab42dc33781253dac4d0c1"
FREEZE_SHA256 = "c28d8bc7277799842b6e21c818e2f997bcbb8530c4b0aee2108ae24569029633"
CONTROL_SHA256 = "4f72b18f98de277e9381b2fa2a715500ea8692f70995f8bbd9914f0ac591a559"
FREEZE_MANIFEST_HASH = (
    "70052add49b9212b90b6999e1d501b97f2e9c91cfa47c26468c1475a8a09a203"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dtac_evaluator_freeze_is_hash_locked_and_outcome_unopened() -> None:
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
        "hold_bars": 96,
        "exact_cluster_max": 20,
        "cluster_draws": 20_000,
        "cluster_seed": 20_260_719,
    }


def test_dtac_frozen_control_family_and_stage_counts_are_exact() -> None:
    report = cast(dict[str, Any], json.loads(FREEZE.read_text()))
    records = cast(dict[str, dict[str, Any]], report["schedule_records"])

    assert set(records) == set(evaluator.ALL_CONTROLS)
    expected = {
        "primary": (695, [143, 190, 247, 115]),
        "direction_flip": (695, [143, 190, 247, 115]),
        "all_six_premium_side": (695, [143, 190, 247, 115]),
        "all_six_flow_fade_side": (695, [143, 190, 247, 115]),
        "deterministic_random_side": (695, [143, 190, 247, 115]),
        "symbol_permuted_premium_pairing": (703, [170, 188, 242, 103]),
        "stale_premium_pairing_24h": (679, [167, 177, 221, 114]),
        "extra_latency_1h": (695, [143, 190, 247, 115]),
    }
    for name, (events, counts) in expected.items():
        assert records[name]["events"] == events
        assert records[name]["stage_counts"] == dict(
            zip(evaluator.STAGE_ORDER, counts, strict=True)
        )

    assert {name: row["schedule_hash"] for name, row in records.items()} == {
        "primary": "788fd9878671918c73934527cfd5eefb532dd15e5c94093bcf0919c580b8ccb3",
        "direction_flip": "5767719370569e7742f67ce916456dc3c43458c4e54ae34709fab03c5cc9a0e2",
        "all_six_premium_side": "0b61d808a36cfd5c98f8fd37e6b79a9e682b75c022e891cf3728e345c32fe402",
        "all_six_flow_fade_side": "e88b95ec38082b3b0d3fa3c29fd7c0977bfbee5e10e98cf2eebfc458ead0757a",
        "deterministic_random_side": "4239ffb7b7be1a78a5de206a504841eae2dadfc42a789b21d75cb5927d727d3d",
        "symbol_permuted_premium_pairing": "b4b4f2264990ad80a7e2b183a41c7a2770cb3ca2c6d85bd02aeebc727f48997e",
        "stale_premium_pairing_24h": "e8ab940f98a0a59314afe4af50572eba3dac1f4ca51255c073065ccec08a98a0",
        "extra_latency_1h": "f91f784d57053dcf69e7fd8b357fdb522c41354790e6eef64a9906a998853cdb",
    }


def test_dtac_freeze_verifier_recomputes_source_only_control_semantics() -> None:
    report = evaluator.verify_evaluator_freeze(FREEZE)

    assert report["manifest_hash"] == FREEZE_MANIFEST_HASH
