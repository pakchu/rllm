from __future__ import annotations

import json

from training import evaluate_funding_adjusted_delivery_carry as evaluate


EXPECTED_MANIFEST_HASH = "f705ca3b2f4fe43f789de6a14359c15306d991961f27119ac64083c07ab13417"
EXPECTED_SOURCE_SHA256 = "9fffb213e235d83dcbfe2fecfef34876cb1f6a2ac856a3fb3f575fa68e2af588"


def test_evaluator_freeze_is_hash_bound_and_outcome_closed() -> None:
    report = json.loads(evaluate.EVALUATOR_FREEZE.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert evaluate._canonical_hash(core) == EXPECTED_MANIFEST_HASH
    assert report["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert report["evaluator_source_sha256"] == EXPECTED_SOURCE_SHA256
    assert evaluate._sha256(evaluate.EVALUATOR_SOURCE) == EXPECTED_SOURCE_SHA256
    assert report["opened_windows"] == []
    assert report["sealed_windows"] == [
        "stage1_2021_2022",
        "stage2_2023",
        "2024",
        "2025",
        "2026_ytd",
    ]
    assert report["mutable_parameters"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_settlement_marks_loaded_during_freeze"] == 0
    assert report["execution_simulation_run_during_freeze"] is False


def test_control_clocks_are_frozen_before_outcomes() -> None:
    report = evaluate.verify_evaluator_freeze()
    assert {
        name: (
            row["stage1"],
            row["stage2"],
            row["all_pre2024"],
        )
        for name, row in report["control_schedules"].items()
    } == {
        "primary": (30, 9, 39),
        "direction_flip": (30, 9, 39),
        "basis_only": (11, 4, 15),
        "constant_long_perp_short_quarter": (8, 4, 12),
        "one_funding_event_delay": (30, 9, 39),
    }
