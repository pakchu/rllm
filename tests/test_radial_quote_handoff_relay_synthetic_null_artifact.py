from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import build_radial_quote_handoff_relay_support as rqhr


ARTIFACT = Path(
    "results/radial_quote_handoff_relay_synthetic_null_2026-07-23.json"
)
ARTIFACT_SHA256 = (
    "b96965b9e7ea452d9258fa8e0b4bec36d894a0e5b593cef9d24628a5f8101860"
)
MANIFEST_HASH = (
    "b41d9df8600fcab7c90927fbeacf12cf96c4e50b9992e57101e2ede31581ea38"
)
SUPPORT_SOURCE_SHA256 = (
    "9bb5890962838419b2452ef9f62e3e1db4a8db59b74b5d8bb3f72e5d4b713222"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_synthetic_null_artifact_is_immutable_and_self_consistent() -> None:
    report = payload()
    assert sha256(ARTIFACT) == ARTIFACT_SHA256
    assert report["manifest_hash"] == MANIFEST_HASH
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == rqhr.canonical_hash(core)
    assert report["support_source"] == {
        "path": str(rqhr.SCRIPT_PATH),
        "sha256": SUPPORT_SOURCE_SHA256,
    }
    assert sha256(rqhr.SCRIPT_PATH) == SUPPORT_SOURCE_SHA256
    rqhr.validate_synthetic_artifact_payload(report)


def test_synthetic_null_opened_no_real_source_comparator_or_outcome() -> None:
    report = payload()
    assert report["passed"] is True
    assert report["real_source_rows_read"] == 0
    assert report["real_rqhr_columns_read"] == 0
    assert report["comparator_rows_read"] == 0
    assert report["market_or_outcome_rows_read"] == 0
    null = report["synthetic_null"]
    assert null["source_rows_read"] == 0
    assert null["real_rqhr_columns_read"] == 0
    assert null["comparator_rows_read"] == 0
    assert null["market_or_outcome_rows_read"] == 0


def test_every_frozen_scenario_has_zero_confirmations_and_events() -> None:
    scenarios = payload()["synthetic_null"]["scenarios"]
    assert set(scenarios) == set(rqhr.NULL_SCENARIOS)
    assert {
        name: (
            row["complete_rows"],
            row["raw_confirmations"],
            row["accepted_events"],
            row["passed"],
        )
        for name, row in scenarios.items()
    } == {
        "smooth_symmetric": (105_120, 0, 0, True),
        "tick_rounded_anchor": (105_120, 0, 0, True),
        "stepped_asymmetric": (105_120, 0, 0, True),
        "missing_rows": (104_805, 0, 0, True),
        "discrete_asymmetric_ladder": (105_120, 0, 0, True),
    }
    assert all(
        row["grid_rows"] == 105_120
        and row["scheduled_snapshot_slots"] == 1_051_200
        for row in scenarios.values()
    )


def test_discrete_ladder_only_arms_and_all_races_timeout() -> None:
    scenarios = payload()["synthetic_null"]["scenarios"]
    assert scenarios["discrete_asymmetric_ladder"]["race_audit"] == {
        "ambiguities": 0,
        "arms": 1_829,
        "cancellations": 0,
        "confirmations": 0,
        "incomplete_cancellations": 0,
        "terminal_consumed_rows": 1_828,
        "timeouts": 1_828,
    }
    for name in rqhr.NULL_SCENARIOS[:-1]:
        assert all(
            count == 0
            for count in scenarios[name]["race_audit"].values()
        )
