from __future__ import annotations

import hashlib
import json

from training import evaluate_overnight_rrp_flow_release as evaluator


EXPECTED_JSON_SHA256 = "57dcfc8d5cf945250f8e1ee18e95dc341d81c5dad372ead166c64ebc38e4d63d"
EXPECTED_MANIFEST = "db7e3333913a0f2d1eb2c38fdca7144121b957ad980c25479c7267b8d3fce939"


def test_stage1_rejection_artifact_is_hash_bound() -> None:
    payload = evaluator.STAGE1_OUTPUT.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_JSON_SHA256
    report = json.loads(payload)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == EXPECTED_MANIFEST
    assert evaluator._canonical_hash(core) == EXPECTED_MANIFEST
    assert report["gate_passed"] is False
    assert report["disposition"] == "REJECT_KEEP_2023_SEALED"


def test_stage1_reports_required_absolute_and_strict_metrics() -> None:
    report = json.loads(evaluator.STAGE1_OUTPUT.read_text())
    primary = report["headline_by_clock"]["primary"]
    assert primary["absolute_return_pct"] == 57.85712801188077
    assert primary["cagr_pct"] == 25.660851220629823
    assert primary["strict_mdd_pct"] == 17.961974840699323
    assert primary["cagr_to_strict_mdd"] == 1.4286208197155426
    assert primary["trades"] == 111
    assert report["primary_subperiod_headlines"]["2022"][
        "absolute_return_pct"
    ] < 0.0
    assert report["execution_diagnostics"]["physical_window"] == [
        "2021-01-01T00:00:00+00:00",
        "2023-01-01T00:00:00+00:00",
    ]
