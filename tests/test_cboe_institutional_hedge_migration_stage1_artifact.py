from __future__ import annotations

import json

import pytest

from training import evaluate_cboe_institutional_hedge_migration as evaluator


def test_cihm_stage1_artifact_is_failed_and_physically_bounded() -> None:
    report = json.loads(evaluator.STAGE1_OUTPUT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == evaluator._canonical_hash(core)
    assert report["manifest_hash"] == (
        "e39d43f7d485a1f55fa45699c28a99137a99bac7657abfce7e92fceb4e6a66cf"
    )
    assert report["gate_passed"] is False
    assert report["disposition"] == "REJECT_KEEP_2023_SEALED"
    primary = report["headline_by_clock"]["primary"]
    assert primary["absolute_return_pct"] == pytest.approx(-2.3057602393)
    assert primary["cagr_pct"] == pytest.approx(-1.1603931441)
    assert primary["strict_mdd_pct"] == pytest.approx(35.3129854679)
    assert primary["cagr_to_strict_mdd"] == pytest.approx(-0.0328602391)
    assert primary["trades"] == 151
    diagnostics = report["execution_diagnostics"]
    assert diagnostics["market"]["rows"] == 210_240
    assert diagnostics["market"]["stopped_before_parsing_end_boundary"] is True
    assert diagnostics["funding"]["rows"] == 2_190
    assert diagnostics["funding"]["stopped_before_parsing_end_boundary"] is True


def test_failed_cihm_stage1_cannot_open_2023() -> None:
    freeze = evaluator.verify_evaluator_freeze()
    with pytest.raises(ValueError, match="2023 remains sealed"):
        evaluator._verified_passing_stage1(freeze["manifest_hash"])
