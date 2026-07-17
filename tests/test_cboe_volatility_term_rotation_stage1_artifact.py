from __future__ import annotations

import json

import pytest

from training import evaluate_cboe_volatility_term_rotation as evaluator


def test_cvtr_stage1_artifact_is_failed_and_physically_bounded() -> None:
    report = json.loads(evaluator.STAGE1_OUTPUT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == evaluator._canonical_hash(core)
    assert report["manifest_hash"] == (
        "9f5a5f42d4686c04566b2a1916bfe7959b3e0359e6bd9db3b464ae70a0cfd120"
    )
    assert report["gate_passed"] is False
    assert report["disposition"] == "REJECT_KEEP_2023_SEALED"
    primary = report["headline_by_clock"]["primary"]
    assert primary["absolute_return_pct"] == pytest.approx(-11.4373997648)
    assert primary["cagr_pct"] == pytest.approx(-5.8962100505)
    assert primary["strict_mdd_pct"] == pytest.approx(39.5061374458)
    assert primary["cagr_to_strict_mdd"] == pytest.approx(-0.1492479506)
    assert primary["trades"] == 281
    diagnostics = report["execution_diagnostics"]
    assert diagnostics["market"]["rows"] == 210_240
    assert diagnostics["market"]["stopped_before_parsing_end_boundary"] is True
    assert diagnostics["funding"]["rows"] == 2_190
    assert diagnostics["funding"]["stopped_before_parsing_end_boundary"] is True


def test_failed_cvtr_stage1_cannot_open_2023() -> None:
    freeze = evaluator.verify_evaluator_freeze()
    with pytest.raises(ValueError, match="2023 remains sealed"):
        evaluator._verified_passing_stage1(freeze["manifest_hash"])
