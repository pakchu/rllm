from __future__ import annotations

import json

import pytest

from training import evaluate_cboe_tail_hedge_disagreement as evaluator


def test_cthd_stage1_artifact_is_failed_and_physically_bounded() -> None:
    report = json.loads(evaluator.STAGE1_OUTPUT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == evaluator._canonical_hash(core)
    assert report["manifest_hash"] == (
        "22b07be2336bc56e92ff36f96cf87cfd4695298e36fe94035304c166192a2b69"
    )
    assert report["gate_passed"] is False
    assert report["disposition"] == "REJECT_KEEP_2023_SEALED"
    primary = report["headline_by_clock"]["primary"]
    assert primary["absolute_return_pct"] == pytest.approx(-13.5191623336)
    assert primary["cagr_pct"] == pytest.approx(-7.0095517840)
    assert primary["strict_mdd_pct"] == pytest.approx(25.3488784153)
    assert primary["cagr_to_strict_mdd"] == pytest.approx(-0.2765231530)
    assert primary["trades"] == 156
    diagnostics = report["execution_diagnostics"]
    assert diagnostics["market"]["rows"] == 210_240
    assert diagnostics["market"]["stopped_before_parsing_end_boundary"] is True
    assert diagnostics["funding"]["rows"] == 2_190
    assert diagnostics["funding"]["stopped_before_parsing_end_boundary"] is True


def test_failed_cthd_stage1_cannot_open_2023() -> None:
    freeze = evaluator.verify_evaluator_freeze()
    with pytest.raises(ValueError, match="2023 remains sealed"):
        evaluator._verified_passing_stage1(freeze["manifest_hash"])
