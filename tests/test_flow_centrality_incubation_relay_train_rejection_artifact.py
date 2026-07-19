from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from training import evaluate_flow_centrality_incubation_relay as evaluator


RESULT = Path("results/flow_centrality_incubation_relay_train_2023_2026-07-19.json")
DOC = Path("docs/flow-centrality-incubation-relay-train-result-2026-07-19.md")
RESULT_SHA256 = "1c30eb39917c5d8557120fb668ffccb76bad9c66e05567621360de33b95cf846"
DOC_SHA256 = "acc17a10170ecbb3966a2389317da21c113a564dbd9f5babd575737595c333f1"
MANIFEST_HASH = "a7e8d1bcf4acf5ae352f752c44f961359b1932543ee87e15741a48ea25898a33"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(RESULT.read_text()))


def test_fcir_train_rejection_is_hash_locked_and_later_windows_remain_sealed() -> None:
    assert _sha256(RESULT) == RESULT_SHA256
    assert _sha256(DOC) == DOC_SHA256
    report = _report()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == MANIFEST_HASH
    assert report["manifest_hash"] == evaluator._canonical_hash(core)
    assert report["candidate"] == evaluator.POLICY_ID
    assert report["stage"] == "train"
    assert report["stage_passed"] is False
    assert report["disposition"] == "REJECT_NO_REPAIR"
    assert report["opened_windows"] == ["train"]
    assert report["sealed_windows"] == ["test", "eval", "final"]
    assert report["evaluator_source_sha256"] == (
        "036b22442a2080e7ea5ffe914c605a9b1b1a55b128a315a2f2f05be7b37a736d"
    )
    assert report["evaluator_freeze_manifest_hash"] == (
        "db449785021045afda156a1e1772c11b2b8bcdaf19db9714c3424f0e4e2e88d9"
    )
    for stage in ("test", "eval", "final"):
        assert not evaluator.STAGE_OUTPUTS[stage].exists()
        assert not evaluator.STAGE_DOCS[stage].exists()


def test_fcir_train_failed_economic_significance_and_stability_gates() -> None:
    report = _report()
    headline = report["primary"]["headline"]

    assert headline["absolute_return_pct"] == pytest.approx(-2.4164982406338575)
    assert headline["cagr_pct"] == pytest.approx(-2.4181332014948986)
    assert headline["strict_mdd_pct"] == pytest.approx(6.0655811185508774)
    assert headline["cagr_to_strict_mdd"] == pytest.approx(-0.3986647205325996)
    assert headline["trades"] == 62
    assert headline["longs"] == 26
    assert headline["shorts"] == 36
    assert headline["mean_gross_underlying_bp"] == pytest.approx(4.156803753803208)
    assert headline["weekly_cluster_signflip_p"] == pytest.approx(0.6848657567121644)
    assert report["primary"]["stress_headline"]["absolute_return_pct"] == (
        pytest.approx(-4.81068570029668)
    )
    assert report["primary"]["contained_half_headlines"]["2023_h1"][
        "absolute_return_pct"
    ] == pytest.approx(-0.3406612354590943)
    assert report["primary"]["contained_half_headlines"]["2023_h2"][
        "absolute_return_pct"
    ] == pytest.approx(-2.0829327496133088)
    assert set(report["failed_gates"]) == {
        "absolute_return_positive",
        "cagr_to_strict_mdd_at_least_3",
        "weekly_cluster_signflip_p_at_most_10pct",
        "mean_gross_underlying_at_least_20bp",
        "each_contained_half_absolute_return_positive",
        "stress_absolute_return_positive",
        "stress_cagr_to_strict_mdd_at_least_2_5",
        "mechanism_control_margin_at_least_0_25",
    }


def test_fcir_train_used_exact_2023_execution_window_and_blocks_test() -> None:
    report = _report()
    diagnostics = report["execution_diagnostics"]

    assert diagnostics["physical_window"] == [
        "2023-01-01T00:00:00+00:00",
        "2024-01-01T00:00:00+00:00",
    ]
    assert diagnostics["market"]["rows"] == 105_120
    assert diagnostics["funding"]["rows"] == 1_095
    assert diagnostics["funding"]["maximum_absolute_grid_offset_ms"] == 29.0
    with pytest.raises(ValueError, match="did not pass; test remains sealed"):
        evaluator._verified_prior_reports(
            "test",
            freeze_hash=report["evaluator_freeze_manifest_hash"],
        )
