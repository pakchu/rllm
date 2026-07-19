from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from training import evaluate_discordant_tail_absorption_consensus as evaluator


RESULT = Path("results/discordant_tail_absorption_consensus_train_2023_2026-07-19.json")
DOC = Path("docs/discordant-tail-absorption-consensus-train-result-2026-07-19.md")
RESULT_SHA256 = "d7819070e6ecf45ca27ae43bf6c51bdf6d6ece6cf7390f3a4697a3b3e6ed16b9"
DOC_SHA256 = "d57b669354589d835e757fb9f7abc8ae56cfa4f7a21232acc47a1f7c034352b8"
MANIFEST_HASH = "2b56eaebadbd206733986bc419d5ea514a02815744c98997f93f701ef639c30f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(RESULT.read_text()))


def test_dtac_train_rejection_is_hash_locked_and_later_windows_remain_sealed() -> None:
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
        "f2d87eb64f40c4c0e55cf1f670193a803b3268172eab42dc33781253dac4d0c1"
    )
    assert report["evaluator_freeze_manifest_hash"] == (
        "70052add49b9212b90b6999e1d501b97f2e9c91cfa47c26468c1475a8a09a203"
    )
    for stage in ("test", "eval", "final"):
        assert not evaluator.STAGE_OUTPUTS[stage].exists()
        assert not evaluator.STAGE_DOCS[stage].exists()


def test_dtac_train_failed_economic_significance_and_stability_gates() -> None:
    report = _report()
    headline = report["primary"]["headline"]

    assert headline["absolute_return_pct"] == pytest.approx(-18.33437302950287)
    assert headline["cagr_pct"] == pytest.approx(-18.345701223666723)
    assert headline["strict_mdd_pct"] == pytest.approx(19.714081895063817)
    assert headline["cagr_to_strict_mdd"] == pytest.approx(-0.9305886686135902)
    assert headline["trades"] == 143
    assert headline["longs"] == 84
    assert headline["shorts"] == 59
    assert headline["mean_gross_underlying_bp"] == pytest.approx(-15.762405817608618)
    assert headline["weekly_cluster_signflip_p"] == pytest.approx(0.0071996400179991)
    assert report["primary"]["stress_headline"]["absolute_return_pct"] == (
        pytest.approx(-22.88775915437471)
    )
    assert report["primary"]["contained_half_headlines"]["2023_h1"][
        "absolute_return_pct"
    ] == pytest.approx(-12.472432283908862)
    assert report["primary"]["contained_half_headlines"]["2023_h2"][
        "absolute_return_pct"
    ] == pytest.approx(-6.697250818859701)
    assert set(report["failed_gates"]) == {
        "absolute_return_positive",
        "cagr_to_strict_mdd_at_least_3",
        "strict_mdd_at_most_15pct",
        "mean_gross_underlying_at_least_20bp",
        "each_contained_half_absolute_return_positive",
        "stress_absolute_return_positive",
        "stress_cagr_to_strict_mdd_at_least_2_5",
        "mechanism_control_margin_at_least_0_25",
    }


def test_dtac_train_used_exact_2023_execution_window_and_blocks_test() -> None:
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
