from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import evaluate_btcdom_leverage_polarity_decomposition as evaluator


RESULT = Path(
    "results/btcdom_leverage_polarity_decomposition_train_2022_2026-07-20.json"
)
DOCUMENT = Path(
    "docs/btcdom-leverage-polarity-decomposition-train-result-2026-07-20.md"
)
RESULT_SHA256 = "495e165c43a1d70e7788d40a6cedca6e5096f0f0ac51e1df168a670bb82f3c87"
DOCUMENT_SHA256 = "3e27a7eb8835c35823b799d990b3b9e2378fae88a376454a4842405b906d6016"
MANIFEST_HASH = "3a714ad877e6e2e613ffd7c8168d747b118bb10bf948a8d9d294acab5abb4693"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dlpd_train_result_is_hash_locked_and_rejected() -> None:
    assert _sha256(RESULT) == RESULT_SHA256
    assert _sha256(DOCUMENT) == DOCUMENT_SHA256
    report = json.loads(RESULT.read_text())
    assert report["manifest_hash"] == MANIFEST_HASH
    assert report["stage"] == "train"
    assert report["stage_passed"] is False
    assert report["disposition"] == "REJECT_NO_REPAIR"
    assert report["opened_windows"] == ["train"]
    assert report["sealed_windows"] == ["test", "eval", "final"]
    assert report["component_controls_cannot_repair_primary"] is True
    assert report["primary"]["headline"]["absolute_return_pct"] == pytest.approx(
        -26.290716517492207
    )
    assert report["primary"]["headline"]["cagr_pct"] == pytest.approx(
        -26.306115172738686
    )
    assert report["primary"]["headline"]["strict_mdd_pct"] == pytest.approx(
        34.392135453385194
    )
    assert report["primary"]["headline"]["cagr_to_strict_mdd"] == pytest.approx(
        -0.7648875193688908
    )
    assert report["primary"]["headline"]["trades"] == 237
    assert report["primary"]["stress_headline"]["absolute_return_pct"] == pytest.approx(
        -32.96028280418406
    )
    assert report["controls"]["direction_flip"]["headline"][
        "absolute_return_pct"
    ] == pytest.approx(-2.1082594882880556)
    assert report["controls"]["deterministic_random_side"]["headline"][
        "weekly_cluster_signflip_p"
    ] == pytest.approx(0.8094095295235239)


def test_dlpd_rejection_keeps_all_later_outcome_files_absent() -> None:
    for path in (
        evaluator.STAGE_SOURCE_MANIFESTS["test"],
        evaluator.STAGE_OUTPUTS["test"],
        evaluator.STAGE_OUTPUTS["eval"],
        evaluator.STAGE_OUTPUTS["final"],
        evaluator.STAGE_SOURCE_DIRS["test"] / "BTCUSDT_5m.csv.gz",
    ):
        assert not path.exists(), f"rejected DLPD stage was opened: {path}"
