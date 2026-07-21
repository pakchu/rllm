from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training import evaluate_refined_product_divergence_shock_support as rpds


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "results/refined_product_divergence_shock_source_support_2026-07-21.json"
)
CLOCK = ROOT / "data/refined_product_divergence_shock_clocks_2019_2023.csv.gz"
ARTIFACT_SHA256 = "e9ab44864ddb0e5c92c69c4eb50bc32a941f50f9fe7ab064df388e0f618993b6"
CLOCK_SHA256 = "729f9a236923909ff906f499fb2fe1bada8c38b1db5382b38e1cf4a189f9f52e"
MANIFEST_HASH = "b1e03654bc2ebf2ef2dd90df935e4264c8f58b77f725633081653a944532cadf"


def _report() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_committed_support_artifact_is_hash_bound_and_reproducible(
    tmp_path: Path,
) -> None:
    assert rpds.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    assert rpds.sha256_file(CLOCK) == CLOCK_SHA256
    reproduced = rpds.build_report(clock_output=tmp_path / "clocks.csv.gz")
    assert reproduced == _report()
    assert rpds.sha256_file(tmp_path / "clocks.csv.gz") == CLOCK_SHA256
    assert reproduced["manifest_hash"] == MANIFEST_HASH


def test_support_passes_but_novelty_fails_before_outcomes() -> None:
    report = _report()
    assert report["support"]["passed"] is True
    primary = report["support"]["summaries"]["primary"]
    assert primary["train"]["events"] == 54
    assert primary["selection"]["events"] == 15

    novelty = report["novelty"]
    assert novelty["evaluated"] is True
    assert novelty["passed"] is False
    assert novelty["metrics"]["epsb:crude_only"]["rpds_tolerant_coverage"] == 1.0
    assert (
        novelty["metrics"]["epsb:refined_products_only"]["rpds_tolerant_coverage"]
        == 1.0
    )
    assert report["decision"]["status"] == "retired_before_outcomes"
    assert report["decision"]["outcome_evaluator_authorized"] is False


def test_artifact_preserves_the_outcome_boundary() -> None:
    report = _report()
    boundary = report["outcome_boundary"]
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert boundary["post_2023_source_rows_read"] == 0
    assert report["decision"]["economic_outcomes_opened"] is False
    assert report["decision"]["repair_authorized"] is False
