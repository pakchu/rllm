from __future__ import annotations

import json
from pathlib import Path

from training import preregister_refined_product_divergence_shock as rpds


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "results/refined_product_divergence_shock_preregistration_2026-07-21.json"
)
ARTIFACT_SHA256 = "03ce0d29e0c67f6366959690feebeea9e96b854389e4b15804d9a7ee5cd277b2"
MANIFEST_HASH = "b58a259e128740058d7501188ab479cca1cfe0e40e8a18dc81df3f2e5058320a"


def test_committed_preregistration_is_hash_bound_and_reproducible() -> None:
    assert rpds.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    committed = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert rpds.build_preregistration() == committed
    assert committed["manifest_hash"] == MANIFEST_HASH


def test_artifact_authorizes_only_source_support_and_novelty() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert report["authorization"]["next_action"] == "source support and novelty only"
    assert report["authorization"]["outcome_evaluator"] is False
    assert (
        report["outcome_boundary"]["prefreeze_source_value_rows_read_for_schema"] == 1
    )
    assert (
        report["outcome_boundary"]["prefreeze_comparator_clock_rows_read_for_schema"]
        == 10
    )
    assert report["outcome_boundary"]["rpds_predicate_evaluations"] == 0
    assert report["outcome_boundary"]["comparator_overlap_metrics_computed"] == 0
    assert report["outcome_boundary"]["btc_market_rows_read"] == 0
