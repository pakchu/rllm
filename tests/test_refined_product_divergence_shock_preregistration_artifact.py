from __future__ import annotations

import json
from pathlib import Path

from training import preregister_refined_product_divergence_shock as rpds


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "results/refined_product_divergence_shock_preregistration_2026-07-21.json"
)
ARTIFACT_SHA256 = "b66454c6739080171d26199addc0db553affb79f43cf7c3aab868de3f656e97a"
MANIFEST_HASH = "df7bc965823165e7cffb20353c9edb0aa472544c562dd656b5cd4f92bfc5d8c0"


def test_committed_preregistration_is_hash_bound_and_reproducible() -> None:
    assert rpds.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    committed = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert rpds.build_preregistration() == committed
    assert committed["manifest_hash"] == MANIFEST_HASH


def test_artifact_authorizes_only_source_support_and_novelty() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert report["authorization"]["next_action"] == "source support and novelty only"
    assert report["authorization"]["outcome_evaluator"] is False
    assert report["outcome_boundary"]["source_value_rows_read"] == 0
    assert report["outcome_boundary"]["comparator_clock_rows_read"] == 0
    assert report["outcome_boundary"]["btc_market_rows_read"] == 0
