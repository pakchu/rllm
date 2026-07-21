from __future__ import annotations

import json
from pathlib import Path

from training import preregister_refined_product_divergence_shock as rpds


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "results/refined_product_divergence_shock_preregistration_2026-07-21.json"
)
ARTIFACT_SHA256 = "082026f1091d3b5cdc7a2271d011479f949f8b7a1a9be71a1e2dbfc971b44c8d"
MANIFEST_HASH = "eb68c2d510c68223cd391fb219fb58d851153a34dfbaf73542978b2b9e56e4f9"


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
