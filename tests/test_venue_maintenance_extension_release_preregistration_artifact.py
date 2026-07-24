from __future__ import annotations

import json
from pathlib import Path

import training.preregister_venue_maintenance_extension_release as vmer


ARTIFACT = Path(
    "results/venue_maintenance_extension_release_preregistration_2026-07-24.json"
)
ARTIFACT_SHA256 = "de5cc97ddd7b1c3bfb155a1d3e3cd11e501e43148047c6e8fd9b4d48100e5809"
CONTRACT_SHA256 = "e583ce62b13516d7825c3454a226fe4dc990a693d5ecaebba40bdc97e416c303"
MANIFEST_SHA256 = "05ebb83b40a3d2e9e0b0776d22e78f8282e229fd07970e2e4352a3ffd08e09ac"
SOURCE_SHA256 = "2d457a0fecce6490ce7d0dbf59c50bb72f66ce762cb036706035c2607d38f06f"
DATASETS = {
    "train": (
        Path(
            "data/venue_maintenance_extension_release_synthetic_train_2026-07-24.jsonl"
        ),
        "c1e6a9f6f667451e741f4982989f24565e9983b720cc9f644f8ebfb61d5e98be",
        384,
    ),
    "calibration": (
        Path(
            "data/venue_maintenance_extension_release_"
            "synthetic_calibration_2026-07-24.jsonl"
        ),
        "5b16f0ab7ca24eff3199f2de768a127f415286d84693510fea9790c3d9353d2e",
        144,
    ),
    "adversarial": (
        Path(
            "data/venue_maintenance_extension_release_"
            "synthetic_adversarial_2026-07-24.jsonl"
        ),
        "a610ff7d04b8a55d1c0104ecacc2ebc1a31207d5be3dd62b03e30fcdfe6ef721",
        144,
    ),
    "swaps": (
        Path(
            "data/venue_maintenance_extension_release_synthetic_swaps_2026-07-24.jsonl"
        ),
        "150662f4e8de89f91db8c79cccdbfc6b8af3d6a93345d95172dac92af24e7ec2",
        96,
    ),
}


def test_preregistration_artifact_is_byte_exact_and_self_consistent() -> None:
    assert vmer.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["contract_sha256"] == CONTRACT_SHA256
    assert artifact["manifest_sha256"] == MANIFEST_SHA256
    assert vmer.canonical_hash(artifact["contract"]) == CONTRACT_SHA256
    without_manifest = dict(artifact)
    without_manifest.pop("manifest_sha256")
    assert vmer.canonical_hash(without_manifest) == MANIFEST_SHA256


def test_preregistration_source_and_synthetic_datasets_are_frozen() -> None:
    assert vmer.sha256_file(Path(vmer.__file__)) == SOURCE_SHA256
    artifact = json.loads(ARTIFACT.read_text())
    manifest = artifact["contract"]["synthetic_dataset_manifest"]
    for name, (path, digest, rows) in DATASETS.items():
        assert vmer.sha256_file(path) == digest
        assert manifest[name]["sha256"] == digest
        assert manifest[name]["rows"] == rows
        assert len(path.read_text().splitlines()) == rows


def test_preregistration_opens_no_historical_or_market_evidence() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    counters = artifact["evidence_counters"]
    assert counters["synthetic_rows_created"] == 768
    assert all(
        value == 0 for key, value in counters.items() if key != "synthetic_rows_created"
    )
    contract = artifact["contract"]
    assert contract["source"]["history"]["sealed_month_rows_materialized"] == 0
    assert contract["adaptation"]["historical_update_bodies"] == 0
    assert contract["adaptation"]["market_rows"] == 0
    assert contract["adaptation"]["returns_or_rewards"] == 0
    assert contract["artifact_write_policy"]["overwrite"] is False
    assert (
        contract["synthetic_partition_invariants"][
            "exact_decision_line_overlap_across_partitions"
        ]
        == 0
    )
    assert (
        contract["support_and_evaluation"]["post_selection"]["2024_plus_opened"]
        is False
    )
    assert contract["immutable_retirement"].endswith("without repair")


def test_model_and_raw_anchor_bindings_match_the_frozen_contract() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    validation = artifact["model_validation"]
    assert validation["model_id"] == vmer.MODEL_ID
    assert validation["revision"] == vmer.MODEL_REVISION
    assert validation["runtime_versions"] == dict(vmer.RUNTIME_VERSIONS)
    contract = artifact["contract"]
    assert contract["frozen_artifacts"]["market"]["raw_sha256"] == (
        vmer.MARKET_ARTIFACT_SHA256
    )
    observed = {
        row["family"]: row["raw_sha256"]
        for row in contract["frozen_artifacts"]["comparators"]
    }
    assert observed == {
        family: digest for family, (_, digest) in vmer.COMPARATOR_ARTIFACTS.items()
    }
    selectors = {
        row["family"]: row["primary_selector"]
        for row in contract["support_and_evaluation"]["novelty_gate"]["cohort"]
    }
    assert selectors == {
        family: dict(selector)
        for family, selector in vmer.COMPARATOR_PRIMARY_SELECTORS.items()
    }
    assert not {row["path"] for row in contract["frozen_artifacts"]["comparators"]} & {
        str(path) for path in vmer.FORBIDDEN_COMPARATORS
    }


def test_downstream_formulas_and_rllm_reward_boundary_are_explicit() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    downstream = artifact["contract"]["support_and_evaluation"]
    source = downstream["source_only_gate"]
    assert "divided by all source events" in source["venue_share"]
    assert "before revelation qualification" in source["event_universe"]
    market = downstream["causal_market_gate"]
    assert "before threshold and nonoverlap" in market["qualified_denominator"]
    assert "retained after global nonoverlap" in market["qualified_numerator"]
    novelty = downstream["novelty_gate"]
    assert "divided by candidate entry count" in novelty["exact_entry_overlap_share"]
    assert "one-to-one" in novelty["near_overlap_matching"]
    assert "every family" in novelty["aggregation"]
    economics = downstream["economic_gate"]
    assert economics["base"]["absolute_return"] == {
        "operator": ">",
        "threshold": {"selection": 0.0, "train": 0.0},
    }
    assert economics["stress"]["absolute_return"]["operator"] == ">"
    rllm = downstream["post_selection"]
    forbidden = set(rllm["rllm_forbidden"])
    assert "select or tune on selection reward" in forbidden
    assert "train on sealed-extension reward" in forbidden
    assert "select or tune on sealed-extension reward" in forbidden
    assert "train-only causal folds" in rllm["rllm_checkpoint_selection"]
