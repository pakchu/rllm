from __future__ import annotations

from pathlib import Path

import pytest

from training import evaluate_bitmex_trollbox_attention_saturation as evaluator


RESULT_SHA256 = (
    "922eecd8bdcfb09df15937d8e571813dd90eca0ada6ca547d5aa5f1db7bc56a6"
)
DOCUMENT_SHA256 = (
    "480f53fb915b45a40dd98f8a2b096c1927d7ae9141c8aadf8e40941d39b91d27"
)


def test_train_rejection_is_hash_pinned_and_calendar_2022_remains_sealed() -> None:
    result_path = evaluator.STAGE_OUTPUTS["train"]
    document_path = evaluator.STAGE_DOCS["train"]
    assert evaluator._sha256(result_path) == RESULT_SHA256
    assert evaluator._sha256(document_path) == DOCUMENT_SHA256

    report = evaluator._load_json(result_path)
    evaluator._verify_manifest(report, label="train artifact")

    assert report["manifest_hash"] == (
        "20212dde331ad0d99f290a269af574167a0435d7f467ba224b176f534b3b5550"
    )
    assert report["candidate"] == evaluator.POLICY_ID
    assert report["stage"] == "train"
    assert report["stage_passed"] is False
    assert report["stage_passed"] == all(report["gate_checks"].values())
    assert report["opened_windows"] == ["train"]
    assert report["sealed_windows"] == ["test"]
    assert report["parameter_search_performed"] is False
    assert report["post_failure_repair_performed"] is False
    assert report["source_diagnostics"]["future_rows_decoded"] == 0
    assert report["source_diagnostics"]["market"]["last_timestamp"] == (
        "2021-12-31T23:55:00+00:00"
    )
    assert report["source_diagnostics"]["funding"]["last_timestamp"] == (
        "2021-12-31T16:00:00+00:00"
    )
    assert report["base_headline"] == {
        "absolute_return_pct": -6.606343725298236,
        "cagr_pct": -4.445281958056658,
        "strict_mdd_pct": 40.636993290690235,
        "cagr_to_strict_mdd": -0.1093900310551532,
        "trades": 358,
        "longs": 209,
        "shorts": 149,
        "mean_gross_underlying_bp": 11.44424148982395,
        "weekly_cluster_signflip_p": 0.9088545572721364,
        "weekly_clusters": 77,
        "weekly_test_method": "monte_carlo",
        "largest_absolute_week_share": 0.06756059674198527,
    }
    assert report["incidence"] == {
        "clear_semantic_events": 1_728,
        "boundary_exclusions": 0,
        "reference_ready_events": 1_728,
        "material_events": 519,
        "aligned_material_events_before_overlap": 383,
        "primary_events": 358,
        "primary_overlaps_skipped": 25,
        "alignment_ablation_events": 471,
        "alignment_ablation_overlaps_skipped": 48,
        "primary_longs": 209,
        "primary_shorts": 149,
    }
    assert report["gate_checks"]["minimum_trades"] is True
    assert report["gate_checks"]["minimum_longs"] is True
    assert report["gate_checks"]["minimum_shorts"] is True
    assert report["gate_checks"]["minimum_weekly_clusters"] is True
    assert report["gate_checks"]["absolute_return_positive"] is False
    assert report["gate_checks"]["strict_mdd_at_most_15pct"] is False
    assert report["gate_checks"][
        "weekly_cluster_signflip_p_at_most_10pct"
    ] is False
    assert len(report["base_metrics"]["trade_details"]) == 358
    assert all(
        item["bars_held"] == 24
        for item in report["base_metrics"]["trade_details"]
    )
    assert not Path(evaluator.STAGE_OUTPUTS["test"]).exists()
    assert not Path(evaluator.STAGE_DOCS["test"]).exists()


def test_failed_train_blocks_test_before_any_row_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("test outcome row was accessed after failed train")

    monkeypatch.setattr(evaluator, "_parse_market_months", forbidden)
    monkeypatch.setattr(evaluator, "_parse_funding_prefix", forbidden)

    with pytest.raises(ValueError, match="test remains sealed"):
        evaluator.load_execution_window("test")
