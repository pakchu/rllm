from __future__ import annotations

import json

from training import seal_gdelt_gnrc_source_access as seal


SOURCE_ACCESS_SEAL_SHA256 = (
    "267cbc8c1edd3bbfbbb290f39536a79ff90d51a26eb942c1dabcab5113cc81e8"
)
EXPECTED_SOURCE_ACCESS_SEAL = {
    "daily_source_path": "data/gdelt_bitcoin_narrative_daily_2020_2023.csv.gz",
    "daily_source_sha256": (
        "52d98ee9d63049ca9b12a70f7728a56dfad520f6379feef4e173140e7581347b"
    ),
    "evaluator_source_path": "training/evaluate_gdelt_narrative_source_support.py",
    "evaluator_source_sha256": (
        "b09ae64c831376bce686e55de4bcbe630924faad7acc8cf81bc6cd31ff2b735a"
    ),
    "feature_values_inspected_before_seal": False,
    "market_outcomes_opened_before_seal": False,
    "preregistration_path": (
        "results/gdelt_narrative_rotation_clearing_preregistration_2026-07-20.json"
    ),
    "preregistration_sha256": (
        "ae175a242db1fa850164789e4a3e6f3f39b4ac8eae0fb877ce79e915ae3d67f3"
    ),
    "protocol_document_path": (
        "docs/gdelt-narrative-rotation-clearing-source-support-protocol-2026-07-20.md"
    ),
    "protocol_document_sha256": (
        "dfcf20bb5a5191ebe084feb0e9c23bdcc911f89828a78564874b8e03f02dd5ca"
    ),
    "protocol_version": "gdelt_gnrc_source_access_seal_v1",
    "raw_source_path": (
        "data/gdelt_bitcoin_narrative_timeline_raw_2020_2023.jsonl.gz"
    ),
    "raw_source_sha256": (
        "eb2ca3c2353f1a2c589b850e4e33df16e70027596dd872f77753608ed6463044"
    ),
    "sealed_at": "2026-07-22T00:00:00Z",
    "source_manifest_path": (
        "results/gdelt_bitcoin_narrative_source_manifest_2026-07-20.json"
    ),
    "source_manifest_sha256": (
        "b6e413cca8ba62ca614c5343c81c59e08c04b9819b6f061d123cfa2f0dbc0c68"
    ),
    "transport_amendment_path": (
        "results/gdelt_source_transport_v2_amendment_2026-07-22.json"
    ),
    "transport_amendment_sha256": (
        "9244fc5ab203abe1866a1960c9b652ec725a8e37a1196ea5e784c742d1bc9f18"
    ),
}


def _manifest() -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol_version": "gdelt_bitcoin_narrative_daily_source_v2",
        "contract": {
            "start_date": "2020-01-01",
            "end_date_exclusive": "2024-01-01",
            "required_date_resolution": "day",
            "availability": "source_date UTC midnight + 48h15m",
            "queries": [
                {"query_id": query_id} for query_id in sorted(seal.EXPECTED_QUERY_IDS)
            ],
            "sparse_bin_policy": {"all_query_absence": "all counts zero"},
        },
        "builder": {
            "path": str(seal.V2_BUILDER),
            "sha256": seal.V2_BUILDER_SHA256,
            "v1_dependency_path": str(seal.V1_BUILDER),
            "v1_dependency_sha256": seal.V1_BUILDER_SHA256,
        },
        "requests": {
            "count": 4,
            "response_hashes": [
                {
                    "query_id": query_id,
                    "start": "2020-01-01",
                    "end_exclusive": "2024-01-01",
                    "response_sha256": f"{index + 1:x}" * 64,
                }
                for index, query_id in enumerate(sorted(seal.EXPECTED_QUERY_IDS))
            ],
        },
        "source_audit": {
            "daily_rows": 1461,
            "first_date": "2020-01-01",
            "last_date": "2023-12-31",
            "date_resolution": "day",
            "global_outage_dates": list(seal.EXPECTED_OUTAGES),
            "global_outage_days": 2,
            "known_global_outage_dates_match": True,
            "global_norm_consistent_across_available_queries": True,
        },
        "outputs": {
            "daily_path": str(seal.DAILY_SOURCE),
            "daily_sha256": "a" * 64,
            "daily_columns": list(seal.EXPECTED_COLUMNS),
            "raw_bundle_path": str(seal.RAW_SOURCE),
            "raw_bundle_sha256": "b" * 64,
        },
        "outcome_boundary": {
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_news_rows_requested": 0,
            "economic_metrics_computed": False,
        },
    }
    payload["manifest_hash"] = seal.canonical_hash(payload)
    return payload


def test_manifest_metadata_validation_is_source_only_and_fail_closed() -> None:
    payload = _manifest()
    seal.validate_manifest_metadata(payload)
    boundary = payload["outcome_boundary"]
    assert isinstance(boundary, dict)
    boundary["economic_metrics_computed"] = True
    payload["manifest_hash"] = seal.canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    try:
        seal.validate_manifest_metadata(payload)
    except ValueError as error:
        assert "outcome boundary" in str(error)
    else:
        raise AssertionError("tampered outcome boundary was accepted")


def test_seal_generator_is_bound_to_committed_ancestry_and_evaluator() -> None:
    expected = {
        seal.PREREGISTRATION: seal.PREREGISTRATION_SHA256,
        seal.TRANSPORT_AMENDMENT: seal.TRANSPORT_AMENDMENT_SHA256,
        seal.EVALUATOR_SOURCE: seal.EVALUATOR_SOURCE_SHA256,
        seal.PROTOCOL_DOCUMENT: seal.PROTOCOL_DOCUMENT_SHA256,
        seal.V2_BUILDER: seal.V2_BUILDER_SHA256,
        seal.V1_BUILDER: seal.V1_BUILDER_SHA256,
    }
    for path, expected_hash in expected.items():
        assert seal.sha256_file(path) == expected_hash


def test_committed_source_access_seal_matches_exact_content_and_hash() -> None:
    assert seal.sha256_file(seal.DEFAULT_OUTPUT) == SOURCE_ACCESS_SEAL_SHA256
    with seal.repository_path(seal.DEFAULT_OUTPUT).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload == EXPECTED_SOURCE_ACCESS_SEAL
    assert seal.build_seal() == EXPECTED_SOURCE_ACCESS_SEAL
