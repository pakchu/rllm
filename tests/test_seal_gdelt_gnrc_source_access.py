from __future__ import annotations

from training import seal_gdelt_gnrc_source_access as seal


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
