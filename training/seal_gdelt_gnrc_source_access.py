"""Seal GNRC source artifacts before any daily feature value is inspected."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "gdelt_gnrc_source_access_seal_v1"
PREREGISTRATION = Path(
    "results/gdelt_narrative_rotation_clearing_preregistration_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "ae175a242db1fa850164789e4a3e6f3f39b4ac8eae0fb877ce79e915ae3d67f3"
)
TRANSPORT_AMENDMENT = Path(
    "results/gdelt_source_transport_v2_amendment_2026-07-22.json"
)
TRANSPORT_AMENDMENT_SHA256 = (
    "9244fc5ab203abe1866a1960c9b652ec725a8e37a1196ea5e784c742d1bc9f18"
)
SOURCE_MANIFEST = Path(
    "results/gdelt_bitcoin_narrative_source_manifest_2026-07-20.json"
)
DAILY_SOURCE = Path("data/gdelt_bitcoin_narrative_daily_2020_2023.csv.gz")
RAW_SOURCE = Path("data/gdelt_bitcoin_narrative_timeline_raw_2020_2023.jsonl.gz")
EVALUATOR_SOURCE = Path("training/evaluate_gdelt_narrative_source_support.py")
EVALUATOR_SOURCE_SHA256 = (
    "b09ae64c831376bce686e55de4bcbe630924faad7acc8cf81bc6cd31ff2b735a"
)
PROTOCOL_DOCUMENT = Path(
    "docs/gdelt-narrative-rotation-clearing-source-support-protocol-2026-07-20.md"
)
PROTOCOL_DOCUMENT_SHA256 = (
    "dfcf20bb5a5191ebe084feb0e9c23bdcc911f89828a78564874b8e03f02dd5ca"
)
V2_BUILDER = Path("training/download_gdelt_bitcoin_narrative_daily_v2.py")
V2_BUILDER_SHA256 = "10f22a4a7e45080369dd989add6765caac3ad3a91c72f5e4bb26986904671569"
V1_BUILDER = Path("training/download_gdelt_bitcoin_narrative_daily.py")
V1_BUILDER_SHA256 = "d756990d979e901033891ad6a8c565783dc58e8a4a9e286d6e866929dd74889e"
DEFAULT_OUTPUT = Path("results/gdelt_gnrc_source_access_seal_2026-07-22.json")
SEALED_AT = "2026-07-22T00:00:00Z"
EXPECTED_COLUMNS = (
    "date",
    "available_at",
    "global_article_count",
    "broad_article_count",
    "failure_article_count",
    "constraint_article_count",
    "adoption_article_count",
)
EXPECTED_QUERY_IDS = {"broad", "failure", "constraint", "adoption"}
EXPECTED_OUTAGES = ("2020-10-20", "2023-03-23")


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    with repository_path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"GNRC source seal input is not a JSON object: {path}")
    return payload


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_manifest_metadata(payload: Mapping[str, Any]) -> None:
    manifest_hash = payload.get("manifest_hash")
    unhashed = dict(payload)
    unhashed.pop("manifest_hash", None)
    if not _is_sha256(manifest_hash) or manifest_hash != canonical_hash(unhashed):
        raise ValueError("GNRC source manifest internal hash changed")
    if payload.get("protocol_version") != "gdelt_bitcoin_narrative_daily_source_v2":
        raise ValueError("GNRC source manifest protocol changed")
    contract = payload.get("contract", {})
    if (
        contract.get("start_date") != "2020-01-01"
        or contract.get("end_date_exclusive") != "2024-01-01"
        or contract.get("required_date_resolution") != "day"
        or contract.get("availability") != "source_date UTC midnight + 48h15m"
        or {row.get("query_id") for row in contract.get("queries", [])}
        != EXPECTED_QUERY_IDS
        or "sparse_bin_policy" not in contract
    ):
        raise ValueError("GNRC source manifest contract changed")
    builder = payload.get("builder", {})
    if builder != {
        "path": str(V2_BUILDER),
        "sha256": V2_BUILDER_SHA256,
        "v1_dependency_path": str(V1_BUILDER),
        "v1_dependency_sha256": V1_BUILDER_SHA256,
    }:
        raise ValueError("GNRC source manifest builder changed")
    requests = payload.get("requests", {})
    if (
        requests.get("count") != 4
        or len(requests.get("response_hashes", [])) != 4
        or {row.get("query_id") for row in requests.get("response_hashes", [])}
        != EXPECTED_QUERY_IDS
        or any(
            row.get("start") != "2020-01-01"
            or row.get("end_exclusive") != "2024-01-01"
            or not _is_sha256(row.get("response_sha256"))
            for row in requests.get("response_hashes", [])
        )
    ):
        raise ValueError("GNRC source manifest request family changed")
    audit = payload.get("source_audit", {})
    if (
        audit.get("daily_rows") != 1461
        or audit.get("first_date") != "2020-01-01"
        or audit.get("last_date") != "2023-12-31"
        or audit.get("date_resolution") != "day"
        or tuple(audit.get("global_outage_dates", [])) != EXPECTED_OUTAGES
        or audit.get("global_outage_days") != 2
        or audit.get("known_global_outage_dates_match") is not True
        or audit.get("global_norm_consistent_across_available_queries") is not True
    ):
        raise ValueError("GNRC source manifest structural audit changed")
    outputs = payload.get("outputs", {})
    if (
        outputs.get("daily_path") != str(DAILY_SOURCE)
        or outputs.get("raw_bundle_path") != str(RAW_SOURCE)
        or tuple(outputs.get("daily_columns", [])) != EXPECTED_COLUMNS
        or not _is_sha256(outputs.get("daily_sha256"))
        or not _is_sha256(outputs.get("raw_bundle_sha256"))
    ):
        raise ValueError("GNRC source manifest output metadata changed")
    if payload.get("outcome_boundary") != {
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "return_or_pnl_fields_read": 0,
        "post_2023_news_rows_requested": 0,
        "economic_metrics_computed": False,
    }:
        raise ValueError("GNRC source manifest crossed the outcome boundary")


def build_seal() -> dict[str, Any]:
    frozen_inputs = {
        PREREGISTRATION: PREREGISTRATION_SHA256,
        TRANSPORT_AMENDMENT: TRANSPORT_AMENDMENT_SHA256,
        EVALUATOR_SOURCE: EVALUATOR_SOURCE_SHA256,
        PROTOCOL_DOCUMENT: PROTOCOL_DOCUMENT_SHA256,
        V2_BUILDER: V2_BUILDER_SHA256,
        V1_BUILDER: V1_BUILDER_SHA256,
    }
    for path, expected_hash in frozen_inputs.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"GNRC source seal frozen input changed: {path}")
    source_manifest = _load_json(SOURCE_MANIFEST)
    validate_manifest_metadata(source_manifest)
    daily_sha256 = sha256_file(DAILY_SOURCE)
    raw_sha256 = sha256_file(RAW_SOURCE)
    outputs = source_manifest["outputs"]
    if (
        outputs["daily_sha256"] != daily_sha256
        or outputs["raw_bundle_sha256"] != raw_sha256
    ):
        raise ValueError("GNRC source artifact hash differs from its manifest")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "preregistration_path": str(PREREGISTRATION),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "transport_amendment_path": str(TRANSPORT_AMENDMENT),
        "transport_amendment_sha256": TRANSPORT_AMENDMENT_SHA256,
        "source_manifest_path": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "daily_source_path": str(DAILY_SOURCE),
        "daily_source_sha256": daily_sha256,
        "raw_source_path": str(RAW_SOURCE),
        "raw_source_sha256": raw_sha256,
        "evaluator_source_path": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": EVALUATOR_SOURCE_SHA256,
        "protocol_document_path": str(PROTOCOL_DOCUMENT),
        "protocol_document_sha256": PROTOCOL_DOCUMENT_SHA256,
        "feature_values_inspected_before_seal": False,
        "market_outcomes_opened_before_seal": False,
        "sealed_at": SEALED_AT,
    }


def write_once(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = repository_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"GNRC source-access seal is write-once: {destination}")
    seal = build_seal()
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(seal, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"GNRC source-access seal is write-once: {destination}"
        ) from error
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(write_once(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
