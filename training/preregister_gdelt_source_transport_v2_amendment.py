"""Write the outcome-blind GDELT sparse-bin transport amendment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "gdelt_source_transport_v2_amendment_v1"
AS_OF_DATE = "2026-07-22"
V2_COMMIT = "ec3749c7c16a6cb598a4b749abfaf83d175e40ed"
V1_BUILDER = Path("training/download_gdelt_bitcoin_narrative_daily.py")
V1_BUILDER_SHA256 = "d756990d979e901033891ad6a8c565783dc58e8a4a9e286d6e866929dd74889e"
V2_BUILDER = Path("training/download_gdelt_bitcoin_narrative_daily_v2.py")
V2_BUILDER_SHA256 = "10f22a4a7e45080369dd989add6765caac3ad3a91c72f5e4bb26986904671569"
PREREGISTRATION = Path(
    "results/gdelt_narrative_rotation_clearing_preregistration_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "ae175a242db1fa850164789e4a3e6f3f39b4ac8eae0fb877ce79e915ae3d67f3"
)
AMENDMENT_DOCUMENT = Path(
    "docs/gdelt-bitcoin-narrative-source-v2-amendment-2026-07-22.md"
)
AMENDMENT_DOCUMENT_SHA256 = (
    "c426b7277314dd2181849c3e47bc36bc93c6d2c1cb4ef7c8fe219cde0c0a5699"
)
AMENDMENT_SOURCE = Path("training/preregister_gdelt_source_transport_v2_amendment.py")
DEFAULT_OUTPUT = Path("results/gdelt_source_transport_v2_amendment_2026-07-22.json")
UTC = timezone.utc


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
        raise ValueError("GDELT transport amendment input must be a JSON object")
    return payload


def validate_frozen_inputs() -> tuple[dict[str, Any], Any]:
    expected_hashes = {
        V1_BUILDER: V1_BUILDER_SHA256,
        V2_BUILDER: V2_BUILDER_SHA256,
        PREREGISTRATION: PREREGISTRATION_SHA256,
        AMENDMENT_DOCUMENT: AMENDMENT_DOCUMENT_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"GDELT transport amendment frozen input changed: {path}")
    preregistration = _load_json(PREREGISTRATION)
    manifest_hash = preregistration.get("manifest_hash")
    unhashed = dict(preregistration)
    unhashed.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(unhashed):
        raise ValueError("GDELT transport amendment preregistration hash changed")
    prereg_source = Path(str(preregistration["preregistration_source"]))
    prereg_document = Path(str(preregistration["preregistration_document"]))
    if (
        sha256_file(prereg_source) != preregistration["preregistration_source_sha256"]
        or sha256_file(prereg_document)
        != preregistration["preregistration_document_sha256"]
    ):
        raise ValueError("GDELT transport amendment policy definition changed")
    v2 = importlib.import_module("training.download_gdelt_bitcoin_narrative_daily_v2")
    return preregistration, v2


def build_payload() -> dict[str, Any]:
    preregistration, v2 = validate_frozen_inputs()
    old_contract = preregistration["source_transport"]["contract"]
    new_contract = v2.source_contract(v2.Config())
    unchanged_fields = (
        "endpoint",
        "mode",
        "format",
        "queries",
        "start_date",
        "end_date_exclusive",
        "windowing",
        "required_date_resolution",
        "availability",
    )
    if any(old_contract[field] != new_contract[field] for field in unchanged_fields):
        raise ValueError("GDELT v2 changed a frozen feature-source field")
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "decision": "amend_transport_only_for_empirical_global_outage_bins",
        "original_preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": preregistration["manifest_hash"],
            "policy_source": preregistration["preregistration_source"],
            "policy_source_sha256": preregistration["preregistration_source_sha256"],
            "policy_document": preregistration["preregistration_document"],
            "policy_document_sha256": preregistration[
                "preregistration_document_sha256"
            ],
        },
        "transport": {
            "v1_builder": str(V1_BUILDER),
            "v1_builder_sha256": V1_BUILDER_SHA256,
            "v2_commit": V2_COMMIT,
            "v2_builder": str(V2_BUILDER),
            "v2_builder_sha256": V2_BUILDER_SHA256,
            "v2_contract": new_contract,
            "unchanged_contract_fields": list(unchanged_fields),
            "only_added_contract_field": "sparse_bin_policy",
        },
        "date_only_diagnostic": {
            "full_broad_rows": 1459,
            "expected_rows": 1461,
            "first_date": "2020-01-01",
            "last_date": "2023-12-31",
            "missing_dates": list(v2.KNOWN_GLOBAL_OUTAGE_DATES),
            "two_year_windows_reproduced_same_dates": True,
            "unrelated_anchor_query": "(economy OR government OR market)",
            "unrelated_anchor_omitted_both_dates": True,
            "article_count_values_inspected": 0,
        },
        "policy_invariants": {
            "feature_queries_changed": False,
            "feature_formulas_changed": False,
            "score_signs_changed": False,
            "windows_thresholds_holds_changed": False,
            "source_or_family_support_gates_changed": False,
            "market_protocol_changed": False,
        },
        "outcome_boundary": {
            "final_source_artifact_opened": False,
            "source_feature_values_inspected": False,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "outcomes_opened": False,
        },
        "next_action": (
            "commit this amendment, freeze source-support evaluator, then run v2; "
            "commit an outer source seal before parsing feature counts"
        ),
        "amendment_document": str(AMENDMENT_DOCUMENT),
        "amendment_document_sha256": AMENDMENT_DOCUMENT_SHA256,
        "amendment_source": str(AMENDMENT_SOURCE),
        "amendment_source_sha256": sha256_file(AMENDMENT_SOURCE),
        "created_at": datetime(2026, 7, 22, tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def write_once(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = repository_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"GDELT transport amendment is write-once: {destination}")
    payload = build_payload()
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"GDELT transport amendment is write-once: {destination}"
        ) from error
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(write_once(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
