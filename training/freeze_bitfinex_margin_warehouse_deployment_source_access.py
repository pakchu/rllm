"""Seal BFMWD source/evaluator bytes before any source feature is inspected."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "bitfinex_margin_warehouse_deployment_source_access_seal_v1"
PREREGISTRATION = Path(
    "results/bitfinex_margin_warehouse_deployment_preregistration_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "6e478bac6becb58d282867f4ee612d9d13e803d01985474477d6e3073cd49e58"
)
COMPARATOR_FREEZE = Path(
    "results/bitfinex_margin_warehouse_deployment_comparator_freeze_2026-07-20.json"
)
COMPARATOR_FREEZE_SHA256 = (
    "37ee403d33b5361c752b84ef94d46a05d991d82e1b0a77338b41a6c49e8410de"
)
SOURCE_MANIFEST = Path(
    "results/bitfinex_margin_funding_stats_source_manifest_2026-07-20.json"
)
SOURCE_MANIFEST_SHA256 = (
    "9d7c13d56983d7d33fec1c17e24f1794baca64fcfc666599b798d5d5b49cf9b9"
)
SOURCE_DATA = Path("data/bitfinex_margin_funding_stats_2020_2023.csv.gz")
SOURCE_DATA_SHA256 = (
    "71635b9f3a38efa7422a6fcf616859e6a41636bbb79ff0f85e160ef395b0d53c"
)
RAW_SOURCE = Path("data/bitfinex_margin_funding_stats_raw_2020_2023.jsonl.gz")
RAW_SOURCE_SHA256 = (
    "2f5ca2b344806be5bbfa63090fb79a86259d722e03c4f136cd316eb5787f8adb"
)
TRANSPORT_AMENDMENT = Path(
    "results/bitfinex_margin_funding_stats_transport_v2_amendment_2026-07-20.json"
)
TRANSPORT_AMENDMENT_SHA256 = (
    "1fc2d1b35242e7a1bd8232b3b0dfe65d479d0f8e2c4240c523efea1937dd00e9"
)
SQFD_PREFIX_TRANSPORT_FREEZE = Path(
    "results/bfmwd_sqfd_2023_comparator_prefix_transport_freeze_2026-07-20.json"
)
SQFD_PREFIX_TRANSPORT_FREEZE_SHA256 = (
    "c90a2370a76ba81a33b6b9c4102a0be27dbc08c89151d5905aee688403576913"
)
SQFD_PREFIX_MANIFEST = Path(
    "results/bfmwd_sqfd_2023_comparator_prefix_manifest_2026-07-20.json"
)
SQFD_PREFIX_MANIFEST_SHA256 = (
    "09c86f119e24a3379e8d35abf563b81c669e286c2b84e71a5868798c95e3e521"
)
SQFD_PREFIX = Path("data/bfmwd_sqfd_primary_clocks_2023_prefix.csv.gz")
SQFD_PREFIX_SHA256 = (
    "0afc8f0cce62e4276e3a6c0cfc66a0c91a868904236f7857445b88eb84db935a"
)
EVALUATOR_SOURCE = Path(
    "training/evaluate_bitfinex_margin_warehouse_deployment_support.py"
)
PROTOCOL_DOCUMENT = Path(
    "docs/bitfinex-margin-warehouse-deployment-support-protocol-2026-07-20.md"
)
DEFAULT_OUTPUT = Path(
    "results/bitfinex_margin_warehouse_deployment_source_access_seal_2026-07-20.json"
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
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
    payload = json.loads(repository_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"BFMWD seal input must be an object: {path}")
    return payload


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def build_payload() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("BFMWD preregistration changed before source seal")
    if sha256_file(COMPARATOR_FREEZE) != COMPARATOR_FREEZE_SHA256:
        raise ValueError("BFMWD comparator freeze changed before source seal")
    manifest = _load_json(SOURCE_MANIFEST)
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Bitfinex source manifest hash mismatch")
    if sha256_file(TRANSPORT_AMENDMENT) != TRANSPORT_AMENDMENT_SHA256:
        raise ValueError("Bitfinex transport amendment hash mismatch")
    if manifest.get("protocol_version") != "bitfinex_margin_funding_stats_source_v2":
        raise ValueError("Bitfinex source manifest is not transport v2")
    if manifest.get("transport_amendment", {}).get("sha256") != (
        TRANSPORT_AMENDMENT_SHA256
    ):
        raise ValueError("Bitfinex source manifest lost transport amendment")
    contract = manifest.get("source_contract", {})
    if contract.get("outcomes_opened") is not False:
        raise ValueError("Bitfinex source manifest opened outcomes")
    if contract.get("market_or_pnl_columns_loaded") is not False:
        raise ValueError("Bitfinex source manifest loaded market/PnL columns")
    if contract.get("post_2023_rows_requested") is not False:
        raise ValueError("Bitfinex source manifest requested post-2023 rows")
    files = manifest.get("files", {})
    if files.get("canonical", {}).get("path") != str(SOURCE_DATA):
        raise ValueError("Bitfinex canonical source path changed")
    if sha256_file(SOURCE_DATA) != SOURCE_DATA_SHA256:
        raise ValueError("Bitfinex canonical source frozen hash mismatch")
    if files.get("canonical", {}).get("sha256") != SOURCE_DATA_SHA256:
        raise ValueError("Bitfinex canonical source hash mismatch")
    if files.get("raw", {}).get("path") != str(RAW_SOURCE):
        raise ValueError("Bitfinex raw source path changed")
    if sha256_file(RAW_SOURCE) != RAW_SOURCE_SHA256:
        raise ValueError("Bitfinex raw source frozen hash mismatch")
    if files.get("raw", {}).get("sha256") != RAW_SOURCE_SHA256:
        raise ValueError("Bitfinex raw source hash mismatch")
    builder = files.get("builder", {})
    builder_path = Path(str(builder.get("path", "")))
    if builder_path != Path("training/download_bitfinex_margin_funding_stats_v2.py"):
        raise ValueError("Bitfinex source-builder path changed")
    if builder.get("sha256") != sha256_file(builder_path):
        raise ValueError("Bitfinex source-builder hash mismatch")
    frozen_prefix = {
        SQFD_PREFIX_TRANSPORT_FREEZE: SQFD_PREFIX_TRANSPORT_FREEZE_SHA256,
        SQFD_PREFIX_MANIFEST: SQFD_PREFIX_MANIFEST_SHA256,
        SQFD_PREFIX: SQFD_PREFIX_SHA256,
    }
    for path, expected_hash in frozen_prefix.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"SQFD comparator-prefix hash mismatch: {path}")
    prefix_manifest = _load_json(SQFD_PREFIX_MANIFEST)
    prefix_unhashed = dict(prefix_manifest)
    prefix_hash = prefix_unhashed.pop("manifest_hash", None)
    if prefix_hash != canonical_hash(prefix_unhashed):
        raise ValueError("SQFD comparator-prefix manifest hash mismatch")
    if prefix_manifest.get("output", {}).get("sha256") != SQFD_PREFIX_SHA256:
        raise ValueError("SQFD comparator-prefix output binding changed")

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "decision": "freeze_support_evaluator_before_source_feature_access",
        "feature_values_inspected_before_seal": False,
        "market_outcomes_opened_before_seal": False,
        "bindings": {
            "preregistration": binding(PREREGISTRATION),
            "comparator_freeze": binding(COMPARATOR_FREEZE),
            "transport_amendment": binding(TRANSPORT_AMENDMENT),
            "source_manifest": binding(SOURCE_MANIFEST),
            "canonical_source": binding(SOURCE_DATA),
            "raw_source": binding(RAW_SOURCE),
            "sqfd_prefix_transport_freeze": binding(SQFD_PREFIX_TRANSPORT_FREEZE),
            "sqfd_prefix_manifest": binding(SQFD_PREFIX_MANIFEST),
            "sqfd_prefix": binding(SQFD_PREFIX),
            "evaluator_source": binding(EVALUATOR_SOURCE),
            "protocol_document": binding(PROTOCOL_DOCUMENT),
        },
        "outcome_boundary": {
            "btc_market_rows_read": 0,
            "funding_paid_rows_read": 0,
            "post_2023_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "outcomes_opened": False,
        },
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def write_once(path: str | Path, payload: dict[str, Any]) -> None:
    target = repository_path(path)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite frozen BFMWD seal: {path}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload = build_payload()
    write_once(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
