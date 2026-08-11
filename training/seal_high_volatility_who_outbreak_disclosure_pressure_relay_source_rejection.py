"""Seal the first frozen WHO source-contract rejection for HVWODP-24."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_who_outbreak_disclosure_pressure_relay as prereg


RESULT = Path("results/high_volatility_who_outbreak_disclosure_pressure_relay_source_rejection_2026-08-12.json")
BUILDER = Path("training/build_high_volatility_who_outbreak_disclosure_pressure_relay_support.py")
BUILDER_SHA = "685463a339d26aec9b4b8b64d78da0fbd568ab2ee6251997a0be577e2992c153"
BUILDER_COMMIT = "c85ef264"
PREREG_SHA = "ebbd2baf00eba25359f9a4adbfff116c2954696ad3df4b8b12537e8f13fe11dc"
PAGE_HASH = "1541db529c178908cafad9e4cbc117f32f4e97872e869f9831e9d023a40958f5"
OFFENDING_ITEM = {
    "DateCreated": "2022-01-05T13:01:12Z",
    "DonId": "2021-DON313",
    "Id": "00e00a3e-5b87-4118-9662-551357bcbabb",
    "LastModified": "2022-05-30T11:59:36Z",
    "PublicationDate": "2022-01-05T13:01:12Z",
    "PublicationDateAndTime": "2021-02-26T19:00:00Z",
    "SystemSourceKey": None,
    "Title": "Human infection with avian influenza A (H5N8) - Russian Federation",
    "UrlName": "2021-DON313",
}
OFFENDING_ITEM_HASH = "4e2c6d729d8cc40e39c2b3786fb4528bb0f0daae5308598ab102cd50e2fdf280"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA or sha(BUILDER) != BUILDER_SHA:
        raise RuntimeError("HVWODP frozen implementation drift")
    if canonical_hash(OFFENDING_ITEM) != OFFENDING_ITEM_HASH:
        raise RuntimeError("HVWODP offending item drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    core = {
        "protocol_version": "high_volatility_who_outbreak_disclosure_pressure_relay_source_rejection_v1",
        "policy_id": "HVWODP-24",
        "as_of_date": "2026-08-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"], "commit": "1a5fea23"},
        "frozen_source_builder": {"path": str(BUILDER), "sha256": BUILDER_SHA, "commit": BUILDER_COMMIT},
        "source_binding": {"collection_api": prereg.API, "first_page_payload_hash": PAGE_HASH, "first_page_rows": 50, "offending_position_one_based": 1, "offending_item": OFFENDING_ITEM, "offending_item_hash": OFFENDING_ITEM_HASH},
        "failed_contract": {
            "requirement": "every collection member must have a unique nonempty Id, SystemSourceKey, DonId, and UrlName",
            "json_path": "value[0].SystemSourceKey",
            "observed_value": None,
            "failure_class": "RuntimeError",
            "failure_message": "WHO SystemSourceKey identity is empty",
            "first_failure_short_circuit": True,
            "additional_unreached_violation_observed_in_same_item": "PublicationDate and PublicationDateAndTime disagree; the latter is outside the filtered publication interval",
        },
        "research_boundary": {"first_collection_page_opened": True, "full_collection_incidence_opened": False, "who_source_snapshot_written": False, "btc_variation_rows_opened": False, "gross9_rows_opened": False, "execution_prices_opened": False, "funding_rows_opened": False, "postentry_return_or_pnl_opened": False},
        "support_passed": False,
        "advance_to_gross9_novelty": False,
        "advance_to_economic_outcomes": False,
        "missing_key_synthesis_authorized": False,
        "membership_repair_authorized": False,
        "decision": "terminal_source_contract_reject",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVWODP source rejection drift")


if __name__ == "__main__":
    report = build()
    validate(report)
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(RESULT)
