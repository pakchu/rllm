"""Seal the first frozen source-availability rejection for HVAANV-24."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_active_address_network_value_relay as prereg


RESULT = Path("results/high_volatility_active_address_network_value_relay_source_rejection_2026-08-12.json")
BUILDER = Path("training/build_high_volatility_active_address_network_value_relay_support.py")
BUILDER_SHA = "f36925be26eefebbabb6c829c77a0b464f346268af3a5cb7fe785b8fc7851b06"
PREREG_SHA = "38458f49a5568135d8d23be2e3cf1d62e3dee9e50cd9f9aec42283961a337d07"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA or sha(BUILDER) != BUILDER_SHA:
        raise RuntimeError("HVAANV frozen source implementation drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    row = {
        "AdrActCnt": "977386",
        "AssetEODCompletionTime": "1659614699",
        "CapMrktCurUSD": "435930163094.03583788655418767",
        "asset": "btc",
        "time": "2022-08-03T00:00:00.000000000Z",
    }
    core = {
        "protocol_version": "high_volatility_active_address_network_value_relay_source_rejection_v1",
        "policy_id": "HVAANV-24",
        "as_of_date": "2026-08-12",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"], "commit": "d3c50f64",
        },
        "frozen_source_builder": {"path": str(BUILDER), "sha256": BUILDER_SHA, "commit": "1929b79c"},
        "failed_contract": {
            "source": prereg.ENDPOINT,
            "operation": "strict chronological normalization of the frozen current-vintage daily response",
            "first_offending_response_index_zero_based": 214,
            "row": row,
            "row_canonical_hash": canonical_hash(row),
            "required_availability": "D+1 00:00 UTC < AssetEODCompletionTime <= D+1 12:00 UTC",
            "observation_time": "2022-08-03T00:00:00Z",
            "required_latest_completion": "2022-08-04T12:00:00Z",
            "observed_completion": "2022-08-04T12:04:59Z",
            "lateness_seconds": 299,
            "failure_class": "ValueError",
            "failure_message": "AssetEODCompletionTime must be after D+1 00:00 and no later than D+1 12:00 UTC",
            "first_failure_short_circuit": True,
        },
        "research_boundary": {
            "coin_metrics_first_response_page_opened": True,
            "response_rows_opened_before_failure": 215,
            "candidate_clock_or_support_incidence_computed": False,
            "btc_variation_rows_opened": False,
            "gross9_rows_opened": False,
            "execution_prices_opened": False,
            "funding_rows_opened": False,
            "postentry_return_or_pnl_opened": False,
        },
        "support_passed": False,
        "advance_to_gross9_novelty": False,
        "advance_to_economic_outcomes": False,
        "completion_window_repair_authorized": False,
        "repair_authorized": False,
        "decision": "terminal_source_contract_reject",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVAANV source rejection drift")


if __name__ == "__main__":
    report = build()
    validate(report)
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(RESULT)
