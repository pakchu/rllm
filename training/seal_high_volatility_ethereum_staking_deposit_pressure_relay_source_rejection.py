"""Seal the first fail-closed source-transport rejection for HVESDP-24."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_ethereum_staking_deposit_pressure_relay as prereg


RESULT = Path("results/high_volatility_ethereum_staking_deposit_pressure_relay_source_rejection_2026-08-12.json")
BUILDER = Path("training/build_high_volatility_ethereum_staking_deposit_pressure_relay_support.py")
BUILDER_SHA = "32da6db8820538f3c2800c980c1c9828b91241046a568851a01544910b952b6d"
BUILDER_COMMIT = "154748ed"
HELPER = Path("training/build_ethereum_stablecoin_issuance_redemption.py")
HELPER_SHA = "66936f283ecf8060c32d60767af504475329dfa556fb554596c2e27229f6c952"
PREREG_SHA = "3779ce7d9ebf120c92ed979e9d44747b9926c828967413a2cd646a39820dbeef"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVESDP preregistration drift")
    if sha(BUILDER) != BUILDER_SHA or sha(HELPER) != HELPER_SHA:
        raise RuntimeError("HVESDP frozen source implementation drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    core = {
        "protocol_version": "high_volatility_ethereum_staking_deposit_pressure_relay_source_rejection_v1",
        "policy_id": "HVESDP-24",
        "as_of_date": "2026-08-12",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
            "commit": "3c2d36c5",
        },
        "frozen_source_builder": {
            "path": str(BUILDER),
            "sha256": BUILDER_SHA,
            "commit": BUILDER_COMMIT,
        },
        "frozen_rpc_helper": {"path": str(HELPER), "sha256": HELPER_SHA},
        "failed_contract": {
            "transport_role": "primary",
            "rpc_url": prereg.PRIMARY_RPC,
            "method": "eth_getBlockByNumber",
            "operation": "binary search for frozen end-exclusive UTC boundary block",
            "required_interval_end_exclusive": "2026-07-30T00:00:00Z",
            "failure_class": "urllib.error.HTTPError",
            "http_status": 408,
            "failure_message": "HTTP Error 408: Request Timeout",
            "first_run_commit": "a7050424",
            "bounded_transport_correction_commit": "154748ed",
            "bounded_transport_correction": "HTTP 408 added to the existing finite transient retry set; source identity and semantics unchanged",
            "maximum_retries_after_correction": 6,
            "failure_recurred_after_all_retries": True,
            "first_failure_short_circuit": True,
        },
        "research_boundary": {
            "historical_boundary_headers_partially_opened": True,
            "historical_deposit_logs_opened": False,
            "source_incidence_opened": False,
            "btc_variation_rows_opened": False,
            "gross9_rows_opened": False,
            "execution_prices_opened": False,
            "funding_rows_opened": False,
            "postentry_return_or_pnl_opened": False,
        },
        "support_passed": False,
        "advance_to_gross9_novelty": False,
        "advance_to_economic_outcomes": False,
        "replacement_provider_authorized": False,
        "repair_authorized": False,
        "decision": "terminal_source_contract_reject",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVESDP source rejection drift")
    if value["research_boundary"]["historical_deposit_logs_opened"] is not False:
        raise RuntimeError("HVESDP source outcome boundary drift")


if __name__ == "__main__":
    report = build()
    validate(report)
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(RESULT)
