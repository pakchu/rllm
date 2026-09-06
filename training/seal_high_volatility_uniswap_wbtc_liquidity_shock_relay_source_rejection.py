"""Seal the first frozen source-contract rejection for HVUWLS-24."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_uniswap_wbtc_liquidity_shock_relay as prereg


RESULT = Path(
    "results/high_volatility_uniswap_wbtc_liquidity_shock_relay_source_rejection_2026-08-12.json"
)
BUILDER = Path("training/build_high_volatility_uniswap_wbtc_liquidity_shock_relay_support.py")
BUILDER_SHA = "b84dbae2d75150e22d7066fe434b2e759de400afb6b4d5111e1139089009b319"
BUILDER_COMMIT = "89b06572"
HELPER = Path("training/build_ethereum_stablecoin_issuance_redemption.py")
HELPER_SHA = "66936f283ecf8060c32d60767af504475329dfa556fb554596c2e27229f6c952"
PREREG_SOURCE_SHA = "b342449f28512c79b0204070ef70d1b91b2aff1e11d47386cc86bcda3883ff5c"
PREREG_ARTIFACT_SHA = "3761bc07e309e4d9f156cba286c385a5a016df6fcbd99ac4d21bd1c3b85fc9ea"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    prereg_source = Path(prereg.__file__)
    if sha(prereg_source) != PREREG_SOURCE_SHA or sha(prereg.DEFAULT_OUTPUT) != PREREG_ARTIFACT_SHA:
        raise RuntimeError("HVUWLS preregistration drift")
    if sha(BUILDER) != BUILDER_SHA or sha(HELPER) != HELPER_SHA:
        raise RuntimeError("HVUWLS frozen source implementation drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    failed_response = {
        "jsonrpc": "2.0",
        "error": {
            "code": -32001,
            "message": (
                "You've reached the usage limit for your current plan. To continue with higher limits and "
                "uninterrupted access, please upgrade here: https://www.1rpc.io/#pricing"
            ),
        },
        "id": None,
    }
    core = {
        "protocol_version": "high_volatility_uniswap_wbtc_liquidity_shock_relay_source_rejection_v1",
        "policy_id": "HVUWLS-24",
        "as_of_date": "2026-08-12",
        "preregistration": {
            "source_path": str(prereg_source.relative_to(Path.cwd())),
            "source_sha256": PREREG_SOURCE_SHA,
            "artifact_path": str(prereg.DEFAULT_OUTPUT),
            "artifact_sha256": PREREG_ARTIFACT_SHA,
            "manifest_hash": registration["manifest_hash"],
            "commit": "c40a57c7",
        },
        "frozen_source_builder": {
            "path": str(BUILDER),
            "sha256": BUILDER_SHA,
            "commit": BUILDER_COMMIT,
        },
        "frozen_rpc_helper": {"path": str(HELPER), "sha256": HELPER_SHA},
        "failed_contract": {
            "transport_role": "verification",
            "rpc_url": prereg.VERIFY_RPC,
            "method": "eth_chainId",
            "operation": "frozen role-to-host source-contract preflight before historical Swap replay",
            "request": {"jsonrpc": "2.0", "id": 1, "method": "eth_chainId", "params": []},
            "http_status": 200,
            "response": failed_response,
            "response_canonical_hash": canonical_hash(failed_response),
            "required_contract": "JSON-RPC response id must exactly equal request id before result/error handling",
            "observed_violation": "request id 1 received response id null with provider usage-limit error -32001",
            "failure_class": "RuntimeError",
            "failure_message": "Ethereum RPC returned a malformed response identity",
            "first_failure_short_circuit": True,
        },
        "bounded_transport_history": {
            "initial_builder_commit": "50c27dd7",
            "exact_boundary_batch_commit": "369bf3d7",
            "two_role_preflight_commit": BUILDER_COMMIT,
            "semantic_or_source_identity_change": False,
            "historical_incidence_used_for_correction": False,
        },
        "research_boundary": {
            "primary_historical_boundary_headers_partially_opened_during_interrupted_transport_runs": True,
            "historical_uniswap_swap_logs_opened": False,
            "candidate_source_incidence_opened": False,
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
        raise RuntimeError("HVUWLS source rejection drift")
    if value["research_boundary"]["historical_uniswap_swap_logs_opened"] is not False:
        raise RuntimeError("HVUWLS source outcome boundary drift")


if __name__ == "__main__":
    report = build()
    validate(report)
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(RESULT)
