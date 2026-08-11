"""Seal the terminal HVEMSD-24 source transport failure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OUTPUT = Path(
    "results/high_volatility_ethereum_missed_slot_delta_relay_source_failure_2026-08-12.json"
)
PREREG = Path(
    "results/high_volatility_ethereum_missed_slot_delta_relay_preregistration_2026-08-12.json"
)
BUILDER = Path("training/build_high_volatility_ethereum_missed_slot_delta_relay_support.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "hvemsd_24_source_transport_failure_v1",
        "policy_id": "HVEMSD-24",
        "status": "terminal_source_reject_no_repair",
        "preregistration": {"path": str(PREREG), "sha256": sha(PREREG)},
        "source_evaluator": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "failure": {
            "stage": "dual_rpc_historical_boundary_header_collection",
            "host": "https://eth-mainnet.public.blastapi.io",
            "rpc_method": "eth_getBlockByNumber",
            "exception_type": "HTTPError",
            "http_status": 429,
            "message": "Too Many Requests after the frozen four-attempt transport contract",
        },
        "opened_before_failure": {
            "partial_ethereum_header_responses_in_memory": True,
            "complete_daily_boundary_panel": False,
            "daily_missed_slot_counts_or_changes": False,
            "candidate_incidence": False,
            "btc_preentry_variation": False,
            "gross9_rows": False,
            "execution_price": False,
            "postentry_return_or_pnl": False,
            "funding": False,
        },
        "durable_source_outputs_published": False,
        "repair_forbidden": [
            "change or add an RPC host",
            "change request cadence, retry count, retry delay, batching, date range or source method",
            "reuse partial headers to resume",
            "replace execution headers with an explorer, chart, beacon endpoint or prior local source",
            "change threshold, side, hold, clock, subset or control",
        ],
        "next_action": "leave the Ethereum missed-slot source axis and preregister an independent mechanism",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    value = build()
    OUTPUT.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(OUTPUT)
