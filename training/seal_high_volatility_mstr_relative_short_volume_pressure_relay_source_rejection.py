"""Seal HVMRSVP-24's first fail-closed FINRA transport violation."""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from training import build_high_volatility_mstr_relative_short_volume_pressure_relay_support as support


RESULT = Path(
    "results/high_volatility_mstr_relative_short_volume_pressure_relay_source_rejection_2026-08-10.json"
)
EVALUATOR = Path(
    "training/build_high_volatility_mstr_relative_short_volume_pressure_relay_support.py"
)
EVALUATOR_SHA256 = "4483b04fd56b8340c327460ea05a07eb46e8964cf389eba729d8cc466f45811e"
EVALUATOR_COMMIT = "3a838abc026343ba92ba2549565c75588d3e79bf"
FIRST_URL = support.TEMPLATE.format(date="20230101")


def fetch_first_response() -> tuple[int, bytes, dict[str, str | None]]:
    request = urllib.request.Request(
        FIRST_URL,
        headers={"Accept-Encoding": "identity", "User-Agent": "rllm-source-audit/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read(), {
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
            }
    except urllib.error.HTTPError as error:
        return error.code, error.read(), {
            "etag": error.headers.get("ETag"),
            "last_modified": error.headers.get("Last-Modified"),
        }


def build_report(status: int, raw: bytes, headers: dict[str, str | None]) -> dict[str, Any]:
    if status != 403:
        raise RuntimeError(f"HVMRSVP observed first-failure status drift: {status}")
    if support.sha(EVALUATOR) != EVALUATOR_SHA256:
        raise RuntimeError("HVMRSVP frozen source evaluator drift")
    if support.sha(support.prereg.DEFAULT_OUTPUT) != support.PREREG_SHA256:
        raise RuntimeError("HVMRSVP preregistration drift")
    core: dict[str, Any] = {
        "protocol_version": "hvmrsvp_24_source_rejection_v1",
        "policy_id": "HVMRSVP-24",
        "as_of_date": "2026-08-10",
        "preregistration": {
            "path": str(support.prereg.DEFAULT_OUTPUT),
            "sha256": support.PREREG_SHA256,
            "manifest_hash": support.prereg.build()["manifest_hash"],
        },
        "frozen_evaluator": {
            "path": str(EVALUATOR),
            "sha256": EVALUATOR_SHA256,
            "commit": EVALUATOR_COMMIT,
        },
        "source_binding": {
            "url": FIRST_URL,
            "requested_date": "2023-01-01",
            "http_status": status,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "response_bytes": len(raw),
            **headers,
        },
        "failed_contract": {
            "requirement": "every requested calendar date must return HTTP 200 or HTTP 404",
            "observed_status": status,
            "failure_class": "RuntimeError",
            "failure_message": f"FINRA fail-closed HTTP status 403 for {FIRST_URL}",
            "first_failure_short_circuit": True,
        },
        "access_boundary": {
            "finra_http_responses_opened": 1,
            "finra_target_rows_opened": 0,
            "candidate_incidence_derived": False,
            "postgres_connected": False,
            "market_rows_loaded": 0,
            "gross9_rows_opened": False,
            "return_or_pnl_fields_opened": False,
            "economic_outcomes_opened": False,
        },
        "decision": {
            "pass": False,
            "status": "terminal_source_contract_rejection",
            "reason": "the first requested FINRA calendar date returned forbidden HTTP 403",
            "source_support_authorized": False,
            "gross9_novelty_authorized": False,
            "economics_authorized": False,
            "repair_authorized": False,
            "next_action": "new independently preregistered alpha only",
        },
    }
    return {**core, "manifest_hash": support.canonical_hash(core)}


def run() -> dict[str, Any]:
    if RESULT.exists():
        raise FileExistsError("HVMRSVP source rejection artifact is immutable")
    report = build_report(*fetch_first_response())
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
