"""Audit the frozen HVGMR-24 GFZ nowcast-history contract without incidence or outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_geomagnetic_activity_rotation_relay as prereg


PREREG_SHA = "c56f8fd080fc286ea163c3e5ad6f4c857ffdd7427220320788acf3416b878989"
SCRIPT = Path("training/audit_high_volatility_geomagnetic_activity_rotation_source_contract.py")
RESULT = Path("results/high_volatility_geomagnetic_activity_rotation_relay_source_contract_failure_2026-08-12.json")
PROBE_START = "2023-07-01T00:00:00Z"
PROBE_END = "2023-07-03T23:59:59Z"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_status_contract(document: dict[str, Any]) -> dict[str, Any]:
    timestamps = document.get("datetime")
    values = document.get("Kp")
    statuses = document.get("status")
    if not isinstance(timestamps, list) or not isinstance(values, list) or not isinstance(statuses, list):
        raise RuntimeError("HVGMR GFZ response schema drift")
    if not timestamps or len(timestamps) != len(values) or len(values) != len(statuses):
        raise RuntimeError("HVGMR GFZ response cardinality drift")
    counts: dict[str, int] = {}
    for status in statuses:
        counts[str(status)] = counts.get(str(status), 0) + 1
    return {
        "rows": len(statuses),
        "status_counts": dict(sorted(counts.items())),
        "all_rows_preserve_preregistered_nowcast_status": all(status == "nowcast" for status in statuses),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVGMR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    query = urlencode({"start": PROBE_START, "end": PROBE_END, "index": "Kp"})
    url = prereg.GFZ_API + "?" + query
    request = Request(url, headers={"User-Agent": "rllm-hvgmr-source-contract/1.0"})
    with urlopen(request, timeout=60) as response:
        document = json.loads(response.read())
    audit = validate_status_contract(document)
    if audit["all_rows_preserve_preregistered_nowcast_status"]:
        raise RuntimeError("HVGMR probe unexpectedly preserves nowcast history; full builder required")
    core = {
        "protocol_version": "hvgmr_24_source_contract_failure_v1",
        "policy_id": "HVGMR-24",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_evaluator": {"path": str(SCRIPT), "sha256": sha(SCRIPT)},
        "failure_stage": "source_contract_validation",
        "probe": {"url": url, "window": [PROBE_START, PROBE_END], **audit},
        "failure": "preregistered archived nowcast-status Kp contract violated because the GFZ API currently substitutes definitive values for the historical probe",
        "causal_consequence": "the exact near-real-time Kp values available at the frozen D+1 12:00 decision cannot be reconstructed from this preregistered endpoint",
        "kp_numeric_values_published": False,
        "candidate_incidence_opened": False,
        "btc_rows_opened": False,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "support_passed": False,
        "advance_to_gross9_novelty": False,
        "advance_to_economic_outcomes": False,
        "decision": "terminal_source_contract_reject_no_repair",
        "repair_authorized": False,
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"decision": report["decision"], "status_counts": report["probe"]["status_counts"]}, indent=2))
