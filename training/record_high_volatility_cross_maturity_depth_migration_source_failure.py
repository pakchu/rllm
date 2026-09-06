"""Seal the first genuine HVCMDM-8 source-contract failure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OUTPUT = Path(
    "results/high_volatility_cross_maturity_depth_migration_relay_source_contract_failure_2026-08-10.json"
)
PREREG = Path(
    "results/high_volatility_cross_maturity_depth_migration_relay_preregistration_2026-08-10.json"
)
EVALUATOR = Path(
    "training/build_high_volatility_cross_maturity_depth_migration_relay_support.py"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "hvcmdm_8_terminal_source_contract_failure_v1",
        "policy_id": "HVCMDM-8",
        "as_of_date": "2026-08-10",
        "preregistration": {
            "path": str(PREREG),
            "sha256": sha(PREREG),
            "manifest_hash": json.loads(PREREG.read_text())["manifest_hash"],
        },
        "source_evaluator": {"path": str(EVALUATOR), "sha256": sha(EVALUATOR)},
        "failure": {
            "stage": "required_official_archive_acquisition",
            "reason": "required_contract_day_checksum_missing",
            "http_status": 404,
            "contract": "BTCUSD_230929",
            "day": "2023-09-25",
            "required_checksum_url": (
                "https://data.binance.vision/data/futures/cm/daily/bookDepth/"
                "BTCUSD_230929/BTCUSD_230929-bookDepth-2023-09-25.zip.CHECKSUM"
            ),
            "required_zip_url": (
                "https://data.binance.vision/data/futures/cm/daily/bookDepth/"
                "BTCUSD_230929/BTCUSD_230929-bookDepth-2023-09-25.zip"
            ),
            "contract_requirement": "every_required_contract_day_checksum_verified",
            "repair_forbidden": (
                "the missing day may not be filled, interpolated, replaced by a different maturity, "
                "or removed from the candidate window"
            ),
        },
        "attempt_audit": {
            "first_attempt": "pre-output implementation mismatch; malformed snapshot was incorrectly escalated from snapshot reject to archive reject",
            "first_attempt_candidate_incidence_created": False,
            "first_attempt_gross9_or_outcomes_opened": False,
            "implementation_correction": "aa3402e7",
            "second_attempt": "first genuine frozen source-contract evaluation",
            "second_attempt_candidate_incidence_created": False,
            "second_attempt_gross9_or_outcomes_opened": False,
        },
        "output_boundary": {
            "source_support_artifact_created": False,
            "candidate_clock_created": False,
            "candidate_incidence_opened": False,
            "gross9_rows_opened": False,
            "postentry_execution_price_rows_opened": False,
            "return_pnl_cagr_mdd_opened": False,
        },
        "support_passed": False,
        "advance_to_gross9_novelty": False,
        "advance_to_economic_outcomes": False,
        "decision": "terminal_source_contract_reject_no_repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    result = build()
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(OUTPUT)
