"""Seal the first fail-closed source violation observed for HVILFT-8."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_high_volatility_inverse_linear_funding_transfer_relay_support as support


RESULT = Path("results/high_volatility_inverse_linear_funding_transfer_relay_source_rejection_2026-08-10.json")
EVALUATOR = Path("training/build_high_volatility_inverse_linear_funding_transfer_relay_support.py")
EVALUATOR_SHA256 = "c6e7c1b3cb37012d34cfa0ae07f8284cbddc2c803e416dd23a9336f6bee32dc9"
EVALUATOR_COMMIT = "213bb613328267c53f9d47d3cf209264870f0436"
FIRST_ARCHIVE_URL = support.archive_url("cm", "BTCUSD_PERP", "2023-01")


def first_row(raw: bytes) -> dict[str, str]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise RuntimeError("HVILFT first archive member count drift")
        reader = csv.DictReader(io.StringIO(archive.read(names[0]).decode("utf-8")))
        if reader.fieldnames != ["calc_time", "funding_interval_hours", "last_funding_rate"]:
            raise RuntimeError("HVILFT first archive schema drift")
        row = next(reader, None)
    if row is None or any(not isinstance(value, str) for value in row.values()):
        raise RuntimeError("HVILFT first archive has no valid physical row")
    return row


def build_report(raw: bytes, binding: dict[str, Any]) -> dict[str, Any]:
    row = first_row(raw)
    if row != {
        "calc_time": "1672531200005",
        "funding_interval_hours": "8",
        "last_funding_rate": "-0.00001609",
    }:
        raise RuntimeError("HVILFT observed first-failure row drift")
    timestamp = pd.Timestamp(int(row["calc_time"]), unit="ms", tz="UTC")
    floor = timestamp.floor("5min")
    if timestamp == floor:
        raise RuntimeError("HVILFT expected off-boundary source violation disappeared")
    if support.sha(EVALUATOR) != EVALUATOR_SHA256:
        raise RuntimeError("HVILFT frozen source evaluator drift")
    if support.sha(support.prereg.DEFAULT_OUTPUT) != support.PREREG_SHA256:
        raise RuntimeError("HVILFT preregistration drift")
    report: dict[str, Any] = {
        "protocol_version": "high_volatility_inverse_linear_funding_transfer_relay_source_rejection_v1",
        "policy_id": "HVILFT-8",
        "as_of_date": "2026-08-10",
        "preregistration": {
            "path": str(support.prereg.DEFAULT_OUTPUT),
            "sha256": support.PREREG_SHA256,
            "manifest_hash": support.prereg.build()["manifest_hash"],
        },
        "frozen_evaluator": {
            "path": str(EVALUATOR), "sha256": EVALUATOR_SHA256, "commit": EVALUATOR_COMMIT,
        },
        "source_binding": {
            **binding,
            "instrument": "BTCUSD_PERP", "month": "2023-01",
            "first_physical_row_number": 2,
            "first_physical_row": row,
        },
        "failed_contract": {
            "requirement": "calc_time must be on an exact five-minute boundary",
            "observed_calc_time": timestamp.isoformat(),
            "required_floor": floor.isoformat(),
            "offset_milliseconds": int((timestamp - floor) / pd.Timedelta(milliseconds=1)),
            "failure_class": "ValueError",
            "failure_message": "funding archive row value invalid",
            "first_failure_short_circuit": True,
        },
        "access_boundary": {
            "funding_source_values_opened": True,
            "archives_opened": 1,
            "source_rows_read_for_seal": 1,
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
            "reason": "the first official COIN-M archive row violates the preregistered exact-boundary row contract",
            "source_support_authorized": False,
            "gross9_novelty_authorized": False,
            "economics_authorized": False,
            "repair_authorized": False,
            "next_action": "new independently preregistered alpha only",
        },
    }
    report["manifest_hash"] = support.canonical_hash(report)
    return report


def run() -> dict[str, Any]:
    if RESULT.exists():
        raise FileExistsError("HVILFT source rejection artifact is immutable")
    raw, binding = support.verified_archive(FIRST_ARCHIVE_URL)
    report = build_report(raw, binding)
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
