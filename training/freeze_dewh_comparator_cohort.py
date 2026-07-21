"""Freeze the eight-member pure-clock comparator cohort for DEWH-144."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from training.freeze_lvrt_comparator_cohort import (
    AFCS_CLOCK,
    AFCS_CLOCK_SHA256,
    BAFR_CLOCK,
    BAFR_CLOCK_SHA256,
    LIVE_CLOCK,
    LIVE_CLOCK_SHA256,
    LIVE_MANIFEST,
    LIVE_MANIFEST_SHA256,
    MFIC_CLOCK,
    MFIC_CLOCK_SHA256,
    MFIC_MANIFEST,
    MFIC_MANIFEST_SHA256,
    _load_afcs,
    _load_bafr,
    _load_json,
    _load_pure_clock,
    canonical_hash,
    sha256_file,
)


POLICY_ID = "DEWH-144"
PROTOCOL_VERSION = "dewh_comparator_cohort_freeze_v1"
IMPLEMENTATION = Path("training/freeze_dewh_comparator_cohort.py")
MECHANISM_DOCUMENT = Path(
    "docs/deribit-expiry-wall-handoff-mechanism-decision-2026-07-21.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "f5b378b75e3d32c18e32b245f62674a7a7b25f90ec7761d865ddb6c627a93ce8"
)
DEWH_SOURCE = Path(
    "data/deribit_btc_expiry_wall_2019_2023/"
    "BTC_deribit_expiry_wall_2019-01-01_2023-12-31.csv.gz"
)
DEWH_SOURCE_SHA256 = "53e8c829d8dd49eb669218067409a1b5175900c88fd75652c0ad420f6b6167f5"
DEWH_SOURCE_MANIFEST = Path(
    "data/deribit_btc_expiry_wall_2019_2023/build_manifest.json"
)
DEWH_SOURCE_MANIFEST_SHA256 = (
    "dde10a20d6efc3026be253daefe88bff0bae4ba379deaa90b4d08431dd741c36"
)
DEWH_SOURCE_MANIFEST_HASH = (
    "dbecf89849c356e4b5900600e2727ad5f972c9a62487b8d897407a0e22da104e"
)
DEHR_MANIFEST = Path("results/dewh_dehr_comparator_clock_manifest_2026-07-21.json")
DEHR_MANIFEST_SHA256 = (
    "0b972dd106c4013e4229ed72b5fc78039862dd4611d285f48a2407dd5a2fa2bb"
)
DEHR_MANIFEST_HASH = "703fc475d80128390f71b619bb2169aabcb9d0fc756f2be862627d164a9d102f"
DEHR_CLOCK = Path("results/dewh_dehr_comparator_clocks_2026-07-21.csv.gz")
DEHR_CLOCK_SHA256 = "d3072326da413a6d28a00827876ac4295c6d8c2ea65766bc5ec42112a192aae9"
DEFAULT_OUTPUT = Path("results/dewh_comparator_cohort_freeze_2026-07-21.json")
REQUIRED_MEMBERS = (
    "afcs:afcs_144",
    "bafr:primary",
    "dehr:dehr_72_normalized",
    "live:cand_rex_veto_7",
    "live:new_long_minimal_funding_premium",
    "live:oi_upbit_ratio288_low",
    "mfic:mfic_fast",
    "mfic:mfic_slow",
)


def _validate_member_rows(
    rows: Iterable[tuple[str, datetime, datetime, int]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[datetime, datetime, int]]] = {}
    for candidate_id, entry, exit_time, side in rows:
        if candidate_id not in REQUIRED_MEMBERS:
            raise ValueError(f"unexpected DEWH comparator member: {candidate_id}")
        if side not in {-1, 1} or entry >= exit_time:
            raise ValueError(f"invalid DEWH comparator clock: {candidate_id}")
        for label, value in (("entry", entry), ("exit", exit_time)):
            if value.utcoffset() != timedelta(0):
                raise ValueError(f"{candidate_id} {label} is not UTC")
            if value.second or value.microsecond or value.minute % 5:
                raise ValueError(f"{candidate_id} {label} left the five-minute grid")
        grouped.setdefault(candidate_id, []).append((entry, exit_time, side))

    summary: dict[str, dict[str, Any]] = {}
    for candidate_id, candidate_rows in grouped.items():
        ordered = sorted(candidate_rows, key=lambda row: row[0])
        if ordered != candidate_rows:
            raise ValueError(f"comparator rows are not chronological: {candidate_id}")
        if len({row[0] for row in ordered}) != len(ordered):
            raise ValueError(f"duplicate comparator entry: {candidate_id}")
        if any(
            current[0] < previous[1] for previous, current in zip(ordered, ordered[1:])
        ):
            raise ValueError(f"overlapping comparator schedule: {candidate_id}")
        summary[candidate_id] = {
            "rows": len(ordered),
            "first_entry": ordered[0][0].isoformat(),
            "last_exit": ordered[-1][1].isoformat(),
            "year_counts": dict(
                sorted(Counter(str(row[0].year) for row in ordered).items())
            ),
            "longs": sum(row[2] == 1 for row in ordered),
            "shorts": sum(row[2] == -1 for row in ordered),
        }
    if set(summary) != set(REQUIRED_MEMBERS):
        missing = sorted(set(REQUIRED_MEMBERS) - set(summary))
        raise ValueError(f"empty required DEWH comparators: {missing}")
    return dict(sorted(summary.items()))


def build_freeze() -> dict[str, Any]:
    expected_hashes = {
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        DEWH_SOURCE: DEWH_SOURCE_SHA256,
        DEWH_SOURCE_MANIFEST: DEWH_SOURCE_MANIFEST_SHA256,
        AFCS_CLOCK: AFCS_CLOCK_SHA256,
        BAFR_CLOCK: BAFR_CLOCK_SHA256,
        MFIC_MANIFEST: MFIC_MANIFEST_SHA256,
        MFIC_CLOCK: MFIC_CLOCK_SHA256,
        LIVE_MANIFEST: LIVE_MANIFEST_SHA256,
        LIVE_CLOCK: LIVE_CLOCK_SHA256,
        DEHR_MANIFEST: DEHR_MANIFEST_SHA256,
        DEHR_CLOCK: DEHR_CLOCK_SHA256,
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"DEWH comparator binding changed: {path}")

    source_manifest = _load_json(DEWH_SOURCE_MANIFEST)
    if source_manifest.get("manifest_hash") != DEWH_SOURCE_MANIFEST_HASH:
        raise ValueError("DEWH source manifest hash changed")
    if source_manifest.get("candidate_incidence_computed") is not False:
        raise ValueError("DEWH source artifact opened candidate incidence")
    if (
        source_manifest.get("outcome_boundary", {}).get("economic_outcomes_computed")
        is not False
    ):
        raise ValueError("DEWH source artifact computed outcomes")

    mfic_manifest = _load_json(MFIC_MANIFEST)
    if mfic_manifest.get("clock", {}).get("sha256") != MFIC_CLOCK_SHA256:
        raise ValueError("MFIC pure-clock manifest binding changed")
    if (
        mfic_manifest.get("outcome_boundary", {}).get("economic_outcomes_computed")
        is not False
    ):
        raise ValueError("MFIC comparator computed outcomes")

    live_manifest = _load_json(LIVE_MANIFEST)
    if live_manifest.get("clock", {}).get("sha256") != LIVE_CLOCK_SHA256:
        raise ValueError("live pure-clock manifest binding changed")
    if live_manifest.get("outcomes_opened") is not False:
        raise ValueError("live pure-clock comparator opened outcomes")

    dehr_manifest = _load_json(DEHR_MANIFEST)
    if dehr_manifest.get("manifest_hash") != DEHR_MANIFEST_HASH:
        raise ValueError("normalized DEHR comparator manifest hash changed")
    if dehr_manifest.get("clock", {}).get("sha256") != DEHR_CLOCK_SHA256:
        raise ValueError("normalized DEHR comparator clock binding changed")
    if dehr_manifest.get("normalization", {}).get("selection_changed") is not False:
        raise ValueError("normalized DEHR comparator changed selection")
    if dehr_manifest.get("normalization", {}).get("side_changed") is not False:
        raise ValueError("normalized DEHR comparator changed side")
    if (
        dehr_manifest.get("outcome_boundary", {}).get("economic_outcomes_computed")
        is not False
    ):
        raise ValueError("normalized DEHR comparator computed outcomes")

    mfic_schema = [
        "candidate_id",
        "split",
        "causal_origin",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
    ]
    live_schema = [
        "candidate_id",
        "split",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
    ]
    dehr_schema = [
        "candidate_id",
        "causal_origin",
        "decision_time",
        "original_entry_time",
        "entry_time",
        "exit_time",
        "side",
    ]
    all_rows = [
        *_load_afcs(),
        *_load_bafr(),
        *_load_pure_clock(MFIC_CLOCK, expected_schema=mfic_schema),
        *_load_pure_clock(LIVE_CLOCK, expected_schema=live_schema),
        *_load_pure_clock(DEHR_CLOCK, expected_schema=dehr_schema),
    ]
    members = _validate_member_rows(all_rows)
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": "2026-07-21",
        "implementation_binding": {
            "path": str(IMPLEMENTATION),
            "sha256": sha256_file(IMPLEMENTATION),
        },
        "decision_binding": {
            "path": str(MECHANISM_DOCUMENT),
            "sha256": MECHANISM_DOCUMENT_SHA256,
        },
        "source_binding": {
            "path": str(DEWH_SOURCE),
            "sha256": DEWH_SOURCE_SHA256,
            "manifest": str(DEWH_SOURCE_MANIFEST),
            "manifest_sha256": DEWH_SOURCE_MANIFEST_SHA256,
            "manifest_hash": DEWH_SOURCE_MANIFEST_HASH,
        },
        "clock_bindings": {
            "afcs": {"path": str(AFCS_CLOCK), "sha256": AFCS_CLOCK_SHA256},
            "bafr": {"path": str(BAFR_CLOCK), "sha256": BAFR_CLOCK_SHA256},
            "mfic_manifest": {
                "path": str(MFIC_MANIFEST),
                "sha256": MFIC_MANIFEST_SHA256,
                "manifest_hash": mfic_manifest["manifest_hash"],
            },
            "mfic_clock": {"path": str(MFIC_CLOCK), "sha256": MFIC_CLOCK_SHA256},
            "live_manifest": {
                "path": str(LIVE_MANIFEST),
                "sha256": LIVE_MANIFEST_SHA256,
                "manifest_hash": live_manifest["manifest_hash"],
            },
            "live_clock": {"path": str(LIVE_CLOCK), "sha256": LIVE_CLOCK_SHA256},
            "dehr_manifest": {
                "path": str(DEHR_MANIFEST),
                "sha256": DEHR_MANIFEST_SHA256,
                "manifest_hash": DEHR_MANIFEST_HASH,
            },
            "dehr_clock": {"path": str(DEHR_CLOCK), "sha256": DEHR_CLOCK_SHA256},
        },
        "required_member_count": len(REQUIRED_MEMBERS),
        "required_members": list(REQUIRED_MEMBERS),
        "members": members,
        "all_required_members_available": True,
        "outcome_boundary": {
            "comparator_clock_rows_read": len(all_rows),
            "dewh_source_rows_read": 0,
            "dewh_incidence_rows_derived": 0,
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "performance_artifacts_parsed": 0,
            "return_or_pnl_fields_read": 0,
            "network_calls": 0,
            "economic_outcomes_computed": False,
        },
        "authorization": {
            "dewh_source_only_incidence_after_this_artifact": True,
            "dewh_market_outcomes": False,
        },
    }
    core["manifest_hash"] = canonical_hash(core)
    return {**core, "created_at": datetime.now(timezone.utc).isoformat()}


def publish(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = build_freeze()
    publish(DEFAULT_OUTPUT, report)
    print(
        json.dumps(
            {
                "output": str(DEFAULT_OUTPUT),
                "manifest_hash": report["manifest_hash"],
                "required_members": report["required_member_count"],
                "comparator_rows": report["outcome_boundary"][
                    "comparator_clock_rows_read"
                ],
                "dewh_incidence_rows_derived": report["outcome_boundary"][
                    "dewh_incidence_rows_derived"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
