"""Freeze the seven-member pure-clock comparator cohort for LVRT-72."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


POLICY_ID = "LVRT-72"
PROTOCOL_VERSION = "lvrt_comparator_cohort_freeze_v1"
IMPLEMENTATION = Path("training/freeze_lvrt_comparator_cohort.py")
MECHANISM_DOCUMENT = Path(
    "docs/liquidity-vacuum-replenishment-transition-mechanism-decision-2026-07-21.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "9c2400a49b77a6e93594c65ae5bc8b17f6c676743a1fdfadf367979887dd77b9"
)
AFCS_CLOCK = Path("results/aggregate_fill_compression_sweep_clock_2026-07-17.csv")
AFCS_CLOCK_SHA256 = (
    "bf1611554604c1930ba2212e674ea434f7c9793377b3f33ef531b3b4e0381688"
)
BAFR_CLOCK = Path("results/binance_aggressor_frustration_clock_2026-07-20.csv")
BAFR_CLOCK_SHA256 = (
    "f3b816a76decce31136ed23d22f043eb8e80ef1b8697b869241b060062f01747"
)
MFIC_MANIFEST = Path("results/lvrt_mfic_pure_clock_manifest_2026-07-21.json")
MFIC_MANIFEST_SHA256 = (
    "e14911867d672ce6eda5e6ffac82352712e22d436bb05d4600b7c46a5275c72e"
)
MFIC_CLOCK = Path("results/lvrt_mfic_pure_clocks_2026-07-21.csv.gz")
MFIC_CLOCK_SHA256 = (
    "d7d889bd2c8137682e244d399a42f14e2c04c48e88a336e07d05e48ddb0605bd"
)
LIVE_MANIFEST = Path(
    "results/cchr_live_portfolio_pure_clock_manifest_2026-07-21.json"
)
LIVE_MANIFEST_SHA256 = (
    "6c53ae482cf72bba0f286a47626842bf43070276ff5fe359be718e44864af57d"
)
LIVE_CLOCK = Path("results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz")
LIVE_CLOCK_SHA256 = (
    "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08"
)
DEFAULT_OUTPUT = Path("results/lvrt_comparator_cohort_freeze_2026-07-21.json")
REQUIRED_MEMBERS = (
    "afcs:afcs_144",
    "bafr:primary",
    "live:cand_rex_veto_7",
    "live:new_long_minimal_funding_premium",
    "live:oi_upbit_ratio288_low",
    "mfic:mfic_fast",
    "mfic:mfic_slow",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be UTC")
    return parsed.astimezone(timezone.utc)


def _validate_member_rows(
    rows: Iterable[tuple[str, datetime, datetime, int]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[datetime, datetime, int]]] = {}
    for candidate_id, entry, exit_time, side in rows:
        if candidate_id not in REQUIRED_MEMBERS:
            raise ValueError(f"unexpected LVRT comparator member: {candidate_id}")
        if side not in {-1, 1} or entry >= exit_time:
            raise ValueError(f"invalid comparator clock: {candidate_id}")
        if entry.second or entry.microsecond or entry.minute % 5:
            raise ValueError(f"comparator entry left five-minute grid: {candidate_id}")
        if exit_time.second or exit_time.microsecond or exit_time.minute % 5:
            raise ValueError(f"comparator exit left five-minute grid: {candidate_id}")
        grouped.setdefault(candidate_id, []).append((entry, exit_time, side))

    summary: dict[str, dict[str, Any]] = {}
    for candidate_id, candidate_rows in grouped.items():
        ordered = sorted(candidate_rows, key=lambda row: row[0])
        if ordered != candidate_rows:
            raise ValueError(f"comparator rows are not chronological: {candidate_id}")
        if len({row[0] for row in ordered}) != len(ordered):
            raise ValueError(f"duplicate comparator entry: {candidate_id}")
        if any(current[0] < previous[1] for previous, current in zip(ordered, ordered[1:])):
            raise ValueError(f"overlapping comparator schedule: {candidate_id}")
        summary[candidate_id] = {
            "rows": len(ordered),
            "first_entry": ordered[0][0].isoformat(),
            "last_exit": ordered[-1][1].isoformat(),
            "year_counts": dict(sorted(Counter(str(row[0].year) for row in ordered).items())),
            "longs": sum(row[2] == 1 for row in ordered),
            "shorts": sum(row[2] == -1 for row in ordered),
        }
    if set(summary) != set(REQUIRED_MEMBERS):
        missing = sorted(set(REQUIRED_MEMBERS) - set(summary))
        raise ValueError(f"empty required LVRT comparators: {missing}")
    return dict(sorted(summary.items()))


def _load_afcs() -> list[tuple[str, datetime, datetime, int]]:
    rows: list[tuple[str, datetime, datetime, int]] = []
    with AFCS_CLOCK.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = [
            "origin_position",
            "signal_position",
            "entry_position",
            "exit_position",
            "origin_date",
            "signal_date",
            "entry_date",
            "exit_date",
            "side",
            "branch",
            "delay_bars",
            "hold_bars",
        ]
        if reader.fieldnames != expected:
            raise ValueError("AFCS comparator schema changed")
        for raw in reader:
            if raw["branch"] != "afcs_144" or int(raw["hold_bars"]) != 144:
                raise ValueError("AFCS comparator identity changed")
            rows.append(
                (
                    "afcs:afcs_144",
                    _timestamp(raw["entry_date"], field="AFCS entry_date"),
                    _timestamp(raw["exit_date"], field="AFCS exit_date"),
                    int(raw["side"]),
                )
            )
    if len(rows) != 573:
        raise ValueError("AFCS comparator row count changed")
    return rows


def _load_bafr() -> list[tuple[str, datetime, datetime, int]]:
    rows: list[tuple[str, datetime, datetime, int]] = []
    with BAFR_CLOCK.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = [
            "signal_position",
            "entry_position",
            "exit_position",
            "signal_date",
            "entry_date",
            "exit_date",
            "side",
            "hold_bars",
        ]
        if reader.fieldnames != expected:
            raise ValueError("BAFR comparator schema changed")
        for raw in reader:
            if int(raw["hold_bars"]) != 24:
                raise ValueError("BAFR comparator hold changed")
            rows.append(
                (
                    "bafr:primary",
                    _timestamp(raw["entry_date"], field="BAFR entry_date"),
                    _timestamp(raw["exit_date"], field="BAFR exit_date"),
                    int(raw["side"]),
                )
            )
    if len(rows) != 11248:
        raise ValueError("BAFR comparator row count changed")
    return rows


def _load_pure_clock(
    path: Path,
    *,
    expected_schema: list[str],
) -> list[tuple[str, datetime, datetime, int]]:
    rows: list[tuple[str, datetime, datetime, int]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_schema:
            raise ValueError(f"pure comparator schema changed: {path}")
        for raw in reader:
            rows.append(
                (
                    raw["candidate_id"],
                    _timestamp(raw["entry_time"], field="entry_time"),
                    _timestamp(raw["exit_time"], field="exit_time"),
                    int(raw["side"]),
                )
            )
    return rows


def build_freeze() -> dict[str, Any]:
    expected_hashes = {
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        AFCS_CLOCK: AFCS_CLOCK_SHA256,
        BAFR_CLOCK: BAFR_CLOCK_SHA256,
        MFIC_MANIFEST: MFIC_MANIFEST_SHA256,
        MFIC_CLOCK: MFIC_CLOCK_SHA256,
        LIVE_MANIFEST: LIVE_MANIFEST_SHA256,
        LIVE_CLOCK: LIVE_CLOCK_SHA256,
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"LVRT comparator binding changed: {path}")

    mfic_manifest = _load_json(MFIC_MANIFEST)
    if mfic_manifest.get("protocol_version") != "lvrt_mfic_pure_clock_export_v1":
        raise ValueError("MFIC pure-clock protocol changed")
    if mfic_manifest.get("clock", {}).get("sha256") != MFIC_CLOCK_SHA256:
        raise ValueError("MFIC pure-clock manifest binding changed")
    if mfic_manifest.get("outcome_boundary", {}).get("economic_outcomes_computed") is not False:
        raise ValueError("MFIC pure clock computed outcomes")

    live_manifest = _load_json(LIVE_MANIFEST)
    if live_manifest.get("protocol_version") != "cchr_pure_clock_export_manifest_v1":
        raise ValueError("live pure-clock protocol changed")
    if live_manifest.get("clock", {}).get("sha256") != LIVE_CLOCK_SHA256:
        raise ValueError("live pure-clock manifest binding changed")
    if live_manifest.get("outcomes_opened") is not False:
        raise ValueError("live pure-clock comparator opened outcomes")

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
    all_rows = [
        *_load_afcs(),
        *_load_bafr(),
        *_load_pure_clock(MFIC_CLOCK, expected_schema=mfic_schema),
        *_load_pure_clock(LIVE_CLOCK, expected_schema=live_schema),
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
        },
        "required_member_count": len(REQUIRED_MEMBERS),
        "required_members": list(REQUIRED_MEMBERS),
        "members": members,
        "all_required_members_available": True,
        "outcome_boundary": {
            "comparator_clock_rows_read": len(all_rows),
            "lvrt_source_rows_read": 0,
            "lvrt_incidence_rows_derived": 0,
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "performance_artifacts_parsed": 0,
            "return_or_pnl_fields_read": 0,
            "network_calls": 0,
            "economic_outcomes_computed": False,
        },
        "authorization": {
            "lvrt_source_only_incidence_after_this_artifact": True,
            "lvrt_market_outcomes": False,
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
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
