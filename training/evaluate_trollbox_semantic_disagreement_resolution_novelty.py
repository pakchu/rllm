"""Export TSDR-72 pure clocks and run its frozen comparator novelty gate."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd

import training.evaluate_bitmex_trollbox_attention_saturation as tbasr
from training.evaluate_trollbox_semantic_disagreement_resolution_support import (
    POLICY_ID,
    SELECTION_END,
    TRAIN_END,
    TRAIN_START,
    build_primary_candidates,
    canonical_hash,
    load_json,
    load_semantic_events,
    schedule_candidates,
    sha256_file,
)


PROTOCOL_VERSION = "trollbox_semantic_disagreement_resolution_novelty_v1"
BAR = timedelta(minutes=5)
TOLERANCE = timedelta(hours=6)
IMPLEMENTATION = Path(
    "training/evaluate_trollbox_semantic_disagreement_resolution_novelty.py"
)
SUPPORT_ARTIFACT = Path(
    "results/trollbox_semantic_disagreement_resolution_support_2026-07-21.json"
)
SUPPORT_ARTIFACT_SHA256 = (
    "e4da84347ba903c14e479b24754029352a7b7913eac32061a1820bd9428d660e"
)
SUPPORT_RESULT_HASH = (
    "475633a3262ee46bee30b56c6accdb71dc5ce61b40e0ba575b224cdb0ea71589"
)
SUPPORT_IMPLEMENTATION_SHA256 = (
    "a76cc643e1f0da56c726f02efa5fc4244b50df87d7000715ab9771e8ffe9dee3"
)
TBASR_IMPLEMENTATION = Path(
    "training/evaluate_bitmex_trollbox_attention_saturation.py"
)
TBASR_IMPLEMENTATION_SHA256 = (
    "d32055317913bd80b00d0115bb0d5f26fa70b9f7d456d3718852e535a70ff193"
)
TBASR_FREEZE = Path(
    "results/bitmex_trollbox_attention_saturation_evaluator_freeze_2026-07-20.json"
)
TBASR_FREEZE_SHA256 = (
    "36dde44985f26896fcc6ef861dc3a45c81479915038e5ea537cc2582f0b3b45a"
)
LIVE_CLOCK_MANIFEST = Path(
    "results/cchr_live_portfolio_pure_clock_manifest_2026-07-21.json"
)
LIVE_CLOCK_MANIFEST_SHA256 = (
    "6c53ae482cf72bba0f286a47626842bf43070276ff5fe359be718e44864af57d"
)
LIVE_CLOCK = Path("results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz")
LIVE_CLOCK_SHA256 = (
    "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08"
)
DEFAULT_CLOCK_OUTPUT = Path("results/tsdr_pure_clocks_2026-07-21.csv.gz")
DEFAULT_REPORT_OUTPUT = Path(
    "results/trollbox_semantic_disagreement_resolution_novelty_2026-07-21.json"
)
PURE_CLOCK_FIELDS = (
    "candidate_id",
    "split",
    "causal_origin",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
)


@dataclass(frozen=True)
class ClockRow:
    candidate_id: str
    split: str
    causal_origin: datetime
    decision_time: datetime
    entry_time: datetime
    exit_time: datetime
    side: int


def _timestamp(value: Any, *, field: str) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid {field}: {value!r}") from exc
    else:
        raise ValueError(f"{field} must be a timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_clock_rows(rows: Sequence[ClockRow], *, label: str) -> None:
    by_candidate: dict[str, list[ClockRow]] = {}
    for row in rows:
        if row.side not in {-1, 1}:
            raise ValueError(f"{label} side changed")
        if not (
            row.causal_origin <= row.decision_time <= row.entry_time < row.exit_time
        ):
            raise ValueError(f"{label} causal clock order changed")
        if row.entry_time.microsecond or row.entry_time.second:
            raise ValueError(f"{label} entry left the minute grid")
        if row.entry_time.minute % 5:
            raise ValueError(f"{label} entry left the five-minute grid")
        by_candidate.setdefault(row.candidate_id, []).append(row)
    for candidate_id, candidate_rows in by_candidate.items():
        ordered = sorted(candidate_rows, key=lambda row: row.entry_time)
        if ordered != candidate_rows:
            raise ValueError(f"{label} {candidate_id} rows are not chronological")
        if len({row.entry_time for row in ordered}) != len(ordered):
            raise ValueError(f"{label} {candidate_id} has duplicate entries")
        if any(
            current.entry_time < previous.exit_time
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise ValueError(f"{label} {candidate_id} schedule overlaps")


def _load_support() -> dict[str, Any]:
    if sha256_file(SUPPORT_ARTIFACT) != SUPPORT_ARTIFACT_SHA256:
        raise ValueError("TSDR-72 support artifact changed")
    if sha256_file(
        Path("training/evaluate_trollbox_semantic_disagreement_resolution_support.py")
    ) != SUPPORT_IMPLEMENTATION_SHA256:
        raise ValueError("TSDR-72 support implementation changed")
    support = load_json(SUPPORT_ARTIFACT)
    if support.get("protocol_version") != (
        "trollbox_semantic_disagreement_resolution_support_v1"
    ):
        raise ValueError("TSDR-72 support protocol changed")
    if support.get("result_hash") != SUPPORT_RESULT_HASH:
        raise ValueError("TSDR-72 support result changed")
    gate = support.get("support_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("TSDR-72 source support did not pass")
    boundary = support.get("outcome_boundary")
    if not isinstance(boundary, dict) or boundary.get("outcomes_opened") is not False:
        raise ValueError("TSDR-72 support opened outcomes")
    return support


def _build_tsdr_clocks(support: Mapping[str, Any]) -> list[ClockRow]:
    events, _ = load_semantic_events()
    raw, _ = build_primary_candidates(events)
    accepted, _ = schedule_candidates(raw)
    rows = [
        ClockRow(
            candidate_id="tsdr:primary",
            split=cast(str, row.split),
            causal_origin=row.onset_end,
            decision_time=row.resolution_end,
            entry_time=row.entry,
            exit_time=row.exit,
            side=row.side,
        )
        for row in accepted
    ]
    _validate_clock_rows(rows, label="TSDR-72")
    serialized = [
        {
            "split": row.split,
            "onset_end": row.causal_origin.isoformat(),
            "resolution_end": row.decision_time.isoformat(),
            "entry": row.entry_time.isoformat(),
            "exit": row.exit_time.isoformat(),
            "side": row.side,
        }
        for row in rows
    ]
    if canonical_hash(serialized) != support["primary"]["clock_hash"]:
        raise ValueError("TSDR-72 support clock commitment changed")
    expected = sum(
        support["primary"]["splits"][name]["accepted_events"]
        for name in ("train", "selection")
    )
    if len(rows) != expected:
        raise ValueError("TSDR-72 support incidence changed")
    return rows


def _build_tbasr_train_clocks() -> tuple[list[ClockRow], dict[str, Any]]:
    if sha256_file(TBASR_IMPLEMENTATION) != TBASR_IMPLEMENTATION_SHA256:
        raise ValueError("TBASR comparator implementation changed")
    if sha256_file(TBASR_FREEZE) != TBASR_FREEZE_SHA256:
        raise ValueError("TBASR comparator freeze changed")
    freeze = tbasr.verify_evaluator_freeze()
    contracts = freeze["source_contracts"]["stage_market_months"]["train"]
    market, market_diagnostics = tbasr._parse_market_months(
        contracts,
        end=cast(pd.Timestamp, tbasr.STAGE_WINDOWS["train"][1]),
    )
    schedules, _, incidence = tbasr.build_stage_schedules(market, "train")
    primary = schedules[tbasr.PRIMARY]
    rows = [
        ClockRow(
            candidate_id="tbasr:primary",
            split="train",
            causal_origin=_timestamp(row["observation_end"], field="observation_end"),
            decision_time=_timestamp(
                row["feature_available_time"], field="feature_available_time"
            ),
            entry_time=_timestamp(row["entry_time"], field="entry_time"),
            exit_time=_timestamp(row["exit_time"], field="exit_time"),
            side=int(row["side"]),
        )
        for row in primary.to_dict(orient="records")
    ]
    _validate_clock_rows(rows, label="TBASR-24")
    if len(rows) != incidence["primary_events"]:
        raise ValueError("TBASR comparator incidence changed")
    return rows, {
        "market_rows_loaded": len(market),
        "market_diagnostics": market_diagnostics,
        "incidence": incidence,
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
    }


def _load_live_clocks() -> tuple[dict[str, list[ClockRow]], dict[str, Any]]:
    if sha256_file(LIVE_CLOCK_MANIFEST) != LIVE_CLOCK_MANIFEST_SHA256:
        raise ValueError("live comparator manifest changed")
    if sha256_file(LIVE_CLOCK) != LIVE_CLOCK_SHA256:
        raise ValueError("live comparator clock changed")
    manifest = load_json(LIVE_CLOCK_MANIFEST)
    if manifest.get("protocol_version") != "cchr_pure_clock_export_manifest_v1":
        raise ValueError("live comparator protocol changed")
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("live comparator opened outcomes")
    clock = manifest.get("clock")
    if not isinstance(clock, dict) or clock.get("sha256") != LIVE_CLOCK_SHA256:
        raise ValueError("live comparator clock binding changed")
    if clock.get("schema") != [
        "candidate_id",
        "split",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
    ]:
        raise ValueError("live comparator schema changed")

    by_candidate: dict[str, list[ClockRow]] = {}
    with gzip.open(LIVE_CLOCK, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != clock["schema"]:
            raise ValueError("live comparator CSV header changed")
        for raw in reader:
            decision = _timestamp(raw["decision_time"], field="decision_time")
            row = ClockRow(
                candidate_id=raw["candidate_id"],
                split=raw["split"],
                causal_origin=decision,
                decision_time=decision,
                entry_time=_timestamp(raw["entry_time"], field="entry_time"),
                exit_time=_timestamp(raw["exit_time"], field="exit_time"),
                side=int(raw["side"]),
            )
            by_candidate.setdefault(row.candidate_id, []).append(row)
    rows = [row for group in by_candidate.values() for row in group]
    if len(rows) != clock.get("rows"):
        raise ValueError("live comparator row count changed")
    for candidate_id, candidate_rows in by_candidate.items():
        _validate_clock_rows(candidate_rows, label=f"live {candidate_id}")
    if set(by_candidate) != {
        "live:cand_rex_veto_7",
        "live:new_long_minimal_funding_premium",
        "live:oi_upbit_ratio288_low",
    }:
        raise ValueError("live comparator membership changed")
    return by_candidate, {
        "manifest_hash": manifest["manifest_hash"],
        "rows": len(rows),
        "members": sorted(by_candidate),
    }


def _in_window(rows: Sequence[ClockRow], start: datetime, end: datetime) -> list[ClockRow]:
    return [row for row in rows if start <= row.entry_time < end and row.exit_time <= end]


def exact_entry_jaccard(primary: Sequence[ClockRow], comparator: Sequence[ClockRow]) -> tuple[float, int]:
    left = {row.entry_time for row in primary}
    right = {row.entry_time for row in comparator}
    intersection = len(left & right)
    union = len(left | right)
    return (intersection / union if union else 0.0), intersection


def maximum_tolerant_matches(
    primary: Sequence[ClockRow],
    comparator: Sequence[ClockRow],
    *,
    tolerance: timedelta = TOLERANCE,
) -> int:
    left = sorted(row.entry_time for row in primary)
    right = sorted(row.entry_time for row in comparator)
    i = 0
    j = 0
    matches = 0
    while i < len(left) and j < len(right):
        if right[j] < left[i] - tolerance:
            j += 1
        elif right[j] > left[i] + tolerance:
            i += 1
        else:
            matches += 1
            i += 1
            j += 1
    return matches


def _exposure_arrays(
    primary: Sequence[ClockRow],
    comparator: Sequence[ClockRow],
    *,
    start: datetime,
    end: datetime,
) -> tuple[np.ndarray, np.ndarray]:
    duration = end - start
    if duration <= timedelta(0) or duration % BAR != timedelta(0):
        raise ValueError("exposure comparison window left the five-minute grid")
    bars = int(duration / BAR)
    left = np.zeros(bars, dtype=np.int8)
    right = np.zeros(bars, dtype=np.int8)
    for rows, target in ((primary, left), (comparator, right)):
        for row in rows:
            entry = max(row.entry_time, start)
            exit_time = min(row.exit_time, end)
            if entry >= exit_time:
                continue
            if (entry - start) % BAR != timedelta(0):
                raise ValueError("entry left the five-minute comparison grid")
            if (exit_time - start) % BAR != timedelta(0):
                raise ValueError("exit left the five-minute comparison grid")
            first = int((entry - start) / BAR)
            last = int((exit_time - start) / BAR)
            if bool(target[first:last].any()):
                raise ValueError("comparator exposure schedule overlaps")
            target[first:last] = row.side
    return left, right


def exposure_metrics(
    primary: Sequence[ClockRow],
    comparator: Sequence[ClockRow],
    *,
    start: datetime,
    end: datetime,
) -> tuple[float, float]:
    left, right = _exposure_arrays(primary, comparator, start=start, end=end)
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(left, right)[0, 1])
    left_occupied = left != 0
    right_occupied = right != 0
    union = int(np.logical_or(left_occupied, right_occupied).sum())
    intersection = int(np.logical_and(left_occupied, right_occupied).sum())
    return correlation, (intersection / union if union else 0.0)


def _novelty_metrics(
    primary: Sequence[ClockRow],
    comparator: Sequence[ClockRow],
    *,
    candidate_id: str,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    left = _in_window(primary, start, end)
    right = _in_window(comparator, start, end)
    if not left or not right:
        raise ValueError(f"empty required comparator coverage: {candidate_id}")
    jaccard, exact_matches = exact_entry_jaccard(left, right)
    tolerant_matches = maximum_tolerant_matches(left, right)
    correlation, position_jaccard = exposure_metrics(
        left,
        right,
        start=start,
        end=end,
    )
    return {
        "candidate_id": candidate_id,
        "common_start": _iso(start),
        "common_end_exclusive": _iso(end),
        "tsdr_events": len(left),
        "comparator_events": len(right),
        "exact_entry_matches": exact_matches,
        "exact_entry_jaccard": jaccard,
        "tolerant_match_window_hours": 6,
        "maximum_one_to_one_tolerant_matches": tolerant_matches,
        "tsdr_tolerant_match_coverage": tolerant_matches / len(left),
        "signed_occupied_exposure_correlation": correlation,
        "position_bar_jaccard": position_jaccard,
    }


def _pure_clock_bytes(rows: Sequence[ClockRow]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=PURE_CLOCK_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "candidate_id": row.candidate_id,
                "split": row.split,
                "causal_origin": _iso(row.causal_origin),
                "decision_time": _iso(row.decision_time),
                "entry_time": _iso(row.entry_time),
                "exit_time": _iso(row.exit_time),
                "side": row.side,
            }
        )
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.getvalue().encode("utf-8"))
    return output.getvalue()


def build_outputs() -> tuple[dict[str, Any], bytes]:
    support = _load_support()
    tsdr_rows = _build_tsdr_clocks(support)
    tbasr_rows, tbasr_audit = _build_tbasr_train_clocks()
    live_rows, live_audit = _load_live_clocks()

    tbasr_metrics = _novelty_metrics(
        tsdr_rows,
        tbasr_rows,
        candidate_id="tbasr:primary",
        start=TRAIN_START,
        end=TRAIN_END,
    )
    live_metrics: dict[str, dict[str, Any]] = {}
    for candidate_id, rows in sorted(live_rows.items()):
        eligible = [row for row in rows if row.entry_time < SELECTION_END]
        if not eligible:
            raise ValueError(f"empty pre-2023 live comparator: {candidate_id}")
        start = max(TRAIN_START, min(row.entry_time for row in eligible))
        end = min(SELECTION_END, max(row.exit_time for row in eligible))
        live_metrics[candidate_id] = _novelty_metrics(
            tsdr_rows,
            eligible,
            candidate_id=candidate_id,
            start=start,
            end=end,
        )

    checks = {
        "tbasr_comparator_nonempty": len(tbasr_rows) > 0,
        "tbasr_exact_entry_jaccard_at_most_0_20": (
            tbasr_metrics["exact_entry_jaccard"] <= 0.20
        ),
        "tbasr_tolerant_coverage_at_most_0_35": (
            tbasr_metrics["tsdr_tolerant_match_coverage"] <= 0.35
        ),
        "tbasr_absolute_exposure_correlation_at_most_0_40": (
            abs(tbasr_metrics["signed_occupied_exposure_correlation"]) <= 0.40
        ),
    }
    for candidate_id, metrics in live_metrics.items():
        prefix = candidate_id.replace(":", "_")
        checks[f"{prefix}_nonempty"] = metrics["comparator_events"] > 0
        checks[f"{prefix}_exact_entry_jaccard_at_most_0_20"] = (
            metrics["exact_entry_jaccard"] <= 0.20
        )
        checks[f"{prefix}_tolerant_coverage_at_most_0_35"] = (
            metrics["tsdr_tolerant_match_coverage"] <= 0.35
        )
        checks[f"{prefix}_absolute_exposure_correlation_at_most_0_40"] = (
            abs(metrics["signed_occupied_exposure_correlation"]) <= 0.40
        )
    passed = all(checks.values())

    clock_rows = sorted(
        [*tsdr_rows, *tbasr_rows],
        key=lambda row: (row.candidate_id, row.entry_time),
    )
    clock_bytes = _pure_clock_bytes(clock_rows)
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": "2026-07-21",
        "implementation_binding": {
            "path": str(IMPLEMENTATION),
            "sha256": sha256_file(IMPLEMENTATION),
        },
        "support_binding": {
            "path": str(SUPPORT_ARTIFACT),
            "sha256": SUPPORT_ARTIFACT_SHA256,
            "result_hash": SUPPORT_RESULT_HASH,
        },
        "comparator_bindings": {
            "tbasr_implementation": {
                "path": str(TBASR_IMPLEMENTATION),
                "sha256": TBASR_IMPLEMENTATION_SHA256,
            },
            "tbasr_evaluator_freeze": {
                "path": str(TBASR_FREEZE),
                "sha256": TBASR_FREEZE_SHA256,
                "manifest_hash": tbasr_audit["evaluator_freeze_manifest_hash"],
            },
            "live_clock_manifest": {
                "path": str(LIVE_CLOCK_MANIFEST),
                "sha256": LIVE_CLOCK_MANIFEST_SHA256,
                "manifest_hash": live_audit["manifest_hash"],
            },
            "live_clock": {
                "path": str(LIVE_CLOCK),
                "sha256": LIVE_CLOCK_SHA256,
            },
        },
        "pure_clock": {
            "path": str(DEFAULT_CLOCK_OUTPUT),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "gzip_mtime": 0,
            "schema": list(PURE_CLOCK_FIELDS),
            "rows": len(clock_rows),
            "rows_by_candidate": {
                "tbasr:primary": len(tbasr_rows),
                "tsdr:primary": len(tsdr_rows),
            },
        },
        "outcome_boundary": {
            "market_rows_loaded_for_frozen_tbasr_causal_feature": tbasr_audit[
                "market_rows_loaded"
            ],
            "funding_rows_loaded": 0,
            "performance_artifacts_parsed": 0,
            "return_or_pnl_fields_read": 0,
            "strict_simulation_calls": 0,
            "tbasr_test_or_later_market_rows_loaded": 0,
            "post_2022_semantic_rows_loaded": 0,
            "raw_private_text_opened": False,
            "network_calls": 0,
            "economic_outcomes_computed": False,
        },
        "comparator_availability": {
            "tbasr_train_rows": len(tbasr_rows),
            "live_clock_rows": live_audit["rows"],
            "live_members": live_audit["members"],
        },
        "tbasr_source_audit": {
            "market": tbasr_audit["market_diagnostics"],
            "incidence": tbasr_audit["incidence"],
        },
        "novelty_metrics": {
            "tbasr:primary": tbasr_metrics,
            **live_metrics,
        },
        "novelty_gate": {"checks": checks, "passed": passed},
        "failure_action": None if passed else "retire_before_economic_evaluation",
        "next_action": (
            "freeze strict train evaluator"
            if passed
            else "retire TSDR-72; do not open economic outcomes or repair thresholds"
        ),
        "parameter_search_performed": False,
        "post_failure_repair_performed": False,
    }
    core["result_hash"] = canonical_hash(core)
    return {**core, "created_at": datetime.now(timezone.utc).isoformat()}, clock_bytes


def publish_outputs(
    report_path: Path,
    clock_path: Path,
    report: Mapping[str, Any],
    clock_bytes: bytes,
) -> None:
    if str(report["pure_clock"]["path"]) != str(clock_path):
        raise ValueError("pure-clock output path changed after freeze")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        clock_fd = os.open(clock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created.append(clock_path)
        with os.fdopen(clock_fd, "wb") as handle:
            handle.write(clock_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        report_fd = os.open(
            report_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        created.append(report_path)
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with os.fdopen(report_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--clock-output", type=Path, default=DEFAULT_CLOCK_OUTPUT)
    args = parser.parse_args()
    report, clock_bytes = build_outputs()
    publish_outputs(args.report_output, args.clock_output, report, clock_bytes)
    failed = [
        name for name, passed in report["novelty_gate"]["checks"].items() if not passed
    ]
    print(
        json.dumps(
            {
                "report_output": str(args.report_output),
                "clock_output": str(args.clock_output),
                "result_hash": report["result_hash"],
                "passed": report["novelty_gate"]["passed"],
                "failed_checks": failed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
