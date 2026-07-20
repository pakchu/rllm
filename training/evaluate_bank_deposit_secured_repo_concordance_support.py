"""Build the source-only BDRC-864 clocks and support decision.

The module is intentionally unable to load BTC market or funding data.  It
reads only the hash-bound H.8 and SOFR panels plus frozen parent event clocks
used for source-clock diagnostics.
"""
from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from training import fed_h8_deposit_migration_clock as h8_clock
from training import sofr_rate_dislocation_clock as sofr_clock


POLICY_ID = "BDRC-864"
PROTOCOL_VERSION = "bank_deposit_secured_repo_concordance_support_v1"
MECHANISM_DECISION = Path(
    "docs/bank-deposit-secured-repo-concordance-mechanism-decision-2026-07-20.md"
)
MECHANISM_DECISION_SHA256 = (
    "142db103f211800fde8233e4e39e6907bb02ad7646c150f97d6c0adb8b92d09c"
)
MECHANISM_COMMIT = "a0b7c319b874c662e1daad9cb5533c0984445dd6"
SOFR_SOURCE = Path(sofr_clock.SOURCE_PATH)
SOFR_SOURCE_SHA256 = "4993eda2b659e346b4d7b6e3aa0e2ff31cacf868f0e1fe2e1a5a76a03d1b5852"
SOFR_MANIFEST = Path(
    "data/new_york_fed_sofr_distribution_2018_2023/build_manifest.json"
)
SOFR_MANIFEST_SHA256 = (
    "873afb5234fd013e3bc454a83713abf34d9f4a4bffc9895683add7891c636598"
)
DEFAULT_OUTPUT = Path(
    "results/bank_deposit_secured_repo_concordance_support_2026-07-20.json"
)
DEFAULT_CLOCKS = Path(
    "results/bank_deposit_secured_repo_concordance_clocks_2026-07-20.csv.gz"
)
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc
HOLD_BARS = 864
BAR_MINUTES = 5
SOFR_LAG_OBSERVATIONS = 5
SOFR_FRESHNESS_HOURS = 36
ELIGIBLE_RELEASE_WEEKDAYS = frozenset({"Thursday", "Friday"})

PARENT_CLOCKS = {
    "H8DM-1": {
        "path": Path("results/fed_h8_deposit_migration_preregistered_clock_2026-07-18.csv.gz"),
        "sha256": "20405f79b86861adcc784c81223baae1c40fdf3c73edda339578471a6a6d1b40",
    },
    "SFRD-1": {
        "path": Path("results/sofr_rate_dislocation_preregistered_clock_2026-07-17.csv.gz"),
        "sha256": "391c42dd2b0d5b87ffcd73058dd9fa0c4d18fd2f535597effff5a4c8edea2e69",
    },
    "FLCC-1": {
        "path": Path("results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz"),
        "sha256": "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c",
    },
}

CLOCK_ORDER = (
    "primary",
    "h8_only",
    "sofr_only_h8_schedule",
    "discordant_state",
    "nsa_h8",
    "stale_h8_one_release",
    "stale_sofr_one_observation",
    "direction_flip",
    "deterministic_random_side",
)

WINDOWS = {
    "train_2020_2022": ("2020-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
    "2020": ("2020-01-01T00:00:00+00:00", "2021-01-01T00:00:00+00:00"),
    "2021": ("2021-01-01T00:00:00+00:00", "2022-01-01T00:00:00+00:00"),
    "2022": ("2022-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
    "selection_2023": ("2023-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
    "2023_h1": ("2023-01-01T00:00:00+00:00", "2023-07-01T00:00:00+00:00"),
    "2023_h2": ("2023-07-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
}


@dataclass(frozen=True)
class Event:
    clock_name: str
    release_date: str
    decision_time: str
    entry_time: str
    exit_time: str
    side: int
    h8_sign: int
    repo_sign: int
    repo5_bp: int
    sofr_effective_date: str
    sofr_available_at: str


EVENT_COLUMNS = tuple(Event.__dataclass_fields__)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_frozen_inputs() -> None:
    expected = {
        MECHANISM_DECISION: MECHANISM_DECISION_SHA256,
        h8_clock.SOURCE: h8_clock.SOURCE_SHA256,
        h8_clock.BUILD_MANIFEST: h8_clock.BUILD_MANIFEST_SHA256,
        SOFR_SOURCE: SOFR_SOURCE_SHA256,
        SOFR_MANIFEST: SOFR_MANIFEST_SHA256,
        **{item["path"]: item["sha256"] for item in PARENT_CLOCKS.values()},
    }
    for path, digest in expected.items():
        if not Path(path).is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"BDRC frozen input drift: {path}")


def _parse_utc(value: str | datetime | pd.Timestamp) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ValueError(f"BDRC timestamp is naive: {value!r}")
    return parsed.tz_convert("UTC").to_pydatetime()


def _h8_sign(row: pd.Series) -> int:
    score = float(row["stress_score"])
    if not math.isfinite(score) or score == 0.0:
        return 0
    sign = 1 if score > 0.0 else -1
    return sign if int(row["agreement_count"]) >= 2 else 0


def _decision_clock(release: datetime) -> tuple[datetime, datetime, datetime]:
    local = release.astimezone(NEW_YORK)
    decision_local = datetime.combine(local.date(), time(17, 0), tzinfo=NEW_YORK)
    if decision_local <= local:
        raise ValueError("BDRC decision must follow the archived H.8 release")
    decision = decision_local.astimezone(UTC)
    entry = decision + timedelta(minutes=BAR_MINUTES)
    exit_time = entry + timedelta(minutes=BAR_MINUTES * HOLD_BARS)
    return decision, entry, exit_time


def _latest_sofr_index(
    availability: list[datetime], decision: datetime
) -> int | None:
    index = bisect.bisect_right(availability, decision) - 1
    return index if index >= 0 else None


def _repo_state(
    rows: list[sofr_clock.SourceRow],
    availability: list[datetime],
    decision: datetime,
    *,
    stale_observations: int = 0,
) -> tuple[int, int, int] | None:
    current = _latest_sofr_index(availability, decision)
    if current is None:
        return None
    freshness = decision - rows[current].available_at
    if freshness < timedelta(0) or freshness > timedelta(hours=SOFR_FRESHNESS_HOURS):
        return None
    right = current - stale_observations
    left = right - SOFR_LAG_OBSERVATIONS
    if left < 0:
        return None
    delta = rows[right].rate_bp - rows[left].rate_bp
    sign = 1 if delta > 0 else -1 if delta < 0 else 0
    return sign, delta, right


def _event(
    *,
    clock_name: str,
    row: pd.Series,
    decision: datetime,
    entry: datetime,
    exit_time: datetime,
    side: int,
    h8_sign: int,
    repo_sign: int,
    repo5_bp: int,
    sofr_row: sofr_clock.SourceRow,
) -> Event:
    return Event(
        clock_name=clock_name,
        release_date=str(row["release_date"]),
        decision_time=decision.isoformat(),
        entry_time=entry.isoformat(),
        exit_time=exit_time.isoformat(),
        side=side,
        h8_sign=h8_sign,
        repo_sign=repo_sign,
        repo5_bp=repo5_bp,
        sofr_effective_date=sofr_row.effective_date.isoformat(),
        sofr_available_at=sofr_row.available_at.isoformat(),
    )


def _schedule(candidates: Iterable[Event]) -> list[Event]:
    ordered = sorted(candidates, key=lambda event: (event.entry_time, event.release_date))
    accepted: list[Event] = []
    reserved_until = datetime.min.replace(tzinfo=UTC)
    for event in ordered:
        entry = _parse_utc(event.entry_time)
        exit_time = _parse_utc(event.exit_time)
        if event.side not in (-1, 1) or exit_time <= entry:
            raise ValueError("BDRC candidate has invalid side or interval")
        if entry < reserved_until:
            continue
        accepted.append(event)
        reserved_until = exit_time
    return accepted


def build_clocks() -> dict[str, list[Event]]:
    h8 = h8_clock.load_source()
    sa = h8_clock.build_features(h8, adjustment="sa", tail_quantile=0.5)
    nsa = h8_clock.build_features(h8, adjustment="nsa", tail_quantile=0.5)
    sofr = sofr_clock.read_source(SOFR_SOURCE)
    availability = [row.available_at for row in sofr]
    candidates: dict[str, list[Event]] = {name: [] for name in CLOCK_ORDER[:-2]}
    previous_valid_h8_sign = 0

    for index, row in sa.iterrows():
        if str(row["release_weekday"]) not in ELIGIBLE_RELEASE_WEEKDAYS:
            continue
        if str(row["release_date"]) in h8_clock.STRUCTURAL_EXCLUSION_RELEASES:
            continue
        release = _parse_utc(row["release_time"])
        decision, entry, exit_time = _decision_clock(release)
        repo = _repo_state(sofr, availability, decision)
        if repo is None:
            continue
        repo_sign, repo_delta, repo_index = repo
        current_h8_sign = _h8_sign(row)
        current_nsa_sign = _h8_sign(nsa.loc[index])
        current_sofr = sofr[repo_index]

        def add(name: str, side: int, bank_sign: int, secured_sign: int, delta: int, sidx: int) -> None:
            candidates[name].append(
                _event(
                    clock_name=name,
                    row=row,
                    decision=decision,
                    entry=entry,
                    exit_time=exit_time,
                    side=side,
                    h8_sign=bank_sign,
                    repo_sign=secured_sign,
                    repo5_bp=delta,
                    sofr_row=sofr[sidx],
                )
            )

        if current_h8_sign:
            add("h8_only", -current_h8_sign, current_h8_sign, repo_sign, repo_delta, repo_index)
        if repo_sign:
            add(
                "sofr_only_h8_schedule",
                -repo_sign,
                current_h8_sign,
                repo_sign,
                repo_delta,
                repo_index,
            )
        if current_h8_sign and current_h8_sign == repo_sign:
            add("primary", -current_h8_sign, current_h8_sign, repo_sign, repo_delta, repo_index)
        if current_h8_sign and current_h8_sign == -repo_sign:
            add(
                "discordant_state",
                -current_h8_sign,
                current_h8_sign,
                repo_sign,
                repo_delta,
                repo_index,
            )
        if current_nsa_sign and current_nsa_sign == repo_sign:
            add("nsa_h8", -current_nsa_sign, current_nsa_sign, repo_sign, repo_delta, repo_index)
        if previous_valid_h8_sign and previous_valid_h8_sign == repo_sign:
            add(
                "stale_h8_one_release",
                -previous_valid_h8_sign,
                previous_valid_h8_sign,
                repo_sign,
                repo_delta,
                repo_index,
            )
        stale_repo = _repo_state(sofr, availability, decision, stale_observations=1)
        if stale_repo is not None:
            stale_sign, stale_delta, stale_index = stale_repo
            if current_h8_sign and current_h8_sign == stale_sign:
                add(
                    "stale_sofr_one_observation",
                    -current_h8_sign,
                    current_h8_sign,
                    stale_sign,
                    stale_delta,
                    stale_index,
                )
        if current_h8_sign:
            previous_valid_h8_sign = current_h8_sign

    clocks = {name: _schedule(rows) for name, rows in candidates.items()}
    primary = clocks["primary"]
    clocks["direction_flip"] = [
        Event(**{**asdict(event), "clock_name": "direction_flip", "side": -event.side})
        for event in primary
    ]
    random_rows: list[Event] = []
    for event in primary:
        digest = hashlib.sha256(
            f"BDRC-864-random-side-20260720|{event.entry_time}".encode("utf-8")
        ).digest()
        side = 1 if digest[0] < 128 else -1
        random_rows.append(
            Event(
                **{
                    **asdict(event),
                    "clock_name": "deterministic_random_side",
                    "side": side,
                }
            )
        )
    clocks["deterministic_random_side"] = random_rows
    if tuple(clocks) != CLOCK_ORDER:
        clocks = {name: clocks[name] for name in CLOCK_ORDER}
    return clocks


def _contained(events: Iterable[Event], start: str, end: str) -> list[Event]:
    lower, upper = _parse_utc(start), _parse_utc(end)
    return [
        event
        for event in events
        if _parse_utc(event.entry_time) >= lower and _parse_utc(event.exit_time) <= upper
    ]


def _summary(events: Iterable[Event], start: str, end: str) -> dict[str, Any]:
    selected = _contained(events, start, end)
    months: dict[str, int] = {}
    for event in selected:
        month = event.entry_time[:7]
        months[month] = months.get(month, 0) + 1
    count = len(selected)
    return {
        "events": count,
        "long": sum(event.side == 1 for event in selected),
        "short": sum(event.side == -1 for event in selected),
        "month_counts": dict(sorted(months.items())),
        "max_month_share": max(months.values(), default=0) / count if count else None,
        "repo_abs_le_3bp": sum(abs(event.repo5_bp) <= 3 for event in selected),
    }


def _write_clocks(path: Path, clocks: dict[str, list[Event]]) -> str:
    rows = [event for name in CLOCK_ORDER for event in clocks[name]]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=list(EVENT_COLUMNS), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(asdict(event) for event in rows)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return sha256_file(path)


def _read_parent_clocks() -> dict[str, pd.DataFrame]:
    parents: dict[str, pd.DataFrame] = {}
    h8 = pd.read_csv(PARENT_CLOCKS["H8DM-1"]["path"])
    parents["H8DM-1"] = h8.loc[
        h8["clock_mode"].eq("primary"), ["entry_time", "exit_time", "side"]
    ].copy()
    sofr = pd.read_csv(PARENT_CLOCKS["SFRD-1"]["path"])
    parents["SFRD-1"] = sofr[["entry_time", "exit_time", "side"]].copy()
    flcc = pd.read_csv(PARENT_CLOCKS["FLCC-1"]["path"])
    for candidate_id, frame in flcc.loc[flcc["clock_name"].eq("primary")].groupby("candidate_id"):
        parents[f"FLCC-1/{candidate_id}"] = frame[["entry_time", "exit_time", "side"]].copy()
    for frame in parents.values():
        frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="raise")
        frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True, errors="raise")
        frame["side"] = pd.to_numeric(frame["side"], errors="raise").astype(int)
    return parents


def _tolerant_jaccard(left: list[datetime], right: list[datetime], tolerance: timedelta) -> dict[str, Any]:
    left, right = sorted(left), sorted(right)
    left_index = 0
    right_index = 0
    matches = 0
    while left_index < len(left) and right_index < len(right):
        left_value = left[left_index]
        right_value = right[right_index]
        if right_value < left_value - tolerance:
            right_index += 1
        elif right_value > left_value + tolerance:
            left_index += 1
        else:
            matches += 1
            left_index += 1
            right_index += 1
    union = len(left) + len(right) - matches
    return {"matches": matches, "left": len(left), "right": len(right), "jaccard": matches / union if union else None}


def _exposure(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    size = int((end - start) / pd.Timedelta(minutes=BAR_MINUTES))
    result = np.zeros(size, dtype=np.int8)
    for row in frame.itertuples(index=False):
        entry = max(pd.Timestamp(row.entry_time), start)
        exit_time = min(pd.Timestamp(row.exit_time), end)
        if exit_time <= entry:
            continue
        left = int((entry - start) / pd.Timedelta(minutes=BAR_MINUTES))
        right = int((exit_time - start) / pd.Timedelta(minutes=BAR_MINUTES))
        result[left:right] = int(row.side)
    return result


def _parent_diagnostics(primary: list[Event]) -> dict[str, Any]:
    start = pd.Timestamp("2020-01-01T00:00:00Z")
    end = pd.Timestamp("2024-01-01T00:00:00Z")
    primary_frame = pd.DataFrame([asdict(event) for event in primary])
    for column in ("entry_time", "exit_time"):
        primary_frame[column] = pd.to_datetime(primary_frame[column], utc=True, errors="raise")
    primary_frame = primary_frame.loc[
        primary_frame["entry_time"].lt(end) & primary_frame["exit_time"].gt(start)
    ].copy()
    primary_exposure = _exposure(primary_frame, start, end)
    primary_entries = [timestamp.to_pydatetime() for timestamp in primary_frame["entry_time"]]
    result: dict[str, Any] = {}
    for name, frame in _read_parent_clocks().items():
        bounded = frame.loc[frame["entry_time"].lt(end) & frame["exit_time"].gt(start)].copy()
        other_exposure = _exposure(bounded, start, end)
        if np.std(primary_exposure) == 0.0 or np.std(other_exposure) == 0.0:
            correlation = None
        else:
            correlation = float(np.corrcoef(primary_exposure, other_exposure)[0, 1])
        result[name] = {
            "entry_overlap_plus_minus_6h": _tolerant_jaccard(
                primary_entries,
                [timestamp.to_pydatetime() for timestamp in bounded["entry_time"]],
                timedelta(hours=6),
            ),
            "signed_5m_occupied_exposure_correlation": correlation,
        }
    return result


def _control_diagnostics(clocks: dict[str, list[Event]]) -> dict[str, Any]:
    start = pd.Timestamp("2020-01-01T00:00:00Z")
    end = pd.Timestamp("2024-01-01T00:00:00Z")

    def bounded_frame(events: list[Event]) -> pd.DataFrame:
        frame = pd.DataFrame([asdict(event) for event in events])
        for column in ("entry_time", "exit_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
        return frame.loc[
            frame["entry_time"].lt(end) & frame["exit_time"].gt(start)
        ].copy()

    primary = bounded_frame(clocks["primary"])
    primary_entries = [
        timestamp.to_pydatetime() for timestamp in primary["entry_time"]
    ]
    primary_exposure = _exposure(primary, start, end)
    result: dict[str, Any] = {}
    for name in CLOCK_ORDER:
        if name == "primary":
            continue
        frame = bounded_frame(clocks[name])
        exposure = _exposure(frame, start, end)
        if np.std(primary_exposure) == 0.0 or np.std(exposure) == 0.0:
            correlation = None
        else:
            correlation = float(np.corrcoef(primary_exposure, exposure)[0, 1])
        result[name] = {
            "exact_entry_overlap": _tolerant_jaccard(
                primary_entries,
                [timestamp.to_pydatetime() for timestamp in frame["entry_time"]],
                timedelta(0),
            ),
            "signed_5m_occupied_exposure_correlation": correlation,
        }
    return result


def _support_gates(summaries: dict[str, dict[str, Any]]) -> dict[str, bool]:
    train = summaries["train_2020_2022"]
    selection = summaries["selection_2023"]
    return {
        "train_total_at_least_45": train["events"] >= 45,
        "each_train_year_at_least_10": all(summaries[str(year)]["events"] >= 10 for year in (2020, 2021, 2022)),
        "train_each_side_at_least_15": min(train["long"], train["short"]) >= 15,
        "train_max_month_share_at_most_25pct": train["max_month_share"] is not None and train["max_month_share"] <= 0.25,
        "selection_total_at_least_12": selection["events"] >= 12,
        "selection_each_half_at_least_5": min(summaries["2023_h1"]["events"], summaries["2023_h2"]["events"]) >= 5,
        "selection_each_side_at_least_4": min(selection["long"], selection["short"]) >= 4,
        "selection_max_month_share_at_most_25pct": selection["max_month_share"] is not None and selection["max_month_share"] <= 0.25,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def evaluate_support(
    output: str | Path = DEFAULT_OUTPUT,
    clocks_output: str | Path = DEFAULT_CLOCKS,
) -> dict[str, Any]:
    _verify_frozen_inputs()
    clocks = build_clocks()
    clocks_path = Path(clocks_output)
    clocks_sha = _write_clocks(clocks_path, clocks)
    primary_summaries = {
        name: _summary(clocks["primary"], *bounds) for name, bounds in WINDOWS.items()
    }
    gates = _support_gates(primary_summaries)
    controls = {
        clock_name: {name: _summary(events, *bounds) for name, bounds in WINDOWS.items()}
        for clock_name, events in clocks.items()
        if clock_name != "primary"
    }
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "mechanism_decision": {
            "path": str(MECHANISM_DECISION),
            "sha256": MECHANISM_DECISION_SHA256,
            "commit": MECHANISM_COMMIT,
        },
        "sources": {
            "h8": {"path": str(h8_clock.SOURCE), "sha256": h8_clock.SOURCE_SHA256, "rows_read": 365},
            "h8_manifest": {"path": str(h8_clock.BUILD_MANIFEST), "sha256": h8_clock.BUILD_MANIFEST_SHA256},
            "sofr": {"path": str(SOFR_SOURCE), "sha256": SOFR_SOURCE_SHA256, "rows_read": 1436},
            "sofr_manifest": {"path": str(SOFR_MANIFEST), "sha256": SOFR_MANIFEST_SHA256},
        },
        "policy": {
            "eligible_release_weekdays": sorted(ELIGIBLE_RELEASE_WEEKDAYS),
            "h8_robust_window_releases": h8_clock.ROBUST_WINDOW_RELEASES,
            "h8_minimum_component_agreement": 2,
            "sofr_lag_observations": SOFR_LAG_OBSERVATIONS,
            "sofr_freshness_hours": SOFR_FRESHNESS_HOURS,
            "decision": "17:00 America/New_York",
            "entry": "17:05 America/New_York",
            "hold_bars_5m": HOLD_BARS,
            "direction": "positive stress+tightening SHORT; negative relief+easing LONG",
        },
        "primary": primary_summaries,
        "controls": controls,
        "support_gates": gates,
        "support_passed": all(gates.values()),
        "decision": "PASS_SOURCE_SUPPORT" if all(gates.values()) else "REJECT_SOURCE_SUPPORT_NO_REPAIR",
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "control_clock_diagnostics": _control_diagnostics(clocks),
        "parent_clock_diagnostics": _parent_diagnostics(clocks["primary"]),
        "clocks": {
            "path": str(clocks_path),
            "sha256": clocks_sha,
            "rows": sum(len(events) for events in clocks.values()),
            "primary_rows": len(clocks["primary"]),
            "control_rows": sum(len(events) for name, events in clocks.items() if name != "primary"),
        },
        "outcome_boundary": {
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "return_rows_loaded": 0,
            "market_values_read": 0,
            "funding_values_read": 0,
            "return_or_pnl_fields": 0,
            "post_2023_source_rows_loaded": 0,
            "outcomes_opened": False,
        },
        "evaluator_authorized": all(gates.values()),
        "no_repair": True,
    }
    artifact = {**core, "manifest_hash": canonical_hash(core)}
    _atomic_write_json(Path(output), artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--clocks-output", default=str(DEFAULT_CLOCKS))
    args = parser.parse_args()
    print(json.dumps(evaluate_support(args.output, args.clocks_output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
