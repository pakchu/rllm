"""Build the outcome-blind RPDS-576 source-support and novelty verdict."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from training import preregister_refined_product_divergence_shock as prereg


PROTOCOL_VERSION = "refined_product_divergence_shock_support_v1"
POLICY_ID = "RPDS-576"
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = Path(
    "results/refined_product_divergence_shock_preregistration_2026-07-21.json"
)
PREREGISTRATION_SHA256 = (
    "03ce0d29e0c67f6366959690feebeea9e96b854389e4b15804d9a7ee5cd277b2"
)
PREREGISTRATION_MANIFEST_HASH = (
    "b58a259e128740058d7501188ab479cca1cfe0e40e8a18dc81df3f2e5058320a"
)
EVALUATOR_SOURCE = Path("training/evaluate_refined_product_divergence_shock_support.py")
DEFAULT_CLOCK_OUTPUT = Path(
    "data/refined_product_divergence_shock_clocks_2019_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/refined_product_divergence_shock_source_support_2026-07-21.json"
)

UTC = timezone.utc
BAR = timedelta(minutes=5)
HOLD = timedelta(hours=48)
TOLERANCE = timedelta(hours=6)
COMPARISON_START = datetime(2020, 1, 1, tzinfo=UTC)
COMPARISON_END = datetime(2024, 1, 1, tzinfo=UTC)
CONTROL_NAMES = (
    "primary",
    "direction_flip",
    "refined_only",
    "crude_only",
    "epsb_concordance_48h",
    "one_release_delay",
    "deterministic_random_side",
    "latency_plus_5m",
)
SPLITS = {
    "history": (
        datetime(2019, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 1, tzinfo=UTC),
    ),
    "train": (COMPARISON_START, datetime(2023, 1, 1, tzinfo=UTC)),
    "selection": (datetime(2023, 1, 1, tzinfo=UTC), COMPARISON_END),
}
SOURCE_COLUMNS = (
    "release_date",
    "available_time_utc",
    "data_week_ending",
    "previous_week_ending",
    "commercial_crude_stock_mmbbl",
    "commercial_crude_change_mmbbl",
    "commercial_crude_arithmetic_change_mmbbl",
    "commercial_crude_change_discrepancy_mmbbl",
    "gasoline_stock_mmbbl",
    "gasoline_change_mmbbl",
    "gasoline_arithmetic_change_mmbbl",
    "gasoline_change_discrepancy_mmbbl",
    "distillate_stock_mmbbl",
    "distillate_change_mmbbl",
    "distillate_arithmetic_change_mmbbl",
    "distillate_change_discrepancy_mmbbl",
    "published_difference_consistent",
    "archive_page_url",
    "table1_csv_url",
    "table1_sha256",
    "source_complete",
)
ALLOWED_SOURCE_COLUMNS = (
    "release_date",
    "available_time_utc",
    "source_complete",
    "published_difference_consistent",
    "commercial_crude_change_mmbbl",
    "gasoline_change_mmbbl",
    "distillate_change_mmbbl",
)
ALLOWED_SOURCE_INDEXES = {
    SOURCE_COLUMNS.index(column): column for column in ALLOWED_SOURCE_COLUMNS
}
CLOCK_COLUMNS = (
    "candidate_id",
    "control",
    "split",
    "origin_release_date",
    "release_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "commercial_crude_change_mmbbl",
    "gasoline_change_mmbbl",
    "distillate_change_mmbbl",
)
EPSB_COMPARATOR_COLUMNS = (
    "control",
    "release_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "commercial_crude_change_mmbbl",
    "gasoline_change_mmbbl",
    "distillate_change_mmbbl",
    "archive_page_url",
    "table1_csv_url",
)
CCHR_COMPARATOR_COLUMNS = (
    "candidate_id",
    "split",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
)
FORBIDDEN_OUTCOME_TOKENS = (
    "return",
    "pnl",
    "profit",
    "cagr",
    "mdd",
    "drawdown",
    "equity",
    "sharpe",
    "label",
    "future",
)


@dataclass(frozen=True)
class SourceRow:
    release_date: date
    available_time: datetime
    source_complete: bool
    published_difference_consistent: bool
    crude: Decimal
    gasoline: Decimal
    distillate: Decimal


@dataclass(frozen=True)
class Event:
    control: str
    origin_release_date: date
    release_date: date
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    side: int
    crude: Decimal
    gasoline: Decimal
    distillate: Decimal


@dataclass(frozen=True)
class ClockRow:
    candidate_id: str
    control: str
    split: str
    origin_release_date: date
    release_date: date
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    side: int
    crude: Decimal
    gasoline: Decimal
    distillate: Decimal


@dataclass(frozen=True)
class ComparatorClock:
    candidate_id: str
    entry_time: datetime
    exit_time: datetime
    side: int
    release_date: date | None = None


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return parsed.astimezone(UTC)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("cannot serialize naive timestamp")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid frozen boolean: {value!r}")


def _parse_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid source decimal") from exc
    if not parsed.is_finite():
        raise ValueError("non-finite source decimal")
    return parsed


def _sign(value: Decimal) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _quality_ready(row: SourceRow) -> bool:
    return row.source_complete and row.published_difference_consistent


def _signal_eligible(row: SourceRow) -> bool:
    return _quality_ready(row) and all(
        value != 0 for value in (row.crude, row.gasoline, row.distillate)
    )


def verify_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("RPDS preregistration file hash drift")
    registration = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    if registration.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("RPDS preregistration manifest drift")
    core = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if canonical_hash(core) != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("RPDS preregistration canonical hash mismatch")
    if registration.get("policy_id") != POLICY_ID:
        raise RuntimeError("RPDS policy identity drift")
    if registration.get("later_outcome_contract", {}).get("authorized") is not False:
        raise RuntimeError("RPDS preregistration opened outcomes")
    return registration


def _project_source_record(raw_record: bytes) -> dict[str, str]:
    """Parse only preregistered fields from the frozen unquoted source CSV."""
    if not raw_record.endswith(b"\n"):
        raise RuntimeError("RPDS source record lacks a newline terminator")
    if b'"' in raw_record:
        raise RuntimeError("RPDS source unexpectedly requires quoted CSV parsing")
    end = len(raw_record) - (2 if raw_record.endswith(b"\r\n") else 1)
    field_index = 0
    selected = ALLOWED_SOURCE_INDEXES.get(field_index)
    buffer = bytearray() if selected is not None else None
    projected: dict[str, str] = {}

    def finish_field() -> None:
        if selected is not None and buffer is not None:
            try:
                projected[selected] = buffer.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError("RPDS selected source field is not UTF-8") from exc

    for position in range(end):
        value = raw_record[position]
        if value == ord(","):
            finish_field()
            field_index += 1
            selected = ALLOWED_SOURCE_INDEXES.get(field_index)
            buffer = bytearray() if selected is not None else None
        elif value in (ord("\r"), ord("\n")):
            raise RuntimeError("RPDS source contains an embedded line ending")
        elif buffer is not None:
            buffer.append(value)
    finish_field()
    if field_index + 1 != len(SOURCE_COLUMNS):
        raise RuntimeError("RPDS source data-column count drift")
    if tuple(projected) != tuple(
        column for column in SOURCE_COLUMNS if column in ALLOWED_SOURCE_COLUMNS
    ):
        raise RuntimeError("RPDS source projection drift")
    return projected


def load_source() -> tuple[list[SourceRow], int]:
    for binding in prereg.SOURCE_BINDINGS.values():
        prereg.verify_binding(binding)
    panel = _path(prereg.SOURCE_BINDINGS["panel"]["path"])
    rows: list[SourceRow] = []
    physical_rows = 0
    with gzip.open(panel, "rb") as handle:
        raw_header = handle.readline()
        try:
            header = tuple(raw_header.decode("ascii").rstrip("\r\n").split(","))
        except UnicodeDecodeError as exc:
            raise RuntimeError("RPDS source header is not ASCII") from exc
        if header != SOURCE_COLUMNS:
            raise RuntimeError("RPDS source schema drift")
        for raw_record in handle:
            physical_rows += 1
            raw = _project_source_record(raw_record)
            row = SourceRow(
                release_date=date.fromisoformat(raw["release_date"]),
                available_time=_parse_time(raw["available_time_utc"]),
                source_complete=_parse_bool(raw["source_complete"]),
                published_difference_consistent=_parse_bool(
                    raw["published_difference_consistent"]
                ),
                crude=_parse_decimal(raw["commercial_crude_change_mmbbl"]),
                gasoline=_parse_decimal(raw["gasoline_change_mmbbl"]),
                distillate=_parse_decimal(raw["distillate_change_mmbbl"]),
            )
            expected_available_date = row.release_date + timedelta(days=1)
            if row.available_time.date() != expected_available_date:
                raise RuntimeError("RPDS source availability date drift")
            if row.available_time.time() != datetime(2000, 1, 1, 13, tzinfo=UTC).time():
                raise RuntimeError("RPDS source availability clock drift")
            rows.append(row)
    if physical_rows != 259 or len(rows) != 259:
        raise RuntimeError("RPDS source row count drift")
    if sum(_quality_ready(row) for row in rows) != 258:
        raise RuntimeError("RPDS complete source count drift")
    if sum(not _quality_ready(row) for row in rows) != 1:
        raise RuntimeError("RPDS quarantine count drift")
    if any(row.release_date.year > 2023 for row in rows):
        raise RuntimeError("RPDS source opened post-2023 rows")
    if any(
        current.available_time <= previous.available_time
        for previous, current in zip(rows, rows[1:])
    ):
        raise RuntimeError("RPDS source clock is not strictly increasing")
    return rows, physical_rows


def _primary_side(row: SourceRow) -> int:
    if not _signal_eligible(row):
        return 0
    crude = _sign(row.crude)
    gasoline = _sign(row.gasoline)
    distillate = _sign(row.distillate)
    return gasoline if gasoline == distillate and crude == -gasoline else 0


def _event(
    row: SourceRow,
    *,
    control: str,
    side: int,
    origin_release_date: date | None = None,
    entry_delay: timedelta = BAR,
) -> Event:
    if side not in (-1, 1):
        raise ValueError("RPDS event side must be signed")
    entry = row.available_time + entry_delay
    return Event(
        control=control,
        origin_release_date=origin_release_date or row.release_date,
        release_date=row.release_date,
        signal_time=row.available_time,
        entry_time=entry,
        exit_time=entry + HOLD,
        side=side,
        crude=row.crude,
        gasoline=row.gasoline,
        distillate=row.distillate,
    )


def deterministic_random_side(release_date_value: date) -> int:
    digest = hashlib.sha256(
        f"{POLICY_ID}|{release_date_value.isoformat()}".encode()
    ).digest()
    return 1 if digest[0] & 1 else -1


def build_raw_events(rows: Sequence[SourceRow]) -> dict[str, list[Event]]:
    primary = [
        _event(row, control="primary", side=side)
        for row in rows
        if (side := _primary_side(row)) != 0
    ]
    controls: dict[str, list[Event]] = {
        "primary": primary,
        "direction_flip": [
            replace(event, control="direction_flip", side=-event.side)
            for event in primary
        ],
        "deterministic_random_side": [
            replace(
                event,
                control="deterministic_random_side",
                side=deterministic_random_side(event.release_date),
            )
            for event in primary
        ],
        "latency_plus_5m": [
            replace(
                event,
                control="latency_plus_5m",
                entry_time=event.entry_time + BAR,
                exit_time=event.exit_time + BAR,
            )
            for event in primary
        ],
    }
    refined: list[Event] = []
    crude_only: list[Event] = []
    epsb: list[Event] = []
    delayed: list[Event] = []
    pending: tuple[date, int] | None = None
    for row in rows:
        if not _quality_ready(row):
            pending = None
            continue
        if not _signal_eligible(row):
            continue
        crude = _sign(row.crude)
        gasoline = _sign(row.gasoline)
        distillate = _sign(row.distillate)
        if gasoline == distillate:
            refined.append(_event(row, control="refined_only", side=gasoline))
        crude_only.append(_event(row, control="crude_only", side=-crude))
        if crude == gasoline == distillate:
            epsb.append(_event(row, control="epsb_concordance_48h", side=crude))
        if pending is not None:
            origin_release_date, pending_side = pending
            delayed.append(
                _event(
                    row,
                    control="one_release_delay",
                    side=pending_side,
                    origin_release_date=origin_release_date,
                )
            )
        current_primary = _primary_side(row)
        pending = (row.release_date, current_primary) if current_primary else None
    controls["refined_only"] = refined
    controls["crude_only"] = crude_only
    controls["epsb_concordance_48h"] = epsb
    controls["one_release_delay"] = delayed
    if set(controls) != set(CONTROL_NAMES):
        raise RuntimeError("RPDS control construction drift")
    return controls


def schedule_events(events: Iterable[Event]) -> list[Event]:
    accepted: list[Event] = []
    reserved_until: datetime | None = None
    for event in sorted(
        events,
        key=lambda item: (
            item.entry_time,
            item.origin_release_date,
            item.release_date,
            item.control,
        ),
    ):
        if reserved_until is not None and event.entry_time < reserved_until:
            continue
        accepted.append(event)
        reserved_until = event.exit_time
    return accepted


def _split_for_event(event: Event) -> str | None:
    origin = datetime.combine(event.origin_release_date, datetime.min.time(), UTC)
    release = datetime.combine(event.release_date, datetime.min.time(), UTC)
    for name, (start, end) in SPLITS.items():
        if (
            start <= origin < end
            and start <= release < end
            and start <= event.signal_time < end
            and start <= event.entry_time < end
            and event.exit_time <= end
        ):
            return name
    return None


def build_clock_rows(rows: Sequence[SourceRow]) -> list[ClockRow]:
    ledger: list[ClockRow] = []
    for control, raw_events in build_raw_events(rows).items():
        for event in schedule_events(raw_events):
            split = _split_for_event(event)
            if split is None:
                continue
            ledger.append(
                ClockRow(
                    candidate_id=f"{POLICY_ID}:{control}",
                    control=control,
                    split=split,
                    origin_release_date=event.origin_release_date,
                    release_date=event.release_date,
                    signal_time=event.signal_time,
                    entry_time=event.entry_time,
                    exit_time=event.exit_time,
                    side=event.side,
                    crude=event.crude,
                    gasoline=event.gasoline,
                    distillate=event.distillate,
                )
            )
    ledger.sort(key=lambda row: (row.control, row.entry_time, row.release_date))
    validate_clock_rows(ledger)
    return ledger


def validate_clock_rows(rows: Sequence[ClockRow]) -> None:
    for row in rows:
        if row.side not in (-1, 1):
            raise RuntimeError("RPDS clock contains invalid side")
        expected_delay = 2 * BAR if row.control == "latency_plus_5m" else BAR
        if row.entry_time != row.signal_time + expected_delay:
            raise RuntimeError("RPDS clock entry latency drift")
        if row.exit_time != row.entry_time + HOLD:
            raise RuntimeError("RPDS clock hold drift")
        if (
            _split_for_event(
                Event(
                    control=row.control,
                    origin_release_date=row.origin_release_date,
                    release_date=row.release_date,
                    signal_time=row.signal_time,
                    entry_time=row.entry_time,
                    exit_time=row.exit_time,
                    side=row.side,
                    crude=row.crude,
                    gasoline=row.gasoline,
                    distillate=row.distillate,
                )
            )
            != row.split
        ):
            raise RuntimeError("RPDS clock split containment drift")
    grouped: dict[tuple[str, str], list[ClockRow]] = {}
    for row in rows:
        grouped.setdefault((row.control, row.split), []).append(row)
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: row.entry_time)
        if any(
            current.entry_time < previous.exit_time
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise RuntimeError(f"RPDS clock overlaps within {key}")
        if len({row.entry_time for row in ordered}) != len(ordered):
            raise RuntimeError(f"RPDS clock duplicates within {key}")


def _clock_values(row: ClockRow) -> tuple[str | int, ...]:
    return (
        row.candidate_id,
        row.control,
        row.split,
        row.origin_release_date.isoformat(),
        row.release_date.isoformat(),
        _iso(row.signal_time),
        _iso(row.entry_time),
        _iso(row.exit_time),
        row.side,
        str(row.crude),
        str(row.gasoline),
        str(row.distillate),
    )


def write_clocks(rows: Sequence[ClockRow], output: str | Path) -> None:
    target = _path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CLOCK_COLUMNS)
    for row in rows:
        writer.writerow(_clock_values(row))
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(buffer.getvalue().encode())


def summarize(rows: Sequence[ClockRow], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row.split == split]
    months = Counter(row.entry_time.strftime("%Y-%m") for row in selected)
    years = Counter(str(row.entry_time.year) for row in selected)
    halves = Counter(
        f"{row.entry_time.year}H{1 if row.entry_time.month <= 6 else 2}"
        for row in selected
    )
    longs = sum(row.side == 1 for row in selected)
    shorts = sum(row.side == -1 for row in selected)
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "long_share": longs / len(selected) if selected else 0.0,
        "short_share": shorts / len(selected) if selected else 0.0,
        "year_counts": dict(sorted(years.items())),
        "half_counts": dict(sorted(halves.items())),
        "month_counts": dict(sorted(months.items())),
        "maximum_month_share": (
            max(months.values(), default=0) / len(selected) if selected else 0.0
        ),
    }


def evaluate_support(rows: Sequence[ClockRow]) -> dict[str, Any]:
    grouped = {
        control: [row for row in rows if row.control == control]
        for control in CONTROL_NAMES
    }
    summaries = {
        control: {split: summarize(control_rows, split) for split in SPLITS}
        for control, control_rows in grouped.items()
    }
    train = summaries["primary"]["train"]
    selection = summaries["primary"]["selection"]
    train_years = train["year_counts"]
    selection_halves = selection["half_counts"]
    primary_releases = {row.release_date for row in grouped["primary"]}
    epsb_releases = {row.release_date for row in grouped["epsb_concordance_48h"]}
    checks = {
        "all_controls_present": all(grouped.values()),
        "train_events_min": train["events"] >= 24,
        "train_events_max": train["events"] <= 75,
        "train_each_year_min": all(
            train_years.get(str(year), 0) >= 5 for year in (2020, 2021, 2022)
        ),
        "train_side_balance": min(train["long_share"], train["short_share"]) >= 0.25,
        "train_month_concentration": train["maximum_month_share"] <= 0.25,
        "selection_events_min": selection["events"] >= 8,
        "selection_events_max": selection["events"] <= 24,
        "selection_each_half_min": all(
            selection_halves.get(name, 0) >= 3 for name in ("2023H1", "2023H2")
        ),
        "selection_both_sides": selection["longs"] >= 1 and selection["shorts"] >= 1,
        "selection_month_concentration": selection["maximum_month_share"] <= 0.25,
        "epsb_release_state_disjoint": not (primary_releases & epsb_releases),
        "direction_flip_exact": [
            (row.entry_time, row.exit_time, row.release_date, row.side)
            for row in grouped["direction_flip"]
        ]
        == [
            (row.entry_time, row.exit_time, row.release_date, -row.side)
            for row in grouped["primary"]
        ],
        "random_side_exact": all(
            row.side == deterministic_random_side(row.release_date)
            for row in grouped["deterministic_random_side"]
        ),
    }
    return {
        "summaries": summaries,
        "checks": checks,
        "passed": all(checks.values()),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
    }


def _forbidden_header(header: Sequence[str]) -> bool:
    return any(
        token in column.lower()
        for column in header
        for token in FORBIDDEN_OUTCOME_TOKENS
    )


def _validate_comparator_group(
    rows: Sequence[ComparatorClock], candidate_id: str
) -> None:
    if not rows:
        raise RuntimeError(f"empty required comparator: {candidate_id}")
    ordered = sorted(rows, key=lambda row: row.entry_time)
    if any(row.side not in (-1, 1) for row in ordered):
        raise RuntimeError(f"invalid comparator side: {candidate_id}")
    if len({row.entry_time for row in ordered}) != len(ordered):
        raise RuntimeError(f"duplicate comparator entry: {candidate_id}")
    if any(row.entry_time >= row.exit_time for row in ordered):
        raise RuntimeError(f"invalid comparator interval: {candidate_id}")
    if any(
        current.entry_time < previous.exit_time
        for previous, current in zip(ordered, ordered[1:])
    ):
        raise RuntimeError(f"overlapping comparator: {candidate_id}")
    for row in ordered:
        if row.entry_time < COMPARISON_START or row.exit_time > COMPARISON_END:
            raise RuntimeError(f"comparator left frozen grid: {candidate_id}")
        if (row.entry_time - COMPARISON_START) % BAR != timedelta(0):
            raise RuntimeError(f"off-grid comparator entry: {candidate_id}")
        if (row.exit_time - COMPARISON_START) % BAR != timedelta(0):
            raise RuntimeError(f"off-grid comparator exit: {candidate_id}")


def _validate_comparator_times(
    *,
    family: str,
    split: str | None,
    decision_time: datetime,
    entry_time: datetime,
    exit_time: datetime,
    release_date: date | None,
) -> bool:
    if not decision_time <= entry_time < exit_time:
        raise RuntimeError(f"invalid comparator clock order: {family}")
    for name, value in (
        ("decision", decision_time),
        ("entry", entry_time),
        ("exit", exit_time),
    ):
        if (value - COMPARISON_START) % BAR != timedelta(0):
            raise RuntimeError(f"off-grid comparator {name}: {family}")

    if family == "epsb":
        if split is not None or release_date is None:
            raise RuntimeError("invalid EPSB comparator metadata")
        if decision_time.date() != release_date + timedelta(days=1):
            raise RuntimeError("EPSB release/signal chronology drift")
        if exit_time <= COMPARISON_START:
            return False
        if (
            decision_time < COMPARISON_START
            or entry_time < COMPARISON_START
            or exit_time > COMPARISON_END
        ):
            raise RuntimeError("EPSB comparator straddles frozen novelty grid")
        return True

    if split not in {"train", "selection"} or release_date is not None:
        raise RuntimeError(f"invalid CCHR comparator metadata: {family}")
    split_start, split_end = SPLITS[split]
    if not (
        split_start <= decision_time < split_end
        and split_start <= entry_time < split_end
        and split_start < exit_time <= split_end
    ):
        raise RuntimeError(f"CCHR comparator left declared split: {family}:{split}")
    return True


def load_comparators() -> tuple[dict[str, list[ComparatorClock]], int]:
    groups: dict[str, list[ComparatorClock]] = {}
    rows_read = 0
    for family, binding in prereg.COMPARATOR_BINDINGS.items():
        prereg.verify_binding(binding)
        with gzip.open(
            _path(binding["path"]), "rt", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames or ())
            expected = (
                EPSB_COMPARATOR_COLUMNS if family == "epsb" else CCHR_COMPARATOR_COLUMNS
            )
            if header != expected:
                raise RuntimeError(f"comparator schema drift: {family}")
            if _forbidden_header(header):
                raise RuntimeError(f"comparator outcome field: {family}")
            for raw in reader:
                rows_read += 1
                candidate_id = (
                    f"epsb:{raw['control']}"
                    if family == "epsb"
                    else f"{family}:{raw['candidate_id']}"
                )
                decision = _parse_time(
                    raw["signal_time"] if family == "epsb" else raw["decision_time"]
                )
                entry = _parse_time(raw["entry_time"])
                exit_time = _parse_time(raw["exit_time"])
                release = (
                    date.fromisoformat(raw["release_date"])
                    if family == "epsb"
                    else None
                )
                split = None if family == "epsb" else raw["split"]
                if not _validate_comparator_times(
                    family=family,
                    split=split,
                    decision_time=decision,
                    entry_time=entry,
                    exit_time=exit_time,
                    release_date=release,
                ):
                    continue
                groups.setdefault(candidate_id, []).append(
                    ComparatorClock(
                        candidate_id=candidate_id,
                        entry_time=entry,
                        exit_time=exit_time,
                        side=int(raw["side"]),
                        release_date=release,
                    )
                )
    for candidate_id, rows in groups.items():
        _validate_comparator_group(rows, candidate_id)
    if "epsb:primary" not in groups:
        raise RuntimeError("missing EPSB primary comparator")
    return groups, rows_read


def exact_entry_jaccard(
    left: Sequence[ComparatorClock], right: Sequence[ComparatorClock]
) -> tuple[float, int]:
    a = {row.entry_time for row in left}
    b = {row.entry_time for row in right}
    matches = len(a & b)
    union = len(a | b)
    return (matches / union if union else 0.0), matches


def maximum_tolerant_matches(
    left: Sequence[ComparatorClock],
    right: Sequence[ComparatorClock],
    tolerance: timedelta = TOLERANCE,
) -> int:
    a = sorted(row.entry_time for row in left)
    b = sorted(row.entry_time for row in right)
    i = 0
    j = 0
    matches = 0
    while i < len(a) and j < len(b):
        if b[j] < a[i] - tolerance:
            j += 1
        elif b[j] > a[i] + tolerance:
            i += 1
        else:
            matches += 1
            i += 1
            j += 1
    return matches


def _exposure(rows: Sequence[ComparatorClock]) -> np.ndarray:
    duration = COMPARISON_END - COMPARISON_START
    bars = int(duration / BAR)
    values = np.zeros(bars, dtype=np.int8)
    for row in rows:
        first = int((row.entry_time - COMPARISON_START) / BAR)
        last = int((row.exit_time - COMPARISON_START) / BAR)
        if first < 0 or last > bars or first >= last:
            raise RuntimeError("comparator exposure left frozen grid")
        if bool(values[first:last].any()):
            raise RuntimeError("comparator exposure overlaps")
        values[first:last] = row.side
    return values


def signed_exposure_correlation(
    left: Sequence[ComparatorClock], right: Sequence[ComparatorClock]
) -> float:
    a = _exposure(left)
    b = _exposure(right)
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    correlation = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(correlation):
        raise RuntimeError("non-finite exposure correlation")
    return correlation


def evaluate_novelty(
    primary_rows: Sequence[ClockRow],
    comparator_groups: Mapping[str, Sequence[ComparatorClock]],
) -> dict[str, Any]:
    primary = [
        ComparatorClock(
            candidate_id=POLICY_ID,
            entry_time=row.entry_time,
            exit_time=row.exit_time,
            side=row.side,
            release_date=row.release_date,
        )
        for row in primary_rows
        if row.control == "primary" and row.split in {"train", "selection"}
    ]
    _validate_comparator_group(primary, POLICY_ID)
    metrics: dict[str, dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for candidate_id, comparator in sorted(comparator_groups.items()):
        jaccard, exact_matches = exact_entry_jaccard(primary, comparator)
        tolerant_matches = maximum_tolerant_matches(primary, comparator)
        primary_coverage = tolerant_matches / len(primary)
        comparator_coverage = tolerant_matches / len(comparator)
        max_coverage = max(primary_coverage, comparator_coverage)
        correlation = signed_exposure_correlation(primary, comparator)
        metrics[candidate_id] = {
            "rpds_events": len(primary),
            "comparator_events": len(comparator),
            "exact_entry_matches": exact_matches,
            "exact_entry_jaccard": jaccard,
            "maximum_one_to_one_matches_within_6h": tolerant_matches,
            "rpds_tolerant_coverage": primary_coverage,
            "comparator_tolerant_coverage": comparator_coverage,
            "maximum_bidirectional_containment": max_coverage,
            "signed_occupied_exposure_correlation": correlation,
        }
        checks[f"{candidate_id}:exact_entry_jaccard"] = jaccard <= 0.10
        checks[f"{candidate_id}:tolerant_containment"] = max_coverage <= 0.25
        checks[f"{candidate_id}:exposure_correlation"] = abs(correlation) <= 0.35

    epsb = comparator_groups["epsb:primary"]
    rpds_release_dates = {row.release_date for row in primary}
    epsb_release_dates = {row.release_date for row in epsb}
    exact_release_overlap = len(rpds_release_dates & epsb_release_dates)
    exact_entry_overlap = len(
        {row.entry_time for row in primary} & {row.entry_time for row in epsb}
    )
    checks["epsb_primary:exact_release_overlap_zero"] = exact_release_overlap == 0
    checks["epsb_primary:exact_entry_overlap_zero"] = exact_entry_overlap == 0
    return {
        "evaluated": True,
        "comparison_start": _iso(COMPARISON_START),
        "comparison_end_exclusive": _iso(COMPARISON_END),
        "metrics": metrics,
        "epsb_primary_exact_release_overlap": exact_release_overlap,
        "epsb_primary_exact_entry_overlap": exact_entry_overlap,
        "checks": checks,
        "passed": all(checks.values()),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
    }


ComparatorLoader = Callable[[], tuple[dict[str, list[ComparatorClock]], int]]


def maybe_evaluate_novelty(
    primary_rows: Sequence[ClockRow],
    support_passed: bool,
    loader: ComparatorLoader = load_comparators,
) -> tuple[dict[str, Any], int]:
    if not support_passed:
        return {
            "evaluated": False,
            "passed": False,
            "failed_checks": [],
            "reason": "source support failed before comparator access",
        }, 0
    comparators, rows_read = loader()
    return evaluate_novelty(primary_rows, comparators), rows_read


def build_report(
    *,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    write_clock: bool = True,
    comparator_loader: ComparatorLoader = load_comparators,
) -> dict[str, Any]:
    registration = verify_preregistration()
    source_rows, physical_rows = load_source()
    clocks = build_clock_rows(source_rows)
    if write_clock:
        write_clocks(clocks, clock_output)
    support = evaluate_support(clocks)
    novelty, comparator_rows_read = maybe_evaluate_novelty(
        clocks, support["passed"], comparator_loader
    )
    outcome_authorized = support["passed"] and novelty.get("passed") is True
    if not support["passed"]:
        status = "retired_before_novelty"
        next_action = "new independent candidate"
    elif not novelty.get("passed"):
        status = "retired_before_outcomes"
        next_action = "new independent candidate"
    else:
        status = "support_and_novelty_pass"
        next_action = "freeze strict train outcome evaluator"

    clock_path = _path(clock_output)
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "evaluator": {
            "path": str(EVALUATOR_SOURCE),
            "sha256": sha256_file(EVALUATOR_SOURCE),
        },
        "source": {
            "bindings": registration["source_bindings"],
            "physical_rows": physical_rows,
            "quality_ready_rows": sum(_quality_ready(row) for row in source_rows),
            "quarantined_rows": sum(not _quality_ready(row) for row in source_rows),
        },
        "clock_output": {
            "path": str(DEFAULT_CLOCK_OUTPUT),
            "sha256": sha256_file(clock_path) if write_clock else None,
            "rows": len(clocks),
            "columns": list(CLOCK_COLUMNS),
        },
        "support": support,
        "novelty": novelty,
        "outcome_boundary": {
            "prefreeze_source_value_rows_read_for_schema": 1,
            "prefreeze_comparator_clock_rows_read_for_schema": 10,
            "source_rows_read_for_support": physical_rows,
            "candidate_clock_rows_created": len(clocks),
            "comparator_clock_rows_read_for_novelty": comparator_rows_read,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_source_rows_read": 0,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
        "decision": {
            "status": status,
            "source_support_pass": support["passed"],
            "novelty_pass": novelty.get("passed") is True,
            "economic_outcomes_opened": False,
            "outcome_evaluator_authorized": outcome_authorized,
            "repair_authorized": False,
            "next_action": next_action,
        },
    }
    report["manifest_hash"] = canonical_hash(report)
    return report


def write_report(report: Mapping[str, Any], output: str | Path) -> None:
    target = _path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(clock_output=args.clock_output)
    write_report(report, args.report_output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
