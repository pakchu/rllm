"""Evaluate frozen ORPB-21 source support and novelty without market outcomes."""

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
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from training import preregister_overnight_rrp_participant_breadth as prereg


PROTOCOL_VERSION = "overnight_rrp_participant_breadth_support_v1"
POLICY_ID = "ORPB-21"
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = Path(
    "results/overnight_rrp_participant_breadth_preregistration_2026-07-21.json"
)
PREREGISTRATION_SHA256 = (
    "62855414b6926ff3e0f2bc37fe3c4c5c6f46f78803c66d6da564ec65de937b30"
)
PREREGISTRATION_MANIFEST_HASH = (
    "cdc0a7297df71417fe6a00198c296ddf8899b4e9d581e5a2b5f8c55b3b8ba1dd"
)
EVALUATOR_SOURCE = Path(
    "training/evaluate_overnight_rrp_participant_breadth_support.py"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "results/overnight_rrp_participant_breadth_support_clocks_2026-07-21.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/overnight_rrp_participant_breadth_source_support_2026-07-21.json"
)

UTC = timezone.utc
BAR = timedelta(minutes=5)
ONE_OPERATION = 1
OTHER_TOLERANCE = timedelta(hours=6)
LOOKBACK = 21
LOWER_TAIL = 0.10
UPPER_TAIL = 0.90
COMPARISON_START = datetime(2021, 1, 1, tzinfo=UTC)
COMPARISON_END = datetime(2024, 1, 1, tzinfo=UTC)
SPLITS = {
    "train": (COMPARISON_START, datetime(2023, 1, 1, tzinfo=UTC)),
    "selection": (datetime(2023, 1, 1, tzinfo=UTC), COMPARISON_END),
}
CONTROL_NAMES = (
    "primary",
    "amount_only_tail",
    "raw_accepted_breadth_tail",
    "participating_breadth_residual",
    "direction_flip",
    "one_release_delay",
    "deterministic_random_side",
)
SOURCE_COLUMNS = (
    "operation_id",
    "operation_date",
    "settlement_date",
    "maturity_date",
    "close_time_et",
    "result_available_at_utc",
    "last_updated_et",
    "total_amount_submitted_usd",
    "total_amount_accepted_usd",
    "participating_counterparties",
    "accepted_counterparties",
    "source_complete",
    "quarantine_reason",
)
ALLOWED_SOURCE_COLUMNS = (
    "operation_date",
    "result_available_at_utc",
    "total_amount_accepted_usd",
    "participating_counterparties",
    "accepted_counterparties",
    "source_complete",
)
ALLOWED_SOURCE_INDEXES = {
    SOURCE_COLUMNS.index(column): column for column in ALLOWED_SOURCE_COLUMNS
}
CLOCK_COLUMNS = (
    "candidate_id",
    "control",
    "split",
    "origin_operation_date",
    "operation_date",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "score",
    "rank",
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
    operation_date: date
    available_time: datetime
    amount_usd: int | None
    participating_counterparties: int | None
    accepted_counterparties: int | None
    source_complete: bool


@dataclass(frozen=True)
class RegressionFeature:
    residual: float | None
    rank: float | None


@dataclass(frozen=True)
class FeatureRow:
    source_index: int
    operation_date: date
    available_time: datetime
    amount_log: float
    accepted_log: float
    participating_log: float
    primary: RegressionFeature
    participating: RegressionFeature
    amount_rank: float
    accepted_rank: float


@dataclass(frozen=True)
class ClockRow:
    candidate_id: str
    control: str
    split: str
    origin_operation_date: date
    operation_date: date
    decision_time: datetime
    entry_time: datetime
    exit_time: datetime
    side: int
    score: float
    rank: float


@dataclass(frozen=True)
class ComparatorClock:
    candidate_id: str
    entry_time: datetime
    exit_time: datetime
    side: int
    operation_date: date | None = None


@dataclass(frozen=True)
class OrfrFeature:
    operation_date: date
    innovation: float


@dataclass(frozen=True)
class ComparatorBundle:
    clocks: dict[str, list[ComparatorClock]]
    orfr_features: list[OrfrFeature]
    rows_read: int


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
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"timestamp must be UTC: {value!r}")
    return parsed.astimezone(UTC)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("cannot serialize a naive timestamp")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_nonnegative_int(value: str, *, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid integer {field}: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"negative integer {field}")
    return parsed


def _parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"invalid source_complete value: {value!r}")


def _parse_finite_float(value: str, *, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid float {field}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite float {field}")
    return parsed


def verify_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("ORPB preregistration file hash drift")
    registration = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    if registration.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("ORPB preregistration manifest drift")
    core = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if canonical_hash(core) != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("ORPB preregistration canonical hash mismatch")
    if registration.get("policy_id") != POLICY_ID:
        raise RuntimeError("ORPB policy identity drift")
    if registration.get("later_outcome_contract", {}).get("authorized") is not False:
        raise RuntimeError("ORPB preregistration opened outcomes")
    return registration


def _project_source_record(raw_record: bytes) -> dict[str, str]:
    """Materialize only the preregistered fields from the unquoted source CSV."""
    if not raw_record.endswith(b"\n"):
        raise RuntimeError("ORPB source record lacks a newline terminator")
    if b'"' in raw_record:
        raise RuntimeError("ORPB source unexpectedly requires quoted CSV parsing")
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
                raise RuntimeError("ORPB selected source field is not UTF-8") from exc

    for position in range(end):
        value = raw_record[position]
        if value == ord(","):
            finish_field()
            field_index += 1
            selected = ALLOWED_SOURCE_INDEXES.get(field_index)
            buffer = bytearray() if selected is not None else None
        elif value in (ord("\r"), ord("\n")):
            raise RuntimeError("ORPB source contains an embedded line ending")
        elif buffer is not None:
            buffer.append(value)
    finish_field()
    if field_index + 1 != len(SOURCE_COLUMNS):
        raise RuntimeError("ORPB source data-column count drift")
    expected = tuple(
        column for column in SOURCE_COLUMNS if column in ALLOWED_SOURCE_COLUMNS
    )
    if tuple(projected) != expected:
        raise RuntimeError("ORPB source projection drift")
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
            raise RuntimeError("ORPB source header is not ASCII") from exc
        if header != SOURCE_COLUMNS:
            raise RuntimeError("ORPB source schema drift")
        for raw_record in handle:
            physical_rows += 1
            raw = _project_source_record(raw_record)
            complete = _parse_bool(raw["source_complete"])
            value_fields = (
                "total_amount_accepted_usd",
                "participating_counterparties",
                "accepted_counterparties",
            )
            if complete and any(not raw[field] for field in value_fields):
                raise RuntimeError("ORPB complete row lacks a frozen value")
            if not complete and any(raw[field] for field in value_fields):
                raise RuntimeError("ORPB quarantined row exposes a frozen value")
            rows.append(
                SourceRow(
                    operation_date=date.fromisoformat(raw["operation_date"]),
                    available_time=_parse_time(raw["result_available_at_utc"]),
                    amount_usd=(
                        _parse_nonnegative_int(
                            raw["total_amount_accepted_usd"],
                            field="total_amount_accepted_usd",
                        )
                        if complete
                        else None
                    ),
                    participating_counterparties=(
                        _parse_nonnegative_int(
                            raw["participating_counterparties"],
                            field="participating_counterparties",
                        )
                        if complete
                        else None
                    ),
                    accepted_counterparties=(
                        _parse_nonnegative_int(
                            raw["accepted_counterparties"],
                            field="accepted_counterparties",
                        )
                        if complete
                        else None
                    ),
                    source_complete=complete,
                )
            )
    if physical_rows != 1498 or len(rows) != 1498:
        raise RuntimeError("ORPB source row-count drift")
    if sum(row.source_complete for row in rows) != 1489:
        raise RuntimeError("ORPB complete source count drift")
    if sum(not row.source_complete for row in rows) != 9:
        raise RuntimeError("ORPB quarantine count drift")
    if rows[0].operation_date != date(2018, 1, 2):
        raise RuntimeError("ORPB source start-date drift")
    if rows[-1].operation_date != date(2023, 12, 29):
        raise RuntimeError("ORPB source end-date drift")
    if any(row.operation_date.year > 2023 for row in rows):
        raise RuntimeError("ORPB source opened post-2023 rows")
    if any(
        current.available_time <= previous.available_time
        for previous, current in zip(rows, rows[1:])
    ):
        raise RuntimeError("ORPB source clock is not strictly increasing")
    if len({row.operation_date for row in rows}) != len(rows):
        raise RuntimeError("ORPB source contains duplicate operation dates")
    return rows, physical_rows


def strict_prior_midrank(current: float, prior: Sequence[float]) -> float:
    if len(prior) != LOOKBACK:
        raise ValueError("ORPB strict-prior rank requires exactly 21 values")
    less = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return (less + 0.5 * equal) / LOOKBACK


def _regression_feature(
    prior_amount: Sequence[float],
    prior_breadth: Sequence[float],
    current_a: float,
    current_b: float,
) -> RegressionFeature:
    if len(prior_amount) != LOOKBACK or len(prior_breadth) != LOOKBACK:
        raise ValueError("ORPB regression requires exactly 21 prior operations")
    mean_a = math.fsum(prior_amount) / LOOKBACK
    mean_b = math.fsum(prior_breadth) / LOOKBACK
    centered_a = [value - mean_a for value in prior_amount]
    denominator = math.fsum(value * value for value in centered_a)
    if denominator == 0.0:
        return RegressionFeature(None, None)
    beta = (
        math.fsum(
            delta_a * (breadth - mean_b)
            for delta_a, breadth in zip(centered_a, prior_breadth)
        )
        / denominator
    )
    alpha = mean_b - beta * mean_a
    prior_residuals = [
        breadth - (alpha + beta * amount)
        for amount, breadth in zip(prior_amount, prior_breadth)
    ]
    current_residual = current_b - (alpha + beta * current_a)
    values = [alpha, beta, current_residual, *prior_residuals]
    if not all(math.isfinite(value) for value in values):
        return RegressionFeature(None, None)
    return RegressionFeature(
        residual=current_residual,
        rank=strict_prior_midrank(current_residual, prior_residuals),
    )


def build_features(rows: Iterable[SourceRow]) -> list[FeatureRow]:
    source_rows = list(rows)
    segment: list[tuple[float, float, float]] = []
    features: list[FeatureRow] = []
    for index, row in enumerate(source_rows):
        if not row.source_complete:
            segment.clear()
            continue
        assert row.amount_usd is not None
        assert row.accepted_counterparties is not None
        assert row.participating_counterparties is not None
        amount = math.log1p(row.amount_usd / 1_000_000_000.0)
        accepted = math.log1p(row.accepted_counterparties)
        participating = math.log1p(row.participating_counterparties)
        if len(segment) >= LOOKBACK:
            prior = segment[-LOOKBACK:]
            prior_amount = [item[0] for item in prior]
            prior_accepted = [item[1] for item in prior]
            prior_participating = [item[2] for item in prior]
            features.append(
                FeatureRow(
                    source_index=index,
                    operation_date=row.operation_date,
                    available_time=row.available_time,
                    amount_log=amount,
                    accepted_log=accepted,
                    participating_log=participating,
                    primary=_regression_feature(
                        prior_amount, prior_accepted, amount, accepted
                    ),
                    participating=_regression_feature(
                        prior_amount, prior_participating, amount, participating
                    ),
                    amount_rank=strict_prior_midrank(amount, prior_amount),
                    accepted_rank=strict_prior_midrank(accepted, prior_accepted),
                )
            )
        segment.append((amount, accepted, participating))
    return features


def _tail_side(rank: float) -> int:
    if rank <= LOWER_TAIL:
        return 1
    if rank >= UPPER_TAIL:
        return -1
    return 0


def deterministic_random_side(entry_time: datetime) -> int:
    token = f"ORPB-21-random-side-20260721|{_iso(entry_time)}".encode()
    return 1 if hashlib.sha256(token).digest()[0] < 128 else -1


def _split_for_clock(
    *,
    origin_operation_date: date,
    operation_date: date,
    decision_time: datetime,
    entry_time: datetime,
    exit_time: datetime,
) -> str | None:
    origin = datetime.combine(origin_operation_date, datetime.min.time(), UTC)
    operation = datetime.combine(operation_date, datetime.min.time(), UTC)
    for name, (start, end) in SPLITS.items():
        if (
            start <= origin < end
            and start <= operation < end
            and start <= decision_time < end
            and start <= entry_time < end
            and start < exit_time <= end
        ):
            return name
    return None


def _clock_row(
    *,
    control: str,
    origin_operation_date: date,
    operation_date: date,
    decision_time: datetime,
    exit_time: datetime,
    side: int,
    score: float,
    rank: float,
) -> ClockRow | None:
    if side not in (-1, 1):
        raise ValueError("ORPB clock side must be signed")
    entry_time = decision_time + BAR
    split = _split_for_clock(
        origin_operation_date=origin_operation_date,
        operation_date=operation_date,
        decision_time=decision_time,
        entry_time=entry_time,
        exit_time=exit_time,
    )
    if split is None:
        return None
    return ClockRow(
        candidate_id=f"{POLICY_ID}:{control}",
        control=control,
        split=split,
        origin_operation_date=origin_operation_date,
        operation_date=operation_date,
        decision_time=decision_time,
        entry_time=entry_time,
        exit_time=exit_time,
        side=side,
        score=score,
        rank=rank,
    )


def build_clock_rows(
    rows: Sequence[SourceRow], features: Sequence[FeatureRow]
) -> list[ClockRow]:
    controls: dict[str, list[ClockRow]] = {name: [] for name in CONTROL_NAMES}
    primary_by_source_index: dict[int, ClockRow] = {}
    for feature in features:
        if feature.source_index + 1 >= len(rows):
            continue
        exit_time = rows[feature.source_index + 1].available_time + BAR
        definitions = [
            (
                "amount_only_tail",
                feature.amount_log,
                feature.amount_rank,
            ),
            (
                "raw_accepted_breadth_tail",
                feature.accepted_log,
                feature.accepted_rank,
            ),
        ]
        if feature.primary.residual is not None and feature.primary.rank is not None:
            definitions.append(
                ("primary", feature.primary.residual, feature.primary.rank)
            )
        if (
            feature.participating.residual is not None
            and feature.participating.rank is not None
        ):
            definitions.append(
                (
                    "participating_breadth_residual",
                    feature.participating.residual,
                    feature.participating.rank,
                )
            )
        for control, score, rank in definitions:
            side = _tail_side(rank)
            if side == 0:
                continue
            clock = _clock_row(
                control=control,
                origin_operation_date=feature.operation_date,
                operation_date=feature.operation_date,
                decision_time=feature.available_time,
                exit_time=exit_time,
                side=side,
                score=score,
                rank=rank,
            )
            if clock is None:
                continue
            controls[control].append(clock)
            if control == "primary":
                primary_by_source_index[feature.source_index] = clock

    for source_index, primary in sorted(primary_by_source_index.items()):
        controls["direction_flip"].append(
            replace(
                primary,
                candidate_id=f"{POLICY_ID}:direction_flip",
                control="direction_flip",
                side=-primary.side,
            )
        )
        controls["deterministic_random_side"].append(
            replace(
                primary,
                candidate_id=f"{POLICY_ID}:deterministic_random_side",
                control="deterministic_random_side",
                side=deterministic_random_side(primary.entry_time),
            )
        )
        if source_index + 2 >= len(rows):
            continue
        delayed_source = rows[source_index + 1]
        delayed_exit = rows[source_index + 2].available_time + BAR
        delayed = _clock_row(
            control="one_release_delay",
            origin_operation_date=primary.origin_operation_date,
            operation_date=delayed_source.operation_date,
            decision_time=delayed_source.available_time,
            exit_time=delayed_exit,
            side=primary.side,
            score=primary.score,
            rank=primary.rank,
        )
        if delayed is not None:
            controls["one_release_delay"].append(delayed)

    ledger = [row for name in CONTROL_NAMES for row in controls[name]]
    ledger.sort(key=lambda row: (row.control, row.entry_time, row.operation_date))
    validate_clock_rows(ledger, rows)
    return ledger


def validate_clock_rows(rows: Sequence[ClockRow], source: Sequence[SourceRow]) -> None:
    source_indexes = {row.operation_date: index for index, row in enumerate(source)}
    grouped: dict[tuple[str, str], list[ClockRow]] = {}
    for row in rows:
        if row.control not in CONTROL_NAMES or row.side not in (-1, 1):
            raise RuntimeError("ORPB clock identity or side drift")
        if row.entry_time != row.decision_time + BAR:
            raise RuntimeError("ORPB clock entry latency drift")
        if not math.isfinite(row.score) or not math.isfinite(row.rank):
            raise RuntimeError("ORPB clock score is non-finite")
        if _tail_side(row.rank) != row.side and row.control not in {
            "direction_flip",
            "deterministic_random_side",
            "one_release_delay",
        }:
            raise RuntimeError("ORPB clock tail direction drift")
        split = _split_for_clock(
            origin_operation_date=row.origin_operation_date,
            operation_date=row.operation_date,
            decision_time=row.decision_time,
            entry_time=row.entry_time,
            exit_time=row.exit_time,
        )
        if split != row.split:
            raise RuntimeError("ORPB clock split-containment drift")
        index = source_indexes[row.operation_date]
        origin_index = source_indexes[row.origin_operation_date]
        if row.decision_time != source[index].available_time:
            raise RuntimeError("ORPB decision clock drift")
        if row.exit_time != source[index + 1].available_time + BAR:
            raise RuntimeError("ORPB next-operation exit drift")
        if row.control == "one_release_delay":
            if index != origin_index + 1:
                raise RuntimeError("ORPB delayed-control origin drift")
        elif index != origin_index:
            raise RuntimeError("ORPB primary-clock origin drift")
        grouped.setdefault((row.control, row.split), []).append(row)
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: row.entry_time)
        if len({row.entry_time for row in ordered}) != len(ordered):
            raise RuntimeError(f"ORPB duplicate clock entry: {key}")
        if any(
            current.entry_time < previous.exit_time
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise RuntimeError(f"ORPB overlapping clock: {key}")

    by_control = {
        control: [row for row in rows if row.control == control]
        for control in CONTROL_NAMES
    }
    primary = by_control["primary"]
    if [
        (row.entry_time, row.exit_time, row.origin_operation_date, row.side)
        for row in by_control["direction_flip"]
    ] != [
        (row.entry_time, row.exit_time, row.origin_operation_date, -row.side)
        for row in primary
    ]:
        raise RuntimeError("ORPB direction-flip control drift")
    if any(
        row.side != deterministic_random_side(row.entry_time)
        for row in by_control["deterministic_random_side"]
    ):
        raise RuntimeError("ORPB random-side control drift")


def _clock_values(row: ClockRow) -> tuple[str | int, ...]:
    return (
        row.candidate_id,
        row.control,
        row.split,
        row.origin_operation_date.isoformat(),
        row.operation_date.isoformat(),
        _iso(row.decision_time),
        _iso(row.entry_time),
        _iso(row.exit_time),
        row.side,
        format(row.score, ".17g"),
        format(row.rank, ".17g"),
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
    years = Counter(str(row.entry_time.year) for row in selected)
    halves = Counter(
        f"{row.entry_time.year}H{1 if row.entry_time.month <= 6 else 2}"
        for row in selected
    )
    months = Counter(row.entry_time.strftime("%Y-%m") for row in selected)
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


def _feature_windows_are_prior_only(
    features: Sequence[FeatureRow], source: Sequence[SourceRow]
) -> bool:
    for feature in features:
        if feature.source_index < LOOKBACK:
            return False
        prior = source[feature.source_index - LOOKBACK : feature.source_index]
        if len(prior) != LOOKBACK or not all(row.source_complete for row in prior):
            return False
        if source[feature.source_index].operation_date != feature.operation_date:
            return False
    return True


def _optional_close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _independent_regression_replay(
    prior_amount: Sequence[float],
    prior_breadth: Sequence[float],
    current_amount: float,
    current_breadth: float,
) -> RegressionFeature:
    amount = np.asarray(prior_amount, dtype=float)
    breadth = np.asarray(prior_breadth, dtype=float)
    if len(amount) != LOOKBACK or len(breadth) != LOOKBACK:
        raise RuntimeError("ORPB independent replay received a non-21 window")
    if bool(np.all(amount == amount[0])):
        return RegressionFeature(None, None)
    design = np.column_stack((np.ones(LOOKBACK, dtype=float), amount))
    coefficients, _, _, _ = np.linalg.lstsq(design, breadth, rcond=None)
    prior_residuals = breadth - design @ coefficients
    current_residual = current_breadth - (
        coefficients[0] + coefficients[1] * current_amount
    )
    values = [current_residual, *prior_residuals.tolist()]
    if not all(math.isfinite(float(value)) for value in values):
        return RegressionFeature(None, None)
    less = int(np.sum(prior_residuals < current_residual))
    equal = int(np.sum(prior_residuals == current_residual))
    return RegressionFeature(
        residual=float(current_residual),
        rank=(less + 0.5 * equal) / LOOKBACK,
    )


def independent_replay_checks(
    features: Sequence[FeatureRow],
    clocks: Sequence[ClockRow],
    source: Sequence[SourceRow],
) -> dict[str, bool]:
    expected: dict[int, FeatureRow] = {}
    segment: list[tuple[float, float, float]] = []
    for index, row in enumerate(source):
        if not row.source_complete:
            segment.clear()
            continue
        assert row.amount_usd is not None
        assert row.accepted_counterparties is not None
        assert row.participating_counterparties is not None
        amount = math.log1p(row.amount_usd / 1_000_000_000.0)
        accepted = math.log1p(row.accepted_counterparties)
        participating = math.log1p(row.participating_counterparties)
        if len(segment) >= LOOKBACK:
            prior = segment[-LOOKBACK:]
            prior_amount = [item[0] for item in prior]
            prior_accepted = [item[1] for item in prior]
            prior_participating = [item[2] for item in prior]
            amount_less = sum(value < amount for value in prior_amount)
            amount_equal = sum(value == amount for value in prior_amount)
            accepted_less = sum(value < accepted for value in prior_accepted)
            accepted_equal = sum(value == accepted for value in prior_accepted)
            expected[index] = FeatureRow(
                source_index=index,
                operation_date=row.operation_date,
                available_time=row.available_time,
                amount_log=amount,
                accepted_log=accepted,
                participating_log=participating,
                primary=_independent_regression_replay(
                    prior_amount, prior_accepted, amount, accepted
                ),
                participating=_independent_regression_replay(
                    prior_amount, prior_participating, amount, participating
                ),
                amount_rank=(amount_less + 0.5 * amount_equal) / LOOKBACK,
                accepted_rank=(accepted_less + 0.5 * accepted_equal) / LOOKBACK,
            )
        segment.append((amount, accepted, participating))

    observed = {feature.source_index: feature for feature in features}
    index_set_exact = observed.keys() == expected.keys()
    ols_exact = index_set_exact
    rank_exact = index_set_exact
    for index, expected_feature in expected.items():
        observed_feature = observed.get(index)
        if observed_feature is None:
            ols_exact = False
            rank_exact = False
            continue
        ols_exact = ols_exact and all(
            (
                _optional_close(
                    observed_feature.primary.residual,
                    expected_feature.primary.residual,
                ),
                _optional_close(
                    observed_feature.participating.residual,
                    expected_feature.participating.residual,
                ),
            )
        )
        rank_exact = rank_exact and all(
            (
                _optional_close(
                    observed_feature.primary.rank, expected_feature.primary.rank
                ),
                _optional_close(
                    observed_feature.participating.rank,
                    expected_feature.participating.rank,
                ),
                _optional_close(
                    observed_feature.amount_rank, expected_feature.amount_rank
                ),
                _optional_close(
                    observed_feature.accepted_rank, expected_feature.accepted_rank
                ),
            )
        )

    expected_primary: dict[date, tuple[float, float, int]] = {}
    for index, feature in expected.items():
        residual = feature.primary.residual
        rank = feature.primary.rank
        if residual is None or rank is None or index + 1 >= len(source):
            continue
        side = 1 if rank <= LOWER_TAIL else -1 if rank >= UPPER_TAIL else 0
        if side == 0:
            continue
        row = source[index]
        entry = row.available_time + BAR
        exit_time = source[index + 1].available_time + BAR
        if (
            _split_for_clock(
                origin_operation_date=row.operation_date,
                operation_date=row.operation_date,
                decision_time=row.available_time,
                entry_time=entry,
                exit_time=exit_time,
            )
            is not None
        ):
            expected_primary[row.operation_date] = (residual, rank, side)
    observed_primary = {
        row.operation_date: row for row in clocks if row.control == "primary"
    }
    clock_exact = observed_primary.keys() == expected_primary.keys()
    for operation_date, (residual, rank, side) in expected_primary.items():
        row = observed_primary.get(operation_date)
        if row is None:
            clock_exact = False
            continue
        clock_exact = clock_exact and (
            _optional_close(row.score, residual)
            and _optional_close(row.rank, rank)
            and row.side == side
        )
    return {
        "feature_index_set_exact": index_set_exact,
        "prior_only_ols_replay_exact": ols_exact,
        "prior_only_rank_replay_exact": rank_exact,
        "primary_clock_score_rank_replay_exact": clock_exact,
    }


def evaluate_support(
    clocks: Sequence[ClockRow],
    source: Sequence[SourceRow],
    features: Sequence[FeatureRow],
) -> dict[str, Any]:
    grouped = {
        control: [row for row in clocks if row.control == control]
        for control in CONTROL_NAMES
    }
    summaries = {
        control: {split: summarize(rows, split) for split in SPLITS}
        for control, rows in grouped.items()
    }
    train = summaries["primary"]["train"]
    selection = summaries["primary"]["selection"]
    replay = independent_replay_checks(features, clocks, source)
    integrity = {
        "source_binding_hashes_exact": True,
        "source_schema_exact": True,
        "source_row_and_quarantine_counts_exact": (
            len(source) == 1498
            and sum(row.source_complete for row in source) == 1489
            and sum(not row.source_complete for row in source) == 9
        ),
        "quarantined_values_blank": all(
            (
                row.amount_usd is not None
                and row.participating_counterparties is not None
                and row.accepted_counterparties is not None
            )
            if row.source_complete
            else (
                row.amount_usd is None
                and row.participating_counterparties is None
                and row.accepted_counterparties is None
            )
            for row in source
        ),
        "quarantine_clears_feature_window": _feature_windows_are_prior_only(
            features, source
        ),
        **replay,
        "decision_clock_exact": True,
        "entry_delay_exact": all(
            row.entry_time == row.decision_time + BAR for row in clocks
        ),
        "next_operation_exit_exact": True,
        "last_source_row_omitted": all(
            row.operation_date != source[-1].operation_date for row in clocks
        ),
        "split_containment_exact": True,
        "nonoverlap_exact": True,
    }
    checks = {
        "train_events_min": train["events"] >= 50,
        "train_events_max": train["events"] <= 130,
        "train_each_year_min": all(
            train["year_counts"].get(str(year), 0) >= 20 for year in (2021, 2022)
        ),
        "train_side_balance": min(train["long_share"], train["short_share"]) >= 0.25,
        "train_month_concentration": train["maximum_month_share"] <= 0.20,
        "selection_events_min": selection["events"] >= 25,
        "selection_events_max": selection["events"] <= 80,
        "selection_each_half_min": all(
            selection["half_counts"].get(name, 0) >= 8 for name in ("2023H1", "2023H2")
        ),
        "selection_side_balance": min(selection["long_share"], selection["short_share"])
        >= 0.20,
        "selection_month_concentration": selection["maximum_month_share"] <= 0.25,
        **{f"integrity:{name}": value for name, value in integrity.items()},
    }
    return {
        "summaries": summaries,
        "integrity": integrity,
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


def _group_id(binding: Mapping[str, Any], raw: Mapping[str, str]) -> str:
    identifiers = binding["identifier_columns"]
    if identifiers:
        return "|".join(raw[column] for column in identifiers)
    fixed = binding.get("fixed_candidate_id")
    if not isinstance(fixed, str):
        raise RuntimeError("ORPB comparator lacks a fixed candidate identity")
    return fixed


def _comparator_fields(
    family: str, raw: Mapping[str, str]
) -> tuple[datetime, datetime, datetime, int, date | None]:
    if family == "orfr_clocks":
        decision = _parse_time(raw["signal_time"])
        entry = _parse_time(raw["entry_time"])
        exit_time = _parse_time(raw["exit_time"])
        operation_date = date.fromisoformat(raw["operation_date"])
    elif family == "flcc":
        decision = _parse_time(raw["signal_time"])
        entry = _parse_time(raw["entry_time"])
        exit_time = _parse_time(raw["exit_time"])
        operation_date = None
    elif family in {"dffb_primary", "dffb_controls"}:
        decision = _parse_time(raw["decision_time_utc"])
        entry = _parse_time(raw["entry_time_utc"])
        exit_time = _parse_time(raw["exit_time_utc"])
        operation_date = None
    elif family == "sfrd":
        decision = _parse_time(raw["sofr_available_at_utc"])
        entry = _parse_time(raw["entry_time"])
        exit_time = _parse_time(raw["exit_time"])
        operation_date = None
    elif family == "bdrc":
        decision = _parse_time(raw["decision_time"])
        entry = _parse_time(raw["entry_time"])
        exit_time = _parse_time(raw["exit_time"])
        operation_date = None
    else:
        raise RuntimeError(f"unknown ORPB comparator family: {family}")
    try:
        side = int(raw["side"])
    except ValueError as exc:
        raise RuntimeError(f"invalid ORPB comparator side: {family}") from exc
    return decision, entry, exit_time, side, operation_date


def _include_comparator_interval(
    *, family: str, decision: datetime, entry: datetime, exit_time: datetime
) -> bool:
    if not decision <= entry < exit_time:
        raise RuntimeError(f"invalid comparator clock order: {family}")
    for name, value in (("decision", decision), ("entry", entry), ("exit", exit_time)):
        if (value - COMPARISON_START) % BAR != timedelta(0):
            raise RuntimeError(f"off-grid comparator {name}: {family}")
    if exit_time <= COMPARISON_START:
        return False
    if decision < COMPARISON_START or entry < COMPARISON_START:
        raise RuntimeError(f"comparator straddles novelty start: {family}")
    if (
        decision >= COMPARISON_END
        or entry >= COMPARISON_END
        or exit_time > COMPARISON_END
    ):
        raise RuntimeError(f"comparator opened post-2023 clock: {family}")
    return True


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
    if any(
        current.entry_time < previous.exit_time
        for previous, current in zip(ordered, ordered[1:])
    ):
        raise RuntimeError(f"overlapping comparator: {candidate_id}")


def load_comparators() -> ComparatorBundle:
    groups: dict[str, list[ComparatorClock]] = {}
    orfr_features: list[OrfrFeature] = []
    rows_read = 0
    for family, binding in prereg.COMPARATOR_BINDINGS.items():
        prereg.verify_binding(binding)
        counts: Counter[str] = Counter()
        family_rows = 0
        with gzip.open(
            _path(binding["path"]), "rt", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames or ())
            if header != tuple(binding["columns"]):
                raise RuntimeError(f"comparator schema drift: {family}")
            if _forbidden_header(header):
                raise RuntimeError(f"comparator outcome field: {family}")
            for raw in reader:
                rows_read += 1
                family_rows += 1
                group = _group_id(binding, raw)
                counts[group] += 1
                if family == "orfr_features":
                    decision = _parse_time(raw["decision_time"])
                    entry = _parse_time(raw["entry_time"])
                    exit_time = _parse_time(raw["scheduled_exit_time"])
                    if _include_comparator_interval(
                        family=family,
                        decision=decision,
                        entry=entry,
                        exit_time=exit_time,
                    ):
                        orfr_features.append(
                            OrfrFeature(
                                operation_date=date.fromisoformat(
                                    raw["operation_date"]
                                ),
                                innovation=_parse_finite_float(
                                    raw["innovation"], field="ORFR innovation"
                                ),
                            )
                        )
                    continue
                decision, entry, exit_time, side, operation_date = _comparator_fields(
                    family, raw
                )
                if not _include_comparator_interval(
                    family=family,
                    decision=decision,
                    entry=entry,
                    exit_time=exit_time,
                ):
                    continue
                candidate_id = f"{family}:{group}"
                groups.setdefault(candidate_id, []).append(
                    ComparatorClock(
                        candidate_id=candidate_id,
                        entry_time=entry,
                        exit_time=exit_time,
                        side=side,
                        operation_date=operation_date,
                    )
                )
        if family_rows != binding["expected_rows"]:
            raise RuntimeError(f"comparator row-count drift: {family}")
        if dict(sorted(counts.items())) != dict(
            sorted(binding["required_groups"].items())
        ):
            raise RuntimeError(f"comparator candidate cohort drift: {family}")
    if rows_read != 7104:
        raise RuntimeError("ORPB comparator total row-count drift")
    for candidate_id, rows in groups.items():
        _validate_comparator_group(rows, candidate_id)
    expected_clock_groups = 44
    if len(groups) != expected_clock_groups:
        raise RuntimeError("ORPB comparator clock-group count drift")
    if len({row.operation_date for row in orfr_features}) != len(orfr_features):
        raise RuntimeError("ORPB duplicate ORFR feature operation date")
    if not orfr_features:
        raise RuntimeError("ORPB empty ORFR feature comparator")
    return ComparatorBundle(groups, orfr_features, rows_read)


def exact_entry_jaccard(
    left: Sequence[ComparatorClock], right: Sequence[ComparatorClock]
) -> tuple[float, int]:
    left_entries = {row.entry_time for row in left}
    right_entries = {row.entry_time for row in right}
    matches = len(left_entries & right_entries)
    union = len(left_entries | right_entries)
    return (matches / union if union else 0.0), matches


def maximum_tolerant_matches(
    left: Sequence[ComparatorClock],
    right: Sequence[ComparatorClock],
    tolerance: timedelta = OTHER_TOLERANCE,
) -> int:
    left_entries = sorted(row.entry_time for row in left)
    right_entries = sorted(row.entry_time for row in right)
    left_index = 0
    right_index = 0
    matches = 0
    while left_index < len(left_entries) and right_index < len(right_entries):
        if right_entries[right_index] < left_entries[left_index] - tolerance:
            right_index += 1
        elif right_entries[right_index] > left_entries[left_index] + tolerance:
            left_index += 1
        else:
            matches += 1
            left_index += 1
            right_index += 1
    return matches


def maximum_operation_matches(
    left: Sequence[ComparatorClock],
    right: Sequence[ComparatorClock],
    operation_indexes: Mapping[date, int],
) -> int:
    def operation_index(row: ComparatorClock) -> int:
        if row.operation_date is None:
            raise RuntimeError("ORPB/ORFR comparator lacks an operation date")
        return operation_indexes[row.operation_date]

    try:
        left_indexes = sorted(operation_index(row) for row in left)
        right_indexes = sorted(operation_index(row) for row in right)
    except KeyError as exc:
        raise RuntimeError("ORPB/ORFR operation-date identity drift") from exc
    left_index = 0
    right_index = 0
    matches = 0
    while left_index < len(left_indexes) and right_index < len(right_indexes):
        if right_indexes[right_index] < left_indexes[left_index] - ONE_OPERATION:
            right_index += 1
        elif right_indexes[right_index] > left_indexes[left_index] + ONE_OPERATION:
            left_index += 1
        else:
            matches += 1
            left_index += 1
            right_index += 1
    return matches


def _exposure(rows: Sequence[ComparatorClock]) -> np.ndarray:
    bars = int((COMPARISON_END - COMPARISON_START) / BAR)
    values = np.zeros(bars, dtype=np.int8)
    for row in rows:
        first = int((row.entry_time - COMPARISON_START) / BAR)
        last = int((row.exit_time - COMPARISON_START) / BAR)
        if first < 0 or last > bars or first >= last:
            raise RuntimeError("ORPB comparator exposure left frozen grid")
        if bool(values[first:last].any()):
            raise RuntimeError("ORPB comparator exposure overlaps")
        values[first:last] = row.side
    return values


def signed_exposure_correlation(
    left: Sequence[ComparatorClock], right: Sequence[ComparatorClock]
) -> float:
    left_exposure = _exposure(left)
    right_exposure = _exposure(right)
    if float(left_exposure.std()) == 0.0 or float(right_exposure.std()) == 0.0:
        raise RuntimeError("ORPB comparator exposure has zero variance")
    correlation = float(np.corrcoef(left_exposure, right_exposure)[0, 1])
    if not math.isfinite(correlation):
        raise RuntimeError("ORPB exposure correlation is non-finite")
    return correlation


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise RuntimeError("ORPB Spearman comparison lacks common observations")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if float(left_ranks.std()) == 0.0 or float(right_ranks.std()) == 0.0:
        raise RuntimeError("ORPB Spearman rank has zero variance")
    correlation = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
    if not math.isfinite(correlation):
        raise RuntimeError("ORPB Spearman correlation is non-finite")
    return correlation


def _containment(
    matches: int, left_count: int, right_count: int
) -> tuple[float, float, float]:
    if left_count == 0 or right_count == 0:
        raise RuntimeError("ORPB novelty comparator is empty")
    left_share = matches / left_count
    right_share = matches / right_count
    return left_share, right_share, max(left_share, right_share)


def evaluate_novelty(
    primary_rows: Sequence[ClockRow],
    features: Sequence[FeatureRow],
    source: Sequence[SourceRow],
    bundle: ComparatorBundle,
) -> dict[str, Any]:
    primary = [
        ComparatorClock(
            candidate_id=POLICY_ID,
            entry_time=row.entry_time,
            exit_time=row.exit_time,
            side=row.side,
            operation_date=row.operation_date,
        )
        for row in primary_rows
        if row.control == "primary"
    ]
    _validate_comparator_group(primary, POLICY_ID)
    operation_indexes = {row.operation_date: index for index, row in enumerate(source)}
    metrics: dict[str, dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for candidate_id, comparator in sorted(bundle.clocks.items()):
        jaccard, exact_matches = exact_entry_jaccard(primary, comparator)
        correlation = signed_exposure_correlation(primary, comparator)
        if candidate_id.startswith("orfr_clocks:"):
            tolerant_matches = maximum_operation_matches(
                primary, comparator, operation_indexes
            )
            left_share, right_share, max_share = _containment(
                tolerant_matches, len(primary), len(comparator)
            )
            metrics[candidate_id] = {
                "orpb_events": len(primary),
                "comparator_events": len(comparator),
                "exact_entry_matches": exact_matches,
                "exact_entry_jaccard": jaccard,
                "maximum_one_to_one_matches_within_one_operation": tolerant_matches,
                "orpb_tolerant_coverage": left_share,
                "comparator_tolerant_coverage": right_share,
                "maximum_bidirectional_containment": max_share,
                "signed_occupied_exposure_correlation": correlation,
            }
            checks[f"{candidate_id}:exact_entry_jaccard"] = jaccard <= 0.15
            checks[f"{candidate_id}:operation_containment"] = max_share <= 0.35
            checks[f"{candidate_id}:exposure_correlation"] = abs(correlation) <= 0.35
        else:
            tolerant_matches = maximum_tolerant_matches(primary, comparator)
            left_share, right_share, max_share = _containment(
                tolerant_matches, len(primary), len(comparator)
            )
            metrics[candidate_id] = {
                "orpb_events": len(primary),
                "comparator_events": len(comparator),
                "exact_entry_matches": exact_matches,
                "exact_entry_jaccard": jaccard,
                "maximum_one_to_one_matches_within_6h": tolerant_matches,
                "orpb_tolerant_coverage": left_share,
                "comparator_tolerant_coverage": right_share,
                "maximum_bidirectional_containment": max_share,
                "signed_occupied_exposure_correlation": correlation,
            }
            checks[f"{candidate_id}:exact_entry_jaccard"] = jaccard <= 0.10
            checks[f"{candidate_id}:six_hour_containment"] = max_share <= 0.25
            checks[f"{candidate_id}:exposure_correlation"] = abs(correlation) <= 0.35

    orpb_residuals = {
        feature.operation_date: feature.primary.residual
        for feature in features
        if feature.primary.residual is not None
        and COMPARISON_START.date() <= feature.operation_date < COMPARISON_END.date()
    }
    orfr_innovations = {
        feature.operation_date: feature.innovation for feature in bundle.orfr_features
    }
    common_dates = sorted(orpb_residuals.keys() & orfr_innovations.keys())
    residual_correlation = spearman_correlation(
        [float(orpb_residuals[current]) for current in common_dates],
        [orfr_innovations[current] for current in common_dates],
    )
    checks["orfr_features:residual_amount_innovation_spearman"] = (
        abs(residual_correlation) <= 0.35
    )
    return {
        "evaluated": True,
        "comparison_start": _iso(COMPARISON_START),
        "comparison_end_exclusive": _iso(COMPARISON_END),
        "comparator_clock_groups": len(bundle.clocks),
        "comparator_rows_read": bundle.rows_read,
        "metrics": metrics,
        "orfr_residual_amount_innovation": {
            "common_operation_dates": len(common_dates),
            "spearman_correlation": residual_correlation,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
    }


ComparatorLoader = Callable[[], ComparatorBundle]


def maybe_evaluate_novelty(
    primary_rows: Sequence[ClockRow],
    features: Sequence[FeatureRow],
    source: Sequence[SourceRow],
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
    bundle = loader()
    return evaluate_novelty(primary_rows, features, source, bundle), bundle.rows_read


def build_report(
    *,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    write_clock: bool = True,
    comparator_loader: ComparatorLoader = load_comparators,
) -> dict[str, Any]:
    registration = verify_preregistration()
    source, physical_rows = load_source()
    features = build_features(source)
    clocks = build_clock_rows(source, features)
    if write_clock:
        write_clocks(clocks, clock_output)
    support = evaluate_support(clocks, source, features)
    novelty, comparator_rows_read = maybe_evaluate_novelty(
        clocks, features, source, support["passed"], comparator_loader
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
        next_action = "freeze separate strict train outcome evaluator"

    clock_path = _path(clock_output)
    core: dict[str, Any] = {
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
            "bindings": registration["source"]["bindings"],
            "physical_rows": physical_rows,
            "complete_rows": sum(row.source_complete for row in source),
            "quarantined_rows": sum(not row.source_complete for row in source),
            "features_computed": len(features),
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
            "source_rows_read_for_support": physical_rows,
            "candidate_clock_rows_created": len(clocks),
            "comparator_rows_read_for_novelty": comparator_rows_read,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_source_rows_read": 0,
            "economic_outcomes_opened": False,
        },
        "decision": {
            "status": status,
            "next_action": next_action,
            "support_passed": support["passed"],
            "novelty_passed": novelty.get("passed") is True,
            "economic_outcomes_opened": False,
            "outcome_evaluator_authorized": outcome_authorized,
            "repair_authorized": False,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def write_report(report: Mapping[str, Any], output: str | Path) -> None:
    target = _path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_REPORT_OUTPUT))
    parser.add_argument("--clocks", default=str(DEFAULT_CLOCK_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(clock_output=args.clocks)
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
