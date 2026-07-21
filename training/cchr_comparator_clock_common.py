"""Outcome-blind primitives for frozen CCHR comparator clocks.

The family exporters may materialize only causal signal inputs and the exact
six-column clock projection declared here.  This module deliberately has no
dependency on legacy search/evaluation modules or outcome-bearing artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, BinaryIO, Iterable, Mapping, Sequence, cast

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIVE_MINUTES = pd.Timedelta(minutes=5)
CLOCK_COLUMNS = (
    "candidate_id",
    "split",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
)


@dataclass(frozen=True)
class ResearchSplit:
    name: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ClockCandidate:
    """One raw causal onset before split containment and non-overlap."""

    candidate_id: str
    causal_origins: tuple[datetime, ...]
    signal_time: datetime
    decision_time: datetime
    entry_time: datetime
    exit_time: datetime
    side: int


def research_splits() -> tuple[ResearchSplit, ...]:
    utc = timezone.utc
    return (
        ResearchSplit(
            "train",
            datetime(2021, 8, 8, tzinfo=utc),
            datetime(2023, 1, 1, tzinfo=utc),
        ),
        ResearchSplit(
            "selection",
            datetime(2023, 1, 1, tzinfo=utc),
            datetime(2024, 1, 1, tzinfo=utc),
        ),
    )


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def normalize_utc(value: datetime | pd.Timestamp, *, label: str) -> datetime:
    stamp = pd.Timestamp(value)
    if not isinstance(stamp, pd.Timestamp) or stamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware UTC")
    stamp = stamp.tz_convert("UTC")
    if stamp.second or stamp.microsecond or stamp.nanosecond or stamp.minute % 5:
        raise ValueError(f"{label} must be aligned to the five-minute grid")
    converted = stamp.to_pydatetime()
    if not isinstance(converted, datetime):
        raise ValueError(f"{label} must be a valid timestamp")
    return converted


def format_utc(value: datetime | pd.Timestamp) -> str:
    return normalize_utc(value, label="clock timestamp").strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_candidate(candidate: ClockCandidate) -> None:
    if (
        not candidate.candidate_id
        or candidate.candidate_id.strip() != candidate.candidate_id
    ):
        raise ValueError("candidate_id must be non-empty and byte-exact")
    if not candidate.causal_origins:
        raise ValueError("clock candidate requires at least one causal origin")
    origins = tuple(
        normalize_utc(value, label="causal origin")
        for value in candidate.causal_origins
    )
    signal = normalize_utc(candidate.signal_time, label="signal_time")
    decision = normalize_utc(candidate.decision_time, label="decision_time")
    entry = normalize_utc(candidate.entry_time, label="entry_time")
    exit_time = normalize_utc(candidate.exit_time, label="exit_time")
    if any(origin > decision for origin in origins):
        raise ValueError("causal origins cannot follow decision_time")
    if signal > decision:
        raise ValueError("signal_time cannot follow decision_time")
    if decision > entry:
        raise ValueError("decision_time cannot follow entry_time")
    if entry >= exit_time:
        raise ValueError("clock interval must be non-empty")
    hold_seconds = (exit_time - entry).total_seconds()
    if hold_seconds % FIVE_MINUTES.total_seconds():
        raise ValueError("clock hold must be an integer number of five-minute bars")
    if candidate.side not in (-1, 1):
        raise ValueError("clock side must be exactly -1 or +1")


def candidate_split(
    candidate: ClockCandidate,
    splits: Sequence[ResearchSplit] | None = None,
) -> str | None:
    """Return the unique split containing every causal and execution timestamp."""

    validate_candidate(candidate)
    declared = research_splits() if splits is None else tuple(splits)
    origins = tuple(
        normalize_utc(value, label="causal origin")
        for value in candidate.causal_origins
    )
    non_exit_times = (
        *origins,
        normalize_utc(candidate.signal_time, label="signal_time"),
        normalize_utc(candidate.decision_time, label="decision_time"),
        normalize_utc(candidate.entry_time, label="entry_time"),
    )
    exit_time = normalize_utc(candidate.exit_time, label="exit_time")
    matches: list[str] = []
    for split in declared:
        start = normalize_utc(split.start, label=f"{split.name} split start")
        end = normalize_utc(split.end, label=f"{split.name} split end")
        if start >= end:
            raise ValueError(f"split {split.name!r} must be non-empty")
        if all(start <= value < end for value in non_exit_times) and (
            start < exit_time < end
        ):
            matches.append(split.name)
    if len(matches) > 1:
        raise ValueError("clock candidate is contained by overlapping splits")
    return matches[0] if matches else None


def schedule_candidates(
    candidates: Iterable[ClockCandidate],
    *,
    splits: Sequence[ResearchSplit] | None = None,
) -> pd.DataFrame:
    """Apply containment first, then independent [entry, exit) non-overlap."""

    retained: list[tuple[ClockCandidate, str]] = []
    raw_keys: set[tuple[str, datetime]] = set()
    for candidate in candidates:
        validate_candidate(candidate)
        entry = normalize_utc(candidate.entry_time, label="entry_time")
        key = (candidate.candidate_id, entry)
        if key in raw_keys:
            raise ValueError("duplicate raw (candidate_id, entry_time)")
        raw_keys.add(key)
        split = candidate_split(candidate, splits)
        if split is not None:
            retained.append((candidate, split))

    retained.sort(
        key=lambda item: (
            item[0].candidate_id,
            normalize_utc(item[0].entry_time, label="entry_time"),
            normalize_utc(item[0].decision_time, label="decision_time"),
        )
    )
    rows: list[dict[str, Any]] = []
    prior_exit_by_id: dict[str, datetime] = {}
    for candidate, split in retained:
        entry = normalize_utc(candidate.entry_time, label="entry_time")
        exit_time = normalize_utc(candidate.exit_time, label="exit_time")
        prior_exit = prior_exit_by_id.get(candidate.candidate_id)
        if prior_exit is not None and entry < prior_exit:
            continue
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "split": split,
                "decision_time": format_utc(candidate.decision_time),
                "entry_time": format_utc(entry),
                "exit_time": format_utc(exit_time),
                "side": candidate.side,
            }
        )
        prior_exit_by_id[candidate.candidate_id] = exit_time
    data_frame = cast(Any, pd.DataFrame)
    return cast(pd.DataFrame, data_frame(rows, columns=list(CLOCK_COLUMNS)))


def validate_clock_frame(
    frame: pd.DataFrame,
    *,
    expected_candidate_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    if tuple(frame.columns) != CLOCK_COLUMNS:
        raise ValueError("clock frame must have the exact frozen six-column schema")
    normalized = frame.copy()
    normalized["candidate_id"] = normalized["candidate_id"].astype(str)
    normalized["split"] = normalized["split"].astype(str)
    duplicate_mask = normalized[["candidate_id", "entry_time"]].duplicated()
    if bool(duplicate_mask.to_numpy(dtype=bool).any()):
        raise ValueError("clock contains duplicate (candidate_id, entry_time)")
    split_series = cast(pd.Series, normalized["split"])
    valid_splits = [split.name for split in research_splits()]
    if not bool(split_series.isin(valid_splits).to_numpy(dtype=bool).all()):
        raise ValueError("clock contains an unknown split")
    side_series = cast(pd.Series, pd.to_numeric(normalized["side"], errors="raise"))
    if not bool(side_series.isin([-1, 1]).to_numpy(dtype=bool).all()):
        raise ValueError("clock contains an invalid side")
    for column in ("decision_time", "entry_time", "exit_time"):
        normalized[column] = [
            format_utc(cast(pd.Timestamp, pd.Timestamp(value)))
            for value in normalized[column]
        ]
    decisions = pd.to_datetime(normalized["decision_time"], utc=True, errors="raise")
    entries = pd.to_datetime(normalized["entry_time"], utc=True, errors="raise")
    exits = pd.to_datetime(normalized["exit_time"], utc=True, errors="raise")
    if not ((decisions <= entries) & (entries < exits)).all():
        raise ValueError("clock execution timestamps are not causal and ordered")
    observed_ids = set(normalized["candidate_id"])
    if expected_candidate_ids is not None:
        expected_ids = set(expected_candidate_ids)
        if len(expected_ids) != len(tuple(expected_candidate_ids)):
            raise ValueError("expected candidate IDs contain duplicates")
        if observed_ids != expected_ids:
            raise ValueError("clock member IDs differ from the frozen candidate map")
    normalized["side"] = side_series.astype(int)
    return normalized.sort_values(
        ["candidate_id", "entry_time"], kind="mergesort"
    ).reset_index(drop=True)


def clock_csv_bytes(
    frame: pd.DataFrame,
    *,
    expected_candidate_ids: Sequence[str] | None = None,
) -> bytes:
    normalized = validate_clock_frame(
        frame, expected_candidate_ids=expected_candidate_ids
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CLOCK_COLUMNS)
    for row in normalized.itertuples(index=False, name=None):
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def write_deterministic_gzip_clock(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    expected_candidate_ids: Sequence[str] | None = None,
) -> str:
    target = repository_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = clock_csv_bytes(frame, expected_candidate_ids=expected_candidate_ids)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as raw_handle:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, mtime=0
            ) as gzip_handle:
                gzip_handle.write(csv_bytes)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return sha256_file(target)


def read_hash_bound_columns(
    path: str | Path,
    *,
    expected_sha256: str,
    columns: Sequence[str],
    parse_dates: Sequence[str] = (),
) -> pd.DataFrame:
    """Hash first, then materialize only the frozen causal allowlist."""

    if not columns or len(set(columns)) != len(tuple(columns)):
        raise ValueError("input column allowlist must be non-empty and unique")
    if not expected_sha256 or sha256_file(path) != expected_sha256:
        raise ValueError(f"input hash differs from frozen value: {path}")
    unknown_dates = sorted(set(parse_dates) - set(columns))
    if unknown_dates:
        raise ValueError(f"parse_dates not present in input allowlist: {unknown_dates}")
    read_csv = cast(Any, pd.read_csv)
    frame = cast(
        pd.DataFrame,
        read_csv(
            repository_path(path),
            usecols=list(columns),
            parse_dates=list(parse_dates) or None,
        ),
    )
    return frame.loc[:, list(columns)]


class _BoundaryPrefixStream(io.RawIOBase):
    """Expose complete rows strictly before a timestamp boundary.

    CSV fields are scanned as opaque bytes until the physical timestamp field.
    Once the first sealed timestamp is found, no later field in that row can
    reach pandas or be materialized as a column value.
    """

    def __init__(
        self,
        path: Path,
        *,
        date_column: str,
        required_columns: Sequence[str],
        boundary: pd.Timestamp,
    ) -> None:
        super().__init__()
        source = (
            gzip.open(path, "rb")
            if path.suffix.lower() == ".gz"
            else path.open("rb", buffering=0)
        )
        self._source = cast(BinaryIO, source)
        self._boundary = boundary
        self._previous: pd.Timestamp | None = None
        self._done = False
        self.reached_boundary = False
        self.source_ended_before_boundary = False
        try:
            header = self._source.readline()
            if not header:
                raise ValueError("hash-bound prefix source has no CSV header")
            decoded_header = header.decode("utf-8-sig")
            names = next(csv.reader([decoded_header]))
            if date_column not in names:
                raise ValueError("hash-bound prefix date column is missing")
            self._date_field_index = names.index(date_column)
            missing = sorted(set(required_columns) - set(names))
            if missing:
                raise ValueError(
                    f"hash-bound prefix source is missing columns: {missing}"
                )
            bom = b"\xef\xbb\xbf"
            self._pending = bytearray(
                header[len(bom) :] if header.startswith(bom) else header
            )
        except BaseException:
            self._source.close()
            raise

    def readable(self) -> bool:
        return True

    def _one_byte(self) -> bytes:
        return self._source.read(1)

    def _physical_field(self) -> tuple[bytes, bytes, bool] | None:
        first = self._one_byte()
        if not first:
            return None
        prefix = bytearray(first)
        value = bytearray()
        if first == b",":
            return bytes(prefix), b"", False
        if first == b"\n":
            return bytes(prefix), b"", True
        if first == b"\r":
            newline = self._one_byte()
            if newline != b"\n":
                raise ValueError("prefix CSV uses an invalid CR line ending")
            prefix.extend(newline)
            return bytes(prefix), b"", True
        if first == b'"':
            while True:
                character = self._one_byte()
                if not character:
                    raise ValueError("unterminated quoted prefix date field")
                prefix.extend(character)
                if character in (b"\r", b"\n"):
                    raise ValueError("prefix date field cannot contain a newline")
                if character != b'"':
                    value.extend(character)
                    continue
                delimiter = self._one_byte()
                if delimiter == b'"':
                    prefix.extend(delimiter)
                    value.extend(b'"')
                    continue
                if delimiter:
                    prefix.extend(delimiter)
                if delimiter not in (b"", b",", b"\r", b"\n"):
                    raise ValueError("invalid delimiter after quoted prefix date field")
                if delimiter == b"\r":
                    newline = self._one_byte()
                    if newline != b"\n":
                        raise ValueError("prefix CSV uses an invalid CR line ending")
                    prefix.extend(newline)
                return bytes(prefix), bytes(value), delimiter != b","

        value.extend(first)
        while True:
            delimiter = self._one_byte()
            if not delimiter:
                return bytes(prefix), bytes(value), True
            prefix.extend(delimiter)
            if delimiter == b",":
                return bytes(prefix), bytes(value), False
            if delimiter == b"\n":
                return bytes(prefix), bytes(value), True
            if delimiter == b"\r":
                newline = self._one_byte()
                if newline != b"\n":
                    raise ValueError("prefix CSV uses an invalid CR line ending")
                prefix.extend(newline)
                return bytes(prefix), bytes(value), True
            value.extend(delimiter)

    def _date_field(self) -> tuple[bytes, str, bool] | None:
        prefix = bytearray()
        for field_index in range(self._date_field_index + 1):
            field = self._physical_field()
            if field is None:
                if field_index == 0:
                    return None
                raise ValueError("prefix CSV row ended before its date field")
            raw_field, raw_value, row_complete = field
            prefix.extend(raw_field)
            if field_index < self._date_field_index:
                if row_complete:
                    raise ValueError("prefix CSV row ended before its date field")
                continue
            return bytes(prefix), raw_value.decode("utf-8"), row_complete
        raise AssertionError("unreachable prefix date-field state")

    def _next_retained_row(self) -> bytes | None:
        field = self._date_field()
        if field is None:
            self._done = True
            self.source_ended_before_boundary = True
            return None
        prefix, raw_date, row_complete = field
        stamp_value = pd.Timestamp(raw_date)
        if not isinstance(stamp_value, pd.Timestamp) or pd.isna(stamp_value):
            raise ValueError("prefix source contains an invalid timestamp")
        stamp = stamp_value
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        if self._previous is not None and stamp < self._previous:
            raise ValueError("hash-bound prefix source must be timestamp ordered")
        self._previous = stamp
        if stamp >= self._boundary:
            self._done = True
            self.reached_boundary = True
            return None
        if row_complete:
            return prefix
        return prefix + self._source.readline()

    def readinto(self, buffer: Any) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed prefix stream")
        view = memoryview(buffer).cast("B")
        while len(self._pending) < len(view) and not self._done:
            row = self._next_retained_row()
            if row is None:
                break
            self._pending.extend(row)
        count = min(len(view), len(self._pending))
        view[:count] = self._pending[:count]
        del self._pending[:count]
        return count

    def close(self) -> None:
        if not self.closed:
            self._source.close()
        super().close()


def read_hash_bound_prefix(
    path: str | Path,
    *,
    expected_sha256: str,
    columns: Sequence[str],
    date_column: str,
    end_exclusive: str | pd.Timestamp,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Hash-bind a source, then retain only its ordered causal prefix.

    The CSV parser is restricted to the frozen allowlist.  Iteration stops as
    soon as the ordered source reaches ``end_exclusive``; no retained frame can
    contain a sealed row.
    """

    if date_column not in columns:
        raise ValueError("prefix date column must be present in the input allowlist")
    if chunksize < 1:
        raise ValueError("prefix chunksize must be positive")
    if not columns or len(set(columns)) != len(tuple(columns)):
        raise ValueError("input column allowlist must be non-empty and unique")
    if not expected_sha256 or sha256_file(path) != expected_sha256:
        raise ValueError(f"input hash differs from frozen value: {path}")
    boundary_value = pd.Timestamp(end_exclusive)
    if not isinstance(boundary_value, pd.Timestamp):
        raise ValueError("prefix boundary must be a valid timestamp")
    boundary = boundary_value
    if boundary.tzinfo is None:
        boundary = boundary.tz_localize("UTC")
    else:
        boundary = boundary.tz_convert("UTC")
    raw_stream = _BoundaryPrefixStream(
        repository_path(path),
        date_column=date_column,
        required_columns=columns,
        boundary=boundary,
    )
    buffered_stream = io.BufferedReader(raw_stream)
    read_csv = cast(Any, pd.read_csv)
    chunks: list[pd.DataFrame] = []
    previous_timestamp: pd.Timestamp | None = None
    try:
        reader = read_csv(
            buffered_stream,
            usecols=list(columns),
            chunksize=chunksize,
        )
        for raw_chunk in reader:
            chunk = cast(pd.DataFrame, raw_chunk).loc[:, list(columns)].copy()
            dates = pd.to_datetime(
                chunk[date_column], utc=True, errors="raise", format="mixed"
            )
            if not isinstance(dates, pd.Series):
                raise TypeError("prefix date parser did not return a Series")
            if not dates.is_monotonic_increasing or (
                previous_timestamp is not None
                and len(dates)
                and dates.iloc[0] < previous_timestamp
            ):
                raise ValueError("hash-bound prefix source must be timestamp ordered")
            if len(dates):
                previous_timestamp = cast(pd.Timestamp, dates.iloc[-1])
            if bool((dates >= boundary).to_numpy(dtype=bool).any()):
                raise RuntimeError("sealed row reached the pandas prefix parser")
            chunk[date_column] = dates.to_numpy()
            chunks.append(chunk)
        reached_boundary = raw_stream.reached_boundary
        source_ended_before_boundary = raw_stream.source_ended_before_boundary
    finally:
        buffered_stream.close()
    if not chunks:
        raise ValueError(f"no rows before {boundary.isoformat()} in {path}")
    frame = cast(pd.DataFrame, pd.concat(chunks, ignore_index=True))
    parsed = pd.to_datetime(frame[date_column], utc=True, errors="raise")
    if not isinstance(parsed, pd.Series) or bool(
        (parsed >= boundary).to_numpy(dtype=bool).any()
    ):
        raise RuntimeError("sealed rows escaped the hash-bound prefix loader")
    if not reached_boundary and source_ended_before_boundary:
        # A naturally truncated pre-boundary source is valid; this flag is
        # intentionally informational rather than an admission requirement.
        frame.attrs["source_ended_before_boundary"] = True
    return frame.loc[:, list(columns)].reset_index(drop=True)


def candidate_map_hash(candidate_map: Mapping[str, Mapping[str, Any]]) -> str:
    if not candidate_map or any(key.strip() != key or not key for key in candidate_map):
        raise ValueError("candidate map keys must be non-empty and byte-exact")
    return canonical_hash(dict(sorted(candidate_map.items())))
