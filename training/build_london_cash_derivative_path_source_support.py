"""Build outcome-blind LCDP-D1 source and categorical-token support.

The official ``run`` subcommand is unavailable until the committed runner and
tests are bound by a write-once execution seal. The support stage never opens
funding, execution prices, future returns, rewards, model rows, actions,
trades, PnL, CAGR, MDD, or at-or-after-2023 non-date source values.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training import preregister_london_cash_derivative_path as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = prereg.POLICY_ID
PROTOCOL_VERSION = "london_cash_derivative_path_source_support_v1"
SEAL_PROTOCOL_VERSION = "lcdp_d1_source_support_execution_seal_v1"

RUNNER_PATH = "training/build_london_cash_derivative_path_source_support.py"
TEST_PATH = "tests/test_build_london_cash_derivative_path_source_support.py"
EXECUTION_SEAL_PATH = (
    "results/lcdp_d1_source_support_execution_seal_2026-07-25.json"
)
TOKEN_OUTPUT = "data/lcdp_d1_source_support/token_support.csv.gz"
PASS_REPORT = "results/lcdp_d1_source_support_2026-07-25.json"
REJECTION_REPORT = (
    "results/lcdp_d1_source_support_rejection_2026-07-25.json"
)

CONTRACT_PATH = "docs/lcdp-d1-source-support-implementation-contract-2026-07-25.md"
CONTRACT_SHA256 = (
    "bcf83b989458484f396ab0e180f336bdd0cd0d36b4d4ecfda6ac7aa2d185f312"
)
CONTRACT_COMMIT = "21583d6b4ab5835bf98ac102a0f809834754849a"
BOUNDARY_PATH = prereg.BOUNDARY_DOCUMENT
BOUNDARY_SHA256 = prereg.BOUNDARY_DOCUMENT_SHA256
PREREGISTRATION_PATH = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "da0dd2f24236c3b64e31604268b0ad9d9b342723629790d1ecf061d0a02f4ad4"
)
PREREGISTRATION_MANIFEST_HASH = (
    "0cbeeaad957c67187381405681e8e7935c7039c7c9f9e2d0a19cbe5e912d5dac"
)
PREREGISTRATION_PRODUCER_SHA256 = (
    "a20118ab1b7cfe1a12c050bfc4a612689286383d9fa8fd2043dfcce62fd7368f"
)

OUTPUT_COLUMNS = (
    "london_date",
    "boundary_utc",
    "expected_slots",
    "source_state",
    "rank_ready",
    "model_eligible",
    *prereg.TOKEN_COLUMNS,
    "primary_line",
    "primary_sequence_hash",
    *prereg.CONTROL_IDS,
)

READY = "READY"
SOURCE_INVALID = "SOURCE_INVALID"
SOURCE_INVALID_START = "SOURCE_INVALID_START"
RANK_UNREADY = "RANK_UNREADY"
CONTROL_UNREADY = "CONTROL_UNREADY"
SAFETY_OR_CONTROL_TOKENS = frozenset(
    (*prereg.PRIMARY_SAFETY_TOKENS, *prereg.CONTROL_TOKENS)
)


@dataclass(frozen=True)
class FrozenPolicy:
    source_start_inclusive: str = "2020-01-01"
    source_end_exclusive: str = "2023-01-01"
    rank_lookback_lines: int = 126
    minimum_prior_valid_values: int = 63
    sequence_lines: int = 21
    annual_valid_share_min: float = 0.97
    quarter_valid_share_min: float = 0.95
    ready_2020_min: int = 280
    ready_2021_2022_min: int = 350
    ready_post_2020q1_quarter_min: int = 80
    category_share_min: float = 0.03
    category_share_max: float = 0.94
    control_difference_share_min: float = 0.05


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    quote_notional: float
    source_complete: bool


@dataclass(frozen=True)
class VenueMetrics:
    start_price: float
    end_price: float
    total_return: float
    first_return: float
    second_return: float
    efficiency: float
    log_range: float
    quote_notional: float


@dataclass(frozen=True)
class DayPair:
    london_date: date
    cash: VenueMetrics | None
    perp: VenueMetrics | None
    cash_reason: str
    perp_reason: str

    @property
    def source_valid(self) -> bool:
        return self.cash is not None and self.perp is not None


@dataclass(frozen=True)
class ParseAudit:
    path: str
    physical_date_rows: int
    selected_non_date_rows: int
    post_cutoff_date_only_rows: int
    at_or_after_2023_date_only_rows: int
    post_cutoff_non_date_rows: int
    at_or_after_2023_non_date_rows: int
    first_timestamp: str
    last_timestamp: str


@dataclass(frozen=True)
class StreamLine:
    london_date: date
    state: str
    tokens: dict[str, str]
    serialized: str
    cash_sign: int
    perp_sign: int

    @property
    def ready(self) -> bool:
        return self.state == READY


@dataclass
class SourceInputs:
    end_exclusive: str
    dates: list[date]
    pairs: list[DayPair]
    coinbase_audit: ParseAudit
    binance_audit: ParseAudit


@dataclass
class SupportBundle(SourceInputs):
    primary: list[StreamLine]
    relational_controls: dict[str, list[StreamLine]]
    derived_controls: dict[str, list[str]]
    rows: list[dict[str, Any]]


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(repository_path(path).read_bytes()).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_tracked_clean(path: str) -> str:
    _git_output("ls-files", "--error-unmatch", "--", path)
    for args in (
        ("diff", "--quiet", "--", path),
        ("diff", "--cached", "--quiet", "--", path),
    ):
        if subprocess.run(
            ("git", *args),
            cwd=REPOSITORY_ROOT,
            check=False,
        ).returncode:
            raise RuntimeError(f"LCDP sealed path is dirty: {path}")
    return _git_output("log", "-1", "--format=%H", "--", path)


def _safe_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if (
        str(path).startswith("~")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("LCDP output path must be repository-relative")
    target = REPOSITORY_ROOT / candidate
    root = REPOSITORY_ROOT.resolve(strict=True)
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise RuntimeError("LCDP output path escapes repository") from error
    current = REPOSITORY_ROOT
    for part in candidate.parent.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("LCDP output parent contains a symlink")
    return target


def write_once_bytes(path: str | Path, content: bytes) -> str:
    target = _safe_output_path(path)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise RuntimeError("LCDP write-once target is not a regular file")
        if target.read_bytes() != content:
            raise RuntimeError(f"LCDP write-once artifact drift: {target}")
        return hashlib.sha256(content).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise RuntimeError("LCDP output parent contains a symlink")
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
    except FileExistsError:
        if target.is_symlink() or not target.is_file():
            raise RuntimeError("LCDP write-once target is not a regular file")
        if target.read_bytes() != content:
            raise RuntimeError(f"LCDP write-once artifact drift: {target}")
        return hashlib.sha256(content).hexdigest()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return hashlib.sha256(content).hexdigest()


def write_once_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    content = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return write_once_bytes(path, content)


def _validate_policy(policy: FrozenPolicy) -> None:
    if policy != FrozenPolicy():
        raise ValueError("LCDP source-support policy is frozen")


def build_execution_seal() -> dict[str, Any]:
    runner_commit = _assert_tracked_clean(RUNNER_PATH)
    test_commit = _assert_tracked_clean(TEST_PATH)
    head = _git_output("rev-parse", "HEAD")
    if runner_commit != test_commit or runner_commit != head:
        raise RuntimeError(
            "LCDP execution seal requires runner and tests at current HEAD"
        )
    if _assert_tracked_clean(CONTRACT_PATH) != CONTRACT_COMMIT:
        raise RuntimeError("LCDP implementation contract commit drift")
    core = {
        "protocol_version": SEAL_PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "source_values_opened": False,
        "outcomes_opened": False,
        "runner": {
            "path": RUNNER_PATH,
            "commit": runner_commit,
            "sha256": sha256_file(RUNNER_PATH),
        },
        "tests": {
            "path": TEST_PATH,
            "commit": test_commit,
            "sha256": sha256_file(TEST_PATH),
        },
        "contract": {
            "path": CONTRACT_PATH,
            "commit": CONTRACT_COMMIT,
            "sha256": CONTRACT_SHA256,
        },
        "boundary": {
            "path": BOUNDARY_PATH,
            "sha256": BOUNDARY_SHA256,
        },
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def create_execution_seal() -> dict[str, Any]:
    seal = build_execution_seal()
    write_once_json(EXECUTION_SEAL_PATH, seal)
    return seal


def validate_execution_seal() -> dict[str, Any]:
    seal_path = repository_path(EXECUTION_SEAL_PATH)
    if not seal_path.is_file() or seal_path.is_symlink():
        raise RuntimeError("LCDP execution seal is absent or unsafe")
    _assert_tracked_clean(EXECUTION_SEAL_PATH)
    seal = json.loads(seal_path.read_text())
    expected_keys = {
        "protocol_version",
        "policy_id",
        "source_values_opened",
        "outcomes_opened",
        "runner",
        "tests",
        "contract",
        "boundary",
        "preregistration",
        "manifest_hash",
    }
    if set(seal) != expected_keys:
        raise RuntimeError("LCDP execution seal schema mismatch")
    core = {key: value for key, value in seal.items() if key != "manifest_hash"}
    if seal.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("LCDP execution seal manifest hash mismatch")
    if seal.get("protocol_version") != SEAL_PROTOCOL_VERSION:
        raise RuntimeError("LCDP execution seal protocol mismatch")
    if seal.get("source_values_opened") is not False:
        raise RuntimeError("LCDP execution seal opened source values")
    if seal.get("outcomes_opened") is not False:
        raise RuntimeError("LCDP execution seal opened outcomes")
    if seal.get("policy_id") != POLICY_ID:
        raise RuntimeError("LCDP execution seal policy mismatch")
    commits: set[str] = set()
    for key, current_path in (("runner", RUNNER_PATH), ("tests", TEST_PATH)):
        binding = seal.get(key, {})
        if set(binding) != {"path", "commit", "sha256"}:
            raise RuntimeError(f"LCDP execution seal {key} schema mismatch")
        if binding.get("path") != current_path:
            raise RuntimeError(f"LCDP execution seal {key} path mismatch")
        commit = str(binding.get("commit", ""))
        commits.add(commit)
        sealed_bytes = subprocess.run(
            ["git", "show", f"{commit}:{current_path}"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(sealed_bytes).hexdigest() != binding.get("sha256"):
            raise RuntimeError(f"LCDP execution seal {key} bytes mismatch")
        if sha256_file(current_path) != binding.get("sha256"):
            raise RuntimeError(f"LCDP current {key} bytes drift")
        _assert_tracked_clean(current_path)
    if len(commits) != 1:
        raise RuntimeError("LCDP runner/test seal commits differ")
    expected_static = {
        "contract": {
            "path": CONTRACT_PATH,
            "commit": CONTRACT_COMMIT,
            "sha256": CONTRACT_SHA256,
        },
        "boundary": {
            "path": BOUNDARY_PATH,
            "sha256": BOUNDARY_SHA256,
        },
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
    }
    for key, expected in expected_static.items():
        if seal.get(key) != expected:
            raise RuntimeError(f"LCDP execution seal {key} mismatch")
    return seal


def validate_frozen_authority() -> dict[str, Any]:
    anchors = {
        CONTRACT_PATH: CONTRACT_SHA256,
        BOUNDARY_PATH: BOUNDARY_SHA256,
        prereg.PRODUCER_SCRIPT: PREREGISTRATION_PRODUCER_SHA256,
        PREREGISTRATION_PATH: PREREGISTRATION_SHA256,
        prereg.COINBASE_SOURCE: prereg.COINBASE_SHA256,
        prereg.BINANCE_SOURCE: prereg.BINANCE_SHA256,
        prereg.SOURCE_MANIFEST: prereg.SOURCE_MANIFEST_SHA256,
    }
    for path, expected in anchors.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"LCDP frozen anchor mismatch: {path}")
    if _assert_tracked_clean(CONTRACT_PATH) != CONTRACT_COMMIT:
        raise RuntimeError("LCDP implementation contract commit mismatch")
    prereg_payload = json.loads(repository_path(PREREGISTRATION_PATH).read_text())
    prereg.validate_manifest(prereg_payload)
    if prereg_payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("LCDP preregistration manifest mismatch")
    if prereg.csv_header(prereg.COINBASE_SOURCE) != prereg.COINBASE_HEADER:
        raise RuntimeError("LCDP Coinbase header mismatch")
    if prereg.csv_header(prereg.BINANCE_SOURCE) != prereg.BINANCE_HEADER:
        raise RuntimeError("LCDP Binance header mismatch")
    source_manifest = json.loads(
        repository_path(prereg.SOURCE_MANIFEST).read_text()
    )
    if source_manifest.get("manifest_hash") != prereg.SOURCE_MANIFEST_HASH:
        raise RuntimeError("LCDP source manifest hash mismatch")
    return {
        "anchors": anchors,
        "source_manifest_hash": prereg.SOURCE_MANIFEST_HASH,
    }


def _parse_timestamp(token: str) -> datetime:
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        token,
    ) is None:
        raise ValueError(
            "LCDP physical timestamp must be exact naive-UTC "
            "YYYY-MM-DD HH:MM:SS"
        )
    try:
        timestamp = datetime.strptime(
            token,
            "%Y-%m-%d %H:%M:%S",
        ).replace(tzinfo=prereg.UTC)
    except ValueError as error:
        raise ValueError(
            "LCDP physical timestamp must be exact naive-UTC "
            "YYYY-MM-DD HH:MM:SS"
        ) from error
    if (
        timestamp.second
        or timestamp.microsecond
        or timestamp.minute % 5
    ):
        raise ValueError("LCDP physical timestamp is not five-minute aligned")
    return timestamp


def _format_utc(timestamp: datetime) -> str:
    return timestamp.astimezone(prereg.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _number(token: str) -> float:
    return float(token) if token.strip() else float("nan")


def _boundary(day: date) -> datetime:
    return datetime.combine(day, time(16), tzinfo=prereg.LONDON)


def _source_cutoff(end_exclusive: str) -> datetime:
    final_day = date.fromisoformat(end_exclusive) - timedelta(days=1)
    return _boundary(final_day).astimezone(prereg.UTC)


def calendar_dates(end_exclusive: str) -> list[date]:
    start = date(2020, 1, 1)
    end = date.fromisoformat(end_exclusive)
    if not (start < end <= date(2023, 1, 1)):
        raise ValueError("LCDP prefix end is outside the frozen range")
    return [
        start + timedelta(days=offset)
        for offset in range((end - start).days)
    ]


def expected_timestamps(day: date) -> list[datetime]:
    start = _boundary(day - timedelta(days=1)).astimezone(prereg.UTC)
    end = _boundary(day).astimezone(prereg.UTC)
    seconds = int((end - start).total_seconds())
    if seconds % 300:
        raise RuntimeError("LCDP boundary is not five-minute aligned")
    count = seconds // 300
    if count not in {276, 288, 300}:
        raise RuntimeError("LCDP unexpected DST source-slot count")
    return [start + timedelta(minutes=5 * index) for index in range(count)]


def _assigned_london_day(timestamp: datetime) -> date:
    local = timestamp.astimezone(prereg.LONDON)
    boundary = datetime.combine(local.date(), time(16), tzinfo=prereg.LONDON)
    return local.date() if local < boundary else local.date() + timedelta(days=1)


def read_venue_source(
    path: str | Path,
    *,
    expected_header: Sequence[str],
    end_exclusive: str,
    venue: str,
) -> tuple[dict[date, list[Bar]], ParseAudit]:
    if venue not in {"cash", "perp"}:
        raise ValueError("LCDP venue must be cash or perp")
    source = repository_path(path)
    cutoff = _source_cutoff(end_exclusive)
    source_start = datetime(2020, 1, 1, tzinfo=prereg.UTC)
    physical_rows = 0
    selected_rows = 0
    post_cutoff_rows = 0
    post_2023_rows = 0
    post_cutoff_non_date = 0
    post_2023_non_date = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    prior: datetime | None = None
    groups: dict[date, list[Bar]] = {}
    opened = (
        gzip.open(source, "rt", encoding="utf-8", newline="")
        if source.suffix == ".gz"
        else source.open("rt", encoding="utf-8", newline="")
    )
    with opened as handle:
        header_line = handle.readline()
        header = tuple(next(csv.reader([header_line])))
        if header != tuple(expected_header):
            raise ValueError("LCDP physical header differs from exact contract")
        for raw_line in handle:
            physical_rows += 1
            date_token = raw_line.split(",", 1)[0]
            timestamp = _parse_timestamp(date_token)
            if prior is not None and timestamp <= prior:
                raise ValueError(
                    "LCDP physical timestamps are duplicate or nonchronological"
                )
            prior = timestamp
            first_timestamp = timestamp if first_timestamp is None else first_timestamp
            last_timestamp = timestamp
            if timestamp >= cutoff:
                post_cutoff_rows += 1
                if timestamp >= datetime(2023, 1, 1, tzinfo=prereg.UTC):
                    post_2023_rows += 1
                continue
            if timestamp < source_start:
                raise ValueError("LCDP selected source precedes 2020")
            values = next(csv.reader([raw_line]))
            selected_rows += 1
            if len(values) != len(header):
                raise ValueError("LCDP malformed selected source row")
            raw = dict(zip(header, values))
            if venue == "cash":
                complete_token = raw["source_complete"].strip()
                if complete_token not in {"0", "1"}:
                    raise ValueError("LCDP Coinbase source_complete is not binary")
                complete = complete_token == "1"
                bar = Bar(
                    timestamp=timestamp,
                    open=_number(raw["open"]),
                    high=_number(raw["high"]),
                    low=_number(raw["low"]),
                    close=_number(raw["close"]),
                    quote_notional=(
                        _number(raw["volume"]) * _number(raw["close"])
                    ),
                    source_complete=complete,
                )
            else:
                fields = (
                    _number(raw["open"]),
                    _number(raw["high"]),
                    _number(raw["low"]),
                    _number(raw["close"]),
                    _number(raw["quote_asset_volume"]),
                )
                if not all(math.isfinite(value) for value in fields):
                    raise ValueError("LCDP Binance selected row is nonfinite")
                bar = Bar(
                    timestamp=timestamp,
                    open=fields[0],
                    high=fields[1],
                    low=fields[2],
                    close=fields[3],
                    quote_notional=fields[4],
                    source_complete=True,
                )
            assigned = _assigned_london_day(timestamp)
            groups.setdefault(assigned, []).append(bar)
    if not physical_rows or first_timestamp is None or last_timestamp is None:
        raise ValueError("LCDP physical source is empty")
    return groups, ParseAudit(
        path=str(path),
        physical_date_rows=physical_rows,
        selected_non_date_rows=selected_rows,
        post_cutoff_date_only_rows=post_cutoff_rows,
        at_or_after_2023_date_only_rows=post_2023_rows,
        post_cutoff_non_date_rows=post_cutoff_non_date,
        at_or_after_2023_non_date_rows=post_2023_non_date,
        first_timestamp=_format_utc(first_timestamp),
        last_timestamp=_format_utc(last_timestamp),
    )


def build_venue_metrics(
    bars: Sequence[Bar],
    expected: Sequence[datetime],
) -> tuple[VenueMetrics | None, str]:
    if len(bars) != len(expected):
        return None, "timestamp_count"
    ordered = sorted(bars, key=lambda bar: bar.timestamp)
    if [bar.timestamp for bar in ordered] != list(expected):
        return None, "timestamp_grid"
    if not all(bar.source_complete for bar in ordered):
        return None, "source_incomplete"
    for bar in ordered:
        values = (
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.quote_notional,
        )
        if not all(math.isfinite(value) for value in values):
            return None, "nonfinite"
        if min(bar.open, bar.high, bar.low, bar.close) <= 0:
            return None, "nonpositive_price"
        if bar.quote_notional < 0:
            return None, "negative_notional"
        if not (
            bar.low
            <= min(bar.open, bar.close)
            <= max(bar.open, bar.close)
            <= bar.high
        ):
            return None, "ohlc_order"
    steps: list[float] = []
    previous = math.log(ordered[0].open)
    for bar in ordered:
        current = math.log(bar.close)
        steps.append(current - previous)
        previous = current
    total_return = math.fsum(steps)
    path = math.fsum(abs(value) for value in steps)
    half = len(steps) // 2
    return (
        VenueMetrics(
            start_price=ordered[0].open,
            end_price=ordered[-1].close,
            total_return=total_return,
            first_return=math.fsum(steps[:half]),
            second_return=math.fsum(steps[half:]),
            efficiency=abs(total_return) / path if path > 0 else 0.0,
            log_range=math.log(
                max(bar.high for bar in ordered)
                / min(bar.low for bar in ordered)
            ),
            quote_notional=math.fsum(
                bar.quote_notional for bar in ordered
            ),
        ),
        "valid",
    )


def build_day_pairs(
    dates: Sequence[date],
    cash_groups: Mapping[date, Sequence[Bar]],
    perp_groups: Mapping[date, Sequence[Bar]],
) -> list[DayPair]:
    pairs: list[DayPair] = []
    for day in dates:
        expected = expected_timestamps(day)
        cash, cash_reason = build_venue_metrics(
            cash_groups.get(day, ()),
            expected,
        )
        perp, perp_reason = build_venue_metrics(
            perp_groups.get(day, ()),
            expected,
        )
        if day == date(2020, 1, 1):
            cash = None
            perp = None
            cash_reason = SOURCE_INVALID_START
            perp_reason = SOURCE_INVALID_START
        pairs.append(
            DayPair(
                london_date=day,
                cash=cash,
                perp=perp,
                cash_reason=cash_reason,
                perp_reason=perp_reason,
            )
        )
    return pairs


def exact_quantile(values: Sequence[float], quantile: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        raise ValueError("LCDP quantile requires finite values")
    h = (len(finite) - 1) * quantile
    low = math.floor(h)
    high = math.ceil(h)
    return finite[low] + (h - low) * (finite[high] - finite[low])


def _calendar_context(day: date) -> str:
    if day.weekday() < 5:
        return "WEEKDAY"
    return "SATURDAY" if day.weekday() == 5 else "SUNDAY"


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _pair_share(pair: DayPair) -> float | None:
    if not pair.source_valid:
        return None
    assert pair.cash is not None and pair.perp is not None
    denominator = pair.cash.quote_notional + pair.perp.quote_notional
    if not math.isfinite(denominator) or denominator <= 0:
        return None
    share = pair.cash.quote_notional / denominator
    return share if math.isfinite(share) else None


def _daily_alignment(cash_sign: int, perp_sign: int) -> str:
    if cash_sign > 0 and perp_sign > 0:
        return "BOTH_RISE"
    if cash_sign < 0 and perp_sign < 0:
        return "BOTH_FALL"
    if cash_sign > 0 and perp_sign < 0:
        return "CASH_RISE_PERP_FALL"
    if cash_sign < 0 and perp_sign > 0:
        return "CASH_FALL_PERP_RISE"
    return "RETURN_MIXED_OR_FLAT"


def _daily_leader(cash: VenueMetrics, perp: VenueMetrics) -> str:
    cash_sign = _sign(cash.total_return)
    perp_sign = _sign(perp.total_return)
    if not cash_sign or cash_sign != perp_sign:
        return "NO_CLEAR_LEADER"
    if abs(cash.total_return) == abs(perp.total_return):
        return "NO_CLEAR_LEADER"
    cash_leads = abs(cash.total_return) > abs(perp.total_return)
    if cash_sign > 0:
        return "CASH_LEADS_RISE" if cash_leads else "PERP_LEADS_RISE"
    return "CASH_LEADS_FALL" if cash_leads else "PERP_LEADS_FALL"


def _basis_path(cash: VenueMetrics, perp: VenueMetrics) -> str:
    start = math.log(cash.start_price / perp.start_price)
    end = math.log(cash.end_price / perp.end_price)
    if start != 0 and end != 0 and _sign(start) != _sign(end):
        return "BASIS_ROTATES"
    if end > start:
        return "CASH_RICHENS"
    if end < start:
        return "CASH_CHEAPENS"
    return "BASIS_FLAT"


def _arc_transfer(cash: VenueMetrics, perp: VenueMetrics) -> str:
    first = cash.first_return - perp.first_return
    second = cash.second_return - perp.second_return
    first_sign = _sign(first)
    second_sign = _sign(second)
    if first_sign > 0 and second_sign > 0:
        return "CASH_LEAD_EXTENDS"
    if first_sign > 0 and second_sign < 0:
        return "CASH_LEAD_REVERSES"
    if first_sign < 0 and second_sign < 0:
        return "PERP_LEAD_EXTENDS"
    if first_sign < 0 and second_sign > 0:
        return "PERP_LEAD_REVERSES"
    return "ARC_MIXED"


def _compare(
    left: float,
    right: float,
    left_token: str,
    right_token: str,
    tie_token: str,
) -> str:
    if left > right:
        return left_token
    if left < right:
        return right_token
    return tie_token


def _participation_state(share: float, low: float, high: float) -> str:
    if share < low:
        return "CASH_PARTICIPATION_LOW"
    if share > high:
        return "CASH_PARTICIPATION_HIGH"
    return "CASH_PARTICIPATION_MID"


def _leader_venue(token: str) -> str | None:
    if token.startswith("CASH_LEADS"):
        return "cash"
    if token.startswith("PERP_LEADS"):
        return "perp"
    return None


def ready_tokens(
    pair: DayPair,
    *,
    low_quantile: float,
    high_quantile: float,
    previous: StreamLine | None,
) -> tuple[dict[str, str], int, int]:
    if not pair.source_valid:
        raise ValueError("LCDP ready token builder received invalid pair")
    assert pair.cash is not None and pair.perp is not None
    share = _pair_share(pair)
    if share is None:
        raise ValueError("LCDP ready pair has invalid participation share")
    cash_sign = _sign(pair.cash.total_return)
    perp_sign = _sign(pair.perp.total_return)
    alignment = _daily_alignment(cash_sign, perp_sign)
    leader = _daily_leader(pair.cash, pair.perp)
    participation = _participation_state(
        share,
        low_quantile,
        high_quantile,
    )
    if previous is None or not previous.ready:
        participation_transition = "PARTICIPATION_UNKNOWN"
        alignment_transition = "ALIGNMENT_MIXED"
        leader_transition = "LEAD_MIXED"
    else:
        prior_participation = previous.tokens["participation_state"]
        order = {
            "CASH_PARTICIPATION_LOW": 0,
            "CASH_PARTICIPATION_MID": 1,
            "CASH_PARTICIPATION_HIGH": 2,
        }
        if order[participation] > order[prior_participation]:
            participation_transition = "CASH_SHARE_RISING"
        elif order[participation] < order[prior_participation]:
            participation_transition = "CASH_SHARE_FALLING"
        else:
            participation_transition = "CASH_SHARE_STABLE"
        prior_alignment = previous.tokens["daily_alignment"]
        aligned = {"BOTH_RISE", "BOTH_FALL"}
        if alignment == prior_alignment and alignment in aligned:
            alignment_transition = "ALIGNMENT_PERSISTS"
        elif alignment in aligned and prior_alignment in aligned:
            alignment_transition = "ALIGNMENT_FLIPS"
        elif prior_alignment in aligned:
            alignment_transition = "ALIGNMENT_DISSIPATES"
        else:
            alignment_transition = "ALIGNMENT_MIXED"
        prior_venue = _leader_venue(previous.tokens["daily_leader"])
        current_venue = _leader_venue(leader)
        if prior_venue is None or current_venue is None:
            leader_transition = "LEAD_MIXED"
        elif prior_venue == current_venue == "cash":
            leader_transition = "CASH_LEAD_PERSISTS"
        elif prior_venue == current_venue == "perp":
            leader_transition = "PERP_LEAD_PERSISTS"
        elif current_venue == "cash":
            leader_transition = "LEAD_ROTATES_TO_CASH"
        else:
            leader_transition = "LEAD_ROTATES_TO_PERP"
    tokens = {
        "calendar_context": _calendar_context(pair.london_date),
        "daily_alignment": alignment,
        "daily_leader": leader,
        "relative_basis_path": _basis_path(pair.cash, pair.perp),
        "arc_transfer": _arc_transfer(pair.cash, pair.perp),
        "path_efficiency": _compare(
            pair.cash.efficiency,
            pair.perp.efficiency,
            "CASH_CLEANER",
            "PERP_CLEANER",
            "BOTH_CHOPPY_OR_TIE",
        ),
        "range_relation": _compare(
            pair.cash.log_range,
            pair.perp.log_range,
            "CASH_RANGE_DOMINANT",
            "PERP_RANGE_DOMINANT",
            "RANGE_BALANCED",
        ),
        "participation_state": participation,
        "participation_transition": participation_transition,
        "alignment_transition": alignment_transition,
        "leader_transition": leader_transition,
    }
    return tokens, cash_sign, perp_sign


def _uniform_tokens(day: date, token: str) -> dict[str, str]:
    return {
        field: _calendar_context(day) if field == "calendar_context" else token
        for field in prereg.TOKEN_COLUMNS
    }


def build_relational_stream(
    pairs: Sequence[DayPair],
    *,
    control: bool,
    invalid_token: str,
    first_token: str,
    policy: FrozenPolicy = FrozenPolicy(),
) -> list[StreamLine]:
    _validate_policy(policy)
    shares: list[float | None] = []
    lines: list[StreamLine] = []
    for index, pair in enumerate(pairs):
        share = _pair_share(pair)
        prior = [
            value
            for value in shares[max(0, index - policy.rank_lookback_lines) : index]
            if value is not None and math.isfinite(value)
        ]
        if index == 0:
            state = first_token
            tokens = _uniform_tokens(pair.london_date, first_token)
            cash_sign = 0
            perp_sign = 0
        elif not pair.source_valid or share is None:
            state = invalid_token
            tokens = _uniform_tokens(pair.london_date, invalid_token)
            cash_sign = 0
            perp_sign = 0
        elif len(prior) < policy.minimum_prior_valid_values:
            state = RANK_UNREADY
            tokens = _uniform_tokens(pair.london_date, RANK_UNREADY)
            cash_sign = 0
            perp_sign = 0
        else:
            low = exact_quantile(prior, 1.0 / 3.0)
            high = exact_quantile(prior, 2.0 / 3.0)
            tokens, cash_sign, perp_sign = ready_tokens(
                pair,
                low_quantile=low,
                high_quantile=high,
                previous=lines[-1] if lines else None,
            )
            state = READY
        serialized = prereg.serialize_line(
            tokens,
            control=control,
            allow_source_invalid_start=(
                index == 0 and state == SOURCE_INVALID_START
            ),
        )
        lines.append(
            StreamLine(
                london_date=pair.london_date,
                state=state,
                tokens=tokens,
                serialized=serialized,
                cash_sign=cash_sign,
                perp_sign=perp_sign,
            )
        )
        shares.append(share)
    return lines


def transformed_pairs(
    pairs: Sequence[DayPair],
) -> dict[str, list[DayPair]]:
    swap: list[DayPair] = []
    stale_cash: list[DayPair] = []
    stale_perp: list[DayPair] = []
    lag_7: list[DayPair] = []
    for index, pair in enumerate(pairs):
        swap.append(
            DayPair(
                pair.london_date,
                pair.perp,
                pair.cash,
                pair.perp_reason,
                pair.cash_reason,
            )
        )
        previous = pairs[index - 1] if index >= 1 else None
        stale_cash.append(
            DayPair(
                pair.london_date,
                previous.cash if previous is not None else None,
                pair.perp,
                previous.cash_reason if previous is not None else CONTROL_UNREADY,
                pair.perp_reason,
            )
        )
        stale_perp.append(
            DayPair(
                pair.london_date,
                pair.cash,
                previous.perp if previous is not None else None,
                pair.cash_reason,
                previous.perp_reason if previous is not None else CONTROL_UNREADY,
            )
        )
        lagged = pairs[index - 7] if index >= 7 else None
        lag_7.append(
            DayPair(
                pair.london_date,
                lagged.cash if lagged is not None else None,
                lagged.perp if lagged is not None else None,
                lagged.cash_reason if lagged is not None else CONTROL_UNREADY,
                lagged.perp_reason if lagged is not None else CONTROL_UNREADY,
            )
        )
    return {
        "cash_perp_role_swap": swap,
        "cash_stale_one_day": stale_cash,
        "perp_stale_one_day": stale_perp,
        "lag_7_calendar_days": lag_7,
    }


def build_derived_controls(
    primary: Sequence[StreamLine],
    pairs: Sequence[DayPair],
) -> dict[str, list[str]]:
    result = {
        "calendar_context_mask": [],
        "cash_only_language": [],
        "perp_only_language": [],
    }
    for line, pair in zip(primary, pairs):
        if not line.ready:
            for key in result:
                result[key].append(
                    prereg.serialize_line(
                        line.tokens,
                        control=True,
                        allow_source_invalid_start=(
                            line.state == SOURCE_INVALID_START
                        ),
                    )
                )
            continue
        masked_calendar = dict(line.tokens)
        masked_calendar["calendar_context"] = "CALENDAR_MASKED"
        result["calendar_context_mask"].append(
            prereg.serialize_line(masked_calendar, control=True)
        )
        assert pair.cash is not None and pair.perp is not None
        for key, prefix, sign in (
            ("cash_only_language", "CASH_ONLY", _sign(pair.cash.total_return)),
            ("perp_only_language", "PERP_ONLY", _sign(pair.perp.total_return)),
        ):
            direction = "RISE" if sign > 0 else "FALL" if sign < 0 else "FLAT"
            tokens = {
                field: (
                    line.tokens["calendar_context"]
                    if field == "calendar_context"
                    else f"{prefix}_{direction}"
                    if field == "daily_alignment"
                    else "ABLATION_MASKED"
                )
                for field in prereg.TOKEN_COLUMNS
            }
            result[key].append(prereg.serialize_line(tokens, control=True))
    return result


def _sequence_hash(lines: Sequence[StreamLine], index: int) -> str:
    if index + 1 < 21:
        return ""
    joined = "\n".join(
        line.serialized for line in lines[index - 20 : index + 1]
    ).encode("ascii")
    return hashlib.sha256(joined).hexdigest()


def assemble_rows(
    dates: Sequence[date],
    pairs: Sequence[DayPair],
    primary: Sequence[StreamLine],
    relational_controls: Mapping[str, Sequence[StreamLine]],
    derived_controls: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (day, pair, line) in enumerate(zip(dates, pairs, primary)):
        model_eligible = (
            line.ready
            and index >= 20
            and not (day.month == 12 and day.day == 31)
        )
        row: dict[str, Any] = {
            "london_date": day.isoformat(),
            "boundary_utc": _format_utc(
                _boundary(day).astimezone(prereg.UTC)
            ),
            "expected_slots": len(expected_timestamps(day)),
            "source_state": line.state,
            "rank_ready": line.ready,
            "model_eligible": model_eligible,
        }
        row.update(line.tokens)
        row["primary_line"] = line.serialized
        row["primary_sequence_hash"] = _sequence_hash(primary, index)
        for control_id in (
            "cash_perp_role_swap",
            "cash_stale_one_day",
            "perp_stale_one_day",
            "lag_7_calendar_days",
        ):
            row[control_id] = relational_controls[control_id][index].serialized
        for control_id in (
            "calendar_context_mask",
            "cash_only_language",
            "perp_only_language",
        ):
            row[control_id] = derived_controls[control_id][index]
        if tuple(row) != OUTPUT_COLUMNS:
            raise RuntimeError("LCDP support output schema drift")
        rows.append(row)
    return rows


def load_source_inputs(
    *,
    end_exclusive: str,
    cash_path: str | Path = prereg.COINBASE_SOURCE,
    perp_path: str | Path = prereg.BINANCE_SOURCE,
    policy: FrozenPolicy = FrozenPolicy(),
) -> SourceInputs:
    _validate_policy(policy)
    dates = calendar_dates(end_exclusive)
    cash_groups, cash_audit = read_venue_source(
        cash_path,
        expected_header=prereg.COINBASE_HEADER,
        end_exclusive=end_exclusive,
        venue="cash",
    )
    perp_groups, perp_audit = read_venue_source(
        perp_path,
        expected_header=prereg.BINANCE_HEADER,
        end_exclusive=end_exclusive,
        venue="perp",
    )
    pairs = build_day_pairs(dates, cash_groups, perp_groups)
    return SourceInputs(
        end_exclusive=end_exclusive,
        dates=dates,
        pairs=pairs,
        coinbase_audit=cash_audit,
        binance_audit=perp_audit,
    )


def finish_bundle(
    inputs: SourceInputs,
    *,
    policy: FrozenPolicy = FrozenPolicy(),
) -> SupportBundle:
    _validate_policy(policy)
    dates = inputs.dates
    pairs = inputs.pairs
    primary = build_relational_stream(
        pairs,
        control=False,
        invalid_token=SOURCE_INVALID,
        first_token=SOURCE_INVALID_START,
        policy=policy,
    )
    transformed = transformed_pairs(pairs)
    relational_controls = {
        "cash_perp_role_swap": build_relational_stream(
            transformed["cash_perp_role_swap"],
            control=True,
            invalid_token=SOURCE_INVALID,
            first_token=SOURCE_INVALID_START,
            policy=policy,
        ),
        "cash_stale_one_day": build_relational_stream(
            transformed["cash_stale_one_day"],
            control=True,
            invalid_token=CONTROL_UNREADY,
            first_token=CONTROL_UNREADY,
            policy=policy,
        ),
        "perp_stale_one_day": build_relational_stream(
            transformed["perp_stale_one_day"],
            control=True,
            invalid_token=CONTROL_UNREADY,
            first_token=CONTROL_UNREADY,
            policy=policy,
        ),
        "lag_7_calendar_days": build_relational_stream(
            transformed["lag_7_calendar_days"],
            control=True,
            invalid_token=CONTROL_UNREADY,
            first_token=CONTROL_UNREADY,
            policy=policy,
        ),
    }
    derived = build_derived_controls(primary, pairs)
    rows = assemble_rows(
        dates,
        pairs,
        primary,
        relational_controls,
        derived,
    )
    return SupportBundle(
        end_exclusive=inputs.end_exclusive,
        dates=dates,
        pairs=pairs,
        coinbase_audit=inputs.coinbase_audit,
        binance_audit=inputs.binance_audit,
        primary=primary,
        relational_controls=relational_controls,
        derived_controls=derived,
        rows=rows,
    )


def build_bundle(
    *,
    end_exclusive: str,
    cash_path: str | Path = prereg.COINBASE_SOURCE,
    perp_path: str | Path = prereg.BINANCE_SOURCE,
    policy: FrozenPolicy = FrozenPolicy(),
) -> SupportBundle:
    inputs = load_source_inputs(
        end_exclusive=end_exclusive,
        cash_path=cash_path,
        perp_path=perp_path,
        policy=policy,
    )
    return finish_bundle(inputs, policy=policy)


def _quarter(day: date) -> str:
    return f"{day.year}Q{((day.month - 1) // 3) + 1}"


def source_validity_metrics(
    bundle: SourceInputs | SupportBundle,
) -> dict[str, Any]:
    year_total = Counter(day.year for day in bundle.dates)
    year_valid = Counter(
        pair.london_date.year for pair in bundle.pairs if pair.source_valid
    )
    quarter_total = Counter(_quarter(day) for day in bundle.dates)
    quarter_valid = Counter(
        _quarter(pair.london_date)
        for pair in bundle.pairs
        if pair.source_valid
    )
    annual = {
        str(year): {
            "valid": year_valid[year],
            "total": total,
            "share": year_valid[year] / total,
        }
        for year, total in sorted(year_total.items())
    }
    quarterly = {
        quarter: {
            "valid": quarter_valid[quarter],
            "total": total,
            "share": quarter_valid[quarter] / total,
        }
        for quarter, total in sorted(quarter_total.items())
    }
    return {"annual": annual, "quarterly": quarterly}


def readiness_metrics(bundle: SupportBundle) -> dict[str, Any]:
    eligible_dates = [
        date.fromisoformat(row["london_date"])
        for row in bundle.rows
        if row["model_eligible"]
    ]
    annual = Counter(day.year for day in eligible_dates)
    quarterly = Counter(_quarter(day) for day in eligible_dates)
    return {
        "annual": {str(year): annual[year] for year in (2020, 2021, 2022)},
        "quarterly": {
            quarter: quarterly[quarter]
            for quarter in sorted({_quarter(day) for day in bundle.dates})
        },
    }


def diversity_metrics(bundle: SupportBundle) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    directions: dict[str, dict[str, int]] = {}
    for year in (2020, 2021, 2022):
        indices = [
            index
            for index, row in enumerate(bundle.rows)
            if row["model_eligible"]
            and date.fromisoformat(row["london_date"]).year == year
        ]
        year_fields: dict[str, Any] = {}
        for field in prereg.TOKEN_COLUMNS[1:]:
            counts = Counter(bundle.rows[index][field] for index in indices)
            total = sum(counts.values())
            year_fields[field] = {
                "counts": dict(sorted(counts.items())),
                "shares": {
                    key: value / total for key, value in sorted(counts.items())
                }
                if total
                else {},
            }
        fields[str(year)] = year_fields
        cash = Counter(bundle.primary[index].cash_sign for index in indices)
        perp = Counter(bundle.primary[index].perp_sign for index in indices)
        directions[str(year)] = {
            "cash_positive": cash[1],
            "cash_negative": cash[-1],
            "cash_zero": cash[0],
            "perp_positive": perp[1],
            "perp_negative": perp[-1],
            "perp_zero": perp[0],
        }
    return {"fields": fields, "directions": directions}


def _line_has_control_or_safety(serialized: str) -> bool:
    return any(token in serialized for token in SAFETY_OR_CONTROL_TOKENS)


def _deserialize_line(serialized: str) -> dict[str, str]:
    parts = serialized.split("|")
    tokens = dict(part.split("=", 1) for part in parts)
    if tuple(tokens) != prereg.TOKEN_COLUMNS:
        raise ValueError("LCDP serialized control field order drift")
    return tokens


def control_metrics(bundle: SupportBundle) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for control_id, lines in bundle.relational_controls.items():
        jointly_ready = 0
        different = 0
        for primary, control in zip(bundle.primary, lines):
            if not primary.ready or not control.ready:
                continue
            if _line_has_control_or_safety(control.serialized):
                continue
            jointly_ready += 1
            different += control.serialized != primary.serialized
        metrics[control_id] = {
            "jointly_ready": jointly_ready,
            "different": different,
            "difference_share": (
                different / jointly_ready if jointly_ready else 0.0
            ),
        }
    for control_id, lines in bundle.derived_controls.items():
        ready = 0
        different = 0
        required_correct = 0
        for primary, control in zip(bundle.primary, lines):
            if not primary.ready:
                continue
            ready += 1
            different += control != primary.serialized
            tokens = _deserialize_line(control)
            if control_id == "calendar_context_mask":
                correct = tokens["calendar_context"] == "CALENDAR_MASKED"
                correct = bool(
                    correct
                    and all(
                        tokens[field] == primary.tokens[field]
                        for field in prereg.TOKEN_COLUMNS[1:]
                    )
                )
            else:
                prefix = (
                    "CASH_ONLY_"
                    if control_id == "cash_only_language"
                    else "PERP_ONLY_"
                )
                correct = (
                    tokens["calendar_context"]
                    == primary.tokens["calendar_context"]
                    and tokens["daily_alignment"].startswith(prefix)
                    and all(
                        tokens[field] == "ABLATION_MASKED"
                        for field in prereg.TOKEN_COLUMNS[2:]
                    )
                )
            required_correct += bool(correct)
        metrics[control_id] = {
            "jointly_ready": ready,
            "different": different,
            "required_field_correct": required_correct,
            "difference_share": different / ready if ready else 0.0,
        }
    return metrics


def parser_integrity_checks(
    inputs: SourceInputs | SupportBundle,
) -> dict[str, bool]:
    return {
        "coinbase_post_cutoff_non_date_zero": (
            inputs.coinbase_audit.post_cutoff_non_date_rows == 0
        ),
        "binance_post_cutoff_non_date_zero": (
            inputs.binance_audit.post_cutoff_non_date_rows == 0
        ),
        "coinbase_post_2023_non_date_zero": (
            inputs.coinbase_audit.at_or_after_2023_non_date_rows == 0
        ),
        "binance_post_2023_non_date_zero": (
            inputs.binance_audit.at_or_after_2023_non_date_rows == 0
        ),
    }


def calendar_integrity_checks(
    inputs: SourceInputs | SupportBundle,
) -> dict[str, bool]:
    return {
        "line_count": len(inputs.dates) == 1096,
        "unique_dates": len(set(inputs.dates)) == len(inputs.dates),
        "start": inputs.dates[0] == date(2020, 1, 1),
        "end": inputs.dates[-1] == date(2022, 12, 31),
        "first_state": (
            inputs.pairs[0].cash_reason == SOURCE_INVALID_START
            and inputs.pairs[0].perp_reason == SOURCE_INVALID_START
        ),
        "first_state_count": sum(
            pair.cash_reason == SOURCE_INVALID_START
            and pair.perp_reason == SOURCE_INVALID_START
            for pair in inputs.pairs
        )
        == 1,
        "slot_counts": Counter(
            len(expected_timestamps(day)) for day in inputs.dates
        )
        == Counter({276: 3, 288: 1090, 300: 3}),
    }


def validity_gate_checks(
    validity: Mapping[str, Any],
    policy: FrozenPolicy,
) -> dict[str, bool]:
    return {
        "annual": all(
            record["share"] >= policy.annual_valid_share_min
            for record in validity["annual"].values()
        ),
        "quarterly": all(
            record["share"] >= policy.quarter_valid_share_min
            for record in validity["quarterly"].values()
        ),
    }


def readiness_gate_checks(
    readiness: Mapping[str, Any],
    policy: FrozenPolicy,
) -> dict[str, bool]:
    return {
        "2020": readiness["annual"]["2020"] >= policy.ready_2020_min,
        "2021": (
            readiness["annual"]["2021"] >= policy.ready_2021_2022_min
        ),
        "2022": (
            readiness["annual"]["2022"] >= policy.ready_2021_2022_min
        ),
        "post_2020q1_quarters": all(
            count >= policy.ready_post_2020q1_quarter_min
            for quarter, count in readiness["quarterly"].items()
            if quarter != "2020Q1"
        ),
    }


def diversity_gate_checks(
    diversity: Mapping[str, Any],
    policy: FrozenPolicy,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for year, year_fields in diversity["fields"].items():
        for field, record in year_fields.items():
            shares = record["shares"]
            supported = [
                share
                for share in shares.values()
                if share >= policy.category_share_min
            ]
            checks[f"{year}:{field}:two_supported"] = len(supported) >= 2
            checks[f"{year}:{field}:dominance"] = (
                bool(shares)
                and max(shares.values()) <= policy.category_share_max
            )
        direction = diversity["directions"][year]
        checks[f"{year}:cash_both_directions"] = (
            direction["cash_positive"] > 0 and direction["cash_negative"] > 0
        )
        checks[f"{year}:perp_both_directions"] = (
            direction["perp_positive"] > 0 and direction["perp_negative"] > 0
        )
    return checks


def control_gate_checks(
    controls: Mapping[str, Any],
    policy: FrozenPolicy,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    relational = {
        "cash_perp_role_swap",
        "cash_stale_one_day",
        "perp_stale_one_day",
        "lag_7_calendar_days",
    }
    for control_id, record in controls.items():
        if control_id in relational:
            checks[control_id] = (
                record["difference_share"]
                >= policy.control_difference_share_min
            )
        else:
            checks[control_id] = bool(
                record["jointly_ready"] > 0
                and record["different"] == record["jointly_ready"]
                and record.get("required_field_correct")
                == record["jointly_ready"]
            )
    return checks


def canonical_row(record: Mapping[str, Any]) -> bytes:
    return canonical_bytes(dict(record))


def compare_prefix_records(
    full_rows: Sequence[Mapping[str, Any]],
    prefix_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    full_by_date = {str(row["london_date"]): row for row in full_rows}
    mismatches: list[str] = []
    encoded: list[bytes] = []
    for row in prefix_rows:
        key = str(row["london_date"])
        current = canonical_row(row)
        encoded.append(current)
        if key not in full_by_date or current != canonical_row(full_by_date[key]):
            mismatches.append(key)
    return {
        "rows": len(prefix_rows),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
        "prefix_hash": hashlib.sha256(b"\n".join(encoded)).hexdigest(),
        "passed": not mismatches,
    }


def run_append_replay(
    full: SupportBundle,
    *,
    cash_path: str | Path = prereg.COINBASE_SOURCE,
    perp_path: str | Path = prereg.BINANCE_SOURCE,
    policy: FrozenPolicy = FrozenPolicy(),
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for cutoff in ("2021-01-01", "2022-01-01", "2023-01-01"):
        prefix = build_bundle(
            end_exclusive=cutoff,
            cash_path=cash_path,
            perp_path=perp_path,
            policy=policy,
        )
        comparison = compare_prefix_records(full.rows, prefix.rows)
        comparison["coinbase_later_non_date_rows"] = (
            prefix.coinbase_audit.post_cutoff_non_date_rows
        )
        comparison["binance_later_non_date_rows"] = (
            prefix.binance_audit.post_cutoff_non_date_rows
        )
        comparison["passed"] = bool(
            comparison["passed"]
            and comparison["coinbase_later_non_date_rows"] == 0
            and comparison["binance_later_non_date_rows"] == 0
        )
        results[cutoff] = comparison
    return {
        "prefixes": results,
        "passed": all(record["passed"] for record in results.values()),
    }


def forbidden_counters(
    bundle: SourceInputs | SupportBundle | None = None,
) -> dict[str, int]:
    return {
        "funding_rows_opened": 0,
        "execution_or_post_boundary_rows_opened": 0,
        "future_return_rows_built": 0,
        "reward_rows_built": 0,
        "model_rows_built": 0,
        "action_rows_built": 0,
        "trade_rows_built": 0,
        "pnl_values_computed": 0,
        "cagr_values_computed": 0,
        "mdd_values_computed": 0,
        "at_or_after_2023_non_date_source_rows_parsed": (
            0
            if bundle is None
            else bundle.coinbase_audit.at_or_after_2023_non_date_rows
            + bundle.binance_audit.at_or_after_2023_non_date_rows
        ),
    }


def deterministic_token_gzip(rows: Sequence[Mapping[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=OUTPUT_COLUMNS,
        extrasaction="raise",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for record in rows:
        encoded = {
            key: (
                "true"
                if value is True
                else "false"
                if value is False
                else value
            )
            for key, value in record.items()
        }
        writer.writerow(encoded)
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=buffer,
        compresslevel=9,
        mtime=0,
    ) as handle:
        handle.write(text.getvalue().encode("utf-8"))
    return buffer.getvalue()


def _gate_record(
    gate_id: int,
    name: str,
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "name": name,
        "checks": dict(checks),
        "passed": bool(checks) and all(checks.values()),
    }


def _authority_report(
    seal: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "execution_seal": {
            "path": EXECUTION_SEAL_PATH,
            "manifest_hash": seal["manifest_hash"],
            "sha256": sha256_file(EXECUTION_SEAL_PATH),
        },
        "runner": seal["runner"],
        "tests": seal["tests"],
        "contract": seal["contract"],
        "boundary": seal["boundary"],
        "preregistration": seal["preregistration"],
        "source_anchors": authority["anchors"],
        "source_manifest_hash": authority["source_manifest_hash"],
    }


def _report(
    *,
    decision: str,
    failure_action: str | None,
    pass_action: str | None,
    authority: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    details: Mapping[str, Any],
    append_replay: Mapping[str, Any] | None,
    counters: Mapping[str, int],
    row_hash: str | None,
    token_output_sha256: str | None,
    error: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "decision": decision,
        "failure_action": failure_action,
        "pass_action": pass_action,
        "profitability_result": False,
        "outcomes_opened": False,
        "authority": dict(authority),
        "policy": asdict(FrozenPolicy()),
        "gates": list(gates),
        "details": dict(details),
        "append_replay": append_replay,
        "forbidden_counters": dict(counters),
        "source_token_row_hash": row_hash,
        "token_output": (
            {
                "path": TOKEN_OUTPUT,
                "sha256": token_output_sha256,
                "rows": 1096,
            }
            if token_output_sha256 is not None
            else None
        ),
        "error": error,
    }
    return {**core, "result_hash": canonical_hash(core)}


def run_official() -> dict[str, Any]:
    policy = FrozenPolicy()
    _validate_policy(policy)
    gates: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    inputs: SourceInputs | None = None
    bundle: SupportBundle | None = None
    authority_report: dict[str, Any] = {
        "execution_seal": {"path": EXECUTION_SEAL_PATH},
        "contract": {
            "path": CONTRACT_PATH,
            "commit": CONTRACT_COMMIT,
            "sha256": CONTRACT_SHA256,
        },
        "boundary": {
            "path": BOUNDARY_PATH,
            "sha256": BOUNDARY_SHA256,
        },
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
    }

    def fail(
        *,
        append_replay: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        row_hash = (
            None
            if bundle is None
            else hashlib.sha256(
                b"\n".join(canonical_row(row) for row in bundle.rows)
            ).hexdigest()
        )
        report = _report(
            decision="fail",
            failure_action="retire_lcdp_d1_unchanged_before_outcomes",
            pass_action=None,
            authority=authority_report,
            gates=gates,
            details=details,
            append_replay=append_replay,
            counters=forbidden_counters(bundle or inputs),
            row_hash=row_hash,
            token_output_sha256=None,
            error=(
                None
                if error is None
                else {"type": type(error).__name__, "message": str(error)}
            ),
        )
        write_once_json(REJECTION_REPORT, report)
        return report

    try:
        seal = validate_execution_seal()
        authority = validate_frozen_authority()
        authority_report = _authority_report(seal, authority)
    except Exception as error:
        gates.append(
            _gate_record(
                1,
                "protocol_source_integrity",
                {"authority_validation_completed": False},
            )
        )
        return fail(error=error)

    try:
        inputs = load_source_inputs(
            end_exclusive="2023-01-01",
            policy=policy,
        )
    except Exception as error:
        gates.append(
            _gate_record(
                1,
                "protocol_source_integrity",
                {"source_build_completed": False},
            )
        )
        return fail(error=error)

    details["parser_audit"] = {
        "coinbase": asdict(inputs.coinbase_audit),
        "binance": asdict(inputs.binance_audit),
    }
    gate = _gate_record(
        1,
        "protocol_source_integrity",
        parser_integrity_checks(inputs),
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    gate = _gate_record(
        2,
        "calendar_dst_integrity",
        calendar_integrity_checks(inputs),
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    validity = source_validity_metrics(inputs)
    details["source_validity"] = validity
    gate = _gate_record(
        3,
        "source_validity",
        validity_gate_checks(validity, policy),
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    try:
        bundle = finish_bundle(inputs, policy=policy)
    except Exception as error:
        gates.append(
            _gate_record(
                4,
                "readiness",
                {"token_build_completed": False},
            )
        )
        return fail(error=error)

    readiness = readiness_metrics(bundle)
    details["readiness"] = readiness
    gate = _gate_record(
        4,
        "readiness",
        readiness_gate_checks(readiness, policy),
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    diversity = diversity_metrics(bundle)
    details["diversity"] = diversity
    gate = _gate_record(
        5,
        "token_diversity",
        diversity_gate_checks(diversity, policy),
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    controls = control_metrics(bundle)
    details["controls"] = controls
    gate = _gate_record(
        6,
        "control_distinctness",
        control_gate_checks(controls, policy),
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    append_replay = run_append_replay(bundle, policy=policy)
    append_gate = _gate_record(
        7,
        "append_replay",
        {"all_prefixes_byte_identical": append_replay["passed"]},
    )
    gates.append(append_gate)
    if not append_gate["passed"]:
        return fail(append_replay=append_replay)
    counters = forbidden_counters(bundle)
    forbidden_gate = _gate_record(
        8,
        "forbidden_access",
        {key: value == 0 for key, value in counters.items()},
    )
    gates.append(forbidden_gate)
    row_hash = hashlib.sha256(
        b"\n".join(canonical_row(row) for row in bundle.rows)
    ).hexdigest()
    if not forbidden_gate["passed"]:
        return fail(append_replay=append_replay)
    token_bytes = deterministic_token_gzip(bundle.rows)
    token_sha = write_once_bytes(TOKEN_OUTPUT, token_bytes)
    report = _report(
        decision="pass",
        failure_action=None,
        pass_action="authorize_economic_rllm_evaluator_freeze_only",
        authority=authority_report,
        gates=gates,
        details=details,
        append_replay=append_replay,
        counters=counters,
        row_hash=row_hash,
        token_output_sha256=token_sha,
    )
    write_once_json(PASS_REPORT, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("create-seal")
    subparsers.add_parser("validate-seal")
    subparsers.add_parser("run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create-seal":
        payload = create_execution_seal()
        print(
            json.dumps(
                {
                    "path": EXECUTION_SEAL_PATH,
                    "manifest_hash": payload["manifest_hash"],
                    "source_values_opened": False,
                    "outcomes_opened": False,
                },
                indent=2,
            )
        )
        return
    if args.command == "validate-seal":
        payload = validate_execution_seal()
        print(json.dumps({"manifest_hash": payload["manifest_hash"]}, indent=2))
        return
    report = run_official()
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "result_hash": report["result_hash"],
                "failure_action": report["failure_action"],
                "pass_action": report["pass_action"],
            },
            indent=2,
        )
    )
    if report["decision"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
