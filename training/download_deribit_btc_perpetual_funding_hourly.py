"""Download the frozen pre-2024 Deribit BTC perpetual funding source.

The source is hourly and outcome-blind.  It persists only Deribit's historical
funding-memory fields and index endpoints, plus a conservative synthetic
availability timestamp.  No Binance market, Binance funding, future return,
held path, or PnL is requested or written.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, TypedDict


ENDPOINT = (
    "https://www.deribit.com/api/v2/public/get_funding_rate_history"
)
OFFICIAL_DOCS = (
    "https://docs.deribit.com/api-reference/market-data/"
    "public-get_funding_rate_history"
)
OFFICIAL_USAGE_POLICY = "https://docs.deribit.com/articles/api-usage-policy"
SOURCE_DECISION = (
    "docs/deribit-funding-impulse-absorption-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "3aa1e0dbea5afa55b82e3693da945c384efe2187a971dfe4f8aa52d5677e7281"
)
INSTRUMENT = "BTC-PERPETUAL"
RESULT_ROW_KEYS = frozenset(
    {
        "timestamp",
        "interest_1h",
        "interest_8h",
        "index_price",
        "prev_index_price",
    }
)
RESPONSE_KEYS = frozenset(
    {"jsonrpc", "result", "testnet", "usIn", "usOut", "usDiff"}
)
OUTPUT_COLUMNS = (
    "timestamp",
    "available_at",
    "interest_1h",
    "interest_8h",
    "index_price",
    "prev_index_price",
)
USER_AGENT = "rllm-private-research/1.0"
UTC_TIMESTAMP_RE = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?(?P<offset>Z|\+00:00)$"
)
HOUR = timedelta(hours=1)


class OutputRow(TypedDict):
    timestamp: str
    available_at: str
    interest_1h: str
    interest_8h: str
    index_price: str
    prev_index_price: str


@dataclass(frozen=True)
class Config:
    output_csv: str = (
        "data/deribit_btc_perpetual_funding_2019_2023.csv.gz"
    )
    manifest_output: str = (
        "results/deribit_btc_perpetual_funding_source_manifest_2026-07-20.json"
    )
    start: str = "2019-04-30T10:00:00Z"
    end_exclusive: str = "2024-01-01T00:00:00Z"
    chunk_hours: int = 28 * 24
    synthetic_availability_delay_minutes: int = 5
    maximum_memory_abs_error: str = "0.00005"
    timeout_sec: float = 30.0
    request_pause_sec: float = 0.20
    maximum_retries: int = 8
    base_url: str = ENDPOINT


Fetch = Callable[[str], dict[str, Any]]


def canonical_hash(payload: Any) -> str:
    def normalise(value: Any) -> Any:
        if isinstance(value, Decimal):
            return {"__exact_json_decimal__": str(value)}
        if isinstance(value, dict):
            return {str(key): normalise(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalise(item) for item in value]
        return value

    encoded = json.dumps(
        normalise(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty UTC timestamp string")
    match = UTC_TIMESTAMP_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"{name} must use an exact ISO-8601 UTC timestamp")
    fraction = match.group("fraction") or ""
    if len(fraction) > 6 and any(digit != "0" for digit in fraction[6:]):
        raise ValueError(f"{name} has unsupported non-zero sub-microseconds")
    microseconds = fraction[:6].ljust(6, "0")
    try:
        parsed = datetime.fromisoformat(
            f"{match.group('head')}.{microseconds}+00:00"
        )
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid UTC timestamp") from exc
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    utc = value.astimezone(timezone.utc)
    if utc.microsecond:
        return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return utc.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_decimal(value: Any, name: str, *, positive: bool) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ValueError(f"{name} must be an exact JSON number")
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an exact JSON number") from exc
    if not number.is_finite():
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if number == 0:
        number = Decimal(0)
    return number


def _format_decimal(value: Decimal) -> str:
    return format(value, "f")


def _validate_config(cfg: Config) -> tuple[datetime, datetime, Decimal]:
    start = _parse_utc(cfg.start, "start")
    end = _parse_utc(cfg.end_exclusive, "end_exclusive")
    if start >= end:
        raise ValueError("start must precede end_exclusive")
    for name, value in (("start", start), ("end_exclusive", end)):
        if value.minute or value.second or value.microsecond:
            raise ValueError(f"{name} must be aligned to a whole UTC hour")
    if (
        isinstance(cfg.chunk_hours, bool)
        or not isinstance(cfg.chunk_hours, int)
        or not 1 <= cfg.chunk_hours <= 720
    ):
        raise ValueError("chunk_hours must be an integer in [1, 720]")
    if (
        isinstance(cfg.synthetic_availability_delay_minutes, bool)
        or not isinstance(cfg.synthetic_availability_delay_minutes, int)
        or cfg.synthetic_availability_delay_minutes < 5
    ):
        raise ValueError(
            "synthetic_availability_delay_minutes must be an integer >= 5"
        )
    try:
        memory_error = Decimal(cfg.maximum_memory_abs_error)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            "maximum_memory_abs_error must be a positive exact decimal"
        ) from exc
    if not memory_error.is_finite() or memory_error <= 0:
        raise ValueError(
            "maximum_memory_abs_error must be a positive exact decimal"
        )
    if not math.isfinite(cfg.timeout_sec) or cfg.timeout_sec <= 0.0:
        raise ValueError("timeout_sec must be positive and finite")
    if not math.isfinite(cfg.request_pause_sec) or cfg.request_pause_sec < 0.0:
        raise ValueError("request_pause_sec must be finite and non-negative")
    if (
        isinstance(cfg.maximum_retries, bool)
        or not isinstance(cfg.maximum_retries, int)
        or cfg.maximum_retries < 0
    ):
        raise ValueError("maximum_retries must be a non-negative integer")
    base = urllib.parse.urlparse(cfg.base_url)
    expected = urllib.parse.urlparse(ENDPOINT)
    if (
        base.scheme != expected.scheme
        or base.netloc != expected.netloc
        or base.path != expected.path
        or base.params
        or base.query
        or base.fragment
    ):
        raise ValueError("base_url must be the frozen Deribit funding endpoint")
    return start, end, memory_error


def request_windows(cfg: Config) -> list[tuple[datetime, datetime, str]]:
    start, end, _ = _validate_config(cfg)
    windows: list[tuple[datetime, datetime, str]] = []
    current = start
    while current < end:
        stop = min(current + timedelta(hours=cfg.chunk_hours), end)
        params = {
            "instrument_name": INSTRUMENT,
            # Deribit start_timestamp behaves exclusively.  Request one hour
            # earlier so the desired current boundary is included.
            "start_timestamp": int((current - HOUR).timestamp() * 1000),
            "end_timestamp": int((stop - HOUR).timestamp() * 1000),
        }
        windows.append(
            (current, stop, f"{cfg.base_url}?{urllib.parse.urlencode(params)}")
        )
        current = stop
    return windows


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Deribit response used non-standard JSON value {value}")


def _decode_payload(raw: bytes) -> dict[str, Any]:
    payload = json.loads(
        raw.decode("utf-8"),
        parse_float=Decimal,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Deribit funding response is not an object")
    return payload


def _http_page(cfg: Config, url: str) -> dict[str, Any]:
    for attempt in range(cfg.maximum_retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(
                request, timeout=cfg.timeout_sec
            ) as response:
                return _decode_payload(response.read())
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt >= cfg.maximum_retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2.0**attempt
            except ValueError:
                delay = 2.0**attempt
            time.sleep(max(cfg.request_pause_sec, min(60.0, delay)))
        except (TimeoutError, urllib.error.URLError):
            if attempt >= cfg.maximum_retries:
                raise
            time.sleep(max(cfg.request_pause_sec, min(60.0, 2.0**attempt)))
    raise AssertionError("unreachable retry loop")


def _validate_payload(
    payload: dict[str, Any], *, maximum_rows: int
) -> list[dict[str, Any]]:
    if payload.get("error") is not None:
        raise RuntimeError(f"Deribit funding API error: {payload['error']!r}")
    if frozenset(payload) != RESPONSE_KEYS:
        raise ValueError("Deribit funding response schema drift")
    if payload.get("jsonrpc") != "2.0" or payload.get("testnet") is not False:
        raise RuntimeError("Deribit funding server environment changed")
    timing: list[int] = []
    for key in ("usIn", "usOut", "usDiff"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Deribit funding server timing metadata is invalid")
        timing.append(value)
    if timing[1] < timing[0] or timing[2] != timing[1] - timing[0]:
        raise ValueError("Deribit funding server timing metadata is inconsistent")
    result = payload.get("result")
    if not isinstance(result, list):
        raise ValueError("Deribit funding result must be a list")
    # A bounded probe found an undocumented 744-row ceiling.  Frozen requests
    # are smaller; touching that ceiling or exceeding the expected overlap is
    # treated as truncation/scope drift.
    if len(result) >= 744 or len(result) > maximum_rows:
        raise RuntimeError("Deribit funding response reached a truncation cap")
    if any(not isinstance(row, dict) for row in result):
        raise ValueError("Deribit funding result row must be an object")
    return result


def _normalise_row(
    raw: dict[str, Any],
    *,
    allowed_start: datetime,
    allowed_end: datetime,
    availability_delay_minutes: int,
) -> OutputRow:
    keys = frozenset(raw)
    if keys != RESULT_ROW_KEYS:
        missing = sorted(RESULT_ROW_KEYS - keys)
        unexpected = sorted(keys - RESULT_ROW_KEYS)
        raise ValueError(
            "Deribit funding row schema drift: "
            f"missing={missing!r} unexpected={unexpected!r}"
        )
    timestamp_ms = raw["timestamp"]
    if (
        isinstance(timestamp_ms, bool)
        or not isinstance(timestamp_ms, int)
        or timestamp_ms <= 0
        or timestamp_ms % 3_600_000 != 0
    ):
        raise ValueError(
            "Deribit funding timestamp must be a positive whole UTC hour"
        )
    try:
        observed = datetime.fromtimestamp(
            timestamp_ms / 1000, tz=timezone.utc
        )
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("Deribit funding timestamp is outside UTC range") from exc
    if not allowed_start <= observed < allowed_end:
        raise ValueError("Deribit funding row is outside its frozen request window")
    interest_1h = _parse_decimal(raw["interest_1h"], "interest_1h", positive=False)
    interest_8h = _parse_decimal(raw["interest_8h"], "interest_8h", positive=False)
    index_price = _parse_decimal(raw["index_price"], "index_price", positive=True)
    prev_index_price = _parse_decimal(
        raw["prev_index_price"], "prev_index_price", positive=True
    )
    available = observed + timedelta(minutes=availability_delay_minutes)
    return {
        "timestamp": _format_utc(observed),
        "available_at": _format_utc(available),
        "interest_1h": _format_decimal(interest_1h),
        "interest_8h": _format_decimal(interest_8h),
        "index_price": _format_decimal(index_price),
        "prev_index_price": _format_decimal(prev_index_price),
    }


def _memory_audit(
    rows: list[OutputRow], *, maximum_abs_error: Decimal
) -> dict[str, Any]:
    by_time = {
        _parse_utc(row["timestamp"], "timestamp"): row for row in rows
    }
    errors: list[Decimal] = []
    for timestamp, row in by_time.items():
        window = [timestamp - timedelta(hours=offset) for offset in range(7, -1, -1)]
        if any(point not in by_time for point in window):
            continue
        rolling = sum(
            (Decimal(by_time[point]["interest_1h"]) for point in window),
            Decimal(0),
        )
        error = abs(rolling - Decimal(row["interest_8h"]))
        if error > maximum_abs_error:
            raise RuntimeError(
                "Deribit funding 1h/8h memory invariant drifted: "
                f"timestamp={_format_utc(timestamp)} error={error} "
                f"maximum={maximum_abs_error}"
            )
        errors.append(error)
    ordered = sorted(errors)
    median = (
        ordered[len(ordered) // 2]
        if ordered
        else Decimal(0)
    )
    return {
        "contiguous_eight_hour_windows": len(errors),
        "maximum_absolute_sum1h_minus_8h": _format_decimal(max(errors, default=Decimal(0))),
        "median_absolute_sum1h_minus_8h": _format_decimal(median),
        "maximum_allowed_absolute_error": _format_decimal(maximum_abs_error),
        "all_windows_within_tolerance": True,
    }


def download_rows(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[OutputRow], dict[str, Any]]:
    start, end, maximum_memory_error = _validate_config(cfg)
    fetch = fetch or (lambda url: _http_page(cfg, url))
    windows = request_windows(cfg)
    by_time: dict[str, OutputRow] = {}
    result_hashes: list[str] = []
    result_lengths: list[int] = []
    exact_boundary_duplicates = 0

    for window_index, (window_start, window_end, url) in enumerate(windows):
        payload = fetch(url)
        if not isinstance(payload, dict):
            raise RuntimeError("Deribit funding response is not an object")
        width_hours = int((window_end - window_start).total_seconds() // 3600)
        raw_rows = _validate_payload(payload, maximum_rows=width_hours + 1)
        result_hashes.append(canonical_hash(raw_rows))
        result_lengths.append(len(raw_rows))
        prior_observed: datetime | None = None
        for raw in raw_rows:
            row = _normalise_row(
                raw,
                allowed_start=window_start - HOUR,
                allowed_end=window_end,
                availability_delay_minutes=cfg.synthetic_availability_delay_minutes,
            )
            observed = _parse_utc(row["timestamp"], "timestamp")
            if prior_observed is not None and observed <= prior_observed:
                raise RuntimeError(
                    "Deribit funding response is not strictly time-ascending"
                )
            prior_observed = observed
            if observed < start:
                if window_index != 0 or observed != window_start - HOUR:
                    raise RuntimeError("Deribit funding crossed the frozen start")
                continue
            existing = by_time.get(row["timestamp"])
            if existing is not None:
                if observed != window_start - HOUR:
                    raise RuntimeError(
                        "Deribit funding source contains a non-boundary duplicate"
                    )
                if existing != row:
                    raise RuntimeError(
                        "Deribit funding boundary duplicate conflicts"
                    )
                exact_boundary_duplicates += 1
                continue
            by_time[row["timestamp"]] = row
        if window_index + 1 < len(windows):
            sleep(cfg.request_pause_sec)

    rows = [by_time[key] for key in sorted(by_time)]
    if not rows:
        raise RuntimeError("Deribit funding source is empty")
    expected_first = _format_utc(start)
    expected_last = _format_utc(end - HOUR)
    if rows[0]["timestamp"] != expected_first:
        raise RuntimeError("Deribit funding source does not start at frozen boundary")
    if rows[-1]["timestamp"] != expected_last:
        raise RuntimeError("Deribit funding source does not reach frozen end boundary")

    observed_times = [_parse_utc(row["timestamp"], "timestamp") for row in rows]
    observed_set = set(observed_times)
    expected_hours = int((end - start).total_seconds() // 3600)
    missing: list[datetime] = []
    current = start
    while current < end:
        if current not in observed_set:
            missing.append(current)
        current += HOUR
    gaps = [
        int((right - left).total_seconds() // 3600)
        for left, right in zip(observed_times, observed_times[1:])
    ]

    contiguous_price_links = 0
    for previous, current_row, gap in zip(rows, rows[1:], gaps):
        if gap != 1:
            continue
        contiguous_price_links += 1
        if Decimal(current_row["prev_index_price"]) != Decimal(
            previous["index_price"]
        ):
            raise RuntimeError(
                "Deribit funding contiguous index-price chain changed"
            )

    memory = _memory_audit(rows, maximum_abs_error=maximum_memory_error)
    audit = {
        "request_windows": len(windows),
        "request_window_hours_max": cfg.chunk_hours,
        "response_result_lengths": result_lengths,
        "response_result_sha256": result_hashes,
        "response_chain_sha256": canonical_hash(result_hashes),
        "response_environment": {
            "jsonrpc": "2.0",
            "testnet": False,
            "server_timing_validated_not_persisted": True,
        },
        "requested_hours": expected_hours,
        "observed_rows": len(rows),
        "coverage_ratio": len(rows) / expected_hours,
        "first_observation": rows[0]["timestamp"],
        "last_observation": rows[-1]["timestamp"],
        "missing_hours": len(missing),
        "missing_hours_head": [_format_utc(value) for value in missing[:24]],
        "maximum_observation_gap_hours": max(gaps, default=0),
        "exact_boundary_duplicates": exact_boundary_duplicates,
        "conflicting_duplicates": 0,
        "contiguous_index_price_links_checked": contiguous_price_links,
        "memory_identity": memory,
        "unexpected_row_fields": 0,
    }
    return rows, audit


def _write_deterministic_csv(path: Path, rows: list[OutputRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8", newline="") as wrapper:
                    writer = csv.DictWriter(
                        wrapper,
                        fieldnames=list(OUTPUT_COLUMNS),
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(
                            {column: row[column] for column in OUTPUT_COLUMNS}
                        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if sha256_file(SOURCE_DECISION) != SOURCE_DECISION_SHA256:
        raise RuntimeError("DFIA-72 source decision hash changed")
    rows, source_audit = download_rows(cfg, fetch=fetch, sleep=sleep)
    output = Path(cfg.output_csv)
    _write_deterministic_csv(output, rows)
    manifest: dict[str, Any] = {
        "protocol_version": 1,
        "candidate": "DFIA-72",
        "config": asdict(cfg),
        "official_docs": OFFICIAL_DOCS,
        "official_usage_policy": OFFICIAL_USAGE_POLICY,
        "source_decision": SOURCE_DECISION,
        "source_decision_sha256": SOURCE_DECISION_SHA256,
        "source_audit": source_audit,
        "output": str(output),
        "output_columns": list(OUTPUT_COLUMNS),
        "output_sha256": sha256_file(output),
        "source_semantics": {
            "interest_1h": "Deribit one-hour perpetual interest rate",
            "interest_8h": "Deribit eight-hour perpetual interest rate",
            "index_price": "Deribit BTC index price at the hourly row",
            "prev_index_price": "Deribit BTC index price at the prior hourly point",
        },
        "causal_availability": {
            "historical_synthetic_delay_minutes": (
                cfg.synthetic_availability_delay_minutes
            ),
            "live_rule": "use actual first successful observation when later; never backdate",
            "gap_rule": "a missing hour breaks the feature chain and is never filled",
        },
        "revision_boundary": (
            "the output hash freezes this downloaded Deribit source vintage; "
            "it is not a historical archive of every publication revision"
        ),
        "distribution_boundary": (
            "raw responses and the local hourly source remain ignored; commit "
            "only code, hashes, source-quality summaries, derived clocks, and "
            "aggregate results under Deribit API policies"
        ),
        "outcome_boundary": {
            "binance_market_rows_loaded": 0,
            "binance_funding_rows_loaded": 0,
            "return_or_pnl_fields": 0,
            "post_2023_source_rows_loaded": 0,
            "raw_deribit_responses_persisted": False,
        },
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_manifest(Path(cfg.manifest_output), manifest)
    return manifest


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", default=Config.output_csv)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end-exclusive", default=Config.end_exclusive)
    parser.add_argument("--chunk-hours", type=int, default=Config.chunk_hours)
    parser.add_argument(
        "--synthetic-availability-delay-minutes",
        type=int,
        default=Config.synthetic_availability_delay_minutes,
    )
    parser.add_argument(
        "--maximum-memory-abs-error",
        default=Config.maximum_memory_abs_error,
    )
    parser.add_argument("--timeout-sec", type=float, default=Config.timeout_sec)
    parser.add_argument(
        "--request-pause-sec", type=float, default=Config.request_pause_sec
    )
    parser.add_argument(
        "--maximum-retries", type=int, default=Config.maximum_retries
    )
    parser.add_argument("--base-url", default=Config.base_url)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
