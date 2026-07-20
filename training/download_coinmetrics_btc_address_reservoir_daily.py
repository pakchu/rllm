"""Download the frozen BTC address-reservoir source from Coin Metrics.

The artifact is source-only and availability-aware.  It contains the stock of
funded addresses (``AdrBalCnt``), daily active-address flow (``AdrActCnt``),
and the recorded end-of-day completion time.  No market, funding, return, or
post-entry outcome is requested or persisted.
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
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, TypedDict


ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
OFFICIAL_CATALOG_URL = (
    "https://community-api.coinmetrics.io/v4/catalog-all-v2/asset-metrics?"
    "assets=btc&metrics=AdrBalCnt%2CAdrActCnt%2CAssetEODCompletionTime"
)
METRICS = ("AdrBalCnt", "AdrActCnt", "AssetEODCompletionTime")
RAW_ROW_KEYS = frozenset({"asset", "time", *METRICS})
OUTPUT_COLUMNS = (
    "observation_date",
    "available_at",
    "AdrBalCnt",
    "AdrActCnt",
)
USER_AGENT = "rllm-private-research/1.0"
UTC_TIMESTAMP_RE = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?(?P<offset>Z|\+00:00)$"
)


class OutputRow(TypedDict):
    observation_date: str
    available_at: str
    AdrBalCnt: str
    AdrActCnt: str


@dataclass(frozen=True)
class Config:
    output_csv: str = (
        "data/coinmetrics_btc_address_reservoir_2019_2023.csv.gz"
    )
    manifest_output: str = (
        "results/coinmetrics_btc_address_reservoir_source_manifest_2026-07-20.json"
    )
    start: str = "2019-01-01"
    end_exclusive: str = "2024-01-01"
    page_size: int = 10_000
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


def _parse_day(value: str, name: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{name} must be an exact YYYY-MM-DD UTC day")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an exact YYYY-MM-DD UTC day") from exc
    return parsed.replace(tzinfo=timezone.utc)


def _parse_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty UTC timestamp string")
    text = value.strip()
    match = UTC_TIMESTAMP_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"{name} must use an exact ISO-8601 UTC timestamp")
    fraction = match.group("fraction") or ""
    if len(fraction) > 6 and any(digit != "0" for digit in fraction[6:]):
        raise ValueError(f"{name} has unsupported non-zero sub-microseconds")
    microseconds = fraction[:6].ljust(6, "0")
    text = f"{match.group('head')}.{microseconds}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid UTC timestamp") from exc
    return parsed.astimezone(timezone.utc)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def _completion_time(value: Any) -> datetime:
    if isinstance(value, bool):
        raise ValueError("AssetEODCompletionTime must be positive and finite")
    try:
        seconds = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            "AssetEODCompletionTime must be positive and finite"
        ) from exc
    if not seconds.is_finite() or seconds <= 0:
        raise ValueError("AssetEODCompletionTime must be positive and finite")
    whole_seconds_decimal = seconds.to_integral_value(rounding=ROUND_FLOOR)
    microseconds_decimal = (seconds - whole_seconds_decimal) * Decimal(1_000_000)
    if microseconds_decimal != microseconds_decimal.to_integral_value():
        raise ValueError(
            "AssetEODCompletionTime has unsupported sub-microsecond precision"
        )
    try:
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=int(whole_seconds_decimal),
            microseconds=int(microseconds_decimal),
        )
    except (OverflowError, ValueError) as exc:
        raise ValueError("AssetEODCompletionTime is outside the UTC range") from exc


def _format_utc(value: datetime) -> str:
    utc = value.astimezone(timezone.utc)
    if utc.microsecond:
        return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return utc.isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_config(cfg: Config) -> tuple[datetime, datetime]:
    start = _parse_day(cfg.start, "start")
    end = _parse_day(cfg.end_exclusive, "end_exclusive")
    if start >= end:
        raise ValueError("start must precede end_exclusive")
    if (
        isinstance(cfg.page_size, bool)
        or not isinstance(cfg.page_size, int)
        or not 1 <= cfg.page_size <= 10_000
    ):
        raise ValueError("page_size must be in [1, 10000]")
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
    return start, end


def source_url(cfg: Config) -> str:
    start, end = _validate_config(cfg)
    # Coin Metrics end_time is inclusive.  Keep the artifact contract
    # end-exclusive by asking for the preceding UTC calendar day.
    inclusive_end = end - timedelta(days=1)
    params = {
        "assets": "btc",
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "start_time": start.date().isoformat(),
        "end_time": inclusive_end.date().isoformat(),
        "page_size": str(cfg.page_size),
    }
    return f"{cfg.base_url}?{urllib.parse.urlencode(params)}"


def _validate_next_page_url(first_url: str, candidate: str) -> str:
    resolved = urllib.parse.urljoin(first_url, candidate)
    first = urllib.parse.urlparse(first_url)
    next_page = urllib.parse.urlparse(resolved)
    if (
        next_page.scheme != first.scheme
        or next_page.netloc != first.netloc
        or next_page.path != first.path
        or next_page.params
        or next_page.fragment
    ):
        raise ValueError(
            "Coin Metrics next_page_url left the frozen source endpoint"
        )
    first_query = urllib.parse.parse_qs(
        first.query, keep_blank_values=True, strict_parsing=True
    )
    next_query = urllib.parse.parse_qs(
        next_page.query, keep_blank_values=True, strict_parsing=True
    )
    frozen_keys = {
        "assets",
        "metrics",
        "frequency",
        "start_time",
        "end_time",
        "page_size",
    }
    if any(next_query.get(key) != first_query.get(key) for key in frozen_keys):
        raise ValueError(
            "Coin Metrics next_page_url changed the frozen source query"
        )
    if set(next_query) != frozen_keys | {"next_page_token"}:
        raise ValueError(
            "Coin Metrics next_page_url has unexpected query fields"
        )
    token = next_query.get("next_page_token")
    if token is None or len(token) != 1 or not token[0]:
        raise ValueError("Coin Metrics next_page_url has an invalid page token")
    return resolved


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Coin Metrics response used non-standard JSON value {value}")


def _decode_payload(raw: bytes) -> dict[str, Any]:
    payload = json.loads(
        raw.decode("utf-8"),
        parse_float=Decimal,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Coin Metrics response is not an object")
    return payload


def _http_page(cfg: Config, url: str) -> dict[str, Any]:
    for attempt in range(cfg.maximum_retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_sec) as response:
                payload = _decode_payload(response.read())
            return payload
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
            time.sleep(
                max(cfg.request_pause_sec, min(60.0, 2.0**attempt))
            )
    raise AssertionError("unreachable retry loop")


def _normalise_row(
    raw: dict[str, Any], *, start: datetime, end: datetime
) -> OutputRow:
    keys = frozenset(raw)
    if keys != RAW_ROW_KEYS:
        missing = sorted(RAW_ROW_KEYS - keys)
        unexpected = sorted(keys - RAW_ROW_KEYS)
        raise ValueError(
            "Coin Metrics row schema drift: "
            f"missing={missing!r} unexpected={unexpected!r}"
        )
    if raw["asset"] != "btc":
        raise ValueError("Coin Metrics row asset must be exactly 'btc'")
    observed = _parse_utc(raw["time"], "time")
    if observed != observed.replace(hour=0, minute=0, second=0, microsecond=0):
        raise ValueError("Coin Metrics daily observation must be UTC midnight")
    if not start <= observed < end:
        raise ValueError("Coin Metrics row is outside the frozen source interval")
    available = _completion_time(raw["AssetEODCompletionTime"])
    earliest = observed + timedelta(days=1)
    if available < earliest:
        raise ValueError(
            "AssetEODCompletionTime precedes observation + 1 day: "
            f"observation={_format_utc(observed)} "
            f"available={_format_utc(available)} required>={_format_utc(earliest)}"
        )
    return {
        "observation_date": _format_utc(observed),
        "available_at": _format_utc(available),
        "AdrBalCnt": str(_positive_integer(raw["AdrBalCnt"], "AdrBalCnt")),
        "AdrActCnt": str(_positive_integer(raw["AdrActCnt"], "AdrActCnt")),
    }


def download_rows(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[OutputRow], dict[str, Any]]:
    start, end = _validate_config(cfg)
    first_url = source_url(cfg)
    fetch = fetch or (lambda url: _http_page(cfg, url))
    url: str | None = first_url
    seen_urls: set[str] = set()
    by_day: dict[str, OutputRow] = {}
    page_hashes: list[str] = []
    page_lengths: list[int] = []
    expected_rows = (end - start).days
    maximum_pages = math.ceil(expected_rows / cfg.page_size) + 4

    while url is not None:
        if len(page_hashes) >= maximum_pages:
            raise RuntimeError(
                "Coin Metrics pagination exceeded the frozen maximum page count"
            )
        if url in seen_urls:
            raise RuntimeError(f"Coin Metrics pagination loop detected at {url}")
        seen_urls.add(url)
        payload = fetch(url)
        if not isinstance(payload, dict):
            raise RuntimeError("Coin Metrics response is not an object")
        page_hashes.append(canonical_hash(payload))
        if payload.get("error") is not None:
            raise RuntimeError(f"Coin Metrics API error: {payload['error']!r}")
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Coin Metrics response field 'data' must be a list")
        if len(data) > cfg.page_size:
            raise RuntimeError("Coin Metrics page exceeds frozen page_size")
        page_lengths.append(len(data))
        for raw in data:
            if not isinstance(raw, dict):
                raise ValueError("Coin Metrics row must be an object")
            row = _normalise_row(raw, start=start, end=end)
            key = row["observation_date"]
            if key in by_day:
                raise RuntimeError(
                    f"Coin Metrics source contains duplicate day {key}"
                )
            by_day[key] = row
            if len(by_day) > expected_rows:
                raise RuntimeError(
                    "Coin Metrics source exceeded the frozen daily row count"
                )

        next_url = payload.get("next_page_url")
        if next_url is None:
            url = None
        elif isinstance(next_url, str) and next_url:
            if not data:
                raise RuntimeError(
                    "Coin Metrics pagination returned an empty non-terminal page"
                )
            url = _validate_next_page_url(first_url, next_url)
        else:
            raise ValueError(
                "Coin Metrics next_page_url must be a non-empty string or null"
            )
        if url is not None:
            sleep(cfg.request_pause_sec)

    expected_days: list[str] = []
    current = start
    while current < end:
        expected_days.append(_format_utc(current))
        current += timedelta(days=1)
    observed_days = sorted(by_day)
    missing = sorted(set(expected_days) - set(observed_days))
    extra = sorted(set(observed_days) - set(expected_days))
    if missing or extra:
        raise RuntimeError(
            "Coin Metrics daily source is incomplete: "
            f"missing_count={len(missing)} missing_head={missing[:5]!r} "
            f"extra_count={len(extra)} extra_head={extra[:5]!r}"
        )
    rows = [by_day[day] for day in expected_days]
    audit = {
        "source_url": first_url,
        "response_pages": len(page_hashes),
        "response_page_lengths": page_lengths,
        "response_page_sha256": page_hashes,
        "response_chain_sha256": canonical_hash(page_hashes),
        "expected_rows": len(expected_days),
        "observed_rows": len(rows),
        "first_observation": rows[0]["observation_date"],
        "last_observation": rows[-1]["observation_date"],
        "maximum_observation_gap_days": 1,
        "duplicates": 0,
        "missing_days": 0,
        "unexpected_row_fields": 0,
    }
    return rows, audit


def _write_deterministic_csv(path: Path, rows: list[OutputRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with io.TextIOWrapper(
                    gz, encoding="utf-8", newline=""
                ) as wrapper:
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
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
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
    rows, source_audit = download_rows(cfg, fetch=fetch, sleep=sleep)
    output = Path(cfg.output_csv)
    _write_deterministic_csv(output, rows)
    manifest: dict[str, Any] = {
        "protocol_version": 1,
        "candidate": "ARCR-864",
        "config": asdict(cfg),
        "official_catalog_url": OFFICIAL_CATALOG_URL,
        "source_audit": source_audit,
        "output": str(output),
        "output_columns": list(OUTPUT_COLUMNS),
        "output_sha256": sha256_file(output),
        "source_semantics": {
            "AdrBalCnt": (
                "unique addresses holding any positive native-unit balance at interval end"
            ),
            "AdrActCnt": (
                "unique addresses active as originator or recipient during the interval"
            ),
            "AssetEODCompletionTime": (
                "recorded completion timestamp used as source availability"
            ),
        },
        "causal_availability": (
            "available_at equals AssetEODCompletionTime and must be no earlier "
            "than observation UTC midnight plus one day"
        ),
        "revision_boundary": (
            "the output hash freezes this downloaded source vintage; it is not "
            "a historical archive of Coin Metrics revisions"
        ),
        "outcome_boundary": {
            "btc_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "return_or_pnl_fields": 0,
            "post_2023_source_rows_loaded": 0,
            "raw_api_pages_persisted": False,
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
    parser.add_argument("--page-size", type=int, default=Config.page_size)
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
