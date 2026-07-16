"""Download BTC daily network metrics from the Coin Metrics Community API.

The output is deliberately availability-aware: Coin Metrics daily observations are
not usable at the observation timestamp itself.  ``AssetEODCompletionTime`` is
converted from Unix time to a naive UTC timestamp and every row is required to be
available no earlier than ``time + 1 day`` before it is written.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import hashlib
import json
import math
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

BASE_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ("AdrActCnt", "TxCnt", "TxTfrCnt", "AssetEODCompletionTime")
OUTPUT_COLUMNS = ("observation_date", "available_at", "AdrActCnt", "TxCnt", "TxTfrCnt")
USER_AGENT = "rllm-research/1.0"


@dataclass(frozen=True)
class CoinMetricsBTCNetworkDailyConfig:
    output: str
    manifest: str
    start: str = "2020-09-01"
    end: str = "2026-06-01"
    timeout_sec: float = 30.0
    page_size: int = 10000
    base_url: str = BASE_URL


def source_url(cfg: CoinMetricsBTCNetworkDailyConfig) -> str:
    params = {
        "assets": "btc",
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "start_time": cfg.start,
        "end_time": cfg.end,
        "page_size": str(int(cfg.page_size)),
    }
    return f"{cfg.base_url}?{urllib.parse.urlencode(params)}"


def get_json_url(url: str, *, timeout_sec: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_utc_naive(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"expected non-empty UTC timestamp string, got {value!r}")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Coin Metrics timestamps can carry nanosecond precision while Python's
    # stdlib datetime accepts at most microseconds.  Truncate, do not round, so
    # an observation at midnight remains that exact UTC day.
    if "." in text:
        head, tail = text.split(".", 1)
        offset = ""
        for marker in ("+", "-"):
            if marker in tail:
                frac, offset_tail = tail.split(marker, 1)
                offset = marker + offset_tail
                break
        else:
            frac = tail
        text = f"{head}.{frac[:6].ljust(6, '0')}{offset}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_unix_utc_naive(value: Any) -> datetime:
    seconds = _parse_positive_number(value, "AssetEODCompletionTime")
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


def _parse_positive_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return format(value, ".15g")


def _format_dt(value: datetime) -> str:
    if value.microsecond:
        return value.isoformat(sep=" ", timespec="microseconds")
    return value.isoformat(sep=" ", timespec="seconds")


def _normalise_row(raw: dict[str, Any]) -> dict[str, str]:
    observed = _parse_utc_naive(raw.get("time"))
    available = _parse_unix_utc_naive(raw.get("AssetEODCompletionTime"))
    earliest_usable = observed + timedelta(days=1)
    if available < earliest_usable:
        raise ValueError(
            "AssetEODCompletionTime availability precedes required daily lag: "
            f"time={_format_dt(observed)} availability={_format_dt(available)} required>={_format_dt(earliest_usable)}"
        )
    return {
        "observation_date": _format_dt(observed),
        "available_at": _format_dt(available),
        "AdrActCnt": _format_number(_parse_positive_number(raw.get("AdrActCnt"), "AdrActCnt")),
        "TxCnt": _format_number(_parse_positive_number(raw.get("TxCnt"), "TxCnt")),
        "TxTfrCnt": _format_number(_parse_positive_number(raw.get("TxTfrCnt"), "TxTfrCnt")),
    }


def download_rows(
    cfg: CoinMetricsBTCNetworkDailyConfig,
    *,
    fetch: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, str]], str]:
    fetch = fetch or (lambda url: get_json_url(url, timeout_sec=cfg.timeout_sec))
    first_url = source_url(cfg)
    url: str | None = first_url
    seen_urls: set[str] = set()
    by_time: dict[str, dict[str, str]] = {}
    while url:
        if url in seen_urls:
            raise RuntimeError(f"Coin Metrics pagination loop detected at {url}")
        seen_urls.add(url)
        payload = fetch(url)
        if "error" in payload:
            raise RuntimeError(f"Coin Metrics error: {payload['error']}")
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise ValueError("Coin Metrics response field 'data' must be a list")
        for raw in data:
            if not isinstance(raw, dict):
                raise ValueError(f"Coin Metrics row must be an object, got {raw!r}")
            row = _normalise_row(raw)
            by_time[row["observation_date"]] = row
        next_url = payload.get("next_page_url")
        if not next_url:
            url = None
        elif isinstance(next_url, str):
            url = urllib.parse.urljoin(first_url, next_url)
        else:
            raise ValueError(f"next_page_url must be a string or null, got {next_url!r}")
    return [by_time[key] for key in sorted(by_time)], first_url


def _write_csv_gz(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            # newline="" plus an explicit lineterminator keeps gzip output
            # deterministic across platforms.
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as wrapper:
                writer = csv.DictWriter(wrapper, fieldnames=list(OUTPUT_COLUMNS), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cfg: CoinMetricsBTCNetworkDailyConfig) -> dict[str, Any]:
    if _parse_utc_naive(cfg.end) < _parse_utc_naive(cfg.start):
        raise ValueError("end must be on or after start")
    rows, first_url = download_rows(cfg)
    output = Path(cfg.output)
    _write_csv_gz(output, rows)
    digest = _sha256(output)
    manifest = {
        "config": asdict(cfg),
        "source_url": first_url,
        "output": str(output),
        "rows": len(rows),
        "row_range": {
            "start": rows[0]["observation_date"] if rows else None,
            "end": rows[-1]["observation_date"] if rows else None,
        },
        "columns": list(OUTPUT_COLUMNS),
        "sha256": digest,
        "availability_rule": "AssetEODCompletionTime must be >= observation time + 1 day; timestamps are naive UTC",
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return manifest


def parse_args() -> CoinMetricsBTCNetworkDailyConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=CoinMetricsBTCNetworkDailyConfig.start)
    parser.add_argument("--end", default=CoinMetricsBTCNetworkDailyConfig.end)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--timeout-sec", type=float, default=CoinMetricsBTCNetworkDailyConfig.timeout_sec)
    parser.add_argument("--page-size", type=int, default=CoinMetricsBTCNetworkDailyConfig.page_size)
    parser.add_argument("--base-url", default=CoinMetricsBTCNetworkDailyConfig.base_url)
    return CoinMetricsBTCNetworkDailyConfig(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
