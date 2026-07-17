"""Download availability-aware BTC miner-security metrics from Coin Metrics."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from training.download_coinmetrics_btc_network_daily import (
    BASE_URL,
    _format_dt,
    _format_number,
    _parse_unix_utc_naive,
    _parse_utc_naive,
    get_json_url,
)


METRICS = (
    "HashRate",
    "IssTotNtv",
    "FeeTotNtv",
    "BlkCnt",
    "AssetEODCompletionTime",
)
OUTPUT_COLUMNS = (
    "observation_date",
    "available_at",
    "HashRate",
    "IssTotNtv",
    "FeeTotNtv",
    "BlkCnt",
)


@dataclass(frozen=True)
class Config:
    output: str
    manifest: str
    start: str = "2019-01-01"
    end: str = "2023-12-31"
    timeout_sec: float = 30.0
    page_size: int = 10_000
    base_url: str = BASE_URL


def source_url(cfg: Config) -> str:
    params = {
        "assets": "btc",
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "start_time": cfg.start,
        "end_time": cfg.end,
        "page_size": str(int(cfg.page_size)),
    }
    return f"{cfg.base_url}?{urllib.parse.urlencode(params)}"


def _parse_nonnegative_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    return number


def _parse_positive_number(value: Any, name: str) -> float:
    number = _parse_nonnegative_number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _normalise_row(raw: dict[str, Any]) -> dict[str, str]:
    observed = _parse_utc_naive(raw.get("time"))
    available = _parse_unix_utc_naive(raw.get("AssetEODCompletionTime"))
    earliest_usable = observed + timedelta(days=1)
    if available < earliest_usable:
        raise ValueError(
            "AssetEODCompletionTime availability precedes required daily lag: "
            f"time={_format_dt(observed)} availability={_format_dt(available)} "
            f"required>={_format_dt(earliest_usable)}"
        )
    return {
        "observation_date": _format_dt(observed),
        "available_at": _format_dt(available),
        "HashRate": _format_number(
            _parse_positive_number(raw.get("HashRate"), "HashRate")
        ),
        "IssTotNtv": _format_number(
            _parse_positive_number(raw.get("IssTotNtv"), "IssTotNtv")
        ),
        "FeeTotNtv": _format_number(
            _parse_nonnegative_number(raw.get("FeeTotNtv"), "FeeTotNtv")
        ),
        "BlkCnt": _format_number(
            _parse_positive_number(raw.get("BlkCnt"), "BlkCnt")
        ),
    }


def download_rows(
    cfg: Config,
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
        rows = payload.get("data", [])
        if not isinstance(rows, list):
            raise ValueError("Coin Metrics response field 'data' must be a list")
        for raw in rows:
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
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as wrapper:
                writer = csv.DictWriter(
                    wrapper, fieldnames=list(OUTPUT_COLUMNS), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cfg: Config) -> dict[str, Any]:
    if _parse_utc_naive(cfg.end) < _parse_utc_naive(cfg.start):
        raise ValueError("end must be on or after start")
    rows, first_url = download_rows(cfg)
    output = Path(cfg.output)
    _write_csv_gz(output, rows)
    digest = _sha256(output)
    manifest = {
        "config": asdict(cfg),
        "source_url": first_url,
        "official_catalog_url": (
            "https://community-api.coinmetrics.io/v4/catalog/metrics?"
            "metrics=HashRate%2CIssTotNtv%2CFeeTotNtv%2CBlkCnt%2CAssetEODCompletionTime"
        ),
        "output": str(output),
        "rows": len(rows),
        "row_range": {
            "start": rows[0]["observation_date"] if rows else None,
            "end": rows[-1]["observation_date"] if rows else None,
        },
        "columns": list(OUTPUT_COLUMNS),
        "sha256": digest,
        "availability_rule": (
            "AssetEODCompletionTime must be >= observation time + 1 day; "
            "timestamps are naive UTC"
        ),
        "source_semantics": {
            "HashRate": "Coin Metrics mean estimated network hash-solving rate",
            "IssTotNtv": "new native units issued in the interval",
            "FeeTotNtv": "transactor-paid fees excluding new issuance",
            "BlkCnt": "main-chain blocks created in the interval",
        },
        "revision_boundary": (
            "AssetEODCompletionTime freezes semantic publication latency and the file "
            "hash freezes this downloaded vintage; it is not a historical revision-vintage archive"
        ),
        "excluded_on_purpose": {
            "price_or_market_cap_metrics": "keep the source independent from BTC market outcomes",
            "exchange_flow_metrics": (
                "address-tag history can change and is inadmissible without point-in-time tags"
            ),
            "post_2023_rows": "preserve 2024 and later source and outcome windows",
        },
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return manifest


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end", default=Config.end)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--timeout-sec", type=float, default=Config.timeout_sec)
    parser.add_argument("--page-size", type=int, default=Config.page_size)
    parser.add_argument("--base-url", default=Config.base_url)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
