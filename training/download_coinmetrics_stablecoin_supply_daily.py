"""Download a fixed, chain-specific stablecoin supply basket from Coin Metrics.

Composite ``usdt`` and ``usdc`` histories are intentionally excluded: their
historical rows can carry recent completion timestamps after constituent-chain
reconstruction.  The fixed basket uses only pre-2020 assets and preserves each
row's ``AssetEODCompletionTime`` so research can reject stale backfills rather
than pretending they were known on the observation day.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


BASE_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
ASSETS = (
    "usdt_eth",
    "usdt_trx",
    "usdt_omni",
    "usdc_eth",
    "dai",
    "busd",
    "gusd",
    "pax",
)
METRICS = ("SplyCur", "AssetEODCompletionTime")
OUTPUT_COLUMNS = ("asset", "observation_date", "available_at", "supply")
USER_AGENT = "rllm-research/1.0"


@dataclass(frozen=True)
class CoinMetricsStablecoinSupplyConfig:
    output: str
    manifest: str
    start: str = "2020-01-01"
    end: str = "2026-06-01"
    timeout_sec: float = 30.0
    page_size: int = 10_000
    status: str = "reviewed"
    base_url: str = BASE_URL


def source_url(cfg: CoinMetricsStablecoinSupplyConfig) -> str:
    params = {
        "assets": ",".join(ASSETS),
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "start_time": cfg.start,
        "end_time": cfg.end,
        "page_size": str(int(cfg.page_size)),
        "status": cfg.status,
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
    if "." in text:
        head, tail = text.split(".", 1)
        offset = ""
        for marker in ("+", "-"):
            if marker in tail:
                fraction, offset_tail = tail.split(marker, 1)
                offset = marker + offset_tail
                break
        else:
            fraction = tail
        text = f"{head}.{fraction[:6].ljust(6, '0')}{offset}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_positive_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return format(value, ".15g")


def _format_dt(value: datetime) -> str:
    timespec = "microseconds" if value.microsecond else "seconds"
    return value.isoformat(sep=" ", timespec=timespec)


def _normalise_row(raw: dict[str, Any]) -> dict[str, str]:
    asset = str(raw.get("asset", "")).strip().lower()
    if asset not in ASSETS:
        raise ValueError(f"unexpected stablecoin asset: {asset!r}")
    observed = _parse_utc_naive(raw.get("time"))
    available_seconds = _parse_positive_number(
        raw.get("AssetEODCompletionTime"), "AssetEODCompletionTime"
    )
    available = datetime.fromtimestamp(available_seconds, tz=timezone.utc).replace(tzinfo=None)
    if available < observed + timedelta(days=1):
        raise ValueError(
            "stablecoin supply was marked complete before the UTC day ended: "
            f"asset={asset} observation={_format_dt(observed)} availability={_format_dt(available)}"
        )
    return {
        "asset": asset,
        "observation_date": _format_dt(observed),
        "available_at": _format_dt(available),
        "supply": _format_number(_parse_positive_number(raw.get("SplyCur"), "SplyCur")),
    }


def download_rows(
    cfg: CoinMetricsStablecoinSupplyConfig,
    *,
    fetch: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, str]], str]:
    fetch = fetch or (lambda url: get_json_url(url, timeout_sec=cfg.timeout_sec))
    first_url = source_url(cfg)
    url: str | None = first_url
    seen_urls: set[str] = set()
    by_key: dict[tuple[str, str], dict[str, str]] = {}
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
            by_key[(row["observation_date"], row["asset"])] = row
        next_url = payload.get("next_page_url")
        if not next_url:
            url = None
        elif isinstance(next_url, str):
            url = urllib.parse.urljoin(first_url, next_url)
        else:
            raise ValueError(f"next_page_url must be a string or null, got {next_url!r}")
    rows = [by_key[key] for key in sorted(by_key)]
    return rows, first_url


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
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cfg: CoinMetricsStablecoinSupplyConfig) -> dict[str, Any]:
    if _parse_utc_naive(cfg.end) < _parse_utc_naive(cfg.start):
        raise ValueError("end must be on or after start")
    rows, first_url = download_rows(cfg)
    output = Path(cfg.output)
    _write_csv_gz(output, rows)
    counts = {asset: 0 for asset in ASSETS}
    maximum_lag_days = {asset: 0.0 for asset in ASSETS}
    for row in rows:
        asset = row["asset"]
        counts[asset] += 1
        lag = (
            _parse_utc_naive(row["available_at"])
            - _parse_utc_naive(row["observation_date"])
        ).total_seconds() / 86_400.0
        maximum_lag_days[asset] = max(maximum_lag_days[asset], lag)
    manifest = {
        "config": asdict(cfg),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "provider_api_version": "Coin Metrics API v4",
        "source_url": first_url,
        "fixed_asset_universe": list(ASSETS),
        "excluded_composites": ["usdt", "usdc"],
        "output": str(output),
        "rows": len(rows),
        "row_counts": counts,
        "row_range": {
            "start": rows[0]["observation_date"] if rows else None,
            "end": rows[-1]["observation_date"] if rows else None,
        },
        "maximum_completion_lag_days": maximum_lag_days,
        "columns": list(OUTPUT_COLUMNS),
        "sha256": _sha256(output),
        "availability_rule": (
            "retain exact AssetEODCompletionTime; research may emit a signal only when all "
            "basket rows completed between observation+1d and observation+3d"
        ),
        "vintage_limitation": (
            "reviewed latest-snapshot history, not a point-in-time vintage archive; completion "
            "timestamps and hashes do not prove the value matched the value published historically"
        ),
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return manifest


def parse_args() -> CoinMetricsStablecoinSupplyConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=CoinMetricsStablecoinSupplyConfig.start)
    parser.add_argument("--end", default=CoinMetricsStablecoinSupplyConfig.end)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--timeout-sec", type=float, default=CoinMetricsStablecoinSupplyConfig.timeout_sec)
    parser.add_argument("--page-size", type=int, default=CoinMetricsStablecoinSupplyConfig.page_size)
    parser.add_argument("--status", default=CoinMetricsStablecoinSupplyConfig.status)
    parser.add_argument("--base-url", default=CoinMetricsStablecoinSupplyConfig.base_url)
    return CoinMetricsStablecoinSupplyConfig(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
