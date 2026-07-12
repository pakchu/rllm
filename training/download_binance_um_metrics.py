"""Download Binance USD-M daily historical metrics archives.

The public Binance archive exposes 5-minute snapshots containing open interest,
top-trader/global long-short ratios, and taker long-short volume ratios.  The
archive timestamp is treated as the observation timestamp only; downstream
research must delay every row by at least one complete source bar because the
archive does not record the original publication/arrival timestamp.

Official archive root:
https://data.binance.vision/?prefix=data/futures/um/daily/metrics/
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
METRIC_COLUMNS = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)


@dataclass(frozen=True)
class MetricsDownloadConfig:
    output_csv: str
    start: str = "2020-09-01"
    end: str = "2026-06-01"
    symbol: str = "BTCUSDT"
    workers: int = 16
    retries: int = 3
    timeout_sec: float = 30.0
    base_url: str = BASE_URL


def archive_url(*, base_url: str, symbol: str, day: str) -> str:
    return f"{base_url.rstrip('/')}/{symbol}/{symbol}-metrics-{day}.zip"


def _parse_archive(blob: bytes, *, symbol: str, day: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in {symbol} {day} archive, found {names}")
        with archive.open(names[0]) as fh:
            frame = pd.read_csv(fh)
    missing = [column for column in METRIC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{symbol} {day} metrics archive missing columns: {missing}")
    frame = frame.loc[:, list(METRIC_COLUMNS)].copy()
    frame["create_time"] = pd.to_datetime(frame["create_time"], utc=True, errors="raise").dt.tz_convert(None)
    if not frame["symbol"].astype(str).eq(symbol).all():
        raise ValueError(f"{symbol} {day} archive contains a different symbol")
    for column in METRIC_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.sort_values("create_time").drop_duplicates("create_time", keep="last").reset_index(drop=True)


def _download_day(cfg: MetricsDownloadConfig, day: str) -> pd.DataFrame:
    url = archive_url(base_url=cfg.base_url, symbol=cfg.symbol, day=day)
    request = urllib.request.Request(url, headers={"User-Agent": "rllm-research/1.0"})
    last_error: Exception | None = None
    for attempt in range(max(1, int(cfg.retries))):
        try:
            with urllib.request.urlopen(request, timeout=float(cfg.timeout_sec)) as response:
                return _parse_archive(response.read(), symbol=cfg.symbol, day=day)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, zipfile.BadZipFile) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                break
            if attempt + 1 < max(1, int(cfg.retries)):
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"failed to download {cfg.symbol} metrics for {day}: {last_error}")


def run(cfg: MetricsDownloadConfig) -> dict[str, Any]:
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    if end < start:
        raise ValueError("end must be on or after start")
    days = [day.strftime("%Y-%m-%d") for day in pd.date_range(start, end, freq="1D")]
    frames: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(cfg.workers))) as pool:
        futures = {pool.submit(_download_day, cfg, day): day for day in days}
        for future in as_completed(futures):
            day = futures[future]
            frames[day] = future.result()
    metrics = pd.concat([frames[day] for day in days], ignore_index=True)
    metrics = metrics.sort_values("create_time").drop_duplicates("create_time", keep="last").reset_index(drop=True)
    output = Path(cfg.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False, compression="infer")
    expected_rows = len(days) * 288
    report = {
        "config": asdict(cfg),
        "output_csv": str(output),
        "days": len(days),
        "rows": int(len(metrics)),
        "expected_rows_at_5m": int(expected_rows),
        "coverage_fraction": float(len(metrics) / expected_rows) if expected_rows else 0.0,
        "start": str(metrics["create_time"].min()),
        "end": str(metrics["create_time"].max()),
        "columns": list(metrics.columns),
        "causality_requirement": "shift by >=1 complete 5m source bar before model or rule use",
        "official_archive": "https://data.binance.vision/?prefix=data/futures/um/daily/metrics/",
    }
    report_path = output.with_suffix(output.suffix + ".summary.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> MetricsDownloadConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--start", default=MetricsDownloadConfig.start)
    parser.add_argument("--end", default=MetricsDownloadConfig.end)
    parser.add_argument("--symbol", default=MetricsDownloadConfig.symbol)
    parser.add_argument("--workers", type=int, default=MetricsDownloadConfig.workers)
    parser.add_argument("--retries", type=int, default=MetricsDownloadConfig.retries)
    parser.add_argument("--timeout-sec", type=float, default=MetricsDownloadConfig.timeout_sec)
    parser.add_argument("--base-url", default=MetricsDownloadConfig.base_url)
    return MetricsDownloadConfig(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
