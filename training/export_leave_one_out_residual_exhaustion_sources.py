"""Physically freeze the 2023-2024 LORE selection source prefix."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.preregister_leave_one_out_residual_exhaustion import canonical_hash, protocol


SYMBOLS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")
EXPECTED_PROTOCOL_HASH = "18480ed99902cecc126fcd4e5d9f5df40c98e65878bfecfb547e2941084be840"
MARKET_COLUMNS = (
    "date", "open", "high", "low", "close", "volume", "quote_asset_volume",
    "number_of_trades", "taker_buy_base", "taker_buy_quote", "tic", "day",
)
FUNDING_COLUMNS = ("date", "symbol", "funding_rate", "funding_time", "mark_price")
DEFAULT_MARKET_DIR = "data/binance_um_pool_5m_2023_2026"
DEFAULT_AUX_DIR = "data/binance_um_aux_2023_2026"
DEFAULT_OUTPUT_DIR = "data/binance_um_lore_2023_2024"
DEFAULT_MANIFEST = "results/leave_one_out_residual_exhaustion_v1_source_manifest_2026-07-17.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    fallback = Path("/home/pakchu/rllm") / path
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(path)


def deterministic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(
                    text,
                    index=False,
                    date_format="%Y-%m-%d %H:%M:%S.%f",
                    float_format="%.17g",
                    lineterminator="\n",
                )


def validate_market(
    frame: pd.DataFrame,
    symbol: str,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
    *,
    exact_grid: bool = True,
) -> pd.DataFrame:
    missing = set(MARKET_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{symbol} missing market columns: {sorted(missing)}")
    out = frame.loc[:, MARKET_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="raise").dt.tz_convert(None)
    if out["date"].duplicated().any() or not out["date"].is_monotonic_increasing:
        raise ValueError(f"{symbol} duplicate or unsorted market timestamps")
    out = out.loc[(out["date"] >= start) & (out["date"] < end)].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{symbol} empty market prefix")
    if not out["tic"].astype(str).eq(symbol).all():
        raise ValueError(f"{symbol} market identity mismatch")
    numeric = [c for c in MARKET_COLUMNS if c not in {"date", "tic"}]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="raise")
    if not np.isfinite(out[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{symbol} non-finite market value")
    prices = out[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if (prices <= 0).any():
        raise ValueError(f"{symbol} non-positive OHLC")
    if (out["high"] < out[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"{symbol} high below open/close")
    if (out["low"] > out[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"{symbol} low above open/close")
    if (out["quote_asset_volume"] < 0).any():
        raise ValueError(f"{symbol} negative quote volume")
    tolerance = np.maximum(1e-6, out["quote_asset_volume"].to_numpy(dtype=float) * 1e-10)
    buy = out["taker_buy_quote"].to_numpy(dtype=float)
    quote = out["quote_asset_volume"].to_numpy(dtype=float)
    if (buy < -tolerance).any() or (buy > quote + tolerance).any():
        raise ValueError(f"{symbol} taker buy outside quote volume")
    if ((quote == 0.0) & (np.abs(buy) > tolerance)).any():
        raise ValueError(f"{symbol} nonzero taker buy on zero-volume bar")
    if exact_grid:
        expected = pd.date_range(start, end - pd.Timedelta(minutes=5), freq="5min")
        actual = pd.DatetimeIndex(out["date"])
        if not actual.equals(expected):
            missing_count = len(expected.difference(actual))
            extra_count = len(actual.difference(expected))
            raise ValueError(f"{symbol} market grid mismatch missing={missing_count} extra={extra_count}")
    if not (out["date"] < end).all():
        raise ValueError(f"{symbol} future market row escaped cutoff")
    return out


def validate_funding(
    frame: pd.DataFrame,
    symbol: str,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
) -> pd.DataFrame:
    missing = set(FUNDING_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{symbol} missing funding columns: {sorted(missing)}")
    out = frame.loc[:, FUNDING_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="raise").dt.tz_convert(None)
    out["funding_time"] = pd.to_numeric(out["funding_time"], errors="raise").astype("int64")
    out["funding_rate"] = pd.to_numeric(out["funding_rate"], errors="raise")
    out["mark_price"] = pd.to_numeric(out["mark_price"], errors="coerce")
    event_time = pd.to_datetime(out["funding_time"], unit="ms", utc=True).dt.tz_convert(None)
    out = out.loc[(event_time >= start) & (event_time < end)].copy().reset_index(drop=True)
    event_time = pd.to_datetime(out["funding_time"], unit="ms", utc=True).dt.tz_convert(None)
    if out.empty:
        raise ValueError(f"{symbol} empty funding prefix")
    if not out["symbol"].astype(str).eq(symbol).all():
        raise ValueError(f"{symbol} funding identity mismatch")
    if out["funding_time"].duplicated().any() or not out["funding_time"].is_monotonic_increasing:
        raise ValueError(f"{symbol} duplicate or unsorted funding timestamps")
    if not np.isfinite(out["funding_rate"].to_numpy(dtype=float)).all():
        raise ValueError(f"{symbol} non-finite funding rate")
    date_error = (out["date"] - event_time).abs()
    if (date_error > pd.Timedelta(seconds=1)).any():
        raise ValueError(f"{symbol} funding date/time mismatch")
    if not (event_time < end).all():
        raise ValueError(f"{symbol} future funding row escaped cutoff")
    return out


def run(
    market_dir: str = DEFAULT_MARKET_DIR,
    aux_dir: str = DEFAULT_AUX_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    manifest_path: str = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    frozen_protocol = protocol()
    if canonical_hash(frozen_protocol) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("LORE preregistration protocol drifted")
    target = Path(output_dir)
    records: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        market_input = resolve(Path(market_dir) / f"{symbol}_5m_2023-01_2026-05.csv.gz")
        funding_input = resolve(Path(aux_dir) / f"{symbol}_funding_2023-01-01_2026-06-01.csv.gz")
        market = validate_market(pd.read_csv(market_input), symbol)
        funding = validate_funding(pd.read_csv(funding_input), symbol)
        market_output = target / f"{symbol}_5m_2023_2024.csv.gz"
        funding_output = target / f"{symbol}_funding_2023_2024.csv.gz"
        deterministic_csv_gz(market, market_output)
        deterministic_csv_gz(funding, funding_output)
        reread_market = validate_market(pd.read_csv(market_output), symbol)
        reread_funding = validate_funding(pd.read_csv(funding_output), symbol)
        records.append({
            "symbol": symbol,
            "input_market": str(market_input),
            "input_market_sha256": sha256_file(market_input),
            "output_market": str(market_output),
            "output_market_sha256": sha256_file(market_output),
            "market_rows": len(reread_market),
            "zero_quote_volume_bars": int((reread_market["quote_asset_volume"] == 0).sum()),
            "market_min": str(reread_market["date"].min()),
            "market_max": str(reread_market["date"].max()),
            "input_funding": str(funding_input),
            "input_funding_sha256": sha256_file(funding_input),
            "output_funding": str(funding_output),
            "output_funding_sha256": sha256_file(funding_output),
            "funding_rows": len(reread_funding),
            "funding_min": str(pd.to_datetime(reread_funding["funding_time"], unit="ms").min()),
            "funding_max": str(pd.to_datetime(reread_funding["funding_time"], unit="ms").max()),
        })
    manifest: dict[str, Any] = {
        "protocol_version": "lore_v1_source_freeze_2026-07-17",
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "outcomes_calculated": False,
        "selection_prefix": {"start": str(START), "end_exclusive": str(END)},
        "future_2025_plus_rows_written": 0,
        "symbols": list(SYMBOLS),
        "market_contract": "exact complete 5m grid; no fill; valid OHLC/taker identity",
        "funding_contract": "exact reported event rows; no fill or synthetic rate",
        "records": records,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    out = Path(manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-dir", default=DEFAULT_MARKET_DIR)
    parser.add_argument("--aux-dir", default=DEFAULT_AUX_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(run(args.market_dir, args.aux_dir, args.output_dir, args.manifest), indent=2))


if __name__ == "__main__":
    main()
