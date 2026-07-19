"""Build a deterministic, price-free six-alt USD-M hourly flow panel.

The transform reads only timestamp, quote-volume, trade-count, and taker-buy
quote fields from the frozen repository-local Binance USD-M five-minute pool.
OHLC, base volume, and every BTC outcome are deliberately outside the read
schema.  Each output row becomes available only at the right edge of its
completed UTC hour.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
BUILDER_PATH = Path("training/build_six_alt_price_free_flow_panel.py")
SYMBOLS = (
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
)
RAW_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "tic",
    "day",
)
ALLOWED_READ_COLUMNS = (
    "date",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_quote",
)
OUTPUT_COLUMNS = (
    "source_hour_open_utc",
    "feature_available_time_utc",
    "symbol",
    "source_hour_open_us",
    "feature_available_time_us",
    "quote_volume_usdt",
    "trade_count",
    "taker_buy_quote_usdt",
    "taker_sell_quote_usdt",
    "signed_taker_flow_usdt",
    "taker_flow_fraction",
    "mean_ticket_usdt",
    "source_bar_count",
    "positive_activity_bar_count",
    "source_complete",
    "feature_valid",
    "feature_invalid_reason",
)
FROZEN_INPUT_SHA256 = {
    "ETHUSDT": "68492d340e6617cda65dcc6cb42ba9138d0866ed0ce7fc1328d66f4e01b55e82",
    "SOLUSDT": "4216e63393ad66a0e4aab6d8b07199f209188a513c680a489c6d055a604607d1",
    "BNBUSDT": "b173cb40adf6a81b23a1b1ede988e001b8f7abbf7c184c1a3f1f96b1fc5e1263",
    "XRPUSDT": "ccb993e92efa586db0d8c8c5d4d1734221b50a50db04da14f447f7905cf50593",
    "DOGEUSDT": "a60e33dc12e1480d284e5fdbda18a06e5aeeb59ee3aa542521bcea211d01861a",
    "ADAUSDT": "9f63ea66c2872f7941b9fde742746ba808fd0efb2a3b35a81bf4eb8cfc6479e1",
}
FROZEN_SUMMARY_SHA256 = (
    "9010bf1110e2555581942d6a0400d98466b64a5bee355c05ce5b7d2c85af173a"
)


@dataclass(frozen=True)
class BuildConfig:
    input_dir: str = "data/binance_um_pool_5m_2023_2026"
    input_summary: str | None = (
        "data/binance_um_pool_5m_2023_2026/"
        "download_summary_5m_2023-01_2026-05.json"
    )
    output_dir: str = "data/binance_six_alt_price_free_flow_2023_2026"
    start: str = "2023-01-01"
    end: str = "2026-06-01"
    symbols: tuple[str, ...] = SYMBOLS
    enforce_frozen_inputs: bool = True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def deterministic_gzip_csv(frame: pd.DataFrame, path: str | Path) -> None:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.encode())
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(buffer.getvalue())


def _read_header(path: Path) -> tuple[str, ...]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = tuple(next(csv.reader(handle)))
    if header != RAW_COLUMNS:
        raise ValueError(f"unexpected six-alt source columns in {path}: {header}")
    return header


def _source_path(input_dir: str | Path, symbol: str) -> Path:
    paths = sorted(Path(input_dir).glob(f"{symbol}_5m_*.csv.gz"))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one source file for {symbol}, found {paths}")
    return paths[0]


def _expected_five_minute_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="5min", inclusive="left")


def _expected_hour_boundaries(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start + pd.Timedelta(hours=1), end, freq="1h")


def load_symbol_hourly(
    path: str | Path,
    *,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one source using the frozen price-free read schema."""
    source_path = Path(path)
    _read_header(source_path)
    raw = cast(
        pd.DataFrame,
        pd.read_csv(
            source_path,
            usecols=cast(Any, list(ALLOWED_READ_COLUMNS)),
            dtype={
                "quote_asset_volume": "float64",
                "number_of_trades": "int64",
                "taker_buy_quote": "float64",
            },
        ),
    )
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="raise").dt.tz_localize(
        None
    )
    expected = _expected_five_minute_grid(start, end)
    observed = pd.DatetimeIndex(raw["date"])
    if raw["date"].duplicated().any() or not observed.equals(expected):
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        raise RuntimeError(
            f"{symbol} source is not the exact frozen five-minute grid; "
            f"missing={missing.tolist()[:10]}, extra={extra.tolist()[:10]}"
        )
    numeric = cast(pd.DataFrame, raw.loc[:, list(ALLOWED_READ_COLUMNS[1:])])
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"{symbol} source contains non-finite flow observables")
    if (numeric < 0.0).any().any():
        raise ValueError(f"{symbol} source contains negative flow observables")
    tolerance = np.maximum(1e-6, raw["quote_asset_volume"].abs() * 1e-12)
    if (raw["taker_buy_quote"] > raw["quote_asset_volume"] + tolerance).any():
        raise ValueError(f"{symbol} taker-buy quote exceeds total quote volume")

    raw["source_hour_open_utc"] = raw["date"].dt.floor("1h")
    raw["positive_activity"] = (
        raw["quote_asset_volume"].gt(0.0) & raw["number_of_trades"].gt(0)
    )
    grouped = raw.groupby("source_hour_open_utc", sort=True, observed=True)
    hourly = grouped.agg(
        quote_volume_usdt=("quote_asset_volume", "sum"),
        trade_count=("number_of_trades", "sum"),
        taker_buy_quote_usdt=("taker_buy_quote", "sum"),
        source_bar_count=("date", "size"),
        positive_activity_bar_count=("positive_activity", "sum"),
    ).reset_index()
    expected_boundaries = _expected_hour_boundaries(start, end)
    hourly["feature_available_time_utc"] = hourly["source_hour_open_utc"] + pd.Timedelta(
        hours=1
    )
    if not pd.DatetimeIndex(hourly["feature_available_time_utc"]).equals(
        expected_boundaries
    ):
        raise RuntimeError(f"{symbol} completed-hour boundary grid changed")
    hourly["symbol"] = symbol
    hourly["taker_sell_quote_usdt"] = (
        hourly["quote_volume_usdt"] - hourly["taker_buy_quote_usdt"]
    ).clip(lower=0.0)
    hourly["signed_taker_flow_usdt"] = (
        2.0 * hourly["taker_buy_quote_usdt"] - hourly["quote_volume_usdt"]
    )
    hourly["source_complete"] = hourly["source_bar_count"].eq(12)
    hourly["feature_valid"] = (
        hourly["source_complete"]
        & hourly["positive_activity_bar_count"].eq(12)
        & hourly["quote_volume_usdt"].gt(0.0)
        & hourly["trade_count"].gt(0)
    )
    hourly["taker_flow_fraction"] = (
        hourly["signed_taker_flow_usdt"] / hourly["quote_volume_usdt"]
    ).where(hourly["feature_valid"])
    hourly["mean_ticket_usdt"] = (
        hourly["quote_volume_usdt"] / hourly["trade_count"]
    ).where(hourly["feature_valid"])
    hourly["feature_invalid_reason"] = np.where(
        hourly["feature_valid"], "ok", "nonpositive_activity_within_completed_hour"
    )
    hourly["source_hour_open_us"] = (
        hourly["source_hour_open_utc"].astype("int64") // 1_000
    )
    hourly["feature_available_time_us"] = (
        hourly["feature_available_time_utc"].astype("int64") // 1_000
    )
    hourly = hourly.loc[:, OUTPUT_COLUMNS]
    if not hourly["source_complete"].all():
        raise RuntimeError(f"{symbol} hourly aggregation lost a five-minute bar")
    valid_fraction = hourly.loc[hourly["feature_valid"], "taker_flow_fraction"]
    if not valid_fraction.between(-1.0 - 1e-12, 1.0 + 1e-12).all():
        raise ValueError(f"{symbol} taker-flow fraction lies outside [-1,1]")

    invalid = hourly.loc[~hourly["feature_valid"], "feature_available_time_utc"]
    metadata = {
        "symbol": symbol,
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "input_rows": int(len(raw)),
        "output_rows": int(len(hourly)),
        "first_input_time_utc": cast(pd.Timestamp, raw["date"].min()).isoformat(),
        "last_input_time_utc": cast(pd.Timestamp, raw["date"].max()).isoformat(),
        "invalid_feature_rows": int(len(invalid)),
        "invalid_feature_boundaries_utc": [value.isoformat() for value in invalid],
    }
    return hourly, metadata


def _validate_config(cfg: BuildConfig) -> tuple[pd.Timestamp, pd.Timestamp, tuple[str, ...]]:
    start = cast(pd.Timestamp, pd.Timestamp(cfg.start))
    end = cast(pd.Timestamp, pd.Timestamp(cfg.end))
    symbols = tuple(symbol.strip().upper() for symbol in cfg.symbols)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start.minute or start.second or end.minute or end.second:
        raise ValueError("source boundaries must be exact UTC hours")
    if symbols != SYMBOLS:
        raise ValueError("six-alt price-free source universe is frozen")
    return start, end, symbols


def _validate_frozen_inputs(cfg: BuildConfig, paths: dict[str, Path]) -> None:
    if not cfg.enforce_frozen_inputs:
        return
    if cfg.start != BuildConfig.start or cfg.end != BuildConfig.end:
        raise ValueError("frozen input enforcement requires the declared full range")
    for symbol, path in paths.items():
        if sha256_file(path) != FROZEN_INPUT_SHA256[symbol]:
            raise RuntimeError(f"frozen six-alt input hash mismatch: {path}")
    if cfg.input_summary is None:
        raise RuntimeError("frozen source summary is required")
    summary_path = Path(cfg.input_summary)
    if sha256_file(summary_path) != FROZEN_SUMMARY_SHA256:
        raise RuntimeError("frozen six-alt source summary hash mismatch")
    summary = json.loads(summary_path.read_text())
    if tuple(summary["config"]["symbols"].split(",")) != SYMBOLS:
        raise RuntimeError("frozen six-alt source summary universe changed")


def build(cfg: BuildConfig = BuildConfig()) -> dict[str, Any]:
    start, end, symbols = _validate_config(cfg)
    paths = {symbol: _source_path(cfg.input_dir, symbol) for symbol in symbols}
    _validate_frozen_inputs(cfg, paths)

    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, Any]] = []
    for symbol in symbols:
        frame, metadata = load_symbol_hourly(
            paths[symbol], symbol=symbol, start=start, end=end
        )
        frames.append(frame)
        inputs.append(metadata)
    panel = cast(
        pd.DataFrame,
        pd.concat(frames, ignore_index=True)
        .sort_values(["feature_available_time_utc", "symbol"], kind="mergesort")
        .reset_index(drop=True),
    )
    expected_pairs = pd.MultiIndex.from_product(
        [_expected_hour_boundaries(start, end), sorted(symbols)],
        names=["feature_available_time_utc", "symbol"],
    )
    observed_frame = cast(
        pd.DataFrame, panel.loc[:, ["feature_available_time_utc", "symbol"]]
    )
    observed_pairs = pd.MultiIndex.from_frame(observed_frame)
    if observed_pairs.has_duplicates or not observed_pairs.equals(expected_pairs):
        raise RuntimeError("combined six-alt panel does not match its exact hour-symbol grid")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        "six_alt_price_free_flow_1h_"
        f"{start:%Y-%m-%d}_{end:%Y-%m-%d}.csv.gz"
    )
    deterministic_gzip_csv(panel, output_path)
    config_record = asdict(cfg)
    config_record["symbols"] = list(symbols)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "builder": str(BUILDER_PATH),
        "builder_sha256": sha256_file(BUILDER_PATH),
        "config": config_record,
        "protocol": {
            "source": (
                "repository-local SHA-256-locked Binance USD-M five-minute pool"
            ),
            "source_end_is_exclusive": True,
            "hourly_bucket": (
                "twelve left-closed UTC five-minute bars; feature becomes available "
                "only at the completed hour right edge"
            ),
            "allowed_input_values": list(ALLOWED_READ_COLUMNS),
            "price_values_read": False,
            "base_volume_values_read": False,
            "btc_data_read": False,
            "return_or_label_computed": False,
            "post_entry_outcomes_opened": False,
            "fill_policy": "no nearest join, forward fill, backward fill, or interpolation",
            "invalid_activity_policy": (
                "retain aggregate diagnostics but blank derived flow fraction and ticket "
                "size unless all twelve source bars have positive volume and trades"
            ),
            "upstream_archive_checksum_provenance": (
                "not asserted by this transform; exact repository-local input bytes are frozen"
            ),
        },
        "combined_output": str(output_path),
        "combined_sha256": sha256_file(output_path),
        "rows": int(len(panel)),
        "expected_rows": int(len(expected_pairs)),
        "source_complete_rows": int(panel["source_complete"].sum()),
        "feature_valid_rows": int(panel["feature_valid"].sum()),
        "invalid_feature_rows": int((~panel["feature_valid"]).sum()),
        "first_source_hour_open_utc": cast(
            pd.Timestamp, panel["source_hour_open_utc"].min()
        ).isoformat(),
        "last_feature_available_time_utc": cast(
            pd.Timestamp, panel["feature_available_time_utc"].max()
        ).isoformat(),
        "symbols": list(symbols),
        "columns": list(panel.columns),
        "input_summary": (
            None if cfg.input_summary is None else Path(cfg.input_summary).name
        ),
        "input_summary_sha256": (
            None if cfg.input_summary is None else sha256_file(cfg.input_summary)
        ),
        "inputs": inputs,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=BuildConfig.input_dir)
    parser.add_argument("--input-summary", default=BuildConfig.input_summary)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--start", default=BuildConfig.start)
    parser.add_argument("--end", default=BuildConfig.end)
    parser.add_argument(
        "--no-enforce-frozen-inputs",
        action="store_true",
        help="intended only for unit fixtures",
    )
    args = parser.parse_args()
    manifest = build(
        BuildConfig(
            input_dir=args.input_dir,
            input_summary=args.input_summary,
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
            enforce_frozen_inputs=not args.no_enforce_frozen_inputs,
        )
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "combined_output",
                    "combined_sha256",
                    "rows",
                    "feature_valid_rows",
                    "invalid_feature_rows",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
