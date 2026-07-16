"""Create immutable 2020-2022 BTC market/funding prefixes for WAD selection."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.export_wikimedia_attention_source import deterministic_gzip_csv, sha256_file
from training.preregister_wikimedia_attention_divergence_alpha import (
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    SELECTION_END,
    canonical_hash,
    validate_manifest as validate_preregistration,
)


DEFAULT_INPUT = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
DEFAULT_FUNDING = (
    "data/binance_um_aux_btc_2020_2026/"
    "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
)
DEFAULT_MARKET_OUTPUT = "data/wikimedia_alpha_btcusdt_5m_2020_2022.csv.gz"
DEFAULT_FUNDING_OUTPUT = "data/wikimedia_alpha_funding_2020_2022.csv.gz"
DEFAULT_MANIFEST = (
    "results/wikimedia_attention_selection_market_prefix_manifest_2026-07-16.json"
)


@dataclass(frozen=True)
class Config:
    input_csv: str = DEFAULT_INPUT
    funding_csv: str = DEFAULT_FUNDING
    market_output: str = DEFAULT_MARKET_OUTPUT
    funding_output: str = DEFAULT_FUNDING_OUTPUT
    manifest_output: str = DEFAULT_MANIFEST
    preregistration: str = DEFAULT_PREREGISTRATION
    cutoff: str = SELECTION_END


def resolve_existing(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    fallback = Path("/home/pakchu/rllm") / candidate
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(path)


def prefix_frame(
    path: str | Path,
    *,
    date_column: str,
    cutoff: str,
    usecols: list[str],
) -> pd.DataFrame:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    cutoff_ts = pd.Timestamp(cutoff)
    opener = gzip.open if source.suffix == ".gz" else Path.open
    previous: pd.Timestamp | None = None
    sentinel: pd.Timestamp | None = None
    with opener(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(usecols).issubset(reader.fieldnames):
            raise ValueError(f"source columns missing: {sorted(set(usecols) - set(reader.fieldnames or []))}")
        for raw in reader:
            timestamp = pd.Timestamp(raw[date_column])
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert("UTC").tz_localize(None)
            if previous is not None and timestamp < previous:
                raise RuntimeError("prefix source is not chronological")
            previous = timestamp
            if timestamp >= cutoff_ts:
                sentinel = timestamp
                break
            row = {column: raw[column] for column in usecols}
            row[date_column] = timestamp
            rows.append(row)
    if not rows:
        raise ValueError(f"no source rows before cutoff: {path}")
    frame = pd.DataFrame(rows, columns=usecols)
    if frame[date_column].max() >= cutoff_ts:
        raise RuntimeError("prefix extraction crossed cutoff")
    frame.attrs["cutoff_sentinel_date"] = str(sentinel) if sentinel is not None else None
    frame.attrs["future_value_rows_parsed"] = 0
    return frame


def validate_market(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise RuntimeError("market prefix has duplicate dates")
    intervals = frame["date"].diff().dropna()
    if not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("market prefix is not a complete five-minute grid")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all() or (frame[column] <= 0.0).any():
            raise ValueError(f"invalid market prefix column: {column}")
    return frame


def validate_funding(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
    if not np.isfinite(frame["funding_rate"]).all():
        raise ValueError("invalid funding prefix")
    return frame


def run(cfg: Config) -> dict[str, Any]:
    if cfg.cutoff != SELECTION_END:
        raise RuntimeError("WAD market prefix cutoff is frozen at 2023-01-01")
    prereg_path = resolve_existing(cfg.preregistration)
    prereg = json.loads(prereg_path.read_text())
    validate_preregistration(prereg)
    market = validate_market(
        prefix_frame(
            resolve_existing(cfg.input_csv),
            date_column="date",
            cutoff=cfg.cutoff,
            usecols=["date", "open", "high", "low", "close"],
        )
    )
    funding = validate_funding(
        prefix_frame(
            resolve_existing(cfg.funding_csv),
            date_column="date",
            cutoff=cfg.cutoff,
            usecols=["date", "funding_rate"],
        )
    )
    deterministic_gzip_csv(market, cfg.market_output)
    deterministic_gzip_csv(funding, cfg.funding_output)
    core: dict[str, Any] = {
        "protocol_version": "wikimedia_selection_market_prefix_v1",
        "future_outcomes_opened": False,
        "prefix_materialization_contract": {
            "reader": "chronological CSV row stream",
            "stop": "before parsing non-date values of the first row at or after cutoff",
            "future_value_rows_parsed": 0,
            "market_cutoff_sentinel_date": market.attrs.get("cutoff_sentinel_date"),
            "funding_cutoff_sentinel_date": funding.attrs.get("cutoff_sentinel_date"),
        },
        "cutoff_exclusive": cfg.cutoff,
        "preregistration_path": str(prereg_path),
        "preregistration_file_sha256": sha256_file(prereg_path),
        "preregistration_manifest_hash": prereg["manifest_hash"],
        "market": {
            "path": cfg.market_output,
            "rows": int(len(market)),
            "first_date": str(market["date"].min()),
            "last_date": str(market["date"].max()),
            "bytes": Path(cfg.market_output).stat().st_size,
            "sha256": sha256_file(cfg.market_output),
        },
        "funding": {
            "path": cfg.funding_output,
            "rows": int(len(funding)),
            "first_date": str(funding["date"].min()),
            "last_date": str(funding["date"].max()),
            "bytes": Path(cfg.funding_output).stat().st_size,
            "sha256": sha256_file(cfg.funding_output),
        },
    }
    payload = {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(cfg.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for name in asdict(defaults):
        parser.add_argument(f"--{name.replace('_', '-')}", default=getattr(defaults, name))
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
