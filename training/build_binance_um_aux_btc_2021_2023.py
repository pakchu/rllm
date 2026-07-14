"""Physically cut and verify Binance BTCUSDT funding/premium before 2024."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.build_bybit_linear_aux_btc_2021_2023 import (
    _write_deterministic_gzip,
)


@dataclass(frozen=True)
class Config:
    start: str = "2021-01-01"
    end: str = "2024-01-01"
    funding_source: str = (
        "data/binance_um_aux_btc_2020_2026/"
        "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
    )
    funding_source_sha256: str = (
        "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7"
    )
    premium_source: str = (
        "data/binance_um_aux_btc_2020_2026/"
        "BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
    )
    premium_source_sha256: str = (
        "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7"
    )
    output_dir: str = "data/binance_um_aux_btc_2021_2023"
    manifest: str = "results/binance_um_aux_btc_2021_2023_manifest.json"
    maximum_funding_timestamp_jitter_ms: int = 1_000


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_funding_time(
    funding_time: pd.Series,
    *,
    maximum_jitter_ms: int,
) -> tuple[pd.Series, pd.Series]:
    raw = pd.to_datetime(
        pd.to_numeric(funding_time, errors="raise"),
        unit="ms",
    )
    normalized = raw.dt.round("8h")
    jitter_ms = (raw - normalized).abs().dt.total_seconds() * 1_000.0
    if jitter_ms.max() > maximum_jitter_ms:
        raise ValueError("Binance funding timestamp jitter exceeds tolerance")
    return normalized, jitter_ms


def _validate_hourly(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    output = frame.sort_values("date").drop_duplicates("date", keep="last")
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    if not pd.DatetimeIndex(output["date"]).equals(expected):
        raise ValueError("Binance premium is not a complete hourly grid")
    numeric = output[["open", "high", "low", "close"]]
    if not numeric.notna().all().all():
        raise ValueError("Binance premium contains non-finite values")
    if not (
        output["high"].ge(numeric[["open", "close"]].max(axis=1)).all()
        and output["low"].le(numeric[["open", "close"]].min(axis=1)).all()
    ):
        raise ValueError("Binance premium OHLC invariants failed")
    return output.reset_index(drop=True)


def build(cfg: Config) -> dict[str, Any]:
    if pd.Timestamp(cfg.end) > pd.Timestamp("2024-01-01"):
        raise ValueError("preselection cut cannot include 2024+")
    for path, expected in (
        (cfg.funding_source, cfg.funding_source_sha256),
        (cfg.premium_source, cfg.premium_source_sha256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"Binance auxiliary source hash changed: {path}")

    funding = pd.read_csv(cfg.funding_source, compression="gzip")
    normalized, jitter_ms = normalize_funding_time(
        funding["funding_time"],
        maximum_jitter_ms=cfg.maximum_funding_timestamp_jitter_ms,
    )
    funding["date"] = normalized
    funding = funding.loc[
        funding["date"].ge(cfg.start) & funding["date"].lt(cfg.end)
    ].copy()
    funding = funding.sort_values("date").drop_duplicates("date", keep="last")
    expected_funding = pd.date_range(cfg.start, cfg.end, freq="8h", inclusive="left")
    if not pd.DatetimeIndex(funding["date"]).equals(expected_funding):
        raise ValueError("Binance funding is not a complete 8-hour grid")
    if not funding["funding_rate"].notna().all():
        raise ValueError("Binance funding contains missing rates")

    premium = pd.read_csv(
        cfg.premium_source,
        compression="gzip",
        parse_dates=["date"],
    )
    premium = premium.loc[
        premium["date"].ge(cfg.start) & premium["date"].lt(cfg.end)
    ].copy()
    premium = _validate_hourly(premium, cfg.start, cfg.end)

    output_dir = Path(cfg.output_dir)
    funding_path = output_dir / "BTCUSDT_funding_2021-01-01_2023-12-31.csv.gz"
    premium_path = output_dir / "BTCUSDT_premium_1h_2021-01-01_2023-12-31.csv.gz"
    funding_hash = _write_deterministic_gzip(funding, funding_path)
    premium_hash = _write_deterministic_gzip(premium, premium_path)

    manifest = {
        "protocol": {
            "name": "Binance USD-M BTCUSDT physical pre-2024 auxiliary cut",
            "outcomes_opened": False,
            "start_inclusive": str(pd.Timestamp(cfg.start)),
            "end_exclusive": str(pd.Timestamp(cfg.end)),
            "post_2023_rows_written": False,
        },
        "config": asdict(cfg),
        "source_hashes": {
            "funding": cfg.funding_source_sha256,
            "premium": cfg.premium_source_sha256,
        },
        "funding_timestamp_jitter_ms": {
            "maximum_full_source": float(jitter_ms.max()),
            "normalization": "nearest UTC 8-hour boundary",
        },
        "files": {
            "funding": {
                "path": str(funding_path),
                "sha256": funding_hash,
                "rows": int(len(funding)),
                "first_date": str(funding["date"].min()),
                "last_date": str(funding["date"].max()),
            },
            "premium": {
                "path": str(premium_path),
                "sha256": premium_hash,
                "rows": int(len(premium)),
                "first_date": str(premium["date"].min()),
                "last_date": str(premium["date"].max()),
            },
        },
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(cfg.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--manifest", default=Config.manifest)
    args = parser.parse_args()
    result = build(Config(output_dir=args.output_dir, manifest=args.manifest))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
