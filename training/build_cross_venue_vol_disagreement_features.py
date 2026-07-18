"""Build a physically pre-2024 BTC BVOL/DVOL disagreement feature frame."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.build_binance_aggtrade_microstructure import _write_gzip_csv


SEALED_END_EXCLUSIVE = pd.Timestamp("2024-01-01")
OUTPUT_COLUMNS = (
    "signal_time_utc",
    "feature_available_time_utc",
    "trade_earliest_time_utc",
    "bvol_close",
    "dvol_close",
    "log_bvol_dvol_ratio",
    "btc_close",
    "btc_return_4h",
    "feature_valid",
    "feature_invalid_reason",
)


@dataclass(frozen=True)
class Config:
    bvol_csv: str = (
        "/home/pakchu/rllm/data/binance_btc_bvol_hourly/"
        "BTCBVOLUSDT_1h_2023-06-20_2023-12-31.csv.gz"
    )
    dvol_csv: str = (
        "/home/pakchu/rllm/data/deribit_btc_dvol_1h_2020-09-01_2026-06-02.csv.gz"
    )
    market_csv: str = (
        "/home/pakchu/rllm/data/binance_um_kline_reference_btc_2020_2023/"
        "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
    )
    output_dir: str = "/home/pakchu/rllm/data/cross_venue_vol_disagreement_btc"
    cutoff: str = "2024-01-01"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bvol(path: str | Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        compression="infer",
        parse_dates=["date", "feature_available_time_utc", "trade_earliest_time_utc"],
    ).sort_values("date").reset_index(drop=True)
    if frame.empty or frame["date"].duplicated().any():
        raise ValueError("BVOL source is empty or has duplicate timestamps")
    if not frame["date"].equals(
        pd.Series(pd.date_range(frame["date"].iloc[0], frame["date"].iloc[-1], freq="1h"), name="date")
    ):
        raise ValueError("BVOL source is not an hourly grid")
    expected_available = frame["date"] + pd.Timedelta("1h")
    if not frame["feature_available_time_utc"].equals(expected_available):
        raise ValueError("BVOL availability is not the completed-hour boundary")
    if not frame["trade_earliest_time_utc"].equals(expected_available):
        raise ValueError("BVOL earliest trade time violates source contract")
    frame = frame.loc[frame["feature_available_time_utc"] < cutoff].copy()
    return frame.rename(
        columns={
            "feature_available_time_utc": "signal_time_utc",
            "close": "bvol_close",
            "feature_valid": "bvol_valid",
            "feature_invalid_reason": "bvol_invalid_reason",
        }
    )[["signal_time_utc", "bvol_close", "bvol_valid", "bvol_invalid_reason"]]


def _load_dvol(path: str | Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="infer", parse_dates=["close_time"])
    frame = frame.rename(columns={"close_time": "signal_time_utc", "close": "dvol_close"})
    frame = frame.loc[frame["signal_time_utc"] < cutoff, ["signal_time_utc", "dvol_close"]]
    frame = frame.sort_values("signal_time_utc").reset_index(drop=True)
    if frame.empty or frame["signal_time_utc"].duplicated().any():
        raise ValueError("DVOL source is empty or has duplicate close times")
    frame["dvol_close"] = pd.to_numeric(frame["dvol_close"], errors="raise")
    return frame


def _load_hourly_btc_close(path: str | Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="infer", parse_dates=["date"], usecols=["date", "close"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty or frame["date"].duplicated().any():
        raise ValueError("market source is empty or has duplicate timestamps")
    expected = pd.date_range(frame["date"].iloc[0], frame["date"].iloc[-1], freq="5min")
    if not frame["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("market source is not a gapless five-minute grid")
    frame["btc_close"] = pd.to_numeric(frame["close"], errors="raise")
    frame["signal_time_utc"] = frame["date"] + pd.Timedelta("5min")
    hourly = frame.loc[
        frame["signal_time_utc"].dt.minute.eq(0)
        & frame["signal_time_utc"].dt.second.eq(0)
        & frame["signal_time_utc"].lt(cutoff),
        ["signal_time_utc", "btc_close"],
    ].reset_index(drop=True)
    hourly["btc_return_4h"] = hourly["btc_close"].div(hourly["btc_close"].shift(4)).sub(1.0)
    return hourly


def build_frame(cfg: Config) -> pd.DataFrame:
    cutoff = pd.Timestamp(cfg.cutoff)
    if cutoff > SEALED_END_EXCLUSIVE:
        raise ValueError("feature builder is physically sealed before 2024")
    bvol = _load_bvol(cfg.bvol_csv, cutoff)
    dvol = _load_dvol(cfg.dvol_csv, cutoff)
    market = _load_hourly_btc_close(cfg.market_csv, cutoff)
    output = bvol.merge(dvol, on="signal_time_utc", how="left", validate="one_to_one")
    output = output.merge(market, on="signal_time_utc", how="left", validate="one_to_one")
    output["log_bvol_dvol_ratio"] = np.log(output["bvol_close"] / output["dvol_close"])
    finite_positive = (
        output[["bvol_close", "dvol_close", "btc_close"]].gt(0.0).all(axis=1)
        & np.isfinite(output[["bvol_close", "dvol_close", "btc_close"]].to_numpy(float)).all(axis=1)
    )
    derived_finite = np.isfinite(
        output[["log_bvol_dvol_ratio", "btc_return_4h"]].to_numpy(float)
    ).all(axis=1)
    output["feature_valid"] = output["bvol_valid"].fillna(False).astype(bool) & finite_positive & derived_finite
    output["feature_invalid_reason"] = np.select(
        [
            ~output["bvol_valid"].fillna(False).astype(bool),
            output["dvol_close"].isna(),
            output["btc_close"].isna(),
            output["btc_return_4h"].isna(),
            ~(finite_positive & derived_finite),
        ],
        [
            "bvol_" + output["bvol_invalid_reason"].fillna("invalid").astype(str),
            "dvol_missing",
            "btc_close_missing",
            "btc_4h_history_missing",
            "nonfinite_or_nonpositive",
        ],
        default="ok",
    )
    quarantine = ["bvol_close", "dvol_close", "log_bvol_dvol_ratio", "btc_close", "btc_return_4h"]
    output.loc[~output["feature_valid"], quarantine] = np.nan
    output["feature_available_time_utc"] = output["signal_time_utc"]
    output["trade_earliest_time_utc"] = output["signal_time_utc"] + pd.Timedelta("5min")
    return output.loc[:, OUTPUT_COLUMNS]


def build(cfg: Config) -> dict[str, Any]:
    output = build_frame(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "BTC_cross_venue_vol_disagreement_1h_pre2024.csv.gz"
    _write_gzip_csv(output, path)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "outcomes_opened": False,
            "physically_truncated_before": cfg.cutoff,
            "joins": "exact UTC completed-hour close times",
            "btc_return": "backward-looking completed 4h close-to-close return",
            "entry_delay": "five minutes after all source candles close",
            "invalid_rows_imputed": False,
        },
        "sources": {
            "bvol": {"path": cfg.bvol_csv, "sha256": sha256_file(cfg.bvol_csv)},
            "dvol": {"path": cfg.dvol_csv, "sha256": sha256_file(cfg.dvol_csv)},
            "market": {"path": cfg.market_csv, "sha256": sha256_file(cfg.market_csv)},
        },
        "output": str(path),
        "output_sha256": sha256_file(path),
        "rows": int(len(output)),
        "feature_valid_rows": int(output["feature_valid"].sum()),
        "first_signal_time": str(output["signal_time_utc"].min()),
        "last_signal_time": str(output["signal_time_utc"].max()),
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvol-csv", default=Config.bvol_csv)
    parser.add_argument("--dvol-csv", default=Config.dvol_csv)
    parser.add_argument("--market-csv", default=Config.market_csv)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--cutoff", default=Config.cutoff)
    manifest = build(Config(**vars(parser.parse_args())))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
