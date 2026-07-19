"""Validate and freeze post-preregistration OPDR-24 feature sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_options_perpetual_demand_relay as prereg  # noqa: E402


PREREG_COMMIT = "c4b9c4f22d24783f8176897ec4159a5ae1f6e68c"
PREREG_SHA256 = (
    "9673fe0fc0cc929514c730a56157f6ed409dd1063486c7df082c215e459ba696"
)
PREREG_MANIFEST_HASH = (
    "c5f61217324c51faeb46324ff31906205e2fd71b84fbb1c39b067b2e4ce4cf6c"
)
BVOL_SHA256 = (
    "40c0d1aecb15119e7fab31aae4108c632d25de136401a6896896852c7f4032b1"
)
BVOL_MANIFEST_SHA256 = (
    "6c62a389cbc8d6524444f5e5fe1d2945c20bafa9fa707b7f2a4801c74221a7e4"
)
DVOL_SHA256 = (
    "26b768f81c2fa49fd59d9f1a173a829329a7ed5bb94c2d71af7c33b46f4f02cf"
)
DVOL_SUMMARY_SHA256 = (
    "22e0a6e311fcad34a51f5b0844b7807e7c851eecc4a367f89b7a7d6ce438bf74"
)


@dataclass(frozen=True)
class Config:
    preregistration: str = prereg.DEFAULT_OUTPUT
    bvol: str = (
        "data/binance_btc_bvol_hourly_opdr_2023_2026/"
        "BTCBVOLUSDT_1h_2023-06-20_2026-06-30.csv.gz"
    )
    bvol_manifest: str = (
        "data/binance_btc_bvol_hourly_opdr_2023_2026/build_manifest.json"
    )
    dvol: str = "data/deribit_btc_dvol_1h_2023-06-20_2026-07-01.csv.gz"
    dvol_summary: str = (
        "data/deribit_btc_dvol_1h_2023-06-20_2026-07-01.csv.gz.summary.json"
    )
    output: str = (
        "results/options_perpetual_demand_relay_source_freeze_2026-07-19.json"
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_registration(cfg: Config) -> dict[str, Any]:
    if _sha256(cfg.preregistration) != PREREG_SHA256:
        raise ValueError("OPDR-24 preregistration bytes changed")
    report = json.loads(Path(cfg.preregistration).read_text(encoding="utf-8"))
    prereg.validate_manifest(report, verify_sources=False)
    if report["manifest_hash"] != PREREG_MANIFEST_HASH:
        raise ValueError("OPDR-24 preregistration manifest changed")
    if report["outcomes_opened"] is not False:
        raise ValueError("OPDR-24 preregistration opened outcomes")
    return report


def load_bvol(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    if _sha256(cfg.bvol) != BVOL_SHA256:
        raise ValueError("OPDR-24 BVOL bytes changed")
    if _sha256(cfg.bvol_manifest) != BVOL_MANIFEST_SHA256:
        raise ValueError("OPDR-24 BVOL manifest bytes changed")
    manifest = json.loads(Path(cfg.bvol_manifest).read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("OPDR-24 BVOL manifest opened outcomes")
    if protocol.get("valid_rows_checksum_verified") is not True:
        raise ValueError("OPDR-24 BVOL rows were not checksum verified")
    if protocol.get("missing_archives_are_invalid_not_imputed") is not True:
        raise ValueError("OPDR-24 BVOL missing archives were not fail-closed")
    if manifest.get("combined_sha256") != BVOL_SHA256:
        raise ValueError("OPDR-24 BVOL manifest data hash changed")
    frame = pd.read_csv(
        cfg.bvol,
        compression="gzip",
        parse_dates=["date", "feature_available_time_utc", "trade_earliest_time_utc"],
    )
    expected = pd.date_range("2023-06-20", "2026-07-01", freq="1h", inclusive="left")
    if not pd.DatetimeIndex(frame["date"]).equals(expected):
        raise ValueError("OPDR-24 BVOL hourly grid changed")
    expected_available = cast(pd.Series, frame["date"]) + pd.Timedelta(hours=1)
    if not cast(pd.Series, frame["feature_available_time_utc"]).equals(
        expected_available
    ):
        raise ValueError("OPDR-24 BVOL availability changed")
    if not cast(pd.Series, frame["trade_earliest_time_utc"]).equals(
        expected_available
    ):
        raise ValueError("OPDR-24 BVOL earliest time changed")
    valid = cast(pd.Series, frame["feature_valid"]).astype(bool)
    ohlc = ["open", "high", "low", "close"]
    if not np.isfinite(frame.loc[valid, ohlc].to_numpy(float)).all():
        raise ValueError("OPDR-24 valid BVOL row is nonfinite")
    if not bool(frame.loc[valid, "source_rows"].eq(3_600).all()):
        raise ValueError("OPDR-24 valid BVOL row is incomplete")
    if not bool(frame.loc[~valid, ohlc].isna().all().all()):
        raise ValueError("OPDR-24 invalid BVOL row retained values")
    if not bool(
        frame.loc[valid, "high"]
        .ge(frame.loc[valid, ["open", "close"]].max(axis=1))
        .all()
    ):
        raise ValueError("OPDR-24 BVOL high envelope changed")
    if not bool(
        frame.loc[valid, "low"]
        .le(frame.loc[valid, ["open", "close"]].min(axis=1))
        .all()
    ):
        raise ValueError("OPDR-24 BVOL low envelope changed")
    return frame, manifest


def load_dvol(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    if _sha256(cfg.dvol) != DVOL_SHA256:
        raise ValueError("OPDR-24 DVOL bytes changed")
    if _sha256(cfg.dvol_summary) != DVOL_SUMMARY_SHA256:
        raise ValueError("OPDR-24 DVOL summary bytes changed")
    summary = json.loads(Path(cfg.dvol_summary).read_text(encoding="utf-8"))
    expected_config = {
        "output_csv": cfg.dvol,
        "start": "2023-06-20",
        "end": "2026-07-01",
        "currency": "BTC",
        "resolution": 3600,
        "timeout_sec": 30.0,
    }
    if summary.get("config") != expected_config:
        raise ValueError("OPDR-24 DVOL acquisition config changed")
    if summary.get("availability") != (
        "candle values join on close_time, never date/open time"
    ):
        raise ValueError("OPDR-24 DVOL availability contract changed")
    frame = pd.read_csv(
        cfg.dvol,
        compression="gzip",
        parse_dates=["date", "close_time"],
    )
    expected = pd.date_range("2023-06-20", "2026-07-01", freq="1h")
    if not pd.DatetimeIndex(frame["date"]).equals(expected):
        raise ValueError("OPDR-24 DVOL hourly grid changed")
    if not cast(pd.Series, frame["close_time"]).equals(
        cast(pd.Series, frame["date"]) + pd.Timedelta(hours=1)
    ):
        raise ValueError("OPDR-24 DVOL close-time availability changed")
    ohlc = ["open", "high", "low", "close"]
    if not np.isfinite(frame[ohlc].to_numpy(float)).all():
        raise ValueError("OPDR-24 DVOL contains nonfinite values")
    if not bool(frame["high"].ge(frame[["open", "close"]].max(axis=1)).all()):
        raise ValueError("OPDR-24 DVOL high envelope changed")
    if not bool(frame["low"].le(frame[["open", "close"]].min(axis=1)).all()):
        raise ValueError("OPDR-24 DVOL low envelope changed")
    return frame, summary


def build_report(cfg: Config = Config()) -> dict[str, Any]:
    registration = _load_registration(cfg)
    bvol, bvol_manifest = load_bvol(cfg)
    dvol, dvol_summary = load_dvol(cfg)
    valid = cast(pd.Series, bvol["feature_valid"]).astype(bool)
    bvol_years: dict[str, Any] = {}
    for year, group in bvol.groupby(cast(pd.Series, bvol["date"]).dt.year):
        group_valid = cast(pd.Series, group["feature_valid"]).astype(bool)
        bvol_years[str(year)] = {
            "rows": int(len(group)),
            "valid_rows": int(group_valid.sum()),
            "valid_share": float(group_valid.mean()),
        }
    archive_status = Counter(
        archive["status"]
        for month in bvol_manifest["months"]
        for archive in month["archives"]
    )
    core: dict[str, Any] = {
        "protocol_version": "options_perpetual_demand_relay_source_freeze_v1",
        "as_of_date": "2026-07-19",
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "preregistration": {
            "path": cfg.preregistration,
            "commit": PREREG_COMMIT,
            "sha256": PREREG_SHA256,
            "manifest_hash": registration["manifest_hash"],
        },
        "bvol": {
            "path": cfg.bvol,
            "sha256": BVOL_SHA256,
            "manifest": cfg.bvol_manifest,
            "manifest_sha256": BVOL_MANIFEST_SHA256,
            "rows": int(len(bvol)),
            "valid_rows": int(valid.sum()),
            "invalid_rows": int((~valid).sum()),
            "archive_status": dict(sorted(archive_status.items())),
            "by_year": bvol_years,
        },
        "dvol": {
            "path": cfg.dvol,
            "sha256": DVOL_SHA256,
            "summary": cfg.dvol_summary,
            "summary_sha256": DVOL_SUMMARY_SHA256,
            "rows": int(len(dvol)),
            "first_date": str(dvol["date"].min()),
            "last_date": str(dvol["date"].max()),
            "last_close_time": str(dvol["close_time"].max()),
            "acquisition": dvol_summary["config"],
        },
        "premium": {
            "path": prereg.PREMIUM_PATH,
            "sha256": registration["source_contract"]["premium_sha256"],
            "manifest": prereg.PREMIUM_MANIFEST,
            "manifest_sha256": registration["source_contract"][
                "premium_manifest_sha256"
            ],
            "values_opened_in_this_stage": False,
        },
        "btc_execution_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "ready_for_outcome_blind_support": True,
        "implementation_sha256": _sha256(__file__),
    }
    return {**core, "manifest_hash": _canonical_hash(core)}


def run(cfg: Config = Config()) -> dict[str, Any]:
    report = build_report(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--bvol", default=Config.bvol)
    parser.add_argument("--bvol-manifest", default=Config.bvol_manifest)
    parser.add_argument("--dvol", default=Config.dvol)
    parser.add_argument("--dvol-summary", default=Config.dvol_summary)
    parser.add_argument("--output", default=Config.output)
    report = run(Config(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                "bvol": report["bvol"],
                "dvol": report["dvol"],
                "ready_for_outcome_blind_support": report[
                    "ready_for_outcome_blind_support"
                ],
                "manifest_hash": report["manifest_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
