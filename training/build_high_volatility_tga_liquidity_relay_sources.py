"""Freeze official FiscalData TGA closing balances for HVTGAL-24."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_tga_liquidity_relay as prereg


SOURCE_DIR = prereg.SOURCE.parent
RAW = SOURCE_DIR / "fiscaldata_operating_cash_balance.json"
MANIFEST = SOURCE_DIR / "build_manifest.json"
BUILDER = Path("training/build_high_volatility_tga_liquidity_relay_sources.py")
PREREG_SHA = "a076cda574a55ce909f28a26ca0afdad687a281d4657cea16a69c07a3e0705e8"
ACCOUNT_TYPE = "Treasury General Account (TGA) Closing Balance"
PARAMS = {
    "filter": "record_date:gte:2020-01-01,record_date:lte:2026-07-27",
    "sort": "record_date",
    "page[size]": "10000",
    "format": "json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_url() -> str:
    return prereg.API + "?" + urllib.parse.urlencode(PARAMS)


def fetch() -> bytes:
    request = urllib.request.Request(source_url(), headers={"User-Agent": "rllm-causal-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def parse(raw: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(raw)
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("HVTGAL FiscalData response missing data")
    rows = [item for item in data if item.get("account_type") == ACCOUNT_TYPE]
    frame = pd.DataFrame({"record_date": [item.get("record_date") for item in rows], "tga_close_millions": [item.get("open_today_bal") for item in rows]})
    frame["record_date"] = pd.to_datetime(frame.record_date, errors="coerce")
    frame["tga_close_millions"] = pd.to_numeric(frame.tga_close_millions, errors="coerce")
    frame = frame.sort_values("record_date").reset_index(drop=True)
    valid = frame.record_date.notna() & np.isfinite(frame.tga_close_millions) & frame.tga_close_millions.gt(0)
    if frame.empty or not valid.all() or frame.record_date.duplicated().any():
        raise RuntimeError("HVTGAL invalid or duplicate official rows")
    # Treasury introduced the explicit TGA Closing Balance account label on
    # 2022-04-18; older rows use a different aggregate account taxonomy.
    if frame.record_date.iloc[0] != pd.Timestamp("2022-04-18") or frame.record_date.iloc[-1] < pd.Timestamp("2026-07-24"):
        raise RuntimeError("HVTGAL official coverage drift")
    meta = payload.get("meta") or {}
    return frame, {"response_rows": len(data), "selected_rows": len(frame), "meta_total_count": meta.get("total-count"), "first_record_date": str(frame.record_date.iloc[0].date()), "last_record_date": str(frame.record_date.iloc[-1].date())}


def write_gzip(frame: pd.DataFrame, path: Path) -> None:
    payload = frame.to_csv(index=False, lineterminator="\n").encode()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(payload)


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVTGAL preregistration drift")
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(fetch())
    frame, metadata = parse(RAW.read_bytes())
    write_gzip(frame, prereg.SOURCE)
    core = {
        "protocol_version": "hvtgal_24_sources_v1",
        "preregistration_sha256": PREREG_SHA,
        "completed_preentry_sources_opened": True,
        "candidate_incidence_opened": False,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "official_source": {"endpoint": prereg.API, "url": source_url(), "raw_path": str(RAW), "raw_sha256": sha(RAW), "account_type": ACCOUNT_TYPE, **metadata},
        "output": {"path": str(prereg.SOURCE), "sha256": sha(prereg.SOURCE), "rows": len(frame)},
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps(report["output"], sort_keys=True))
