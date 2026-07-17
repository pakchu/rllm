from __future__ import annotations

import hashlib
import json

import pandas as pd


DATA = "data/coinmetrics_btc_miner_security_daily_2019_2023.csv.gz"
MANIFEST = (
    "results/coinmetrics_btc_miner_security_daily_2019_2023_manifest_2026-07-17.json"
)
EXPECTED_SHA256 = "448a101834df33f69abaeafe9aadfccd8ce9c3d6ad7816c1c2448189a12b8379"


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_frozen_miner_source_identity_and_boundary() -> None:
    manifest = json.load(open(MANIFEST))
    frame = pd.read_csv(DATA)

    assert sha256(DATA) == EXPECTED_SHA256 == manifest["sha256"]
    assert len(frame) == manifest["rows"] == 1_826
    assert frame.columns.tolist() == manifest["columns"]
    assert frame["observation_date"].is_unique
    assert not frame.isna().any().any()
    assert pd.to_datetime(frame["observation_date"]).max() == pd.Timestamp(
        "2023-12-31"
    )
    assert (frame[["HashRate", "IssTotNtv", "BlkCnt"]] > 0.0).all().all()
    assert (frame["FeeTotNtv"] >= 0.0).all()
    assert manifest["excluded_on_purpose"]["post_2023_rows"]


def test_recorded_availability_never_precedes_required_daily_lag() -> None:
    frame = pd.read_csv(DATA, parse_dates=["observation_date", "available_at"])
    lag = frame["available_at"] - frame["observation_date"]
    assert (lag >= pd.Timedelta(days=1)).all()
    assert frame.loc[frame["observation_date"] == "2023-12-31", "available_at"].iat[
        0
    ] == pd.Timestamp("2024-01-01 04:50:54")
