from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


MANIFEST = Path(
    "results/binance_coinm_liquidation_snapshot_eth_2023_2024_manifest.json"
)
EXPECTED_MANIFEST_SHA256 = (
    "c515731a9029d1786c8650f5106923d4cfbe8c35ed7a947f5420a16154601f5d"
)
EXPECTED_DATA_SHA256 = (
    "8d17ab3d5f9592f5254fef2e649065233be1777b8976983b4af38c77a8cc5bff"
)
MISSING_DATES = {
    "2023-08-05",
    "2023-09-09",
    "2023-09-23",
    "2023-09-25",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_eth_manifest_is_source_only_and_self_consistent() -> None:
    assert _sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256
    manifest = json.loads(MANIFEST.read_text())
    protocol = manifest["protocol"]
    assert protocol["outcomes_opened"] is False
    assert protocol["source_only"] is True
    assert protocol["prices_retained"] is False
    assert protocol["raw_archives_retained"] is False
    assert manifest["symbol"] == "ETHUSD_PERP"
    assert not any(
        forbidden in column
        for column in manifest["retained_columns"]
        for forbidden in ("price", "notional", "usd")
    )
    assert set(manifest["missing_archive_dates"]) == MISSING_DATES
    assert len(manifest["archives"]) == 478
    available = [item for item in manifest["archives"] if item["available"]]
    assert len(available) == 474
    assert sum(item["snapshot_rows"] for item in available) == 28_092
    assert sum(item["raw_rows"] for item in available) == 56_186
    assert sum(item["duplicate_rows_removed"] for item in available) == 28_094

    ordinary = [item for item in available if item["date"] != "2023-09-21"]
    assert all(item["raw_rows"] == 2 * item["snapshot_rows"] for item in ordinary)
    anomaly = next(item for item in available if item["date"] == "2023-09-21")
    assert (
        anomaly["raw_rows"],
        anomaly["snapshot_rows"],
        anomaly["duplicate_rows_removed"],
    ) == (76, 37, 39)


def test_frozen_eth_panel_is_causal_complete_and_price_free() -> None:
    manifest = json.loads(MANIFEST.read_text())
    data_path = Path(manifest["file"]["path"])
    assert _sha256(data_path) == EXPECTED_DATA_SHA256 == manifest["file"]["sha256"]
    frame = pd.read_csv(
        data_path,
        compression="gzip",
        parse_dates=["date", "feature_available_time"],
    )
    assert len(frame) == manifest["file"]["rows"] == 137_664
    expected = pd.Series(
        pd.date_range("2023-06-25", "2024-10-15", freq="5min", inclusive="left"),
        name="date",
    )
    assert frame["date"].equals(expected)
    assert not frame["date"].duplicated().any()
    delay = cast(pd.Series, frame["feature_available_time"].sub(frame["date"]))
    assert bool(delay.eq(pd.Timedelta(minutes=5, seconds=1)).all())
    assert int(frame["source_valid"].sum()) == 136_512
    assert int(frame["event_count"].fillna(0).sum()) == 28_092
    assert int(frame["event_count"].fillna(0).gt(0).sum()) == 11_637
    assert not any(
        forbidden in column
        for column in frame.columns
        for forbidden in ("price", "notional", "usd")
    )

    missing_mask = cast(
        pd.Series, frame["date"].dt.strftime("%Y-%m-%d").isin(MISSING_DATES)
    )
    missing = frame.loc[missing_mask]
    assert len(missing) == len(MISSING_DATES) * 288
    assert not bool(cast(pd.Series, missing["source_valid"]).any())
    features = frame.columns.difference(
        ["date", "feature_available_time", "source_valid"]
    )
    assert bool(missing.loc[:, features].isna().all().all())

    valid = frame.loc[cast(pd.Series, frame["source_valid"]).astype(bool)]
    assert bool(valid.loc[:, features].notna().all().all())
    assert np.allclose(
        valid["short_liquidation_contracts"]
        + valid["long_liquidation_contracts"],
        valid["total_liquidation_contracts"],
    )
    assert np.allclose(
        valid["short_liquidation_contracts"]
        - valid["long_liquidation_contracts"],
        valid["signed_liquidation_contracts"],
    )
    total = cast(pd.Series, valid["total_liquidation_contracts"])
    expected_imbalance = cast(
        pd.Series, valid["signed_liquidation_contracts"]
    ).div(total.where(total.gt(0.0))).fillna(0.0)
    assert np.allclose(valid["liquidation_imbalance"], expected_imbalance)
    assert np.allclose(np.mod(total.to_numpy(float), 1.0), 0.0)
