from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from training import build_binance_btcdom_premium_decomposition_source as builder


PANEL = Path(
    "data/binance_btcdom_premium_decomposition_2021_2023/"
    "BTCUSDT_BTCDOMUSDT_premium_close_1h_2021-07-02_2023-12-31.csv.gz"
)
MANIFEST = Path(
    "data/binance_btcdom_premium_decomposition_2021_2023/build_manifest.json"
)
PANEL_SHA256 = "75fb36b33810134746515e3ad99234e2a52f6f721551792788f6d3950ff5b1d9"
MANIFEST_SHA256 = "885014743c299250c85cec42561db0dc99b09a60ecb1adfe893d8cac95651c05"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    return cast(
        pd.Series,
        values.astype("string").str.lower().map({"true": True, "false": False}),
    ).astype(bool)


def test_frozen_artifact_hashes_and_identity() -> None:
    assert _sha(PANEL) == PANEL_SHA256
    assert _sha(MANIFEST) == MANIFEST_SHA256
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["combined_sha256"] == PANEL_SHA256
    assert manifest["builder_sha256"] == _sha(builder.BUILDER_PATH)
    assert manifest["checksum_inventory_sha256"] == builder.INVENTORY_SHA256
    assert manifest["source_decision_sha256"] == builder.SOURCE_DECISION_SHA256


def test_frozen_panel_has_exact_grid_and_disclosed_missingness() -> None:
    frame = pd.read_csv(PANEL, parse_dates=["date", "source_close_time", "feature_available_time"])
    assert frame.columns.tolist() == list(builder.OUTPUT_COLUMNS)
    expected = pd.date_range(builder.START, builder.END, freq="1h", inclusive="left")
    assert frame["date"].equals(pd.Series(expected, name="date"))
    assert frame["source_close_time"].equals(
        pd.Series(
            expected + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            name="source_close_time",
        )
    )
    assert frame["feature_available_time"].equals(
        pd.Series(expected + pd.Timedelta(hours=1, seconds=1), name="feature_available_time")
    )

    btc = _bool(cast(pd.Series, frame["btcusdt_valid"]))
    dom = _bool(cast(pd.Series, frame["btcdomusdt_valid"]))
    valid = _bool(cast(pd.Series, frame["source_valid"]))
    assert int(btc.sum()) == 21_768
    assert int(dom.sum()) == 21_768
    assert int(valid.sum()) == 21_744
    assert int((~btc & ~dom).sum()) == 120
    assert int((btc & ~dom).sum()) == 24
    assert int((~btc & dom).sum()) == 24
    assert valid.equals(btc & dom)

    assert frame.loc[~btc, "btcusdt_premium_close"].isna().all()
    assert frame.loc[~dom, "btcdomusdt_premium_close"].isna().all()
    assert np.isfinite(frame.loc[btc, "btcusdt_premium_close"].to_numpy(float)).all()
    assert np.isfinite(frame.loc[dom, "btcdomusdt_premium_close"].to_numpy(float)).all()


def test_manifest_preserves_unopened_boundary() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    assert protocol["source_only"] is True
    assert protocol["outcomes_opened"] is False
    assert protocol["post_2023_rows_requested"] is False
    assert protocol["btc_or_btcdom_contract_ohlc_retained"] is False
    assert protocol["btc_or_btcdom_index_prices_retained"] is False
    assert protocol["funding_returns_labels_or_pnl_retained"] is False
    assert protocol["premium_ohlc_paths_retained"] is False
    assert protocol["premium_closes_retained"] is True
    assert manifest["rows"] == 21_912
    assert manifest["valid_rows"] == 21_744
    assert len(manifest["archives"]) == 60
