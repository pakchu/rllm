from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PANEL = Path(
    "data/binance_stablecoin_denominator_btc_2023/"
    "BTC_stablecoin_denominator_1h_2023-08-04T08_2023-12-31T23.csv.gz"
)
MANIFEST = Path("data/binance_stablecoin_denominator_btc_2023/build_manifest.json")
PANEL_SHA256 = "aab063f0f9d898d5cdafffb57f552244083cd93fe69a3c6ebaf97faf6e27b642"
MANIFEST_SHA256 = "863e96b4325d051731c92852c6760986204a9df62f77ff0dd0e01ab08d8a15d3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_stablecoin_denominator_source_artifact() -> None:
    assert _sha256(PANEL) == PANEL_SHA256
    assert _sha256(MANIFEST) == MANIFEST_SHA256
    manifest = json.loads(MANIFEST.read_text())
    protocol = manifest["protocol"]
    assert protocol["outcomes_opened"] is False
    assert protocol["perpetual_ohlc_or_funding_opened"] is False
    assert protocol["future_returns_labels_or_pnl_opened"] is False
    assert protocol["post_2023_rows_requested"] is False
    assert protocol["raw_btc_prices_retained"] is False
    assert protocol["flow_or_volume_fields_retained"] is False
    assert manifest["combined_sha256"] == PANEL_SHA256
    assert manifest["rows"] == manifest["complete_rows"] == 3_592

    frame = pd.read_csv(PANEL, parse_dates=["date", "source_available_at"])
    expected = pd.date_range(
        "2023-08-04 08:00", "2024-01-01", freq="1h", inclusive="left"
    )
    assert frame["date"].equals(pd.Series(expected, name="date"))
    assert frame["source_available_at"].equals(
        pd.Series(expected + pd.Timedelta(hours=1), name="source_available_at")
    )
    assert bool(frame["source_complete"].all())
    values = frame[
        [
            "usdc_vs_usdt",
            "fdusd_vs_usdt",
            "alt_consensus",
            "alt_disagreement",
        ]
    ].to_numpy(float)
    assert np.isfinite(values).all()
    assert np.allclose(
        frame["alt_consensus"],
        (frame["usdc_vs_usdt"] + frame["fdusd_vs_usdt"]) / 2.0,
    )
    assert np.allclose(
        frame["alt_disagreement"],
        (frame["usdc_vs_usdt"] - frame["fdusd_vs_usdt"]).abs(),
    )
