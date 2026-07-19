from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd

from training import build_six_alt_price_free_flow_panel as source


PANEL = Path(
    "data/binance_six_alt_price_free_flow_2023_2026/"
    "six_alt_price_free_flow_1h_2023-01-01_2026-06-01.csv.gz"
)
MANIFEST = Path("data/binance_six_alt_price_free_flow_2023_2026/build_manifest.json")
PANEL_SHA256 = "bf4d67ee02948444712a6ff7862a0d4f4ae4ae2a704c9d0586538043c169f6b9"
MANIFEST_SHA256 = "eab61cbc7f5fc51e78f574e8bef163b3a3b91bd027136cae8efd7aaf26edc0f1"
BUILDER_SHA256 = "7e6a212ab2eeb30ef69e4f9ea5772b757c46fa46dae5f1ea52c0732289b28506"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_price_free_flow_source_is_hash_locked_and_outcome_blind() -> None:
    assert _sha256(PANEL) == PANEL_SHA256
    assert _sha256(MANIFEST) == MANIFEST_SHA256
    assert _sha256(source.BUILDER_PATH) == BUILDER_SHA256
    manifest = json.loads(MANIFEST.read_text())
    protocol = manifest["protocol"]
    assert manifest["builder_sha256"] == BUILDER_SHA256
    assert manifest["combined_sha256"] == PANEL_SHA256
    assert protocol["allowed_input_values"] == list(source.ALLOWED_READ_COLUMNS)
    assert protocol["price_values_read"] is False
    assert protocol["base_volume_values_read"] is False
    assert protocol["btc_data_read"] is False
    assert protocol["return_or_label_computed"] is False
    assert protocol["post_entry_outcomes_opened"] is False
    assert protocol["fill_policy"].startswith("no nearest join")
    assert {row["symbol"]: row["source_sha256"] for row in manifest["inputs"]} == (
        source.FROZEN_INPUT_SHA256
    )


def test_price_free_flow_panel_has_exact_causal_hour_symbol_grid() -> None:
    frame = pd.read_csv(
        PANEL,
        parse_dates=["source_hour_open_utc", "feature_available_time_utc"],
    )
    assert tuple(frame.columns) == source.OUTPUT_COLUMNS
    assert len(frame) == 179_568
    assert set(frame["symbol"]) == set(source.SYMBOLS)
    assert not frame[["feature_available_time_utc", "symbol"]].duplicated().any()
    assert bool(
        cast(pd.Series, frame["feature_available_time_utc"])
        .eq(frame["source_hour_open_utc"] + pd.Timedelta(hours=1))
        .all()
    )
    assert frame["source_hour_open_utc"].min() == pd.Timestamp("2023-01-01")
    assert frame["feature_available_time_utc"].max() == pd.Timestamp("2026-06-01")
    assert bool(cast(pd.Series, frame["source_bar_count"]).eq(12).all())
    assert bool(cast(pd.Series, frame["source_complete"]).all())
    assert not {"open", "high", "low", "close", "return", "pnl"}.intersection(
        frame.columns
    )


def test_invalid_activity_hours_fail_closed_for_every_symbol() -> None:
    frame = pd.read_csv(PANEL, parse_dates=["feature_available_time_utc"])
    invalid = cast(pd.DataFrame, frame.loc[~frame["feature_valid"]])
    assert len(invalid) == 36
    counts = invalid.groupby("feature_available_time_utc")["symbol"].nunique()
    assert len(counts) == 6
    assert bool(counts.eq(6).all())
    assert bool(invalid["taker_flow_fraction"].isna().all())
    assert bool(invalid["mean_ticket_usdt"].isna().all())
    valid = cast(pd.DataFrame, frame.loc[frame["feature_valid"]])
    assert bool(valid["taker_flow_fraction"].between(-1.0, 1.0).all())
    assert bool(valid["mean_ticket_usdt"].gt(0.0).all())
