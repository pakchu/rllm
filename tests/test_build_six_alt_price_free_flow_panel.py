from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from training import build_six_alt_price_free_flow_panel as source


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def _write_source(
    path: Path,
    *,
    start: str = "2023-01-01",
    periods: int = 24,
    zero_at: int | None = None,
) -> None:
    dates = pd.date_range(start, periods=periods, freq="5min")
    total = np.linspace(100.0, 200.0, periods)
    trades = np.arange(periods, dtype=np.int64) + 10
    taker_buy = total * 0.6
    if zero_at is not None:
        total[zero_at] = 0.0
        trades[zero_at] = 0
        taker_buy[zero_at] = 0.0
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": ["not-read"] * periods,
            "high": ["not-read"] * periods,
            "low": ["not-read"] * periods,
            "close": ["not-read"] * periods,
            "volume": ["not-read"] * periods,
            "quote_asset_volume": total,
            "number_of_trades": trades,
            "taker_buy_base": ["not-read"] * periods,
            "taker_buy_quote": taker_buy,
            "tic": ["not-read"] * periods,
            "day": ["not-read"] * periods,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def test_load_symbol_reads_only_price_free_columns_and_delays_availability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ETHUSDT_5m_fixture.csv.gz"
    _write_source(path, zero_at=13)
    original = pd.read_csv
    observed_usecols: list[str] = []

    def guarded_read_csv(*args: Any, **kwargs: Any) -> Any:
        observed_usecols.extend(cast(list[str], kwargs["usecols"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(source.pd, "read_csv", guarded_read_csv)
    hourly, metadata = source.load_symbol_hourly(
        path,
        symbol="ETHUSDT",
        start=_ts("2023-01-01"),
        end=_ts("2023-01-01 02:00"),
    )
    assert tuple(observed_usecols) == source.ALLOWED_READ_COLUMNS
    assert tuple(hourly.columns) == source.OUTPUT_COLUMNS
    assert hourly["feature_available_time_utc"].tolist() == [
        pd.Timestamp("2023-01-01 01:00"),
        pd.Timestamp("2023-01-01 02:00"),
    ]
    assert bool(hourly["source_bar_count"].eq(12).all())
    assert hourly["feature_valid"].tolist() == [True, False]
    assert hourly.loc[0, "taker_flow_fraction"] == pytest.approx(0.2)
    assert bool(pd.isna(hourly.loc[1, "taker_flow_fraction"]))
    assert bool(pd.isna(hourly.loc[1, "mean_ticket_usdt"]))
    assert metadata["invalid_feature_rows"] == 1


def test_load_symbol_rejects_gap_duplicate_and_taker_bound(tmp_path: Path) -> None:
    path = tmp_path / "ETHUSDT_5m_fixture.csv.gz"
    _write_source(path, periods=12)
    frame = pd.read_csv(path)
    frame = frame.drop(index=3)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    with pytest.raises(RuntimeError, match="exact frozen five-minute grid"):
        source.load_symbol_hourly(
            path,
            symbol="ETHUSDT",
            start=_ts("2023-01-01"),
            end=_ts("2023-01-01 01:00"),
        )

    _write_source(path, periods=12)
    frame = pd.read_csv(path)
    frame.loc[0, "taker_buy_quote"] = frame.loc[0, "quote_asset_volume"] + 1.0
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    with pytest.raises(ValueError, match="exceeds total"):
        source.load_symbol_hourly(
            path,
            symbol="ETHUSDT",
            start=_ts("2023-01-01"),
            end=_ts("2023-01-01 01:00"),
        )


def test_header_contract_rejects_missing_or_reordered_columns(tmp_path: Path) -> None:
    path = tmp_path / "ETHUSDT_5m_fixture.csv.gz"
    _write_source(path, periods=12)
    frame = pd.read_csv(path)
    frame = frame.loc[:, list(reversed(frame.columns))]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    with pytest.raises(ValueError, match="unexpected six-alt source columns"):
        source.load_symbol_hourly(
            path,
            symbol="ETHUSDT",
            start=_ts("2023-01-01"),
            end=_ts("2023-01-01 01:00"),
        )


def test_build_is_byte_deterministic_and_outcome_blind(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "output"
    for symbol in source.SYMBOLS:
        _write_source(input_dir / f"{symbol}_5m_fixture.csv.gz")
    cfg = source.BuildConfig(
        input_dir=str(input_dir),
        input_summary=None,
        output_dir=str(output_dir),
        start="2023-01-01",
        end="2023-01-01 02:00",
        enforce_frozen_inputs=False,
    )
    first = source.build(cfg)
    panel_path = Path(first["combined_output"])
    first_panel = panel_path.read_bytes()
    first_manifest = (output_dir / "build_manifest.json").read_bytes()
    second = source.build(cfg)
    assert panel_path.read_bytes() == first_panel
    assert (output_dir / "build_manifest.json").read_bytes() == first_manifest
    assert first == second
    assert first["rows"] == first["expected_rows"] == 12
    assert first["protocol"]["price_values_read"] is False
    assert first["protocol"]["btc_data_read"] is False
    assert first["protocol"]["post_entry_outcomes_opened"] is False
    assert not {"open", "high", "low", "close"}.intersection(first["columns"])


def test_build_rejects_universe_mutation_and_default_hashes_are_complete() -> None:
    with pytest.raises(ValueError, match="universe is frozen"):
        source._validate_config(source.BuildConfig(symbols=("ETHUSDT",)))
    assert tuple(source.FROZEN_INPUT_SHA256) == source.SYMBOLS
    assert all(len(value) == 64 for value in source.FROZEN_INPUT_SHA256.values())
