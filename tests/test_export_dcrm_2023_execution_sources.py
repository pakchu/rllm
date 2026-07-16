from __future__ import annotations

import pandas as pd
import pytest

from training import export_dcrm_2023_execution_sources as export


def test_market_prefix_rejects_any_timestamp_at_2024(monkeypatch) -> None:
    index = pd.date_range(export.START, export.END - pd.Timedelta(minutes=5), freq="5min")
    frame = pd.DataFrame(
        {
            "date": index,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "tic": "ETHUSDT",
        }
    )
    validated = export.validate_market_prefix(frame, "ETHUSDT")
    assert len(validated) == export.MARKET_ROWS
    frame.loc[len(frame) - 1, "date"] = export.END
    with pytest.raises(ValueError, match="prefix/grid"):
        export.validate_market_prefix(frame, "ETHUSDT")


def test_funding_prefix_requires_exact_8h_grid() -> None:
    event_time = pd.date_range(export.START, export.END - pd.Timedelta(hours=8), freq="8h")
    event_time = event_time + pd.to_timedelta(
        [index % 19 for index in range(len(event_time))], unit="ms"
    )
    frame = pd.DataFrame({"event_time": event_time, "funding_rate": 0.0001})
    validated = export.validate_funding_prefix(frame, "ETHUSDT")
    assert len(validated) == export.FUNDING_ROWS
    frame = frame.drop(index=10).reset_index(drop=True)
    with pytest.raises(ValueError, match="prefix/grid"):
        export.validate_funding_prefix(frame, "ETHUSDT")


def test_source_contract_does_not_hash_combined_inputs() -> None:
    source = open("training/export_dcrm_2023_execution_sources.py").read()
    run_body = source.split("def run(", 1)[1]
    assert "sha256_file(input_market)" not in run_body
    assert "sha256_file(input_funding)" not in run_body
    assert "nrows=MARKET_ROWS" in run_body
    assert "nrows=FUNDING_ROWS" in run_body
