from __future__ import annotations

import gzip
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from training import build_deribit_expiry_wall_handoff_source as wall


def _ms(value: str) -> int:
    timestamp = cast(pd.Timestamp, pd.Timestamp(value, tz="UTC"))
    return int(timestamp.timestamp() * 1000)


def _row(
    instrument: str,
    timestamp: str,
    *,
    position: float,
    index_price: float = 105.0,
) -> dict[str, Any]:
    return {
        "position": position,
        "timestamp": _ms(timestamp),
        "type": "delivery",
        "instrument_name": instrument,
        "index_price": index_price,
        "mark_price": 0.0,
        "profit_loss": 999.0,
        "session_profit_loss": 888.0,
    }


def _expiry_rows(date_code: str, date: str) -> list[dict[str, Any]]:
    timestamp = f"{date} 08:00:00.100"
    return [
        _row(f"BTC-{date_code}", timestamp, position=1_000.0),
        _row(f"BTC-{date_code}-90-C", timestamp, position=2.0),
        _row(f"BTC-{date_code}-90-P", timestamp, position=3.0),
        _row(f"BTC-{date_code}-100-C", timestamp, position=20.0),
        _row(f"BTC-{date_code}-100-P", timestamp, position=10.0),
        _row(f"BTC-{date_code}-110-C", timestamp, position=4.0),
        _row(f"BTC-{date_code}-110-P", timestamp, position=6.0),
    ]


def _payload(
    rows: list[dict[str, Any]], continuation: str | None
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "result": {"settlements": rows, "continuation": continuation},
    }


def test_wall_aggregate_combines_call_and_put_at_strike() -> None:
    aggregate, audit = wall.aggregate_wall_deliveries(
        _expiry_rows("3JAN20", "2020-01-03"),
        wall._delivery_config(wall.Config()),
    )

    row = aggregate.iloc[0]
    expected_spacing = abs(math.log(110.0 / 100.0))
    assert row["distinct_strike_count"] == 3
    assert row["total_position"] == pytest.approx(45.0)
    assert row["dominant_strike"] == pytest.approx(100.0)
    assert row["dominant_strike_position"] == pytest.approx(30.0)
    assert row["wall_share"] == pytest.approx(30.0 / 45.0)
    assert row["strike_position_hhi"] == pytest.approx(
        (5.0 / 45.0) ** 2 + (30.0 / 45.0) ** 2 + (10.0 / 45.0) ** 2
    )
    assert row["largest_individual_instrument_share"] == pytest.approx(
        20.0 / 45.0
    )
    assert row["local_log_spacing"] == pytest.approx(expected_spacing)
    assert row["signed_normalized_wall_distance"] == pytest.approx(
        math.log(105.0 / 100.0) / expected_spacing
    )
    assert row["wall_tie_count"] == 1
    assert audit["option_rows_selected"] == 6
    assert audit["futures_rows_excluded"] == 1
    assert audit["source_expiry_events"] == 1
    assert audit["wall_valid_expiry_events"] == 1


def test_tied_wall_and_too_few_strikes_are_source_only_exclusions() -> None:
    tied = _expiry_rows("3JAN20", "2020-01-03")
    for row in tied:
        if row["instrument_name"] == "BTC-3JAN20-90-C":
            row["position"] = 27.0
    two_strikes = [
        row
        for row in _expiry_rows("10JAN20", "2020-01-10")
        if "-110-" not in str(row["instrument_name"])
    ]
    valid = _expiry_rows("17JAN20", "2020-01-17")

    aggregate, audit = wall.aggregate_wall_deliveries(
        tied + two_strikes + valid,
        wall._delivery_config(wall.Config()),
    )

    assert len(aggregate) == 1
    assert audit["source_expiry_events"] == 3
    assert audit["wall_valid_expiry_events"] == 1
    assert audit["invalid_tied_wall"] == 1
    assert audit["invalid_too_few_strikes"] == 1
    assert audit["invalid_spacing"] == 0


def test_source_run_is_deterministic_and_retains_no_instrument_rows(
    tmp_path: Path,
) -> None:
    recent = _expiry_rows("29DEC23", "2023-12-29")
    before = _expiry_rows("28DEC18", "2018-12-28")
    pages = {
        None: _payload(recent, "old"),
        "old": _payload(before, None),
    }

    def fetch(params: dict[str, Any]) -> dict[str, Any]:
        return pages[params.get("continuation")]

    cfg = replace(
        wall.Config(),
        output_csv=str(tmp_path / "wall.csv.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
        page_size=len(recent),
        request_pause_sec=0.0,
    )
    first = wall.run(cfg, fetch=fetch, sleep=lambda _: None)
    first_csv = Path(cfg.output_csv).read_bytes()
    first_manifest = Path(cfg.manifest_output).read_bytes()
    second = wall.run(cfg, fetch=fetch, sleep=lambda _: None)

    assert Path(cfg.output_csv).read_bytes() == first_csv
    assert Path(cfg.manifest_output).read_bytes() == first_manifest
    assert first["manifest_hash"] == second["manifest_hash"]
    assert first["aggregate"]["columns"] == list(wall.SOURCE_COLUMNS)
    assert first["candidate_incidence_computed"] is False
    assert first["parameter_search_performed"] is False
    assert first["outcome_boundary"] == {
        "binance_market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "future_return_rows_loaded": 0,
        "performance_artifacts_parsed": 0,
        "return_or_pnl_fields_retained": 0,
        "economic_outcomes_computed": False,
        "raw_deribit_rows_persisted": False,
    }
    with gzip.open(cfg.output_csv, "rt", encoding="utf-8") as handle:
        public_csv = handle.read()
    assert "BTC-29DEC23" not in public_csv
    assert "option_type" not in public_csv
    assert "profit_loss" not in public_csv


def test_source_run_binds_mechanism_and_shared_downloader(tmp_path: Path) -> None:
    recent = _expiry_rows("29DEC23", "2023-12-29")
    before = _expiry_rows("28DEC18", "2018-12-28")
    pages = iter([_payload(recent, "old"), _payload(before, None)])
    cfg = replace(
        wall.Config(),
        output_csv=str(tmp_path / "wall.csv.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
        page_size=len(recent),
        request_pause_sec=0.0,
    )

    manifest = wall.run(
        cfg,
        fetch=lambda _: next(pages),
        sleep=lambda _: None,
    )

    assert manifest["mechanism_binding"]["sha256"] == (
        wall.MECHANISM_DOCUMENT_SHA256
    )
    assert manifest["shared_downloader_binding"]["sha256"] == (
        wall.SHARED_DOWNLOADER_SHA256
    )
