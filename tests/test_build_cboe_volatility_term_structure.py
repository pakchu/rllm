from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import pytest

from training import build_cboe_volatility_term_structure as builder


def _response(rows: list[tuple[str, str, str, str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(builder.SOURCE_COLUMNS)
    writer.writerows(rows)
    return output.getvalue().encode()


def test_source_url_is_official_daily_history() -> None:
    assert builder.source_url("VIX9D") == (
        "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv"
    )
    with pytest.raises(ValueError, match="unsupported"):
        builder.source_url("BTC")


def test_normalize_response_filters_horizon_and_normalizes_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        builder.FROZEN_SOURCE_COVERAGE,
        "VIX",
        (2, "2018-01-02", "2023-12-29"),
    )
    payload = _response(
        [
            ("12/29/2017", "10", "11", "9", "10"),
            ("01/02/2018", "10", "12", "9", "11"),
            ("12/29/2023", "15", "16", "14", "15.5"),
            ("01/02/2024", "20", "21", "19", "20"),
        ]
    )
    normalized = builder.normalize_response(payload, symbol="VIX").decode()
    assert "2017" not in normalized and "2024" not in normalized
    assert "2018-01-02,10.000000,12.000000,9.000000,11.000000" in normalized
    assert normalized.endswith("2023-12-29,15.000000,16.000000,14.000000,15.500000\n")


def test_normalize_rejects_duplicate_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        builder.FROZEN_SOURCE_COVERAGE,
        "VIX3M",
        (2, "2018-01-02", "2018-01-02"),
    )
    payload = _response(
        [
            ("01/02/2018", "10", "11", "9", "10"),
            ("01/02/2018", "10", "11", "9", "10"),
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        builder.normalize_response(payload, symbol="VIX3M")


def test_normalize_rejects_bad_ohlc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        builder.FROZEN_SOURCE_COVERAGE,
        "VIX9D",
        (1, "2018-01-02", "2018-01-02"),
    )
    payload = _response([("01/02/2018", "10", "9", "8", "11")])
    with pytest.raises(ValueError, match="OHLC invariant"):
        builder.normalize_response(payload, symbol="VIX9D")


def test_panel_uses_exact_three_way_intersection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builder,
        "FROZEN_SOURCE_COVERAGE",
        {symbol: (2, "2018-01-02", "2018-01-03") for symbol in builder.SYMBOLS},
    )
    monkeypatch.setattr(builder, "FROZEN_PANEL_COVERAGE", (2, "2018-01-02", "2018-01-03"))
    normalized = (
        "date,open,high,low,close\n"
        "2018-01-02,10.000000,11.000000,9.000000,10.000000\n"
        "2018-01-03,11.000000,12.000000,10.000000,11.000000\n"
    ).encode()
    panel = builder.build_panel({symbol: normalized for symbol in builder.SYMBOLS}).decode()
    assert panel.splitlines()[0] == ",".join(builder.PANEL_COLUMNS)
    assert panel.count("2018-") == 2


def test_deterministic_gzip_round_trip(tmp_path: Path) -> None:
    payload = b"date,close\n2018-01-02,10\n"
    left = tmp_path / "left.csv.gz"
    right = tmp_path / "right.csv.gz"
    builder.write_gzip(left, payload)
    builder.write_gzip(right, payload)
    assert left.read_bytes() == right.read_bytes()
    with gzip.open(left, "rb") as handle:
        assert handle.read() == payload
