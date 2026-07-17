from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import pytest

from training import build_cboe_tail_risk_panel as builder


def _response(columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return output.getvalue().encode()


def test_source_urls_are_official_daily_histories() -> None:
    assert builder.source_url("SKEW") == (
        "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"
    )
    assert builder.source_url("VVIX").endswith("/VVIX_History.csv")
    with pytest.raises(ValueError, match="unsupported"):
        builder.source_url("BTC")


@pytest.mark.parametrize("symbol", ["SKEW", "VVIX"])
def test_normalize_simple_index_response(
    monkeypatch: pytest.MonkeyPatch, symbol: str
) -> None:
    monkeypatch.setitem(
        builder.FROZEN_SOURCE_COVERAGE,
        symbol,
        (2, "2018-01-02", "2023-12-29"),
    )
    column = builder.VALUE_COLUMN[symbol]
    payload = _response(
        ("DATE", column),
        [
            ("12/29/2017", "100"),
            ("01/02/2018", "101.25"),
            ("12/29/2023", "111.5"),
            ("01/02/2024", "120"),
        ],
    )
    normalized = builder.normalize_response(payload, symbol=symbol).decode()
    assert "2017" not in normalized and "2024" not in normalized
    assert "2018-01-02,101.250000" in normalized
    assert normalized.endswith("2023-12-29,111.500000\n")


def test_normalize_vix_validates_ohlc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        builder.FROZEN_SOURCE_COVERAGE,
        "VIX",
        (1, "2018-01-02", "2018-01-02"),
    )
    payload = _response(
        builder.SOURCE_COLUMNS["VIX"],
        [("01/02/2018", "10", "9", "8", "11")],
    )
    with pytest.raises(ValueError, match="OHLC invariant"):
        builder.normalize_response(payload, symbol="VIX")


def test_normalize_rejects_schema_change() -> None:
    payload = _response(("DATE", "VALUE"), [("01/02/2018", "100")])
    with pytest.raises(ValueError, match="schema changed"):
        builder.normalize_response(payload, symbol="SKEW")


def test_panel_uses_exact_three_way_intersection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builder,
        "FROZEN_SOURCE_COVERAGE",
        {symbol: (2, "2018-01-02", "2018-01-03") for symbol in builder.SYMBOLS},
    )
    monkeypatch.setattr(
        builder, "FROZEN_PANEL_COVERAGE", (2, "2018-01-02", "2018-01-03")
    )
    normalized = (
        "date,close\n"
        "2018-01-02,100.000000\n"
        "2018-01-03,101.000000\n"
    ).encode()
    panel = builder.build_panel(
        {symbol: normalized for symbol in builder.SYMBOLS}
    ).decode()
    assert panel.splitlines()[0] == ",".join(builder.PANEL_COLUMNS)
    assert panel.count("2018-") == 2


def test_deterministic_gzip_round_trip(tmp_path: Path) -> None:
    payload = b"date,close\n2018-01-02,100\n"
    left = tmp_path / "left.csv.gz"
    right = tmp_path / "right.csv.gz"
    builder.write_gzip(left, payload)
    builder.write_gzip(right, payload)
    assert left.read_bytes() == right.read_bytes()
    with gzip.open(left, "rb") as handle:
        assert handle.read() == payload
