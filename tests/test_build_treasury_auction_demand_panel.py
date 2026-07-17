from __future__ import annotations

import gzip
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from training import build_treasury_auction_demand_panel as builder


def _row(
    auction_date: str,
    *,
    cusip: str = "TEST12345",
    term: str = "5-Year",
    reopening: str = "No",
    tips: str = "No",
    floating: str = "No",
    bid_to_cover: str = "2.50",
    primary: str = "40",
    direct: str = "10",
    indirect: str = "50",
    competitive: str = "100",
    updated: str | None = None,
) -> dict[str, object]:
    year = auction_date[:4]
    return {
        "auctionDate": f"{auction_date}T00:00:00",
        "securityType": "Note" if term != "30-Year" else "Bond",
        "originalSecurityTerm": term,
        "cusip": cusip,
        "reopening": reopening,
        "tips": tips,
        "floatingRate": floating,
        "bidToCoverRatio": bid_to_cover,
        "competitiveAccepted": competitive,
        "primaryDealerAccepted": primary,
        "directBidderAccepted": direct,
        "indirectBidderAccepted": indirect,
        "closingTimeCompetitive": "01:00 PM",
        "updatedTimestamp": updated or f"{auction_date}T13:03:00",
        "pdfFilenameCompetitiveResults": f"R_{year}0101_1.pdf",
        "xmlFilenameCompetitiveResults": f"R_{year}0101_1.xml",
    }


def _payload(*rows: dict[str, object], total: int = 4000) -> bytes:
    return json.dumps(
        {"totalResultsCount": total, "securityList": list(rows)},
        separators=(",", ":"),
    ).encode()


class _FakeFetcher:
    def __init__(self, payloads: dict[int, bytes]) -> None:
        self.payloads = payloads

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        page = int(url.split("pagenum=")[1].split("&")[0])
        return self.payloads[page]


def test_page_url_is_official_and_bounded() -> None:
    assert builder.page_url(1) == (
        "https://www.treasurydirect.gov/TA_WS/securities/jqsearch?"
        "format=json&pagenum=1&pagesize=2000"
    )
    with pytest.raises(ValueError, match="non-negative"):
        builder.page_url(-1)


def test_parse_page_response_rejects_bad_envelopes() -> None:
    total, rows = builder.parse_page_response(_payload(_row("2023-01-01")))
    assert total == 4000
    assert len(rows) == 1
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        builder.parse_page_response(b"not-json")
    with pytest.raises(ValueError, match="invalid result envelope"):
        builder.parse_page_response(b"{}")


def test_normalize_panel_filters_to_original_nominal_coupon_auctions() -> None:
    rows = [
        _row("2023-01-25", cusip="KEEP"),
        _row("2023-01-26", cusip="REOPEN", reopening="Yes"),
        _row("2023-01-27", cusip="TIPS", tips="Yes"),
        _row("2023-01-28", cusip="FRN", floating="Yes"),
        _row("2024-01-01", cusip="FUTURE"),
    ]
    panel = builder.normalize_panel(
        rows,
        start=builder.MIN_AUCTION_DATE,
        end=builder.MAX_AUCTION_DATE,
    )
    assert len(panel) == 1
    row = panel[0]
    assert row["cusip"] == "KEEP"
    assert row["result_available_at_utc"] == "2023-01-25T22:00:00+00:00"
    assert row["indirect_competitive_share"] == "0.500000000000000"
    assert row["competitive_results_pdf_url"].endswith(
        "/2023/R_20230101_1.pdf"
    )
    assert row["competitive_results_xml_url"].endswith("/xml/R_20230101_1.xml")


def test_normalize_panel_rejects_accounting_and_quarantines_late_updates() -> None:
    with pytest.raises(ValueError, match="do not sum"):
        builder.normalize_panel(
            [_row("2023-01-25", competitive="99")],
            start=builder.MIN_AUCTION_DATE,
            end=builder.MAX_AUCTION_DATE,
        )
    late = builder.normalize_panel(
        [_row("2023-01-25", updated="2023-04-28T11:33:00")],
        start=builder.MIN_AUCTION_DATE,
        end=builder.MAX_AUCTION_DATE,
    )
    assert late[0]["source_complete"] == "false"
    assert late[0]["bid_to_cover_ratio"] == ""
    assert late[0]["indirect_competitive_share"] == ""
    with pytest.raises(ValueError, match=r"2024\+"):
        builder.normalize_panel(
            [], start=builder.MIN_AUCTION_DATE, end=date(2024, 1, 1)
        )


def test_build_is_deterministic_and_opens_no_crypto_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page0 = _payload(_row("2023-12-28", cusip="NEW", term="30-Year"))
    page1 = _payload(_row("2016-02-24", cusip="OLD", term="2-Year"))
    payloads = {0: page0, 1: page1}
    monkeypatch.setattr(builder, "FROZEN_PAGES", (0, 1))
    monkeypatch.setattr(
        builder,
        "FROZEN_RESPONSE_SHA256",
        {page: hashlib.sha256(payload).hexdigest() for page, payload in payloads.items()},
    )
    monkeypatch.setattr(
        builder,
        "FROZEN_PAGE_COVERAGE",
        {
            0: (1, "2023-12-28", "2023-12-28"),
            1: (1, "2016-02-24", "2016-02-24"),
        },
    )
    expected_coverage = {
        "rows": 2,
        "complete_rows": 2,
        "incomplete_rows": 0,
        "first_auction_date": "2016-02-24",
        "last_auction_date": "2023-12-28",
        "year_counts": {"2016": 1, "2023": 1},
        "term_counts": {"2-Year": 1, "30-Year": 1},
    }
    monkeypatch.setattr(builder, "FROZEN_PANEL_COVERAGE", expected_coverage)
    cfg = builder.BuildConfig(output_dir=str(tmp_path))
    first = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    first_output = Path(first["output"]).read_bytes()
    first_manifest = (tmp_path / "build_manifest.json").read_bytes()
    second = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    assert Path(second["output"]).read_bytes() == first_output
    assert (tmp_path / "build_manifest.json").read_bytes() == first_manifest
    assert first["coverage"] == expected_coverage
    assert first["protocol"]["crypto_market_fields_opened"] is False
    assert first["protocol"]["outcomes_opened"] is False
    for source in first["sources"]:
        raw_path = Path(source["raw_path"])
        assert raw_path.exists()
        with gzip.open(raw_path, "rb") as handle:
            assert handle.read() == payloads[source["page"]]


def test_build_rejects_mutated_source_and_future_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "FROZEN_PAGES", (0,))
    monkeypatch.setattr(builder, "FROZEN_RESPONSE_SHA256", {0: "0" * 64})
    with pytest.raises(ValueError, match="page 0 changed"):
        builder.build(
            builder.BuildConfig(output_dir=str(tmp_path)),
            fetcher=_FakeFetcher({0: _payload(_row("2023-01-01"))}),
        )
    with pytest.raises(ValueError, match="2016-2023"):
        builder.build(
            builder.BuildConfig(output_dir=str(tmp_path), end_date="2024-01-01"),
            fetcher=_FakeFetcher({}),
        )
