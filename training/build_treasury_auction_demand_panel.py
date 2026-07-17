"""Freeze a causal, source-only U.S. Treasury auction-demand panel.

The TreasuryDirect auction query contains official competitive auction results.
This builder keeps only original-issue, nominal fixed-rate coupon auctions and
never imports crypto prices, returns, labels, or portfolio outcomes.

Historical rows expose an Eastern-time update timestamp but no explicit time
zone.  Rather than infer a faster public-release clock, every result is made
available at 22:00 UTC on its auction date.  That is no earlier than 17:00 ET
in either standard or daylight time and is therefore deliberately later than
TreasuryDirect's documented after-17:00 account-availability guarantee.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as wall_time, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, cast


BASE_URL = "https://www.treasurydirect.gov/TA_WS/securities/jqsearch"
USER_AGENT = "rllm-treasury-auction-source-freeze/1.0"
SOURCE_SNAPSHOT_DATE = "2026-07-17"
SCHEMA_VERSION = 1
PAGE_SIZE = 2000
FROZEN_PAGES = (0, 1)
MIN_AUCTION_DATE = date(2016, 1, 1)
MAX_AUCTION_DATE = date(2023, 12, 31)
RESULT_AVAILABLE_UTC = wall_time(22, 0, tzinfo=timezone.utc)
ORIGINAL_TERMS = (
    "2-Year",
    "3-Year",
    "5-Year",
    "7-Year",
    "10-Year",
    "20-Year",
    "30-Year",
)

FROZEN_RESPONSE_SHA256 = {
    0: "c3b3ec599a5d8e94e41ba2588433b4fb899cd3bdd8172ce04bb515dba9977b2d",
    1: "1be85936b3fb04321665d4980f1b7f7191529a6f7b08a9355ee459b4bb94886f",
}
FROZEN_PAGE_COVERAGE = {
    0: (2000, "2021-11-09", "2026-07-23"),
    1: (2000, "2016-02-18", "2021-11-08"),
}
FROZEN_PANEL_COVERAGE = {
    "rows": 445,
    "complete_rows": 440,
    "incomplete_rows": 5,
    "first_auction_date": "2016-02-24",
    "last_auction_date": "2023-12-28",
    "year_counts": {
        "2016": 48,
        "2017": 52,
        "2018": 56,
        "2019": 51,
        "2020": 59,
        "2021": 60,
        "2022": 59,
        "2023": 60,
    },
    "term_counts": {
        "2-Year": 90,
        "3-Year": 93,
        "5-Year": 90,
        "7-Year": 95,
        "10-Year": 31,
        "20-Year": 15,
        "30-Year": 31,
    },
}

RAW_REQUIRED_FIELDS = (
    "auctionDate",
    "securityType",
    "originalSecurityTerm",
    "cusip",
    "reopening",
    "tips",
    "floatingRate",
    "bidToCoverRatio",
    "competitiveAccepted",
    "primaryDealerAccepted",
    "directBidderAccepted",
    "indirectBidderAccepted",
    "closingTimeCompetitive",
    "updatedTimestamp",
    "pdfFilenameCompetitiveResults",
    "xmlFilenameCompetitiveResults",
)
OUTPUT_COLUMNS = (
    "auction_date",
    "result_available_at_utc",
    "security_type",
    "original_security_term",
    "cusip",
    "bid_to_cover_ratio",
    "competitive_accepted_usd",
    "primary_dealer_accepted_usd",
    "direct_bidder_accepted_usd",
    "indirect_bidder_accepted_usd",
    "indirect_competitive_share",
    "closing_time_competitive_et",
    "updated_timestamp_et",
    "competitive_results_pdf_url",
    "competitive_results_xml_url",
    "source_complete",
)


@dataclass(frozen=True)
class BuildConfig:
    output_dir: str = "data/us_treasury_auction_demand_2016_2023"
    start_date: str = MIN_AUCTION_DATE.isoformat()
    end_date: str = MAX_AUCTION_DATE.isoformat()
    retries: int = 5
    timeout_seconds: int = 60


def page_url(page: int, *, base_url: str = BASE_URL) -> str:
    if page < 0:
        raise ValueError("page must be non-negative")
    params = {"format": "json", "pagenum": page, "pagesize": PAGE_SIZE}
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def _fetch_bytes(url: str, *, retries: int, timeout: int) -> bytes:
    error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt + 1 < retries:
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts") from error


def _parse_iso_datetime(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO datetime string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid ISO datetime: {value!r}") from exc
    if parsed.tzinfo is not None:
        raise ValueError(f"{field} unexpectedly contains a timezone: {value!r}")
    return parsed


def _parse_decimal(value: Any, *, field: str) -> Decimal:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty base-10 string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} is not base-10 numeric: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite")
    return parsed


def parse_page_response(payload: bytes) -> tuple[int, list[dict[str, Any]]]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("TreasuryDirect response is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("TreasuryDirect response must be an object")
    total = document.get("totalResultsCount")
    rows = document.get("securityList")
    if not isinstance(total, int) or total < 1 or not isinstance(rows, list):
        raise ValueError("TreasuryDirect response has an invalid result envelope")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("TreasuryDirect securityList must contain objects")
    return total, rows


def _result_url(*, auction_year: int, filename: str, xml: bool) -> str:
    if not filename or "/" in filename or "\\" in filename:
        raise ValueError(f"unsafe TreasuryDirect result filename: {filename!r}")
    if xml:
        return f"https://www.treasurydirect.gov/xml/{filename}"
    return (
        "https://www.treasurydirect.gov/instit/annceresult/press/preanre/"
        f"{auction_year}/{filename}"
    )


def normalize_panel(
    raw_rows: list[dict[str, Any]], *, start: date, end: date
) -> list[dict[str, str]]:
    if start > end:
        raise ValueError("start date must not exceed end date")
    if end > MAX_AUCTION_DATE:
        raise ValueError("2024+ source rows are sealed by this research snapshot")

    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_rows:
        missing = set(RAW_REQUIRED_FIELDS).difference(raw)
        if missing:
            raise ValueError(f"TreasuryDirect row is missing: {sorted(missing)}")
        auction_dt = _parse_iso_datetime(raw["auctionDate"], field="auctionDate")
        auction_date = auction_dt.date()
        if not start <= auction_date <= end:
            continue
        if (
            raw["reopening"] != "No"
            or raw["tips"] != "No"
            or raw["floatingRate"] != "No"
            or raw["originalSecurityTerm"] not in ORIGINAL_TERMS
            or raw["securityType"] not in {"Note", "Bond"}
        ):
            continue

        key = (auction_date.isoformat(), str(raw["cusip"]))
        if key in seen:
            raise ValueError(f"duplicate Treasury auction row: {key}")
        seen.add(key)

        bid_to_cover = _parse_decimal(
            raw["bidToCoverRatio"], field="bidToCoverRatio"
        )
        competitive = _parse_decimal(
            raw["competitiveAccepted"], field="competitiveAccepted"
        )
        primary = _parse_decimal(
            raw["primaryDealerAccepted"], field="primaryDealerAccepted"
        )
        direct = _parse_decimal(
            raw["directBidderAccepted"], field="directBidderAccepted"
        )
        indirect = _parse_decimal(
            raw["indirectBidderAccepted"], field="indirectBidderAccepted"
        )
        if bid_to_cover <= 0 or competitive <= 0:
            raise ValueError("auction demand values must be positive")
        if primary < 0 or direct < 0 or indirect < 0:
            raise ValueError("bidder accepted amounts must be non-negative")
        if primary + direct + indirect != competitive:
            raise ValueError("bidder accepted amounts do not sum to competitiveAccepted")

        updated = _parse_iso_datetime(
            raw["updatedTimestamp"], field="updatedTimestamp"
        )
        source_complete = (
            updated.date() == auction_date and updated.time() < wall_time(22, 0)
        )
        closing = raw["closingTimeCompetitive"]
        if not isinstance(closing, str) or not closing:
            raise ValueError("closingTimeCompetitive must be non-empty text")

        available = datetime.combine(auction_date, RESULT_AVAILABLE_UTC)
        selected.append(
            {
                "auction_date": auction_date.isoformat(),
                "result_available_at_utc": available.isoformat(),
                "security_type": str(raw["securityType"]),
                "original_security_term": str(raw["originalSecurityTerm"]),
                "cusip": str(raw["cusip"]),
                "bid_to_cover_ratio": format(bid_to_cover, "f") if source_complete else "",
                "competitive_accepted_usd": format(competitive, "f") if source_complete else "",
                "primary_dealer_accepted_usd": format(primary, "f") if source_complete else "",
                "direct_bidder_accepted_usd": format(direct, "f") if source_complete else "",
                "indirect_bidder_accepted_usd": format(indirect, "f") if source_complete else "",
                "indirect_competitive_share": (
                    format(indirect / competitive, ".15f") if source_complete else ""
                ),
                "closing_time_competitive_et": closing,
                "updated_timestamp_et": updated.isoformat(),
                "competitive_results_pdf_url": _result_url(
                    auction_year=auction_date.year,
                    filename=str(raw["pdfFilenameCompetitiveResults"]),
                    xml=False,
                ),
                "competitive_results_xml_url": _result_url(
                    auction_year=auction_date.year,
                    filename=str(raw["xmlFilenameCompetitiveResults"]),
                    xml=True,
                ),
                "source_complete": "true" if source_complete else "false",
            }
        )
    selected.sort(key=lambda row: (row["auction_date"], row["cusip"]))
    return selected


def _write_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(payload)


def _write_gzip_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=list(OUTPUT_COLUMNS), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(cast(Any, rows))


def _coverage(rows: list[dict[str, str]]) -> dict[str, Any]:
    years: dict[str, int] = {}
    terms: dict[str, int] = {}
    for row in rows:
        year = row["auction_date"][:4]
        term = row["original_security_term"]
        years[year] = years.get(year, 0) + 1
        terms[term] = terms.get(term, 0) + 1
    return {
        "rows": len(rows),
        "complete_rows": sum(row["source_complete"] == "true" for row in rows),
        "incomplete_rows": sum(row["source_complete"] != "true" for row in rows),
        "first_auction_date": rows[0]["auction_date"] if rows else None,
        "last_auction_date": rows[-1]["auction_date"] if rows else None,
        "year_counts": dict(sorted(years.items())),
        "term_counts": dict(sorted(terms.items())),
    }


def build(
    cfg: BuildConfig, *, fetcher: Callable[..., bytes] = _fetch_bytes
) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start_date)
    end = date.fromisoformat(cfg.end_date)
    if start < MIN_AUCTION_DATE or end > MAX_AUCTION_DATE:
        raise ValueError("builder supports only the frozen 2016-2023 source window")
    if cfg.retries < 1 or cfg.timeout_seconds < 1:
        raise ValueError("retries and timeout_seconds must be positive")

    output_dir = Path(cfg.output_dir)
    raw_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    total_counts: set[int] = set()
    for page in FROZEN_PAGES:
        url = page_url(page)
        payload = fetcher(url, retries=cfg.retries, timeout=cfg.timeout_seconds)
        response_sha = hashlib.sha256(payload).hexdigest()
        expected_sha = FROZEN_RESPONSE_SHA256.get(page)
        if response_sha != expected_sha:
            raise ValueError(
                f"TreasuryDirect page {page} changed: expected={expected_sha}, "
                f"actual={response_sha}"
            )
        total, rows = parse_page_response(payload)
        total_counts.add(total)
        dates = sorted(_parse_iso_datetime(r["auctionDate"], field="auctionDate").date() for r in rows)
        page_coverage = (
            len(rows),
            dates[0].isoformat(),
            dates[-1].isoformat(),
        )
        if page_coverage != FROZEN_PAGE_COVERAGE.get(page):
            raise ValueError(
                f"TreasuryDirect page {page} coverage changed: {page_coverage}"
            )
        raw_path = output_dir / "raw" / f"auction_query_page_{page}.json.gz"
        _write_gzip(raw_path, payload)
        raw_rows.extend(rows)
        sources.append(
            {
                "page": page,
                "url": url,
                "response_sha256": response_sha,
                "raw_path": str(raw_path),
                "raw_gzip_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "rows": len(rows),
                "first_auction_date": page_coverage[1],
                "last_auction_date": page_coverage[2],
            }
        )
    if len(total_counts) != 1:
        raise ValueError("TreasuryDirect pages disagree on totalResultsCount")

    panel = normalize_panel(raw_rows, start=start, end=end)
    coverage = _coverage(panel)
    if coverage != FROZEN_PANEL_COVERAGE:
        raise ValueError(
            "Treasury auction panel coverage changed: "
            f"expected={FROZEN_PANEL_COVERAGE}, actual={coverage}"
        )
    output_path = output_dir / "us_treasury_nominal_original_auctions_2016_2023.csv.gz"
    _write_gzip_csv(output_path, panel)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": asdict(cfg),
        "protocol": {
            "source": "U.S. Treasury Bureau of the Fiscal Service / TreasuryDirect",
            "source_measure": "official competitive auction results",
            "universe": "original-issue nominal fixed-rate 2y/3y/5y/7y/10y/20y/30y coupon auctions",
            "availability": "22:00 UTC on auction date; conservative after-17:00-ET guard",
            "future_source_rows_used": False,
            "crypto_market_fields_opened": False,
            "outcomes_opened": False,
        },
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "coverage": coverage,
        "columns": list(OUTPUT_COLUMNS),
        "source_total_results_count": total_counts.pop(),
        "sources": sources,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--start-date", default=BuildConfig.start_date)
    parser.add_argument("--end-date", default=BuildConfig.end_date)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument(
        "--timeout-seconds", type=int, default=BuildConfig.timeout_seconds
    )
    result = build(BuildConfig(**vars(parser.parse_args())))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
