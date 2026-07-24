from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
LEDGER = (
    ROOT
    / "data/federal_reserve_deliberation_communication_identity_2012_2020.csv"
)
EXPECTED_SHA256 = (
    "a8586f749e2e1d3f3a83fb14de579f3c286fd1a8077af8fb9a5b67d247012bea"
)
EXPECTED_FIELDS = [
    "document_class",
    "meeting_date",
    "release_date",
    "last_update_date",
    "encoding",
    "source_eligible",
    "official_url",
    "index_url",
]
EXPECTED_QUARANTINES = {
    (
        "2018-01-31",
        "2018-02-01",
        "https://www.federalreserve.gov/newsevents/pressreleases/"
        "monetary20180131a.htm",
    ),
    (
        "2019-10-11",
        "2019-10-15",
        "https://www.federalreserve.gov/newsevents/pressreleases/"
        "monetary20191011a.htm",
    ),
}
EXPECTED_NON_UTF8 = (
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes20111213.htm"
)

STATEMENT_PATH = re.compile(
    r"/newsevents/pressreleases/monetary(?P<date>\d{8})a\.htm"
)
MINUTES_PATH = re.compile(
    r"/monetarypolicy/fomcminutes(?P<date>\d{8})\.htm"
)
INDEX_URL = re.compile(
    r"https://www\.federalreserve\.gov/monetarypolicy/"
    r"fomchistorical(?P<year>201[1-9]|2020)\.htm"
)


def _read() -> tuple[bytes, list[str], list[dict[str, str]]]:
    raw = LEDGER.read_bytes()
    text = raw.decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    rows = list(reader)
    return raw, list(reader.fieldnames or ()), rows


def _iso(value: str) -> date:
    parsed = date.fromisoformat(value)
    assert parsed.isoformat() == value
    return parsed


def test_identity_ledger_exact_bytes_schema_hash_and_order() -> None:
    raw, fields, rows = _read()
    assert len(raw) == 29_677
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    assert fields == EXPECTED_FIELDS
    assert len(rows) == 147
    assert len({row["official_url"] for row in rows}) == 147
    assert rows == sorted(
        rows,
        key=lambda row: (
            row["release_date"],
            row["document_class"],
            row["official_url"],
        ),
    )


def test_identity_ledger_freezes_counts_quarantines_and_encodings() -> None:
    _, _, rows = _read()
    assert Counter(row["document_class"] for row in rows) == {
        "fomc_statement": 75,
        "fomc_minutes": 72,
    }
    assert Counter(row["source_eligible"] for row in rows) == {
        "true": 145,
        "false": 2,
    }
    eligible = [row for row in rows if row["source_eligible"] == "true"]
    assert Counter(row["document_class"] for row in eligible) == {
        "fomc_statement": 73,
        "fomc_minutes": 72,
    }
    assert Counter(row["release_date"][:4] for row in rows) == {
        "2012": 16,
        "2013": 16,
        "2014": 16,
        "2015": 16,
        "2016": 16,
        "2017": 16,
        "2018": 16,
        "2019": 17,
        "2020": 18,
    }
    assert {
        (
            row["release_date"],
            row["last_update_date"],
            row["official_url"],
        )
        for row in rows
        if row["source_eligible"] == "false"
    } == EXPECTED_QUARANTINES
    assert Counter(row["encoding"] for row in rows) == {
        "utf-8": 146,
        "windows-1252": 1,
    }
    assert [
        row["official_url"] for row in rows if row["encoding"] != "utf-8"
    ] == [EXPECTED_NON_UTF8]
    assert all(
        row["last_update_date"] == row["release_date"] for row in eligible
    )
    assert sum(
        count - 1
        for count in Counter(row["document_class"] for row in eligible).values()
    ) == 143


def test_identity_ledger_seals_release_window_and_current_calendar() -> None:
    _, _, rows = _read()
    release_dates = [_iso(row["release_date"]) for row in rows]
    assert min(release_dates) == date(2012, 1, 3)
    assert max(release_dates) == date(2020, 12, 16)
    assert all(date(2012, 1, 1) <= value <= date(2020, 12, 31) for value in release_dates)
    assert all("fomccalendars" not in row["index_url"] for row in rows)
    assert all("fomccalendars" not in row["official_url"] for row in rows)
    assert all("beige" not in row["official_url"].lower() for row in rows)


def test_identity_ledger_urls_clocks_and_index_years_are_exact() -> None:
    _, _, rows = _read()
    for row in rows:
        assert row["source_eligible"] in {"true", "false"}
        assert row["encoding"] in {"utf-8", "windows-1252"}

        meeting = _iso(row["meeting_date"])
        release = _iso(row["release_date"])
        last_update = _iso(row["last_update_date"])
        assert meeting <= release
        assert last_update >= release

        parsed = urlsplit(row["official_url"])
        assert parsed.scheme == "https"
        assert parsed.netloc == "www.federalreserve.gov"
        assert not parsed.query
        assert not parsed.fragment

        index_match = INDEX_URL.fullmatch(row["index_url"])
        assert index_match is not None
        assert int(index_match.group("year")) == meeting.year

        if row["document_class"] == "fomc_statement":
            match = STATEMENT_PATH.fullmatch(parsed.path)
            assert match is not None
            path_date = date.fromisoformat(
                f"{match.group('date')[:4]}-"
                f"{match.group('date')[4:6]}-"
                f"{match.group('date')[6:]}"
            )
            assert path_date == meeting == release
        else:
            assert row["document_class"] == "fomc_minutes"
            match = MINUTES_PATH.fullmatch(parsed.path)
            assert match is not None
            path_date = date.fromisoformat(
                f"{match.group('date')[:4]}-"
                f"{match.group('date')[4:6]}-"
                f"{match.group('date')[6:]}"
            )
            assert path_date == meeting
            assert release > meeting
