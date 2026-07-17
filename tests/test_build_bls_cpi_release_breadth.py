from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from training import build_bls_cpi_release_breadth as builder


ARTIFACT_DIR = Path("data/bls_cpi_release_breadth_2019_2023")


def test_parse_schedule_binds_official_clock_and_reference_month() -> None:
    markdown = """Title: Schedule of Selected Releases 2022
URL Source: http://www.bls.gov/schedule/2022/home.htm
Markdown Content:
""" + "\n".join(
        f"| Wednesday, {month} {day:02d}, 2022 | 08:30 AM | **Consumer Price Index** for {reference} 2021 |"
        for month, day, reference in [
            ("January", 12, "December"),
            ("February", 10, "January"),
            ("March", 10, "February"),
            ("April", 12, "March"),
            ("May", 11, "April"),
            ("June", 10, "May"),
            ("July", 13, "June"),
            ("August", 10, "July"),
            ("September", 13, "August"),
            ("October", 13, "September"),
            ("November", 10, "October"),
            ("December", 13, "November"),
        ]
    )
    rows = builder.parse_schedule(markdown, year=2022)
    assert len(rows) == 12
    assert rows[0]["release_date"] == date(2022, 1, 12)
    assert rows[0]["reference_month"] == date(2021, 12, 1)
    assert rows[0]["release_url"].endswith("cpi_01122022.htm")


def test_parse_release_reads_point_in_time_table_values() -> None:
    markdown = """Title: Consumer Price Index News Release
URL Source: http://www.bls.gov/news.release/archives/cpi_09132022.htm
Markdown Content:
Transmission of material in this release is embargoed until
8:30 a.m. (ET) Tuesday, September 13, 2022
CONSUMER PRICE INDEX - AUGUST 2022
Over the last 12 months, the all items index increased 8.3 percent before seasonal adjustment.
The all items less food and energy index rose 6.3 percent over the last 12 months.
| All items | 0.8 | 1.2 | 0.3 | 1.0 | 1.3 | 0.0 | 0.1 | 8.3 |
| All items less food and energy | 0.5 | 0.3 | 0.6 | 0.6 | 0.7 | 0.3 | 0.6 | 6.3 |
"""
    values = builder.parse_release(
        markdown,
        release_date=date(2022, 9, 13),
        reference_month=date(2022, 8, 1),
    )
    assert values == {
        "headline_yoy_pct": Decimal("8.3"),
        "core_yoy_pct": Decimal("6.3"),
    }


def test_parse_release_accepts_legacy_bls_separator_encoding() -> None:
    markdown = """Title: Consumer Price Index News Release
URL Source: http://www.bls.gov/news.release/archives/cpi_01112019.htm
Markdown Content:
8:30 a.m. (EST) January 11, 2019
CONSUMER PRICE INDEX �DECEMBER 2018
Over the last 12 months, the all items index increased 1.9 percent before seasonal adjustment.
The all items less food and energy index rose 2.2 percent over the last 12 months.
"""
    values = builder.parse_release(
        markdown,
        release_date=date(2019, 1, 11),
        reference_month=date(2018, 12, 1),
    )
    assert values["headline_yoy_pct"] == Decimal("1.9")


def test_release_clock_respects_eastern_daylight_saving() -> None:
    winter = builder.release_time_utc(date(2022, 1, 12))
    summer = builder.release_time_utc(date(2022, 9, 13))
    assert winter == pd.Timestamp("2022-01-12T13:30:00Z")
    assert summer == pd.Timestamp("2022-09-13T12:30:00Z")


def test_build_panel_requires_independent_fred_crosscheck() -> None:
    release_date = date(2022, 9, 13)
    reference = date(2022, 8, 1)
    release_url = builder.official_release_url(release_date)
    release = """Title: Consumer Price Index News Release
URL Source: http://www.bls.gov/news.release/archives/cpi_09132022.htm
Markdown Content:
8:30 a.m. (ET) Tuesday, September 13, 2022
CONSUMER PRICE INDEX - AUGUST 2022
Over the last 12 months, the all items index increased 8.3 percent before seasonal adjustment.
The all items less food and energy index rose 6.3 percent over the last 12 months.
| All items | 8.3 |
| All items less food and energy | 6.3 |
"""
    fred = {
        "headline": {
            date(2021, 8, 1): Decimal("273.567"),
            reference: Decimal("296.171"),
        },
        "core": {
            date(2021, 8, 1): Decimal("279.507"),
            reference: Decimal("297.178"),
        },
    }
    frame = builder.build_panel(
        [
            {
                "release_date": release_date,
                "reference_month": reference,
                "schedule_url": builder.official_schedule_url(2022),
                "release_url": release_url,
            }
        ],
        {release_url: release},
        fred,
    )
    assert len(frame) == 1
    assert bool(frame.iloc[0]["fred_crosscheck_passed"])
    assert frame.iloc[0]["release_time_utc"] == pd.Timestamp("2022-09-13T12:30:00Z")


def test_frozen_source_artifact_identity_and_boundary() -> None:
    manifest = json.loads((ARTIFACT_DIR / "build_manifest.json").read_text())
    data_path = ARTIFACT_DIR / "bls_cpi_release_breadth_2019_2023.csv.gz"
    frame = pd.read_csv(data_path, parse_dates=["release_time_utc"])
    assert manifest["rows"] == 60 == len(frame)
    assert manifest["output_sha256"] == builder.sha256_file(data_path)
    assert manifest["output_sha256"] == (
        "d199f409952d8cb83218864d0a96573bed82b59e649067b22fc97580a06d1059"
    )
    assert manifest["market_or_funding_rows_read"] == 0
    assert manifest["all_fred_crosschecks_passed"] is True
    assert bool(frame["source_complete"].all())
    assert frame["release_time_utc"].is_monotonic_increasing
    assert frame.iloc[-1]["release_time_utc"] < pd.Timestamp("2024-01-01T00:00:00Z")
