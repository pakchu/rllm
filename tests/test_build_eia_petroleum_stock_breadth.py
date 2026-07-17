from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from training import build_eia_petroleum_stock_breadth as builder


ARTIFACT_DIR = Path("data/eia_petroleum_stock_breadth_2019_2023")


def _sample_table() -> bytes:
    return (
        '"STUB_1","12/31/21","12/24/21","Difference","Percent Change"\r\n'
        '"Crude Oil","1,011.533","1,015.023","-3.490","-0.300"\r\n'
        '"Commercial (Excluding SPR)","417.851","419.995","-2.144","-0.500"\r\n'
        '"Strategic Petroleum Reserve (SPR)","593.682","595.028","-1.346","-0.200"\r\n'
        '"Total Motor Gasoline","232.787","222.659","10.128","4.500"\r\n'
        '"Distillate Fuel Oil","126.846","122.428","4.418","3.600"\r\n'
        '"STUB_1","STUB_2","12/31/21","12/24/21","Difference"\r\n'
    ).encode("cp1252")


def test_parse_archive_index_binds_release_and_table_urls() -> None:
    html = """
    <a href="/petroleum/supply/weekly/archive/2022/2022_01_05/wpsr_2022_01_05.php">5</a>
    <a href="/petroleum/supply/weekly/archive/2022/2022_01_12/wpsr_2022_01_12.php">12</a>
    """
    rows = builder.parse_archive_index(
        html,
        start_year=2022,
        end_year=2022,
        require_annual_density=False,
    )
    assert len(rows) == 2
    assert rows[0]["release_date"] == date(2022, 1, 5)
    assert rows[0]["table1_csv_url"].endswith("2022_01_05/csv/table1.csv")


def test_parse_table1_uses_issue_local_current_prior_and_difference() -> None:
    values = builder.parse_table1(_sample_table(), release_date=date(2022, 1, 5))
    assert values["data_week_ending"] == date(2021, 12, 31)
    assert values["previous_week_ending"] == date(2021, 12, 24)
    assert float(values["commercial_crude_change_mmbbl"]) == -2.144
    assert float(values["commercial_crude_change_discrepancy_mmbbl"]) == 0.0
    assert float(values["gasoline_change_mmbbl"]) == 10.128
    assert float(values["distillate_change_mmbbl"]) == 4.418


def test_availability_is_conservatively_after_release_date() -> None:
    available = builder.conservative_available_time_utc(date(2022, 1, 5))
    assert available == pd.Timestamp("2022-01-06T13:00:00Z")


def test_build_panel_reads_no_market_outcome() -> None:
    url = (
        "https://www.eia.gov/petroleum/supply/weekly/archive/2022/"
        "2022_01_05/csv/table1.csv"
    )
    frame = builder.build_panel(
        [
            {
                "release_date": date(2022, 1, 5),
                "archive_page_url": url.replace("/csv/table1.csv", "/wpsr.php"),
                "table1_csv_url": url,
            }
        ],
        {url: _sample_table()},
    )
    assert len(frame) == 1
    assert bool(frame.iloc[0]["source_complete"])
    assert frame.iloc[0]["available_time_utc"] == pd.Timestamp(
        "2022-01-06T13:00:00Z"
    )


def test_frozen_source_artifact_identity_and_boundary() -> None:
    manifest = json.loads((ARTIFACT_DIR / "build_manifest.json").read_text())
    data_path = ARTIFACT_DIR / "eia_petroleum_stock_breadth_2019_2023.csv.gz"
    frame = pd.read_csv(data_path, parse_dates=["available_time_utc"])
    assert manifest["rows"] == 259 == len(frame)
    assert manifest["output_sha256"] == builder.sha256_file(data_path)
    assert manifest["output_sha256"] == (
        "26cbe6a91079a64fd9bbcb1cb5e1f81e15df25e45ed2171f7c464d048b34757b"
    )
    assert manifest["market_or_funding_rows_read"] == 0
    assert manifest["source_complete_rows"] == 258
    assert manifest["source_quarantined_rows"] == 1
    quarantined = frame.loc[~frame["source_complete"].astype(bool)]
    assert quarantined["release_date"].tolist() == ["2023-12-28"]
    assert frame["available_time_utc"].is_monotonic_increasing
    assert frame.iloc[-1]["available_time_utc"] < pd.Timestamp(
        "2024-01-01T00:00:00Z"
    )
