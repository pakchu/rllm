from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from training import build_cftc_institutional_transfer_absorption as builder


ARTIFACT_DIR = Path("data/cftc_institutional_transfer_absorption_2018_2023")


def _sample_zip() -> bytes:
    header = [
        "Market_and_Exchange_Names",
        "Report_Date_as_YYYY-MM-DD",
        "CFTC_Contract_Market_Code_Quotes",
        "Open_Interest_All",
        "Dealer_Positions_Long_All",
        "Dealer_Positions_Short_All",
        "Asset_Mgr_Positions_Long_All",
        "Asset_Mgr_Positions_Short_All",
        "Lev_Money_Positions_Long_All",
        "Lev_Money_Positions_Short_All",
        "Change_in_Dealer_Long_All",
        "Change_in_Dealer_Short_All",
        "Change_in_Asset_Mgr_Long_All",
        "Change_in_Asset_Mgr_Short_All",
        "Change_in_Lev_Money_Long_All",
        "Change_in_Lev_Money_Short_All",
    ]
    rows = [
        [
            builder.MARKET_NAME,
            "2017-01-10",
            builder.CONTRACT_CODE,
            "1000",
            "120",
            "180",
            "300",
            "100",
            "140",
            "260",
            "20",
            "30",
            "40",
            "10",
            "-20",
            "20",
        ],
        [
            builder.MARKET_NAME,
            "2017-01-03",
            builder.CONTRACT_CODE,
            "900",
            "100",
            "150",
            "260",
            "90",
            "160",
            "240",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
        ],
    ]
    text = "\n".join(
        [",".join(f'"{value}"' for value in header)]
        + [",".join(f'"{value}"' for value in row) for row in rows]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("FinFutYY.txt", text)
    return buffer.getvalue()


def test_parse_and_reconcile_published_participant_changes() -> None:
    annual = builder.parse_annual_zip(_sample_zip(), year=2017)
    panel = builder.build_panel([annual])
    assert len(panel) == 2
    assert not bool(panel.iloc[0]["source_complete"])
    assert bool(panel.iloc[1]["source_complete"])
    assert panel.iloc[1]["asset_mgr_published_net_change"] == 30
    assert panel.iloc[1]["asset_mgr_arithmetic_net_change"] == 30
    assert panel.iloc[1]["lev_money_published_net_change"] == -40
    assert panel.iloc[1]["lev_money_arithmetic_net_change"] == -40


def test_availability_is_conservative_and_honors_ion_backlog() -> None:
    assert builder.conservative_available_time_utc(
        date(2022, 1, 4)
    ) == pd.Timestamp("2022-01-12T00:00:00Z")
    assert builder.conservative_available_time_utc(
        date(2023, 1, 31)
    ) == pd.Timestamp("2023-02-25T00:00:00Z")


def test_frozen_source_artifact_identity_and_boundary() -> None:
    manifest = json.loads((ARTIFACT_DIR / "build_manifest.json").read_text())
    data_path = (
        ARTIFACT_DIR
        / "cftc_institutional_transfer_absorption_2018_2023.csv.gz"
    )
    frame = pd.read_csv(data_path, parse_dates=["available_time_utc"])
    assert manifest["rows"] == 299 == len(frame)
    assert manifest["output_sha256"] == builder.sha256_file(data_path)
    assert manifest["output_sha256"] == (
        "064eed3fa340b1701f4686d1176de2a10f39128abc5ebf846e8b6319b8144ee6"
    )
    assert manifest["source_complete_rows"] == 298
    assert manifest["source_quarantined_rows"] == 1
    assert manifest["quarantined_report_dates"] == ["2018-04-10"]
    assert manifest["special_publication_overrides"] == 7
    assert manifest["market_or_funding_rows_read"] == 0
    assert frame["available_time_utc"].is_monotonic_increasing
