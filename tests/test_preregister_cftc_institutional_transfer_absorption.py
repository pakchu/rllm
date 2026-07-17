from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

from training import cftc_institutional_transfer_absorption_clock as clock
from training import preregister_cftc_institutional_transfer_absorption as prereg


def _row(asset: str, leveraged: str) -> clock.SourceRow:
    return clock.SourceRow(
        report_date="2022-01-04",
        available_time=pd.Timestamp("2022-01-12T00:00:00Z"),
        asset_mgr_net_change=Decimal(asset),
        lev_money_net_change=Decimal(leveraged),
        official_zip_url="https://www.cftc.gov/example.zip",
        special_publication_override=False,
        source_complete=True,
    )


def test_primary_direction_is_institutional_transfer_not_a_fitted_sign() -> None:
    assert clock.side_for_row(_row("10", "-5")) == 1
    assert clock.side_for_row(_row("-10", "5")) == -1
    assert clock.side_for_row(_row("10", "5")) == 0
    assert clock.side_for_row(_row("10", "5"), mode="asset_manager_only") == 1
    assert (
        clock.side_for_row(_row("10", "5"), mode="leveraged_contrarian_only")
        == -1
    )


def test_source_and_primary_clock_are_frozen_and_nonoverlapping() -> None:
    rows = clock.load_source()
    assert len(rows) == 299
    assert sum(row.source_complete for row in rows) == 298
    frame = clock.events_frame(clock.build_events(rows))
    stage1 = frame.loc[
        frame["entry_time"].ge(pd.Timestamp("2020-01-01T00:00:00Z"))
        & frame["exit_time"].le(pd.Timestamp("2023-01-01T00:00:00Z"))
    ]
    stage2 = frame.loc[
        frame["entry_time"].ge(pd.Timestamp("2023-01-01T00:00:00Z"))
        & frame["exit_time"].le(pd.Timestamp("2024-01-01T00:00:00Z"))
    ]
    assert len(stage1) == 98
    assert len(stage2) == 23
    assert frame["entry_time"].eq(
        frame["signal_time"] + pd.Timedelta(minutes=5)
    ).all()
    assert frame["exit_time"].eq(
        frame["entry_time"] + pd.Timedelta(days=7)
    ).all()
    assert frame["entry_time"].iloc[1:].reset_index(drop=True).ge(
        frame["exit_time"].iloc[:-1].reset_index(drop=True)
    ).all()


def test_preregistration_opens_no_outcome_and_replays(tmp_path: Path) -> None:
    output = tmp_path / "prereg.json"
    clock_output = tmp_path / "clock.csv.gz"
    docs = tmp_path / "prereg.md"
    first = prereg.run(output, clock_output, docs)
    second = prereg.run(output, clock_output, docs)
    assert first == second
    assert first["policy_id"] == "CITA-1"
    assert first["opened_outcome_windows"] == []
    assert first["source_contract"]["market_or_funding_rows_loaded"] == 0
    assert first["policy"]["mutable_parameters"] == []
    assert first["source_only_distributions"]["stage1_2020_2022"]["trades"] == 98
    assert first["source_only_distributions"]["stage2_2023"]["trades"] == 23


def test_frozen_preregistration_artifacts_replay() -> None:
    stored = json.loads(Path(prereg.DEFAULT_OUTPUT).read_text())
    replay = prereg.build_report(Path(prereg.DEFAULT_CLOCK))
    assert replay == stored
    assert prereg.sha256_file(Path(prereg.DEFAULT_CLOCK)) == stored["clock_sha256"]
