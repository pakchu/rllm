from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from training import audit_cftc_dealer_absorption_client_consensus_source as audit


def row(dealer: float, asset: float, leveraged: float) -> audit.SourceRow:
    return audit.SourceRow(
        report_date="2020-01-01",
        available_time=pd.Timestamp("2020-01-09T00:00:00Z"),
        dealer_change=dealer,
        asset_manager_change=asset,
        leveraged_money_change=leveraged,
        official_zip_url="https://www.cftc.gov/example.zip",
        special_publication_override=False,
        source_complete=True,
    )


@pytest.mark.parametrize(
    ("dealer", "asset", "leveraged", "expected"),
    [
        (-1, 2, 3, 1),
        (1, -2, -3, -1),
        (1, 2, 3, 0),
        (-1, -2, -3, 0),
        (-1, 2, -3, 0),
        (0, 2, 3, 0),
        (-1, 0, 3, 0),
    ],
)
def test_daic_side_requires_same_nonzero_clients_and_opposite_dealer(
    dealer: float, asset: float, leveraged: float, expected: int
) -> None:
    assert audit.daic_side(row(dealer, asset, leveraged)) == expected


def test_cita_and_daic_conditions_are_mutually_exclusive() -> None:
    examples = [
        row(-1, 2, 3),
        row(1, -2, -3),
        row(1, 2, -3),
        row(-1, -2, 3),
    ]
    assert all(
        not (audit.daic_side(item) and audit.cita_side(item))
        for item in examples
    )


def test_nonoverlap_skips_compressed_release() -> None:
    first = audit._event(row(-1, 2, 3), 1)
    compressed_row = audit.SourceRow(
        **{
            **row(1, -2, -3).__dict__,
            "report_date": "2020-01-02",
            "available_time": pd.Timestamp("2020-01-12T00:00:00Z"),
        }
    )
    compressed = audit._event(compressed_row, -1)
    accepted = audit.nonoverlapping([first, compressed])
    assert accepted == [first]
    assert pd.Timestamp(first.entry_time) - pd.Timestamp(first.signal_time) == pd.Timedelta(minutes=5)
    assert pd.Timestamp(first.exit_time) - pd.Timestamp(first.entry_time) == pd.Timedelta(hours=168)


def test_real_audit_rejects_without_opening_outcomes() -> None:
    report = audit.build_report()
    assert report["disposition"] == "REJECT_BEFORE_OUTCOMES_NO_REPAIR"
    assert report["research_integrity"] == {
        "formal_preregistration_committed_before_exact_incidence": False,
        "exact_source_incidence_opened": True,
        "btc_outcomes_opened_for_daic": False,
        "performance_values_opened_for_daic": False,
        "conservative_action": "retire unchanged candidate before outcomes",
    }
    incidence = report["source_incidence"]
    assert incidence["daic_raw_events"] == 63
    assert incidence["daic_accepted_events"] == 61
    assert incidence["daic_suppressed_overlap_events"] == 2
    assert incidence["cita_raw_events"] == 183
    assert incidence["cita_accepted_events"] == 181
    assert incidence["daic_cita_raw_report_overlap"] == 0
    assert incidence["daic_cita_accepted_entry_jaccard"] == 0.0
    assert incidence["windows"]["train_2020_2022"] == {
        "events": 30,
        "longs": 15,
        "shorts": 15,
        "special_publication_overrides": 0,
        "max_single_month_share": 0.1,
    }
    assert incidence["windows"]["selection_2023"]["events"] == 25
    assert incidence["windows"]["2020"]["events"] == 7
    assert incidence["windows"]["2021"]["events"] == 6
    assert incidence["windows"]["2022"]["events"] == 16
    assert all(report["checks"].values())
    assert report["outcome_boundary"]["btc_market_rows_read"] == 0
    assert report["outcome_boundary"]["pnl_cagr_mdd_opened"] is False


def test_source_schema_contains_no_market_or_outcome_fields() -> None:
    forbidden = {"open", "high", "low", "close", "return", "pnl", "cagr", "mdd", "funding"}
    assert not forbidden.intersection(
        column.lower() for column in audit.ALLOWED_SOURCE_COLUMNS
    )


def test_repository_path_rejects_escape() -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        audit._repository_path("../outside.json")


def test_written_artifact_replays() -> None:
    stored = audit.DEFAULT_OUTPUT
    assert Path(stored).exists()
    assert audit.build_report() == __import__("json").loads(Path(stored).read_text())
