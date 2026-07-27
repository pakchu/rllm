from __future__ import annotations

import pandas as pd
import pytest

from training import bctp_transition_labels as bctp
from training import psim_semantic_transition_labels as semantic


def _row(index: int, decision: str, *, year: int = 2020) -> dict[str, object]:
    digest = f"{index + 1:064x}"
    return {
        "schema_version": semantic.PSIM_SOURCE_SCHEMA_VERSION,
        "row_index": index,
        "row_hash": digest,
        "decision_at": decision,
        "split": "train",
        "split_year": year,
        "schedule": "ARCHIVE_D90",
        "source_payload": {"events": [], "relation_edges": []},
        "source_payload_sha256": f"{index + 100:064x}",
    }


def test_transition_adapter_preserves_daily_identity_and_exact_schema() -> None:
    rows = [
        _row(0, "2020-01-01T12:05:00Z"),
        _row(1, "2020-01-02T12:05:00Z"),
    ]
    frame = semantic.transition_state_frame(rows)

    assert tuple(frame.columns) == bctp.SOURCE_COLUMNS
    assert frame["sequence_id"].tolist() == [
        rows[0]["row_hash"],
        rows[1]["row_hash"],
    ]
    assert frame["entry_time"].tolist() == [
        pd.Timestamp("2020-01-01T12:05:00Z"),
        pd.Timestamp("2020-01-02T12:05:00Z"),
    ]
    assert set(frame.loc[:, bctp.TOKEN_COLUMNS].stack()) == {
        semantic.UNUSED_TOKEN
    }


def test_source_validation_rejects_calendar_or_index_drift() -> None:
    rows = [
        _row(0, "2020-01-01T12:05:00Z"),
        _row(2, "2020-01-02T12:05:00Z"),
    ]
    with pytest.raises(ValueError, match="row index"):
        semantic.validate_source_rows(rows)

    rows[1]["row_index"] = 1
    rows[1]["decision_at"] = "2020-01-02T12:10:00Z"
    with pytest.raises(ValueError, match="decision clock"):
        semantic.validate_source_rows(rows)


def test_select_year_does_not_read_or_infer_outcomes() -> None:
    rows = [
        _row(0, "2020-12-31T12:05:00Z"),
        _row(1, "2021-01-01T12:05:00Z", year=2021),
    ]
    rows[1]["split"] = "train"

    selected = semantic.select_year(rows, 2020)

    assert len(selected) == 1
    assert selected[0]["decision_at"] == "2020-12-31T12:05:00Z"


def test_transition_adapter_accepts_contiguous_later_year_row_indices() -> None:
    rows = [
        _row(366, "2021-01-01T12:05:00Z", year=2021),
        _row(367, "2021-01-02T12:05:00Z", year=2021),
    ]
    rows[0]["row_hash"] = f"{367:064x}"
    rows[1]["row_hash"] = f"{368:064x}"

    frame = semantic.transition_state_frame(rows)

    assert len(frame) == 2
    assert frame["entry_time"].iloc[0] == pd.Timestamp(
        "2021-01-01T12:05:00Z"
    )
