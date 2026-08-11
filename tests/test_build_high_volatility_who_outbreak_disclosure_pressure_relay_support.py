import pandas as pd
import pytest

from training import build_high_volatility_who_outbreak_disclosure_pressure_relay_support as b


def _record() -> dict:
    return {"Id": "4dfe0e00-b342-41c4-ba5f-96e2a18639a4", "SystemSourceKey": "WHO-DON-source-key", "DonId": "2023-DON001", "UrlName": "sample", "Title": "Sample outbreak", "PublicationDate": "2023-01-01T10:00:00Z", "PublicationDateAndTime": "2023-01-01T10:00:00Z", "DateCreated": "2023-01-01T09:00:00Z", "LastModified": "2023-01-01T11:00:00Z"}


def test_normalize_record_fixes_identity_and_publication_time():
    row = b.normalize_record(_record())
    assert row["don_id"] == "2023-DON001"
    assert row["system_source_key"] == "WHO-DON-source-key"
    assert row["publication_at"] == "2023-01-01T10:00:00+00:00"


def test_publication_timestamp_disagreement_fails_closed():
    raw = _record() | {"PublicationDateAndTime": "2023-01-01T11:00:00Z"}
    with pytest.raises(RuntimeError, match="timestamps disagree"):
        b.normalize_record(raw)


def test_daily_pressure_uses_disjoint_twenty_eight_day_windows():
    records = []
    for day in ("2022-01-01", "2022-01-29", "2022-02-25"):
        records.append({"publication_at": pd.Timestamp(day, tz="UTC").isoformat()})
    frame = b.build_daily_panel(records)
    row = frame.loc[frame.source_day.eq(pd.Timestamp("2022-02-25T00:00:00Z"))].iloc[0]
    assert row.daily_count == 1
    assert row.pressure == 1
    assert row.result_side == 1
    assert row.source_candidate
    assert row.decision_time == pd.Timestamp("2022-02-26T12:00:00Z")


def test_strict_prior_midrank_excludes_current():
    rank = b.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0, 4.0]), lookback=3, minimum=3)
    assert rank.iloc[:3].isna().all()
    assert rank.iloc[3] == 1.0
