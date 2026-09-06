import pandas as pd

from training import build_high_volatility_doi_research_attention_relay_support as b


def item(doi: str, title: str, created: str, deposited: str, work_type: str = "journal-article") -> dict:
    return {"doi": doi, "titles": [title], "type": work_type, "created": created, "deposited": deposited}


def test_normalize_item_freezes_selected_fields():
    value = b.normalize_item(
        {
            "DOI": "10.1/ABC",
            "title": ["  Bitcoin   Attention  "],
            "type": "journal-article",
            "created": {"date-time": "2025-01-01T01:02:03Z"},
            "deposited": {"date-time": "2025-01-01T01:02:09Z"},
        }
    )
    assert value == {
        "doi": "10.1/abc",
        "titles": ["Bitcoin Attention"],
        "type": "journal-article",
        "created": "2025-01-01T01:02:03Z",
        "deposited": "2025-01-01T01:02:09Z",
    }


def test_daily_panel_uses_same_day_deposits_and_seven_day_change():
    documents = [
        item("10.1/a", "Bitcoin study", "2022-01-08T01:00:00Z", "2022-01-08T02:00:00Z"),
        item("10.1/b", "Cryptocurrency study", "2022-01-15T01:00:00Z", "2022-01-15T02:00:00Z"),
        item("10.1/c", "Cryptoasset study", "2022-01-15T03:00:00Z", "2022-01-15T04:00:00Z"),
        item("10.1/d", "Bitcoin revised later", "2022-01-15T05:00:00Z", "2022-01-16T05:00:00Z"),
        item("10.1/e", "Unrelated title", "2022-01-15T05:00:00Z", "2022-01-15T05:00:01Z"),
    ]
    panel = b.build_daily_panel(documents)
    row = panel.loc[panel.source_day.eq(pd.Timestamp("2022-01-15T00:00:00Z"))].iloc[0]
    assert row.daily_count == 2
    assert row.attention_change == 1
    assert row.result_side == 1
    assert row.decision_time == pd.Timestamp("2022-01-17T12:00:00Z")


def test_strict_prior_midrank_uses_270_180_contract():
    values = pd.Series(range(181), dtype=float)
    ranks = b.strict_prior_midrank(values)
    assert ranks.iloc[:180].isna().all()
    assert ranks.iloc[180] == 1.0


def test_crossref_filter_uses_api_accepted_utc_shape():
    source = open(b.__file__).read()
    assert "from-created-date:2022-01-01T00:00:00,until-created-date:2026-07-30T23:59:59" in source
    assert "from-created-date:2022-01-01T00:00:00Z" not in source


def test_primary_clock_holds_24_hours():
    features = pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2023-07-03T12:00:00Z")],
            "result_side": [-1],
            "raw_day_over_day_side": [1],
            "daily_count": [2],
            "attention_change": [-1],
            "btc_realized_variation": [0.02],
            "btc_variation_rank": [0.8],
        }
    )
    clock = b.build_clock(features)
    assert len(clock) == 1
    assert clock.iloc[0].side == -1
    assert clock.iloc[0].exit_time - clock.iloc[0].entry_time == pd.Timedelta(hours=24)
