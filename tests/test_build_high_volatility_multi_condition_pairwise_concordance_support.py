from __future__ import annotations

import pandas as pd
import pytest

from training import build_high_volatility_multi_condition_pairwise_concordance_support as support
from training import preregister_high_volatility_multi_condition_pairwise_concordance as prereg


def _clock(component: str, rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    values = []
    for entry_raw, side, decision_raw, available_raw in rows:
        entry = pd.Timestamp(entry_raw)
        values.append(
            {
                "candidate": component,
                "control": "primary",
                "split": "train",
                "decision_time": pd.Timestamp(decision_raw),
                "feature_available_time": pd.Timestamp(available_raw),
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta("8h"),
                "side": side,
            }
        )
    return pd.DataFrame(values, columns=support.INPUT_COLUMNS)


def test_exact_intersection_requires_same_entry_and_strict_same_side() -> None:
    left, right = prereg.COMPONENT_ORDER[:2]
    clocks = {
        left: _clock(
            left,
            [
                ("2023-07-01T00:00:00Z", 1, "2023-06-30T23:00:00Z", "2023-06-30T23:30:00Z"),
                ("2023-07-01T08:00:00Z", -1, "2023-07-01T07:00:00Z", "2023-07-01T07:30:00Z"),
                ("2023-07-01T16:00:00Z", 1, "2023-07-01T15:00:00Z", "2023-07-01T15:30:00Z"),
            ],
        ),
        right: _clock(
            right,
            [
                ("2023-07-01T00:00:00Z", 1, "2023-06-30T23:15:00Z", "2023-06-30T23:45:00Z"),
                ("2023-07-01T08:00:00Z", 1, "2023-07-01T07:15:00Z", "2023-07-01T07:45:00Z"),
                ("2023-07-01T16:05:00Z", 1, "2023-07-01T15:15:00Z", "2023-07-01T15:45:00Z"),
            ],
        ),
    }
    result = support.intersect_pair(left, right, clocks)
    assert result["entry_time"].tolist() == [pd.Timestamp("2023-07-01T00:00:00Z")]
    assert result.loc[0, "side"] == 1
    assert result.loc[0, "exit_time"] == result.loc[0, "entry_time"] + pd.Timedelta("8h")
    assert result.loc[0, "decision_time"] == pd.Timestamp("2023-06-30T23:15:00Z")
    assert result.loc[0, "feature_available_time"] == pd.Timestamp("2023-06-30T23:45:00Z")
    assert result.loc[0, "left_component_id"] == left
    assert result.loc[0, "right_component_id"] == right


def test_half_open_reservation_is_chronological_and_allows_entry_at_exit() -> None:
    frame = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2023-07-01T04:00:00Z", "2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z"], utc=True
            ),
            "exit_time": pd.to_datetime(
                ["2023-07-01T12:00:00Z", "2023-07-01T08:00:00Z", "2023-07-01T16:00:00Z"], utc=True
            ),
        }
    )
    result = support.reserve_half_open(frame)
    assert result["entry_time"].tolist() == [
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2023-07-01T08:00:00Z"),
    ]


def test_support_stats_and_gates_cover_each_stage() -> None:
    entries = pd.to_datetime(
        ["2023-07-01T00:00:00Z", "2023-07-02T00:00:00Z", "2023-08-01T00:00:00Z"], utc=True
    )
    frame = pd.DataFrame({"split": ["train"] * 3, "entry_time": entries, "side": [1, -1, 1]})
    stats = support.support_stats(frame, "train")
    assert stats == {
        "events": 3, "longs": 2, "shorts": 1,
        "minority_side_share": pytest.approx(1 / 3),
        "max_month_share": pytest.approx(2 / 3),
    }
    all_stats = {name: support.support_stats(frame, name) for name in prereg.build()["stages"]}
    checks = support._support_checks(all_stats)
    assert set(checks) == {
        f"{stage}_{gate}"
        for stage in prereg.build()["stages"]
        for gate in ("minimum_events", "side_balance", "month_concentration")
    }


def test_frozen_contract_and_artifact_hashes() -> None:
    assert support.PREREG_SHA == "3cdd3edbedfda4e581bb95b9fac2db7309a7b54fe72c37ff5e5004ce8bba8d14"
    verified = support.verify_frozen_inputs()
    assert tuple(verified) == prereg.COMPONENT_ORDER
    assert all(value["gross9_passed"] is True for value in verified.values())
    assert all(
        artifact["verified"] is True
        for value in verified.values()
        for name, artifact in value.items()
        if name != "gross9_passed"
    )


def test_run_keeps_combination_outcomes_sealed_and_writes_deterministically(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(support, "CLOCK_DIR", tmp_path / "clocks")
    monkeypatch.setattr(support, "RESULT", tmp_path / "support.json")
    first = support.run()
    first_bytes = support.RESULT.read_bytes()
    first_hashes = {name: row["clock"]["sha256"] for name, row in first["pairs"].items()}
    second = support.run()
    assert support.RESULT.read_bytes() == first_bytes
    assert {name: row["clock"]["sha256"] for name, row in second["pairs"].items()} == first_hashes
    assert first["combination_outcomes_opened"] is False
    assert first["combination_postentry_returns_or_pnl_opened"] is False
    assert first["entry_exit_prices_opened"] is False
    assert first["funding_opened"] is False
    assert first["gross9_rows_opened"] is False
    assert first["advance_to_combination_economic_outcomes"] is False
    assert first["eligible_pairs_for_combination_gross9"] == [
        candidate for candidate, value in first["pairs"].items() if value["support_passed"]
    ]
