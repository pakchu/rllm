from __future__ import annotations

import pandas as pd

from training import build_cross_collateral_inventory_pressure_absorption_support as s


def _feature() -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=4, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "feature_complete": [True] * 4,
            "oi_rotation": [1.0, 1.0, -1.0, -1.0],
            "taker_gap": [-1.0, 1.0, 1.0, -1.0],
            "oi_rotation_rank": [0.9] * 4,
            "taker_gap_rank": [0.8] * 4,
        },
        index=index,
    )


def test_primary_uses_only_opposed_quadrant() -> None:
    flags = s.build_flags(
        _feature(), {"policy": {"oi_rotation_rank_min": 0.8, "taker_gap_rank_min": 0.6}}
    )
    assert flags["primary"].tolist() == [True, False, True, False]


def test_support_build_is_outcome_blind_and_rejects_concentration(tmp_path) -> None:
    report = s.build(tmp_path / "support.json", tmp_path / "clock.csv.gz")
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["source"]["execution_market_rows_loaded"] == 0
    assert report["source"]["funding_rows_loaded"] == 0
    assert report["support_passed"] is False
    assert report["failed_checks"] == ["2023_month_concentration"]


def test_clock_is_causal_nonoverlapping_and_opposed(tmp_path) -> None:
    report = s.build(tmp_path / "support.json", tmp_path / "clock.csv.gz")
    clock = pd.read_csv(report["clock"]["path"])
    primary = clock.loc[clock["control"].eq("primary")].copy()
    for column in ("signal_time", "entry_time", "exit_time"):
        primary[column] = pd.to_datetime(primary[column], utc=True)
    assert (
        primary["entry_time"]
        .eq(primary["signal_time"] + pd.Timedelta(minutes=10))
        .all()
    )
    assert primary["exit_time"].eq(primary["entry_time"] + pd.Timedelta(hours=4)).all()
    assert (primary["oi_rotation"] * primary["taker_gap"] < 0).all()
    assert (
        primary["side"]
        .eq(primary["oi_rotation"].apply(lambda value: 1 if value > 0 else -1))
        .all()
    )
    assert (
        primary["entry_time"].iloc[1:].reset_index(drop=True)
        >= primary["exit_time"].iloc[:-1].reset_index(drop=True)
    ).all()
