from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd


CLOCKS = Path("data/eth_btc_liquidation_relay_clocks_2023_2024.csv.gz")
RESULT = Path("results/eth_btc_liquidation_relay_support_2026-07-19.json")
CLOCKS_SHA256 = "b4b35a0e9ae0cf26bf08df67b5c2fc832393c638c97f5b91a86894ee693b430e"
RESULT_SHA256 = "58726a37bd632cccdf7a5320ec81b2f190f6448e1fd8392295d251683ab8ca17"
MANIFEST_HASH = "c3884a55f16c2677e01689a972f90289ab134519c257ec385f561b40116fa0a7"
IMPLEMENTATION_SHA256 = (
    "fe40b29ab1cabc115304adce574033ab61a76d441ea730c379e63b1c7aa2d8cf"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_frozen_eblr_support_is_outcome_blind_and_passes() -> None:
    assert _sha256(CLOCKS) == CLOCKS_SHA256
    assert _sha256(RESULT) == RESULT_SHA256
    result = json.loads(RESULT.read_text())
    protocol = result["protocol"]
    assert protocol["candidate"] == "EBLR-60/30"
    assert protocol["candidate_count"] == 1
    assert protocol["outcomes_opened"] is False
    assert protocol["market_prices_opened"] is False
    assert protocol["funding_opened"] is False
    assert protocol["return_labels_constructed"] is False
    assert protocol["candidate_repair_after_outcomes"] is False
    assert protocol["later_stage_outcomes_opened"] is False
    assert result["support_passes"] is True
    assert result["support"]["passes"] is True
    assert result["clocks"]["sha256"] == CLOCKS_SHA256
    assert result["clocks"]["rows"] == 266
    assert result["implementation"] == {
        "path": "training/preregister_eth_btc_liquidation_relay.py",
        "sha256": IMPLEMENTATION_SHA256,
    }

    expected_support = {
        "train": (21, 8, 13, 0.38095238095238093),
        "test": (132, 49, 83, 0.20454545454545456),
        "eval": (113, 25, 88, 0.24778761061946902),
    }
    for split, (count, long_count, short_count, month_share) in (
        expected_support.items()
    ):
        support = result["support"][split]
        assert support["count"] == count
        assert support["long"] == long_count
        assert support["short"] == short_count
        assert support["maximum_month_share"] == month_share
        assert support["passes"] is True

    overlap = result["clock_overlap"]
    assert overlap["CLBR"]["intersection_entries"] == 0
    assert overlap["CLBR"]["entry_jaccard"] == 0.0
    assert overlap["ICLA"]["intersection_entries"] == 0
    assert overlap["ICLA"]["entry_jaccard"] == 0.0
    assert overlap["passes"] is True

    core = dict(result)
    assert core.pop("manifest_hash") == MANIFEST_HASH
    assert _canonical_hash(core) == MANIFEST_HASH


def test_frozen_eblr_clocks_are_causal_nonoverlapping_and_price_free() -> None:
    timestamp_columns = [
        "first_bar_open_time",
        "last_bar_open_time",
        "wave_completed_time",
        "feature_available_time",
        "entry_time",
        "planned_exit_time",
    ]
    clocks = pd.read_csv(
        CLOCKS,
        compression="gzip",
        parse_dates=timestamp_columns,
    )
    assert len(clocks) == 266
    assert bool(clocks["candidate"].eq("EBLR-60/30").all())
    assert bool(clocks["direction"].isin([-1, 1]).all())
    assert clocks["entry_time"].is_monotonic_increasing
    assert not clocks["entry_time"].duplicated().any()
    assert clocks["entry_time"].iloc[0] == pd.Timestamp("2023-07-26 20:50:00")
    assert clocks["planned_exit_time"].iloc[-1] == pd.Timestamp(
        "2024-10-09 22:30:00"
    )
    assert clocks["entry_time"].iloc[0] >= pd.Timestamp("2023-07-23")

    assert bool(
        clocks["last_bar_open_time"]
        .sub(clocks["first_bar_open_time"])
        .eq(pd.Timedelta(minutes=55))
        .all()
    )
    assert bool(
        clocks["wave_completed_time"]
        .sub(clocks["last_bar_open_time"])
        .eq(pd.Timedelta(minutes=5))
        .all()
    )
    assert bool(
        clocks["feature_available_time"]
        .sub(clocks["wave_completed_time"])
        .eq(pd.Timedelta(seconds=1))
        .all()
    )
    assert bool(clocks["entry_time"].gt(clocks["feature_available_time"]).all())
    assert bool(
        clocks["planned_exit_time"]
        .sub(clocks["entry_time"])
        .eq(pd.Timedelta(minutes=30))
        .all()
    )
    assert bool(
        clocks["entry_time"]
        .iloc[1:]
        .reset_index(drop=True)
        .ge(clocks["planned_exit_time"].iloc[:-1].reset_index(drop=True))
        .all()
    )
    assert bool(clocks["eth_event_count_60m"].ge(3).all())
    assert bool(clocks["eth_severity"].ge(1.0).all())
    assert bool(clocks["eth_wave_imbalance"].abs().ge(0.70).all())
    assert bool(clocks["btc_quiet_severity"].le(0.50).all())
    expected_direction = cast(pd.Series, clocks["eth_wave_imbalance"]).map(
        lambda value: 1 if value > 0.0 else -1
    )
    assert clocks["direction"].equals(expected_direction)
    assert set(clocks.columns).isdisjoint(
        {"open", "high", "low", "close", "return", "pnl", "entry_price", "exit_price"}
    )

    split_bounds = {
        "train": (pd.Timestamp("2023-06-25"), pd.Timestamp("2023-10-15")),
        "test": (pd.Timestamp("2023-10-15"), pd.Timestamp("2024-04-15")),
        "eval": (pd.Timestamp("2024-04-15"), pd.Timestamp("2024-10-15")),
    }
    for split, (start, end) in split_bounds.items():
        subset = clocks.loc[clocks["split"].eq(split)]
        assert bool(subset["entry_time"].ge(start).all())
        assert bool(subset["planned_exit_time"].lt(end).all())
