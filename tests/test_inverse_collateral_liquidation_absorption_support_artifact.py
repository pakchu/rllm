from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


CLOCKS = Path(
    "data/inverse_collateral_liquidation_absorption_clocks_2023_2024.csv.gz"
)
RESULT = Path(
    "results/inverse_collateral_liquidation_absorption_support_2026-07-19.json"
)
CLOCKS_SHA256 = "a55c23a7a0c296b98bb7a8958f713548c4313c0c682f1693c8f8be80b70dd053"
RESULT_SHA256 = "3cd925b2028c9b685c980fe4a4accfb0f58cbcad76c0498bb786771828354636"
MANIFEST_HASH = "22b98432da41ca69e01b3c37d2ca2903f959cb2f24a37262d7d248b3e16a0712"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_frozen_icla_support_artifact() -> None:
    assert _sha256(CLOCKS) == CLOCKS_SHA256
    assert _sha256(RESULT) == RESULT_SHA256

    result = json.loads(RESULT.read_text())
    protocol = result["protocol"]
    assert protocol["outcomes_opened"] is False
    assert protocol["execution_prices_opened"] is False
    assert protocol["funding_opened"] is False
    assert protocol["return_labels_constructed"] is False
    assert protocol["candidate_count"] == 1
    assert protocol["candidate_repair_after_outcomes"] is False
    assert protocol["later_stage_outcomes_opened"] is False
    assert result["support_passes"] is True
    assert result["clocks"]["sha256"] == CLOCKS_SHA256

    expected_support = {
        "train": (30, 15, 15, 0.30),
        "test": (111, 69, 42, 0.1981981981981982),
        "eval": (108, 74, 34, 0.25),
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
    assert overlap["intersection_entries"] == 4
    assert overlap["entry_jaccard"] == 0.007662835249042145
    assert overlap["entry_jaccard"] <= overlap["maximum_entry_jaccard_allowed"]
    assert overlap["passes"] is True

    core = dict(result)
    assert core.pop("manifest_hash") == MANIFEST_HASH
    assert _canonical_hash(core) == MANIFEST_HASH


def test_frozen_icla_clocks_respect_latency_hold_and_nonoverlap() -> None:
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

    assert len(clocks) == 249
    assert bool(clocks["candidate"].eq("ICLA-60").all())
    assert bool(clocks["direction"].isin([-1, 1]).all())
    assert clocks["entry_time"].is_monotonic_increasing
    assert not clocks["entry_time"].duplicated().any()
    assert clocks["entry_time"].iloc[0] == pd.Timestamp("2023-06-30 17:35:00")
    assert clocks["planned_exit_time"].iloc[-1] == pd.Timestamp(
        "2024-10-07 16:25:00"
    )

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
        .eq(pd.Timedelta(hours=1))
        .all()
    )
    assert bool(
        clocks["entry_time"]
        .iloc[1:]
        .reset_index(drop=True)
        .ge(clocks["planned_exit_time"].iloc[:-1].reset_index(drop=True))
        .all()
    )

    side_counts = clocks.groupby(["split", "direction"]).size().to_dict()
    assert side_counts == {
        ("eval", -1): 34,
        ("eval", 1): 74,
        ("test", -1): 42,
        ("test", 1): 69,
        ("train", -1): 15,
        ("train", 1): 15,
    }
