from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd

from training import preregister_coinm_liquidation_burst_release as prereg


CLOCKS = Path("data/coinm_liquidation_burst_release_clocks_2023_2024.csv.gz")
RESULT = Path("results/coinm_liquidation_burst_release_support_2026-07-19.json")
EXPECTED_CLOCK_SHA256 = (
    "df619a5ffc3b849d3c35fc7112641c33105ba76c81cbb7b8c7f3c975fd80bee0"
)
EXPECTED_RESULT_SHA256 = (
    "362c1b45fd52b278e2c7f3f06214812fd02b5a1a311aae716ad3c8621852ead3"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_support_artifacts_match_single_candidate_contract() -> None:
    assert _sha256(CLOCKS) == EXPECTED_CLOCK_SHA256
    assert _sha256(RESULT) == EXPECTED_RESULT_SHA256
    result = json.loads(RESULT.read_text())
    assert result["protocol"]["outcomes_opened"] is False
    assert result["protocol"]["candidate_count"] == 1
    assert result["protocol"]["market_prices_opened"] is False
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == prereg.canonical_hash(core)
    assert result["clocks"]["sha256"] == EXPECTED_CLOCK_SHA256
    assert result["clocks"]["rows"] == 277
    assert result["support"] == {
        "train": {
            "count": 40,
            "long": 29,
            "short": 11,
            "minimum_required": 40,
            "passes": True,
        },
        "test": {
            "count": 128,
            "long": 82,
            "short": 46,
            "minimum_required": 100,
            "passes": True,
        },
        "eval": {
            "count": 109,
            "long": 77,
            "short": 32,
            "minimum_required": 100,
            "passes": True,
        },
    }

    clocks = pd.read_csv(
        CLOCKS,
        compression="gzip",
        parse_dates=[
            "burst_time",
            "release_time",
            "feature_available_time",
            "entry_time",
            "planned_exit_time",
        ],
    )
    assert len(clocks) == 277
    assert bool(cast(pd.Series, clocks["candidate"]).eq("CLBR-24").all())
    assert clocks["entry_time"].is_unique
    assert clocks["entry_time"].is_monotonic_increasing
    assert bool(
        cast(pd.Series, clocks["release_time"].sub(clocks["burst_time"]))
        .eq(pd.Timedelta(minutes=5))
        .all()
    )
    assert bool(
        cast(
            pd.Series, clocks["feature_available_time"].sub(clocks["release_time"])
        )
        .eq(pd.Timedelta(minutes=5, seconds=1))
        .all()
    )
    assert bool(
        cast(pd.Series, clocks["entry_time"].sub(clocks["release_time"]))
        .eq(pd.Timedelta(minutes=10))
        .all()
    )
    assert bool(
        cast(pd.Series, clocks["planned_exit_time"].sub(clocks["entry_time"]))
        .eq(pd.Timedelta(hours=2))
        .all()
    )
    assert bool(
        cast(pd.Series, clocks["burst_total_liquidation_usd"])
        .ge(cast(pd.Series, clocks["burst_threshold_usd"]))
        .all()
    )
    assert bool(cast(pd.Series, clocks["burst_imbalance"]).abs().ge(0.8).all())
    assert bool(
        cast(pd.Series, clocks["release_total_liquidation_usd"])
        .le(cast(pd.Series, clocks["burst_total_liquidation_usd"]) * 0.25)
        .all()
    )
    assert bool(
        cast(pd.Series, clocks["release_counterflow_usd"])
        .le(cast(pd.Series, clocks["burst_total_liquidation_usd"]) * 0.10)
        .all()
    )
    long = cast(pd.Series, clocks["direction"]).gt(0)
    assert bool(
        cast(pd.Series, clocks.loc[long, "stop_price"])
        .lt(cast(pd.Series, clocks.loc[long, "stop_anchor"]))
        .all()
    )
    assert bool(
        cast(pd.Series, clocks.loc[~long, "stop_price"])
        .gt(cast(pd.Series, clocks.loc[~long, "stop_anchor"]))
        .all()
    )
    for split in prereg.SPLITS:
        subset = clocks[clocks["split"].eq(split)]
        previous_exit = cast(pd.Series, subset["planned_exit_time"]).shift(1)
        entry_time = cast(pd.Series, subset["entry_time"])
        assert bool(entry_time.iloc[1:].ge(previous_exit.iloc[1:]).all())
