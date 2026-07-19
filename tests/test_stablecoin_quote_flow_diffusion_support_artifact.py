from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd

from training import build_stablecoin_quote_flow_diffusion_support as support


RESULT = Path("results/stablecoin_quote_flow_diffusion_support_2026-07-19.json")
CLOCKS = Path("data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz")
EXPECTED_RESULT_SHA256 = (
    "07230e9e579f1b16e07712a022e572026b4fbfa17070e998970b3fd8ee21d4b5"
)
EXPECTED_CLOCK_SHA256 = (
    "a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_clocks() -> pd.DataFrame:
    return pd.read_csv(
        CLOCKS,
        parse_dates=[
            "source_hour_start",
            "decision_time",
            "feature_available_time",
            "entry_time",
            "exit_time",
        ],
    )


def test_support_artifacts_are_hash_locked_and_outcome_blind() -> None:
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    assert _sha256(RESULT) == EXPECTED_RESULT_SHA256
    assert _sha256(CLOCKS) == EXPECTED_CLOCK_SHA256
    assert report["manifest_hash"] == support._canonical_hash(
        {key: value for key, value in report.items() if key != "manifest_hash"}
    )
    assert report["builder"] == str(support.BUILDER_PATH)
    assert report["builder_sha256"] == _sha256(support.BUILDER_PATH)
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["btc_execution_rows_loaded"] == 0
    assert report["funding_rows_loaded"] == 0
    assert report["support_passed"] is True
    assert report["failed_checks"] == []
    assert report["advance_to_train_outcomes"] is True


def test_primary_support_and_novelty_pass_the_frozen_gates() -> None:
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    assert {name: row["events"] for name, row in report["support"].items()} == {
        "train": 55,
        "2023_q3": 15,
        "2023_q4": 40,
        "test": 185,
        "2024_h1": 99,
        "2024_h2": 86,
        "eval": 217,
        "2025_h1": 94,
        "2025_h2": 123,
        "final": 93,
        "2026_q1": 40,
        "2026_q2": 53,
    }
    assert report["support"]["train"]["long"] == 32
    assert report["support"]["train"]["short"] == 23
    assert report["support"]["final"]["long"] == 50
    assert report["support"]["final"]["short"] == 43
    support_windows = {
        "train",
        "2023_q3",
        "2023_q4",
        "test",
        "2024_h1",
        "2024_h2",
        "eval",
        "2025_h1",
        "2025_h2",
        "final",
        "2026_q1",
        "2026_q2",
    }
    parent_windows = {"train", "test", "eval", "final"}
    comparators = {"OPDR-24", "PCBR-12", "PSR-30/6", "FQPR-3"}
    expected_checks = {
        *(f"{name}_events" for name in support_windows),
        *(f"{name}_side_balance" for name in parent_windows),
        *(f"{name}_month_concentration" for name in parent_windows),
        *(f"{name}_exact_jaccard" for name in comparators),
        *(f"{name}_near_containment" for name in comparators),
    }
    assert set(report["checks"]) == expected_checks
    assert all(report["checks"].values())
    for novelty in report["novelty"].values():
        assert novelty["exact_jaccard"] <= 0.10
        assert novelty["max_bidirectional_near_share"] <= 0.35


def test_clock_contract_is_causal_nonoverlapping_and_price_free() -> None:
    clocks = _load_clocks()
    assert tuple(clocks.columns) == support.CLOCK_COLUMNS
    assert len(clocks) == 8_914
    forbidden = ("price", "open", "high", "low", "close", "return", "pnl", "funding")
    assert not any(
        token in column.lower() for column in clocks.columns for token in forbidden
    )
    assert clocks[["control", "entry_time"]].duplicated().sum() == 0
    assert bool(cast(pd.Series, clocks["side"]).isin([-1, 1]).all())
    assert bool(
        cast(pd.Series, clocks["feature_available_time"])
        .eq(clocks["decision_time"])
        .all()
    )
    assert bool(
        clocks["decision_time"]
        .eq(clocks["source_hour_start"] + pd.Timedelta(hours=1))
        .all()
    )
    assert bool(
        clocks["exit_time"].eq(clocks["entry_time"] + pd.Timedelta(hours=6)).all()
    )

    assert set(clocks["split"]) == set(support.SPLITS)
    for split, (start, end) in support.SPLITS.items():
        window = cast(
            pd.DataFrame,
            clocks[cast(pd.Series, clocks["split"]).eq(split)],
        )
        assert bool(cast(pd.Series, window["source_hour_start"]).ge(start).all())
        assert bool(cast(pd.Series, window["entry_time"]).ge(start).all())
        assert bool(cast(pd.Series, window["exit_time"]).le(end).all())

    for control, untyped_group in clocks.groupby("control", sort=True):
        group = cast(pd.DataFrame, untyped_group).sort_values("entry_time")
        entries = cast(pd.Series, group["entry_time"]).reset_index(drop=True)
        exits = cast(pd.Series, group["exit_time"]).reset_index(drop=True)
        assert bool(entries.iloc[1:].ge(exits.iloc[:-1].to_numpy()).all()), control
        delay = group["entry_time"] - group["decision_time"]
        expected = pd.Timedelta(minutes=65 if control == "extra_latency_1h" else 5)
        assert bool(delay.eq(expected).all()), control


def test_same_clock_controls_preserve_primary_reservations() -> None:
    clocks = _load_clocks()
    primary = cast(
        pd.DataFrame,
        clocks[cast(pd.Series, clocks["control"]).eq("primary")],
    ).sort_values("entry_time")
    for control in ("direction_flip", "deterministic_random_side"):
        comparison = cast(
            pd.DataFrame,
            clocks[cast(pd.Series, clocks["control"]).eq(control)],
        ).sort_values("entry_time")
        pd.testing.assert_series_equal(
            primary["entry_time"].reset_index(drop=True),
            comparison["entry_time"].reset_index(drop=True),
            check_names=False,
        )
    flipped = cast(
        pd.DataFrame,
        clocks[cast(pd.Series, clocks["control"]).eq("direction_flip")],
    ).sort_values("entry_time")
    assert (
        primary["side"].reset_index(drop=True)
        == -flipped["side"].reset_index(drop=True)
    ).all()

    delayed = cast(
        pd.DataFrame,
        clocks[cast(pd.Series, clocks["control"]).eq("extra_latency_1h")],
    ).sort_values("entry_time")
    assert (
        delayed["entry_time"].reset_index(drop=True)
        == primary["entry_time"].reset_index(drop=True) + pd.Timedelta(hours=1)
    ).all()
