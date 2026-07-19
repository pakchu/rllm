from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import preregister_ticket_gap_release as tgr


RESULT = Path("results/ticket_gap_release_support_2026-07-19.json")
CLOCK = Path("data/ticket_gap_release_clocks_2023_2026.csv.gz")
DOC = Path("docs/ticket-gap-release-preregistration-2026-07-19.md")
SOURCE_SHA256 = "cc15eb3461d7520c1d759be8dfc69ebe5dcf56b128a97d38e277ffe37835138e"
RESULT_SHA256 = "00e91c0bb8d1ae658eb7192ceb35db10aa800e26837d6c1a1de8e7731870bcba"
CLOCK_SHA256 = "166c6e214f43d19eb0c33adbc7deed5fd81eeee222de6a737daf061fd2c3ffc2"
DOC_SHA256 = "bd1b13ffac38674331cc7afb7418095840065ca03fa512a2df2840afa2402b1f"
MANIFEST_HASH = "acbbf299b58837fe8bc70b9042f63355a62894f6c0e278cf75a68184c429c43f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(RESULT.read_text()))


def _all(value: Any) -> bool:
    return bool(cast(pd.Series, value).all())


def test_tgr_support_artifacts_are_hash_locked_and_outcome_blind() -> None:
    assert _sha256(tgr.PREREGISTRATION_SOURCE) == SOURCE_SHA256
    assert _sha256(RESULT) == RESULT_SHA256
    assert _sha256(CLOCK) == CLOCK_SHA256
    assert _sha256(DOC) == DOC_SHA256
    report = _report()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == MANIFEST_HASH
    assert report["manifest_hash"] == tgr.canonical_hash(core)
    assert report["preregistration_source_sha256"] == SOURCE_SHA256
    assert report["clock_sha256"] == CLOCK_SHA256
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["btc_execution_rows_loaded"] == 0
    assert report["funding_rows_loaded"] == 0
    assert report["future_source_values_opened_before_selection"] is False
    assert report["source_support_passed"] is True
    assert report["support_passed"] is False
    assert report["advance_to_evaluator_freeze"] is False
    assert report["disposition"] == "REJECT_SOURCE_INCIDENCE_NO_OUTCOME_OPEN"
    assert report["checks"]["test_minimum_trade_incidence"] is False


def test_tgr_selected_strongest_source_supported_cell_and_incidence() -> None:
    report = _report()
    assert report["selected"] == {
        "top_flow_quantile": 0.9,
        "ticket_gap_quantile": 0.7,
        "selection_rule_used_future_source_metrics": False,
        "selection_rule_used_outcomes": False,
    }
    cells = cast(list[dict[str, Any]], report["tested_cells"])
    assert len(cells) == 25
    selected = tgr.select_support_cell(cells)
    assert selected["top_flow_quantile"] == 0.9
    assert selected["ticket_gap_quantile"] == 0.7
    support = cast(dict[str, dict[str, Any]], report["support"])
    assert {
        split: (row["events"], row["long"], row["short"])
        for split, row in support.items()
    } == {
        "train": (60, 33, 27),
        "test": (69, 29, 40),
        "eval": (79, 43, 36),
        "final": (42, 21, 21),
    }
    assert support["train"]["subwindows"] == {"train_h1": 24, "train_h2": 36}
    assert support["train"]["quarter_counts"] == {
        "2023Q1": 6,
        "2023Q2": 18,
        "2023Q3": 21,
        "2023Q4": 15,
    }
    assert support["train"]["maximum_month_share"] == 10 / 60


def test_tgr_clock_is_causal_nonoverlapping_and_mechanism_valid() -> None:
    clock = pd.read_csv(
        CLOCK,
        parse_dates=[
            "source_hour_open_utc",
            "decision_time",
            "feature_available_time",
            "entry_time",
            "exit_time",
        ],
    )
    assert tuple(clock.columns) == tgr.EVENT_COLUMNS
    assert len(clock) == 250
    assert clock["decision_time"].min() == pd.Timestamp("2023-03-01 05:00:00")
    assert _all(clock["top_symbol_1"].isin(tgr.SYMBOLS))
    assert _all(clock["top_symbol_2"].isin(tgr.SYMBOLS))
    assert _all(clock["top_symbol_1"].ne(clock["top_symbol_2"]))
    assert _all(clock["feature_available_time"].eq(clock["decision_time"]))
    assert _all(
        clock["decision_time"].eq(clock["source_hour_open_utc"] + pd.Timedelta(hours=1))
    )
    assert _all(
        clock["entry_time"].eq(clock["decision_time"] + pd.Timedelta(minutes=5))
    )
    assert _all(clock["exit_time"].eq(clock["entry_time"] + pd.Timedelta(hours=12)))
    expected_side = np.where(clock["top_ticket_flow"].gt(0.0), 1, -1)
    assert _all(clock["side"].eq(expected_side))
    assert _all(clock["top_ticket_flow"].abs().ge(clock["top_flow_abs_threshold"]))
    assert _all(clock["bottom_crowd_flow"].abs().le(clock["bottom_quiet_threshold"]))
    assert _all(clock["ticket_gap"].ge(clock["ticket_gap_threshold"]))

    for split, (start, end) in tgr.SPLITS.items():
        selected = cast(pd.DataFrame, clock.loc[clock["split"].eq(split)]).sort_values(
            "entry_time"
        )
        assert _all(selected["entry_time"].ge(start))
        assert _all(selected["exit_time"].le(end))
        entries = cast(pd.Series, selected["entry_time"]).reset_index(drop=True)
        exits = cast(pd.Series, selected["exit_time"]).reset_index(drop=True)
        assert _all(entries.iloc[1:].ge(exits.iloc[:-1].to_numpy()))


def test_tgr_clock_passes_every_frozen_novelty_gate() -> None:
    report = _report()
    novelty = cast(dict[str, dict[str, float]], report["novelty"])
    assert set(novelty) == set(tgr.COMPARATORS)
    for values in novelty.values():
        assert values["exact_jaccard"] <= 0.05
        assert values["max_bidirectional_near_share"] <= 0.35
    assert novelty["SQFD-6"]["max_bidirectional_near_share"] == (64 / 210)
