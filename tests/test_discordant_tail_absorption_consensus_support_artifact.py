from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from training import preregister_discordant_tail_absorption_consensus as dtac


RESULT = Path("results/discordant_tail_absorption_consensus_support_2026-07-19.json")
CLOCK = Path("data/discordant_tail_absorption_consensus_clocks_2023_2026.csv.gz")
DOC = Path("docs/discordant-tail-absorption-consensus-preregistration-2026-07-19.md")
SOURCE_SHA256 = "092302ca733c9498a5472e55e6a9868fbaa0e7849be26900af4c48237248744b"
RESULT_SHA256 = "b3b34619443b92a458f0babece588881bd2079d91828d0af034a3a988777fe9e"
CLOCK_SHA256 = "c71685bb6285c07e90e328ffe2f69a11de37445f8113a47d7ffee6ef16eece79"
DOC_SHA256 = "3f8833f84429a343b5ddd6eb4a4fdc7192f87420c40d29eb5b9138c7f4fcd870"
MANIFEST_HASH = "978ab1603298bd02f6f622364d51f3f7dccd0fff4bc85f2f5e30b6d7c9b23cf3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(RESULT.read_text()))


def _all(value: Any) -> bool:
    return bool(cast(pd.Series, value).all())


def test_dtac_support_artifacts_are_hash_locked_and_outcome_blind() -> None:
    assert _sha256(dtac.PREREGISTRATION_SOURCE) == SOURCE_SHA256
    assert _sha256(RESULT) == RESULT_SHA256
    assert _sha256(CLOCK) == CLOCK_SHA256
    assert _sha256(DOC) == DOC_SHA256
    report = _report()
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == MANIFEST_HASH
    assert report["manifest_hash"] == dtac.canonical_hash(core)
    assert report["preregistration_source_sha256"] == SOURCE_SHA256
    assert report["clock_sha256"] == CLOCK_SHA256
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["btc_execution_rows_loaded"] == 0
    assert report["btc_funding_rows_loaded"] == 0
    assert report["future_source_values_opened_before_selection"] is False
    assert report["support_passed"] is True
    assert report["advance_to_evaluator_freeze"] is True
    assert report["disposition"] == "ADVANCE_TO_EVALUATOR_FREEZE"
    assert all(cast(dict[str, bool], report["checks"]).values())


def test_dtac_selected_strongest_supported_cell_and_balanced_incidence() -> None:
    report = _report()
    assert report["selected"] == {
        "flow_tail_quantile": 0.8,
        "premium_tail_quantile": 0.6,
        "consensus_count": 2,
        "selection_rule_used_future_source_metrics": False,
        "selection_rule_used_outcomes": False,
    }
    cells = cast(list[dict[str, Any]], report["tested_cells"])
    assert len(cells) == 75
    selected = dtac.select_support_cell(cells)
    assert selected["flow_tail_quantile"] == 0.8
    assert selected["premium_tail_quantile"] == 0.6
    assert selected["consensus_count"] == 2
    support = cast(dict[str, dict[str, Any]], report["support"])
    assert {
        split: (row["events"], row["long"], row["short"])
        for split, row in support.items()
    } == {
        "train": (143, 84, 59),
        "test": (190, 120, 70),
        "eval": (247, 148, 99),
        "final": (115, 54, 61),
    }
    assert support["train"]["subwindows"] == {"train_h1": 71, "train_h2": 72}
    assert support["train"]["quarter_counts"] == {
        "2023Q1": 17,
        "2023Q2": 54,
        "2023Q3": 41,
        "2023Q4": 31,
    }
    assert min(row["side_share_min"] for row in support.values()) >= 0.35


def test_dtac_clock_is_causal_nonoverlapping_and_mechanism_valid() -> None:
    clock = pd.read_csv(
        CLOCK,
        keep_default_na=False,
        parse_dates=[
            "source_hour_open_utc",
            "decision_time",
            "feature_available_time",
            "entry_time",
            "exit_time",
        ],
    )
    assert tuple(clock.columns) == dtac.EVENT_COLUMNS
    assert len(clock) == 695
    assert clock["decision_time"].min() == pd.Timestamp("2023-02-01 09:00:00")
    assert _all(clock["feature_available_time"].eq(clock["decision_time"]))
    assert _all(
        clock["decision_time"].eq(clock["source_hour_open_utc"] + pd.Timedelta(hours=1))
    )
    assert _all(
        clock["entry_time"].eq(clock["decision_time"] + pd.Timedelta(minutes=5))
    )
    assert _all(clock["exit_time"].eq(clock["entry_time"] + pd.Timedelta(hours=8)))
    assert _all(clock["consensus_count"].eq(2))
    assert _all(clock["flow_tail_quantile"].eq(0.8))
    assert _all(clock["premium_tail_quantile"].eq(0.6))

    long_clock = cast(pd.DataFrame, clock.loc[clock["side"].eq(1)])
    short_clock = cast(pd.DataFrame, clock.loc[clock["side"].eq(-1)])
    assert _all(long_clock["long_votes"].ge(2))
    assert _all(long_clock["short_votes"].le(1))
    assert _all(long_clock["long_vote_symbols"].ne(""))
    assert _all(long_clock["mean_vote_flow"].lt(0.0))
    assert _all(long_clock["mean_vote_premium_impulse"].gt(0.0))
    assert _all(short_clock["short_votes"].ge(2))
    assert _all(short_clock["long_votes"].le(1))
    assert _all(short_clock["short_vote_symbols"].ne(""))
    assert _all(short_clock["mean_vote_flow"].gt(0.0))
    assert _all(short_clock["mean_vote_premium_impulse"].lt(0.0))

    for split, (start, end) in dtac.SPLITS.items():
        selected = cast(pd.DataFrame, clock.loc[clock["split"].eq(split)]).sort_values(
            "entry_time"
        )
        assert _all(selected["entry_time"].ge(start))
        assert _all(selected["exit_time"].le(end))
        entries = cast(pd.Series, selected["entry_time"]).reset_index(drop=True)
        exits = cast(pd.Series, selected["exit_time"]).reset_index(drop=True)
        assert _all(entries.iloc[1:].ge(exits.iloc[:-1].to_numpy()))


def test_dtac_clock_passes_every_frozen_novelty_gate() -> None:
    report = _report()
    novelty = cast(dict[str, dict[str, float]], report["novelty"])
    assert set(novelty) == set(dtac.COMPARATORS)
    for values in novelty.values():
        assert values["exact_jaccard"] <= 0.05
        assert values["max_bidirectional_near_share"] <= 0.35
    assert (
        max(values["max_bidirectional_near_share"] for values in novelty.values())
        == novelty["TGR-12"]["max_bidirectional_near_share"]
    )
    assert novelty["TGR-12"]["max_bidirectional_near_share"] < 0.145
