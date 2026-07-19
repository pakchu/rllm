from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from training import preregister_flow_centrality_incubation_relay as fcir


RESULT = Path("results/flow_centrality_incubation_relay_support_2026-07-19.json")
CLOCK = Path("data/flow_centrality_incubation_relay_clocks_2023_2026.csv.gz")
DOCS = Path(
    "docs/flow-centrality-incubation-relay-preregistration-2026-07-19.md"
)
RESULT_SHA256 = "2ecd30c6a4f8678207053522aabe7ef6bfbc24e3f5d29b4e52743c218e4d2e89"
CLOCK_SHA256 = "d4bb6245f0bac34885e780e35ff1edb9b5cf2114dc3c13088ec19613ad8056ea"
DOCS_SHA256 = "fa657716d0bc5c3be4dfaf975d97e80459cce8fbf5882d742e317ccc7d5067ed"
SOURCE_SHA256 = "2c0af80ed23c9de9f875d6d53bb93c7cc6bd31fae710b514204c80e9df68b1d3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict[str, object]:
    return json.loads(RESULT.read_text())


def test_fcir_preregistration_is_hash_locked_and_outcome_blind() -> None:
    assert _sha256(RESULT) == RESULT_SHA256
    assert _sha256(CLOCK) == CLOCK_SHA256
    assert _sha256(DOCS) == DOCS_SHA256
    assert _sha256(fcir.PREREGISTRATION_SOURCE) == SOURCE_SHA256
    report = _report()
    body = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == fcir.canonical_hash(body)
    assert report["preregistration_source_sha256"] == SOURCE_SHA256
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["btc_execution_rows_loaded"] == 0
    assert report["funding_rows_loaded"] == 0
    assert report["future_source_values_opened_before_selection"] is False
    assert report["support_passed"] is True
    assert report["advance_to_evaluator_freeze"] is True
    protocol = cast(dict[str, object], report["protocol"])
    execution = cast(dict[str, object], protocol["eventual_execution"])
    assert execution["funding"] == (
        "conservative exact-time accounting: interior events symmetric; "
        "exact entry/exit credits dropped and debits retained; every settlement "
        "mark visited"
    )


def test_fcir_selection_uses_only_train_support_and_is_the_strongest_pass() -> None:
    report = _report()
    selected = report["selected"]
    assert selected == {
        "central_flow_quantile": 0.75,
        "minimum_effective_names": 3.0,
        "selection_rule_used_future_source_metrics": False,
        "selection_rule_used_outcomes": False,
    }
    cells = cast(list[dict[str, object]], report["tested_cells"])
    assert len(cells) == 12
    assert fcir.select_support_cell(cells)["central_flow_quantile"] == 0.75
    assert fcir.select_support_cell(cells)["minimum_effective_names"] == 3.0
    assert all(cast(dict[str, bool], report["checks"]).values())
    config = cast(dict[str, object], report["config"])
    assert config["threshold_window_hours"] == 2160
    assert config["threshold_minimum_hours"] == 720
    assert report["support"] == {
        "train": {
            "events": 62,
            "long": 26,
            "short": 36,
            "side_share_min": 26 / 62,
            "maximum_month_share": 12 / 62,
            "month_counts": {
                "2023-03": 12,
                "2023-04": 9,
                "2023-05": 11,
                "2023-06": 4,
                "2023-07": 2,
                "2023-08": 5,
                "2023-09": 10,
                "2023-10": 1,
                "2023-11": 1,
                "2023-12": 7,
            },
            "quarter_counts": {
                "2023Q1": 12,
                "2023Q2": 24,
                "2023Q3": 17,
                "2023Q4": 9,
            },
            "subwindows": {"train_h1": 36, "train_h2": 26},
        },
        **{
            split: cast(dict[str, object], report["support"])[split]
            for split in ("test", "eval", "final")
        },
    }
    support = cast(dict[str, object], report["support"])
    headline = {
        split: {
            key: cast(dict[str, object], support[split])[key]
            for key in ("events", "long", "short")
        }
        for split in fcir.SPLITS
    }
    assert headline == {
        "train": {"events": 62, "long": 26, "short": 36},
        "test": {"events": 90, "long": 46, "short": 44},
        "eval": {"events": 61, "long": 32, "short": 29},
        "final": {"events": 34, "long": 16, "short": 18},
    }


def test_fcir_clock_is_causal_nonoverlapping_and_mechanism_valid() -> None:
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
    assert tuple(clock.columns) == fcir.EVENT_COLUMNS
    assert len(clock) == 247
    assert clock["decision_time"].min() == pd.Timestamp("2023-03-01 04:00:00")
    assert not clock[["split", "entry_time"]].duplicated().any()
    assert bool(clock["feature_available_time"].eq(clock["decision_time"]).all())
    assert bool(
        clock["decision_time"]
        .eq(clock["source_hour_open_utc"] + pd.Timedelta(hours=1))
        .all()
    )
    assert bool(
        clock["entry_time"].eq(clock["decision_time"] + pd.Timedelta(minutes=5)).all()
    )
    assert bool(
        clock["exit_time"].eq(clock["entry_time"] + pd.Timedelta(hours=12)).all()
    )
    assert bool(
        cast(pd.Series, clock["side"])
        .eq(np.where(clock["central_flow"].gt(0.0), 1, -1))
        .all()
    )
    assert bool(clock["central_flow"].abs().ge(clock["central_abs_threshold"]).all())
    assert bool(
        clock["equal_weight_flow"].abs().le(clock["crowd_quiet_threshold"]).all()
    )
    assert bool(clock["effective_names"].ge(3.0).all())
    weights = cast(
        pd.DataFrame,
        clock[[f"weight_{symbol.lower()}" for symbol in fcir.SYMBOLS]],
    )
    assert bool((weights.to_numpy(float) >= 0.0).all())
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-10)

    for split, (start, end) in fcir.SPLITS.items():
        group = cast(pd.DataFrame, clock.loc[clock["split"].eq(split)]).sort_values(
            "entry_time"
        )
        assert bool(group["entry_time"].ge(start).all())
        assert bool(group["exit_time"].le(end).all())
        entries = cast(pd.Series, group["entry_time"]).reset_index(drop=True)
        exits = cast(pd.Series, group["exit_time"]).reset_index(drop=True)
        assert bool(entries.iloc[1:].ge(exits.iloc[:-1].to_numpy()).all())


def test_fcir_clock_overlap_passes_every_frozen_novelty_gate() -> None:
    report = _report()
    novelty = cast(dict[str, dict[str, float]], report["novelty"])
    assert set(novelty) == set(fcir.COMPARATORS)
    for values in novelty.values():
        assert values["exact_jaccard"] <= 0.05
        assert values["max_bidirectional_near_share"] <= 0.35
    assert novelty["SQFD-6"]["max_bidirectional_near_share"] == 1 / 3
