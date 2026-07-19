from __future__ import annotations

import csv
import gzip
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from training import preregister_ticket_gap_release as tgr


def _source_frame(periods: int = 12) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    symbols = sorted(tgr.SYMBOLS)
    for position, timestamp in enumerate(
        pd.date_range("2023-01-01T00:00:00", periods=periods, freq="1h")
    ):
        for index, symbol in enumerate(symbols):
            rows.append(
                {
                    "feature_available_time_utc": timestamp,
                    "symbol": symbol,
                    "taker_flow_fraction": (index - 2.5) / 20.0,
                    "mean_ticket_usdt": 100.0 + index * 10.0 + position,
                    "feature_valid": True,
                }
            )
    return pd.DataFrame(rows)


def _small_config() -> tgr.Config:
    return replace(
        tgr.Config(),
        ticket_window_hours=3,
        ticket_minimum_hours=2,
        threshold_window_hours=3,
        threshold_minimum_hours=2,
        top_flow_quantiles=(0.75,),
        ticket_gap_quantiles=(0.75,),
    )


def test_source_prefix_stops_before_boundary_without_parsing_future_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "source.csv.gz"
    first = cast(pd.Timestamp, pd.Timestamp("2023-01-01T01:00:00"))
    boundary = cast(pd.Timestamp, pd.Timestamp("2024-01-01T00:00:00"))
    rows: list[dict[str, str]] = []
    for symbol in sorted(tgr.SYMBOLS):
        row = {column: "" for column in tgr.SOURCE_SCHEMA}
        row.update(
            {
                "feature_available_time_utc": first.isoformat(),
                "symbol": symbol,
                "taker_flow_fraction": "0.1",
                "mean_ticket_usdt": "100",
                "feature_valid": "true",
            }
        )
        rows.append(row)
    poison = {column: "" for column in tgr.SOURCE_SCHEMA}
    poison.update(
        {
            "feature_available_time_utc": boundary.isoformat(),
            "symbol": sorted(tgr.SYMBOLS)[0],
            "taker_flow_fraction": "not-a-float",
            "mean_ticket_usdt": "not-a-float",
            "feature_valid": "true",
        }
    )
    rows.append(poison)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tgr.SOURCE_SCHEMA)
        writer.writeheader()
        writer.writerows(cast(Any, rows))
    monkeypatch.setattr(tgr, "SOURCE_PANEL", path)

    frame = tgr.load_source_prefix(end_exclusive=boundary)

    assert len(frame) == len(tgr.SYMBOLS)
    assert frame["feature_available_time_utc"].max() == first


def test_feature_thresholds_are_strictly_prior_and_future_invariant() -> None:
    cfg = _small_config()
    source = _source_frame()
    flow, ticket = tgr.source_matrices(source)
    base = tgr.base_feature_panel(flow, ticket, cfg)
    original = tgr.feature_panel(
        base,
        top_flow_quantile=0.75,
        ticket_gap_quantile=0.75,
        cfg=cfg,
    )
    changed_source = source.copy()
    last = changed_source["feature_available_time_utc"].max()
    changed_source.loc[
        changed_source["feature_available_time_utc"].eq(last), "mean_ticket_usdt"
    ] *= 1_000.0
    changed_flow, changed_ticket = tgr.source_matrices(changed_source)
    changed_base = tgr.base_feature_panel(changed_flow, changed_ticket, cfg)
    changed = tgr.feature_panel(
        changed_base,
        top_flow_quantile=0.75,
        ticket_gap_quantile=0.75,
        cfg=cfg,
    )

    prior_rows = original.index < last
    assert original.loc[prior_rows].equals(changed.loc[prior_rows])
    assert (
        original.loc[last, "top_flow_abs_threshold"]
        == changed.loc[last, "top_flow_abs_threshold"]
    )
    assert (
        original.loc[last, "ticket_gap_threshold"]
        == changed.loc[last, "ticket_gap_threshold"]
    )


def test_ticket_mad_uses_the_same_strictly_prior_window_median() -> None:
    cfg = replace(
        _small_config(),
        ticket_window_hours=3,
        ticket_minimum_hours=3,
    )
    index = pd.date_range("2023-01-01", periods=4, freq="1h")
    ticket = pd.DataFrame({"ETHUSDT": np.exp([1.0, 2.0, 10.0, 4.0])}, index=index)

    observed = tgr.robust_ticket_z(ticket, cfg).loc[index[-1], "ETHUSDT"]

    prior = np.array([1.0, 2.0, 10.0])
    center = np.median(prior)
    mad = np.median(np.abs(prior - center))
    expected = (4.0 - center) / (1.4826 * mad)
    assert observed == pytest.approx(expected)


def test_signal_requires_two_agreeing_leaders_and_quiet_bottom_crowd() -> None:
    cfg = _small_config()
    features = pd.DataFrame(
        {
            "top_ticket_flow": [0.10, 0.10, 0.10],
            "top_flow_abs_threshold": [0.08, 0.08, 0.08],
            "ticket_gap": [2.0, 2.0, 2.0],
            "ticket_gap_threshold": [1.5, 1.5, 1.5],
            "bottom_crowd_flow": [0.01, 0.01, 0.05],
            "bottom_quiet_threshold": [0.02, 0.02, 0.02],
            "top_agreement": [2, 1, 2],
            "side": [1.0, 1.0, 1.0],
            "top_symbol_1": ["ETHUSDT"] * 3,
            "top_symbol_2": ["SOLUSDT"] * 3,
        }
    )

    assert tgr.signal_state(features, cfg).tolist() == [True, False, False]


def test_support_selector_uses_no_outcome_field_and_maximizes_strength() -> None:
    checks = {"source_gate": True}
    cells = [
        {
            "top_flow_quantile": 0.85,
            "ticket_gap_quantile": 0.80,
            "checks": checks,
            "passes": True,
        },
        {
            "top_flow_quantile": 0.875,
            "ticket_gap_quantile": 0.75,
            "checks": checks,
            "passes": True,
        },
    ]

    selected = tgr.select_support_cell(cells)

    assert selected["top_flow_quantile"] == 0.875
    assert selected["ticket_gap_quantile"] == 0.75
    with pytest.raises(ValueError, match="forbidden outcome field"):
        tgr.select_support_cell([{**cells[0], "future_return": 1.0}])


def test_train_support_gate_requires_balance_halves_active_quarters_and_months() -> (
    None
):
    cfg = tgr.Config()
    passing = {
        "events": 56,
        "side_share_min": 25 / 56,
        "subwindows": {"train_h1": 20, "train_h2": 36},
        "quarter_counts": {"2023Q2": 19, "2023Q3": 19, "2023Q4": 17},
        "maximum_month_share": 10 / 56,
    }

    assert all(tgr.train_support_checks(passing, cfg).values())
    failing = {**passing, "side_share_min": 0.39}
    assert tgr.train_support_checks(failing, cfg)["train_side_balance"] is False


def test_source_only_run_selects_frozen_cell_and_never_opens_btc(
    tmp_path: Path,
) -> None:
    cfg = tgr.Config(
        result_output=str(tmp_path / "result.json"),
        clock_output=str(tmp_path / "clocks.csv.gz"),
        docs_output=str(tmp_path / "docs.md"),
    )

    report = tgr.run(cfg)

    assert report["selected"] == {
        "top_flow_quantile": 0.9,
        "ticket_gap_quantile": 0.7,
        "selection_rule_used_future_source_metrics": False,
        "selection_rule_used_outcomes": False,
    }
    assert report["clock_rows"] == 250
    assert {
        split: cast(dict[str, Any], report["support"])[split]["events"]
        for split in tgr.SPLITS
    } == {"train": 60, "test": 69, "eval": 79, "final": 42}
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["btc_execution_rows_loaded"] == 0
    assert report["funding_rows_loaded"] == 0
    assert report["future_source_values_opened_before_selection"] is False
    assert report["source_support_passed"] is True
    assert report["support_passed"] is False
    assert report["advance_to_evaluator_freeze"] is False
    assert report["disposition"] == "REJECT_SOURCE_INCIDENCE_NO_OUTCOME_OPEN"
    checks = cast(dict[str, bool], report["checks"])
    assert checks["test_minimum_trade_incidence"] is False
    assert sum(not value for value in checks.values()) == 1


def test_unregistered_signal_mutation_is_rejected() -> None:
    with pytest.raises(ValueError, match="configuration is frozen"):
        tgr.run(replace(tgr.Config(), hold_hours=24))
