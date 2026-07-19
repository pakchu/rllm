from __future__ import annotations

import csv
import gzip
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from training import preregister_discordant_tail_absorption_consensus as dtac


def _small_config() -> dtac.Config:
    return replace(
        dtac.Config(),
        threshold_window_hours=4,
        threshold_minimum_sign_observations=1,
        flow_tail_quantiles=(0.5,),
        premium_tail_quantiles=(0.5,),
        consensus_counts=(2,),
    )


def test_flow_prefix_stops_before_boundary_without_parsing_future_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "flow.csv.gz"
    first = cast(pd.Timestamp, pd.Timestamp("2023-01-01T01:00:00"))
    boundary = cast(pd.Timestamp, pd.Timestamp("2024-01-01T00:00:00"))
    rows: list[dict[str, str]] = []
    for symbol in sorted(dtac.SYMBOLS):
        row = {column: "" for column in dtac.FLOW_SCHEMA}
        row.update(
            {
                "feature_available_time_utc": first.isoformat(),
                "symbol": symbol,
                "taker_flow_fraction": "0.1",
                "feature_valid": "true",
            }
        )
        rows.append(row)
    poison = {column: "" for column in dtac.FLOW_SCHEMA}
    poison.update(
        {
            "feature_available_time_utc": boundary.isoformat(),
            "symbol": sorted(dtac.SYMBOLS)[0],
            "taker_flow_fraction": "not-a-float",
            "feature_valid": "true",
        }
    )
    rows.append(poison)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dtac.FLOW_SCHEMA)
        writer.writeheader()
        writer.writerows(cast(Any, rows))
    monkeypatch.setattr(dtac, "FLOW_PANEL", path)

    frame = dtac.load_flow_prefix(end_exclusive=boundary)

    assert len(frame) == len(dtac.SYMBOLS)
    assert frame["feature_available_time_utc"].max() == first


def test_premium_prefix_stops_before_boundary_without_parsing_future_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    boundary = cast(pd.Timestamp, pd.Timestamp("2024-01-01T00:00:00"))
    first_timestamp = cast(pd.Timestamp, pd.Timestamp("2023-01-01T00:59:59.999"))
    boundary_timestamp = cast(pd.Timestamp, pd.Timestamp("2023-12-31T23:59:59.999"))
    first_close = int(first_timestamp.timestamp() * 1000)
    boundary_close = int(boundary_timestamp.timestamp() * 1000)
    monkeypatch.setattr(dtac, "PREMIUM_DIR", tmp_path)
    for symbol in sorted(dtac.SYMBOLS):
        path = dtac.premium_path(symbol)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=dtac.PREMIUM_SCHEMA)
            writer.writeheader()
            writer.writerow(
                {
                    "date": "2023-01-01 00:00:00",
                    "symbol": symbol,
                    "open": "0.1",
                    "high": "0.3",
                    "low": "0.0",
                    "close": "0.2",
                    "close_time": str(first_close),
                }
            )
            writer.writerow(
                {
                    "date": "2023-12-31 23:00:00",
                    "symbol": symbol,
                    "open": "not-a-float",
                    "high": "not-a-float",
                    "low": "not-a-float",
                    "close": "not-a-float",
                    "close_time": str(boundary_close),
                }
            )

    frame = dtac.load_premium_prefix(end_exclusive=boundary)

    assert frame.shape == (1, len(dtac.SYMBOLS))
    assert frame.index[0] == pd.Timestamp("2023-01-01T01:00:00")
    assert np.allclose(frame.iloc[0].to_numpy(float), 0.1)


def test_sign_tail_thresholds_are_separate_and_strictly_prior() -> None:
    cfg = _small_config()
    index = pd.date_range("2023-01-01", periods=5, freq="1h")
    frame = pd.DataFrame(
        {"ETHUSDT": [1.0, -2.0, 3.0, -4.0, 1_000.0]},
        index=index,
    )

    positive, negative = dtac._sign_tail_thresholds(frame, 0.5, cfg)

    assert positive.loc[index[-1], "ETHUSDT"] == pytest.approx(2.0)
    assert negative.loc[index[-1], "ETHUSDT"] == pytest.approx(3.0)
    changed = frame.copy()
    changed.loc[index[-1], "ETHUSDT"] = -1_000.0
    changed_positive, changed_negative = dtac._sign_tail_thresholds(changed, 0.5, cfg)
    assert changed_positive.loc[index[-1], "ETHUSDT"] == pytest.approx(2.0)
    assert changed_negative.loc[index[-1], "ETHUSDT"] == pytest.approx(3.0)


def test_feature_panel_emits_mirrored_two_symbol_absorption_consensus() -> None:
    cfg = _small_config()
    index = pd.date_range("2023-01-01", periods=5, freq="1h")
    symbols = sorted(dtac.SYMBOLS)
    flow = pd.DataFrame(index=index, columns=pd.Index(symbols), dtype=float)
    premium = pd.DataFrame(index=index, columns=pd.Index(symbols), dtype=float)
    for offset, symbol in enumerate(symbols):
        flow[symbol] = [0.1, -0.1, 0.2, -0.2, 0.01 + offset / 1_000]
        premium[symbol] = [-0.1, 0.1, -0.2, 0.2, 0.01 + offset / 1_000]
    for symbol in symbols[:2]:
        flow.loc[index[-1], symbol] = -1.0
        premium.loc[index[-1], symbol] = 1.0

    panel = dtac.feature_panel(
        flow,
        premium,
        flow_tail_quantile=0.5,
        premium_tail_quantile=0.5,
        consensus_count=2,
        cfg=cfg,
    )

    assert panel.loc[index[-1], "side"] == 1
    assert panel.loc[index[-1], "long_votes"] == 2
    assert panel.loc[index[-1], "short_votes"] == 0
    assert panel.loc[index[-1], "long_vote_symbols"] == ";".join(symbols[:2])
    assert panel.loc[index[-1], "mean_vote_flow"] == pytest.approx(-1.0)
    assert panel.loc[index[-1], "mean_vote_premium_impulse"] == pytest.approx(1.0)


def test_signal_onset_includes_direct_polarity_change() -> None:
    features = pd.DataFrame({"side": [0, 1, 1, -1, -1, 0, 1]})

    assert dtac.signal_onset(features).tolist() == [
        False,
        True,
        False,
        True,
        False,
        False,
        True,
    ]


def test_support_selector_uses_no_outcome_and_maximizes_frozen_strength() -> None:
    cells = [
        {
            "flow_tail_quantile": 0.80,
            "premium_tail_quantile": 0.60,
            "consensus_count": 2,
            "checks": {"source": True},
            "passes": True,
        },
        {
            "flow_tail_quantile": 0.75,
            "premium_tail_quantile": 0.80,
            "consensus_count": 2,
            "checks": {"source": True},
            "passes": True,
        },
    ]

    selected = dtac.select_support_cell(cells)

    assert selected["flow_tail_quantile"] == 0.80
    assert selected["premium_tail_quantile"] == 0.60
    with pytest.raises(ValueError, match="forbidden outcome field"):
        dtac.select_support_cell([{**cells[0], "future_return": 1.0}])


def test_source_only_run_freezes_balanced_feasible_novel_clock(tmp_path: Path) -> None:
    cfg = dtac.Config(
        result_output=str(tmp_path / "result.json"),
        clock_output=str(tmp_path / "clock.csv.gz"),
        docs_output=str(tmp_path / "docs.md"),
    )

    report = dtac.run(cfg)

    assert report["selected"] == {
        "flow_tail_quantile": 0.8,
        "premium_tail_quantile": 0.6,
        "consensus_count": 2,
        "selection_rule_used_future_source_metrics": False,
        "selection_rule_used_outcomes": False,
    }
    assert report["clock_rows"] == 695
    assert {
        split: (
            cast(dict[str, Any], report["support"])[split]["events"],
            cast(dict[str, Any], report["support"])[split]["long"],
            cast(dict[str, Any], report["support"])[split]["short"],
        )
        for split in dtac.SPLITS
    } == {
        "train": (143, 84, 59),
        "test": (190, 120, 70),
        "eval": (247, 148, 99),
        "final": (115, 54, 61),
    }
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["btc_execution_rows_loaded"] == 0
    assert report["btc_funding_rows_loaded"] == 0
    assert report["future_source_values_opened_before_selection"] is False
    assert report["support_passed"] is True
    assert report["advance_to_evaluator_freeze"] is True
    assert all(cast(dict[str, bool], report["checks"]).values())


def test_unregistered_signal_mutation_is_rejected() -> None:
    with pytest.raises(ValueError, match="configuration is frozen"):
        dtac.run(replace(dtac.Config(), hold_hours=12))
