from __future__ import annotations

import ast
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import training.export_cchr_pdlh_pure_clocks as pdlh

UTC = timezone.utc


def _dates(start: str, periods: int) -> pd.Series:
    return pd.Series(pd.date_range(start, periods=periods, freq="5min", tz="UTC"))


def _states(values: list[float], other: float | list[float] = 0.0) -> pd.DataFrame:
    other_values = other if isinstance(other, list) else [other] * len(values)
    return pd.DataFrame(
        {
            "top_position_minus_global": values,
            "top_account_minus_global": other_values,
        }
    )


def test_load_causal_inputs_is_hash_bound_to_exact_allowlists() -> None:
    assert pdlh.MARKET_COLUMNS == ("date",)
    assert pdlh.METRICS_COLUMNS == (
        "create_time",
        "sum_toptrader_long_short_ratio",
        "count_toptrader_long_short_ratio",
        "count_long_short_ratio",
    )
    assert (
        pdlh.MARKET_SHA256
        == "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
    )
    assert (
        pdlh.METRICS_SHA256
        == "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106"
    )


def test_module_has_no_cli_and_no_legacy_or_outcome_imports() -> None:
    tree = ast.parse(Path(pdlh.__file__).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "training.search_positioning_lifecycle_hazard_alpha" not in imported
    assert all("result" not in name and "evaluator" not in name for name in imported)
    assert not any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in ast.walk(tree)
    )


def test_attach_delayed_metrics_uses_one_complete_bar_and_rejects_stale() -> None:
    market = pd.DataFrame({"date": _dates("2021-01-01T00:00:00Z", 4)})
    metrics = pd.DataFrame(
        {
            "create_time": _dates("2021-01-01T00:00:00Z", 4),
            "sum_toptrader_long_short_ratio": [10.0, 11.0, 12.0, 13.0],
            "count_toptrader_long_short_ratio": [20.0, 21.0, 22.0, 23.0],
            "count_long_short_ratio": [30.0, 31.0, 32.0, 33.0],
        }
    )
    joined = pdlh.attach_delayed_metrics(market, metrics)
    assert pd.isna(joined.loc[0, "positioning_source_time"])
    assert joined.loc[2, "positioning_source_time"] == pd.Timestamp(
        "2021-01-01T00:05:00Z"
    )
    assert joined.loc[2, "sum_toptrader_long_short_ratio"] == 11.0
    assert bool(joined.loc[2, "positioning_available"])

    stale_market = pd.DataFrame({"date": _dates("2021-01-01T00:00:00Z", 5)})
    stale_metrics = metrics.iloc[:1]
    bounded = pdlh.attach_delayed_metrics(stale_market, stale_metrics)
    available = bounded["positioning_available"]
    ages = (
        bounded.loc[available, "date"]
        - bounded.loc[available, "positioning_source_time"]
    )
    assert ages.max() == pd.Timedelta("10min")
    assert available.tolist() == [False, True, True, False, False]


def test_prior_z_excludes_current_row() -> None:
    values = pd.Series([0.0, 2.0, 4.0, 100.0])
    z = pdlh.prior_z(values, window=3, minimum=3)
    assert np.isnan(z.iloc[2])
    assert z.iloc[3] == pytest.approx((100.0 - 2.0) / np.std([0.0, 2.0, 4.0]))


def test_lifecycle_resets_on_invalid_and_2022_quarantine() -> None:
    dates = _dates("2021-12-31T12:00:00Z", 150)
    valid = pdlh.positioning_valid_mask(dates, [True] * len(dates))
    assert not valid[144]  # 2022-01-01T00:00Z is quarantined.
    signals = pdlh.lifecycle_signals_for_state(
        dates,
        [2.0] * 144 + [0.0] + [2.0] * 5,
        valid,
        state="top_position_minus_global",
        min_age=144,
        trigger="contraction",
    )
    assert signals == []

    dates_2023 = _dates("2023-01-01T00:00:00Z", 146)
    valid_with_gap = np.array([True] * 100 + [False] + [True] * 45)
    signals = pdlh.lifecycle_signals_for_state(
        dates_2023,
        [2.0] * 144 + [0.0, 0.0],
        valid_with_gap,
        state="top_position_minus_global",
        min_age=144,
        trigger="contraction",
    )
    assert signals == []


def test_exact_member_map_and_ids_match_cchr_prereg_subset() -> None:
    members = pdlh.pdlh_candidate_map()
    assert len(members) == 16
    assert tuple(members) == pdlh.expected_candidate_ids()
    candidate_id = "pdlh:top_position_minus_global:age=144:trigger=contraction:hold=72"
    assert members[candidate_id] == {
        "family": "pdlh",
        "parameters": {
            "disagreement": "top_position_minus_global",
            "min_age": 144,
            "trigger": "contraction",
        },
        "hold_bars": 72,
        "component_weight": None,
    }
    assert (
        "pdlh:top_account_minus_global:age=432:trigger=zero_cross:hold=216" in members
    )


def test_next_bar_timing_hold_and_side_for_first_resolution() -> None:
    dates = _dates("2023-01-01T00:00:00Z", 146)
    states = _states([2.0] * 144 + [1.0, 0.0])
    frame = pdlh.build_pdlh_clock_from_states(dates, states, [True] * len(dates))
    row = frame.loc[
        frame["candidate_id"]
        == "pdlh:top_position_minus_global:age=144:trigger=contraction:hold=72"
    ].iloc[0]
    assert row["decision_time"] == "2023-01-01T12:05:00Z"
    assert row["entry_time"] == "2023-01-01T12:05:00Z"
    assert row["exit_time"] == "2023-01-01T18:05:00Z"
    assert row["side"] == -1


def test_split_containment_uses_episode_origin_not_just_execution_time() -> None:
    dates = _dates("2022-12-31T12:00:00Z", 146)
    states = _states([2.0] * 144 + [1.0, 0.0])
    frame = pdlh.build_pdlh_clock_from_states(dates, states, [True] * len(dates))
    assert frame.empty


def test_nonoverlap_is_independent_per_candidate_id() -> None:
    dates = _dates("2023-01-01T00:00:00Z", 370)
    values = [2.0] * 144 + [1.0, 0.0] + [0.0] * 10 + [2.0] * 144 + [1.0, 0.0]
    values.extend([0.0] * (len(dates) - len(values)))
    states = _states(values, other=values)
    frame = pdlh.build_pdlh_clock_from_states(dates, states, [True] * len(dates))

    long_hold = frame[
        frame["candidate_id"]
        == "pdlh:top_position_minus_global:age=144:trigger=contraction:hold=216"
    ]
    short_hold = frame[
        frame["candidate_id"]
        == "pdlh:top_position_minus_global:age=144:trigger=contraction:hold=72"
    ]
    other_state = frame[
        frame["candidate_id"]
        == "pdlh:top_account_minus_global:age=144:trigger=contraction:hold=216"
    ]
    assert len(long_hold) == 1
    assert len(short_hold) == 2
    assert len(other_state) == 1
    assert set(other_state["entry_time"]) == set(long_hold["entry_time"])
