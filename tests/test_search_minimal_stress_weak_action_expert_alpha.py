from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd

from training.search_minimal_stress_weak_action_expert_alpha import (
    ACTION_SETS,
    EXPERT_SPEC,
    FIT_END,
    FIT_START,
    _implementation_hash,
    actions_from_prediction,
    event_weights,
    fit_event_mask,
    fit_weighted_action_ridge,
    schedule_window,
)


def test_action_expert_grid_discloses_all_972_cells() -> None:
    assert EXPERT_SPEC["grid_cells"] == 972


def test_fit_event_mask_purges_labels_that_cross_2023() -> None:
    dates = pd.Series(
        pd.date_range("2022-12-29 00:00:00", periods=5, freq="24h")
    )
    context = {"dates": dates}
    events = {
        "signals": np.asarray([0, 1, 2], dtype=np.int64),
        "max_exits": np.asarray([1, 2, 3], dtype=np.int64),
    }

    mask = fit_event_mask(context, events)

    assert FIT_START == pd.Timestamp("2020-07-01")
    assert FIT_END == pd.Timestamp("2023-01-01")
    assert mask.tolist() == [True, True, False]


def test_year_weights_give_each_fit_year_equal_total_mass() -> None:
    dates = pd.Series(
        pd.to_datetime(
            ["2020-07-01", "2021-01-01", "2021-02-01", "2022-01-01"]
        )
    )
    fit = np.ones(4, dtype=bool)
    weights = event_weights(dates, np.ones(4), fit, "year")

    years = dates.dt.year.to_numpy()
    totals = [float(weights[years == year].sum()) for year in (2020, 2021, 2022)]
    assert np.allclose(totals, totals[0])
    assert np.isclose(weights.mean(), 1.0)


def test_year_source_weights_equalize_joint_groups() -> None:
    dates = pd.Series(
        pd.to_datetime(
            ["2021-01-01", "2021-02-01", "2021-03-01", "2022-01-01"]
        )
    )
    source = np.asarray([1.0, 1.0, -1.0, -1.0])
    fit = np.ones(4, dtype=bool)
    weights = event_weights(dates, source, fit, "year_source")

    groups = np.asarray(["2021:1", "2021:1", "2021:-1", "2022:-1"])
    totals = [float(weights[groups == group].sum()) for group in np.unique(groups)]
    assert np.allclose(totals, totals[0])


def test_weighted_ridge_ignores_appended_nonfit_target_rows() -> None:
    design = np.asarray([[-1.0], [0.0], [1.0], [99.0]])
    fit = np.asarray([True, True, True, False])
    target_fit = np.asarray([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
    weight = np.ones(4)

    _, full_model = fit_weighted_action_ridge(
        design, target_fit, fit, weight, ridge=0.1
    )
    _, prefix_model = fit_weighted_action_ridge(
        design[:3], target_fit, np.ones(3, dtype=bool), weight[:3], ridge=0.1
    )

    assert full_model == prefix_model


def test_prediction_uses_neutral_action_on_nonpositive_or_exact_tie() -> None:
    actions = ACTION_SETS["tp4_time"]
    prediction = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-0.1, -0.2, -0.3, -0.4],
            [0.1, 0.3, 0.2, 0.05],
        ]
    )

    routed = actions_from_prediction(
        6, np.asarray([1, 3, 5]), prediction, actions
    )

    assert routed[1].tolist() == [0, 0]
    assert routed[3].tolist() == [0, 0]
    assert routed[5].tolist() == list(actions[1])


def test_counterfactual_trade_interface_documents_both_sides() -> None:
    # Cheap contract guard: all non-neutral actions are executable sides and the
    # frozen family contains both LONG and SHORT choices.
    actions = ACTION_SETS["tp4_tp8_tp12_time"]
    assert {side for side, _ in actions} == {-1, 1}
    assert all(take_bps > 0 for _, take_bps in actions)


def test_implementation_hash_covers_selection_and_oos_guards() -> None:
    source = inspect.getsource(_implementation_hash)
    for function_name in (
        "_grid",
        "_freeze_payload",
        "_freeze_hash",
        "_validate_manifest",
        "_write_manifest_once",
        "_selection_payload",
        "_mark_oos_opened",
        "_oos",
    ):
        assert function_name in source


def test_schedule_rejects_trade_whose_exit_crosses_period_end() -> None:
    class _Engine:
        def trade_at(self, signal, side, hold, take_bps, stop_bps):
            del signal, side, hold, take_bps, stop_bps
            return SimpleNamespace(exit_position=4)

    dates = pd.Series(pd.date_range("2022-12-31", periods=6, freq="h"))
    context = {
        "dates": dates,
        "active": np.asarray([False, True, False, False, False, False]),
        "engine": _Engine(),
    }
    actions = np.zeros((6, 2), dtype=np.int32)
    actions[1] = (1, 400)

    trades = schedule_window(
        context,
        actions,
        start="2022-12-31 00:00:00",
        end="2022-12-31 03:00:00",
    )

    assert trades == []
