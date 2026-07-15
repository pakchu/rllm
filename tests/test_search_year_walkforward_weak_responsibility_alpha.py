import numpy as np
import pandas as pd
import pytest

from training.search_year_walkforward_weak_responsibility_alpha import (
    FOLDS,
    fit_event_mask,
    mark_adjacent_stability,
    source_action,
    validate_fold_plan,
)


def test_fold_plan_is_expanding_and_never_opens_2024():
    validate_fold_plan()
    assert FOLDS[-1]["predict_end"] == "2024-01-01"
    broken = (
        {
            "name": "leaky",
            "fit_end": "2021-01-02",
            "predict_start": "2021-01-01",
            "predict_end": "2022-01-01",
        },
    )
    with pytest.raises(ValueError, match="non-causal"):
        validate_fold_plan(broken)


def test_fit_event_mask_purges_full_maximum_path():
    dates = pd.Series(pd.date_range("2020-06-01", periods=80, freq="D"))
    events = {
        "signals": np.array([0, 10, 20], dtype=np.int64),
        "max_path_end": np.array([20, 30, 40], dtype=np.int64),
    }
    mask = fit_event_mask(dates, events, fit_end="2020-07-01")
    assert mask.tolist() == [True, False, False]


def test_source_actions_keep_exact_execution_contract():
    assert source_action(True) == (576, 400, 1_000_000)
    assert source_action(False) == (144, 1_000_000, 300)


def test_adjacent_stability_rejects_isolated_pass():
    rows = [
        {
            "spec": {"hazard_hours": 168, "form": "linear", "ridge": 10.0, "margin": 0.0},
            "selection_passed": True,
        },
        {
            "spec": {"hazard_hours": 168, "form": "linear", "ridge": 10.0, "margin": 0.001},
            "selection_passed": False,
        },
        {
            "spec": {"hazard_hours": 168, "form": "linear", "ridge": 100.0, "margin": 0.0},
            "selection_passed": False,
        },
    ]
    mark_adjacent_stability(rows)
    assert rows[0]["accepted"] is False
    rows[1]["selection_passed"] = True
    mark_adjacent_stability(rows)
    assert rows[0]["accepted"] is True
    assert rows[1]["accepted"] is True
