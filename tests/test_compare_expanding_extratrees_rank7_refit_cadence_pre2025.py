from __future__ import annotations

import numpy as np
import pandas as pd

from training.compare_expanding_extratrees_rank7_refit_cadence_pre2025 import (
    cadence_windows,
    cutoff_masks,
    frozen_rank7_spec,
    select_cadence,
)


def test_monthly_windows_cover_pre2025_selection_without_gaps() -> None:
    windows = cadence_windows("monthly", "2023-01-01", "2025-01-01")
    assert len(windows) == 24
    assert windows[0] == ("month_2023_01", "2023-01-01", "2023-02-01")
    assert windows[-1] == ("month_2024_12", "2024-12-01", "2025-01-01")
    assert all(left[2] == right[1] for left, right in zip(windows, windows[1:]))


def test_annual_windows_support_a_partial_final_year() -> None:
    assert cadence_windows("annual", "2025-01-01", "2026-06-02") == (
        ("year_2025", "2025-01-01", "2026-01-01"),
        ("year_2026", "2026-01-01", "2026-06-02"),
    )


def test_cutoff_masks_purge_labels_exiting_on_or_after_cutoff() -> None:
    base = {
        "signal_dates": pd.Series(pd.to_datetime(["2022-12-01", "2022-12-20", "2023-01-10"])),
        "targets": np.ones((3, 2), dtype=float),
        "exit_dates": pd.to_datetime(["2022-12-10", "2023-01-01", "2023-01-12"]).to_numpy(),
    }
    fit, predict = cutoff_masks(base, "2023-01-01", "2023-02-01")
    assert fit.tolist() == [True, False, False]
    assert predict.tolist() == [False, False, True]


def test_cutoff_masks_allow_a_zero_event_prediction_month() -> None:
    base = {
        "signal_dates": pd.Series(pd.to_datetime(["2022-12-01"])),
        "targets": np.ones((1, 2), dtype=float),
        "exit_dates": pd.to_datetime(["2022-12-10"]).to_numpy(),
    }
    fit, predict = cutoff_masks(base, "2023-04-01", "2023-05-01")
    assert fit.tolist() == [True]
    assert predict.tolist() == [False]


def test_cadence_selection_uses_rank_and_prefers_annual_on_exact_tie() -> None:
    annual = {"cadence": "annual", "rank": [1, 3.0, 3.2, 40, 20.0]}
    monthly = {"cadence": "monthly", "rank": [1, 3.0, 3.2, 40, 20.0]}
    assert select_cadence((monthly, annual)) == "annual"
    monthly["rank"] = [1, 3.1, 3.0, 38, 19.0]
    assert select_cadence((annual, monthly)) == "monthly"


def test_frozen_rank7_spec_is_loaded_from_parent_manifest() -> None:
    manifest_hash, learner, policy, row = frozen_rank7_spec()
    assert len(manifest_hash) == 64
    assert row["rank_position"] == 7
    assert (learner.max_depth, learner.min_samples_leaf, learner.max_features) == (2, 32, 0.8)
    assert (
        policy.risk_lambda,
        policy.funding_quantile,
        policy.premium_quantile,
        policy.risk_quantile,
    ) == (0.25, 0.40, 0.55, 0.75)
