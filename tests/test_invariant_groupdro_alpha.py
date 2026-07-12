import numpy as np
import pandas as pd
import torch

from training.search_invariant_groupdro_alpha import (
    environment_risk_objective,
    stable_feature_ranking,
    tail_labels,
)


def test_tail_labels_fit_thresholds_only():
    targets = np.array([-3.0, -1.0, 1.0, 3.0, 999.0])
    fit = np.array([True, True, True, True, False])

    labels, thresholds = tail_labels(
        targets, fit, low_quantile=0.25, high_quantile=0.75
    )

    np.testing.assert_allclose(thresholds, [-1.5, 1.5])
    np.testing.assert_array_equal(labels, [0, 1, 1, 2, 2])


def test_vrex_penalizes_environment_loss_variance():
    losses = torch.tensor([1.0, 3.0])

    objective, weights = environment_risk_objective(
        losses, objective="vrex", vrex_penalty=2.0
    )

    assert objective.item() == 4.0
    assert weights is None


def test_stable_feature_ranking_ignores_future_and_deduplicates_sign_flips():
    dates = pd.Series(
        pd.to_datetime(
            [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2021-01-01",
                "2021-02-01",
                "2021-03-01",
                "2022-01-01",
                "2022-02-01",
                "2022-03-01",
                "2023-01-01",
            ]
        )
    )
    stable = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 100], dtype=float)
    matrix = np.column_stack([stable, -stable, np.arange(10, dtype=float)])
    targets = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, -999], dtype=float)
    fit = dates.dt.year.to_numpy() <= 2022

    first = stable_feature_ranking(
        matrix,
        ["stable", "stable_inverse", "unstable"],
        targets,
        dates,
        fit,
        minimum_samples_per_year=3,
    )
    mutated = matrix.copy()
    mutated[-1] = [-1e9, 1e9, 1e9]
    second = stable_feature_ranking(
        mutated,
        ["stable", "stable_inverse", "unstable"],
        targets,
        dates,
        fit,
        minimum_samples_per_year=3,
    )

    assert [row["name"] for row in first] == [row["name"] for row in second]
    assert len({"stable", "stable_inverse"} & {row["name"] for row in first}) == 1
