from __future__ import annotations

import numpy as np

from training.search_causal_weak_tensor_exit_router_alpha import (
    ACTIONS,
    TENSOR_SPEC,
    actions_from_prediction,
    fit_ridge_router,
    tensor_design,
)


def test_tensor_grid_discloses_all_1008_cells() -> None:
    assert TENSOR_SPEC["grid_cells"] == 1008


def test_tensor_design_has_linear_and_cross_dimensions() -> None:
    weak = np.asarray([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    bocpd = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    design, metadata = tensor_design(
        weak, bocpd, np.asarray([True, True, True]), form="tensor"
    )

    assert design.shape == (3, 8)  # 2 weak + 2 BOCPD + 2x2 interactions
    assert metadata["dimensions"] == 8


def test_ridge_router_predicts_two_advantages() -> None:
    design = np.asarray([[-1.0], [0.0], [1.0], [2.0]])
    target = np.column_stack([design[:, 0], -design[:, 0]])
    prediction, model = fit_ridge_router(
        design, target, np.ones(4, dtype=bool), ridge=0.1
    )

    assert prediction.shape == (4, 2)
    assert len(model["intercept"]) == 2


def test_ridge_fit_is_invariant_to_appended_oos_rows() -> None:
    design = np.asarray([[-1.0], [0.0], [1.0], [9.0]])
    target = np.asarray([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [99.0, 99.0]])
    fit_mask = np.asarray([True, True, True, False])

    _, full_model = fit_ridge_router(design, target, fit_mask, ridge=0.1)
    _, prefix_model = fit_ridge_router(
        design[:3], target[:3], np.ones(3, dtype=bool), ridge=0.1
    )

    assert full_model == prefix_model


def test_action_router_preserves_tp12_on_exact_tie() -> None:
    prediction = np.asarray([[0.0, 0.0], [1.0, 0.2], [0.1, 2.0]])
    actions = actions_from_prediction(6, np.asarray([1, 3, 5]), prediction)

    assert actions.tolist() == [-1, 12, -1, 4, -1, 8]
    assert set(actions[actions > 0]).issubset(set(ACTIONS))
