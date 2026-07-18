import numpy as np
import pandas as pd

from training.search_kalman_state_gated_alpha import (
    kalman_local_linear,
    kalman_local_linear_checkpointed,
    map_hourly_state,
    top_k_promotions,
)
from training.search_gaussian_hmm_regime_alpha import filtered, fit_hmm


def test_hmm_filter_is_prefix_causal():
    rng = np.random.default_rng(7)
    x = np.r_[rng.normal(-1.0, 0.25, (80, 2)), rng.normal(1.0, 0.25, (80, 2))]
    model = fit_hmm(x[:120], 2, seed=3, max_iter=4)

    full = filtered(x, model)
    prefix = filtered(x[:100], model)

    np.testing.assert_allclose(full[:100], prefix, rtol=0, atol=1e-12)
    np.testing.assert_allclose(full.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_hmm_transition_matrix_is_stochastic():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(120, 3))
    model = fit_hmm(x, 3, seed=5, max_iter=3)

    assert np.all(model["A"] >= 0)
    np.testing.assert_allclose(model["A"].sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_kalman_filter_is_prefix_causal():
    rng = np.random.default_rng(13)
    log_price = 8.0 + np.cumsum(rng.normal(0.0, 0.01, 300))

    full = kalman_local_linear(log_price, q_level=1.0, q_slope=0.01, r_obs=0.5, train_var=1e-4)
    prefix = kalman_local_linear(
        log_price[:173], q_level=1.0, q_slope=0.01, r_obs=0.5, train_var=1e-4
    )

    np.testing.assert_allclose(full[:173], prefix, rtol=0, atol=1e-12)


def test_kalman_checkpoint_resume_is_exactly_batch_equivalent():
    rng = np.random.default_rng(715)
    log_price = 8.0 + np.cumsum(rng.normal(0.0, 0.01, 480))
    full = kalman_local_linear(
        log_price, q_level=1.0, q_slope=0.01, r_obs=0.5, train_var=1e-4
    )
    first, checkpoint = kalman_local_linear_checkpointed(
        log_price[:173], q_level=1.0, q_slope=0.01, r_obs=0.5, train_var=1e-4
    )
    second, checkpoint = kalman_local_linear_checkpointed(
        log_price[173:311],
        q_level=1.0,
        q_slope=0.01,
        r_obs=0.5,
        train_var=1e-4,
        checkpoint=checkpoint,
    )
    third, _ = kalman_local_linear_checkpointed(
        log_price[311:],
        q_level=1.0,
        q_slope=0.01,
        r_obs=0.5,
        train_var=1e-4,
        checkpoint=checkpoint,
    )
    np.testing.assert_array_equal(np.vstack([first, second, third]), full)


def test_kalman_hourly_mapping_never_uses_future_state():
    dates = pd.Series(pd.to_datetime(["2026-01-01 00:30", "2026-01-01 01:00", "2026-01-01 01:30"]))
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
            "state": [4, 7],
        }
    )

    np.testing.assert_array_equal(map_hourly_state(dates, hourly), np.array([4, 7, 7]))


def test_kalman_top10_later_winner_is_promoted_but_rank11_is_not():
    frozen_loser = {"passes_alpha_pool": False, "passes_live_grade": False}
    later_winner = {"passes_alpha_pool": True, "passes_live_grade": True}
    rank11_winner = {"passes_alpha_pool": True, "passes_live_grade": True}

    selected = [frozen_loser, later_winner, *([frozen_loser] * 8), rank11_winner]
    assert top_k_promotions(selected) == ([later_winner], [later_winner])
