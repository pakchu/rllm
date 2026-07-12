import numpy as np

from training.search_kalman_state_gated_alpha import kalman_local_linear
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
