import numpy as np

from training.search_bocpd_state_gated_alpha import bocpd_student_t


def test_bocpd_filter_is_prefix_causal():
    rng = np.random.default_rng(17)
    observations = np.r_[rng.normal(-0.3, 0.8, 90), rng.normal(0.4, 0.8, 110)]

    full = bocpd_student_t(observations, hazard_lambda=72, max_run_length=100)
    prefix = bocpd_student_t(observations[:137], hazard_lambda=72, max_run_length=100)

    for key in full:
        np.testing.assert_allclose(full[key][:137], prefix[key], rtol=0, atol=1e-12)
    assert np.all((full["short_mass"] >= 0.0) & (full["short_mass"] <= 1.0))
