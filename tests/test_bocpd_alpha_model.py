import numpy as np
import pandas as pd

from training.search_bocpd_state_gated_alpha import (
    _map_output,
    bocpd_student_t,
    frozen_winner_promotions,
)


def test_bocpd_filter_is_prefix_causal():
    rng = np.random.default_rng(17)
    observations = np.r_[rng.normal(-0.3, 0.8, 90), rng.normal(0.4, 0.8, 110)]

    full = bocpd_student_t(observations, hazard_lambda=72, max_run_length=100)
    prefix = bocpd_student_t(observations[:137], hazard_lambda=72, max_run_length=100)

    for key in full:
        np.testing.assert_allclose(full[key][:137], prefix[key], rtol=0, atol=1e-12)
    assert np.all((full["short_mass"] >= 0.0) & (full["short_mass"] <= 1.0))


def test_bocpd_hourly_mapping_never_uses_future_output():
    dates = pd.Series(pd.to_datetime(["2026-01-01 00:30", "2026-01-01 01:00"]))
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
            "primary": [1.0, 2.0],
            "short_mass": [0.1, 0.2],
            "run_drop": [0.0, 0.1],
            "secondary": [3.0, 4.0],
            "surprise": [5.0, 6.0],
        }
    )

    mapped = _map_output(dates, output)
    np.testing.assert_allclose(mapped["primary"].to_numpy(), np.array([1.0, 2.0]))


def test_bocpd_diagnostic_later_winner_is_not_promoted():
    frozen_loser = {"passes_alpha_pool": False, "passes_live_grade": False}
    later_winner = {"passes_alpha_pool": True, "passes_live_grade": True}

    assert frozen_winner_promotions([frozen_loser, later_winner]) == ([], [])
