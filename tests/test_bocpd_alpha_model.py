import numpy as np
import pandas as pd
import pytest

from training.search_bocpd_state_gated_alpha import (
    _map_output,
    bocpd_student_t,
    bocpd_student_t_checkpointed,
    top_k_promotions,
)


def test_bocpd_filter_is_prefix_causal():
    rng = np.random.default_rng(17)
    observations = np.r_[rng.normal(-0.3, 0.8, 90), rng.normal(0.4, 0.8, 110)]

    full = bocpd_student_t(observations, hazard_lambda=72, max_run_length=100)
    prefix = bocpd_student_t(observations[:137], hazard_lambda=72, max_run_length=100)

    for key in full:
        np.testing.assert_allclose(full[key][:137], prefix[key], rtol=0, atol=1e-12)
    assert np.all((full["short_mass"] >= 0.0) & (full["short_mass"] <= 1.0))


def test_bocpd_checkpoint_resume_is_exactly_batch_equivalent():
    rng = np.random.default_rng(715)
    observations = rng.normal(size=(480, 2))
    full = bocpd_student_t(observations, hazard_lambda=72, max_run_length=100)

    first, checkpoint = bocpd_student_t_checkpointed(
        observations[:137], hazard_lambda=72, max_run_length=100
    )
    second, checkpoint = bocpd_student_t_checkpointed(observations[137:311], state=checkpoint)
    third, _ = bocpd_student_t_checkpointed(observations[311:], state=checkpoint)

    for key, expected in full.items():
        resumed = np.concatenate([first[key], second[key], third[key]], axis=0)
        np.testing.assert_array_equal(resumed, expected)


def test_bocpd_checkpoint_rejects_parameter_drift():
    _, checkpoint = bocpd_student_t_checkpointed(
        np.arange(20.0), hazard_lambda=72, max_run_length=100
    )
    with pytest.raises(ValueError, match="hazard_lambda mismatch"):
        bocpd_student_t_checkpointed(
            np.arange(20.0, 30.0), state=checkpoint, hazard_lambda=336
        )


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


def test_bocpd_top10_later_winner_is_promoted_but_rank11_is_not():
    frozen_loser = {"passes_alpha_pool": False, "passes_live_grade": False}
    later_winner = {"passes_alpha_pool": True, "passes_live_grade": True}
    rank11_winner = {"passes_alpha_pool": True, "passes_live_grade": True}

    selected = [frozen_loser, later_winner, *([frozen_loser] * 8), rank11_winner]
    assert top_k_promotions(selected) == ([later_winner], [later_winner])
