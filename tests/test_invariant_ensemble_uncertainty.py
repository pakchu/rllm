import numpy as np

from training.evaluate_invariant_ensemble_uncertainty import (
    signed_dynamic_policy_masks,
    uncertainty_score_streams,
)


def test_uncertainty_transforms_shrink_disagreement():
    members = {
        f"m{i}": values
        for i, values in enumerate(
            [
                np.array([0.8, 0.2]),
                np.array([0.8, 0.2]),
                np.array([0.8, 0.2]),
                np.array([0.8, -0.2]),
                np.array([0.8, -0.2]),
                np.array([0.8, -0.2]),
            ]
        )
    }

    streams, diagnostics = uncertainty_score_streams(members)

    assert streams["agree_6of6"][0] > 0.0
    assert streams["agree_6of6"][1] == 0.0
    assert streams["shrink_k1.0"][0] > streams["shrink_k1.0"][1]
    assert diagnostics["member_count"] == 6


def test_signed_policy_treats_zero_confidence_as_abstain():
    scores = np.array([0.0, 0.5, -0.5])
    positions = np.array([2, 5, 8])
    low = np.array([0.0, 0.0, 0.0])
    high = np.array([0.0, 0.0, 0.0])

    long_active, short_active = signed_dynamic_policy_masks(
        scores,
        positions,
        12,
        side_policy="both",
        low_thresholds=low,
        high_thresholds=high,
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [5])
    np.testing.assert_array_equal(np.flatnonzero(short_active), [8])
