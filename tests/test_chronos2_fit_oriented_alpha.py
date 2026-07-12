import numpy as np

from training.evaluate_chronos2_fit_oriented_alpha import fit_score_orientation


def test_fit_orientation_ignores_later_scores_and_flips_negative_fit_relation():
    scores = np.array([3.0, 2.0, 1.0, 999.0])
    targets = np.array([1.0, 2.0, 3.0, -999.0])
    fit = np.array([True, True, True, False])

    oriented, metadata = fit_score_orientation(scores, targets, fit)
    changed = scores.copy()
    changed[-1] = -1e9
    changed_oriented, changed_metadata = fit_score_orientation(changed, targets, fit)

    assert metadata["orientation"] == -1
    assert metadata == changed_metadata
    np.testing.assert_allclose(oriented[:3], [-3.0, -2.0, -1.0])
    np.testing.assert_allclose(changed_oriented[:3], oriented[:3])
