import numpy as np

from training.evaluate_invariant_groupdro_rolling_gate import ensemble_score_streams


def test_ensemble_definitions_are_fixed_within_feature_set():
    scores = {
        "linear_erm_stable8": np.array([1.0, 2.0]),
        "linear_vrex_stable8": np.array([3.0, 4.0]),
        "mlp_erm_stable8": np.array([5.0, 6.0]),
        "mlp_groupdro_stable8": np.array([7.0, 8.0]),
    }
    metadata = {
        "linear_erm_stable8": {
            "feature_set": "stable8",
            "architecture": "linear",
            "objective": "erm",
        },
        "linear_vrex_stable8": {
            "feature_set": "stable8",
            "architecture": "linear",
            "objective": "vrex",
        },
        "mlp_erm_stable8": {
            "feature_set": "stable8",
            "architecture": "mlp",
            "objective": "erm",
        },
        "mlp_groupdro_stable8": {
            "feature_set": "stable8",
            "architecture": "mlp",
            "objective": "groupdro",
        },
    }

    ensembles = ensemble_score_streams(scores, metadata)

    np.testing.assert_allclose(ensembles["ensemble_all_stable8"], [4.0, 5.0])
    np.testing.assert_allclose(ensembles["ensemble_mlp_stable8"], [6.0, 7.0])
    np.testing.assert_allclose(ensembles["ensemble_invariant_stable8"], [5.0, 6.0])
