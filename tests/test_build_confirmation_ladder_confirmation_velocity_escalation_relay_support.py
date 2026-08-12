import pandas as pd

from training import build_confirmation_ladder_confirmation_velocity_escalation_relay_support as support


def test_cached_panel_recomputes_return_per_second_velocity():
    row = {
        "anchor_height": 36,
        "confirmation_height": 42,
        "feature_available_time": "2025-01-01T00:00:00Z",
        "source_valid": True,
    }
    row.update({f"interval_return_{i}": value for i, value in enumerate([.01, .01, .01, .02, .02, .02], 1)})
    row.update({f"interval_duration_{i}": 600 for i in range(1, 7)})
    result = support.build_features_from_cache(pd.DataFrame([row])).iloc[0]
    assert result["late_unanimous"]
    assert result["velocity_escalation"]
    assert result["late_velocity"] == 2 * result["early_velocity"]


def test_primary_uses_false_to_true_velocity_onset():
    features = pd.DataFrame(
        {
            "source_valid": [True, True, True, True],
            "late_return": [1.0, 1.0, -1.0, -1.0],
            "late_unanimous": [True] * 4,
            "velocity_escalation": [False, True, True, False],
            "eligible_state": [False, True, True, False],
        }
    )
    active, side = support.active_and_side(features)
    assert active.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, -1, -1]
