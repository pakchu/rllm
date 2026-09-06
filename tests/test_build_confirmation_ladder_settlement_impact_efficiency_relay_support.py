import pandas as pd

from training import build_confirmation_ladder_settlement_impact_efficiency_relay_support as support


def _features():
    return pd.DataFrame(
        {
            "source_valid": [True, True, True, True],
            "late_return": [0.01, 0.01, -0.01, -0.01],
            "late_unanimous": [True, True, True, True],
            "impact_efficiency_escalation": [False, True, True, False],
            "eligible_state": [False, True, True, False],
        }
    )


def test_primary_uses_false_to_true_impact_efficiency_onset():
    active, side = support.active_and_side(_features())
    assert active.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, -1, -1]


def test_impact_only_control_does_not_require_unanimity():
    features = _features()
    features.loc[1, "late_unanimous"] = False
    active, _ = support.active_and_side(features, "impact_efficiency_only")
    assert active.tolist() == [False, True, False, False]


def test_cached_panel_recomputes_per_weight_impacts():
    row = {
        "anchor_height": 36,
        "confirmation_height": 42,
        "feature_available_time": "2025-01-01T00:00:00Z",
        "source_valid": True,
    }
    row.update({f"interval_return_{i}": value for i, value in enumerate([.01, .01, .01, .02, .02, .02], 1)})
    row.update({f"block_weight_{i}": 1_000_000 for i in range(1, 7)})
    result = support.build_features_from_cache(pd.DataFrame([row])).iloc[0]
    assert result["late_unanimous"]
    assert result["impact_efficiency_escalation"]
    assert result["late_impact"] == 2 * result["early_impact"]
