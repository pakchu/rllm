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
