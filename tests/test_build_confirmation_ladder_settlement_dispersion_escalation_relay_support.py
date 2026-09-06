import pandas as pd
from training import build_confirmation_ladder_settlement_dispersion_escalation_relay_support as support


def _features():
    return pd.DataFrame({"source_valid":[True]*4,"late_return":[.01,.01,-.01,-.01],"late_unanimous":[True]*4,"dispersion_escalation":[False,True,True,False],"eligible_state":[False,True,True,False]})


def test_primary_uses_false_to_true_dispersion_onset():
    active, side = support.active_and_side(_features())
    assert active.tolist() == [False, True, False, False]
    assert side.tolist() == [1, 1, -1, -1]


def test_dispersion_only_control_does_not_require_unanimity():
    features = _features(); features.loc[1,"late_unanimous"] = False
    active, _ = support.active_and_side(features, "dispersion_only")
    assert active.tolist() == [False, True, False, False]


def test_cached_panel_recomputes_normalized_weight_dispersion():
    row={"anchor_height":36,"confirmation_height":42,"feature_available_time":"2025-01-01T00:00:00Z","source_valid":True}
    row.update({f"interval_return_{i}": v for i,v in enumerate([.01,.01,.01,.02,.02,.02],1)})
    row.update({f"block_weight_{i}": v for i,v in enumerate([100,100,100,50,100,150],1)})
    result=support.build_features_from_cache(pd.DataFrame([row])).iloc[0]
    assert result["late_unanimous"]
    assert result["dispersion_escalation"]
    assert result["early_dispersion"] == 0
    assert abs(result["late_dispersion"] - 1/3) < 1e-12
