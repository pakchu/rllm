from training.export_dual_regime_predictions import _match
from training.sweep_conjunctive_event_gates import Gate


def test_match_uses_row_feature_snapshot():
    row = {"_fs": {"range_vol": 0.03, "kimchi_premium_change": 0.0}}
    assert _match(row, (Gate("range_vol", ">=", 0.02), Gate("kimchi_premium_change", "<=", 0.0)))
    assert not _match(row, (Gate("range_vol", ">=", 0.04),))
