from training.sweep_conjunctive_event_gates import Gate, filter_rows, make_primitives


def test_gate_filters_signal_time_feature_snapshot():
    rows = [
        {"_fs": {"range_vol": 0.01, "drawdown": 0.02}},
        {"_fs": {"range_vol": 0.03, "drawdown": 0.01}},
        {"_fs": {"range_vol": 0.04, "drawdown": 0.03}},
    ]
    gates = (Gate("range_vol", ">=", 0.02), Gate("drawdown", ">=", 0.02))
    assert filter_rows(rows, gates) == [rows[2]]


def test_make_primitives_uses_numeric_train_features():
    rows = [{"_fs": {"x": float(i), "flat": 1.0, "bad": "n/a"}} for i in range(60)]
    gates = make_primitives(rows, (0.5,))
    assert {g.feature for g in gates} == {"x"}
    assert {g.op for g in gates} == {">=", "<="}
