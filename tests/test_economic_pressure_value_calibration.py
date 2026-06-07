from training.economic_pressure_value_calibration import bucket, compact_features, choose_action, make_stats


def test_bucket_assigns_ordered_ranges():
    assert bucket(0.1, [0.2, 0.5]) == "lt0p2"
    assert bucket(0.3, [0.2, 0.5]) == "lt0p5"
    assert bucket(0.7, [0.2, 0.5]) == "hi"


def test_compact_features_parses_prompt_json():
    row = {"prompt": 'x\nCompact features: {"state":{"regime":"UPTREND"}}\n\nTrain-only structured teacher hint: {}'}
    assert compact_features(row)["state"]["regime"] == "UPTREND"


def test_choose_action_abstains_below_threshold():
    row = {"prompt": 'Compact features: {"state":{"regime":"RANGE","trend_alignment":"BULL_STACK"}}\n\n', "teacher": {"teacher_pressure": "LONG_FAVORED", "teacher_confidence": 0.9}}
    key = ("LONG", "coarse", "LONG_FAVORED", "RANGE", "BULL_STACK")
    action, info = choose_action(row, {key: make_stats([-0.01, -0.02])}, level="coarse", min_n=2, min_score=0.0, score_mode="mean", side_gate="free")
    assert action == "NONE"
    assert info["score"] < 0
