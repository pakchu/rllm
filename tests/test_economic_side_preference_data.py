from training.economic_side_preference_data import side_target, split_for_date


def test_side_target_json_shape():
    assert side_target("LONG") == '{"risk": "MEDIUM", "side": "LONG"}'


def test_split_for_date_boundaries():
    assert split_for_date("2024-01-01 00:00:00") == "train"
    assert split_for_date("2025-03-01 00:00:00") == "val"
    assert split_for_date("2025-09-01 00:00:00") == "eval"
