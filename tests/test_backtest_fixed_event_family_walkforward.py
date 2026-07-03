from training.backtest_fixed_event_family_walkforward import _folds, _leverage_values, _split_name


def test_fold_builder_and_split_names_are_chronological():
    folds = _folds("2025-01-01", "2025-04-01", 1)
    assert [f["name"] for f in folds] == ["202501_202502", "202502_202503", "202503_202504"]
    assert _split_name("2024-12-01") == "history_oos"
    assert _split_name("2025-06-01") == "test_2025"
    assert _split_name("2026-01-01") == "eval_2026h1"


def test_leverage_grid_parser():
    assert _leverage_values("0.5,1,1.5") == [0.5, 1.0, 1.5]
