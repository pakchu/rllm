from training.economic_walkforward_policy_check import in_range, parse_dt


def test_in_range_includes_bounds():
    row = {"date": "2025-01-01 00:00:00"}
    assert in_range(row, parse_dt("2025-01-01 00:00:00"), parse_dt("2025-01-01 00:00:00"))


def test_in_range_excludes_before_start():
    row = {"date": "2024-12-31 23:59:59"}
    assert not in_range(row, parse_dt("2025-01-01 00:00:00"), None)
