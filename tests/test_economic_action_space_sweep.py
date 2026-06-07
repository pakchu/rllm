from training.economic_action_space_sweep import parse_floats, parse_ints


def test_parse_floats_csv():
    assert parse_floats("0.5, 1.0") == [0.5, 1.0]


def test_parse_ints_csv():
    assert parse_ints("36,72") == [36, 72]
