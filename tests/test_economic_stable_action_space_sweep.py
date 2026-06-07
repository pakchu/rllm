from training.economic_stable_action_space_sweep import config_grid, parse_floats, parse_ints


def test_config_grid_nonempty():
    assert config_grid()


def test_parse_csv_helpers():
    assert parse_ints("36, 72") == [36, 72]
    assert parse_floats("1.2, 2.5") == [1.2, 2.5]
