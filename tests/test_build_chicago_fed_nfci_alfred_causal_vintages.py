from datetime import date

import pytest

from training import build_chicago_fed_nfci_alfred_causal_vintages as builder


def test_parse_requires_exact_reference_and_vintage_column():
    payload = b"observation_date,NFCI_20230714\n2023-06-30,-0.30\n2023-07-07,-0.32\n"
    assert builder.parse_exact_observation(payload, date(2023, 7, 7), date(2023, 7, 14)) == -0.32
    with pytest.raises(ValueError, match="unexpected ALFRED schema"):
        builder.parse_exact_observation(payload, date(2023, 7, 14), date(2023, 7, 21))


def test_missing_exact_reference_is_causal_unavailability_not_imputation():
    payload = b"observation_date,NFCI_20221223\n2022-12-09,-0.10\n"
    assert builder.parse_exact_observation(payload, date(2022, 12, 16), date(2022, 12, 23)) is None


def test_bulk_parser_selects_diagonal_causal_vintages_without_fill():
    payload = (
        b"observation_date,NFCI_20230714,NFCI_20230721,NFCI_20230728\n"
        b"2023-07-07,-0.32,-0.33,-0.34\n"
        b"2023-07-14,.,-0.35,-0.36\n"
    )
    rows = builder.parse_bulk(payload, [date(2023, 7, 7), date(2023, 7, 14)])
    assert [row["nfci"] for row in rows] == [-0.32, -0.35]
    missing = builder.parse_bulk(payload, [date(2023, 7, 7), date(2023, 7, 14), date(2023, 7, 21)])
    assert not missing[-1]["available"]


def test_reference_grid_is_strictly_weekly_friday():
    values = builder.fridays(date(2020, 1, 3), date(2020, 1, 17))
    assert values == [date(2020, 1, 3), date(2020, 1, 10), date(2020, 1, 17)]
    assert all(value.weekday() == 4 for value in values)
