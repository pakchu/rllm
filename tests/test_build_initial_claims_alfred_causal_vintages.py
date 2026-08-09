from datetime import date

from training import build_initial_claims_alfred_causal_vintages as build


def test_parse_bulk_binds_first_and_revised_vintages():
    payload = (
        "observation_date,ICSA_20200109,ICSA_20200116\n"
        "2020-01-04,210000,205000\n"
    ).encode()
    rows = build.parse_bulk(payload, [date(2020, 1, 4)])
    assert rows == [{
        "reference_date": "2020-01-04",
        "first_vintage_date": "2020-01-09",
        "revised_vintage_date": "2020-01-16",
        "first_available": True,
        "revised_available": True,
        "icsa_first": 210000.0,
        "icsa_revised": 205000.0,
    }]


def test_bulk_url_has_both_causal_vintages():
    url = build.bulk_url([date(2020, 1, 4)])
    assert "id=ICSA,ICSA" in url
    assert "vintage_date=2020-01-09,2020-01-16" in url


def test_six_adjacent_references_fit_seven_unique_vintage_columns():
    references = build.saturdays(date(2020, 1, 4), date(2020, 2, 8))
    vintages = build.bulk_url(references).split("vintage_date=", 1)[1].split(",")
    assert len(set(vintages)) == 7
