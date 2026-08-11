from datetime import date

from training import build_commercial_paper_alfred_causal_vintages as build


def test_parse_bulk_binds_first_and_revised_vintages():
    payload = (
        "observation_date,COMPOUT_20200102,COMPOUT_20200109\n"
        "2020-01-01,1500.0,1495.0\n"
    ).encode()
    rows = build.parse_bulk(payload, [date(2020, 1, 1)])
    assert rows == [{
        "reference_date": "2020-01-01",
        "first_vintage_date": "2020-01-02",
        "revised_vintage_date": "2020-01-09",
        "first_available": True,
        "revised_available": True,
        "compout_first": 1500.0,
        "compout_revised": 1495.0,
    }]


def test_bulk_url_has_both_causal_vintages():
    url = build.bulk_url([date(2020, 1, 1)])
    assert "id=COMPOUT,COMPOUT" in url
    assert "vintage_date=2020-01-02,2020-01-09" in url


def test_six_adjacent_references_fit_seven_unique_vintage_columns():
    references = build.wednesdays(date(2020, 1, 1), date(2020, 2, 5))
    vintages = build.bulk_url(references).split("vintage_date=", 1)[1].split(",")
    assert len(set(vintages)) == 7
