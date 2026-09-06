import pandas as pd

from training import build_high_volatility_month_phase_seasonality_relay_model as builder


def test_model_selects_four_each_sign_deterministically():
    rows = []
    for day in range(1, 13):
        score = (13 - day) / 1000 if day <= 6 else -(day - 6) / 1000
        rows.extend({"day_of_month": day, "label": score} for _ in range(30))
    model = builder.fit_model(pd.DataFrame(rows))
    selected = model["selected"]
    assert sum(item["side"] == 1 for item in selected) == 4
    assert sum(item["side"] == -1 for item in selected) == 4
    assert {item["day_of_month"] for item in selected if item["side"] == 1} == {1, 2, 3, 4}
    assert {item["day_of_month"] for item in selected if item["side"] == -1} == {9, 10, 11, 12}
