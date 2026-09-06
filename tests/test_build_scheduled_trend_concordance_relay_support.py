import pandas as pd

from training import build_scheduled_trend_concordance_relay_support as support


def test_primary_clock_requires_sign_agreement():
    rows = []
    for day, r3, r14 in [
        ("2023-07-03", 0.1, 0.2),
        ("2023-07-06", -0.1, 0.2),
        ("2023-07-10", -0.2, -0.3),
    ]:
        rows.append({
            "decision_time": pd.Timestamp(day, tz="UTC"), "source_valid": True,
            "return_3d": r3, "return_14d": r14, "rv20": 0.5,
            "rv20_threshold": 0.7, "rv20_q90_active": False,
        })
    clock = support.build_clock(pd.DataFrame(rows))
    assert clock.side.tolist() == [1, -1]


def test_fixed_source_support_contract():
    assert support.MINIMUM == {"train": 8, "test": 12, "eval": 12, "final": 8}
    assert support.CONTROLS == ("three_day_only", "fourteen_day_only", "direction_flip")
