import pandas as pd

from training import build_funding_settlement_volatility_unwind_relay_support as support


def test_fsvur_clock_uses_completed_settlement_hour_and_fixed_six_hour_hold():
    settlement = pd.Timestamp("2024-07-01T08:00:00Z")
    frame = pd.DataFrame(
        {
            "settlement_time": [settlement],
            "decision_time": [settlement + pd.Timedelta(hours=1)],
            "base_valid": [True],
            "funding_rate": [0.0002],
            "prior_abs_funding_q60": [0.0001],
            "pre_settlement_return_8h": [0.02],
            "prior_abs_pre_return_q60": [0.01],
            "post_settlement_return_1h": [-0.004],
            "bvol_body": [-0.01],
            "dvol_body": [-0.02],
        }
    )

    clock = support.build_clock(frame)

    assert len(clock) == 1
    assert clock.iloc[0]["side"] == -1
    assert clock.iloc[0]["entry_time"] == settlement + pd.Timedelta(hours=1, minutes=5)
    assert clock.iloc[0]["exit_time"] - clock.iloc[0]["entry_time"] == pd.Timedelta(hours=6)


def test_fsvur_primary_rejects_nonreversal_or_expanding_volatility():
    settlement = pd.Timestamp("2024-07-01T08:00:00Z")
    common = {
        "settlement_time": settlement,
        "decision_time": settlement + pd.Timedelta(hours=1),
        "base_valid": True,
        "funding_rate": 0.0002,
        "prior_abs_funding_q60": 0.0001,
        "pre_settlement_return_8h": 0.02,
        "prior_abs_pre_return_q60": 0.01,
        "post_settlement_return_1h": -0.004,
        "bvol_body": -0.01,
        "dvol_body": -0.02,
    }
    same_direction = {**common, "post_settlement_return_1h": 0.004}
    expanding = {**common, "dvol_body": 0.02}

    assert support.build_clock(pd.DataFrame([same_direction])).empty
    assert support.build_clock(pd.DataFrame([expanding])).empty
