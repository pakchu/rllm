import pandas as pd

from training import build_options_crowding_deleveraging_relay_support_v3 as s


def test_raw_oi_observations_join_backward_without_timestamp_snapping() -> None:
    decisions = pd.to_datetime(["2023-08-01T01:00:00Z", "2023-08-01T02:00:00Z"])
    bvol = pd.DataFrame(
        {
            "feature_available_time_utc": decisions,
            "open": [100, 100], "close": [101, 101], "feature_valid": [True, True],
        }
    )
    dvol = pd.DataFrame(
        {"close_time": decisions, "open": [100, 100], "close": [102, 102]}
    )
    oi = pd.DataFrame(
        {
            "ts": pd.to_datetime([
                "2023-07-31T23:59:58Z", "2023-08-01T00:59:58Z",
                "2023-08-01T01:59:58Z",
            ]),
            "sum_open_interest": [90, 100, 120],
        }
    )
    funding = pd.DataFrame(
        {
            "funding_time": pd.to_datetime(["2023-07-01T00:00:00Z"]),
            "funding_rate": [0.001],
        }
    )
    joined = s.joined_features(bvol, dvol, oi, funding)
    assert joined["oi_current_time"].tolist() == list(
        pd.to_datetime(["2023-08-01T00:59:58Z", "2023-08-01T01:59:58Z"])
    )
    assert joined["oi_prior_time"].tolist() == list(
        pd.to_datetime(["2023-07-31T23:59:58Z", "2023-08-01T00:59:58Z"])
    )
    assert joined["oi_change"].round(8).tolist() == [0.11111111, 0.2]


def test_support_v3_still_cannot_open_economic_outcomes() -> None:
    assert '"advance_to_economic_outcomes": False' in open(s.__file__).read()
