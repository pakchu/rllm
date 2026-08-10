from pathlib import Path

import pandas as pd

from training import build_high_volatility_treasury_belly_curvature_relay_support as support


def test_xml_parser_extracts_three_maturities_and_curvature():
    frame = support.load_treasury_xml([
        Path("data/treasury_parallel_yield_shock_relay_sources_2023_2026/official_xml/daily_treasury_yield_curve_2024.xml")
    ])
    assert {"yield_2y", "yield_5y", "yield_10y", "curvature", "curvature_change"} <= set(frame)
    first = frame.iloc[0]
    assert first.source_day == pd.Timestamp("2024-01-02T00:00:00Z")
    assert first.yield_2y == 4.33
    assert first.yield_5y == 3.93
    assert first.yield_10y == 3.95
    assert abs(first.curvature - (2 * 3.93 - 4.33 - 3.95)) < 1e-12


def test_clock_uses_negative_curvature_change_and_global_reservation():
    states = pd.DataFrame({
        "source_day": pd.to_datetime(["2023-07-02", "2023-07-03", "2023-07-04"], utc=True),
        "decision_time": pd.to_datetime(["2023-07-03T12:00Z", "2023-07-04T12:00Z", "2023-07-05T12:00Z"], utc=True),
        "source_valid": [True, True, True], "curvature_change_valid": [True, True, True],
        "yield_2y": [4.0, 4.0, 4.0], "yield_5y": [4.0, 4.1, 4.0],
        "yield_10y": [4.0, 4.0, 4.0], "curvature": [0.0, 0.2, 0.0],
        "curvature_change": [0.1, 0.2, -0.2], "btc_variation": [0.1, 0.1, 0.1],
        "btc_variation_rank": [0.8, 0.8, 0.8],
    })
    clock = support.build_clock(states)
    assert clock.side.tolist() == [-1, -1, 1]
    assert clock.entry_time.tolist() == pd.to_datetime([
        "2023-07-03T12:05Z", "2023-07-04T12:05Z", "2023-07-05T12:05Z"
    ], utc=True).tolist()
