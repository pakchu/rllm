from __future__ import annotations

import pandas as pd

from training import select_psim_d8_cross_protocol_disagreement_2022 as selection


def test_trade_return_includes_side_cost_and_funding() -> None:
    long_result = selection.trade_net_return(
        side=1,
        entry_price=100.0,
        exit_price=110.0,
        funding_cashflows=[0.001, -0.0005],
        cost_rate=0.0006,
    )
    short_result = selection.trade_net_return(
        side=-1,
        entry_price=100.0,
        exit_price=90.0,
        funding_cashflows=[0.001],
        cost_rate=0.0006,
    )

    assert abs(long_result - (0.10 - 0.0005 - 0.0006 - 0.00066)) < 1e-12
    assert abs(short_result - (0.10 + 0.001 - 0.0006 - 0.00054)) < 1e-12


def test_strict_trade_path_uses_adverse_intrabar_extreme() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2022-01-01 12:10:00",
                    "2022-01-01 12:15:00",
                    "2022-01-01 12:20:00",
                ],
                utc=True,
            ),
            "open": [100.0, 100.0, 101.0],
            "high": [101.0, 102.0, 102.0],
            "low": [80.0, 99.0, 100.0],
            "close": [100.0, 101.0, 101.0],
        }
    )
    path = selection.strict_trade_path(
        market,
        side=1,
        entry_time=pd.Timestamp("2022-01-01 12:10:00", tz="UTC"),
        exit_time=pd.Timestamp("2022-01-01 12:20:00", tz="UTC"),
        funding=[],
        start_equity=1.0,
        cost_rate=0.0,
    )

    assert path["strict_drawdown"] >= 0.20
    assert abs(path["end_equity"] - 1.01) < 1e-12


def test_selection_order_uses_ratio_then_stress_then_trades_then_id() -> None:
    rows = [
        {
            "candidate_id": "B",
            "base": {"cagr_to_strict_mdd": 1.0},
            "stress": {"absolute_return": 0.10},
            "closed_trades": 30,
        },
        {
            "candidate_id": "A",
            "base": {"cagr_to_strict_mdd": 1.0},
            "stress": {"absolute_return": 0.10},
            "closed_trades": 30,
        },
    ]
    assert selection.rank_candidates(rows)[0]["candidate_id"] == "A"
