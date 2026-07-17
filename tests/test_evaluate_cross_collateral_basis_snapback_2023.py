from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.evaluate_cross_collateral_basis_snapback_2023 import (
    cm_inverse_pnl_usd,
    monthly_signflip_pvalue,
    run_ledger,
    select_exit,
    summarize_ledger,
    um_pnl_usd,
)


def _market() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    time = pd.date_range("2023-01-01", periods=12, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open_time": time,
            "available_time": time + pd.Timedelta(minutes=5),
            "um_open": np.linspace(100.0, 101.1, len(time)),
            "um_high": np.linspace(100.2, 101.3, len(time)),
            "um_low": np.linspace(99.8, 100.9, len(time)),
            "um_close": np.linspace(100.1, 101.2, len(time)),
            "um_ohlc_valid": True,
            "cm_open": np.linspace(99.0, 100.1, len(time)),
            "cm_high": np.linspace(99.2, 100.3, len(time)),
            "cm_low": np.linspace(98.8, 99.9, len(time)),
            "cm_close": np.linspace(99.1, 100.2, len(time)),
            "cm_ohlc_valid": True,
            "source_complete": True,
            "delivery_time": pd.Timestamp("2023-03-31 08:00", tz="UTC"),
            "contract_segment": "20230331",
        }
    )
    features = frame[["open_time"]].copy()
    features["zscore"] = [2.0, 2.0, 2.0, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    event = {
        "open_time": time[0].isoformat(),
        "entry_time": time[2].isoformat(),
        "maximum_exit_time": time[8].isoformat(),
        "delivery_time": pd.Timestamp("2023-03-31 08:00", tz="UTC").isoformat(),
        "contract_segment": "20230331",
        "zscore": 2.0,
    }
    return frame, features, [event]


def test_inverse_and_linear_leg_pnl_have_expected_sign_and_scale() -> None:
    assert um_pnl_usd(side=1, quantity_btc=1.0, entry_price=100.0, mark_price=110.0) == 10.0
    assert um_pnl_usd(side=-1, quantity_btc=1.0, entry_price=100.0, mark_price=90.0) == 10.0
    long_cm = cm_inverse_pnl_usd(
        side=1,
        contracts=1.0,
        multiplier_usd=100.0,
        entry_price=100.0,
        mark_price=110.0,
    )
    short_cm = cm_inverse_pnl_usd(
        side=-1,
        contracts=1.0,
        multiplier_usd=100.0,
        entry_price=100.0,
        mark_price=90.0,
    )
    assert long_cm == pytest.approx(10.0)
    assert short_cm == pytest.approx(10.0)


def test_normalization_exit_waits_a_full_bar_after_availability() -> None:
    _, features, events = _market()
    decision = select_exit(events[0], features, normalization_z=0.5, maximum_hold_bars=6)
    assert decision.trigger_open_time == features.iloc[3]["open_time"]
    assert decision.exit_time == features.iloc[3]["open_time"] + pd.Timedelta(minutes=10)
    assert decision.reason == "normalization"


def test_ledger_charges_both_legs_and_marks_strict_path() -> None:
    frame, features, events = _market()
    result = run_ledger(
        frame,
        features,
        events,
        cost_rate=0.001,
        normalization_z=0.5,
        maximum_hold_bars=6,
        cm_multiplier=100.0,
    )
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["entry_fee"] == pytest.approx(0.001)
    assert trade["exit_fee"] > 0.0
    assert result["transaction_cost"] == pytest.approx(trade["entry_fee"] + trade["exit_fee"])
    assert result["strict_mdd"] > 0.0


def test_ledger_fails_closed_on_incomplete_held_path() -> None:
    frame, features, events = _market()
    frame.loc[3, "source_complete"] = False
    with pytest.raises(ValueError, match="incomplete source row"):
        run_ledger(
            frame,
            features,
            events,
            cost_rate=0.001,
            normalization_z=0.5,
            maximum_hold_bars=6,
            cm_multiplier=100.0,
        )


def test_favorable_before_adverse_cross_venue_marks_raise_strict_mdd() -> None:
    time = pd.date_range("2023-01-01", periods=5, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open_time": time,
            "um_open": 100.0,
            "um_high": [100.0, 110.0, 100.0, 100.0, 100.0],
            "um_low": [100.0, 90.0, 100.0, 100.0, 100.0],
            "cm_open": 100.0,
            "cm_high": [100.0, 110.0, 100.0, 100.0, 100.0],
            "cm_low": [100.0, 90.0, 100.0, 100.0, 100.0],
            "source_complete": True,
            "um_ohlc_valid": True,
            "cm_ohlc_valid": True,
            "contract_segment": "20230331",
        }
    )
    features = pd.DataFrame({"open_time": time, "zscore": 2.0})
    events = [
        {
            "open_time": time[0].isoformat(),
            "entry_time": time[1].isoformat(),
            "maximum_exit_time": time[3].isoformat(),
            "delivery_time": pd.Timestamp("2023-03-31 08:00", tz="UTC").isoformat(),
            "contract_segment": "20230331",
            "zscore": 2.0,
        }
    ]
    result = run_ledger(
        frame,
        features,
        events,
        cost_rate=0.0,
        normalization_z=0.5,
        maximum_hold_bars=2,
        cm_multiplier=100.0,
    )
    # z>0 shorts UM and longs CM: the combined favorable extrema mark 1.10,
    # then the adversarial extrema mark 0.90 in the same held bar.
    assert result["strict_mdd"] == pytest.approx(1.0 - 0.90 / 1.10)


def test_monthly_signflip_is_exact_and_excludes_empty_months() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                [
                    "2023-01-01T00:00:00Z",
                    "2023-01-10T00:00:00Z",
                    "2023-02-01T00:00:00Z",
                    "2023-03-01T00:00:00Z",
                ]
            ),
            "net_return": [0.01, 0.02, 0.03, -0.01],
        }
    )
    pvalue, months, sums = monthly_signflip_pvalue(trades)
    assert months == 3
    assert sums == {"2023-01": 0.03, "2023-02": 0.03, "2023-03": -0.01}
    assert pvalue == 0.25


def test_summary_annualizes_full_declared_wall_clock() -> None:
    ledger = {
        "ending_equity": 1.10,
        "strict_mdd": 0.05,
        "pre_cost_pnl": 0.12,
        "transaction_cost": 0.02,
        "trades": [
            {
                "entry_time": "2023-01-01T00:00:00+00:00",
                "net_return": 0.10,
                "rich_leg": "um",
                "signed_wedge_convergence": 0.01,
            }
        ],
    }
    summary = summarize_ledger(
        ledger,
        period_start=pd.Timestamp("2023-01-01", tz="UTC"),
        period_end=pd.Timestamp("2024-01-01", tz="UTC"),
    )
    expected = (1.10 ** (365.25 / 365.0) - 1.0) * 100.0
    assert summary["absolute_return_pct"] == pytest.approx(10.0)
    assert summary["cagr_pct"] == pytest.approx(expected)
    assert summary["strict_mdd_pct"] == pytest.approx(5.0)
