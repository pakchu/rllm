from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import evaluate_crrc_2023 as crrc


def synthetic_bundle(
    *,
    funding: pd.DataFrame | None = None,
    high_multiplier: float = 1.01,
    low_multiplier: float = 0.99,
) -> crrc.MarketBundle:
    dates = pd.date_range("2023-01-01", periods=15, freq="5min")
    values = np.full(len(dates), 100.0)
    if funding is None:
        funding = pd.DataFrame(columns=["event_time", "funding_rate"])
    return crrc.MarketBundle(
        dates=dates,
        open=values.copy(),
        high=values * high_multiplier,
        low=values * low_multiplier,
        close=values.copy(),
        funding=funding,
        source_hashes={},
    )


def synthetic_clock(*, side: int = 1) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=15, freq="5min")
    return pd.DataFrame(
        [
            {
                "quarter": "q1",
                "signal_position": 0,
                "entry_position": 2,
                "exit_position": 12,
                "signal_date": dates[0],
                "entry_date": dates[2],
                "exit_date": dates[12],
                "side": side,
                "hold_bars": 10,
            }
        ]
    )


def simulate_synthetic(
    bundle: crrc.MarketBundle,
    clock: pd.DataFrame,
    *,
    cost_bp: float = 6.0,
) -> dict:
    return crrc.simulate(
        bundle,
        clock,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-01 01:15"),
        cost_bp=cost_bp,
    )


def test_half_gross_keeps_cash_and_charges_active_notional_only() -> None:
    stats = simulate_synthetic(
        synthetic_bundle(high_multiplier=1.0, low_multiplier=1.0),
        synthetic_clock(),
    )
    assert stats["trades"] == 1
    assert stats["absolute_return_pct"] == pytest.approx(-0.06, abs=1e-9)
    assert stats["transaction_cost_pct_initial"] == pytest.approx(0.06, abs=1e-9)


def test_strict_mdd_uses_global_peak_and_favorable_before_adverse_ohlc() -> None:
    stats = simulate_synthetic(synthetic_bundle(), synthetic_clock())
    assert stats["strict_mdd_pct"] > stats["close_mdd_pct"]
    assert stats["strict_mdd_pct"] > 0.5


def test_funding_excludes_exact_entry_and_includes_post_entry_and_exit() -> None:
    funding = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2023-01-01 00:10:00.000",
                    "2023-01-01 00:10:00.008",
                    "2023-01-01 01:00:00.000",
                ]
            ),
            "funding_rate": [-0.08, -0.08, -0.08],
        }
    )
    stats = simulate_synthetic(
        synthetic_bundle(
            funding=funding, high_multiplier=1.0, low_multiplier=1.0
        ),
        synthetic_clock(),
        cost_bp=0.0,
    )
    assert stats["funding_cash_pct_initial"] == pytest.approx(8.0, abs=1e-9)
    assert stats["absolute_return_pct"] == pytest.approx(8.0, abs=1e-9)


def test_direction_delay_and_side_controls_change_only_frozen_dimension() -> None:
    clock = synthetic_clock()
    flipped = crrc.transform_clock(clock, "direction_flip")
    assert flipped.loc[0, "side"] == -1
    delayed = crrc.transform_clock(clock, "delay_five_minutes")
    assert delayed.loc[0, "signal_date"] == clock.loc[0, "signal_date"]
    assert delayed.loc[0, "entry_date"] == clock.loc[0, "entry_date"] + pd.Timedelta(minutes=5)
    assert delayed.loc[0, "exit_position"] == clock.loc[0, "exit_position"] + 1
    assert len(crrc.transform_clock(clock, "long_only")) == 1
    assert crrc.transform_clock(clock, "short_only").empty


def test_selection_checks_match_every_preregistered_gate() -> None:
    annual = {
        "absolute_return_pct": 10.0,
        "cagr_pct": 10.0,
        "strict_mdd_pct": 3.0,
        "cagr_to_strict_mdd": 3.1,
        "trades": 156,
    }
    primary = {"2023": annual, **{name: {**annual, "trades": 30} for name in ("q1", "q2", "q3", "q4")}}
    checks = crrc.selection_checks(
        primary,
        {"absolute_return_pct": 1.0},
        {"absolute_return_pct": 1.0},
        {"absolute_return_pct": 1.0},
        {"absolute_return_pct": 1.0},
        {"cagr_pct": 0.0},
        {"raw_p_value": 0.05},
    )
    assert len(checks) == 11
    assert all(checks.values())


def test_full_calendar_2023_is_exactly_one_cagr_year() -> None:
    assert crrc.calendar_year_fraction(crrc.START, crrc.END) == 1.0
    equity = 1.25
    years = crrc.calendar_year_fraction(crrc.START, crrc.END)
    assert (equity ** (1.0 / years) - 1.0) * 100.0 == pytest.approx(25.0)


def test_loader_is_physically_2023_only() -> None:
    assert crrc.source_export.MARKET_ROWS == 105_120
    assert crrc.source_export.FUNDING_ROWS == 1_095
    assert crrc.END == pd.Timestamp("2024-01-01")
    source = open(crrc.EVALUATION_SOURCE).read()
    loader = source.split("def load_bundle_2023", 1)[1].split("def transform_clock", 1)[0]
    assert "2020_2023" not in loader
    assert "2021_2023" not in loader
    assert "nrows=" not in loader


def test_invalid_freeze_stops_before_outcome_load(tmp_path, monkeypatch) -> None:
    loaded = False

    def forbidden_load(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("outcomes must not load")

    monkeypatch.setattr(
        crrc,
        "verify_preoutcome_artifacts",
        lambda: ({}, pd.DataFrame(), {}),
    )
    monkeypatch.setattr(
        crrc,
        "verify_evaluation_freeze",
        lambda: (_ for _ in ()).throw(RuntimeError("freeze invalid")),
    )
    monkeypatch.setattr(crrc, "load_bundle_2023", forbidden_load)
    monkeypatch.setattr(crrc, "DEFAULT_OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(crrc, "DEFAULT_DOCS", tmp_path / "result.md")
    with pytest.raises(RuntimeError, match="freeze invalid"):
        crrc.run()
    assert loaded is False


def test_arbitrary_outcome_path_is_rejected_before_loading(tmp_path, monkeypatch) -> None:
    loaded = False

    def forbidden_load(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("outcomes must not load")

    monkeypatch.setattr(crrc, "load_bundle_2023", forbidden_load)
    with pytest.raises(ValueError, match="immutable"):
        crrc.run(tmp_path / "second-open.json", tmp_path / "second-open.md")
    assert loaded is False
