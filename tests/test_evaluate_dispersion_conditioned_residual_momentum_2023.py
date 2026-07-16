from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import evaluate_dispersion_conditioned_residual_momentum_2023 as dcrm


def synthetic_bundle(
    *,
    funding: dict[str, pd.DataFrame] | None = None,
    high_multiplier: float = 1.01,
    low_multiplier: float = 0.99,
) -> dcrm.MarketBundle:
    dates = pd.date_range("2023-01-01", periods=13, freq="5min")
    market = {}
    for symbol in ("ADAUSDT", "XRPUSDT"):
        values = np.full(len(dates), 100.0)
        market[symbol] = {
            "open": values.copy(),
            "high": values * high_multiplier,
            "low": values * low_multiplier,
            "close": values.copy(),
        }
    if funding is None:
        funding = {
            symbol: pd.DataFrame(columns=["event_time", "funding_rate"])
            for symbol in market
        }
    return dcrm.MarketBundle(dates=dates, market=market, funding=funding, source_hashes={})


def synthetic_clock(*, gross_scale: float = 0.25) -> pd.DataFrame:
    decision = pd.Timestamp("2023-01-01 00:00")
    return pd.DataFrame(
        [
            {
                "decision_time": decision,
                "entry_time": decision + pd.Timedelta(minutes=5),
                "exit_time": decision + pd.Timedelta(minutes=55),
                "long_symbol": "ADAUSDT",
                "short_symbol": "XRPUSDT",
                "long_weight": gross_scale / 2,
                "short_weight_abs": gross_scale / 2,
                "base_long_weight": 0.5,
                "base_short_weight_abs": 0.5,
                "gross_scale": gross_scale,
                "long_beta": 1.0,
                "short_beta": 1.0,
            }
        ]
    )


def simulate_synthetic(
    bundle: dcrm.MarketBundle,
    clock: pd.DataFrame,
    *,
    cost_bp: float = 6.0,
) -> dict:
    return dcrm.simulate(
        bundle,
        clock,
        start=pd.Timestamp("2023-01-01 00:00"),
        end=pd.Timestamp("2023-01-01 01:05"),
        cost_bp=cost_bp,
    )


def test_frozen_support_maps_to_causal_scaled_execution_clock() -> None:
    support, clock = dcrm.verify_support_and_clock()
    execution = dcrm.execution_clock(clock)
    assert support["post_entry_returns_or_pnl_calculated"] is False
    assert len(execution) == 92
    assert (execution["last_feature_time"] < execution["decision_time"]).all()
    assert (execution["decision_time"] < execution["entry_time"]).all()
    assert set(execution["gross_scale"]) == {0.25, 1.0}


def test_quarter_gross_keeps_cash_and_charges_only_active_notional() -> None:
    stats = simulate_synthetic(
        synthetic_bundle(high_multiplier=1.0, low_multiplier=1.0),
        synthetic_clock(gross_scale=0.25),
    )
    assert stats["trades"] == 1
    assert stats["absolute_return_pct"] == pytest.approx(-0.03, abs=1e-9)
    assert stats["transaction_cost_pct_initial"] == pytest.approx(0.03, abs=1e-9)


def test_strict_mdd_uses_favorable_before_adverse_held_ohlc() -> None:
    stats = simulate_synthetic(synthetic_bundle(), synthetic_clock())
    assert stats["strict_mdd_pct"] > stats["close_mdd_pct"]
    assert stats["strict_mdd_pct"] > 0.25


def test_funding_interval_excludes_entry_boundary_and_includes_exit_boundary() -> None:
    long_funding = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2023-01-01 00:00", "2023-01-01 00:10", "2023-01-01 00:55"]
            ),
            "funding_rate": [-0.08, -0.08, -0.08],
        }
    )
    funding = {
        "ADAUSDT": long_funding,
        "XRPUSDT": pd.DataFrame(columns=["event_time", "funding_rate"]),
    }
    stats = simulate_synthetic(
        synthetic_bundle(funding=funding, high_multiplier=1.0, low_multiplier=1.0),
        synthetic_clock(),
        cost_bp=0.0,
    )
    assert stats["funding_cash_pct_initial"] == pytest.approx(2.0, abs=1e-9)
    assert stats["absolute_return_pct"] == pytest.approx(2.0, abs=1e-9)


def test_controls_change_only_the_frozen_dimension() -> None:
    base = synthetic_clock()
    flipped = dcrm.transform_clock(base, "direction_flip")
    assert flipped.loc[0, "long_symbol"] == "XRPUSDT"
    assert flipped.loc[0, "short_symbol"] == "ADAUSDT"
    delayed = dcrm.transform_clock(base, "delay_five_minutes")
    assert delayed.loc[0, "decision_time"] == base.loc[0, "decision_time"]
    assert delayed.loc[0, "entry_time"] == base.loc[0, "entry_time"] + pd.Timedelta(minutes=5)
    full = dcrm.transform_clock(base, "full_gross")
    assert full.loc[0, "gross_scale"] == 1.0
    assert full.loc[0, "long_weight"] + full.loc[0, "short_weight_abs"] == 1.0
    inverted = dcrm.transform_clock(base, "inverted_dispersion_scale")
    assert inverted.loc[0, "gross_scale"] == 1.0


def test_selection_checks_match_preregistered_2023_gate() -> None:
    annual = {
        "absolute_return_pct": 10.0,
        "cagr_to_strict_mdd": 2.1,
        "strict_mdd_pct": 5.0,
        "trades": 38,
        "cagr_pct": 10.0,
    }
    half = {**annual, "trades": 18}
    primary = {"2023": annual, "2023_h1": half, "2023_h2": half}
    checks = dcrm.selection_checks(
        primary,
        {"absolute_return_pct": 1.0},
        {"absolute_return_pct": 1.0},
        {"cagr_pct": 0.0},
        {"raw_p_value": 0.05},
    )
    assert len(checks) == 10
    assert all(checks.values())


def test_loader_contract_physically_stops_before_2024() -> None:
    assert dcrm.MARKET_ROWS_2023 == 105_120
    assert dcrm.FUNDING_ROWS_2023 == 1_095
    assert dcrm.END == pd.Timestamp("2024-01-01")
    assert "2024" not in str(dcrm.DEFAULT_OUTPUT)
    assert str(dcrm.EXECUTION_SOURCE_DIR).endswith("dcrm_2023_execution")
    source = open(dcrm.EVALUATION_SOURCE).read()
    loader = source.split("def load_bundle_2023", 1)[1].split("def _funding_events_by_bar", 1)[0]
    assert "2023_2024" not in loader
    assert "nrows=" not in loader


def test_invalid_freeze_stops_before_outcome_load(tmp_path, monkeypatch) -> None:
    loaded = False

    def forbidden_load(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("outcomes must not load")

    monkeypatch.setattr(dcrm, "verify_support_and_clock", lambda: ({}, pd.DataFrame()))
    monkeypatch.setattr(
        dcrm,
        "verify_evaluation_freeze",
        lambda: (_ for _ in ()).throw(RuntimeError("freeze invalid")),
    )
    monkeypatch.setattr(dcrm, "load_bundle_2023", forbidden_load)
    with pytest.raises(RuntimeError, match="freeze invalid"):
        dcrm.run(tmp_path / "result.json", tmp_path / "result.md")
    assert loaded is False
