from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import evaluate_post_funding_crowding_release_episode_v2_2023_2024 as pfcr2


def synthetic_bundle() -> pfcr2.lore.MarketBundle:
    dates = pd.date_range("2023-01-01", periods=6, freq="5min")
    market = {}
    for symbol in ("ADAUSDT", "XRPUSDT"):
        values = np.full(len(dates), 100.0)
        market[symbol] = {
            "open": values.copy(),
            "high": values * 1.01,
            "low": values * 0.99,
            "close": values.copy(),
        }
    funding = {
        symbol: pd.DataFrame(columns=["event_time", "funding_rate"])
        for symbol in market
    }
    return pfcr2.lore.MarketBundle(dates, market, funding, {})


def synthetic_clock() -> pd.DataFrame:
    settlement = pd.Timestamp("2023-01-01 00:00")
    return pd.DataFrame(
        [
            {
                "policy_id": "PFCR02",
                "settlement_time": settlement,
                "signal_time": settlement,
                "feature_available_time": settlement + pd.Timedelta(minutes=5),
                "entry_time": settlement + pd.Timedelta(minutes=10),
                "exit_time": settlement + pd.Timedelta(minutes=25),
                "long_symbol": "ADAUSDT",
                "short_symbol": "XRPUSDT",
                "long_weight": 0.4,
                "short_weight_abs": 0.6,
                "long_beta": 1.5,
                "short_beta": 1.0,
                "choice": "crowding_release",
                "gross_scale": 1.0,
                "predicted_edge": 0.001,
                "confidence_threshold": 0.001,
                "current_funding_spread": 0.002,
                "prior_spread_q90": 0.001,
            }
        ]
    )


def test_frozen_support_maps_to_causal_execution_clock() -> None:
    support, clock = pfcr2.verify_support_and_clock()
    execution = pfcr2.execution_clock(clock)
    assert support["post_entry_returns_calculated"] is False
    assert len(execution) == 82
    assert (execution["signal_time"] < execution["feature_available_time"]).all()
    assert (execution["feature_available_time"] < execution["entry_time"]).all()
    assert execution["choice"].eq("crowding_release").all()


def test_strict_wrapper_preserves_favorable_before_adverse_mdd() -> None:
    stats = pfcr2._simulate(
        synthetic_bundle(),
        synthetic_clock(),
        pfcr2.START,
        pfcr2.MID,
        cost_bp=6.0,
    )
    assert stats["trades"] == 1
    assert stats["strict_mdd_pct"] > stats["close_mdd_pct"]
    assert stats["strict_mdd_pct"] > 1.0
    assert stats["calendar_start"] == "2023-01-01"
    assert stats["calendar_end_exclusive"] == "2024-01-01"


def test_controls_preserve_pair_bundle_and_shift_only_declared_fields() -> None:
    base = synthetic_clock()
    flipped = pfcr2.transform_clock(base, "direction_flip")
    assert flipped.loc[0, "long_symbol"] == "XRPUSDT"
    assert flipped.loc[0, "short_symbol"] == "ADAUSDT"
    assert flipped.loc[0, "long_weight"] == pytest.approx(0.6)
    assert flipped.loc[0, "short_weight_abs"] == pytest.approx(0.4)
    delayed = pfcr2.transform_clock(base, "delay_five_minutes")
    assert delayed.loc[0, "signal_time"] == base.loc[0, "signal_time"]
    assert delayed.loc[0, "entry_time"] == base.loc[0, "entry_time"] + pd.Timedelta(minutes=5)
    fake = pfcr2.transform_clock(base, "fake_settlement_plus_four_hours")
    assert fake.loc[0, "signal_time"] == base.loc[0, "signal_time"] + pd.Timedelta(hours=4)
    assert fake.loc[0, "exit_time"] == base.loc[0, "exit_time"] + pd.Timedelta(hours=4)


def test_selection_checks_match_preregistered_gate_exactly() -> None:
    good_year = {
        "absolute_return_pct": 10.0,
        "cagr_to_strict_mdd": 2.0,
        "strict_mdd_pct": 5.0,
        "trades": 40,
        "cagr_pct": 10.0,
    }
    combined = {**good_year, "cagr_to_strict_mdd": 3.1, "trades": 80}
    primary = {"2023": good_year, "2024": good_year, "combined_2023_2024": combined}
    stress = {"absolute_return_pct": 1.0}
    delay = {"absolute_return_pct": 1.0}
    opposite = {"cagr_pct": 0.0}
    signflip = {"raw_p_value": 0.05}
    checks = pfcr2.selection_checks(primary, stress, delay, opposite, signflip)
    assert len(checks) == 11
    assert all(checks.values())
    bad = {key: dict(value) for key, value in primary.items()}
    bad["2024"]["absolute_return_pct"] = -0.1
    assert not all(
        pfcr2.selection_checks(bad, stress, delay, opposite, signflip).values()
    )


def test_evaluator_does_not_default_to_any_2025_or_2026_source() -> None:
    assert pfcr2.END == pd.Timestamp("2025-01-01")
    assert "2025" not in str(pfcr2.CLOCK_PATH)
    assert "2026" not in str(pfcr2.CLOCK_PATH)
    assert pfcr2.lore.END == pfcr2.END


def test_invalid_freeze_stops_before_outcome_bundle_load(tmp_path, monkeypatch) -> None:
    loaded = False

    def forbidden_load():
        nonlocal loaded
        loaded = True
        raise AssertionError("outcome bundle must not load")

    monkeypatch.setattr(pfcr2, "verify_support_and_clock", lambda: ({}, pd.DataFrame()))
    monkeypatch.setattr(
        pfcr2,
        "verify_evaluation_freeze",
        lambda: (_ for _ in ()).throw(RuntimeError("freeze invalid")),
    )
    monkeypatch.setattr(pfcr2, "load_bundle", forbidden_load)
    with pytest.raises(RuntimeError, match="freeze invalid"):
        pfcr2.run(tmp_path / "result.json", tmp_path / "result.md")
    assert loaded is False
