import numpy as np
import pandas as pd
import pytest

import training.search_funding_premium_oi_transition_gate_alpha as module
from training.search_funding_premium_oi_transition_gate_alpha import (
    LR_IMPACT_LOWER,
    LR_IMPACT_UPPER,
    OITransitionGateConfig,
    _apply_policy,
    _build_base_components,
    _candidate_specs,
    _completed_hourly_oi,
    _fit_state_thresholds,
    _manifest_core_hash,
    _select_top,
    _validate_manifest,
)


def _cfg(tmp_path) -> OITransitionGateConfig:
    return OITransitionGateConfig(
        input_csv="market.csv",
        metrics_csv="metrics.csv",
        funding_csv="funding.csv",
        premium_csv="premium.csv",
        output=str(tmp_path / "out.json"),
        manifest_output=str(tmp_path / "manifest.json"),
    )


def test_completed_hourly_oi_is_exposed_only_after_a_full_source_hour():
    dates = pd.date_range("2022-01-01", periods=30, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "sum_open_interest": np.arange(100.0, 130.0),
            "positioning_source_time": dates - pd.Timedelta("5min"),
            "oi_available": 1.0,
        }
    )

    hourly = _completed_hourly_oi(market)

    assert hourly.loc[0, "effective_time"] == pd.Timestamp("2022-01-01 01:00:00")
    assert hourly.loc[0, "log_oi"] == pytest.approx(np.log(111.0))
    assert hourly.loc[1, "log_oi"] == pytest.approx(np.log(123.0))
    assert np.isnan(hourly.loc[2, "log_oi"]), "a partial source hour must not become usable"


def test_state_thresholds_use_only_frozen_fit_window():
    fit_times = pd.date_range("2021-04-15", periods=20, freq="1h")
    future_times = pd.date_range("2024-01-01", periods=20, freq="1h")
    hourly = pd.DataFrame(
        {
            "effective_time": fit_times.append(future_times),
            "oi_logchg24": np.r_[np.arange(20.0), np.full(20, 1e9)],
            "oi_z168": np.r_[np.arange(20.0) / 10.0, np.full(20, -1e9)],
        }
    )

    thresholds = _fit_state_thresholds(hourly, min_observations=10)

    assert thresholds["oi_logchg24_q70"] < 20.0
    assert thresholds["oi_z168_median"] > 0.0
    assert thresholds["change_observations"] == 20


def test_fixed_base_applies_lr_gate_to_funding_only(monkeypatch):
    n = 100
    market = pd.DataFrame(
        {
            "close": np.linspace(100.0, 110.0, n),
            "funding_rate": -1.0,
            "premium_index_change": np.r_[np.zeros(96), -1.0, -1.0, np.zeros(2)],
            "funding_available": 1.0,
            "premium_available": 1.0,
        }
    )
    monkeypatch.setattr(
        module,
        "_completed_timeframe_features",
        lambda *args, **kwargs: pd.DataFrame({"htf_1d_return_4": np.r_[np.zeros(96), 1.0, 1.0, 0.0, 0.0]}),
    )
    monkeypatch.setattr(
        module,
        "build_liquidity_features",
        lambda *args, **kwargs: pd.DataFrame(
            {"lr_impact_72": np.r_[np.full(96, (LR_IMPACT_LOWER + LR_IMPACT_UPPER) / 2.0), 99.0, 99.0, 99.0, 99.0]}
        ),
    )

    funding, premium, _ = _build_base_components(market)

    assert funding[95]
    assert not funding[96]
    assert premium[96:98].tolist() == [True, True]


def test_policy_requires_oi_for_targeted_leg_but_preserves_untargeted_premium():
    funding = np.array([True, True, False, False])
    premium = np.array([False, False, True, True])
    features = pd.DataFrame(
        {
            "oi_transition3": [1.0, 2.0, 1.0, np.nan],
            "oi_transition6": [1.0, 2.0, 1.0, np.nan],
            "oi_transition_available": [1.0, 1.0, 1.0, 0.0],
        }
    )
    spec = {
        "schema_states": 3,
        "mode": "allow_top_positive",
        "allowed_states": [1],
        "vetoed_states": [],
        "target_component": "funding",
    }

    active = _apply_policy(funding, premium, features, spec)

    assert active.tolist() == [True, False, True, True]


def test_candidate_search_is_bounded_to_declared_policy_family(monkeypatch, tmp_path):
    def fake_scores(*args, schema_states, **kwargs):
        rows = []
        for state in range(schema_states * schema_states):
            positive = state < (schema_states * schema_states // 2)
            rows.append(
                {
                    "state": state,
                    "previous": state // schema_states,
                    "current": state % schema_states,
                    "trades": 9,
                    "mean_trade_return_pct": 0.5 if positive else -0.5,
                    "median_trade_return_pct": 0.3 if positive else -0.3,
                }
            )
        return rows

    monkeypatch.setattr(module, "_state_score_table", fake_scores)
    cfg = _cfg(tmp_path)
    specs = _candidate_specs(
        pd.DataFrame(),
        pd.Series(dtype="datetime64[ns]"),
        np.array([], dtype=bool),
        np.array([], dtype=bool),
        pd.DataFrame(),
        cfg,
    )

    assert 0 < len(specs) <= 36
    assert {spec["schema_states"] for spec in specs} == {3, 6}
    assert {spec["target_component"] for spec in specs} == {"funding", "all"}
    assert {spec["k"] for spec in specs}.issubset({3, 5, 8})


def test_selection_is_deterministic_and_manifest_mutation_is_rejected():
    def row(schema, mode, target, k, score):
        return {
            "schema_states": schema,
            "mode": mode,
            "target_component": target,
            "k": k,
            "selection_score": score,
            "selection_stats": {"select_2023": {"ratio": score, "return_pct": score}},
        }

    rows = [
        row(3, "allow_top_positive", "funding", 3, 4.0),
        row(3, "allow_top_positive", "all", 5, 3.0),
        row(3, "allow_top_positive", "all", 8, 2.0),
        row(6, "veto_bottom_negative", "funding", 3, 1.0),
    ]
    first = _select_top(rows, top_n=4, top_per_schema_mode=2)
    second = _select_top(list(reversed(rows)), top_n=4, top_per_schema_mode=2)
    assert first == second
    assert len(first) == 3

    core = {"protocol": {}, "selected": []}
    manifest = {"as_of": "now", "sha256": _manifest_core_hash(core), **core}
    _validate_manifest(manifest)
    manifest["selected"].append({"future": "leak"})
    with pytest.raises(RuntimeError, match="frozen SHA-256"):
        _validate_manifest(manifest)
