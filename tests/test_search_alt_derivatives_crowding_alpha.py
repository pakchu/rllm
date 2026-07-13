import numpy as np
import pandas as pd
import pytest

from training.search_alt_derivatives_crowding_alpha import (
    AltCrowdingConfig,
    _merge_source,
    _selection_score,
    _validate_manifest,
    feature_admission,
    rule_mask,
)


def _stats(cagr: float = 10.0, ratio: float = 2.0, trades: int = 20, mdd: float = 5.0) -> dict:
    return {
        "cagr_pct": cagr,
        "ratio": ratio,
        "trades": trades,
        "strict_mdd_pct": mdd,
    }


def test_external_merge_is_backward_only_and_staleness_bounded():
    dates = pd.Series(pd.date_range("2023-01-01 00:00", periods=5, freq="30min"))
    source = pd.DataFrame(
        {
            "time": ["2023-01-01 00:00", "2023-01-01 01:00"],
            "value": [1.0, 2.0],
        }
    )
    values, source_times = _merge_source(
        dates,
        source,
        source_time="time",
        value_column="value",
        tolerance="45min",
    )
    assert values.iloc[:4].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert np.isnan(values.iloc[4])
    valid = source_times.notna()
    assert (source_times.loc[valid].to_numpy() <= dates.loc[valid].to_numpy()).all()


def test_external_merge_restores_unsorted_caller_order():
    dates = pd.Series(pd.to_datetime(["2023-01-01 01:00", "2023-01-01 00:00"]))
    source = pd.DataFrame(
        {
            "time": ["2023-01-01 00:00", "2023-01-01 01:00"],
            "value": [1.0, 2.0],
        }
    )
    values, _ = _merge_source(
        dates,
        source,
        source_time="time",
        value_column="value",
        tolerance="65min",
    )
    assert values.tolist() == [2.0, 1.0]


def test_feature_admission_rejects_correlated_input():
    n = 400
    trend = pd.Series(np.linspace(-1.0, 1.0, n))
    base = pd.DataFrame(
        {
            "btc_funding_rate": trend,
            "btc_premium_change": np.sin(np.arange(n)),
            "btc_trend_96": trend,
            "btc_daily_momentum_4": np.cos(np.arange(n)),
            "btc_lr_impact_72": np.sin(np.arange(n) * 0.37),
        }
    )
    features = pd.DataFrame(
        {
            "copied_trend": trend,
            "independent_cycle": np.sin(np.arange(n) * 0.113),
            "alt_derivatives_available": 1.0,
        }
    )
    admitted, audit = feature_admission(
        features,
        base,
        np.ones(n, dtype=bool),
        max_abs_spearman=0.30,
        min_observations=300,
    )
    assert "copied_trend" not in admitted
    assert "independent_cycle" in admitted
    assert audit["copied_trend"]["max_abs_spearman"] > 0.99


def test_rule_mask_requires_availability_and_all_terms():
    features = pd.DataFrame(
        {
            "alt_derivatives_available": [1.0, 1.0, 0.0, 1.0],
            "a": [-2.0, 0.0, -2.0, -2.0],
            "b": [2.0, 2.0, 2.0, 0.0],
        }
    )
    terms = [
        {"feature": "a", "op": "le", "threshold": -1.0},
        {"feature": "b", "op": "ge", "threshold": 1.0},
    ]
    assert rule_mask(features, terms).tolist() == [True, False, False, False]


def test_selection_score_requires_positive_both_halves():
    cfg = AltCrowdingConfig(
        input_csv="x",
        aux_dir="a",
        btc_funding_csv="f",
        btc_premium_csv="p",
        output="o",
        manifest_output="m",
    )
    stable = {"fit_2023_h1": _stats(), "select_2023_h2": _stats()}
    assert _selection_score(stable, cfg) > -1e11
    unstable = {key: dict(value) for key, value in stable.items()}
    unstable["select_2023_h2"]["cagr_pct"] = -1.0
    assert _selection_score(unstable, cfg) <= -1e11


def test_manifest_validation_detects_mutation():
    import hashlib
    import json

    core = {"protocol": {"fit": "past"}, "selected": []}
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True)
    manifest = {"as_of": "now", "sha256": hashlib.sha256(canonical.encode()).hexdigest(), **core}
    _validate_manifest(manifest)
    manifest["selected"].append({"future": True})
    with pytest.raises(RuntimeError, match="frozen SHA-256"):
        _validate_manifest(manifest)
