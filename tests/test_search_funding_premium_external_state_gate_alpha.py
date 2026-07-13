import numpy as np
import pandas as pd
import pytest

from training.search_funding_premium_external_state_gate_alpha import (
    BASE_ADMISSION_FEATURES,
    ExternalStateGateConfig,
    _manifest_core_hash,
    _read_premium_before,
    _select_top,
    _source_hashes,
    _validate_manifest,
    external_gate_mask,
    feature_admission,
)


def _base_frame(n: int, trend: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "btc_funding_rate": trend,
            "btc_premium_index_change": np.sin(np.arange(n) * 0.17),
            "btc_trend_96": np.cos(np.arange(n) * 0.11),
            "btc_daily_mom4": np.sin(np.arange(n) * 0.07),
            "btc_lr_impact_72": np.cos(np.arange(n) * 0.041),
        }
    )


def test_feature_admission_rejects_fit_spearman_correlated_external_state():
    n = 500
    trend = pd.Series(np.linspace(-1.0, 1.0, n))
    base = _base_frame(n, trend)
    features = pd.DataFrame(
        {
            "copied_btc_funding": trend,
            "low_corr_cycle": np.sin(np.arange(n) * 0.313),
        }
    )
    availability = {"copied_btc_funding": "positioning_available", "low_corr_cycle": "dvol_available"}

    admitted, audit = feature_admission(
        features,
        base,
        availability,
        np.ones(n, dtype=bool),
        max_abs_spearman=0.30,
        min_observations=300,
    )

    assert "copied_btc_funding" not in admitted
    assert "low_corr_cycle" in admitted
    assert audit["copied_btc_funding"]["max_abs_spearman"] > 0.99
    assert set(audit["low_corr_cycle"]["pair_counts"]) == set(BASE_ADMISSION_FEATURES)


def test_external_gate_mask_requires_finite_feature_and_specific_availability():
    features = pd.DataFrame({"dvol_z2016": [-2.0, -2.0, np.nan, 0.0]})
    availability_frame = pd.DataFrame({"dvol_available": [1.0, 0.0, 1.0, 1.0]})
    spec = {"feature": "dvol_z2016", "gate_mode": "lower", "lower": -1.0, "upper": 1.0}

    mask = external_gate_mask(features, spec, availability_frame, {"dvol_z2016": "dvol_available"})

    assert mask.tolist() == [True, False, False, False]


def test_source_hashes_detect_input_mutation(tmp_path):
    paths = {}
    for name in ("input", "metrics", "dvol", "funding", "premium"):
        path = tmp_path / f"{name}.csv"
        path.write_text(f"{name},1\n")
        paths[name] = path
    cfg = ExternalStateGateConfig(
        input_csv=str(paths["input"]),
        metrics_csv=str(paths["metrics"]),
        dvol_csv=str(paths["dvol"]),
        funding_csv=str(paths["funding"]),
        premium_csv=str(paths["premium"]),
        output=str(tmp_path / "out.json"),
        manifest_output=str(tmp_path / "manifest.json"),
    )
    before = _source_hashes(cfg)

    paths["dvol"].write_text("dvol,2\n")
    after = _source_hashes(cfg)

    assert before[str(paths["dvol"])] != after[str(paths["dvol"])]
    for key in ("input", "metrics", "funding", "premium"):
        assert before[str(paths[key])] == after[str(paths[key])]


def test_premium_reader_physically_truncates_future_rows(tmp_path):
    path = tmp_path / "premium.csv"
    pd.DataFrame(
        {
            "close_time": [
                int(pd.Timestamp("2023-12-31 23:59:59", tz="UTC").timestamp() * 1000),
                int(pd.Timestamp("2024-01-01 00:59:59", tz="UTC").timestamp() * 1000),
            ],
            "close": [0.1, 99.0],
        }
    ).to_csv(path, index=False)

    loaded = _read_premium_before(str(path), "2024-01-01")

    assert loaded["close"].tolist() == [0.1]


def test_selection_stability_is_deterministic_and_top_per_feature_bounded():
    def row(feature, score, ratio, ret, gate="lower"):
        return {
            "feature": feature,
            "gate_mode": gate,
            "target_component": "all",
            "selection_score": score,
            "selection_stats": {"select_2023": {"ratio": ratio, "return_pct": ret}},
        }

    rows = [
        row("oi_z2016", 2.0, 3.0, 4.0, "upper"),
        row("oi_z2016", 1.9, 9.0, 9.0, "lower"),
        row("oi_z2016", 1.8, 8.0, 8.0, "outer"),
        row("dvol_z2016", 1.95, 2.0, 2.0, "lower"),
        row("pos_smart_size_z144", 1.7, 10.0, 10.0, "lower"),
    ]

    first = _select_top(rows, top_n=4, top_per_feature=2)
    second = _select_top(list(reversed(rows)), top_n=4, top_per_feature=2)

    assert first == second
    assert [item["feature"] for item in first].count("oi_z2016") == 2
    assert [item["feature"] for item in first] == ["oi_z2016", "dvol_z2016", "oi_z2016", "pos_smart_size_z144"]


def test_manifest_validation_detects_mutation():
    core = {
        "protocol": {"threshold_fit": ["2021-04-15", "2023-01-01"]},
        "source_hashes": {},
        "selected": [],
    }
    manifest = {"as_of": "now", "sha256": _manifest_core_hash(core), **core}

    _validate_manifest(manifest)
    manifest["selected"].append({"feature": "future_leak"})

    with pytest.raises(RuntimeError, match="frozen SHA-256"):
        _validate_manifest(manifest)
