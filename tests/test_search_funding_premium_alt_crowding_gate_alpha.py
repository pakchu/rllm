import json

import numpy as np
import pandas as pd
import pytest

from training import search_funding_premium_alt_crowding_gate_alpha as mod
from training.search_funding_premium_alt_crowding_gate_alpha import (
    AltCrowdingGateConfig,
    _manifest_core_hash,
    _selection_score,
    _source_prefix_hashes,
    _validate_manifest,
    alt_gate_mask,
)


def _stats(cagr: float = 8.0, ratio: float = 2.0, trades: int = 20, mdd: float = 5.0) -> dict:
    return {"cagr_pct": cagr, "ratio": ratio, "trades": trades, "strict_mdd_pct": mdd}


def test_selection_score_uses_fit_and_2023h2_only():
    cfg = AltCrowdingGateConfig(
        input_csv="x",
        aux_dir="a",
        btc_funding_csv="f",
        btc_premium_csv="p",
        output="o",
        manifest_output="m",
    )
    stable = {"fit_2023_h1": _stats(trades=20), "select_2023_h2": _stats(ratio=2.4, trades=24)}
    assert _selection_score(stable, cfg) > -1e11

    bad = {key: dict(value) for key, value in stable.items()}
    bad["select_2023_h2"]["trades"] = cfg.min_select_trades - 1
    assert _selection_score(bad, cfg) <= -1e11

    bad = {key: dict(value) for key, value in stable.items()}
    bad["fit_2023_h1"]["cagr_pct"] = -0.1
    assert _selection_score(bad, cfg) <= -1e11


def test_alt_gate_mask_requires_feature_specific_availability():
    features = pd.DataFrame(
        {
            "alt_funding_median_z2016": [-2.0, -2.0, 0.0, -2.0],
            "alt_crowding_concordance": [2.0, 2.0, 2.0, 0.0],
        }
    )
    availability = pd.DataFrame(
        {
            "alt_funding_available": [1.0, 0.0, 1.0, 1.0],
            "alt_premium_available": [1.0, 1.0, 1.0, 1.0],
            "alt_derivatives_available": [1.0, 0.0, 1.0, 1.0],
        }
    )
    funding_spec = {"feature": "alt_funding_median_z2016", "lower": -1.0, "upper": 1.0, "gate_mode": "lower"}
    assert alt_gate_mask(features, funding_spec, availability).tolist() == [True, False, False, True]

    both_spec = {"feature": "alt_crowding_concordance", "lower": -1.0, "upper": 1.0, "gate_mode": "upper"}
    assert alt_gate_mask(features, both_spec, availability).tolist() == [True, False, True, False]


def test_manifest_validation_detects_mutation_and_run_reuses_existing_manifest(tmp_path, monkeypatch):
    core = {"protocol": {"selection": "2023H2"}, "feature_admission_audit": {}, "selected": []}
    manifest = {"as_of": "now", "sha256": _manifest_core_hash(core), **core}
    _validate_manifest(manifest)
    mutated = json.loads(json.dumps(manifest))
    mutated["selected"].append({"future": True})
    with pytest.raises(RuntimeError, match="frozen SHA-256"):
        _validate_manifest(mutated)

    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    cfg = AltCrowdingGateConfig(
        input_csv="x",
        aux_dir="a",
        btc_funding_csv="f",
        btc_premium_csv="p",
        output=str(output_path),
        manifest_output=str(manifest_path),
    )
    called = {"select": 0}

    def fail_select(_cfg):
        called["select"] += 1
        raise AssertionError("manifest should have been reused")

    def fake_replay(_cfg, loaded):
        return {
            "manifest_sha256": loaded["sha256"],
            "protocol": loaded["protocol"],
            "selected": [],
            "alpha_pool_qualifiers": [],
            "live_grade": [],
        }

    monkeypatch.setattr(mod, "_select_manifest", fail_select)
    monkeypatch.setattr(mod, "_replay", fake_replay)
    report = mod.run(cfg)
    assert called["select"] == 0
    assert report["manifest_sha256"] == manifest["sha256"]
    assert json.loads(output_path.read_text())["manifest_sha256"] == manifest["sha256"]


def _write_csv(path, frame):
    frame.to_csv(path, index=False)


def _premium_frame(values):
    close_times = pd.to_datetime(["2023-12-31 23:00", "2024-01-01 00:00"], utc=True)
    return pd.DataFrame({"close_time": (close_times.view("int64") // 1_000_000).astype("int64"), "close": values})


def test_source_prefix_hashes_ignore_post_2023_mutations_but_track_prefix(tmp_path):
    market = pd.DataFrame(
        {
            "date": ["2023-12-31 23:55", "2024-01-01 00:00"],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [10.0, 20.0],
        }
    )
    funding = pd.DataFrame({"date": ["2023-12-31 16:00", "2024-01-01 00:00"], "funding_rate": [0.1, 99.0]})
    input_csv = tmp_path / "market.csv"
    btc_funding = tmp_path / "btc_funding.csv"
    btc_premium = tmp_path / "btc_premium.csv"
    aux_dir = tmp_path / "aux"
    aux_dir.mkdir()
    _write_csv(input_csv, market)
    _write_csv(btc_funding, funding)
    _write_csv(btc_premium, _premium_frame([1.0, 99.0]))
    for symbol in mod.SYMBOLS:
        _write_csv(aux_dir / f"{symbol}_funding_test.csv.gz", funding)
        _write_csv(aux_dir / f"{symbol}_premium_1h_test.csv.gz", _premium_frame([1.0, 99.0]))

    cfg = AltCrowdingGateConfig(
        input_csv=str(input_csv),
        aux_dir=str(aux_dir),
        btc_funding_csv=str(btc_funding),
        btc_premium_csv=str(btc_premium),
        output="o",
        manifest_output="m",
    )
    original = _source_prefix_hashes(cfg)

    market.loc[1, "close"] = 12345.0
    _write_csv(input_csv, market)
    _write_csv(btc_funding, pd.DataFrame({"date": ["2023-12-31 16:00", "2024-01-01 00:00"], "funding_rate": [0.1, 12345.0]}))
    _write_csv(btc_premium, _premium_frame([1.0, 12345.0]))
    _write_csv(aux_dir / f"{mod.SYMBOLS[0]}_funding_test.csv.gz", pd.DataFrame({"date": ["2023-12-31 16:00", "2024-01-01 00:00"], "funding_rate": [0.1, 12345.0]}))
    assert _source_prefix_hashes(cfg) == original

    market.loc[0, "close"] = 7.0
    _write_csv(input_csv, market)
    changed = _source_prefix_hashes(cfg)
    assert changed != original
    assert changed["market"] != original["market"]
