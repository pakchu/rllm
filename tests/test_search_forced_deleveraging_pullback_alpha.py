from __future__ import annotations

import json

import numpy as np
import pytest

import training.search_forced_deleveraging_pullback_alpha as module
from training.search_forced_deleveraging_pullback_alpha import (
    SPEC,
    Config,
    _freeze_hash,
    _frozen_execution_config,
    _mark_oos_opened,
    _spec_hash,
    _validate_manifest,
    activation_hash,
    build_masks,
    selection_passes,
)


def test_build_masks_requires_all_five_causal_witnesses() -> None:
    features = {
        "flow_accel": np.array([2.0, 2.0, -2.0, -2.0, 2.0]),
        "oi_purge": np.array([2.0, 2.0, 2.0, 2.0, 0.0]),
        "vol_expansion": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        "range_position_288": np.array([-0.4, -0.4, 0.4, 0.4, -0.4]),
        "kimchi_impulse": np.array([1.0, -1.0, -1.0, 1.0, 1.0]),
        "kimchi_available": np.array([True, False, True, True, True]),
    }
    thresholds = {
        "1": {
            "flow_accel": 1.0,
            "oi_purge": 1.0,
            "vol_expansion": 1.0,
            "range_pullback": 0.2,
            "kimchi_impulse": 0.0,
        },
        "-1": {
            "flow_accel": 1.0,
            "oi_purge": 1.0,
            "vol_expansion": 1.0,
            "range_pullback": 0.2,
            "kimchi_impulse": 0.0,
        },
    }

    long_active, short_active = build_masks(features, np.arange(5), thresholds)

    assert long_active.tolist() == [True, False, False, False, False]
    assert short_active.tolist() == [False, False, True, False, False]


def _stats(*, fit_ratio: float = 1.5, select_ratio: float = 3.0) -> dict:
    def row(return_pct: float, ratio: float, trades: int, longs: int, shorts: int) -> dict:
        return {
            "absolute_return_pct": return_pct,
            "cagr_to_strict_mdd": ratio,
            "trades": trades,
            "longs": longs,
            "shorts": shorts,
        }

    return {
        "fit": row(10.0, fit_ratio, 40, 20, 20),
        "fit_2021": row(2.0, 1.0, 20, 10, 10),
        "fit_2022": row(2.0, 1.0, 20, 10, 10),
        "select_2023": row(8.0, select_ratio, 16, 8, 8),
        "select_2023_h1": row(4.0, 3.0, 8, 4, 4),
        "select_2023_h2": row(4.0, 3.0, 8, 4, 4),
    }


def test_selection_contract_requires_fit_and_both_2023_halves() -> None:
    assert selection_passes(_stats())
    assert not selection_passes(_stats(fit_ratio=1.49))

    broken_half = _stats()
    broken_half["select_2023_h2"]["cagr_to_strict_mdd"] = 2.99
    assert not selection_passes(broken_half)


def test_activation_hash_changes_with_side_or_position() -> None:
    anchors = np.array([10, 20, 30], dtype=np.int64)
    first = activation_hash(
        anchors,
        np.array([True, False, False]),
        np.array([False, True, False]),
    )
    side_changed = activation_hash(
        anchors,
        np.array([False, False, False]),
        np.array([True, True, False]),
    )
    position_changed = activation_hash(
        anchors,
        np.array([True, False, True]),
        np.array([False, False, False]),
    )

    assert first != side_changed
    assert first != position_changed


def _config(tmp_path) -> Config:
    return Config(
        input_csv="market.csv",
        metrics_csv="metrics.csv",
        funding_csv="funding.csv",
        output=str(tmp_path / "oos.json"),
        manifest_output=str(tmp_path / "manifest.json"),
    )


def _manifest(cfg: Config) -> dict:
    payload = {
        "phase": "pre_2024_freeze",
        "oos_opened": False,
        "selection_end": "2024-01-01",
        "spec": SPEC,
        "spec_hash": _spec_hash(),
        "frozen_execution_config": _frozen_execution_config(cfg),
        "source_prefix_hashes": {"market": "a", "metrics": "b", "funding": "c"},
        "thresholds": {"1": {"flow": 1.0}, "-1": {"flow": 1.0}},
        "activation_hash": "activation",
        "selection_passed": True,
        "selection_stats": _stats(),
        "selection_schedule_hashes": {"fit": "schedule"},
    }
    payload["freeze_hash"] = _freeze_hash(payload)
    return payload


def test_manifest_freeze_rejects_threshold_or_runtime_mutation(tmp_path) -> None:
    cfg = _config(tmp_path)
    manifest = _manifest(cfg)
    _validate_manifest(cfg, manifest)

    threshold_mutation = {**manifest, "thresholds": {"1": {"flow": 2.0}}}
    with pytest.raises(RuntimeError, match="freeze hash"):
        _validate_manifest(cfg, threshold_mutation)

    changed_cost = Config(**{**cfg.__dict__, "fee_rate": 0.0004})
    with pytest.raises(RuntimeError, match="execution config"):
        _validate_manifest(changed_cost, manifest)


def test_oos_seal_is_written_before_replay_and_cannot_be_reused(tmp_path) -> None:
    cfg = _config(tmp_path)
    manifest = _manifest(cfg)
    path = tmp_path / "manifest.json"

    opened = _mark_oos_opened(path, manifest, cfg.output)

    assert opened["oos_opened"] is True
    assert path.exists()
    assert '"oos_opened": true' in path.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="already been opened"):
        _validate_manifest(cfg, opened)


def test_oos_seal_precedes_even_prefix_source_reads(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    manifest = _manifest(cfg)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def guarded_load_sources(*args, **kwargs):
        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert on_disk["oos_opened"] is True
        raise RuntimeError("stop-after-seal")

    monkeypatch.setattr(module, "_load_sources", guarded_load_sources)

    with pytest.raises(RuntimeError, match="stop-after-seal"):
        module._oos(cfg)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["oos_opened"] is True
