from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import training.search_asymmetric_jump_rejection_alpha as module
from training.search_asymmetric_jump_rejection_alpha import (
    SPEC,
    Config,
    _freeze_hash,
    _frozen_execution_config,
    _implementation_hash,
    _mark_oos_opened,
    _schedule_window,
    _spec_hash,
    _validate_manifest,
    activation_hash,
    build_masks,
    feature_hash,
    fit_thresholds,
    selection_passes,
    validate_feature_prefix,
)
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine


def test_build_masks_requires_each_side_specific_witness_and_dxy_availability() -> None:
    features = {
        "long_signed_jump": np.array([2.0, 2.0, 0.0, 0.0, 2.0]),
        "long_oi_build": np.array([2.0, 0.0, 0.0, 0.0, 2.0]),
        "long_path_clean": np.array([2.0, 2.0, 0.0, 0.0, 2.0]),
        "short_upper_rejection": np.array([0.0, 0.0, 2.0, 2.0, 0.0]),
        "short_dxy_strength": np.array([0.0, 0.0, 2.0, 2.0, 0.0]),
        "short_fast_volume_clock": np.array([0.0, 0.0, 2.0, 2.0, 0.0]),
        "dxy_available": np.array([True, True, True, False, True]),
    }
    thresholds = {
        "long": {
            "long_signed_jump": 1.0,
            "long_oi_build": 1.0,
            "long_path_clean": 1.0,
        },
        "short": {
            "short_upper_rejection": 1.0,
            "short_dxy_strength": 1.0,
            "short_fast_volume_clock": 1.0,
        },
    }

    long_active, short_active = build_masks(features, np.arange(5), thresholds)

    assert long_active.tolist() == [True, False, False, False, True]
    assert short_active.tolist() == [False, False, True, False, False]


def _stats(*, fit_ratio: float = 4.0, select_ratio: float = 4.0) -> dict:
    def row(return_pct: float, ratio: float, trades: int, longs: int, shorts: int) -> dict:
        return {
            "absolute_return_pct": return_pct,
            "cagr_to_strict_mdd": ratio,
            "trades": trades,
            "longs": longs,
            "shorts": shorts,
        }

    return {
        "fit": row(100.0, fit_ratio, 200, 110, 90),
        "fit_2021": row(30.0, 3.0, 80, 45, 35),
        "fit_2021_h1": row(15.0, 2.0, 40, 22, 18),
        "fit_2021_h2": row(12.0, 2.0, 40, 23, 17),
        "fit_2022": row(20.0, 3.0, 90, 50, 40),
        "fit_2022_h1": row(10.0, 2.0, 45, 25, 20),
        "fit_2022_h2": row(9.0, 2.0, 45, 25, 20),
        "select_2023": row(25.0, select_ratio, 70, 40, 30),
        "select_2023_h1": row(12.0, 4.0, 35, 20, 15),
        "select_2023_h2": row(11.0, 4.0, 35, 20, 15),
    }


def test_selection_contract_requires_fit_and_both_2023_halves() -> None:
    assert selection_passes(_stats())
    assert not selection_passes(_stats(fit_ratio=3.99))

    broken_half = _stats()
    broken_half["select_2023_h2"]["cagr_to_strict_mdd"] = 3.99
    assert not selection_passes(broken_half)

    broken_stability = _stats()
    broken_stability["fit_2022_h2"]["absolute_return_pct"] = -0.01
    assert not selection_passes(broken_stability)


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


def test_fit_thresholds_ignore_values_after_fit_window() -> None:
    dates = pd.Series(pd.date_range("2021-01-01", periods=12_000, freq="5min"))
    dates.iloc[11_000:] = pd.date_range("2024-01-01", periods=1_000, freq="5min")
    names = (
        "long_signed_jump",
        "long_oi_build",
        "long_path_clean",
        "short_upper_rejection",
        "short_dxy_strength",
        "short_fast_volume_clock",
    )
    first = {name: np.linspace(-1.0, 1.0, len(dates)) for name in names}
    second = {name: values.copy() for name, values in first.items()}
    for values in second.values():
        values[11_000:] = 1_000_000.0

    assert fit_thresholds(first, dates) == fit_thresholds(second, dates)


def test_full_run_feature_prefix_must_match_physically_truncated_features() -> None:
    selection_dates = pd.Series(pd.date_range("2023-12-31", periods=12, freq="5min"))
    frozen = {
        "long_signed_jump": np.linspace(0.0, 1.0, len(selection_dates)),
        "dxy_available": np.ones(len(selection_dates), dtype=bool),
    }
    manifest = {"feature_prefix_hash": feature_hash(frozen)}
    full_dates = pd.concat(
        [selection_dates, pd.Series(pd.date_range("2024-01-01", periods=4, freq="5min"))],
        ignore_index=True,
    )
    causal = {
        name: np.concatenate([values, np.repeat(values[-1], 4)])
        for name, values in frozen.items()
    }
    validate_feature_prefix(manifest, causal, full_dates)

    leaked = {name: values.copy() for name, values in causal.items()}
    leaked["long_signed_jump"][0] += 0.01
    with pytest.raises(RuntimeError, match="feature prefix"):
        validate_feature_prefix(manifest, leaked, full_dates)


def test_schedule_enters_next_open_and_resolves_same_bar_against_strategy() -> None:
    rows = 400
    dates = pd.date_range("2021-01-01", periods=rows, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.0),
            "low": np.full(rows, 100.0),
        }
    )
    # Signal at 10 enters at 11. Both 3% stop and 4% take are touched there;
    # the engine must take the conservative stop first.
    market.loc[11, "high"] = 105.0
    market.loc[11, "low"] = 96.0
    funding = pd.DataFrame(
        {"date": pd.Series([], dtype="datetime64[ns]"), "funding_rate": pd.Series([], dtype=float)}
    )
    cfg = Config(
        input_csv="market.csv",
        metrics_csv="metrics.csv",
        funding_csv="funding.csv",
        output="out.json",
        manifest_output="manifest.json",
    )
    engine = ExecutionEngine(market, funding, module._engine_config(cfg))

    trades = _schedule_window(
        engine,
        np.array([10], dtype=np.int64),
        np.array([True]),
        np.array([False]),
        "fit_2021_h1",
    )

    assert len(trades) == 1
    assert trades[0].entry_position == 11
    assert trades[0].exit_position == 11
    assert trades[0].gross_return == pytest.approx(-0.03)


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
        "implementation_hash": _implementation_hash(),
        "frozen_execution_config": _frozen_execution_config(cfg),
        "source_prefix_hashes": {"market": "a", "metrics": "b", "funding": "c"},
        "feature_prefix_hash": "feature-prefix",
        "thresholds": {"long": {"jump": 1.0}, "short": {"rejection": 1.0}},
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

    threshold_mutation = {**manifest, "thresholds": {"long": {"jump": 2.0}}}
    with pytest.raises(RuntimeError, match="freeze hash"):
        _validate_manifest(cfg, threshold_mutation)

    changed_cost = Config(**{**cfg.__dict__, "fee_rate": 0.0004})
    with pytest.raises(RuntimeError, match="execution config"):
        _validate_manifest(changed_cost, manifest)


def test_manifest_requires_explicit_clean_pre_oos_state(tmp_path) -> None:
    cfg = _config(tmp_path)
    manifest = _manifest(cfg)

    missing_flag = {key: value for key, value in manifest.items() if key != "oos_opened"}
    with pytest.raises(RuntimeError, match="explicitly pre-OOS"):
        _validate_manifest(cfg, missing_flag)

    null_flag = {**manifest, "oos_opened": None}
    with pytest.raises(RuntimeError, match="explicitly pre-OOS"):
        _validate_manifest(cfg, null_flag)

    stale_metadata = {**manifest, "oos_opened_at": "2026-07-15T00:00:00Z"}
    with pytest.raises(RuntimeError, match="stale OOS metadata"):
        _validate_manifest(cfg, stale_metadata)


def test_oos_seal_is_written_before_replay_and_cannot_be_reused(tmp_path) -> None:
    cfg = _config(tmp_path)
    manifest = _manifest(cfg)
    path = tmp_path / "manifest.json"

    opened = _mark_oos_opened(path, manifest, cfg.output)

    assert opened["oos_opened"] is True
    assert path.exists()
    assert '"oos_opened": true' in path.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="explicitly pre-OOS"):
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

    frozen_implementation_hash = manifest["implementation_hash"]
    monkeypatch.setattr(module, "_implementation_hash", lambda: frozen_implementation_hash)
    monkeypatch.setattr(module, "_load_sources", guarded_load_sources)

    with pytest.raises(RuntimeError, match="stop-after-seal"):
        module._oos(cfg)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["oos_opened"] is True
