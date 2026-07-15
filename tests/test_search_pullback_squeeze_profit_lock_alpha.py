from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import training.search_pullback_squeeze_profit_lock_alpha as module
from training.search_pullback_squeeze_profit_lock_alpha import (
    HOLD_GRID,
    MANIFEST_PAYLOAD_FIELDS,
    NO_STOP_BPS,
    TAKE_PROFIT_GRID_BPS,
    Config,
    _frozen_protocol,
    _manifest_hash,
    _passes_oos_gate,
    _passes_pre_oos_gate,
    _selection_score,
    _shift_signal,
    _spec_name,
    _validate_manifest,
    _write_manifest_once,
)


def _stats(ratio: float = 3.2, trades: int = 20, return_pct: float = 5.0, mdd: float = 8.0) -> dict:
    return {
        "absolute_return_pct": return_pct,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": mdd,
        "trades": trades,
    }


def test_grid_records_take_profit_only_and_time_only_controls() -> None:
    assert 576 in HOLD_GRID
    assert 1_000 in TAKE_PROFIT_GRID_BPS
    assert NO_STOP_BPS in TAKE_PROFIT_GRID_BPS
    assert _spec_name(576, 1_000) == "hold_576_tp_1000bps_no_stop"
    assert _spec_name(576, NO_STOP_BPS) == "hold_576_time_only"


def test_shift_signal_delays_without_wrapping() -> None:
    active = np.array([True, False, True, False, False])
    assert _shift_signal(active, 2).tolist() == [False, False, True, False, True]
    assert _shift_signal(active, 0).tolist() == active.tolist()


def test_pre_oos_gate_requires_train_selection_and_combined_floor() -> None:
    names = (
        "train",
        "train_2020h2",
        "train_2021",
        "train_2022",
        "select_2023",
        "select_2023_h1",
        "select_2023_h2",
        "pre_2024",
    )
    values = {name: _stats(trades=70 if name == "train" else 20) for name in names}
    assert _passes_pre_oos_gate(values)
    values["pre_2024"] = _stats(ratio=2.49)
    assert not _passes_pre_oos_gate(values)


def test_selection_score_prioritizes_stable_segments_then_worst_primary_ratio() -> None:
    names = (
        "train",
        "train_2020h2",
        "train_2021",
        "train_2022",
        "select_2023",
        "select_2023_h1",
        "select_2023_h2",
        "pre_2024",
    )
    stable = {name: _stats() for name in names}
    unstable = {name: _stats(ratio=4.0) for name in names}
    unstable["train_2022"] = _stats(ratio=4.0, return_pct=-1.0)
    assert _selection_score(stable) > _selection_score(unstable)


def test_oos_gate_uses_2024_test_and_2025_eval_not_short_holdout_for_primary_admission() -> None:
    values = {
        "test_2024": _stats(trades=15),
        "eval_2025": _stats(trades=15),
        "holdout_2026": _stats(ratio=-1.0, trades=2, return_pct=-1.0),
        "oos_2024_2026": _stats(trades=32),
    }
    assert _passes_oos_gate(values)
    values["eval_2025"] = _stats(ratio=2.99, trades=15)
    assert not _passes_oos_gate(values)


def test_manifest_rejects_payload_and_runtime_protocol_changes() -> None:
    cfg = Config()
    payload = {
        "selection_end": "2024-01-01",
        "source_hashes": {"market": "a", "funding": "b", "premium": "c"},
        "activation_hash": "signals",
        "thresholds": {"q": 1.0},
        "spec": {
            "name": "hold_576_tp_1000bps_no_stop",
            "hold_bars": 576,
            "take_bps": 1_000,
            "stop_bps": NO_STOP_BPS,
        },
        "selection_score": [5.0, 3.0],
        "passes_pre_oos_gate": True,
        "frozen_protocol": _frozen_protocol(cfg),
    }
    assert tuple(payload) == MANIFEST_PAYLOAD_FIELDS
    manifest = {
        "phase": "pre_oos_frozen",
        "manifest_hash": _manifest_hash(payload),
        **payload,
    }
    _validate_manifest(manifest, cfg, expected_hash=manifest["manifest_hash"])

    tampered = {**manifest, "spec": {**manifest["spec"], "take_bps": 1_200}}
    try:
        _validate_manifest(tampered, cfg, expected_hash=manifest["manifest_hash"])
    except RuntimeError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("tampered manifest was accepted")

    changed_cost = Config(slippage_rate=0.0002)
    try:
        _validate_manifest(manifest, changed_cost, expected_hash=manifest["manifest_hash"])
    except RuntimeError as exc:
        assert "protocol" in str(exc)
    else:
        raise AssertionError("changed runtime protocol was accepted")

    rehashed_tamper = {**tampered}
    rehashed_payload = {key: rehashed_tamper[key] for key in MANIFEST_PAYLOAD_FIELDS}
    rehashed_tamper["manifest_hash"] = _manifest_hash(rehashed_payload)
    with pytest.raises(RuntimeError, match="externally pinned"):
        _validate_manifest(rehashed_tamper, cfg, expected_hash=manifest["manifest_hash"])


def test_manifest_is_write_once(tmp_path) -> None:
    cfg = Config()
    # Use the real committed manifest because its externally pinned hash must
    # match the complete frozen payload, not merely a synthetic fixture.
    committed = module.json.loads(
        module.Path("results/pullback_squeeze_profit_lock_manifest_2026-07-15.json").read_text()
    )
    path = tmp_path / "manifest.json"
    assert _write_manifest_once(path, committed, cfg) == committed

    same_payload = {**committed, "created_at": "later"}
    assert _write_manifest_once(path, same_payload, cfg) == committed
    assert module.json.loads(path.read_text()) == committed

    changed = {**committed, "selection_score": [999.0]}
    changed_payload = {key: changed[key] for key in MANIFEST_PAYLOAD_FIELDS}
    changed["manifest_hash"] = _manifest_hash(changed_payload)
    with pytest.raises(RuntimeError):
        _write_manifest_once(path, changed, cfg)


def test_oos_loads_future_only_after_validation_and_replays_frozen_spec(tmp_path, monkeypatch) -> None:
    manifest_source = module.Path("results/pullback_squeeze_profit_lock_manifest_2026-07-15.json")
    manifest = module.json.loads(manifest_source.read_text())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(module.json.dumps(manifest))
    cfg = Config(manifest_output=str(manifest_path), output=str(tmp_path / "oos.json"))

    selection_dates = pd.Series(pd.to_datetime(["2023-12-31 22:00", "2023-12-31 23:00"]))
    full_dates = pd.Series(
        pd.to_datetime(["2023-12-31 22:00", "2023-12-31 23:00", "2024-01-01 00:00", "2024-01-01 01:00"])
    )
    load_cutoffs: list[str] = []
    evaluation_calls: list[tuple[dict[str, Any], float]] = []

    def fake_load(_cfg, *, cutoff, premium_tolerance):
        load_cutoffs.append(cutoff)
        dates = selection_dates if cutoff == module.SELECTION_END else full_dates
        market = pd.DataFrame({"date": dates})
        features = pd.DataFrame(index=range(len(dates)))
        hashes = manifest["source_hashes"] if cutoff == module.SELECTION_END else {}
        return market, features, pd.DataFrame(), hashes

    def fake_fit(_features, dates, _decisions):
        active = np.zeros(len(dates), dtype=bool)
        return active, manifest["thresholds"]

    def fake_evaluate(_market, _funding, _active, eval_cfg, **kwargs):
        evaluation_calls.append((kwargs, eval_cfg.slippage_rate))
        return {
            name: {
                "absolute_return_pct": 5.0,
                "cagr_pct": 10.0,
                "strict_mdd_pct": 2.0,
                "cagr_to_strict_mdd": 5.0,
                "trades": 20,
                "mean_net_bps": 10.0,
                "win_rate": 0.6,
            }
            for name in kwargs["windows"]
        }

    monkeypatch.setattr(module, "_load_bundle", fake_load)
    monkeypatch.setattr(module, "live_decision_features", lambda frame: frame)
    monkeypatch.setattr(module, "_fit_active", fake_fit)
    monkeypatch.setattr(module, "_activation_hash", lambda _active, _dates: manifest["activation_hash"])
    monkeypatch.setattr(module, "_evaluate", fake_evaluate)

    report = module._oos_run(cfg)
    assert report["verdict"] == "ALPHA_QUALIFIED"
    assert load_cutoffs == [module.SELECTION_END, cfg.exclude_from]
    assert len(evaluation_calls) == 2
    for kwargs, _slippage in evaluation_calls:
        assert kwargs["hold_bars"] == manifest["spec"]["hold_bars"]
        assert kwargs["take_bps"] == manifest["spec"]["take_bps"]
        assert kwargs["stop_bps"] == manifest["spec"]["stop_bps"]
