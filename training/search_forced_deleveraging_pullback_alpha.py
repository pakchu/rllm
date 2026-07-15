"""Freeze and replay a causal forced-deleveraging pullback alpha.

The selection process is physically truncated before 2024.  The exact rule
requires five weak witnesses to agree:

* directional taker-flow acceleration,
* a 12-hour open-interest purge,
* 24-hour versus 7-day volatility expansion,
* entry from the opposing side of the trailing 24-hour range, and
* a same-direction local kimchi-premium impulse.

The rule enters at the next 5-minute open, holds at most 24 hours, and uses a
loose 6% catastrophe stop.  ``--open-oos`` is refused until a compatible
pre-2024 manifest has already been written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_inventory_purge_reclaim_alpha import (
    Config as EngineConfig,
    ExecutionEngine,
    WINDOWS,
    _build_features,
    _load_sources,
    _schedule_hash,
    _stats_by_window,
)


SELECTION_END = "2024-01-01"
FIT_WINDOWS = (
    "fit",
    "fit_2020q4",
    "fit_2021",
    "fit_2022",
    "select_2023",
    "select_2023_h1",
    "select_2023_h2",
)
OOS_WINDOWS = ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026")
SPEC = {
    "anchor_stride_bars": 72,
    "warmup_bars": 2016,
    "hold_bars": 288,
    "catastrophe_stop": 0.06,
    "take_profit": 10.0,
    "flow_quantile": 0.70,
    "oi_purge_quantile": 0.70,
    "vol_expansion_quantile": 0.70,
    "range_pullback_quantile": 0.60,
    "kimchi_impulse_threshold": 0.0,
}
FROZEN_CONFIG_KEYS = (
    "input_csv",
    "metrics_csv",
    "funding_csv",
    "exclude_from",
    "metrics_tolerance",
    "source_delay_bars",
    "leverage",
    "fee_rate",
    "slippage_rate",
)


@dataclass(frozen=True)
class Config:
    input_csv: str
    metrics_csv: str
    funding_csv: str
    output: str
    manifest_output: str
    docs_output: str = ""
    exclude_from: str = "2026-06-02"
    metrics_tolerance: str = "10min"
    source_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    open_oos: bool = False


def _engine_config(cfg: Config) -> EngineConfig:
    return EngineConfig(
        input_csv=cfg.input_csv,
        metrics_csv=cfg.metrics_csv,
        funding_csv=cfg.funding_csv,
        output=cfg.output,
        manifest_output=cfg.manifest_output,
        docs_output=cfg.docs_output,
        exclude_from=cfg.exclude_from,
        metrics_tolerance=cfg.metrics_tolerance,
        source_delay_bars=cfg.source_delay_bars,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        open_oos=cfg.open_oos,
    )


def _spec_hash() -> str:
    encoded = json.dumps(SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frozen_execution_config(cfg: Config) -> dict[str, Any]:
    values = asdict(cfg)
    return {key: values[key] for key in FROZEN_CONFIG_KEYS}


def _freeze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_end": payload["selection_end"],
        "spec": payload["spec"],
        "spec_hash": payload["spec_hash"],
        "frozen_execution_config": payload["frozen_execution_config"],
        "source_prefix_hashes": payload["source_prefix_hashes"],
        "thresholds": payload["thresholds"],
        "activation_hash": payload["activation_hash"],
        "selection_passed": payload["selection_passed"],
        "selection_stats": payload["selection_stats"],
        "selection_schedule_hashes": payload["selection_schedule_hashes"],
    }


def _freeze_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _freeze_payload(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest(cfg: Config, manifest: dict[str, Any]) -> None:
    if manifest.get("oos_opened"):
        raise RuntimeError("manifest OOS has already been opened")
    if manifest.get("spec_hash") != _spec_hash():
        raise RuntimeError("manifest strategy spec is incompatible")
    if not manifest.get("selection_passed"):
        raise RuntimeError("manifest did not pass pre-2024 selection")
    if manifest.get("frozen_execution_config") != _frozen_execution_config(cfg):
        raise RuntimeError("runtime execution config differs from frozen manifest")
    if manifest.get("freeze_hash") != _freeze_hash(manifest):
        raise RuntimeError("manifest freeze hash mismatch")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _mark_oos_opened(manifest_path: Path, manifest: dict[str, Any], output: str) -> dict[str, Any]:
    opened = {
        **manifest,
        "phase": "oos_opening",
        "oos_opened": True,
        "oos_opened_at": datetime.now(timezone.utc).isoformat(),
        "oos_output": output,
    }
    _atomic_write_json(manifest_path, opened)
    return opened


def build_signal_features(market: pd.DataFrame) -> dict[str, np.ndarray]:
    model, _ = _build_features(market)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_expansion = np.log(
            model["realized_vol_288"].to_numpy(float)
            / model["realized_vol_2016"].to_numpy(float)
        )
    return {
        "flow_accel": (
            model["taker_imbalance_12"] - model["taker_imbalance_48"]
        ).to_numpy(float),
        "oi_purge": -model["oi_return_144"].to_numpy(float),
        "vol_expansion": vol_expansion,
        # build_model_features() centers this field in [-0.5, 0.5].
        "range_position_288": model["range_position_288"].to_numpy(float),
        "kimchi_impulse": model["kimchi_premium_change"].to_numpy(float),
        "kimchi_available": (
            pd.to_numeric(market["kimchi_available"], errors="coerce").to_numpy(float) > 0.5
        ),
    }


def _fit_mask(dates: pd.Series) -> np.ndarray:
    start, end = WINDOWS["fit"]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _quantile(values: np.ndarray, fit_mask: np.ndarray, quantile: float) -> float:
    sample = values[fit_mask & np.isfinite(values)]
    if len(sample) < 10_000:
        raise ValueError(f"insufficient fit rows for q{quantile}: {len(sample)}")
    return float(np.quantile(sample, quantile))


def fit_thresholds(features: dict[str, np.ndarray], dates: pd.Series) -> dict[str, dict[str, float]]:
    fit_mask = _fit_mask(dates)
    thresholds: dict[str, dict[str, float]] = {}
    for side in (-1, 1):
        thresholds[str(side)] = {
            "flow_accel": _quantile(side * features["flow_accel"], fit_mask, SPEC["flow_quantile"]),
            "oi_purge": _quantile(features["oi_purge"], fit_mask, SPEC["oi_purge_quantile"]),
            "vol_expansion": _quantile(
                features["vol_expansion"], fit_mask, SPEC["vol_expansion_quantile"]
            ),
            "range_pullback": _quantile(
                -side * features["range_position_288"],
                fit_mask,
                SPEC["range_pullback_quantile"],
            ),
            "kimchi_impulse": float(SPEC["kimchi_impulse_threshold"]),
        }
    return thresholds


def build_masks(
    features: dict[str, np.ndarray],
    anchors: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for side in (-1, 1):
        threshold = thresholds[str(side)]
        flow = side * features["flow_accel"][anchors]
        purge = features["oi_purge"][anchors]
        vol = features["vol_expansion"][anchors]
        pullback = -side * features["range_position_288"][anchors]
        kimchi = side * features["kimchi_impulse"][anchors]
        masks[side] = (
            np.isfinite(flow)
            & np.isfinite(purge)
            & np.isfinite(vol)
            & np.isfinite(pullback)
            & np.isfinite(kimchi)
            & (flow >= threshold["flow_accel"])
            & (purge >= threshold["oi_purge"])
            & (vol >= threshold["vol_expansion"])
            & (pullback >= threshold["range_pullback"])
            & (kimchi >= threshold["kimchi_impulse"])
            & features["kimchi_available"][anchors]
        )
    both = masks[1] & masks[-1]
    masks[1] &= ~both
    masks[-1] &= ~both
    return masks[1], masks[-1]


def activation_hash(anchors: np.ndarray, long_active: np.ndarray, short_active: np.ndarray) -> str:
    records = np.column_stack(
        [anchors[long_active | short_active], long_active[long_active | short_active].astype(np.int8)]
    )
    return hashlib.sha256(records.astype("<i8", copy=False).tobytes(order="C")).hexdigest()


def selection_passes(stats: dict[str, dict[str, Any]]) -> bool:
    enough = (
        stats["fit"]["trades"] >= 40
        and stats["select_2023"]["trades"] >= 15
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 6
        and min(stats["fit"]["longs"], stats["fit"]["shorts"]) >= 6
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 4
    )
    robust = all(
        stats[window]["absolute_return_pct"] > 0.0
        for window in ("fit_2021", "fit_2022", "select_2023_h1", "select_2023_h2")
    )
    return bool(
        enough
        and robust
        and stats["fit"]["cagr_to_strict_mdd"] >= 1.5
        and stats["select_2023"]["cagr_to_strict_mdd"] >= 3.0
        and stats["select_2023_h1"]["cagr_to_strict_mdd"] >= 3.0
        and stats["select_2023_h2"]["cagr_to_strict_mdd"] >= 3.0
    )


def _schedule_windows(
    engine: ExecutionEngine,
    anchors: np.ndarray,
    long_active: np.ndarray,
    short_active: np.ndarray,
    windows: tuple[str, ...],
) -> dict[str, list[Any]]:
    return {
        window: engine.schedule(
            anchors,
            long_active,
            short_active,
            window=window,
            hold=int(SPEC["hold_bars"]),
            tp=float(SPEC["take_profit"]),
            sl=float(SPEC["catastrophe_stop"]),
        )
        for window in windows
    }


def _write_docs(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    stats = payload["selection_stats"]
    lines = [
        "# Forced-deleveraging pullback candidate",
        "",
        "- Selection data are physically truncated before `2024-01-01`.",
        "- Entry: next 5-minute open; 24-hour maximum hold; 6% catastrophe stop.",
        "- Cost: 6bp/notional/side plus realized funding; leverage 0.5x.",
        "- Strict MDD: global high-water plus conservative favorable-then-adverse intratrade path.",
        "- Status: frozen pre-2024 candidate, not yet an OOS-confirmed alpha.",
        "",
        "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | L/S |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window in FIT_WINDOWS:
        value = stats[window]
        lines.append(
            f"| {window} | {value['absolute_return_pct']:.2f}% | {value['cagr_pct']:.2f}% | "
            f"{value['strict_mdd_pct']:.2f}% | {value['cagr_to_strict_mdd']:.2f} | "
            f"{value['trades']} | {value['longs']}/{value['shorts']} |"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection(cfg: Config) -> dict[str, Any]:
    engine_cfg = _engine_config(cfg)
    market, funding, prefix_hashes = _load_sources(engine_cfg, cutoff=SELECTION_END)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection source was not physically truncated")
    features = build_signal_features(market)
    thresholds = fit_thresholds(features, dates)
    anchors = np.arange(
        int(SPEC["warmup_bars"]),
        len(market) - int(SPEC["hold_bars"]) - 2,
        int(SPEC["anchor_stride_bars"]),
        dtype=np.int64,
    )
    long_active, short_active = build_masks(features, anchors, thresholds)
    engine = ExecutionEngine(market, funding, engine_cfg)
    schedules = _schedule_windows(engine, anchors, long_active, short_active, FIT_WINDOWS)
    stats = _stats_by_window(schedules, engine_cfg)
    passed = selection_passes(stats)
    if not passed:
        raise RuntimeError("pre-2024 candidate no longer passes its frozen selection contract")
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_2024_freeze",
        "oos_opened": False,
        "selection_end": SELECTION_END,
        "config": asdict(cfg),
        "frozen_execution_config": _frozen_execution_config(cfg),
        "spec": SPEC,
        "spec_hash": _spec_hash(),
        "source_prefix_hashes": prefix_hashes,
        "thresholds": thresholds,
        "activation_hash": activation_hash(anchors, long_active, short_active),
        "selection_passed": passed,
        "selection_stats": stats,
        "selection_schedule_hashes": {
            window: _schedule_hash(schedules[window]) for window in FIT_WINDOWS
        },
    }
    payload["freeze_hash"] = _freeze_hash(payload)
    manifest_path = Path(cfg.manifest_output)
    _atomic_write_json(manifest_path, payload)
    output_path = Path(cfg.output)
    _atomic_write_json(output_path, payload)
    _write_docs(cfg.docs_output, payload)
    return payload


def _oos(cfg: Config) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if not manifest_path.exists():
        raise FileNotFoundError("pre-2024 manifest must exist before --open-oos")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(cfg, manifest)

    # Burn the one-shot OOS seal before opening any source file.  Even a
    # prefix-verification failure remains visibly contaminated instead of
    # permitting another future-data attempt.
    opened_manifest = _mark_oos_opened(manifest_path, manifest, cfg.output)
    engine_cfg = _engine_config(cfg)
    _, _, prefix_hashes = _load_sources(engine_cfg, cutoff=SELECTION_END)
    if prefix_hashes != manifest.get("source_prefix_hashes"):
        raise RuntimeError("pre-2024 source prefix changed after freeze")

    market, funding, _ = _load_sources(engine_cfg, cutoff=cfg.exclude_from)
    dates = pd.to_datetime(market["date"])
    features = build_signal_features(market)
    anchors = np.arange(
        int(SPEC["warmup_bars"]),
        len(market) - int(SPEC["hold_bars"]) - 2,
        int(SPEC["anchor_stride_bars"]),
        dtype=np.int64,
    )
    thresholds = manifest["thresholds"]
    long_active, short_active = build_masks(features, anchors, thresholds)
    engine = ExecutionEngine(market, funding, engine_cfg)
    schedules = _schedule_windows(engine, anchors, long_active, short_active, OOS_WINDOWS)
    stats = _stats_by_window(schedules, engine_cfg)
    result = {
        **opened_manifest,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "frozen_oos_replay",
        "data_end": str(dates.max()),
        "oos_stats": stats,
        "oos_schedule_hashes": {
            window: _schedule_hash(schedules[window]) for window in OOS_WINDOWS
        },
    }
    output_path = Path(cfg.output)
    _atomic_write_json(output_path, result)
    return result


def run(cfg: Config) -> dict[str, Any]:
    return _oos(cfg) if cfg.open_oos else _selection(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--docs-output", default="")
    parser.add_argument("--exclude-from", default="2026-06-02")
    parser.add_argument("--open-oos", action="store_true")
    args = parser.parse_args()
    payload = run(Config(**vars(args)))
    stats = payload.get("oos_stats", payload.get("selection_stats", {}))
    print(json.dumps({"phase": payload["phase"], "spec_hash": payload["spec_hash"], "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
