"""Freeze and replay a causal asymmetric jump/rejection alpha.

The selection process is physically truncated before 2024.  It combines two
different weak-signal mechanisms rather than forcing one symmetric rule:

* long: positive signed jump variation + OI build + clean 6-hour path;
* short: upper-wick rejection + rising DXY + fast causal volume clock.

Every witness is evaluated on a completed 5-minute bar.  Entry is the next
bar open, the maximum hold is 24 hours, and conservative same-bar execution
checks a 3% stop before a 4% take.  ``--open-oos`` is refused until a matching
pre-2024 manifest has already been written and committed.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import (
    Config as EngineConfig,
    ExecutionEngine,
    Trade,
    WINDOWS as ENGINE_WINDOWS,
    _build_features,
    _load_sources,
    _schedule_hash,
    equity_stats,
)
from training.search_jump_variation_bidirectional_alpha import features as jump_features
from training.search_liquidity_recovery_bidirectional_alpha import features as liquidity_features
from training.search_path_memory_bidirectional_alpha import features as path_features
from training.search_volume_clock_bidirectional_alpha import features as volume_clock_features


SELECTION_END = "2024-01-01"
WINDOWS = {
    **ENGINE_WINDOWS,
    "fit_2021_h1": ("2021-01-01", "2021-07-01"),
    "fit_2021_h2": ("2021-07-01", "2022-01-01"),
    "fit_2022_h1": ("2022-01-01", "2022-07-01"),
    "fit_2022_h2": ("2022-07-01", "2023-01-01"),
}
FIT_WINDOWS = (
    "fit",
    "fit_2020q4",
    "fit_2021",
    "fit_2021_h1",
    "fit_2021_h2",
    "fit_2022",
    "fit_2022_h1",
    "fit_2022_h2",
    "select_2023",
    "select_2023_h1",
    "select_2023_h2",
)
OOS_WINDOWS = ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026")
SPEC = {
    "anchor_stride_bars": 72,
    "warmup_bars": 2016,
    "hold_bars": 288,
    "stop_loss": 0.03,
    "take_profit": 0.04,
    "witness_quantile": 0.70,
    "long_rule": ["signed_jump_72", "oi_build_144", "path_clean_72"],
    "short_rule": ["upper_rejection_12", "dxy_strength", "fast_volume_clock_0p25"],
    "same_bar_policy": "stop_before_take",
    "discovery_accounting": {
        "symmetric_source_rules": 448,
        "asymmetric_pairs": 625,
        "unique_pair_stat_signatures": 460,
        "hold_pairs": 36,
        "slow_regime_variants": 45,
        "barrier_variants": 40,
        "barrier_variants_passing_contract": 10,
    },
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
        "oos_opened": payload["oos_opened"],
        "selection_end": payload["selection_end"],
        "spec": payload["spec"],
        "spec_hash": payload["spec_hash"],
        "implementation_hash": payload["implementation_hash"],
        "frozen_execution_config": payload["frozen_execution_config"],
        "source_prefix_hashes": payload["source_prefix_hashes"],
        "feature_prefix_hash": payload["feature_prefix_hash"],
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
    if manifest.get("oos_opened") is not False:
        raise RuntimeError("manifest must be explicitly pre-OOS")
    if "oos_opened_at" in manifest or "oos_output" in manifest:
        raise RuntimeError("pre-OOS manifest contains stale OOS metadata")
    if manifest.get("spec_hash") != _spec_hash():
        raise RuntimeError("manifest strategy spec is incompatible")
    if manifest.get("implementation_hash") != _implementation_hash():
        raise RuntimeError("manifest implementation is incompatible")
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
    """Build completed-bar witnesses with the exact frozen discovery semantics."""

    model, _ = _build_features(market)
    empty = pd.DataFrame(index=market.index)
    jump = jump_features(market, empty)
    liquidity = liquidity_features(market, empty)
    path = path_features(market, empty)
    volume_clock = volume_clock_features(market, empty)
    rejection_balance = (
        liquidity["lr_lower_rejection"] - liquidity["lr_upper_rejection"]
    ).to_numpy(float)
    return {
        "long_signed_jump": jump["jv_signed_jump_72"].to_numpy(float),
        "long_oi_build": model["oi_return_144"].to_numpy(float),
        "long_path_clean": np.abs(path["pm_eff_72"].to_numpy(float)),
        # Short orientation: upper rejection dominates when -balance is large.
        "short_upper_rejection": -rejection_balance,
        # For a short, the discovery orientation of dxy_support is +DXY momentum.
        "short_dxy_strength": model["dxy_momentum"].to_numpy(float),
        "short_fast_volume_clock": -volume_clock["vc_duration_0p25"].to_numpy(float),
        "dxy_available": (
            pd.to_numeric(market["dxy_available"], errors="coerce").to_numpy(float) > 0.5
        ),
    }


def feature_hash(features: dict[str, np.ndarray], mask: np.ndarray | None = None) -> str:
    ordered = {}
    for name in sorted(features):
        values = np.asarray(features[name])
        ordered[name] = values if mask is None else values[mask]
    return _frame_hash(pd.DataFrame(ordered).reset_index(drop=True))


def validate_feature_prefix(
    manifest: dict[str, Any], features: dict[str, np.ndarray], dates: pd.Series
) -> None:
    prefix = (dates < pd.Timestamp(SELECTION_END)).to_numpy(bool)
    if feature_hash(features, prefix) != manifest["feature_prefix_hash"]:
        raise RuntimeError("full-run pre-2024 feature prefix changed after freeze")


def _implementation_hash() -> str:
    functions = (
        build_signal_features,
        feature_hash,
        validate_feature_prefix,
        fit_thresholds,
        build_masks,
        activation_hash,
        _schedule_window,
        _load_sources,
        _build_features,
        jump_features,
        liquidity_features,
        path_features,
        volume_clock_features,
        ExecutionEngine.trade_at,
        equity_stats,
    )
    source = "\n\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


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
    quantile = float(SPEC["witness_quantile"])
    return {
        "long": {
            name: _quantile(features[name], fit_mask, quantile)
            for name in ("long_signed_jump", "long_oi_build", "long_path_clean")
        },
        "short": {
            name: _quantile(features[name], fit_mask, quantile)
            for name in (
                "short_upper_rejection",
                "short_dxy_strength",
                "short_fast_volume_clock",
            )
        },
    }


def build_masks(
    features: dict[str, np.ndarray],
    anchors: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    long_values = {
        name: features[name][anchors]
        for name in ("long_signed_jump", "long_oi_build", "long_path_clean")
    }
    short_values = {
        name: features[name][anchors]
        for name in (
            "short_upper_rejection",
            "short_dxy_strength",
            "short_fast_volume_clock",
        )
    }
    long_active = np.logical_and.reduce(
        [
            np.isfinite(values) & (values >= thresholds["long"][name])
            for name, values in long_values.items()
        ]
    )
    short_active = np.logical_and.reduce(
        [
            np.isfinite(values) & (values >= thresholds["short"][name])
            for name, values in short_values.items()
        ]
    ) & features["dxy_available"][anchors]
    both = long_active & short_active
    return long_active & ~both, short_active & ~both


def activation_hash(anchors: np.ndarray, long_active: np.ndarray, short_active: np.ndarray) -> str:
    records = np.column_stack(
        [anchors[long_active | short_active], long_active[long_active | short_active].astype(np.int8)]
    )
    return hashlib.sha256(records.astype("<i8", copy=False).tobytes(order="C")).hexdigest()


def selection_passes(stats: dict[str, dict[str, Any]]) -> bool:
    enough = (
        stats["fit"]["trades"] >= 180
        and stats["select_2023"]["trades"] >= 60
        and min(
            stats[window]["trades"]
            for window in (
                "fit_2021_h1",
                "fit_2021_h2",
                "fit_2022_h1",
                "fit_2022_h2",
                "select_2023_h1",
                "select_2023_h2",
            )
        ) >= 30
        and min(stats["fit"]["longs"], stats["fit"]["shorts"]) >= 75
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 25
    )
    stability_windows = (
        "fit_2021_h1",
        "fit_2021_h2",
        "fit_2022_h1",
        "fit_2022_h2",
        "select_2023_h1",
        "select_2023_h2",
    )
    robust = all(
        stats[window]["absolute_return_pct"] > 0.0
        for window in stability_windows
    )
    return bool(
        enough
        and robust
        and stats["fit"]["cagr_to_strict_mdd"] >= 4.0
        and stats["fit_2021"]["cagr_to_strict_mdd"] >= 3.0
        and stats["fit_2022"]["cagr_to_strict_mdd"] >= 3.0
        and stats["select_2023"]["cagr_to_strict_mdd"] >= 4.0
        and stats["select_2023_h1"]["cagr_to_strict_mdd"] >= 4.0
        and stats["select_2023_h2"]["cagr_to_strict_mdd"] >= 4.0
    )


def _schedule_window(
    engine: ExecutionEngine,
    anchors: np.ndarray,
    long_active: np.ndarray,
    short_active: np.ndarray,
    window: str,
) -> list[Trade]:
    start, end = WINDOWS[window]
    period = ((engine.dates >= pd.Timestamp(start)) & (engine.dates < pd.Timestamp(end))).to_numpy(bool)
    selected = period[anchors] & np.logical_xor(long_active, short_active)
    trades: list[Trade] = []
    next_allowed = 0
    for anchor_index in np.flatnonzero(selected):
        signal = int(anchors[anchor_index])
        if signal < next_allowed:
            continue
        side = 1 if bool(long_active[anchor_index]) else -1
        trade = engine.trade_at(
            signal,
            side,
            int(SPEC["hold_bars"]),
            int(round(float(SPEC["take_profit"]) * 10_000)),
            int(round(float(SPEC["stop_loss"]) * 10_000)),
        )
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def _schedule_windows(
    engine: ExecutionEngine,
    anchors: np.ndarray,
    long_active: np.ndarray,
    short_active: np.ndarray,
    windows: tuple[str, ...],
) -> dict[str, list[Any]]:
    return {
        window: _schedule_window(engine, anchors, long_active, short_active, window)
        for window in windows
    }


def _stats_by_window(
    schedules: dict[str, list[Trade]], engine_cfg: EngineConfig
) -> dict[str, dict[str, Any]]:
    return {
        window: equity_stats(
            trades,
            start=WINDOWS[window][0],
            end=WINDOWS[window][1],
            cfg=engine_cfg,
        )
        for window, trades in schedules.items()
    }


def _write_docs(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    stats = payload["selection_stats"]
    lines = [
        "# Asymmetric jump/rejection candidate",
        "",
        "- Selection data are physically truncated before `2024-01-01`.",
        "- Long: signed jump variation + OI build + clean path.",
        "- Short: upper rejection + rising DXY + fast causal volume clock.",
        "- Entry: next 5-minute open; 24-hour max hold; 3% stop; 4% take.",
        "- Same-bar ambiguity is conservative: stop is evaluated before take.",
        "- Cost: 6bp/notional/side plus realized funding; leverage 0.5x.",
        "- Strict MDD: global high-water plus conservative favorable-then-adverse intratrade path.",
        "- Multiplicity: 448 source rules, 625 asymmetric pairs (460 unique stat signatures), then bounded execution refinements.",
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
        "implementation_hash": _implementation_hash(),
        "source_prefix_hashes": prefix_hashes,
        "feature_prefix_hash": feature_hash(features),
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
    selection_market, selection_funding, prefix_hashes = _load_sources(
        engine_cfg, cutoff=SELECTION_END
    )
    if prefix_hashes != manifest.get("source_prefix_hashes"):
        raise RuntimeError("pre-2024 source prefix changed after freeze")

    selection_features = build_signal_features(selection_market)
    selection_anchors = np.arange(
        int(SPEC["warmup_bars"]),
        len(selection_market) - int(SPEC["hold_bars"]) - 2,
        int(SPEC["anchor_stride_bars"]),
        dtype=np.int64,
    )
    selection_long, selection_short = build_masks(
        selection_features, selection_anchors, manifest["thresholds"]
    )
    if (
        activation_hash(selection_anchors, selection_long, selection_short)
        != manifest["activation_hash"]
    ):
        raise RuntimeError("pre-2024 activation schedule changed after freeze")
    selection_engine = ExecutionEngine(selection_market, selection_funding, engine_cfg)
    selection_schedules = _schedule_windows(
        selection_engine, selection_anchors, selection_long, selection_short, FIT_WINDOWS
    )
    selection_schedule_hashes = {
        window: _schedule_hash(selection_schedules[window]) for window in FIT_WINDOWS
    }
    if selection_schedule_hashes != manifest["selection_schedule_hashes"]:
        raise RuntimeError("pre-2024 trade schedules changed after freeze")

    market, funding, _ = _load_sources(engine_cfg, cutoff=cfg.exclude_from)
    dates = pd.to_datetime(market["date"])
    features = build_signal_features(market)
    validate_feature_prefix(manifest, features, dates)
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
