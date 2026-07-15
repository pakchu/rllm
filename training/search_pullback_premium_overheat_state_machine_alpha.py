"""Freeze and replay a three-state confirmed-pullback execution policy.

The entry clock remains the causal confirmed pullback squeeze.  Weak features
are not averaged.  They form mutually exclusive economic states:

* capitulation: weak completed week and either a wide 48h range or dry quote
  activity -> retain the position but lock a 4% take;
* premium overheat: strong premium-index change while price is high in its
  completed 48h range -> skip the trade;
* orderly remainder -> retain the 12% take.

The overheat family contains four predeclared interactions and two actions
(``skip`` or ``tp4``), for eight pre-2024 cells.  The exact champion must be
committed in a one-shot manifest before ``--open-oos`` can read 2024+ rows.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_confirmed_pullback_squeeze_live_parity import (
    AuditConfig,
    PRE2024_WINDOWS,
    _activation_hash,
    _execution_config,
    _fit_active,
    _load_bundle,
    decision_mask,
    live_decision_features,
)
from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)


SELECTION_END = "2024-01-01"
FIT_START = "2020-07-01"
FIT_END = "2023-01-01"
NO_STOP_BPS = 1_000_000
SELECTION_WINDOWS = tuple(PRE2024_WINDOWS)
FUTURE_WINDOWS: dict[str, tuple[str, str]] = {
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}
OOS_GATE_WINDOWS = ("test_2024", "eval_2025", "oos_2024_2026")
OVERHEAT_FAMILIES: dict[str, tuple[str, str]] = {
    "late_weekly_range": ("htf_1w_return_1", "rex_576_range_pos"),
    "bb_quote_displacement": ("bb_z", "quote_vol_z_1d"),
    "premium_range_overheat": ("premium_index_change", "rex_576_range_pos"),
    "three_day_activity_overheat": ("htf_3d_return_1", "quote_vol_z_1d"),
}
OVERHEAT_ACTIONS = ("skip", "tp4")
FROZEN_CHAMPION = {"overheat": "premium_range_overheat", "action": "skip"}
FEATURE_QUANTILES: dict[str, tuple[float, ...]] = {
    "htf_1w_return_1": (0.50, 0.67),
    "rex_576_range_width_pct": (0.50,),
    "quote_vol_z_1d": (0.20, 0.67),
    "rex_576_range_pos": (0.67,),
    "bb_z": (0.67,),
    "premium_index_change": (0.67,),
    "htf_3d_return_1": (0.67,),
}
SPEC: dict[str, Any] = {
    "name": "pullback_premium_overheat_state_machine",
    "side": 1,
    "hold_bars": 576,
    "hold_hours": 48.0,
    "capitulation_take_bps": 400,
    "normal_take_bps": 1_200,
    "stop_bps": NO_STOP_BPS,
    "leverage": 0.50,
    "capitulation": "week_q50_low AND (range_width_q50_high OR quote_activity_q20_low)",
    "overheat_quantile": 0.67,
    "overheat_families": OVERHEAT_FAMILIES,
    "overheat_actions": list(OVERHEAT_ACTIONS),
    "frozen_champion": FROZEN_CHAMPION,
    "state_priority": ["capitulation", "overheat", "normal"],
    "same_bar_policy": "stop_before_take",
    "discovery_accounting": {
        "state_machine_cells": 8,
        "prior_routed_take_cells": 29,
        "rejected_transition_cells": {
            "centroid_oi_unwind_and_mirror": 16,
            "crowded_endpoint_rejection_and_continuation": 16,
            "premium_wick_participant_migration": 8,
        },
        "status": "retrospective multiplicity disclosure; exact state machine selected pre-2024",
    },
}
FROZEN_CONFIG_KEYS = (
    "input_csv",
    "funding_csv",
    "premium_csv",
    "exclude_from",
    "window_size",
    "leverage",
    "fee_rate",
    "slippage_rate",
    "funding_tolerance",
    "live_premium_tolerance",
)


@dataclass(frozen=True)
class Config(AuditConfig):
    output: str = "results/pullback_premium_overheat_state_machine_selection_2026-07-15.json"
    manifest_output: str = "results/pullback_premium_overheat_state_machine_manifest_2026-07-15.json"
    docs_output: str = "docs/pullback-premium-overheat-state-machine-selection-2026-07-15.md"
    leverage: float = 0.50
    open_oos: bool = False


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _spec_hash() -> str:
    encoded = json.dumps(SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _frozen_execution_config(cfg: Config) -> dict[str, Any]:
    values = asdict(cfg)
    return {key: values[key] for key in FROZEN_CONFIG_KEYS}


def _freeze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "oos_opened",
            "selection_end",
            "spec",
            "spec_hash",
            "implementation_hash",
            "frozen_execution_config",
            "source_prefix_hashes",
            "feature_prefix_hash",
            "base_thresholds",
            "state_thresholds",
            "activation_hash",
            "capitulation_hash",
            "overheat_hash",
            "selection_passed",
            "selection_stats",
            "selection_schedule_hashes",
            "selection_grid",
        )
    }


def _freeze_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _freeze_payload(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest(cfg: Config, manifest: dict[str, Any]) -> None:
    if manifest.get("oos_opened") is not False:
        raise RuntimeError("manifest must be explicitly pre-OOS")
    if "oos_opened_at" in manifest or "oos_output" in manifest:
        raise RuntimeError("pre-OOS manifest contains stale OOS metadata")
    if manifest.get("spec_hash") != _spec_hash():
        raise RuntimeError("manifest strategy specification is incompatible")
    if manifest.get("implementation_hash") != _implementation_hash():
        raise RuntimeError("manifest implementation is incompatible")
    if manifest.get("frozen_execution_config") != _frozen_execution_config(cfg):
        raise RuntimeError("runtime execution configuration differs from the freeze")
    if not manifest.get("selection_passed"):
        raise RuntimeError("manifest did not pass the pre-2024 selection contract")
    if manifest.get("freeze_hash") != _freeze_hash(manifest):
        raise RuntimeError("manifest freeze hash mismatch")


def _write_manifest_once(path: Path, payload: dict[str, Any], cfg: Config) -> dict[str, Any]:
    _validate_manifest(cfg, payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _validate_manifest(cfg, existing)
        if existing["freeze_hash"] != payload["freeze_hash"]:
            raise RuntimeError("refusing to overwrite a different frozen manifest")
        return existing
    _atomic_write_json(path, payload)
    return payload


def _mark_oos_opened(path: Path, manifest: dict[str, Any], output: str) -> dict[str, Any]:
    opened = {
        **manifest,
        "phase": "oos_opening",
        "oos_opened": True,
        "oos_opened_at": datetime.now(timezone.utc).isoformat(),
        "oos_output": output,
    }
    _atomic_write_json(path, opened)
    return opened


def state_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_QUANTILES).difference(features.columns)
    if missing:
        raise ValueError(f"missing state features: {sorted(missing)}")
    return pd.DataFrame(
        {
            name: pd.to_numeric(features[name], errors="coerce")
            for name in FEATURE_QUANTILES
        }
    )


def feature_hash(features: pd.DataFrame, mask: np.ndarray | None = None) -> str:
    selected = features if mask is None else features.loc[np.asarray(mask, dtype=bool)]
    return _frame_hash(selected.reset_index(drop=True))


def _threshold_key(column: str, quantile: float) -> str:
    return f"{column}_q{int(round(100 * quantile)):02d}"


def fit_state_thresholds(
    features: pd.DataFrame,
    dates: pd.Series,
    active: np.ndarray,
    *,
    fit_start: str | pd.Timestamp = FIT_START,
    fit_end: str | pd.Timestamp = FIT_END,
    minimum_events: int = 50,
) -> dict[str, Any]:
    if len(features) != len(dates) or len(active) != len(features):
        raise ValueError("features, dates and active must have equal length")
    parsed = pd.to_datetime(dates)
    fit = (
        np.asarray(active, dtype=bool)
        & (parsed >= pd.Timestamp(fit_start)).to_numpy(bool)
        & (parsed < pd.Timestamp(fit_end)).to_numpy(bool)
    )
    output: dict[str, Any] = {"fit_active_events": int(fit.sum())}
    for column, quantiles in FEATURE_QUANTILES.items():
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(float)
        sample = values[fit & np.isfinite(values)]
        if len(sample) < int(minimum_events):
            raise ValueError(f"insufficient active fit events for {column}: {len(sample)}")
        for quantile in quantiles:
            output[_threshold_key(column, quantile)] = float(np.quantile(sample, quantile))
    return output


def build_state_masks(
    features: pd.DataFrame,
    thresholds: dict[str, Any],
    overheat_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if overheat_name not in OVERHEAT_FAMILIES:
        raise KeyError(overheat_name)

    def values(column: str) -> np.ndarray:
        return pd.to_numeric(features[column], errors="coerce").to_numpy(float)

    week = values("htf_1w_return_1")
    width = values("rex_576_range_width_pct")
    quote = values("quote_vol_z_1d")
    capitulation = (
        np.isfinite(week)
        & np.isfinite(width)
        & np.isfinite(quote)
        & (week <= thresholds[_threshold_key("htf_1w_return_1", 0.50)])
        & (
            (width >= thresholds[_threshold_key("rex_576_range_width_pct", 0.50)])
            | (quote <= thresholds[_threshold_key("quote_vol_z_1d", 0.20)])
        )
    )
    first_name, second_name = OVERHEAT_FAMILIES[overheat_name]
    first = values(first_name)
    second = values(second_name)
    overheat = (
        np.isfinite(first)
        & np.isfinite(second)
        & (first >= thresholds[_threshold_key(first_name, 0.67)])
        & (second >= thresholds[_threshold_key(second_name, 0.67)])
    )
    return capitulation, overheat


def schedule_window(
    engine: ExecutionEngine,
    active: np.ndarray,
    capitulation: np.ndarray,
    overheat: np.ndarray,
    *,
    overheat_action: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> list[Trade]:
    if overheat_action not in OVERHEAT_ACTIONS:
        raise KeyError(overheat_action)
    dates = pd.to_datetime(engine.market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    trades: list[Trade] = []
    next_allowed = 0
    for signal in np.flatnonzero(np.asarray(active, dtype=bool) & period):
        signal = int(signal)
        if signal < next_allowed:
            continue
        if capitulation[signal]:
            take_bps = int(SPEC["capitulation_take_bps"])
        elif overheat[signal]:
            if overheat_action == "skip":
                continue
            take_bps = int(SPEC["capitulation_take_bps"])
        else:
            take_bps = int(SPEC["normal_take_bps"])
        trade = engine.trade_at(
            signal,
            int(SPEC["side"]),
            int(SPEC["hold_bars"]),
            take_bps,
            int(SPEC["stop_bps"]),
        )
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: stats[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "trades",
            "mean_net_bps",
            "win_rate",
        )
    }


def _schedules_and_stats(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    active: np.ndarray,
    capitulation: np.ndarray,
    overheat: np.ndarray,
    cfg: Config,
    *,
    overheat_action: str,
    windows: dict[str, tuple[str, str]],
) -> tuple[dict[str, list[Trade]], dict[str, dict[str, Any]]]:
    engine_cfg = _execution_config(cfg, cfg.leverage)
    engine = ExecutionEngine(market, funding, engine_cfg)
    schedules = {
        name: schedule_window(
            engine,
            active,
            capitulation,
            overheat,
            overheat_action=overheat_action,
            start=start,
            end=end,
        )
        for name, (start, end) in windows.items()
    }
    stats = {
        name: _slim(equity_stats(schedules[name], start=start, end=end, cfg=engine_cfg))
        for name, (start, end) in windows.items()
    }
    return schedules, stats


def selection_passes(stats: dict[str, dict[str, Any]]) -> bool:
    support = (
        stats["train"]["trades"] >= 60
        and stats["select_2023"]["trades"] >= 12
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"])
        >= 5
    )
    stable = all(
        stats[name]["absolute_return_pct"] > 0.0
        for name in (
            "train_2020h2",
            "train_2021",
            "train_2022",
            "select_2023_h1",
            "select_2023_h2",
        )
    )
    target = all(
        stats[name]["cagr_to_strict_mdd"] >= 3.0
        for name in ("train", "select_2023", "pre_2024")
    )
    risk = all(
        stats[name]["strict_mdd_pct"] <= 15.0
        for name in ("train", "select_2023", "pre_2024")
    )
    return bool(support and stable and target and risk)


def oos_passes(stats: dict[str, dict[str, Any]]) -> bool:
    return bool(
        all(stats[name]["absolute_return_pct"] > 0.0 for name in OOS_GATE_WINDOWS)
        and all(stats[name]["cagr_to_strict_mdd"] >= 3.0 for name in OOS_GATE_WINDOWS)
        and all(stats[name]["strict_mdd_pct"] <= 15.0 for name in OOS_GATE_WINDOWS)
        and stats["test_2024"]["trades"] >= 12
        and stats["eval_2025"]["trades"] >= 12
        and stats["oos_2024_2026"]["trades"] >= 30
    )


def _selection_score(stats: dict[str, dict[str, Any]]) -> list[float]:
    ratios = [stats[name]["cagr_to_strict_mdd"] for name in ("train", "select_2023", "pre_2024")]
    return [float(min(ratios)), float(np.median(ratios)), float(stats["pre_2024"]["trades"])]


def _implementation_hash() -> str:
    functions = (
        state_feature_frame,
        feature_hash,
        fit_state_thresholds,
        build_state_masks,
        schedule_window,
        selection_passes,
        oos_passes,
        decision_mask,
        live_decision_features,
        _fit_active,
        _load_bundle,
        ExecutionEngine.trade_at,
        equity_stats,
    )
    source = "\n\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(source.encode()).hexdigest()


def _write_docs(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    is_oos = "oos_stats" in payload
    stats = payload["oos_stats"] if is_oos else payload["selection_stats"]
    windows = tuple(FUTURE_WINDOWS) if is_oos else SELECTION_WINDOWS
    title = (
        "# Pullback premium-overheat state machine OOS"
        if is_oos
        else "# Pullback premium-overheat state machine selection"
    )
    status = (
        ("accepted" if payload["oos_passed"] else "rejected") + " by the frozen OOS gate"
        if is_oos
        else "frozen pre-2024 candidate; 2024+ not opened for this exact policy"
    )
    lines = [
        title,
        "",
        f"- Status: **{status}**.",
        "- Entry: causal confirmed pullback squeeze; next 5-minute open.",
        "- Capitulation: weak week AND (wide 48h range OR dry quote activity) -> 4% take.",
        "- Premium overheat: high premium-index change AND high 48h range position -> skip.",
        "- Orderly remainder: 12% take. All routes use a 48h cap and no stop.",
        "- Leverage 0.50x; cost 6bp/notional/side plus realized funding.",
        "- Strict MDD uses global/pre-entry HWM and favorable-then-adverse position envelopes.",
        "- Eight state-machine cells were selected on physically truncated pre-2024 data.",
        "- Family-level 2024+ is not pristine because related pullback variants were previously inspected; the exact policy was not selected from future outcomes.",
        "",
        "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in windows:
        value = stats[name]
        lines.append(
            f"| {name} | {value['absolute_return_pct']:.2f}% | {value['cagr_pct']:.2f}% | "
            f"{value['strict_mdd_pct']:.2f}% | {value['cagr_to_strict_mdd']:.2f} | "
            f"{value['trades']} |"
        )
    if is_oos:
        lines.extend(
            [
                "",
                "`holdout_2026` is a short diagnostic and is not part of the frozen OOS gate.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Selection grid",
                "",
                "Only `premium_range_overheat + skip` passed. COUA, CERF and PWPM transition families were also tested and rejected before this state-machine refinement.",
                "",
                "```json",
                json.dumps(payload["state_thresholds"], indent=2, ensure_ascii=False),
                "```",
            ]
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection(cfg: Config) -> dict[str, Any]:
    if not np.isclose(cfg.leverage, float(SPEC["leverage"])):
        raise ValueError("selection leverage differs from the frozen specification")
    market, raw_features, funding, source_hashes = _load_bundle(
        cfg, cutoff=SELECTION_END, premium_tolerance=cfg.live_premium_tolerance
    )
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection source was not physically truncated")
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    features = live_decision_features(raw_features)
    active, base_thresholds = _fit_active(features, dates, decisions)
    state_features = state_feature_frame(features)
    state_thresholds = fit_state_thresholds(state_features, dates, active)

    grid: list[dict[str, Any]] = []
    schedules_by_key: dict[tuple[str, str], dict[str, list[Trade]]] = {}
    for overheat_name in OVERHEAT_FAMILIES:
        capitulation, overheat = build_state_masks(state_features, state_thresholds, overheat_name)
        for action in OVERHEAT_ACTIONS:
            schedules, stats = _schedules_and_stats(
                market,
                funding,
                active,
                capitulation,
                overheat,
                cfg,
                overheat_action=action,
                windows=PRE2024_WINDOWS,
            )
            key = (overheat_name, action)
            schedules_by_key[key] = schedules
            grid.append(
                {
                    "overheat": overheat_name,
                    "action": action,
                    "selection_passed": selection_passes(stats),
                    "score": _selection_score(stats),
                    "stats": stats,
                }
            )
    grid.sort(key=lambda row: (row["selection_passed"], *row["score"]), reverse=True)
    champion = grid[0]
    if {key: champion[key] for key in FROZEN_CHAMPION} != FROZEN_CHAMPION:
        raise RuntimeError("the frozen state-machine champion is no longer rank one")
    if not champion["selection_passed"]:
        raise RuntimeError("the frozen state-machine champion no longer passes selection")
    selected_key = (champion["overheat"], champion["action"])
    selected_schedules = schedules_by_key[selected_key]
    capitulation, selected_overheat = build_state_masks(
        state_features, state_thresholds, champion["overheat"]
    )

    payload: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_2024_freeze",
        "oos_opened": False,
        "selection_end": SELECTION_END,
        "config": asdict(cfg),
        "spec": SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "frozen_execution_config": _frozen_execution_config(cfg),
        "source_prefix_hashes": source_hashes,
        "feature_prefix_hash": feature_hash(state_features),
        "base_thresholds": base_thresholds,
        "state_thresholds": state_thresholds,
        "activation_hash": _activation_hash(active, dates),
        "capitulation_hash": _activation_hash(capitulation, dates),
        "overheat_hash": _activation_hash(selected_overheat, dates),
        "selection_passed": True,
        "selection_stats": champion["stats"],
        "selection_schedule_hashes": {
            name: _schedule_hash(selected_schedules[name]) for name in SELECTION_WINDOWS
        },
        "selection_grid": grid,
    }
    payload["freeze_hash"] = _freeze_hash(payload)
    stored = _write_manifest_once(Path(cfg.manifest_output), payload, cfg)
    _atomic_write_json(Path(cfg.output), stored)
    _write_docs(cfg.docs_output, stored)
    return stored


def _validate_selection_prefix(cfg: Config, manifest: dict[str, Any]) -> pd.DataFrame:
    market, raw_features, funding, source_hashes = _load_bundle(
        cfg, cutoff=SELECTION_END, premium_tolerance=cfg.live_premium_tolerance
    )
    if source_hashes != manifest["source_prefix_hashes"]:
        raise RuntimeError("pre-2024 source prefix changed after freeze")
    dates = pd.to_datetime(market["date"])
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    features = live_decision_features(raw_features)
    active, base_thresholds = _fit_active(features, dates, decisions)
    if base_thresholds != manifest["base_thresholds"]:
        raise RuntimeError("base thresholds changed after freeze")
    state_features = state_feature_frame(features)
    if feature_hash(state_features) != manifest["feature_prefix_hash"]:
        raise RuntimeError("state feature prefix changed after freeze")
    state_thresholds = fit_state_thresholds(state_features, dates, active)
    if state_thresholds != manifest["state_thresholds"]:
        raise RuntimeError("state thresholds changed after freeze")
    capitulation, overheat = build_state_masks(
        state_features, state_thresholds, FROZEN_CHAMPION["overheat"]
    )
    hashes = (
        _activation_hash(active, dates),
        _activation_hash(capitulation, dates),
        _activation_hash(overheat, dates),
    )
    expected = (
        manifest["activation_hash"],
        manifest["capitulation_hash"],
        manifest["overheat_hash"],
    )
    if hashes != expected:
        raise RuntimeError("pre-2024 state activation changed after freeze")
    schedules, stats = _schedules_and_stats(
        market,
        funding,
        active,
        capitulation,
        overheat,
        cfg,
        overheat_action=FROZEN_CHAMPION["action"],
        windows=PRE2024_WINDOWS,
    )
    schedule_hashes = {name: _schedule_hash(schedules[name]) for name in SELECTION_WINDOWS}
    if schedule_hashes != manifest["selection_schedule_hashes"] or stats != manifest["selection_stats"]:
        raise RuntimeError("pre-2024 trade replay changed after freeze")
    return market


def _oos(cfg: Config) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if not manifest_path.exists():
        raise FileNotFoundError("pre-2024 manifest must exist before --open-oos")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(cfg, manifest)
    opened = _mark_oos_opened(manifest_path, manifest, cfg.output)
    selection_market = _validate_selection_prefix(cfg, manifest)

    market, raw_features, funding, _ = _load_bundle(
        cfg, cutoff=cfg.exclude_from, premium_tolerance=cfg.live_premium_tolerance
    )
    dates = pd.to_datetime(market["date"])
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    features = live_decision_features(raw_features)
    active, base_thresholds = _fit_active(features, dates, decisions)
    if base_thresholds != manifest["base_thresholds"]:
        raise RuntimeError("full replay changed frozen base thresholds")
    state_features = state_feature_frame(features)
    prefix = (dates < pd.Timestamp(SELECTION_END)).to_numpy(bool)
    prefix_dates = dates.loc[prefix].reset_index(drop=True)
    selection_dates = pd.to_datetime(selection_market["date"]).reset_index(drop=True)
    if not prefix_dates.equals(selection_dates):
        raise RuntimeError("full-run date prefix changed after freeze")
    if feature_hash(state_features, prefix) != manifest["feature_prefix_hash"]:
        raise RuntimeError("full-run state feature prefix changed after freeze")
    capitulation, overheat = build_state_masks(
        state_features, manifest["state_thresholds"], FROZEN_CHAMPION["overheat"]
    )
    actual_hashes = (
        _activation_hash(active[prefix], prefix_dates),
        _activation_hash(capitulation[prefix], prefix_dates),
        _activation_hash(overheat[prefix], prefix_dates),
    )
    expected_hashes = (
        manifest["activation_hash"],
        manifest["capitulation_hash"],
        manifest["overheat_hash"],
    )
    if actual_hashes != expected_hashes:
        raise RuntimeError("full-run state prefix changed after freeze")
    schedules, stats = _schedules_and_stats(
        market,
        funding,
        active,
        capitulation,
        overheat,
        cfg,
        overheat_action=FROZEN_CHAMPION["action"],
        windows=FUTURE_WINDOWS,
    )
    result = {
        **opened,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "frozen_oos_replay",
        "data_end": str(dates.max()),
        "oos_passed": oos_passes(stats),
        "oos_stats": stats,
        "oos_schedule_hashes": {
            name: _schedule_hash(schedules[name]) for name in FUTURE_WINDOWS
        },
    }
    _atomic_write_json(Path(cfg.output), result)
    _write_docs(cfg.docs_output, result)
    return result


def run(cfg: Config) -> dict[str, Any]:
    return _oos(cfg) if cfg.open_oos else _selection(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, field in Config.__dataclass_fields__.items():
        default = field.default
        flag = "--" + name.replace("_", "-")
        if name == "open_oos":
            parser.add_argument(flag, action="store_true")
        elif isinstance(default, float):
            parser.add_argument(flag, type=float, default=default)
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, default=default)
        else:
            parser.add_argument(flag, default=default)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    stats = payload.get("oos_stats", payload.get("selection_stats", {}))
    print(
        json.dumps(
            {
                "phase": payload["phase"],
                "selection_passed": payload.get("selection_passed"),
                "oos_passed": payload.get("oos_passed"),
                "stats": stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
