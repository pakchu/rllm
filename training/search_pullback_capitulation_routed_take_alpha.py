"""Freeze and replay a conditional-exit pullback-squeeze candidate.

This experiment does not blend alpha scores.  It keeps the already-causal
confirmed pullback-squeeze entry and routes each trade to one of two fixed
take-profit levels from a train-only interaction:

* completed weekly return is weak; and
* either the completed 48-hour range is wide or one-day quote activity is dry.

The interaction is intended to identify capitulation/exhaustion squeezes whose
large favorable envelope otherwise dominates the conservative strict-MDD
metric.  Stress/exhaustion trades lock 4%; other trades retain a 12% target.
Both routes have a 48-hour cap and no stop.

Selection is physically truncated before 2024.  ``--open-oos`` requires a
matching frozen manifest, burns its one-shot seal before reading future rows,
and replays only the frozen specification.
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
LEVERAGE_GRID = (0.50, 0.55, 0.60, 0.65, 0.70)
SPEC: dict[str, Any] = {
    "name": "pullback_capitulation_routed_take",
    "side": 1,
    "hold_bars": 576,
    "hold_hours": 48.0,
    "stress_take_bps": 400,
    "normal_take_bps": 1_200,
    "stop_bps": NO_STOP_BPS,
    "week_feature": "htf_1w_return_1",
    "week_quantile": 0.50,
    "range_feature": "rex_576_range_width_pct",
    "range_quantile": 0.50,
    "quote_activity_feature": "quote_vol_z_1d",
    "quote_activity_quantile": 0.20,
    "interaction": "week_low AND (range_wide OR quote_activity_dry)",
    "leverage_grid": list(LEVERAGE_GRID),
    "selected_leverage": 0.60,
    "same_bar_policy": "stop_before_take",
    "discovery_accounting": {
        "audited_trade_features": 23,
        "conditional_route_cells": 12,
        "exit_refinement_cells": 12,
        "operating_leverage_points": 5,
        "earlier_rejected_mechanism_families": 8,
        "status": "retrospective multiplicity disclosure; 2024+ remained sealed",
    },
}
ROUTE_FEATURES = (
    str(SPEC["week_feature"]),
    str(SPEC["range_feature"]),
    str(SPEC["quote_activity_feature"]),
)
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
    output: str = "results/pullback_capitulation_routed_take_selection_2026-07-15.json"
    manifest_output: str = "results/pullback_capitulation_routed_take_manifest_2026-07-15.json"
    docs_output: str = "docs/pullback-capitulation-routed-take-selection-2026-07-15.md"
    leverage: float = 0.60
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
            "route_thresholds",
            "activation_hash",
            "route_hash",
            "selection_passed",
            "selection_stats",
            "selection_schedule_hashes",
            "operating_sweep",
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


def _write_manifest_once(path: Path, payload: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """Create the pre-OOS seal once and refuse incompatible replacement."""

    _validate_manifest(cfg, payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _validate_manifest(cfg, existing)
        if existing["freeze_hash"] != payload["freeze_hash"]:
            raise RuntimeError("refusing to overwrite a different frozen manifest")
        return existing
    _atomic_write_json(path, payload)
    return payload


def route_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    missing = set(ROUTE_FEATURES).difference(features.columns)
    if missing:
        raise ValueError(f"missing route features: {sorted(missing)}")
    return pd.DataFrame(
        {name: pd.to_numeric(features[name], errors="coerce") for name in ROUTE_FEATURES}
    )


def feature_hash(features: pd.DataFrame, mask: np.ndarray | None = None) -> str:
    selected = features if mask is None else features.loc[np.asarray(mask, dtype=bool)]
    return _frame_hash(selected.reset_index(drop=True))


def fit_route_thresholds(
    features: pd.DataFrame,
    dates: pd.Series,
    active: np.ndarray,
    *,
    fit_start: str | pd.Timestamp = FIT_START,
    fit_end: str | pd.Timestamp = FIT_END,
    minimum_events: int = 50,
) -> dict[str, float]:
    """Fit route cutoffs on active fit events only."""

    if len(features) != len(dates) or len(active) != len(features):
        raise ValueError("features, dates and active must have equal length")
    parsed = pd.to_datetime(dates)
    fit = (
        np.asarray(active, dtype=bool)
        & (parsed >= pd.Timestamp(fit_start)).to_numpy(bool)
        & (parsed < pd.Timestamp(fit_end)).to_numpy(bool)
    )
    output: dict[str, float] = {}
    settings = (
        (str(SPEC["week_feature"]), float(SPEC["week_quantile"]), "week_low"),
        (str(SPEC["range_feature"]), float(SPEC["range_quantile"]), "range_wide"),
        (
            str(SPEC["quote_activity_feature"]),
            float(SPEC["quote_activity_quantile"]),
            "quote_activity_dry",
        ),
    )
    for column, quantile, label in settings:
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(float)
        sample = values[fit & np.isfinite(values)]
        if len(sample) < int(minimum_events):
            raise ValueError(f"insufficient active fit events for {column}: {len(sample)}")
        output[label] = float(np.quantile(sample, quantile))
    output.update(
        {
            "week_quantile": float(SPEC["week_quantile"]),
            "range_quantile": float(SPEC["range_quantile"]),
            "quote_activity_quantile": float(SPEC["quote_activity_quantile"]),
            "fit_active_events": int(fit.sum()),
        }
    )
    return output


def build_stress_mask(features: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    week = pd.to_numeric(features[str(SPEC["week_feature"])], errors="coerce").to_numpy(float)
    width = pd.to_numeric(features[str(SPEC["range_feature"])], errors="coerce").to_numpy(float)
    quote = pd.to_numeric(
        features[str(SPEC["quote_activity_feature"])], errors="coerce"
    ).to_numpy(float)
    finite = np.isfinite(week) & np.isfinite(width) & np.isfinite(quote)
    return finite & (week <= float(thresholds["week_low"])) & (
        (width >= float(thresholds["range_wide"]))
        | (quote <= float(thresholds["quote_activity_dry"]))
    )


def schedule_window(
    engine: ExecutionEngine,
    active: np.ndarray,
    stress: np.ndarray,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> list[Trade]:
    dates = pd.to_datetime(engine.market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    trades: list[Trade] = []
    next_allowed = 0
    for signal in np.flatnonzero(np.asarray(active, dtype=bool) & period):
        signal = int(signal)
        if signal < next_allowed:
            continue
        take_bps = int(SPEC["stress_take_bps"] if stress[signal] else SPEC["normal_take_bps"])
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
    stress: np.ndarray,
    cfg: Config,
    *,
    leverage: float,
    windows: dict[str, tuple[str, str]],
) -> tuple[dict[str, list[Trade]], dict[str, dict[str, Any]]]:
    engine_cfg = _execution_config(cfg, leverage)
    engine = ExecutionEngine(market, funding, engine_cfg)
    schedules = {
        name: schedule_window(engine, active, stress, start=start, end=end)
        for name, (start, end) in windows.items()
    }
    stats = {
        name: _slim(equity_stats(trades, start=start, end=end, cfg=engine_cfg))
        for name, trades in schedules.items()
        for start, end in (windows[name],)
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


def _implementation_hash() -> str:
    functions = (
        route_feature_frame,
        feature_hash,
        fit_route_thresholds,
        build_stress_mask,
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
        "# Pullback capitulation-routed take OOS"
        if is_oos
        else "# Pullback capitulation-routed take selection"
    )
    status = (
        ("accepted" if payload["oos_passed"] else "rejected") + " by the frozen OOS gate"
        if is_oos
        else "frozen pre-2024 candidate; 2024+ not opened"
    )
    lines = [
        title,
        "",
        f"- Status: **{status}**.",
        "- Entry signal: confirmed pullback squeeze at a completed hourly boundary; enter next 5-minute open.",
        "- Route: weak completed weekly return AND (wide completed 48h range OR dry 1d quote activity).",
        "- Stress route: 4% take; normal route: 12% take; both use 48h cap and no stop.",
        f"- Leverage: {float(SPEC['selected_leverage']):.2f}x; cost: 6bp/notional/side plus realized funding.",
        "- Strict MDD: global/pre-entry HWM plus position favorable envelope before adverse envelope.",
        "- Selection sources were physically truncated before `2024-01-01`; route thresholds use active fit events only.",
        "- Multiplicity disclosed in the frozen specification; this is a conditional interaction, not a score blend.",
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
                "`holdout_2026` is reported as a shorter diagnostic window and is not part of the frozen OOS gate.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Frozen thresholds",
                "",
                "```json",
                json.dumps(payload["route_thresholds"], indent=2, ensure_ascii=False),
                "```",
                "",
                "## Search accounting",
                "",
                "The route emerged after 23 trade-feature diagnostics, 12 conditional-route cells, "
                "12 bounded exit refinements, five leverage points, and eight earlier rejected mechanism families. "
                "This multiplicity is retrospective and is not presented as a pristine single-hypothesis test.",
            ]
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection(cfg: Config) -> dict[str, Any]:
    if not np.isclose(cfg.leverage, float(SPEC["selected_leverage"])):
        raise ValueError("selection leverage must equal the frozen operating leverage")
    market, raw_features, funding, source_hashes = _load_bundle(
        cfg, cutoff=SELECTION_END, premium_tolerance=cfg.live_premium_tolerance
    )
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection source was not physically truncated")
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    features = live_decision_features(raw_features)
    active, base_thresholds = _fit_active(features, dates, decisions)
    route_features = route_feature_frame(features)
    route_thresholds = fit_route_thresholds(route_features, dates, active)
    stress = build_stress_mask(route_features, route_thresholds)

    sweep: list[dict[str, Any]] = []
    selected_schedules: dict[str, list[Trade]] | None = None
    selected_stats: dict[str, dict[str, Any]] | None = None
    for leverage in LEVERAGE_GRID:
        schedules, stats = _schedules_and_stats(
            market,
            funding,
            active,
            stress,
            cfg,
            leverage=leverage,
            windows=PRE2024_WINDOWS,
        )
        row = {"leverage": leverage, "selection_passed": selection_passes(stats), "stats": stats}
        sweep.append(row)
        if np.isclose(leverage, float(SPEC["selected_leverage"])):
            selected_schedules, selected_stats = schedules, stats
    passing = [float(row["leverage"]) for row in sweep if row["selection_passed"]]
    if not passing or not np.isclose(min(passing), float(SPEC["selected_leverage"])):
        raise RuntimeError("the frozen leverage is no longer the smallest passing operating point")
    if selected_schedules is None or selected_stats is None or not selection_passes(selected_stats):
        raise RuntimeError("the frozen candidate no longer passes pre-2024 selection")

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
        "feature_prefix_hash": feature_hash(route_features),
        "base_thresholds": base_thresholds,
        "route_thresholds": route_thresholds,
        "activation_hash": _activation_hash(active, dates),
        "route_hash": _activation_hash(stress, dates),
        "selection_passed": True,
        "selection_stats": selected_stats,
        "selection_schedule_hashes": {
            name: _schedule_hash(selected_schedules[name]) for name in SELECTION_WINDOWS
        },
        "operating_sweep": sweep,
    }
    payload["freeze_hash"] = _freeze_hash(payload)
    stored = _write_manifest_once(Path(cfg.manifest_output), payload, cfg)
    _atomic_write_json(Path(cfg.output), stored)
    _write_docs(cfg.docs_output, stored)
    return stored


def _validate_selection_prefix(
    cfg: Config, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
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
        raise RuntimeError("base fit thresholds changed after freeze")
    route_features = route_feature_frame(features)
    if feature_hash(route_features) != manifest["feature_prefix_hash"]:
        raise RuntimeError("pre-2024 route feature prefix changed after freeze")
    route_thresholds = fit_route_thresholds(route_features, dates, active)
    if route_thresholds != manifest["route_thresholds"]:
        raise RuntimeError("route thresholds changed after freeze")
    stress = build_stress_mask(route_features, manifest["route_thresholds"])
    if _activation_hash(active, dates) != manifest["activation_hash"]:
        raise RuntimeError("pre-2024 entry activation changed after freeze")
    if _activation_hash(stress, dates) != manifest["route_hash"]:
        raise RuntimeError("pre-2024 route activation changed after freeze")
    schedules, stats = _schedules_and_stats(
        market,
        funding,
        active,
        stress,
        cfg,
        leverage=float(SPEC["selected_leverage"]),
        windows=PRE2024_WINDOWS,
    )
    hashes = {name: _schedule_hash(schedules[name]) for name in SELECTION_WINDOWS}
    if hashes != manifest["selection_schedule_hashes"] or stats != manifest["selection_stats"]:
        raise RuntimeError("pre-2024 trade replay changed after freeze")
    return market, raw_features, funding, active, stress


def _oos(cfg: Config) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if not manifest_path.exists():
        raise FileNotFoundError("pre-2024 manifest must exist before --open-oos")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(cfg, manifest)

    # Burn the one-shot seal before any future source is opened.  A subsequent
    # prefix failure remains visible and cannot be retried as a pristine OOS.
    opened = _mark_oos_opened(manifest_path, manifest, cfg.output)
    selection_market, _, _, _, _ = _validate_selection_prefix(cfg, manifest)

    market, raw_features, funding, _ = _load_bundle(
        cfg, cutoff=cfg.exclude_from, premium_tolerance=cfg.live_premium_tolerance
    )
    dates = pd.to_datetime(market["date"])
    features = live_decision_features(raw_features)
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    active, base_thresholds = _fit_active(features, dates, decisions)
    if base_thresholds != manifest["base_thresholds"]:
        raise RuntimeError("full replay changed frozen base thresholds")
    route_features = route_feature_frame(features)
    prefix = (dates < pd.Timestamp(SELECTION_END)).to_numpy(bool)
    prefix_dates = dates.loc[prefix].reset_index(drop=True)
    selection_dates = pd.to_datetime(selection_market["date"]).reset_index(drop=True)
    if not prefix_dates.equals(selection_dates):
        raise RuntimeError("full-run date prefix changed after freeze")
    if feature_hash(route_features, prefix) != manifest["feature_prefix_hash"]:
        raise RuntimeError("full-run route feature prefix changed after freeze")
    if _activation_hash(active[prefix], prefix_dates) != manifest["activation_hash"]:
        raise RuntimeError("full-run entry activation prefix changed after freeze")
    stress = build_stress_mask(route_features, manifest["route_thresholds"])
    if _activation_hash(stress[prefix], prefix_dates) != manifest["route_hash"]:
        raise RuntimeError("full-run route activation prefix changed after freeze")

    schedules, stats = _schedules_and_stats(
        market,
        funding,
        active,
        stress,
        cfg,
        leverage=float(SPEC["selected_leverage"]),
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
