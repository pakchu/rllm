"""Freeze and replay a causal BOCPD exit router for the pullback alpha.

The frozen pullback premium-overheat state machine keeps ownership of entries,
the overheat skip, and the capitulation TP4 action.  This experiment gives a
causal BOCPD state only one bounded responsibility: route otherwise-normal
trades between TP4 and TP12.  All router parameters and state actions are fit
before 2024 and written to a one-shot manifest before later rows may be read.
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

from training.audit_causal_bocpd_pullback_overlay import (
    build_bocpd_state,
    completed_hour_features,
    exact_hour_map,
)
from training.audit_confirmed_pullback_squeeze_live_parity import (
    PRE2024_WINDOWS,
    _activation_hash,
    _execution_config,
    _fit_active,
    _load_bundle,
    decision_mask,
    live_decision_features,
)
from training.search_bocpd_state_gated_alpha import _model_output
from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.search_pullback_premium_overheat_state_machine_alpha import (
    Config as PullbackConfig,
    FIT_END,
    FIT_START,
    FROZEN_CHAMPION as PULLBACK_CHAMPION,
    FUTURE_WINDOWS,
    NO_STOP_BPS,
    SELECTION_END,
    SPEC as PULLBACK_SPEC,
    _selection_score,
    _schedules_and_stats,
    build_state_masks,
    fit_state_thresholds,
    oos_passes,
    selection_passes,
    state_feature_frame,
)


ROUTER_SPEC: dict[str, Any] = {
    "name": "causal_bocpd_pullback_exit_router",
    "base": "pullback_premium_overheat_state_machine",
    "hourly_observation": "[H-1h,H) labelled H; exact H boundary map",
    "columns": ["ret1", "flow24"],
    "hazard_hours": [168, 336],
    "primary_quantiles": [[0.25, 0.75], [0.33, 0.67]],
    "short_mass_quantile": 0.75,
    "secondary_quantile": 0.50,
    "risk_lambdas": [0.0, 0.25],
    "minimum_state_trades": 5,
    "normal_actions": ["tp4", "tp12"],
    "grid_cells": 8,
    "selection_rule": "pass absolute gate and lexicographically beat frozen base",
}
FROZEN_CHAMPION = {
    "hazard_hours": 168,
    "primary_quantiles": [0.33, 0.67],
    "risk_lambda": 0.0,
    "minimum_state_trades": 5,
    "actions": ["tp4", "tp12"],
}
FROZEN_CONFIG_KEYS = (
    "exclude_from",
    "window_size",
    "leverage",
    "fee_rate",
    "slippage_rate",
    "funding_tolerance",
    "live_premium_tolerance",
)


@dataclass(frozen=True)
class Config(PullbackConfig):
    output: str = "results/causal_bocpd_pullback_exit_router_selection_2026-07-15.json"
    manifest_output: str = "results/causal_bocpd_pullback_exit_router_manifest_2026-07-15.json"
    docs_output: str = "docs/causal-bocpd-pullback-exit-router-selection-2026-07-15.md"
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
    encoded = json.dumps(
        ROUTER_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _state_hash(states: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(states, dtype=np.int16).tobytes()).hexdigest()


def _frozen_execution_config(cfg: Config) -> dict[str, Any]:
    values = asdict(cfg)
    return {key: values[key] for key in FROZEN_CONFIG_KEYS}


def trade_utility(
    trade: Trade,
    risk_lambda: float,
    *,
    leverage: float = 0.5,
    cost: float = 0.0006,
) -> float:
    net = (
        (1.0 - leverage * cost) ** 2
        * trade.price_factor
        * trade.funding_factor
        - 1.0
    )
    adverse_factor = (
        (1.0 - leverage * cost)
        * trade.funding_debit_factor
        * trade.adverse_price_factor
    )
    adverse = max(0.0, 1.0 - adverse_factor)
    return float(net - float(risk_lambda) * adverse)


def fit_state_actions(
    engine: ExecutionEngine,
    train_trades: list[Trade],
    states: np.ndarray,
    capitulation: np.ndarray,
    *,
    risk_lambda: float,
    minimum_state_trades: int = 5,
) -> tuple[dict[int, str], dict[str, Any]]:
    records: dict[int, dict[str, list[float]]] = {}
    for base_trade in train_trades:
        signal = int(base_trade.signal_position)
        if bool(capitulation[signal]):
            continue
        state_id = int(states[signal])
        if state_id < 0:
            continue
        tp4 = engine.trade_at(
            signal, 1, int(PULLBACK_SPEC["hold_bars"]), 400, NO_STOP_BPS
        )
        tp12 = engine.trade_at(
            signal, 1, int(PULLBACK_SPEC["hold_bars"]), 1200, NO_STOP_BPS
        )
        if tp4 is None or tp12 is None:
            continue
        bank = records.setdefault(state_id, {"tp4": [], "tp12": []})
        bank["tp4"].append(trade_utility(tp4, risk_lambda))
        bank["tp12"].append(trade_utility(tp12, risk_lambda))

    actions: dict[int, str] = {}
    quality: dict[str, Any] = {}
    for state_id, values in records.items():
        count = len(values["tp4"])
        means = {
            action: float(np.mean(action_values))
            for action, action_values in values.items()
        }
        selected = "tp12"
        if count >= int(minimum_state_trades) and means["tp4"] > means["tp12"]:
            selected = "tp4"
        actions[state_id] = selected
        quality[str(state_id)] = {
            "n": count,
            "mean_utility": means,
            "selected": selected,
        }
    return actions, quality


def schedule_window(
    engine: ExecutionEngine,
    active: np.ndarray,
    capitulation: np.ndarray,
    overheat: np.ndarray,
    states: np.ndarray,
    actions: dict[int, str],
    *,
    start: str,
    end: str,
) -> list[Trade]:
    dates = engine.dates
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(
        bool
    )
    trades: list[Trade] = []
    next_allowed = 0
    for signal in np.flatnonzero(np.asarray(active, dtype=bool) & period):
        signal = int(signal)
        if signal < next_allowed:
            continue
        if bool(capitulation[signal]):
            take_bps = 400
        elif bool(overheat[signal]):
            continue
        else:
            action = actions.get(int(states[signal]), "tp12")
            take_bps = 400 if action == "tp4" else 1200
        trade = engine.trade_at(
            signal,
            1,
            int(PULLBACK_SPEC["hold_bars"]),
            take_bps,
            NO_STOP_BPS,
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


def schedules_and_stats(
    engine: ExecutionEngine,
    cfg: Config,
    active: np.ndarray,
    capitulation: np.ndarray,
    overheat: np.ndarray,
    states: np.ndarray,
    actions: dict[int, str],
    *,
    windows: dict[str, tuple[str, str]],
) -> tuple[dict[str, list[Trade]], dict[str, dict[str, Any]]]:
    schedules = {
        name: schedule_window(
            engine,
            active,
            capitulation,
            overheat,
            states,
            actions,
            start=start,
            end=end,
        )
        for name, (start, end) in windows.items()
    }
    engine_cfg = _execution_config(cfg, cfg.leverage)
    stats = {
        name: _slim(equity_stats(trades, start=start, end=end, cfg=engine_cfg))
        for (name, (start, end)), trades in zip(windows.items(), schedules.values())
    }
    return schedules, stats


def _base_context(
    cfg: Config, *, cutoff: str
) -> tuple[dict[str, Any], dict[str, str]]:
    market, raw_features, funding, source_hashes = _load_bundle(
        cfg, cutoff=cutoff, premium_tolerance=cfg.live_premium_tolerance
    )
    dates = pd.to_datetime(market["date"])
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=cfg.window_size)
    features = live_decision_features(raw_features)
    active, base_thresholds = _fit_active(features, dates, decisions)
    execution_features = state_feature_frame(features)
    execution_thresholds = fit_state_thresholds(execution_features, dates, active)
    capitulation, overheat = build_state_masks(
        execution_features,
        execution_thresholds,
        PULLBACK_CHAMPION["overheat"],
    )
    base_schedules, base_stats = _schedules_and_stats(
        market,
        funding,
        active,
        capitulation,
        overheat,
        cfg,
        overheat_action=PULLBACK_CHAMPION["action"],
        windows=PRE2024_WINDOWS,
    )
    if not selection_passes(base_stats):
        raise RuntimeError("frozen base no longer passes pre-2024 selection")
    engine_cfg = _execution_config(cfg, cfg.leverage)
    return (
        {
            "market": market,
            "funding": funding,
            "dates": dates,
            "active": active,
            "capitulation": capitulation,
            "overheat": overheat,
            "base_thresholds": base_thresholds,
            "execution_thresholds": execution_thresholds,
            "base_schedules": base_schedules,
            "base_stats": base_stats,
            "engine": ExecutionEngine(market, funding, engine_cfg),
            "hourly": completed_hour_features(market),
        },
        source_hashes,
    )


def _bocpd_state(
    context: dict[str, Any], *, hazard_hours: int, primary_quantiles: list[float]
) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
    hourly = context["hourly"]
    fit_hour = np.asarray(
        (hourly.index >= pd.Timestamp(FIT_START))
        & (hourly.index < pd.Timestamp(FIT_END)),
        dtype=bool,
    )
    hourly_output, metadata = _model_output(
        hourly,
        fit_hour,
        columns=("ret1", "flow24"),
        secondary_index=1,
        hazard_lambda=int(hazard_hours),
    )
    fit_output = hourly_output[
        (hourly_output["date"] >= pd.Timestamp(FIT_START))
        & (hourly_output["date"] < pd.Timestamp(FIT_END))
    ]
    low_q, high_q = map(float, primary_quantiles)
    thresholds = {
        "primary_low": float(fit_output["primary"].quantile(low_q)),
        "primary_high": float(fit_output["primary"].quantile(high_q)),
        "short_mass_high": float(fit_output["short_mass"].quantile(0.75)),
        "secondary_high": float(fit_output["secondary"].quantile(0.50)),
    }
    mapped = exact_hour_map(context["dates"], hourly_output)
    return build_bocpd_state(mapped, thresholds), metadata, thresholds


def _grid(context: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_score = _selection_score(context["base_stats"])
    for hazard_hours in ROUTER_SPEC["hazard_hours"]:
        for primary_quantiles in ROUTER_SPEC["primary_quantiles"]:
            states, metadata, thresholds = _bocpd_state(
                context,
                hazard_hours=int(hazard_hours),
                primary_quantiles=list(primary_quantiles),
            )
            for risk_lambda in ROUTER_SPEC["risk_lambdas"]:
                actions, quality = fit_state_actions(
                    context["engine"],
                    context["base_schedules"]["train"],
                    states,
                    context["capitulation"],
                    risk_lambda=float(risk_lambda),
                    minimum_state_trades=int(ROUTER_SPEC["minimum_state_trades"]),
                )
                schedules, stats = schedules_and_stats(
                    context["engine"],
                    cfg,
                    context["active"],
                    context["capitulation"],
                    context["overheat"],
                    states,
                    actions,
                    windows=PRE2024_WINDOWS,
                )
                score = _selection_score(stats)
                passed = selection_passes(stats)
                beats_base = bool(tuple(score) > tuple(base_score))
                rows.append(
                    {
                        "spec": {
                            "hazard_hours": int(hazard_hours),
                            "primary_quantiles": list(primary_quantiles),
                            "risk_lambda": float(risk_lambda),
                            "minimum_state_trades": int(
                                ROUTER_SPEC["minimum_state_trades"]
                            ),
                            "actions": list(ROUTER_SPEC["normal_actions"]),
                        },
                        "model": metadata,
                        "state_thresholds": thresholds,
                        "state_actions": {
                            str(state_id): action
                            for state_id, action in actions.items()
                        },
                        "state_quality": quality,
                        "state_hash": _state_hash(states),
                        "selection_schedule_hashes": {
                            name: _schedule_hash(window_trades)
                            for name, window_trades in schedules.items()
                        },
                        "selection_passed": passed,
                        "beats_base": beats_base,
                        "accepted": bool(passed and beats_base),
                        "score": score,
                        "stats": stats,
                    }
                )
    rows.sort(
        key=lambda row: (
            row["accepted"],
            row["selection_passed"],
            *row["score"],
        ),
        reverse=True,
    )
    return rows


def _implementation_hash() -> str:
    functions = (
        completed_hour_features,
        exact_hour_map,
        build_bocpd_state,
        trade_utility,
        fit_state_actions,
        schedule_window,
        schedules_and_stats,
        _bocpd_state,
        decision_mask,
        live_decision_features,
        _fit_active,
        _load_bundle,
        ExecutionEngine.trade_at,
        equity_stats,
    )
    source = "\n\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(source.encode()).hexdigest()


def _freeze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "oos_opened",
            "selection_end",
            "router_spec",
            "spec_hash",
            "implementation_hash",
            "frozen_execution_config",
            "source_prefix_hashes",
            "hourly_feature_prefix_hash",
            "base_thresholds",
            "execution_thresholds",
            "base_activation_hash",
            "capitulation_hash",
            "overheat_hash",
            "base_score",
            "base_stats",
            "selected",
            "selection_stats",
            "selection_schedule_hashes",
            "selection_grid",
        )
    }


def _freeze_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _freeze_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selection_payload(cfg: Config) -> dict[str, Any]:
    context, source_hashes = _base_context(cfg, cutoff=SELECTION_END)
    if len(context["dates"]) and context["dates"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection source was not physically truncated")
    grid = _grid(context, cfg)
    champion = grid[0]
    if champion["spec"] != FROZEN_CHAMPION:
        raise RuntimeError("frozen exit-router champion is no longer rank one")
    if not champion["accepted"]:
        raise RuntimeError("frozen exit-router champion no longer beats the base")
    payload: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_2024_freeze",
        "oos_opened": False,
        "selection_end": SELECTION_END,
        "config": asdict(cfg),
        "router_spec": ROUTER_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "frozen_execution_config": _frozen_execution_config(cfg),
        "source_prefix_hashes": source_hashes,
        "hourly_feature_prefix_hash": _frame_hash(
            context["hourly"].reset_index().rename(columns={"index": "date"})
        ),
        "base_thresholds": context["base_thresholds"],
        "execution_thresholds": context["execution_thresholds"],
        "base_activation_hash": _activation_hash(
            context["active"], context["dates"]
        ),
        "capitulation_hash": _activation_hash(
            context["capitulation"], context["dates"]
        ),
        "overheat_hash": _activation_hash(context["overheat"], context["dates"]),
        "base_score": _selection_score(context["base_stats"]),
        "base_stats": context["base_stats"],
        "selected": {
            key: champion[key]
            for key in (
                "spec",
                "model",
                "state_thresholds",
                "state_actions",
                "state_quality",
                "state_hash",
                "score",
            )
        },
        "selection_stats": champion["stats"],
        "selection_schedule_hashes": champion["selection_schedule_hashes"],
        "selection_grid": grid,
    }
    payload["freeze_hash"] = _freeze_hash(payload)
    return payload


def _validate_manifest(cfg: Config, manifest: dict[str, Any]) -> None:
    if manifest.get("oos_opened") is not False:
        raise RuntimeError("manifest must be pre-OOS")
    if manifest.get("spec_hash") != _spec_hash():
        raise RuntimeError("router specification changed after freeze")
    if manifest.get("implementation_hash") != _implementation_hash():
        raise RuntimeError("router implementation changed after freeze")
    if manifest.get("frozen_execution_config") != _frozen_execution_config(cfg):
        raise RuntimeError("execution configuration changed after freeze")
    if manifest.get("selected", {}).get("spec") != FROZEN_CHAMPION:
        raise RuntimeError("manifest champion mismatch")
    if manifest.get("freeze_hash") != _freeze_hash(manifest):
        raise RuntimeError("manifest freeze hash mismatch")


def _write_manifest_once(path: Path, payload: dict[str, Any], cfg: Config) -> None:
    _validate_manifest(cfg, payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _validate_manifest(cfg, existing)
        if existing["freeze_hash"] != payload["freeze_hash"]:
            raise RuntimeError("refusing to overwrite a different router freeze")
        return
    _atomic_write_json(path, payload)


def _metric(row: dict[str, Any]) -> str:
    return (
        f"{row['absolute_return_pct']:.2f}% / {row['cagr_pct']:.2f}% / "
        f"{row['strict_mdd_pct']:.2f}% / "
        f"{row['cagr_to_strict_mdd']:.2f} / {row['trades']}"
    )


def _write_docs(path: str, payload: dict[str, Any]) -> None:
    oos = bool(payload.get("oos_opened"))
    lines = [
        (
            "# Causal BOCPD pullback exit router OOS"
            if oos
            else "# Causal BOCPD pullback exit router selection"
        ),
        "",
        "Metric format: absolute return / CAGR / strict MDD / CAGR-MDD / trades.",
        "",
    ]
    if not oos:
        base = payload["base_stats"]
        selected = payload["selection_stats"]
        lines += [
            "## Verdict",
            "",
            "**Frozen for OOS.** BOCPD does not gate entries. It routes only normal "
            "state-5 trades to TP4 while preserving every base trade and all "
            "overheat/capitulation behavior.",
            "",
            "| Policy | Train | 2023 selection | Pre-2024 | Score |",
            "|---|---:|---:|---:|---:|",
            f"| Frozen base | {_metric(base['train'])} | {_metric(base['select_2023'])} | {_metric(base['pre_2024'])} | `{payload['base_score']}` |",
            f"| Exit router | {_metric(selected['train'])} | {_metric(selected['select_2023'])} | {_metric(selected['pre_2024'])} | `{payload['selected']['score']}` |",
            "",
            "## Leakage controls",
            "",
            "- The market source is physically truncated before 2024.",
            "- Hour H is built from `[H-1h,H)` and mapped only at exact H.",
            "- Standardization, state thresholds, and TP action quality use only 2020-07 through 2022-12.",
            "- The one-shot manifest must be committed before `--open-oos` can read later rows.",
            "- Next-open entry, realized funding, 6 bp/notional/side, non-overlap, split-contained exits, and strict MDD remain unchanged.",
        ]
    else:
        lines += [
            "## Verdict",
            "",
            ("**OOS gate passed.**" if payload["oos_passed"] else "**OOS gate failed.**"),
            "",
            "| Window | Result |",
            "|---|---:|",
        ]
        for name, row in payload["oos_stats"].items():
            lines.append(f"| {name} | {_metric(row)} |")
        lines += [
            "",
            "The underlying pullback family has prior research-history exposure; "
            "this exact router is a manifest-frozen shadow result, not pristine market OOS.",
        ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection(cfg: Config) -> dict[str, Any]:
    payload = _selection_payload(cfg)
    _write_manifest_once(Path(cfg.manifest_output), payload, cfg)
    _atomic_write_json(Path(cfg.output), payload)
    _write_docs(cfg.docs_output, payload)
    return payload


def _mark_oos_opened(path: Path, manifest: dict[str, Any], output: str) -> None:
    opened = {
        **manifest,
        "phase": "oos_opening",
        "oos_opened": True,
        "oos_opened_at": datetime.now(timezone.utc).isoformat(),
        "oos_output": output,
    }
    _atomic_write_json(path, opened)


def _oos(cfg: Config) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if not manifest_path.exists():
        raise FileNotFoundError("pre-2024 router manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(cfg, manifest)
    prefix = _selection_payload(cfg)
    if prefix["freeze_hash"] != manifest["freeze_hash"]:
        raise RuntimeError("pre-2024 router replay changed after freeze")
    _mark_oos_opened(manifest_path, manifest, cfg.output)

    context, full_source_hashes = _base_context(cfg, cutoff=cfg.exclude_from)
    selected_spec = manifest["selected"]["spec"]
    states, metadata, thresholds = _bocpd_state(
        context,
        hazard_hours=int(selected_spec["hazard_hours"]),
        primary_quantiles=list(selected_spec["primary_quantiles"]),
    )
    if metadata != manifest["selected"]["model"]:
        raise RuntimeError("full-run BOCPD model differs from freeze")
    if thresholds != manifest["selected"]["state_thresholds"]:
        raise RuntimeError("full-run BOCPD thresholds differ from freeze")
    prefix_mask = (context["dates"] < pd.Timestamp(SELECTION_END)).to_numpy(bool)
    if _state_hash(states[prefix_mask]) != manifest["selected"]["state_hash"]:
        raise RuntimeError("full-run BOCPD state prefix differs from freeze")
    actions, quality = fit_state_actions(
        context["engine"],
        context["base_schedules"]["train"],
        states,
        context["capitulation"],
        risk_lambda=float(selected_spec["risk_lambda"]),
        minimum_state_trades=int(selected_spec["minimum_state_trades"]),
    )
    serialized_actions = {str(state_id): action for state_id, action in actions.items()}
    if serialized_actions != manifest["selected"]["state_actions"]:
        raise RuntimeError("full-run state actions differ from freeze")
    if quality != manifest["selected"]["state_quality"]:
        raise RuntimeError("full-run state action quality differs from freeze")
    schedules, stats = schedules_and_stats(
        context["engine"],
        cfg,
        context["active"],
        context["capitulation"],
        context["overheat"],
        states,
        actions,
        windows=FUTURE_WINDOWS,
    )
    payload = {
        **manifest,
        "phase": "oos_result",
        "oos_opened": True,
        "oos_opened_at": datetime.now(timezone.utc).isoformat(),
        "full_source_hashes": full_source_hashes,
        "oos_stats": stats,
        "oos_schedule_hashes": {
            name: _schedule_hash(window_trades)
            for name, window_trades in schedules.items()
        },
        "oos_passed": oos_passes(stats),
    }
    _atomic_write_json(manifest_path, payload)
    _atomic_write_json(Path(cfg.output), payload)
    _write_docs(cfg.docs_output, payload)
    return payload


def run(cfg: Config) -> dict[str, Any]:
    return _oos(cfg) if cfg.open_oos else _selection(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--premium-csv", default=Config.premium_csv)
    parser.add_argument("--exclude-from", default=Config.exclude_from)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    parser.add_argument("--open-oos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(Config(**vars(args)))
    if payload.get("oos_opened"):
        summary = {
            "phase": payload["phase"],
            "oos_passed": payload["oos_passed"],
            "oos_stats": payload["oos_stats"],
        }
    else:
        summary = {
            "phase": payload["phase"],
            "freeze_hash": payload["freeze_hash"],
            "selected": payload["selected"],
            "selection_stats": payload["selection_stats"],
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
