"""Combine frozen continual DVOL and regime-routed REX signals on one path."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_positioning_continual_hgb_alpha import (
    ContinualConfig,
    _load_market,
    _policy_key,
    build_continual_features,
    generate_prequential_signals,
)
from training.search_positioning_hgb_path_alpha import PositioningHgbConfig, _feature_hash
from training.search_rex_pre2024_ml_alpha import (
    FIT_END,
    SELECT_END,
    build_model,
    completed_before,
    feature_matrix,
    feature_names,
    fit_medians,
    load_market_before,
    manifest_hash,
    read_jsonl,
    target_vector,
)
from training.search_rex_regime_routed_alpha import routed_prediction_rows
from training.strict_bar_backtest import _trade_stats, load_market_bars


WINDOWS = {
    "select2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
    "select2023_h1": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-07-01")),
    "select2023_h2": (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-01-01")),
    "test2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
    "eval2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
    "ytd2026": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-02")),
}


@dataclass(frozen=True)
class DvolRexComplementConfig:
    positioning_input_csv: str
    metrics_csv: str
    dvol_csv: str
    positioning_manifest: str
    rex_market_csv: str
    rex_train_jsonl: str
    rex_test_jsonl: str
    rex_eval_jsonl: str
    rex_specialist_manifest: str
    rex_routed_manifest: str
    output: str
    manifest_output: str
    top_n: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    random_state: int = 42


def dvol_events(
    signals: tuple[np.ndarray, np.ndarray],
    policy: dict[str, Any],
    dates: pd.Series,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> list[dict[str, Any]]:
    long_signal, short_signal = signals
    hold = int(policy["hold_bars"])
    stride = int(policy["stride_bars"])
    start, end = window
    positions = np.arange(0, len(dates) - hold - 2, stride, dtype=np.int64)
    mask = (dates.iloc[positions].to_numpy() >= np.datetime64(start)) & (dates.iloc[positions].to_numpy() < np.datetime64(end))
    positions = positions[mask & (long_signal[positions] | short_signal[positions])]
    events: list[dict[str, Any]] = []
    for position in positions:
        side = 1 if long_signal[position] and not short_signal[position] else -1 if short_signal[position] and not long_signal[position] else 0
        exit_position = int(position) + 1 + hold
        if side == 0 or exit_position >= len(dates) or pd.Timestamp(dates.iloc[exit_position]) >= end:
            continue
        events.append({"signal_pos": int(position), "side": side, "hold_bars": hold, "source": "dvol"})
    return events


def rex_events(
    rows: list[dict[str, Any]],
    score_bank: dict[str, np.ndarray],
    pair: dict[str, Any],
    market_dates: pd.Series,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> list[dict[str, Any]]:
    start, end = window
    predictions = routed_prediction_rows(
        rows,
        score_bank,
        pair,
        market_dates=market_dates,
        start=start,
        end=end,
        hold_bars=144,
    )
    return [
        {
            "signal_pos": int(row["signal_pos"]),
            "side": 1 if row["prediction"]["side"] == "LONG" else -1,
            "hold_bars": int(row["prediction"]["hold_bars"]),
            "source": "rex",
        }
        for row in predictions
    ]


def build_schedule(events: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    ordered = sorted(events, key=lambda row: (int(row["signal_pos"]), 0 if row["source"] == "rex" else 1))
    by_position: dict[int, dict[str, Any]] = {}
    for event in ordered:
        by_position.setdefault(int(event["signal_pos"]), event)
    ordered = [by_position[pos] for pos in sorted(by_position)]
    schedule: list[dict[str, Any]] = []
    cursor = 0
    i = 0
    while i < len(ordered):
        while i < len(ordered) and int(ordered[i]["signal_pos"]) < cursor:
            i += 1
        if i >= len(ordered):
            break
        event = dict(ordered[i])
        entry = int(event["signal_pos"]) + 1
        scheduled_exit = entry + int(event["hold_bars"])
        actual_exit = scheduled_exit
        if mode == "rex_short_preempt" and event["source"] == "dvol" and int(event["side"]) > 0:
            for future in ordered[i + 1 :]:
                future_signal = int(future["signal_pos"])
                if future_signal + 1 >= scheduled_exit:
                    break
                if future["source"] == "rex" and int(future["side"]) < 0:
                    actual_exit = future_signal + 1
                    break
        event["entry_pos"] = entry
        event["exit_pos"] = actual_exit
        event["preempted"] = actual_exit < scheduled_exit
        schedule.append(event)
        cursor = actual_exit - 1 if event["preempted"] else actual_exit + 1
        i += 1
    return schedule


def simulate_schedule(
    market: pd.DataFrame,
    events: list[dict[str, Any]],
    *,
    mode: str,
    window: tuple[pd.Timestamp, pd.Timestamp],
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    start, end = window
    schedule = build_schedule(events, mode)
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    dates = pd.to_datetime(market["date"])
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    equity = peak = 1.0
    max_dd = 0.0
    returns: list[float] = []
    sources: Counter[str] = Counter()
    preemptions = 0
    for event in schedule:
        entry = int(event["entry_pos"])
        exit_position = int(event["exit_pos"])
        if entry >= len(market) or exit_position >= len(market):
            continue
        if not (start <= pd.Timestamp(dates.iloc[entry]) < end) or pd.Timestamp(dates.iloc[exit_position]) >= end:
            continue
        entry_price = float(opens[entry])
        if entry_price <= 0 or exit_position <= entry:
            continue
        side = int(event["side"])
        entry_equity = equity
        equity *= 1.0 - cost
        max_dd = max(max_dd, 1.0 - equity / peak)
        segment_high = float(np.max(highs[entry:exit_position]))
        segment_low = float(np.min(lows[entry:exit_position]))
        favorable = segment_high if side > 0 else segment_low
        adverse = segment_low if side > 0 else segment_high
        favorable_eq = max(0.0, equity * (1.0 + leverage * side * (favorable / entry_price - 1.0)))
        intratrade_peak = max(peak, favorable_eq)
        adverse_eq = max(0.0, equity * (1.0 + leverage * side * (adverse / entry_price - 1.0)))
        max_dd = max(max_dd, 1.0 - adverse_eq / intratrade_peak)
        peak = max(peak, intratrade_peak)
        raw_return = side * (float(opens[exit_position]) / entry_price - 1.0)
        equity *= max(0.0, 1.0 + leverage * raw_return)
        equity *= 1.0 - cost
        max_dd = max(max_dd, 1.0 - equity / peak)
        peak = max(peak, equity)
        returns.append(equity / entry_equity - 1.0)
        sources[str(event["source"])] += 1
        preemptions += int(bool(event["preempted"]))
    years = (end - start).total_seconds() / (365.25 * 86400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0 else -100.0
    mdd = max_dd * 100.0
    stats = _trade_stats(returns)
    return {
        "return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "ratio": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "trades": len(returns),
        "source_trades": dict(sources),
        "preemptions": int(preemptions),
        "p_value": float(stats["p_value_mean_ret_approx"]),
    }


def _continual_cfg(cfg: DvolRexComplementConfig) -> tuple[ContinualConfig, PositioningHgbConfig]:
    continual = ContinualConfig(
        input_csv=cfg.positioning_input_csv,
        metrics_csv=cfg.metrics_csv,
        dvol_csv=cfg.dvol_csv,
        output=cfg.output,
        manifest_output=cfg.manifest_output,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        random_state=cfg.random_state,
    )
    hgb = PositioningHgbConfig(
        input_csv=cfg.positioning_input_csv,
        metrics_csv=cfg.metrics_csv,
        output=cfg.output,
        manifest_output=cfg.manifest_output,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        random_state=cfg.random_state,
    )
    return continual, hgb


def _fit_rex_phase(
    cfg: DvolRexComplementConfig,
    phase_market: pd.DataFrame,
    specialist_manifest: dict[str, Any],
    routed_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    pre_rows = read_jsonl(cfg.rex_train_jsonl)
    dates = np.asarray([np.datetime64(str(row["date"])) for row in pre_rows])
    fit_rows = [
        row
        for row, date in zip(pre_rows, dates)
        if date < np.datetime64(FIT_END) and completed_before(row, phase_market["date"], FIT_END, 144)
    ]
    select_rows = [row for row, date in zip(pre_rows, dates) if np.datetime64(FIT_END) <= date < np.datetime64(SELECT_END)]
    names = feature_names(fit_rows)
    if names != specialist_manifest["feature_names"]:
        raise RuntimeError("REX specialist feature names changed")
    medians = fit_medians(fit_rows, names)
    x_fit = feature_matrix(fit_rows, names, medians)
    x_select = feature_matrix(select_rows, names, medians)
    model_specs: dict[str, dict[str, Any]] = {}
    for pair in routed_manifest["pairs"]:
        for key in ("long_policy", "short_policy"):
            policy = pair[key]
            model_specs[str(policy["model"])] = policy["model_spec"]
    models: dict[str, Any] = {}
    scores: dict[str, np.ndarray] = {}
    for name, spec in model_specs.items():
        model = build_model(spec, cfg.random_state)
        model.fit(x_fit, target_vector(fit_rows, str(spec["target"])))
        models[name] = model
        scores[name] = np.asarray(model.predict(x_select), dtype=np.float64)
    return select_rows, names, medians, models, scores


def _continual_policy_key(policy: dict[str, Any]) -> str:
    return _policy_key(
        str(policy["model"]),
        int(policy["train_days"]),
        int(policy["calibration_days"]),
        float(policy["score_quantile"]),
        str(policy["side"]),
    )


def run(cfg: DvolRexComplementConfig) -> dict[str, Any]:
    positioning_manifest = json.loads(Path(cfg.positioning_manifest).read_text())
    specialist_manifest = json.loads(Path(cfg.rex_specialist_manifest).read_text())
    routed_manifest = json.loads(Path(cfg.rex_routed_manifest).read_text())
    continual_cfg, hgb_cfg = _continual_cfg(cfg)

    phase_market = _load_market(continual_cfg, hgb_cfg, cutoff=str(SELECT_END))
    phase_dates = pd.to_datetime(phase_market["date"])
    phase_features = build_continual_features(phase_market, include_dvol=True)
    dvol_policies = positioning_manifest["policies"]
    unique_dvol = [
        {key: policy[key] for key in ("model", "train_days", "calibration_days", "score_quantile", "side")}
        for policy in dvol_policies
    ]
    dvol_signal_bank, _ = generate_prequential_signals(
        phase_market,
        phase_features,
        start="2023-01-01",
        end=str(SELECT_END),
        policies=unique_dvol,
        cfg=continual_cfg,
    )

    rex_phase_market = load_market_before(cfg.rex_market_csv, str(SELECT_END))
    if not phase_dates.reset_index(drop=True).equals(pd.to_datetime(rex_phase_market["date"]).reset_index(drop=True)):
        raise RuntimeError("positioning and REX phase timestamps differ")
    rex_select_rows, rex_names, rex_medians, rex_models, rex_select_scores = _fit_rex_phase(
        cfg, rex_phase_market, specialist_manifest, routed_manifest
    )

    candidates: list[dict[str, Any]] = []
    for dvol_index, dvol_policy in enumerate(dvol_policies):
        dvol_key = _continual_policy_key(dvol_policy)
        for rex_index, rex_pair in enumerate(routed_manifest["pairs"]):
            for mode in ("union", "rex_short_preempt"):
                row: dict[str, Any] = {
                    "dvol_index": dvol_index,
                    "dvol_policy": dvol_policy,
                    "rex_index": rex_index,
                    "rex_pair": rex_pair,
                    "mode": mode,
                }
                metrics: dict[str, Any] = {}
                for window_name in ("select2023", "select2023_h1", "select2023_h2"):
                    window = WINDOWS[window_name]
                    events = dvol_events(dvol_signal_bank[dvol_key], dvol_policy, phase_dates, window)
                    events += rex_events(rex_select_rows, rex_select_scores, rex_pair, rex_phase_market["date"], window)
                    metrics[window_name] = simulate_schedule(
                        phase_market,
                        events,
                        mode=mode,
                        window=window,
                        leverage=cfg.leverage,
                        fee_rate=cfg.fee_rate,
                        slippage_rate=cfg.slippage_rate,
                    )
                row.update(metrics)
                h1, h2 = metrics["select2023_h1"], metrics["select2023_h2"]
                row["selection_score"] = {
                    "positive_halves": int(h1["cagr_pct"] > 0) + int(h2["cagr_pct"] > 0),
                    "min_half_ratio": min(float(h1["ratio"]), float(h2["ratio"])),
                    "full_ratio": float(metrics["select2023"]["ratio"]),
                    "min_half_trades": min(int(h1["trades"]), int(h2["trades"])),
                }
                candidates.append(row)

    def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
        score = row["selection_score"]
        return (
            float(score["positive_halves"]),
            float(score["min_half_trades"] >= 12),
            float(score["min_half_ratio"]),
            float(score["full_ratio"]),
            -float(row["select2023"]["p_value"]),
        )

    selected = sorted(candidates, key=rank_key, reverse=True)[: int(cfg.top_n)]
    manifest_core = {
        "phase": "pre_future_manifest",
        "positioning_manifest_hash": positioning_manifest["manifest_hash"],
        "rex_routed_manifest_hash": routed_manifest["manifest_hash"],
        "phase_positioning_feature_hash": _feature_hash(phase_features, phase_dates),
        "selection": "2023 Top-10 chronological single-position combinations",
        "combinations": [
            {key: row[key] for key in ("dvol_index", "dvol_policy", "rex_index", "rex_pair", "mode")}
            for row in selected
        ],
    }
    frozen_hash = manifest_hash(manifest_core)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), **manifest_core, "manifest_hash": frozen_hash}
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    full_market = _load_market(continual_cfg, hgb_cfg)
    full_dates = pd.to_datetime(full_market["date"])
    prefix = full_market.iloc[: len(phase_market)].reset_index(drop=True)
    if not prefix.equals(phase_market.reset_index(drop=True)):
        raise RuntimeError("positioning pre-2024 market prefix changed")
    full_features = build_continual_features(full_market, include_dvol=True)
    if _feature_hash(full_features.iloc[: len(phase_features)], full_dates.iloc[: len(phase_dates)]) != manifest_core["phase_positioning_feature_hash"]:
        raise RuntimeError("positioning pre-2024 feature prefix changed")
    future_dvol_bank, _ = generate_prequential_signals(
        full_market,
        full_features,
        start="2024-01-01",
        end="2026-06-02",
        policies=unique_dvol,
        cfg=continual_cfg,
    )
    full_dvol_bank: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for policy in unique_dvol:
        key = _continual_policy_key(policy)
        long_signal = np.zeros(len(full_market), dtype=bool)
        short_signal = np.zeros(len(full_market), dtype=bool)
        long_signal[: len(phase_market)] = dvol_signal_bank[key][0]
        short_signal[: len(phase_market)] = dvol_signal_bank[key][1]
        long_signal |= future_dvol_bank[key][0]
        short_signal |= future_dvol_bank[key][1]
        full_dvol_bank[key] = (long_signal, short_signal)

    rex_market = load_market_bars(cfg.rex_market_csv)
    if not full_dates.reset_index(drop=True).equals(pd.to_datetime(rex_market["date"]).reset_index(drop=True)):
        raise RuntimeError("positioning and REX full timestamps differ")
    rex_future_sets = {"test2024": read_jsonl(cfg.rex_test_jsonl), "eval2025_2026": read_jsonl(cfg.rex_eval_jsonl)}
    rex_future_scores: dict[str, dict[str, np.ndarray]] = {}
    for split, rows in rex_future_sets.items():
        x = feature_matrix(rows, rex_names, rex_medians)
        rex_future_scores[split] = {name: np.asarray(model.predict(x), dtype=np.float64) for name, model in rex_models.items()}

    for row in selected:
        dvol_key = _continual_policy_key(row["dvol_policy"])
        for window_name in ("test2024", "eval2025", "ytd2026"):
            window = WINDOWS[window_name]
            rex_split = "test2024" if window_name == "test2024" else "eval2025_2026"
            events = dvol_events(full_dvol_bank[dvol_key], row["dvol_policy"], full_dates, window)
            events += rex_events(
                rex_future_sets[rex_split],
                rex_future_scores[rex_split],
                row["rex_pair"],
                rex_market["date"],
                window,
            )
            row[window_name] = simulate_schedule(
                full_market,
                events,
                mode=row["mode"],
                window=window,
                leverage=cfg.leverage,
                fee_rate=cfg.fee_rate,
                slippage_rate=cfg.slippage_rate,
            )
        row["passes_alpha_target"] = all(
            row[name]["cagr_pct"] > 0
            and row[name]["ratio"] >= 3
            and row[name]["strict_mdd_pct"] <= 15
            and row[name]["trades"] >= 30
            for name in ("test2024", "eval2025")
        )
        row["passes_live_target"] = bool(
            row["passes_alpha_target"]
            and row["ytd2026"]["cagr_pct"] > 0
            and row["ytd2026"]["ratio"] >= 3
            and row["ytd2026"]["strict_mdd_pct"] <= 15
            and row["ytd2026"]["trades"] >= 10
        )

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "component_manifests_frozen": True,
            "combined_manifest_written_before_future": True,
            "full_market_loaded_after_manifest": True,
            "execution": "one position at a time; optional causal REX-short preemption",
            "strict_mdd": "worst-order favorable-to-adverse OHLC high-water path drawdown",
            "cagr": "full configured calendar window including idle time",
        },
        "tested": len(candidates),
        "manifest_hash": frozen_hash,
        "selected": selected,
        "alpha_qualifiers": [row for row in selected if row["passes_alpha_target"]],
        "live_qualifiers": [row for row in selected if row["passes_live_target"]],
    }
    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> DvolRexComplementConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in DvolRexComplementConfig.__dataclass_fields__.values():
        required = field.default is MISSING and field.default_factory is MISSING
        parser.add_argument("--" + field.name.replace("_", "-"), default=None if required else field.default, required=required)
    ns = parser.parse_args()
    for name in ("top_n", "random_state"):
        setattr(ns, name, int(getattr(ns, name)))
    for name in ("leverage", "fee_rate", "slippage_rate"):
        setattr(ns, name, float(getattr(ns, name)))
    return DvolRexComplementConfig(**vars(ns))


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "manifest_hash": report["manifest_hash"],
                "tested": report["tested"],
                "alpha_qualifiers": len(report["alpha_qualifiers"]),
                "live_qualifiers": len(report["live_qualifiers"]),
                "selected": report["selected"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
