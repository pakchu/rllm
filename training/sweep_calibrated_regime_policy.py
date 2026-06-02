"""Sweep train-calibrated symbolic policies without rebuilding path records.

This is intentionally leakage-safe for a train->validation search: records contain
past-only summaries plus realized action outcomes, rules are fit on train records,
and validation metrics use fixed train rules.  A separate untouched eval period is
still required before accepting a selected configuration.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from training.calibrated_regime_policy import (
    CalibratedPolicyConfig,
    _aggregate_action,
    _metrics_from_trades,
    _summary_key,
    build_calibration_records,
)
from training.text_analyzer_trader_data import load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates


def _parse_key_sets(raw: str) -> list[tuple[str, ...]]:
    sets: list[tuple[str, ...]] = []
    for chunk in str(raw).split(";"):
        fields = tuple(x.strip() for x in chunk.split(",") if x.strip())
        if fields:
            sets.append(fields)
    if not sets:
        raise ValueError("at least one key set is required")
    return sets


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _copy_with_keys(records: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    keyed: list[dict[str, Any]] = []
    for row in records:
        clone = dict(row)
        clone["key"] = _summary_key(row["summary"], key_fields)
        keyed.append(clone)
    return keyed


def _precompute_train_action_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(str(row["key"]), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for key, rows in groups.items():
        action_keys = sorted({a for r in rows for a in r["actions"]})
        out[key] = {
            "group_samples": len(rows),
            "actions": [_aggregate_action(rows, action_key) for action_key in action_keys],
        }
    return out


def _fit_rules_from_stats(group_stats: dict[str, dict[str, Any]], cfg: CalibratedPolicyConfig) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    for key, stats in group_stats.items():
        candidates = [c for c in stats["actions"] if int(c.get("samples", 0)) >= int(cfg.min_train_samples)]
        qualified = [
            c
            for c in candidates
            if float(c["mean_net_return"]) >= float(cfg.min_train_mean_net)
            and float(c["mean_utility"]) >= float(cfg.min_train_mean_utility)
            and float(c["win_rate"]) >= float(cfg.min_train_win_rate)
            and float(c["mean_mae"]) <= float(cfg.max_train_mean_mae)
        ]
        if not qualified:
            continue
        best = max(qualified, key=lambda c: (float(c["mean_utility"]), float(c["mean_net_return"]), float(c["win_rate"])))
        rules[key] = {
            "key": key,
            "train_samples_in_group": int(stats["group_samples"]),
            "action": best,
            "qualified_actions": qualified[:5],
        }
    return rules


def _precompute_action_rows(records: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_key: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in records:
        action_map = by_key.setdefault(str(row["key"]), {})
        for action_key, outcome in row["actions"].items():
            action_map.setdefault(str(action_key), []).append({"date": row.get("date", ""), "signal_pos": int(row.get("signal_pos", -1)), "key": row["key"], **outcome})
    return by_key


def _evaluate_rules_precomputed(
    *,
    records_count: int,
    action_rows: dict[str, dict[str, list[dict[str, Any]]]],
    rules: dict[str, dict[str, Any]],
    non_overlapping: bool = False,
    include_intratrade_mdd: bool = False,
) -> dict[str, Any]:
    candidate_trades: list[dict[str, Any]] = []
    for key, rule in rules.items():
        action = rule["action"]
        action_key = f"{action['side']}_{int(action['hold_bars'])}"
        hold_bars = int(action["hold_bars"])
        for row in action_rows.get(str(key), {}).get(action_key, []):
            candidate_trades.append({**row, "_selected_hold_bars": hold_bars})
    if non_overlapping:
        trades: list[dict[str, Any]] = []
        next_available_pos = -1
        for trade in sorted(candidate_trades, key=lambda x: int(x.get("signal_pos", -1))):
            signal_pos = int(trade.get("signal_pos", -1))
            if signal_pos <= next_available_pos:
                continue
            trades.append(trade)
            next_available_pos = signal_pos + int(trade.get("_selected_hold_bars", trade.get("hold_bars", 0)))
    else:
        trades = candidate_trades
    metrics = _metrics_from_trades(trades, records_count=int(records_count), include_intratrade_mdd=include_intratrade_mdd)
    metrics["non_overlapping"] = bool(non_overlapping)
    return metrics


def _period_years(records: list[dict[str, Any]]) -> float:
    if len(records) < 2:
        return 0.0
    dates = pd.to_datetime([r["date"] for r in records])
    days = max(1.0, float((dates.max() - dates.min()).days))
    return days / 365.25


def _augment_metrics(metrics: dict[str, Any], *, years: float) -> dict[str, Any]:
    out = dict(metrics)
    compounded = float(out.get("compounded_return", 0.0) or 0.0)
    mdd = float(out.get("strict_mdd_proxy", 0.0) or 0.0)
    if years > 0 and compounded > -0.999:
        cagr = (1.0 + compounded) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0
    out["years"] = years
    out["cagr_proxy"] = cagr
    out["cagr_to_mdd_proxy"] = cagr / mdd if mdd > 0 else (float("inf") if cagr > 0 else 0.0)
    return out


def _score(metrics: dict[str, Any], *, min_trades: int) -> tuple[float, float, float, int]:
    trades = int(metrics.get("trades", 0) or 0)
    if trades < min_trades:
        return (-1e9, float(metrics.get("cagr_to_mdd_proxy", 0.0) or 0.0), float(metrics.get("cagr_proxy", 0.0) or 0.0), trades)
    ratio = float(metrics.get("cagr_to_mdd_proxy", 0.0) or 0.0)
    cagr = float(metrics.get("cagr_proxy", 0.0) or 0.0)
    win = float(metrics.get("win_rate", 0.0) or 0.0)
    return (ratio, cagr, win, trades)


def run_sweep(
    *,
    market_csv: str,
    output: str,
    wave_trading_root: str = "",
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    key_sets: str = "regime,trend_alignment,location,oscillator,risk_state",
    min_train_samples: str = "12,24,48",
    min_train_mean_net: str = "0.0005,0.001,0.002,0.003",
    min_train_mean_utility: str = "-0.001,0,0.001",
    min_train_win_rate: str = "0.50,0.52,0.55",
    max_train_mean_mae: str = "0.005,0.0075,0.01",
    min_validation_trades: int = 30,
    top_k: int = 50,
) -> dict[str, Any]:
    base_cfg = CalibratedPolicyConfig(hold_candidates=parse_hold_candidates(hold_candidates))
    market = load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    train_records_base = build_calibration_records(market, base_cfg, start_date=train_start, end_date=train_end, stride_bars=stride_bars)
    validation_records_base = build_calibration_records(
        market, base_cfg, start_date=validation_start, end_date=validation_end, stride_bars=stride_bars
    )
    validation_years = _period_years(validation_records_base)
    train_years = _period_years(train_records_base)

    results: list[dict[str, Any]] = []
    total = 0
    for key_fields in _parse_key_sets(key_sets):
        train_records = _copy_with_keys(train_records_base, key_fields)
        validation_records = _copy_with_keys(validation_records_base, key_fields)
        train_group_stats = _precompute_train_action_stats(train_records)
        train_action_rows = _precompute_action_rows(train_records)
        validation_action_rows = _precompute_action_rows(validation_records)
        for samples, mean_net, mean_utility, win_rate, max_mae in itertools.product(
            _parse_ints(min_train_samples),
            _parse_floats(min_train_mean_net),
            _parse_floats(min_train_mean_utility),
            _parse_floats(min_train_win_rate),
            _parse_floats(max_train_mean_mae),
        ):
            total += 1
            cfg = CalibratedPolicyConfig(
                hold_candidates=base_cfg.hold_candidates,
                min_train_samples=samples,
                min_train_mean_net=mean_net,
                min_train_mean_utility=mean_utility,
                min_train_win_rate=win_rate,
                max_train_mean_mae=max_mae,
                key_fields=key_fields,
            )
            rules = _fit_rules_from_stats(train_group_stats, cfg)
            train_proxy_metrics = _augment_metrics(
                _evaluate_rules_precomputed(records_count=len(train_records), action_rows=train_action_rows, rules=rules),
                years=train_years,
            )
            validation_proxy_metrics = _augment_metrics(
                _evaluate_rules_precomputed(records_count=len(validation_records), action_rows=validation_action_rows, rules=rules),
                years=validation_years,
            )
            train_metrics = _augment_metrics(
                _evaluate_rules_precomputed(
                    records_count=len(train_records),
                    action_rows=train_action_rows,
                    rules=rules,
                    non_overlapping=True,
                    include_intratrade_mdd=True,
                ),
                years=train_years,
            )
            validation_metrics = _augment_metrics(
                _evaluate_rules_precomputed(
                    records_count=len(validation_records),
                    action_rows=validation_action_rows,
                    rules=rules,
                    non_overlapping=True,
                    include_intratrade_mdd=True,
                ),
                years=validation_years,
            )
            results.append(
                {
                    "score": _score(validation_metrics, min_trades=int(min_validation_trades)),
                    "config": asdict(cfg),
                    "rules_count": len(rules),
                    "train_metrics": train_metrics,
                    "validation_metrics": validation_metrics,
                    "train_proxy_metrics": train_proxy_metrics,
                    "validation_proxy_metrics": validation_proxy_metrics,
                    "rules_preview": list(rules.values())[:10],
                }
            )

    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    report = {
        "market_csv": str(Path(market_csv).resolve()),
        "periods": {"train": [train_start, train_end], "validation": [validation_start, validation_end]},
        "records": {"train": len(train_records_base), "validation": len(validation_records_base)},
        "sweep": {"total_configs": total, "min_validation_trades": int(min_validation_trades)},
        "top": ranked[: int(top_k)],
        "leakage_guard": {
            "rules_fit_on_train_period_only": True,
            "validation_uses_fixed_train_rules": True,
            "selected_config_still_requires_untouched_eval": True,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep leakage-safe train-calibrated symbolic policy settings")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--validation-start", required=True)
    p.add_argument("--validation-end", required=True)
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--hold-candidates", default="48,96,144,288")
    p.add_argument("--key-sets", default="regime,trend_alignment,location,oscillator,risk_state")
    p.add_argument("--min-train-samples", default="12,24,48")
    p.add_argument("--min-train-mean-net", default="0.0005,0.001,0.002,0.003")
    p.add_argument("--min-train-mean-utility", default="-0.001,0,0.001")
    p.add_argument("--min-train-win-rate", default="0.50,0.52,0.55")
    p.add_argument("--max-train-mean-mae", default="0.005,0.0075,0.01")
    p.add_argument("--min-validation-trades", type=int, default=30)
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_sweep(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
