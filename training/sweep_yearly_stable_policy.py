"""Fast sweep for yearly-stable calibrated policies."""

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
    build_calibration_records,
    evaluate_rules,
)
from training.sweep_calibrated_regime_policy import (
    _augment_metrics,
    _copy_with_keys,
    _parse_floats,
    _parse_ints,
    _parse_key_sets,
    _score,
)
from training.text_analyzer_trader_data import load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates
from training.yearly_stable_regime_policy import YearlyStableConfig


def _year(row: dict[str, Any]) -> int:
    return int(pd.to_datetime(row["date"]).year)


def _precompute_group_year_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    years = sorted({_year(r) for r in records})
    for row in records:
        groups.setdefault(str(row["key"]), []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for key, rows in groups.items():
        action_keys = sorted({action_key for row in rows for action_key in row["actions"]})
        actions = []
        for action_key in action_keys:
            yearly = {
                str(year): _aggregate_action(
                    [row for row in rows if _year(row) == year], action_key
                )
                for year in years
            }
            actions.append({"overall": _aggregate_action(rows, action_key), "yearly": yearly})
        out[key] = {"group_samples": len(rows), "actions": actions}
    return out


def _passes_overall(action: dict[str, Any], cfg: CalibratedPolicyConfig) -> bool:
    return (
        int(action.get("samples", 0)) >= cfg.min_train_samples
        and float(action["mean_net_return"]) >= cfg.min_train_mean_net
        and float(action["mean_utility"]) >= cfg.min_train_mean_utility
        and float(action["win_rate"]) >= cfg.min_train_win_rate
        and float(action["mean_mae"]) <= cfg.max_train_mean_mae
    )


def _passes_yearly(action: dict[str, Any], stable: YearlyStableConfig) -> bool:
    return all(
        int(year_action.get("samples", 0)) >= stable.min_year_samples
        and float(year_action["mean_net_return"]) >= stable.min_year_mean_net
        and float(year_action["win_rate"]) >= stable.min_year_win_rate
        and float(year_action["mean_mae"]) <= stable.max_year_mean_mae
        for year_action in action["yearly"].values()
    )


def _fit_from_stats(
    stats: dict[str, dict[str, Any]],
    cfg: CalibratedPolicyConfig,
    stable: YearlyStableConfig,
) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    for key, group in stats.items():
        qualified = []
        for item in group["actions"]:
            overall = item["overall"]
            if not _passes_overall(overall, cfg):
                continue
            if not _passes_yearly(item, stable):
                continue
            qualified.append({**overall, "yearly": item["yearly"]})

        if not qualified:
            continue
        best = max(
            qualified,
            key=lambda candidate: (
                float(candidate["mean_utility"]),
                float(candidate["mean_net_return"]),
                float(candidate["win_rate"]),
            ),
        )
        rules[key] = {
            "key": key,
            "train_samples_in_group": int(group["group_samples"]),
            "action": best,
            "qualified_actions": qualified[:5],
        }
    return rules


def run_sweep(
    *,
    market_csv: str,
    output: str,
    wave_trading_root: str = "",
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    key_sets: str = "regime,trend_alignment,location,risk_state",
    min_train_samples: str = "6,8,12",
    min_train_mean_net: str = "-0.001,0,0.0005",
    min_train_mean_utility: str = "-0.005,-0.002",
    min_train_win_rate: str = "0.48,0.50",
    max_train_mean_mae: str = "0.02",
    min_year_samples: str = "3,4,6",
    min_year_mean_net: str = "-0.002,-0.001,0",
    min_year_win_rate: str = "0.45,0.48,0.50",
    min_eval_trades: int = 30,
    top_k: int = 50,
) -> dict[str, Any]:
    market = load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    base = CalibratedPolicyConfig(hold_candidates=parse_hold_candidates(hold_candidates))
    train_base = build_calibration_records(
        market, base, start_date=train_start, end_date=train_end, stride_bars=stride_bars
    )
    eval_base = build_calibration_records(
        market, base, start_date=eval_start, end_date=eval_end, stride_bars=stride_bars
    )
    years = max(1e-9, (pd.to_datetime(eval_end) - pd.to_datetime(eval_start)).days / 365.25)

    results = []
    total = 0
    for keys in _parse_key_sets(key_sets):
        train_records = _copy_with_keys(train_base, keys)
        eval_records = _copy_with_keys(eval_base, keys)
        stats = _precompute_group_year_stats(train_records)
        grid = itertools.product(
            _parse_ints(min_train_samples),
            _parse_floats(min_train_mean_net),
            _parse_floats(min_train_mean_utility),
            _parse_floats(min_train_win_rate),
            _parse_floats(max_train_mean_mae),
            _parse_ints(min_year_samples),
            _parse_floats(min_year_mean_net),
            _parse_floats(min_year_win_rate),
        )
        for min_samples, mean_net, mean_utility, win_rate, max_mae, year_samples, year_net, year_win in grid:
            total += 1
            cfg = CalibratedPolicyConfig(
                hold_candidates=base.hold_candidates,
                min_train_samples=min_samples,
                min_train_mean_net=mean_net,
                min_train_mean_utility=mean_utility,
                min_train_win_rate=win_rate,
                max_train_mean_mae=max_mae,
                key_fields=keys,
            )
            stable = YearlyStableConfig(
                min_year_samples=year_samples,
                min_year_mean_net=year_net,
                min_year_win_rate=year_win,
                max_year_mean_mae=max_mae,
            )
            rules = _fit_from_stats(stats, cfg, stable)
            metrics = _augment_metrics(
                evaluate_rules(
                    eval_records,
                    rules,
                    non_overlapping=True,
                    include_intratrade_mdd=True,
                ),
                years=years,
            )
            results.append(
                {
                    "score": _score(metrics, min_trades=min_eval_trades),
                    "config": asdict(cfg),
                    "stable_config": asdict(stable),
                    "rules_count": len(rules),
                    "eval_metrics": metrics,
                    "rules_preview": list(rules.values())[:10],
                }
            )

    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    report = {
        "periods": {"train": [train_start, train_end], "eval": [eval_start, eval_end]},
        "records": {"train": len(train_base), "eval": len(eval_base)},
        "sweep": {"total_configs": total, "min_eval_trades": min_eval_trades},
        "top": ranked[:top_k],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast yearly-stable policy sweep")
    for arg in ["market-csv", "output", "train-start", "train-end", "eval-start", "eval-end"]:
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--wave-trading-root", default="")
    parser.add_argument("--stride-bars", type=int, default=12)
    parser.add_argument("--hold-candidates", default="48,96,144,288")
    parser.add_argument("--key-sets", default="regime,trend_alignment,location,risk_state")
    parser.add_argument("--min-train-samples", default="6,8,12")
    parser.add_argument("--min-train-mean-net", default="-0.001,0,0.0005")
    parser.add_argument("--min-train-mean-utility", default="-0.005,-0.002")
    parser.add_argument("--min-train-win-rate", default="0.48,0.50")
    parser.add_argument("--max-train-mean-mae", default="0.02")
    parser.add_argument("--min-year-samples", default="3,4,6")
    parser.add_argument("--min-year-mean-net", default="-0.002,-0.001,0")
    parser.add_argument("--min-year-win-rate", default="0.45,0.48,0.50")
    parser.add_argument("--min-eval-trades", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_sweep(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
