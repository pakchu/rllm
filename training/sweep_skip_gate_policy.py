"""Sweep a past-only skip gate over candidate policy trades.

The skip gate is trained on long-horizon calibration records to decide which
coarse analyzer buckets should not be traded.  It is policy-agnostic: first
produce candidate trades from a rule book, then drop trades whose router bucket
has unstable/negative realized outcomes in train.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from training.calibrated_regime_policy import CalibratedPolicyConfig, _aggregate_action, _metrics_from_trades, _summary_key
from training.sweep_calibrated_regime_policy import _augment_metrics, _parse_floats, _parse_ints, _parse_key_sets, _score
from training.sweep_hybrid_stable_policy import _filter_uncovered_records, _load_top_rules
from training.sweep_yearly_stable_policy import _fit_from_stats, _load_or_build_records, _precompute_group_year_stats, _records_cache_exists
from training.text_analyzer_trader_data import load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates
from training.yearly_stable_regime_policy import YearlyStableConfig


def _year(row: dict[str, Any]) -> int:
    return int(pd.to_datetime(row["date"]).year)


def _group_rows(records: list[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(_summary_key(row["summary"], key_fields), []).append(row)
    return groups


def _precompute_skip_stats(
    train_records: list[dict[str, Any]],
    *,
    router_fields: tuple[str, ...],
    action_side: str,
    action_hold_bars: int,
) -> dict[str, dict[str, Any]]:
    """Precompute router/year aggregates once for threshold-only sweeps."""
    action_key = f"{action_side}_{int(action_hold_bars)}"
    stats: dict[str, dict[str, Any]] = {}
    for key, rows in _group_rows(train_records, router_fields).items():
        by_year: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_year.setdefault(_year(row), []).append(row)
        stats[key] = {
            "router_key": key,
            "overall": _aggregate_action(rows, action_key),
            "yearly": {str(year): _aggregate_action(year_rows, action_key) for year, year_rows in sorted(by_year.items())},
        }
    return stats


def fit_skip_allowlist_from_stats(
    stats: dict[str, dict[str, Any]],
    *,
    min_samples: int,
    min_mean_net: float,
    min_win_rate: float,
    max_mean_mae: float,
    min_good_years: int,
    min_year_samples: int,
    min_year_mean_net: float,
    min_year_win_rate: float,
    max_bad_year_mean_net: float,
) -> dict[str, dict[str, Any]]:
    allow: dict[str, dict[str, Any]] = {}
    for key, item in stats.items():
        overall = item["overall"]
        if int(overall.get("samples", 0)) < int(min_samples):
            continue
        if float(overall.get("mean_net_return", 0.0)) < float(min_mean_net):
            continue
        if float(overall.get("win_rate", 0.0)) < float(min_win_rate):
            continue
        if float(overall.get("mean_mae", 1e9)) > float(max_mean_mae):
            continue
        good_years = 0
        bad_year = False
        yearly = item["yearly"]
        for ystats in yearly.values():
            if int(ystats.get("samples", 0)) < int(min_year_samples):
                continue
            mean_net = float(ystats.get("mean_net_return", 0.0))
            if mean_net <= float(max_bad_year_mean_net):
                bad_year = True
                break
            if mean_net >= float(min_year_mean_net) and float(ystats.get("win_rate", 0.0)) >= float(min_year_win_rate):
                good_years += 1
        if bad_year or good_years < int(min_good_years):
            continue
        allow[key] = {"router_key": key, "overall": overall, "yearly": yearly, "good_years": good_years}
    return allow


def fit_skip_allowlist(
    train_records: list[dict[str, Any]],
    *,
    router_fields: tuple[str, ...],
    action_side: str,
    action_hold_bars: int,
    min_samples: int,
    min_mean_net: float,
    min_win_rate: float,
    max_mean_mae: float,
    min_good_years: int,
    min_year_samples: int,
    min_year_mean_net: float,
    min_year_win_rate: float,
    max_bad_year_mean_net: float,
) -> dict[str, dict[str, Any]]:
    return fit_skip_allowlist_from_stats(
        _precompute_skip_stats(
            train_records,
            router_fields=router_fields,
            action_side=action_side,
            action_hold_bars=action_hold_bars,
        ),
        min_samples=min_samples,
        min_mean_net=min_mean_net,
        min_win_rate=min_win_rate,
        max_mean_mae=max_mean_mae,
        min_good_years=min_good_years,
        min_year_samples=min_year_samples,
        min_year_mean_net=min_year_mean_net,
        min_year_win_rate=min_year_win_rate,
        max_bad_year_mean_net=max_bad_year_mean_net,
    )


def _candidate_rule_for_row(
    row: dict[str, Any],
    *,
    base_rules: dict[str, dict[str, Any]],
    base_key_fields: tuple[str, ...],
    addon_rules: dict[str, dict[str, Any]] | None = None,
    addon_key_fields: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    base_key = _summary_key(row["summary"], base_key_fields)
    rule = base_rules.get(base_key)
    if rule is not None:
        return rule, base_key
    if addon_rules and addon_key_fields:
        addon_key = _summary_key(row["summary"], addon_key_fields)
        rule = addon_rules.get(addon_key)
        if rule is not None:
            return rule, addon_key
    return None, base_key


def evaluate_policy_with_skip(
    records: list[dict[str, Any]],
    *,
    base_rules: dict[str, dict[str, Any]],
    base_key_fields: tuple[str, ...],
    skip_allowlist: dict[str, dict[str, Any]],
    skip_router_fields: tuple[str, ...],
    addon_rules: dict[str, dict[str, Any]] | None = None,
    addon_key_fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    next_available_pos = -1
    for row in records:
        signal_pos = int(row.get("signal_pos", -1))
        if signal_pos <= next_available_pos:
            continue
        router_key = _summary_key(row["summary"], skip_router_fields)
        if router_key not in skip_allowlist:
            continue
        rule, policy_key = _candidate_rule_for_row(
            row,
            base_rules=base_rules,
            base_key_fields=base_key_fields,
            addon_rules=addon_rules,
            addon_key_fields=addon_key_fields,
        )
        if rule is None:
            continue
        action = rule["action"]
        hold_bars = int(action["hold_bars"])
        outcome = row["actions"].get(f"{action['side']}_{hold_bars}")
        if not outcome:
            continue
        trades.append({"date": row["date"], "signal_pos": signal_pos, "router_key": router_key, "policy_key": policy_key, **outcome})
        next_available_pos = signal_pos + hold_bars
    metrics = _metrics_from_trades(trades, records_count=len(records), include_intratrade_mdd=True)
    metrics["non_overlapping"] = True
    return metrics


def _allowlist_signature(allow: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(allow.keys()))


def _fit_addon_from_report(train_records: list[dict[str, Any]], base_rules: dict[str, dict[str, Any]], base_key_fields: tuple[str, ...], addon_report: str, addon_top_index: int) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    if not addon_report:
        return {}, ()
    item = json.loads(Path(addon_report).read_text())["top"][int(addon_top_index)]
    addon_key_fields = tuple(item["addon_config"]["key_fields"])
    cfg = CalibratedPolicyConfig(
        hold_candidates=tuple(int(x) for x in item["addon_config"].get("hold_candidates", (48, 96, 144, 288))),
        min_train_samples=int(item["addon_config"]["min_train_samples"]),
        min_train_mean_net=float(item["addon_config"]["min_train_mean_net"]),
        min_train_mean_utility=float(item["addon_config"]["min_train_mean_utility"]),
        min_train_win_rate=float(item["addon_config"]["min_train_win_rate"]),
        max_train_mean_mae=float(item["addon_config"]["max_train_mean_mae"]),
        key_fields=addon_key_fields,
    )
    stable = YearlyStableConfig(**item["addon_stable_config"])
    uncovered = _filter_uncovered_records(train_records, base_rules, base_key_fields)
    addon_records = []
    for row in uncovered:
        clone = dict(row)
        clone["key"] = _summary_key(row["summary"], addon_key_fields)
        addon_records.append(clone)
    return _fit_from_stats(_precompute_group_year_stats(addon_records), cfg, stable), addon_key_fields


def run_sweep(
    *,
    market_csv: str,
    output: str,
    base_report: str,
    wave_trading_root: str = "",
    addon_report: str = "",
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    records_cache_dir: str = "",
    base_top_index: int = 0,
    include_base_rule_indices: str = "",
    addon_top_index: int = 0,
    skip_router_fields: str = "regime,risk_state,Volume State",
    skip_action_side: str = "LONG",
    skip_action_hold_bars: int = 144,
    min_samples: str = "24,36,48",
    min_mean_net: str = "-0.001,0,0.0005",
    min_win_rate: str = "0.45,0.48,0.50",
    max_mean_mae: str = "0.015,0.02,0.03",
    min_good_years: str = "3,4",
    min_year_samples: str = "3,4,6",
    min_year_mean_net: str = "-0.001,0,0.0005",
    min_year_win_rate: str = "0.43,0.45,0.48",
    max_bad_year_mean_net: str = "-0.02,-0.015,-0.01",
    min_eval_trades: int = 40,
    top_k: int = 50,
) -> dict[str, Any]:
    cfg0 = CalibratedPolicyConfig(hold_candidates=parse_hold_candidates(hold_candidates))
    market = None
    if not _records_cache_exists(records_cache_dir, cfg0, train_start=train_start, train_end=train_end, eval_start=eval_start, eval_end=eval_end, stride_bars=stride_bars):
        market = load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    train_records = _load_or_build_records(market, cfg0, start_date=train_start, end_date=train_end, stride_bars=stride_bars, split="train", records_cache_dir=records_cache_dir)
    eval_records = _load_or_build_records(market, cfg0, start_date=eval_start, end_date=eval_end, stride_bars=stride_bars, split="eval", records_cache_dir=records_cache_dir)
    base_rules, base_key_fields = _load_top_rules(base_report, top_index=base_top_index, include_indices=include_base_rule_indices)
    addon_rules, addon_key_fields = _fit_addon_from_report(train_records, base_rules, base_key_fields, addon_report, addon_top_index)
    router_fields = tuple(x.strip() for x in skip_router_fields.split(",") if x.strip())
    skip_stats = _precompute_skip_stats(train_records, router_fields=router_fields, action_side=skip_action_side, action_hold_bars=skip_action_hold_bars)
    years = max(1e-9, (pd.to_datetime(eval_end) - pd.to_datetime(eval_start)).days / 365.25)
    results: list[dict[str, Any]] = []
    total = 0
    metrics_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    metrics_cache_hits = 0
    metrics_cache_misses = 0
    for ms, mn, wr, mae, gy, ys, yn, yw, bad in itertools.product(
        _parse_ints(min_samples),
        _parse_floats(min_mean_net),
        _parse_floats(min_win_rate),
        _parse_floats(max_mean_mae),
        _parse_ints(min_good_years),
        _parse_ints(min_year_samples),
        _parse_floats(min_year_mean_net),
        _parse_floats(min_year_win_rate),
        _parse_floats(max_bad_year_mean_net),
    ):
        total += 1
        allow = fit_skip_allowlist_from_stats(
            skip_stats,
            min_samples=ms,
            min_mean_net=mn,
            min_win_rate=wr,
            max_mean_mae=mae,
            min_good_years=gy,
            min_year_samples=ys,
            min_year_mean_net=yn,
            min_year_win_rate=yw,
            max_bad_year_mean_net=bad,
        )
        signature = _allowlist_signature(allow)
        if signature in metrics_cache:
            metrics_cache_hits += 1
            metrics = metrics_cache[signature]
        else:
            metrics_cache_misses += 1
            metrics = _augment_metrics(
                evaluate_policy_with_skip(
                    eval_records,
                    base_rules=base_rules,
                    base_key_fields=base_key_fields,
                    addon_rules=addon_rules,
                    addon_key_fields=addon_key_fields,
                    skip_allowlist=allow,
                    skip_router_fields=router_fields,
                ),
                years=years,
            )
            metrics_cache[signature] = metrics
        results.append(
            {
                "score": _score(metrics, min_trades=min_eval_trades),
                "skip_rules_count": len(allow),
                "skip_config": {
                    "router_fields": list(router_fields),
                    "action_side": skip_action_side,
                    "action_hold_bars": int(skip_action_hold_bars),
                    "min_samples": ms,
                    "min_mean_net": mn,
                    "min_win_rate": wr,
                    "max_mean_mae": mae,
                    "min_good_years": gy,
                    "min_year_samples": ys,
                    "min_year_mean_net": yn,
                    "min_year_win_rate": yw,
                    "max_bad_year_mean_net": bad,
                },
                "eval_metrics": metrics,
                "skip_allow_preview": list(allow.values())[:20],
            }
        )
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    report = {
        "periods": {"train": [train_start, train_end], "eval": [eval_start, eval_end]},
        "records": {"train": len(train_records), "eval": len(eval_records)},
        "policy": {"base_rules_count": len(base_rules), "addon_rules_count": len(addon_rules)},
        "sweep": {
            "total_configs": total,
            "min_eval_trades": int(min_eval_trades),
            "metrics_cache_hits": metrics_cache_hits,
            "metrics_cache_misses": metrics_cache_misses,
        },
        "top": ranked[:top_k],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep skip gates over candidate policy trades")
    for arg in ["market-csv", "output", "base-report", "train-start", "train-end", "eval-start", "eval-end"]:
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--wave-trading-root", default="")
    parser.add_argument("--addon-report", default="")
    parser.add_argument("--stride-bars", type=int, default=12)
    parser.add_argument("--hold-candidates", default="48,96,144,288")
    parser.add_argument("--records-cache-dir", default="")
    parser.add_argument("--base-top-index", type=int, default=0)
    parser.add_argument("--include-base-rule-indices", default="")
    parser.add_argument("--addon-top-index", type=int, default=0)
    parser.add_argument("--skip-router-fields", default="regime,risk_state,Volume State")
    parser.add_argument("--skip-action-side", default="LONG")
    parser.add_argument("--skip-action-hold-bars", type=int, default=144)
    parser.add_argument("--min-samples", default="24,36,48")
    parser.add_argument("--min-mean-net", default="-0.001,0,0.0005")
    parser.add_argument("--min-win-rate", default="0.45,0.48,0.50")
    parser.add_argument("--max-mean-mae", default="0.015,0.02,0.03")
    parser.add_argument("--min-good-years", default="3,4")
    parser.add_argument("--min-year-samples", default="3,4,6")
    parser.add_argument("--min-year-mean-net", default="-0.001,0,0.0005")
    parser.add_argument("--min-year-win-rate", default="0.43,0.45,0.48")
    parser.add_argument("--max-bad-year-mean-net", default="-0.02,-0.015,-0.01")
    parser.add_argument("--min-eval-trades", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_sweep(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
