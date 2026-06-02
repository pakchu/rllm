"""Sweep router-first specialist policies.

A router chooses a coarse market regime from past-only analyzer fields.  Each
router bucket owns a separate symbolic specialist rule book.  This avoids forcing
one global rule table to work in bull, bear, and chop regimes simultaneously.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from training.calibrated_regime_policy import CalibratedPolicyConfig, _metrics_from_trades, _summary_key
from training.sweep_calibrated_regime_policy import _augment_metrics, _parse_floats, _parse_ints, _parse_key_sets, _score
from training.sweep_yearly_stable_policy import (
    _fit_from_stats,
    _load_or_build_records,
    _passes_overall,
    _precompute_group_year_stats,
    _records_cache_exists,
    _rule_signature,
)
from training.text_analyzer_trader_data import load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates
from training.yearly_stable_regime_policy import YearlyStableConfig


def _with_key(records: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for row in records:
        clone = dict(row)
        clone["key"] = _summary_key(row["summary"], key_fields)
        out.append(clone)
    return out


def _router_key(row: dict[str, Any], router_fields: tuple[str, ...]) -> str:
    return _summary_key(row["summary"], router_fields)


def _group_by_router(records: list[dict[str, Any]], router_fields: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(_router_key(row, router_fields), []).append(row)
    return groups


def _fit_k_of_years_from_stats(
    stats: dict[str, dict[str, Any]],
    cfg: CalibratedPolicyConfig,
    stable: YearlyStableConfig,
    *,
    min_good_years: int,
    max_bad_year_mean_net: float,
) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    for key, group in stats.items():
        qualified = []
        for item in group["actions"]:
            overall = item["overall"]
            if not _passes_overall(overall, cfg):
                continue
            good_years = 0
            bad_year = False
            for year_action in item["yearly"].values():
                samples = int(year_action.get("samples", 0))
                if samples < stable.min_year_samples:
                    continue
                mean_net = float(year_action.get("mean_net_return", 0.0))
                win_rate = float(year_action.get("win_rate", 0.0))
                mean_mae = float(year_action.get("mean_mae", 0.0))
                if mean_net <= max_bad_year_mean_net:
                    bad_year = True
                    break
                if (
                    mean_net >= stable.min_year_mean_net
                    and win_rate >= stable.min_year_win_rate
                    and mean_mae <= stable.max_year_mean_mae
                ):
                    good_years += 1
            if bad_year or good_years < min_good_years:
                continue
            qualified.append({**overall, "yearly": item["yearly"], "good_years": good_years})
        if not qualified:
            continue
        best = max(
            qualified,
            key=lambda candidate: (
                int(candidate["good_years"]),
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


def _precompute_router_stats(
    train_records: list[dict[str, Any]],
    *,
    router_fields: tuple[str, ...],
    specialist_key_fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    router_groups = _group_by_router(train_records, router_fields)
    return {
        router: {
            "train_records": len(rows),
            "stats": _precompute_group_year_stats(_with_key(rows, specialist_key_fields)),
        }
        for router, rows in router_groups.items()
    }


def fit_router_specialists_from_stats(
    router_stats: dict[str, dict[str, Any]],
    *,
    cfg: CalibratedPolicyConfig,
    stable: YearlyStableConfig,
    min_good_years: int = 0,
    max_bad_year_mean_net: float = -1.0,
) -> dict[str, dict[str, Any]]:
    specialists: dict[str, dict[str, Any]] = {}
    for router, payload in router_stats.items():
        stats = payload["stats"]
        if min_good_years > 0:
            rules = _fit_k_of_years_from_stats(
                stats,
                cfg,
                stable,
                min_good_years=min_good_years,
                max_bad_year_mean_net=max_bad_year_mean_net,
            )
        else:
            rules = _fit_from_stats(stats, cfg, stable)
        if rules:
            specialists[router] = {"router_key": router, "rules": rules, "train_records": int(payload["train_records"])}
    return specialists


def fit_router_specialists(
    train_records: list[dict[str, Any]],
    *,
    router_fields: tuple[str, ...],
    specialist_key_fields: tuple[str, ...],
    cfg: CalibratedPolicyConfig,
    stable: YearlyStableConfig,
    min_good_years: int = 0,
    max_bad_year_mean_net: float = -1.0,
) -> dict[str, dict[str, Any]]:
    router_stats = _precompute_router_stats(
        train_records, router_fields=router_fields, specialist_key_fields=specialist_key_fields
    )
    return fit_router_specialists_from_stats(
        router_stats,
        cfg=cfg,
        stable=stable,
        min_good_years=min_good_years,
        max_bad_year_mean_net=max_bad_year_mean_net,
    )


def _specialist_signature(specialists: dict[str, dict[str, Any]]) -> tuple[tuple[str, tuple[tuple[str, str, int], ...]], ...]:
    return tuple(sorted((router, _rule_signature(book["rules"])) for router, book in specialists.items()))


def evaluate_router_specialists(
    records: list[dict[str, Any]],
    specialists: dict[str, dict[str, Any]],
    *,
    router_fields: tuple[str, ...],
    specialist_key_fields: tuple[str, ...],
    include_intratrade_mdd: bool = True,
) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    next_available_pos = -1
    for row in records:
        signal_pos = int(row.get("signal_pos", -1))
        if signal_pos <= next_available_pos:
            continue
        router = _router_key(row, router_fields)
        book = specialists.get(router)
        if not book:
            continue
        key = _summary_key(row["summary"], specialist_key_fields)
        rule = book["rules"].get(key)
        if not rule:
            continue
        action = rule["action"]
        hold_bars = int(action["hold_bars"])
        outcome = row["actions"].get(f"{action['side']}_{hold_bars}")
        if not outcome:
            continue
        trades.append({"date": row["date"], "signal_pos": signal_pos, "router_key": router, "key": key, **outcome})
        next_available_pos = signal_pos + hold_bars
    metrics = _metrics_from_trades(trades, records_count=len(records), include_intratrade_mdd=include_intratrade_mdd)
    metrics["non_overlapping"] = True
    return metrics


def _precompute_router_action_rows(
    records: list[dict[str, Any]],
    *,
    router_fields: tuple[str, ...],
    specialist_key_fields: tuple[str, ...],
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    by_router: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for row in records:
        router = _router_key(row, router_fields)
        key = _summary_key(row["summary"], specialist_key_fields)
        action_map = by_router.setdefault(router, {}).setdefault(key, {})
        for action_key, outcome in row["actions"].items():
            action_map.setdefault(str(action_key), []).append(
                {"date": row.get("date", ""), "signal_pos": int(row.get("signal_pos", -1)), "router_key": router, "key": key, **outcome}
            )
    return by_router


def evaluate_router_specialists_precomputed(
    *,
    records_count: int,
    action_rows: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    specialists: dict[str, dict[str, Any]],
    include_intratrade_mdd: bool = True,
) -> dict[str, Any]:
    candidate_trades: list[dict[str, Any]] = []
    for router, book in specialists.items():
        router_rows = action_rows.get(router, {})
        for key, rule in book["rules"].items():
            action = rule["action"]
            action_key = f"{action['side']}_{int(action['hold_bars'])}"
            hold_bars = int(action["hold_bars"])
            for row in router_rows.get(str(key), {}).get(action_key, []):
                candidate_trades.append({**row, "_selected_hold_bars": hold_bars})

    trades: list[dict[str, Any]] = []
    next_available_pos = -1
    for trade in sorted(candidate_trades, key=lambda x: int(x.get("signal_pos", -1))):
        signal_pos = int(trade.get("signal_pos", -1))
        if signal_pos <= next_available_pos:
            continue
        trades.append(trade)
        next_available_pos = signal_pos + int(trade.get("_selected_hold_bars", trade.get("hold_bars", 0)))

    metrics = _metrics_from_trades(trades, records_count=int(records_count), include_intratrade_mdd=include_intratrade_mdd)
    metrics["non_overlapping"] = True
    return metrics


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
    records_cache_dir: str = "",
    router_fields: str = "regime,risk_state",
    specialist_key_sets: str = "trend_alignment,location,Volume State",
    min_train_samples: str = "8,12,16",
    min_train_mean_net: str = "0,0.0005,0.001",
    min_train_mean_utility: str = "-0.003,-0.001,0",
    min_train_win_rate: str = "0.48,0.50,0.52",
    max_train_mean_mae: str = "0.01,0.015,0.02",
    min_year_samples: str = "3,4,6",
    min_year_mean_net: str = "0,0.0005",
    min_year_win_rate: str = "0.48,0.50",
    min_good_years: str = "0,3,4",
    max_bad_year_mean_net: str = "-0.015,-0.01,-0.005",
    min_eval_trades: int = 60,
    top_k: int = 50,
) -> dict[str, Any]:
    base = CalibratedPolicyConfig(hold_candidates=parse_hold_candidates(hold_candidates))
    market = None
    if not _records_cache_exists(
        records_cache_dir,
        base,
        train_start=train_start,
        train_end=train_end,
        eval_start=eval_start,
        eval_end=eval_end,
        stride_bars=stride_bars,
    ):
        market = load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    train_records = _load_or_build_records(
        market,
        base,
        start_date=train_start,
        end_date=train_end,
        stride_bars=stride_bars,
        split="train",
        records_cache_dir=records_cache_dir,
    )
    eval_records = _load_or_build_records(
        market,
        base,
        start_date=eval_start,
        end_date=eval_end,
        stride_bars=stride_bars,
        split="eval",
        records_cache_dir=records_cache_dir,
    )
    router = tuple(x.strip() for x in router_fields.split(",") if x.strip())
    years = max(1e-9, (pd.to_datetime(eval_end) - pd.to_datetime(eval_start)).days / 365.25)
    results: list[dict[str, Any]] = []
    total = 0
    hits = 0
    misses = 0
    for specialist_fields in _parse_key_sets(specialist_key_sets):
        router_stats = _precompute_router_stats(
            train_records, router_fields=router, specialist_key_fields=specialist_fields
        )
        eval_action_rows = _precompute_router_action_rows(
            eval_records, router_fields=router, specialist_key_fields=specialist_fields
        )
        cache: dict[tuple[tuple[str, tuple[tuple[str, str, int], ...]], ...], dict[str, Any]] = {}
        grid = itertools.product(
            _parse_ints(min_train_samples),
            _parse_floats(min_train_mean_net),
            _parse_floats(min_train_mean_utility),
            _parse_floats(min_train_win_rate),
            _parse_floats(max_train_mean_mae),
            _parse_ints(min_year_samples),
            _parse_floats(min_year_mean_net),
            _parse_floats(min_year_win_rate),
            _parse_ints(min_good_years),
            _parse_floats(max_bad_year_mean_net),
        )
        for min_samples, mean_net, mean_utility, win_rate, max_mae, year_samples, year_net, year_win, good_years, bad_net in grid:
            total += 1
            cfg = CalibratedPolicyConfig(
                hold_candidates=base.hold_candidates,
                min_train_samples=min_samples,
                min_train_mean_net=mean_net,
                min_train_mean_utility=mean_utility,
                min_train_win_rate=win_rate,
                max_train_mean_mae=max_mae,
                key_fields=specialist_fields,
            )
            stable = YearlyStableConfig(
                min_year_samples=year_samples,
                min_year_mean_net=year_net,
                min_year_win_rate=year_win,
                max_year_mean_mae=max_mae,
            )
            specialists = fit_router_specialists_from_stats(
                router_stats,
                cfg=cfg,
                stable=stable,
                min_good_years=good_years,
                max_bad_year_mean_net=bad_net,
            )
            signature = _specialist_signature(specialists)
            if signature in cache:
                hits += 1
                metrics = cache[signature]
            else:
                misses += 1
                metrics = _augment_metrics(
                    evaluate_router_specialists_precomputed(
                        records_count=len(eval_records),
                        action_rows=eval_action_rows,
                        specialists=specialists,
                    ),
                    years=years,
                )
                cache[signature] = metrics
            results.append(
                {
                    "score": _score(metrics, min_trades=min_eval_trades),
                    "router_fields": list(router),
                    "specialist_key_fields": list(specialist_fields),
                    "config": asdict(cfg),
                    "stable_config": asdict(stable),
                    "min_good_years": int(good_years),
                    "max_bad_year_mean_net": float(bad_net),
                    "router_count": len(specialists),
                    "rules_count": sum(len(book["rules"]) for book in specialists.values()),
                    "eval_metrics": metrics,
                    "routers_preview": list(specialists.values())[:10],
                }
            )
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    report = {
        "periods": {"train": [train_start, train_end], "eval": [eval_start, eval_end]},
        "records": {"train": len(train_records), "eval": len(eval_records)},
        "sweep": {"total_configs": total, "hits": hits, "misses": misses, "min_eval_trades": int(min_eval_trades)},
        "top": ranked[:top_k],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep router-first specialist policies")
    for arg in ["market-csv", "output", "train-start", "train-end", "eval-start", "eval-end"]:
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--wave-trading-root", default="")
    parser.add_argument("--stride-bars", type=int, default=12)
    parser.add_argument("--hold-candidates", default="48,96,144,288")
    parser.add_argument("--records-cache-dir", default="")
    parser.add_argument("--router-fields", default="regime,risk_state")
    parser.add_argument("--specialist-key-sets", default="trend_alignment,location,Volume State")
    parser.add_argument("--min-train-samples", default="8,12,16")
    parser.add_argument("--min-train-mean-net", default="0,0.0005,0.001")
    parser.add_argument("--min-train-mean-utility", default="-0.003,-0.001,0")
    parser.add_argument("--min-train-win-rate", default="0.48,0.50,0.52")
    parser.add_argument("--max-train-mean-mae", default="0.01,0.015,0.02")
    parser.add_argument("--min-year-samples", default="3,4,6")
    parser.add_argument("--min-year-mean-net", default="0,0.0005")
    parser.add_argument("--min-year-win-rate", default="0.48,0.50")
    parser.add_argument("--min-good-years", default="0,3,4")
    parser.add_argument("--max-bad-year-mean-net", default="-0.015,-0.01,-0.005")
    parser.add_argument("--min-eval-trades", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_sweep(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
