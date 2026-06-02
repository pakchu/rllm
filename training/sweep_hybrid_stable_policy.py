"""Sweep add-on yearly-stable rules behind a fixed base policy.

The hybrid policy is intentionally asymmetric:
1. Apply a high-confidence base policy first.
2. Only when the base policy has no rule for the analyzer summary, try add-on
   rules fit with richer key fields such as macro, volume, or kimchi-premium
   state.

This keeps the proven sparse policy intact while searching for extra trades in
previously uncovered regimes.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from training.calibrated_regime_policy import CalibratedPolicyConfig, _summary_key
from training.sweep_calibrated_regime_policy import (
    _augment_metrics,
    _copy_with_keys,
    _evaluate_rules_precomputed,
    _parse_floats,
    _parse_ints,
    _parse_key_sets,
    _precompute_action_rows,
    _score,
)
from training.sweep_yearly_stable_policy import (
    _fit_from_stats,
    _load_or_build_records,
    _precompute_group_year_stats,
    _records_cache_exists,
    _rule_signature,
)
from training.text_analyzer_trader_data import load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates
from training.yearly_stable_regime_policy import YearlyStableConfig


def _parse_indices(raw: str) -> set[int]:
    return {int(x.strip()) for x in str(raw).split(",") if x.strip()}


def _load_top_rules(
    report_path: str,
    *,
    top_index: int = 0,
    include_indices: str = "",
    exclude_indices: str = "",
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    report = json.loads(Path(report_path).read_text())
    item = report["top"][int(top_index)]
    key_fields = tuple(item["config"]["key_fields"])
    include = _parse_indices(include_indices)
    exclude = _parse_indices(exclude_indices)
    rules: dict[str, dict[str, Any]] = {}
    for idx, rule in enumerate(item["rules_preview"], start=1):
        if include and idx not in include:
            continue
        if idx in exclude:
            continue
        clone = dict(rule)
        clone["key_fields"] = key_fields
        rules[str(clone["key"])] = clone
    return rules, key_fields


def _filter_uncovered_records(
    records: list[dict[str, Any]],
    base_rules: dict[str, dict[str, Any]],
    base_key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not base_rules:
        return list(records)
    base_keys = set(base_rules)
    return [row for row in records if _summary_key(row["summary"], base_key_fields) not in base_keys]


def _merge_action_rows(
    left: dict[str, dict[str, list[dict[str, Any]]]],
    right: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    merged = {key: {action: list(rows) for action, rows in actions.items()} for key, actions in left.items()}
    for key, actions in right.items():
        dst = merged.setdefault(key, {})
        for action, rows in actions.items():
            dst.setdefault(action, []).extend(rows)
    return merged


def _evaluate_hybrid_precomputed(
    *,
    records_count: int,
    base_action_rows: dict[str, dict[str, list[dict[str, Any]]]],
    addon_action_rows: dict[str, dict[str, list[dict[str, Any]]]],
    base_rules: dict[str, dict[str, Any]],
    addon_rules: dict[str, dict[str, Any]],
    years: float,
) -> dict[str, Any]:
    action_rows = _merge_action_rows(base_action_rows, addon_action_rows)
    rules = {**base_rules, **addon_rules}
    return _augment_metrics(
        _evaluate_rules_precomputed(
            records_count=records_count,
            action_rows=action_rows,
            rules=rules,
            non_overlapping=True,
            include_intratrade_mdd=True,
        ),
        years=years,
    )


def run_hybrid_sweep(
    *,
    market_csv: str,
    output: str,
    base_report: str,
    wave_trading_root: str = "",
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    records_cache_dir: str = "",
    base_top_index: int = 0,
    include_base_rule_indices: str = "",
    exclude_base_rule_indices: str = "",
    addon_key_sets: str = "regime,trend_alignment,location,risk_state,Volume State",
    min_train_samples: str = "4,6,8",
    min_train_mean_net: str = "-0.001,0",
    min_train_mean_utility: str = "-0.003,-0.002",
    min_train_win_rate: str = "0.43,0.45",
    max_train_mean_mae: str = "0.018,0.02",
    min_year_samples: str = "4,5,6",
    min_year_mean_net: str = "-0.003,-0.002,-0.001,0",
    min_year_win_rate: str = "0.43,0.45,0.48",
    min_eval_trades: int = 60,
    top_k: int = 50,
) -> dict[str, Any]:
    base_cfg = CalibratedPolicyConfig(hold_candidates=parse_hold_candidates(hold_candidates))
    market = None
    if not _records_cache_exists(
        records_cache_dir,
        base_cfg,
        train_start=train_start,
        train_end=train_end,
        eval_start=eval_start,
        eval_end=eval_end,
        stride_bars=stride_bars,
    ):
        market = load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    train_base = _load_or_build_records(
        market,
        base_cfg,
        start_date=train_start,
        end_date=train_end,
        stride_bars=stride_bars,
        split="train",
        records_cache_dir=records_cache_dir,
    )
    eval_base = _load_or_build_records(
        market,
        base_cfg,
        start_date=eval_start,
        end_date=eval_end,
        stride_bars=stride_bars,
        split="eval",
        records_cache_dir=records_cache_dir,
    )
    years = max(1e-9, (pd.to_datetime(eval_end) - pd.to_datetime(eval_start)).days / 365.25)
    base_rules, base_key_fields = _load_top_rules(
        base_report,
        top_index=base_top_index,
        include_indices=include_base_rule_indices,
        exclude_indices=exclude_base_rule_indices,
    )
    train_uncovered = _filter_uncovered_records(train_base, base_rules, base_key_fields)
    eval_uncovered = _filter_uncovered_records(eval_base, base_rules, base_key_fields)
    base_eval_records = _copy_with_keys(eval_base, base_key_fields)
    base_action_rows = _precompute_action_rows(base_eval_records)
    base_metrics = _evaluate_hybrid_precomputed(
        records_count=len(eval_base),
        base_action_rows=base_action_rows,
        addon_action_rows={},
        base_rules=base_rules,
        addon_rules={},
        years=years,
    )

    results: list[dict[str, Any]] = []
    total = 0
    metrics_cache_hits = 0
    metrics_cache_misses = 0
    for addon_keys in _parse_key_sets(addon_key_sets):
        train_records = _copy_with_keys(train_uncovered, addon_keys)
        eval_records = _copy_with_keys(eval_uncovered, addon_keys)
        stats = _precompute_group_year_stats(train_records)
        addon_action_rows = _precompute_action_rows(eval_records)
        metrics_cache: dict[tuple[tuple[str, str, int], ...], dict[str, Any]] = {}
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
                hold_candidates=base_cfg.hold_candidates,
                min_train_samples=min_samples,
                min_train_mean_net=mean_net,
                min_train_mean_utility=mean_utility,
                min_train_win_rate=win_rate,
                max_train_mean_mae=max_mae,
                key_fields=addon_keys,
            )
            stable = YearlyStableConfig(
                min_year_samples=year_samples,
                min_year_mean_net=year_net,
                min_year_win_rate=year_win,
                max_year_mean_mae=max_mae,
            )
            addon_rules = _fit_from_stats(stats, cfg, stable)
            for rule in addon_rules.values():
                rule["key_fields"] = addon_keys
            signature = (_rule_signature(base_rules), _rule_signature(addon_rules))
            if signature in metrics_cache:
                metrics_cache_hits += 1
                metrics = metrics_cache[signature]
            else:
                metrics_cache_misses += 1
                metrics = _evaluate_hybrid_precomputed(
                    records_count=len(eval_base),
                    base_action_rows=base_action_rows,
                    addon_action_rows=addon_action_rows,
                    base_rules=base_rules,
                    addon_rules=addon_rules,
                    years=years,
                )
                metrics_cache[signature] = metrics
            results.append(
                {
                    "score": _score(metrics, min_trades=int(min_eval_trades)),
                    "base_rules_count": len(base_rules),
                    "addon_rules_count": len(addon_rules),
                    "addon_config": asdict(cfg),
                    "addon_stable_config": asdict(stable),
                    "eval_metrics": metrics,
                    "addon_rules_preview": list(addon_rules.values())[:10],
                }
            )

    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    report = {
        "periods": {"train": [train_start, train_end], "eval": [eval_start, eval_end]},
        "records": {
            "train": len(train_base),
            "eval": len(eval_base),
            "train_uncovered": len(train_uncovered),
            "eval_uncovered": len(eval_uncovered),
        },
        "base": {
            "report": base_report,
            "top_index": int(base_top_index),
            "key_fields": list(base_key_fields),
            "rules_count": len(base_rules),
            "eval_metrics": base_metrics,
        },
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
    parser = argparse.ArgumentParser(description="Sweep add-on yearly-stable hybrid policies")
    for arg in ["market-csv", "output", "base-report", "train-start", "train-end", "eval-start", "eval-end"]:
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--wave-trading-root", default="")
    parser.add_argument("--stride-bars", type=int, default=12)
    parser.add_argument("--hold-candidates", default="48,96,144,288")
    parser.add_argument("--records-cache-dir", default="")
    parser.add_argument("--base-top-index", type=int, default=0)
    parser.add_argument("--include-base-rule-indices", default="")
    parser.add_argument("--exclude-base-rule-indices", default="")
    parser.add_argument("--addon-key-sets", default="regime,trend_alignment,location,risk_state,Volume State")
    parser.add_argument("--min-train-samples", default="4,6,8")
    parser.add_argument("--min-train-mean-net", default="-0.001,0")
    parser.add_argument("--min-train-mean-utility", default="-0.003,-0.002")
    parser.add_argument("--min-train-win-rate", default="0.43,0.45")
    parser.add_argument("--max-train-mean-mae", default="0.018,0.02")
    parser.add_argument("--min-year-samples", default="4,5,6")
    parser.add_argument("--min-year-mean-net", default="-0.003,-0.002,-0.001,0")
    parser.add_argument("--min-year-win-rate", default="0.43,0.45,0.48")
    parser.add_argument("--min-eval-trades", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_hybrid_sweep(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
