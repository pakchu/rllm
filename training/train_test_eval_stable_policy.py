"""Train/test/eval validation for yearly-stable calibrated policies.

Train fits rules, test selects hyperparameters, eval is an untouched final report.
This keeps model/parameter choice out of the eval window while reusing the same
strict non-overlapping, intratrade-MDD-aware policy evaluation used by sweeps.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from training.calibrated_regime_policy import CalibratedPolicyConfig
from training.sweep_calibrated_regime_policy import _augment_metrics, _copy_with_keys, _parse_floats, _parse_ints, _parse_key_sets, _score, _evaluate_rules_precomputed, _precompute_action_rows
from training.sweep_yearly_stable_policy import _fit_from_stats, _load_or_build_records, _precompute_group_year_stats, _records_cache_path, _rule_signature
from training.text_analyzer_trader_data import load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates
from training.yearly_stable_regime_policy import YearlyStableConfig


def _period_years(start: str, end: str) -> float:
    return max(1e-9, (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25)


def _cache_exists(cache_dir: str, *, split: str, start: str, end: str, stride_bars: int, cfg: CalibratedPolicyConfig) -> bool:
    return bool(cache_dir) and _records_cache_path(cache_dir, split=split, start_date=start, end_date=end, stride_bars=stride_bars, cfg=cfg).exists()


def _evaluate_with_cache(
    *,
    records_count: int,
    action_rows: list[dict[str, Any]],
    rules: dict[str, dict[str, Any]],
    years: float,
    cache: dict[tuple[tuple[str, str, int], ...], dict[str, Any]],
) -> dict[str, Any]:
    signature = _rule_signature(rules)
    if signature not in cache:
        cache[signature] = _augment_metrics(
            _evaluate_rules_precomputed(
                records_count=records_count,
                action_rows=action_rows,
                rules=rules,
                non_overlapping=True,
                include_intratrade_mdd=True,
            ),
            years=years,
        )
    return cache[signature]


def run_validate(
    *,
    market_csv: str,
    output: str,
    wave_trading_root: str = "",
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    eval_start: str,
    eval_end: str,
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    key_sets: str = "regime,trend_alignment,location,risk_state",
    min_train_samples: str = "12,24,36",
    min_train_mean_net: str = "-0.001,0,0.0005",
    min_train_mean_utility: str = "-0.005,-0.002",
    min_train_win_rate: str = "0.48,0.50",
    max_train_mean_mae: str = "0.015,0.02",
    min_year_samples: str = "1,2,3",
    min_year_mean_net: str = "-0.002,0",
    min_year_win_rate: str = "0.40,0.45,0.50",
    min_test_trades: int = 30,
    min_eval_trades: int = 30,
    top_k: int = 20,
    records_cache_dir: str = "",
) -> dict[str, Any]:
    base = CalibratedPolicyConfig(hold_candidates=parse_hold_candidates(hold_candidates))
    have_cache = all(
        _cache_exists(records_cache_dir, split=split, start=start, end=end, stride_bars=stride_bars, cfg=base)
        for split, start, end in (
            ("train", train_start, train_end),
            ("test", test_start, test_end),
            ("eval", eval_start, eval_end),
        )
    )
    market = None if have_cache else load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    train_base = _load_or_build_records(market, base, start_date=train_start, end_date=train_end, stride_bars=stride_bars, split="train", records_cache_dir=records_cache_dir)
    test_base = _load_or_build_records(market, base, start_date=test_start, end_date=test_end, stride_bars=stride_bars, split="test", records_cache_dir=records_cache_dir)
    eval_base = _load_or_build_records(market, base, start_date=eval_start, end_date=eval_end, stride_bars=stride_bars, split="eval", records_cache_dir=records_cache_dir)

    test_years = _period_years(test_start, test_end)
    eval_years = _period_years(eval_start, eval_end)
    test_candidates: list[dict[str, Any]] = []
    total = 0
    test_cache_hits = test_cache_misses = 0

    for keys in _parse_key_sets(key_sets):
        train_records = _copy_with_keys(train_base, keys)
        test_records = _copy_with_keys(test_base, keys)
        test_action_rows = _precompute_action_rows(test_records)
        stats = _precompute_group_year_stats(train_records)
        test_metrics_cache: dict[tuple[tuple[str, str, int], ...], dict[str, Any]] = {}
        for min_samples, mean_net, mean_utility, win_rate, max_mae, year_samples, year_net, year_win in itertools.product(
            _parse_ints(min_train_samples),
            _parse_floats(min_train_mean_net),
            _parse_floats(min_train_mean_utility),
            _parse_floats(min_train_win_rate),
            _parse_floats(max_train_mean_mae),
            _parse_ints(min_year_samples),
            _parse_floats(min_year_mean_net),
            _parse_floats(min_year_win_rate),
        ):
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
            sig = _rule_signature(rules)
            before = len(test_metrics_cache)
            test_metrics = _evaluate_with_cache(records_count=len(test_records), action_rows=test_action_rows, rules=rules, years=test_years, cache=test_metrics_cache)
            if len(test_metrics_cache) == before:
                test_cache_hits += 1
            else:
                test_cache_misses += 1
            test_candidates.append(
                {
                    "test_score": _score(test_metrics, min_trades=min_test_trades),
                    "signature": list(sig),
                    "config": asdict(cfg),
                    "stable_config": asdict(stable),
                    "rules_count": len(rules),
                    "test_metrics": test_metrics,
                    "rules_preview": list(rules.values())[:10],
                }
            )

    ranked = sorted(test_candidates, key=lambda r: r["test_score"], reverse=True)[:top_k]
    eval_results: list[dict[str, Any]] = []
    eval_metrics_cache: dict[tuple[tuple[str, str, int], ...], dict[str, Any]] = {}
    for item in ranked:
        keys = tuple(item["config"]["key_fields"])
        cfg = CalibratedPolicyConfig(**{k: v for k, v in item["config"].items() if k != "hold_candidates"}, hold_candidates=tuple(item["config"]["hold_candidates"]))
        stable = YearlyStableConfig(**item["stable_config"])
        train_records = _copy_with_keys(train_base, keys)
        eval_records = _copy_with_keys(eval_base, keys)
        rules = _fit_from_stats(_precompute_group_year_stats(train_records), cfg, stable)
        eval_metrics = _evaluate_with_cache(records_count=len(eval_records), action_rows=_precompute_action_rows(eval_records), rules=rules, years=eval_years, cache=eval_metrics_cache)
        eval_results.append({**item, "eval_score": _score(eval_metrics, min_trades=min_eval_trades), "eval_metrics": eval_metrics})

    eval_ranked = sorted(eval_results, key=lambda r: (r["eval_score"], r["test_score"]), reverse=True)
    report = {
        "periods": {"train": [train_start, train_end], "test": [test_start, test_end], "eval": [eval_start, eval_end]},
        "records": {"train": len(train_base), "test": len(test_base), "eval": len(eval_base)},
        "sweep": {"total_configs": total, "min_test_trades": int(min_test_trades), "min_eval_trades": int(min_eval_trades), "test_metrics_cache_hits": test_cache_hits, "test_metrics_cache_misses": test_cache_misses},
        "leakage_guard": {"selection_window": "test", "eval_used_for_selection": False},
        "top_by_test_then_eval": eval_results,
        "top_by_eval_report_only": eval_ranked,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe train/test/eval yearly-stable policy validation")
    for arg in ["market-csv", "output", "train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"]:
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--wave-trading-root", default="")
    parser.add_argument("--stride-bars", type=int, default=12)
    parser.add_argument("--hold-candidates", default="48,96,144,288")
    parser.add_argument("--key-sets", default="regime,trend_alignment,location,risk_state")
    parser.add_argument("--min-train-samples", default="12,24,36")
    parser.add_argument("--min-train-mean-net", default="-0.001,0,0.0005")
    parser.add_argument("--min-train-mean-utility", default="-0.005,-0.002")
    parser.add_argument("--min-train-win-rate", default="0.48,0.50")
    parser.add_argument("--max-train-mean-mae", default="0.015,0.02")
    parser.add_argument("--min-year-samples", default="1,2,3")
    parser.add_argument("--min-year-mean-net", default="-0.002,0")
    parser.add_argument("--min-year-win-rate", default="0.40,0.45,0.50")
    parser.add_argument("--min-test-trades", type=int, default=30)
    parser.add_argument("--min-eval-trades", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--records-cache-dir", default="")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_validate(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
