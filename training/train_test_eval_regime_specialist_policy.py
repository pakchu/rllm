"""Train/test/eval validation for router-first specialist policies.

The router/specialist configuration is selected on the test window only.  Eval
is reported after selection and is not used to choose parameters.
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
from training.sweep_calibrated_regime_policy import _augment_metrics, _parse_floats, _parse_ints, _parse_key_sets, _score
from training.sweep_regime_specialist_policy import (
    _precompute_router_stats,
    _specialist_signature,
    evaluate_router_specialists,
    fit_router_specialists_from_stats,
)
from training.sweep_yearly_stable_policy import _load_or_build_records, _records_cache_path
from training.text_analyzer_trader_data import load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates
from training.yearly_stable_regime_policy import YearlyStableConfig


def _period_years(start: str, end: str) -> float:
    return max(1e-9, (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25)


def _cache_exists(cache_dir: str, *, split: str, start: str, end: str, stride_bars: int, cfg: CalibratedPolicyConfig) -> bool:
    return bool(cache_dir) and _records_cache_path(cache_dir, split=split, start_date=start, end_date=end, stride_bars=stride_bars, cfg=cfg).exists()


def _eval_specialists_cached(
    records: list[dict[str, Any]],
    specialists: dict[str, dict[str, Any]],
    *,
    router_fields: tuple[str, ...],
    specialist_key_fields: tuple[str, ...],
    years: float,
    cache: dict[tuple[tuple[str, tuple[tuple[str, str, int], ...]], ...], dict[str, Any]],
) -> dict[str, Any]:
    signature = _specialist_signature(specialists)
    if signature not in cache:
        cache[signature] = _augment_metrics(
            evaluate_router_specialists(
                records,
                specialists,
                router_fields=router_fields,
                specialist_key_fields=specialist_key_fields,
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
    records_cache_dir: str = "",
    router_fields: str = "regime,risk_state",
    specialist_key_sets: str = "trend_alignment,location,Volume State",
    min_train_samples: str = "8,12,16",
    min_train_mean_net: str = "0,0.0005,0.001",
    min_train_mean_utility: str = "-0.003,-0.001,0",
    min_train_win_rate: str = "0.48,0.50,0.52",
    max_train_mean_mae: str = "0.01,0.015,0.02",
    min_year_samples: str = "1,2,3",
    min_year_mean_net: str = "-0.002,0",
    min_year_win_rate: str = "0.40,0.45,0.50",
    min_good_years: str = "0,2,3",
    max_bad_year_mean_net: str = "-0.015,-0.01,-0.005",
    min_test_trades: int = 30,
    min_eval_trades: int = 30,
    top_k: int = 20,
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
    train_records = _load_or_build_records(market, base, start_date=train_start, end_date=train_end, stride_bars=stride_bars, split="train", records_cache_dir=records_cache_dir)
    test_records = _load_or_build_records(market, base, start_date=test_start, end_date=test_end, stride_bars=stride_bars, split="test", records_cache_dir=records_cache_dir)
    eval_records = _load_or_build_records(market, base, start_date=eval_start, end_date=eval_end, stride_bars=stride_bars, split="eval", records_cache_dir=records_cache_dir)

    router = tuple(x.strip() for x in router_fields.split(",") if x.strip())
    test_years = _period_years(test_start, test_end)
    eval_years = _period_years(eval_start, eval_end)
    candidates: list[dict[str, Any]] = []
    total = hits = misses = 0

    for specialist_fields in _parse_key_sets(specialist_key_sets):
        router_stats = _precompute_router_stats(train_records, router_fields=router, specialist_key_fields=specialist_fields)
        test_cache: dict[tuple[tuple[str, tuple[tuple[str, str, int], ...]], ...], dict[str, Any]] = {}
        for min_samples, mean_net, mean_utility, win_rate, max_mae, year_samples, year_net, year_win, good_years, bad_net in itertools.product(
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
        ):
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
            stable = YearlyStableConfig(min_year_samples=year_samples, min_year_mean_net=year_net, min_year_win_rate=year_win, max_year_mean_mae=max_mae)
            specialists = fit_router_specialists_from_stats(router_stats, cfg=cfg, stable=stable, min_good_years=good_years, max_bad_year_mean_net=bad_net)
            before = len(test_cache)
            test_metrics = _eval_specialists_cached(test_records, specialists, router_fields=router, specialist_key_fields=specialist_fields, years=test_years, cache=test_cache)
            if len(test_cache) == before:
                hits += 1
            else:
                misses += 1
            candidates.append(
                {
                    "test_score": _score(test_metrics, min_trades=min_test_trades),
                    "router_fields": list(router),
                    "specialist_key_fields": list(specialist_fields),
                    "config": asdict(cfg),
                    "stable_config": asdict(stable),
                    "min_good_years": int(good_years),
                    "max_bad_year_mean_net": float(bad_net),
                    "router_count": len(specialists),
                    "rules_count": sum(len(book["rules"]) for book in specialists.values()),
                    "test_metrics": test_metrics,
                    "routers_preview": list(specialists.values())[:10],
                }
            )

    top_by_test = sorted(candidates, key=lambda r: r["test_score"], reverse=True)[:top_k]
    eval_results: list[dict[str, Any]] = []
    eval_cache: dict[tuple[tuple[str, tuple[tuple[str, str, int], ...]], ...], dict[str, Any]] = {}
    for item in top_by_test:
        specialist_fields = tuple(item["specialist_key_fields"])
        cfg = CalibratedPolicyConfig(**{k: v for k, v in item["config"].items() if k != "hold_candidates"}, hold_candidates=tuple(item["config"]["hold_candidates"]))
        stable = YearlyStableConfig(**item["stable_config"])
        router_stats = _precompute_router_stats(train_records, router_fields=router, specialist_key_fields=specialist_fields)
        specialists = fit_router_specialists_from_stats(router_stats, cfg=cfg, stable=stable, min_good_years=int(item["min_good_years"]), max_bad_year_mean_net=float(item["max_bad_year_mean_net"]))
        eval_metrics = _eval_specialists_cached(eval_records, specialists, router_fields=router, specialist_key_fields=specialist_fields, years=eval_years, cache=eval_cache)
        eval_results.append({**item, "eval_score": _score(eval_metrics, min_trades=min_eval_trades), "eval_metrics": eval_metrics})

    report = {
        "periods": {"train": [train_start, train_end], "test": [test_start, test_end], "eval": [eval_start, eval_end]},
        "records": {"train": len(train_records), "test": len(test_records), "eval": len(eval_records)},
        "sweep": {"total_configs": total, "min_test_trades": int(min_test_trades), "min_eval_trades": int(min_eval_trades), "test_metrics_cache_hits": hits, "test_metrics_cache_misses": misses},
        "leakage_guard": {"selection_window": "test", "eval_used_for_selection": False},
        "top_by_test_then_eval": eval_results,
        "top_by_eval_report_only": sorted(eval_results, key=lambda r: (r["eval_score"], r["test_score"]), reverse=True),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe train/test/eval router-specialist policy validation")
    for arg in ["market-csv", "output", "train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"]:
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
    parser.add_argument("--min-year-samples", default="1,2,3")
    parser.add_argument("--min-year-mean-net", default="-0.002,0")
    parser.add_argument("--min-year-win-rate", default="0.40,0.45,0.50")
    parser.add_argument("--min-good-years", default="0,2,3")
    parser.add_argument("--max-bad-year-mean-net", default="-0.015,-0.01,-0.005")
    parser.add_argument("--min-test-trades", type=int, default=30)
    parser.add_argument("--min-eval-trades", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_validate(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
