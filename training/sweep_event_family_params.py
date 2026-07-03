"""No-leak family/parameter sweep for event candidate alpha probes.

The selector experiments showed that a weak candidate book cannot be rescued by
an LLM router.  This script searches for a *candidate surface* first: thresholds
are fit on train only, candidate family/quantile/hold combinations are ranked on
train+test only, and eval metrics are emitted only as a final holdout check.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.event_candidate_pool_probe import (
    EventPoolConfig,
    _candidate_rows_for_family,
    _feature_candidates,
    _load_market,
    _simulate_rows,
    _split_mask,
)


def _csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _metric_row(result: dict[str, Any]) -> dict[str, Any]:
    sim = result.get("sim", {})
    stats = result.get("trade_stats", {})
    return {
        "ret_pct": sim.get("ret_pct"),
        "cagr_pct": sim.get("cagr_pct"),
        "strict_mdd_pct": sim.get("strict_mdd_pct"),
        "cagr_to_strict_mdd": sim.get("cagr_to_strict_mdd"),
        "trade_entries": sim.get("trade_entries"),
        "side_counts": sim.get("side_counts"),
        "p_value_mean_ret_approx": stats.get("p_value_mean_ret_approx"),
        "mean_trade_ret_pct": stats.get("mean_trade_ret_pct"),
        "effect_size_d": stats.get("effect_size_d"),
    }


def _rank_metric(row: dict[str, Any], *, min_trades: int) -> float:
    trades = int(row.get("trade_entries", 0) or 0)
    ratio = float(row.get("cagr_to_strict_mdd", -1e9) or -1e9)
    cagr = float(row.get("cagr_pct", 0.0) or 0.0)
    p_value = float(row.get("p_value_mean_ret_approx", 1.0) or 1.0)
    if trades < int(min_trades) or cagr <= 0.0 or not math.isfinite(ratio):
        return -1e9 + trades
    return ratio + 0.02 * cagr - 0.20 * max(0.0, p_value - 0.20) + 0.05 * math.log1p(trades)


def _sort_dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-priority same-time/same-side row before strict sim.

    The strict candidate simulator accepts one position until exit and iterates
    rows in order, so duplicate same-side signals at the same timestamp should
    not multiply-count when combining prefix ensembles.
    """
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("signal_date")), str(row.get("side")))
        old = seen.get(key)
        if old is None or float(row.get("score_mean", 0.0) or 0.0) > float(old.get("score_mean", 0.0) or 0.0):
            seen[key] = row
    return sorted(seen.values(), key=lambda r: (str(r.get("signal_date")), -float(r.get("score_mean", 0.0) or 0.0)))


def run(cfg: EventPoolConfig, *, hold_bars_list: list[int], quantiles: list[float], ensemble_hold: int, ensemble_prefixes: list[int], top_n: int) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    dates = pd.to_datetime(market["date"])
    masks = {
        "train": _split_mask(dates, cfg.train_start, cfg.train_end),
        "test": _split_mask(dates, cfg.val_start, cfg.val_end),
        "eval": _split_mask(dates, cfg.eval_start, cfg.eval_end),
    }
    families = _feature_candidates(features)
    if cfg.family_include:
        needles = [x.strip() for x in str(cfg.family_include).split(",") if x.strip()]
        families = {name: value for name, value in families.items() if any(needle in name for needle in needles)}
        if not families:
            raise ValueError(f"no candidate families matched --family-include={cfg.family_include!r}")

    candidates: list[dict[str, Any]] = []
    row_cache: dict[tuple[str, int, float, str], list[dict[str, Any]]] = {}

    def rows_for(family: str, hold: int, quantile: float, split: str, *, priority: float = 1.0) -> list[dict[str, Any]]:
        key = (family, int(hold), float(quantile), split)
        if key in row_cache:
            rows = [dict(r) for r in row_cache[key]]
        else:
            local_cfg = replace(cfg, hold_bars=int(hold), quantile=float(quantile))
            strength, direction = families[family]
            x = strength[masks["train"] & np.isfinite(strength) & (strength > 0.0)]
            if x.size < 100:
                rows = []
            else:
                threshold = float(np.quantile(x, float(quantile)))
                rows = _candidate_rows_for_family(market, strength, direction, family=family, threshold=threshold, mask=masks[split], cfg=local_cfg)
            row_cache[key] = [dict(r) for r in rows]
        for row in rows:
            row["score_mean"] = float(priority)
            row["family_param"] = f"{family}@h{hold}@q{quantile:g}"
        return rows

    for hold in hold_bars_list:
        local_cfg = replace(cfg, hold_bars=int(hold))
        for quantile in quantiles:
            for family, (strength, _direction) in families.items():
                x = strength[masks["train"] & np.isfinite(strength) & (strength > 0.0)]
                if x.size < 100:
                    continue
                threshold = float(np.quantile(x, float(quantile)))
                split_metrics: dict[str, Any] = {}
                for split in ("train", "test", "eval"):
                    rows = rows_for(family, int(hold), float(quantile), split)
                    split_metrics[split] = _metric_row(_simulate_rows(rows, market, local_cfg)) | {"candidate_rows": len(rows)}
                train_rank = _rank_metric(split_metrics["train"], min_trades=int(cfg.min_train_trades))
                test_rank = _rank_metric(split_metrics["test"], min_trades=int(cfg.min_val_trades))
                if train_rank <= -1e8 or test_rank <= -1e8:
                    continue
                candidates.append(
                    {
                        "family": family,
                        "hold_bars": int(hold),
                        "quantile": float(quantile),
                        "threshold": threshold,
                        "selection_score": float(test_rank + 0.25 * train_rank),
                        **split_metrics,
                    }
                )

    candidates.sort(key=lambda r: float(r["selection_score"]), reverse=True)
    top = candidates[: int(top_n)]

    ensemble_results: list[dict[str, Any]] = []
    ensemble_combos = [row for row in top if int(row["hold_bars"]) == int(ensemble_hold)]
    for prefix in ensemble_prefixes:
        chosen = ensemble_combos[: int(prefix)]
        if not chosen:
            continue
        split_metrics = {}
        local_cfg = replace(cfg, hold_bars=int(ensemble_hold))
        for split in ("train", "test", "eval"):
            rows: list[dict[str, Any]] = []
            for idx, combo in enumerate(chosen):
                priority = 1.0 + 0.01 * (len(chosen) - idx)
                rows.extend(rows_for(combo["family"], int(combo["hold_bars"]), float(combo["quantile"]), split, priority=priority))
            rows = _sort_dedup_rows(rows)
            split_metrics[split] = _metric_row(_simulate_rows(rows, market, local_cfg)) | {"candidate_rows": len(rows)}
        ensemble_results.append(
            {
                "n_prefix": int(prefix),
                "combos": [{"family": c["family"], "hold_bars": c["hold_bars"], "quantile": c["quantile"]} for c in chosen],
                **split_metrics,
            }
        )

    eligible_ensembles = [
        row
        for row in ensemble_results
        if float(row["train"].get("cagr_pct", 0.0) or 0.0) > 0.0 and float(row["test"].get("cagr_pct", 0.0) or 0.0) > 0.0
    ]
    selected_ensemble = max(
        eligible_ensembles,
        key=lambda row: (float(row["test"].get("cagr_to_strict_mdd", -1e9) or -1e9), int(row["test"].get("trade_entries", 0) or 0)),
        default=None,
    )

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "sweep": {"hold_bars_list": hold_bars_list, "quantiles": quantiles, "candidate_count": len(candidates), "top": top},
        "ensemble": {"hold_bars": int(ensemble_hold), "prefix_results": ensemble_results, "selected_by_test": selected_ensemble},
        "leakage_guard": {
            "thresholds_fit_on_train_only": True,
            "sweep_selection_uses_train_and_test_only": True,
            "ensemble_prefix_selected_by_test_only": True,
            "eval_not_used_for_selection": True,
            "same_hold_only_for_prefix_ensemble": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-leak event-family parameter sweep")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default="2021-01-01")
    p.add_argument("--train-end", default="2025-01-01")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--test-end", default="2026-01-01")
    p.add_argument("--eval-start", default="2026-01-01")
    p.add_argument("--eval-end", default="2026-06-01")
    p.add_argument("--hold-bars", default="96,144,216,288,432,576")
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--stride-bars", type=int, default=24)
    p.add_argument("--quantiles", default="0.70,0.75,0.80,0.85,0.90,0.92,0.95")
    p.add_argument("--min-train-trades", type=int, default=80)
    p.add_argument("--min-test-trades", type=int, default=20)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--family-include", default="")
    p.add_argument("--ensemble-hold", type=int, default=144)
    p.add_argument("--ensemble-prefixes", default="1,2,3,5,8,12")
    p.add_argument("--top-n", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EventPoolConfig(
        input_csv=args.input_csv,
        output=args.output,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.test_start,
        val_end=args.test_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        hold_bars=int(_csv_ints(args.hold_bars)[0]),
        entry_delay_bars=args.entry_delay_bars,
        window_size=args.window_size,
        stride_bars=args.stride_bars,
        quantile=float(_csv_floats(args.quantiles)[0]),
        min_train_trades=args.min_train_trades,
        min_val_trades=args.min_test_trades,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        family_include=args.family_include,
    )
    report = run(
        cfg,
        hold_bars_list=_csv_ints(args.hold_bars),
        quantiles=_csv_floats(args.quantiles),
        ensemble_hold=args.ensemble_hold,
        ensemble_prefixes=_csv_ints(args.ensemble_prefixes),
        top_n=args.top_n,
    )
    print(json.dumps({"top": report["sweep"]["top"][:10], "selected_ensemble": report["ensemble"].get("selected_by_test")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
