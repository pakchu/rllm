"""Leakage-safe risk-sensitive state/action policy sweep.

This moves beyond imitating a preselected symbolic rule book.  For each analyzer
state bucket, it chooses among all LONG/SHORT hold candidates using a train-only
risk-sensitive objective, selects objective hyperparameters on test, then reports
eval without using eval for selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training.calibrated_regime_policy import CalibratedPolicyConfig, _aggregate_action
from training.sweep_calibrated_regime_policy import _augment_metrics, _copy_with_keys, _evaluate_rules_precomputed, _parse_key_sets, _precompute_action_rows, _score
from training.sweep_yearly_stable_policy import _load_or_build_records, _records_cache_path, _rule_signature
from training.text_analyzer_trader_data import load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates


@dataclass(frozen=True)
class RiskSensitiveConfig:
    min_samples: int = 12
    min_mean_net: float = -0.002
    min_win_rate: float = 0.45
    max_mean_mae: float = 0.03
    mae_weight: float = 0.5
    cvar_weight: float = 1.0
    downside_weight: float = 0.0
    cvar_alpha: float = 0.2
    key_fields: tuple[str, ...] = ("regime", "trend_alignment", "location", "risk_state")


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split(",") if x.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()]


def _period_years(start: str, end: str) -> float:
    return max(1e-9, (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25)


def _cache_exists(cache_dir: str, *, split: str, start: str, end: str, stride_bars: int, cfg: CalibratedPolicyConfig) -> bool:
    return bool(cache_dir) and _records_cache_path(cache_dir, split=split, start_date=start, end_date=end, stride_bars=stride_bars, cfg=cfg).exists()


def _action_distribution(rows: list[dict[str, Any]], action_key: str) -> dict[str, Any]:
    vals = [r["actions"][action_key] for r in rows if action_key in r["actions"]]
    base = _aggregate_action(rows, action_key)
    if not vals:
        return {**base, "cvar_loss": 0.0, "downside_mean": 0.0, "risk_score": -1e9}
    nets = sorted(float(v["net_return"]) for v in vals)
    losses = [-x for x in nets if x < 0.0]
    tail_n = max(1, int(math.ceil(len(nets) * float(0.2))))
    tail = nets[:tail_n]
    cvar_loss = max(0.0, -sum(tail) / len(tail)) if tail else 0.0
    downside_mean = sum(losses) / len(losses) if losses else 0.0
    return {**base, "cvar_loss": cvar_loss, "downside_mean": downside_mean}


def _risk_objective(stats: dict[str, Any], cfg: RiskSensitiveConfig) -> float:
    return (
        float(stats.get("mean_net_return", 0.0))
        - float(cfg.mae_weight) * float(stats.get("mean_mae", 0.0))
        - float(cfg.cvar_weight) * float(stats.get("cvar_loss", 0.0))
        - float(cfg.downside_weight) * float(stats.get("downside_mean", 0.0))
    )


def fit_risk_sensitive_rules(records: list[dict[str, Any]], cfg: RiskSensitiveConfig) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(str(row["key"]), []).append(row)
    rules: dict[str, dict[str, Any]] = {}
    for key, rows in groups.items():
        action_keys = sorted({a for row in rows for a in row["actions"]})
        qualified = []
        for action_key in action_keys:
            stats = _action_distribution(rows, action_key)
            # recompute cvar at configured alpha
            vals = sorted(float(r["actions"][action_key]["net_return"]) for r in rows if action_key in r["actions"])
            tail_n = max(1, int(math.ceil(len(vals) * float(cfg.cvar_alpha)))) if vals else 0
            tail = vals[:tail_n] if tail_n else []
            stats["cvar_loss"] = max(0.0, -sum(tail) / len(tail)) if tail else 0.0
            stats["risk_objective"] = _risk_objective(stats, cfg)
            if int(stats.get("samples", 0)) < int(cfg.min_samples):
                continue
            if float(stats.get("mean_net_return", 0.0)) < float(cfg.min_mean_net):
                continue
            if float(stats.get("win_rate", 0.0)) < float(cfg.min_win_rate):
                continue
            if float(stats.get("mean_mae", 1e9)) > float(cfg.max_mean_mae):
                continue
            qualified.append(stats)
        if not qualified:
            continue
        best = max(qualified, key=lambda s: (float(s["risk_objective"]), float(s["mean_net_return"]), float(s["win_rate"])))
        rules[key] = {"key": key, "train_samples_in_group": len(rows), "action": best, "qualified_actions": qualified[:5]}
    return rules


def _eval_rules(records: list[dict[str, Any]], rules: dict[str, dict[str, Any]], *, years: float) -> dict[str, Any]:
    return _augment_metrics(
        _evaluate_rules_precomputed(records_count=len(records), action_rows=_precompute_action_rows(records), rules=rules, non_overlapping=True, include_intratrade_mdd=True),
        years=years,
    )


def run_sweep(
    *,
    market_csv: str,
    output: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    eval_start: str,
    eval_end: str,
    wave_trading_root: str = "",
    records_cache_dir: str = "",
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    key_sets: str = "regime,trend_alignment,location,risk_state,Volume State",
    min_samples: str = "12,24,48",
    min_mean_net: str = "-0.002,-0.001,0",
    min_win_rate: str = "0.40,0.45,0.50",
    max_mean_mae: str = "0.015,0.02,0.03",
    mae_weight: str = "0,0.5,1.0",
    cvar_weight: str = "0,0.5,1.0,2.0",
    downside_weight: str = "0,0.5",
    cvar_alpha: str = "0.1,0.2,0.3",
    min_test_trades: int = 30,
    min_eval_trades: int = 30,
    top_k: int = 30,
) -> dict[str, Any]:
    base = CalibratedPolicyConfig(hold_candidates=parse_hold_candidates(hold_candidates))
    have_cache = all(
        _cache_exists(records_cache_dir, split=split, start=start, end=end, stride_bars=stride_bars, cfg=base)
        for split, start, end in (("train", train_start, train_end), ("test", test_start, test_end), ("eval", eval_start, eval_end))
    )
    market = None if have_cache else load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    train_base = _load_or_build_records(market, base, start_date=train_start, end_date=train_end, stride_bars=stride_bars, split="train", records_cache_dir=records_cache_dir)
    test_base = _load_or_build_records(market, base, start_date=test_start, end_date=test_end, stride_bars=stride_bars, split="test", records_cache_dir=records_cache_dir)
    eval_base = _load_or_build_records(market, base, start_date=eval_start, end_date=eval_end, stride_bars=stride_bars, split="eval", records_cache_dir=records_cache_dir)
    test_years = _period_years(test_start, test_end)
    eval_years = _period_years(eval_start, eval_end)
    candidates: list[dict[str, Any]] = []
    total = hits = misses = 0
    for keys in _parse_key_sets(key_sets):
        train = _copy_with_keys(train_base, keys)
        test = _copy_with_keys(test_base, keys)
        metrics_cache: dict[tuple[tuple[str, str, int], ...], dict[str, Any]] = {}
        for ms, mn, wr, mae, mw, cw, dw, ca in itertools.product(_parse_ints(min_samples), _parse_floats(min_mean_net), _parse_floats(min_win_rate), _parse_floats(max_mean_mae), _parse_floats(mae_weight), _parse_floats(cvar_weight), _parse_floats(downside_weight), _parse_floats(cvar_alpha)):
            total += 1
            cfg = RiskSensitiveConfig(min_samples=ms, min_mean_net=mn, min_win_rate=wr, max_mean_mae=mae, mae_weight=mw, cvar_weight=cw, downside_weight=dw, cvar_alpha=ca, key_fields=keys)
            rules = fit_risk_sensitive_rules(train, cfg)
            sig = _rule_signature(rules)
            if sig in metrics_cache:
                hits += 1
                test_metrics = metrics_cache[sig]
            else:
                misses += 1
                test_metrics = _eval_rules(test, rules, years=test_years)
                metrics_cache[sig] = test_metrics
            candidates.append({"test_score": _score(test_metrics, min_trades=min_test_trades), "config": asdict(cfg), "rules_count": len(rules), "test_metrics": test_metrics, "rules_preview": list(rules.values())[:10]})
    top = sorted(candidates, key=lambda r: r["test_score"], reverse=True)[:top_k]
    eval_results = []
    for item in top:
        cfg = RiskSensitiveConfig(**{**item["config"], "key_fields": tuple(item["config"]["key_fields"])})
        train = _copy_with_keys(train_base, cfg.key_fields)
        eval_rows = _copy_with_keys(eval_base, cfg.key_fields)
        rules = fit_risk_sensitive_rules(train, cfg)
        eval_metrics = _eval_rules(eval_rows, rules, years=eval_years)
        eval_results.append({**item, "eval_score": _score(eval_metrics, min_trades=min_eval_trades), "eval_metrics": eval_metrics})
    out = {
        "periods": {"train": [train_start, train_end], "test": [test_start, test_end], "eval": [eval_start, eval_end]},
        "records": {"train": len(train_base), "test": len(test_base), "eval": len(eval_base)},
        "sweep": {"total_configs": total, "min_test_trades": min_test_trades, "min_eval_trades": min_eval_trades, "metrics_cache_hits": hits, "metrics_cache_misses": misses},
        "leakage_guard": {"rules_fit_on_train_only": True, "hyperparams_selected_on_test_only": True, "eval_used_for_selection": False},
        "top_by_test_then_eval": eval_results,
        "top_by_eval_report_only": sorted(eval_results, key=lambda r: (r["eval_score"], r["test_score"]), reverse=True),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep risk-sensitive analyzer-state policies")
    for arg in ["market-csv", "output", "train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"]:
        p.add_argument("--" + arg, required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--records-cache-dir", default="")
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--hold-candidates", default="48,96,144,288")
    p.add_argument("--key-sets", default="regime,trend_alignment,location,risk_state,Volume State")
    p.add_argument("--min-samples", default="12,24,48")
    p.add_argument("--min-mean-net", default="-0.002,-0.001,0")
    p.add_argument("--min-win-rate", default="0.40,0.45,0.50")
    p.add_argument("--max-mean-mae", default="0.015,0.02,0.03")
    p.add_argument("--mae-weight", default="0,0.5,1.0")
    p.add_argument("--cvar-weight", default="0,0.5,1.0,2.0")
    p.add_argument("--downside-weight", default="0,0.5")
    p.add_argument("--cvar-alpha", default="0.1,0.2,0.3")
    p.add_argument("--min-test-trades", type=int, default=30)
    p.add_argument("--min-eval-trades", type=int, default=30)
    p.add_argument("--top-k", type=int, default=30)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_sweep(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
