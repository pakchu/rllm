"""Analyze selected TTE policy drift between test and eval windows.

This is a diagnostic bridge for the analyzer/router redesign: after a policy is
selected on test, report which rule/action buckets invert or decay on eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from training.calibrated_regime_policy import CalibratedPolicyConfig, _aggregate_action, _summary_key
from training.sweep_yearly_stable_policy import _fit_from_stats, _load_or_build_records, _precompute_group_year_stats, _records_cache_path
from training.text_analyzer_trader_data import load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates
from training.yearly_stable_regime_policy import YearlyStableConfig


def _cache_exists(cache_dir: str, *, split: str, start: str, end: str, stride_bars: int, cfg: CalibratedPolicyConfig) -> bool:
    return bool(cache_dir) and _records_cache_path(cache_dir, split=split, start_date=start, end_date=end, stride_bars=stride_bars, cfg=cfg).exists()


def _year_month(date: str) -> str:
    dt = pd.to_datetime(date)
    return f"{dt.year}-{dt.month:02d}"


def _selected_policy_item(report: dict[str, Any], index: int) -> dict[str, Any]:
    rows = report.get("top_by_test_then_eval") or report.get("top") or []
    if not rows:
        raise ValueError("report has no selectable top rows")
    return rows[int(index)]


def _copy_with_key(records: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for row in records:
        clone = dict(row)
        clone["key"] = _summary_key(row["summary"], key_fields)
        out.append(clone)
    return out


def _matched_outcomes(records: list[dict[str, Any]], rules: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in records:
        key = str(row["key"])
        rule = rules.get(key)
        if not rule:
            continue
        action = rule["action"]
        action_key = f"{action['side']}_{int(action['hold_bars'])}"
        outcome = row["actions"].get(action_key)
        if not outcome:
            continue
        rows.append({"date": row["date"], "key": key, "action_key": action_key, **outcome})
    return rows


def _aggregate_flat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, "mean_net_return": 0.0, "win_rate": 0.0, "mean_mae": 0.0, "mean_utility": 0.0}
    nets = [float(r["net_return"]) for r in rows]
    maes = [float(r["mae"]) for r in rows]
    utils = [float(r["utility"]) for r in rows]
    return {
        "samples": len(rows),
        "mean_net_return": sum(nets) / len(nets),
        "win_rate": sum(1 for x in nets if x > 0.0) / len(nets),
        "mean_mae": sum(maes) / len(maes),
        "mean_utility": sum(utils) / len(utils),
        "sum_net_return": sum(nets),
    }


def _by_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["key"]), []).append(row)
    return {key: _aggregate_flat(group) for key, group in sorted(groups.items())}


def _by_month(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_year_month(row["date"]), []).append(row)
    return {key: _aggregate_flat(group) for key, group in sorted(groups.items())}


def run_analysis(
    *,
    report: str,
    output: str,
    market_csv: str,
    wave_trading_root: str = "",
    records_cache_dir: str = "",
    top_index: int = 0,
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
) -> dict[str, Any]:
    payload = json.loads(Path(report).read_text())
    periods = payload["periods"]
    item = _selected_policy_item(payload, top_index)
    cfg_payload = item["config"]
    cfg = CalibratedPolicyConfig(**{k: v for k, v in cfg_payload.items() if k != "hold_candidates"}, hold_candidates=tuple(cfg_payload.get("hold_candidates") or parse_hold_candidates(hold_candidates)))
    stable = YearlyStableConfig(**item["stable_config"])
    have_cache = all(
        _cache_exists(records_cache_dir, split=split, start=period[0], end=period[1], stride_bars=stride_bars, cfg=cfg)
        for split, period in (("train", periods["train"]), ("test", periods["test"]), ("eval", periods["eval"]))
    )
    market = None if have_cache else load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    train = _load_or_build_records(market, cfg, start_date=periods["train"][0], end_date=periods["train"][1], stride_bars=stride_bars, split="train", records_cache_dir=records_cache_dir)
    test = _load_or_build_records(market, cfg, start_date=periods["test"][0], end_date=periods["test"][1], stride_bars=stride_bars, split="test", records_cache_dir=records_cache_dir)
    eval_rows = _load_or_build_records(market, cfg, start_date=periods["eval"][0], end_date=periods["eval"][1], stride_bars=stride_bars, split="eval", records_cache_dir=records_cache_dir)
    keys = tuple(cfg.key_fields)
    train_k = _copy_with_key(train, keys)
    rules = _fit_from_stats(_precompute_group_year_stats(train_k), cfg, stable)
    test_hits = _matched_outcomes(_copy_with_key(test, keys), rules)
    eval_hits = _matched_outcomes(_copy_with_key(eval_rows, keys), rules)
    train_key_stats = {
        key: _aggregate_action([row for row in train_k if str(row["key"]) == key], f"{rule['action']['side']}_{int(rule['action']['hold_bars'])}")
        for key, rule in rules.items()
    }
    test_by_key = _by_key(test_hits)
    eval_by_key = _by_key(eval_hits)
    drift = []
    for key in sorted(set(test_by_key) | set(eval_by_key)):
        t = test_by_key.get(key, {"samples": 0, "mean_net_return": 0.0, "win_rate": 0.0})
        e = eval_by_key.get(key, {"samples": 0, "mean_net_return": 0.0, "win_rate": 0.0})
        drift.append(
            {
                "key": key,
                "action": rules.get(key, {}).get("action", {}),
                "train": train_key_stats.get(key, {}),
                "test": t,
                "eval": e,
                "mean_net_delta_eval_minus_test": float(e.get("mean_net_return", 0.0)) - float(t.get("mean_net_return", 0.0)),
            }
        )
    drift.sort(key=lambda r: (float(r["eval"].get("sum_net_return", 0.0)), float(r["mean_net_delta_eval_minus_test"])))
    out = {
        "source_report": report,
        "selected_top_index": int(top_index),
        "periods": periods,
        "policy": {"rules_count": len(rules), "key_fields": list(keys), "config": cfg_payload, "stable_config": item["stable_config"]},
        "test_overall": _aggregate_flat(test_hits),
        "eval_overall": _aggregate_flat(eval_hits),
        "test_by_month": _by_month(test_hits),
        "eval_by_month": _by_month(eval_hits),
        "worst_eval_rule_drifts": drift[:25],
        "best_eval_rule_drifts": list(reversed(drift[-10:])),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze selected TTE policy drift from test to eval")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--market-csv", required=True)
    parser.add_argument("--wave-trading-root", default="")
    parser.add_argument("--records-cache-dir", default="")
    parser.add_argument("--top-index", type=int, default=0)
    parser.add_argument("--stride-bars", type=int, default=12)
    parser.add_argument("--hold-candidates", default="48,96,144,288")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_analysis(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
