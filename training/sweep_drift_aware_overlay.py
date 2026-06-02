"""Leakage-safe sweep for rolling drift-risk overlays on a fixed stable policy.

Rules are fit on train. Drift overlay thresholds are selected on test only from
completed prior outcomes, then the selected overlays are reported on eval.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from training.calibrated_regime_policy import CalibratedPolicyConfig, _aggregate_action, _metrics_from_trades
from training.export_drift_aware_policy_labels import _aggregate_outcomes, _rule_action_key, classify_drift_risk
from training.sweep_calibrated_regime_policy import _copy_with_keys, _score
from training.sweep_yearly_stable_policy import _fit_from_stats, _load_or_build_records, _precompute_group_year_stats, _records_cache_path
from training.text_analyzer_trader_data import load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates
from training.yearly_stable_regime_policy import YearlyStableConfig


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split(",") if x.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()]


def _period_years(start: str, end: str) -> float:
    return max(1e-9, (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25)


def _cache_exists(cache_dir: str, *, split: str, start: str, end: str, stride_bars: int, cfg: CalibratedPolicyConfig) -> bool:
    return bool(cache_dir) and _records_cache_path(cache_dir, split=split, start_date=start, end_date=end, stride_bars=stride_bars, cfg=cfg).exists()


def _selected_item(report: dict[str, Any], top_index: int) -> dict[str, Any]:
    rows = report.get("top_by_test_then_eval") or report.get("top") or []
    if not rows:
        raise ValueError("report has no selected rows")
    return rows[int(top_index)]


def _fit_selected_rules(report: dict[str, Any], top_index: int, train_records: list[dict[str, Any]]) -> tuple[CalibratedPolicyConfig, YearlyStableConfig, dict[str, dict[str, Any]]]:
    item = _selected_item(report, top_index)
    cfg_payload = item["config"]
    cfg = CalibratedPolicyConfig(**{k: v for k, v in cfg_payload.items() if k != "hold_candidates"}, hold_candidates=tuple(cfg_payload["hold_candidates"]))
    stable = YearlyStableConfig(**item["stable_config"])
    keyed_train = _copy_with_keys(train_records, tuple(cfg.key_fields))
    rules = _fit_from_stats(_precompute_group_year_stats(keyed_train), cfg, stable)
    return cfg, stable, rules


def _baseline_stats(train_records: list[dict[str, Any]], rules: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, rule in rules.items():
        rows = [row for row in train_records if str(row["key"]) == key]
        out[key] = _aggregate_action(rows, _rule_action_key(rule))
    return out


def simulate_drift_overlay(
    records: list[dict[str, Any]],
    *,
    rules: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    seed_history: dict[str, list[dict[str, Any]]],
    recent_window: int,
    min_recent_samples: int,
    high_min_mean_net: float,
    high_min_win_rate: float,
    high_max_mean_mae: float,
    medium_mean_drop: float,
    skip_risks: tuple[str, ...],
    years: float,
) -> dict[str, Any]:
    from collections import defaultdict, deque

    history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=int(recent_window)))
    for key, rows in seed_history.items():
        for row in rows[-int(recent_window) :]:
            history[key].append(row)
    pending: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    next_available_pos = -1
    drift_counts: dict[str, int] = {}
    skip_counts: dict[str, int] = {}
    for row in sorted(records, key=lambda r: int(r.get("signal_pos", 0))):
        signal_pos = int(row["signal_pos"])
        keep = []
        for item in pending:
            if int(item["complete_pos"]) < signal_pos:
                history[item["key"]].append(item["outcome"])
            else:
                keep.append(item)
        pending = keep
        key = str(row["key"])
        rule = rules.get(key)
        if signal_pos <= next_available_pos:
            skip_counts["POSITION_OPEN_SKIP"] = skip_counts.get("POSITION_OPEN_SKIP", 0) + 1
            continue
        if not rule:
            skip_counts["NO_RULE"] = skip_counts.get("NO_RULE", 0) + 1
            continue
        recent = _aggregate_outcomes(list(history.get(key, [])))
        risk = classify_drift_risk(
            baseline=baseline.get(key, {}),
            recent=recent,
            min_recent_samples=min_recent_samples,
            high_min_mean_net=high_min_mean_net,
            high_min_win_rate=high_min_win_rate,
            high_max_mean_mae=high_max_mean_mae,
            medium_mean_drop=medium_mean_drop,
        )
        drift_counts[risk] = drift_counts.get(risk, 0) + 1
        outcome = row["actions"].get(_rule_action_key(rule))
        hold_bars = int(rule["action"]["hold_bars"])
        if outcome:
            pending.append({"key": key, "complete_pos": signal_pos + hold_bars, "outcome": {"signal_pos": signal_pos, **outcome}})
        if risk in skip_risks:
            skip_counts[f"DRIFT_{risk}_SKIP"] = skip_counts.get(f"DRIFT_{risk}_SKIP", 0) + 1
            continue
        if not outcome:
            skip_counts["MISSING_OUTCOME"] = skip_counts.get("MISSING_OUTCOME", 0) + 1
            continue
        trades.append({"date": row["date"], "signal_pos": signal_pos, "policy_key": key, "drift_risk": risk, **outcome})
        next_available_pos = signal_pos + hold_bars
    metrics = _metrics_from_trades(trades, records_count=len(records), include_intratrade_mdd=True)
    compounded = float(metrics.get("compounded_return", 0.0) or 0.0)
    mdd = float(metrics.get("strict_mdd_proxy", 0.0) or 0.0)
    cagr = (1.0 + compounded) ** (1.0 / years) - 1.0 if compounded > -0.999 else 0.0
    metrics.update({"years": years, "cagr_proxy": cagr, "cagr_to_mdd_proxy": cagr / mdd if mdd > 0 else (float("inf") if cagr > 0 else 0.0), "non_overlapping": True})
    return {"metrics": metrics, "drift_counts": drift_counts, "skip_counts": skip_counts}


def _seed_history(train_records: list[dict[str, Any]], rules: dict[str, dict[str, Any]], recent_window: int) -> dict[str, list[dict[str, Any]]]:
    hist: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(train_records, key=lambda r: int(r.get("signal_pos", 0))):
        key = str(row["key"])
        rule = rules.get(key)
        if not rule:
            continue
        outcome = row["actions"].get(_rule_action_key(rule))
        if outcome:
            hist.setdefault(key, []).append({"signal_pos": int(row["signal_pos"]), **outcome})
    return {k: v[-int(recent_window) :] for k, v in hist.items()}


def run_sweep(
    *,
    report: str,
    output: str,
    market_csv: str,
    records_cache_dir: str,
    wave_trading_root: str = "",
    top_index: int = 0,
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    recent_window: str = "8,12,24",
    min_recent_samples: str = "3,6",
    high_min_mean_net: str = "0,0.001,0.002",
    high_min_win_rate: str = "0.45,0.50,0.55",
    high_max_mean_mae: str = "0.012,0.015,0.02",
    medium_mean_drop: str = "0.002,0.004",
    skip_risks: str = "HIGH;HIGH,MEDIUM",
    min_test_trades: int = 20,
    min_eval_trades: int = 30,
    top_k: int = 20,
) -> dict[str, Any]:
    payload = json.loads(Path(report).read_text())
    periods = payload["periods"]
    base = CalibratedPolicyConfig(hold_candidates=parse_hold_candidates(hold_candidates))
    have_cache = all(
        _cache_exists(records_cache_dir, split=split, start=periods[split][0], end=periods[split][1], stride_bars=stride_bars, cfg=base)
        for split in ("train", "test", "eval")
    )
    market = None if have_cache else load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    train = _load_or_build_records(market, base, start_date=periods["train"][0], end_date=periods["train"][1], stride_bars=stride_bars, split="train", records_cache_dir=records_cache_dir)
    test = _load_or_build_records(market, base, start_date=periods["test"][0], end_date=periods["test"][1], stride_bars=stride_bars, split="test", records_cache_dir=records_cache_dir)
    eval_rows = _load_or_build_records(market, base, start_date=periods["eval"][0], end_date=periods["eval"][1], stride_bars=stride_bars, split="eval", records_cache_dir=records_cache_dir)
    cfg, stable, rules = _fit_selected_rules(payload, top_index, train)
    train_k = _copy_with_keys(train, tuple(cfg.key_fields))
    test_k = _copy_with_keys(test, tuple(cfg.key_fields))
    eval_k = _copy_with_keys(eval_rows, tuple(cfg.key_fields))
    baseline = _baseline_stats(train_k, rules)
    test_years = _period_years(periods["test"][0], periods["test"][1])
    eval_years = _period_years(periods["eval"][0], periods["eval"][1])
    rows = []
    total = 0
    for rw, mrs, hmn, hwr, hmae, md, skip_raw in itertools.product(
        _parse_ints(recent_window),
        _parse_ints(min_recent_samples),
        _parse_floats(high_min_mean_net),
        _parse_floats(high_min_win_rate),
        _parse_floats(high_max_mean_mae),
        _parse_floats(medium_mean_drop),
        [tuple(x.strip() for x in group.split(",") if x.strip()) for group in str(skip_risks).split(";") if group.strip()],
    ):
        total += 1
        seed = _seed_history(train_k, rules, rw)
        test_rep = simulate_drift_overlay(test_k, rules=rules, baseline=baseline, seed_history=seed, recent_window=rw, min_recent_samples=mrs, high_min_mean_net=hmn, high_min_win_rate=hwr, high_max_mean_mae=hmae, medium_mean_drop=md, skip_risks=skip_raw, years=test_years)
        rows.append(
            {
                "test_score": _score(test_rep["metrics"], min_trades=min_test_trades),
                "drift_config": {"recent_window": rw, "min_recent_samples": mrs, "high_min_mean_net": hmn, "high_min_win_rate": hwr, "high_max_mean_mae": hmae, "medium_mean_drop": md, "skip_risks": list(skip_raw)},
                "test": test_rep,
            }
        )
    top = sorted(rows, key=lambda r: r["test_score"], reverse=True)[:top_k]
    for row in top:
        c = row["drift_config"]
        seed = _seed_history(train_k, rules, int(c["recent_window"]))
        row["eval"] = simulate_drift_overlay(eval_k, rules=rules, baseline=baseline, seed_history=seed, recent_window=int(c["recent_window"]), min_recent_samples=int(c["min_recent_samples"]), high_min_mean_net=float(c["high_min_mean_net"]), high_min_win_rate=float(c["high_min_win_rate"]), high_max_mean_mae=float(c["high_max_mean_mae"]), medium_mean_drop=float(c["medium_mean_drop"]), skip_risks=tuple(c["skip_risks"]), years=eval_years)
        row["eval_score"] = _score(row["eval"]["metrics"], min_trades=min_eval_trades)
    out = {
        "source_report": report,
        "periods": periods,
        "records": {"train": len(train), "test": len(test), "eval": len(eval_rows)},
        "policy": {"rules_count": len(rules), "key_fields": list(cfg.key_fields), "config": asdict(cfg), "stable_config": asdict(stable)},
        "sweep": {"total_configs": total, "min_test_trades": min_test_trades, "min_eval_trades": min_eval_trades},
        "leakage_guard": {"rules_fit_on_train_only": True, "drift_overlay_selected_on_test_only": True, "eval_used_for_selection": False},
        "top_by_test_then_eval": top,
        "top_by_eval_report_only": sorted(top, key=lambda r: (r["eval_score"], r["test_score"]), reverse=True),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep test-selected drift-aware overlays and report eval")
    for arg in ["report", "output", "market-csv", "records-cache-dir"]:
        p.add_argument("--" + arg, required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--top-index", type=int, default=0)
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--hold-candidates", default="48,96,144,288")
    p.add_argument("--recent-window", default="8,12,24")
    p.add_argument("--min-recent-samples", default="3,6")
    p.add_argument("--high-min-mean-net", default="0,0.001,0.002")
    p.add_argument("--high-min-win-rate", default="0.45,0.50,0.55")
    p.add_argument("--high-max-mean-mae", default="0.012,0.015,0.02")
    p.add_argument("--medium-mean-drop", default="0.002,0.004")
    p.add_argument("--skip-risks", default="HIGH;HIGH,MEDIUM")
    p.add_argument("--min-test-trades", type=int, default=20)
    p.add_argument("--min-eval-trades", type=int, default=30)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_sweep(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
