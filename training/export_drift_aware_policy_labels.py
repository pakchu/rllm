"""Export analyzer/trader rows for a drift-aware calibrated policy.

The policy rules are fit on the train period.  At each label timestamp, a
past-only rolling monitor compares completed prior outcomes for the same policy
key/action against the train baseline.  The analyzer target exposes that drift
state; the trader target skips otherwise-valid trades when drift risk is high.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd

from training.calibrated_regime_policy import CalibratedPolicyConfig, _aggregate_action, _metrics_from_trades, _summary_key
from training.export_calibrated_policy_labels import build_policy_trader_input, format_policy_book
from training.sweep_calibrated_regime_policy import _copy_with_keys
from training.sweep_yearly_stable_policy import _fit_from_stats, _load_or_build_records, _precompute_group_year_stats, _records_cache_path
from training.text_analyzer_trader_data import analyzer_summary_to_text, write_jsonl
from training.text_step_analyzer_data import parse_hold_candidates
from training.yearly_stable_regime_policy import YearlyStableConfig


def _aggregate_outcomes(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    }


def classify_drift_risk(
    *,
    baseline: dict[str, Any],
    recent: dict[str, Any],
    min_recent_samples: int,
    high_min_mean_net: float,
    high_min_win_rate: float,
    high_max_mean_mae: float,
    medium_mean_drop: float,
) -> str:
    if int(recent.get("samples", 0)) < int(min_recent_samples):
        return "UNKNOWN"
    mean_net = float(recent.get("mean_net_return", 0.0))
    win_rate = float(recent.get("win_rate", 0.0))
    mean_mae = float(recent.get("mean_mae", 0.0))
    if mean_net < float(high_min_mean_net) or win_rate < float(high_min_win_rate) or mean_mae > float(high_max_mean_mae):
        return "HIGH"
    baseline_mean = float(baseline.get("mean_net_return", 0.0))
    if baseline_mean - mean_net >= float(medium_mean_drop):
        return "MEDIUM"
    return "LOW"


def _load_records_from_cache(cache_dir: str, *, split: str, start_date: str, end_date: str, stride_bars: int, cfg: CalibratedPolicyConfig) -> list[dict[str, Any]]:
    return json.loads(_records_cache_path(cache_dir, split=split, start_date=start_date, end_date=end_date, stride_bars=stride_bars, cfg=cfg).read_text())


def _selected_item(report: dict[str, Any], top_index: int) -> dict[str, Any]:
    rows = report.get("top_by_test_then_eval") or report.get("top") or []
    if not rows:
        raise ValueError("report has no selected rows")
    return rows[int(top_index)]


def _rule_action_key(rule: dict[str, Any]) -> str:
    action = rule["action"]
    return f"{action['side']}_{int(action['hold_bars'])}"


def _seed_history(train_records: list[dict[str, Any]], rules: dict[str, dict[str, Any]], *, recent_window: int) -> dict[str, deque[dict[str, Any]]]:
    history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=int(recent_window)))
    for row in sorted(train_records, key=lambda r: int(r.get("signal_pos", 0))):
        key = str(row["key"])
        rule = rules.get(key)
        if not rule:
            continue
        outcome = row["actions"].get(_rule_action_key(rule))
        if outcome:
            history[key].append({"signal_pos": int(row.get("signal_pos", 0)), **outcome})
    return history


def _update_completed_history(
    pending: list[dict[str, Any]],
    history: dict[str, deque[dict[str, Any]]],
    *,
    current_signal_pos: int,
) -> list[dict[str, Any]]:
    keep = []
    for item in pending:
        if int(item["complete_pos"]) < int(current_signal_pos):
            history[item["key"]].append(item["outcome"])
        else:
            keep.append(item)
    return keep


def run_export(
    *,
    report: str,
    records_cache_dir: str,
    analyzer_output: str,
    trader_output: str,
    summary_output: str,
    train_start: str = "",
    train_end: str = "",
    label_split: str = "eval",
    label_start: str = "",
    label_end: str = "",
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    top_index: int = 0,
    recent_window: int = 12,
    min_recent_samples: int = 6,
    high_min_mean_net: float = 0.0,
    high_min_win_rate: float = 0.45,
    high_max_mean_mae: float = 0.02,
    medium_mean_drop: float = 0.004,
) -> dict[str, Any]:
    payload = json.loads(Path(report).read_text())
    periods = payload.get("periods", {})
    train_start = train_start or periods["train"][0]
    train_end = train_end or periods["train"][1]
    label_period = periods.get(label_split)
    if not label_period and (not label_start or not label_end):
        raise ValueError("label period must be provided or present in report periods")
    label_start = label_start or label_period[0]
    label_end = label_end or label_period[1]
    item = _selected_item(payload, top_index)
    cfg_payload = item["config"]
    cfg = CalibratedPolicyConfig(**{k: v for k, v in cfg_payload.items() if k != "hold_candidates"}, hold_candidates=tuple(cfg_payload.get("hold_candidates") or parse_hold_candidates(hold_candidates)))
    stable = YearlyStableConfig(**item["stable_config"])
    train_records = _load_records_from_cache(records_cache_dir, split="train", start_date=train_start, end_date=train_end, stride_bars=stride_bars, cfg=cfg)
    label_records = _load_records_from_cache(records_cache_dir, split=label_split, start_date=label_start, end_date=label_end, stride_bars=stride_bars, cfg=cfg)
    train_records = _copy_with_keys(train_records, tuple(cfg.key_fields))
    label_records = _copy_with_keys(label_records, tuple(cfg.key_fields))
    rules = _fit_from_stats(_precompute_group_year_stats(train_records), cfg, stable)
    policy_book = format_policy_book(rules)
    baseline = {
        key: _aggregate_action([row for row in train_records if str(row["key"]) == key], _rule_action_key(rule))
        for key, rule in rules.items()
    }
    history = _seed_history(train_records, rules, recent_window=recent_window)
    pending: list[dict[str, Any]] = []
    next_available_pos = -1
    analyzer_rows: list[dict[str, Any]] = []
    trader_rows: list[dict[str, Any]] = []
    simulated_trades: list[dict[str, Any]] = []
    target_counts: dict[str, int] = {}
    drift_counts: dict[str, int] = {}
    for row in sorted(label_records, key=lambda r: int(r.get("signal_pos", 0))):
        signal_pos = int(row["signal_pos"])
        pending = _update_completed_history(pending, history, current_signal_pos=signal_pos)
        key = str(row["key"])
        rule = rules.get(key)
        recent = _aggregate_outcomes(list(history.get(key, [])))
        base = baseline.get(key, {"samples": 0})
        drift_risk = "NO_RULE" if rule is None else classify_drift_risk(
            baseline=base,
            recent=recent,
            min_recent_samples=min_recent_samples,
            high_min_mean_net=high_min_mean_net,
            high_min_win_rate=high_min_win_rate,
            high_max_mean_mae=high_max_mean_mae,
            medium_mean_drop=medium_mean_drop,
        )
        drift_counts[drift_risk] = drift_counts.get(drift_risk, 0) + 1
        summary = dict(row["summary"])
        summary["policy_context"] = {
            "policy_key": key,
            "train_action": rule["action"] if rule else None,
            "train_baseline": base,
            "rolling_recent": recent,
            "rolling_drift_risk": drift_risk,
            "recent_window": int(recent_window),
        }
        summary_text = analyzer_summary_to_text(summary)
        if signal_pos <= next_available_pos:
            target = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "policy_key": key, "drift_risk": drift_risk, "reason": "POSITION_OPEN_SKIP"}
        elif rule is None:
            target = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "policy_key": key, "drift_risk": drift_risk, "reason": "NO_CALIBRATED_EDGE"}
        elif drift_risk == "HIGH":
            target = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "policy_key": key, "drift_risk": drift_risk, "reason": "DRIFT_RISK_SKIP"}
        else:
            action = rule["action"]
            hold_bars = int(action["hold_bars"])
            target = {"gate": "TRADE", "side": str(action["side"]), "hold_bars": hold_bars, "policy_key": key, "drift_risk": drift_risk, "reason": "DRIFT_AWARE_CALIBRATED_EDGE"}
            outcome = row["actions"].get(_rule_action_key(rule))
            if outcome:
                simulated_trades.append({"date": row["date"], "signal_pos": signal_pos, "policy_key": key, "drift_risk": drift_risk, **outcome})
            next_available_pos = signal_pos + hold_bars
        target_counts[str(target["reason"])] = target_counts.get(str(target["reason"]), 0) + 1
        if rule is not None:
            outcome = row["actions"].get(_rule_action_key(rule))
            if outcome:
                pending.append({"key": key, "complete_pos": signal_pos + int(rule["action"]["hold_bars"]), "outcome": {"signal_pos": signal_pos, **outcome}})
        analyzer_rows.append(
            {
                "task": "drift_aware_policy_analyzer",
                "date": row["date"],
                "signal_pos": signal_pos,
                "prompt": "Produce a past-only analyzer summary plus rolling policy drift risk for the current state.",
                "target": summary_text,
                "policy_key": key,
                "rolling_drift_risk": drift_risk,
                "leakage_guard": {"current_label_future_not_used_for_drift": True, "rolling_history_uses_completed_prior_outcomes_only": True},
            }
        )
        trader_rows.append(
            {
                "task": "drift_aware_policy_trader",
                "date": row["date"],
                "signal_pos": signal_pos,
                "prompt": build_policy_trader_input(summary_text, hold_candidates=cfg.hold_candidates, entry_delay_bars=cfg.entry_delay_bars, current_policy_key=key, policy_book=policy_book),
                "target": json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                "trade_gate_label": target["gate"],
                "trade_side_label": target["side"],
                "hold_bars_label": int(target["hold_bars"]),
                "policy_key": key,
                "rolling_drift_risk": drift_risk,
                "leakage_guard": {"rules_fit_on_train_only": True, "drift_state_is_past_completed_outcomes_only": True},
            }
        )
    write_jsonl(analyzer_output, analyzer_rows)
    write_jsonl(trader_output, trader_rows)
    metrics = _metrics_from_trades(simulated_trades, records_count=len(label_records), include_intratrade_mdd=True)
    years = max(1e-9, (pd.to_datetime(label_end) - pd.to_datetime(label_start)).days / 365.25)
    compounded = float(metrics.get("compounded_return", 0.0) or 0.0)
    mdd = float(metrics.get("strict_mdd_proxy", 0.0) or 0.0)
    cagr = (1.0 + compounded) ** (1.0 / years) - 1.0 if compounded > -0.999 else 0.0
    metrics.update({"years": years, "cagr_proxy": cagr, "cagr_to_mdd_proxy": cagr / mdd if mdd > 0 else (float("inf") if cagr > 0 else 0.0), "non_overlapping": True})
    summary = {
        "source_report": report,
        "periods": {"train": [train_start, train_end], "label": [label_start, label_end]},
        "records": {"train": len(train_records), "label": len(label_records), "analyzer": len(analyzer_rows), "trader": len(trader_rows)},
        "rules_count": len(rules),
        "key_fields": list(cfg.key_fields),
        "target_counts": target_counts,
        "drift_counts": drift_counts,
        "simulated_label_policy_metrics": metrics,
        "drift_config": {"recent_window": recent_window, "min_recent_samples": min_recent_samples, "high_min_mean_net": high_min_mean_net, "high_min_win_rate": high_min_win_rate, "high_max_mean_mae": high_max_mean_mae, "medium_mean_drop": medium_mean_drop},
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export drift-aware analyzer/trader SFT rows")
    for arg in ["report", "records-cache-dir", "analyzer-output", "trader-output", "summary-output"]:
        p.add_argument("--" + arg, required=True)
    p.add_argument("--train-start", default="")
    p.add_argument("--train-end", default="")
    p.add_argument("--label-split", default="eval")
    p.add_argument("--label-start", default="")
    p.add_argument("--label-end", default="")
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--hold-candidates", default="48,96,144,288")
    p.add_argument("--top-index", type=int, default=0)
    p.add_argument("--recent-window", type=int, default=12)
    p.add_argument("--min-recent-samples", type=int, default=6)
    p.add_argument("--high-min-mean-net", type=float, default=0.0)
    p.add_argument("--high-min-win-rate", type=float, default=0.45)
    p.add_argument("--high-max-mean-mae", type=float, default=0.02)
    p.add_argument("--medium-mean-drop", type=float, default=0.004)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_export(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
