"""Build and evaluate sparse-setup event-action records for LLM/RL policies.

This stage bridges the current sparse alpha search and the LLM/RL direction: a
sparse setup emits an event from past-only thresholds, then the policy chooses
NO_TRADE or an executable side/hold action.  Future OHLC outcomes are stored only
as labels/audit targets; walk-forward evaluation fits action rules from previous
folds only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.eval_calibrated_policy_model import _metrics_from_actions
from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome
from training.sparse_setup_ensemble_audit import EnsembleCfg, _candidate_events, _candidate_key, _load_market
from training.wave_feature_ridge_policy import build_wave_feature_frame


@dataclass(frozen=True)
class SparseActionPolicyCfg:
    sparse_report: str
    market_csv: str
    output_jsonl: str
    output_report: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    include_external_components: bool = False
    window_size: int = 144
    candidate_limit: int = 16
    hold_candidates: tuple[int, ...] = (24, 36, 72, 144)
    include_opposite_side: bool = True
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    mae_penalty: float = 1.0
    min_history_folds: int = 1
    min_action_samples: int = 8
    min_action_mean_net: float = 0.0
    min_action_mean_utility: float = -0.002
    min_action_win_rate: float = 0.50
    max_action_mean_mae: float = 0.03
    cold_start_source_action: bool = True
    fallback_source_action: str = "cold_start"  # never | cold_start | always_when_no_rule
    rule_key_mode: str = "candidate"  # candidate | feature_sides | source_side


def _parse_holds(csv: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(csv, tuple):
        return tuple(int(x) for x in csv)
    return tuple(int(x.strip()) for x in str(csv).split(",") if x.strip())


def _features(market: pd.DataFrame, cfg: SparseActionPolicyCfg) -> pd.DataFrame:
    return pd.concat(
        [
            build_market_feature_frame(market, window_size=int(cfg.window_size)).add_prefix("mkt__"),
            build_wave_feature_frame(market, window=int(cfg.window_size)).add_prefix("wave__"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0).loc[:, lambda df: ~df.columns.duplicated(keep="last")]


def _load_inputs(cfg: SparseActionPolicyCfg) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series, list[dict[str, Any]]]:
    sparse = json.loads(Path(cfg.sparse_report).read_text())
    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(
            market,
            wave_trading_root=cfg.wave_trading_root,
            tolerance=cfg.external_tolerance,
            include_forex_components=bool(cfg.include_external_components),
        )
    feats = _features(market, cfg)
    dates = pd.to_datetime(market["date"])
    candidates = [dict(c, _candidate_index=i) for i, c in enumerate(sparse.get("top_strict", [])[: int(cfg.candidate_limit)])]
    return sparse, market, feats, dates, candidates


def _ensemble_cfg(cfg: SparseActionPolicyCfg) -> EnsembleCfg:
    return EnsembleCfg(
        sparse_report=cfg.sparse_report,
        market_csv=cfg.market_csv,
        output=cfg.output_report,
        wave_trading_root=cfg.wave_trading_root,
        external_tolerance=cfg.external_tolerance,
        window_size=cfg.window_size,
        candidate_limit=cfg.candidate_limit,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        entry_delay_bars=cfg.entry_delay_bars,
        execution_horizon_bars=0,
    )


def _fold_order(sparse: dict[str, Any]) -> list[str]:
    return [str(f["name"]) for f in sorted(sparse.get("folds", []), key=lambda x: str(x["eval_start"]))]


def _source_side_label(ev: dict[str, Any]) -> str:
    return "LONG" if int(ev.get("side", 0)) > 0 else "SHORT"


def _opposite(side: str) -> str:
    return "SHORT" if side == "LONG" else "LONG"


def _bucket(value: float, edges: tuple[float, float, float], labels: tuple[str, str, str, str]) -> str:
    v = float(value)
    if v < edges[0]:
        return labels[0]
    if v < edges[1]:
        return labels[1]
    if v < edges[2]:
        return labels[2]
    return labels[3]


def _candidate_tokens(cand: dict[str, Any], ev: dict[str, Any], feats: pd.DataFrame) -> list[str]:
    tokens: list[str] = [
        f"candidate={_candidate_key(cand)}",
        f"source_side={_source_side_label(ev)}",
        f"source_horizon={int(ev.get('source_horizon', ev.get('horizon', 0)))}",
        f"event_horizon={int(ev.get('horizon', 0))}",
        f"prior_mean={_bucket(float(ev.get('prior_mean_ret', 0.0) or 0.0), (-0.001, 0.0, 0.001), ('neg','flat','pos','strong'))}",
        f"prior_std={_bucket(float(ev.get('prior_std_ret', 0.0) or 0.0), (0.002, 0.006, 0.012), ('low','mid','high','extreme'))}",
        f"prior_n={_bucket(float(ev.get('prior_n', 0) or 0), (50, 200, 1000), ('tiny','small','mid','large'))}",
    ]
    for f in cand.get("features", []):
        name = str(f["name"])
        side = str(f["side"])
        val = float(feats.iloc[int(ev["signal_pos"])].get(name, 0.0))
        tokens.append(f"predicate={name}:{side}")
        tokens.append(f"{name}={_bucket(val, (-1.0, 0.0, 1.0), ('very_low','low','high','very_high'))}")
    return tokens


def _action_outcomes(market: pd.DataFrame, signal_pos: int, source_side: str, cfg: SparseActionPolicyCfg) -> dict[str, dict[str, Any]]:
    sides = [source_side]
    if bool(cfg.include_opposite_side):
        sides.append(_opposite(source_side))
    actions: dict[str, dict[str, Any]] = {}
    for side in sides:
        for hold in cfg.hold_candidates:
            pcfg = PathOutcomeConfig(
                hold_bars=int(hold),
                entry_delay_bars=int(cfg.entry_delay_bars),
                fee_rate=float(cfg.fee_rate),
                slippage_rate=float(cfg.slippage_rate),
                leverage=float(cfg.leverage),
                mae_penalty=float(cfg.mae_penalty),
            )
            out = compute_trade_path_outcome(market, int(signal_pos), side, pcfg)
            if out is None:
                continue
            actions[f"{side}_{int(hold)}"] = {
                "side": side,
                "hold_bars": int(hold),
                "net_return": float(out.net_return),
                "mae": float(out.mae),
                "mfe": float(out.mfe),
                "utility": float(out.utility),
                "entry_pos": int(out.entry_pos),
                "exit_pos": int(out.exit_pos),
            }
    return actions


def build_sparse_action_records(cfg: SparseActionPolicyCfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sparse, market, feats, dates, candidates = _load_inputs(cfg)
    base_cfg = _ensemble_cfg(cfg)
    records: list[dict[str, Any]] = []
    per_candidate_counts: Counter[str] = Counter()
    for cand in candidates:
        events = _candidate_events(cand=cand, report=sparse, dates=dates, features=feats, market=market, cfg=base_cfg)
        for ev in events:
            actions = _action_outcomes(market, int(ev["signal_pos"]), _source_side_label(ev), cfg)
            if not actions:
                continue
            key = str(ev["candidate_key"])
            best_key, best = max(actions.items(), key=lambda kv: (float(kv[1]["utility"]), float(kv[1]["net_return"])))
            records.append(
                {
                    "task": "sparse_setup_event_action_policy",
                    "date": str(ev["date"]),
                    "signal_pos": int(ev["signal_pos"]),
                    "fold": str(ev["fold"]),
                    "key": key,
                    "candidate_index": int(ev.get("candidate_index", cand.get("_candidate_index", -1))),
                    "state_tokens": _candidate_tokens(cand, ev, feats),
                    "source_action": {"gate": "TRADE", "side": _source_side_label(ev), "hold_bars": int(ev.get("horizon", cand.get("horizon", 0)))},
                    "actions": actions,
                    "target_action_audit": {"action_key": best_key, **best},
                    "reward_audit": {
                        k: {"net_return": v["net_return"], "mae": v["mae"], "mfe": v["mfe"], "utility": v["utility"]}
                        for k, v in actions.items()
                    },
                    "leakage_guard": {
                        "event_thresholds_and_side_fit_before_fold_start": True,
                        "state_tokens_use_current_or_past_features_only": True,
                        "actions_use_future_ohlc_for_training_label_only": True,
                    },
                }
            )
            per_candidate_counts[key] += 1
    records.sort(key=lambda r: (int(r["signal_pos"]), str(r["key"])))
    summary = {
        "records": len(records),
        "folds": dict(Counter(str(r["fold"]) for r in records)),
        "candidate_count": len(candidates),
        "candidate_event_counts": dict(per_candidate_counts.most_common(20)),
        "action_keys": dict(Counter(k for r in records for k in r["actions"])),
        "period": {"start": records[0]["date"] if records else None, "end": records[-1]["date"] if records else None},
    }
    return records, summary


def _aggregate_action(rows: list[dict[str, Any]], action_key: str) -> dict[str, Any]:
    vals = [r["actions"][action_key] for r in rows if action_key in r["actions"]]
    if not vals:
        return {"samples": 0}
    nets = [float(v["net_return"]) for v in vals]
    maes = [float(v["mae"]) for v in vals]
    utils = [float(v["utility"]) for v in vals]
    return {
        "samples": len(vals),
        "side": str(vals[0]["side"]),
        "hold_bars": int(vals[0]["hold_bars"]),
        "mean_net_return": float(np.mean(nets)),
        "mean_mae": float(np.mean(maes)),
        "mean_utility": float(np.mean(utils)),
        "win_rate": float(np.mean([x > 0.0 for x in nets])),
    }


def _policy_key(row: dict[str, Any], cfg: SparseActionPolicyCfg) -> str:
    mode = str(cfg.rule_key_mode)
    key = str(row["key"])
    if mode == "candidate":
        return key
    if mode == "feature_sides":
        return key.split("|", 2)[-1] if "|" in key else key
    if mode == "source_side":
        return "source_side=" + str(row.get("source_action", {}).get("side", "NONE"))
    raise ValueError("rule_key_mode must be one of {'candidate','feature_sides','source_side'}")


def fit_action_rules(train_records: list[dict[str, Any]], cfg: SparseActionPolicyCfg) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_records:
        groups[_policy_key(row, cfg)].append(row)
    rules: dict[str, dict[str, Any]] = {}
    for key, rows in groups.items():
        action_keys = sorted({a for row in rows for a in row["actions"]})
        aggs = [_aggregate_action(rows, a) | {"action_key": a} for a in action_keys]
        qualified = [
            a
            for a in aggs
            if int(a.get("samples", 0)) >= int(cfg.min_action_samples)
            and float(a.get("mean_net_return", 0.0)) >= float(cfg.min_action_mean_net)
            and float(a.get("mean_utility", 0.0)) >= float(cfg.min_action_mean_utility)
            and float(a.get("win_rate", 0.0)) >= float(cfg.min_action_win_rate)
            and float(a.get("mean_mae", 1.0)) <= float(cfg.max_action_mean_mae)
        ]
        if not qualified:
            continue
        best = max(qualified, key=lambda a: (float(a["mean_utility"]), float(a["mean_net_return"]), float(a["win_rate"]), int(a["samples"])))
        rules[key] = {"key": key, "train_samples_in_group": len(rows), "action": best, "qualified_actions": qualified[:8]}
    return rules


def _source_or_no_trade(row: dict[str, Any], cfg: SparseActionPolicyCfg) -> dict[str, Any]:
    if not bool(cfg.cold_start_source_action):
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "reason": "COLD_START_NO_HISTORY"}
    src = row.get("source_action", {})
    side = str(src.get("side", "NONE"))
    hold = int(src.get("hold_bars", 0) or 0)
    if f"{side}_{hold}" in row.get("actions", {}):
        return {"gate": "TRADE", "side": side, "hold_bars": hold, "reason": "COLD_START_SOURCE_ACTION"}
    return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "reason": "COLD_START_SOURCE_ACTION_UNAVAILABLE"}


def _actions_for_records(records: list[dict[str, Any]], rules: dict[str, dict[str, Any]], cfg: SparseActionPolicyCfg, *, allow_cold_start: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in records:
        rule = rules.get(_policy_key(row, cfg))
        if rule:
            a = rule["action"]
            out.append({"gate": "TRADE", "side": str(a["side"]), "hold_bars": int(a["hold_bars"]), "reason": "HISTORY_ACTION_RULE"})
        elif str(cfg.fallback_source_action) == "always_when_no_rule" or (allow_cold_start and str(cfg.fallback_source_action) == "cold_start"):
            out.append(_source_or_no_trade(row, cfg))
        else:
            reason = "COLD_START_NO_HISTORY" if allow_cold_start else "NO_HISTORY_RULE"
            out.append({"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "reason": reason})
    return out


def _add_annualized(metrics: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(metrics)
    if not records:
        return out
    start = pd.Timestamp(records[0]["date"])
    end = pd.Timestamp(records[-1]["date"])
    years = max(1.0 / 365.25, float((end - start).days) / 365.25)
    gross = 1.0 + float(out.get("compounded_return", 0.0) or 0.0)
    cagr = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = float(out.get("strict_mdd_proxy", 0.0) or 0.0) * 100.0
    out.update({"period": {"start": str(start), "end": str(end), "years": years}, "cagr_pct": cagr, "strict_mdd_pct": mdd_pct, "cagr_to_strict_mdd": cagr / mdd_pct if mdd_pct > 1e-12 else float("inf")})
    return out


def walkforward_action_eval(records: list[dict[str, Any]], sparse: dict[str, Any], cfg: SparseActionPolicyCfg) -> dict[str, Any]:
    folds = _fold_order(sparse)
    by_fold: dict[str, list[dict[str, Any]]] = {f: [r for r in records if str(r["fold"]) == f] for f in folds}
    fold_reports: list[dict[str, Any]] = []
    final_records: list[dict[str, Any]] = []
    final_actions: list[dict[str, Any]] = []
    for idx, fold in enumerate(folds):
        history_folds = folds[:idx]
        train = [r for hf in history_folds for r in by_fold.get(hf, [])]
        current = by_fold.get(fold, [])
        rules = fit_action_rules(train, cfg) if len(history_folds) >= int(cfg.min_history_folds) else {}
        actions = _actions_for_records(current, rules, cfg, allow_cold_start=len(history_folds) < int(cfg.min_history_folds))
        metrics = _add_annualized(_metrics_from_actions(current, actions), current)
        fold_reports.append(
            {
                "fold": fold,
                "history_folds": history_folds,
                "records": len(current),
                "rules_count": len(rules),
                "metrics": metrics,
                "action_reasons": dict(Counter(str(a.get("reason", "")) for a in actions)),
                "rules_preview": list(rules.values())[:8],
            }
        )
        final_records.extend(current)
        final_actions.extend(actions)
    return {
        "folds": fold_reports,
        "final_metrics": _add_annualized(_metrics_from_actions(final_records, final_actions), final_records),
        "leakage_guard": {
            "action_rules_fit_on_previous_folds_only": True,
            "current_fold_rewards_not_used_for_current_action_selection": True,
            "fallback_source_action": str(cfg.fallback_source_action),
            "source_setup_action_is_fit_before_each_fold_start": True,
            "record_labels_use_future_ohlc_but_only_after_signal_for_training_audit": True,
        },
    }


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(cfg: SparseActionPolicyCfg) -> dict[str, Any]:
    sparse, _, _, _, _ = _load_inputs(cfg)
    records, dataset_summary = build_sparse_action_records(cfg)
    write_jsonl(cfg.output_jsonl, records)
    wf = walkforward_action_eval(records, sparse, cfg)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg) | {"hold_candidates": list(cfg.hold_candidates)},
        "dataset": dataset_summary,
        "walkforward": wf,
        "source_sparse_report": cfg.sparse_report,
    }
    Path(cfg.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build sparse setup event-action data and evaluate past-only action rules")
    p.add_argument("--sparse-report", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=SparseActionPolicyCfg.external_tolerance)
    p.add_argument("--include-external-components", action="store_true", default=SparseActionPolicyCfg.include_external_components)
    p.add_argument("--window-size", type=int, default=SparseActionPolicyCfg.window_size)
    p.add_argument("--candidate-limit", type=int, default=SparseActionPolicyCfg.candidate_limit)
    p.add_argument("--hold-candidates", default="24,36,72,144")
    p.add_argument("--include-opposite-side", action=argparse.BooleanOptionalAction, default=SparseActionPolicyCfg.include_opposite_side)
    p.add_argument("--leverage", type=float, default=SparseActionPolicyCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=SparseActionPolicyCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=SparseActionPolicyCfg.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=SparseActionPolicyCfg.entry_delay_bars)
    p.add_argument("--mae-penalty", type=float, default=SparseActionPolicyCfg.mae_penalty)
    p.add_argument("--min-history-folds", type=int, default=SparseActionPolicyCfg.min_history_folds)
    p.add_argument("--min-action-samples", type=int, default=SparseActionPolicyCfg.min_action_samples)
    p.add_argument("--min-action-mean-net", type=float, default=SparseActionPolicyCfg.min_action_mean_net)
    p.add_argument("--min-action-mean-utility", type=float, default=SparseActionPolicyCfg.min_action_mean_utility)
    p.add_argument("--min-action-win-rate", type=float, default=SparseActionPolicyCfg.min_action_win_rate)
    p.add_argument("--max-action-mean-mae", type=float, default=SparseActionPolicyCfg.max_action_mean_mae)
    p.add_argument("--cold-start-source-action", action=argparse.BooleanOptionalAction, default=SparseActionPolicyCfg.cold_start_source_action)
    p.add_argument("--fallback-source-action", choices=["never", "cold_start", "always_when_no_rule"], default=SparseActionPolicyCfg.fallback_source_action)
    p.add_argument("--rule-key-mode", choices=["candidate", "feature_sides", "source_side"], default=SparseActionPolicyCfg.rule_key_mode)
    ns = p.parse_args()
    ns.hold_candidates = _parse_holds(ns.hold_candidates)
    return ns


def main() -> None:
    rep = run(SparseActionPolicyCfg(**vars(parse_args())))
    print(json.dumps({"dataset": rep["dataset"], "final_metrics": rep["walkforward"]["final_metrics"], "folds": [{"fold": f["fold"], "metrics": f["metrics"], "reasons": f["action_reasons"]} for f in rep["walkforward"]["folds"]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
