"""Past-only walk-forward selector for sparse setup ensembles.

This is stricter than ``sparse_setup_ensemble_audit.py``.  The audit greedily
selects an ensemble on the full replay period, which is useful for family
research but optimistic for deployment.  This script reuses the same fold-local
candidate event construction, then selects candidates for each fold only from
previous fold outcomes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.sparse_setup_ensemble_audit import (
    EnsembleCfg,
    _candidate_events,
    _candidate_key,
    _load_market,
    _score,
    _simulate_events,
)
from training.wave_feature_ridge_policy import build_wave_feature_frame


@dataclass(frozen=True)
class WalkForwardCfg:
    sparse_report: str
    market_csv: str
    output: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    window_size: int = 144
    candidate_limit: int = 20
    max_ensemble_size: int = 6
    leverage: float = 1.2
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    cooldown_bars: int = 0
    max_same_bar_signals: int = 1
    min_trades: int = 30
    trade_stop_loss_pct: float = 6.0
    trade_take_profit_pct: float = 6.0
    atr_trailing_stop_mult: float = 0.0
    atr_period: int = 45
    rolling_window_trades: int = 0
    rolling_loss_stop_pct: float = 0.0
    pause_bars: int = 288
    min_recent_fold_trades: int = 0
    min_active_folds: int = 0
    setup_sizing: str = "prior_sharpe"
    min_position_scale: float = 0.35
    max_position_scale: float = 1.0
    execution_horizon_bars: int = 288
    min_history_folds: int = 1
    seed_candidates: str = "top_prior"  # top_prior | all_when_cold
    include_external_components: bool = False


def _ensemble_cfg(cfg: WalkForwardCfg) -> EnsembleCfg:
    data = asdict(cfg)
    data.pop("min_history_folds")
    data.pop("seed_candidates")
    data.pop("include_external_components")
    return EnsembleCfg(**data)


def _features(market: pd.DataFrame, cfg: WalkForwardCfg) -> pd.DataFrame:
    return pd.concat(
        [
            build_market_feature_frame(market, window_size=int(cfg.window_size)).add_prefix("mkt__"),
            build_wave_feature_frame(market, window=int(cfg.window_size)).add_prefix("wave__"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0).loc[:, lambda df: ~df.columns.duplicated(keep="last")]


def _fold_order(sparse: dict[str, Any]) -> list[str]:
    return [str(f["name"]) for f in sorted(sparse["folds"], key=lambda x: str(x["eval_start"]))]


def _filter_events(events: list[dict[str, Any]], folds: set[str]) -> list[dict[str, Any]]:
    return [e for e in events if str(e.get("fold")) in folds]


def _candidate_prior_score(events: list[dict[str, Any]]) -> float:
    if not events:
        return -1e9
    vals = []
    for e in events:
        mean = float(e.get("prior_mean_ret", 0.0) or 0.0)
        std = float(e.get("prior_std_ret", 0.0) or 0.0)
        n = int(e.get("prior_n", 0) or 0)
        vals.append((mean / std if std > 1e-12 else mean) * min(1.0, n / 500.0))
    return float(np.nanmean(vals)) if vals else -1e9


def _select_for_history(*, candidate_ids: list[int], event_cache: dict[int, list[dict[str, Any]]], history_folds: list[str], dates: pd.Series, market: pd.DataFrame, cfg: EnsembleCfg) -> tuple[list[int], list[dict[str, Any]]]:
    history_set = set(history_folds)
    if len(history_folds) <= 0:
        ranked = sorted(candidate_ids, key=lambda i: _candidate_prior_score(event_cache[i]), reverse=True)
        return ranked[: max(1, int(cfg.max_ensemble_size))], []

    scored = []
    for cid in candidate_ids:
        evs = _filter_events(event_cache[cid], history_set)
        res = _simulate_events(evs, dates=dates, market=market, cfg=cfg)
        scored.append({"candidate_index": cid, "score": _score(res, cfg), "result": {k: v for k, v in res.items() if k != "executed"}})
    scored.sort(key=lambda r: float(r["score"]), reverse=True)

    selected: list[int] = []
    current: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = []
    for _ in range(int(cfg.max_ensemble_size)):
        best = None
        for row in scored:
            cid = int(row["candidate_index"])
            if cid in selected:
                continue
            trial_events: list[dict[str, Any]] = []
            for tid in selected + [cid]:
                trial_events.extend(_filter_events(event_cache[tid], history_set))
            trial_events.sort(key=lambda e: (int(e["signal_pos"]), str(e["candidate_key"])))
            res = _simulate_events(trial_events, dates=dates, market=market, cfg=cfg)
            sc = _score(res, cfg)
            if best is None or sc > best[0]:
                best = (sc, cid, res)
        if best is None:
            break
        if current is not None and best[0] <= _score(current, cfg) + 1e-9:
            break
        selected.append(best[1])
        current = best[2]
        steps.append({"added_candidate_index": best[1], "history_score": best[0], "history_result": {k: v for k, v in best[2].items() if k != "executed"}})
    return selected, steps


def run(cfg: WalkForwardCfg) -> dict[str, Any]:
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
    base_cfg = _ensemble_cfg(cfg)
    candidates = [dict(c, _candidate_index=i) for i, c in enumerate(sparse.get("top_strict", [])[: int(cfg.candidate_limit)])]
    candidate_ids = [int(c["_candidate_index"]) for c in candidates]
    event_cache = {int(c["_candidate_index"]): _candidate_events(cand=c, report=sparse, dates=dates, features=feats, market=market, cfg=base_cfg) for c in candidates}
    fold_names = _fold_order(sparse)

    fold_rows: list[dict[str, Any]] = []
    final_events: list[dict[str, Any]] = []
    for pos, fold in enumerate(fold_names):
        history = fold_names[:pos]
        if len(history) < int(cfg.min_history_folds):
            if str(cfg.seed_candidates) == "all_when_cold":
                selected = candidate_ids[: max(1, int(cfg.max_ensemble_size))]
                steps = []
            else:
                selected, steps = _select_for_history(candidate_ids=candidate_ids, event_cache=event_cache, history_folds=[], dates=dates, market=market, cfg=base_cfg)
        else:
            selected, steps = _select_for_history(candidate_ids=candidate_ids, event_cache=event_cache, history_folds=history, dates=dates, market=market, cfg=base_cfg)
        fold_events: list[dict[str, Any]] = []
        for cid in selected:
            fold_events.extend(_filter_events(event_cache[cid], {fold}))
        fold_events.sort(key=lambda e: (int(e["signal_pos"]), str(e["candidate_key"])))
        final_events.extend(fold_events)
        fold_res = _simulate_events(fold_events, dates=dates, market=market, cfg=base_cfg)
        fold_rows.append({"fold": fold, "history_folds": history, "selected": selected, "steps": steps, "fold_result": {k: v for k, v in fold_res.items() if k != "executed"}})

    final_events.sort(key=lambda e: (int(e["signal_pos"]), str(e["candidate_key"])))
    final = _simulate_events(final_events, dates=dates, market=market, cfg=base_cfg)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "folds": fold_rows,
        "final": {k: v for k, v in final.items() if k != "executed"},
        "candidate_keys": {str(int(c["_candidate_index"])): _candidate_key(c) for c in candidates},
        "leakage_guard": {
            "candidate_thresholds_and_side_fit_before_each_eval_fold": True,
            "selector_uses_only_previous_fold_outcomes": True,
            "current_fold_metrics_not_used_for_current_selection": True,
            "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled",
            "known_limit": "candidate pool is inherited from sparse report; use this to audit ensemble-selection leakage, not final production discovery",
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Past-only walk-forward sparse setup ensemble selector")
    p.add_argument("--sparse-report", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=WalkForwardCfg.external_tolerance)
    p.add_argument("--window-size", type=int, default=WalkForwardCfg.window_size)
    p.add_argument("--candidate-limit", type=int, default=WalkForwardCfg.candidate_limit)
    p.add_argument("--max-ensemble-size", type=int, default=WalkForwardCfg.max_ensemble_size)
    p.add_argument("--leverage", type=float, default=WalkForwardCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=WalkForwardCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=WalkForwardCfg.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=WalkForwardCfg.entry_delay_bars)
    p.add_argument("--cooldown-bars", type=int, default=WalkForwardCfg.cooldown_bars)
    p.add_argument("--max-same-bar-signals", type=int, default=WalkForwardCfg.max_same_bar_signals)
    p.add_argument("--min-trades", type=int, default=WalkForwardCfg.min_trades)
    p.add_argument("--trade-stop-loss-pct", type=float, default=WalkForwardCfg.trade_stop_loss_pct)
    p.add_argument("--trade-take-profit-pct", type=float, default=WalkForwardCfg.trade_take_profit_pct)
    p.add_argument("--atr-trailing-stop-mult", type=float, default=WalkForwardCfg.atr_trailing_stop_mult)
    p.add_argument("--atr-period", type=int, default=WalkForwardCfg.atr_period)
    p.add_argument("--rolling-window-trades", type=int, default=WalkForwardCfg.rolling_window_trades)
    p.add_argument("--rolling-loss-stop-pct", type=float, default=WalkForwardCfg.rolling_loss_stop_pct)
    p.add_argument("--pause-bars", type=int, default=WalkForwardCfg.pause_bars)
    p.add_argument("--min-recent-fold-trades", type=int, default=WalkForwardCfg.min_recent_fold_trades)
    p.add_argument("--min-active-folds", type=int, default=WalkForwardCfg.min_active_folds)
    p.add_argument("--setup-sizing", choices=["fixed", "prior_sharpe"], default=WalkForwardCfg.setup_sizing)
    p.add_argument("--min-position-scale", type=float, default=WalkForwardCfg.min_position_scale)
    p.add_argument("--max-position-scale", type=float, default=WalkForwardCfg.max_position_scale)
    p.add_argument("--execution-horizon-bars", type=int, default=WalkForwardCfg.execution_horizon_bars)
    p.add_argument("--min-history-folds", type=int, default=WalkForwardCfg.min_history_folds)
    p.add_argument("--seed-candidates", choices=["top_prior", "all_when_cold"], default=WalkForwardCfg.seed_candidates)
    p.add_argument("--include-external-components", action="store_true", default=WalkForwardCfg.include_external_components)
    return p.parse_args()


def main() -> None:
    rep = run(WalkForwardCfg(**vars(parse_args())))
    s = rep["final"]["sim"]
    print(json.dumps({"final": s, "trade_stats": rep["final"].get("trade_stats", {}), "fold_selected": [{"fold": f["fold"], "selected": f["selected"], "trades": f["fold_result"]["sim"].get("trade_entries", 0)} for f in rep["folds"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
