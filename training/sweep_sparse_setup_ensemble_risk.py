"""In-process risk overlay sweep for sparse setup ensembles."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
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


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split(",") if x.strip()]


def _compact_result(res: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in res.items() if k != "executed"}


def _run_greedy(*, individual: list[dict[str, Any]], event_cache: dict[int, list[dict[str, Any]]], dates: pd.Series, market: pd.DataFrame, cfg: EnsembleCfg) -> dict[str, Any]:
    ranked = []
    for row in individual:
        idx = int(row["candidate_index"])
        res = _simulate_events(event_cache[idx], dates=dates, market=market, cfg=cfg)
        ranked.append({**row, "result": _compact_result(res), "score": _score(res, cfg)})
    ranked.sort(key=lambda r: float(r["score"]), reverse=True)
    selected: list[int] = []
    current = None
    steps = []
    for _ in range(int(cfg.max_ensemble_size)):
        best = None
        for row in ranked:
            idx = int(row["candidate_index"])
            if idx in selected:
                continue
            events = []
            for sid in selected + [idx]:
                events.extend(event_cache[sid])
            events.sort(key=lambda e: (int(e["signal_pos"]), str(e["candidate_key"])))
            res = _simulate_events(events, dates=dates, market=market, cfg=cfg)
            sc = _score(res, cfg)
            if best is None or sc > best[0]:
                best = (sc, idx, res)
        if best is None:
            break
        if current is not None and best[0] <= _score(current, cfg) + 1e-9:
            break
        selected.append(best[1])
        current = best[2]
        steps.append({"added_candidate_index": best[1], "score": best[0], "result": _compact_result(best[2])})
    return {"selected_candidate_indices": selected, "steps": steps, "final": None if current is None else _compact_result(current), "individual_top": ranked[:10]}


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    base_cfg = EnsembleCfg(
        sparse_report=args.sparse_report,
        market_csv=args.market_csv,
        output=args.output,
        wave_trading_root=args.wave_trading_root,
        candidate_limit=args.candidate_limit,
        max_ensemble_size=args.max_ensemble_size,
        min_trades=args.min_trades,
        min_recent_fold_trades=args.min_recent_fold_trades,
        min_active_folds=args.min_active_folds,
        setup_sizing=args.setup_sizing,
        min_position_scale=args.min_position_scale,
        max_position_scale=args.max_position_scale,
    )
    sparse = json.loads(Path(args.sparse_report).read_text())
    market = _load_market(args.market_csv)
    if args.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=args.wave_trading_root, tolerance=base_cfg.external_tolerance)
    features = pd.concat(
        [
            build_market_feature_frame(market, window_size=int(base_cfg.window_size)).add_prefix("mkt__"),
            build_wave_feature_frame(market, window=int(base_cfg.window_size)).add_prefix("wave__"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    dates = pd.to_datetime(market["date"])
    candidates = [dict(c, _candidate_index=i) for i, c in enumerate(sparse.get("top_strict", [])[: int(args.candidate_limit)])]
    event_cache = {int(c["_candidate_index"]): _candidate_events(cand=c, report=sparse, dates=dates, features=features, market=market, cfg=base_cfg) for c in candidates}
    individual_meta = [{"candidate_index": int(c["_candidate_index"]), "key": _candidate_key(c), "candidate": c} for c in candidates]

    rows = []
    for lev in _parse_floats(args.leverages):
        for stop in _parse_floats(args.stop_losses):
            for take in _parse_floats(args.take_profits):
                for atr in _parse_floats(args.atr_mults):
                    if stop > 0 and atr > 0:
                        continue
                    for rolling_window in _parse_ints(args.rolling_windows):
                        for rolling_loss in _parse_floats(args.rolling_losses):
                            cfg = replace(
                                base_cfg,
                                leverage=float(lev),
                                trade_stop_loss_pct=float(stop),
                                trade_take_profit_pct=float(take),
                                atr_trailing_stop_mult=float(atr),
                                rolling_window_trades=int(rolling_window),
                                rolling_loss_stop_pct=float(rolling_loss),
                                pause_bars=int(args.pause_bars),
                                min_recent_fold_trades=int(args.min_recent_fold_trades),
                                min_active_folds=int(args.min_active_folds),
                                setup_sizing=str(args.setup_sizing),
                                min_position_scale=float(args.min_position_scale),
                                max_position_scale=float(args.max_position_scale),
                            )
                            greedy = _run_greedy(individual=individual_meta, event_cache=event_cache, dates=dates, market=market, cfg=cfg)
                            final = greedy["final"] or {"sim": {"cagr_pct": -100, "strict_mdd_pct": 100, "cagr_to_strict_mdd": -1, "trade_entries": 0}, "trade_stats": {}}
                            row = {"params": {"leverage": lev, "stop_loss": stop, "take_profit": take, "atr_mult": atr, "rolling_window": rolling_window, "rolling_loss": rolling_loss}, "selected": greedy["selected_candidate_indices"], "final": final, "score": _score(final, cfg)}
                            rows.append(row)
                            s = final["sim"]
                            print(json.dumps({"params": row["params"], "selected": row["selected"], "cagr": s["cagr_pct"], "mdd": s["strict_mdd_pct"], "ratio": s["cagr_to_strict_mdd"], "trades": s["trade_entries"], "p": final.get("trade_stats", {}).get("p_value_mean_ret_approx")}, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: (float(r["final"]["sim"].get("cagr_to_strict_mdd", -999)), float(r["final"]["sim"].get("cagr_pct", -999))), reverse=True)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": vars(args), "rows": rows, "top": rows[:30], "leakage_guard": {"events_precomputed_with_prior_only_fold_thresholds": True, "risk_overlay_uses_completed_trade_history_only": True}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sparse-report", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--candidate-limit", type=int, default=20)
    p.add_argument("--max-ensemble-size", type=int, default=6)
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--leverages", default="0.5,0.8,1.0")
    p.add_argument("--stop-losses", default="0,4,6")
    p.add_argument("--take-profits", default="0,6,10")
    p.add_argument("--atr-mults", default="0,2.5")
    p.add_argument("--rolling-windows", default="0,8")
    p.add_argument("--rolling-losses", default="0,4")
    p.add_argument("--pause-bars", type=int, default=288)
    p.add_argument("--min-recent-fold-trades", type=int, default=0)
    p.add_argument("--min-active-folds", type=int, default=0)
    p.add_argument("--setup-sizing", choices=["fixed", "prior_sharpe"], default="fixed")
    p.add_argument("--min-position-scale", type=float, default=0.25)
    p.add_argument("--max-position-scale", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    rep = run_sweep(parse_args())
    print("BEST", json.dumps(rep["top"][:5], ensure_ascii=False))


if __name__ == "__main__":
    main()
