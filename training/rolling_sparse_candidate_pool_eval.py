"""Rolling candidate-pool evaluation for sparse setup alpha.

Unlike reports that rank sparse candidates over all folds, this script chooses the
candidate pool for each target fold from information available before that fold:
previous fold outcomes when available, otherwise pre-fold training priors.  It
then applies the selected candidates unchanged to the target fold and stitches the
fold events into one live-style continuous replay.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.alpha_feature_backtest import _forward_return
from training.rolling_sparse_setup_miner import SparseSetupCfg, _default_folds, _event_stats, _feature_columns, _predicate_mask, _score_event_folds
from training.sparse_setup_ensemble_audit import EnsembleCfg, _candidate_events, _candidate_key, _load_market, _score, _simulate_events
from training.wave_feature_ridge_policy import build_wave_feature_frame


@dataclass(frozen=True)
class RollingPoolCfg:
    input_csv: str
    output: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    window_size: int = 144
    horizons: str = "288"
    quantiles: str = "0.15,0.20,0.25"
    max_features: int = 30
    top_history_candidates: int = 30
    max_ensemble_size: int = 5
    min_train_rows: int = 20_000
    min_fold_events: int = 20
    max_fold_events: int = 220
    min_positive_folds: int = 1
    leverage: float = 1.2
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    cooldown_bars: int = 0
    max_same_bar_signals: int = 1
    min_trades: int = 30
    trade_stop_loss_pct: float = 6.0
    trade_take_profit_pct: float = 6.0
    setup_sizing: str = "prior_sharpe"
    min_position_scale: float = 0.35
    max_position_scale: float = 1.0
    seed_mode: str = "prior"  # prior | skip_first


def _parse_list(raw: str, typ):
    return [typ(x.strip()) for x in str(raw).split(",") if x.strip()]


def _build_features(market: pd.DataFrame, cfg: RollingPoolCfg) -> pd.DataFrame:
    features = pd.concat(
        [
            build_market_feature_frame(market, window_size=int(cfg.window_size)).add_prefix("mkt__"),
            build_wave_feature_frame(market, window=int(cfg.window_size)).add_prefix("wave__"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return features.loc[:, ~features.columns.duplicated(keep="last")]


def _fold_meta(folds: list[dict[str, str]], dates: pd.Series, fwd: np.ndarray) -> list[dict[str, Any]]:
    finite_y = np.isfinite(fwd)
    out = []
    for fold in folds:
        start = pd.Timestamp(fold["eval_start"])
        end = pd.Timestamp(fold["eval_end"])
        out.append({"fold": fold, "train": np.asarray(dates < start, dtype=bool) & finite_y, "eval": np.asarray((dates >= start) & (dates <= end), dtype=bool) & finite_y})
    return out


def _sparse_score_cfg(cfg: RollingPoolCfg) -> SparseSetupCfg:
    return SparseSetupCfg(
        input_csv=cfg.input_csv,
        output=cfg.output,
        wave_trading_root=cfg.wave_trading_root,
        external_tolerance=cfg.external_tolerance,
        window_size=cfg.window_size,
        horizons=cfg.horizons,
        quantiles=cfg.quantiles,
        min_train_rows=cfg.min_train_rows,
        min_fold_events=cfg.min_fold_events,
        max_fold_events=cfg.max_fold_events,
        min_positive_folds=cfg.min_positive_folds,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        entry_delay_bars=cfg.entry_delay_bars,
        max_features=cfg.max_features,
    )


def _ensemble_cfg(cfg: RollingPoolCfg, horizon: int) -> EnsembleCfg:
    return EnsembleCfg(
        sparse_report="",
        market_csv=cfg.input_csv,
        output=cfg.output,
        wave_trading_root=cfg.wave_trading_root,
        external_tolerance=cfg.external_tolerance,
        window_size=cfg.window_size,
        candidate_limit=cfg.top_history_candidates,
        max_ensemble_size=cfg.max_ensemble_size,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        entry_delay_bars=cfg.entry_delay_bars,
        cooldown_bars=cfg.cooldown_bars,
        max_same_bar_signals=cfg.max_same_bar_signals,
        min_trades=cfg.min_trades,
        trade_stop_loss_pct=cfg.trade_stop_loss_pct,
        trade_take_profit_pct=cfg.trade_take_profit_pct,
        setup_sizing=cfg.setup_sizing,
        min_position_scale=cfg.min_position_scale,
        max_position_scale=cfg.max_position_scale,
        execution_horizon_bars=horizon,
    )


def _candidate_from_parts(feat_a: str, side_a: str, feat_b: str, side_b: str, horizon: int, q: float, idx: int) -> dict[str, Any]:
    return {"features": [{"name": feat_a, "side": side_a}, {"name": feat_b, "side": side_b}], "horizon": int(horizon), "quantile": float(q), "_candidate_index": int(idx)}


def _history_score_candidate(cand: dict[str, Any], *, history_meta: list[dict[str, Any]], X: dict[str, np.ndarray], fwd: np.ndarray, score_cfg: SparseSetupCfg) -> tuple[float, list[dict[str, Any]]]:
    fa, fb = cand["features"][0], cand["features"][1]
    q = float(cand["quantile"])
    fold_rows = []
    for fm in history_meta:
        train = fm["train"] & np.isfinite(X[fa["name"]]) & np.isfinite(X[fb["name"]])
        if int(train.sum()) < int(score_cfg.min_train_rows):
            fold_rows.append({"fold": fm["fold"]["name"], "n": 0, "skip": "not_enough_train", "train_n": int(train.sum())})
            continue
        ma, ta = _predicate_mask(X[fa["name"]], X[fa["name"]][train], fa["side"], q)
        mb, tb = _predicate_mask(X[fb["name"]], X[fb["name"]][train], fb["side"], q)
        active_train = train & ma & mb
        if int(active_train.sum()) < int(score_cfg.min_fold_events):
            fold_rows.append({"fold": fm["fold"]["name"], "n": 0, "skip": "not_enough_active_train", "train_n": int(train.sum()), "active_train_n": int(active_train.sum())})
            continue
        trade_side = 1 if float(np.mean(fwd[active_train])) >= 0.0 else -1
        active_eval = fm["eval"] & ma & mb
        raw = fwd[active_eval] * trade_side - (float(score_cfg.fee_rate) + float(score_cfg.slippage_rate)) * 2.0 * float(score_cfg.leverage)
        st = _event_stats(raw.astype(float))
        st.update({"fold": fm["fold"]["name"], "train_n": int(train.sum()), "active_train_n": int(active_train.sum()), "side": "LONG" if trade_side > 0 else "SHORT", "thresholds": {fa["name"]: {"side": fa["side"], "threshold": ta}, fb["name"]: {"side": fb["side"], "threshold": tb}}})
        fold_rows.append(st)
    return _score_event_folds(fold_rows, score_cfg), fold_rows


def _prior_score_candidate(cand: dict[str, Any], *, train: np.ndarray, X: dict[str, np.ndarray], fwd: np.ndarray, score_cfg: SparseSetupCfg) -> float:
    fa, fb = cand["features"][0], cand["features"][1]
    valid_train = train & np.isfinite(X[fa["name"]]) & np.isfinite(X[fb["name"]])
    if int(valid_train.sum()) < int(score_cfg.min_train_rows):
        return -1e9
    ma, _ = _predicate_mask(X[fa["name"]], X[fa["name"]][valid_train], fa["side"], float(cand["quantile"]))
    mb, _ = _predicate_mask(X[fb["name"]], X[fb["name"]][valid_train], fb["side"], float(cand["quantile"]))
    active = valid_train & ma & mb
    if int(active.sum()) < int(score_cfg.min_fold_events):
        return -1e9 + int(active.sum())
    vals = np.abs(fwd[active]) - (float(score_cfg.fee_rate) + float(score_cfg.slippage_rate)) * 2.0 * float(score_cfg.leverage)
    st = _event_stats(vals.astype(float))
    return float(st["t_stat"]) + float(st["mean_pct"]) * 0.25 + min(2.0, int(st["n"]) / 100.0)


def _select_greedy_history(scored: list[dict[str, Any]], *, report: dict[str, Any], dates: pd.Series, features: pd.DataFrame, market: pd.DataFrame, ens_cfg: EnsembleCfg, history_folds: list[str]) -> tuple[list[int], list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    top = scored[: int(ens_cfg.candidate_limit)]
    candidate_by_id = {int(r["candidate"]["_candidate_index"]): r["candidate"] for r in top}
    event_cache = {cid: _candidate_events(cand=c, report=report, dates=dates, features=features, market=market, cfg=ens_cfg) for cid, c in candidate_by_id.items()}
    history_set = set(history_folds)
    selected: list[int] = []
    current = None
    steps = []
    for _ in range(int(ens_cfg.max_ensemble_size)):
        best = None
        for cid in candidate_by_id:
            if cid in selected:
                continue
            events: list[dict[str, Any]] = []
            for tid in selected + [cid]:
                events.extend([e for e in event_cache[tid] if str(e.get("fold")) in history_set])
            if not events:
                continue
            res = _simulate_events(events, dates=dates, market=market, cfg=ens_cfg)
            sc = _score(res, ens_cfg)
            if best is None or sc > best[0]:
                best = (sc, cid, res)
        if best is None:
            break
        if current is not None and best[0] <= _score(current, ens_cfg) + 1e-9:
            break
        selected.append(best[1])
        current = best[2]
        steps.append({"candidate_index": best[1], "history_score": best[0], "history_result": {k: v for k, v in best[2].items() if k != "executed"}})
    if not selected:
        selected = [int(r["candidate"]["_candidate_index"]) for r in top[: int(ens_cfg.max_ensemble_size)]]
    return selected, steps, event_cache


def run(cfg: RollingPoolCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features = _build_features(market, cfg)
    cols = _feature_columns(features)
    if int(cfg.max_features) > 0:
        cols = cols[: int(cfg.max_features)]
    X = {c: features[c].to_numpy(dtype=float) for c in cols}
    dates = pd.to_datetime(market["date"])
    folds = _default_folds()
    score_cfg = _sparse_score_cfg(cfg)

    pred_specs = [(c, s) for c in cols for s in ("low", "high")]
    candidate_specs: list[dict[str, Any]] = []
    idx = 0
    for horizon in _parse_list(cfg.horizons, int):
        for q in _parse_list(cfg.quantiles, float):
            for i, a in enumerate(pred_specs):
                for b in pred_specs[i + 1:]:
                    if a[0] == b[0]:
                        continue
                    candidate_specs.append(_candidate_from_parts(a[0], a[1], b[0], b[1], int(horizon), float(q), idx))
                    idx += 1

    fold_rows = []
    final_events: list[dict[str, Any]] = []
    for fold_pos, fold in enumerate(folds):
        horizon = int(_parse_list(cfg.horizons, int)[0])
        fwd = _forward_return(market["open"].astype(float), horizon=horizon, entry_delay_bars=int(cfg.entry_delay_bars))
        metas = _fold_meta(folds, dates, fwd)
        target_meta = metas[fold_pos]
        history_meta = metas[:fold_pos]
        if not history_meta and str(cfg.seed_mode) == "skip_first":
            fold_rows.append({"fold": fold["name"], "selected": [], "skip": "no_history"})
            continue
        if history_meta:
            scored = []
            for cand in candidate_specs:
                sc, rows = _history_score_candidate(cand, history_meta=history_meta, X=X, fwd=fwd, score_cfg=score_cfg)
                scored.append({"candidate": cand, "history_event_score": float(sc), "history_event_folds": rows})
            scored.sort(key=lambda r: float(r["history_event_score"]), reverse=True)
        else:
            scored = []
            for cand in candidate_specs:
                sc = _prior_score_candidate(cand, train=target_meta["train"], X=X, fwd=fwd, score_cfg=score_cfg)
                scored.append({"candidate": cand, "history_event_score": float(sc), "history_event_folds": []})
            scored.sort(key=lambda r: float(r["history_event_score"]), reverse=True)

        fold_report = {"folds": history_meta and [m["fold"] for m in history_meta] or [fold]}
        target_report = {"folds": [fold]}
        ens_cfg = _ensemble_cfg(cfg, horizon)
        if history_meta:
            selected, steps, _ = _select_greedy_history(scored, report=fold_report, dates=dates, features=features, market=market, ens_cfg=ens_cfg, history_folds=[m["fold"]["name"] for m in history_meta])
        else:
            selected = [int(r["candidate"]["_candidate_index"]) for r in scored[: int(cfg.max_ensemble_size)]]
            steps = []
        cand_by_id = {int(r["candidate"]["_candidate_index"]): r["candidate"] for r in scored[: max(int(cfg.top_history_candidates), int(cfg.max_ensemble_size), 1)]}
        # Ensure greedy-selected ids beyond the retained top map are available.
        for r in scored:
            cid = int(r["candidate"]["_candidate_index"])
            if cid in selected and cid not in cand_by_id:
                cand_by_id[cid] = r["candidate"]
        target_events: list[dict[str, Any]] = []
        for cid in selected:
            target_events.extend(_candidate_events(cand=cand_by_id[cid], report=target_report, dates=dates, features=features, market=market, cfg=ens_cfg))
        target_events.sort(key=lambda e: (int(e["signal_pos"]), str(e["candidate_key"])))
        fold_res = _simulate_events(target_events, dates=dates, market=market, cfg=ens_cfg)
        final_events.extend(target_events)
        fold_rows.append({
            "fold": fold["name"],
            "history_folds": [m["fold"]["name"] for m in history_meta],
            "selected": selected,
            "selected_keys": [_candidate_key(cand_by_id[cid]) for cid in selected],
            "top_history": [{"candidate_index": int(r["candidate"]["_candidate_index"]), "key": _candidate_key(r["candidate"]), "score": r["history_event_score"]} for r in scored[:10]],
            "greedy_steps": steps,
            "fold_result": {k: v for k, v in fold_res.items() if k != "executed"},
        })

    final_events.sort(key=lambda e: (int(e["signal_pos"]), str(e["candidate_key"])))
    final_cfg = _ensemble_cfg(cfg, int(_parse_list(cfg.horizons, int)[0]))
    final = _simulate_events(final_events, dates=dates, market=market, cfg=final_cfg)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": int(len(market)), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "feature_count": len(cols),
        "candidate_count": len(candidate_specs),
        "folds": fold_rows,
        "final": {k: v for k, v in final.items() if k != "executed"},
        "leakage_guard": {
            "candidate_pool_ranked_per_target_fold": True,
            "history_scores_use_only_previous_eval_folds_or_prefold_prior": True,
            "target_fold_not_used_for_selection": True,
            "thresholds_and_side_fit_before_target_fold": True,
            "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled",
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling sparse candidate-pool evaluation")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=RollingPoolCfg.external_tolerance)
    p.add_argument("--window-size", type=int, default=RollingPoolCfg.window_size)
    p.add_argument("--horizons", default=RollingPoolCfg.horizons)
    p.add_argument("--quantiles", default=RollingPoolCfg.quantiles)
    p.add_argument("--max-features", type=int, default=RollingPoolCfg.max_features)
    p.add_argument("--top-history-candidates", type=int, default=RollingPoolCfg.top_history_candidates)
    p.add_argument("--max-ensemble-size", type=int, default=RollingPoolCfg.max_ensemble_size)
    p.add_argument("--min-train-rows", type=int, default=RollingPoolCfg.min_train_rows)
    p.add_argument("--min-fold-events", type=int, default=RollingPoolCfg.min_fold_events)
    p.add_argument("--max-fold-events", type=int, default=RollingPoolCfg.max_fold_events)
    p.add_argument("--min-positive-folds", type=int, default=RollingPoolCfg.min_positive_folds)
    p.add_argument("--leverage", type=float, default=RollingPoolCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=RollingPoolCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=RollingPoolCfg.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=RollingPoolCfg.entry_delay_bars)
    p.add_argument("--cooldown-bars", type=int, default=RollingPoolCfg.cooldown_bars)
    p.add_argument("--max-same-bar-signals", type=int, default=RollingPoolCfg.max_same_bar_signals)
    p.add_argument("--min-trades", type=int, default=RollingPoolCfg.min_trades)
    p.add_argument("--trade-stop-loss-pct", type=float, default=RollingPoolCfg.trade_stop_loss_pct)
    p.add_argument("--trade-take-profit-pct", type=float, default=RollingPoolCfg.trade_take_profit_pct)
    p.add_argument("--setup-sizing", choices=["fixed", "prior_sharpe"], default=RollingPoolCfg.setup_sizing)
    p.add_argument("--min-position-scale", type=float, default=RollingPoolCfg.min_position_scale)
    p.add_argument("--max-position-scale", type=float, default=RollingPoolCfg.max_position_scale)
    p.add_argument("--seed-mode", choices=["prior", "skip_first"], default=RollingPoolCfg.seed_mode)
    return p.parse_args()


def main() -> None:
    rep = run(RollingPoolCfg(**vars(parse_args())))
    print(json.dumps({"final": rep["final"]["sim"], "trade_stats": rep["final"].get("trade_stats", {}), "folds": [{"fold": f["fold"], "selected": f.get("selected", []), "trades": f.get("fold_result", {}).get("sim", {}).get("trade_entries", 0)} for f in rep["folds"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
