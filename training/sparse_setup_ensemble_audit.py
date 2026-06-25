"""Continuous rolling sparse-setup ensemble audit.

Takes top candidates from ``rolling_sparse_setup_miner.py`` and replays them as a
single live-style equity curve from the first eval fold through the last. For each
fold, every setup's thresholds and side are fit only on data before fold start.

Selection is intentionally simple and transparent: rank individual candidates by
continuous CAGR/MDD, then greedily add candidates that improve the ensemble score.
This tests whether sparse opportunity families diversify each other without using
future bars inside a fold.
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
from training.strict_bar_backtest import _drawdown_from_trough, _trade_stats
from training.wave_feature_ridge_policy import build_wave_feature_frame
from training.price_action_extreme_feature_audit import build_extreme_bar_features


@dataclass(frozen=True)
class EnsembleCfg:
    sparse_report: str
    market_csv: str
    output: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    window_size: int = 144
    candidate_limit: int = 20
    max_ensemble_size: int = 6
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    cooldown_bars: int = 0
    max_same_bar_signals: int = 1
    min_trades: int = 30
    trade_stop_loss_pct: float = 0.0
    trade_take_profit_pct: float = 0.0
    atr_trailing_stop_mult: float = 0.0
    atr_period: int = 45
    rolling_window_trades: int = 0
    rolling_loss_stop_pct: float = 0.0
    pause_bars: int = 288
    min_recent_fold_trades: int = 0
    min_active_folds: int = 0
    setup_sizing: str = "fixed"  # fixed | prior_sharpe
    min_position_scale: float = 0.25
    max_position_scale: float = 1.0
    execution_horizon_bars: int = 0  # 0 keeps candidate horizon
    include_price_action_extremes: bool = False
    price_action_lookbacks: str = "36,72,144,288,576,2016"


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _fit_mask(values: np.ndarray, train: np.ndarray, finite_y: np.ndarray, side: str, q: float) -> tuple[np.ndarray, float]:
    train_values = values[train & finite_y & np.isfinite(values)]
    if train_values.size < 100:
        raise ValueError("not enough train values")
    if side == "low":
        thr = float(np.quantile(train_values, q))
        return values <= thr, thr
    thr = float(np.quantile(train_values, 1.0 - q))
    return values >= thr, thr


def _mask_from_threshold(values: np.ndarray, side: str, threshold: float) -> np.ndarray:
    if side == "low":
        return values <= float(threshold)
    return values >= float(threshold)


def _stored_fold_spec(cand: dict[str, Any], fold_name: str) -> dict[str, Any] | None:
    for row in cand.get("strict_folds", []):
        if str(row.get("fold")) == str(fold_name):
            return row
    return None


def _candidate_events(*, cand: dict[str, Any], report: dict[str, Any], dates: pd.Series, features: pd.DataFrame, market: pd.DataFrame, cfg: EnsembleCfg) -> list[dict[str, Any]]:
    horizon = int(cfg.execution_horizon_bars) if int(cfg.execution_horizon_bars) > 0 else int(cand["horizon"])
    q = float(cand["quantile"])
    fa, fb = cand["features"][0], cand["features"][1]
    fwd = _forward_return(market["open"].astype(float), horizon=horizon, entry_delay_bars=int(cfg.entry_delay_bars))
    finite_y = np.isfinite(fwd)
    xa = features[fa["name"]].to_numpy(dtype=float)
    xb = features[fb["name"]].to_numpy(dtype=float)
    events: list[dict[str, Any]] = []
    for fold in report["folds"]:
        start = pd.Timestamp(fold["eval_start"])
        end = pd.Timestamp(fold["eval_end"])
        train = np.asarray(dates < start, dtype=bool)
        stored = _stored_fold_spec(cand, str(fold["name"]))
        if stored is not None:
            stored_sim = stored.get("result", {}).get("sim", {})
            if int(stored_sim.get("samples", stored_sim.get("trade_entries", 0)) or 0) <= 0:
                continue
            thresholds = stored.get("thresholds", {})
            try:
                ta = float(thresholds[fa["name"]]["threshold"])
                tb = float(thresholds[fb["name"]]["threshold"])
            except KeyError:
                continue
            ma = _mask_from_threshold(xa, fa["side"], ta)
            mb = _mask_from_threshold(xb, fb["side"], tb)
            side = 1 if str(stored.get("side", "LONG")).upper() == "LONG" else -1
        else:
            try:
                ma, ta = _fit_mask(xa, train, finite_y, fa["side"], q)
                mb, tb = _fit_mask(xb, train, finite_y, fb["side"], q)
            except ValueError:
                continue
            active_train_for_side = train & ma & mb & finite_y
            if int(active_train_for_side.sum()) <= 0:
                continue
            side = 1 if float(np.mean(fwd[active_train_for_side])) >= 0.0 else -1
        active_train = train & ma & mb & finite_y
        if int(active_train.sum()) <= 0:
            continue
        prior_rets = fwd[active_train] * float(side)
        prior_mean = float(np.mean(prior_rets)) if prior_rets.size else 0.0
        prior_std = float(np.std(prior_rets, ddof=1)) if prior_rets.size > 1 else 0.0
        idx = np.flatnonzero(np.asarray((dates >= start) & (dates <= end), dtype=bool) & ma & mb)
        for pos in idx:
            events.append(
                {
                    "signal_pos": int(pos),
                    "date": str(dates.iloc[int(pos)]),
                    "side": int(side),
                    "horizon": horizon,
                    "source_horizon": int(cand["horizon"]),
                    "candidate_index": int(cand.get("_candidate_index", -1)),
                    "candidate_key": _candidate_key(cand),
                    "fold": fold["name"],
                    "prior_mean_ret": prior_mean,
                    "prior_std_ret": prior_std,
                    "prior_n": int(prior_rets.size),
                    "thresholds": {fa["name"]: ta, fb["name"]: tb},
                }
            )
    return events


def _candidate_key(cand: dict[str, Any]) -> str:
    parts = [f"{x['name']}:{x['side']}" for x in cand["features"]]
    return f"h{cand['horizon']}|q{cand['quantile']}|" + "&".join(parts)


def _rolling_atr(highs: np.ndarray, lows: np.ndarray, opens: np.ndarray, period: int) -> np.ndarray:
    period = max(1, int(period))
    prev_close = np.roll(opens, 1)
    prev_close[0] = opens[0]
    true_range = np.maximum.reduce([highs - lows, np.abs(highs - prev_close), np.abs(lows - prev_close)])
    out = np.empty_like(true_range, dtype=float)
    csum = np.cumsum(true_range, dtype=float)
    for i in range(len(true_range)):
        start = max(0, i - period + 1)
        out[i] = (csum[i] - (csum[start - 1] if start > 0 else 0.0)) / float(i - start + 1)
    return out


def _recent_loss(trade_returns: list[float], n: int) -> float:
    if n <= 0 or not trade_returns:
        return 0.0
    eq = 1.0
    for ret in trade_returns[-int(n):]:
        eq *= max(0.0, 1.0 + float(ret))
    return min(0.0, eq - 1.0)


def _event_position_scale(ev: dict[str, Any], cfg: EnsembleCfg) -> float:
    if str(cfg.setup_sizing) == "fixed":
        return 1.0
    if str(cfg.setup_sizing) != "prior_sharpe":
        return 1.0
    mean = float(ev.get("prior_mean_ret", 0.0) or 0.0)
    std = float(ev.get("prior_std_ret", 0.0) or 0.0)
    n = int(ev.get("prior_n", 0) or 0)
    if n < 20 or std <= 1e-12 or mean <= 0.0:
        return float(cfg.min_position_scale)
    # Smoothly maps positive prior risk-adjusted edge to [min,max]. Conservative
    # because prior labels are noisy and fold-local.
    raw = mean / std
    scale = float(cfg.min_position_scale) + (float(cfg.max_position_scale) - float(cfg.min_position_scale)) * min(1.0, max(0.0, raw / 0.35))
    return float(min(float(cfg.max_position_scale), max(float(cfg.min_position_scale), scale)))


def _simulate_events(events: list[dict[str, Any]], *, dates: pd.Series, market: pd.DataFrame, cfg: EnsembleCfg) -> dict[str, Any]:
    if not events:
        return {"sim": {"cagr_pct": -100.0, "strict_mdd_pct": 100.0, "cagr_to_strict_mdd": -1.0, "trade_entries": 0}, "trade_stats": _trade_stats([]), "executed": []}
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for ev in sorted(events, key=lambda e: (int(e["signal_pos"]), str(e["candidate_key"]))):
        grouped.setdefault(int(ev["signal_pos"]), []).append(ev)
    eq = peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    trade_returns: list[float] = []
    executed: list[dict[str, Any]] = []
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)
    atr = _rolling_atr(highs, lows, opens, int(cfg.atr_period)) if float(cfg.atr_trailing_stop_mult) > 0.0 else None
    paused_until = -1
    skipped_overlay = 0
    exit_reasons: dict[str, int] = {}
    for signal_pos in sorted(grouped):
        if signal_pos < next_allowed:
            continue
        if signal_pos < paused_until:
            skipped_overlay += 1
            continue
        if int(cfg.rolling_window_trades) > 0 and len(trade_returns) >= int(cfg.rolling_window_trades):
            loss_pct = -_recent_loss(trade_returns, int(cfg.rolling_window_trades)) * 100.0
            if float(cfg.rolling_loss_stop_pct) > 0.0 and loss_pct >= float(cfg.rolling_loss_stop_pct):
                paused_until = signal_pos + max(1, int(cfg.pause_bars))
                skipped_overlay += 1
                continue
        # If multiple candidates fire on the same bar, use the first N deterministic
        # keys. This avoids multiplying leverage on duplicate condition hits.
        for ev in grouped[signal_pos][: max(1, int(cfg.max_same_bar_signals))]:
            entry_pos = int(ev["signal_pos"]) + int(cfg.entry_delay_bars)
            exit_pos = entry_pos + int(ev["horizon"])
            if entry_pos >= len(market) - 1 or exit_pos >= len(market):
                continue
            side = int(ev["side"])
            position_scale = _event_position_scale(ev, cfg)
            trade_leverage = float(cfg.leverage) * position_scale
            trade_cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * trade_leverage
            entry_eq = eq
            eq *= max(0.0, 1.0 - trade_cost)
            max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
            position_start_eq = eq
            entry_price = float(opens[entry_pos])
            exit_reason = "time"
            atr_stop_price = None
            if atr is not None and entry_price > 0.0:
                atr_distance = float(atr[max(0, entry_pos - 1)]) * float(cfg.atr_trailing_stop_mult)
                if atr_distance > 0.0:
                    atr_stop_price = entry_price - atr_distance if side > 0 else entry_price + atr_distance
            for j in range(entry_pos, exit_pos):
                open_j = float(opens[j])
                if open_j <= 0.0:
                    continue
                if side > 0:
                    adverse_ret = (float(lows[j]) - open_j) / open_j
                    close_ret = (float(opens[j + 1]) - open_j) / open_j
                    from_entry_low = (float(lows[j]) - entry_price) / entry_price if entry_price > 0.0 else 0.0
                    from_entry_high = (float(highs[j]) - entry_price) / entry_price if entry_price > 0.0 else 0.0
                    atr_stop_hit = atr_stop_price is not None and float(lows[j]) <= float(atr_stop_price)
                    atr_stop_ret = (float(atr_stop_price) - entry_price) / entry_price if atr_stop_hit and entry_price > 0.0 else 0.0
                else:
                    adverse_ret = (open_j - float(highs[j])) / open_j
                    close_ret = (open_j - float(opens[j + 1])) / open_j
                    from_entry_low = (entry_price - float(highs[j])) / entry_price if entry_price > 0.0 else 0.0
                    from_entry_high = (entry_price - float(lows[j])) / entry_price if entry_price > 0.0 else 0.0
                    atr_stop_hit = atr_stop_price is not None and float(highs[j]) >= float(atr_stop_price)
                    atr_stop_ret = (entry_price - float(atr_stop_price)) / entry_price if atr_stop_hit and entry_price > 0.0 else 0.0
                max_dd = max(max_dd, _drawdown_from_trough(peak, eq * (1.0 + trade_leverage * adverse_ret)))
                stop_hit = float(cfg.trade_stop_loss_pct) > 0.0 and trade_leverage * from_entry_low * 100.0 <= -float(cfg.trade_stop_loss_pct)
                take_hit = float(cfg.trade_take_profit_pct) > 0.0 and trade_leverage * from_entry_high * 100.0 >= float(cfg.trade_take_profit_pct)
                if stop_hit:
                    eq = position_start_eq * max(0.0, 1.0 - float(cfg.trade_stop_loss_pct) / 100.0)
                    max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
                    exit_reason = "stop_loss"
                    exit_pos = j + 1
                    break
                if atr_stop_hit:
                    eq = position_start_eq * max(0.0, 1.0 + trade_leverage * atr_stop_ret)
                    max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
                    exit_reason = "atr_trailing_stop"
                    exit_pos = j + 1
                    break
                if take_hit:
                    eq = position_start_eq * (1.0 + float(cfg.trade_take_profit_pct) / 100.0)
                    peak = max(peak, eq)
                    exit_reason = "take_profit"
                    exit_pos = j + 1
                    break
                eq *= max(0.0, 1.0 + trade_leverage * close_ret)
                peak = max(peak, eq)
                if atr_stop_price is not None and entry_price > 0.0:
                    if side > 0:
                        atr_stop_price = max(float(atr_stop_price), float(highs[j]) - float(atr[j]) * float(cfg.atr_trailing_stop_mult))
                    else:
                        atr_stop_price = min(float(atr_stop_price), float(lows[j]) + float(atr[j]) * float(cfg.atr_trailing_stop_mult))
                if eq <= 0.0:
                    exit_reason = "liquidation"
                    break
            eq *= max(0.0, 1.0 - trade_cost)
            max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
            peak = max(peak, eq)
            trade_returns.append(eq / entry_eq - 1.0)
            exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
            executed.append({**ev, "ret_pct": (eq / entry_eq - 1.0) * 100.0, "exit_reason": exit_reason, "position_scale": position_scale})
            next_allowed = exit_pos + max(0, int(cfg.cooldown_bars))
            if eq <= 0.0:
                break
        if eq <= 0.0:
            break
    start_dt = pd.Timestamp(events[0]["date"]).to_pydatetime()
    end_dt = pd.Timestamp(events[-1]["date"]).to_pydatetime()
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd = max_dd * 100.0
    return {
        "period": {"start": str(start_dt), "end": str(end_dt), "years": years},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr,
            "strict_mdd_pct": mdd,
            "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else float("inf"),
            "trade_entries": len(trade_returns),
            "skipped_overlay": skipped_overlay,
            "exit_reasons": exit_reasons,
            "return_application": "continuous_sparse_setup_ensemble_actual_ohlc",
        },
        "trade_stats": _trade_stats(trade_returns),
        "fold_trade_counts": {fold: sum(1 for e in executed if e["fold"] == fold) for fold in sorted({e["fold"] for e in events})},
        "executed": executed[:200],
    }


def _score(result: dict[str, Any], cfg: EnsembleCfg) -> float:
    sim = result["sim"]
    trades = int(sim.get("trade_entries", 0))
    cagr = float(sim.get("cagr_pct", -100.0))
    mdd = float(sim.get("strict_mdd_pct", 100.0))
    ratio = float(sim.get("cagr_to_strict_mdd", -1.0))
    fold_counts = result.get("fold_trade_counts", {}) or {}
    active_folds = sum(1 for v in fold_counts.values() if int(v) > 0)
    if int(cfg.min_active_folds) > 0 and active_folds < int(cfg.min_active_folds):
        return -800.0 + active_folds + cagr / 100.0 - mdd / 100.0
    if int(cfg.min_recent_fold_trades) > 0:
        recent_keys = sorted(fold_counts)[-2:]
        if any(int(fold_counts.get(k, 0)) < int(cfg.min_recent_fold_trades) for k in recent_keys):
            return -700.0 + sum(int(fold_counts.get(k, 0)) for k in recent_keys) / 10.0 + cagr / 100.0 - mdd / 100.0
    if trades < int(cfg.min_trades) or cagr <= 0.0:
        return -1000.0 + trades / 1000.0 + cagr / 100.0 - mdd / 100.0
    return ratio * 10.0 + min(50.0, cagr) / 10.0 - max(0.0, mdd - 15.0) / 5.0 + min(2.0, trades / 100.0)


def run(cfg: EnsembleCfg) -> dict[str, Any]:
    sparse = json.loads(Path(cfg.sparse_report).read_text())
    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    feature_parts = [
        build_market_feature_frame(market, window_size=int(cfg.window_size)).add_prefix("mkt__"),
        build_wave_feature_frame(market, window=int(cfg.window_size)).add_prefix("wave__"),
    ]
    if bool(cfg.include_price_action_extremes):
        feature_parts.append(build_extreme_bar_features(market, tuple(int(x.strip()) for x in str(cfg.price_action_lookbacks).split(",") if x.strip())).add_prefix("pa__"))
    features = pd.concat(feature_parts, axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    dates = pd.to_datetime(market["date"])
    candidates = [dict(c, _candidate_index=i) for i, c in enumerate(sparse.get("top_strict", [])[: int(cfg.candidate_limit)])]
    event_cache: dict[int, list[dict[str, Any]]] = {}
    individual = []
    for cand in candidates:
        idx = int(cand["_candidate_index"])
        events = _candidate_events(cand=cand, report=sparse, dates=dates, features=features, market=market, cfg=cfg)
        event_cache[idx] = events
        res = _simulate_events(events, dates=dates, market=market, cfg=cfg)
        individual.append({"candidate_index": idx, "key": _candidate_key(cand), "candidate": cand, "result": {k: v for k, v in res.items() if k != "executed"}, "score": _score(res, cfg)})
    individual.sort(key=lambda r: float(r["score"]), reverse=True)

    selected: list[int] = []
    current: dict[str, Any] | None = None
    steps = []
    for _ in range(int(cfg.max_ensemble_size)):
        best = None
        for row in individual:
            idx = int(row["candidate_index"])
            if idx in selected:
                continue
            trial_ids = selected + [idx]
            events = []
            for tid in trial_ids:
                events.extend(event_cache[tid])
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
        steps.append({"added_candidate_index": best[1], "score": best[0], "result": {k: v for k, v in best[2].items() if k != "executed"}})
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "source_sparse_report": cfg.sparse_report,
        "individual": individual,
        "greedy_ensemble": {"selected_candidate_indices": selected, "steps": steps, "final": None if current is None else {k: v for k, v in current.items() if k != "executed"}},
        "leakage_guard": {
            "source_candidates_from_sparse_report": True,
            "each_fold_thresholds_and_side_fit_before_eval_start": True,
            "ensemble_replay_is_continuous_not_fold_reset": True,
            "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled",
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuous sparse setup ensemble audit")
    p.add_argument("--sparse-report", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=EnsembleCfg.external_tolerance)
    p.add_argument("--window-size", type=int, default=EnsembleCfg.window_size)
    p.add_argument("--candidate-limit", type=int, default=EnsembleCfg.candidate_limit)
    p.add_argument("--max-ensemble-size", type=int, default=EnsembleCfg.max_ensemble_size)
    p.add_argument("--leverage", type=float, default=EnsembleCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=EnsembleCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=EnsembleCfg.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=EnsembleCfg.entry_delay_bars)
    p.add_argument("--cooldown-bars", type=int, default=EnsembleCfg.cooldown_bars)
    p.add_argument("--max-same-bar-signals", type=int, default=EnsembleCfg.max_same_bar_signals)
    p.add_argument("--min-trades", type=int, default=EnsembleCfg.min_trades)
    p.add_argument("--trade-stop-loss-pct", type=float, default=EnsembleCfg.trade_stop_loss_pct)
    p.add_argument("--trade-take-profit-pct", type=float, default=EnsembleCfg.trade_take_profit_pct)
    p.add_argument("--atr-trailing-stop-mult", type=float, default=EnsembleCfg.atr_trailing_stop_mult)
    p.add_argument("--atr-period", type=int, default=EnsembleCfg.atr_period)
    p.add_argument("--rolling-window-trades", type=int, default=EnsembleCfg.rolling_window_trades)
    p.add_argument("--rolling-loss-stop-pct", type=float, default=EnsembleCfg.rolling_loss_stop_pct)
    p.add_argument("--pause-bars", type=int, default=EnsembleCfg.pause_bars)
    p.add_argument("--min-recent-fold-trades", type=int, default=EnsembleCfg.min_recent_fold_trades)
    p.add_argument("--min-active-folds", type=int, default=EnsembleCfg.min_active_folds)
    p.add_argument("--setup-sizing", choices=["fixed", "prior_sharpe"], default=EnsembleCfg.setup_sizing)
    p.add_argument("--min-position-scale", type=float, default=EnsembleCfg.min_position_scale)
    p.add_argument("--max-position-scale", type=float, default=EnsembleCfg.max_position_scale)
    p.add_argument("--execution-horizon-bars", type=int, default=EnsembleCfg.execution_horizon_bars)
    p.add_argument("--include-price-action-extremes", action="store_true", default=EnsembleCfg.include_price_action_extremes)
    p.add_argument("--price-action-lookbacks", default=EnsembleCfg.price_action_lookbacks)
    return p.parse_args()


def main() -> None:
    rep = run(EnsembleCfg(**vars(parse_args())))
    print(json.dumps({"best_individual": rep["individual"][:5], "ensemble": rep["greedy_ensemble"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
