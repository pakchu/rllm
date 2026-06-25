"""Rolling sparse setup miner for event/opportunity alpha.

Searches two-predicate setups such as:
  htf weekly trend is high AND flow imbalance is low -> choose LONG/SHORT from prior data

For each eval fold, predicate thresholds and side are fit only from rows before the
fold starts. Event-level ranking is used only to select top candidates for strict
bar-by-bar OHLC replay. This is meant to find sparse opportunity regimes rather
than broad always-on direction predictors.
"""
from __future__ import annotations

import argparse
import json
import math
import re
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


@dataclass(frozen=True)
class SparseSetupCfg:
    input_csv: str
    output: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    window_size: int = 144
    horizons: str = "72,144,288"
    quantiles: str = "0.15,0.25"
    folds_json: str = ""
    min_train_rows: int = 20_000
    min_fold_events: int = 20
    max_fold_events: int = 220
    min_positive_folds: int = 5
    top_event_candidates: int = 80
    max_strict_candidates: int = 20
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    max_features: int = 0
    include_external_components: bool = False
    feature_include_regex: str = ""


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _parse_list(raw: str, typ):
    return [typ(x.strip()) for x in str(raw).split(",") if x.strip()]


def _default_folds() -> list[dict[str, str]]:
    return [
        {"name": "eval_2023h1", "eval_start": "2023-01-01", "eval_end": "2023-06-30 23:59:59"},
        {"name": "eval_2023h2", "eval_start": "2023-07-01", "eval_end": "2023-12-31 23:59:59"},
        {"name": "eval_2024h1", "eval_start": "2024-01-01", "eval_end": "2024-06-30 23:59:59"},
        {"name": "eval_2024h2", "eval_start": "2024-07-01", "eval_end": "2024-12-31 23:59:59"},
        {"name": "eval_2025h1", "eval_start": "2025-01-01", "eval_end": "2025-06-30 23:59:59"},
        {"name": "eval_2025h2", "eval_start": "2025-07-01", "eval_end": "2025-12-31 23:59:59"},
        {"name": "eval_2026h1", "eval_start": "2026-01-01", "eval_end": "2026-06-01 00:00:00"},
    ]


def _feature_columns(features: pd.DataFrame) -> list[str]:
    cols = []
    deny = ("available", "external_any")
    for c in features.columns:
        if any(x in c for x in deny):
            continue
        x = features[c].to_numpy(dtype=float)
        if float(np.nanstd(x)) <= 1e-12:
            continue
        # Avoid raw level columns that mostly encode cache coverage/date drift.
        if c in {"mkt__dxy", "mkt__kimchi_premium", "wave__dxy", "wave__kimchi_premium"}:
            continue
        cols.append(c)
    preferred_prefixes = (
        "mkt__htf_", "mkt__weekly_", "mkt__trend_", "mkt__range_", "mkt__window_", "mkt__volume_",
        "mkt__taker_", "mkt__dxy_", "mkt__kimchi_", "mkt__usdkrw_", "wave__eff_", "wave__gk_",
        "mkt__btckrw_", "mkt__fx_", "wave__btckrw_", "wave__fx_",
        "wave__cvd_", "wave__flow_", "wave__taker_", "wave__vol_", "wave__vwap_", "wave__mom_",
    )
    preferred = [c for c in cols if c.startswith(preferred_prefixes)]
    return preferred or cols


def _event_stats(xs: np.ndarray) -> dict[str, Any]:
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return {"n": 0, "mean_pct": 0.0, "t_stat": 0.0, "win_rate": 0.0}
    sd = float(np.std(xs, ddof=1)) if xs.size > 1 else 0.0
    t = float(np.mean(xs) / (sd / math.sqrt(xs.size))) if sd > 1e-12 else 0.0
    return {"n": int(xs.size), "mean_pct": float(np.mean(xs) * 100.0), "t_stat": t, "win_rate": float(np.mean(xs > 0.0))}


def _score_event_folds(folds: list[dict[str, Any]], cfg: SparseSetupCfg) -> float:
    usable = [f for f in folds if int(f.get("n", 0)) >= int(cfg.min_fold_events)]
    if len(usable) < 5:
        return -1e9 + len(usable)
    means = np.asarray([float(f["mean_pct"]) for f in usable], dtype=float)
    ts = np.asarray([float(f["t_stat"]) for f in usable], dtype=float)
    ns = np.asarray([int(f["n"]) for f in usable], dtype=float)
    pos = int(np.sum(means > 0.0))
    too_many_penalty = float(np.mean(np.maximum(0.0, ns - int(cfg.max_fold_events)))) / 50.0
    if pos < int(cfg.min_positive_folds):
        return -1e6 + pos * 100.0 + float(np.median(means))
    worst = float(np.min(means))
    return pos * 20.0 + float(np.median(means)) * 2.0 + min(3.0, float(np.median(ts))) + min(0.0, worst) * 3.0 - too_many_penalty


def _predicate_mask(values: np.ndarray, train_values: np.ndarray, side: str, q: float) -> tuple[np.ndarray, float]:
    if side == "low":
        thr = float(np.quantile(train_values, q))
        return values <= thr, thr
    thr = float(np.quantile(train_values, 1.0 - q))
    return values >= thr, thr


def _simulate_indices(*, market: pd.DataFrame, dates: pd.Series, indices: np.ndarray, side: int, horizon: int, cfg: SparseSetupCfg) -> dict[str, Any]:
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    eq = peak = 1.0
    max_dd = 0.0
    entries = 0
    trade_returns: list[float] = []
    next_allowed = 0
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)
    for pos in indices:
        pos = int(pos)
        if pos < next_allowed:
            continue
        entry_pos = pos + int(cfg.entry_delay_bars)
        exit_pos = entry_pos + int(horizon)
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            continue
        entry_eq = eq
        entries += 1
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        for j in range(entry_pos, exit_pos):
            open_j = float(opens[j])
            if open_j <= 0:
                continue
            if side > 0:
                adverse_ret = (float(lows[j]) - open_j) / open_j
                close_ret = (float(opens[j + 1]) - open_j) / open_j
            else:
                adverse_ret = (open_j - float(highs[j])) / open_j
                close_ret = (open_j - float(opens[j + 1])) / open_j
            max_dd = max(max_dd, _drawdown_from_trough(peak, eq * (1.0 + float(cfg.leverage) * adverse_ret)))
            eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        next_allowed = exit_pos
        if eq <= 0.0:
            break
    if len(indices):
        start_dt = pd.Timestamp(dates.iloc[int(indices[0])]).to_pydatetime()
        end_dt = pd.Timestamp(dates.iloc[int(indices[-1])]).to_pydatetime()
    else:
        start_dt = pd.Timestamp(dates.iloc[0]).to_pydatetime(); end_dt = pd.Timestamp(dates.iloc[-1]).to_pydatetime()
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd = max_dd * 100.0
    return {"period": {"start": str(start_dt), "end": str(end_dt), "years": years}, "sim": {"ret_pct": ret_pct, "cagr_pct": cagr, "strict_mdd_pct": mdd, "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else float("inf"), "trade_entries": entries, "samples": int(len(indices)), "hold_bars": int(horizon), "side": "LONG" if side > 0 else "SHORT", "return_application": "sparse_setup_actual_ohlc_bar_by_bar"}, "trade_stats": _trade_stats(trade_returns)}


def run(cfg: SparseSetupCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(
            market,
            wave_trading_root=cfg.wave_trading_root,
            tolerance=cfg.external_tolerance,
            include_forex_components=bool(cfg.include_external_components),
        )
    base = build_market_feature_frame(market, window_size=int(cfg.window_size)).add_prefix("mkt__")
    wave = build_wave_feature_frame(market, window=int(cfg.window_size)).add_prefix("wave__")
    features = pd.concat([base, wave], axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    cols = _feature_columns(features)
    if str(cfg.feature_include_regex).strip():
        pattern = re.compile(str(cfg.feature_include_regex))
        cols = [c for c in cols if pattern.search(c)]
    if int(cfg.max_features) > 0:
        cols = cols[: int(cfg.max_features)]
    X = {c: features[c].to_numpy(dtype=float) for c in cols}
    dates = pd.to_datetime(market["date"])
    folds = json.loads(cfg.folds_json) if cfg.folds_json else _default_folds()
    fold_meta = []
    for fold in folds:
        start = pd.Timestamp(fold["eval_start"])
        fit_end = start - pd.Timedelta(seconds=1)
        fold_meta.append({"fold": fold, "train": np.asarray(dates <= fit_end, dtype=bool), "eval": np.asarray((dates >= start) & (dates <= pd.Timestamp(fold["eval_end"])), dtype=bool)})

    candidates: list[dict[str, Any]] = []
    quantiles = _parse_list(cfg.quantiles, float)
    pred_specs = [(c, s) for c in cols for s in ("low", "high")]
    pairs = []
    for i, a in enumerate(pred_specs):
        for b in pred_specs[i + 1 :]:
            if a[0] == b[0]:
                continue
            pairs.append((a, b))

    for horizon in _parse_list(cfg.horizons, int):
        fwd = _forward_return(market["open"].astype(float), horizon=int(horizon), entry_delay_bars=int(cfg.entry_delay_bars))
        finite_y = np.isfinite(fwd)
        for q in quantiles:
            for (feat_a, side_a), (feat_b, side_b) in pairs:
                fold_rows = []
                for fm in fold_meta:
                    train = fm["train"] & finite_y & np.isfinite(X[feat_a]) & np.isfinite(X[feat_b])
                    if int(train.sum()) < int(cfg.min_train_rows):
                        fold_rows.append({"fold": fm["fold"]["name"], "n": 0, "skip": "not_enough_train", "train_n": int(train.sum())})
                        continue
                    mask_a, thr_a = _predicate_mask(X[feat_a], X[feat_a][train], side_a, float(q))
                    mask_b, thr_b = _predicate_mask(X[feat_b], X[feat_b][train], side_b, float(q))
                    active_train = train & mask_a & mask_b
                    if int(active_train.sum()) < int(cfg.min_fold_events):
                        fold_rows.append({"fold": fm["fold"]["name"], "n": 0, "skip": "not_enough_active_train", "train_n": int(train.sum()), "active_train_n": int(active_train.sum())})
                        continue
                    train_mean = float(np.mean(fwd[active_train]))
                    trade_side = 1 if train_mean >= 0.0 else -1
                    active_eval = fm["eval"] & finite_y & mask_a & mask_b
                    raw = fwd[active_eval] * trade_side - (float(cfg.fee_rate) + float(cfg.slippage_rate)) * 2.0 * float(cfg.leverage)
                    st = _event_stats(raw.astype(float))
                    st.update({"fold": fm["fold"]["name"], "train_n": int(train.sum()), "active_train_n": int(active_train.sum()), "side": "LONG" if trade_side > 0 else "SHORT", "thresholds": {feat_a: {"side": side_a, "threshold": thr_a}, feat_b: {"side": side_b, "threshold": thr_b}}})
                    fold_rows.append(st)
                score = _score_event_folds(fold_rows, cfg)
                candidates.append({"features": [{"name": feat_a, "side": side_a}, {"name": feat_b, "side": side_b}], "horizon": int(horizon), "quantile": float(q), "event_score": float(score), "event_folds": fold_rows})
    candidates.sort(key=lambda r: float(r["event_score"]), reverse=True)

    strict_rows = []
    for cand in candidates[: int(cfg.max_strict_candidates)]:
        strict_folds = []
        fa, fb = cand["features"][0], cand["features"][1]
        horizon = int(cand["horizon"])
        fwd = _forward_return(market["open"].astype(float), horizon=horizon, entry_delay_bars=int(cfg.entry_delay_bars))
        finite_y = np.isfinite(fwd)
        for fm in fold_meta:
            train = fm["train"] & finite_y & np.isfinite(X[fa["name"]]) & np.isfinite(X[fb["name"]])
            mask_a, thr_a = _predicate_mask(X[fa["name"]], X[fa["name"]][train], fa["side"], float(cand["quantile"]))
            mask_b, thr_b = _predicate_mask(X[fb["name"]], X[fb["name"]][train], fb["side"], float(cand["quantile"]))
            active_train = train & mask_a & mask_b
            trade_side = 1 if float(np.mean(fwd[active_train])) >= 0.0 else -1
            idx = np.flatnonzero(fm["eval"] & mask_a & mask_b)
            result = _simulate_indices(market=market, dates=dates, indices=idx, side=trade_side, horizon=horizon, cfg=cfg)
            strict_folds.append({"fold": fm["fold"]["name"], "side": "LONG" if trade_side > 0 else "SHORT", "thresholds": {fa["name"]: {"side": fa["side"], "threshold": thr_a}, fb["name"]: {"side": fb["side"], "threshold": thr_b}}, "result": result})
        row = dict(cand)
        row["strict_folds"] = strict_folds
        sims = [f["result"]["sim"] for f in strict_folds]
        row["strict_summary"] = {"positive_folds": int(sum(float(s["cagr_pct"]) > 0 for s in sims)), "ratio3_mdd15_folds": int(sum(float(s["cagr_to_strict_mdd"]) >= 3 and float(s["strict_mdd_pct"]) <= 15 for s in sims)), "total_trades": int(sum(int(s["trade_entries"]) for s in sims)), "median_cagr_pct": float(np.median([float(s["cagr_pct"]) for s in sims])), "median_strict_mdd_pct": float(np.median([float(s["strict_mdd_pct"]) for s in sims])), "worst_cagr_pct": float(np.min([float(s["cagr_pct"]) for s in sims]))}
        strict_rows.append(row)
    strict_rows.sort(key=lambda r: (r["strict_summary"]["positive_folds"], r["strict_summary"]["ratio3_mdd15_folds"], r["strict_summary"]["median_cagr_pct"]), reverse=True)

    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "input": {"rows": int(len(market)), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])}, "feature_count": len(cols), "folds": folds, "top_event": candidates[: int(cfg.top_event_candidates)], "top_strict": strict_rows, "leakage_guard": {"features_use_rows_at_or_before_t": True, "each_fold_thresholds_and_side_fit_before_eval_start": True, "strict_replay_uses_actual_ohlc_bar_by_bar": True, "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled"}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling sparse two-predicate setup miner")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=SparseSetupCfg.external_tolerance)
    p.add_argument("--window-size", type=int, default=SparseSetupCfg.window_size)
    p.add_argument("--horizons", default=SparseSetupCfg.horizons)
    p.add_argument("--quantiles", default=SparseSetupCfg.quantiles)
    p.add_argument("--folds-json", default="")
    p.add_argument("--min-train-rows", type=int, default=SparseSetupCfg.min_train_rows)
    p.add_argument("--min-fold-events", type=int, default=SparseSetupCfg.min_fold_events)
    p.add_argument("--max-fold-events", type=int, default=SparseSetupCfg.max_fold_events)
    p.add_argument("--min-positive-folds", type=int, default=SparseSetupCfg.min_positive_folds)
    p.add_argument("--top-event-candidates", type=int, default=SparseSetupCfg.top_event_candidates)
    p.add_argument("--max-strict-candidates", type=int, default=SparseSetupCfg.max_strict_candidates)
    p.add_argument("--leverage", type=float, default=SparseSetupCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=SparseSetupCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=SparseSetupCfg.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=SparseSetupCfg.entry_delay_bars)
    p.add_argument("--max-features", type=int, default=SparseSetupCfg.max_features)
    p.add_argument("--include-external-components", action="store_true", default=SparseSetupCfg.include_external_components)
    p.add_argument("--feature-include-regex", default=SparseSetupCfg.feature_include_regex)
    return p.parse_args()


def main() -> None:
    rep = run(SparseSetupCfg(**vars(parse_args())))
    for row in rep["top_strict"][:10]:
        print(json.dumps({"features": row["features"], "h": row["horizon"], "q": row["quantile"], "event_score": row["event_score"], "strict_summary": row["strict_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
