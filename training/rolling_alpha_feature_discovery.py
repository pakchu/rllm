"""Rolling prior-only alpha discovery for market/wave/external features.

This is a pre-LLM alpha miner. It searches for feature/horizon/quantile rules that
keep directional edge across chronological folds. Candidate selection is based on
prior-only expanding fits and fold-level event returns; strict OHLC replay is then
run only for top candidates.
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

from preprocessing.binance_aux_features import attach_binance_um_aux_features
from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.alpha_feature_backtest import FeatureRuleConfig, _forward_return, fit_rule, simulate_rule
from training.strict_bar_backtest import _trade_stats
from training.wave_feature_ridge_policy import build_wave_feature_frame


@dataclass(frozen=True)
class RollingAlphaCfg:
    input_csv: str
    output: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    binance_funding_csv: str = ""
    binance_premium_csv: str = ""
    binance_funding_tolerance: str = "12h"
    binance_premium_tolerance: str = "2h"
    window_size: int = 144
    horizons: str = "36,72,144,288"
    quantiles: str = "0.10,0.20,0.30"
    folds_json: str = ""
    min_train_rows: int = 20_000
    min_eval_events: int = 120
    top_event_candidates: int = 40
    leverage: float = 1.0
    max_strict_candidates: int = 12
    max_features: int = 0
    include_external_components: bool = False


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


def _event_stats(returns: list[float]) -> dict[str, Any]:
    xs = np.asarray([r for r in returns if np.isfinite(r)], dtype=float)
    if xs.size == 0:
        return {"n": 0, "mean_pct": 0.0, "t_stat": 0.0, "positive_rate": 0.0}
    std = float(np.std(xs, ddof=1)) if xs.size > 1 else 0.0
    t_stat = float(np.mean(xs) / (std / math.sqrt(xs.size))) if std > 0 else 0.0
    return {"n": int(xs.size), "mean_pct": float(np.mean(xs) * 100.0), "t_stat": t_stat, "positive_rate": float(np.mean(xs > 0.0))}


def _event_fold_eval(*, dates: pd.Series, values: np.ndarray, fwd: np.ndarray, fold: dict[str, str], rule: dict[str, Any], cost: float) -> dict[str, Any]:
    mask = np.asarray((dates >= pd.Timestamp(fold["eval_start"])) & (dates <= pd.Timestamp(fold["eval_end"])), dtype=bool)
    rets: list[float] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    for pos in np.flatnonzero(mask):
        v = float(values[pos])
        if not np.isfinite(v) or not np.isfinite(fwd[pos]):
            continue
        sig = 0
        if v >= float(rule["high_threshold"]):
            sig = 1 if rule["high_side"] == "LONG" else -1
        elif v <= float(rule["low_threshold"]):
            sig = 1 if rule["low_side"] == "LONG" else -1
        if sig == 0:
            continue
        side_counts["LONG" if sig > 0 else "SHORT"] += 1
        rets.append(float(sig) * float(fwd[pos]) - float(cost))
    stats = _event_stats(rets)
    stats["side_counts"] = side_counts
    return stats


def _candidate_score(folds: list[dict[str, Any]], min_eval_events: int) -> float:
    usable = [f for f in folds if int(f.get("n", 0)) >= int(min_eval_events)]
    if len(usable) < 5:
        return -1e9 + len(usable)
    means = np.asarray([float(f["mean_pct"]) for f in usable], dtype=float)
    ts = np.asarray([float(f["t_stat"]) for f in usable], dtype=float)
    positive_folds = int(np.sum(means > 0.0))
    worst = float(np.min(means))
    median = float(np.median(means))
    # Real alpha should survive bad folds; this intentionally punishes one-regime spikes.
    return positive_folds * 10.0 + median + min(2.0, float(np.median(ts))) + min(0.0, worst) * 2.0


def _strict_score(row: dict[str, Any]) -> float:
    folds = row.get("strict_folds", [])
    if not folds:
        return -1e9
    positive = 0
    ratios = []
    cagr = []
    mdds = []
    trades = 0
    for f in folds:
        sim = f["result"]["sim"]
        positive += int(float(sim["cagr_pct"]) > 0)
        ratios.append(float(sim["cagr_to_strict_mdd"]))
        cagr.append(float(sim["cagr_pct"]))
        mdds.append(float(sim["strict_mdd_pct"]))
        trades += int(sim["trade_entries"])
    return positive * 20.0 + float(np.median(ratios)) * 3.0 + min(float(np.median(cagr)), 50.0) - max(0.0, float(np.median(mdds)) - 15.0) * 2.0 + min(5.0, trades / 100.0)


def run(cfg: RollingAlphaCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(
            market,
            wave_trading_root=cfg.wave_trading_root,
            tolerance=cfg.external_tolerance,
            include_forex_components=bool(cfg.include_external_components),
        )
    if cfg.binance_funding_csv or cfg.binance_premium_csv:
        market = attach_binance_um_aux_features(
            market,
            funding_csv=cfg.binance_funding_csv or None,
            premium_csv=cfg.binance_premium_csv or None,
            funding_tolerance=cfg.binance_funding_tolerance,
            premium_tolerance=cfg.binance_premium_tolerance,
        )
    base = build_market_feature_frame(market, window_size=int(cfg.window_size))
    wave = build_wave_feature_frame(market, window=int(cfg.window_size))
    features = pd.concat([base.add_prefix("mkt__"), wave.add_prefix("wave__")], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    dates = pd.to_datetime(market["date"])
    folds = json.loads(cfg.folds_json) if cfg.folds_json else _default_folds()
    feature_cols = [c for c in features.columns if float(np.nanstd(features[c].to_numpy(dtype=float))) > 1e-12]
    if int(cfg.max_features) > 0:
        feature_cols = feature_cols[: int(cfg.max_features)]

    horizons = _parse_list(cfg.horizons, int)
    quantiles = _parse_list(cfg.quantiles, float)
    cost = (0.0004 + 0.0001) * 2.0 * float(cfg.leverage)
    event_rows: list[dict[str, Any]] = []
    fold_masks = []
    for fold in folds:
        eval_start = pd.Timestamp(fold["eval_start"])
        fit_end = eval_start - pd.Timedelta(seconds=1)
        fold_masks.append(
            {
                "fold": fold,
                "fit_end": fit_end,
                "train_base": np.asarray(dates <= fit_end, dtype=bool),
                "eval_base": np.asarray((dates >= eval_start) & (dates <= pd.Timestamp(fold["eval_end"])), dtype=bool),
            }
        )

    def fit_rule_fast(x_train: np.ndarray, y_train: np.ndarray, q: float) -> dict[str, Any]:
        q = float(np.clip(q, 0.01, 0.49))
        lo = float(np.quantile(x_train, q))
        hi = float(np.quantile(x_train, 1.0 - q))
        high_mean = float(np.mean(y_train[x_train >= hi])) if np.any(x_train >= hi) else 0.0
        low_mean = float(np.mean(y_train[x_train <= lo])) if np.any(x_train <= lo) else 0.0
        high_side = "LONG" if high_mean >= low_mean else "SHORT"
        return {
            "fit_n": int(x_train.size),
            "low_threshold": lo,
            "high_threshold": hi,
            "fit_low_mean_pct": low_mean * 100.0,
            "fit_high_mean_pct": high_mean * 100.0,
            "fit_high_minus_low_pct": (high_mean - low_mean) * 100.0,
            "high_side": high_side,
            "low_side": "SHORT" if high_side == "LONG" else "LONG",
        }

    for horizon in horizons:
        fwd = _forward_return(market["open"].astype(float), horizon=int(horizon), entry_delay_bars=1)
        finite_y = np.isfinite(fwd)
        for feature in feature_cols:
            values = features[feature].to_numpy(dtype=float)
            finite_x = np.isfinite(values)
            for q in quantiles:
                fold_rows: list[dict[str, Any]] = []
                for fm in fold_masks:
                    train_mask = fm["train_base"] & finite_x & finite_y
                    train_n = int(np.sum(train_mask))
                    fold = fm["fold"]
                    if train_n < int(cfg.min_train_rows):
                        fold_rows.append({"fold": fold["name"], "n": 0, "skip": "not_enough_train", "train_n": train_n})
                        continue
                    x_train = values[train_mask]
                    y_train = fwd[train_mask]
                    rule = fit_rule_fast(x_train, y_train, float(q))
                    eval_mask = fm["eval_base"] & finite_x & finite_y
                    xv = values[eval_mask]
                    yv = fwd[eval_mask]
                    sig = np.zeros_like(xv, dtype=float)
                    if rule["high_side"] == "LONG":
                        sig[xv >= float(rule["high_threshold"])] = 1.0
                        sig[xv <= float(rule["low_threshold"])] = -1.0
                    else:
                        sig[xv >= float(rule["high_threshold"])] = -1.0
                        sig[xv <= float(rule["low_threshold"])] = 1.0
                    active = sig != 0.0
                    rets = (sig[active] * yv[active] - cost).astype(float)
                    st = _event_stats(rets.tolist())
                    st["side_counts"] = {"LONG": int(np.sum(sig[active] > 0.0)), "SHORT": int(np.sum(sig[active] < 0.0))}
                    st.update({"fold": fold["name"], "train_n": train_n, "rule": rule})
                    fold_rows.append(st)
                score = _candidate_score(fold_rows, int(cfg.min_eval_events))
                event_rows.append({"feature": feature, "horizon": int(horizon), "quantile": float(q), "event_score": float(score), "event_folds": fold_rows})

    event_rows.sort(key=lambda r: float(r["event_score"]), reverse=True)
    strict_rows: list[dict[str, Any]] = []
    for row in event_rows[: int(cfg.max_strict_candidates)]:
        values = features[row["feature"]].to_numpy(dtype=float)
        fwd = _forward_return(market["open"].astype(float), horizon=int(row["horizon"]), entry_delay_bars=1)
        strict_folds = []
        for fold in folds:
            fit_end = str(pd.Timestamp(fold["eval_start"]) - pd.Timedelta(seconds=1))
            fold_cfg = FeatureRuleConfig(
                input_csv=cfg.input_csv,
                output="",
                feature=row["feature"],
                horizon=int(row["horizon"]),
                fit_start=str(dates.iloc[0]),
                fit_end=fit_end,
                eval_start=fold["eval_start"],
                eval_end=fold["eval_end"],
                quantile=float(row["quantile"]),
                window_size=int(cfg.window_size),
                entry_delay_bars=1,
                leverage=float(cfg.leverage),
                wave_trading_root=cfg.wave_trading_root,
                external_tolerance=cfg.external_tolerance,
                binance_funding_csv=cfg.binance_funding_csv,
                binance_premium_csv=cfg.binance_premium_csv,
                binance_funding_tolerance=cfg.binance_funding_tolerance,
                binance_premium_tolerance=cfg.binance_premium_tolerance,
            )
            try:
                rule = fit_rule(dates=dates, feature_values=values, forward_returns=fwd, cfg=fold_cfg)
                result = simulate_rule(market=market, feature_values=values, dates=dates, rule=rule, cfg=fold_cfg)
            except Exception as exc:
                strict_folds.append({"fold": fold["name"], "error": str(exc)})
                continue
            strict_folds.append({"fold": fold["name"], "rule": rule, "result": result})
        strict_row = dict(row)
        strict_row["strict_folds"] = strict_folds
        strict_row["strict_score"] = _strict_score(strict_row)
        strict_rows.append(strict_row)
    strict_rows.sort(key=lambda r: float(r["strict_score"]), reverse=True)

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": int(len(market)), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "folds": folds,
        "feature_count": len(feature_cols),
        "top_event": event_rows[: int(cfg.top_event_candidates)],
        "top_strict": strict_rows,
        "leakage_guard": {
            "features_use_rows_at_or_before_t": True,
            "each_fold_rule_fit_uses_only_dates_before_eval_start": True,
            "strict_replay_uses_actual_ohlc_bar_by_bar": True,
            "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled",
            "binance_aux_join": "backward_asof_no_future" if (cfg.binance_funding_csv or cfg.binance_premium_csv) else "disabled",
            "premium_index_uses_close_time_when_available": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling prior-only alpha feature discovery")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=RollingAlphaCfg.external_tolerance)
    p.add_argument("--binance-funding-csv", default="")
    p.add_argument("--binance-premium-csv", default="")
    p.add_argument("--binance-funding-tolerance", default=RollingAlphaCfg.binance_funding_tolerance)
    p.add_argument("--binance-premium-tolerance", default=RollingAlphaCfg.binance_premium_tolerance)
    p.add_argument("--window-size", type=int, default=RollingAlphaCfg.window_size)
    p.add_argument("--horizons", default=RollingAlphaCfg.horizons)
    p.add_argument("--quantiles", default=RollingAlphaCfg.quantiles)
    p.add_argument("--folds-json", default="")
    p.add_argument("--min-train-rows", type=int, default=RollingAlphaCfg.min_train_rows)
    p.add_argument("--min-eval-events", type=int, default=RollingAlphaCfg.min_eval_events)
    p.add_argument("--top-event-candidates", type=int, default=RollingAlphaCfg.top_event_candidates)
    p.add_argument("--leverage", type=float, default=RollingAlphaCfg.leverage)
    p.add_argument("--max-strict-candidates", type=int, default=RollingAlphaCfg.max_strict_candidates)
    p.add_argument("--max-features", type=int, default=RollingAlphaCfg.max_features)
    p.add_argument("--include-external-components", action="store_true", default=RollingAlphaCfg.include_external_components)
    return p.parse_args()


def main() -> None:
    rep = run(RollingAlphaCfg(**vars(parse_args())))
    for row in rep["top_strict"][:10]:
        compact = []
        for f in row["strict_folds"]:
            if "result" not in f:
                compact.append({"fold": f.get("fold"), "error": f.get("error")})
                continue
            sim = f["result"]["sim"]
            compact.append({"fold": f["fold"], "cagr": sim["cagr_pct"], "mdd": sim["strict_mdd_pct"], "ratio": sim["cagr_to_strict_mdd"], "trades": sim["trade_entries"]})
        print(json.dumps({"feature": row["feature"], "horizon": row["horizon"], "q": row["quantile"], "event_score": row["event_score"], "strict_score": row["strict_score"], "folds": compact}, ensure_ascii=False))


if __name__ == "__main__":
    main()
