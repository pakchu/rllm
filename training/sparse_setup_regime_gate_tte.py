"""Train/test/eval regime gate for sparse setup candidate events.

Sparse setup candidates are treated as weak event proposals.  A leakage-safe ridge
model learns from past regime/context features to score candidate events, then a
score quantile is selected on test only and evaluated on untouched eval folds.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.price_action_extreme_feature_audit import build_extreme_bar_features
from training.sparse_setup_ensemble_audit import EnsembleCfg, _candidate_events, _load_market, _simulate_events
from training.sparse_setup_train_test_eval_validator import _load_folds
from training.sparse_setup_walkforward_selector import WalkForwardCfg, _features as _sparse_features
from training.wave_feature_ridge_policy import build_wave_feature_frame


@dataclass(frozen=True)
class RegimeGateTTECfg:
    sparse_report: str
    market_csv: str
    output: str
    train_folds_json: str
    test_folds_json: str
    eval_folds_json: str
    candidate_limit: int = 80
    ridge_alpha: float = 100.0
    quantiles: str = "0.50,0.60,0.70,0.80,0.85,0.90,0.95"
    min_test_trades: int = 30
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    trade_stop_loss_pct: float = 6.0
    trade_take_profit_pct: float = 6.0
    setup_sizing: str = "prior_sharpe"
    window_size: int = 144
    include_price_action_extremes: bool = True
    price_action_lookbacks: str = "36,72,144,288,576,2016"
    feature_include_regex: str = ""
    max_features: int = 96
    target: str = "utility"  # net | utility
    include_failure_regime_classes: bool = True


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _fold_names(folds: list[dict[str, str]]) -> set[str]:
    return {str(f["name"]) for f in folds}


def _feature_frame(market: pd.DataFrame, cfg: RegimeGateTTECfg) -> pd.DataFrame:
    parts = [
        build_market_feature_frame(market, window_size=int(cfg.window_size)).add_prefix("mkt__"),
        build_wave_feature_frame(market, window=int(cfg.window_size)).add_prefix("wave__"),
    ]
    if bool(cfg.include_price_action_extremes):
        lookbacks = tuple(int(x.strip()) for x in str(cfg.price_action_lookbacks).split(",") if x.strip())
        parts.append(build_extreme_bar_features(market, lookbacks).add_prefix("pa__"))
    frame = pd.concat(parts, axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0).loc[:, lambda df: ~df.columns.duplicated(keep="last")]
    if bool(cfg.include_failure_regime_classes):
        frame = append_failure_regime_classes(frame)
    return frame


def _col(features: pd.DataFrame, name: str) -> pd.Series:
    if name in features.columns:
        return features[name].astype(float)
    return pd.Series(np.zeros(len(features), dtype=float), index=features.index)


def append_failure_regime_classes(features: pd.DataFrame) -> pd.DataFrame:
    """Append explicit past-only failure regime class flags.

    These are intentionally simple, fixed-threshold descriptors rather than
    distribution-fitted gates so they do not learn from eval-period outcomes.
    They expose nonlinear regimes to the ridge gate: chop/compression, macro
    impulse, premium shock, volatility transition, and range-extreme pressure.
    """
    out = features.copy()
    trend24 = _col(out, "mkt__trend_24")
    trend96 = _col(out, "mkt__trend_96")
    range_vol = _col(out, "mkt__range_vol")
    range_pos = _col(out, "mkt__range_pos")
    dxy_z = _col(out, "mkt__dxy_zscore")
    dxy_mom = _col(out, "mkt__dxy_momentum")
    kimchi_z = _col(out, "mkt__kimchi_premium_zscore")
    kimchi_chg = _col(out, "mkt__kimchi_premium_change")
    usdkrw_z = _col(out, "mkt__usdkrw_zscore")
    usdkrw_mom = _col(out, "mkt__usdkrw_momentum")
    vol_spike = _col(out, "wave__vol_spike")
    vol_regime = _col(out, "wave__vol_regime")
    flow_mom = _col(out, "wave__flow_mom")
    pa144_pos = _col(out, "pa__pa_ext_144_range_pos")
    pa288_pos = _col(out, "pa__pa_ext_288_range_pos")
    pa144_to_high = _col(out, "pa__pa_ext_144_to_max_high_pct")
    pa144_to_low = _col(out, "pa__pa_ext_144_to_min_low_pct")
    pa288_overlap = _col(out, "pa__pa_ext_288_extreme_bar_overlap_pct")
    out["fr__chop_compression"] = ((trend24.abs() < 0.0035) & (range_vol < 0.018)).astype(float)
    out["fr__chop_midrange"] = ((trend24.abs() < 0.004) & (range_pos.abs() < 0.35)).astype(float)
    out["fr__trend_conflict"] = ((np.sign(trend24) != np.sign(trend96)) & (trend24.abs() > 0.0025) & (trend96.abs() > 0.006)).astype(float)
    out["fr__volatility_transition"] = ((vol_spike > 2.0) | (vol_regime > 3.0)).astype(float)
    out["fr__dxy_impulse_up"] = ((dxy_z > 1.8) | (dxy_mom > 0.0032)).astype(float)
    out["fr__dxy_impulse_down"] = ((dxy_z < -1.8) | (dxy_mom < -0.0032)).astype(float)
    out["fr__usdkrw_impulse"] = ((usdkrw_z.abs() > 1.8) | (usdkrw_mom.abs() > 0.004)).astype(float)
    out["fr__kimchi_shock"] = ((kimchi_z.abs() > 1.8) | (kimchi_chg.abs() > 0.007)).astype(float)
    out["fr__flow_dislocation"] = (flow_mom.abs() > 0.065).astype(float)
    out["fr__near_upper_extreme"] = ((pa144_pos > 0.90) | (pa288_pos > 0.90) | (pa144_to_high > -0.002)).astype(float)
    out["fr__near_lower_extreme"] = ((pa144_pos < 0.10) | (pa288_pos < 0.10) | (pa144_to_low < 0.0025)).astype(float)
    out["fr__extreme_overlap_compression"] = (pa288_overlap > -0.006).astype(float)
    out["fr__macro_shock_cluster"] = ((out["fr__dxy_impulse_up"] + out["fr__usdkrw_impulse"] + out["fr__kimchi_shock"]) >= 2.0).astype(float)
    out["fr__chop_or_conflict"] = ((out["fr__chop_compression"] + out["fr__chop_midrange"] + out["fr__trend_conflict"]) >= 1.0).astype(float)
    out["fr__breakout_failure_risk"] = ((out["fr__volatility_transition"] > 0) & ((out["fr__near_upper_extreme"] > 0) | (out["fr__near_lower_extreme"] > 0))).astype(float)
    return out


def _feature_names(features: pd.DataFrame, cfg: RegimeGateTTECfg) -> list[str]:
    deny = ("available", "external_any")
    names = []
    for c in features.columns:
        if any(x in c for x in deny):
            continue
        if float(np.nanstd(features[c].to_numpy(dtype=float))) <= 1e-12:
            continue
        names.append(str(c))
    if str(cfg.feature_include_regex).strip():
        pat = re.compile(str(cfg.feature_include_regex))
        names = [n for n in names if pat.search(n)]
    preferred = [n for n in names if n.startswith(("fr__", "mkt__htf_", "mkt__dxy_", "mkt__kimchi_", "mkt__usdkrw_", "wave__mom_", "wave__flow_", "wave__cvd_", "pa__pa_ext_"))]
    names = preferred or names
    return names[: max(1, int(cfg.max_features))]


def _event_reward(ev: dict[str, Any], market: pd.DataFrame, cfg: RegimeGateTTECfg) -> dict[str, float]:
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    signal_pos = int(ev["signal_pos"])
    entry_pos = signal_pos + int(cfg.entry_delay_bars)
    horizon = int(ev["horizon"])
    exit_pos = entry_pos + horizon
    if entry_pos >= len(market) - 1 or exit_pos >= len(market):
        return {"net_return_pct": 0.0, "mae_pct": 0.0, "mfe_pct": 0.0, "utility": -1.0}
    side = int(ev["side"])
    entry = float(opens[entry_pos])
    if entry <= 0.0:
        return {"net_return_pct": 0.0, "mae_pct": 0.0, "mfe_pct": 0.0, "utility": -1.0}
    end = float(opens[exit_pos])
    raw = ((end - entry) / entry) if side > 0 else ((entry - end) / entry)
    path_high = highs[entry_pos:exit_pos]
    path_low = lows[entry_pos:exit_pos]
    if side > 0:
        adverse = float(np.min((path_low - entry) / entry)) if path_low.size else 0.0
        favorable = float(np.max((path_high - entry) / entry)) if path_high.size else 0.0
    else:
        adverse = float(np.min((entry - path_high) / entry)) if path_high.size else 0.0
        favorable = float(np.max((entry - path_low) / entry)) if path_low.size else 0.0
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * 2.0 * float(cfg.leverage)
    net = raw * float(cfg.leverage) - cost
    mae = abs(min(0.0, adverse * float(cfg.leverage)))
    mfe = max(0.0, favorable * float(cfg.leverage))
    utility = net - 0.75 * mae + 0.10 * mfe
    return {"net_return_pct": net * 100.0, "mae_pct": mae * 100.0, "mfe_pct": mfe * 100.0, "utility": utility * 100.0}


def _build_events(cfg: RegimeGateTTECfg, sparse: dict[str, Any], market: pd.DataFrame, dates: pd.Series) -> list[dict[str, Any]]:
    wf = WalkForwardCfg(
        sparse_report=cfg.sparse_report,
        market_csv=cfg.market_csv,
        output=cfg.output,
        window_size=cfg.window_size,
        candidate_limit=cfg.candidate_limit,
        max_ensemble_size=1,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        entry_delay_bars=cfg.entry_delay_bars,
        trade_stop_loss_pct=cfg.trade_stop_loss_pct,
        trade_take_profit_pct=cfg.trade_take_profit_pct,
        setup_sizing=cfg.setup_sizing,
        include_price_action_extremes=cfg.include_price_action_extremes,
        price_action_lookbacks=cfg.price_action_lookbacks,
    )
    setup_features = _sparse_features(market, wf)
    ecfg = EnsembleCfg(
        cfg.sparse_report,
        cfg.market_csv,
        cfg.output,
        window_size=cfg.window_size,
        candidate_limit=cfg.candidate_limit,
        max_ensemble_size=1,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        entry_delay_bars=cfg.entry_delay_bars,
        trade_stop_loss_pct=cfg.trade_stop_loss_pct,
        trade_take_profit_pct=cfg.trade_take_profit_pct,
        setup_sizing=cfg.setup_sizing,
        include_price_action_extremes=cfg.include_price_action_extremes,
        price_action_lookbacks=cfg.price_action_lookbacks,
    )
    events: list[dict[str, Any]] = []
    for i, cand in enumerate(sparse.get("top_strict", [])[: int(cfg.candidate_limit)]):
        for ev in _candidate_events(cand=dict(cand, _candidate_index=i), report=sparse, dates=dates, features=setup_features, market=market, cfg=ecfg):
            ev = dict(ev)
            ev["reward"] = _event_reward(ev, market, cfg)
            events.append(ev)
    events.sort(key=lambda e: (int(e["signal_pos"]), int(e.get("candidate_index", -1)), str(e.get("candidate_key", ""))))
    return events


def _matrix(rows: list[dict[str, Any]], features: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), len(names) * 2 + 8), dtype=float)
    y = np.zeros(len(rows), dtype=float)
    for i, ev in enumerate(rows):
        pos = int(ev["signal_pos"])
        side = 1.0 if int(ev.get("side", 0)) > 0 else -1.0
        vals = np.asarray([float(features[n].iloc[pos]) for n in names], dtype=float)
        x[i, : len(names)] = vals
        x[i, len(names) : len(names) * 2] = vals * side
        base = len(names) * 2
        prior_std = float(ev.get("prior_std_ret", 0.0) or 0.0)
        prior_mean = float(ev.get("prior_mean_ret", 0.0) or 0.0)
        x[i, base : base + 8] = [
            1.0 if side > 0 else 0.0,
            1.0 if side < 0 else 0.0,
            side,
            float(ev.get("horizon", 0) or 0) / 288.0,
            float(ev.get("prior_n", 0) or 0) / 1000.0,
            prior_mean,
            prior_std,
            prior_mean / prior_std if prior_std > 1e-12 else 0.0,
        ]
        reward = ev.get("reward", {})
        y[i] = float(reward.get("utility", 0.0) or 0.0)
    return x, y


def _target(rows: list[dict[str, Any]], cfg: RegimeGateTTECfg) -> np.ndarray:
    key = "net_return_pct" if str(cfg.target) == "net" else "utility"
    return np.asarray([float(r.get("reward", {}).get(key, 0.0) or 0.0) for r in rows], dtype=float)


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0) if len(x) else np.zeros(x.shape[1], dtype=float)
    sd = x.std(axis=0) if len(x) else np.ones(x.shape[1], dtype=float)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (x - mu) / sd, mu, sd


def _standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xb = np.c_[np.ones(len(x)), x]
    reg = np.eye(xb.shape[1], dtype=float) * float(alpha)
    reg[0, 0] = 0.0
    return np.linalg.pinv(xb.T @ xb + reg) @ xb.T @ y


def _predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(x)), x] @ w if len(x) else np.zeros(0, dtype=float)


def _select_best_events(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    best: dict[int, tuple[float, dict[str, Any]]] = {}
    for row, score in zip(rows, scores):
        pos = int(row["signal_pos"])
        cur = best.get(pos)
        if cur is None or float(score) > cur[0]:
            best[pos] = (float(score), row)
    out = [dict(row, pred_score=float(score)) for _pos, (score, row) in sorted(best.items()) if float(score) >= float(threshold)]
    return out


def _score_period(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float, market: pd.DataFrame, cfg: RegimeGateTTECfg) -> dict[str, Any]:
    chosen = _select_best_events(rows, scores, threshold)
    ecfg = EnsembleCfg(cfg.sparse_report, cfg.market_csv, cfg.output, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, entry_delay_bars=cfg.entry_delay_bars, trade_stop_loss_pct=cfg.trade_stop_loss_pct, trade_take_profit_pct=cfg.trade_take_profit_pct, setup_sizing=cfg.setup_sizing)
    sim = _simulate_events(chosen, dates=pd.to_datetime(market["date"]), market=market, cfg=ecfg)
    return {k: v for k, v in sim.items() if k != "executed"} | {"selected_events": len(chosen)}


def _selection_score(sim: dict[str, Any], min_trades: int) -> float:
    trades = int(sim.get("trade_entries", 0) or 0)
    cagr = float(sim.get("cagr_pct", -100.0) or -100.0)
    mdd = float(sim.get("strict_mdd_pct", 100.0) or 100.0)
    ratio = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
    if trades < int(min_trades) or cagr <= 0.0:
        capped_cagr = min(100.0, max(-100.0, cagr))
        return -1000.0 + trades / max(1.0, float(min_trades)) + capped_cagr / 100.0 - min(100.0, max(0.0, mdd)) / 100.0
    return ratio + min(2.0, trades / 100.0) - max(0.0, mdd - 15.0) / 10.0


def run(cfg: RegimeGateTTECfg) -> dict[str, Any]:
    train_folds = _load_folds(cfg.train_folds_json)
    test_folds = _load_folds(cfg.test_folds_json)
    eval_folds = _load_folds(cfg.eval_folds_json)
    sparse = json.loads(Path(cfg.sparse_report).read_text())
    sparse = dict(sparse)
    sparse["folds"] = sorted(train_folds + test_folds + eval_folds, key=lambda f: f["eval_start"])
    market = _load_market(cfg.market_csv)
    dates = pd.to_datetime(market["date"])
    regime_features = _feature_frame(market, cfg)
    names = _feature_names(regime_features, cfg)
    events = _build_events(cfg, sparse, market, dates)
    train_names, test_names, eval_names = _fold_names(train_folds), _fold_names(test_folds), _fold_names(eval_folds)
    train_rows = [e for e in events if str(e.get("fold")) in train_names]
    test_rows = [e for e in events if str(e.get("fold")) in test_names]
    eval_rows = [e for e in events if str(e.get("fold")) in eval_names]

    xtr, _ = _matrix(train_rows, regime_features, names)
    ytr = _target(train_rows, cfg)
    xtrz, mu, sd = _standardize_fit(xtr)
    w = _fit_ridge(xtrz, ytr, cfg.ridge_alpha)
    train_scores = _predict(xtrz, w)
    xtest, _ = _matrix(test_rows, regime_features, names)
    test_scores = _predict(_standardize_apply(xtest, mu, sd), w)

    candidates: list[dict[str, Any]] = []
    for q in _parse_floats(cfg.quantiles):
        threshold = float(np.quantile(train_scores, q)) if len(train_scores) else 1e9
        train_res = _score_period(train_rows, train_scores, threshold, market, cfg)
        test_res = _score_period(test_rows, test_scores, threshold, market, cfg)
        test_sim = test_res["sim"]
        score = _selection_score(test_sim, int(cfg.min_test_trades))
        candidates.append({"q": q, "threshold": threshold, "score": score, "train": train_res, "test": test_res})
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = candidates[0] if candidates else {"q": 1.0, "threshold": 1e9, "score": -1e9}

    train_test_rows = train_rows + test_rows
    xtt, _ = _matrix(train_test_rows, regime_features, names)
    ytt = _target(train_test_rows, cfg)
    xttz, mu2, sd2 = _standardize_fit(xtt)
    w2 = _fit_ridge(xttz, ytt, cfg.ridge_alpha)
    tt_scores = _predict(xttz, w2)
    eval_threshold = float(np.quantile(tt_scores, float(selected["q"]))) if len(tt_scores) else 1e9
    xeval, _ = _matrix(eval_rows, regime_features, names)
    eval_scores = _predict(_standardize_apply(xeval, mu2, sd2), w2)
    eval_res = _score_period(eval_rows, eval_scores, eval_threshold, market, cfg)
    all_selected = {
        "train": selected.get("train"),
        "test": selected.get("test"),
        "eval": eval_res,
    }
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": {"events": len(events), "train": len(train_rows), "test": len(test_rows), "eval": len(eval_rows)},
        "features": {"numeric": len(names), "expanded": len(names) * 2 + 8, "names": names},
        "selected_by_test": {"q": selected.get("q"), "threshold": selected.get("threshold"), "score": selected.get("score")},
        "top_test_candidates": candidates[:10],
        "final_eval_threshold": eval_threshold,
        "periods": all_selected,
        "leakage_guard": {
            "fit_uses_train_rows_only_for_test_selection": True,
            "test_labels_select_quantile_only": True,
            "eval_not_used_for_fit_or_selection": True,
            "final_eval_refits_on_train_plus_test_then_applies_selected_quantile": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime-aware sparse setup event gate with train/test/eval validation")
    for field in RegimeGateTTECfg.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        required = field.default is MISSING and field.default_factory is MISSING
        if field.name == "include_price_action_extremes":
            default = True if required else bool(field.default)
            p.add_argument(name, action=argparse.BooleanOptionalAction, default=default, required=False)
        else:
            p.add_argument(name, default=None if required else field.default, required=required)
    ns = p.parse_args()
    ints = {"candidate_limit", "min_test_trades", "entry_delay_bars", "window_size", "max_features"}
    floats = {"ridge_alpha", "leverage", "fee_rate", "slippage_rate", "trade_stop_loss_pct", "trade_take_profit_pct"}
    data = vars(ns)
    for k in ints:
        data[k] = int(data[k])
    for k in floats:
        data[k] = float(data[k])
    data["include_price_action_extremes"] = str(data["include_price_action_extremes"]).lower() not in {"false", "0", "no"} if not isinstance(data["include_price_action_extremes"], bool) else data["include_price_action_extremes"]
    data["include_failure_regime_classes"] = str(data["include_failure_regime_classes"]).lower() not in {"false", "0", "no"} if not isinstance(data["include_failure_regime_classes"], bool) else data["include_failure_regime_classes"]
    return argparse.Namespace(**data)


def main() -> None:
    rep = run(RegimeGateTTECfg(**vars(parse_args())))
    print(json.dumps({"selected_by_test": rep["selected_by_test"], "periods": {k: v["sim"] for k, v in rep["periods"].items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
