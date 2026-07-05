"""Rolling validation for composite long-regime activity/flow score gates.

This is the next step after the single activity-gate scan: keep the same
leak-safe long entry skeleton, but replace one brittle activity threshold with a
train-fitted composite score that tries to separate reboundable pullbacks from
low-interest bear-continuation traps.

Candidate skeleton per fold:
  pb30_funding entry + premium high + funding non-negative + score gate

All quantile thresholds and score standardization statistics are fitted on each
fold's train window only. Validation windows are report-only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.long_regime_gate_scan import LongRegimeGateConfig, _entry_active, _gate_active
from training.long_regime_combo_scan import _load_market, _split_mask, _strict_long_sim
from training.long_regime_interest_gate_validation import build_interest_features
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LongRegimeScoreGateConfig(LongRegimeGateConfig):
    score_quantiles: str = "0.5,0.6,0.7,0.8"
    score_variants: str = "activity,activity_flow,activity_flow_htf,activity_flow_deriv_htf"
    entry_quantile: float = 0.8
    premium_quantile: float = 0.8
    funding_quantile: float = 0.7
    hold_bars: str = "144"
    stride_bars: str = "12"
    leverage: float = 0.5
    folds: str = (
        "2020-01-01:2023-01-01:2023-01-01:2024-01-01,"
        "2020-01-01:2024-01-01:2024-01-01:2025-01-01,"
        "2020-01-01:2025-01-01:2025-01-01:2026-01-01,"
        "2020-01-01:2026-01-01:2026-01-01:2026-06-02"
    )
    fold_preset: str = "explicit"
    rolling_train_start: str = "2020-01-01"
    rolling_validation_start: str = "2023-01-01"
    rolling_validation_end: str = "2026-06-02"
    rolling_months: int = 6


def _parse_list(raw: str, cast: Any) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_fold(raw: str) -> tuple[str, str, str, str]:
    parts = [x.strip() for x in raw.split(":")]
    if len(parts) != 4:
        raise ValueError(f"fold must be train_start:train_end:val_start:val_end, got {raw!r}")
    return tuple(parts)  # type: ignore[return-value]


def _build_folds(cfg: LongRegimeScoreGateConfig) -> list[tuple[str, str, str, str]]:
    if str(cfg.fold_preset).strip().lower() in ("", "explicit"):
        return [_parse_fold(x) for x in _parse_list(cfg.folds, str)]
    if str(cfg.fold_preset).strip().lower() != "anchored_rolling":
        raise ValueError(f"unknown fold preset: {cfg.fold_preset!r}")
    start = pd.Timestamp(cfg.rolling_validation_start)
    end = pd.Timestamp(cfg.rolling_validation_end)
    months = int(cfg.rolling_months)
    if months <= 0:
        raise ValueError("rolling_months must be positive")
    folds: list[tuple[str, str, str, str]] = []
    val_start = start
    while val_start < end:
        val_end = min(val_start + pd.DateOffset(months=months), end)
        folds.append((str(pd.Timestamp(cfg.rolling_train_start).date()), str(val_start.date()), str(val_start.date()), str(val_end.date())))
        val_start = val_end
    return folds


def _train_zscore(series: pd.Series, train_mask: np.ndarray) -> tuple[pd.Series, dict[str, float]] | None:
    arr = series.to_numpy(float)
    ref = arr[train_mask & np.isfinite(arr)]
    if ref.size < 500:
        return None
    mean = float(np.mean(ref))
    std = float(np.std(ref))
    if std <= 1e-12:
        return None
    z = ((series.astype(float) - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z, {"mean": mean, "std": std}


def _past_return(market: pd.DataFrame, bars: int) -> pd.Series:
    close = market["close"].astype(float)
    return close.pct_change(bars).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_score_frame(market: pd.DataFrame, features: pd.DataFrame, interest: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=market.index)
    for col in interest.columns:
        out[col] = interest[col]
    # Past-only HTF pressure proxies. Positive values mean the market is not in a
    # severe higher-timeframe slide; negative values penalize long pullback bets.
    out["ret_1d"] = _past_return(market, 288)
    out["ret_3d"] = _past_return(market, 864)
    out["ret_1w"] = _past_return(market, 2016)
    out["bear_pressure"] = -np.minimum(out["ret_1d"], 0.0) - np.minimum(out["ret_3d"], 0.0) - np.minimum(out["ret_1w"], 0.0)
    # Derivative context from already past-aligned feature builder.
    out["premium_index_zscore"] = features.get("premium_index_zscore", pd.Series(0.0, index=market.index)).astype(float)
    out["funding_zscore"] = features.get("funding_zscore", pd.Series(0.0, index=market.index)).astype(float)
    out["taker_imbalance"] = features.get("taker_imbalance", pd.Series(0.0, index=market.index)).astype(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _score_variant(raw: pd.DataFrame, train_mask: np.ndarray, variant: str) -> tuple[pd.Series, dict[str, Any]] | None:
    # Equal-weight, interpretable score. Every component is standardized using
    # train-only statistics before fold validation.
    variant_cols: dict[str, float]
    if variant == "activity":
        variant_cols = {
            "quote_vol_rel_1d_30d": 1.0,
            "quote_vol_z_1d": 1.0,
            "trades_rel_1d_30d": 1.0,
        }
    elif variant == "activity_flow":
        variant_cols = {
            "quote_vol_rel_1d_30d": 1.0,
            "quote_vol_z_1d": 0.75,
            "dollar_flow_rel_4h_30d": 1.0,
            "taker_imbalance": 0.5,
        }
    elif variant == "activity_flow_htf":
        variant_cols = {
            "quote_vol_rel_1d_30d": 1.0,
            "quote_vol_z_1d": 0.75,
            "dollar_flow_rel_4h_30d": 1.0,
            "taker_imbalance": 0.5,
            "bear_pressure": -0.75,
        }
    elif variant == "activity_flow_deriv_htf":
        variant_cols = {
            "quote_vol_rel_1d_30d": 1.0,
            "quote_vol_z_1d": 0.75,
            "dollar_flow_rel_4h_30d": 1.0,
            "taker_imbalance": 0.5,
            "premium_index_zscore": 0.35,
            "funding_zscore": 0.25,
            "bear_pressure": -0.75,
        }
    else:
        raise ValueError(f"unknown score variant: {variant}")

    score = pd.Series(0.0, index=raw.index)
    stats: dict[str, Any] = {"variant": variant, "components": []}
    total_abs = 0.0
    for col, weight in variant_cols.items():
        if col not in raw.columns:
            return None
        z = _train_zscore(raw[col], train_mask)
        if z is None:
            return None
        comp, st = z
        score = score + float(weight) * comp
        total_abs += abs(float(weight))
        stats["components"].append({"feature": col, "weight": float(weight), **st})
    if total_abs <= 0:
        return None
    score = (score / total_abs).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return score, stats


def _fit_score_gate(score: pd.Series, train_mask: np.ndarray, q: float) -> tuple[np.ndarray, dict[str, Any]] | None:
    arr = score.to_numpy(float)
    ref = arr[train_mask & np.isfinite(arr)]
    if ref.size < 500 or float(np.nanstd(ref)) <= 1e-12:
        return None
    threshold = float(np.quantile(ref, q))
    return (arr >= threshold) & np.isfinite(arr), {"op": "ge", "quantile": float(q), "threshold": threshold}


def _run_fold(
    *,
    market: pd.DataFrame,
    features: pd.DataFrame,
    score_raw: pd.DataFrame,
    cfg: LongRegimeScoreGateConfig,
    variant: str,
    score_quantile: float,
    fold: tuple[str, str, str, str],
    hold: int,
    stride: int,
) -> dict[str, Any] | None:
    train_start, train_end, val_start, val_end = fold
    dates = pd.to_datetime(market["date"])
    train_mask = _split_mask(dates, train_start, train_end)
    val_mask = _split_mask(dates, val_start, val_end)
    entry = _entry_active(features, rule="pb30_funding", train_mask=train_mask, quantile=float(cfg.entry_quantile))
    premium = _gate_active(features, feature="premium_index_zscore", op="ge", train_mask=train_mask, quantile=float(cfg.premium_quantile))
    funding = _gate_active(features, feature="funding_zscore", op="ge", train_mask=train_mask, quantile=float(cfg.funding_quantile))
    score_pack = _score_variant(score_raw, train_mask, variant)
    if entry is None or premium is None or funding is None or score_pack is None:
        return None
    score, score_stats = score_pack
    score_gate = _fit_score_gate(score, train_mask, score_quantile)
    if score_gate is None:
        return None
    entry_mask, entry_spec = entry
    premium_mask, premium_spec = premium
    funding_mask, funding_spec = funding
    score_mask, score_spec = score_gate
    active = entry_mask & premium_mask & funding_mask & score_mask
    positions = np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - hold - int(cfg.entry_delay_bars) - 1), stride, dtype=np.int64)
    rows = {}
    for split, mask in (("train", train_mask), ("validation", val_mask)):
        p = positions[active[positions] & mask[positions]]
        sim, returns = _strict_long_sim(
            p,
            market=market,
            hold_bars=hold,
            entry_delay_bars=int(cfg.entry_delay_bars),
            leverage=float(cfg.leverage),
            fee_rate=float(cfg.fee_rate),
            slippage_rate=float(cfg.slippage_rate),
        )
        rows[split] = {"sim": sim, "trade_stats": _trade_stats(returns), "candidate_count": int(len(p))}
    return {
        "fold": {"train_start": train_start, "train_end": train_end, "validation_start": val_start, "validation_end": val_end},
        "score_gate": {"variant": variant, **score_spec},
        "score_stats": score_stats,
        "entry_spec": entry_spec,
        "base_gates": [premium_spec, funding_spec],
        **rows,
    }


def _score_fold_result(folds: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [f["validation"]["sim"] for f in folds]
    valid = [v for v in vals if int(v["trade_entries"]) > 0]
    if not valid:
        return {"folds": len(folds), "positive_folds": 0, "min_ratio": -999.0, "median_ratio": -999.0, "total_validation_trades": 0}
    ratios = [float(v["cagr_to_strict_mdd"]) for v in valid]
    return {
        "folds": len(folds),
        "positive_folds": sum(1 for v in valid if float(v["cagr_pct"]) > 0.0),
        "min_ratio": float(min(ratios)),
        "median_ratio": float(np.median(ratios)),
        "mean_ratio": float(np.mean(ratios)),
        "total_validation_trades": int(sum(int(v["trade_entries"]) for v in valid)),
        "max_validation_mdd": float(max(float(v["strict_mdd_pct"]) for v in valid)),
        "min_validation_trades": int(min(int(v["trade_entries"]) for v in valid)),
    }


def run(cfg: LongRegimeScoreGateConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    interest = build_interest_features(market, features)
    score_raw = _build_score_frame(market, features, interest)
    folds = _build_folds(cfg)
    rows = []
    for variant in _parse_list(cfg.score_variants, str):
        for q in _parse_list(cfg.score_quantiles, float):
            for hold in _parse_list(cfg.hold_bars, int):
                for stride in _parse_list(cfg.stride_bars, int):
                    fold_rows = []
                    for fold in folds:
                        fr = _run_fold(market=market, features=features, score_raw=score_raw, cfg=cfg, variant=variant, score_quantile=q, fold=fold, hold=hold, stride=stride)
                        if fr is not None:
                            fold_rows.append(fr)
                    score = _score_fold_result(fold_rows)
                    rows.append({"score_variant": variant, "score_quantile": q, "hold_bars": hold, "hold_hours": hold * 5 / 60, "stride_bars": stride, "fold_score": score, "folds": fold_rows})
    ranked = sorted(rows, key=lambda r: (r["fold_score"]["positive_folds"], r["fold_score"].get("min_ratio", -999), r["fold_score"].get("total_validation_trades", 0)), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "selection_protocol": "Each fold fits entry, premium/funding gates, score z-statistics, and score threshold on that fold train only; validation is report-only.",
        "folds": [{"train_start": a, "train_end": b, "validation_start": c, "validation_end": d} for a, b, c, d in folds],
        "top": ranked[:50],
        "all_count": len(rows),
        "leakage_guard": {"fold_train_only_thresholds": True, "fold_train_only_score_standardization": True, "validation_not_used_for_thresholds": True, "features_past_only_rolling_or_pct_change": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    for field in ("score-quantiles", "score-variants", "hold-bars", "stride-bars", "folds", "fold-preset", "rolling-train-start", "rolling-validation-start", "rolling-validation-end", "exclude-from"):
        p.add_argument(f"--{field}", default=getattr(LongRegimeScoreGateConfig, field.replace("-", "_")))
    p.add_argument("--entry-quantile", type=float, default=LongRegimeScoreGateConfig.entry_quantile)
    p.add_argument("--premium-quantile", type=float, default=LongRegimeScoreGateConfig.premium_quantile)
    p.add_argument("--funding-quantile", type=float, default=LongRegimeScoreGateConfig.funding_quantile)
    p.add_argument("--window-size", type=int, default=LongRegimeScoreGateConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=LongRegimeScoreGateConfig.entry_delay_bars)
    p.add_argument("--rolling-months", type=int, default=LongRegimeScoreGateConfig.rolling_months)
    p.add_argument("--leverage", type=float, default=LongRegimeScoreGateConfig.leverage)
    p.add_argument("--fee-rate", type=float, default=LongRegimeScoreGateConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=LongRegimeScoreGateConfig.slippage_rate)
    return p.parse_args()


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "score_variant": row["score_variant"],
        "score_quantile": row["score_quantile"],
        "hold_bars": row["hold_bars"],
        "stride_bars": row["stride_bars"],
        "fold_score": row["fold_score"],
        "folds": [{"fold": f["fold"], "score_gate": f["score_gate"], "validation": f["validation"]["sim"]} for f in row["folds"]],
    }


def main() -> None:
    report = run(LongRegimeScoreGateConfig(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output"], "input": report["input"], "all_count": report["all_count"], "top": [_compact(r) for r in report["top"][:10]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
