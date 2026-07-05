"""Rolling validation for long-regime interest/activity gate candidates.

Promotes the exploratory finding that REX long pullbacks need enough market
participation.  The candidate is fixed as:
  pb30_funding entry + premium high + funding non-negative + activity gate.

All thresholds are fitted on each fold's train window only.  Validation/eval
windows are report-only for that fold.
"""
from __future__ import annotations

import argparse, json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.long_regime_gate_scan import LongRegimeGateConfig, _entry_active, _gate_active
from training.long_regime_combo_scan import _load_market, _split_mask, _strict_long_sim
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class InterestGateValidationConfig(LongRegimeGateConfig):
    activity_features: str = "dollar_flow_rel_4h_30d,quote_vol_z_1d,quote_vol_rel_1d_30d,interest_score"
    activity_quantiles: str = "0.3,0.5,0.7"
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


def _parse_list(raw: str, cast: Any) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(',') if x.strip()]


def _parse_fold(raw: str) -> tuple[str, str, str, str]:
    a = [x.strip() for x in raw.split(':')]
    if len(a) != 4:
        raise ValueError(f"fold must be train_start:train_end:val_start:val_end, got {raw!r}")
    return tuple(a)  # type: ignore[return-value]


def _z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(10, window // 5)).mean()
    std = series.rolling(window, min_periods=max(10, window // 5)).std(ddof=0)
    return ((series - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rel(series: pd.Series, fast: int, slow: int) -> pd.Series:
    f = series.rolling(fast, min_periods=max(5, fast // 5)).mean()
    s = series.rolling(slow, min_periods=max(10, slow // 5)).mean()
    return np.log((f + 1.0) / (s + 1.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_interest_features(market: pd.DataFrame, base_features: pd.DataFrame) -> pd.DataFrame:
    quote = market["quote_asset_volume"].astype(float)
    trades = market["number_of_trades"].astype(float)
    volume = market["volume"].astype(float)
    out = pd.DataFrame(index=market.index)
    out["quote_vol_z_1d"] = _z(np.log1p(quote), 288)
    out["quote_vol_rel_1d_30d"] = _rel(quote, 288, 8640)
    out["trades_rel_1d_30d"] = _rel(trades, 288, 8640)
    out["volume_rel_1d_30d"] = _rel(volume, 288, 8640)
    out["dollar_flow_rel_4h_30d"] = _rel(quote, 48, 8640)
    premium = pd.Series(np.abs(base_features.get("premium_index", pd.Series(0.0, index=market.index)).to_numpy(float)), index=market.index)
    funding = pd.Series(np.abs(base_features.get("funding_rate", pd.Series(0.0, index=market.index)).to_numpy(float)), index=market.index)
    out["premium_abs_z"] = _z(premium, 288)
    out["funding_abs_z"] = _z(funding, 288)
    out["interest_score"] = out[["quote_vol_rel_1d_30d", "trades_rel_1d_30d", "volume_rel_1d_30d", "dollar_flow_rel_4h_30d", "premium_abs_z", "funding_abs_z"]].mean(axis=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_activity_gate(values: np.ndarray, train_mask: np.ndarray, q: float) -> tuple[np.ndarray, dict[str, Any]] | None:
    ref = values[train_mask & np.isfinite(values)]
    if ref.size < 500 or float(np.nanstd(ref)) <= 1e-12:
        return None
    threshold = float(np.quantile(ref, q))
    return (values >= threshold) & np.isfinite(values), {"op": "ge", "quantile": float(q), "threshold": threshold}


def _run_fold(
    *, market: pd.DataFrame, features: pd.DataFrame, interest: pd.DataFrame, cfg: InterestGateValidationConfig,
    activity_feature: str, activity_quantile: float, fold: tuple[str, str, str, str], hold: int, stride: int,
) -> dict[str, Any] | None:
    train_start, train_end, val_start, val_end = fold
    dates = pd.to_datetime(market["date"])
    train_mask = _split_mask(dates, train_start, train_end)
    val_mask = _split_mask(dates, val_start, val_end)
    entry = _entry_active(features, rule="pb30_funding", train_mask=train_mask, quantile=float(cfg.entry_quantile))
    premium = _gate_active(features, feature="premium_index_zscore", op="ge", train_mask=train_mask, quantile=float(cfg.premium_quantile))
    funding = _gate_active(features, feature="funding_zscore", op="ge", train_mask=train_mask, quantile=float(cfg.funding_quantile))
    if entry is None or premium is None or funding is None:
        return None
    values = interest[activity_feature].to_numpy(float)
    activity = _fit_activity_gate(values, train_mask, activity_quantile)
    if activity is None:
        return None
    entry_mask, entry_spec = entry
    premium_mask, premium_spec = premium
    funding_mask, funding_spec = funding
    activity_mask, activity_spec = activity
    active = entry_mask & premium_mask & funding_mask & activity_mask
    positions = np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - hold - int(cfg.entry_delay_bars) - 1), stride, dtype=np.int64)
    rows = {}
    for split, mask in (("train", train_mask), ("validation", val_mask)):
        p = positions[active[positions] & mask[positions]]
        sim, returns = _strict_long_sim(p, market=market, hold_bars=hold, entry_delay_bars=int(cfg.entry_delay_bars), leverage=float(cfg.leverage), fee_rate=float(cfg.fee_rate), slippage_rate=float(cfg.slippage_rate))
        rows[split] = {"sim": sim, "trade_stats": _trade_stats(returns), "candidate_count": int(len(p))}
    return {
        "fold": {"train_start": train_start, "train_end": train_end, "validation_start": val_start, "validation_end": val_end},
        "activity_gate": {"feature": activity_feature, **activity_spec},
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
    }


def run(cfg: InterestGateValidationConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    interest = build_interest_features(market, features)
    folds = [_parse_fold(x) for x in _parse_list(cfg.folds, str)]
    rows = []
    for activity_feature in _parse_list(cfg.activity_features, str):
        if activity_feature not in interest.columns:
            continue
        for q in _parse_list(cfg.activity_quantiles, float):
            for hold in _parse_list(cfg.hold_bars, int):
                for stride in _parse_list(cfg.stride_bars, int):
                    fold_rows = []
                    for fold in folds:
                        fr = _run_fold(market=market, features=features, interest=interest, cfg=cfg, activity_feature=activity_feature, activity_quantile=q, fold=fold, hold=hold, stride=stride)
                        if fr is not None:
                            fold_rows.append(fr)
                    score = _score_fold_result(fold_rows)
                    rows.append({"activity_feature": activity_feature, "activity_quantile": q, "hold_bars": hold, "hold_hours": hold * 5 / 60, "stride_bars": stride, "fold_score": score, "folds": fold_rows})
    ranked = sorted(rows, key=lambda r: (r["fold_score"]["positive_folds"], r["fold_score"].get("min_ratio", -999), r["fold_score"].get("total_validation_trades", 0)), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "selection_protocol": "Each fold fits entry, premium/funding gates, and activity threshold on that fold train only; validation is report-only.",
        "top": ranked[:50],
        "all_count": len(rows),
        "leakage_guard": {"fold_train_only_thresholds": True, "validation_not_used_for_thresholds": True, "activity_features_past_only_rolling": True},
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
    for field in ("activity-features", "activity-quantiles", "hold-bars", "stride-bars", "folds", "exclude-from"):
        p.add_argument(f"--{field}", default=getattr(InterestGateValidationConfig, field.replace('-', '_')))
    p.add_argument("--entry-quantile", type=float, default=InterestGateValidationConfig.entry_quantile)
    p.add_argument("--premium-quantile", type=float, default=InterestGateValidationConfig.premium_quantile)
    p.add_argument("--funding-quantile", type=float, default=InterestGateValidationConfig.funding_quantile)
    p.add_argument("--window-size", type=int, default=InterestGateValidationConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=InterestGateValidationConfig.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=InterestGateValidationConfig.leverage)
    p.add_argument("--fee-rate", type=float, default=InterestGateValidationConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=InterestGateValidationConfig.slippage_rate)
    return p.parse_args()


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {"activity_feature": row["activity_feature"], "activity_quantile": row["activity_quantile"], "hold_bars": row["hold_bars"], "stride_bars": row["stride_bars"], "fold_score": row["fold_score"], "folds": [{"fold": f["fold"], "activity_gate": f["activity_gate"], "validation": f["validation"]["sim"]} for f in row["folds"]]}


def main() -> None:
    report = run(InterestGateValidationConfig(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output"], "input": report["input"], "all_count": report["all_count"], "top": [_compact(r) for r in report["top"][:10]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
