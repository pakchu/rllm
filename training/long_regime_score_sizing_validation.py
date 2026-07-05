"""Rolling validation for score-tiered long-regime position sizing.

Hard activity gates can miss trades or overcommit in marginal regimes. This
validator keeps the leak-safe long entry skeleton, then sizes positions from a
train-fitted score:

  pb30_funding entry + premium high + funding non-negative + score tier

Tiers are fitted on the fold train window only. Validation windows are report-only.
The simulator is strict long OHLC with non-overlapping holds, intrabar adverse low
included in MDD, and per-trade fee/slippage scaled by leverage.
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
from training.long_regime_combo_scan import _load_market, _split_mask
from training.long_regime_gate_scan import LongRegimeGateConfig, _entry_active, _gate_active
from training.long_regime_score_gate_validation import _build_folds, _build_score_frame, _score_variant
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LongRegimeScoreSizingConfig(LongRegimeGateConfig):
    score_variants: str = "activity_flow,activity_flow_htf,activity_flow_deriv_htf,activity"
    tier_specs: str = "0.5:0.25:0.7:0.5,0.5:0.2:0.8:0.5,0.6:0.25:0.8:0.5,0.5:0.15:0.75:0.5"
    entry_quantile: float = 0.8
    premium_quantile: float = 0.8
    funding_quantile: float = 0.7
    hold_bars: str = "144"
    stride_bars: str = "12"
    folds: str = (
        "2020-01-01:2023-01-01:2023-01-01:2024-01-01,"
        "2020-01-01:2024-01-01:2024-01-01:2025-01-01,"
        "2020-01-01:2025-01-01:2025-01-01:2026-01-01,"
        "2020-01-01:2026-01-01:2026-01-01:2026-06-02"
    )
    fold_preset: str = "anchored_rolling"
    rolling_train_start: str = "2020-01-01"
    rolling_validation_start: str = "2023-01-01"
    rolling_validation_end: str = "2026-06-02"
    rolling_months: int = 6
    max_leverage: float = 0.5


def _parse_list(raw: str, cast: Any) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_tier_spec(raw: str) -> dict[str, float]:
    parts = [float(x.strip()) for x in raw.split(":")]
    if len(parts) != 4:
        raise ValueError(f"tier spec must be low_q:low_lev:high_q:high_lev, got {raw!r}")
    low_q, low_lev, high_q, high_lev = parts
    if not 0.0 <= low_q <= high_q <= 1.0:
        raise ValueError(f"invalid tier quantiles in {raw!r}")
    if low_lev < 0 or high_lev < 0:
        raise ValueError(f"invalid negative leverage in {raw!r}")
    return {"low_q": low_q, "low_leverage": low_lev, "high_q": high_q, "high_leverage": high_lev}


def _fit_tier_leverage(score: pd.Series, train_mask: np.ndarray, tier: dict[str, float]) -> tuple[np.ndarray, dict[str, Any]] | None:
    arr = score.to_numpy(float)
    ref = arr[train_mask & np.isfinite(arr)]
    if ref.size < 500 or float(np.nanstd(ref)) <= 1e-12:
        return None
    low_thr = float(np.quantile(ref, tier["low_q"]))
    high_thr = float(np.quantile(ref, tier["high_q"]))
    lev = np.zeros(len(arr), dtype=float)
    lev[(arr >= low_thr) & np.isfinite(arr)] = float(tier["low_leverage"])
    lev[(arr >= high_thr) & np.isfinite(arr)] = float(tier["high_leverage"])
    return lev, {**tier, "low_threshold": low_thr, "high_threshold": high_thr}


def _strict_long_sim_variable_leverage(
    signal_positions: np.ndarray,
    leverage_by_pos: np.ndarray,
    *,
    market: pd.DataFrame,
    hold_bars: int,
    entry_delay_bars: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[dict[str, Any], list[float]]:
    opens = market["open"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    dates = pd.to_datetime(market["date"]).to_numpy()
    signal_positions = np.asarray(signal_positions, dtype=np.int64)

    eq = peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    trade_returns: list[float] = []
    leverages: list[float] = []
    first_signal: int | None = None
    last_signal: int | None = None
    for pos in signal_positions:
        if pos < next_allowed:
            continue
        leverage = float(leverage_by_pos[int(pos)])
        if leverage <= 0.0:
            continue
        entry_pos = int(pos) + int(entry_delay_bars)
        exit_pos = entry_pos + int(hold_bars)
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            continue
        first_signal = int(pos) if first_signal is None else first_signal
        last_signal = int(pos)
        entry_eq = eq
        cost = (float(fee_rate) + float(slippage_rate)) * leverage
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak if peak > 0.0 else 0.0)
        for j in range(entry_pos, exit_pos):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            adverse_ret = (float(lows[j]) - open_j) / open_j
            max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 + leverage * adverse_ret)) / peak)
            close_ret = (float(opens[j + 1]) - open_j) / open_j
            eq *= max(0.0, 1.0 + leverage * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak if peak > 0.0 else 0.0)
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        leverages.append(leverage)
        next_allowed = exit_pos
        if eq <= 0.0:
            break

    if first_signal is None or last_signal is None:
        start_dt = end_dt = datetime.now()
        years = 1.0 / 365.25
    else:
        start_dt = pd.Timestamp(dates[first_signal]).to_pydatetime()
        end_dt = pd.Timestamp(dates[last_signal]).to_pydatetime()
        years = max((end_dt - start_dt).total_seconds() / (365.25 * 24 * 3600), 1.0 / 365.25)
    total_return = eq - 1.0
    cagr = (eq ** (1.0 / years) - 1.0) if eq > 0.0 else -1.0
    sim = {
        "period": {"start": str(start_dt), "end": str(end_dt), "years": years},
        "cagr_pct": float(cagr * 100.0),
        "strict_mdd_pct": float(max_dd * 100.0),
        "cagr_to_strict_mdd": float((cagr * 100.0) / (max_dd * 100.0)) if max_dd > 0.0 else float("inf"),
        "trade_entries": int(len(trade_returns)),
        "win_rate": float(np.mean(np.asarray(trade_returns) > 0.0)) if trade_returns else 0.0,
        "total_return_pct": float(total_return * 100.0),
        "avg_leverage": float(np.mean(leverages)) if leverages else 0.0,
        "max_leverage": float(max(leverages)) if leverages else 0.0,
        "hold_bars": int(hold_bars),
        "entry_delay_bars": int(entry_delay_bars),
        "return_application": "long_only_variable_leverage_actual_ohlc_strict_mdd",
    }
    return sim, trade_returns


def _run_fold(
    *,
    market: pd.DataFrame,
    features: pd.DataFrame,
    score_raw: pd.DataFrame,
    cfg: LongRegimeScoreSizingConfig,
    variant: str,
    tier: dict[str, float],
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
    tier_pack = _fit_tier_leverage(score, train_mask, tier)
    if tier_pack is None:
        return None
    leverage_by_pos, tier_spec = tier_pack
    leverage_by_pos = np.minimum(leverage_by_pos, float(cfg.max_leverage))
    entry_mask, entry_spec = entry
    premium_mask, premium_spec = premium
    funding_mask, funding_spec = funding
    active = entry_mask & premium_mask & funding_mask & (leverage_by_pos > 0.0)
    positions = np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - hold - int(cfg.entry_delay_bars) - 1), stride, dtype=np.int64)
    rows = {}
    for split, mask in (("train", train_mask), ("validation", val_mask)):
        p = positions[active[positions] & mask[positions]]
        sim, returns = _strict_long_sim_variable_leverage(
            p,
            leverage_by_pos,
            market=market,
            hold_bars=hold,
            entry_delay_bars=int(cfg.entry_delay_bars),
            fee_rate=float(cfg.fee_rate),
            slippage_rate=float(cfg.slippage_rate),
        )
        rows[split] = {"sim": sim, "trade_stats": _trade_stats(returns), "candidate_count": int(len(p))}
    return {
        "fold": {"train_start": train_start, "train_end": train_end, "validation_start": val_start, "validation_end": val_end},
        "tier_spec": tier_spec,
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
    total_returns = [float(v["total_return_pct"]) for v in valid]
    return {
        "folds": len(folds),
        "active_folds": len(valid),
        "positive_folds": sum(1 for v in valid if float(v["total_return_pct"]) > 0.0),
        "min_ratio": float(min(ratios)),
        "median_ratio": float(np.median(ratios)),
        "mean_ratio": float(np.mean(ratios)),
        "min_total_return_pct": float(min(total_returns)),
        "sum_total_return_pct": float(sum(total_returns)),
        "total_validation_trades": int(sum(int(v["trade_entries"]) for v in valid)),
        "max_validation_mdd": float(max(float(v["strict_mdd_pct"]) for v in valid)),
        "min_validation_trades": int(min(int(v["trade_entries"]) for v in valid)),
    }


def run(cfg: LongRegimeScoreSizingConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    from training.long_regime_interest_gate_validation import build_interest_features
    interest = build_interest_features(market, features)
    score_raw = _build_score_frame(market, features, interest)
    folds = _build_folds(cfg)  # type: ignore[arg-type]
    rows = []
    for variant in _parse_list(cfg.score_variants, str):
        for tier in _parse_list(cfg.tier_specs, _parse_tier_spec):
            for hold in _parse_list(cfg.hold_bars, int):
                for stride in _parse_list(cfg.stride_bars, int):
                    fold_rows = []
                    for fold in folds:
                        fr = _run_fold(market=market, features=features, score_raw=score_raw, cfg=cfg, variant=variant, tier=tier, fold=fold, hold=hold, stride=stride)
                        if fr is not None:
                            fold_rows.append(fr)
                    score = _score_fold_result(fold_rows)
                    rows.append({"score_variant": variant, "tier": tier, "hold_bars": hold, "hold_hours": hold * 5 / 60, "stride_bars": stride, "fold_score": score, "folds": fold_rows})
    ranked = sorted(rows, key=lambda r: (r["fold_score"].get("positive_folds", 0), r["fold_score"].get("min_total_return_pct", -999), r["fold_score"].get("min_ratio", -999), r["fold_score"].get("total_validation_trades", 0)), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "selection_protocol": "Each fold fits entry, premium/funding gates, score z-statistics, and score tier thresholds on fold train only; validation is report-only.",
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
    for field in ("score-variants", "tier-specs", "hold-bars", "stride-bars", "folds", "fold-preset", "rolling-train-start", "rolling-validation-start", "rolling-validation-end", "exclude-from"):
        p.add_argument(f"--{field}", default=getattr(LongRegimeScoreSizingConfig, field.replace("-", "_")))
    p.add_argument("--entry-quantile", type=float, default=LongRegimeScoreSizingConfig.entry_quantile)
    p.add_argument("--premium-quantile", type=float, default=LongRegimeScoreSizingConfig.premium_quantile)
    p.add_argument("--funding-quantile", type=float, default=LongRegimeScoreSizingConfig.funding_quantile)
    p.add_argument("--window-size", type=int, default=LongRegimeScoreSizingConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=LongRegimeScoreSizingConfig.entry_delay_bars)
    p.add_argument("--rolling-months", type=int, default=LongRegimeScoreSizingConfig.rolling_months)
    p.add_argument("--max-leverage", type=float, default=LongRegimeScoreSizingConfig.max_leverage)
    p.add_argument("--fee-rate", type=float, default=LongRegimeScoreSizingConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=LongRegimeScoreSizingConfig.slippage_rate)
    return p.parse_args()


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "score_variant": row["score_variant"],
        "tier": row["tier"],
        "hold_bars": row["hold_bars"],
        "stride_bars": row["stride_bars"],
        "fold_score": row["fold_score"],
        "folds": [{"fold": f["fold"], "tier_spec": f["tier_spec"], "validation": f["validation"]["sim"]} for f in row["folds"]],
    }


def main() -> None:
    report = run(LongRegimeScoreSizingConfig(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output"], "input": report["input"], "all_count": report["all_count"], "top": [_compact(r) for r in report["top"][:10]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
