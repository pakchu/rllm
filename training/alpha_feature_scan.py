"""Leak-safe univariate alpha scan for market/external features.

This diagnostic is intentionally model-free.  It asks whether any past-only
feature has stable forward-return edge before spending more GPU time on LLM/RL
training.  It reports IC, directional accuracy, and top-vs-bottom quantile
spreads by chronological split.
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
from preprocessing.market_features import EXTENDED_MARKET_FEATURE_COLUMNS, build_market_feature_frame


@dataclass(frozen=True)
class ScanConfig:
    input_csv: str
    output: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    binance_funding_csv: str = ""
    binance_premium_csv: str = ""
    binance_funding_tolerance: str = "12h"
    binance_premium_tolerance: str = "2h"
    window_size: int = 144
    entry_delay_bars: int = 1
    horizons: tuple[int, ...] = (36, 72, 144, 288)
    quantile: float = 0.2
    min_abs_feature: float = 1e-12


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson(xr, yr)


def _t_stat(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0.0
    std = float(np.std(values, ddof=1))
    if std <= 0.0:
        return 0.0
    return float(np.mean(values) / (std / math.sqrt(values.size)))


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def _split_masks(dates: pd.Series) -> dict[str, np.ndarray]:
    # Current no-leak chronology used in recent experiments.
    return {
        "train_2023_2024h1": (dates >= "2023-01-01") & (dates <= "2024-06-30 23:59:59"),
        "test_2024h2_2025aug": (dates >= "2024-07-01") & (dates <= "2025-08-31 23:59:59"),
        "eval_2025sep_2026feb": (dates >= "2025-09-01") & (dates <= "2026-02-28 23:59:59"),
        "recent_train_2024_2025aug": (dates >= "2024-01-01") & (dates <= "2025-08-31 23:59:59"),
    }


def _feature_metrics(
    *,
    feature: np.ndarray,
    fwd_ret: np.ndarray,
    mask: np.ndarray,
    quantile: float,
    min_abs_feature: float,
) -> dict[str, Any]:
    valid = mask & np.isfinite(feature) & np.isfinite(fwd_ret) & (np.abs(feature) > float(min_abs_feature))
    x = feature[valid].astype(float)
    y = fwd_ret[valid].astype(float)
    if x.size < 30:
        return {"n": int(x.size)}
    x_center = x - float(np.median(x))
    q = float(np.clip(quantile, 0.01, 0.49))
    lo = float(np.quantile(x, q))
    hi = float(np.quantile(x, 1.0 - q))
    high = y[x >= hi]
    low = y[x <= lo]
    signed = np.sign(x_center) * y
    active = np.abs(x_center) > float(min_abs_feature)
    dir_acc = float(np.mean(np.sign(x_center[active]) == np.sign(y[active]))) if np.any(active) else 0.0
    spread = _safe_mean(high) - _safe_mean(low)
    return {
        "n": int(x.size),
        "pearson_ic": _pearson(x, y),
        "spearman_ic": _spearman(x, y),
        "direction_accuracy": dir_acc,
        "signed_mean_pct": _safe_mean(signed) * 100.0,
        "signed_t": _t_stat(signed),
        "q_low_threshold": lo,
        "q_high_threshold": hi,
        "q_low_mean_pct": _safe_mean(low) * 100.0,
        "q_high_mean_pct": _safe_mean(high) * 100.0,
        "q_high_minus_low_pct": spread * 100.0,
        "q_spread_t": _t_stat(np.concatenate([high, -low])) if high.size and low.size else 0.0,
    }


def _forward_return(open_: pd.Series, *, horizon: int, entry_delay_bars: int) -> np.ndarray:
    entry = open_.shift(-int(entry_delay_bars))
    exit_ = open_.shift(-(int(entry_delay_bars) + int(horizon)))
    return ((exit_ - entry) / entry.replace(0.0, np.nan)).to_numpy(dtype=float)


def scan_alpha_features(cfg: ScanConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(
            market,
            wave_trading_root=cfg.wave_trading_root,
            tolerance=cfg.external_tolerance,
        )
    if cfg.binance_funding_csv or cfg.binance_premium_csv:
        market = attach_binance_um_aux_features(
            market,
            funding_csv=cfg.binance_funding_csv or None,
            premium_csv=cfg.binance_premium_csv or None,
            funding_tolerance=cfg.binance_funding_tolerance,
            premium_tolerance=cfg.binance_premium_tolerance,
        )
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    feature_cols = [c for c in EXTENDED_MARKET_FEATURE_COLUMNS if c in features.columns]
    dates = pd.to_datetime(market["date"])
    split_masks = _split_masks(dates)

    rows: list[dict[str, Any]] = []
    open_ = market["open"].astype(float)
    for horizon in cfg.horizons:
        fwd = _forward_return(open_, horizon=int(horizon), entry_delay_bars=int(cfg.entry_delay_bars))
        for col in feature_cols:
            x = features[col].to_numpy(dtype=float)
            per_split = {
                name: _feature_metrics(
                    feature=x,
                    fwd_ret=fwd,
                    mask=np.asarray(mask, dtype=bool),
                    quantile=float(cfg.quantile),
                    min_abs_feature=float(cfg.min_abs_feature),
                )
                for name, mask in split_masks.items()
            }
            eval_m = per_split["eval_2025sep_2026feb"]
            test_m = per_split["test_2024h2_2025aug"]
            train_m = per_split["train_2023_2024h1"]
            # Robustness score favors same-sign IC/spread across train/test/eval.
            ic_values = [
                float(m.get("spearman_ic", 0.0))
                for m in (train_m, test_m, eval_m)
                if int(m.get("n", 0)) >= 30
            ]
            spread_values = [
                float(m.get("q_high_minus_low_pct", 0.0))
                for m in (train_m, test_m, eval_m)
                if int(m.get("n", 0)) >= 30
            ]
            ic_signs = [math.copysign(1.0, v) for v in ic_values if abs(v) > 1e-9]
            spread_signs = [math.copysign(1.0, v) for v in spread_values if abs(v) > 1e-9]
            ic_consistency = float(abs(sum(ic_signs)) / max(1, len(ic_signs))) if ic_signs else 0.0
            spread_consistency = float(abs(sum(spread_signs)) / max(1, len(spread_signs))) if spread_signs else 0.0
            sign_consistency = 0.5 * (ic_consistency + spread_consistency)
            eval_ic = abs(float(eval_m.get("spearman_ic", 0.0)))
            eval_spread = abs(float(eval_m.get("q_high_minus_low_pct", 0.0)))
            test_ic = abs(float(test_m.get("spearman_ic", 0.0)))
            test_spread = abs(float(test_m.get("q_high_minus_low_pct", 0.0)))
            # Penalize one-period flukes: eval-only edge without test confirmation
            # should not outrank weaker but more stable signals.
            stable_ic = min(eval_ic, test_ic)
            stable_spread = min(eval_spread, test_spread)
            score = (
                stable_ic * 100.0
                + stable_spread
                + 0.25 * max(0.0, min(abs(float(eval_m.get("signed_t", 0.0))), abs(float(test_m.get("signed_t", 0.0)))) - 1.0)
                + sign_consistency
            )
            rows.append(
                {
                    "feature": col,
                    "horizon": int(horizon),
                    "score": float(score),
                    "sign_consistency": sign_consistency,
                    "splits": per_split,
                }
            )

    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg) | {"horizons": list(cfg.horizons)},
        "inputs": {"rows": int(len(market)), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "features": feature_cols,
        "top": rows[:50],
        "all": rows,
        "leakage_guard": {
            "features_use_rows_at_or_before_t": True,
            "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled",
            "binance_aux_join": "backward_asof_no_future" if (cfg.binance_funding_csv or cfg.binance_premium_csv) else "disabled",
            "premium_index_uses_close_time_when_available": True,
            "future_returns_used_for_diagnostic_labels_only": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _parse_horizons(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(value).split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leak-safe feature alpha scan")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="30min")
    p.add_argument("--binance-funding-csv", default="")
    p.add_argument("--binance-premium-csv", default="")
    p.add_argument("--binance-funding-tolerance", default=ScanConfig.binance_funding_tolerance)
    p.add_argument("--binance-premium-tolerance", default=ScanConfig.binance_premium_tolerance)
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--horizons", default="36,72,144,288")
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument("--min-abs-feature", type=float, default=1e-12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ScanConfig(
        input_csv=args.input_csv,
        output=args.output,
        wave_trading_root=args.wave_trading_root,
        external_tolerance=args.external_tolerance,
        binance_funding_csv=args.binance_funding_csv,
        binance_premium_csv=args.binance_premium_csv,
        binance_funding_tolerance=args.binance_funding_tolerance,
        binance_premium_tolerance=args.binance_premium_tolerance,
        window_size=int(args.window_size),
        entry_delay_bars=int(args.entry_delay_bars),
        horizons=_parse_horizons(args.horizons),
        quantile=float(args.quantile),
        min_abs_feature=float(args.min_abs_feature),
    )
    report = scan_alpha_features(cfg)
    print(json.dumps({"output": cfg.output, "top": report["top"][:10], "leakage_guard": report["leakage_guard"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
