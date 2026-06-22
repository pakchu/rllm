"""Audit causal feature coverage before widening alpha/RLLM experiments.

The alpha failures can come from weak signal, but also from feature availability
holes, constant availability flags, or stale external joins. This report measures
coverage, variance, and per-year availability for market, external, HTF, and wave
features built with the same causal feature builders used by alpha scans.
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

from preprocessing.binance_aux_features import attach_binance_um_aux_features
from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.wave_feature_ridge_policy import build_wave_feature_frame


@dataclass(frozen=True)
class FeatureCoverageConfig:
    input_csv: str
    output: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    binance_funding_csv: str = ""
    binance_premium_csv: str = ""
    binance_funding_tolerance: str = "12h"
    binance_premium_tolerance: str = "2h"
    window_size: int = 144
    include_wave_features: bool = True
    min_nonzero_fraction: float = 0.01
    min_std: float = 1e-12


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _family(name: str) -> str:
    raw = name.removeprefix("mkt__").removeprefix("wave__")
    if raw.startswith(("dxy", "usdkrw", "kimchi", "btckrw", "external_")):
        return "external_macro_kimchi"
    if raw.startswith(("htf_", "weekly_")):
        return "higher_timeframe"
    if raw.startswith(("taker", "trades_", "volume", "flow", "cvd", "trade_intensity")):
        return "flow_volume"
    if raw.startswith(("funding", "oi_", "premium")):
        return "derivatives_aux"
    if raw.endswith("available") or raw.endswith("_available"):
        return "availability_flag"
    if name.startswith("wave__"):
        return "wave"
    return "market"


def _feature_stats(values: pd.Series, years: pd.Series, *, min_nonzero_fraction: float, min_std: float) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.replace([np.inf, -np.inf], np.nan).notna()
    clean = numeric.where(finite)
    n = int(len(clean))
    non_null = int(finite.sum())
    nonzero = int((clean.fillna(0.0).abs() > 1e-12).sum())
    std = float(clean.fillna(0.0).std(ddof=0)) if n else 0.0
    per_year: dict[str, Any] = {}
    for year, idx in years.groupby(years).groups.items():
        sub = clean.iloc[list(idx)]
        per_year[str(year)] = {
            "rows": int(len(sub)),
            "non_null_fraction": float(sub.notna().mean()) if len(sub) else 0.0,
            "nonzero_fraction": float((sub.fillna(0.0).abs() > 1e-12).mean()) if len(sub) else 0.0,
            "std": float(sub.fillna(0.0).std(ddof=0)) if len(sub) else 0.0,
            "mean": float(sub.fillna(0.0).mean()) if len(sub) else 0.0,
        }
    nonzero_fraction = nonzero / max(1, n)
    return {
        "rows": n,
        "non_null_fraction": non_null / max(1, n),
        "nonzero_fraction": nonzero_fraction,
        "std": std,
        "mean": float(clean.fillna(0.0).mean()) if n else 0.0,
        "min": float(clean.min()) if non_null else 0.0,
        "max": float(clean.max()) if non_null else 0.0,
        "usable": bool(nonzero_fraction >= float(min_nonzero_fraction) and std >= float(min_std)),
        "per_year": per_year,
    }


def build_feature_frame(market: pd.DataFrame, cfg: FeatureCoverageConfig) -> pd.DataFrame:
    base = build_market_feature_frame(market, window_size=int(cfg.window_size)).add_prefix("mkt__")
    if not cfg.include_wave_features:
        return base.replace([np.inf, -np.inf], np.nan)
    wave = build_wave_feature_frame(market, window=int(cfg.window_size)).add_prefix("wave__")
    features = pd.concat([base, wave], axis=1)
    return features.loc[:, ~features.columns.duplicated(keep="last")].replace([np.inf, -np.inf], np.nan)


def run(cfg: FeatureCoverageConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    attach_error = None
    if cfg.wave_trading_root:
        try:
            market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
        except Exception as exc:  # keep audit useful even if external cache is absent
            attach_error = str(exc)
    if cfg.binance_funding_csv or cfg.binance_premium_csv:
        market = attach_binance_um_aux_features(
            market,
            funding_csv=cfg.binance_funding_csv or None,
            premium_csv=cfg.binance_premium_csv or None,
            funding_tolerance=cfg.binance_funding_tolerance,
            premium_tolerance=cfg.binance_premium_tolerance,
        )
    features = build_feature_frame(market, cfg)
    years = pd.to_datetime(market["date"]).dt.year.astype(str)
    feature_reports: dict[str, Any] = {}
    family_counts: dict[str, dict[str, int]] = {}
    for col in features.columns:
        fam = _family(str(col))
        stats = _feature_stats(features[col], years, min_nonzero_fraction=float(cfg.min_nonzero_fraction), min_std=float(cfg.min_std))
        feature_reports[str(col)] = {"family": fam, **stats}
        fam_counts = family_counts.setdefault(fam, {"features": 0, "usable": 0})
        fam_counts["features"] += 1
        fam_counts["usable"] += int(bool(stats["usable"]))
    critical = {k: v for k, v in feature_reports.items() if any(token in k for token in ("dxy", "usdkrw", "kimchi", "htf_", "weekly_", "funding", "premium", "taker", "trades_ratio"))}
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": int(len(market)), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1]), "columns": list(market.columns)},
        "external_attach_error": attach_error,
        "feature_count": int(len(features.columns)),
        "family_counts": family_counts,
        "critical_features": critical,
        "unusable_features": {k: v for k, v in feature_reports.items() if not v["usable"]},
        "leakage_guard": {
            "uses_same_causal_feature_builders_as_alpha_scans": True,
            "external_join_is_backward_asof_when_enabled": True,
            "binance_aux_join_is_backward_asof_when_enabled": True,
            "premium_index_uses_close_time_when_available": True,
            "audit_does_not_use_future_returns": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit feature coverage and availability")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=FeatureCoverageConfig.external_tolerance)
    p.add_argument("--binance-funding-csv", default="")
    p.add_argument("--binance-premium-csv", default="")
    p.add_argument("--binance-funding-tolerance", default=FeatureCoverageConfig.binance_funding_tolerance)
    p.add_argument("--binance-premium-tolerance", default=FeatureCoverageConfig.binance_premium_tolerance)
    p.add_argument("--window-size", type=int, default=FeatureCoverageConfig.window_size)
    p.add_argument("--no-wave-features", action="store_true")
    p.add_argument("--min-nonzero-fraction", type=float, default=FeatureCoverageConfig.min_nonzero_fraction)
    p.add_argument("--min-std", type=float, default=FeatureCoverageConfig.min_std)
    args = p.parse_args()
    args.include_wave_features = not bool(args.no_wave_features)
    delattr(args, "no_wave_features")
    return args


def main() -> None:
    print(json.dumps(run(FeatureCoverageConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
