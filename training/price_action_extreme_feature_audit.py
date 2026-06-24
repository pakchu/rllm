"""Leak-safe audit for price-action extreme-bar features.

For each lookback window, find:
- the candle with the highest high and expose that candle's low as support-like context;
- the candle with the lowest low and expose that candle's high as resistance-like context.

All features at row t use only candles <= t. Forward returns are diagnostic labels
only and assume entry after `entry_delay_bars`.
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


@dataclass(frozen=True)
class PriceActionExtremeAuditCfg:
    input_csv: str
    output: str
    lookbacks: tuple[int, ...] = (36, 72, 144, 288, 576, 2016)
    horizons: tuple[int, ...] = (36, 72, 144, 288)
    entry_delay_bars: int = 1
    quantile: float = 0.2
    min_samples: int = 500


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=np.abs(den) > 1e-12)


def build_extreme_bar_features(market: pd.DataFrame, lookbacks: tuple[int, ...]) -> pd.DataFrame:
    high = market["high"].to_numpy(dtype=float)
    low = market["low"].to_numpy(dtype=float)
    open_ = market["open"].to_numpy(dtype=float)
    close = market["close"].to_numpy(dtype=float) if "close" in market.columns else open_
    n = len(market)
    out: dict[str, np.ndarray] = {}
    for lb_raw in lookbacks:
        lb = int(lb_raw)
        max_high = np.full(n, np.nan, dtype=float)
        min_low = np.full(n, np.nan, dtype=float)
        low_at_max_high = np.full(n, np.nan, dtype=float)
        high_at_min_low = np.full(n, np.nan, dtype=float)
        max_high_age = np.full(n, np.nan, dtype=float)
        min_low_age = np.full(n, np.nan, dtype=float)
        for i in range(n):
            start = max(0, i - lb + 1)
            hi_window = high[start : i + 1]
            lo_window = low[start : i + 1]
            if hi_window.size < lb:
                continue
            arg_hi = int(np.argmax(hi_window)) + start
            arg_lo = int(np.argmin(lo_window)) + start
            max_high[i] = high[arg_hi]
            min_low[i] = low[arg_lo]
            low_at_max_high[i] = low[arg_hi]
            high_at_min_low[i] = high[arg_lo]
            max_high_age[i] = (i - arg_hi) / max(1.0, float(lb - 1))
            min_low_age[i] = (i - arg_lo) / max(1.0, float(lb - 1))
        prefix = f"pa_ext_{lb}"
        width = max_high - min_low
        out[f"{prefix}_range_pos"] = _safe_div(close - min_low, width)
        out[f"{prefix}_to_max_high_pct"] = _safe_div(close - max_high, close)
        out[f"{prefix}_to_min_low_pct"] = _safe_div(close - min_low, close)
        out[f"{prefix}_to_low_of_max_high_pct"] = _safe_div(close - low_at_max_high, close)
        out[f"{prefix}_to_high_of_min_low_pct"] = _safe_div(close - high_at_min_low, close)
        out[f"{prefix}_max_high_bar_spread_pct"] = _safe_div(max_high - low_at_max_high, close)
        out[f"{prefix}_min_low_bar_spread_pct"] = _safe_div(high_at_min_low - min_low, close)
        out[f"{prefix}_extreme_bar_overlap_pct"] = _safe_div(high_at_min_low - low_at_max_high, close)
        out[f"{prefix}_max_high_age_frac"] = max_high_age
        out[f"{prefix}_min_low_age_frac"] = min_low_age
        out[f"{prefix}_age_diff"] = max_high_age - min_low_age
        out[f"{prefix}_open_to_low_of_max_high_pct"] = _safe_div(open_ - low_at_max_high, open_)
        out[f"{prefix}_open_to_high_of_min_low_pct"] = _safe_div(open_ - high_at_min_low, open_)
    return pd.DataFrame(out, index=market.index)


def _forward_return(open_: pd.Series, *, horizon: int, entry_delay_bars: int) -> np.ndarray:
    entry = open_.shift(-int(entry_delay_bars))
    exit_ = open_.shift(-(int(entry_delay_bars) + int(horizon)))
    return ((exit_ - entry) / entry.replace(0.0, np.nan)).to_numpy(dtype=float)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return 0.0
    sx = float(np.std(x)); sy = float(np.std(y))
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(pd.Series(x).rank(method="average").to_numpy(dtype=float), pd.Series(y).rank(method="average").to_numpy(dtype=float))


def _t_stat(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0.0
    sd = float(np.std(values, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean(values) / (sd / math.sqrt(values.size)))


def _mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def _split_masks(dates: pd.Series) -> dict[str, np.ndarray]:
    years = sorted({int(x.year) for x in pd.to_datetime(dates)})
    masks: dict[str, np.ndarray] = {}
    for y in years:
        masks[str(y)] = np.asarray((dates >= f"{y}-01-01") & (dates < f"{y + 1}-01-01"), dtype=bool)
    masks["pre2026"] = np.asarray(dates < "2026-01-01", dtype=bool)
    masks["eval2026"] = np.asarray(dates >= "2026-01-01", dtype=bool)
    return masks


def _metrics(feature: np.ndarray, fwd_ret: np.ndarray, mask: np.ndarray, cfg: PriceActionExtremeAuditCfg) -> dict[str, Any]:
    valid = mask & np.isfinite(feature) & np.isfinite(fwd_ret)
    x = feature[valid].astype(float)
    y = fwd_ret[valid].astype(float)
    if x.size < int(cfg.min_samples):
        return {"n": int(x.size)}
    q = float(np.clip(cfg.quantile, 0.01, 0.49))
    lo = float(np.quantile(x, q))
    hi = float(np.quantile(x, 1.0 - q))
    low_y = y[x <= lo]
    high_y = y[x >= hi]
    spread_samples = np.concatenate([high_y, -low_y]) if high_y.size and low_y.size else np.asarray([], dtype=float)
    centered = x - float(np.median(x))
    signed = np.sign(centered) * y
    return {
        "n": int(x.size),
        "spearman_ic": _spearman(x, y),
        "pearson_ic": _pearson(x, y),
        "q_low": lo,
        "q_high": hi,
        "q_low_mean_pct": _mean(low_y) * 100.0,
        "q_high_mean_pct": _mean(high_y) * 100.0,
        "q_high_minus_low_pct": (_mean(high_y) - _mean(low_y)) * 100.0,
        "q_spread_t": _t_stat(spread_samples),
        "signed_mean_pct": _mean(signed) * 100.0,
        "signed_t": _t_stat(signed),
    }


def _stable_score(per_split: dict[str, dict[str, Any]]) -> float:
    years = [k for k in per_split if k.isdigit()]
    vals = [float(per_split[y].get("q_high_minus_low_pct", 0.0)) for y in years if int(per_split[y].get("n", 0)) > 0]
    ics = [float(per_split[y].get("spearman_ic", 0.0)) for y in years if int(per_split[y].get("n", 0)) > 0]
    if not vals or not ics:
        return 0.0
    signs = [math.copysign(1.0, v) for v in vals if abs(v) > 1e-12]
    consistency = abs(sum(signs)) / max(1, len(signs)) if signs else 0.0
    recent = abs(float(per_split.get("eval2026", {}).get("q_high_minus_low_pct", 0.0)))
    robust = min(abs(v) for v in vals) if vals else 0.0
    ic_robust = min(abs(v) for v in ics) * 100.0 if ics else 0.0
    return float(robust + 0.5 * recent + ic_robust + consistency)


def run(cfg: PriceActionExtremeAuditCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    features = build_extreme_bar_features(market, cfg.lookbacks)
    dates = pd.to_datetime(market["date"])
    masks = _split_masks(dates)
    rows: list[dict[str, Any]] = []
    for horizon in cfg.horizons:
        fwd = _forward_return(market["open"].astype(float), horizon=int(horizon), entry_delay_bars=int(cfg.entry_delay_bars))
        for col in features.columns:
            x = features[col].to_numpy(dtype=float)
            per_split = {name: _metrics(x, fwd, mask, cfg) for name, mask in masks.items()}
            rows.append({"feature": col, "horizon": int(horizon), "score": _stable_score(per_split), "splits": per_split})
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg) | {"lookbacks": list(cfg.lookbacks), "horizons": list(cfg.horizons)},
        "inputs": {"rows": int(len(market)), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "feature_count": int(len(features.columns)),
        "top": rows[:50],
        "all": rows,
        "leakage_guard": {
            "features_use_only_candles_at_or_before_t": True,
            "highest_high_bar_low_uses_same_past_window_only": True,
            "lowest_low_bar_high_uses_same_past_window_only": True,
            "future_returns_used_for_diagnostic_labels_only": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit price-action extreme-bar features")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lookbacks", default="36,72,144,288,576,2016")
    p.add_argument("--horizons", default="36,72,144,288")
    p.add_argument("--entry-delay-bars", type=int, default=PriceActionExtremeAuditCfg.entry_delay_bars)
    p.add_argument("--quantile", type=float, default=PriceActionExtremeAuditCfg.quantile)
    p.add_argument("--min-samples", type=int, default=PriceActionExtremeAuditCfg.min_samples)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PriceActionExtremeAuditCfg(
        input_csv=args.input_csv,
        output=args.output,
        lookbacks=_parse_ints(args.lookbacks),
        horizons=_parse_ints(args.horizons),
        entry_delay_bars=int(args.entry_delay_bars),
        quantile=float(args.quantile),
        min_samples=int(args.min_samples),
    )
    report = run(cfg)
    print(json.dumps({"output": cfg.output, "top": report["top"][:10], "leakage_guard": report["leakage_guard"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
