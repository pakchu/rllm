"""Leak-safe BTC alpha search from path efficiency and volatility-term structure.

The family is deliberately different from single-return/indicator gates: it asks
whether price travelled directionally per unit of realised path length, and
whether short-horizon volatility is expanding or compressing relative to the
long horizon.  All thresholds are frozen on the pre-2024 train split; 2024 is
the only ranking split and 2025/2026 are report-only diagnostics.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market, _split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config, W, mk, sim


def quantile(frame: pd.DataFrame, mask: np.ndarray, column: str, level: float) -> float:
    values = frame.loc[mask, column].to_numpy(float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, level))


def features(market: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    close = market.close.astype(float)
    logp = np.log(close.replace(0.0, np.nan))
    ret = logp.diff()
    abs_ret = ret.abs()

    for n in (24, 72, 144, 288):
        displacement = logp.diff(n)
        path = abs_ret.rolling(n, min_periods=n).sum()
        rv = ret.pow(2).rolling(n, min_periods=n).mean().pow(0.5)
        high = market.high.astype(float).rolling(n, min_periods=n).max()
        low = market.low.astype(float).rolling(n, min_periods=n).min()
        span = (high - low).replace(0.0, np.nan)
        out[f"ev_signed_eff_{n}"] = displacement / path.replace(0.0, np.nan)
        out[f"ev_eff_{n}"] = displacement.abs() / path.replace(0.0, np.nan)
        out[f"ev_rv_{n}"] = rv
        out[f"ev_range_{n}"] = span / close.replace(0.0, np.nan)
        out[f"ev_range_pos_{n}"] = (close - low) / span

    out["ev_volterm_24_288"] = out.ev_rv_24 / out.ev_rv_288.replace(0.0, np.nan)
    out["ev_volterm_72_288"] = out.ev_rv_72 / out.ev_rv_288.replace(0.0, np.nan)
    out["ev_range_term_24_288"] = out.ev_range_24 / out.ev_range_288.replace(0.0, np.nan)
    out["ev_eff_accel_24_144"] = out.ev_eff_24 - out.ev_eff_144
    out["ev_signed_eff_gap_24_144"] = out.ev_signed_eff_24 - out.ev_signed_eff_144
    return out.replace([np.inf, -np.inf], np.nan)


def specifications(frame: pd.DataFrame, train: np.ndarray):
    specs = []

    def frozen(terms):
        return [(column, op, quantile(frame, train, column, q)) for column, op, q in terms]

    for horizon in (72, 144, 288):
        for efficiency_q, vol_q, location_q in itertools.product((0.7, 0.8, 0.9), (0.6, 0.75, 0.9), (0.7, 0.8)):
            specs.append((
                f"efficient_expansion_{horizon}_{efficiency_q}_{vol_q}_{location_q}",
                frozen([(f"ev_signed_eff_{horizon}", "ge", efficiency_q), ("ev_volterm_24_288", "ge", vol_q), (f"ev_range_pos_{horizon}", "ge", location_q)]),
                frozen([(f"ev_signed_eff_{horizon}", "le", 1.0-efficiency_q), ("ev_volterm_24_288", "ge", vol_q), (f"ev_range_pos_{horizon}", "le", 1.0-location_q)]),
            ))
        for compression_q, accel_q, location_q in itertools.product((0.1, 0.2, 0.3), (0.7, 0.8, 0.9), (0.65, 0.75)):
            specs.append((
                f"compression_release_{horizon}_{compression_q}_{accel_q}_{location_q}",
                frozen([("ev_range_term_24_288", "le", compression_q), ("ev_eff_accel_24_144", "ge", accel_q), (f"ev_range_pos_{horizon}", "ge", location_q)]),
                frozen([("ev_range_term_24_288", "le", compression_q), ("ev_eff_accel_24_144", "ge", accel_q), (f"ev_range_pos_{horizon}", "le", 1.0-location_q)]),
            ))
        for exhaustion_q, vol_q, location_q in itertools.product((0.1, 0.2, 0.3), (0.75, 0.9), (0.8, 0.9)):
            specs.append((
                f"inefficient_extreme_revert_{horizon}_{exhaustion_q}_{vol_q}_{location_q}",
                frozen([(f"ev_eff_{horizon}", "le", exhaustion_q), ("ev_volterm_24_288", "ge", vol_q), (f"ev_range_pos_{horizon}", "le", 1.0-location_q)]),
                frozen([(f"ev_eff_{horizon}", "le", exhaustion_q), ("ev_volterm_24_288", "ge", vol_q), (f"ev_range_pos_{horizon}", "ge", location_q)]),
            ))
    return specs


def run(cfg: Config) -> dict:
    market = _load_market(cfg)
    base = build_market_feature_frame(market, window_size=cfg.window_size)
    frame = features(market, pd.concat([base, build_interest_features(market, base)], axis=1))
    dates = pd.to_datetime(market.date)
    train = _split_mask(dates, *W["train"])
    rows = []
    for name, long_conditions, short_conditions in specifications(frame, train):
        long_active, short_active = mk(frame, long_conditions), mk(frame, short_conditions)
        if int((long_active | short_active)[train].sum()) < 100:
            continue
        for hold, stride, (tp, sl) in itertools.product(
            (24, 48, 72, 96, 144), (6, 12, 24), ((.01, .008), (.015, .01), (.025, .015), (.04, .025))
        ):
            score = sim(market, dates, long_active, short_active, cfg, hold, stride, tp, sl, "test2024")
            if score["trades"] >= cfg.min_test_trades and score["longs"] >= cfg.min_each_side and score["shorts"] >= cfg.min_each_side:
                rows.append({
                    "name": name,
                    "long_conditions": [{"feature": c, "op": o, "threshold": t} for c, o, t in long_conditions],
                    "short_conditions": [{"feature": c, "op": o, "threshold": t} for c, o, t in short_conditions],
                    "hold_bars": hold, "stride_bars": stride, "tp": tp, "sl": sl, "test2024": score,
                })
    rows.sort(key=lambda row: (row["test2024"]["ratio"], row["test2024"]["return_pct"]), reverse=True)
    selected = rows[:cfg.top_n]
    for row in selected:
        lc = [(x["feature"], x["op"], x["threshold"]) for x in row["long_conditions"]]
        sc = [(x["feature"], x["op"], x["threshold"]) for x in row["short_conditions"]]
        long_active, short_active = mk(frame, lc), mk(frame, sc)
        for split in ("train", "eval2025", "ytd2026"):
            row[split] = sim(market, dates, long_active, short_active, cfg, row["hold_bars"], row["stride_bars"], row["tp"], row["sl"], split)
        eval_ok = row["eval2025"]["trades"] >= 16 and row["eval2025"]["longs"] >= 4 and row["eval2025"]["shorts"] >= 4
        row["passes_alpha_pool"] = bool(eval_ok and row["test2024"]["ratio"] >= 2.5 and row["eval2025"]["ratio"] >= 2.5)
        row["passes_live_grade"] = bool(eval_ok and all(row[x]["ratio"] >= 3 for x in ("test2024", "eval2025")))
        row["passes_2026_target"] = bool(row["ytd2026"]["trades"] >= 8 and row["ytd2026"]["ratio"] >= 5)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg),
        "protocol": "causal path-efficiency/volatility-term bidirectional BTC; train-frozen thresholds; test2024-only rank; sealed eval/2026 diagnostics; entry delay 1; 6bp/side; strict adverse-excursion MDD",
        "tested": len(rows), "selected": selected,
        "alpha_pool_qualifiers": [row for row in selected if row["passes_alpha_pool"]],
        "live_grade": [row for row in selected if row["passes_live_grade"] and row["passes_2026_target"]],
    }
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--funding-csv", default="")
    parser.add_argument("--premium-csv", default="")
    parser.add_argument("--exclude-from", default="2026-06-02")
    args = parser.parse_args()
    report = run(Config(**vars(args)))
    print(json.dumps({"tested": report["tested"], "qualifiers": len(report["alpha_pool_qualifiers"]), "live": len(report["live_grade"]), "top": report["selected"][:5]}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
