"""Audit train-fitted setup-quality buckets for fixed episode templates.

This checks whether weak price-action events become tradeable after conditioning
on causal setup-quality attributes such as stop distance, candle body, wick, and
close location.  Bucket thresholds are fit on train triggers only, then applied
unchanged to test/eval.
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

from training.alpha_linear_combo_scan import _load_market, _parse_list
from training.fixed_episode_template_backtest import _parse_specs
from training.price_action_episode_policy import EpisodePolicyCfg, add_sequence_context_features, build_episode_event_features, simulate_triggers, template_triggers


@dataclass(frozen=True)
class SetupQualityAuditCfg:
    input_csv: str
    output: str
    specs: str
    train_start: str = "2024-01-01"
    train_end: str = "2024-12-31 23:59:59"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    windows: str = "36,72,144,288,576,2016,4032,8640"
    include_sequence_context: bool = True
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    min_train_trades: int = 20
    top_k: int = 60


def _mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)), dtype=bool)


def _policy_cfg(cfg: SetupQualityAuditCfg) -> EpisodePolicyCfg:
    return EpisodePolicyCfg(
        input_csv=cfg.input_csv,
        output=cfg.output,
        train_start=cfg.train_start,
        train_end=cfg.train_end,
        test_start=cfg.test_start,
        test_end=cfg.test_end,
        eval_start=cfg.eval_start,
        eval_end=cfg.eval_end,
        windows=cfg.windows,
        horizons="1",
        entry_delay_bars=cfg.entry_delay_bars,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        include_sequence_context=cfg.include_sequence_context,
    )


def _quality_frame(market: pd.DataFrame, specs: list[dict[str, Any]], features: pd.DataFrame, dates: pd.Series, cfg: SetupQualityAuditCfg) -> pd.DataFrame:
    high = market["high"].to_numpy(dtype=float)
    low = market["low"].to_numpy(dtype=float)
    open_ = market["open"].to_numpy(dtype=float)
    close = market["close"].to_numpy(dtype=float)
    rows = []
    for spec in specs:
        ev = features[spec["event"]].to_numpy(dtype=float)
        for pos in np.flatnonzero(ev > 0.5):
            entry_pos = int(pos) + int(cfg.entry_delay_bars)
            if entry_pos >= len(market):
                continue
            rng = max(1e-12, high[pos] - low[pos])
            close_pos = (close[pos] - low[pos]) / rng
            body_frac = abs(close[pos] - open_[pos]) / rng
            upper_wick_frac = (high[pos] - max(open_[pos], close[pos])) / rng
            lower_wick_frac = (min(open_[pos], close[pos]) - low[pos]) / rng
            entry = open_[entry_pos]
            if entry <= 0:
                continue
            if spec["side"] == "LONG":
                risk_bps = max(0.0, (entry - low[pos]) / entry * 10_000.0)
                favorable_wick_frac = lower_wick_frac
                close_quality = close_pos
            else:
                risk_bps = max(0.0, (high[pos] - entry) / entry * 10_000.0)
                favorable_wick_frac = upper_wick_frac
                close_quality = 1.0 - close_pos
            rows.append(dict(spec) | {
                "pos": int(pos),
                "date": str(dates.iloc[int(pos)]),
                "risk_bps": float(risk_bps),
                "body_frac": float(body_frac),
                "favorable_wick_frac": float(favorable_wick_frac),
                "close_quality": float(close_quality),
                "range_bps": float(rng / max(1e-12, close[pos]) * 10_000.0),
            })
    return pd.DataFrame(rows)


def _bucket_masks(train_vals: pd.Series, all_vals: pd.Series) -> dict[str, np.ndarray]:
    if train_vals.empty:
        return {}
    q1, q2 = train_vals.quantile([1 / 3, 2 / 3]).tolist()
    return {
        "low": np.asarray(all_vals <= q1, dtype=bool),
        "mid": np.asarray((all_vals > q1) & (all_vals <= q2), dtype=bool),
        "high": np.asarray(all_vals > q2, dtype=bool),
    }


def _rows_to_triggers(rows: pd.DataFrame) -> list[dict[str, Any]]:
    keys = ["event", "window", "event_type", "episode", "side", "horizon", "pos"]
    return [{k: row[k] for k in keys} | {"score": 0.0, "train_score": 0.0} for row in rows.to_dict("records")]


def _score(sim: dict[str, Any]) -> float:
    s = sim["sim"]
    p = float(sim["trade_stats"].get("p_value_mean_ret_approx", 1.0) or 1.0)
    return float(s["cagr_to_strict_mdd"]) + 0.02 * float(s["cagr_pct"]) + min(1.0, float(s["trade_entries"]) / 100.0) - p


def run(cfg: SetupQualityAuditCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    dates = pd.to_datetime(market["date"])
    windows = _parse_list(cfg.windows, int)
    features = build_episode_event_features(market, windows)
    if cfg.include_sequence_context:
        features = add_sequence_context_features(market, features, windows)
    specs = _parse_specs(cfg.specs)
    qf = _quality_frame(market, specs, features, dates, cfg)
    train_mask = _mask(pd.to_datetime(qf["date"]), cfg.train_start, cfg.train_end) if len(qf) else np.array([], dtype=bool)
    policy_cfg = _policy_cfg(cfg)
    rows = []
    for spec_key, sub in qf.groupby(["event", "horizon", "side"], dropna=False):
        train_sub = sub.loc[train_mask[sub.index.to_numpy()]] if len(sub) else sub
        for feature in ["risk_bps", "range_bps", "body_frac", "favorable_wick_frac", "close_quality"]:
            buckets = _bucket_masks(train_sub[feature], sub[feature])
            for bucket, bm in buckets.items():
                filt = sub.loc[bm].copy()
                triggers = _rows_to_triggers(filt)
                train = simulate_triggers(market, dates, triggers, start=cfg.train_start, end=cfg.train_end, cfg=policy_cfg)
                if int(train["sim"]["trade_entries"]) < int(cfg.min_train_trades):
                    continue
                test = simulate_triggers(market, dates, triggers, start=cfg.test_start, end=cfg.test_end, cfg=policy_cfg)
                ev = simulate_triggers(market, dates, triggers, start=cfg.eval_start, end=cfg.eval_end, cfg=policy_cfg)
                rows.append({
                    "spec": {"event": spec_key[0], "horizon": int(spec_key[1]), "side": spec_key[2]},
                    "filter": {"feature": feature, "bucket": bucket, "threshold_source": "train_quantiles"},
                    "train": {"sim": train["sim"], "trade_stats": train["trade_stats"]},
                    "test": {"sim": test["sim"], "trade_stats": test["trade_stats"]},
                    "eval_diagnostic": {"sim": ev["sim"], "trade_stats": ev["trade_stats"]},
                    "train_score": _score(train),
                    "test_score": _score(test),
                })
    ranked = sorted(rows, key=lambda r: (float(r["test_score"]), float(r["train_score"]), int(r["test"]["sim"]["trade_entries"])), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "trigger_rows": int(len(qf)),
        "candidates": len(rows),
        "top": ranked[: int(cfg.top_k)],
        "protocol": "bucket thresholds fit on train triggers only; ranking shown for train/test diagnostics; eval is diagnostic holdout",
        "leakage_guard": {"bucket_thresholds_fit_on_eval": False, "features_use_signal_bar_or_earlier": True, "entry_uses_next_open": int(cfg.entry_delay_bars) >= 1},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--specs", required=True)
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(SetupQualityAuditCfg, name.replace("-", "_")))
    p.add_argument("--windows", default=SetupQualityAuditCfg.windows)
    p.add_argument("--no-sequence-context", dest="include_sequence_context", action="store_false")
    p.set_defaults(include_sequence_context=SetupQualityAuditCfg.include_sequence_context)
    p.add_argument("--entry-delay-bars", type=int, default=SetupQualityAuditCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=SetupQualityAuditCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=SetupQualityAuditCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=SetupQualityAuditCfg.slippage_rate)
    p.add_argument("--min-train-trades", type=int, default=SetupQualityAuditCfg.min_train_trades)
    p.add_argument("--top-k", type=int, default=SetupQualityAuditCfg.top_k)
    return p.parse_args()


def main() -> None:
    r = run(SetupQualityAuditCfg(**vars(parse_args())))
    print(json.dumps({
        "output": r["config"]["output"],
        "trigger_rows": r["trigger_rows"],
        "candidates": r["candidates"],
        "top": [
            row["spec"] | {"filter": row["filter"], "train": row["train"]["sim"], "test": row["test"]["sim"], "eval": row["eval_diagnostic"]["sim"], "test_p": row["test"]["trade_stats"].get("p_value_mean_ret_approx")}
            for row in r["top"][:10]
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
