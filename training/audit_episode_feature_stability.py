"""Audit episode feature stability before further optimization.

This is intentionally diagnostic, not a selector.  It checks whether structural
price-action episode features are sparse, duplicated, or directionally unstable
across chronological splits, which are the usual causes of fast overfit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_linear_combo_scan import _load_market, _parse_list
from training.price_action_episode_policy import EPISODE_SIDES, add_sequence_context_features, build_episode_event_features
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class EpisodeFeatureStabilityCfg:
    input_csv: str
    output: str
    train_start: str = "2024-01-01"
    train_end: str = "2024-12-31 23:59:59"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    windows: str = "36,72,144,288,576,2016,4032,8640"
    horizons: str = "36,72,144,288,432"
    include_sequence_context: bool = True
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    min_split_triggers: int = 10
    high_overlap_jaccard: float = 0.80
    top_k: int = 80


def _mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)), dtype=bool)


def _forward_returns_by_horizon(open_: np.ndarray, horizons: list[int], cfg: EpisodeFeatureStabilityCfg) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)
    safe_open = np.where(open_ > 0.0, open_, np.nan)
    for h in horizons:
        h = int(h)
        entry = int(cfg.entry_delay_bars)
        future_idx = np.arange(len(open_)) + entry + h
        entry_idx = np.arange(len(open_)) + entry
        vals = np.full(len(open_), np.nan, dtype=float)
        ok = (entry_idx >= 0) & (future_idx < len(open_))
        raw = safe_open[future_idx[ok]] / safe_open[entry_idx[ok]] - 1.0
        vals[ok] = float(cfg.leverage) * raw - 2.0 * cost
        out[h] = vals
    return out


def _event_type(event: str) -> str:
    for suffix in sorted(EPISODE_SIDES, key=len, reverse=True):
        if event.endswith("_" + suffix):
            return suffix
    return "unknown"


def _split_stats(vals: list[float], total_bars: int) -> dict[str, Any]:
    st = _trade_stats(vals)
    return {
        "n": len(vals),
        "rate_per_10k_bars": len(vals) / max(1, total_bars) * 10_000.0,
        "mean_ret_pct": float(np.mean(vals)) * 100.0 if vals else 0.0,
        "loss_rate": sum(v <= 0 for v in vals) / len(vals) if vals else 0.0,
        "p_value": st.get("p_value_mean_ret_approx", 1.0),
        "effect_size_d": st.get("effect_size_d", 0.0),
    }


def _stability_flags(train: dict[str, Any], test: dict[str, Any], ev: dict[str, Any], cfg: EpisodeFeatureStabilityCfg) -> list[str]:
    flags = []
    for name, st in (("train", train), ("test", test), ("eval", ev)):
        if int(st["n"]) < int(cfg.min_split_triggers):
            flags.append(f"{name}_sparse")
    if train["n"] >= cfg.min_split_triggers and test["n"] >= cfg.min_split_triggers and np.sign(train["mean_ret_pct"]) != np.sign(test["mean_ret_pct"]):
        flags.append("train_test_sign_flip")
    if test["n"] >= cfg.min_split_triggers and ev["n"] >= cfg.min_split_triggers and np.sign(test["mean_ret_pct"]) != np.sign(ev["mean_ret_pct"]):
        flags.append("test_eval_sign_flip")
    if test["mean_ret_pct"] > 0.0 and ev["mean_ret_pct"] < 0.0:
        flags.append("positive_test_negative_eval")
    if test["n"] > 0 and ev["n"] > 0:
        rate_ratio = max(test["rate_per_10k_bars"], ev["rate_per_10k_bars"]) / max(1e-9, min(test["rate_per_10k_bars"], ev["rate_per_10k_bars"]))
        if rate_ratio >= 3.0:
            flags.append("event_rate_drift_3x")
    return flags


def run(cfg: EpisodeFeatureStabilityCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    dates = pd.to_datetime(market["date"])
    windows = _parse_list(cfg.windows, int)
    horizons = _parse_list(cfg.horizons, int)
    feats = build_episode_event_features(market, windows)
    if cfg.include_sequence_context:
        feats = add_sequence_context_features(market, feats, windows)
    train_mask = _mask(dates, cfg.train_start, cfg.train_end)
    test_mask = _mask(dates, cfg.test_start, cfg.test_end)
    eval_mask = _mask(dates, cfg.eval_start, cfg.eval_end)
    split_masks = {"train": train_mask, "test": test_mask, "eval": eval_mask}
    split_sizes = {k: int(v.sum()) for k, v in split_masks.items()}
    open_ = market["open"].to_numpy(dtype=float)
    fwd_by_horizon = _forward_returns_by_horizon(open_, horizons, cfg)

    candidate_events = []
    for col in feats.columns:
        et = _event_type(col)
        if et == "unknown" or et not in EPISODE_SIDES:
            continue
        if float(feats[col].sum()) <= 0.0:
            continue
        side, episode = EPISODE_SIDES[et]
        event_values = feats[col].to_numpy(dtype=float) > 0.5
        for horizon in horizons:
            signed = fwd_by_horizon[int(horizon)] * (1.0 if side == "LONG" else -1.0)
            split_vals: dict[str, list[float]] = {}
            for split_name, split_mask in split_masks.items():
                vals = signed[event_values & split_mask]
                vals = vals[np.isfinite(vals)]
                split_vals[split_name] = vals.astype(float).tolist()
            st = {name: _split_stats(vals, split_sizes[name]) for name, vals in split_vals.items()}
            flags = _stability_flags(st["train"], st["test"], st["eval"], cfg)
            candidate_events.append({
                "event": col,
                "event_type": et,
                "episode": episode,
                "side": side,
                "horizon": int(horizon),
                "splits": st,
                "flags": flags,
                "overfit_score": (st["test"]["mean_ret_pct"] - st["eval"]["mean_ret_pct"]) if st["test"]["mean_ret_pct"] > 0 else 0.0,
                "abs_test_eval_delta": abs(st["test"]["mean_ret_pct"] - st["eval"]["mean_ret_pct"]),
            })

    # Event-level duplicate audit independent of horizon.
    event_cols = [c for c in feats.columns if _event_type(c) in EPISODE_SIDES and float(feats[c].sum()) > 0.0]
    bools = {c: set(np.flatnonzero(feats[c].to_numpy(dtype=float) > 0.5)) for c in event_cols}
    overlaps = []
    for i, left in enumerate(event_cols):
        a = bools[left]
        if not a:
            continue
        for right in event_cols[i + 1 :]:
            b = bools[right]
            if not b:
                continue
            j = len(a & b) / max(1, len(a | b))
            if j >= float(cfg.high_overlap_jaccard):
                overlaps.append({"left": left, "right": right, "jaccard": j, "left_n": len(a), "right_n": len(b)})

    flag_counts: dict[str, int] = {}
    for row in candidate_events:
        for flag in row["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    report = {
        "config": asdict(cfg),
        "feature_columns": len(feats.columns),
        "episode_event_columns": len(event_cols),
        "template_count": len(candidate_events),
        "split_sizes": split_sizes,
        "flag_counts": dict(sorted(flag_counts.items())),
        "top_positive_test_negative_eval": sorted([r for r in candidate_events if "positive_test_negative_eval" in r["flags"]], key=lambda r: (r["overfit_score"], r["splits"]["test"]["n"]), reverse=True)[: int(cfg.top_k)],
        "top_stable_both_positive": sorted([r for r in candidate_events if r["splits"]["train"]["mean_ret_pct"] > 0 and r["splits"]["test"]["mean_ret_pct"] > 0 and r["splits"]["eval"]["mean_ret_pct"] > 0], key=lambda r: (r["splits"]["eval"]["mean_ret_pct"], r["splits"]["eval"]["n"]), reverse=True)[: int(cfg.top_k)],
        "high_overlap_pairs": sorted(overlaps, key=lambda r: r["jaccard"], reverse=True)[: int(cfg.top_k)],
        "all_summary": sorted(candidate_events, key=lambda r: (len(r["flags"]), -r["splits"]["eval"]["mean_ret_pct"]), reverse=True)[: int(cfg.top_k)],
        "leakage_guard": {"episode_features_use_shifted_prior_ranges": True, "sequence_prior_counts_are_shifted": True, "audit_is_diagnostic_not_selector": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default=EpisodeFeatureStabilityCfg.train_start)
    p.add_argument("--train-end", default=EpisodeFeatureStabilityCfg.train_end)
    p.add_argument("--test-start", default=EpisodeFeatureStabilityCfg.test_start)
    p.add_argument("--test-end", default=EpisodeFeatureStabilityCfg.test_end)
    p.add_argument("--eval-start", default=EpisodeFeatureStabilityCfg.eval_start)
    p.add_argument("--eval-end", default=EpisodeFeatureStabilityCfg.eval_end)
    p.add_argument("--windows", default=EpisodeFeatureStabilityCfg.windows)
    p.add_argument("--horizons", default=EpisodeFeatureStabilityCfg.horizons)
    p.add_argument("--no-sequence-context", dest="include_sequence_context", action="store_false")
    p.set_defaults(include_sequence_context=EpisodeFeatureStabilityCfg.include_sequence_context)
    p.add_argument("--entry-delay-bars", type=int, default=EpisodeFeatureStabilityCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=EpisodeFeatureStabilityCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=EpisodeFeatureStabilityCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=EpisodeFeatureStabilityCfg.slippage_rate)
    p.add_argument("--min-split-triggers", type=int, default=EpisodeFeatureStabilityCfg.min_split_triggers)
    p.add_argument("--high-overlap-jaccard", type=float, default=EpisodeFeatureStabilityCfg.high_overlap_jaccard)
    p.add_argument("--top-k", type=int, default=EpisodeFeatureStabilityCfg.top_k)
    return p.parse_args()


def main() -> None:
    r = run(EpisodeFeatureStabilityCfg(**vars(parse_args())))
    print(json.dumps({
        "output": r["config"]["output"],
        "feature_columns": r["feature_columns"],
        "episode_event_columns": r["episode_event_columns"],
        "template_count": r["template_count"],
        "flag_counts": r["flag_counts"],
        "top_positive_test_negative_eval": r["top_positive_test_negative_eval"][:5],
        "top_stable_both_positive": r["top_stable_both_positive"][:5],
        "high_overlap_pairs": r["high_overlap_pairs"][:5],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
