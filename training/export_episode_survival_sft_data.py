"""Export compact SFT rows for episode path-survival classification.

The prompt contains only causal setup/context descriptors and one candidate
(side+horizon+event).  The target is a JSON survival decision derived from
future net/MAE/MFE path labels for offline training.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_linear_combo_scan import _load_market, _parse_list
from training.audit_episode_survival_quality import _future_path_arrays, _quality_arrays
from training.price_action_episode_policy import EPISODE_SIDES, add_sequence_context_features, build_episode_event_features


@dataclass(frozen=True)
class EpisodeSurvivalSftCfg:
    input_csv: str
    output_dir: str
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31 23:59:59"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    windows: str = "288,576,2016,4032"
    horizons: str = "72,144,288,432"
    include_sequence_context: bool = False
    event_types: str = "failed_breakdown_long,low_sweep_reclaim,reclaim_mid_from_below,failed_mid_loss_long,downtrend_pullback_reject,failed_mid_reclaim_short"
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    min_trade_net_pct: float = 0.25
    max_trade_mae_pct: float = 2.0
    min_mfe_to_mae: float = 1.25
    mae_penalty: float = 0.2
    max_negative_per_positive: int = 4
    max_rows_per_split: int = 60000
    seed: int = 42
    gzip_output: bool = True


def _mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)), dtype=bool)


def _event_type(event: str) -> str:
    for suffix in sorted(EPISODE_SIDES, key=len, reverse=True):
        if event.endswith("_" + suffix):
            return suffix
    return "unknown"


def _bucket(value: float, cuts: tuple[float, float], *, labels: tuple[str, str, str] = ("LOW", "MID", "HIGH")) -> str:
    if not np.isfinite(value):
        return "NA"
    if value <= cuts[0]:
        return labels[0]
    if value <= cuts[1]:
        return labels[1]
    return labels[2]


def _quantiles(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (0.0, 0.0)
    return tuple(float(x) for x in np.quantile(vals, [1 / 3, 2 / 3]))  # type: ignore[return-value]


def _macro_context(market: pd.DataFrame, pos: int) -> dict[str, Any]:
    row = market.iloc[int(pos)]
    def f(name: str) -> float:
        return round(float(row.get(name, 0.0) or 0.0), 4)
    return {
        "dxy_z": f("dxy_zscore"),
        "usdkrw_z": f("usdkrw_zscore"),
        "kimchi_z": f("kimchi_premium_zscore"),
        "kimchi_chg": f("kimchi_premium_change"),
    }


def _target(net: float, mae: float, mfe: float, cfg: EpisodeSurvivalSftCfg) -> dict[str, Any]:
    mfe_to_mae = float(mfe) / max(float(mae), 1e-9)
    utility = float(net) - float(cfg.mae_penalty) * float(mae)
    trade = (
        float(net) * 100.0 >= float(cfg.min_trade_net_pct)
        and float(mae) * 100.0 <= float(cfg.max_trade_mae_pct)
        and mfe_to_mae >= float(cfg.min_mfe_to_mae)
        and utility > 0.0
    )
    if trade:
        conf = "HIGH" if utility * 100.0 >= 0.6 else "MID"
        reason = "path_survives_mae_and_net_thresholds"
    else:
        conf = "LOW"
        if float(mae) * 100.0 > float(cfg.max_trade_mae_pct):
            reason = "adverse_excursion_too_large"
        elif float(net) * 100.0 < float(cfg.min_trade_net_pct):
            reason = "net_edge_too_small"
        elif mfe_to_mae < float(cfg.min_mfe_to_mae):
            reason = "favorable_excursion_not_enough"
        else:
            reason = "utility_not_positive"
    return {
        "decision": "TRADE" if trade else "NO_TRADE",
        "confidence": conf,
        "reason": reason,
        "net_pct": round(float(net) * 100.0, 4),
        "mae_pct": round(float(mae) * 100.0, 4),
        "mfe_pct": round(float(mfe) * 100.0, 4),
        "mfe_to_mae": round(mfe_to_mae, 4),
        "utility_pct": round(utility * 100.0, 4),
    }


def _prompt(date: str, candidate: dict[str, Any], setup: dict[str, Any], macro: dict[str, Any]) -> str:
    return "\n".join([
        "You are a BTCUSDT futures survival filter.",
        "Use only the causal setup/context below.",
        "Decide whether this candidate trade is likely to survive path risk.",
        "Return JSON with keys: decision, confidence, reason.",
        f"date: {date}",
        f"candidate: {json.dumps(candidate, sort_keys=True, separators=(',', ':'))}",
        f"setup_quality: {json.dumps(setup, sort_keys=True, separators=(',', ':'))}",
        f"macro_context: {json.dumps(macro, sort_keys=True, separators=(',', ':'))}",
    ])


def _write_jsonl(path: Path, rows: list[dict[str, Any]], gzip_output: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    mode = "wt"
    with opener(path, mode, encoding="utf-8") as f:  # type: ignore[arg-type]
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _split_sample(rows: list[dict[str, Any]], cfg: EpisodeSurvivalSftCfg, rng: random.Random) -> list[dict[str, Any]]:
    pos = [r for r in rows if json.loads(r["target"])["decision"] == "TRADE"]
    neg = [r for r in rows if json.loads(r["target"])["decision"] != "TRADE"]
    rng.shuffle(pos)
    rng.shuffle(neg)
    max_neg = min(len(neg), max(len(pos) * int(cfg.max_negative_per_positive), int(cfg.max_rows_per_split) - min(len(pos), int(cfg.max_rows_per_split))))
    out = pos + neg[:max_neg]
    if len(out) > int(cfg.max_rows_per_split):
        # Preserve all positives up to cap, then fill with negatives.
        pos_cap = min(len(pos), int(cfg.max_rows_per_split) // 2 if len(pos) > int(cfg.max_rows_per_split) // 2 else len(pos))
        out = pos[:pos_cap] + neg[: max(0, int(cfg.max_rows_per_split) - pos_cap)]
    out.sort(key=lambda r: (r["date"], r["candidate"]["event"], r["candidate"]["horizon"]))
    for r in out:
        if "prompt" not in r:
            r["prompt"] = _prompt(r["date"], r["candidate"], r.pop("setup_quality"), r.pop("macro_context"))
    return out


def run(cfg: EpisodeSurvivalSftCfg) -> dict[str, Any]:
    rng = random.Random(int(cfg.seed))
    market = _load_market(cfg.input_csv)
    dates = pd.to_datetime(market["date"])
    windows = _parse_list(cfg.windows, int)
    horizons = _parse_list(cfg.horizons, int)
    allowed_event_types = {x.strip() for x in str(cfg.event_types).split(",") if x.strip()}
    feats = build_episode_event_features(market, windows)
    if cfg.include_sequence_context:
        feats = add_sequence_context_features(market, feats, windows)
    path_cfg = type("PathCfg", (), asdict(cfg))()
    path = _future_path_arrays(market, horizons, path_cfg)
    q = _quality_arrays(market, path_cfg)
    train_mask = _mask(dates, cfg.train_start, cfg.train_end)
    test_mask = _mask(dates, cfg.test_start, cfg.test_end)
    eval_mask = _mask(dates, cfg.eval_start, cfg.eval_end)
    split_masks = {"train": train_mask, "test": test_mask, "eval": eval_mask}

    # Quantile cuts are global train cuts per side/quality feature, not per event.
    cuts: dict[str, tuple[float, float]] = {}
    for side in ("LONG", "SHORT"):
        cuts[f"{side}_risk_bps"] = _quantiles(q[f"{side}_risk_bps"][train_mask])
        cuts[f"{side}_favorable_wick_frac"] = _quantiles(q[f"{side}_favorable_wick_frac"][train_mask])
        cuts[f"{side}_close_quality"] = _quantiles(q[f"{side}_close_quality"][train_mask])
    cuts["range_bps"] = _quantiles(q["range_bps"][train_mask])
    cuts["body_frac"] = _quantiles(q["body_frac"][train_mask])

    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for col in feats.columns:
        et = _event_type(col)
        if et not in EPISODE_SIDES:
            continue
        if allowed_event_types and et not in allowed_event_types:
            continue
        side, episode = EPISODE_SIDES[et]
        ev_idx = np.flatnonzero(feats[col].to_numpy(dtype=float) > 0.5)
        if len(ev_idx) == 0:
            continue
        for h in horizons:
            p = path[int(h)]
            net = p[f"{side}_net"]
            mae = p[f"{side}_mae"]
            mfe = p[f"{side}_mfe"]
            for pos in ev_idx:
                pos = int(pos)
                if not (np.isfinite(net[pos]) and np.isfinite(mae[pos]) and np.isfinite(mfe[pos])):
                    continue
                split = "train" if train_mask[pos] else "test" if test_mask[pos] else "eval" if eval_mask[pos] else None
                if split is None:
                    continue
                candidate = {"event": col, "event_type": et, "episode": episode, "side": side, "horizon": int(h)}
                setup = {
                    "risk_bucket": _bucket(float(q[f"{side}_risk_bps"][pos]), cuts[f"{side}_risk_bps"]),
                    "risk_bps": round(float(q[f"{side}_risk_bps"][pos]), 2),
                    "range_bucket": _bucket(float(q["range_bps"][pos]), cuts["range_bps"]),
                    "range_bps": round(float(q["range_bps"][pos]), 2),
                    "body_bucket": _bucket(float(q["body_frac"][pos]), cuts["body_frac"]),
                    "body_frac": round(float(q["body_frac"][pos]), 4),
                    "wick_bucket": _bucket(float(q[f"{side}_favorable_wick_frac"][pos]), cuts[f"{side}_favorable_wick_frac"]),
                    "wick_frac": round(float(q[f"{side}_favorable_wick_frac"][pos]), 4),
                    "close_quality_bucket": _bucket(float(q[f"{side}_close_quality"][pos]), cuts[f"{side}_close_quality"]),
                    "close_quality": round(float(q[f"{side}_close_quality"][pos]), 4),
                }
                target = _target(float(net[pos]), float(mae[pos]), float(mfe[pos]), cfg)
                rows_by_split[split].append({
                    "task": "episode_survival_filter_sft",
                    "date": str(dates.iloc[pos]),
                    "signal_pos": pos,
                    "candidate": candidate,
                    "setup_quality": setup,
                    "macro_context": _macro_context(market, pos),
                    "target": json.dumps({k: target[k] for k in ("decision", "confidence", "reason")}, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                    "target_audit": target,
                    "leakage_guard": {"prompt_uses_future_path": False, "target_uses_future_path_for_training_only": True, "bucket_thresholds_fit_on_train_only": True},
                })

    out_dir = Path(cfg.output_dir)
    summary = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "splits": {}, "quantile_cuts": cuts}
    for split in ("train", "test", "eval"):
        sampled = _split_sample(rows_by_split[split], cfg, rng)
        suffix = ".jsonl.gz" if cfg.gzip_output else ".jsonl"
        path_out = out_dir / f"episode_survival_{split}{suffix}"
        _write_jsonl(path_out, sampled, bool(cfg.gzip_output))
        dec = Counter(json.loads(r["target"])["decision"] for r in sampled)
        side = Counter(r["candidate"]["side"] for r in sampled)
        summary["splits"][split] = {"raw_rows": len(rows_by_split[split]), "sampled_rows": len(sampled), "decisions": dict(dec), "sides": dict(side), "output": str(path_out)}
    summary_path = out_dir / "episode_survival_sft_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", required=True)
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(EpisodeSurvivalSftCfg, name.replace("-", "_")))
    p.add_argument("--windows", default=EpisodeSurvivalSftCfg.windows)
    p.add_argument("--horizons", default=EpisodeSurvivalSftCfg.horizons)
    p.add_argument("--include-sequence-context", action="store_true")
    p.add_argument("--event-types", default=EpisodeSurvivalSftCfg.event_types)
    p.add_argument("--entry-delay-bars", type=int, default=EpisodeSurvivalSftCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=EpisodeSurvivalSftCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=EpisodeSurvivalSftCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=EpisodeSurvivalSftCfg.slippage_rate)
    p.add_argument("--min-trade-net-pct", type=float, default=EpisodeSurvivalSftCfg.min_trade_net_pct)
    p.add_argument("--max-trade-mae-pct", type=float, default=EpisodeSurvivalSftCfg.max_trade_mae_pct)
    p.add_argument("--min-mfe-to-mae", type=float, default=EpisodeSurvivalSftCfg.min_mfe_to_mae)
    p.add_argument("--mae-penalty", type=float, default=EpisodeSurvivalSftCfg.mae_penalty)
    p.add_argument("--max-negative-per-positive", type=int, default=EpisodeSurvivalSftCfg.max_negative_per_positive)
    p.add_argument("--max-rows-per-split", type=int, default=EpisodeSurvivalSftCfg.max_rows_per_split)
    p.add_argument("--seed", type=int, default=EpisodeSurvivalSftCfg.seed)
    p.add_argument("--no-gzip-output", dest="gzip_output", action="store_false")
    p.set_defaults(gzip_output=EpisodeSurvivalSftCfg.gzip_output)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EpisodeSurvivalSftCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
