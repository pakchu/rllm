"""Build leak-safe REX combo candidate records for text/LLM rankers.

The REX family cluster is currently the only repeatable weak alpha.  This stage
turns fixed REX family thresholds into candidate rows with:
- a compact categorical prompt for LLM fine-tuning;
- numeric/categorical features for cheap ridge baselines;
- future path reward stored only as the supervised label.

Candidate thresholds are fitted on a configured historical period and then held
fixed for validation/eval rows.  No future outcome enters the prompt.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.event_action_policy_data import EventActionPolicyConfig, _action_utility, _date_mask
from training.event_action_verifier_text_data import _state_text
from training.event_candidate_pool_probe import _feature_candidates, _load_market, _split_mask
from training.rex_combo_rolling_validation import _parse_combo
from training.rex_horizon_sweep import _fast_candidate_rows


@dataclass(frozen=True)
class RexCandidateRankerRecordsCfg:
    input_csv: str
    train_output: str
    eval_output: str
    summary_output: str
    combo: str = "rex_htf_pullback_resume:0.85,rex_htf_pullback_reclaim:0.85"
    threshold_start: str = "2020-01-01"
    threshold_end: str = "2025-01-01"
    train_start: str = "2020-01-01"
    train_end: str = "2026-01-01"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    hold_bars: int = 288
    stride_bars: int = 24
    window_size: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    mae_penalty: float = 1.0


def _bucket(x: float, cuts: tuple[float, ...], labels: tuple[str, ...]) -> str:
    v = float(x)
    for cut, label in zip(cuts, labels):
        if v < cut:
            return label
    return labels[-1]


def _feature_snapshot(features: pd.DataFrame, pos: int, strength: float, threshold: float, *, family: str, side: str) -> dict[str, float]:
    row = features.iloc[int(pos)]
    keep = [
        "trend_12",
        "trend_24",
        "trend_96",
        "range_pos",
        "rsi_norm",
        "bb_z",
        "range_vol",
        "window_drawdown",
        "volume_zscore",
        "taker_imbalance",
        "htf_4h_return_1",
        "htf_4h_return_4",
        "htf_1d_return_1",
        "htf_1d_return_4",
        "htf_3d_return_4",
        "htf_1w_return_4",
        "htf_1d_drawdown_4",
        "htf_1w_drawdown_4",
        "htf_1w_range_pos",
        "dxy_zscore",
        "dxy_momentum",
        "usdkrw_zscore",
        "usdkrw_momentum",
        "kimchi_premium_zscore",
        "kimchi_premium_change",
        "funding_zscore",
        "oi_zscore",
        "oi_change",
        "rex_144_range_pos",
        "rex_576_range_pos",
        "rex_2016_range_pos",
        "rex_8640_range_pos",
        "rex_144_range_width_pct",
        "rex_576_range_width_pct",
        "rex_2016_range_width_pct",
        "rex_8640_range_width_pct",
        "rex_144_max_to_cur_pct",
        "rex_144_cur_to_min_pct",
        "rex_2016_max_to_cur_pct",
        "rex_2016_cur_to_min_pct",
    ]
    out = {k: float(row.get(k, 0.0) or 0.0) for k in keep}
    out["rex_candidate_strength"] = float(strength)
    out["rex_candidate_threshold"] = float(threshold)
    out["rex_threshold_excess"] = float(strength) - float(threshold)
    out["rex_threshold_ratio"] = float(strength) / max(abs(float(threshold)), 1e-12)
    out["family_is_resume"] = 1.0 if family == "rex_htf_pullback_resume" else 0.0
    out["family_is_reclaim"] = 1.0 if family == "rex_htf_pullback_reclaim" else 0.0
    out["candidate_side_sign"] = 1.0 if side == "LONG" else -1.0
    return out


def _state_tokens(features: pd.DataFrame, pos: int, *, family: str, side: str, strength: float, threshold: float) -> dict[str, str]:
    state = {k: float(features.iloc[int(pos)].get(k, 0.0) or 0.0) for k in features.columns}
    tokens: dict[str, str] = {}
    for item in _state_text(state):
        if "=" in item:
            k, v = item.split("=", 1)
            tokens[k] = v
    tokens["candidate_family"] = "resume" if family.endswith("resume") else ("reclaim" if family.endswith("reclaim") else family)
    tokens["candidate_side"] = side.lower()
    tokens["candidate_strength"] = _bucket(float(strength) / max(abs(float(threshold)), 1e-12), (1.05, 1.25, 1.60, 2.20), ("barely_above_threshold", "above_threshold", "strong", "very_strong", "extreme"))
    return tokens


def _prompt(date: str, tokens: dict[str, str], snap: dict[str, float], *, family: str, side: str, hold_bars: int) -> str:
    token_text = "; ".join(f"{k}={tokens[k]}" for k in sorted(tokens))
    numeric_keys = [
        "rex_threshold_ratio",
        "rex_144_range_pos",
        "rex_576_range_pos",
        "rex_2016_range_pos",
        "trend_24",
        "trend_96",
        "htf_1d_return_4",
        "htf_1w_return_4",
        "window_drawdown",
        "volume_zscore",
        "taker_imbalance",
        "dxy_zscore",
        "kimchi_premium_zscore",
        "funding_zscore",
        "oi_change",
    ]
    numeric = "; ".join(f"{k}={float(snap.get(k, 0.0)):+.4f}" for k in numeric_keys)
    return "\n".join(
        [
            "You are a BTCUSDT futures candidate verifier.",
            "The candidate comes from a fixed REX higher-timeframe pullback generator.",
            "Use only signal-time price-action, regime, and candidate context.",
            "Decide whether this exact candidate should be traded or skipped.",
            "Output exactly one label: TAKE or SKIP.",
            "Do not assume future path information.",
            "",
            f"Date: {date}",
            f"Candidate: family={family}; side={side}; hold_bars={int(hold_bars)}",
            f"Categorical context: {token_text}",
            f"Numeric context: {numeric}",
        ]
    )


def _target(reward: dict[str, Any]) -> str:
    net = float(reward.get("net_return", 0.0))
    mae = float(reward.get("mae", 1.0))
    mfe_to_mae = float(reward.get("mfe_to_mae", 0.0))
    utility = float(reward.get("utility", net))
    if net >= 0.006 and mae <= 0.018 and mfe_to_mae >= 1.15 and utility > 0.0:
        return "TAKE"
    return "SKIP"


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def _summ(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(r.get("target")) for r in rows)
    fam = Counter(str(r.get("family")) for r in rows)
    side = Counter(str(r.get("side")) for r in rows)
    rewards = np.asarray([float((r.get("reward") or {}).get("net_return_pct", 0.0) or 0.0) for r in rows], dtype=float) if rows else np.asarray([])
    return {
        "rows": len(rows),
        "period": {"start": rows[0]["date"] if rows else None, "end": rows[-1]["date"] if rows else None},
        "target_counts": dict(sorted(labels.items())),
        "family_counts": dict(sorted(fam.items())),
        "side_counts": dict(sorted(side.items())),
        "net_return_pct": {
            "mean": float(np.mean(rewards)) if len(rewards) else 0.0,
            "positive_rate": float(np.mean(rewards > 0.0)) if len(rewards) else 0.0,
        },
    }


def run(cfg: RexCandidateRankerRecordsCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    families = _feature_candidates(features)
    specs = _parse_combo(cfg.combo)
    dates = pd.to_datetime(market["date"])
    date_strings = [str(x) for x in market["date"].tolist()]
    date_to_pos = {date: i for i, date in enumerate(date_strings)}
    threshold_mask = _split_mask(dates, cfg.threshold_start, cfg.threshold_end)
    thresholds: dict[str, float] = {}
    for spec in specs:
        if spec.family not in families:
            raise ValueError(f"family not found: {spec.family}")
        strength, _direction = families[spec.family]
        train_x = strength[threshold_mask & np.isfinite(strength)]
        if train_x.size < 100:
            raise ValueError(f"too few threshold rows for {spec.family}: {train_x.size}")
        thresholds[spec.family] = float(np.quantile(train_x, float(spec.quantile)))

    masks = {
        "train": _date_mask(dates, cfg.train_start, cfg.train_end),
        "eval": _date_mask(dates, cfg.eval_start, cfg.eval_end),
    }
    pcfg = EventActionPolicyConfig(
        market_csv=cfg.input_csv,
        output=cfg.train_output,
        window_size=cfg.window_size,
        stride_bars=cfg.stride_bars,
        hold_bars_list=(int(cfg.hold_bars),),
        entry_delay_bars=cfg.entry_delay_bars,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        mae_penalty=cfg.mae_penalty,
    )
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    for split, mask in masks.items():
        all_rows: list[dict[str, Any]] = []
        for spec in specs:
            strength, direction = families[spec.family]
            threshold = thresholds[spec.family]
            rows = _fast_candidate_rows(
                date_strings,
                strength,
                direction,
                family=spec.family,
                threshold=threshold,
                mask=mask,
                hold_bars=int(cfg.hold_bars),
                entry_delay_bars=int(cfg.entry_delay_bars),
                stride_bars=int(cfg.stride_bars),
                window_size=int(cfg.window_size),
            )
            for row in rows:
                signal_pos = int(date_to_pos[str(row["signal_date"])])
                reward_raw = _action_utility(market, signal_pos, str(row["side"]), int(cfg.hold_bars), pcfg)
                if reward_raw is None:
                    continue
                snap = _feature_snapshot(features, signal_pos, float(row["strength"]), threshold, family=spec.family, side=str(row["side"]))
                tokens = _state_tokens(features, signal_pos, family=spec.family, side=str(row["side"]), strength=float(row["strength"]), threshold=threshold)
                reward = {
                    "net_return_pct": float(reward_raw["net_return"]) * 100.0,
                    "mae_pct": float(reward_raw["mae"]) * 100.0,
                    "mfe_pct": float(reward_raw["mfe"]) * 100.0,
                    "mfe_to_mae": float(reward_raw["mfe_to_mae"]),
                    "utility_pct": float(reward_raw["utility"]) * 100.0,
                    "net_return": float(reward_raw["net_return"]),
                    "mae": float(reward_raw["mae"]),
                    "mfe": float(reward_raw["mfe"]),
                    "utility": float(reward_raw["utility"]),
                }
                all_rows.append(
                    {
                        "task": "rex_candidate_ranker",
                        "split": split,
                        "date": str(row["signal_date"]),
                        "signal_pos": signal_pos,
                        "family": spec.family,
                        "side": str(row["side"]),
                        "candidate": {"family": spec.family, "side": str(row["side"]), "hold_bars": int(cfg.hold_bars)},
                        "prompt": _prompt(str(row["signal_date"]), tokens, snap, family=spec.family, side=str(row["side"]), hold_bars=int(cfg.hold_bars)),
                        "target": _target(reward),
                        "reward": reward,
                        "state_tokens": tokens,
                        "feature_snapshot": snap,
                        "threshold_fit": {"start": cfg.threshold_start, "end": cfg.threshold_end, "quantile": spec.quantile, "threshold": threshold},
                        "leakage_guard": {
                            "prompt_uses_future_path": False,
                            "target_uses_future_path_for_training_only": True,
                            "candidate_threshold_fit_before_validation_eval": True,
                            "features_signal_time_or_prior": True,
                        },
                    }
                )
        all_rows.sort(key=lambda r: (str(r["date"]), int(r["signal_pos"]), str(r["family"]), str(r["side"])))
        outputs[split] = all_rows

    _write_jsonl(cfg.train_output, outputs["train"])
    _write_jsonl(cfg.eval_output, outputs["eval"])
    summary = {
        "config": asdict(cfg),
        "thresholds": thresholds,
        "train": _summ(outputs["train"]),
        "eval": _summ(outputs["eval"]),
        "outputs": {"train": cfg.train_output, "eval": cfg.eval_output},
        "leakage_guard": {
            "thresholds_fit_on_configured_history_only": True,
            "eval_not_used_for_thresholds": True,
            "prompts_are_past_only": True,
            "labels_are_future_path_only": True,
        },
    }
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--combo", default=RexCandidateRankerRecordsCfg.combo)
    p.add_argument("--threshold-start", default=RexCandidateRankerRecordsCfg.threshold_start)
    p.add_argument("--threshold-end", default=RexCandidateRankerRecordsCfg.threshold_end)
    p.add_argument("--train-start", default=RexCandidateRankerRecordsCfg.train_start)
    p.add_argument("--train-end", default=RexCandidateRankerRecordsCfg.train_end)
    p.add_argument("--eval-start", default=RexCandidateRankerRecordsCfg.eval_start)
    p.add_argument("--eval-end", default=RexCandidateRankerRecordsCfg.eval_end)
    p.add_argument("--hold-bars", type=int, default=RexCandidateRankerRecordsCfg.hold_bars)
    p.add_argument("--stride-bars", type=int, default=RexCandidateRankerRecordsCfg.stride_bars)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RexCandidateRankerRecordsCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
