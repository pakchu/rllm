"""Build rich text-state rows for LLM/RL ranking of robust wave candidates.

Rows are generated only from a pre-selected, fold-consistent wave candidate
policy.  Prompt/context fields contain past-only bucketed market/macro states at
signal time.  Realized trade reward is stored as target metadata for supervised
or RL-style training, not as an input feature.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import EXTENDED_MARKET_FEATURE_COLUMNS, build_market_feature_frame
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.sweep_wave_fold_consistency import _load_closes, _parse_folds, _policy_side, _write_policy_predictions
from training.sweep_wave_teacher_rllm_thresholds import _rolling_prob_rows
from training.validate_wave_trading_best import _build_best_features, _load_wave_module


@dataclass(frozen=True)
class WaveLlmStateDatasetConfig:
    wave_root: str
    market_5m_csv: str
    fold_consistency_report: str
    train_output: str
    eval_output: str
    summary_output: str
    start_date: str = "2020-01-01"
    end_date: str = "2026-06-02"
    selection_folds: str = "2021-01-01|2021-06-30 23:59:59,2021-07-01|2021-12-31 23:59:59,2022-01-01|2022-06-30 23:59:59,2022-07-01|2022-12-31 23:59:59,2023-01-01|2023-06-30 23:59:59,2023-07-01|2023-12-31 23:59:59,2024-01-01|2024-06-30 23:59:59"
    eval_start: str = "2024-07-01"
    eval_end: str = "2026-06-01 00:00:00"
    lr_c: float = 0.05
    lr_penalty: str = "l1"
    leverage: float = 1.0
    entry_delay_bars: int = 3
    atr_trailing_stop_mult: float = 3.75
    atr_period: int = 45
    attach_external: bool = True
    external_tolerance: str = "30min"


def _load_market(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)


def _bucket_signed(x: float, *, small: float, large: float) -> str:
    if x <= -large:
        return "strong_down"
    if x <= -small:
        return "down"
    if x < small:
        return "flat"
    if x < large:
        return "up"
    return "strong_up"


def _bucket_abs(x: float, *, low: float, high: float) -> str:
    ax = abs(float(x))
    if ax < low:
        return "low"
    if ax < high:
        return "medium"
    return "high"


def _bucket_z(x: float) -> str:
    return _bucket_signed(float(x), small=0.75, large=1.75)


def _bucket_range_pos(x: float) -> str:
    if x <= -0.65:
        return "near_low"
    if x <= -0.25:
        return "lower_half"
    if x < 0.25:
        return "middle"
    if x < 0.65:
        return "upper_half"
    return "near_high"


def _bucket_rex_width(x: float) -> str:
    return _bucket_abs(float(x), low=0.025, high=0.08)


def _bucket_rex_gap(x: float) -> str:
    ax = abs(float(x))
    if ax < 0.01:
        return "touching"
    if ax < 0.03:
        return "near"
    if ax < 0.08:
        return "mid"
    return "far"


def _safe_feature(features: pd.DataFrame, pos: int, col: str) -> float:
    if col not in features.columns or pos < 0 or pos >= len(features):
        return 0.0
    val = features.iloc[pos][col]
    try:
        if pd.isna(val):
            return 0.0
    except TypeError:
        return 0.0
    return float(val)


def _state_tokens(features: pd.DataFrame, pos: int, side: str) -> dict[str, str]:
    side_sign = 1 if side == "LONG" else -1 if side == "SHORT" else 0
    t12 = _safe_feature(features, pos, "trend_12")
    t96 = _safe_feature(features, pos, "trend_96")
    h4 = _safe_feature(features, pos, "htf_4h_return_4")
    d1 = _safe_feature(features, pos, "htf_1d_return_4")
    w1 = _safe_feature(features, pos, "htf_1w_return_4")
    range_pos = _safe_feature(features, pos, "range_pos")
    rsi = _safe_feature(features, pos, "rsi_norm")
    vol = _safe_feature(features, pos, "range_vol")
    dxy = _safe_feature(features, pos, "dxy_momentum")
    kimchi = _safe_feature(features, pos, "kimchi_premium_zscore")
    usdkrw = _safe_feature(features, pos, "usdkrw_momentum")
    side_t12 = side_sign * t12
    side_d1 = side_sign * d1
    rex_tokens: dict[str, str] = {}
    for rex_window in (36, 144, 576, 2016, 8640):
        prefix = f"rex_{rex_window}"
        rex_tokens[f"{prefix}_loc"] = _bucket_range_pos(_safe_feature(features, pos, f"{prefix}_range_pos"))
        rex_tokens[f"{prefix}_width"] = _bucket_rex_width(_safe_feature(features, pos, f"{prefix}_range_width_pct"))
        rex_tokens[f"{prefix}_upper_gap"] = _bucket_rex_gap(_safe_feature(features, pos, f"{prefix}_max_to_cur_pct"))
        rex_tokens[f"{prefix}_lower_gap"] = _bucket_rex_gap(_safe_feature(features, pos, f"{prefix}_cur_to_min_pct"))
    return {
        "short_trend": _bucket_signed(t12, small=0.003, large=0.01),
        "session_trend": _bucket_signed(t96, small=0.006, large=0.02),
        "four_hour_context": _bucket_signed(h4, small=0.006, large=0.02),
        "daily_context": _bucket_signed(d1, small=0.01, large=0.04),
        "weekly_context": _bucket_signed(w1, small=0.02, large=0.08),
        "range_location": _bucket_signed(range_pos, small=0.25, large=0.65),
        "oscillator_pressure": _bucket_signed(rsi, small=0.20, large=0.55),
        "volatility": _bucket_abs(vol, low=0.025, high=0.07),
        "candidate_alignment_short": _bucket_signed(side_t12, small=0.003, large=0.01),
        "candidate_alignment_daily": _bucket_signed(side_d1, small=0.01, large=0.04),
        "dxy_pressure": _bucket_signed(dxy, small=0.001, large=0.004),
        "kimchi_pressure": _bucket_z(kimchi),
        "usdkrw_pressure": _bucket_signed(usdkrw, small=0.001, large=0.004),
        "external_availability": "available" if _safe_feature(features, pos, "external_any_available") > 0.5 else "missing_or_partial",
        **rex_tokens,
    }


def _prompt(row: dict[str, Any], tokens: dict[str, str], policy: dict[str, Any]) -> str:
    side = str(row["side"])
    prob = float(row.get("teacher_probability_long", 0.5))
    margin = prob - float(policy["long_th"]) if side == "LONG" else float(policy["short_th"]) - prob
    confidence_bucket = "thin" if margin < 0.02 else "normal" if margin < 0.06 else "wide"
    lines = [
        "Task: decide whether to take, reduce, or reject this pre-generated BTCUSDT futures candidate.",
        "Use only the bucketed state below. Do not infer from future outcome metadata.",
        f"Candidate: side={side}; source=15m_wave_teacher; confidence_margin={confidence_bucket}; execution=next_15m_open; exit=atr_trailing_or_time.",
        "State buckets:",
    ]
    for k in sorted(tokens):
        lines.append(f"- {k}: {tokens[k]}")
    lines.append("Return JSON with decision in {TAKE_FULL, TAKE_SMALL, ABSTAIN} and a short risk reason.")
    return "\n".join(lines)


def _target_from_reward(reward_pct: float) -> dict[str, str]:
    if reward_pct >= 0.75:
        return {"decision": "TAKE_FULL", "risk_reason": "realized_reward_strong_positive"}
    if reward_pct > 0.0:
        return {"decision": "TAKE_SMALL", "risk_reason": "realized_reward_positive_but_thin"}
    return {"decision": "ABSTAIN", "risk_reason": "realized_reward_non_positive"}


def _rows_for_period(*, rows: list[dict[str, Any]], market_csv: str, features: pd.DataFrame, closes: np.ndarray, policy: dict[str, Any], tmp: Path, tag: str, cfg: WaveLlmStateDatasetConfig, hold_bars: int, split: str) -> list[dict[str, Any]]:
    pred = tmp / f"{tag}.jsonl"
    _write_policy_predictions(rows, pred, closes=closes, policy=policy, hold_bars=hold_bars)
    bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred), market_csv=market_csv, output=str(tmp / f"{tag}.bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, atr_trailing_stop_mult=cfg.atr_trailing_stop_mult, atr_period=cfg.atr_period))
    source_by_pos = {int(r["signal_pos"]): r for r in rows}
    out: list[dict[str, Any]] = []
    for ex in bt["executed"]:
        pos = int(ex["signal_pos"])
        src = source_by_pos.get(pos)
        if not src:
            continue
        side = str(ex["side"])
        tokens = _state_tokens(features, pos, side)
        reward_pct = float(ex["trade_ret_pct"])
        row = {
            "split": split,
            "date": ex.get("date"),
            "signal_pos": pos,
            "side": side,
            "prompt": _prompt({**src, "side": side}, tokens, policy),
            "target": _target_from_reward(reward_pct),
            "reward": {"trade_ret_pct": reward_pct, "equity_after_trade": float(ex.get("equity", 0.0)), "exit_reason": ex.get("exit_reason")},
            "candidate": {"teacher_probability_long": float(src.get("teacher_probability_long", 0.5)), "policy": policy, "hold_bars": hold_bars},
            "state_tokens": tokens,
            "leakage_guard": {"prompt_uses_future_reward": False, "reward_is_label_only": True, "features_signal_time_or_prior": True},
        }
        out.append(row)
    return out


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def run_build(cfg: WaveLlmStateDatasetConfig) -> dict[str, Any]:
    fc = json.loads(Path(cfg.fold_consistency_report).read_text())
    policy = fc["selected_policies"][0]
    psr = _load_wave_module(cfg.wave_root)
    data = _build_best_features(psr, start_date=cfg.start_date, end_date=cfg.end_date, time_interval="15m")
    hold_bars = int(data["params"]["holding_period"]) * 3
    market = _load_market(cfg.market_5m_csv)
    enriched = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_root, tolerance=cfg.external_tolerance) if cfg.attach_external else market
    features = build_market_feature_frame(enriched)
    for col in EXTENDED_MARKET_FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = 0.0
    closes = _load_closes(cfg.market_5m_csv)
    folds = _parse_folds(cfg.selection_folds)
    train_prob_rows = [_rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=a, eval_end=b, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty) for a, b in folds]
    eval_prob_rows = _rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=cfg.eval_start, eval_end=cfg.eval_end, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty)
    train_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rllm_wave_llm_state_") as tmp_raw:
        tmp = Path(tmp_raw)
        for i, rows in enumerate(train_prob_rows):
            train_rows.extend(_rows_for_period(rows=rows, market_csv=cfg.market_5m_csv, features=features, closes=closes, policy=policy, tmp=tmp, tag=f"train_f{i}", cfg=cfg, hold_bars=hold_bars, split="train"))
        eval_rows = _rows_for_period(rows=eval_prob_rows, market_csv=cfg.market_5m_csv, features=features, closes=closes, policy=policy, tmp=tmp, tag="eval", cfg=cfg, hold_bars=hold_bars, split="eval")
    _write_jsonl(cfg.train_output, train_rows)
    _write_jsonl(cfg.eval_output, eval_rows)
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        rewards = np.asarray([float(r["reward"]["trade_ret_pct"]) for r in rows], dtype=float) if rows else np.asarray([], dtype=float)
        decisions: dict[str, int] = {}
        sides: dict[str, int] = {}
        for r in rows:
            decisions[str(r["target"]["decision"])] = decisions.get(str(r["target"]["decision"]), 0) + 1
            sides[str(r["side"])] = sides.get(str(r["side"]), 0) + 1
        return {
            "rows": len(rows),
            "mean_reward_pct": float(np.mean(rewards)) if len(rewards) else 0.0,
            "std_reward_pct": float(np.std(rewards)) if len(rewards) else 0.0,
            "positive_rate": float(np.mean(rewards > 0.0)) if len(rewards) else 0.0,
            "decisions": decisions,
            "sides": sides,
        }
    summary = {
        "config": asdict(cfg),
        "policy": policy,
        "teacher_params": data["params"],
        "outputs": {"train": cfg.train_output, "eval": cfg.eval_output},
        "train": summarize(train_rows),
        "eval": summarize(eval_rows),
        "prompt_contract": "prompt contains bucketed past-only state; reward and target are label-only",
        "leakage_guard": {"rolling_teacher_train_before_candidate_period": True, "features_signal_time_or_prior": True, "external_join_backward_asof": bool(cfg.attach_external), "eval_rows_not_used_for_training": True},
    }
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build rich text-state LLM/RL rows from robust wave candidates")
    p.add_argument("--wave-root", default="/home/pakchu/workspace/wave_trading")
    p.add_argument("--market-5m-csv", required=True)
    p.add_argument("--fold-consistency-report", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2026-06-02")
    p.add_argument("--selection-folds", default=WaveLlmStateDatasetConfig.selection_folds)
    p.add_argument("--eval-start", default="2024-07-01")
    p.add_argument("--eval-end", default="2026-06-01 00:00:00")
    p.add_argument("--lr-c", type=float, default=0.05)
    p.add_argument("--lr-penalty", default="l1")
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--entry-delay-bars", type=int, default=3)
    p.add_argument("--atr-trailing-stop-mult", type=float, default=3.75)
    p.add_argument("--atr-period", type=int, default=45)
    p.add_argument("--no-external", dest="attach_external", action="store_false")
    p.add_argument("--external-tolerance", default="30min")
    p.set_defaults(attach_external=True)
    return p.parse_args()


def main() -> None:
    summary = run_build(WaveLlmStateDatasetConfig(**vars(parse_args())))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
