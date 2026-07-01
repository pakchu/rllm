"""Build dense take/skip option rows from cached wave probability predictions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import EXTENDED_MARKET_FEATURE_COLUMNS, build_market_feature_frame
from training.build_wave_llm_state_dataset import _bucket_signed, _safe_feature, _state_tokens


@dataclass(frozen=True)
class DenseWaveProbOptionCfg:
    train_predictions_jsonl: str
    eval_predictions_jsonl: str
    market_csv: str
    train_output: str
    eval_output: str
    summary_output: str
    min_long_prob: float = 0.54
    max_short_prob: float = 0.46
    hold_bars: int = 12
    entry_delay_bars: int = 3
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    max_rows_per_split: int = 0
    exit_model: str = "fixed"
    atr_trailing_stop_mult: float = 3.75
    atr_period: int = 45


def _read(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _prob_bucket(p: float, side: str) -> str:
    edge = p - 0.5 if side == "LONG" else 0.5 - p
    if edge < 0.04:
        return "thin"
    if edge < 0.08:
        return "normal"
    if edge < 0.14:
        return "wide"
    return "extreme"


def _rolling_atr(high: np.ndarray, low: np.ndarray, open_: np.ndarray, period: int) -> np.ndarray:
    period = max(1, int(period))
    prev_close = np.roll(open_, 1)
    prev_close[0] = open_[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    out = np.empty_like(tr, dtype=float)
    csum = np.cumsum(tr, dtype=float)
    for i in range(len(tr)):
        start = max(0, i - period + 1)
        total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        out[i] = total / float(i - start + 1)
    return out


def _fixed_hold_reward(market: pd.DataFrame, signal_pos: int, side: str, hold_bars: int, entry_delay_bars: int, cost: float) -> float | None:
    entry = int(signal_pos) + int(entry_delay_bars)
    exit_ = entry + int(hold_bars)
    if entry < 0 or exit_ >= len(market):
        return None
    ep = float(market.iloc[entry]["open"])
    xp = float(market.iloc[exit_]["open"])
    if ep <= 0 or xp <= 0:
        return None
    raw = (xp - ep) / ep if side == "LONG" else (ep - xp) / ep
    return (raw - 2.0 * float(cost)) * 100.0


def _atr_hold_reward(market: pd.DataFrame, atr: np.ndarray, signal_pos: int, side: str, hold_bars: int, entry_delay_bars: int, cost: float, atr_mult: float) -> float | None:
    entry = int(signal_pos) + int(entry_delay_bars)
    exit_ = entry + int(hold_bars)
    if entry < 0 or exit_ >= len(market):
        return None
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    entry_price = float(opens[entry])
    if entry_price <= 0.0:
        return None
    signal = 1 if side == "LONG" else -1
    atr_ref = max(0, entry - 1)
    atr_dist = float(atr[atr_ref]) * float(atr_mult)
    stop = entry_price - atr_dist if signal > 0 else entry_price + atr_dist
    exit_price = float(opens[exit_])
    for j in range(entry, exit_):
        if signal > 0:
            if float(lows[j]) <= stop:
                exit_price = stop
                break
            stop = max(stop, float(highs[j]) - atr_dist)
        else:
            if float(highs[j]) >= stop:
                exit_price = stop
                break
            stop = min(stop, float(lows[j]) + atr_dist)
    raw = (exit_price - entry_price) / entry_price if signal > 0 else (entry_price - exit_price) / entry_price
    return (raw - 2.0 * float(cost)) * 100.0


def _prompt(date: str, signal_pos: int, side: str, prob: float, tokens: dict[str, str], hold_bars: int) -> str:
    lines = [
        "Task: decide whether this BTCUSDT futures candidate should be traded.",
        "Use only the signal-time state card. Answer exactly one letter: A or B.",
        "A = TAKE_TRADE",
        "B = SKIP_TRADE",
        f"Candidate: side={side}; source=wave_probability_dense; probability_long_bucket={_prob_bucket(prob, side)}; hold_bars={hold_bars}; execution=delayed_open.",
        f"Date: {date}",
        f"Signal position: {signal_pos}",
        "State buckets:",
    ]
    for k in sorted(tokens):
        lines.append(f"- {k}: {tokens[k]}")
    return "\n".join(lines)


def _rows(src: list[dict[str, Any]], market: pd.DataFrame, features: pd.DataFrame, cfg: DenseWaveProbOptionCfg, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cost = float(cfg.fee_rate) + float(cfg.slippage_rate)
    atr = _rolling_atr(market["high"].to_numpy(dtype=float), market["low"].to_numpy(dtype=float), market["open"].to_numpy(dtype=float), int(cfg.atr_period)) if str(cfg.exit_model).lower() == "atr" else None
    for r in src:
        pos = int(r.get("signal_pos", -1) or -1)
        prob = float(r.get("teacher_probability_long", 0.5) or 0.5)
        candidates: list[str] = []
        if prob >= float(cfg.min_long_prob):
            candidates.append("LONG")
        if prob <= float(cfg.max_short_prob):
            candidates.append("SHORT")
        for side in candidates:
            if atr is not None:
                reward = _atr_hold_reward(market, atr, pos, side, int(cfg.hold_bars), int(cfg.entry_delay_bars), cost, float(cfg.atr_trailing_stop_mult))
            else:
                reward = _fixed_hold_reward(market, pos, side, int(cfg.hold_bars), int(cfg.entry_delay_bars), cost)
            if reward is None:
                continue
            tokens = _state_tokens(features, pos, side)
            tokens["wave_prob_edge"] = _prob_bucket(prob, side)
            tokens["wave_prob_direction"] = _bucket_signed(prob - 0.5, small=0.04, large=0.10)
            target = "A" if reward > 0.0 else "B"
            rows.append({
                "task": "wave_probability_dense_take_skip_option",
                "split": split,
                "date": r.get("date"),
                "signal_pos": pos,
                "side": side,
                "prompt": _prompt(str(r.get("date")), pos, side, prob, tokens, int(cfg.hold_bars)),
                "target": target,
                "choice_utility": {"A": reward, "B": 0.0},
                "source": {"teacher_probability_long": prob, "state_tokens": tokens, "fixed_hold_reward_pct": reward},
                "leakage_guard": {"prompt_uses_future_reward": False, "target_uses_future_reward_for_training_only": True, "features_signal_time_or_prior": True},
            })
    if int(cfg.max_rows_per_split) > 0 and len(rows) > int(cfg.max_rows_per_split):
        # Chronological thinning preserves time coverage deterministically.
        idx = np.linspace(0, len(rows) - 1, int(cfg.max_rows_per_split)).round().astype(int)
        rows = [rows[int(i)] for i in idx]
    return rows


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(r["choice_utility"]["A"]) for r in rows]
    return {"rows": len(rows), "target_counts": dict(sorted(Counter(str(r["target"]) for r in rows).items())), "side_counts": dict(sorted(Counter(str(r["side"]) for r in rows).items())), "mean_reward_pct": float(np.mean(rewards)) if rewards else 0.0, "positive_rate": float(np.mean(np.asarray(rewards) > 0.0)) if rewards else 0.0}


def run(cfg: DenseWaveProbOptionCfg) -> dict[str, Any]:
    market = _load_market(cfg.market_csv)
    features = build_market_feature_frame(market)
    for c in EXTENDED_MARKET_FEATURE_COLUMNS:
        if c not in features.columns:
            features[c] = 0.0
    train = _rows(_read(cfg.train_predictions_jsonl), market, features, cfg, "train")
    ev = _rows(_read(cfg.eval_predictions_jsonl), market, features, cfg, "eval")
    _write(cfg.train_output, train)
    _write(cfg.eval_output, ev)
    report = {"config": asdict(cfg), "outputs": {"train": cfg.train_output, "eval": cfg.eval_output}, "train": _summary(train), "eval": _summary(ev), "leakage_guard": {"features_signal_time_or_prior": True, "future_rewards_label_only": True}}
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-predictions-jsonl", required=True)
    p.add_argument("--eval-predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--min-long-prob", type=float, default=DenseWaveProbOptionCfg.min_long_prob)
    p.add_argument("--max-short-prob", type=float, default=DenseWaveProbOptionCfg.max_short_prob)
    p.add_argument("--hold-bars", type=int, default=DenseWaveProbOptionCfg.hold_bars)
    p.add_argument("--entry-delay-bars", type=int, default=DenseWaveProbOptionCfg.entry_delay_bars)
    p.add_argument("--fee-rate", type=float, default=DenseWaveProbOptionCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=DenseWaveProbOptionCfg.slippage_rate)
    p.add_argument("--max-rows-per-split", type=int, default=DenseWaveProbOptionCfg.max_rows_per_split)
    p.add_argument("--exit-model", choices=["fixed", "atr"], default=DenseWaveProbOptionCfg.exit_model)
    p.add_argument("--atr-trailing-stop-mult", type=float, default=DenseWaveProbOptionCfg.atr_trailing_stop_mult)
    p.add_argument("--atr-period", type=int, default=DenseWaveProbOptionCfg.atr_period)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(DenseWaveProbOptionCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
