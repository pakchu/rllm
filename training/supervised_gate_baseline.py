"""Leak-free supervised baseline for executable path-outcome trade gates.

This isolates whether the current past-only engineered prompt features contain
learnable gate information before spending more GPU time on LLM GRPO gates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import EXTENDED_MARKET_FEATURE_COLUMNS, build_market_feature_frame
from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LinearGateConfig:
    epochs: int = 400
    learning_rate: float = 0.05
    l2: float = 0.001
    positive_weight: float = 1.0
    seed: int = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _standardize_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _fit_logistic(x: np.ndarray, y: np.ndarray, cfg: LinearGateConfig) -> tuple[np.ndarray, float, list[float]]:
    rng = np.random.default_rng(int(cfg.seed))
    w = rng.normal(0.0, 0.01, size=x.shape[1])
    b = 0.0
    losses: list[float] = []
    sample_w = np.where(y > 0.5, float(cfg.positive_weight), 1.0)
    denom = max(1e-12, float(sample_w.sum()))
    for _ in range(max(1, int(cfg.epochs))):
        logits = x @ w + b
        p = _sigmoid(logits)
        err = (p - y) * sample_w
        grad_w = (x.T @ err) / denom + float(cfg.l2) * w
        grad_b = float(err.sum() / denom)
        w -= float(cfg.learning_rate) * grad_w
        b -= float(cfg.learning_rate) * grad_b
        eps = 1e-9
        loss = -float(np.sum(sample_w * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))) / denom)
        loss += 0.5 * float(cfg.l2) * float(np.dot(w, w))
        losses.append(loss)
    return w, b, losses


def _build_dataset(
    market: pd.DataFrame,
    *,
    window_size: int,
    path_cfg: PathOutcomeConfig,
    stride_bars: int = 1,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    features = build_market_feature_frame(market, window_size=window_size)
    rows: list[dict[str, Any]] = []
    last_signal_pos = len(market) - max(1, int(path_cfg.entry_delay_bars)) - max(1, int(path_cfg.hold_bars)) - 1
    for pos in range(max(0, int(window_size) - 1), max(0, last_signal_pos) + 1, max(1, int(stride_bars))):
        long = compute_trade_path_outcome(market, pos, "LONG", path_cfg)
        short = compute_trade_path_outcome(market, pos, "SHORT", path_cfg)
        if long is None or short is None:
            continue
        best = long if long.utility >= short.utility else short
        gate = (
            float(best.utility) > float(path_cfg.hold_margin)
            and float(best.net_return) > float(path_cfg.min_net_return)
            and float(best.mae) <= float(path_cfg.max_mae)
        )
        feat = features.iloc[pos]
        rows.append(
            {
                "date": str(pd.to_datetime(market.iloc[pos]["date"])),
                "signal_pos": int(pos),
                "target": int(bool(gate)),
                "side": best.side,
                "best_net_return": float(best.net_return),
                "best_mae": float(best.mae),
                "best_utility": float(best.utility),
                **{col: float(feat.get(col, 0.0)) for col in EXTENDED_MARKET_FEATURE_COLUMNS},
            }
        )
    return features, rows


def _slice_rows(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    a = pd.to_datetime(start)
    b = pd.to_datetime(end)
    return [r for r in rows if a <= pd.to_datetime(r["date"]) <= b]


def _rows_to_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(r.get(col, 0.0)) for col in EXTENDED_MARKET_FEATURE_COLUMNS] for r in rows], dtype=np.float64)
    y = np.asarray([float(r["target"]) for r in rows], dtype=np.float64)
    return x, y


def _classification_metrics(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray([int(r["target"]) for r in rows], dtype=np.int64)
    pred = (scores >= float(threshold)).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "num_samples": int(len(rows)),
        "positive_targets": int((y == 1).sum()),
        "negative_targets": int((y == 0).sum()),
        "predicted_positive": int((pred == 1).sum()),
        "accuracy": float((tp + tn) / max(1, len(rows))),
        "precision_trade": float(precision),
        "recall_trade": float(recall),
        "f1_trade": float(f1),
        "balanced_recall": float(0.5 * (tp / pos + tn / neg)),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "threshold": float(threshold),
    }


def _strict_sim_from_scores(
    rows: list[dict[str, Any]],
    market: pd.DataFrame,
    scores: np.ndarray,
    *,
    threshold: float,
    path_cfg: PathOutcomeConfig,
    fee_rate: float,
    slippage_rate: float,
    leverage: float,
    cooldown_bars: int,
) -> dict[str, Any]:
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    trade_returns: list[float] = []
    entries = 0
    next_allowed_pos = 0
    skipped = 0
    for row, score in zip(rows, scores):
        if float(score) < float(threshold):
            continue
        signal_pos = int(row["signal_pos"])
        if signal_pos < next_allowed_pos:
            continue
        entry_pos = signal_pos + max(0, int(path_cfg.entry_delay_bars))
        exit_pos = entry_pos + max(1, int(path_cfg.hold_bars))
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            skipped += 1
            continue
        side = str(row["side"]).upper()
        signal = 1 if side == "LONG" else -1
        entry_eq = eq
        entries += 1
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak if peak > 0 else 0.0)
        for j in range(entry_pos, exit_pos):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            if signal > 0:
                adverse_ret = (float(lows[j]) - open_j) / open_j
                close_ret = (float(opens[j + 1]) - open_j) / open_j
            else:
                adverse_ret = (open_j - float(highs[j])) / open_j
                close_ret = (open_j - float(opens[j + 1])) / open_j
            adverse_eq = eq * (1.0 + float(leverage) * adverse_ret)
            max_dd = max(max_dd, 1.0 - max(0.0, adverse_eq) / peak if peak > 0 else 0.0)
            eq *= max(0.0, 1.0 + float(leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak if peak > 0 else 0.0)
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        next_allowed_pos = exit_pos + max(0, int(cooldown_bars))
        if eq <= 0.0:
            break
    if not rows:
        years = 1.0 / 365.25
        start = end = None
    else:
        start = str(rows[0]["date"])
        end = str(rows[-1]["date"])
        years = max(1.0 / 365.25, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / (365.25 * 24 * 3600))
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "period": {"start": start, "end": end, "years": float(years)},
        "sim": {
            "ret_pct": float(ret_pct),
            "cagr_pct": float(cagr_pct),
            "strict_mdd_pct": float(mdd_pct),
            "cagr_to_strict_mdd": float(cagr_pct / mdd_pct) if mdd_pct > 1e-12 else float("inf"),
            "trade_entries": int(entries),
            "samples": int(len(rows)),
            "skipped_missing_bars": int(skipped),
            "return_application": "supervised_gate_oracle_side_actual_ohlc_bar_by_bar_strict_mdd",
        },
        "trade_stats": _trade_stats(trade_returns),
    }


def _select_threshold(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    market: pd.DataFrame,
    *,
    path_cfg: PathOutcomeConfig,
    fee_rate: float,
    slippage_rate: float,
    leverage: float,
    cooldown_bars: int,
    min_trades: int,
    stride_bars: int = 1,
) -> dict[str, Any]:
    candidates = sorted(set(float(x) for x in np.quantile(scores, np.linspace(0.05, 0.95, 19))))
    candidates.extend([0.25, 0.5, 0.75])
    best = None
    for th in sorted(set(candidates)):
        rep = _strict_sim_from_scores(
            rows,
            market,
            scores,
            threshold=th,
            path_cfg=path_cfg,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            leverage=leverage,
            cooldown_bars=cooldown_bars,
        )
        if int(rep["sim"]["trade_entries"]) < int(min_trades):
            continue
        row = {"threshold": float(th), "strict": rep, "classification": _classification_metrics(rows, scores, th)}
        key = (float(rep["sim"]["cagr_to_strict_mdd"]), float(rep["sim"]["cagr_pct"]), -float(rep["sim"]["strict_mdd_pct"]))
        if best is None or key > best[0]:
            best = (key, row)
    if best is None:
        th = 0.5
        return {"threshold": th, "strict": _strict_sim_from_scores(rows, market, scores, threshold=th, path_cfg=path_cfg, fee_rate=fee_rate, slippage_rate=slippage_rate, leverage=leverage, cooldown_bars=cooldown_bars), "classification": _classification_metrics(rows, scores, th)}
    return best[1]


def run_baseline(
    *,
    market_csv: str,
    output: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    eval_start: str,
    eval_end: str,
    window_size: int,
    hold_bars: int,
    entry_delay_bars: int,
    fee_rate: float,
    slippage_rate: float,
    leverage: float,
    mae_penalty: float,
    mfe_bonus: float,
    min_net_return: float,
    min_utility: float,
    max_mae: float,
    positive_weight: float,
    epochs: int,
    learning_rate: float,
    l2: float,
    cooldown_bars: int,
    min_trades: int,
    stride_bars: int = 1,
) -> dict[str, Any]:
    market = pd.read_csv(market_csv)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(market.columns)
    if missing:
        raise ValueError(f"market csv lacks required columns: {sorted(missing)}")
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    for col in ("open", "high", "low", "close", "volume"):
        market[col] = market[col].astype(float)
    path_cfg = PathOutcomeConfig(
        hold_bars=hold_bars,
        entry_delay_bars=entry_delay_bars,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        leverage=leverage,
        mae_penalty=mae_penalty,
        mfe_bonus=mfe_bonus,
        hold_margin=min_utility,
        min_net_return=min_net_return,
        max_mae=max_mae,
    )
    _, all_rows = _build_dataset(market, window_size=window_size, path_cfg=path_cfg, stride_bars=stride_bars)
    train_rows = _slice_rows(all_rows, train_start, train_end)
    test_rows = _slice_rows(all_rows, test_start, test_end)
    eval_rows = _slice_rows(all_rows, eval_start, eval_end)
    x_train, y_train = _rows_to_xy(train_rows)
    x_train_z, mean, std = _standardize_train(x_train)
    cfg = LinearGateConfig(epochs=epochs, learning_rate=learning_rate, l2=l2, positive_weight=positive_weight)
    w, b, losses = _fit_logistic(x_train_z, y_train, cfg)

    def score(rows: list[dict[str, Any]]) -> np.ndarray:
        x, _ = _rows_to_xy(rows)
        return _sigmoid(_standardize_apply(x, mean, std) @ w + b)

    train_scores = score(train_rows)
    test_scores = score(test_rows)
    eval_scores = score(eval_rows)
    selected = _select_threshold(
        test_rows,
        test_scores,
        market,
        path_cfg=path_cfg,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        leverage=leverage,
        cooldown_bars=cooldown_bars,
        min_trades=min_trades,
    )
    th = float(selected["threshold"])
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "files": {"market_csv": str(Path(market_csv).resolve())},
        "periods": {"train": [train_start, train_end], "test": [test_start, test_end], "eval": [eval_start, eval_end]},
        "path_outcome": path_cfg.__dict__,
        "sampling": {"stride_bars": int(stride_bars)},
        "model": {"type": "standardized_logistic_regression_numpy", **cfg.__dict__, "final_loss": float(losses[-1])},
        "feature_columns": list(EXTENDED_MARKET_FEATURE_COLUMNS),
        "selected_threshold_from_test": th,
        "splits": {
            "train": {"classification": _classification_metrics(train_rows, train_scores, th), "strict": _strict_sim_from_scores(train_rows, market, train_scores, threshold=th, path_cfg=path_cfg, fee_rate=fee_rate, slippage_rate=slippage_rate, leverage=leverage, cooldown_bars=cooldown_bars)},
            "test": selected,
            "eval": {"classification": _classification_metrics(eval_rows, eval_scores, th), "strict": _strict_sim_from_scores(eval_rows, market, eval_scores, threshold=th, path_cfg=path_cfg, fee_rate=fee_rate, slippage_rate=slippage_rate, leverage=leverage, cooldown_bars=cooldown_bars)},
        },
        "leakage_guard": {
            "features_are_past_only": True,
            "threshold_selected_on": "test",
            "eval_used_for_selection": False,
            "side_is_oracle_future_label": True,
            "deployable_policy": False,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised executable gate baseline")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", default="results/supervised_gate_baseline.json")
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--train-end", default="2023-03-31")
    p.add_argument("--test-start", default="2023-04-01")
    p.add_argument("--test-end", default="2023-06-30")
    p.add_argument("--eval-start", default="2023-07-01")
    p.add_argument("--eval-end", default="2023-09-30")
    p.add_argument("--window-size", type=int, default=96)
    p.add_argument("--hold-bars", type=int, default=144)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--mae-penalty", type=float, default=1.0)
    p.add_argument("--mfe-bonus", type=float, default=0.0)
    p.add_argument("--min-net-return", type=float, default=0.001)
    p.add_argument("--min-utility", type=float, default=0.0)
    p.add_argument("--max-mae", type=float, default=0.015)
    p.add_argument("--positive-weight", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=0.001)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--stride-bars", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_baseline(**vars(args))
    print(json.dumps({
        "selected_threshold_from_test": out["selected_threshold_from_test"],
        "train": out["splits"]["train"],
        "test": out["splits"]["test"],
        "eval": out["splits"]["eval"],
        "leakage_guard": out["leakage_guard"],
    }, indent=2))


if __name__ == "__main__":
    main()
