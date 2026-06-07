"""Fold-stable linear value trader over compact LLM/analyzer context.

This is intentionally lightweight and dependency-free (numpy only):
- train split only builds categorical vocabulary and ridge value model
- labels are realized net returns for hypothetical LONG/SHORT actions
- prediction uses expected return minus uncertainty penalty
- config selection is by chronological fold stability, not a single eval period
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.economic_fold_stability_sweep import aggregate_score, default_folds, load_all_rows, score_fold_result, split_rows
from training.economic_pressure_backtest import PressureBacktestConfig, strict_pressure_backtest
from training.economic_pressure_value_calibration import SIDE_TO_LABEL, context_values, simulate_trade_return
from training.strict_bar_backtest import load_market_bars


@dataclass
class LinearValueModel:
    vocab: dict[str, int]
    beta: np.ndarray
    xtx_inv: np.ndarray
    residual_std: float
    alpha: float


def feature_tokens(row: dict[str, Any], side: str) -> list[str]:
    vals = context_values(row)
    tokens = ["bias=1", f"side={side}"]
    for k, v in vals.items():
        if k == "tags":
            for tag in str(v).split("+"):
                if tag:
                    tokens.append(f"tag={tag}")
        else:
            tokens.append(f"{k}={v}")
    # interaction features carry the analyzer's categorical reasoning into side-specific value.
    for k in ["teacher_pressure", "regime", "trend_alignment", "risk_state", "order_flow", "macro_dollar", "kimchi"]:
        tokens.append(f"side:{side}|{k}={vals.get(k, 'NA')}")
    return tokens


def build_vocab(rows: list[dict[str, Any]], *, min_count: int = 5) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for side in ["LONG", "SHORT"]:
            for tok in feature_tokens(row, side):
                counts[tok] = counts.get(tok, 0) + 1
    vocab = {"bias=1": 0}
    for tok, cnt in sorted(counts.items()):
        if tok == "bias=1":
            continue
        if cnt >= min_count:
            vocab[tok] = len(vocab)
    return vocab


def vectorize(row: dict[str, Any], side: str, vocab: dict[str, int]) -> np.ndarray:
    x = np.zeros(len(vocab), dtype=float)
    for tok in feature_tokens(row, side):
        idx = vocab.get(tok)
        if idx is not None:
            x[idx] = 1.0
    return x


def fit_linear_value_model(
    rows: list[dict[str, Any]],
    market,
    *,
    horizon_bars: int,
    target_pct: float,
    stop_pct: float,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    entry_delay_bars: int,
    alpha: float,
    min_feature_count: int = 5,
) -> LinearValueModel:
    vocab = build_vocab(rows, min_count=min_feature_count)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for row in rows:
        for side in ["LONG", "SHORT"]:
            ret = simulate_trade_return(row, market, side=side, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
            if ret is None:
                continue
            xs.append(vectorize(row, side, vocab))
            ys.append(float(ret))
    if not xs:
        raise ValueError("no training samples for linear value model")
    xmat = np.vstack(xs)
    yvec = np.asarray(ys, dtype=float)
    reg = np.eye(xmat.shape[1], dtype=float) * float(alpha)
    reg[0, 0] = 1e-9  # do not materially regularize intercept
    xtx = xmat.T @ xmat + reg
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ xmat.T @ yvec
    residual = yvec - xmat @ beta
    residual_std = float(np.std(residual)) if len(residual) > 1 else 0.0
    return LinearValueModel(vocab=vocab, beta=beta, xtx_inv=xtx_inv, residual_std=residual_std, alpha=alpha)


def predict_value(row: dict[str, Any], side: str, model: LinearValueModel, *, risk_penalty: float) -> dict[str, float]:
    x = vectorize(row, side, model.vocab)
    pred = float(x @ model.beta)
    leverage_uncertainty = float(math.sqrt(max(0.0, x @ model.xtx_inv @ x))) * model.residual_std
    score = pred - float(risk_penalty) * leverage_uncertainty
    return {"pred": pred, "uncertainty": leverage_uncertainty, "score": score}


def policy_rows(rows: list[dict[str, Any]], model: LinearValueModel, *, threshold: float, risk_penalty: float, min_gap: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    counts = {"LONG": 0, "SHORT": 0, "NONE": 0}
    for row in rows:
        long = predict_value(row, "LONG", model, risk_penalty=risk_penalty)
        short = predict_value(row, "SHORT", model, risk_penalty=risk_penalty)
        best_side = "LONG" if long["score"] >= short["score"] else "SHORT"
        best = long if best_side == "LONG" else short
        other = short if best_side == "LONG" else long
        side = best_side if best["score"] >= threshold and (best["score"] - other["score"]) >= min_gap else "NONE"
        r = dict(row)
        r["prediction"] = {"direction_pressure": SIDE_TO_LABEL[side], "linear_value_score": best["score"], "linear_value_pred": best["pred"], "linear_value_uncertainty": best["uncertainty"]}
        r["linear_value_policy"] = {"long": long, "short": short, "threshold": threshold, "risk_penalty": risk_penalty, "min_gap": min_gap}
        out.append(r)
        counts[side] += 1
    return out, counts


def evaluate_linear_config(
    rows: list[dict[str, Any]],
    model: LinearValueModel,
    market,
    *,
    horizon_bars: int,
    target_pct: float,
    stop_pct: float,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    entry_delay_bars: int,
    threshold: float,
    risk_penalty: float,
    min_gap: float,
) -> dict[str, Any]:
    rows2, counts = policy_rows(rows, model, threshold=threshold, risk_penalty=risk_penalty, min_gap=min_gap)
    cfg = PressureBacktestConfig(horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
    bt = strict_pressure_backtest(rows2, market, cfg)
    return {"config": {"threshold": threshold, "risk_penalty": risk_penalty, "min_gap": min_gap, "alpha": model.alpha}, "action_counts": counts, "sim": bt["sim"], "trade_stats": bt["trade_stats"]}


def run_linear_value_sweep(
    *,
    jsonl_paths: list[str],
    market_csv: str,
    output: str,
    horizon_bars: int = 144,
    target_pct: float = 1.8,
    stop_pct: float = 1.5,
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    entry_delay_bars: int = 1,
    min_trades: int = 50,
) -> dict[str, Any]:
    rows = load_all_rows(jsonl_paths)
    market = load_market_bars(market_csv)
    folds = default_folds()
    configs = []
    for alpha in [0.1, 1.0, 10.0, 100.0, 300.0]:
        for threshold in [-0.0005, 0.0, 0.0002, 0.0005, 0.0010, 0.0015, 0.0020]:
            for risk_penalty in [0.0, 0.25, 0.5, 1.0]:
                for min_gap in [0.0, 0.0002, 0.0005, 0.0010]:
                    configs.append({"alpha": alpha, "threshold": threshold, "risk_penalty": risk_penalty, "min_gap": min_gap})

    fold_material = []
    for fold in folds:
        train_rows, test_rows = split_rows(rows, train_end=fold["train_end"], test_start=fold["test_start"], test_end=fold["test_end"])
        models = {}
        for alpha in sorted({float(c["alpha"]) for c in configs}):
            models[alpha] = fit_linear_value_model(train_rows, market, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars, alpha=alpha)
        fold_material.append({"fold": fold, "train_rows": len(train_rows), "test_rows": len(test_rows), "test": test_rows, "models": models})

    summaries = []
    for cfg in configs:
        fold_results = []
        fold_scores = []
        for material in fold_material:
            model = material["models"][float(cfg["alpha"])]
            result = evaluate_linear_config(material["test"], model, market, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars, threshold=float(cfg["threshold"]), risk_penalty=float(cfg["risk_penalty"]), min_gap=float(cfg["min_gap"]))
            fs = score_fold_result(result, min_trades=min_trades)
            fold_scores.append(fs)
            fold_results.append({"fold": material["fold"], "train_rows": material["train_rows"], "test_rows": material["test_rows"], "result": result, "score": fs})
        summaries.append({"config": cfg, "aggregate": aggregate_score(fold_scores, min_trades=min_trades), "folds": fold_results})
    ranked = sorted(summaries, key=lambda x: (x["aggregate"]["stability_score"], x["aggregate"]["positive_folds"], x["aggregate"]["min_ratio"], x["aggregate"]["avg_ratio"]), reverse=True)
    report = {
        "fixed_economics": {"horizon_bars": horizon_bars, "target_pct": target_pct, "stop_pct": stop_pct},
        "selected": ranked[0],
        "top": ranked[:25],
        "sweep_space": {"config_count": len(configs), "min_trades_per_fold": min_trades},
        "leakage_guard": {"each_fold_model_fit_prior_rows_only": True, "feature_vocab_fit_train_only": True, "labels_are_hypothetical_net_returns_after_costs": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-paths", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--horizon-bars", type=int, default=144)
    p.add_argument("--target-pct", type=float, default=1.8)
    p.add_argument("--stop-pct", type=float, default=1.5)
    p.add_argument("--min-trades", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_linear_value_sweep(jsonl_paths=[x for x in args.jsonl_paths.split(",") if x], market_csv=args.market_csv, output=args.output, horizon_bars=args.horizon_bars, target_pct=args.target_pct, stop_pct=args.stop_pct, min_trades=args.min_trades)
    sel = report["selected"]
    print(json.dumps({"fixed_economics": report["fixed_economics"], "selected_config": sel["config"], "aggregate": sel["aggregate"]}, indent=2, ensure_ascii=False))
    for f in sel["folds"]:
        sim = f["result"]["sim"]
        print(f["fold"]["name"], "trades", sim["trade_entries"], "cagr", round(sim["cagr_pct"], 2), "mdd", round(sim["strict_mdd_pct"], 2), "ratio", round(sim["cagr_to_strict_mdd"], 3))


if __name__ == "__main__":
    main()
