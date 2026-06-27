"""Numpy logistic baseline for episode survival SFT data.

This is a pre-finetune sanity check.  It parses only causal prompt fields,
trains a simple logistic classifier on train, chooses an acceptance threshold on
test, and reports eval plus strict backtest of accepted candidates.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_linear_combo_scan import _load_market
from training.price_action_episode_policy import EpisodePolicyCfg, simulate_triggers


@dataclass(frozen=True)
class SurvivalBaselineCfg:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    epochs: int = 250
    lr: float = 0.08
    l2: float = 0.001
    threshold_metric: str = "utility"  # utility or f05
    min_test_predictions: int = 40
    max_features: int = 20000
    seed: int = 42
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31 23:59:59"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"


_JSON_LINE_RE = re.compile(r"^(candidate|setup_quality|macro_context): (\{.*\})$", re.MULTILINE)


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")


def _load(path: str) -> list[dict[str, Any]]:
    with _open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _prompt_parts(prompt: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw in _JSON_LINE_RE.findall(prompt):
        out[key] = json.loads(raw)
    return out


def _target_y(row: dict[str, Any]) -> int:
    return 1 if json.loads(row["target"]).get("decision") == "TRADE" else 0


def _features(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    parts = _prompt_parts(str(row["prompt"]))
    cand = dict(row.get("candidate") or parts.get("candidate") or {})
    setup = dict(parts.get("setup_quality") or {})
    macro = dict(parts.get("macro_context") or {})
    num: dict[str, float] = {}
    cat: dict[str, str] = {}
    for k in ("risk_bps", "range_bps", "body_frac", "wick_frac", "close_quality"):
        num[f"setup.{k}"] = float(setup.get(k, 0.0) or 0.0)
    for k in ("dxy_z", "usdkrw_z", "kimchi_z", "kimchi_chg"):
        num[f"macro.{k}"] = float(macro.get(k, 0.0) or 0.0)
    for k in ("event", "event_type", "episode", "side", "horizon"):
        cat[f"cand.{k}"] = str(cand.get(k, "NA"))
    for k in ("risk_bucket", "range_bucket", "body_bucket", "wick_bucket", "close_quality_bucket"):
        cat[f"setup.{k}"] = str(setup.get(k, "NA"))
    # Simple numeric interactions; still causal.
    num["setup.risk_x_range"] = num["setup.risk_bps"] * num["setup.range_bps"] / 10_000.0
    num["setup.body_x_closeq"] = num["setup.body_frac"] * num["setup.close_quality"]
    return num, cat


def _build_matrix(rows: list[dict[str, Any]], vocab: dict[str, int] | None = None, *, max_features: int = 20000) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, tuple[float, float]]]:
    parsed = [_features(r) for r in rows]
    y = np.asarray([_target_y(r) for r in rows], dtype=float)
    num_keys = sorted({k for n, _ in parsed for k in n})
    stats: dict[str, tuple[float, float]] = {}
    for k in num_keys:
        vals = np.asarray([n.get(k, 0.0) for n, _ in parsed], dtype=float)
        mu = float(vals.mean())
        sd = float(vals.std()) or 1.0
        stats[k] = (mu, sd)
    if vocab is None:
        counts: dict[str, int] = defaultdict(int)
        for _, c in parsed:
            for k, v in c.items():
                counts[f"{k}={v}"] += 1
        vocab = {f"num:{k}": i for i, k in enumerate(num_keys)}
        for token, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(vocab) >= max_features:
                break
            vocab[f"cat:{token}"] = len(vocab)
    x = np.zeros((len(rows), len(vocab) + 1), dtype=np.float32)
    x[:, 0] = 1.0
    for i, (n, c) in enumerate(parsed):
        for k in num_keys:
            j = vocab.get(f"num:{k}")
            if j is not None:
                mu, sd = stats[k]
                x[i, j + 1] = (float(n.get(k, 0.0)) - mu) / sd
        for k, v in c.items():
            j = vocab.get(f"cat:{k}={v}")
            if j is not None:
                x[i, j + 1] = 1.0
    return x, y, vocab, stats


def _build_matrix_with(rows: list[dict[str, Any]], vocab: dict[str, int], stats: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    parsed = [_features(r) for r in rows]
    y = np.asarray([_target_y(r) for r in rows], dtype=float)
    x = np.zeros((len(rows), len(vocab) + 1), dtype=np.float32)
    x[:, 0] = 1.0
    for i, (n, c) in enumerate(parsed):
        for k, (mu, sd) in stats.items():
            j = vocab.get(f"num:{k}")
            if j is not None:
                x[i, j + 1] = (float(n.get(k, 0.0)) - mu) / sd
        for k, v in c.items():
            j = vocab.get(f"cat:{k}={v}")
            if j is not None:
                x[i, j + 1] = 1.0
    return x, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _train_logistic(x: np.ndarray, y: np.ndarray, cfg: SurvivalBaselineCfg) -> np.ndarray:
    rng = np.random.default_rng(int(cfg.seed))
    w = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float64)
    n = max(1, len(y))
    for _ in range(int(cfg.epochs)):
        p = _sigmoid(x @ w)
        grad = (x.T @ (p - y)) / n
        grad[1:] += float(cfg.l2) * w[1:]
        w -= float(cfg.lr) * grad
    return w


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = p >= float(threshold)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    f05 = 1.25 * precision * recall / max(1e-12, 0.25 * precision + recall)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "predicted_trade": int(pred.sum()), "precision": precision, "recall": recall, "f1": f1, "f05": f05, "positive_rate": float(pred.mean())}


def _choose_threshold(y: np.ndarray, p: np.ndarray, rows: list[dict[str, Any]], cfg: SurvivalBaselineCfg) -> tuple[float, list[dict[str, Any]]]:
    audits = [r.get("target_audit") or {} for r in rows]
    grid = sorted(set([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] + [float(x) for x in np.quantile(p, np.linspace(0.1, 0.95, 18))]))
    scored = []
    for t in grid:
        m = _metrics(y, p, t)
        pred = p >= t
        if int(m["predicted_trade"]) < int(cfg.min_test_predictions):
            continue
        if pred.any():
            mean_util = float(np.mean([float(a.get("utility_pct", 0.0) or 0.0) for a, keep in zip(audits, pred) if keep]))
            mean_net = float(np.mean([float(a.get("net_pct", 0.0) or 0.0) for a, keep in zip(audits, pred) if keep]))
        else:
            mean_util = mean_net = 0.0
        score = float(m["f05"]) if cfg.threshold_metric == "f05" else (float(m["precision"]) * 2.0 + mean_util + 0.1 * math.log1p(int(m["predicted_trade"])))
        scored.append({**m, "mean_target_utility_pct": mean_util, "mean_target_net_pct": mean_net, "score": score})
    if not scored:
        return 0.5, []
    best = max(scored, key=lambda r: float(r["score"]))
    return float(best["threshold"]), sorted(scored, key=lambda r: float(r["score"]), reverse=True)


def _accepted_backtest(rows: list[dict[str, Any]], p: np.ndarray, threshold: float, market: pd.DataFrame, start: str, end: str, cfg: SurvivalBaselineCfg) -> dict[str, Any]:
    chosen: dict[int, dict[str, Any]] = {}
    for row, prob in zip(rows, p):
        if float(prob) < float(threshold):
            continue
        pos = int(row["signal_pos"])
        cand = dict(row.get("candidate") or {})
        cur = chosen.get(pos)
        if cur is None or float(prob) > float(cur["score"]):
            chosen[pos] = {
                "pos": pos,
                "event": cand.get("event"),
                "window": int(str(cand.get("event", "pae_w0_")).split("_w", 1)[1].split("_", 1)[0]) if "_w" in str(cand.get("event", "")) else 0,
                "event_type": cand.get("event_type"),
                "episode": cand.get("episode"),
                "side": cand.get("side"),
                "horizon": int(cand.get("horizon", 1)),
                "score": float(prob),
                "train_score": 0.0,
            }
    dates = pd.to_datetime(market["date"])
    pcfg = EpisodePolicyCfg(input_csv=cfg.market_csv, output=cfg.output, entry_delay_bars=int(cfg.entry_delay_bars), leverage=float(cfg.leverage), fee_rate=float(cfg.fee_rate), slippage_rate=float(cfg.slippage_rate))
    bt = simulate_triggers(market, dates, list(chosen.values()), start=start, end=end, cfg=pcfg)
    return {"accepted_rows": int((p >= threshold).sum()), "unique_signal_positions": len(chosen), "sim": bt["sim"], "trade_stats": bt["trade_stats"]}


def run(cfg: SurvivalBaselineCfg) -> dict[str, Any]:
    train = _load(cfg.train_jsonl)
    test = _load(cfg.test_jsonl)
    ev = _load(cfg.eval_jsonl)
    x_train, y_train, vocab, stats = _build_matrix(train, max_features=int(cfg.max_features))
    x_test, y_test = _build_matrix_with(test, vocab, stats)
    x_eval, y_eval = _build_matrix_with(ev, vocab, stats)
    w = _train_logistic(x_train, y_train, cfg)
    p_train = _sigmoid(x_train @ w)
    p_test = _sigmoid(x_test @ w)
    p_eval = _sigmoid(x_eval @ w)
    threshold, threshold_grid = _choose_threshold(y_test, p_test, test, cfg)
    market = _load_market(cfg.market_csv)
    report = {
        "config": asdict(cfg),
        "rows": {"train": len(train), "test": len(test), "eval": len(ev)},
        "features": {"vocab_size": len(vocab), "numeric_features": len(stats)},
        "threshold": threshold,
        "threshold_grid_top": threshold_grid[:20],
        "classification": {
            "train": _metrics(y_train, p_train, threshold),
            "test": _metrics(y_test, p_test, threshold),
            "eval": _metrics(y_eval, p_eval, threshold),
        },
        "prob_summary": {
            "train": {"mean": float(p_train.mean()), "p90": float(np.quantile(p_train, 0.9)), "p99": float(np.quantile(p_train, 0.99))},
            "test": {"mean": float(p_test.mean()), "p90": float(np.quantile(p_test, 0.9)), "p99": float(np.quantile(p_test, 0.99))},
            "eval": {"mean": float(p_eval.mean()), "p90": float(np.quantile(p_eval, 0.9)), "p99": float(np.quantile(p_eval, 0.99))},
        },
        "accepted_backtest": {
            "test": _accepted_backtest(test, p_test, threshold, market, cfg.test_start, cfg.test_end, cfg),
            "eval": _accepted_backtest(ev, p_eval, threshold, market, cfg.eval_start, cfg.eval_end, cfg),
        },
        "leakage_guard": {"threshold_chosen_on_eval": False, "features_parse_prompt_only": True, "backtest_uses_model_probability_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=SurvivalBaselineCfg.epochs)
    p.add_argument("--lr", type=float, default=SurvivalBaselineCfg.lr)
    p.add_argument("--l2", type=float, default=SurvivalBaselineCfg.l2)
    p.add_argument("--threshold-metric", default=SurvivalBaselineCfg.threshold_metric)
    p.add_argument("--min-test-predictions", type=int, default=SurvivalBaselineCfg.min_test_predictions)
    p.add_argument("--max-features", type=int, default=SurvivalBaselineCfg.max_features)
    p.add_argument("--seed", type=int, default=SurvivalBaselineCfg.seed)
    p.add_argument("--entry-delay-bars", type=int, default=SurvivalBaselineCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=SurvivalBaselineCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=SurvivalBaselineCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=SurvivalBaselineCfg.slippage_rate)
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(SurvivalBaselineCfg, name.replace("-", "_")))
    return p.parse_args()


def main() -> None:
    r = run(SurvivalBaselineCfg(**vars(parse_args())))
    print(json.dumps({
        "output": r["config"]["output"],
        "threshold": r["threshold"],
        "classification": r["classification"],
        "accepted_backtest": {k: v["sim"] | {"accepted_rows": v["accepted_rows"], "unique_signal_positions": v["unique_signal_positions"], "p": v["trade_stats"].get("p_value_mean_ret_approx")} for k, v in r["accepted_backtest"].items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
