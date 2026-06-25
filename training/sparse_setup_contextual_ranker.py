"""Contextual action ranker for sparse setup event-action records.

This is a lightweight stand-in for the LLM/RL policy layer: it learns from
past-fold textual state/action tokens and ranks executable actions on the current
fold.  The model never uses current-fold rewards for current-fold action choice.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.sparse_setup_action_policy import _add_annualized, _fold_order, _metrics_from_sparse_actions, _source_or_no_trade


@dataclass(frozen=True)
class ContextualRankerCfg:
    records_jsonl: str
    sparse_report: str
    output: str
    l2: float = 10.0
    min_history_folds: int = 1
    min_train_samples: int = 200
    min_pred_utility: float = -0.002
    min_pred_net: float = 0.0
    max_pred_mae: float = 0.03
    target: str = "utility"  # utility | net_return
    fallback_source_action: bool = True
    fallback_risk_profile: str = "base"
    max_features: int = 4096


def load_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: (str(r.get("date", "")), int(r.get("signal_pos", 0)), str(r.get("key", ""))))
    return rows


def _action_tokens(action_key: str, action: dict[str, Any]) -> list[str]:
    return [
        f"action={action_key}",
        f"side={action.get('side')}",
        f"hold={int(action.get('hold_bars', 0) or 0)}",
        f"risk_profile={action.get('risk_profile', 'base')}",
        f"stop={float(action.get('stop_loss_pct', 0.0) or 0.0):.1f}",
        f"take={float(action.get('take_profit_pct', 0.0) or 0.0):.1f}",
    ]


def _example_tokens(row: dict[str, Any], action_key: str, action: dict[str, Any]) -> list[str]:
    state = [str(x) for x in row.get("state_tokens", [])]
    return ["bias", f"candidate_key={row.get('key')}", f"candidate_index={row.get('candidate_index')}"] + state + _action_tokens(action_key, action)


def _build_vocab(rows: list[dict[str, Any]], max_features: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for action_key, action in row.get("actions", {}).items():
            counts.update(_example_tokens(row, str(action_key), action))
    vocab = {tok: i for i, (tok, _n) in enumerate(counts.most_common(max(1, int(max_features))))}
    return vocab


def _vectorize(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    x = np.zeros(len(vocab), dtype=float)
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            x[idx] += 1.0
    return x


def _fit_ridge(train_rows: list[dict[str, Any]], cfg: ContextualRankerCfg) -> tuple[dict[str, int], np.ndarray, dict[str, Any]]:
    vocab = _build_vocab(train_rows, int(cfg.max_features))
    xs: list[np.ndarray] = []
    y: list[float] = []
    for row in train_rows:
        for action_key, action in row.get("actions", {}).items():
            xs.append(_vectorize(_example_tokens(row, str(action_key), action), vocab))
            target = float(action.get(str(cfg.target), action.get("utility", 0.0)) or 0.0)
            y.append(target)
    if not xs:
        return vocab, np.zeros(len(vocab), dtype=float), {"samples": 0, "features": len(vocab)}
    X = np.vstack(xs)
    yy = np.asarray(y, dtype=float)
    reg = float(cfg.l2) * np.eye(X.shape[1], dtype=float)
    if "bias" in vocab:
        reg[vocab["bias"], vocab["bias"]] = 0.0
    w = np.linalg.solve(X.T @ X + reg, X.T @ yy)
    pred = X @ w
    rmse = float(np.sqrt(np.mean((pred - yy) ** 2))) if yy.size else 0.0
    return vocab, w, {"samples": int(len(y)), "features": len(vocab), "target_mean": float(np.mean(yy)), "target_rmse": rmse}


def _predict_action(row: dict[str, Any], vocab: dict[str, int], w: np.ndarray, cfg: ContextualRankerCfg) -> dict[str, Any]:
    best: tuple[float, str, dict[str, Any]] | None = None
    for action_key, action in row.get("actions", {}).items():
        score = float(_vectorize(_example_tokens(row, str(action_key), action), vocab) @ w) if len(vocab) else -1e9
        if best is None or score > best[0]:
            best = (score, str(action_key), action)
    if best is None:
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "reason": "NO_ACTIONS"}
    score, action_key, action = best
    pred_net = score if str(cfg.target) == "net_return" else float(action.get("net_return", 0.0))
    pred_mae = float(action.get("mae", 0.0) or 0.0)
    if score < float(cfg.min_pred_utility) or pred_net < float(cfg.min_pred_net) or pred_mae > float(cfg.max_pred_mae):
        if bool(cfg.fallback_source_action):
            fallback = _source_or_no_trade(row, _dummy_sparse_cfg(), risk_profile=str(cfg.fallback_risk_profile))
            fallback["reason"] = "CONTEXT_REJECT_FALLBACK_SOURCE"
            fallback["pred_score"] = score
            return fallback
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "reason": "CONTEXT_REJECT", "pred_score": score}
    return {
        "gate": "TRADE",
        "side": str(action.get("side")),
        "hold_bars": int(action.get("hold_bars", 0) or 0),
        "action_key": action_key,
        "reason": "CONTEXT_RANKER",
        "pred_score": score,
    }


def _dummy_sparse_cfg():
    from training.sparse_setup_action_policy import SparseActionPolicyCfg

    return SparseActionPolicyCfg("", "", "", "")


def run(cfg: ContextualRankerCfg) -> dict[str, Any]:
    records = load_records(cfg.records_jsonl)
    sparse = json.loads(Path(cfg.sparse_report).read_text())
    folds = _fold_order(sparse)
    by_fold = {fold: [r for r in records if str(r.get("fold")) == fold] for fold in folds}
    final_records: list[dict[str, Any]] = []
    final_actions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for idx, fold in enumerate(folds):
        history_folds = folds[:idx]
        train = [r for hf in history_folds for r in by_fold.get(hf, [])]
        current = by_fold.get(fold, [])
        if len(history_folds) < int(cfg.min_history_folds) or sum(len(r.get("actions", {})) for r in train) < int(cfg.min_train_samples):
            actions = [_source_or_no_trade(r, _dummy_sparse_cfg(), risk_profile=str(cfg.fallback_risk_profile)) if cfg.fallback_source_action else {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "reason": "COLD_START_NO_TRADE"} for r in current]
            fit = {"samples": 0, "features": 0}
        else:
            vocab, w, fit = _fit_ridge(train, cfg)
            actions = [_predict_action(r, vocab, w, cfg) for r in current]
        metrics = _add_annualized(_metrics_from_sparse_actions(current, actions), current)
        fold_reports.append({"fold": fold, "history_folds": history_folds, "records": len(current), "fit": fit, "metrics": metrics, "action_reasons": dict(Counter(str(a.get("reason", "")) for a in actions))})
        final_records.extend(current)
        final_actions.extend(actions)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "records": len(records),
        "folds": fold_reports,
        "final_metrics": _add_annualized(_metrics_from_sparse_actions(final_records, final_actions), final_records),
        "leakage_guard": {"ranker_fit_on_previous_folds_only": True, "current_fold_rewards_not_used_for_current_actions": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Past-only contextual ranker for sparse event-action JSONL")
    p.add_argument("--records-jsonl", required=True)
    p.add_argument("--sparse-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--l2", type=float, default=ContextualRankerCfg.l2)
    p.add_argument("--min-history-folds", type=int, default=ContextualRankerCfg.min_history_folds)
    p.add_argument("--min-train-samples", type=int, default=ContextualRankerCfg.min_train_samples)
    p.add_argument("--min-pred-utility", type=float, default=ContextualRankerCfg.min_pred_utility)
    p.add_argument("--min-pred-net", type=float, default=ContextualRankerCfg.min_pred_net)
    p.add_argument("--max-pred-mae", type=float, default=ContextualRankerCfg.max_pred_mae)
    p.add_argument("--target", choices=["utility", "net_return"], default=ContextualRankerCfg.target)
    p.add_argument("--fallback-source-action", action=argparse.BooleanOptionalAction, default=ContextualRankerCfg.fallback_source_action)
    p.add_argument("--fallback-risk-profile", default=ContextualRankerCfg.fallback_risk_profile)
    p.add_argument("--max-features", type=int, default=ContextualRankerCfg.max_features)
    return p.parse_args()


def main() -> None:
    rep = run(ContextualRankerCfg(**vars(parse_args())))
    print(json.dumps({"final_metrics": rep["final_metrics"], "folds": [{"fold": f["fold"], "metrics": f["metrics"], "reasons": f["action_reasons"]} for f in rep["folds"]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
