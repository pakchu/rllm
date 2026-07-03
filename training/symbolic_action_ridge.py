"""Leakage-safe symbolic action ridge ranker.

Uses the categorical text prompt from event_action_verifier_text as the LLM-style
symbolic feature surface, but learns realized action utility with a lightweight
chronological ridge model instead of asking the LLM to classify ALLOW/BLOCK.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "SYMBOLIC_RIDGE", "confidence": "HIGH"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")


def _prompt_tokens(prompt: str) -> list[str]:
    toks = ["bias"]
    for line in str(prompt).splitlines():
        if line.startswith("Regime tokens:"):
            for part in line.split(":", 1)[1].split(";"):
                tok = part.strip()
                if tok:
                    toks.append(f"regime:{tok}")
        elif line.startswith("Candidate book tokens:"):
            for part in line.split(":", 1)[1].split(";"):
                tok = part.strip()
                if tok:
                    toks.append(f"book:{tok}")
        elif line.startswith("Selected action tokens:"):
            for part in line.split(":", 1)[1].split(";"):
                tok = part.strip()
                if tok:
                    toks.append(f"action:{tok}")
    # side/context interactions are the useful part of the symbolic surface.
    side = next((t.split("=", 1)[1] for t in toks if t.startswith("action:side=")), "NONE")
    horizon = next((t.split("=", 1)[1] for t in toks if t.startswith("action:horizon=")), "NONE")
    family = next((t.split("=", 1)[1] for t in toks if t.startswith("action:family=")), "NONE")
    for tok in list(toks):
        if tok.startswith("regime:"):
            toks.append(f"x:side={side}|{tok}")
            toks.append(f"x:horizon={horizon}|{tok}")
            toks.append(f"x:family={family}|{tok}")
    return toks


def row_tokens(row: dict[str, Any]) -> list[str]:
    toks = _prompt_tokens(str(row.get("prompt", "")))
    action = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
    side = str(action.get("side", "NONE")).upper()
    family = str(action.get("family", "NONE"))
    hold = int(action.get("hold_bars", 0) or 0)
    toks.extend([f"raw_action:side={side}", f"raw_action:family={family}", f"raw_action:hold={hold}"])
    state_tokens = row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {}
    for key, val in sorted(state_tokens.items()):
        tok = f"state:{key}={val}"
        toks.append(tok)
        # Side/action interactions let the ranker learn that the same market
        # structure can mean different things for LONG/SHORT or trend/reversal
        # action families.
        toks.append(f"x:side={side}|{tok}")
        toks.append(f"x:family={family}|{tok}")
        toks.append(f"x:horizon={hold}|{tok}")
    return toks


@dataclass
class FeatureSpace:
    vocab: dict[str, int]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, rows: list[dict[str, Any]], *, min_count: int = 5) -> "FeatureSpace":
        counts: Counter[str] = Counter()
        for row in rows:
            counts.update(row_tokens(row))
        vocab = {"bias": 0}
        for tok, cnt in sorted(counts.items()):
            if tok == "bias":
                continue
            if cnt >= int(min_count):
                vocab[tok] = len(vocab)
        dummy = cls(vocab=vocab, mean=np.zeros(len(vocab)), std=np.ones(len(vocab)))
        x = dummy.matrix(rows, scale=False)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-9] = 1.0
        mean[0] = 0.0
        std[0] = 1.0
        return cls(vocab=vocab, mean=mean, std=std)

    def matrix(self, rows: list[dict[str, Any]], *, scale: bool = True) -> np.ndarray:
        x = np.zeros((len(rows), len(self.vocab)), dtype=np.float64)
        for i, row in enumerate(rows):
            for tok in row_tokens(row):
                j = self.vocab.get(tok)
                if j is not None:
                    x[i, j] = 1.0
        if scale:
            x = (x - self.mean) / self.std
        return x


def target_value(row: dict[str, Any], *, target: str = "utility") -> float:
    audit = row.get("action_audit", {}) if isinstance(row.get("action_audit"), dict) else {}
    # Newer REX ranker rows store the same path quantities under `reward`.
    # Prefer action_audit for backwards compatibility, then fall back to reward
    # so symbolic rankability can be tested before another expensive LLM run.
    reward = row.get("reward", {}) if isinstance(row.get("reward"), dict) else {}
    source = audit if audit else reward
    net = float(source.get("net_return", 0.0) or 0.0)
    mae = max(0.0, float(source.get("mae", 0.0) or 0.0))
    mfe = max(0.0, float(source.get("mfe", 0.0) or 0.0))
    rr = mfe / max(mae, 1e-6)
    action = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
    hold = int(action.get("hold_bars", source.get("hold_bars", 0)) or 0)
    family = str(action.get("family", source.get("family", "")))
    side = str(action.get("side", source.get("side", ""))).upper()
    long_tail_penalty = 0.0
    if hold >= 432:
        long_tail_penalty += 0.15 * mae
    if side == "LONG" and hold >= 432:
        long_tail_penalty += 0.20 * mae
    if family in {"drawdown_reversal", "drawdown_continuation"}:
        long_tail_penalty += 0.20 * mae
    if target == "net_return":
        return net
    if target == "risk_adjusted":
        return net - mae
    if target == "tail_risk":
        # Prefer trades whose expected return survives adverse excursion; this is
        # deliberately continuous, not a hard gate, so the model can still learn
        # safe long-horizon trades when reward/risk is strong.
        return net - 1.5 * mae - long_tail_penalty
    if target == "distributional_safety":
        # Penalize left-tail shape and weak reward/risk asymmetry while preserving
        # upside for high-MFE actions.  A small MFE credit avoids collapsing into
        # pure abstention on all volatile opportunities.
        asym_penalty = 0.004 if rr < 1.2 else 0.0
        loss_penalty = 0.5 * abs(min(0.0, net))
        return net + 0.25 * mfe - 1.25 * mae - long_tail_penalty - asym_penalty - loss_penalty
    return float(source.get("utility", source.get("rank_utility", 0.0)) or 0.0)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    reg = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 1e-9
    return np.linalg.pinv(x.T @ x + reg) @ x.T @ y


def predict_candidates(rows: list[dict[str, Any]], fs: FeatureSpace, w: np.ndarray) -> list[dict[str, Any]]:
    x = fs.matrix(rows)
    preds = x @ w
    out = []
    for row, pred in zip(rows, preds):
        action = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
        out.append({
            "date": row.get("date"),
            "signal_pos": int(row.get("signal_pos", -1) or -1),
            "action": {"family": action.get("family"), "side": str(action.get("side", "NONE")).upper(), "hold_bars": int(action.get("hold_bars", 0) or 0)},
            "predicted_utility": float(pred),
            "actual_utility": target_value(row),
            "target": row.get("target"),
        })
    return out


def choose_best(pred_rows: list[dict[str, Any]], *, threshold: float, min_gap: float = 0.0) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in pred_rows:
        groups[(str(r["date"]), int(r["signal_pos"]))].append(r)
    chosen = []
    for key in sorted(groups):
        gs = sorted(groups[key], key=lambda r: float(r["predicted_utility"]), reverse=True)
        best = gs[0]
        second = float(gs[1]["predicted_utility"]) if len(gs) > 1 else -1e9
        if float(best["predicted_utility"]) >= float(threshold) and float(best["predicted_utility"]) - second >= float(min_gap):
            a = best["action"]
            pred = {"gate": "TRADE", "family": a.get("family", "UNKNOWN"), "side": a.get("side", "NONE"), "hold_bars": int(a.get("hold_bars", 0) or 0), "confidence": "HIGH"}
        else:
            pred = dict(NO_TRADE)
        chosen.append({"date": key[0], "signal_pos": key[1], "prediction": pred, "predicted_utility": float(best["predicted_utility"]), "runner_up_gap": float(best["predicted_utility"] - second), "selected_action": best["action"], "actual_utility": float(best["actual_utility"])})
    return chosen



def _candidate_rows_from_preds(rows: list[dict[str, Any]], preds: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for row, pred in zip(rows, preds):
        action = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
        out.append({
            "date": row.get("date"),
            "signal_pos": int(row.get("signal_pos", -1) or -1),
            "action": {"family": action.get("family"), "side": str(action.get("side", "NONE")).upper(), "hold_bars": int(action.get("hold_bars", 0) or 0)},
            "predicted_utility": float(pred),
            "actual_utility": target_value(row),
            "target": row.get("target"),
        })
    return out


def _write_and_backtest_candidate_preds(preds: np.ndarray, rows: list[dict[str, Any]], *, threshold: float, min_gap: float, predictions_output: str, market_csv: str, bt_output: str) -> dict[str, Any]:
    chosen = choose_best(_candidate_rows_from_preds(rows, preds), threshold=float(threshold), min_gap=float(min_gap))
    write_jsonl(predictions_output, chosen)
    return backtest(predictions_output, market_csv, bt_output)


def train_predict(*, train_jsonl: str, eval_jsonl: str, predictions_output: str, alpha: float, threshold: float, min_gap: float, target: str = "utility", min_feature_count: int = 5) -> dict[str, Any]:
    train = load_jsonl(train_jsonl)
    ev = load_jsonl(eval_jsonl)
    fs = FeatureSpace.fit(train, min_count=min_feature_count)
    x = fs.matrix(train)
    y = np.asarray([target_value(r, target=target) for r in train], dtype=np.float64)
    w = fit_ridge(x, y, alpha=float(alpha))
    train_pred = x @ w
    cand = predict_candidates(ev, fs, w)
    chosen = choose_best(cand, threshold=float(threshold), min_gap=float(min_gap))
    write_jsonl(predictions_output, chosen)
    corr = 0.0 if len(y) < 2 or np.std(y) < 1e-12 or np.std(train_pred) < 1e-12 else float(np.corrcoef(y, train_pred)[0, 1])
    return {"train_jsonl": train_jsonl, "eval_jsonl": eval_jsonl, "predictions_output": predictions_output, "config": {"alpha": alpha, "threshold": threshold, "min_gap": min_gap, "target": target, "min_feature_count": min_feature_count, "features": len(fs.vocab)}, "fit": {"train_rows": len(train), "eval_rows": len(ev), "train_corr": corr, "train_rmse_pct": math.sqrt(float(np.mean((train_pred - y) ** 2))) * 100.0}, "chosen_counts": dict(Counter(f"{r['prediction']['gate']}/{r['prediction'].get('side')}" for r in chosen))}


def backtest(predictions_output: str, market_csv: str, output: str) -> dict[str, Any]:
    return run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=predictions_output, market_csv=market_csv, output=output))


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_csv_targets(raw: str) -> list[str]:
    allowed = {"utility", "net_return", "risk_adjusted", "tail_risk", "distributional_safety"}
    out = [x.strip() for x in str(raw).split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(f"unknown targets: {bad}")
    return out or ["utility"]


def sweep(*, train_jsonl: str, val_jsonl: str, holdout_jsonl: str, market_csv: str, output: str, work_dir: str, min_trades: int = 30, targets: str = "utility,net_return,risk_adjusted", alphas: str = "1,10,100,1000,10000", thresholds: str = "-0.006,-0.003,-0.001,0,0.001,0.003,0.006", min_gaps: str = "0,0.0005,0.0015,0.003", min_val_cagr_pct: float = 0.0, min_val_ratio: float = -999.0, max_val_mdd_pct: float = 0.0, max_val_p_value: float = 1.0, abstain_on_validation_fail: bool = False, min_feature_count: int = 5) -> dict[str, Any]:
    train = load_jsonl(train_jsonl)
    val = load_jsonl(val_jsonl)
    hold = load_jsonl(holdout_jsonl)
    fs = FeatureSpace.fit(train, min_count=int(min_feature_count))
    x_train = fs.matrix(train)
    x_val = fs.matrix(val)
    x_hold = fs.matrix(hold)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    configs = []
    for target in _parse_csv_targets(targets):
        y = np.asarray([target_value(r, target=target) for r in train], dtype=np.float64)
        for alpha in _parse_csv_floats(alphas):
            w = fit_ridge(x_train, y, alpha=float(alpha))
            val_preds = x_val @ w
            train_pred = x_train @ w
            corr = 0.0 if len(y) < 2 or np.std(y) < 1e-12 or np.std(train_pred) < 1e-12 else float(np.corrcoef(y, train_pred)[0, 1])
            for threshold in _parse_csv_floats(thresholds):
                for min_gap in _parse_csv_floats(min_gaps):
                    configs.append({"target": target, "alpha": alpha, "threshold": threshold, "min_gap": min_gap, "w": w, "val_preds": val_preds, "fit": {"train_corr": corr, "train_rmse_pct": math.sqrt(float(np.mean((train_pred - y) ** 2))) * 100.0}})

    scored = []
    for i, cfg in enumerate(configs):
        tag = f"a{cfg['alpha']}_t{cfg['threshold']}_g{cfg['min_gap']}_{cfg['target']}".replace(".", "p").replace("-", "m")
        pred = str(Path(work_dir) / f"val_{i}_{tag}.jsonl")
        btout = str(Path(work_dir) / f"val_{i}_{tag}.bt.json")
        bt = _write_and_backtest_candidate_preds(cfg["val_preds"], val, threshold=float(cfg["threshold"]), min_gap=float(cfg["min_gap"]), predictions_output=pred, market_csv=market_csv, bt_output=btout)
        sim = bt["sim"]
        trades = int(sim["trade_entries"])
        ratio = float(sim["cagr_to_strict_mdd"])
        stats = bt["trade_stats"]
        reject_reasons = []
        if trades < int(min_trades):
            reject_reasons.append("trades_below_min")
        if float(sim["cagr_pct"]) < float(min_val_cagr_pct):
            reject_reasons.append("cagr_below_min")
        if ratio < float(min_val_ratio):
            reject_reasons.append("ratio_below_min")
        if float(max_val_mdd_pct) > 0.0 and float(sim["strict_mdd_pct"]) > float(max_val_mdd_pct):
            reject_reasons.append("mdd_above_max")
        if float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0) > float(max_val_p_value):
            reject_reasons.append("p_value_above_max")
        score = ratio + trades / 1000.0 if not reject_reasons else -999.0 + trades / 1000.0 - len(reject_reasons)

        public_cfg = {k: cfg[k] for k in ("target", "alpha", "threshold", "min_gap")}
        scored.append({"config": public_cfg, "fit": cfg["fit"], "val": {"sim": sim, "trade_stats": stats, "path": btout}, "validation_passed": not reject_reasons, "validation_reject_reasons": reject_reasons, "selection_score": score})
    ranked = sorted(scored, key=lambda r: (r["selection_score"], r["val"]["sim"]["trade_entries"], r["val"]["sim"]["cagr_pct"]), reverse=True)
    selected = ranked[0]
    # Recompute selected weights deterministically from train only.
    y_sel = np.asarray([target_value(r, target=selected["config"]["target"]) for r in train], dtype=np.float64)
    w_sel = fit_ridge(x_train, y_sel, alpha=float(selected["config"]["alpha"]))
    hold_pred = str(Path(work_dir) / "holdout_selected_predictions.jsonl")
    hold_bt_path = str(Path(work_dir) / "holdout_selected.bt.json")
    holdout_abstained = bool(abstain_on_validation_fail and not selected.get("validation_passed", False))
    if holdout_abstained:
        # Strict deployment protocol: if validation found no acceptable config, do not trade untouched holdout.
        pred_rows = []
        for row in hold:
            pred_rows.append({"date": row.get("date"), "signal_pos": int(row.get("signal_pos", -1) or -1), "prediction": dict(NO_TRADE), "predicted_utility": 0.0, "runner_up_gap": 0.0, "selected_action": None, "actual_utility": 0.0})
        write_jsonl(hold_pred, pred_rows)
        hold_bt = backtest(hold_pred, market_csv, hold_bt_path)
    else:
        hold_pred_values = x_hold @ w_sel
        hold_bt = _write_and_backtest_candidate_preds(hold_pred_values, hold, threshold=float(selected["config"]["threshold"]), min_gap=float(selected["config"]["min_gap"]), predictions_output=hold_pred, market_csv=market_csv, bt_output=hold_bt_path)
    report = {
        "selected": selected,
        "holdout": {"sim": hold_bt["sim"], "trade_stats": hold_bt["trade_stats"], "path": hold_bt_path, "predictions_output": hold_pred, "abstained_due_to_validation_fail": holdout_abstained},
        "top": ranked[:20],
        "sweep_space": {"configs": len(configs), "min_trades": min_trades, "targets": _parse_csv_targets(targets), "alphas": _parse_csv_floats(alphas), "thresholds": _parse_csv_floats(thresholds), "min_gaps": _parse_csv_floats(min_gaps), "min_val_cagr_pct": min_val_cagr_pct, "min_val_ratio": min_val_ratio, "max_val_mdd_pct": max_val_mdd_pct, "max_val_p_value": max_val_p_value, "abstain_on_validation_fail": abstain_on_validation_fail, "min_feature_count": min_feature_count, "features": len(fs.vocab), "train_rows": len(train), "val_rows": len(val), "holdout_rows": len(hold)},
        "leakage_guard": {"feature_vocab_and_ridge_fit_train_only": True, "config_selected_on_val_only": True, "holdout_uses_fixed_selected_config": True, "prompts_are_past_only_symbolic_text": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Symbolic text feature ridge action ranker")
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train-predict")
    tr.add_argument("--train-jsonl", required=True); tr.add_argument("--eval-jsonl", required=True); tr.add_argument("--predictions-output", required=True)
    tr.add_argument("--alpha", type=float, default=100.0); tr.add_argument("--threshold", type=float, default=0.0); tr.add_argument("--min-gap", type=float, default=0.0); tr.add_argument("--target", choices=["utility", "net_return", "risk_adjusted", "tail_risk", "distributional_safety"], default="utility"); tr.add_argument("--min-feature-count", type=int, default=5)
    sw = sub.add_parser("sweep")
    sw.add_argument("--train-jsonl", required=True); sw.add_argument("--val-jsonl", required=True); sw.add_argument("--holdout-jsonl", required=True); sw.add_argument("--market-csv", required=True); sw.add_argument("--output", required=True); sw.add_argument("--work-dir", required=True); sw.add_argument("--min-trades", type=int, default=30)
    sw.add_argument("--targets", default="utility,net_return,risk_adjusted")
    sw.add_argument("--alphas", default="1,10,100,1000,10000")
    sw.add_argument("--thresholds", default="-0.006,-0.003,-0.001,0,0.001,0.003,0.006")
    sw.add_argument("--min-gaps", default="0,0.0005,0.0015,0.003")
    sw.add_argument("--min-val-cagr-pct", type=float, default=0.0)
    sw.add_argument("--min-val-ratio", type=float, default=-999.0)
    sw.add_argument("--max-val-mdd-pct", type=float, default=0.0)
    sw.add_argument("--max-val-p-value", type=float, default=1.0)
    sw.add_argument("--abstain-on-validation-fail", action="store_true")
    sw.add_argument("--min-feature-count", type=int, default=5)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.cmd == "train-predict":
        print(json.dumps(train_predict(train_jsonl=a.train_jsonl, eval_jsonl=a.eval_jsonl, predictions_output=a.predictions_output, alpha=a.alpha, threshold=a.threshold, min_gap=a.min_gap, target=a.target, min_feature_count=a.min_feature_count), indent=2, ensure_ascii=False))
    elif a.cmd == "sweep":
        rep = sweep(train_jsonl=a.train_jsonl, val_jsonl=a.val_jsonl, holdout_jsonl=a.holdout_jsonl, market_csv=a.market_csv, output=a.output, work_dir=a.work_dir, min_trades=a.min_trades, targets=a.targets, alphas=a.alphas, thresholds=a.thresholds, min_gaps=a.min_gaps, min_val_cagr_pct=a.min_val_cagr_pct, min_val_ratio=a.min_val_ratio, max_val_mdd_pct=a.max_val_mdd_pct, max_val_p_value=a.max_val_p_value, abstain_on_validation_fail=a.abstain_on_validation_fail, min_feature_count=a.min_feature_count)
        print(json.dumps({"selected_config": rep["selected"]["config"], "val_sim": rep["selected"]["val"]["sim"], "holdout_sim": rep["holdout"]["sim"]}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
