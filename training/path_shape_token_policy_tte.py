"""Train/val/eval token baseline for path-shape trader labels.

This is a cheap learnability probe before LLM fine-tuning: fit a past-summary
categorical token model on train, select confidence/margin thresholds on val by
strict backtest, then report untouched eval performance.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.eval_economic_path_shape_sft import parse_trader

LABELS = ("LONG", "SHORT", "NO_TRADE")


@dataclass(frozen=True)
class PathShapeTokenPolicyCfg:
    train_jsonl: str
    val_jsonl: str
    eval_jsonl: str
    market_csv: str
    work_dir: str
    output: str
    min_count: int = 6
    smoothing: float = 2.0
    top_k_tokens: int = 24
    confidence_thresholds: str = "0.34,0.38,0.42,0.46,0.50,0.55,0.60"
    margin_thresholds: str = "0.00,0.03,0.06,0.10,0.15,0.20"
    min_val_trades: int = 40
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    max_hold_bars: int = 144
    trade_stop_loss_pct: float = 0.6
    trade_take_profit_pct: float = 1.0


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _summary_from_prompt(prompt: str) -> dict[str, Any]:
    marker = "Past-only analyzer summary: "
    if marker not in prompt:
        return {}
    raw = prompt.split(marker, 1)[1]
    raw = raw.split("\n\nAnalyzer path-shape output:", 1)[0].strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _bin_num(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if x <= -3:
        return "<=-3"
    if x <= -1.5:
        return "-3..-1.5"
    if x <= -0.5:
        return "-1.5..-0.5"
    if x < 0.5:
        return "-0.5..0.5"
    if x < 1.5:
        return "0.5..1.5"
    if x < 3:
        return "1.5..3"
    return ">=3"


def tokens_from_row(row: dict[str, Any]) -> list[str]:
    s = _summary_from_prompt(str(row.get("prompt", "")))
    toks: list[str] = []
    for key in ("regime", "trend_alignment", "trend_strength", "momentum", "location", "oscillator", "volatility_level", "volume_state", "risk_state", "candle_pattern"):
        if key in s:
            toks.append(f"{key}={s[key]}")
    for tag in s.get("context_tags", []) if isinstance(s.get("context_tags"), list) else []:
        toks.append(f"tag={tag}")
    sym = s.get("symbolic_features") if isinstance(s.get("symbolic_features"), dict) else {}
    for k, v in sorted(sym.items()):
        toks.append(f"sym.{k}={v}")
    seq = s.get("sequence_stats") if isinstance(s.get("sequence_stats"), dict) else {}
    for k, v in sorted(seq.items()):
        toks.append(f"seq.{k}={_bin_num(v)}")
    ev = s.get("evidence") if isinstance(s.get("evidence"), dict) else {}
    for k, v in sorted(ev.items()):
        toks.append(f"ev.{k}={_bin_num(v)}")
    recent = s.get("recent_bar_sequence") if isinstance(s.get("recent_bar_sequence"), list) else []
    for item in recent[-6:]:
        toks.append(f"recent={item}")
    return toks


def target_label(row: dict[str, Any]) -> str:
    act = parse_trader(str(row.get("target", "{}")))
    if act["gate"] != "TRADE":
        return "NO_TRADE"
    return str(act["side"])


def _fit(rows: list[dict[str, Any]], cfg: PathShapeTokenPolicyCfg) -> dict[str, Any]:
    priors = Counter(target_label(r) for r in rows)
    total = max(1, len(rows))
    prior_p = {lab: (priors.get(lab, 0) + cfg.smoothing) / (total + cfg.smoothing * len(LABELS)) for lab in LABELS}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    token_n: Counter[str] = Counter()
    for r in rows:
        y = target_label(r)
        for tok in set(tokens_from_row(r)):
            counts[tok][y] += 1
            token_n[tok] += 1
    weights: dict[str, dict[str, float]] = {}
    for tok, n in token_n.items():
        if n < int(cfg.min_count):
            continue
        weights[tok] = {}
        for lab in LABELS:
            p = (counts[tok].get(lab, 0) + cfg.smoothing * prior_p[lab]) / (n + cfg.smoothing)
            weights[tok][lab] = math.log(max(1e-9, p) / max(1e-9, prior_p[lab]))
    return {"prior": prior_p, "weights": weights, "token_n": dict(token_n), "target_counts": dict(priors)}


def _predict(row: dict[str, Any], model: dict[str, Any], cfg: PathShapeTokenPolicyCfg) -> dict[str, Any]:
    logits = {lab: math.log(float(model["prior"].get(lab, 1e-9))) for lab in LABELS}
    toks = sorted(set(tokens_from_row(row)), key=lambda t: max([abs(float(v)) for v in model["weights"].get(t, {}).values()] or [0.0]), reverse=True)
    for tok in toks[: int(cfg.top_k_tokens)]:
        for lab, w in model["weights"].get(tok, {}).items():
            logits[lab] += float(w)
    mx = max(logits.values())
    exps = {lab: math.exp(max(-40.0, min(40.0, logits[lab] - mx))) for lab in LABELS}
    denom = sum(exps.values())
    probs = {lab: exps[lab] / denom for lab in LABELS}
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return {"label": ordered[0][0], "prob": ordered[0][1], "margin": ordered[0][1] - ordered[1][1], "probs": probs, "top_tokens": toks[: int(cfg.top_k_tokens)]}


def _classification(rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for r, p in zip(rows, preds) if target_label(r) == p["label"])
    return {"rows": len(rows), "accuracy": correct / max(1, len(rows)), "target_counts": dict(Counter(target_label(r) for r in rows)), "pred_counts": dict(Counter(p["label"] for p in preds))}


def _prediction_rows(rows: list[dict[str, Any]], preds: list[dict[str, Any]], cfg: PathShapeTokenPolicyCfg, *, prob_th: float, margin_th: float) -> list[dict[str, Any]]:
    out = []
    for row, pred in zip(rows, preds):
        label = str(pred["label"])
        if label == "NO_TRADE" or float(pred["prob"]) < prob_th or float(pred["margin"]) < margin_th:
            action = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
        else:
            action = {"gate": "TRADE", "side": label, "hold_bars": int(cfg.max_hold_bars)}
        out.append({"date": row.get("date"), "signal_pos": int(row.get("signal_pos", -1)), "prediction": action, "position_scale": 1.0, "pred_prob": pred["prob"], "pred_margin": pred["margin"], "target_label": target_label(row)})
    return out


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _bt(pred_path: str, cfg: PathShapeTokenPolicyCfg, out: str) -> dict[str, Any]:
    return run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=pred_path, market_csv=cfg.market_csv, output=out, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, entry_delay_bars=cfg.entry_delay_bars, max_hold_bars=cfg.max_hold_bars, trade_stop_loss_pct=cfg.trade_stop_loss_pct, trade_take_profit_pct=cfg.trade_take_profit_pct))


def _score(sim: dict[str, Any], min_trades: int) -> float:
    trades = int(sim.get("trade_entries", 0) or 0)
    cagr = float(sim.get("cagr_pct", -100.0) or -100.0)
    mdd = float(sim.get("strict_mdd_pct", 100.0) or 100.0)
    ratio = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
    if trades < int(min_trades) or cagr <= 0:
        return -1000.0 + trades / max(1, int(min_trades)) + cagr / 100.0 - min(100.0, max(0.0, mdd)) / 100.0
    return ratio + min(2.0, trades / 100.0) - max(0.0, mdd - 15.0) / 10.0


def run(cfg: PathShapeTokenPolicyCfg) -> dict[str, Any]:
    train = _load(cfg.train_jsonl)
    val = _load(cfg.val_jsonl)
    eval_rows = _load(cfg.eval_jsonl)
    model = _fit(train, cfg)
    train_preds = [_predict(r, model, cfg) for r in train]
    val_preds = [_predict(r, model, cfg) for r in val]
    eval_preds = [_predict(r, model, cfg) for r in eval_rows]
    work = Path(cfg.work_dir)
    candidates = []
    for pth in _parse_floats(cfg.confidence_thresholds):
        for mth in _parse_floats(cfg.margin_thresholds):
            pred_rows = _prediction_rows(val, val_preds, cfg, prob_th=pth, margin_th=mth)
            pred_path = work / f"val_p{pth:.2f}_m{mth:.2f}.predictions.jsonl"
            _write_jsonl(pred_path, pred_rows)
            bt = _bt(str(pred_path), cfg, str(work / f"val_p{pth:.2f}_m{mth:.2f}.bt.json"))
            candidates.append({"prob_threshold": pth, "margin_threshold": mth, "score": _score(bt["sim"], cfg.min_val_trades), "val_sim": bt["sim"], "val_trade_stats": bt["trade_stats"]})
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = candidates[0] if candidates else {"prob_threshold": 1.0, "margin_threshold": 1.0, "score": -1e9}
    eval_pred_rows = _prediction_rows(eval_rows, eval_preds, cfg, prob_th=float(selected["prob_threshold"]), margin_th=float(selected["margin_threshold"]))
    eval_pred_path = work / "selected_eval.predictions.jsonl"
    _write_jsonl(eval_pred_path, eval_pred_rows)
    eval_bt = _bt(str(eval_pred_path), cfg, str(work / "selected_eval.bt.json"))
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "learned_token_count": len(model["weights"]),
        "train_classification": _classification(train, train_preds),
        "val_classification": _classification(val, val_preds),
        "eval_classification": _classification(eval_rows, eval_preds),
        "selected_by_val": selected,
        "top_val_candidates": candidates[:20],
        "eval_backtest": {"predictions": str(eval_pred_path), "sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
        "leakage_guard": {"token_model_fit_on_train_only": True, "val_selects_thresholds_only": True, "eval_not_used_for_model_or_threshold_selection": True, "prompts_are_past_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/val/eval token baseline for path-shape trader labels")
    for field in PathShapeTokenPolicyCfg.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        required = field.default is MISSING and field.default_factory is MISSING
        p.add_argument(name, default=None if required else field.default, required=required)
    ns = vars(p.parse_args())
    for k in {"min_count", "top_k_tokens", "min_val_trades", "entry_delay_bars", "max_hold_bars"}:
        ns[k] = int(ns[k])
    for k in {"smoothing", "leverage", "fee_rate", "slippage_rate", "trade_stop_loss_pct", "trade_take_profit_pct"}:
        ns[k] = float(ns[k])
    return argparse.Namespace(**ns)


def main() -> None:
    rep = run(PathShapeTokenPolicyCfg(**vars(parse_args())))
    print(json.dumps({"selected_by_val": rep["selected_by_val"], "eval_sim": rep["eval_backtest"]["sim"], "classification": {"val": rep["val_classification"], "eval": rep["eval_classification"]}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
