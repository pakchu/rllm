"""Val-selected token veto overlay for path-shape token policies."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.path_shape_token_policy_tte import PathShapeTokenPolicyCfg, _fit, _load, _maybe_invert, _parse_floats, _predict, _score, tokens_from_row


@dataclass(frozen=True)
class ValTokenVetoTTECfg:
    train_jsonl: str
    val_jsonl: str
    eval_jsonl: str
    market_csv: str
    work_dir: str
    output: str
    min_count: int = 3
    smoothing: float = 2.0
    top_k_tokens: int = 24
    confidence_thresholds: str = "0.34,0.38,0.42,0.46,0.50"
    margin_thresholds: str = "0.00,0.03,0.06,0.10"
    side_modes: str = "normal"
    veto_sizes: str = "0,3,5,8,12"
    veto_unit_mode: str = "token"  # token | semantic
    exclude_veto_regex: str = ""
    min_token_trades: int = 12
    max_veto_mean_ret_pct: float = -0.05
    min_val_trades: int = 40
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    max_hold_bars: int = 144
    trade_stop_loss_pct: float = 0.6
    trade_take_profit_pct: float = 1.0


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _policy_cfg(cfg: ValTokenVetoTTECfg) -> PathShapeTokenPolicyCfg:
    return PathShapeTokenPolicyCfg(
        train_jsonl=cfg.train_jsonl,
        val_jsonl=cfg.val_jsonl,
        eval_jsonl=cfg.eval_jsonl,
        market_csv=cfg.market_csv,
        work_dir=cfg.work_dir,
        output=cfg.output,
        min_count=cfg.min_count,
        smoothing=cfg.smoothing,
        top_k_tokens=cfg.top_k_tokens,
        confidence_thresholds=cfg.confidence_thresholds,
        margin_thresholds=cfg.margin_thresholds,
        min_val_trades=cfg.min_val_trades,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        entry_delay_bars=cfg.entry_delay_bars,
        max_hold_bars=cfg.max_hold_bars,
        trade_stop_loss_pct=cfg.trade_stop_loss_pct,
        trade_take_profit_pct=cfg.trade_take_profit_pct,
    )


def _side_modes(raw: str) -> list[str]:
    out = [x.strip() for x in str(raw).split(",") if x.strip()]
    if any(x not in {"normal", "invert"} for x in out):
        raise ValueError("side_modes must contain normal/invert")
    return out or ["normal"]


def _semantic_unit(tok: str) -> str:
    raw = str(tok)
    if "=" not in raw:
        return raw
    key, val = raw.split("=", 1)
    key = re.sub(r"\.w\d+\.", ".", key)
    key = re.sub(r"^augnum\.", "augnum.", key)
    return f"{key}={val}"


def _veto_units(row: dict[str, Any], cfg: ValTokenVetoTTECfg) -> set[str]:
    toks = set(tokens_from_row(row))
    pat = re.compile(str(cfg.exclude_veto_regex)) if str(cfg.exclude_veto_regex).strip() else None
    out: set[str] = set()
    for tok in toks:
        if pat and pat.search(tok):
            continue
        out.add(_semantic_unit(tok) if str(cfg.veto_unit_mode) == "semantic" else tok)
    return out

def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _prediction_rows(rows: list[dict[str, Any]], preds: list[dict[str, Any]], *, prob_th: float, margin_th: float, side_mode: str, veto_tokens: set[str], hold_bars: int, cfg: ValTokenVetoTTECfg) -> list[dict[str, Any]]:
    out = []
    for row, pred in zip(rows, preds):
        label = _maybe_invert(str(pred["label"]), side_mode)
        units = _veto_units(row, cfg)
        vetoed = bool(veto_tokens.intersection(units))
        if vetoed or label == "NO_TRADE" or float(pred["prob"]) < prob_th or float(pred["margin"]) < margin_th:
            action = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
        else:
            action = {"gate": "TRADE", "side": label, "hold_bars": int(hold_bars)}
        out.append({"date": row.get("date"), "signal_pos": int(row.get("signal_pos", -1)), "prediction": action, "vetoed": vetoed, "pred_label": pred["label"], "pred_prob": pred["prob"], "pred_margin": pred["margin"]})
    return out


def _bt(path: str, cfg: ValTokenVetoTTECfg, output: str) -> dict[str, Any]:
    return run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=path, market_csv=cfg.market_csv, output=output, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, entry_delay_bars=cfg.entry_delay_bars, max_hold_bars=cfg.max_hold_bars, trade_stop_loss_pct=cfg.trade_stop_loss_pct, trade_take_profit_pct=cfg.trade_take_profit_pct))


def _trade_key(row: dict[str, Any]) -> tuple[str, int, str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1) or -1), str(row.get("side", "")), int(row.get("hold_bars", 0) or 0))


def _bad_tokens(rows: list[dict[str, Any]], preds: list[dict[str, Any]], executed: list[dict[str, Any]], side_mode: str, cfg: ValTokenVetoTTECfg) -> list[dict[str, Any]]:
    by_key = {_trade_key(r): float(r.get("trade_ret_pct", 0.0) or 0.0) for r in executed}
    vals: dict[str, list[float]] = defaultdict(list)
    for row, pred in zip(rows, preds):
        label = _maybe_invert(str(pred["label"]), side_mode)
        if label == "NO_TRADE":
            continue
        key = (str(row.get("date")), int(row.get("signal_pos", -1) or -1), label, int(cfg.max_hold_bars))
        if key not in by_key:
            continue
        ret = by_key[key]
        for tok in _veto_units(row, cfg):
            vals[tok].append(ret)
    out = []
    for tok, xs in vals.items():
        if len(xs) < int(cfg.min_token_trades):
            continue
        mean = sum(xs) / len(xs)
        if mean <= float(cfg.max_veto_mean_ret_pct):
            out.append({"token": tok, "n": len(xs), "mean_ret_pct": mean, "sum_ret_pct": sum(xs), "win_rate": sum(1 for x in xs if x > 0) / len(xs)})
    return sorted(out, key=lambda r: (float(r["mean_ret_pct"]), -int(r["n"])))


def run(cfg: ValTokenVetoTTECfg) -> dict[str, Any]:
    train, val, eval_rows = _load(cfg.train_jsonl), _load(cfg.val_jsonl), _load(cfg.eval_jsonl)
    pcfg = _policy_cfg(cfg)
    model = _fit(train, pcfg)
    val_preds = [_predict(r, model, pcfg) for r in val]
    eval_preds = [_predict(r, model, pcfg) for r in eval_rows]
    work = Path(cfg.work_dir)
    candidates = []
    bad_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in _side_modes(cfg.side_modes):
        base_rows = _prediction_rows(val, val_preds, prob_th=0.0, margin_th=0.0, side_mode=mode, veto_tokens=set(), hold_bars=cfg.max_hold_bars, cfg=cfg)
        base_pred = work / f"val_{mode}_base_for_veto.predictions.jsonl"
        _write_jsonl(base_pred, base_rows)
        base_bt = _bt(str(base_pred), cfg, str(work / f"val_{mode}_base_for_veto.bt.json"))
        bad_by_mode[mode] = _bad_tokens(val, val_preds, base_bt.get("executed", []), mode, cfg)
    for mode in _side_modes(cfg.side_modes):
        bad = bad_by_mode[mode]
        for veto_size in _parse_ints(cfg.veto_sizes):
            veto = {r["token"] for r in bad[: int(veto_size)]}
            for pth in _parse_floats(cfg.confidence_thresholds):
                for mth in _parse_floats(cfg.margin_thresholds):
                    pred_rows = _prediction_rows(val, val_preds, prob_th=pth, margin_th=mth, side_mode=mode, veto_tokens=veto, hold_bars=cfg.max_hold_bars, cfg=cfg)
                    tag = f"val_{mode}_v{veto_size}_p{pth:.2f}_m{mth:.2f}"
                    pred_path = work / f"{tag}.predictions.jsonl"
                    _write_jsonl(pred_path, pred_rows)
                    bt = _bt(str(pred_path), cfg, str(work / f"{tag}.bt.json"))
                    candidates.append({"side_mode": mode, "veto_size": veto_size, "veto_tokens": sorted(veto), "prob_threshold": pth, "margin_threshold": mth, "score": _score(bt["sim"], cfg.min_val_trades), "val_sim": bt["sim"], "val_trade_stats": bt["trade_stats"]})
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = candidates[0] if candidates else {"side_mode": "normal", "veto_size": 0, "veto_tokens": [], "prob_threshold": 1.0, "margin_threshold": 1.0}
    eval_pred_rows = _prediction_rows(eval_rows, eval_preds, prob_th=float(selected["prob_threshold"]), margin_th=float(selected["margin_threshold"]), side_mode=str(selected["side_mode"]), veto_tokens=set(selected.get("veto_tokens", [])), hold_bars=cfg.max_hold_bars, cfg=cfg)
    eval_pred = work / "selected_eval.predictions.jsonl"
    _write_jsonl(eval_pred, eval_pred_rows)
    eval_bt = _bt(str(eval_pred), cfg, str(work / "selected_eval.bt.json"))
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "bad_tokens_by_mode": bad_by_mode,
        "selected_by_val": selected,
        "top_val_candidates": candidates[:20],
        "eval_backtest": {"predictions": str(eval_pred), "sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
        "leakage_guard": {"model_fit_on_train_only": True, "bad_tokens_selected_from_val_executed_returns_only": True, "eval_not_used_for_veto_or_threshold_selection": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Val-selected token veto overlay for path-shape token policy")
    for field in ValTokenVetoTTECfg.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        required = field.default.__class__.__name__ == "_MISSING_TYPE"
        p.add_argument(name, default=None if required else field.default, required=required)
    ns = vars(p.parse_args())
    if ns["veto_unit_mode"] not in {"token", "semantic"}:
        raise ValueError("veto_unit_mode must be token or semantic")
    for k in {"min_count", "top_k_tokens", "min_token_trades", "min_val_trades", "entry_delay_bars", "max_hold_bars"}:
        ns[k] = int(ns[k])
    for k in {"smoothing", "max_veto_mean_ret_pct", "leverage", "fee_rate", "slippage_rate", "trade_stop_loss_pct", "trade_take_profit_pct"}:
        ns[k] = float(ns[k])
    return argparse.Namespace(**ns)


def main() -> None:
    rep = run(ValTokenVetoTTECfg(**vars(parse_args())))
    print(json.dumps({"selected_by_val": rep["selected_by_val"], "eval_sim": rep["eval_backtest"]["sim"], "eval_trade_stats": rep["eval_backtest"]["trade_stats"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
