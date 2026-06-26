"""Monthly adaptive semantic-veto backtest for path-shape token policy."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.path_shape_token_policy_tte import PathShapeTokenPolicyCfg, _fit, _load, _predict
from training.path_shape_val_token_veto_tte import ValTokenVetoTTECfg, _bad_tokens, _prediction_rows


@dataclass(frozen=True)
class AdaptiveSemanticVetoCfg:
    input_jsonl: str
    market_csv: str
    work_dir: str
    output: str
    eval_start: str = "2025-01-01"
    min_train_rows: int = 1000
    history_months: int = 6
    min_count: int = 3
    smoothing: float = 2.0
    top_k_tokens: int = 24
    prob_threshold: float = 0.34
    margin_threshold: float = 0.30
    veto_size: int = 12
    min_token_trades: int = 16
    max_veto_mean_ret_pct: float = -0.05
    exclude_veto_regex: str = "^recent="
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    max_hold_bars: int = 144
    trade_stop_loss_pct: float = 0.6
    trade_take_profit_pct: float = 1.0


def _month_key(date: str) -> str:
    return str(date)[:7]


def _month_start(month: str) -> str:
    return f"{month}-01"


def _months(rows: list[dict[str, Any]], eval_start: str) -> list[str]:
    return sorted({_month_key(str(r["date"])) for r in rows if str(r["date"]) >= str(eval_start)})


def _history_start(month: str, months_back: int) -> str:
    dt = datetime.fromisoformat(f"{month}-01 00:00:00")
    y, m = dt.year, dt.month - int(months_back)
    while m <= 0:
        y -= 1
        m += 12
    return f"{y:04d}-{m:02d}-01"


def _policy_cfg(cfg: AdaptiveSemanticVetoCfg, train: str, val: str, eval_: str) -> PathShapeTokenPolicyCfg:
    return PathShapeTokenPolicyCfg(train, val, eval_, cfg.market_csv, cfg.work_dir, cfg.output, min_count=cfg.min_count, smoothing=cfg.smoothing, top_k_tokens=cfg.top_k_tokens, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, entry_delay_bars=cfg.entry_delay_bars, max_hold_bars=cfg.max_hold_bars, trade_stop_loss_pct=cfg.trade_stop_loss_pct, trade_take_profit_pct=cfg.trade_take_profit_pct)


def _veto_cfg(cfg: AdaptiveSemanticVetoCfg) -> ValTokenVetoTTECfg:
    return ValTokenVetoTTECfg("", "", "", cfg.market_csv, cfg.work_dir, cfg.output, min_count=cfg.min_count, smoothing=cfg.smoothing, top_k_tokens=cfg.top_k_tokens, veto_unit_mode="semantic", exclude_veto_regex=cfg.exclude_veto_regex, min_token_trades=cfg.min_token_trades, max_veto_mean_ret_pct=cfg.max_veto_mean_ret_pct, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, entry_delay_bars=cfg.entry_delay_bars, max_hold_bars=cfg.max_hold_bars, trade_stop_loss_pct=cfg.trade_stop_loss_pct, trade_take_profit_pct=cfg.trade_take_profit_pct)


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _bt(pred_path: str, cfg: AdaptiveSemanticVetoCfg, out: str) -> dict[str, Any]:
    return run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=pred_path, market_csv=cfg.market_csv, output=out, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, entry_delay_bars=cfg.entry_delay_bars, max_hold_bars=cfg.max_hold_bars, trade_stop_loss_pct=cfg.trade_stop_loss_pct, trade_take_profit_pct=cfg.trade_take_profit_pct))


def run(cfg: AdaptiveSemanticVetoCfg) -> dict[str, Any]:
    rows = sorted(_load(cfg.input_jsonl), key=lambda r: str(r["date"]))
    work = Path(cfg.work_dir)
    all_predictions: list[dict[str, Any]] = []
    month_reports: list[dict[str, Any]] = []
    vcfg = _veto_cfg(cfg)
    for month in _months(rows, cfg.eval_start):
        train_rows = [r for r in rows if str(r["date"]) < _month_start(month)]
        if len(train_rows) < int(cfg.min_train_rows):
            continue
        hist_start = _history_start(month, int(cfg.history_months))
        hist_rows = [r for r in rows if hist_start <= str(r["date"]) < _month_start(month)]
        cur_rows = [r for r in rows if _month_key(str(r["date"])) == month]
        pcfg = _policy_cfg(cfg, cfg.input_jsonl, cfg.input_jsonl, cfg.input_jsonl)
        model = _fit(train_rows, pcfg)
        hist_preds = [_predict(r, model, pcfg) for r in hist_rows]
        cur_preds = [_predict(r, model, pcfg) for r in cur_rows]
        hist_pred_rows = _prediction_rows(hist_rows, hist_preds, prob_th=0.0, margin_th=0.0, side_mode="normal", veto_tokens=set(), hold_bars=cfg.max_hold_bars, cfg=vcfg)
        hist_pred_path = work / f"{month}_history.predictions.jsonl"
        _write_jsonl(hist_pred_path, hist_pred_rows)
        hist_bt = _bt(str(hist_pred_path), cfg, str(work / f"{month}_history.bt.json")) if hist_pred_rows else {"executed": []}
        bad = _bad_tokens(hist_rows, hist_preds, hist_bt.get("executed", []), "normal", vcfg)
        veto = {r["token"] for r in bad[: int(cfg.veto_size)]}
        cur_pred_rows = _prediction_rows(cur_rows, cur_preds, prob_th=float(cfg.prob_threshold), margin_th=float(cfg.margin_threshold), side_mode="normal", veto_tokens=veto, hold_bars=cfg.max_hold_bars, cfg=vcfg)
        month_pred_path = work / f"{month}.predictions.jsonl"
        _write_jsonl(month_pred_path, cur_pred_rows)
        month_bt = _bt(str(month_pred_path), cfg, str(work / f"{month}.bt.json")) if cur_pred_rows else {"sim": {}, "trade_stats": {}}
        all_predictions.extend(cur_pred_rows)
        month_reports.append({"month": month, "train_rows": len(train_rows), "history_rows": len(hist_rows), "current_rows": len(cur_rows), "veto_tokens": sorted(veto), "bad_token_candidates": bad[:30], "month_sim": month_bt.get("sim", {}), "month_trade_stats": month_bt.get("trade_stats", {})})
    combined_pred = work / "adaptive_eval.predictions.jsonl"
    _write_jsonl(combined_pred, all_predictions)
    combined_bt = _bt(str(combined_pred), cfg, str(work / "adaptive_eval.bt.json")) if all_predictions else {"sim": {}, "trade_stats": {}}
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "months": month_reports, "combined_predictions": str(combined_pred), "combined_backtest": {"sim": combined_bt.get("sim", {}), "trade_stats": combined_bt.get("trade_stats", {})}, "leakage_guard": {"month_model_fit_uses_rows_before_month_only": True, "veto_history_uses_rows_before_month_only": True, "current_month_not_used_for_veto_selection": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly adaptive semantic-veto backtest for path-shape token policy")
    for field in AdaptiveSemanticVetoCfg.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        required = field.default.__class__.__name__ == "_MISSING_TYPE"
        p.add_argument(name, default=None if required else field.default, required=required)
    ns = vars(p.parse_args())
    for k in {"min_train_rows", "history_months", "min_count", "top_k_tokens", "veto_size", "min_token_trades", "entry_delay_bars", "max_hold_bars"}:
        ns[k] = int(ns[k])
    for k in {"smoothing", "prob_threshold", "margin_threshold", "max_veto_mean_ret_pct", "leverage", "fee_rate", "slippage_rate", "trade_stop_loss_pct", "trade_take_profit_pct"}:
        ns[k] = float(ns[k])
    return argparse.Namespace(**ns)


def main() -> None:
    rep = run(AdaptiveSemanticVetoCfg(**vars(parse_args())))
    print(json.dumps({"combined": rep["combined_backtest"], "months": [{"month": m["month"], "sim": m["month_sim"]} for m in rep["months"]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
