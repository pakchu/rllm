"""Blend two prediction JSONL streams and replay the strict overlay backtest.

This is intentionally simple glue for leakage-resistant ensemble probes: it does
not fit thresholds from the target period.  It combines two already-generated
walk-forward prediction streams signal-by-signal using fixed logical rules.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class BlendPredictionSetsCfg:
    base_predictions: str
    guard_predictions: str
    market_csv: str
    output: str
    predictions_output: str
    mode: str = "intersection"
    leverage: float = 0.5
    entry_delay_bars: int = 1
    trade_stop_loss_pct: float = 0.0
    trade_take_profit_pct: float = 0.0


def _load_many(raw: str) -> dict[tuple[str, int], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in [x.strip() for x in str(raw).replace("\n", ",").split(",") if x.strip()]:
        rows.extend(json.loads(line) for line in Path(p).read_text().splitlines() if line.strip())
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for r in sorted(rows, key=lambda x: (str(x.get("date")), int(x.get("signal_pos", -1) or -1))):
        key = (str(r.get("date")), int(r.get("signal_pos", -1) or -1))
        out.setdefault(key, r)
    if not out:
        raise ValueError(f"no rows loaded from {raw}")
    return out


def _is_trade(row: dict[str, Any] | None) -> bool:
    pred = (row or {}).get("prediction") or {}
    return str(pred.get("gate", "NO_TRADE")) == "TRADE" and str(pred.get("side", "NONE")) in {"LONG", "SHORT"}


def _no_trade_like(source: dict[str, Any], *, reason: str) -> dict[str, Any]:
    out = dict(source)
    out["position_scale"] = 0.0
    out["prediction"] = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "confidence": "LOW", "family": "blend_prediction_sets", "reason": reason}
    return out


def _blend_row(base: dict[str, Any] | None, guard: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    source = base or guard
    if source is None:
        raise ValueError("missing source row")
    bt = _is_trade(base)
    gt = _is_trade(guard)
    same_side = bool(bt and gt and (base or {}).get("prediction", {}).get("side") == (guard or {}).get("prediction", {}).get("side"))
    if mode == "intersection":
        return dict(base) if same_side else _no_trade_like(source, reason="intersection_failed")
    if mode == "base_with_guard_veto":
        return dict(base) if same_side else _no_trade_like(source, reason="guard_veto")
    if mode == "union":
        if bt:
            return dict(base)
        if gt:
            return dict(guard)
        return _no_trade_like(source, reason="both_no_trade")
    if mode == "guard_priority_union":
        if gt:
            return dict(guard)
        if bt:
            return dict(base)
        return _no_trade_like(source, reason="both_no_trade")
    raise ValueError(f"unknown mode: {mode}")


def run(cfg: BlendPredictionSetsCfg) -> dict[str, Any]:
    base = _load_many(cfg.base_predictions)
    guard = _load_many(cfg.guard_predictions)
    keys = sorted(set(base) | set(guard), key=lambda x: (x[0], x[1]))
    blended = [_blend_row(base.get(k), guard.get(k), cfg.mode) for k in keys]
    out_path = Path(cfg.predictions_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in blended))
    bt = run_overlay(OnlineRiskOverlayConfig(
        predictions_jsonl=str(out_path),
        market_csv=cfg.market_csv,
        output=cfg.output,
        leverage=cfg.leverage,
        entry_delay_bars=cfg.entry_delay_bars,
        trade_stop_loss_pct=cfg.trade_stop_loss_pct,
        trade_take_profit_pct=cfg.trade_take_profit_pct,
    ))
    summary = {
        "config": asdict(cfg),
        "rows": {"base": len(base), "guard": len(guard), "blended": len(blended)},
        "trade_counts": {
            "base": sum(_is_trade(r) for r in base.values()),
            "guard": sum(_is_trade(r) for r in guard.values()),
            "blended": sum(_is_trade(r) for r in blended),
        },
        "backtest": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]},
        "leakage_guard": {
            "no_threshold_fit_in_this_script": True,
            "combines_only_prior_walkforward_predictions": True,
            "fixed_logical_mode": cfg.mode,
        },
    }
    Path(cfg.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blend prediction streams and strict-backtest the result")
    p.add_argument("--base-predictions", required=True)
    p.add_argument("--guard-predictions", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--mode", choices=["intersection", "base_with_guard_veto", "union", "guard_priority_union"], default=BlendPredictionSetsCfg.mode)
    p.add_argument("--leverage", type=float, default=BlendPredictionSetsCfg.leverage)
    p.add_argument("--entry-delay-bars", type=int, default=BlendPredictionSetsCfg.entry_delay_bars)
    p.add_argument("--trade-stop-loss-pct", type=float, default=BlendPredictionSetsCfg.trade_stop_loss_pct)
    p.add_argument("--trade-take-profit-pct", type=float, default=BlendPredictionSetsCfg.trade_take_profit_pct)
    return p.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(BlendPredictionSetsCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))
