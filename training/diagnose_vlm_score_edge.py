"""Diagnose whether VLM action scores correlate with realized strict trade edge.

This is diagnostic only: it does not choose thresholds or optimize gates.  It
replays every non-HOLD predicted action from an eval_vlm_policy report with the
same strict delayed-entry fixed-hold execution and reports correlations between
model score features and realized trade returns.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.strict_bar_backtest import BarExecutionConfig, load_market_bars


def _signal(action: str) -> int:
    key = str(action).upper().strip()
    if key in {"BUY", "LONG"}:
        return 1
    if key in {"SELL", "SHORT"}:
        return -1
    return 0


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-24 or vy <= 1e-24:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    vals = sorted(values)
    def q(p: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
        return float(vals[idx])
    return {"min": vals[0], "q25": q(0.25), "median": q(0.5), "q75": q(0.75), "max": vals[-1]}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _realized_trade_return(
    row: dict[str, Any],
    date_to_pos: dict,
    opens,
    highs,
    lows,
    market_len: int,
    cfg: BarExecutionConfig,
    hold_bars: int,
) -> float | None:
    dt = datetime.fromisoformat(str(row["date"]))
    pos = date_to_pos.get(dt.replace(tzinfo=None))
    sig = _signal(str(row.get("pred", "HOLD")))
    if pos is None or sig == 0:
        return None
    entry_pos = pos + int(cfg.entry_delay_bars)
    exit_pos = entry_pos + int(hold_bars)
    if entry_pos >= market_len - 1 or exit_pos >= market_len:
        return None
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)
    eq = max(0.0, 1.0 - cost)
    for j in range(entry_pos, exit_pos):
        open_j = float(opens[j])
        if open_j <= 0.0:
            continue
        close_ret = (float(opens[j + 1]) - open_j) / open_j if sig > 0 else (open_j - float(opens[j + 1])) / open_j
        eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
        if eq <= 0.0:
            break
    eq *= max(0.0, 1.0 - cost)
    return eq - 1.0


def diagnose(eval_report: str, market_csv: str, cfg: BarExecutionConfig, hold_bars: int) -> dict[str, Any]:
    report = json.loads(Path(eval_report).read_text())
    rows = report.get("action_scores") or []
    market = load_market_bars(market_csv)
    date_to_pos = {ts.to_pydatetime().replace(tzinfo=None): int(i) for i, ts in enumerate(market["date"])}
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    market_len = len(market)
    points: list[dict[str, float | str]] = []
    for row in rows:
        scores = row.get("scores") or {}
        pred = str(row.get("pred", "HOLD")).upper()
        sig = _signal(pred)
        ret = _realized_trade_return(row, date_to_pos, opens, highs, lows, market_len, cfg, hold_bars)
        if sig == 0 or ret is None:
            continue
        buy = float(scores.get("BUY", float("nan")))
        hold = float(scores.get("HOLD", float("nan")))
        sell = float(scores.get("SELL", float("nan")))
        if any(math.isnan(x) for x in [buy, hold, sell]):
            continue
        directional_margin = buy - sell
        chosen_score = buy if sig > 0 else sell
        opposite_score = sell if sig > 0 else buy
        signed_margin = directional_margin if sig > 0 else -directional_margin
        edge_over_hold = chosen_score - hold
        points.append({
            "date": str(row.get("date")),
            "pred": pred,
            "ret": float(ret),
            "signed_ret": float(ret),
            "chosen_score": float(chosen_score),
            "opposite_score": float(opposite_score),
            "hold_score": float(hold),
            "signed_margin": float(signed_margin),
            "abs_directional_margin": float(abs(directional_margin)),
            "edge_over_hold": float(edge_over_hold),
            "chosen_minus_opposite": float(chosen_score - opposite_score),
        })
    rets = [float(p["ret"]) for p in points]
    features = ["chosen_score", "opposite_score", "hold_score", "signed_margin", "abs_directional_margin", "edge_over_hold", "chosen_minus_opposite"]
    correlations = {f: _corr([float(p[f]) for p in points], rets) for f in features}
    by_side: dict[str, Any] = {}
    for side in ["BUY", "SELL"]:
        vals = [float(p["ret"]) for p in points if p["pred"] == side]
        by_side[side] = {"n": len(vals), "mean_ret_pct": _mean(vals) * 100.0, "ret_quantiles_pct": None if not vals else {k: v * 100.0 for k, v in (_quantiles(vals) or {}).items()}}
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"eval_report": eval_report, "market_csv": market_csv},
        "execution": cfg.__dict__ | {"hold_bars": int(hold_bars)},
        "n_points": len(points),
        "mean_trade_ret_pct": _mean(rets) * 100.0,
        "ret_quantiles_pct": None if not rets else {k: v * 100.0 for k, v in (_quantiles(rets) or {}).items()},
        "correlations_to_trade_return": correlations,
        "by_side": by_side,
        "leakage_guard": {
            "uses_model_scores_and_predictions_only": True,
            "uses_targets_for_diagnostics": False,
            "optimizes_thresholds_or_gates": False,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose VLM score edge correlations")
    p.add_argument("--eval-report", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hold-bars", type=int, default=144)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BarExecutionConfig(
        leverage=float(args.leverage),
        fee_rate=float(args.fee_rate),
        slippage_rate=float(args.slippage_rate),
        drawdown_stop=1.0,
        pause_bars=0,
        monthly_loss_stop=1.0,
        entry_delay_bars=int(args.entry_delay_bars),
    )
    report = diagnose(args.eval_report, args.market_csv, cfg, int(args.hold_bars))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
