"""Backtest wave state A/B option predictions using realized candidate rewards."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WaveStateOptionBacktestCfg:
    options_jsonl: str
    predictions_jsonl: str
    output: str
    select_prediction: str = "A"
    min_margin: float = 0.0


def _read(path: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for i, r in enumerate(rows):
        r.setdefault("row_index", i)
    return rows


def _trade_stats(rets_pct: list[float]) -> dict[str, Any]:
    xs = np.asarray(rets_pct, dtype=float)
    if xs.size == 0:
        return {"n_trades": 0, "mean_trade_ret_pct": 0.0, "std_trade_ret_pct": 0.0, "t_stat_like": 0.0, "p_value_mean_ret_approx": 1.0, "positive_rate": 0.0}
    std = float(np.std(xs, ddof=1)) if xs.size > 1 else 0.0
    t = float(np.mean(xs) / (std / math.sqrt(xs.size))) if std > 0 else 0.0
    # Normal approximation, enough for quick diagnostics.
    p = float(math.erfc(abs(t) / math.sqrt(2.0)))
    return {"n_trades": int(xs.size), "mean_trade_ret_pct": float(np.mean(xs)), "std_trade_ret_pct": std, "t_stat_like": t, "p_value_mean_ret_approx": p, "positive_rate": float(np.mean(xs > 0.0))}


def _equity_stats(rows: list[dict[str, Any]], rets_pct: list[float]) -> dict[str, Any]:
    if not rets_pct:
        return {"ret_pct_compound": 0.0, "strict_mdd_pct": 0.0, "cagr_pct": 0.0, "cagr_to_strict_mdd": 0.0, "trade_entries": 0}
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for r in rets_pct:
        eq *= 1.0 + float(r) / 100.0
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)
    ret_pct = (eq - 1.0) * 100.0
    dates = [str(r.get("date", "")) for r in rows]
    try:
        import pandas as pd
        ds = pd.to_datetime([d for d in dates if d])
        days = max(1.0, float((ds.max() - ds.min()).days)) if len(ds) >= 2 else 365.0
    except Exception:
        days = 365.0
    cagr = (eq ** (365.0 / days) - 1.0) * 100.0 if eq > 0 else -100.0
    mdd_pct = mdd * 100.0
    return {"ret_pct_compound": ret_pct, "strict_mdd_pct": mdd_pct, "cagr_pct": cagr, "cagr_to_strict_mdd": cagr / mdd_pct if mdd_pct > 0 else 0.0, "trade_entries": len(rets_pct)}


def run(cfg: WaveStateOptionBacktestCfg) -> dict[str, Any]:
    options = {int(r.get("row_index", i)): r for i, r in enumerate(_read(cfg.options_jsonl))}
    preds = _read(cfg.predictions_jsonl)
    selected: list[dict[str, Any]] = []
    skipped = 0
    for p in preds:
        score = p.get("scores") if isinstance(p.get("scores"), dict) else {}
        margin = float(score.get("A", 0.0)) - float(score.get("B", 0.0))
        if str(p.get("prediction")) != cfg.select_prediction or abs(margin) < float(cfg.min_margin):
            skipped += 1
            continue
        opt = options.get(int(p.get("row_index", -1)))
        if not opt:
            skipped += 1
            continue
        selected.append({**opt, "prediction_scores": score, "prediction_margin": margin})
    selected.sort(key=lambda r: str(r.get("date", "")))
    rets = [float(dict(r.get("choice_utility") or {}).get("A", 0.0) or 0.0) for r in selected]
    report = {
        "config": asdict(cfg),
        "rows": {"options": len(options), "predictions": len(preds), "selected": len(selected), "skipped": skipped},
        "sim": _equity_stats(selected, rets),
        "trade_stats": _trade_stats(rets),
        "selected_preview": selected[:5],
        "leakage_guard": {"selection_uses_predictions_only": True, "rewards_used_for_reporting_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--options-jsonl", required=True)
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--select-prediction", default="A")
    p.add_argument("--min-margin", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(WaveStateOptionBacktestCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
