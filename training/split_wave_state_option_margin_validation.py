"""Validate wave option predictions with a chronological test/eval margin split."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.backtest_wave_state_option_predictions import _equity_stats, _trade_stats


@dataclass(frozen=True)
class SplitWaveOptionMarginCfg:
    options_jsonl: str
    predictions_jsonl: str
    output: str
    test_end: str = "2025-12-31 23:59:59"
    margins: str = "0,0.25,0.5,1.0,1.5,2.0"
    min_test_trades: int = 20


def _read(path: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for i, r in enumerate(rows):
        r.setdefault("row_index", i)
    return rows


def _margin(pred: dict[str, Any]) -> float:
    score = pred.get("scores") if isinstance(pred.get("scores"), dict) else {}
    return float(score.get("A", 0.0)) - float(score.get("B", 0.0))


def _selected(rows: list[dict[str, Any]], margin: float) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("prediction")) == "A" and abs(float(r.get("prediction_margin", 0.0))) >= float(margin)]


def _summ(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: str(r.get("date", "")))
    rets = [float(dict(r.get("choice_utility") or {}).get("A", 0.0) or 0.0) for r in rows]
    return {"rows": len(rows), "sim": _equity_stats(rows, rets), "trade_stats": _trade_stats(rets)}


def _score(summary: dict[str, Any], min_trades: int) -> float:
    sim = summary["sim"]
    stats = summary["trade_stats"]
    trades = int(sim.get("trade_entries", 0) or 0)
    if trades < int(min_trades):
        return -1e9 + trades
    return float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0) + 0.02 * float(sim.get("cagr_pct", 0.0) or 0.0) - float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0)


def run(cfg: SplitWaveOptionMarginCfg) -> dict[str, Any]:
    options = {int(r.get("row_index", i)): r for i, r in enumerate(_read(cfg.options_jsonl))}
    preds = _read(cfg.predictions_jsonl)
    joined = []
    for p in preds:
        opt = options.get(int(p.get("row_index", -1)))
        if not opt:
            continue
        joined.append({**opt, "prediction": p.get("prediction"), "prediction_scores": p.get("scores"), "prediction_margin": _margin(p)})
    test = [r for r in joined if str(r.get("date", "")) <= str(cfg.test_end)]
    ev = [r for r in joined if str(r.get("date", "")) > str(cfg.test_end)]
    rows = []
    for margin in [float(x) for x in str(cfg.margins).split(",") if x.strip()]:
        test_s = _summ(_selected(test, margin))
        eval_s = _summ(_selected(ev, margin))
        rows.append({"margin": margin, "test": test_s, "eval": eval_s, "selection_score": _score(test_s, int(cfg.min_test_trades))})
    rows.sort(key=lambda r: float(r["selection_score"]), reverse=True)
    report = {
        "config": asdict(cfg),
        "rows": {"joined": len(joined), "test": len(test), "eval": len(ev)},
        "ranked_by_test": rows,
        "selected_by_test": rows[0] if rows else None,
        "leakage_guard": {"margin_ranked_on_test_only": True, "eval_not_used_for_selection": True, "rewards_reporting_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--options-jsonl", required=True)
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--test-end", default=SplitWaveOptionMarginCfg.test_end)
    p.add_argument("--margins", default=SplitWaveOptionMarginCfg.margins)
    p.add_argument("--min-test-trades", type=int, default=SplitWaveOptionMarginCfg.min_test_trades)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(SplitWaveOptionMarginCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
