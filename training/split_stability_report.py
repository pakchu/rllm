"""Report cross-split stability for train/test/eval policy search artifacts.

This is an anti-overfit gate: a candidate is not considered useful just because
it tops one selection window.  The report emphasizes test->eval performance gap,
minimum holdout ratio, and whether the candidate survives basic robustness
criteria.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metric(row: dict[str, Any], split: str) -> dict[str, Any]:
    block = row.get(split) or row.get(f"{split}_metrics") or {}
    if isinstance(block, dict) and "metrics" in block:
        return block["metrics"]
    return block if isinstance(block, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _summarize_candidate(row: dict[str, Any], index: int) -> dict[str, Any]:
    test = _metric(row, "test")
    evalm = _metric(row, "eval")
    test_ratio = _safe_float(test.get("cagr_to_mdd_proxy", test.get("cagr_to_strict_mdd", 0.0)))
    eval_ratio = _safe_float(evalm.get("cagr_to_mdd_proxy", evalm.get("cagr_to_strict_mdd", 0.0)))
    test_cagr = _safe_float(test.get("cagr_proxy", test.get("cagr_pct", 0.0)))
    eval_cagr = _safe_float(evalm.get("cagr_proxy", evalm.get("cagr_pct", 0.0)))
    test_mdd = _safe_float(test.get("strict_mdd_proxy", test.get("strict_mdd_pct", 0.0)))
    eval_mdd = _safe_float(evalm.get("strict_mdd_proxy", evalm.get("strict_mdd_pct", 0.0)))
    return {
        "rank": int(index),
        "test": {"trades": int(test.get("trades", test.get("trade_entries", 0)) or 0), "cagr": test_cagr, "strict_mdd": test_mdd, "ratio": test_ratio},
        "eval": {"trades": int(evalm.get("trades", evalm.get("trade_entries", 0)) or 0), "cagr": eval_cagr, "strict_mdd": eval_mdd, "ratio": eval_ratio},
        "generalization_gap": {"ratio_eval_minus_test": eval_ratio - test_ratio, "cagr_eval_minus_test": eval_cagr - test_cagr, "mdd_eval_minus_test": eval_mdd - test_mdd},
        "config": row.get("config") or row.get("overlay_config") or row.get("drift_config") or row.get("policy") or {},
    }


def _passes_gate(summary: dict[str, Any], *, min_eval_trades: int, min_eval_ratio: float, max_eval_mdd: float, max_ratio_gap: float) -> bool:
    return (
        int(summary["eval"]["trades"]) >= int(min_eval_trades)
        and float(summary["eval"]["ratio"]) >= float(min_eval_ratio)
        and float(summary["eval"]["strict_mdd"]) <= float(max_eval_mdd)
        and float(summary["generalization_gap"]["ratio_eval_minus_test"]) >= -float(max_ratio_gap)
    )


def run_report(
    *,
    inputs: str,
    output: str,
    min_eval_trades: int = 30,
    min_eval_ratio: float = 3.0,
    max_eval_mdd: float = 0.15,
    max_ratio_gap: float = 3.0,
    top_k: int = 10,
) -> dict[str, Any]:
    paths = [x.strip() for x in str(inputs).split(",") if x.strip()]
    reports = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        rows = payload.get("top_by_test_then_eval") or payload.get("top") or []
        summaries = [_summarize_candidate(row, i + 1) for i, row in enumerate(rows[: int(top_k)])]
        for s in summaries:
            s["passes_gate"] = _passes_gate(s, min_eval_trades=min_eval_trades, min_eval_ratio=min_eval_ratio, max_eval_mdd=max_eval_mdd, max_ratio_gap=max_ratio_gap)
        reports.append(
            {
                "path": path,
                "periods": payload.get("periods", {}),
                "leakage_guard": payload.get("leakage_guard", {}),
                "criteria": {"min_eval_trades": min_eval_trades, "min_eval_ratio": min_eval_ratio, "max_eval_mdd": max_eval_mdd, "max_ratio_gap": max_ratio_gap},
                "num_pass": sum(1 for s in summaries if s["passes_gate"]),
                "candidates": summaries,
            }
        )
    out = {"reports": reports, "overall_pass": any(r["num_pass"] > 0 for r in reports)}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize split stability and anti-overfit gates")
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-eval-trades", type=int, default=30)
    p.add_argument("--min-eval-ratio", type=float, default=3.0)
    p.add_argument("--max-eval-mdd", type=float, default=0.15)
    p.add_argument("--max-ratio-gap", type=float, default=3.0)
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_report(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
