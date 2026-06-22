"""Gate rolling alpha discovery reports before spending LLM/RL cycles.

This module is deliberately conservative. It consumes prior-only rolling alpha
reports and decides whether any candidate is strong enough to become an RLLM
policy prior. It does not run a new search and does not inspect future data beyond
what is already separated into the report's strict folds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlphaGateConfig:
    input_report: str
    output: str
    min_cagr_to_mdd: float = 3.0
    max_strict_mdd_pct: float = 15.0
    min_fold_trades: int = 30
    min_total_trades: int = 300
    min_positive_folds: int = 5
    require_all_folds_mdd: bool = True


def _sim_to_metric(name: str, sim: dict[str, Any]) -> dict[str, Any]:
    return {
        "fold": name,
        "valid": True,
        "cagr_pct": float(sim.get("cagr_pct", 0.0)),
        "strict_mdd_pct": float(sim.get("strict_mdd_pct", 0.0)),
        "cagr_to_strict_mdd": float(sim.get("cagr_to_strict_mdd", 0.0)),
        "trade_entries": int(sim.get("trade_entries", 0)),
    }


def _strict_fold_metrics(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    if "strict_folds" not in candidate and ("test" in candidate or "eval" in candidate):
        folds = []
        for split in ("test", "eval"):
            sim = candidate.get(split, {}).get("sim", {}) if isinstance(candidate.get(split), dict) else {}
            if sim:
                folds.append(_sim_to_metric(split, sim))
        return folds

    out: list[dict[str, Any]] = []
    for fold in candidate.get("strict_folds", []) or []:
        if not isinstance(fold, dict) or "result" not in fold:
            out.append({"fold": fold.get("fold") if isinstance(fold, dict) else None, "valid": False, "error": fold.get("error", "missing_result") if isinstance(fold, dict) else "invalid_fold"})
            continue
        sim = fold.get("result", {}).get("sim", {})
        out.append(_sim_to_metric(str(fold.get("fold")), sim))
    return out


def score_candidate(candidate: dict[str, Any], cfg: AlphaGateConfig) -> dict[str, Any]:
    folds = _strict_fold_metrics(candidate)
    valid = [f for f in folds if f.get("valid")]
    positive_folds = sum(1 for f in valid if float(f["cagr_pct"]) > 0.0)
    passing_ratio_folds = sum(1 for f in valid if float(f["cagr_to_strict_mdd"]) >= float(cfg.min_cagr_to_mdd))
    mdd_ok_folds = sum(1 for f in valid if float(f["strict_mdd_pct"]) <= float(cfg.max_strict_mdd_pct))
    trade_ok_folds = sum(1 for f in valid if int(f["trade_entries"]) >= int(cfg.min_fold_trades))
    total_trades = sum(int(f["trade_entries"]) for f in valid)
    worst_cagr = min((float(f["cagr_pct"]) for f in valid), default=0.0)
    worst_mdd = max((float(f["strict_mdd_pct"]) for f in valid), default=0.0)
    min_ratio = min((float(f["cagr_to_strict_mdd"]) for f in valid), default=0.0)

    failures: list[str] = []
    if not valid:
        failures.append("no_valid_strict_folds")
    if positive_folds < int(cfg.min_positive_folds):
        failures.append("insufficient_positive_folds")
    if passing_ratio_folds < int(cfg.min_positive_folds):
        failures.append("insufficient_cagr_to_mdd_folds")
    if trade_ok_folds < int(cfg.min_positive_folds):
        failures.append("insufficient_trade_count_folds")
    if total_trades < int(cfg.min_total_trades):
        failures.append("insufficient_total_trades")
    if bool(cfg.require_all_folds_mdd) and mdd_ok_folds < len(valid):
        failures.append("mdd_exceeds_limit_in_some_folds")
    elif not bool(cfg.require_all_folds_mdd) and mdd_ok_folds < int(cfg.min_positive_folds):
        failures.append("insufficient_mdd_ok_folds")
    if worst_cagr <= 0.0:
        failures.append("negative_or_zero_worst_fold_cagr")

    return {
        "candidate": {
            "feature": candidate.get("feature"),
            "group": candidate.get("group"),
            "horizon": candidate.get("horizon"),
            "quantile": candidate.get("quantile"),
            "event_score": candidate.get("event_score"),
            "strict_score": candidate.get("strict_score"),
            "selection_score": candidate.get("selection_score"),
            "overlay": candidate.get("overlay"),
        },
        "passed": not failures,
        "failures": failures,
        "summary": {
            "valid_folds": len(valid),
            "positive_folds": positive_folds,
            "passing_ratio_folds": passing_ratio_folds,
            "mdd_ok_folds": mdd_ok_folds,
            "trade_ok_folds": trade_ok_folds,
            "total_trades": total_trades,
            "worst_cagr_pct": worst_cagr,
            "worst_strict_mdd_pct": worst_mdd,
            "min_cagr_to_strict_mdd": min_ratio,
        },
        "folds": folds,
    }


def gate_report(cfg: AlphaGateConfig) -> dict[str, Any]:
    report = json.loads(Path(cfg.input_report).read_text())
    source_key = "top_strict" if "top_strict" in report else "top_by_selection" if "top_by_selection" in report else "top"
    rows = [score_candidate(c, cfg) for c in report.get(source_key, [])]
    rows.sort(
        key=lambda r: (
            bool(r["passed"]),
            int(r["summary"]["positive_folds"]),
            float(r["summary"]["min_cagr_to_strict_mdd"]),
            -float(r["summary"]["worst_strict_mdd_pct"]),
        ),
        reverse=True,
    )
    passed = [r for r in rows if r["passed"]]
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "input_report": str(Path(cfg.input_report).resolve()),
        "config": asdict(cfg),
        "decision": "GO" if passed else "NO_GO",
        "passed_count": len(passed),
        "candidate_count": len(rows),
        "source_key": source_key,
        "top_candidates": rows[:20],
        "blocking_reason": None if passed else "No rolling alpha candidate satisfies strict CAGR/MDD, MDD cap, trade-count, and fold-consistency gates.",
        "leakage_guard": {
            "consumes_existing_rolling_report_only": True,
            "does_not_refit_or_select_on_final_eval": True,
            "requires_strict_fold_or_test_eval_metrics": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate rolling alpha candidates before RLLM training")
    p.add_argument("--input-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-cagr-to-mdd", type=float, default=AlphaGateConfig.min_cagr_to_mdd)
    p.add_argument("--max-strict-mdd-pct", type=float, default=AlphaGateConfig.max_strict_mdd_pct)
    p.add_argument("--min-fold-trades", type=int, default=AlphaGateConfig.min_fold_trades)
    p.add_argument("--min-total-trades", type=int, default=AlphaGateConfig.min_total_trades)
    p.add_argument("--min-positive-folds", type=int, default=AlphaGateConfig.min_positive_folds)
    p.add_argument("--allow-some-fold-mdd-breach", action="store_true")
    args = p.parse_args()
    args.require_all_folds_mdd = not bool(args.allow_some_fold_mdd_breach)
    delattr(args, "allow_some_fold_mdd_breach")
    return args


def main() -> None:
    print(json.dumps(gate_report(AlphaGateConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
