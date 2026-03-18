"""Validate whether trade count is statistically meaningful from a significance report.

Input is expected to be a JSON artifact like:
  results/vlm_qrdqn_ratio3_trade_significance_2026-03-07.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _eval_trade_stats(
    trade_stats: dict[str, Any],
    *,
    alpha: float,
    min_trades_rule: int,
) -> dict[str, Any]:
    n = int(trade_stats.get("n_trades", 0))
    mean_ret = _safe_float(trade_stats.get("mean_trade_ret_pct"))
    p_two = _safe_float(trade_stats.get("p_value_mean_ret_approx"), default=1.0)
    p_sign = _safe_float(trade_stats.get("p_value_sign_test"), default=1.0)
    n_req_power = math.ceil(
        max(0.0, _safe_float(trade_stats.get("n_required_for_80pct_power_alpha5pct"), default=0.0))
    )
    ci = trade_stats.get("bootstrap_mean_95ci", [0.0, 0.0])
    ci_low = _safe_float(ci[0]) if isinstance(ci, list) and len(ci) >= 1 else 0.0
    ci_high = _safe_float(ci[1]) if isinstance(ci, list) and len(ci) >= 2 else 0.0

    passes = {
        "min_trades_rule": bool(n >= int(min_trades_rule)),
        "power_80pct_rule": bool(n >= int(n_req_power)) if n_req_power > 0 else False,
        "mean_positive_rule": bool(mean_ret > 0.0),
        "p_value_rule_two_sided": bool(p_two < float(alpha)),
        "bootstrap_ci_positive_rule": bool(ci_low > 0.0),
        "sign_test_rule": bool(p_sign < float(alpha)),
    }
    strict_pass = bool(
        passes["power_80pct_rule"]
        and passes["mean_positive_rule"]
        and passes["p_value_rule_two_sided"]
        and passes["bootstrap_ci_positive_rule"]
    )

    return {
        "n_trades": n,
        "mean_trade_ret_pct": mean_ret,
        "p_value_mean_ret_approx": p_two,
        "p_value_sign_test": p_sign,
        "bootstrap_mean_95ci": [ci_low, ci_high],
        "n_required_for_80pct_power_alpha5pct": int(n_req_power),
        "n_gap_to_power_rule": int(max(0, n_req_power - n)),
        "passes": passes,
        "strict_pass": strict_pass,
    }


def _extract_datasets(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    datasets = report.get("datasets", {})
    if isinstance(datasets, dict):
        for name, payload in datasets.items():
            if isinstance(payload, dict) and isinstance(payload.get("trade_stats"), dict):
                out[name] = payload["trade_stats"]

    pooled_core = report.get("pooled_core", {})
    if isinstance(pooled_core, dict) and isinstance(pooled_core.get("trade_stats"), dict):
        out["pooled_core"] = pooled_core["trade_stats"]

    pooled_8m = report.get("pooled_monthly_8m", {})
    if isinstance(pooled_8m, dict) and isinstance(pooled_8m.get("trade_stats"), dict):
        out["pooled_monthly_8m"] = pooled_8m["trade_stats"]

    return out


def run_validation(
    input_path: str,
    *,
    alpha: float,
    min_trades_rule: int,
) -> dict[str, Any]:
    report = json.loads(Path(input_path).read_text())
    datasets = _extract_datasets(report)

    evaluations = {
        name: _eval_trade_stats(stats, alpha=alpha, min_trades_rule=min_trades_rule)
        for name, stats in datasets.items()
    }

    strict_passes = [k for k, v in evaluations.items() if bool(v.get("strict_pass"))]
    near_candidates = sorted(
        evaluations.items(),
        key=lambda kv: (
            int(kv[1].get("n_gap_to_power_rule", 10**9)),
            _safe_float(kv[1].get("p_value_mean_ret_approx"), default=1.0),
        ),
    )
    near_positive_candidates = sorted(
        (
            kv
            for kv in evaluations.items()
            if bool(kv[1].get("passes", {}).get("mean_positive_rule", False))
        ),
        key=lambda kv: (
            int(kv[1].get("n_gap_to_power_rule", 10**9)),
            _safe_float(kv[1].get("p_value_mean_ret_approx"), default=1.0),
        ),
    )

    projection = {}
    oos = evaluations.get("oos_2025-09_2026-02")
    if oos:
        n = int(oos.get("n_trades", 0))
        need = int(oos.get("n_gap_to_power_rule", 0))
        start = str(report.get("datasets", {}).get("oos_2025-09_2026-02", {}).get("period", {}).get("start", ""))
        end = str(report.get("datasets", {}).get("oos_2025-09_2026-02", {}).get("period", {}).get("end", ""))
        months = 0.0
        try:
            s = datetime.fromisoformat(start)
            e = datetime.fromisoformat(end)
            months = max(1e-9, (e.year - s.year) * 12 + (e.month - s.month) + (e.day - s.day + 1) / 30.0)
        except Exception:
            months = 0.0
        trades_per_month = (float(n) / months) if months > 0 else 0.0
        months_needed = (float(need) / trades_per_month) if trades_per_month > 0 else None
        projection = {
            "reference_dataset": "oos_2025-09_2026-02",
            "reference_period": {"start": start, "end": end},
            "observed_trades": int(n),
            "observed_trade_rate_per_month": float(trades_per_month),
            "additional_trades_needed_for_power_rule": int(need),
            "estimated_additional_months_needed_at_same_trade_rate": (
                float(months_needed) if months_needed is not None else None
            ),
        }

    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "source_report": str(Path(input_path).resolve()),
        "criteria": {
            "alpha": float(alpha),
            "min_trades_rule": int(min_trades_rule),
            "strict_pass_definition": (
                "power_80pct_rule AND mean_positive_rule "
                "AND p_value_rule_two_sided AND bootstrap_ci_positive_rule"
            ),
        },
        "summary": {
            "num_evaluated_sets": int(len(evaluations)),
            "strict_pass_count": int(len(strict_passes)),
            "strict_pass_sets": strict_passes,
            "closest_set_to_power_rule": near_candidates[0][0] if near_candidates else None,
            "closest_positive_set_to_power_rule": (
                near_positive_candidates[0][0] if near_positive_candidates else None
            ),
        },
        "evaluations": evaluations,
        "sizing_projection": projection,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate statistical significance of trade count")
    p.add_argument(
        "--input",
        type=str,
        default="results/vlm_qrdqn_ratio3_trade_significance_2026-03-07.json",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results/vlm_qrdqn_trade_count_validation_2026-03-07.json",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-trades-rule", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_validation(
        args.input,
        alpha=args.alpha,
        min_trades_rule=args.min_trades_rule,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print("[done]", args.output)
    print(
        json.dumps(
            {
                "strict_pass_count": out["summary"]["strict_pass_count"],
                "closest_set_to_power_rule": out["summary"]["closest_set_to_power_rule"],
                "closest_positive_set_to_power_rule": out["summary"].get(
                    "closest_positive_set_to_power_rule"
                ),
                "projection": out.get("sizing_projection", {}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
