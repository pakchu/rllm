"""Try larger trade pools by recombining existing per-dataset trade statistics.

This script does not regenerate model inference.
It expands pool definitions from an existing significance artifact and
re-estimates significance with summary-stat pooling formulas.
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


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _combine_from_group_stats(groups: list[dict[str, Any]]) -> dict[str, Any]:
    # groups: [{"name":..., "n":..., "mean":..., "std":...}, ...]
    n_total = int(sum(int(g["n"]) for g in groups))
    if n_total <= 1:
        return {
            "n_trades": n_total,
            "mean_trade_ret_pct": 0.0,
            "std_trade_ret_pct": 0.0,
            "t_stat_like": 0.0,
            "p_value_mean_ret_approx": 1.0,
            "effect_size_d": 0.0,
            "n_required_for_80pct_power_alpha5pct": None,
            "n_gap_to_power_rule": None,
        }

    mean = sum(float(g["n"]) * float(g["mean"]) for g in groups) / float(n_total)

    # Combine sample variances:
    # SS_total = sum((n_i-1)s_i^2 + n_i*(mu_i-mu)^2)
    ss = 0.0
    for g in groups:
        n_i = int(g["n"])
        if n_i <= 0:
            continue
        m_i = float(g["mean"])
        s_i = max(0.0, float(g["std"]))
        ss += max(0, n_i - 1) * (s_i**2) + n_i * ((m_i - mean) ** 2)
    var = ss / float(max(1, n_total - 1))
    std = math.sqrt(max(0.0, var))

    se = std / math.sqrt(float(max(1, n_total)))
    t_like = (mean / se) if se > 0.0 else 0.0
    p_two = 2.0 * (1.0 - _norm_cdf(abs(t_like)))
    d = (mean / std) if std > 1e-12 else 0.0
    # One-sample approximation: n ~= ((z_alpha/2 + z_beta)/d)^2
    z_alpha_over_2 = 1.959963984540054
    z_beta_80 = 0.8416212335729143
    n_req = None
    n_gap = None
    if abs(d) > 1e-12:
        n_req = int(math.ceil(((z_alpha_over_2 + z_beta_80) / abs(d)) ** 2))
        n_gap = int(max(0, n_req - n_total))

    ci_half = 1.96 * se
    return {
        "n_trades": int(n_total),
        "mean_trade_ret_pct": float(mean),
        "std_trade_ret_pct": float(std),
        "t_stat_like": float(t_like),
        "p_value_mean_ret_approx": float(p_two),
        "effect_size_d": float(d),
        "bootstrap_mean_95ci_proxy": [float(mean - ci_half), float(mean + ci_half)],
        "n_required_for_80pct_power_alpha5pct": n_req,
        "n_gap_to_power_rule": n_gap,
    }


def _extract_group_map(sig_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    datasets = sig_report.get("datasets", {})
    if not isinstance(datasets, dict):
        return out
    for name, payload in datasets.items():
        if not isinstance(payload, dict):
            continue
        ts = payload.get("trade_stats")
        if not isinstance(ts, dict):
            continue
        out[name] = {
            "n": int(ts.get("n_trades", 0)),
            "mean": _safe_float(ts.get("mean_trade_ret_pct")),
            "std": _safe_float(ts.get("std_trade_ret_pct")),
        }
    return out


def _evaluate(stats: dict[str, Any], *, alpha: float, min_trades_rule: int) -> dict[str, Any]:
    ci = stats.get("bootstrap_mean_95ci_proxy", [0.0, 0.0])
    ci_low = _safe_float(ci[0]) if isinstance(ci, list) and len(ci) >= 1 else 0.0
    passes = {
        "min_trades_rule": bool(int(stats.get("n_trades", 0)) >= int(min_trades_rule)),
        "power_80pct_rule": (
            int(stats.get("n_trades", 0))
            >= int(stats.get("n_required_for_80pct_power_alpha5pct"))
            if stats.get("n_required_for_80pct_power_alpha5pct") is not None
            else False
        ),
        "mean_positive_rule": bool(_safe_float(stats.get("mean_trade_ret_pct")) > 0.0),
        "p_value_rule_two_sided": bool(_safe_float(stats.get("p_value_mean_ret_approx"), 1.0) < alpha),
        "ci_proxy_positive_rule": bool(ci_low > 0.0),
    }
    strict_pass = bool(
        passes["power_80pct_rule"]
        and passes["mean_positive_rule"]
        and passes["p_value_rule_two_sided"]
        and passes["ci_proxy_positive_rule"]
    )
    out = dict(stats)
    out["passes"] = passes
    out["strict_pass"] = strict_pass
    return out


def run_pool_expand(
    input_path: str,
    *,
    alpha: float,
    min_trades_rule: int,
) -> dict[str, Any]:
    sig_report = json.loads(Path(input_path).read_text())
    gm = _extract_group_map(sig_report)

    pools: dict[str, dict[str, Any]] = {}

    # Non-overlap (safer) expansion attempt
    non_overlap_names = ["val_2024h2"] + [f"m{m}" for m in [
        "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"
    ]]
    non_overlap_groups = [
        {"name": n, **gm[n]} for n in non_overlap_names if n in gm
    ]
    if non_overlap_groups:
        stats = _combine_from_group_stats(non_overlap_groups)
        pools["non_overlap_val_plus_8m"] = {
            "members": [g["name"] for g in non_overlap_groups],
            "overlap_risk": "none (period-disjoint)",
            "trade_stats": _evaluate(stats, alpha=alpha, min_trades_rule=min_trades_rule),
        }

    # Exploratory max pool (contains overlap: pooled_core already includes oos Sep-Feb)
    # Keep explicitly flagged for diagnostics only.
    exploratory_names = ["val_2024h2", "oos_2025-09_2026-02"] + [f"m{m}" for m in [
        "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"
    ]]
    exploratory_groups = [
        {"name": n, **gm[n]} for n in exploratory_names if n in gm
    ]
    if exploratory_groups:
        stats = _combine_from_group_stats(exploratory_groups)
        pools["exploratory_overlap_max_pool"] = {
            "members": [g["name"] for g in exploratory_groups],
            "overlap_risk": "high (Sep-Feb duplicated via oos + monthly)",
            "trade_stats": _evaluate(stats, alpha=alpha, min_trades_rule=min_trades_rule),
        }

    # Positive-month-only exploratory (selection-biased; diagnostic only)
    positive_groups = [
        {"name": n, **v} for n, v in gm.items() if float(v["mean"]) > 0.0
    ]
    if positive_groups:
        stats = _combine_from_group_stats(positive_groups)
        pools["exploratory_positive_only"] = {
            "members": [g["name"] for g in positive_groups],
            "overlap_risk": "selection bias (positive-month cherry-pick)",
            "trade_stats": _evaluate(stats, alpha=alpha, min_trades_rule=min_trades_rule),
        }

    strict_passes = [
        name
        for name, payload in pools.items()
        if bool(payload.get("trade_stats", {}).get("strict_pass", False))
    ]

    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "source_significance_report": str(Path(input_path).resolve()),
        "criteria": {
            "alpha": float(alpha),
            "min_trades_rule": int(min_trades_rule),
            "note": "pool stats estimated from per-dataset summary stats (approximate)",
        },
        "pools": pools,
        "summary": {
            "num_pools": int(len(pools)),
            "strict_pass_count": int(len(strict_passes)),
            "strict_pass_pools": strict_passes,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Try larger pools from trade significance report")
    p.add_argument(
        "--input",
        type=str,
        default="results/vlm_qrdqn_ratio3_trade_significance_2026-03-07.json",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results/vlm_qrdqn_trade_pool_expand_2026-03-07.json",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-trades-rule", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_pool_expand(
        args.input,
        alpha=args.alpha,
        min_trades_rule=args.min_trades_rule,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print("[done]", args.output)
    print(json.dumps(out.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
