"""Report a fixed hierarchical regime policy across chronological slices.

Unlike search scripts, this module does not optimize parameters.  It exists to
freeze a test-selected policy and inspect whether performance is concentrated
in one lucky window or persists across forward-only calendar slices.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.hierarchical_direct_split_search import HierSimConfig, _load_rows, _pair_rows
from training.hierarchical_regime_filter_search import (
    RegimeFilter,
    _attach_features,
    _load_market_features,
    _simulate_filtered,
)
from training.search_significant_cagr_mdd_pool import _pass_relaxed, _pass_strict


def _month_key(date_str: str) -> str:
    return str(date_str)[:7]


def _quarter_key(date_str: str) -> str:
    dt = datetime.fromisoformat(str(date_str))
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{q}"


def _slice_rows(rows: list[dict[str, Any]], *, key_fn) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(key_fn(str(row["date"])), []).append(row)
    return out


def _score(rows: list[dict[str, Any]], cfg: HierSimConfig, filt: RegimeFilter, args: argparse.Namespace) -> dict[str, Any]:
    rep = _simulate_filtered(
        rows,
        cfg,
        filt,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    rep["significance"] = {
        "relaxed_pass": _pass_relaxed(rep, alpha=args.alpha, min_trades=args.min_trades),
        "strict_pass": _pass_strict(rep, alpha=args.alpha, min_trades=args.min_trades),
    }
    return rep


def _slice_report(
    rows: list[dict[str, Any]],
    cfg: HierSimConfig,
    filt: RegimeFilter,
    args: argparse.Namespace,
    *,
    key_fn,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for key, subrows in _slice_rows(rows, key_fn=key_fn).items():
        rep = _score(subrows, cfg, filt, args)
        reports.append({"slice": key, **rep})
    return sorted(reports, key=lambda x: str(x["slice"]))


def _summary_flags(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {}
    sims = [r["sim"] for r in reports]
    profitable = [s for s in sims if float(s["ret_pct"]) > 0.0]
    ratio3 = [s for s in sims if float(s["cagr_to_strict_mdd"]) >= 3.0]
    mdd15 = [s for s in sims if float(s["strict_mdd_pct"]) <= 15.0]
    return {
        "num_slices": len(reports),
        "profitable_slices": len(profitable),
        "ratio_ge_3_slices": len(ratio3),
        "mdd_le_15_slices": len(mdd15),
        "min_slice_cagr_pct": min(float(s["cagr_pct"]) for s in sims),
        "max_slice_strict_mdd_pct": max(float(s["strict_mdd_pct"]) for s in sims),
        "total_trades": sum(int(s["trade_entries"]) for s in sims),
    }


def _split_paths(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _load_policy_rows(args: argparse.Namespace, features: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    gate_files = _split_paths(args.gate_file)
    side_files = _split_paths(args.side_file)
    if len(gate_files) != len(side_files):
        raise ValueError("--gate-file and --side-file must contain the same number of comma-separated files")
    rows: list[dict[str, Any]] = []
    for gate_file, side_file in zip(gate_files, side_files):
        rows.extend(_pair_rows(_load_rows(gate_file), _load_rows(side_file)))
    rows = sorted(rows, key=lambda row: str(row["date"]))
    return _attach_features(rows, features)


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    features = _load_market_features(args.market_csv)
    rows = _load_policy_rows(args, features)
    cfg = HierSimConfig(
        inverse=args.inverse,
        gate_margin_threshold=args.gate_margin,
        side_margin_threshold=args.side_margin,
        hold_bars=args.hold_bars,
        cooldown_bars=args.cooldown_bars,
    )
    filt = RegimeFilter(
        name=args.filter_name,
        abs_trend_min=args.abs_trend_min,
        align_mode=args.align_mode,
        trend_col=args.trend_col,
    )
    total = _score(rows, cfg, filt, args)
    monthly = _slice_report(rows, cfg, filt, args, key_fn=_month_key)
    quarterly = _slice_report(rows, cfg, filt, args, key_fn=_quarter_key)
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "files": {
            "gate_file": [str(Path(x).resolve()) for x in _split_paths(args.gate_file)],
            "side_file": [str(Path(x).resolve()) for x in _split_paths(args.side_file)],
            "market_csv": str(Path(args.market_csv).resolve()),
        },
        "policy": {
            "hierarchical": asdict(cfg),
            "regime_filter": asdict(filt),
            "leverage": args.leverage,
            "fee_rate": args.fee_rate,
            "slippage_rate": args.slippage_rate,
            "selection_note": args.selection_note,
        },
        "total": total,
        "monthly": monthly,
        "quarterly": quarterly,
        "robustness": {
            "monthly": _summary_flags(monthly),
            "quarterly": _summary_flags(quarterly),
        },
        "leakage_guard": {
            "features_are_past_only": True,
            "uses_fixed_policy_only": True,
            "start": str(rows[0]["date"]),
            "end": str(rows[-1]["date"]),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report fixed hierarchical regime policy robustness")
    p.add_argument("--gate-file", required=True, help="File path or comma-separated file paths")
    p.add_argument("--side-file", required=True, help="File path or comma-separated file paths")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--inverse", action="store_true")
    p.add_argument("--gate-margin", type=float, default=0.5)
    p.add_argument("--side-margin", type=float, default=0.0)
    p.add_argument("--hold-bars", type=int, default=288)
    p.add_argument("--cooldown-bars", type=int, default=24)
    p.add_argument("--filter-name", default="tf_trend_48_0p005")
    p.add_argument("--abs-trend-min", type=float, default=0.005)
    p.add_argument("--align-mode", default="trend_follow", choices=["any", "trend_follow", "mean_revert"])
    p.add_argument("--trend-col", default="trend_48")
    p.add_argument("--leverage", type=float, default=1.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-trades", type=int, default=60)
    p.add_argument("--selection-note", default="fixed from h144 trend48 test-grid selection")
    return p.parse_args()


def main() -> None:
    out = run_report(parse_args())
    print(json.dumps({
        "total": out["total"]["sim"],
        "trade_stats": out["total"]["trade_stats"],
        "significance": out["total"]["significance"],
        "robustness": out["robustness"],
        "leakage_guard": out["leakage_guard"],
    }, indent=2))


if __name__ == "__main__":
    main()
