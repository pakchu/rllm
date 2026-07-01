"""Audit chronological stability of meta-controller prompt features.

This is a CPU preflight for LLM SFT.  It parses signal-time prompt features,
measures simple feature/TAKE correlations per chronological period, and reports
which clues keep the same sign out of sample.  The intent is to avoid giving
Gemma a large unstable prompt surface that only fits one regime.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.linear_alpha_meta_feature_diagnostic import _read_jsonl, _row_features, _target_decision


@dataclass(frozen=True)
class StabilityConfig:
    inputs: str
    output: str
    period: str = "halfyear"
    min_rows: int = 500
    min_periods: int = 4
    max_features: int = 256
    trade_only: bool = False


def _date(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("date") or row.get("date") or "")


def _period_key(date_s: str, period: str) -> str:
    year = date_s[:4] if len(date_s) >= 4 else "unknown"
    try:
        month = int(date_s[5:7])
    except Exception:
        month = 1
    if period == "year":
        return year
    if period == "quarter":
        return f"{year}Q{((month - 1) // 3) + 1}"
    if period == "halfyear":
        return f"{year}H{1 if month <= 6 else 2}"
    raise ValueError("period must be year|halfyear|quarter")


def _feature_map(row: dict[str, Any]) -> dict[str, float]:
    dense, cats = _row_features(row)
    out = {f"num:{k}": float(v) for k, v in dense.items() if math.isfinite(float(v))}
    out.update({k: 1.0 for k in cats})
    return out


def _feature_space(rows: list[dict[str, Any]], max_features: int) -> list[str]:
    numeric: set[str] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        dense, cats = _row_features(row)
        numeric.update(f"num:{k}" for k in dense)
        counts.update(cats.keys())
    cats = [k for k, _ in counts.most_common(max(0, int(max_features) - len(numeric)))]
    return sorted(numeric) + cats


def _corr(xs: list[float], ys: list[float]) -> float | None:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 30 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run(cfg: StabilityConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw in cfg.inputs.split(","):
        path = raw.strip()
        if path:
            rows.extend(_read_jsonl(path))
    if cfg.trade_only:
        rows = [r for r in rows if str(r.get("metadata", {}).get("candidate_gate", "")).upper() == "TRADE"]
    features = _feature_space(rows, int(cfg.max_features))
    by_period: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_period[_period_key(_date(row), cfg.period)].append(row)

    period_stats: dict[str, Any] = {}
    feature_periods: dict[str, dict[str, float]] = {f: {} for f in features}
    for period, period_rows in sorted(by_period.items()):
        y = [1.0 if _target_decision(r) == "TAKE" else 0.0 for r in period_rows]
        counts = Counter("TAKE" if v == 1.0 else "SKIP" for v in y)
        period_stats[period] = {"rows": len(period_rows), "target_counts": dict(sorted(counts.items()))}
        if len(period_rows) < int(cfg.min_rows):
            continue
        maps = [_feature_map(r) for r in period_rows]
        for feat in features:
            c = _corr([m.get(feat, 0.0) for m in maps], y)
            if c is not None:
                feature_periods[feat][period] = c

    rows_out = []
    eligible_periods = [p for p, rs in sorted(by_period.items()) if len(rs) >= int(cfg.min_rows)]
    for feat, per in feature_periods.items():
        vals = [float(per[p]) for p in eligible_periods if p in per]
        if len(vals) < int(cfg.min_periods):
            continue
        signs = [1 if v > 0 else -1 if v < 0 else 0 for v in vals]
        nonzero = [s for s in signs if s]
        sign_consistency = abs(sum(nonzero)) / max(1, len(nonzero))
        min_abs = min(abs(v) for v in vals)
        mean_abs = float(np.mean([abs(v) for v in vals]))
        last = vals[-1]
        first = vals[0]
        rows_out.append(
            {
                "feature": feat,
                "score": float(sign_consistency * mean_abs + min_abs),
                "sign_consistency": float(sign_consistency),
                "min_abs_corr": float(min_abs),
                "mean_abs_corr": mean_abs,
                "first_corr": float(first),
                "last_corr": float(last),
                "period_corr": {p: per[p] for p in eligible_periods if p in per},
            }
        )
    rows_out.sort(key=lambda r: (float(r["sign_consistency"]), float(r["min_abs_corr"]), float(r["mean_abs_corr"])), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": len(rows),
        "feature_count": len(features),
        "eligible_periods": eligible_periods,
        "period_stats": period_stats,
        "stable_top": rows_out[:50],
        "all": rows_out,
        "leakage_guard": {
            "features_parsed_from_llm_prompt_only": True,
            "future_labels_used_only_for_offline_feature_audit": True,
            "no_fit_on_eval_for_model_weights": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit chronological stability of linear-alpha meta-controller features")
    p.add_argument("--inputs", required=True, help="comma-separated JSONL files")
    p.add_argument("--output", required=True)
    p.add_argument("--period", choices=["year", "halfyear", "quarter"], default=StabilityConfig.period)
    p.add_argument("--min-rows", type=int, default=StabilityConfig.min_rows)
    p.add_argument("--min-periods", type=int, default=StabilityConfig.min_periods)
    p.add_argument("--max-features", type=int, default=StabilityConfig.max_features)
    p.add_argument("--trade-only", action="store_true", default=StabilityConfig.trade_only)
    return p.parse_args()


def main() -> None:
    report = run(StabilityConfig(**vars(parse_args())))
    print(json.dumps({"rows": report["rows"], "periods": report["eligible_periods"], "stable_top": report["stable_top"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
