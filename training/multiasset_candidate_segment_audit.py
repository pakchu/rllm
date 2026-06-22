"""Segment-level audit for rolling multi-asset candidate policies.

This is a diagnostic script: it does not select parameters on eval.  It exposes
whether promising aggregate candidates are stable across half-year regimes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def load_utility_module():
    path = Path(__file__).with_name("multiasset_feature_model_rolling_utility.py")
    spec = importlib.util.spec_from_file_location("rolling_utility", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rolling_utility"] = mod
    spec.loader.exec_module(mod)
    return mod


CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "rolling_excess_spread_0p004",
        "rationale": "best prior diagnostic after rolling retrain + score-spread abstention",
        "params": {"horizon": 288, "stride": 96, "target": "excess", "l2": 100.0, "top_k": 2, "portfolio": "long_short", "regime_filter": "market_mom_pos", "min_score_spread": 0.004, "score_sign": 1.0},
    },
    {
        "name": "utility1_positive_eval_diagnostic",
        "rationale": "utility branch with positive 2025-2026 eval but failed 2024 validation",
        "params": {"horizon": 288, "stride": 96, "target": "utility1_excess", "l2": 100.0, "top_k": 2, "portfolio": "long_short", "regime_filter": "market_mom_pos", "min_score_spread": 0.0, "score_sign": 1.0},
    },
    {
        "name": "utility3_validation_sign_inverted",
        "rationale": "best utility sign-sweep validation return but failed strict MDD and eval",
        "params": {"horizon": 288, "stride": 96, "target": "utility3_excess", "l2": 100.0, "top_k": 2, "portfolio": "long_short", "regime_filter": "market_mom_pos", "min_score_spread": 0.0, "score_sign": -1.0},
    },
]

SEGMENTS = [
    ("2024H1", "2024-01-01", "2024-07-01"),
    ("2024H2", "2024-07-01", "2025-01-01"),
    ("2025H1", "2025-01-01", "2025-07-01"),
    ("2025H2", "2025-07-01", "2026-01-01"),
    ("2026YTD", "2026-01-01", "2026-06-01"),
]


def run(args: argparse.Namespace) -> dict[str, Any]:
    m = load_utility_module()
    cfg = m.Cfg(data_dir=args.data_dir, aux_dir=args.aux_dir, output=args.output, train_start=args.train_start)
    close = m.load_close_matrix(cfg.data_dir)
    feats, feature_names = m.build_features(close, cfg.aux_dir)
    rows = []
    for cand in CANDIDATES:
        segs = []
        for name, start, end in SEGMENTS:
            res = m.rolling_eval_period(close, feats, feature_names, cfg, cand["params"], args.train_days, start, end)
            segs.append({"segment": name, "start": start, "end": end, "sim": res["result"]["sim"], "stats": res["result"]["stats"], "models": len(res["model_windows"])})
        rows.append({"name": cand["name"], "rationale": cand["rationale"], "params": cand["params"], "segments": segs})
    report = {
        "train_days": args.train_days,
        "candidates": rows,
        "conclusion_hint": "A deployable alpha should not depend on one half-year; inspect segment signs, MDD, and p-values before further modeling.",
        "leakage_guard": "Each segment uses monthly models trained only before that segment month; no segment future data is used for features or fitting.",
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--aux-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--train-days", type=int, default=365)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
