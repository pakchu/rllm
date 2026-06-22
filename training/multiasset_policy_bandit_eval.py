"""Online policy selector over rolling candidate policies.

The selector chooses next month's policy using only realized performance from prior
months. It is a diagnostic deployability test for regime/sign switching.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def load_mod():
    path = Path(__file__).with_name("multiasset_feature_model_rolling_utility.py")
    spec = importlib.util.spec_from_file_location("rolling_utility", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec); sys.modules["rolling_utility"] = mod; spec.loader.exec_module(mod); return mod


CANDIDATES = [
    ("excess_spread", {"horizon": 288, "stride": 96, "target": "excess", "l2": 100.0, "top_k": 2, "portfolio": "long_short", "regime_filter": "market_mom_pos", "min_score_spread": 0.004, "score_sign": 1.0}),
    ("utility1_pos", {"horizon": 288, "stride": 96, "target": "utility1_excess", "l2": 100.0, "top_k": 2, "portfolio": "long_short", "regime_filter": "market_mom_pos", "min_score_spread": 0.0, "score_sign": 1.0}),
    ("utility1_inv", {"horizon": 288, "stride": 96, "target": "utility1_excess", "l2": 100.0, "top_k": 2, "portfolio": "long_short", "regime_filter": "market_mom_pos", "min_score_spread": 0.0, "score_sign": -1.0}),
    ("utility3_pos", {"horizon": 288, "stride": 96, "target": "utility3_excess", "l2": 100.0, "top_k": 2, "portfolio": "long_short", "regime_filter": "market_mom_pos", "min_score_spread": 0.0, "score_sign": 1.0}),
    ("utility3_inv", {"horizon": 288, "stride": 96, "target": "utility3_excess", "l2": 100.0, "top_k": 2, "portfolio": "long_short", "regime_filter": "market_mom_pos", "min_score_spread": 0.0, "score_sign": -1.0}),
    ("cash", None),
]


def month_ranges(start: str, end: str) -> list[tuple[str, str, str]]:
    out=[]
    for m in pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="MS"):
        n = min(m + pd.offsets.MonthBegin(1), pd.Timestamp(end))
        if m >= pd.Timestamp(end):
            continue
        out.append((str(m.date()), str(m), str(n)))
    return out


def combine_months(months: list[dict[str, Any]]) -> dict[str, Any]:
    eq = peak = 1.0; mdd = 0.0; rets=[]
    for m in months:
        sim = m["sim"]
        month_mdd = float(sim["strict_mdd_pct"]) / 100.0
        trough = eq * (1.0 - month_mdd)
        mdd = max(mdd, 1.0 - trough / peak if peak else 1.0)
        r = float(sim["ret_pct"]) / 100.0
        eq *= max(0.0, 1.0 + r)
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak if peak else 1.0)
        rets.append(r)
    days = max(1, sum(max(1, (pd.Timestamp(m["end"]) - pd.Timestamp(m["start"])).days) for m in months))
    cagr = (eq ** (365.25 / days) - 1.0) * 100.0 if eq > 0 else -100.0
    arr = pd.Series(rets, dtype=float)
    if len(arr) > 1 and arr.std(ddof=1) > 1e-12:
        t = float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr))))
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
    else:
        t = 0.0; p = 1.0
    return {"ret_pct": (eq-1)*100.0, "cagr_pct": cagr, "strict_mdd_pct": mdd*100.0, "cagr_to_strict_mdd": cagr/(mdd*100.0) if mdd>1e-9 else 0.0, "months": len(months), "month_t_stat_like": t, "month_p_value_mean_ret_approx": p}


def score_trailing(months: list[dict[str, Any]]) -> float:
    if not months:
        return 0.0
    c = combine_months(months)
    # prefer positive low-drawdown recent behavior; negative returns lose to cash.
    return float(c["cagr_to_strict_mdd"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    m = load_mod()
    cfg = m.Cfg(data_dir=args.data_dir, aux_dir=args.aux_dir, output=args.output, train_start=args.train_start)
    close = m.load_close_matrix(cfg.data_dir); feats, feature_names = m.build_features(close, cfg.aux_dir)
    months = month_ranges(args.pre_start, args.eval_end)
    monthly: dict[str, dict[str, Any]] = {}
    for cname, params in CANDIDATES:
        monthly[cname] = {}
        for label, start, end in months:
            if params is None:
                res = {"result": {"sim": {"ret_pct": 0.0, "cagr_pct": 0.0, "strict_mdd_pct": 0.0, "cagr_to_strict_mdd": 0.0, "rebalance_count": 0, "avg_turnover_per_rebalance": 0.0}, "stats": {"rebalance_p_value_mean_ret_approx": 1.0}}}
            else:
                res = m.rolling_eval_period(close, feats, feature_names, cfg, params, args.train_days, start, end)
            monthly[cname][label] = {"start": start, "end": end, "sim": res["result"]["sim"], "stats": res["result"].get("stats", {})}
    decisions=[]
    eval_labels=[label for label,_,_ in months if pd.Timestamp(label) >= pd.Timestamp(args.eval_start)]
    for label in eval_labels:
        current = pd.Timestamp(label)
        trailing_labels=[]
        for i in range(args.lookback_months,0,-1):
            trailing_labels.append(str((current - pd.DateOffset(months=i)).date()))
        scores={}
        for cname,_ in CANDIDATES:
            hist=[monthly[cname][tl] for tl in trailing_labels if tl in monthly[cname]]
            scores[cname]=score_trailing(hist)
        selected=max(scores, key=scores.get)
        decisions.append({"month": label, "selected": selected, "scores": scores, **monthly[selected][label]})
    sim=combine_months(decisions)
    report={"lookback_months": args.lookback_months, "train_days": args.train_days, "candidates":[c[0] for c in CANDIDATES], "decisions": decisions, "sim": sim, "leakage_guard":"monthly candidate chosen only from previous lookback months; each candidate month model trains only on data before that month", "note":"combined MDD uses each selected month strict MDD embedded into chained monthly equity, not full 5m path across candidate switches"}
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report,indent=2,ensure_ascii=False)); return report


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--aux-dir',required=True); p.add_argument('--output',required=True); p.add_argument('--train-start',default='2023-01-01'); p.add_argument('--pre-start',default='2024-01-01'); p.add_argument('--eval-start',default='2025-01-01'); p.add_argument('--eval-end',default='2026-06-01'); p.add_argument('--train-days',type=int,default=365); p.add_argument('--lookback-months',type=int,default=3); return p.parse_args()


def main(): print(json.dumps(run(parse_args()),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
