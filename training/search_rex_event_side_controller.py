"""Search simple causal side filters for the REX HTF pullback reclaim event.

The event-first scan found that the base event is regime fragile: side quality
flips by year.  This probe keeps the base event threshold fixed from train and
searches small, interpretable side allow-filters with thresholds fit on train.
Selection uses train+test; eval is final holdout.
"""
from __future__ import annotations

import argparse, json, math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.event_candidate_pool_probe import EventPoolConfig,_load_market,_feature_candidates,_candidate_rows_for_family,_simulate_rows,_split_mask

FEATURES = [
    "trend_24","trend_96","return_zscore_48","range_pos","bb_z","rsi_norm",
    "htf_4h_return_1","htf_4h_return_4","htf_1d_return_1","htf_1d_return_4","htf_3d_return_4","htf_1w_return_4",
    "taker_imbalance","volume_zscore","window_drawdown","range_vol",
    "dxy_zscore","dxy_momentum","usdkrw_zscore","usdkrw_momentum","kimchi_premium_zscore","kimchi_premium_change",
    "oi_zscore","oi_change","funding_zscore","premium_index_zscore",
    "rex_144_range_pos","rex_576_range_pos","rex_2016_range_pos","rex_8640_range_pos",
    "rex_144_range_width_pct","rex_576_range_width_pct","rex_2016_range_width_pct","rex_8640_range_width_pct",
]
QUANTILES = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]
OPS = ["<=", ">="]

@dataclass(frozen=True)
class Cfg:
    input_csv: str
    output: str
    family: str = "rex_htf_pullback_reclaim"
    hold_bars: int = 144
    quantile: float = 0.75
    stride_bars: int = 24
    train_start: str = "2020-01-01"
    train_end: str = "2025-01-01"
    test_start: str = "2025-01-01"
    test_end: str = "2026-01-01"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    min_train_trades: int = 80
    min_test_trades: int = 20
    top_n: int = 50


def annual(ret_pct: float, start: str, end: str) -> float:
    days=(pd.Timestamp(end)-pd.Timestamp(start)).total_seconds()/86400
    return ((1+ret_pct/100.0)**(365.25/days)-1)*100.0 if days>0 else float("nan")


def metric(res: dict[str,Any], start: str, end: str) -> dict[str,Any]:
    s=res["sim"]; st=res.get("trade_stats",{})
    c=annual(float(s.get("ret_pct",0.0) or 0.0), start, end)
    m=float(s.get("strict_mdd_pct",0.0) or 0.0)
    return {
        "ret_pct": s.get("ret_pct"),
        "full_window_cagr_pct": c,
        "strict_mdd_pct": m,
        "full_window_cagr_to_strict_mdd": c/m if m else None,
        "trade_entries": s.get("trade_entries"),
        "side_counts": s.get("side_counts"),
        "p_value_mean_ret_approx": st.get("p_value_mean_ret_approx"),
        "mean_trade_ret_pct": st.get("mean_trade_ret_pct"),
        "effect_size_d": st.get("effect_size_d"),
    }


def rank(row: dict[str,Any], min_trades: int) -> float:
    tr=int(row.get("trade_entries",0) or 0); c=float(row.get("full_window_cagr_pct",0) or 0); m=float(row.get("strict_mdd_pct",0) or 0)
    r=float(row.get("full_window_cagr_to_strict_mdd",-1e9) or -1e9)
    if tr < min_trades or c <= 0 or m <= 0 or not math.isfinite(r):
        return -1e9 + tr
    return r + 0.03*c + 0.03*math.log1p(tr)


def filter_rows(rows, feat_df, date_to_pos, rule):
    if rule is None:
        return list(rows)
    name, op, thr = rule
    vals=feat_df[name].to_numpy(float)
    out=[]
    for r in rows:
        pos=date_to_pos.get(str(r["signal_date"]))
        if pos is None: continue
        v=float(vals[pos])
        if not math.isfinite(v): continue
        if (op=="<=" and v<=thr) or (op==">=" and v>=thr):
            out.append(r)
    return out


def run(cfg: Cfg) -> dict[str,Any]:
    market=_load_market(cfg.input_csv)
    feat=build_market_feature_frame(market, window_size=144)
    dates=pd.to_datetime(market["date"])
    date_str=[str(x) for x in market["date"].tolist()]
    date_to_pos={d:i for i,d in enumerate(date_str)}
    base_cfg=EventPoolConfig(input_csv=cfg.input_csv, output="", train_start=cfg.train_start, train_end=cfg.train_end, val_start=cfg.test_start, val_end=cfg.test_end, eval_start=cfg.eval_start, eval_end=cfg.eval_end, hold_bars=cfg.hold_bars, stride_bars=cfg.stride_bars, quantile=cfg.quantile)
    strength,direction=_feature_candidates(feat)[cfg.family]
    masks={
        "train": _split_mask(dates,cfg.train_start,cfg.train_end),
        "test": _split_mask(dates,cfg.test_start,cfg.test_end),
        "eval": _split_mask(dates,cfg.eval_start,cfg.eval_end),
    }
    x=strength[masks["train"] & np.isfinite(strength) & (strength>0)]
    base_thr=float(np.quantile(x,cfg.quantile))
    base_rows={sp:_candidate_rows_for_family(market,strength,direction,family=cfg.family,threshold=base_thr,mask=mask,cfg=base_cfg) for sp,mask in masks.items()}

    rules={"LONG":[None],"SHORT":[None]}
    train_rows=base_rows["train"]
    for side in ("LONG","SHORT"):
        side_rows=[r for r in train_rows if r["side"]==side]
        idx=[date_to_pos[str(r["signal_date"])] for r in side_rows if str(r["signal_date"]) in date_to_pos]
        if len(idx)<50: continue
        for f in FEATURES:
            if f not in feat.columns: continue
            vals=feat[f].to_numpy(float)[idx]
            vals=vals[np.isfinite(vals)]
            if vals.size<50 or np.nanstd(vals)<1e-12: continue
            for q in QUANTILES:
                thr=float(np.quantile(vals,q))
                for op in OPS:
                    rules[side].append((f,op,thr))

    trials=[]
    # pre-rank side rules individually on train to avoid combinatorial blowup
    pre={}
    for side in ("LONG","SHORT"):
        scored=[]
        for rule in rules[side]:
            rows=[]
            for r in base_rows["train"]:
                if r["side"]==side:
                    rows.extend(filter_rows([r],feat,date_to_pos,rule))
            res=metric(_simulate_rows(rows,market,base_cfg),cfg.train_start,cfg.train_end)
            scored.append((rank(res, max(20, cfg.min_train_trades//4)), rule, res))
        scored.sort(key=lambda t:t[0], reverse=True)
        pre[side]=scored[:25]

    for _ls,lrule,_lr in pre["LONG"]:
        for _ss,srule,_sr in pre["SHORT"]:
            split_metrics={}
            for sp,(st,en) in {"train":(cfg.train_start,cfg.train_end),"test":(cfg.test_start,cfg.test_end),"eval":(cfg.eval_start,cfg.eval_end)}.items():
                rows=[]
                rows += filter_rows([r for r in base_rows[sp] if r["side"]=="LONG"],feat,date_to_pos,lrule)
                rows += filter_rows([r for r in base_rows[sp] if r["side"]=="SHORT"],feat,date_to_pos,srule)
                rows=sorted(rows,key=lambda r:str(r["signal_date"]))
                split_metrics[sp]=metric(_simulate_rows(rows,market,base_cfg),st,en) | {"candidate_rows":len(rows)}
            tr_rank=rank(split_metrics["train"], cfg.min_train_trades)
            te_rank=rank(split_metrics["test"], cfg.min_test_trades)
            if tr_rank<=-1e8 or te_rank<=-1e8: continue
            trials.append({"long_rule":lrule,"short_rule":srule,"score":te_rank+0.25*tr_rank,**split_metrics})
    trials.sort(key=lambda r:r["score"], reverse=True)
    report={"as_of":datetime.now(timezone.utc).isoformat(),"config":asdict(cfg),"base_threshold":base_thr,"base_event_counts":{k:len(v) for k,v in base_rows.items()},"top":trials[:cfg.top_n],"leakage_guard":{"base_threshold_fit_on_train":True,"side_rule_thresholds_fit_on_train_side_events":True,"selection_uses_train_and_test_only":True,"eval_not_used_for_selection":True}}
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-csv",required=True); p.add_argument("--output",required=True)
    p.add_argument("--family", default=Cfg.family)
    p.add_argument("--hold-bars", type=int, default=Cfg.hold_bars)
    p.add_argument("--quantile", type=float, default=Cfg.quantile)
    p.add_argument("--stride-bars", type=int, default=Cfg.stride_bars)
    p.add_argument("--train-start", default=Cfg.train_start)
    p.add_argument("--train-end", default=Cfg.train_end)
    p.add_argument("--test-start", default=Cfg.test_start)
    p.add_argument("--test-end", default=Cfg.test_end)
    p.add_argument("--eval-start", default=Cfg.eval_start)
    p.add_argument("--eval-end", default=Cfg.eval_end)
    p.add_argument("--min-train-trades", type=int, default=Cfg.min_train_trades)
    p.add_argument("--min-test-trades", type=int, default=Cfg.min_test_trades)
    p.add_argument("--top-n",type=int,default=50)
    ns=p.parse_args()
    args={
        "input_csv": ns.input_csv,
        "output": ns.output,
        "family": ns.family,
        "hold_bars": ns.hold_bars,
        "quantile": ns.quantile,
        "stride_bars": ns.stride_bars,
        "train_start": ns.train_start,
        "train_end": ns.train_end,
        "test_start": ns.test_start,
        "test_end": ns.test_end,
        "eval_start": ns.eval_start,
        "eval_end": ns.eval_end,
        "min_train_trades": ns.min_train_trades,
        "min_test_trades": ns.min_test_trades,
        "top_n": ns.top_n,
    }
    r=run(Cfg(**args))
    print(json.dumps({"base_threshold":r["base_threshold"],"base_counts":r["base_event_counts"],"top":r["top"][:10]},indent=2,ensure_ascii=False))
if __name__=="__main__": main()
