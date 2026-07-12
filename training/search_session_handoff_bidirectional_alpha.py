"""Causal BTC alpha search at completed 8-hour session handoffs."""
from __future__ import annotations

import argparse, itertools, json, sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market, _split_mask
from training.search_bidirectional_state_alpha import Config, W, mk, sim


def q(frame, mask, col, level):
    x = frame.loc[mask, col].to_numpy(float); x = x[np.isfinite(x)]
    return float(np.quantile(x, level))


def features(market: pd.DataFrame) -> pd.DataFrame:
    out = build_market_feature_frame(market, window_size=144)
    close = market.close.astype(float); high = market.high.astype(float); low = market.low.astype(float)
    logret = np.log(close.replace(0, np.nan)).diff()
    quote = market.quote_asset_volume.astype(float)
    buy = market.taker_buy_quote.astype(float)
    flow = (2 * buy / quote.replace(0, np.nan) - 1).clip(-1, 1)
    n = 96  # one completed 8-hour session on 5-minute bars
    out["sh_ret"] = np.log(close / close.shift(n))
    out["sh_path_eff"] = out.sh_ret / logret.abs().rolling(n, min_periods=n).sum().replace(0, np.nan)
    out["sh_range"] = (high.rolling(n, min_periods=n).max() - low.rolling(n, min_periods=n).min()) / close
    out["sh_flow"] = flow.rolling(n, min_periods=n).mean()
    out["sh_flow_recent"] = flow.rolling(24, min_periods=24).mean()
    out["sh_flow_shift"] = out.sh_flow_recent - out.sh_flow
    out["sh_volume_ratio"] = quote.rolling(n, min_periods=n).sum() / quote.rolling(4*n, min_periods=4*n).sum().mul(.25).replace(0, np.nan)
    dt = pd.to_datetime(market.date)
    out["sh_boundary"] = ((dt.dt.minute == 0) & dt.dt.hour.isin([0, 8, 16])).astype(float).to_numpy()
    out["sh_boundary_0"] = ((dt.dt.minute == 0) & (dt.dt.hour == 0)).astype(float).to_numpy()
    out["sh_boundary_8"] = ((dt.dt.minute == 0) & (dt.dt.hour == 8)).astype(float).to_numpy()
    out["sh_boundary_16"] = ((dt.dt.minute == 0) & (dt.dt.hour == 16)).astype(float).to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


def specs(frame, train):
    rows=[]
    def f(terms): return [(c,o,q(frame,train,c,z)) if c != "sh_boundary" and not c.startswith("sh_boundary_") else (c,o,z) for c,o,z in terms]
    for boundary in ("sh_boundary", "sh_boundary_0", "sh_boundary_8", "sh_boundary_16"):
        for rq, fq, vq in itertools.product((.7,.8,.9),(.55,.65,.75),(.4,.6,.8)):
            rows.append((f"handoff_cont_{boundary}_{rq}_{fq}_{vq}",
                f([(boundary,"ge",.5),("sh_ret","ge",rq),("sh_flow","ge",fq),("sh_volume_ratio","ge",vq)]),
                f([(boundary,"ge",.5),("sh_ret","le",1-rq),("sh_flow","le",1-fq),("sh_volume_ratio","ge",vq)])))
        for rq, shiftq, rangeq in itertools.product((.1,.2,.3),(.65,.75,.85),(.6,.8)):
            rows.append((f"handoff_absorb_{boundary}_{rq}_{shiftq}_{rangeq}",
                f([(boundary,"ge",.5),("sh_ret","le",rq),("sh_flow_shift","ge",shiftq),("sh_range","ge",rangeq)]),
                f([(boundary,"ge",.5),("sh_ret","ge",1-rq),("sh_flow_shift","le",1-shiftq),("sh_range","ge",rangeq)])))
        for eq, fq in itertools.product((.8,.9),(.6,.75,.9)):
            rows.append((f"handoff_efficient_{boundary}_{eq}_{fq}",
                f([(boundary,"ge",.5),("sh_path_eff","ge",eq),("sh_flow_recent","ge",fq)]),
                f([(boundary,"ge",.5),("sh_path_eff","le",1-eq),("sh_flow_recent","le",1-fq)])))
    return rows


def run(cfg: Config):
    market=_load_market(cfg); frame=features(market); dates=pd.to_datetime(market.date); train=_split_mask(dates,*W["train"]); rows=[]
    for name,lc,sc in specs(frame,train):
        la,sa=mk(frame,lc),mk(frame,sc)
        if int((la|sa)[train].sum()) < 60: continue
        for hold,(tp,sl) in itertools.product((24,48,72,96,144),((.01,.008),(.015,.01),(.025,.015),(.04,.025))):
            score=sim(market,dates,la,sa,cfg,hold,1,tp,sl,"test2024")
            if score["trades"]>=cfg.min_test_trades and score["longs"]>=cfg.min_each_side and score["shorts"]>=cfg.min_each_side:
                rows.append({"name":name,"long_conditions":[{"feature":c,"op":o,"threshold":t} for c,o,t in lc],"short_conditions":[{"feature":c,"op":o,"threshold":t} for c,o,t in sc],"hold_bars":hold,"stride_bars":1,"tp":tp,"sl":sl,"test2024":score})
    rows.sort(key=lambda r:(r["test2024"]["ratio"],r["test2024"]["return_pct"]),reverse=True); selected=rows[:cfg.top_n]
    for row in selected:
        la=mk(frame,[(x["feature"],x["op"],x["threshold"]) for x in row["long_conditions"]]); sa=mk(frame,[(x["feature"],x["op"],x["threshold"]) for x in row["short_conditions"]])
        for split in ("train","eval2025","ytd2026"): row[split]=sim(market,dates,la,sa,cfg,row["hold_bars"],1,row["tp"],row["sl"],split)
        enough=row["eval2025"]["trades"]>=16 and row["eval2025"]["longs"]>=4 and row["eval2025"]["shorts"]>=4
        row["passes_alpha_pool"]=bool(enough and row["test2024"]["ratio"]>=2.5 and row["eval2025"]["ratio"]>=2.5)
        row["passes_live_grade"]=bool(enough and all(row[s]["ratio"]>=3 for s in ("test2024","eval2025")) and row["ytd2026"]["trades"]>=8 and row["ytd2026"]["ratio"]>=5)
    out={"as_of":datetime.now(timezone.utc).isoformat(),"config":asdict(cfg),"protocol":"completed 8h session handoff; train-frozen thresholds; test2024-only rank; sealed eval/2026; entry+1 bar; 6bp/side; strict MDD","tested":len(rows),"selected":selected,"alpha_pool_qualifiers":[r for r in selected if r["passes_alpha_pool"]],"live_grade":[r for r in selected if r["passes_live_grade"]]}
    Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))); return out


def main():
    p=argparse.ArgumentParser();p.add_argument("--input-csv",required=True);p.add_argument("--output",required=True);p.add_argument("--funding-csv",default="");p.add_argument("--premium-csv",default="");p.add_argument("--exclude-from",default="2026-06-02");a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({"tested":o["tested"],"qualifiers":len(o["alpha_pool_qualifiers"]),"live":len(o["live_grade"]),"top":o["selected"][:5]},indent=2,ensure_ascii=False,default=str))
if __name__=="__main__": main()
