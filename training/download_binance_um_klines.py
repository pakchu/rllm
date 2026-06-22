"""Download Binance USD-M futures monthly kline zips and combine to CSV.GZ."""
from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

BASE="https://data.binance.vision/data/futures/um/monthly/klines"
COLS=["open_time","open","high","low","close","volume","close_time","quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"]

@dataclass(frozen=True)
class Cfg:
    symbols: str
    start_month: str
    end_month: str
    output_dir: str
    interval: str = "5m"


def months(start: str, end: str) -> list[str]:
    y,m=map(int,start.split('-')); ey,em=map(int,end.split('-'))
    out=[]
    while (y,m) <= (ey,em):
        out.append(f"{y:04d}-{m:02d}")
        m+=1
        if m==13: y+=1; m=1
    return out


def fetch_zip(symbol: str, interval: str, ym: str) -> pd.DataFrame | None:
    url=f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data=r.read()
    except Exception:
        return None
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name=z.namelist()[0]
        with z.open(name) as f:
            raw=pd.read_csv(f, header=None)
    if len(raw) and str(raw.iloc[0,0]).strip().lower() == "open_time":
        raw=raw.iloc[1:].reset_index(drop=True)
    if len(raw.columns) >= len(COLS):
        df=raw.iloc[:,:len(COLS)].copy()
        df.columns=COLS
    else:
        return None
    df["open_time"]=pd.to_numeric(df["open_time"], errors="coerce")
    df=df.dropna(subset=["open_time"])
    df["date"]=pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
    for c in ["open","high","low","close","volume","quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote"]:
        df[c]=pd.to_numeric(df[c], errors="coerce")
    df["tic"]=symbol
    df["day"]=df["date"].dt.dayofweek
    return df[["date","open","high","low","close","volume","quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","tic","day"]].dropna(subset=["date","open","high","low","close"])


def run(c: Cfg) -> dict[str, Any]:
    outdir=Path(c.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    report={"config":c.__dict__,"symbols":{}}
    for sym in [x.strip().upper() for x in c.symbols.split(',') if x.strip()]:
        frames=[]; ok=[]; missing=[]
        for ym in months(c.start_month,c.end_month):
            df=fetch_zip(sym,c.interval,ym)
            if df is None or len(df)==0:
                missing.append(ym)
            else:
                frames.append(df); ok.append(ym)
        if frames:
            full=pd.concat(frames,ignore_index=True).sort_values("date").drop_duplicates("date",keep="last")
            path=outdir/f"{sym}_{c.interval}_{c.start_month}_{c.end_month}.csv.gz"
            full.to_csv(path,index=False,compression="gzip")
            report["symbols"][sym]={"rows":int(len(full)),"date_min":str(full['date'].min()),"date_max":str(full['date'].max()),"ok_months":ok,"missing_months":missing,"output":str(path)}
        else:
            report["symbols"][sym]={"rows":0,"ok_months":ok,"missing_months":missing,"output":""}
    summary=outdir/f"download_summary_{c.interval}_{c.start_month}_{c.end_month}.json"
    summary.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def parse_args() -> Cfg:
    p=argparse.ArgumentParser(); p.add_argument('--symbols',required=True); p.add_argument('--start-month',required=True); p.add_argument('--end-month',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--interval',default=Cfg.interval)
    return Cfg(**vars(p.parse_args()))

def main(): print(json.dumps(run(parse_args()),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
