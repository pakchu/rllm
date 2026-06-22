"""Augment event candidates with causal cross-market lead/lag features.

Uses Wave Trading 1m KRW-BTC, BTCUSDT, USDKRW/EURUSD data resampled to 5m.
Features are computed at or before the signal timestamp and are intended to test
whether richer kimchi/FX lead-lag information contains directional BTC edge.
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Cfg:
    train_candidates: str
    eval_candidates: str
    wave_data_dir: str
    train_output: str
    eval_output: str
    summary_output: str
    windows: str = "3,6,12,24,48,96,288"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)+("\n" if rows else ""))


def find_files(root: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in glob.glob(str(Path(root)/"*.csv.gz")):
        try:
            head = pd.read_csv(p, nrows=1)
            tic = str(head.get("tic", pd.Series([""])).iloc[0])
        except Exception:
            continue
        out.setdefault(tic, []).append(p)
    return out


def load_tic(files: list[str], tic: str) -> pd.DataFrame:
    frames=[]
    for p in files:
        usecols=lambda c: c in {"date","ts","tic","open","high","low","close","volume"}
        df=pd.read_csv(p, usecols=usecols)
        if "tic" in df.columns:
            df=df[df["tic"].astype(str)==tic]
        if len(df)==0:
            continue
        dt_col="date" if "date" in df.columns else "ts"
        df["date"]=pd.to_datetime(df[dt_col], utc=True, errors="coerce").dt.tz_convert(None)
        df=df.dropna(subset=["date"])
        frames.append(df[["date","open","high","low","close","volume"]])
    if not frames:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    df=pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates("date", keep="last")
    return df.reset_index(drop=True)


def resample_5m(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if len(df)==0:
        return pd.DataFrame(columns=["date", f"{prefix}_close"])
    d=df.set_index("date").sort_index()
    agg=d.resample("5min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["close"])
    agg=agg.rename(columns={c:f"{prefix}_{c}" for c in agg.columns}).reset_index()
    return agg


def ret(s: pd.Series, w: int) -> pd.Series:
    return (s/s.shift(w)-1.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)


def zscore(s: pd.Series, w: int) -> pd.Series:
    mu=s.rolling(w, min_periods=max(10,w//4)).mean()
    sd=s.rolling(w, min_periods=max(10,w//4)).std().replace(0,np.nan)
    return ((s-mu)/sd).replace([np.inf,-np.inf], np.nan).fillna(0.0)


def build_features(root: str, windows: list[int]) -> pd.DataFrame:
    files=find_files(root)
    krw=resample_5m(load_tic(files.get("KRW-BTC",[]), "KRW-BTC"), "krwbtc")
    btc=resample_5m(load_tic(files.get("BTCUSDT",[]), "BTCUSDT"), "btc")
    usdkrw=resample_5m(load_tic(files.get("USDKRW",[]), "USDKRW"), "usdkrw")
    eur=resample_5m(load_tic(files.get("EURUSD",[]), "EURUSD"), "eurusd")
    # Outer merge then forward fill exogenous markets; no future fill.
    df=btc[["date","btc_close"]].merge(krw[["date","krwbtc_close"]], on="date", how="outer")
    df=df.merge(usdkrw[["date","usdkrw_close"]], on="date", how="outer")
    if len(eur):
        df=df.merge(eur[["date","eurusd_close"]], on="date", how="outer")
    else:
        df["eurusd_close"]=np.nan
    df=df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for col in ["btc_close","krwbtc_close","usdkrw_close","eurusd_close"]:
        df[col]=df[col].ffill()
    # KRW BTC expressed in USD; premium positive means KR market rich vs Binance BTCUSDT.
    df["xmk_krwbtc_usd"] = df["krwbtc_close"] / df["usdkrw_close"]
    df["xmk_kimchi_premium"] = (df["xmk_krwbtc_usd"] / df["btc_close"] - 1.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    for w in windows:
        df[f"xmk_btc_ret_{w}"]=ret(df["btc_close"], w)
        df[f"xmk_krwbtc_usd_ret_{w}"]=ret(df["xmk_krwbtc_usd"], w)
        df[f"xmk_usdkrw_ret_{w}"]=ret(df["usdkrw_close"], w)
        df[f"xmk_eurusd_ret_{w}"]=ret(df["eurusd_close"], w)
        df[f"xmk_kimchi_change_{w}"]=df["xmk_kimchi_premium"]-df["xmk_kimchi_premium"].shift(w)
        df[f"xmk_krw_lead_gap_{w}"]=df[f"xmk_krwbtc_usd_ret_{w}"]-df[f"xmk_btc_ret_{w}"]
        df[f"xmk_riskfx_gap_{w}"]=df[f"xmk_usdkrw_ret_{w}"]-df[f"xmk_eurusd_ret_{w}"]
    for w in [288, 2016, 4032]:
        df[f"xmk_kimchi_z_{w}"]=zscore(df["xmk_kimchi_premium"], w)
        df[f"xmk_usdkrw_z_{w}"]=zscore(df["usdkrw_close"], w)
        df[f"xmk_eurusd_z_{w}"]=zscore(df["eurusd_close"], w)
    feat_cols=[c for c in df.columns if c.startswith("xmk_")]
    df[feat_cols]=df[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return df[["date", *feat_cols]]


def token_bucket(v: float) -> str:
    if v >= 0.02: return "strong_up"
    if v >= 0.005: return "up"
    if v <= -0.02: return "strong_down"
    if v <= -0.005: return "down"
    return "flat"


def augment(rows: list[dict[str, Any]], by_date: dict[str, dict[str, float]]) -> tuple[list[dict[str, Any]], int]:
    out=[]; matched=0
    for row in rows:
        vals=by_date.get(str(row.get("date")))
        r=dict(row)
        snap=dict(row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {})
        toks=dict(row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {})
        if vals:
            snap.update(vals)
            toks["xmk_krw_lead_1h"] = token_bucket(float(vals.get("xmk_krw_lead_gap_12",0.0)))
            toks["xmk_krw_lead_4h"] = token_bucket(float(vals.get("xmk_krw_lead_gap_48",0.0)))
            toks["xmk_kimchi_4h"] = token_bucket(float(vals.get("xmk_kimchi_change_48",0.0)))
            toks["xmk_usdkrw_4h"] = token_bucket(float(vals.get("xmk_usdkrw_ret_48",0.0)))
            matched += 1
        r["feature_snapshot"]=snap; r["state_tokens"]=toks
        lg=dict(row.get("leakage_guard", {}) if isinstance(row.get("leakage_guard"), dict) else {})
        lg["cross_market_features_signal_time_or_prior"] = bool(vals)
        r["leakage_guard"]=lg
        out.append(r)
    return out, matched


def run(c: Cfg) -> dict[str, Any]:
    windows=[int(x) for x in c.windows.split(',') if x.strip()]
    feats=build_features(c.wave_data_dir, windows)
    feat_cols=[x for x in feats.columns if x != "date"]
    by_date={str(rec["date"]): {k: float(rec[k]) for k in feat_cols} for rec in feats.to_dict("records")}
    train=load_jsonl(c.train_candidates); ev=load_jsonl(c.eval_candidates)
    tr, tm=augment(train, by_date); er, em=augment(ev, by_date)
    write_jsonl(c.train_output, tr); write_jsonl(c.eval_output, er)
    report={"config":c.__dict__,"feature_count_added":len(feat_cols),"feature_examples":feat_cols[:30],"feature_date_min":str(feats['date'].min()) if len(feats) else None,"feature_date_max":str(feats['date'].max()) if len(feats) else None,"train":{"rows":len(tr),"matched_rows":tm,"output":c.train_output},"eval":{"rows":len(er),"matched_rows":em,"output":c.eval_output},"leakage_guard":"cross-market features are resampled/rolling/ffill at or before signal timestamp"}
    Path(c.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(c.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> Cfg:
    p=argparse.ArgumentParser()
    p.add_argument('--train-candidates', required=True); p.add_argument('--eval-candidates', required=True); p.add_argument('--wave-data-dir', required=True)
    p.add_argument('--train-output', required=True); p.add_argument('--eval-output', required=True); p.add_argument('--summary-output', required=True); p.add_argument('--windows', default=Cfg.windows)
    return Cfg(**vars(p.parse_args()))

def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
if __name__ == '__main__': main()
