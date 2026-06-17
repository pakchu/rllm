"""Audit entry-context descriptors separating winning and losing trades."""
from __future__ import annotations

import argparse, json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame


@dataclass(frozen=True)
class TradeContextAuditConfig:
    fixed_report: str
    input_csv: str
    output: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    window_size: int = 144
    split: str = "eval"
    min_abs_ret_pct: float = 0.0


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _effect_rows(features: pd.DataFrame, winners: np.ndarray, losers: np.ndarray) -> list[dict]:
    rows=[]
    for col in features.columns:
        w=features.loc[winners,col].replace([np.inf,-np.inf], np.nan).dropna().to_numpy(float)
        l=features.loc[losers,col].replace([np.inf,-np.inf], np.nan).dropna().to_numpy(float)
        if len(w)<10 or len(l)<10: continue
        wm,lm=float(np.mean(w)),float(np.mean(l)); ws,ls=float(np.std(w)),float(np.std(l))
        pooled=np.sqrt((ws*ws+ls*ls)/2) if ws+ls>0 else 0.0
        d=(wm-lm)/pooled if pooled>1e-12 else 0.0
        rows.append({
            "feature": col,
            "winner_mean": wm, "loser_mean": lm,
            "winner_median": float(np.median(w)), "loser_median": float(np.median(l)),
            "winner_p25": float(np.percentile(w,25)), "winner_p75": float(np.percentile(w,75)),
            "loser_p25": float(np.percentile(l,25)), "loser_p75": float(np.percentile(l,75)),
            "effect_d_win_minus_loss": float(d), "abs_effect": abs(float(d)),
        })
    return sorted(rows, key=lambda x:x["abs_effect"], reverse=True)


def _side_effect_rows(features: pd.DataFrame, trades_df: pd.DataFrame, dates: pd.Series, side: str, min_abs_ret_pct: float) -> list[dict]:
    t=trades_df[trades_df["side"].eq(side)].copy()
    if min_abs_ret_pct>0:
        t=t[t["ret_pct"].abs()>=min_abs_ret_pct]
    win_dates=pd.to_datetime(t[t["ret_pct"]>0]["signal_date"])
    lose_dates=pd.to_datetime(t[t["ret_pct"]<=0]["signal_date"])
    date_to_idx={v:i for i,v in enumerate(dates)}
    winners=np.array([date_to_idx[x] for x in win_dates if x in date_to_idx], dtype=int)
    losers=np.array([date_to_idx[x] for x in lose_dates if x in date_to_idx], dtype=int)
    return _effect_rows(features, winners, losers) if len(winners)>=10 and len(losers)>=10 else []


def run(cfg: TradeContextAuditConfig) -> dict:
    fixed=json.load(open(cfg.fixed_report))
    trades=fixed.get("monthly",{}).get(cfg.split,{}).get("trades",[])
    if not trades:
        raise ValueError("fixed report has no full trade list; rerun alpha_fixed_regime_rule_backtest after full-trade export patch")
    market=_load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market=attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features=build_market_feature_frame(market, window_size=cfg.window_size)
    dates=pd.to_datetime(market["date"])
    df=pd.DataFrame(trades)
    if cfg.min_abs_ret_pct>0:
        df=df[df["ret_pct"].abs()>=cfg.min_abs_ret_pct]
    date_to_idx={v:i for i,v in enumerate(dates)}
    winners=np.array([date_to_idx[x] for x in pd.to_datetime(df[df.ret_pct>0].signal_date) if x in date_to_idx], dtype=int)
    losers=np.array([date_to_idx[x] for x in pd.to_datetime(df[df.ret_pct<=0].signal_date) if x in date_to_idx], dtype=int)
    report={
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "trade_counts": {"all": int(len(df)), "winners": int(len(winners)), "losers": int(len(losers)), "long": int((df.side=='LONG').sum()), "short": int((df.side=='SHORT').sum())},
        "overall_win_loss_descriptors": _effect_rows(features, winners, losers)[:40],
        "long_win_loss_descriptors": _side_effect_rows(features, df, dates, 'LONG', cfg.min_abs_ret_pct)[:30],
        "short_win_loss_descriptors": _side_effect_rows(features, df, dates, 'SHORT', cfg.min_abs_ret_pct)[:30],
        "llm_descriptor_guidance": "Prefer descriptor families that separate winners from losers at entry time and are interpretable: macro/DXY, Kimchi level/change, trend/reversion, liquidity/participation.",
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--fixed-report',required=True); p.add_argument('--input-csv',required=True); p.add_argument('--output',required=True)
    p.add_argument('--wave-trading-root',default=''); p.add_argument('--external-tolerance',default='30min'); p.add_argument('--window-size',type=int,default=144); p.add_argument('--split',default='eval'); p.add_argument('--min-abs-ret-pct',type=float,default=0.0)
    return p.parse_args()


def main():
    r=run(TradeContextAuditConfig(**vars(parse_args())))
    print('counts', r['trade_counts'])
    for name in ['overall_win_loss_descriptors','long_win_loss_descriptors','short_win_loss_descriptors']:
        print('\n'+name)
        for x in r[name][:12]: print(json.dumps(x, ensure_ascii=False))

if __name__=='__main__': main()
