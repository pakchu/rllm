"""Audit feature descriptors that separate strong vs weak periods for a fixed regime rule."""
from __future__ import annotations

import argparse, json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.alpha_fixed_regime_rule_backtest import FixedRegimeRuleConfig, run as run_fixed


@dataclass(frozen=True)
class DescriptorAuditConfig:
    fixed_report: str
    output: str
    input_csv: str
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    window_size: int = 144
    good_months: str = "2025-01,2025-02,2025-05,2025-07,2025-09,2025-11"
    bad_months: str = "2023-01,2023-02,2023-09,2023-10,2023-11,2024-01,2024-10"


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _month_mask(dates: pd.Series, months: list[str]) -> np.ndarray:
    ym = dates.dt.to_period("M").astype(str)
    return ym.isin(months).to_numpy()


def _summarize_feature(features: pd.DataFrame, dates: pd.Series, good_months: list[str], bad_months: list[str]) -> list[dict]:
    good_mask = _month_mask(dates, good_months)
    bad_mask = _month_mask(dates, bad_months)
    rows=[]
    for col in features.columns:
        g = features.loc[good_mask, col].replace([np.inf,-np.inf], np.nan).dropna().to_numpy(float)
        b = features.loc[bad_mask, col].replace([np.inf,-np.inf], np.nan).dropna().to_numpy(float)
        if len(g)<100 or len(b)<100: continue
        gm, bm = float(np.mean(g)), float(np.mean(b))
        gs, bs = float(np.std(g)), float(np.std(b))
        pooled = np.sqrt((gs*gs + bs*bs)/2.0) if gs+bs>0 else 0.0
        d = (gm-bm)/pooled if pooled>1e-12 else 0.0
        rows.append({
            "feature": col,
            "good_mean": gm,
            "bad_mean": bm,
            "good_median": float(np.median(g)),
            "bad_median": float(np.median(b)),
            "good_p25": float(np.percentile(g,25)),
            "good_p75": float(np.percentile(g,75)),
            "bad_p25": float(np.percentile(b,25)),
            "bad_p75": float(np.percentile(b,75)),
            "effect_d_good_minus_bad": float(d),
            "abs_effect": abs(float(d)),
        })
    return sorted(rows, key=lambda x:x["abs_effect"], reverse=True)


def _trade_month_quality(fixed_report: dict) -> dict:
    out={}
    for split in ["test","eval"]:
        months=fixed_report.get("monthly",{}).get(split,{}).get("months",{})
        out[split]=sorted([
            {"month":m, **v, "quality": float(v.get("mean_trade_ret_pct",0.0))*np.sqrt(max(1,int(v.get("n",0))))}
            for m,v in months.items()
        ], key=lambda x:x["quality"], reverse=True)
    return out


def run(cfg: DescriptorAuditConfig) -> dict:
    fixed=json.load(open(cfg.fixed_report))
    market=_load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market=attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features=build_market_feature_frame(market, window_size=cfg.window_size)
    dates=pd.to_datetime(market["date"])
    good=[x.strip() for x in cfg.good_months.split(',') if x.strip()]
    bad=[x.strip() for x in cfg.bad_months.split(',') if x.strip()]
    ranked=_summarize_feature(features, dates, good, bad)
    report={
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "fixed_report": cfg.fixed_report,
        "trade_month_quality": _trade_month_quality(fixed),
        "descriptor_candidates": ranked[:40],
        "interpretation": {
            "goal": "find past-only descriptors that separate months where the fixed Kimchi-flow rule worked from months where it failed",
            "llm_use": "convert top stable descriptor families into text regime summaries; LLM should decide whether to activate/abstain, not blindly trade every rule signal",
        }
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--fixed-report', required=True); p.add_argument('--output', required=True); p.add_argument('--input-csv', required=True)
    p.add_argument('--wave-trading-root', default=''); p.add_argument('--external-tolerance', default='30min'); p.add_argument('--window-size', type=int, default=144)
    p.add_argument('--good-months', default='2025-01,2025-02,2025-05,2025-07,2025-09,2025-11')
    p.add_argument('--bad-months', default='2023-01,2023-02,2023-09,2023-10,2023-11,2024-01,2024-10')
    return p.parse_args()


def main():
    r=run(DescriptorAuditConfig(**vars(parse_args())))
    print('top descriptors')
    for x in r['descriptor_candidates'][:15]:
        print(json.dumps(x, ensure_ascii=False))

if __name__=='__main__': main()
