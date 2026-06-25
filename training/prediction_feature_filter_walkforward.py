"""Walk-forward feature filters for an existing prediction stream.

Select simple pass conditions on validation trades and apply them to the next
period's predictions. This is for testing whether market/path features can veto
bad symbolic trades without using eval outcomes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "FEATURE_FILTER", "confidence": "HIGH"}


@dataclass(frozen=True)
class PredictionFeatureFilterCfg:
    predictions_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/prediction_feature_filter_walkforward"
    start_date: str = "2026-03-01"
    end_date: str = "2026-06-01"
    validation_months: int = 1
    feature_names: str = "sma12_ratio,usdkrw_zscore,usdkrw_momentum,kimchi_premium_zscore,trend_24,sma48_ratio,trend_12,dxy_momentum"
    quantiles: str = "0.25,0.5,0.75"
    scopes: str = "ALL,LONG,SHORT"
    min_val_trades: int = 15
    min_val_cagr_pct: float = 0.0
    max_val_mdd_pct: float = 20.0
    max_val_p_value: float = 0.8
    leverage: float = 1.0


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _months(start: str, end: str) -> list[pd.Timestamp]:
    cur = pd.Timestamp(start).replace(day=1)
    out=[]
    while cur < pd.Timestamp(end):
        out.append(cur); cur = cur + pd.offsets.MonthBegin(1)
    return out


def _slice(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    return [r for r in rows if start <= str(r.get("date", "")) < end]


def _is_trade(r: dict[str, Any]) -> bool:
    return (r.get("prediction", {}) if isinstance(r.get("prediction"), dict) else {}).get("gate") == "TRADE"


def _side(r: dict[str, Any]) -> str:
    return str((r.get("prediction", {}) if isinstance(r.get("prediction"), dict) else {}).get("side", "NONE")).upper()


def _apply(rows: list[dict[str, Any]], values: np.ndarray, feature: str, direction: str, threshold: float, scope: str) -> list[dict[str, Any]]:
    out=[]; scope=scope.upper()
    for r in rows:
        if not _is_trade(r):
            out.append(r); continue
        if scope in {"LONG","SHORT"} and _side(r) != scope:
            out.append(r); continue
        pos=int(r.get("signal_pos",-1) or -1)
        v=float(values[pos]) if 0 <= pos < len(values) else float("nan")
        ok=np.isfinite(v) and ((v <= threshold) if direction == "le" else (v >= threshold))
        if ok:
            out.append({**r, "feature_filter": {"feature": feature, "direction": direction, "threshold": threshold, "scope": scope, "value": v}})
        else:
            out.append({**r, "blocked_prediction": r.get("prediction"), "prediction": {**NO_TRADE, "reason": "feature_filter"}, "feature_filter": {"feature": feature, "direction": direction, "threshold": threshold, "scope": scope, "value": v}})
    return out


def _passes(bt: dict[str, Any], cfg: PredictionFeatureFilterCfg) -> tuple[bool, list[str]]:
    sim=bt["sim"]; stats=bt["trade_stats"]; reasons=[]
    if int(sim.get("trade_entries",0) or 0) < cfg.min_val_trades: reasons.append("trades")
    if float(sim.get("cagr_pct",-999) or -999) < cfg.min_val_cagr_pct: reasons.append("cagr")
    if float(sim.get("strict_mdd_pct",999) or 999) > cfg.max_val_mdd_pct: reasons.append("mdd")
    if float(stats.get("p_value_mean_ret_approx",1) or 1) > cfg.max_val_p_value: reasons.append("p")
    return not reasons, reasons


def _score(bt: dict[str, Any], passed: bool, reasons: list[str]) -> float:
    sim=bt["sim"]; stats=bt["trade_stats"]
    score=float(sim.get("cagr_to_strict_mdd",-999) or -999) - float(stats.get("p_value_mean_ret_approx",1) or 1) + min(1.0, float(sim.get("trade_entries",0) or 0)/100.0)
    if not passed: score -= 1000 + 10*len(reasons)
    return score


def run(cfg: PredictionFeatureFilterCfg) -> dict[str, Any]:
    preds=_read_jsonl(cfg.predictions_jsonl)
    market=_load_market(cfg.market_csv)
    feats=build_market_feature_frame(market, window_size=144).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    work=Path(cfg.work_dir); work.mkdir(parents=True, exist_ok=True)
    pred_paths=[]; reports=[]
    qs=[float(x) for x in cfg.quantiles.split(',') if x.strip()]
    features=[x.strip() for x in cfg.feature_names.split(',') if x.strip()]
    scopes=[x.strip().upper() for x in cfg.scopes.split(',') if x.strip()]
    for mstart in _months(cfg.start_date, cfg.end_date):
        mend=min(mstart+pd.offsets.MonthBegin(1), pd.Timestamp(cfg.end_date))
        vstart=mstart-pd.DateOffset(months=int(cfg.validation_months)); vend=mstart
        month=str(mstart.date())[:7]
        val_rows=_slice(preds, str(vstart.date()), str(vend.date()))
        eval_rows=_slice(preds, str(mstart.date()), str(mend.date()))
        candidates=[]
        for feature in features:
            if feature not in feats.columns: continue
            values=feats[feature].to_numpy(dtype=float)
            val_trade_pos=[int(r.get('signal_pos',-1) or -1) for r in val_rows if _is_trade(r)]
            sample=np.asarray([values[p] for p in val_trade_pos if 0 <= p < len(values) and np.isfinite(values[p])], dtype=float)
            if len(sample) < 10: continue
            for q in qs:
                thr=float(np.quantile(sample, q))
                for direction in ('le','ge'):
                    for scope in scopes:
                        tag=f"{month}_{feature}_{direction}_{str(q).replace('.','p')}_{scope}"
                        frows=_apply(val_rows, values, feature, direction, thr, scope)
                        vp=work/f"{tag}_val.jsonl"; _write_jsonl(vp,frows)
                        bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(vp), market_csv=cfg.market_csv, output=str(work/f"{tag}_val.bt.json"), leverage=cfg.leverage))
                        passed,reasons=_passes(bt,cfg)
                        candidates.append({"feature":feature,"direction":direction,"threshold":thr,"quantile":q,"scope":scope,"validation_passed":passed,"validation_reject_reasons":reasons,"score":_score(bt,passed,reasons),"val":{"sim":bt['sim'],"trade_stats":bt['trade_stats']}})
                        vp.unlink(missing_ok=True); (work/f"{tag}_val.bt.json").unlink(missing_ok=True)
        candidates.sort(key=lambda r: float(r['score']), reverse=True)
        selected=candidates[0] if candidates else None
        if selected is None or not selected.get('validation_passed'):
            out_rows=eval_rows
            status='UNFILTERED'
        else:
            values=feats[str(selected['feature'])].to_numpy(dtype=float)
            out_rows=_apply(eval_rows, values, str(selected['feature']), str(selected['direction']), float(selected['threshold']), str(selected['scope']))
            status='FILTERED'
        ep=work/f"{month}_eval_predictions.jsonl"; _write_jsonl(ep,out_rows); pred_paths.append(str(ep))
        ebt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(ep), market_csv=cfg.market_csv, output=str(work/f"{month}_eval.bt.json"), leverage=cfg.leverage))
        reports.append({"month":month,"status":status,"selected":selected,"eval":{"sim":ebt['sim'],"trade_stats":ebt['trade_stats']},"top5":candidates[:5]})
    combined=work/'combined_predictions.jsonl'
    with combined.open('w') as out:
        for p in pred_paths:
            for line in Path(p).read_text().splitlines():
                if line.strip(): out.write(line+'\n')
    bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(combined), market_csv=cfg.market_csv, output=str(work/'combined_backtest.json'), leverage=cfg.leverage))
    report={"config":asdict(cfg),"months":reports,"aggregate":{"prediction_file":str(combined),"backtest":{"sim":bt['sim'],"trade_stats":bt['trade_stats']}},"leakage_guard":{"filter_selected_on_prior_validation_only":True,"eval_month_not_used_for_filter_selection":True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True); Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)); return report


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions-jsonl',required=True); p.add_argument('--market-csv',required=True); p.add_argument('--output',required=True); p.add_argument('--work-dir',default=PredictionFeatureFilterCfg.work_dir)
    p.add_argument('--start-date',default='2026-03-01'); p.add_argument('--end-date',default='2026-06-01'); p.add_argument('--validation-months',type=int,default=1)
    p.add_argument('--feature-names',default=PredictionFeatureFilterCfg.feature_names); p.add_argument('--quantiles',default=PredictionFeatureFilterCfg.quantiles); p.add_argument('--scopes',default=PredictionFeatureFilterCfg.scopes)
    p.add_argument('--min-val-trades',type=int,default=15); p.add_argument('--min-val-cagr-pct',type=float,default=0); p.add_argument('--max-val-mdd-pct',type=float,default=20); p.add_argument('--max-val-p-value',type=float,default=0.8); p.add_argument('--leverage',type=float,default=1.0)
    return p.parse_args()


def main():
    r=run(PredictionFeatureFilterCfg(**vars(parse_args())))
    print(json.dumps({'sim':r['aggregate']['backtest']['sim'],'trade_stats':r['aggregate']['backtest']['trade_stats'],'months':[{k:m[k] for k in ('month','status','selected','eval')} for m in r['months']]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
