"""Backtest LONG/SHORT/NO_TRADE portfolio label predictions."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training.eval_portfolio_text_label import parse_portfolio_label
from training.sparse_setup_ensemble_audit import EnsembleCfg, _load_market, _simulate_events


@dataclass(frozen=True)
class BacktestPortfolioPredCfg:
    predictions_jsonl: str
    market_csv: str
    output: str
    default_hold_bars: int = 288
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    max_same_bar_signals: int = 1
    trade_stop_loss_pct: float = 0.0
    trade_take_profit_pct: float = 0.0


def _load_preds(path: str) -> list[dict[str, Any]]:
    rows=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _events(rows: list[dict[str, Any]], cfg: BacktestPortfolioPredCfg) -> list[dict[str, Any]]:
    events=[]
    for row in rows:
        pred=parse_portfolio_label(str(row.get('prediction','NO_TRADE')))
        if pred == 'NO_TRADE':
            continue
        hold=int(row.get('candidate',{}).get('hold_bars') or cfg.default_hold_bars)
        side=1 if pred == 'LONG' else -1
        events.append({
            'signal_pos': int(row['signal_pos']),
            'date': str(row.get('date')),
            'side': side,
            'horizon': hold,
            'source_horizon': hold,
            'candidate_index': 0,
            'candidate_key': f'portfolio_label|{pred}|h{hold}',
            'fold': 'eval',
            'prior_mean_ret': 0.0,
            'prior_std_ret': 1.0,
            'prior_n': 100,
        })
    return sorted(events, key=lambda e: (int(e['signal_pos']), str(e['candidate_key'])))


def run(cfg: BacktestPortfolioPredCfg) -> dict[str, Any]:
    pred_rows=_load_preds(cfg.predictions_jsonl)
    events=_events(pred_rows,cfg)
    market=_load_market(cfg.market_csv)
    sim_cfg=EnsembleCfg(
        sparse_report='', market_csv=cfg.market_csv, output=cfg.output,
        leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate,
        entry_delay_bars=cfg.entry_delay_bars, max_same_bar_signals=cfg.max_same_bar_signals,
        trade_stop_loss_pct=cfg.trade_stop_loss_pct, trade_take_profit_pct=cfg.trade_take_profit_pct,
    )
    res=_simulate_events(events, dates=pd.to_datetime(market['date']), market=market, cfg=sim_cfg)
    report={'config':cfg.__dict__,'input_rows':len(pred_rows),'events':len(events),'result':{k:v for k,v in res.items() if k!='executed'},'leakage_note':'target_echo mode is oracle only; model predictions are required for deployable validation'}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Backtest portfolio label predictions')
    p.add_argument('--predictions-jsonl',required=True)
    p.add_argument('--market-csv',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--default-hold-bars',type=int,default=288)
    p.add_argument('--leverage',type=float,default=1.0)
    p.add_argument('--fee-rate',type=float,default=0.0004)
    p.add_argument('--slippage-rate',type=float,default=0.0001)
    p.add_argument('--entry-delay-bars',type=int,default=1)
    p.add_argument('--max-same-bar-signals',type=int,default=1)
    p.add_argument('--trade-stop-loss-pct',type=float,default=0.0)
    p.add_argument('--trade-take-profit-pct',type=float,default=0.0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(BacktestPortfolioPredCfg(**vars(parse_args()))),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
