"""Convert decision-ranker predictions into strict overlay backtests."""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

@dataclass(frozen=True)
class DecisionBacktestConfig:
    eval_jsonl: str
    prediction_report: str
    market_csv: str
    output: str
    predictions_output: str
    policy: str = "non_abstain_full"  # non_abstain_full, take_full_only, small_half
    aggregate_duplicates: bool = False
    hold_bars: int = 12
    leverage: float = 1.0
    entry_delay_bars: int = 3
    atr_trailing_stop_mult: float = 3.75
    atr_period: int = 45

def _read(path: str) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def _scale(decision: str, policy: str) -> float:
    if policy == 'non_abstain_full':
        return 1.0 if decision in {'TAKE_FULL','TAKE_SMALL'} else 0.0
    if policy == 'take_full_only':
        return 1.0 if decision == 'TAKE_FULL' else 0.0
    if policy == 'small_half':
        if decision == 'TAKE_FULL': return 1.0
        if decision == 'TAKE_SMALL': return 0.5
        return 0.0
    raise ValueError(f'unknown policy: {policy}')

def run(cfg: DecisionBacktestConfig) -> dict[str, Any]:
    rows=_read(cfg.eval_jsonl)
    rep=json.loads(Path(cfg.prediction_report).read_text())
    preds=rep.get('predictions') or []
    raw=[]
    decision_counts={}
    severity={'ABSTAIN':0,'TAKE_SMALL':1,'TAKE_FULL':2}
    for p in preds:
        i=int(p['index']); decision=str(p['prediction']); row=rows[i]; src=row.get('source') or {}
        decision_counts[decision]=decision_counts.get(decision,0)+1
        raw.append({'date': src.get('date'), 'signal_pos': int(src['signal_pos']), 'side': str(src['side']), 'llm_decision': decision, 'source_reward': src.get('reward'), 'severity': severity.get(decision,0)})
    if cfg.aggregate_duplicates:
        grouped={}
        for r in raw:
            key=(int(r['signal_pos']), str(r['date']))
            prev=grouped.get(key)
            if prev is None or (int(r['severity']), str(r['side'])) > (int(prev['severity']), str(prev['side'])):
                grouped[key]=r
        raw=list(grouped.values())
    out=[]
    for r in raw:
        decision=str(r['llm_decision'])
        scale=_scale(decision, cfg.policy)
        pred={"confidence":"HIGH","family":"wave_llm_decision_ranker","gate":"NO_TRADE","hold_bars":0,"side":"NONE"}
        if scale>0:
            pred={"confidence":"HIGH","family":"wave_llm_decision_ranker","gate":"TRADE","hold_bars":int(cfg.hold_bars),"side":str(r['side'])}
        out.append({'date': r.get('date'), 'signal_pos': int(r['signal_pos']), 'prediction': pred, 'position_scale': scale, 'llm_decision': decision, 'source_reward': r.get('source_reward')})
    out.sort(key=lambda r:(int(r['signal_pos']), str(r['date'])))
    Path(cfg.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.predictions_output).write_text('\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out)+'\n')
    bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=cfg.predictions_output, market_csv=cfg.market_csv, output=cfg.output, leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, atr_trailing_stop_mult=cfg.atr_trailing_stop_mult, atr_period=cfg.atr_period))
    report={**bt, 'decision_backtest_config': asdict(cfg), 'decision_counts': decision_counts, 'note':'Rows are model-scored executed base candidates; skipped replacement opportunities are not generated.'}
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--eval-jsonl', required=True); ap.add_argument('--prediction-report', required=True); ap.add_argument('--market-csv', required=True); ap.add_argument('--output', required=True); ap.add_argument('--predictions-output', required=True)
    ap.add_argument('--policy', choices=['non_abstain_full','take_full_only','small_half'], default='non_abstain_full')
    ap.add_argument('--aggregate-duplicates', action='store_true')
    ap.add_argument('--hold-bars', type=int, default=12); ap.add_argument('--leverage', type=float, default=1.0); ap.add_argument('--entry-delay-bars', type=int, default=3); ap.add_argument('--atr-trailing-stop-mult', type=float, default=3.75); ap.add_argument('--atr-period', type=int, default=45)
    return ap.parse_args()

def main():
    r=run(DecisionBacktestConfig(**vars(parse_args())))
    print(json.dumps({'policy': r['decision_backtest_config']['policy'], 'sim': r['sim'], 'trade_stats': r['trade_stats'], 'decision_counts': r['decision_counts']}, indent=2, ensure_ascii=False))
if __name__=='__main__': main()
