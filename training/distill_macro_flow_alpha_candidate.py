"""Distill the fixed macro-flow alpha into a disabled shadow configuration."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from training import search_meaningful_alpha_combinations as base

ROOT=base.ROOT
SELECTION=ROOT/'research/macro_flow_combinations/selection_freeze.json'
HISTORICAL=ROOT/'research/macro_flow_combinations/report.json'
FRESH=ROOT/'research/macro_flow_fresh/report.json'
CONFIG=ROOT/'configs/shadow/macro_flow_regime_switch_candidate_2026-09-06.json'
DOC=ROOT/'docs/macro-flow-regime-switch-alpha-2026-09-06.md'
HIST_NAME='mix_223_162_0.75';FRESH_NAME='dollar_flow_plus_regime_switch'


def file_hash(path):
 h=hashlib.sha256();h.update(Path(path).read_bytes());return h.hexdigest()


def build():
 selection=json.loads(SELECTION.read_text());historical=json.loads(HISTORICAL.read_text());fresh=json.loads(FRESH.read_text())
 selected=next(row for row in selection['top'] if row['name']==HIST_NAME)
 hist={window:historical['reports'][window]['0.0006'][HIST_NAME] for window in ['report2024','report2025','report2026','combined']}
 hist_stress=historical['reports']['combined']['0.001'][HIST_NAME]
 recent=fresh['reports']['0.0006'][FRESH_NAME];recent_stress=fresh['reports']['0.001'][FRESH_NAME]
 if min(selected['half_sharpes'])<=0 or min(hist['report2024']['return_pct'],hist['report2025']['return_pct'],hist['combined']['return_pct'],hist_stress['return_pct'],recent['return_pct'],recent_stress['return_pct'])<=0:
  raise RuntimeError('Candidate no longer meets distillation evidence')
 config={
  'id':'macro_flow_regime_switch_75_25_v1','status':'shadow_candidate','enabled':False,'live_authorized':False,'side':'bidirectional_net','symbol':'BTCUSDT','decision_interval':'1h','execution_delay':'next_5m_open','position_overlap_allowed':True,'long_short_offset_before_risk_and_cost':True,'net_exposure_cap':1.0,
  'components':{
   'inverse_dollar_aggressive_flow':{'weight':.75,'source':['BTCUSDT taker_buy_quote','BTCUSDT quote_asset_volume','DXY proxy with availability'],'feature':'flow6=sum(2*taker_buy_quote-quote_asset_volume,6h)/sum(quote_asset_volume,6h); dxy_change6=log(DXY).diff(6h), delayed one extra completed hour','entry':'abs(flow6)>0.02 and sign(flow6)*dxy_change6<0','direction':'sign(flow6)','sizing':'clip(0.20/(vol24*sqrt(8766)),0.10,1.0)','signal_refresh_hours':24},
   'long_regime_flow_switch':{'weight':.25,'source':['BTCUSDT close','BTCUSDT aggressive flow'],'features':['mom720=log-close change720h/(vol720*sqrt720)','z24=close deviation from 24h mean/std','flow6'],'entry':'if abs(mom720)>0.75: long only when mom720>0 and flow6>0; else long range-reversion only when z24<-1.5','direction':'long_only','sizing':'1.0','signal_refresh_hours':24}},
  'portfolio_formula':'clip(0.75*inverse_dollar_aggressive_flow + 0.25*long_regime_flow_switch,-1,1)',
  'accounting':{'notional_maintenance':'hourly target fraction; all resulting unit changes charged','base_cost_per_notional_side':.0006,'stress_cost_per_notional_side':.001,'funding':'realized; missing historical mark uses settlement 5m open proxy','mdd':'conservative five-minute high-before-low envelope'},
  'evidence':{'selection_2021_2023':selected,'historical_reports':hist,'historical_combined_stress':hist_stress,'fresh_2026_06_01_to_09_05':recent,'fresh_stress':recent_stress,'fresh_source_availability':fresh['availability'],'fresh_component_nonzero_hours':fresh['component_nonzero_hours']},
  'risks':['Historical 2024/2025/2026H1 were exposed in prior research; not clean OOS.','2026H1 return was negative before the recent recovery.','DXY/USDKRW/kimchi publication-time parity is not independently proven; recent DXY availability is partial.','No liquidation, capacity, market-impact, or tick-order model.','Candidate selected from exposed historical finalists before one-shot recent replay; remain shadow-only.'],
  'artifacts':{str(p.relative_to(ROOT)):file_hash(p) for p in [SELECTION,HISTORICAL,FRESH]},
  'implementation':'training/evaluate_macro_flow_fixed_fresh.py:fixed_positions','research_only':True}
 config['result_hash']=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
 return config


def run():
 config=build();CONFIG.parent.mkdir(parents=True,exist_ok=True);CONFIG.write_text(json.dumps(config,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 e=config['evidence'];h=e['historical_reports'];r=e['fresh_2026_06_01_to_09_05'];s=e['fresh_stress']
 DOC.write_text(f'''# Macro-flow regime-switch alpha — 2026-09-06\n\n## Decision\n\n**SHADOW CANDIDATE; live disabled.** The candidate was fixed before its recent DB replay and passed that one-shot report, but earlier historical periods were already exposed.\n\n## Formula\n\n- 75%: six-hour aggressive futures flow when its direction opposes the six-hour dollar move. Volatility-targeted and refreshed every 24 hours.\n- 25%: long-only 720-hour regime switch. Follow positive trend only with aligned flow; otherwise buy a 24-hour downside displacement in a non-trending regime.\n- Sum the signed sleeves, then cap absolute net exposure at 1x. Overlap is allowed; opposing positions offset before costs and risk.\n\n## Evidence at 6 bp/side\n\n| Window | Return | CAGR | strict MDD | CAGR/MDD | Entry episodes | Rebalance orders | Fees / initial |\n|---|---:|---:|---:|---:|---:|---:|---:|\n| 2024 | {h['report2024']['return_pct']:.2f}% | {h['report2024']['cagr_pct']:.2f}% | {h['report2024']['mdd_pct']:.2f}% | {h['report2024']['calmar']:.2f} | {h['report2024']['entry_episodes']} | {h['report2024']['rebalance_orders']} | {h['report2024']['fees_pct_initial']:.2f}% |\n| 2025 | {h['report2025']['return_pct']:.2f}% | {h['report2025']['cagr_pct']:.2f}% | {h['report2025']['mdd_pct']:.2f}% | {h['report2025']['calmar']:.2f} | {h['report2025']['entry_episodes']} | {h['report2025']['rebalance_orders']} | {h['report2025']['fees_pct_initial']:.2f}% |\n| 2026 H1 | {h['report2026']['return_pct']:.2f}% | {h['report2026']['cagr_pct']:.2f}% | {h['report2026']['mdd_pct']:.2f}% | {h['report2026']['calmar']:.2f} | {h['report2026']['entry_episodes']} | {h['report2026']['rebalance_orders']} | {h['report2026']['fees_pct_initial']:.2f}% |\n| 2024–2026 H1 | {h['combined']['return_pct']:.2f}% | {h['combined']['cagr_pct']:.2f}% | {h['combined']['mdd_pct']:.2f}% | {h['combined']['calmar']:.2f} | {h['combined']['entry_episodes']} | {h['combined']['rebalance_orders']} | {h['combined']['fees_pct_initial']:.2f}% |\n| Recent Jun–Sep 5 | {r['return_pct']:.2f}% | {r['cagr_pct']:.2f}% | {r['mdd_pct']:.2f}% | {r['calmar']:.2f} | {r['entry_episodes']} | {r['rebalance_orders']} | {r['fees_pct_initial']:.2f}% |\n\nRecent 10 bp/side stress return: **{s['return_pct']:.2f}%**. Historical combined stress return: **{e['historical_combined_stress']['return_pct']:.2f}%**.\n\n## Interpretation\n\nThe edge is strongest when dollar direction and actual aggressive crypto flow disagree, while the smaller sleeve supplies long exposure only in an established flow-confirmed regime or a non-trending downside displacement. Fixed ML candidates were tested, but formulaic mixtures were more stable and easier to audit.\n\n## Risks\n\n- Historical report periods are not pristine OOS.\n- 2026 H1 was negative before recent recovery.\n- Recent external-source availability is partial; live publication-time parity is not yet proven.\n- Keep this configuration disabled until a shadow/live-parity audit and additional forward data pass.\n''',encoding='utf-8')
 print(json.dumps({'config':str(CONFIG),'doc':str(DOC),'result_hash':config['result_hash']},indent=2))

if __name__=='__main__':run()
