"""Outcome-blind preregistration for OICER-12."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT=Path('results/options_oi_chase_exhaustion_reversal_preregistration_2026-08-08.json')
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def build()->dict:
 core={'protocol_version':'options_oi_chase_exhaustion_reversal_v1','policy_id':'OICER-12','as_of_date':'2026-08-08','outcomes_opened':False,
 'mechanism':{'claim':'joint implied-volatility expansion, rising OI and a large funding-confirmed completed-hour BTC chase identify crowded directional inventory vulnerable to a twelve-hour reversal','side':'negative sign of the completed-hour BTC return','why_distinct_from_ocdr':'OICER requires a large signed BTC chase and only uses funding for sign concurrence; it has no extreme-funding tail','why_distinct_from_ovepr':'OICER never uses premium-index direction or efficiency and requires OI build plus completed-hour price exhaustion'},
 'clock':{'decision':'T after a completed UTC hour','volatility':'positive normalized BVOL and DVOL bodies with DVOL body strictly larger','oi':'raw-time backward-asof observations at T and T-60m, each age<=5m; positive change >= strictly-prior 720h q75 with 672 observations','price':'60 exact BTCUSDT 1m bars in [T-1h,T); return=last close/first open-1; abs return >= strictly-prior 720h q75 with 672 observations','funding':'latest nonzero event at or before T; sign must equal completed-hour return sign','trigger':'false-to-true onset, prior hour source-valid and consecutive','entry':'exact BTCUSDT T+5m open, all features available by T','side':'opposite completed-hour return','hold':'12 elapsed hours','reservation':'global half-open, exit first on equal open','no_imputation':True},
 'policy':{'prior_hours':720,'prior_min_hours':672,'oi_change_quantile':0.75,'absolute_return_quantile':0.75,'oi_asof_max_age_minutes':5,'entry_delay_minutes':5,'hold_hours':12,'leverage':0.5,'base_cost_per_notional_side':0.0006,'stress_cost_per_notional_side':0.001},
 'stages':{'train':['2023-07-01T00:00:00Z','2024-01-01T00:00:00Z'],'test':['2024-01-01T00:00:00Z','2025-01-01T00:00:00Z'],'eval':['2025-01-01T00:00:00Z','2026-01-01T00:00:00Z'],'final':['2026-01-01T00:00:00Z','2026-08-01T00:00:00Z']},
 'source_support_gates':{'minimum_events':{'train':16,'test':24,'eval':24,'final':16},'minority_side_share_min':0.20,'max_month_share':0.45},
 'novelty_gates':{'exact_entry_jaccard_max':0.10,'candidate_near_6h_share_max':0.45,'occupied_5m_jaccard_max':0.30,'absolute_signed_exposure_pearson_max':0.35,'must_pass_before_economics':True},
 'economic_gates':{'absolute_return_positive':True,'cagr_to_strict_mdd_min':3.0,'strict_mdd_max_pct':15.0,'mean_gross_underlying_min_bp':20.0,'weekly_signflip_one_sided_p_max':0.10,'stress_absolute_return_positive':True,'stress_cagr_to_strict_mdd_min':2.5,'each_calendar_half_positive':True,'stop_on_first_failure':True,'future_can_rank_repair_or_reselect':False,'accounting':'fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR'},
 'source_plan':{'vol_oi_funding':'reuse hash-bound OCDR-12C nonprice snapshot','completed_hour_price':'materialize exact Postgres bars_binance 1m hourly OHLC/count after commit; source support may open only [T-1h,T) feature bars','execution_price':'sealed until source-support and Gross9 novelty pass'},
 'research_boundary':{'ocdr_incidence_known':True,'ocdr_economic_outcomes_known':False,'oicer_candidate_incidence_opened':False,'oicer_post_entry_return_or_pnl_opened':False,'candidate_count':1,'grid':False}}
 return {**core,'manifest_hash':chash(core)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output.write_text(json.dumps(build(),indent=2,ensure_ascii=False)+'\n');print(a.output)
