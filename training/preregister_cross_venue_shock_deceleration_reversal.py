"""Outcome-blind preregistration for CVSDR-6."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT=Path('results/cross_venue_shock_deceleration_reversal_preregistration_2026-08-08.json')
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 core={'protocol_version':'cross_venue_shock_deceleration_reversal_v1','policy_id':'CVSDR-6','as_of_date':'2026-08-08','outcomes_opened':False,
 'mechanism':{'claim':'opposite BVOL and DVOL repricing plus an extreme first-half BTC shock, extreme absolute OI repricing, and same-direction but sharply decelerating second-half flow identify exhausted impact vulnerable to reversal','side':'opposite sign of the first-half BTC return','why_distinct':'CVSDR requires same-direction deceleration, whereas CVDAR required opposite-direction absorption and CVLIR required quiet-to-breakout acceleration'},
 'clock':{'decision':'T after a completed UTC hour','volatility':'normalized BVOL and DVOL bodies have strictly opposite nonzero signs','price_source':'60 exact BTCUSDT 1m bars in [T-1h,T), split at exact minute 30','first_half_shock':'nonzero absolute first-half return >= strictly-prior 720h q75 with 672 observations','deceleration':'second-half return has the same nonzero sign and absolute magnitude <= one half of the first-half absolute return','oi':'absolute one-hour raw-time backward-asof OI change >= strictly-prior 720h q75 with 672 observations; each observation age<=5m','funding':'not a signal input; opened only for later exact PnL accounting','trigger':'false-to-true onset, prior hour source-valid and consecutive','entry':'exact BTCUSDT T+5m open, all features available by T','side':'opposite sign of first-half return','hold':'6 elapsed hours','reservation':'global half-open, exit first on equal open','no_imputation':True},
 'policy':{'prior_hours':720,'prior_min_hours':672,'first_half_absolute_return_quantile':.75,'maximum_deceleration_ratio':.5,'absolute_oi_change_quantile':.75,'oi_asof_max_age_minutes':5,'entry_delay_minutes':5,'hold_hours':6,'leverage':.5,'base_cost_per_notional_side':.0006,'stress_cost_per_notional_side':.001},
 'stages':{'train':['2023-07-01T00:00:00Z','2024-01-01T00:00:00Z'],'test':['2024-01-01T00:00:00Z','2025-01-01T00:00:00Z'],'eval':['2025-01-01T00:00:00Z','2026-01-01T00:00:00Z'],'final':['2026-01-01T00:00:00Z','2026-08-01T00:00:00Z']},
 'source_support_gates':{'minimum_events':{'train':8,'test':12,'eval':12,'final':8},'minority_side_share_min':.2,'max_month_share':.45},
 'novelty_gates':{'exact_entry_jaccard_max':.1,'candidate_near_6h_share_max':.45,'occupied_5m_jaccard_max':.3,'absolute_signed_exposure_pearson_max':.35,'must_pass_before_economics':True},
 'economic_gates':{'absolute_return_positive':True,'cagr_to_strict_mdd_min':3.,'strict_mdd_max_pct':15.,'mean_gross_underlying_min_bp':20.,'weekly_signflip_one_sided_p_max':.1,'stress_absolute_return_positive':True,'stress_cagr_to_strict_mdd_min':2.5,'each_calendar_half_positive':True,'stop_on_first_failure':True,'future_can_rank_repair_or_reselect':False,'accounting':'fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR'},
 'source_plan':{'vol_oi_funding':'reuse hash-bound OCDR-12C nonprice snapshot','intrahour_price':'reuse hash-bound OLIAH completed-hour intrahour snapshot','execution_price':'sealed until source-support and Gross9 novelty pass'},
 'research_boundary':{'prior_candidate_incidence_known':True,'prior_candidate_outcomes_used_for_cvsdr':False,'cvsdr_candidate_incidence_opened':False,'cvsdr_post_entry_return_or_pnl_opened':False,'candidate_count':1,'grid':False,'repair_of_prior_candidate':False}}
 return {**core,'manifest_hash':canonical_hash(core)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output.write_text(json.dumps(build(),indent=2,ensure_ascii=False)+'\n');print(a.output)
