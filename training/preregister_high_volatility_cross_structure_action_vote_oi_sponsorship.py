"""Outcome-blind preregistration for HVCAVOIS-8."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_cross_structure_action_vote as base
POLICY_ID='HVCAVOIS-8';DEFAULT_OUTPUT=Path('results/high_volatility_cross_structure_action_vote_oi_sponsorship_preregistration_2026-08-16.json')
BASE={'preregistration':{'path':'results/high_volatility_cross_structure_action_vote_preregistration_2026-08-16.json','sha256':'340627ccd4928acb6297f0959fd001cc07066e05e00bfde98db88ae0cb0c550e'},'support':{'path':'results/high_volatility_cross_structure_action_vote_support_2026-08-16.json','sha256':'881b16adab6c0b646cbeca6cf3341a3921b8b6792a7790828b1ad65eb85fa0df'},'gross9':{'path':'results/high_volatility_cross_structure_action_vote_gross9_novelty_2026-08-16.json','sha256':'9e9334cdefe333ff8bb02d44a404040299a66ca0da78d0237ee4b322abdcde10'},'clock':{'path':'data/high_volatility_cross_structure_action_vote_clocks_2023_2026.csv.gz','sha256':'4a30d75dbb9c0efe73f2ac929299a7413e97c8812c2b404d22f8328f26bf657d'}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_hash(v:Any):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def build():
 c=base.build();core={'protocol_version':'high_volatility_cross_structure_action_vote_oi_sponsorship_v1','policy_id':POLICY_ID,'as_of_date':'2026-08-16','exploratory_discovery':True,'fresh_confirmatory_evidence':False,'source_incidence_opened':False,'outcomes_opened':False,'gross9_rows_opened':False,'singleton':True,'candidate_family':[POLICY_ID],'candidate_family_size':1,'base_artifacts':BASE,
 'construction':{'base':'immutable HVCAV-8 clock, action vote, side, entry and hold','oi_source':'exact BTCUSDT open_interest_binance period=5m values at D-8h and D','gate':'strict log(OI_D/OI_D_minus_8h)>0; missing, duplicate, nonpositive or zero change is ineligible','additional_or_tuned_thresholds':'none','entry':'immutable D+5m','hold':'8 elapsed hours','alternatives':'none'},
 'mechanism':{'claim':'A strict cross-structure majority during high variance concentration is more durable when aggregate perpetual open interest expands over the same completed eight-hour block.','why_low_gross9_overlap_is_plausible':'the sparse conjunction of variance-concentration action consensus and exact OI sponsorship is absent from Gross9'},
 'clock':{'decision':'immutable base exact 00:00/08:00/16:00 UTC','entry':'D+5m','hold':'8 elapsed hours','funding':'not a signal; held settlements sealed until economics'},'stages':c['stages'],'source_support_gates':c['source_support_gates'],'gross9_novelty_gates':c['gross9_novelty_gates'],'economic_gates':c['economic_gates'],
 'source_plan':{'table':'open_interest_binance','symbol':'BTCUSDT','period':'5m','columns':['ts','sum_open_interest'],'read_after_preregistration':True,'postentry_prices_returns_pnl':'sealed'},
 'research_boundary':{'HVCAV_train_terminal_outcome_known':True,'HVCAV_train_cagr_to_mdd_2_82375_known':True,'exact_oi_sponsorship_incidence_or_outcomes_known':False,'source_incidence_opened':False,'postentry_return_or_pnl_opened':False,'classification':'exploratory discovery; not fresh confirmatory evidence','repair_of_prior_candidate':False,'selection_basis':'independent exact OI sponsorship applied to immutable action-vote consensus'},
 'stopping_rule':'source support, Gross9, train/test/eval/final; stop first failure; no threshold, sign, base, vote, side, clock, hold, subset, substitution, or control repair.'}
 return {**core,'manifest_hash':canonical_hash(core)}
def validate(v):
 if v['manifest_hash']!=canonical_hash({k:x for k,x in v.items() if k!='manifest_hash'}):raise RuntimeError('manifest drift')
 for a in BASE.values():
  if sha(a['path'])!=a['sha256']:raise RuntimeError('base drift')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2)+"\n");print(a.output)
