"""Build source-support clocks for CVSDR-6 without post-entry outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import build_options_led_intrahour_absorption_support as intrahour
from training import preregister_cross_venue_shock_deceleration_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

CLOCK=Path('data/cross_venue_shock_deceleration_reversal_clocks_2023_2026.csv.gz');CONTROLDIR=Path('data/cross_venue_shock_deceleration_reversal_controls_2023_2026');RESULT=Path('results/cross_venue_shock_deceleration_reversal_support_2026-08-08.json');SPLITS=base.SPLITS;MIN={'train':8,'test':12,'eval':12,'final':8};CONTROLS=('no_vol_disagreement','no_first_half_tail','no_abs_oi_tail','no_deceleration_cap','direction_flip');ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=('candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','bvol_body','dvol_body','first_half_return','prior_abs_first_half_q75','second_half_return','deceleration_ratio','oi_change','prior_abs_oi_q75')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def features()->pd.DataFrame:
 j=intrahour.features().copy();j['first_q75']=j.first_half_return.abs().where(j.price_valid).shift(1).rolling(720,min_periods=672).quantile(.75);j['abs_oi_q75']=j.oi_change.abs().where(j.source_valid).shift(1).rolling(720,min_periods=672).quantile(.75);cols=['bvol_body','dvol_body','first_half_return','second_half_return','oi_change'];j['base_valid']=j.source_valid&j.price_valid&np.isfinite(j[cols]).all(axis=1)&j[['bvol_open','bvol_close','dvol_open','dvol_close','oi_current','oi_prior']].gt(0).all(axis=1);return j
def clock(j:pd.DataFrame,control:str='primary')->pd.DataFrame:
 if control=='no_vol_disagreement':vol=j.bvol_body.ne(0)&j.dvol_body.ne(0)
 else:vol=j.bvol_body.ne(0)&j.dvol_body.ne(0)&np.sign(j.bvol_body).eq(-np.sign(j.dvol_body))
 shock=j.first_half_return.ne(0)
 if control!='no_first_half_tail':shock&=j.first_q75.notna()&j.first_half_return.abs().ge(j.first_q75)
 same=j.second_half_return.ne(0)&np.sign(j.second_half_return).eq(np.sign(j.first_half_return));ratio=j.second_half_return.abs()/j.first_half_return.abs();decel=same
 if control!='no_deceleration_cap':decel&=ratio.le(.5)
 oi=j.oi_change.ne(0)
 if control!='no_abs_oi_tail':oi&=j.abs_oi_q75.notna()&j.oi_change.abs().ge(j.abs_oi_q75)
 active=j.base_valid&vol&shock&decel&oi;on=active&~active.shift(1,fill_value=False)&j.base_valid.shift(1,fill_value=False)&j.decision_time.diff().eq(pd.Timedelta(hours=1));rows=[];next_allowed=None
 for i in j.index[on]:
  decision=pd.Timestamp(j.at[i,'decision_time']);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  side=-int(np.sign(j.at[i,'first_half_return']));side=-side if control=='direction_flip' else side;next_allowed=exit_;rows.append({'candidate':'CVSDR-6','control':control,'split':split,'decision_time':decision,'feature_available_time':decision,'entry_time':entry,'exit_time':exit_,'side':side,'bvol_body':float(j.at[i,'bvol_body']),'dvol_body':float(j.at[i,'dvol_body']),'first_half_return':float(j.at[i,'first_half_return']),'prior_abs_first_half_q75':float(j.at[i,'first_q75']),'second_half_return':float(j.at[i,'second_half_return']),'deceleration_ratio':float(ratio.at[i]),'oi_change':float(j.at[i,'oi_change']),'prior_abs_oi_q75':float(j.at[i,'abs_oi_q75'])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':lo,'shorts':sh,'minority_side_share':min(lo,sh)/len(x),'max_month_share':int(m.max())/len(x)}
def run()->dict:
 j=features();primary=clock(j);controls={n:clock(j,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROLDIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROLDIR/f'{n}.csv.gz')
 st={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in st.items():checks[f'{n}_minimum_events']=x['events']>=MIN[n];checks[f'{n}_side_balance']=x['minority_side_share']>=.2;checks[f'{n}_month_concentration']=x['max_month_share']<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());sm=intrahour.PRICE_DIR/'manifest.json';passed=all(checks.values());core={'protocol_version':'cvsdr_6_source_support_v1','policy_id':'CVSDR-6','preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':sha(prereg.DEFAULT_OUTPUT),'manifest_hash':reg['manifest_hash']},'source_manifest':{'path':str(sm),'sha256':sha(sm)},'completed_preentry_feature_price_opened':True,'postentry_return_pnl_execution_price_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(primary)},'controls':{n:{'path':str(CONTROLDIR/f'{n}.csv.gz'),'sha256':sha(CONTROLDIR/f'{n}.csv.gz'),'rows':len(x)} for n,x in controls.items()},'support':st,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':ECONOMIC_OUTCOMES_AUTHORIZED,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return r
if __name__=='__main__':argparse.ArgumentParser().parse_args();r=run();print(json.dumps({'passed':r['support_passed'],'support':r['support']},indent=2))
