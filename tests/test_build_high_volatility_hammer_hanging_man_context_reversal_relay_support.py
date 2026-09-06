import numpy as np
import pandas as pd
from training import build_high_volatility_hammer_hanging_man_context_reversal_relay_support as support

def frame(rows):
 return pd.DataFrame(rows,columns=["bar_open","bar_high","bar_low","bar_close"])

def prefix():
 return [(10.,11.5,9.5,11.) if i%2==0 else (11.,11.5,9.5,10.) for i in range(10)]

def test_official_hammer_context_outputs_long():
 rows=prefix()+[(10.,11.5,9.5,11.),(9.,9.55,7.5,9.5)]
 out=support.context_pattern_outputs(frame(rows),pd.Series([True]*12))
 assert not out.context_pattern_ready.iloc[10] and out.context_pattern_ready.iloc[11]
 assert out.context_pattern_output.iloc[11]==1

def test_official_hanging_man_context_outputs_short():
 rows=prefix()+[(10.,11.5,9.5,11.),(11.2,11.75,9.5,11.7)]
 out=support.context_pattern_outputs(frame(rows),pd.Series([True]*12))
 assert out.context_pattern_output.iloc[11]==-1

def test_long_lower_shadow_and_short_upper_shadow_are_strict():
 rows=prefix()+[(10.,11.5,9.5,11.),(9.,9.55,8.5,9.5)]
 out=support.context_pattern_outputs(frame(rows),pd.Series([True]*12))
 assert out.context_pattern_output.iloc[11]==0 and out.morphology_side.iloc[11]==0

def test_ambiguous_near_context_is_ignored():
 rows=prefix()
 for i in range(5,10):rows[i]=(rows[i][0],20.,0.,rows[i][3])
 rows += [(10.,11.5,9.5,11.),(10.,10.55,8.5,10.5)]
 out=support.context_pattern_outputs(frame(rows),pd.Series([True]*12))
 assert out.morphology_side.iloc[11]==1 and out.context_pattern_output.iloc[11]==0

def test_invalid_hour_resets_readiness():
 rows=prefix()+[(10.,11.5,9.5,11.),(9.,9.55,7.5,9.5)]+prefix()+[(10.,11.5,9.5,11.),(9.,9.55,7.5,9.5)]
 valid=pd.Series([True]*24);valid.iloc[11]=False;out=support.context_pattern_outputs(frame(rows),valid)
 assert not out.context_pattern_ready.iloc[12:23].any() and out.context_pattern_ready.iloc[23]

def test_prior_rank_excludes_current_and_resets():
 v=pd.Series(np.arange(725,dtype=float));ok=pd.Series([True]*725);rank=support.prior_rank(v,ok)
 assert np.isnan(rank.iloc[719]) and rank.iloc[720]==1.;ok.iloc[721]=False;assert np.isnan(support.prior_rank(v,ok).iloc[722])

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"morphology_side":[0,1,0,1,1,0],"no_context_side":[0,-1,0,1,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})

def test_controls_are_fixed_diagnostics():
 a,z,_=support.active(panel());assert a.iloc[2] and z.iloc[2]==1 and not a.iloc[4]
 assert support.active(panel(),"no_variation_gate")[0].iloc[4]
 m,ms,_=support.active(panel(),"morphology_only_lower_wick_direction");assert m.iloc[1] and ms.iloc[1]==1
 n,ns,_=support.active(panel(),"no_near_context");assert n.iloc[1] and ns.iloc[1]==-1
 stale,ss,_=support.active(panel(),"one_hour_stale_pattern");assert stale.iloc[3] and ss.iloc[3]==1
 flip,fs,_=support.active(panel(),"direction_flip");assert flip.iloc[2] and fs.iloc[2]==-1
 forced,fl,_=support.active(panel(),"forced_long");assert forced.iloc[2] and fl.iloc[2]==1
