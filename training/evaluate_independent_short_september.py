"""Fixed historical finalist's standalone short extension; no parameter changes."""
import argparse
import json
import numpy as np
import pandas as pd
from training import search_independent_short_candidates as search
from training import search_meaningful_alpha_combinations as base
from training import search_macro_flow_alpha_combinations as macro
from training import build_g9_september_clock_inputs as builder

OUT=base.ROOT/'research/independent_short_september'
RAW=base.ROOT/'research/g9_september_inputs/raw_enriched_cache.pkl'
FUND=base.ROOT/'research/g9_september_inputs/raw_funding_cache.csv.gz'
NAME='failed_rebound_t0.5_h24_tp0.02_sl0.015'
END='2026-09-05'


def spec():
    r=json.loads((search.OUT/'selection_freeze.json').read_text())
    return next(x['spec'] for x in r['top'] if x['spec']['name']==NAME)


def register():
    r={'version':1,'spec':spec(),'window':['2026-06-01',END],
       'selection':'none in extension; historical2024 rank2 retained for research after exposed reports',
       'hashes':{str(p):base.sha(p) for p in [__file__,search.__file__,RAW,FUND,search.OUT/'report.json']},'live_enabled':False}
    p=OUT/'design.json'
    if p.exists() and json.loads(p.read_text())!=r:raise RuntimeError('Extension drift')
    base.write_json(p,r);return r


def run():
    reg=register();m=pd.read_pickle(RAW);m['date']=pd.to_datetime(m.date,utc=True).dt.tz_convert(None)
    f=pd.read_csv(FUND);f['date']=pd.to_datetime(f.date,utc=True,format='mixed').dt.tz_convert(None)
    f=f[f.date>=m.date.min()];f,diag=builder.canonicalize_funding_aliases(f)
    if not diag['passed']:raise RuntimeError('Funding ambiguity')
    x=base.features(m,f);cols=['date','dxy','usdkrw','kimchi_premium','dxy_available','usdkrw_available','kimchi_available']
    x=pd.concat([x,macro.macro_features(m[cols],x.index)],axis=1)
    dates=pd.DatetimeIndex(m.date);op=m.open.to_numpy(float);transfer=np.zeros(len(m))
    fi=np.searchsorted(dates.to_numpy(),f.date.to_numpy(),side='right')-1;use=(fi>=0)&(fi<len(m));mark=f.loc[use,'mark_price'].to_numpy(float)
    fallback=~np.isfinite(mark)|(mark<=0);mark[fallback]=op[fi[use][fallback]]
    np.add.at(transfer,fi[use],mark*f.loc[use,'funding_rate'].to_numpy(float))
    s=spec();reports={};clocks={}
    for window,start in [('recent','2026-06-01'),('extension_since_aug4','2026-08-04'),('september_only','2026-09-01')]:
        ids=np.flatnonzero((dates>=pd.Timestamp(start))&(dates<pd.Timestamp(END)))
        if ids[-1]+1>=len(m):raise RuntimeError('No terminal open')
        d={'date':dates.to_numpy()[ids],'end_date':dates.to_numpy()[ids+1],'open':op[ids],'end':op[ids+1],
           'high':m.high.to_numpy(float)[ids],'low':m.low.to_numpy(float)[ids],'funding':transfer[ids]}
        expected=int((pd.Timestamp(END)-pd.Timestamp(start))/pd.Timedelta('5min'))
        if len(ids)!=expected or not np.all(np.diff(d['date']).astype('timedelta64[m]')==np.timedelta64(5,'m')):raise RuntimeError('Grid gap')
        labels=x.index+pd.Timedelta('5min');valid=(labels>=pd.Timestamp(start))&(labels<pd.Timestamp(END))
        xx=x.loc[valid];entries=pd.DatetimeIndex(d['date']).get_indexer(labels[valid])
        if (entries<0).any():raise RuntimeError('Signal entry absent')
        potential=search.potential_trades(d,entries,s['hold'],s['tp'],s['sl'])
        trades=search.schedule(potential,search.raw_signal(xx,s))
        reports[window]={str(c):search.exact(d,trades,c) for c in [.0006,.001]}
        clocks[window]=[{'entry_date':str(d['date'][a]),'exit_date':str(d['date'][z]) if z<len(ids) else str(d['end_date'][-1]),
                         'exit_price':float(p),'barrier':bool(b)} for a,z,p,b in zip(trades['entry'],trades['exit'],trades['exit_price'],trades['barrier'])]
    base.write_json(OUT/'report.json',{'registration':reg,'funding_receipt':diag,'reports':reports,'trades':clocks,'live_enabled':False,
                                     'notes':['Window ends Sep5 00:00UTC, not full September.','Windows restart flat independently; results overlap.','No original-parent requirement.']})
    print(json.dumps(reports,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();register() if a.freeze else run()
