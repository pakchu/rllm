"""Recent extension of frozen legacy dollar/rally short, unchanged thresholds."""
import argparse
import json
import pandas as pd
from training import audit_legacy_dollar_rally_short as legacy
from training import evaluate_g9_macro_historical as h
from training import search_meaningful_alpha_combinations as base
from preprocessing.market_features import build_market_feature_frame

OUT=base.ROOT/'research/legacy_short_september'
WINDOWS={'recent':('2026-06-01','2026-09-05'),'since_aug4':('2026-08-04','2026-09-05'),'september_only':('2026-09-01','2026-09-05')}
VARIANTS=['original','delay_dxy_gate_1h']


def register():
    r={'version':1,'candidate':legacy.legacy_top0(),'windows':{k:list(v) for k,v in WINDOWS.items()},'variants':VARIANTS,
       'hashes':{str(p):base.sha(p) for p in [__file__,legacy.__file__,h.RAW,h.FUND,legacy.OUT/'report.json']},
       'no_tuning':True,'live_enabled':False}
    p=OUT/'design.json'
    if p.exists() and json.loads(p.read_text())!=r:raise RuntimeError('Recent legacy design drift')
    base.write_json(p,r);return r


def run():
    reg=register();m,f,_=h.merged_market();r=pd.read_pickle(h.RAW);r.date=pd.to_datetime(r.date,utc=True).dt.tz_convert(None)
    m=pd.concat([m,r[r.date>m.date.max()]],ignore_index=True).sort_values('date').reset_index(drop=True)
    if m.date.duplicated().any() or not m.date.diff().dropna().eq(pd.Timedelta('5min')).all():raise ValueError('Market grid failure')
    features=build_market_feature_frame(m,window_size=144)
    old=legacy.WINDOWS;legacy.WINDOWS=WINDOWS
    try:
        reports={w:{v:legacy.replay_window(m,f,features,w,v) for v in VARIANTS} for w in WINDOWS}
    finally:legacy.WINDOWS=old
    base.write_json(OUT/'report.json',{'registration':reg,'reports':reports,'live_enabled':False,
                                     'limits':['September1-4 only, windows overlap.','Preserves legacy global stride phase and dataset-tail signal cutoff.',
                                               'No stop loss, fixed12h hold; no live capacity/latency validation.']})
    for w in reports:
        for v in VARIANTS:print(w,v,json.dumps(reports[w][v]['metrics']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();register() if a.freeze else run()
