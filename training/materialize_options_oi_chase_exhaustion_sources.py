"""Materialize only the completed-hour BTC feature source for OICER-12."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from preprocessing.live_db_features import postgres_url_from_env
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

START=pd.Timestamp('2023-06-20T00:00:00Z');END=pd.Timestamp('2026-08-01T00:00:00Z')
BASE_MANIFEST=Path('data/options_crowding_deleveraging_relay_sources_v4_2023_2026/manifest.json')
OUTPUT_DIR=Path('data/options_oi_chase_exhaustion_sources_2023_2026')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load_hourly(env_file:str)->pd.DataFrame:
 from sqlalchemy import create_engine,text
 e=create_engine(postgres_url_from_env(env_file),connect_args={'connect_timeout':10})
 q=text("""SELECT date_bin('1 hour',ts,TIMESTAMPTZ '1970-01-01') AS hour_start,(array_agg(open ORDER BY ts))[1] AS open,(array_agg(close ORDER BY ts DESC))[1] AS close,count(*) AS source_rows FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1""")
 with e.connect() as c:d=pd.read_sql_query(q,c,params={'start':START.to_pydatetime(),'end':END.to_pydatetime()})
 e.dispose();d['hour_start']=pd.to_datetime(d.hour_start,utc=True,format='mixed');grid=pd.DataFrame({'hour_start':pd.date_range(START,END,freq='1h',inclusive='left')});d=grid.merge(d,on='hour_start',how='left',validate='one_to_one');d['source_rows']=d.source_rows.fillna(0).astype(int);d['source_valid']=d.source_rows.eq(60)
 for c in ('open','close'):d[c]=pd.to_numeric(d[c],errors='coerce');d.loc[~d.source_valid,c]=np.nan
 finite=np.isfinite(d[['open','close']]).all(axis=1)&d[['open','close']].gt(0).all(axis=1);d['source_valid']&=finite;d.loc[~d.source_valid,['open','close']]=np.nan;d['decision_time']=d.hour_start+pd.Timedelta(hours=1);return d
def run(env_file:str)->dict:
 d=load_hourly(env_file);OUTPUT_DIR.mkdir(parents=True,exist_ok=True);out=OUTPUT_DIR/'btc_completed_hour.csv.gz';_write_gzip_csv(d,out);base=json.loads(BASE_MANIFEST.read_text());core={'protocol_version':'oicer_12_source_snapshot_v1','window':[START.isoformat(),END.isoformat()],'base_nonprice_manifest':{'path':str(BASE_MANIFEST),'sha256':sha(BASE_MANIFEST),'manifest_hash':base['manifest_hash']},'feature_price_scope':'completed [T-1h,T) hourly open/close only','post_entry_return_pnl_or_execution_price_opened':False,'candidate_incidence_opened':False,'output':{'path':str(out),'sha256':sha(out),'rows':len(d),'valid_rows':int(d.source_valid.sum())}};r={**core,'manifest_hash':chash(core)};(OUTPUT_DIR/'manifest.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n');return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--env-file',default='/home/pakchu/rllm/.env');a=p.parse_args();print(json.dumps(run(a.env_file),indent=2))
