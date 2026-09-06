"""Fit the frozen pre-2023H2 CLMSRR-6 ridge model."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_confirmation_ladder_multistate_ridge_relay as prereg
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="7e09b0452fddc9310e57bfb9aa8f611da84d29d6a76556495e4c70a0255bd1f0"
PANEL=Path("data/confirmation_ladder_transaction_density_relay_sources_2023_2026/confirmation_ladders.csv.gz");PANEL_SHA="e62fb5bc98a49e819ca43d8a8b0529a901f07a1ef1d07fe9ae3beb4d5f3585e8"
DURATION=Path("data/confirmation_ladder_tempo_compression_relay_sources_2023_2026/confirmation_ladders.csv.gz");DURATION_SHA="3dbfd8d4418559814c2a711c2f0bd67429d57268a8b031a9d64c131b31956dd8"
OUTPUT=Path("results/confirmation_ladder_multistate_ridge_relay_model_freeze_2026-08-12.json");START=pd.Timestamp("2023-05-01T00:00:00Z");END=pd.Timestamp("2023-07-01T00:00:00Z")
RET=tuple(f"interval_return_{i}" for i in range(1,7));WEIGHT=tuple(f"block_weight_{i}" for i in range(1,7));TX=tuple(f"block_tx_count_{i}" for i in range(1,7));DUR=tuple(f"interval_duration_{i}" for i in range(1,7))
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def feature_frame(panel:pd.DataFrame,duration:pd.DataFrame)->pd.DataFrame:
 keys=["anchor_height","confirmation_height","feature_available_time"]
 required=set(keys+["source_valid",*RET,*WEIGHT,*TX]);required_d=set(keys+["source_valid",*DUR])
 if not required.issubset(panel) or not required_d.issubset(duration):raise RuntimeError("CLMSRR panel schema drift")
 x=panel.loc[:,keys+["source_valid",*RET,*WEIGHT,*TX]].merge(duration.loc[:,keys+["source_valid",*DUR]],on=keys,suffixes=("_panel","_duration"),validate="one_to_one")
 x["feature_available_time"]=pd.to_datetime(x.feature_available_time,utc=True,errors="raise")
 raw_ret=x.loc[:,RET].to_numpy(float);weight=x.loc[:,WEIGHT].to_numpy(float);tx=x.loc[:,TX].to_numpy(float);duration_values=x.loc[:,DUR].to_numpy(float)
 valid=x.source_valid_panel.astype(bool)&x.source_valid_duration.astype(bool)&np.isfinite(raw_ret).all(axis=1)&np.isfinite(weight).all(axis=1)&np.isfinite(tx).all(axis=1)&np.isfinite(duration_values).all(axis=1)&(weight>0).all(axis=1)&(tx>0).all(axis=1)&(duration_values>=60).all(axis=1)&(duration_values<=1800).all(axis=1)
 variation=np.sqrt(np.square(raw_ret).sum(axis=1));valid&=variation>0
 matrices=[raw_ret/variation[:,None]]
 for values in (np.log(weight),np.log(tx/weight),np.log(duration_values)):matrices.append(values-values.mean(axis=1,keepdims=True))
 matrix=np.column_stack(matrices);valid&=np.isfinite(matrix).all(axis=1)
 out=x.loc[:,keys].copy();out["source_valid"]=valid;out["variation"]=variation
 for i,name in enumerate(prereg.FEATURES):out[name]=matrix[:,i]
 return out
def fit_ridge(matrix:np.ndarray,label:np.ndarray,alpha:float)->tuple[np.ndarray,float,np.ndarray,np.ndarray,np.ndarray]:
 mean=matrix.mean(axis=0);scale=matrix.std(axis=0,ddof=0)
 if not np.isfinite(scale).all() or (scale<=0).any():raise RuntimeError("CLMSRR zero feature scale")
 z=(matrix-mean)/scale;design=np.column_stack([np.ones(len(z)),z]);penalty=np.eye(design.shape[1]);penalty[0,0]=0;coef=np.linalg.solve(design.T@design+alpha*penalty,design.T@label);pred=design@coef
 return coef[1:],float(coef[0]),mean,scale,pred
def load_opens(start:pd.Timestamp,end:pd.Timestamp)->pd.Series:
 from sqlalchemy import create_engine,text
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);db=create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
 q=text("SELECT ts,open,count(*) OVER(PARTITION BY ts) duplicate_count FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<=:end ORDER BY ts")
 try:
  with db.connect() as c:frame=pd.read_sql_query(q,c,params={"start":start.to_pydatetime(),"end":end.to_pydatetime()})
 finally:db.dispose()
 frame["ts"]=pd.to_datetime(frame.ts,utc=True);frame["open"]=pd.to_numeric(frame.open,errors="coerce")
 if frame.ts.duplicated().any() or not frame.duplicate_count.eq(1).all() or not np.isfinite(frame.open).all() or not frame.open.gt(0).all():raise RuntimeError("CLMSRR label open source invalid")
 return frame.set_index("ts").open
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA or sha(PANEL)!=PANEL_SHA or sha(DURATION)!=DURATION_SHA:raise RuntimeError("CLMSRR frozen input drift")
 features=feature_frame(pd.read_csv(PANEL,compression="gzip"),pd.read_csv(DURATION,compression="gzip"));features["entry_time"]=features.feature_available_time+pd.Timedelta("5m");features["exit_time"]=features.entry_time+pd.Timedelta("6h")
 fit=features.source_valid&features.entry_time.ge(START)&features.exit_time.lt(END);rows=features.loc[fit].copy();opens=load_opens(START,END);entry=opens.reindex(rows.entry_time).to_numpy(float);exit_open=opens.reindex(rows.exit_time).to_numpy(float)
 if not np.isfinite(entry).all() or not np.isfinite(exit_open).all():raise RuntimeError("CLMSRR incomplete training labels")
 labels=np.log(exit_open/entry);matrix=rows.loc[:,prereg.FEATURES].to_numpy(float);weights,intercept,mean,scale,pred=fit_ridge(matrix,labels,prereg.build()["policy"]["ridge_alpha"]);prediction_q75=float(np.quantile(np.abs(pred),.75,method="linear"));variation_q65=float(np.quantile(rows.variation,.65,method="linear"))
 core={"protocol_version":"clmsrr_6_model_freeze_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA},"inputs":{"panel":{"path":str(PANEL),"sha256":PANEL_SHA},"duration":{"path":str(DURATION),"sha256":DURATION_SHA}},"fit_window":[START.isoformat(),END.isoformat()],"training_rows":len(rows),"ordered_features":list(prereg.FEATURES),"standardization":{"mean":mean.tolist(),"scale":scale.tolist()},"ridge":{"alpha":100.,"intercept":intercept,"weights":weights.tolist()},"frozen_thresholds":{"absolute_prediction_q75":prediction_q75,"variation_q65":variation_q65},"pretraining_label_rows_opened":len(rows)*2,"oos_source_incidence_opened":False,"oos_outcomes_opened":False,"gross9_rows_opened":False,"fit_runs":1};result={**core,"manifest_hash":chash(core)};OUTPUT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return result
if __name__=="__main__":print(json.dumps({"output":str(OUTPUT),"training_rows":run()["training_rows"]}))
