"""Build source-only BHARR-48 clocks from dual-replayed Bitcoin headers."""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,time,urllib.request
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_bitcoin_hashrate_acceleration_retarget_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE="/home/pakchu/rllm/.env";PROVIDERS=("https://blockstream.info/api","https://mempool.space/api");FIRST_HEIGHT=745920;END=pd.Timestamp("2026-08-01T00:00:00Z");BTC_START=pd.Timestamp("2021-11-01T00:00:00Z")
SOURCE_DIR=Path("data/bitcoin_hashrate_acceleration_retarget_relay_sources_2021_2026");CACHE=SOURCE_DIR/"transport_cache";HEADERS=SOURCE_DIR/"dual_retarget_headers.jsonl.gz";EVENTS=SOURCE_DIR/"retarget_events.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/bitcoin_hashrate_acceleration_retarget_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/bitcoin_hashrate_acceleration_retarget_relay_controls_2023_2026");RESULT=Path("results/bitcoin_hashrate_acceleration_retarget_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_volatility_gate","difficulty_level_change","one_retarget_stale_acceleration","direction_flip")
QUERY="SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def get(url:str,cache:Path)->bytes:
 CACHE.mkdir(parents=True,exist_ok=True)
 if cache.exists():return cache.read_bytes()
 req=urllib.request.Request(url,headers={"User-Agent":"rllm-research/1.0"});last=None
 for i in range(5):
  try:raw=urllib.request.urlopen(req,timeout=30).read();cache.write_bytes(raw);time.sleep(.15);return raw
  except Exception as e:last=e;time.sleep(2**i)
 raise RuntimeError(f"transport failed: {url}") from last
def provider_record(provider_index:int,height:int)->dict[str,Any]:
 base=PROVIDERS[provider_index];prefix=f"p{provider_index}-{height}";block_hash=get(f"{base}/block-height/{height}",CACHE/f"{prefix}-hash.txt").decode().strip();header=get(f"{base}/block/{block_hash}/header",CACHE/f"{prefix}-header.txt").decode().strip();meta=json.loads(get(f"{base}/block/{block_hash}",CACHE/f"{prefix}-meta.json"));
 if len(block_hash)!=64 or len(header)!=160 or int(meta["height"])!=height or meta["id"]!=block_hash:raise RuntimeError("Bitcoin transport object identity invalid")
 bits=int.from_bytes(bytes.fromhex(header)[72:76],"little");return {"height":height,"hash":block_hash,"header":header,"bits":bits,"timestamp":int(meta["timestamp"])}
def dual(height:int)->dict[str,Any]:
 a,b=provider_record(0,height),provider_record(1,height)
 if a!=b:raise RuntimeError(f"Bitcoin dual replay mismatch at {height}")
 return a
def target(bits:int)->int:
 exponent=bits>>24;mantissa=bits&0x007fffff
 if mantissa<=0 or exponent<3:raise RuntimeError("invalid compact target")
 return mantissa << (8*(exponent-3))
def tip()->int:
 values=[]
 for i,base in enumerate(PROVIDERS):values.append(int(get(f"{base}/blocks/tip/height",CACHE/f"p{i}-tip.txt")))
 if values[0]!=values[1]:raise RuntimeError("Bitcoin provider tip mismatch")
 return values[0]
def replay()->list[dict[str,Any]]:
 last=(tip()//2016)*2016;rows=[]
 for height in range(FIRST_HEIGHT,last+1,2016):
  current=dual(height);confirm=dual(height+6);rows.append({"height":height,"hash":current["hash"],"header":current["header"],"bits":current["bits"],"target":str(target(current["bits"])),"block_timestamp":current["timestamp"],"confirmation_height":height+6,"confirmation_hash":confirm["hash"],"availability_timestamp":confirm["timestamp"]})
 SOURCE_DIR.mkdir(parents=True,exist_ok=True)
 with HEADERS.open("wb") as f:
  with gzip.GzipFile(filename="",mode="wb",fileobj=f,mtime=0,compresslevel=9) as z:
   for row in rows:z.write(json.dumps(row,sort_keys=True,separators=(",",":")).encode()+b"\n")
 return rows
def rank(values:pd.Series)->pd.Series:
 out=pd.Series(np.nan,index=values.index,dtype=float);hist=[]
 for i,v in pd.to_numeric(values,errors="coerce").items():
  prior=hist[-52:]
  if math.isfinite(v) and len(prior)>=26:a=np.asarray(prior);out.at[i]=(np.sum(a<v)+.5*np.sum(a==v))/len(a)
  if math.isfinite(v):hist.append(float(v))
 return out
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def event_frame(rows,bars):
 f=bars.copy();f.ts=pd.to_datetime(f.ts,utc=True)
 for c in ("open","high","low","close"):f[c]=pd.to_numeric(f[c],errors="coerce")
 f=f.drop_duplicates("ts",keep=False).set_index("ts").sort_index();out=[]
 for j,row in enumerate(rows):
  if j<2:continue
  availability=pd.Timestamp(row["availability_timestamp"],unit="s",tz="UTC");minute=availability.floor("min");expected=pd.date_range(minute-pd.Timedelta(hours=24),minute,freq="1min",inclusive="left");w=f.reindex(expected);valid=len(w)==1440 and np.isfinite(w[["open","high","low","close"]]).all().all() and w[["open","high","low","close"]].gt(0).all().all();variation=float(np.log(w.close).diff().dropna().pow(2).sum()) if valid else np.nan;d0=-math.log(int(row["target"]));d1=-math.log(int(rows[j-1]["target"]));d2=-math.log(int(rows[j-2]["target"]));change=d0-d1;accel=change-(d1-d2);entry=availability.floor("5min")+pd.Timedelta(minutes=5);out.append({"height":row["height"],"availability_time":availability,"entry_time":entry,"difficulty_change":change,"acceleration":accel,"variation_24h":variation,"retarget_hash":row["hash"],"confirmation_hash":row["confirmation_hash"]})
 result=pd.DataFrame(out);result["variation_rank"]=rank(result.variation_24h);return result
def clock(events,control="primary"):
 f=events.copy();side_source=f.acceleration.copy()
 if control=="difficulty_level_change":side_source=f.difficulty_change
 elif control=="one_retarget_stale_acceleration":side_source=side_source.shift(1)
 side=pd.Series(np.where(side_source>0,1,-1),index=f.index);side[~np.isfinite(side_source)|side_source.eq(0)]=np.nan
 if control=="direction_flip":side=-side
 active=np.isfinite(side)&np.isfinite(f.variation_rank)&(f.variation_rank.ge(.35) if control!="no_volatility_gate" else True);rows=[];reserved=None
 for i in f.index[active]:
  entry=pd.Timestamp(f.at[i,"entry_time"]);exit_=entry+pd.Timedelta(hours=48)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":"BHARR-48","control":control,"split":split,"decision_time":f.at[i,"availability_time"],"feature_available_time":f.at[i,"availability_time"],"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"height":int(f.at[i,"height"]),"difficulty_change":float(f.at[i,"difficulty_change"]),"acceleration":float(f.at[i,"acceleration"]),"variation_24h":float(f.at[i,"variation_24h"]),"variation_rank":float(f.at[i,"variation_rank"]),"retarget_hash":f.at[i,"retarget_hash"],"confirmation_hash":f.at[i,"confirmation_hash"]})
 return pd.DataFrame(rows,columns=["candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","height","difficulty_change","acceleration","variation_24h","variation_rank","retarget_hash","confirmation_hash"])
def stats(c,split):
 s=c[c.split.eq(split)]
 if s.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(s.side.eq(1).sum());q=int(s.side.eq(-1).sum());m=pd.to_datetime(s.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(s),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(s),"max_month_share":int(m.max())/len(s)}
def run():
 from sqlalchemy import text
 rows=replay();engine=postgres_engine()
 with engine.connect() as conn:bars=pd.read_sql_query(text(QUERY),conn,params={"start":BTC_START.to_pydatetime(),"end":END.to_pydatetime()})
 engine.dispose();events=event_frame(rows,bars);SOURCE_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(events,EVENTS);primary=clock(events);controls={n:clock(events,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,c in controls.items():_write_gzip_csv(c,CONTROL_DIR/f"{n}.csv.gz")
 source_core={"protocol_version":"bharr_48_source_v1","providers":PROVIDERS,"dual_rows":len(rows),"headers":{"path":str(HEADERS),"sha256":sha(HEADERS)},"events":{"path":str(EVENTS),"sha256":sha(EVENTS),"rows":len(events)},"outcomes_opened":False,"gross9_rows_opened":False};source={**source_core,"manifest_hash":canonical_hash(source_core)};MANIFEST.write_text(json.dumps(source,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"bharr_48_source_support_v1","policy_id":"BHARR-48","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":source["manifest_hash"]},"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f'{n}.csv.gz'),"sha256":sha(CONTROL_DIR/f'{n}.csv.gz'),"rows":len(c),"promotion_authorized":False} for n,c in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
