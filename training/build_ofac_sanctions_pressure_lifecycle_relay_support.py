"""Build source-only OSPLR-24 clocks from official OFAC Recent Actions."""
from __future__ import annotations
import argparse,gzip,hashlib,html,json,math,re,time,unicodedata,urllib.parse,urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_ofac_sanctions_pressure_lifecycle_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE="/home/pakchu/rllm/.env";BASE="https://ofac.treasury.gov/recent-actions";START_DATE="2022-01-01";END_DATE="2026-07-31"
BTC_START=pd.Timestamp("2021-12-30T22:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR=Path("data/ofac_sanctions_pressure_lifecycle_relay_sources_2022_2026");CACHE=SOURCE_DIR/"raw_page_cache";RAW=SOURCE_DIR/"ofac_recent_actions_raw_pages.jsonl.gz";DAYS=SOURCE_DIR/"classified_action_days.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json"
CLOCK=Path("data/ofac_sanctions_pressure_lifecycle_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/ofac_sanctions_pressure_lifecycle_relay_controls_2023_2026");RESULT=Path("results/ofac_sanctions_pressure_lifecycle_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_volatility_gate","title_only_taxonomy","one_event_stale_side","direction_flip")
QUERY="""SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def normalize(x:Any)->str:return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",html.unescape(str(x or ""))).lower()).strip()
def contains(text:str,phrase:str)->bool:
 escaped=re.escape(phrase).replace(r"\ ",r"\s+")
 return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])",text) is not None
def classify(title:str,summary:str,title_only:bool=False)->int:
 t=prereg.build()["taxonomy"];text=normalize(title if title_only else f"{title} {summary}");pressure=any(contains(text,x) for x in t["pressure_terms"]);relief=any(contains(text,x) for x in t["relief_terms"])
 return 0 if pressure==relief else (-1 if pressure else 1)

class RecentParser(HTMLParser):
 def __init__(self):super().__init__();self.depth=0;self.current=None;self.rows=[];self.anchor=None
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=="div" and self.current is None and "search-result" in a.get("class",""):self.current={"text":[],"links":[]};self.depth=1;return
  if self.current is not None:
   if tag=="div":self.depth+=1
   if tag=="a":self.anchor={"href":a.get("href",""),"text":[]}
 def handle_data(self,data):
  if self.current is not None:self.current["text"].append(data)
  if self.anchor is not None:self.anchor["text"].append(data)
 def handle_endtag(self,tag):
  if self.current is None:return
  if tag=="a" and self.anchor is not None:self.current["links"].append({"href":self.anchor["href"],"text":normalize(" ".join(self.anchor["text"]))});self.anchor=None
  if tag=="div":
   self.depth-=1
   if self.depth==0:self.rows.append(self.current);self.current=None

def parse_page(raw:bytes)->list[dict[str,Any]]:
 p=RecentParser();p.feed(raw.decode("utf-8"));out=[]
 for row in p.rows:
  links=row["links"];action=next((x for x in links if re.fullmatch(r"/recent-actions/\d{8}(?:_\d+)?",x["href"])),None)
  if action is None:continue
  text=normalize(" ".join(row["text"]));match=re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december) \d{2}, \d{4}",text)
  if not match:raise RuntimeError("OFAC action date missing")
  category=next((x["text"] for x in links if x is not action and x["href"].startswith("/recent-actions/")),"")
  out.append({"date":pd.Timestamp(match.group(0)).tz_localize("UTC"),"title":action["text"],"summary":category,"url":"https://ofac.treasury.gov"+action["href"]})
 return out

def page_url(page:int)->str:return BASE+"?"+urllib.parse.urlencode({"ra-start-date":START_DATE,"ra-end-date":END_DATE,"page":page})
def fetch(page:int)->bytes:
 CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/f"page-{page:03d}.html"
 if path.exists():return path.read_bytes()
 req=urllib.request.Request(page_url(page),headers={"User-Agent":"rllm-research/1.0"});raw=urllib.request.urlopen(req,timeout=60).read();path.write_bytes(raw);time.sleep(.3);return raw
def download()->tuple[list[dict[str,Any]],int]:
 first=fetch(0);m=re.search(rb'aria-label="Go to Page (\d+)"',first)
 if not m:raise RuntimeError("OFAC last-page identity missing")
 pages=int(m.group(1));raws=[first]+[fetch(i) for i in range(1,pages)];records=[]
 for raw in raws:records.extend(parse_page(raw))
 if len({x["url"] for x in records})!=len(records):raise RuntimeError("OFAC duplicate action URL")
 SOURCE_DIR.mkdir(parents=True,exist_ok=True)
 with RAW.open("wb") as target:
  with gzip.GzipFile(filename="",mode="wb",fileobj=target,mtime=0,compresslevel=9) as stream:
   for raw in raws:stream.write(raw+b"\n")
 return records,pages
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def rank(values:pd.Series)->pd.Series:
 out=pd.Series(np.nan,index=values.index,dtype=float);hist=[]
 for i,v in pd.to_numeric(values,errors="coerce").items():
  prior=hist[-252:]
  if math.isfinite(v) and len(prior)>=126:a=np.asarray(prior);out.at[i]=(np.sum(a<v)+.5*np.sum(a==v))/len(a)
  if math.isfinite(v):hist.append(float(v))
 return out
def variation(records,bars):
 dates=sorted({x["date"] for x in records});f=bars.copy();f["ts"]=pd.to_datetime(f.ts,utc=True)
 for c in ("open","high","low","close"):f[c]=pd.to_numeric(f[c],errors="coerce")
 f=f.drop_duplicates("ts",keep=False).set_index("ts").sort_index();rows=[]
 for date in dates:
  decision=date+pd.Timedelta(hours=22);expected=pd.date_range(decision-pd.Timedelta(hours=24),decision,freq="1min",inclusive="left");w=f.reindex(expected);valid=len(w)==1440 and np.isfinite(w[["open","high","low","close"]]).all().all() and w[["open","high","low","close"]].gt(0).all().all();v=float(np.log(w.close).diff().dropna().pow(2).sum()) if valid else np.nan;rows.append({"date":date,"decision_time":decision,"variation_24h":v})
 out=pd.DataFrame(rows);out["variation_rank"]=rank(out.variation_24h);return out
def classified(records,var,title_only=False):
 rows=[]
 for x in records:
  side=classify(x["title"],x["summary"],title_only)
  if side:rows.append({"date":x["date"],"side":side,"url":x["url"],"title":x["title"]})
 out=[]
 for date,g in pd.DataFrame(rows).groupby("date",sort=True):
  sides=set(g.side)
  if len(sides)==1:
   urls=sorted(g.url);out.append({"date":date,"side":sides.pop(),"action_count":len(urls),"action_hash":hashlib.sha256("\n".join(urls).encode()).hexdigest(),"titles":" | ".join(sorted(g.title))})
 return pd.DataFrame(out).merge(var,on="date",validate="one_to_one")
def clock(days,control="primary"):
 f=days.sort_values("decision_time").reset_index(drop=True);side=f.side.copy()
 if control=="one_event_stale_side":side=side.shift(1)
 elif control=="direction_flip":side=-side
 active=np.isfinite(side)&(f.variation_rank.ge(.65) if control!="no_volatility_gate" else True);rows=[];reserved=None
 for i in f.index[active]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=24)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":"OSPLR-24","control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"variation_24h":float(f.at[i,"variation_24h"]),"variation_rank":float(f.at[i,"variation_rank"]),"action_count":int(f.at[i,"action_count"]),"action_hash":f.at[i,"action_hash"]})
 return pd.DataFrame(rows,columns=["candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","variation_24h","variation_rank","action_count","action_hash"])
def stats(c,split):
 s=c[c.split.eq(split)]
 if s.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(s.side.eq(1).sum());q=int(s.side.eq(-1).sum());months=pd.to_datetime(s.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(s),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(s),"max_month_share":int(months.max())/len(s)}
def run():
 from sqlalchemy import text
 records,pages=download();engine=postgres_engine()
 with engine.connect() as conn:bars=pd.read_sql_query(text(QUERY),conn,params={"start":BTC_START.to_pydatetime(),"end":END.to_pydatetime()})
 engine.dispose();var=variation(records,bars);primary_days=classified(records,var);title_days=classified(records,var,True);SOURCE_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary_days,DAYS);primary=clock(primary_days);controls={n:clock(title_days if n=="title_only_taxonomy" else primary_days,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,c in controls.items():_write_gzip_csv(c,CONTROL_DIR/f"{n}.csv.gz")
 source_core={"protocol_version":"osplr_24_source_v1","pages":pages,"records":len(records),"raw":{"path":str(RAW),"sha256":sha(RAW)},"days":{"path":str(DAYS),"sha256":sha(DAYS),"rows":len(primary_days)},"outcomes_opened":False,"gross9_rows_opened":False};source={**source_core,"manifest_hash":canonical_hash(source_core)};MANIFEST.write_text(json.dumps(source,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"osplr_24_source_support_v1","policy_id":"OSPLR-24","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":source["manifest_hash"]},"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f'{n}.csv.gz'),"sha256":sha(CONTROL_DIR/f'{n}.csv.gz'),"rows":len(c),"promotion_authorized":False} for n,c in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
