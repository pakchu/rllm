"""Freeze an outcome-blind official Cboe tail and term-volatility panel."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,json,math,urllib.request
from pathlib import Path
from typing import Any
BASE="https://cdn.cboe.com/api/global/us_indices/daily_prices";SYMBOLS=("SKEW","VVIX","VIX9D","VIX","VIX3M");OUTPUT_COLUMNS=("observation_date","SKEW_close","VVIX_close","VIX9D_close","VIX_close","VIX3M_close")
def sha_bytes(x:bytes)->str:return hashlib.sha256(x).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def fetch(symbol:str,timeout:int)->bytes:
 req=urllib.request.Request(f"{BASE}/{symbol}_History.csv",headers={"User-Agent":"rllm-cboe-volatility-surface/1.0","Accept":"text/csv"})
 with urllib.request.urlopen(req,timeout=timeout) as response:return response.read()
def parse(payload:bytes,symbol:str,start:str,end:str)->dict[str,str]:
 reader=csv.DictReader(io.StringIO(payload.decode("utf-8-sig")));fields=tuple(reader.fieldnames or ());expected=("DATE",symbol) if symbol in {"SKEW","VVIX"} else ("DATE","OPEN","HIGH","LOW","CLOSE")
 if fields!=expected:raise RuntimeError(f"Cboe {symbol} schema drift: {fields}")
 out={}
 for row in reader:
  from datetime import datetime
  day=datetime.strptime(row["DATE"],"%m/%d/%Y").date().isoformat()
  if not start<=day<end:continue
  value=float(row[symbol] if symbol in {"SKEW","VVIX"} else row["CLOSE"])
  if not math.isfinite(value) or value<=0:raise RuntimeError(f"Cboe {symbol} nonpositive value")
  if day in out:raise RuntimeError(f"Cboe {symbol} duplicate date")
  out[day]=format(value,".6f")
 if not out:raise RuntimeError(f"Cboe {symbol} empty source")
 return out
def gzip_bytes(payload:bytes)->bytes:
 target=io.BytesIO()
 with gzip.GzipFile(fileobj=target,mode="wb",filename="",mtime=0) as handle:handle.write(payload)
 return target.getvalue()
def run(output_dir:Path,start:str,end:str,timeout:int)->dict[str,Any]:
 raw={symbol:fetch(symbol,timeout) for symbol in SYMBOLS};parsed={symbol:parse(raw[symbol],symbol,start,end) for symbol in SYMBOLS};dates=sorted(set.intersection(*(set(x) for x in parsed.values())))
 if not dates or dates[-1]<"2026-07-31":raise RuntimeError("extended Cboe common coverage does not include July 2026")
 buffer=io.StringIO(newline="");writer=csv.DictWriter(buffer,fieldnames=OUTPUT_COLUMNS,lineterminator="\n");writer.writeheader()
 for day in dates:writer.writerow({"observation_date":day,**{f"{s}_close":parsed[s][day] for s in SYMBOLS}})
 payload=gzip_bytes(buffer.getvalue().encode());output_dir.mkdir(parents=True,exist_ok=True);panel=output_dir/f"cboe_volatility_surface_{start}_{dates[-1]}.csv.gz";panel.write_bytes(payload)
 core={"protocol_version":"cboe_volatility_surface_extended_v1","source":"official Cboe daily index-history CSVs","base_url":BASE,"requested_window":[start,end],"common_rows":len(dates),"common_first":dates[0],"common_last":dates[-1],"symbols":list(SYMBOLS),"raw_response_sha256":{s:sha_bytes(raw[s]) for s in SYMBOLS},"raw_rows_in_window":{s:len(parsed[s]) for s in SYMBOLS},"panel":{"path":str(panel),"sha256":sha_bytes(payload),"columns":list(OUTPUT_COLUMNS)},"availability":"daily close values; downstream entry no earlier than next Cboe source session 09:35 America/New_York","raw_responses_persisted":False,"btc_price_return_funding_or_pnl_opened":False,"outcomes_opened":False};manifest={**core,"manifest_hash":canonical_hash(core)};(output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n");return manifest
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=Path("data/cboe_volatility_surface_2021_2026"));p.add_argument("--start",default="2021-01-01");p.add_argument("--end",default="2026-08-08");p.add_argument("--timeout",type=int,default=60);a=p.parse_args();r=run(a.output_dir,a.start,a.end,a.timeout);print(json.dumps({"rows":r["common_rows"],"first":r["common_first"],"last":r["common_last"],"panel":r["panel"]},indent=2))
