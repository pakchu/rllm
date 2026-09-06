"""Record terminal CEVMR duplicate archive identity."""
from __future__ import annotations
import gzip,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_cftc_enforcement_volatility_momentum_relay as prereg
from training import build_cftc_enforcement_volatility_momentum_relay_support as support
PAGES=(Path("data/cftc_enforcement_volatility_momentum_relay_sources_1995_2026/raw_page_cache/page-099.html"),Path("data/cftc_enforcement_volatility_momentum_relay_sources_1995_2026/raw_page_cache/page-100.html"));RAW=Path("data/cftc_enforcement_volatility_momentum_relay_sources_1995_2026/terminal_duplicate_pages.jsonl.gz");OUTPUT=Path("results/cftc_enforcement_volatility_momentum_relay_source_failure_2026-08-09.json")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def run():
 raws=[p.read_bytes() for p in PAGES];records=[support.parse(x) for x in raws];a,b=records[0][8],records[1][1]
 if a!=b or a["action_url"]!="https://www.cftc.gov/PressRoom/PressReleases/7020-14":raise RuntimeError("CEVMR duplicate identity drift")
 RAW.parent.mkdir(parents=True,exist_ok=True)
 with RAW.open("wb") as f:
  with gzip.GzipFile(filename="",mode="wb",fileobj=f,mtime=0,compresslevel=9) as z:
   for raw in raws:z.write(raw+b"\n")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"cevmr_24_terminal_source_failure_v1","policy_id":"CEVMR-24","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"first_failed_gate":"official_archive_unique_action_identity","duplicate":{"action_url":a["action_url"],"first":{"page":99,"index":8},"second":{"page":100,"index":1},"record":a},"raw_pages":{"path":str(RAW),"sha256":sha(RAW)},"candidate_incidence_computed":False,"btc_source_rows_opened":0,"gross9_rows_opened":0,"postentry_return_or_pnl_opened":False,"advance_to_gross9_novelty":False,"advance_to_economic_outcomes":False,"decision":"terminal_source_reject_no_repair","forbidden_repairs":["deduplicate archive rows","drop one page boundary row","prefer one page","restrict archive years"]};result={**core,"manifest_hash":canonical_hash(core)};OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
