"""Audit the broad alpha universe for duplicate PnL paths and provenance leaks."""
from __future__ import annotations
import argparse,hashlib,json,sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
if __package__ is None or __package__=="":sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import training.portfolio_opt_all_discovered_alpha_gross10 as broad
import training.portfolio_opt_combined_rex_new_alpha as base

TARGETS=("oi_upbit_ratio288_low","new_long_minimal_funding_premium","cand_rex_veto_7")

def digest(parts):
 h=hashlib.sha256()
 for a in parts:
  x=np.nan_to_num(np.asarray(a,dtype=np.float64),nan=0,posinf=0,neginf=0)
  h.update(np.round(x,12).tobytes())
 return h.hexdigest()
def provenance(name):
 if name.startswith("cand_calendar_"):return {"status":"contaminated","reason":"source calendar top ranking used test2024/eval2025/ytd2026"}
 if name.startswith("cand_rex_veto_"):return {"status":"contaminated","reason":"source top/tte candidate family used post-train selection evidence"}
 if name.startswith("cand_"):return {"status":"contaminated","reason":"source alpha_pool_qualifier required later-window performance"}
 if name.startswith("new_"):return {"status":"research_seen","reason":"candidate was promoted during iterative 2025/2026 research"}
 return {"status":"legacy_unknown","reason":"legacy sleeve requires its own provenance audit"}
def run(cfg):
 market,feat,masks,years,events,meta=base.build_combined_events(cfg)
 meta["prior_latest_counts"]=broad.add_qualifier_candidates(events,market,masks,cfg)
 meta["calendar_candidate_counts"]=broad.add_calendar_candidates(events,market,masks,cfg)
 meta["rex_veto_candidate_counts"]=broad.add_rex_veto_candidates(events,market,masks,cfg)
 starts,ends=base._split_starts_ends(masks); by_event=defaultdict(list)
 for e in events:by_event[(e['split'],e['sleeve'])].append(e)
 def aggregate(split,sleeve):
  st,en=starts[split],ends[split];r=np.zeros(en-st);a=np.zeros(en-st);trades=0
  for e in by_event.get((split,sleeve),[]):
   r+=e['ret'][st:en];a+=e['adv'][st:en];trades+=int(e.get('trade_count',1))
  return r,a,trades
 hashes=defaultdict(list);active_hashes=defaultdict(list);summary={}
 for sleeve in base.SLEEVES:
  ret_parts=[];adv_parts=[];act_parts=[];trades=0
  for split in ("train","test2024","eval2025","ytd2026"):
   r,a,c=aggregate(split,sleeve);trades+=c
   ret_parts.append(r);adv_parts.append(a);act_parts.append(((np.abs(r)>1e-15)|(np.abs(a)>1e-15)).astype(np.uint8))
  ph=digest([*ret_parts,*adv_parts]);ah=digest(act_parts);hashes[ph].append(sleeve);active_hashes[ah].append(sleeve)
  summary[sleeve]={"pnl_hash":ph,"active_hash":ah,"trades":trades,"provenance":provenance(sleeve)}
 def groups(d):return [v for v in d.values() if len(v)>1]
 pairwise=[]
 for i,x in enumerate(TARGETS):
  for y in TARGETS[i+1:]:
   xs=[];ys=[]
   for split in ("train","test2024","eval2025","ytd2026"):
    xr,_,_=aggregate(split,x);yr,_,_=aggregate(split,y);xs.append(xr);ys.append(yr)
   xa=np.concatenate(xs);ya=np.concatenate(ys);ax=np.abs(xa)>1e-15;ay=np.abs(ya)>1e-15;union=int((ax|ay).sum());inter=int((ax&ay).sum());both=ax&ay
   corr=float(np.corrcoef(xa[both],ya[both])[0,1]) if both.sum()>2 and np.std(xa[both])>0 and np.std(ya[both])>0 else 0.0
   pairwise.append({"a":x,"b":y,"active_jaccard":inter/union if union else 0.0,"overlap_bars":inter,"return_corr_on_overlap":corr})
 out={"as_of":datetime.now(timezone.utc).isoformat(),"config":asdict(cfg),"sleeves":len(base.SLEEVES),"events":len(events),"exact_pnl_duplicate_groups":groups(hashes),"exact_active_duplicate_groups":groups(active_hashes),"target_summaries":{x:summary[x] for x in TARGETS},"target_pairwise":pairwise,"promotion_rule":"No contaminated/research_seen sleeve may be called pristine OOS. Collapse exact PnL duplicates and family-cap near duplicates before optimization.","build_meta":meta}
 Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='results/all_discovered_alpha_universe_audit_2026-07-12.json');p.add_argument('--candidate-calendar-top-n',type=int,default=250);p.add_argument('--candidate-rex-top-n',type=int,default=50);a=p.parse_args();o=run(broad.Config(output=a.output,docs_output='/tmp/unused.md',random_samples=0,candidate_calendar_top_n=a.candidate_calendar_top_n,candidate_rex_top_n=a.candidate_rex_top_n));print(json.dumps({"output":a.output,"sleeves":o['sleeves'],"events":o['events'],"pnl_duplicate_groups":len(o['exact_pnl_duplicate_groups']),"active_duplicate_groups":len(o['exact_active_duplicate_groups']),"targets":o['target_pairwise']},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
