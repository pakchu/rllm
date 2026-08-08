"""Write the settlement-mark-complete OCDR-12C preregistration."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from training import preregister_options_crowding_deleveraging_relay_v3 as v3

DEFAULT_OUTPUT=Path('results/options_crowding_deleveraging_relay_preregistration_v4_2026-08-08.json')
VETO=Path('results/options_crowding_deleveraging_relay_source_support_v3_veto_2026-08-08.json');VETO_SHA='468fca3defbf5d516e8619c436dc9b9611b558a923f10a629e440bbc1fcdca67'
MARKS=Path('data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz');MARKS_SHA='3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def build()->dict:
 if sha(VETO)!=VETO_SHA or sha(MARKS)!=MARKS_SHA:raise RuntimeError('OCDR v4 predecessor/source drift')
 prior=v3.build();core={k:v for k,v in prior.items() if k!='manifest_hash'};core['protocol_version']='options_crowding_deleveraging_relay_v4';core['policy']={**core['policy'],'policy_id':'OCDR-12C'}
 core['v3_terminal_source_support_veto']={'path':str(VETO),'sha256':VETO_SHA,'candidate_incidence_opened':False,'economic_outcomes_opened':False}
 core['source_plan']={**core['source_plan'],'funding':('funding rate/event time comes from Postgres funding_rates_binance; settlement mark uses positive DB mark when available, otherwise exact event-aligned data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz bound to SHA-256 3284bbb6...; no price proxy or forward fill')}
 core['economic_gates']={**core['economic_gates'],'funding_mark_precedence':['positive exact Postgres mark','exact event-aligned frozen official Binance mark'],'funding_mark_missing_action':'terminal failure'}
 core['research_boundary']={**core['research_boundary'],'v3_missing_mark_metadata_known':True,'v4_candidate_incidence_opened':False,'v4_price_or_return_rows_opened':False,'mechanism_threshold_side_hold_changed_from_v3':False}
 return {**core,'manifest_hash':v3.v2.v1.canonical_hash(core)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output.write_text(json.dumps(build(),indent=2,ensure_ascii=False)+'\n');print(a.output)
