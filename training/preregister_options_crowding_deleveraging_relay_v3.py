"""Write the raw-observation-time OCDR-12B preregistration."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from training import preregister_options_crowding_deleveraging_relay_v2 as v2

DEFAULT_OUTPUT=Path('results/options_crowding_deleveraging_relay_preregistration_v3_2026-08-08.json')
VETO=Path('results/options_crowding_deleveraging_relay_source_support_v2_veto_2026-08-08.json')
VETO_SHA='e45f993bc1fcaead3058fc0c31621cc69964c7aa14cf74f78960fb5159e43d97'

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def build()->dict:
 if sha(VETO)!=VETO_SHA:raise RuntimeError('OCDR v2 veto drift')
 prior=v2.build();core={k:v for k,v in prior.items() if k!='manifest_hash'}
 core['protocol_version']='options_crowding_deleveraging_relay_v3'
 core['policy']={**core['policy'],'policy_id':'OCDR-12B','oi_asof_max_age_minutes':5}
 core['v2_terminal_source_support_veto']={'path':str(VETO),'sha256':VETO_SHA,'candidate_incidence_opened':False,'economic_outcomes_opened':False}
 core['causal_clock']={**core['causal_clock'],'oi_change':('at T, current is the latest positive raw OI observation with ts<=T and T-ts<=5m; prior is the latest positive observation with ts<=T-60m and T-60m-ts<=5m; change=current/prior-1'),'oi_archive_availability':('raw database ts is retained without floor, round, snap or fill and is the feature availability time; current availability must be <=T and strictly before T+5m entry')}
 core['source_plan']={**core['source_plan'],'oi':('Postgres open_interest_binance BTCUSDT period=5m source=open_interest_hist; retain raw ts and values exactly, as-of with frozen 5m maximum age, never snap timestamps or impute zeros')}
 core['research_boundary']={**core['research_boundary'],'v2_offgrid_metadata_known':True,'v3_candidate_incidence_opened':False,'v3_price_or_return_rows_opened':False,'mechanism_threshold_side_hold_changed_from_v2':False}
 return {**core,'manifest_hash':v2.v1.canonical_hash(core)}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output.write_text(json.dumps(build(),indent=2,ensure_ascii=False)+'\n');print(a.output)
