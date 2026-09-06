"""Key-alias-only successor for the fixed OI divergence replay."""
from __future__ import annotations
import argparse,json,hashlib
from training import evaluate_oi_divergence_fresh as v1
from training import search_inventory_purge_reclaim_alpha as execmod
from training import search_meaningful_alpha_combinations as base
OUT=base.ROOT/'research/oi_divergence_fresh_v2';DESIGN={**v1.DESIGN,'version':2,'correction':'hold_bars_5m->hold_bars and stride_bars_5m->stride_bars aliases only'}
def register():
 f=base.ROOT/'research/oi_divergence_fresh/runtime_failure.json';p=OUT/'design.json';d={'design':DESIGN,'code_sha256':base.sha(__file__),'v1_sha256':base.sha(v1.__file__),'failure_sha256':base.sha(f)}
 if p.exists() and json.loads(p.read_text())!=d:raise RuntimeError('OI v2 drift')
 base.write_json(p,d);return d
def run():
 r=json.loads((OUT/'design.json').read_text());
 if r!=register():raise RuntimeError('Registration changed')
 payload=json.loads(v1.CONFIG.read_text());signal=payload['signal'];candidate={**signal,'hold_bars':int(signal['hold_bars_5m']),'stride_bars':int(signal['stride_bars_5m'])};market,fund,source,receipt=v1.load_context();trades,feat,cfg,raw=v1.schedule(market,fund,candidate);reports={str(cost):execmod.equity_stats(trades,start=v1.START,end=v1.EVAL_END,cfg=cfg,cost_rate=cost) for cost in DESIGN['costs']}
 result={'registration':r,'source_receipt':receipt,'oi':{'rows':len(source),'first':str(source.date.min()),'last':str(source.date.max()),'sha256':hashlib.sha256(source.to_csv(index=False).encode()).hexdigest()},'signal_candidates':len(raw),'scheduled_nonoverlap_trades':len(trades),'schedule_hash':execmod._schedule_hash(trades),'reports':reports,'live_enabled':False};base.write_json(OUT/'report.json',result);print(json.dumps(reports,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();register() if a.freeze else run()
