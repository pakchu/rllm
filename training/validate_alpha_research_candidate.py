"""Independent mechanical critic for the current leak-safe alpha candidate."""
from __future__ import annotations
import argparse,json,math,re
from pathlib import Path

def validate(result_path: str, source_path: str) -> dict:
 d=json.loads(Path(result_path).read_text());src=Path(source_path).read_text();errors=[]
 q=d.get('alpha_pool_qualifiers',[])
 if not q:errors.append('no alpha_pool qualifier');r={}
 else:r=q[0]
 cfg=d.get('config',{});cost=float(cfg.get('fee_rate',-1))+float(cfg.get('slippage_rate',-1))
 if abs(cost-.0006)>1e-12:errors.append(f'cost is {cost}, expected .0006 per side')
 protocol=str(d.get('protocol','')).lower()
 requirements=(('train thresholds','train-fit'),('test-only rank','test-only selection'),('sealed eval/2026',),('strict mdd',))
 for alternatives in requirements:
  if not any(token in protocol for token in alternatives):errors.append(f'protocol missing one of {alternatives}')
 if re.search(r'shift\(\s*-\d+',src):errors.append('negative shift/future reference found')
 if re.search(r'center\s*=\s*true',src,re.I):errors.append('centered rolling window found')
 if "rows.sort(key=lambda r:(r['test2024']" not in src:errors.append('selection is not visibly test2024-only')
 if "for w in ('train','eval2025','ytd2026')" not in src:errors.append('sealed windows are not attached after selection')
 for split,min_n in (('test2024',30),('eval2025',16)):
  s=r.get(split,{})
  if not math.isfinite(float(s.get('ratio',float('nan')))):errors.append(f'{split} ratio non-finite')
  if float(s.get('ratio',-999))<2.5:errors.append(f'{split} ratio below 2.5')
  if int(s.get('trades',0))<min_n:errors.append(f'{split} trades below {min_n}')
  if int(s.get('longs',0))<4 or int(s.get('shorts',0))<4:errors.append(f'{split} lacks both directions')
  if float(s.get('strict_mdd_pct',0))<=0:errors.append(f'{split} strict MDD invalid')
 for side in ('long_conditions','short_conditions'):
  if not r.get(side):errors.append(f'{side} absent')
  for c in r.get(side,[]):
   if not math.isfinite(float(c.get('threshold',float('nan')))):errors.append(f'non-finite threshold in {side}')
 out={'status':'passed' if not errors else 'failed','passed':not errors,'result_path':result_path,'source_path':source_path,'candidate':r.get('name'),'checks':{'cost_per_side':cost,'test_ratio':r.get('test2024',{}).get('ratio'),'eval_ratio':r.get('eval2025',{}).get('ratio'),'test_trades':r.get('test2024',{}).get('trades'),'eval_trades':r.get('eval2025',{}).get('trades'),'future_shift_scan':'clean' if not re.search(r'shift\(\s*-\d+',src) else 'failed'},'errors':errors}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',default='results/jump_variation_bidirectional_alpha_scan_2026-07-12.json');p.add_argument('--source',default='training/search_jump_variation_bidirectional_alpha.py');p.add_argument('--output',default='results/jump_variation_bidirectional_alpha_validator_2026-07-12.json');a=p.parse_args();o=validate(a.result,a.source);Path(a.output).write_text(json.dumps(o,indent=2));print(json.dumps(o,indent=2));raise SystemExit(0 if o['passed'] else 1)
if __name__=='__main__':main()
