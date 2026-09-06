"""Report all originally frozen G9-plus-new local sensitivity cells, no reselection."""
import argparse
import json
import numpy as np
import pandas as pd
from training import optimize_g9_plus_added_alphas as joint
from training import search_meaningful_alpha_combinations as base
from training import g9_joint_net_ledger as ledger

OUT=joint.OUT/'fixed_additions'


def register():
    w,labels=joint.allocation_grid();indices=[i for i,n in enumerate(labels) if i<2 or n.startswith('g9x')]
    reg={'code_hash':base.sha(__file__),'joint_code_hash':base.sha(joint.__file__),
         'original_design_hash':base.sha(joint.OUT/'design.json'),
         'original_report_hash':base.sha(joint.OUT/'report.json'),
         'cells':{labels[i]:w[i].tolist() for i in indices},
         'purpose':'Descriptive sensitivity of 27 predeclared cells, not replacement of failed June winner.',
         'live_enabled':False}
    path=OUT/'design.json'
    if path.exists() and json.loads(path.read_text())!=reg:raise RuntimeError('Sensitivity drift')
    base.write_json(path,reg);return reg


def run():
    reg=register();d,p,e,b,receipt=joint.context();labels=list(reg['cells']);w=np.array(list(reg['cells'].values()))
    reports={}
    for name,(a,z) in {'common':(joint.new.START,joint.new.END),'july_to_september':('2026-07-01',joint.new.END),'september_only':('2026-09-01',joint.new.END)}.items():
        mask=(d['date']>=pd.Timestamp(a).tz_localize(None).to_datetime64())&(d['end_date']<=pd.Timestamp(z).tz_localize(None).to_datetime64())
        ee=e[mask].copy();ee[0]=True;reports[name]={}
        for cost in [.0006,.001]:
            rows,_=ledger.simulate(base.subset(d,mask),p[mask],ee,b[mask],w,joint.NAMES,cost=cost)
            reports[name][str(cost)]=dict(zip(labels,rows))
    base.write_json(OUT/'report.json',{'registration':reg,'source_receipt':receipt,'reports':reports,'live_enabled':False,'selected_replacement':None})
    for label in labels:
        print(label,[(n,round(r['0.0006'][label]['return_pct'],3),round(r['0.0006'][label]['mdd_pct'],3)) for n,r in reports.items()])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    register() if a.freeze else run()
