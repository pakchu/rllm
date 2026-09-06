import json,numpy as np,pandas as pd
from training import evaluate_oi_divergence_fresh as s
from training import evaluate_oi_llm_selector as ev

def test_frozen_gate_exact_and_oi_required():
 c=json.loads(s.CONFIG.read_text())['signal'];f=pd.DataFrame({g['feature']:[g['threshold']] for g in c['gates']});assert ev._candidate_active(f,c).tolist()==[True]
 assert s.DESIGN['candidate_changes'].endswith('OI availability')
 assert s.DESIGN['costs']==[0.,.0006,.001]
