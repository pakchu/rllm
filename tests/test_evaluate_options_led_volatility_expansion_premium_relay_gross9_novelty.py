from __future__ import annotations
import hashlib,json
import pandas as pd
from training import evaluate_options_led_volatility_expansion_premium_relay_gross9_novelty as n

def test_optimal_matching_maximizes_cardinality_then_minimizes_lag()->None:
 base=pd.Timestamp('2024-01-01T00:00:00Z')
 left=[base,base+pd.Timedelta(hours=10)];right=[base+pd.Timedelta(hours=5),base+pd.Timedelta(hours=11)]
 pairs,lag=n.optimal_near_matches(left,right)
 assert len(pairs)==2
 assert lag==6*3600

def test_frozen_gross9_novelty_replay_passes_without_outcomes(tmp_path)->None:
 out=tmp_path/'novelty.json';d=n.run(out)
 assert d['gross9_novelty_status']=='passed'
 assert d['advance_to_economic_outcomes'] is True
 assert d['every_gross9_sleeve_passed'] is True
 assert d['evidence_boundary']['btc_price_or_return_rows_opened']==0
 assert d['evidence_boundary']['economic_outcome_rows_opened']==0
 for result in d['gross9_sleeves'].values():
  assert result['passed'] is True
  assert result['metrics']['exact_entry_jaccard']==0.0

def test_canonical_novelty_artifact_hash_is_frozen()->None:
 p=n.DEFAULT_OUTPUT
 assert hashlib.sha256(p.read_bytes()).hexdigest()=='b6a5128aa259907df36b39d484f7d7bc3f142134b0d68e5cb22a0ef64ddfdd03'
 d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==n.canonical_hash(core)=='3e5dc6b7382b8979a5b66087a3a5aed63363b746643b87f14dbd49b99f66f6f4'
