import json
from training import preregister_high_volatility_ofr_financial_stress_relay as p
def test_singleton_outcome_blind_canonical():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVOFSR-24";assert x["outcomes_opened"] is x["source_incidence_opened"] is x["gross9_rows_opened"] is False;assert x["manifest_hash"]==p.canonical_hash({k:v for k,v in x.items() if k!="manifest_hash"})
def test_frozen_stress_latency_clock_and_gates():
 x=p.build();q=x["policy"];assert (q["stress_prior_observations"],q["stress_prior_minimum"],q["stress_midrank_min"])==(252,126,.70);assert q["publication_lag_source_rows"]==2;assert x["clock"]["entry"]=="exact BTCUSDT 22:05 UTC five-minute open";assert x["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8};assert len(x["diagnostic_controls"]["names"])==6
def test_artifact_matches():
 x=json.loads(p.DEFAULT_OUTPUT.read_text());p.validate(x);assert x==p.build()
