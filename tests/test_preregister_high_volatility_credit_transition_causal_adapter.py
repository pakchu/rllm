import json
from training import preregister_high_volatility_credit_transition_causal_adapter as p
def test_adapter_is_frozen_before_label_access():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVCQCA-24";assert x["policy"]["memory_labels"]==12;assert x["policy"]["minimum_mature_labels"]==8;assert x["features"]["maturity"]=="label exit_time<=current decision_time";assert x["research_boundary"]["transition_event_level_BTC_labels_opened"] is False;json.dumps(x,allow_nan=False)
def test_source_hash():assert p.sha256(p.SOURCE)==p.SOURCE_SHA
