import json
from training import preregister_high_volatility_spot_underwater_causal_sign_adapter as p
def test_adapter_is_frozen_before_label_access():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVSAUDCA-8";assert x["policy"]["memory_labels"]==12;assert x["policy"]["minimum_mature_labels"]==8;assert "exit_time<=current decision_time" in x["causal_label"]["maturity"];assert x["research_boundary"]["base_event_level_trade_returns_opened"] is False;json.dumps(x,allow_nan=False)
def test_base_artifacts_are_hash_pinned():
 for a in p.BASE.values():assert p.sha256(a["path"])==a["sha256"]
