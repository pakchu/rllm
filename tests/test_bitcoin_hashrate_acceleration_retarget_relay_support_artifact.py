import json
from training import build_bitcoin_hashrate_acceleration_retarget_relay_support as support
def test_terminal_source_rejection_is_hash_bound():
 p=json.loads(support.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==support.canonical_hash(core);assert p["support_passed"] is False;assert p["advance_to_gross9_novelty"] is False;assert p["postentry_return_pnl_execution_price_opened"] is False;assert p["gross9_rows_opened"] is False
def test_only_frozen_train_support_properties_fail():
 p=json.loads(support.RESULT.read_text());failed={k for k,v in p["support_checks"].items() if not v};assert failed=={"train_minimum_events","train_month_concentration"};assert p["support"]["train"]["events"]==4
