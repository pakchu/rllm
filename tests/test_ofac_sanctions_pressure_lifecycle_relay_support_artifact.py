import json
from training import build_ofac_sanctions_pressure_lifecycle_relay_support as support

def test_terminal_source_rejection_is_hash_bound():
 payload=json.loads(support.RESULT.read_text());core={k:v for k,v in payload.items() if k!="manifest_hash"}
 assert payload["manifest_hash"]==support.canonical_hash(core)
 assert payload["support_passed"] is False
 assert payload["advance_to_gross9_novelty"] is False
 assert payload["postentry_return_pnl_execution_price_opened"] is False
 assert payload["gross9_rows_opened"] is False

def test_frozen_test_side_balance_is_first_failed_property():
 payload=json.loads(support.RESULT.read_text());failed={k for k,v in payload["support_checks"].items() if not v}
 assert failed=={"test_side_balance"}
 assert payload["support"]["test"]["minority_side_share"]==6/49
