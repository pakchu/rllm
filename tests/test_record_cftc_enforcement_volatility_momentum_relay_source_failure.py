import json
from training import record_cftc_enforcement_volatility_momentum_relay_source_failure as record
def test_terminal_duplicate_is_hash_bound_pre_market():
 p=json.loads(record.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==record.canonical_hash(core);assert p["first_failed_gate"]=="official_archive_unique_action_identity";assert p["duplicate"]["first"]!={};assert p["btc_source_rows_opened"]==0;assert p["gross9_rows_opened"]==0;assert p["advance_to_economic_outcomes"] is False
