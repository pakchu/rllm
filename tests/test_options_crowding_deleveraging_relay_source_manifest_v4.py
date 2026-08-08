import hashlib,json
from pathlib import Path
from training.build_options_crowding_deleveraging_relay_support_v4 import canonical_hash
P=Path('data/options_crowding_deleveraging_relay_sources_v4_2023_2026/manifest.json')
def test_source_manifest_is_hash_bound_and_price_blind():
 assert hashlib.sha256(P.read_bytes()).hexdigest()=='5a6b8fb8d2da235afee4709d219bc3e537498e15078fafa27d57ba284723cd7e'
 d=json.loads(P.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==canonical_hash(core)=='1fcd9d47cd2e6f9592edcba0e3b8db06d16805e69a13f286eade6a534ed8e180'
 assert d['btc_price_or_return_opened'] is False and d['candidate_incidence_opened'] is False
