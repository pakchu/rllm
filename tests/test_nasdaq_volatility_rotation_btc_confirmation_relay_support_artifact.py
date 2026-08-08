import hashlib,json
from training import build_nasdaq_volatility_rotation_btc_confirmation_relay_support as s
def test_nvxcr_support_is_frozen_pass_before_novelty_and_economics():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='26145cb0626e9e16ecf178310c8fde42daca4c34c7af1a04d2cbaaca0522f114';d=json.loads(s.RESULT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==s.canonical_hash(core)=='f346074cc92b6e19a94547a90996fed45a7e323ab840b43e20bfcff69424d688';assert d['clock']['rows']==184 and d['support_passed'] is True and all(d['support_checks'].values()) and d['advance_to_gross9_novelty'] is True and d['advance_to_economic_outcomes'] is False
