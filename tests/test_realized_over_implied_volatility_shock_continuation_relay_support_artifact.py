import hashlib,json
from training import build_realized_over_implied_volatility_shock_continuation_relay_support as s
def test_rivscr_support_is_frozen_pass_before_novelty_and_economics():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='db7a88628a6468d926de1231c955731bdf7fbc39ac82fe3edbe430a105c7634e';d=json.loads(s.RESULT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==s.chash(core)=='77cd3b8ed09bb056b68df52b153158ccb3b328641c7768aeb4f839da02410919';assert d['clock']['rows']==361 and d['support_passed'] is True and all(d['support_checks'].values()) and d['advance_to_gross9_novelty'] is True and d['advance_to_economic_outcomes'] is False
