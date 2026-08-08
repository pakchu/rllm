import hashlib,json
from training import build_deribit_expansion_partial_absorption_relay_support as s
def test_depar_support_is_frozen_pass_before_novelty_and_economics():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=='d3473bbbe2ef39a7304f63da252e5709d612e293009bb14a4ca598cbb1ac9fff'
 d=json.loads(s.RESULT.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==s.chash(core)=='831a0df5fff238af162ed59be7a5fe72efcc81d318e96d7e5b78379b52f3567a'
 assert d['clock']['sha256']==hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()
 assert d['support_passed'] is True and all(d['support_checks'].values())
 assert d['advance_to_gross9_novelty'] is True and d['advance_to_economic_outcomes'] is False
