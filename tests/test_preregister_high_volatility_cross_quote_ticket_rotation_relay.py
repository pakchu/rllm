import json
from training import preregister_high_volatility_cross_quote_ticket_rotation_relay as prereg
def test_prereg_blind_deterministic_bound():
 a,b=prereg.build(),prereg.build();assert a==b;prereg.validate(a);assert not a["outcomes_opened"];assert not a["source_incidence_opened"];assert not a["gross9_rows_opened"];assert a["policy"]["variation_rank_min"]==.65
def test_controls_frozen(tmp_path):
 x=prereg.build();p=tmp_path/"x";p.write_text(json.dumps(x,allow_nan=False));y=json.loads(p.read_text());prereg.validate(y);assert set(y["diagnostic_controls"]["definitions"])=={"no_variation_gate","no_ticket_rotation","usdt_ticket_expansion","one_block_stale_ticket_rotation","direction_flip","same_clock_forced_long"};assert y["diagnostic_controls"]["cannot_be_promoted"]
