import json
from training import preregister_high_volatility_equity_adjusted_btc_residual_reversal as prereg
def test_prereg_deterministic_blind_bound():
 a,b=prereg.build(),prereg.build();assert a==b;prereg.validate(a);assert not a["outcomes_opened"];assert not a["source_incidence_opened"];assert not a["gross9_rows_opened"];assert a["policy"]["residual_rank_min"]==.75;assert a["policy"]["variation_rank_min"]==.65
def test_roundtrip_controls(tmp_path):
 x=prereg.build();p=tmp_path/"x";p.write_text(json.dumps(x,allow_nan=False));y=json.loads(p.read_text());prereg.validate(y);assert set(y["diagnostic_controls"]["definitions"])=={"no_variation_gate","no_residual_tail","raw_btc_return_reversal","one_session_stale_beta","direction_flip","same_clock_forced_long"};assert y["diagnostic_controls"]["cannot_be_promoted"]
