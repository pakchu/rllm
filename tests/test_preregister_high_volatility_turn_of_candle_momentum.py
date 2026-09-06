import json
from training import preregister_high_volatility_turn_of_candle_momentum as prereg
def test_prereg_is_deterministic_outcome_blind_and_bound():
 a,b=prereg.build(),prereg.build();assert a==b;prereg.validate(a);assert not a["outcomes_opened"];assert not a["source_incidence_opened"];assert not a["gross9_rows_opened"];assert a["clock"]["hold"]=="30 elapsed minutes";assert a["policy"]["variation_rank_min"]==.65
def test_serialization_and_controls(tmp_path):
 x=prereg.build();p=tmp_path/"x.json";p.write_text(json.dumps(x,allow_nan=False));y=json.loads(p.read_text());prereg.validate(y);assert set(y["diagnostic_controls"]["definitions"])=={"no_variation_gate","opening_half_hour_fade","second_half_hour_momentum","one_day_stale_opening_return","same_clock_forced_long"};assert y["diagnostic_controls"]["cannot_be_promoted"]
