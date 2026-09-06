import numpy as np
from training import build_high_volatility_cross_alt_sign_configuration_entropy_relay_support as s
def test_entropy_collapses_for_one_repeated_state():assert s.entropy([6]*24)==0
def test_entropy_is_two_bits_for_four_equal_states():assert s.entropy([6]*6+[4]*6+[-4]*6+[-6]*6)==2
def test_pinned_registration_and_clock():assert s.PREREG_SHA=="421932e169aebd1990eb313abc754c92225e92f1fc274058c51fdeac0b14d611" and s.P["interval_minutes"]==15
