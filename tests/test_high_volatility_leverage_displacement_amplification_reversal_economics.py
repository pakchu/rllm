import gzip
import pandas as pd
from training import evaluate_high_volatility_leverage_displacement_amplification_reversal_economics as e
def test_empty_diagnostic_clock_is_valid(tmp_path):
 p=tmp_path/'e.csv.gz'
 with gzip.open(p,'wt') as f:f.write('entry_time,exit_time,side\n')
 x=e.load_clock_allow_empty(p,'train',pd.Timestamp('2023-07-01T00:00:00Z'),pd.Timestamp('2024-01-01T00:00:00Z'));assert x.empty
def test_frozen_contract():
 assert e.CONTROLS==('no_variation_gate','rank_0_99','cash_displacement_dominance','one_bar_stale_amplification','direction_flip','forced_long');assert e.LEVERAGE==.5 and e.BASE_COST==.0006 and e.STRESS_COST==.001;assert 'load_clock_allow_empty' in open(e.__file__).read()
