import gzip
import pandas as pd
from training import evaluate_high_volatility_leverage_only_wick_rejection_reversal_economics as e
def test_empty_diagnostic_clock_is_valid(tmp_path):
 path=tmp_path/'empty.csv.gz'
 with gzip.open(path,'wt') as stream:stream.write('entry_time,exit_time,side\n')
 result=e.load_clock_allow_empty(path,'train',pd.Timestamp('2023-07-01T00:00:00Z'),pd.Timestamp('2024-01-01T00:00:00Z'));assert result.empty and result.columns.tolist()==['entry_time','exit_time','side'] and str(result.entry_time.dtype)=='datetime64[ns, UTC]'
def test_frozen_controls_and_strict_costs():
 assert e.CONTROLS==('no_variation_gate','rank_0_99','no_close_nonconfirmation','one_bar_stale_rejection','direction_flip','forced_long');assert e.LEVERAGE==.5 and e.BASE_COST==.0006 and e.STRESS_COST==.001;assert 'load_clock_allow_empty' in open(e.__file__).read()
