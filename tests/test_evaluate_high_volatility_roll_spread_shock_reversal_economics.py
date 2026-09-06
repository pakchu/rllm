from pathlib import Path
import pandas as pd
from training import evaluate_high_volatility_roll_spread_shock_reversal_economics as e
def test_evaluator_binds_predecessors_and_empty_controls():
 assert e.POLICY_ID=='HVRSSR-8';assert e.sha256(e.PREREG)==e.PREREG_SHA;assert e.sha256(e.SUPPORT)==e.SUPPORT_SHA;assert e.sha256(e.NOVELTY)==e.NOVELTY_SHA;assert e.sha256(e.CLOCK)==e.CLOCK_SHA
 assert e.CONTROLS==('no_spread_tail_gate','no_variation_gate','all_negative_covariance','one_block_stale_geometry','direction_flip','forced_long')
 source=Path(e.__file__).read_text();assert source.index('def load_clock_allow_empty')<source.index('def evaluate_primary')
def test_empty_clock_loader(tmp_path):
 p=tmp_path/'x.csv.gz';pd.DataFrame(columns=['entry_time','exit_time','side']).to_csv(p,index=False,compression='gzip');assert e.load_clock_allow_empty(p,'train',pd.Timestamp('2023-07-01',tz='UTC'),pd.Timestamp('2024-01-01',tz='UTC')).empty
