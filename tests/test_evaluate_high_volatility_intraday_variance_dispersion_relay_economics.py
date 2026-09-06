from pathlib import Path
import pandas as pd
from training import evaluate_high_volatility_intraday_variance_dispersion_relay_economics as e

def test_bound_and_empty_safe():
 assert e.POLICY_ID=='HVIVDR-8' and e.sha256(e.PREREG)==e.PREREG_SHA and e.sha256(e.SUPPORT)==e.SUPPORT_SHA and e.sha256(e.NOVELTY)==e.NOVELTY_SHA and e.sha256(e.CLOCK)==e.CLOCK_SHA
 assert e.CONTROLS==('no_dispersion_tail','no_variation_gate','one_block_stale_geometry','direction_flip','forced_long')
 source=Path(e.__file__).read_text();assert source.index('def load_clock_allow_empty')<source.index('def evaluate_primary')
def test_empty_clock(tmp_path):
 p=tmp_path/'empty.csv.gz';pd.DataFrame(columns=['entry_time','exit_time','side']).to_csv(p,index=False,compression='gzip')
 assert e.load_clock_allow_empty(p,'train',pd.Timestamp('2023-07-01',tz='UTC'),pd.Timestamp('2024-01-01',tz='UTC')).empty
def test_strict_accounting_constants():
 assert e.LEVERAGE==.5 and e.BASE_COST==.0006 and e.STRESS_COST==.001 and tuple(e.STAGES)==('train','test','eval','final')
