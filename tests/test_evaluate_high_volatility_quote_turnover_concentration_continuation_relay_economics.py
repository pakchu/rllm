from pathlib import Path
import pandas as pd
from training import evaluate_high_volatility_quote_turnover_concentration_continuation_relay_economics as e
def test_evaluator_is_bound_and_empty_safe():
 assert e.POLICY_ID=='HVTCCR-8' and e.sha256(e.PREREG)==e.PREREG_SHA and e.sha256(e.SUPPORT)==e.SUPPORT_SHA and e.sha256(e.NOVELTY)==e.NOVELTY_SHA and e.sha256(e.CLOCK)==e.CLOCK_SHA
 assert e.CONTROLS==('no_concentration_tail','no_variation_gate','unweighted_return_direction','one_boundary_stale_concentration','direction_flip');assert Path(e.__file__).read_text().index('def load_clock_allow_empty')<Path(e.__file__).read_text().index('def evaluate_primary')
def test_empty(tmp_path):
 p=tmp_path/'x.csv.gz';pd.DataFrame(columns=['entry_time','exit_time','side']).to_csv(p,index=False,compression='gzip');assert e.load_clock_allow_empty(p,'train',pd.Timestamp('2023-07-01',tz='UTC'),pd.Timestamp('2024-01-01',tz='UTC')).empty
