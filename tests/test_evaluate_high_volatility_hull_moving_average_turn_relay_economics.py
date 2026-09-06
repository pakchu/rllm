from pathlib import Path
import pandas as pd
from training import evaluate_high_volatility_hull_moving_average_turn_relay_economics as e
def test_bound_and_empty_safe():
 assert e.POLICY_ID=="HVHMA-24" and e.sha256(e.PREREG)==e.PREREG_SHA and e.sha256(e.SUPPORT)==e.SUPPORT_SHA and e.sha256(e.NOVELTY)==e.NOVELTY_SHA and e.sha256(e.CLOCK)==e.CLOCK_SHA
 assert e.CONTROLS==("no_variation_gate","full_length_wma_turn","one_bar_stale_turn","direction_flip")
 source=Path(e.__file__).read_text();assert source.index("def load_clock_allow_empty")<source.index("def evaluate_primary")
def test_empty_clock(tmp_path):
 p=tmp_path/"empty.csv.gz";pd.DataFrame(columns=["entry_time","exit_time","side"]).to_csv(p,index=False,compression="gzip");assert e.load_clock_allow_empty(p,"train",pd.Timestamp("2023-07-01",tz="UTC"),pd.Timestamp("2024-01-01",tz="UTC")).empty
