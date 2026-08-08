from pathlib import Path
import pandas as pd
from training import build_funding_acceleration_reversal_relay_support as support

def test_farr_source_builder_is_outcome_blind_and_hash_bound():
 s=Path(support.__file__).read_text();assert support.PREREG_SHA=="da543375cf42c905c3c193dcdab6c6fa1f8f498bc4ba53594072ec35fb93ef64";assert support.SETTLEMENT_QUERY.startswith("SELECT funding_time,funding_rate");assert "mark_price" not in support.SETTLEMENT_QUERY;assert "postentry_return_pnl_execution_price_opened" in s;assert "gross9_rows_opened" in s

def test_farr_strict_prior_rank_excludes_current_value():
 x=pd.Series(range(130),dtype=float);r=support.strict_prior_rank(x);assert r.iloc[:126].isna().all();assert r.iloc[126]==1.

def test_farr_controls_and_clock_are_frozen():
 assert support.CONTROLS==("no_volatility_gate","no_change_tail","funding_level","one_day_stale_change","direction_flip");assert support.MINIMUM_EVENTS=={"train":8,"test":12,"eval":12,"final":8}
