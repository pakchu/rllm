from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
from training import build_options_led_volatility_expansion_premium_relay_support as s

def digest(path: str|Path)->str:
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def test_frozen_support_replay_is_deterministic_and_outcome_blind() -> None:
 before=(digest(s.DEFAULT_CLOCK),digest(s.DEFAULT_RESULT))
 report=s.run()
 after=(digest(s.DEFAULT_CLOCK),digest(s.DEFAULT_RESULT))
 assert after==before
 assert report['outcomes_opened'] is False
 assert report['outcome_sources_opened']==[]
 assert report['btc_execution_rows_opened']==0
 assert report['funding_rows_opened']==0
 assert report['gross9_rows_opened']==0
 assert report['support_passed'] is True
 assert report['named_family_novelty_passed'] is True
 assert report['gross9_novelty_status']=='pending'
 assert report['advance_to_economic_outcomes'] is False

def test_primary_clock_is_causal_nonoverlapping_and_support_counts_are_frozen() -> None:
 report=json.loads(s.DEFAULT_RESULT.read_text())
 assert {k:v['events'] for k,v in report['support'].items()}=={'train':109,'test':127,'eval':63,'final':52}
 frame=pd.read_csv(s.DEFAULT_CLOCK,compression='gzip',parse_dates=['decision_time','feature_available_time','entry_time','exit_time'])
 assert list(frame.columns)==s.CLOCK_COLS
 assert (frame.feature_available_time<frame.entry_time).all()
 assert (frame.entry_time==frame.decision_time+pd.Timedelta(minutes=5)).all()
 assert (frame.exit_time-frame.entry_time==pd.Timedelta(hours=24)).all()
 ordered=frame.sort_values('entry_time').reset_index(drop=True)
 assert (ordered.entry_time.iloc[1:].reset_index(drop=True)>=ordered.exit_time.iloc[:-1].reset_index(drop=True)).all()
 assert set(frame.side)=={-1,1}

def test_named_family_novelty_limits_pass_without_gross9_substitution() -> None:
 report=json.loads(s.DEFAULT_RESULT.read_text())
 for metrics in report['named_family_novelty'].values():
  assert metrics['exact_entry_jaccard']<=.1
  assert metrics['one_to_one_6h_max_matched_share']<=.45
  assert metrics['occupied_5m_bar_jaccard']<=.3
