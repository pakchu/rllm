import json
from training import evaluate_high_volatility_turn_of_candle_momentum_gross9_novelty as novelty
def test_novelty_artifact_is_terminal_before_economics():
 p=json.loads(novelty.OUTPUT.read_text());assert p["policy_id"]=="HVTOCM-30M";assert p["source_support_passed"];assert not p["every_gross9_sleeve_passed"];assert not p["advance_to_economic_outcomes"];assert not p["gross9_sleeves"]["fresh_kimchi_fx"]["passed"];assert not p["gross9_sleeves"]["markov_transition_long"]["passed"];assert p["evidence_boundary"]["economic_outcome_rows_opened"]==0;h=p.pop("manifest_hash");assert novelty.chash(p)==h
