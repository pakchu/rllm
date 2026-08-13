import gzip
import pandas as pd
from training import evaluate_high_volatility_taker_imbalance_flow_persistence_economics as economics
def test_empty_clock_is_valid(tmp_path):
 p=tmp_path/'e.csv.gz'
 with gzip.open(p,'wt') as f:f.write('entry_time,exit_time,side\n')
 assert economics.load_clock_allow_empty(p,'train',pd.Timestamp('2023-07-01T00:00:00Z'),pd.Timestamp('2024-01-01T00:00:00Z')).empty
def test_contract():
 assert economics.CONTROLS==("no_persistence_gate","no_variation_gate","imbalance_level_tail","one_decision_stale_persistence","direction_flip","forced_long");assert economics.LEVERAGE==.5 and economics.BASE_COST==.0006 and economics.STRESS_COST==.001;assert 'load_clock_allow_empty' in open(economics.__file__).read()
