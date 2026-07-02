import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from training.build_score_direction_regime_dataset import ScoreDirectionRegimeConfig, _quantile_threshold, build_records


def test_direction_label_uses_clean_target_score_rank(tmp_path):
    report = tmp_path / 'report.json'
    market = tmp_path / 'market.csv'
    report.write_text(json.dumps({'folds': [{
        'fold': {'name': 'f', 'start': '2024-02-01', 'end': '2024-03-01'},
        'pre_fold_scoreboard': [
            {'family': 'high', 'score': 10, 'threshold': 0, 'evidence': []},
            {'family': 'low', 'score': -1, 'threshold': 0, 'evidence': []},
        ],
        'top_fold_diagnostic_not_for_selection': [
            {'family': 'low', 'metrics': {'cagr_to_strict_mdd': 2, 'cagr_pct': 10, 'strict_mdd_pct': 5, 'trade_entries': 20}},
        ],
    }]}))
    market.write_text('date,open,high,low,close,volume\n2024-01-01,1,1,1,1,1\n')

    fake_features = pd.DataFrame({'date': pd.to_datetime(['2024-01-01']), 'trend_12': [0.1]})
    with patch('training.build_score_direction_regime_dataset._load_market_features', return_value=fake_features):
        rows = build_records(ScoreDirectionRegimeConfig(selector_report=str(report), market_csv=str(market), output_jsonl=str(tmp_path / 'out.jsonl')))

    assert json.loads(rows[0]['target'])['direction_regime'] == 'LOW_SCORE_WINS'
    assert rows[0]['completion'] == rows[0]['target']
    assert 'market_regime_features' in rows[0]['prompt']
    assert rows[0]['leakage_guard']['features_before_fold_start'] is True


def test_quantile_threshold_uses_interpolated_median():
    assert _quantile_threshold([-1.0, 10.0], 0.5) == 4.5
    assert _quantile_threshold([1.0, 3.0, 9.0], 0.5) == 3.0
