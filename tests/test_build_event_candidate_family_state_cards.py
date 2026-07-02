import json
from pathlib import Path

from training.build_event_candidate_family_state_cards import FamilyStateCardConfig, build_records, run


def _report(path: Path):
    path.write_text(json.dumps({
        'folds': [{
            'fold': {'name': 'eval_x', 'start': '2024-01-01', 'end': '2024-07-01'},
            'selector_mode': 'prefold_prior',
            'selected_family': 'rex_htf_pullback_reclaim',
            'abstained': False,
            'selected_metrics': {'cagr_to_strict_mdd': 1.2},
            'pre_fold_scoreboard': [{
                'family': 'rex_htf_pullback_reclaim',
                'score': 1.5,
                'threshold': 0.2,
                'evidence': [{'fold': 'prefold_train', 'distance': 0.0, 'raw_score': 1.5, 'weighted_score': 1.5, 'metrics': {'cagr_to_strict_mdd': 1.1, 'trade_entries': 50, 'p_value_mean_ret_approx': 0.2}}],
            }],
        }]
    }))


def test_state_cards_include_explicit_position_state(tmp_path):
    report = tmp_path / 'report.json'
    _report(report)
    rows = build_records(FamilyStateCardConfig(selector_report=str(report), output_jsonl=str(tmp_path / 'out.jsonl')))

    assert rows[0]['position_state']['mode'] == 'FLAT'
    assert rows[0]['position_state']['side'] == 'NONE'
    assert 'Current position:' in rows[0]['prompt']
    assert rows[0]['completion'] == 'A'


def test_state_card_run_writes_jsonl(tmp_path):
    report = tmp_path / 'report.json'
    output = tmp_path / 'out.jsonl'
    _report(report)
    summary = run(FamilyStateCardConfig(selector_report=str(report), output_jsonl=str(output)))

    assert summary['rows'] == 1
    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert written[0]['position_state']['mode'] == 'FLAT'


def test_state_card_run_filters_fold_range(tmp_path):
    report = tmp_path / 'report.json'
    output = tmp_path / 'out.jsonl'
    report.write_text(json.dumps({
        'folds': [
            {'fold': {'name': 'old', 'start': '2024-01-01', 'end': '2024-02-01'}, 'selector_mode': 'x', 'selected_family': 'f', 'abstained': False, 'pre_fold_scoreboard': [{'family': 'f', 'score': 1.0, 'threshold': 0.1, 'evidence': []}]},
            {'fold': {'name': 'new', 'start': '2025-01-01', 'end': '2025-02-01'}, 'selector_mode': 'x', 'selected_family': 'f', 'abstained': False, 'pre_fold_scoreboard': [{'family': 'f', 'score': 1.0, 'threshold': 0.1, 'evidence': []}]},
        ]
    }))
    summary = run(FamilyStateCardConfig(selector_report=str(report), output_jsonl=str(output), fold_start='2025-01-01', fold_end='2026-01-01'))

    assert summary['rows'] == 1
    row = json.loads(output.read_text().strip())
    assert row['fold']['name'] == 'new'


def test_randomize_options_moves_selected_family_off_a_when_seeded(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({
        'folds': [{
            'fold': {'name': 'eval_x', 'start': '2024-01-01', 'end': '2024-02-01'},
            'selector_mode': 'x',
            'selected_family': 'fam0',
            'abstained': False,
            'pre_fold_scoreboard': [
                {'family': 'fam0', 'score': 3.0, 'threshold': 0.1, 'evidence': []},
                {'family': 'fam1', 'score': 2.0, 'threshold': 0.1, 'evidence': []},
                {'family': 'fam2', 'score': 1.0, 'threshold': 0.1, 'evidence': []},
            ],
        }]
    }))

    rows = build_records(FamilyStateCardConfig(selector_report=str(report), output_jsonl=str(tmp_path / 'out.jsonl'), randomize_options=True, random_seed=0))

    assert {opt['id'] for opt in rows[0]['options']} >= {'A', 'B', 'C', 'ABSTAIN'}
    assert rows[0]['target']['family'] == 'fam0'
    assert rows[0]['completion'] in {'A', 'B', 'C'}
    assert rows[0]['leakage_guard']['option_order_randomized'] is True
