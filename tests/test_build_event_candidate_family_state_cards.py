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
