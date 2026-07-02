import json
from pathlib import Path

from training.build_event_candidate_family_clean_pairwise_cards import CleanPairwiseFamilyCardConfig, build_records, build_state_rows, run


def _report(path: Path):
    path.write_text(json.dumps({
        'folds': [{
            'fold': {'name': 'eval_x', 'start': '2024-01-01', 'end': '2024-02-01'},
            'selector_mode': 'prefold_prior',
            'selected_family': 'noisy_prefold_winner',
            'abstained': False,
            'pre_fold_scoreboard': [
                {'family': 'noisy_prefold_winner', 'score': 5.0, 'threshold': 0.1, 'evidence': []},
                {'family': 'clean_diagnostic_winner', 'score': 1.0, 'threshold': 0.1, 'evidence': []},
                {'family': 'bad_diagnostic', 'score': 0.5, 'threshold': 0.1, 'evidence': []},
            ],
            'top_fold_diagnostic_not_for_selection': [
                {'family': 'bad_diagnostic', 'metrics': {'cagr_to_strict_mdd': 9.0, 'cagr_pct': 20.0, 'strict_mdd_pct': 2.0, 'trade_entries': 1}},
                {'family': 'clean_diagnostic_winner', 'metrics': {'cagr_to_strict_mdd': 1.5, 'cagr_pct': 15.0, 'strict_mdd_pct': 10.0, 'trade_entries': 12, 'p_value_mean_ret_approx': 0.2}},
                {'family': 'noisy_prefold_winner', 'metrics': {'cagr_to_strict_mdd': -1.0, 'cagr_pct': -5.0, 'strict_mdd_pct': 5.0, 'trade_entries': 20}},
            ],
        }, {
            'fold': {'name': 'eval_y', 'start': '2024-02-01', 'end': '2024-03-01'},
            'selector_mode': 'prefold_prior',
            'pre_fold_scoreboard': [
                {'family': 'only_bad', 'score': 2.0, 'threshold': 0.1, 'evidence': []},
            ],
            'top_fold_diagnostic_not_for_selection': [
                {'family': 'only_bad', 'metrics': {'cagr_to_strict_mdd': -0.5, 'cagr_pct': -10.0, 'strict_mdd_pct': 20.0, 'trade_entries': 20}},
            ],
        }]
    }))


def test_clean_target_uses_diagnostic_label_not_prefold_winner(tmp_path):
    report = tmp_path / 'report.json'
    _report(report)
    rows = build_state_rows(CleanPairwiseFamilyCardConfig(selector_report=str(report), output_jsonl=str(tmp_path / 'out.jsonl'), max_options=3, min_diagnostic_trades=8))

    assert rows[0]['target']['family'] == 'clean_diagnostic_winner'
    assert rows[0]['target']['reason'] == 'best_target_fold_diagnostic_family_with_prefold_option'
    assert rows[0]['target']['diagnostic_target']['trade_entries'] == 12


def test_clean_target_falls_back_to_abstain_when_no_valid_diagnostic(tmp_path):
    report = tmp_path / 'report.json'
    _report(report)
    rows = build_state_rows(CleanPairwiseFamilyCardConfig(selector_report=str(report), output_jsonl=str(tmp_path / 'out.jsonl'), fold_start='2024-02-01'))

    assert rows[0]['target']['family'] == 'ABSTAIN'
    assert rows[0]['target']['diagnostic_target'] is None


def test_clean_pairwise_prompt_excludes_target_fold_metrics(tmp_path):
    report = tmp_path / 'report.json'
    _report(report)
    records = build_records(CleanPairwiseFamilyCardConfig(selector_report=str(report), output_jsonl=str(tmp_path / 'out.jsonl'), fold_end='2024-02-01', max_rejected_per_row=2))

    assert records
    assert records[0]['diagnostic_target']['family'] == 'clean_diagnostic_winner'
    assert records[0]['leakage_guard']['target_fold_metrics_not_in_prompt'] is True
    assert 'diagnostic_target' not in records[0]['prompt']
    assert 'cagr_pct' not in records[0]['prompt']
    assert 'clean_diagnostic_winner' in records[0]['prompt']


def test_clean_pairwise_run_writes_summary(tmp_path):
    report = tmp_path / 'report.json'
    out = tmp_path / 'pairs.jsonl'
    _report(report)
    summary = run(CleanPairwiseFamilyCardConfig(selector_report=str(report), output_jsonl=str(out), max_rejected_per_row=1))

    assert summary['rows'] == 2
    assert out.exists()
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert written[0]['completion'] == 'A'
