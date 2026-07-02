import json
from pathlib import Path

from training.build_event_candidate_family_pairwise_cards import PairwiseFamilyCardConfig, build_records, run


def _state_card(path: Path):
    row = {
        'split': 'train',
        'fold': {'name': 'f', 'start': '2024-01-01', 'end': '2024-02-01'},
        'position_state': {'mode': 'FLAT', 'side': 'NONE'},
        'target': {'choice_id': 'B', 'family': 'chosen_family', 'reason': 'test'},
        'options': [
            {'id': 'A', 'family': 'reject_high', 'pre_fold_score': 1.0, 'threshold': 0.1, 'evidence_count': 1, 'latest_evidence': {}},
            {'id': 'B', 'family': 'chosen_family', 'pre_fold_score': 2.0, 'threshold': 0.2, 'evidence_count': 1, 'latest_evidence': {}},
            {'id': 'C', 'family': 'reject_low', 'pre_fold_score': 0.5, 'threshold': 0.3, 'evidence_count': 1, 'latest_evidence': {}},
            {'id': 'ABSTAIN', 'family': 'ABSTAIN', 'pre_fold_score': 0.0, 'threshold': None, 'evidence_count': 0, 'latest_evidence': {}},
        ],
    }
    path.write_text(json.dumps(row) + '\n')


def test_pairwise_cards_include_position_and_contrast(tmp_path):
    inp = tmp_path / 'cards.jsonl'
    _state_card(inp)
    rows = build_records(PairwiseFamilyCardConfig(input_jsonl=str(inp), output_jsonl=str(tmp_path / 'out.jsonl'), max_rejected_per_row=2))

    assert len(rows) == 2
    assert rows[0]['chosen']['family'] == 'chosen_family'
    assert rows[0]['rejected']['family'] == 'reject_high'
    assert rows[0]['position_state']['mode'] == 'FLAT'
    assert 'position_state' in rows[0]['prompt']
    assert rows[0]['completion'] == 'A'


def test_pairwise_run_writes_jsonl(tmp_path):
    inp = tmp_path / 'cards.jsonl'
    out = tmp_path / 'pairs.jsonl'
    _state_card(inp)
    summary = run(PairwiseFamilyCardConfig(input_jsonl=str(inp), output_jsonl=str(out), max_rejected_per_row=3))

    assert summary['rows'] == 3
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert written[0]['leakage_guard']['position_state_included'] is True
