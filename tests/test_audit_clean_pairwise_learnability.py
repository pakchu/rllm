import json
from pathlib import Path

from training.audit_clean_pairwise_learnability import PairwiseLearnabilityAuditConfig, run


def test_pairwise_learnability_audits_prompt_visible_rules(tmp_path):
    path = tmp_path / 'pairs.jsonl'
    row = {
        'prompt': 'x\n' + json.dumps({'option_a': {'family': 'fa', 'pre_fold_score': 2, 'evidence_count': 1, 'latest_evidence': {}}, 'option_b': {'family': 'fb', 'pre_fold_score': 1, 'evidence_count': 1, 'latest_evidence': {}}}),
        'target': json.dumps({'choice': 'A'}),
        'target_family': 'fa',
        'order_variant': 'chosen_as_a',
    }
    path.write_text(json.dumps(row) + '\n')
    report = run(PairwiseLearnabilityAuditConfig(jsonl=str(path)))

    assert report['rows'] == 1
    assert report['target_counts']['A'] == 1
    assert report['rule_accuracy']['higher_pre_fold_score'] == 1.0
