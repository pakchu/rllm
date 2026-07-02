import json
from pathlib import Path

from training.audit_score_direction_regime_thresholds import DirectionThresholdAuditConfig, run


def _write(path: Path, rows):
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')


def test_threshold_audit_fits_train_only_rule(tmp_path):
    train = tmp_path / 'train.jsonl'; test = tmp_path / 'test.jsonl'; ev = tmp_path / 'eval.jsonl'; out = tmp_path / 'out.json'
    rows = [
        {'features': {'x': 0.0}, 'target': json.dumps({'direction_regime': 'LOW_SCORE_WINS'})},
        {'features': {'x': 1.0}, 'target': json.dumps({'direction_regime': 'HIGH_SCORE_WINS'})},
    ]
    _write(train, rows); _write(test, rows); _write(ev, rows)
    report = run(DirectionThresholdAuditConfig(train_jsonl=str(train), test_jsonl=str(test), eval_jsonl=str(ev), output=str(out), min_train_class_count=1))
    assert report['top_rules'][0]['feature'] == 'x'
    assert report['top_rules'][0]['train_accuracy'] == 1.0
    assert report['top_rules'][0]['test']['accuracy'] == 1.0
