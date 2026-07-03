from training.build_rex_candidate_ranker_records import _target, _bucket


def test_target_requires_clean_positive_path():
    assert _target({"net_return": 0.008, "mae": 0.01, "mfe_to_mae": 1.3, "utility": 0.002}) == "TAKE"
    assert _target({"net_return": 0.008, "mae": 0.03, "mfe_to_mae": 1.3, "utility": 0.002}) == "SKIP"
    assert _target({"net_return": -0.001, "mae": 0.01, "mfe_to_mae": 2.0, "utility": 0.001}) == "SKIP"


def test_bucket_uses_ordered_cut_labels():
    assert _bucket(0.5, (1.0, 2.0), ("low", "mid", "high")) == "low"
    assert _bucket(1.5, (1.0, 2.0), ("low", "mid", "high")) == "mid"
    assert _bucket(2.5, (1.0, 2.0), ("low", "mid", "high")) == "high"
