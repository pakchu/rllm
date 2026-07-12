from training.evaluate_tabicl_rolling_adaptation import (
    FOLDS,
    _frozen_candidates,
    validate_fold_chronology,
)


def test_fold_chronology_is_causal():
    validate_fold_chronology()


def test_fold_chronology_rejects_overlap():
    bad = dict(FOLDS)
    bad["bad"] = {
        "fit": ("2020-01-01", "2024-01-01"),
        "calibration": ("2023-01-01", "2024-01-01"),
        "evaluation": ("2024-01-01", "2025-01-01"),
    }

    try:
        validate_fold_chronology(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping fold chronology must fail")


def test_frozen_candidates_are_copied_before_mutation():
    source = {"top10": [{"signal_hash": "a"}]}
    copied = _frozen_candidates(source)
    copied[0]["rank"] = 1

    assert "rank" not in source["top10"][0]
