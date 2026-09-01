import pytest

from training import eval_pposm_residual_router as router


def _score(base, candidate, margin, target="KEEP"):
    base_identity = (
        base
        if "pposm-counterfactual-action|" in base
        else f"pposm-counterfactual-action|pre_2024|{base}"
    )
    return {
        "identity": f"rid|{base_identity}|{candidate}",
        "base_identity": base_identity,
        "candidate_action": candidate,
        "split": "train",
        "window": "pre_2024",
        "target": target,
        "switch_margin": margin,
        "date": "2023-01-01",
        "signal_time": "2023-01-01",
        "signal_pos": 1,
        "scores": {"KEEP": 0.0, "SWITCH": margin},
        "residual_advantage": 0.0,
    }


def test_freeze_threshold_uses_all_train_margins_and_defaults_when_materiality_impossible():
    spec = router.freeze_threshold([
        _score("a", "SKIP", -5, "KEEP"), _score("a", "TP12", 0.1, "SWITCH"),
        _score("b", "SKIP", 9, "KEEP"), _score("b", "TP12", 0.3, "SWITCH"),
    ])
    assert spec["status"] == "no_feasible_train_materiality_default_tp4"
    assert spec["threshold"] > 9
    assert "all_distinct_pre2024_margins" in spec["threshold_source"]


def test_freeze_threshold_maximizes_train_residual_utility_subject_to_materiality():
    scores = []
    for i in range(100):
        scores.append({**_score(f"s{i}", "SKIP", 0.3 if i < 10 else -0.1), "residual_advantage": 0.01 if i < 10 else -0.02})
        scores.append({**_score(f"s{i}", "TP12", 0.2 if 10 <= i < 20 else -0.2), "residual_advantage": 0.02 if 10 <= i < 20 else -0.01})
    spec = router.freeze_threshold(scores)
    assert spec["status"] == "feasible_train_materiality"
    assert spec["threshold"] == pytest.approx(-0.1)
    mat = spec["selected_train_evaluation"]["materiality"]
    assert mat["non_default_counts"] == {"SKIP": 10, "TP12": 10}


def test_assemble_defaults_tp4_and_switches_above_frozen_threshold():
    spec = {"threshold": 0.2}
    preds, report = router.assemble_routes([
        _score("a", "SKIP", 0.1), _score("a", "TP12", 0.19),
        _score("b", "SKIP", 0.21), _score("b", "TP12", -0.2),
    ], spec)
    assert [row["prediction"] for row in preds] == ["TP4", "SKIP"]
    assert report["difference_rate_vs_always_tp4"] == pytest.approx(0.5)


def test_dual_switch_chooses_larger_margin_and_ties_skip_first():
    preds, _ = router.assemble_routes([
        _score("a", "SKIP", 0.4), _score("a", "TP12", 0.5),
        _score("b", "SKIP", 0.6), _score("b", "TP12", 0.6),
    ], {"threshold": 0.1})
    assert [row["prediction"] for row in preds] == ["TP12", "SKIP"]


def test_validate_score_rows_rejects_missing_pair():
    with pytest.raises(ValueError, match="exactly two"):
        router.validate_score_rows([_score("a", "SKIP", 0.1)])


def test_freeze_threshold_rejects_non_train_or_post_2024_rows():
    scores = [
        _score("pposm-counterfactual-action|test_2024|1", "SKIP", 0.1),
        _score("pposm-counterfactual-action|test_2024|1", "TP12", 0.2),
    ]
    for row in scores:
        row.update(split="oos", window="test_2024", signal_time="2024-01-01")
    with pytest.raises(ValueError, match="pre-2024"):
        router.freeze_threshold(scores)
