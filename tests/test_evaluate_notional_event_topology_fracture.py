from __future__ import annotations

from training import evaluate_notional_event_topology_fracture as evaluate


def _metrics(
    *,
    absolute_return: float = 10.0,
    ratio: float = 4.0,
    mdd: float = 5.0,
    trades: int = 80,
    p_value: float = 0.05,
) -> dict[str, object]:
    return {
        "absolute_return_pct": absolute_return,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": mdd,
        "trade_count": trades,
        "weekly_cluster_sign_flip": {"p_value_one_sided": p_value},
    }


def _candidate(name: str, *, ratio: float = 4.0) -> dict[str, object]:
    item: dict[str, object] = {
        "candidate": {"name": name},
        "windows": {
            "train": _metrics(ratio=ratio),
            "select2023": _metrics(ratio=ratio),
            "select2023_h1": _metrics(trades=30),
            "select2023_h2": _metrics(trades=30),
        },
    }
    item["qualification"] = evaluate.qualification(item)
    return item


def test_frozen_netf_artifacts_and_support_stopping_rule_match() -> None:
    result = evaluate._verify_preregistration()
    assert result["protocol"]["outcomes_opened_for_netf"] is False
    assert result["all_candidates_pass_support"] is True
    assert result["support_calibration"]["selected_tension_quantile"] == 0.875
    assert result["support_calibration"]["further_support_repairs_allowed"] is False


def test_evaluator_opens_no_window_at_or_after_2024() -> None:
    assert set(evaluate.WINDOWS) == {
        "train",
        "select2023",
        "select2023_h1",
        "select2023_h2",
    }
    assert max(end for _, end in evaluate.WINDOWS.values()) == "2024-01-01"


def test_qualification_enforces_half_support_and_strict_p_value() -> None:
    item = _candidate("candidate")
    assert item["qualification"]["qualifies"] is True

    item["windows"]["select2023_h1"]["trade_count"] = 19
    item["windows"]["train"]["weekly_cluster_sign_flip"][
        "p_value_one_sided"
    ] = 0.10
    result = evaluate.qualification(item)
    assert result["qualifies"] is False
    assert "select2023_h1: fewer than 20 trades" in result["failures"]
    assert "train: weekly-cluster p-value not below 0.10" in result["failures"]


def test_selection_rejects_failures_and_uses_frozen_tie_break() -> None:
    failing = _candidate("failed")
    failing["windows"]["select2023"]["absolute_return_pct"] = 0.0
    failing["qualification"] = evaluate.qualification(failing)
    alpha = _candidate("alpha")
    beta = _candidate("beta")
    result = evaluate.select_candidate([failing, beta, alpha])
    assert failing["qualification"]["qualifies"] is False
    assert result["selected_candidate"] == "alpha"
    assert result["rejected"] is False
