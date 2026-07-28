from __future__ import annotations

import copy

import pytest

import training.audit_gross9_invariant_ensemble_top10_marginal as module


def metric(
    absolute_return_pct: float,
    ratio: float,
    mdd: float,
    trades: int = 40,
) -> dict[str, float | int | dict[str, int]]:
    return {
        "absolute_return_pct": absolute_return_pct,
        "cagr_pct": ratio * mdd,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": ratio,
        "trades": trades,
        "win_rate": 0.5,
        "trades_by_sleeve": {},
    }


def preregistration() -> dict:
    return {
        "selection_contract": {
            "gross_cap": 10.0,
            "candidate_family_gross_cap": 1.0,
            "numeric_thresholds": {
                "train_strict_mdd_cap_pct": 40.0,
                "test2024_strict_mdd_cap_pct": 20.0,
                "absolute_return_retention_floor": 0.97,
                "min_candidate_trades_each_selection_window": 30,
                "min_ratio_improvement_over_comparator": 0.05,
                "entry_jaccard_cap": 0.5,
            },
        }
    }


def test_committed_preregistration_hash_and_contract_load() -> None:
    payload = module.load_preregistration(module.PREREGISTRATION)
    assert payload["short_name"] == "G9-IEM-1"
    assert len(payload["candidate_universe"]["candidates"]) == 10
    assert payload["future_veto_contract"]["future_can_rerank"] is False


def test_cell_weights_are_same_gross_and_leave_baseline_unchanged() -> None:
    baseline = {"a": 3.0, "b": 6.0}
    weights, comparator, gross = module.cell_weights(baseline, "candidate", 0.75)
    assert baseline == {"a": 3.0, "b": 6.0}
    assert weights == {"a": 3.0, "b": 6.0, "candidate": 0.75}
    assert gross == pytest.approx(9.75)
    assert sum(comparator.values()) == pytest.approx(gross)
    assert comparator == pytest.approx({"a": 3.25, "b": 6.5})


def test_signed_geometric_mean_is_deterministic_for_nonpassing_rows() -> None:
    assert module.signed_geometric_mean(4.0, 9.0) == pytest.approx(6.0)
    assert module.signed_geometric_mean(-4.0, 9.0) == pytest.approx(-6.0)
    assert module.signed_geometric_mean(-4.0, -9.0) == pytest.approx(-6.0)
    assert module.signed_geometric_mean(0.0, 9.0) == 0.0


def test_selection_row_passes_only_when_all_preregistered_checks_pass() -> None:
    baseline = {
        "train": metric(100.0, 4.0, 20.0, 100),
        "test2024": metric(100.0, 5.0, 10.0, 100),
    }
    comparator = {
        "train": metric(110.0, 4.0, 21.0, 100),
        "test2024": metric(110.0, 5.0, 11.0, 100),
    }
    stats = {
        "train": metric(105.0, 4.2, 19.0, 140),
        "test2024": metric(102.0, 5.2, 9.0, 140),
    }
    standalone = {
        "train": metric(10.0, 2.0, 5.0, 40),
        "test2024": metric(8.0, 2.0, 4.0, 40),
    }
    stressed_stats = {
        "train": metric(101.0, 4.0, 20.0, 140),
        "test2024": metric(99.0, 4.9, 10.0, 140),
    }
    stressed_standalone = {
        "train": metric(8.0, 1.5, 5.0, 40),
        "test2024": metric(6.0, 1.5, 4.0, 40),
    }
    row = module.selection_row(
        preregistration=preregistration(),
        candidate_name=module.CANDIDATE_NAMES[0],
        pre_evaluation_rank=1,
        weight=0.5,
        gross=9.5,
        baseline=baseline,
        stats=stats,
        comparator=comparator,
        standalone=standalone,
        stressed_stats=stressed_stats,
        stressed_standalone=stressed_standalone,
        max_entry_jaccard=0.1,
    )
    assert row["passes"] is True
    assert all(row["checks"].values())
    assert min(row["ratio_improvement_over_comparator"].values()) == pytest.approx(0.2)

    failed_stress = copy.deepcopy(stressed_standalone)
    failed_stress["test2024"]["absolute_return_pct"] = -0.1
    failed = module.selection_row(
        preregistration=preregistration(),
        candidate_name=module.CANDIDATE_NAMES[0],
        pre_evaluation_rank=1,
        weight=0.5,
        gross=9.5,
        baseline=baseline,
        stats=stats,
        comparator=comparator,
        standalone=standalone,
        stressed_stats=stressed_stats,
        stressed_standalone=failed_stress,
        max_entry_jaccard=0.1,
    )
    assert failed["passes"] is False
    assert failed["checks"]["stress_standalone_positive"] is False


def test_selection_input_validator_does_not_resolve_future_records(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = module.Config()
    prereg = module.load_preregistration(cfg.preregistration)
    original = module._resolved
    future_paths = {
        record["path"]
        for record in prereg[
            "future_only_provenance_not_openable_by_selection"
        ].values()
    }

    def guarded(path):
        if str(path) in future_paths:
            raise AssertionError("future-only path was opened")
        return original(path)

    monkeypatch.setattr(module, "_resolved", guarded)
    records = module.validate_selection_inputs(cfg, prereg)
    assert tuple(records) == module.SELECTION_PROVENANCE_KEYS


def test_rex_selection_path_set_excludes_future_source() -> None:
    assert module.SELECTION_REX_PATHS == (
        "data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl",
        "data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl",
    )
    assert all("eval_2025" not in path for path in module.SELECTION_REX_PATHS)


def test_invariant_support_diagnostics_emit_per_rank_hash_and_parity() -> None:
    prereg = module.load_preregistration(module.PREREGISTRATION)
    _, manifest = module.load_support(module.Config(), prereg)
    counts = {
        row["candidate_name"]: {
            "train": int(
                manifest["support_counts_by_rank_and_year"][
                    str(row["pre_evaluation_rank"])
                ]["2023"]
            ),
            "test2024": int(
                manifest["support_counts_by_rank_and_year"][
                    str(row["pre_evaluation_rank"])
                ]["2024"]
            ),
        }
        for row in prereg["candidate_universe"]["candidates"]
    }
    diagnostics = module.invariant_support_diagnostics(
        prereg, manifest, counts
    )
    assert diagnostics["all_rank_date_position_parity"] is True
    assert diagnostics["later_metrics_included"] is False
    assert len(diagnostics["per_rank"]) == 10
    assert all(row["date_position_parity"] for row in diagnostics["per_rank"])
    assert all(
        row["frozen_2023_signal_hash"]
        == row["verified_2023_signal_hash"]
        == row["emitted_2023_signal_hash"]
        for row in diagnostics["per_rank"]
    )
    assert all(len(row["support_rows_sha256"]) == 64 for row in diagnostics["per_rank"])
