from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from training import evaluate_gross9_overlap_net_position_portfolio as evaluator
from training import preregister_gross9_overlap_net_position_validation as validation


def _primary(
    *,
    return_pct: float = 2.0,
    ratio: float = 4.0,
    mdd: float = 2.0,
    edge: float = 25.0,
    stress_return: float = 1.0,
    stress_ratio: float = 3.0,
    pvalue: float = 0.05,
) -> dict[str, object]:
    return {
        "base": {
            "absolute_return_pct": return_pct,
            "cagr_to_strict_mdd": ratio,
            "strict_mdd_pct": mdd,
            "mean_exposure_weighted_gross_edge_bp": edge,
        },
        "stress": {
            "absolute_return_pct": stress_return,
            "cagr_to_strict_mdd": stress_ratio,
        },
        "cluster_signflip": {"pvalue": pvalue},
        "calendar_halves": {
            "first": {"absolute_return_pct": 1.0},
            "second": {"absolute_return_pct": 1.0},
        },
    }


def test_validation_freeze_opens_no_outcomes_and_keeps_stage_order() -> None:
    frozen = validation.build()
    validation.validate(frozen)

    assert frozen["sequence"]["stages"] == [
        "holdout_dec2023",
        "test2024",
        "eval2025",
        "final2026",
    ]
    assert frozen["sequence"]["stop_on_first_failure"] is True
    assert frozen["sequence"]["rerank_repair_or_substitution_authorized"] is False
    assert frozen["evidence_boundary"]["holdout_market_or_funding_rows_opened"] == 0
    assert frozen["evidence_boundary"]["oos_market_or_funding_rows_opened"] == 0
    assert frozen["gates"]["waived_cost_gates"] == [
        "turnover_cap",
        "sleeve_turnover_share_cap",
    ]
    assert set(frozen["implementation"]) == {
        "evaluator",
        "freezer",
        "fixed_ledger",
        "optimizer_utilities",
        "net_risk_metrics",
        "hash_bound_source_loader",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda frozen: frozen["implementation"].pop("evaluator"),
        lambda frozen: frozen["frozen_inputs"]["config"].pop("protocol_hash"),
        lambda frozen: frozen["frozen_inputs"].update({"selected_clocks": []}),
    ],
)
def test_validation_rejects_rehashed_incomplete_receipts(mutation) -> None:
    frozen = copy.deepcopy(validation.build())
    mutation(frozen)
    core = {key: value for key, value in frozen.items() if key != "manifest_hash"}
    frozen["manifest_hash"] = validation.canonical_hash(core)

    with pytest.raises(RuntimeError):
        validation.validate(frozen)


def test_signed_episode_count_uses_atomic_aggregate_net_position() -> None:
    clock = pd.DataFrame(
        [
            {
                "sleeve": "a",
                "weight": 0.3,
                "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "exit_time": pd.Timestamp("2024-01-01T08:00:00Z"),
                "side": 1,
            },
            {
                "sleeve": "b",
                "weight": 0.3,
                "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "exit_time": pd.Timestamp("2024-01-01T08:00:00Z"),
                "side": -1,
            },
            {
                "sleeve": "c",
                "weight": 0.2,
                "entry_time": pd.Timestamp("2024-01-02T00:00:00Z"),
                "exit_time": pd.Timestamp("2024-01-02T08:00:00Z"),
                "side": -1,
            },
        ]
    )

    assert evaluator.aggregate_net_signed_episode_count(clock) == 1


def test_holdout_clock_rows_exist_inside_frozen_window() -> None:
    frozen = validation.build()
    start = evaluator._utc(evaluator.STAGES["holdout_dec2023"][1])
    end = evaluator._utc(evaluator.STAGES["holdout_dec2023"][2])

    clock, receipts = evaluator.load_portfolio_clock(
        frozen,
        "holdout_dec2023",
        start,
        end,
    )

    assert len(clock) > 0
    assert sum(row["stage_rows"] for row in receipts) == len(clock)
    assert clock["entry_time"].ge(start).all()
    assert clock["exit_time"].le(end).all()


def test_holdout_checks_do_not_reject_turnover_or_costs() -> None:
    checks = evaluator.holdout_checks(
        _primary(),
        {"mean_abs_net_position": 0.1, "max_abs_net_position": 0.5},
        intervals=8,
        weeks=3,
        gates=validation.build()["gates"]["holdout_dec2023"],
    )

    assert all(checks.values())
    assert "turnover_cap" not in checks
    assert "sleeve_turnover_share_cap" not in checks


def test_oos_checks_keep_performance_and_net_risk_gates() -> None:
    checks = evaluator.oos_checks(
        "test2024",
        _primary(),
        {"mean_abs_net_position": 0.1, "max_abs_net_position": 0.5},
        signed_episodes=12,
        gates=validation.build()["gates"]["oos"],
    )

    assert all(checks.values())
    assert set(checks) >= {
        "absolute_return_positive",
        "stress_absolute_return_positive",
        "weekly_cluster_signflip_one_sided_p_max",
        "aggregate_net_signed_episode_min",
        "mean_abs_net_position_cap",
        "max_abs_net_position_cap",
    }
    assert "turnover_cap" not in checks


def test_missing_predecessor_blocks_before_any_stage_loader(monkeypatch, tmp_path) -> None:
    opened = False

    def should_not_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("stage data opened before predecessor authorization")

    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("{}", encoding="utf-8")
    fake_freeze = {"manifest_hash": "f" * 64, "implementation": {}}
    monkeypatch.setattr(evaluator, "load_validation_freeze", lambda path: fake_freeze)
    monkeypatch.setattr(evaluator, "load_portfolio_clock", should_not_open)
    monkeypatch.setattr(evaluator, "load_stage_sources", should_not_open)
    outputs = {stage: tmp_path / f"{stage}.json" for stage in evaluator.STAGES}

    with pytest.raises(RuntimeError, match="missing predecessor holdout_dec2023"):
        evaluator.run(
            "test2024",
            tmp_path / "result.json",
            freeze_path=freeze_path,
            outputs=outputs,
        )
    assert opened is False


def test_failed_predecessor_cannot_authorize_next_stage(tmp_path) -> None:
    predecessor = {
        "policy_id": evaluator.POLICY_ID,
        "stage": "holdout_dec2023",
        "passed": False,
        "advance_to_next_stage": False,
    }
    predecessor["manifest_hash"] = evaluator.canonical_hash(predecessor)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(predecessor), encoding="utf-8")
    outputs = {stage: tmp_path / f"{stage}.json" for stage in evaluator.STAGES}
    outputs["holdout_dec2023"] = path

    with pytest.raises(RuntimeError, match="predecessor did not authorize"):
        evaluator.verify_predecessor(
            "test2024",
            {"path": "freeze", "sha256": "a", "manifest_hash": "b"},
            {},
            outputs,
        )


def test_stale_predecessor_freeze_cannot_authorize_next_stage(tmp_path) -> None:
    current_freeze = {"path": "freeze", "sha256": "a", "manifest_hash": "b"}
    predecessor = {
        "policy_id": evaluator.POLICY_ID,
        "stage": "holdout_dec2023",
        "validation_freeze": {"path": "old", "sha256": "c", "manifest_hash": "d"},
        "implementation": {},
        "passed": True,
        "advance_to_next_stage": True,
    }
    predecessor["manifest_hash"] = evaluator.canonical_hash(predecessor)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(predecessor), encoding="utf-8")
    outputs = {stage: tmp_path / f"{stage}.json" for stage in evaluator.STAGES}
    outputs["holdout_dec2023"] = path

    with pytest.raises(RuntimeError, match="predecessor did not authorize"):
        evaluator.verify_predecessor("test2024", current_freeze, {}, outputs)


def test_corrupt_predecessor_chain_cannot_authorize_eval(tmp_path) -> None:
    freeze_receipt = {"path": "freeze", "sha256": "a", "manifest_hash": "b"}
    implementation: dict[str, object] = {}
    holdout = {
        "policy_id": evaluator.POLICY_ID,
        "stage": "holdout_dec2023",
        "validation_freeze": freeze_receipt,
        "implementation": implementation,
        "predecessor": None,
        "passed": True,
        "advance_to_next_stage": True,
    }
    holdout["manifest_hash"] = evaluator.canonical_hash(holdout)
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(holdout), encoding="utf-8")
    test = {
        "policy_id": evaluator.POLICY_ID,
        "stage": "test2024",
        "validation_freeze": freeze_receipt,
        "implementation": implementation,
        "predecessor": {"stage": "holdout_dec2023", "path": "wrong"},
        "passed": True,
        "advance_to_next_stage": True,
    }
    test["manifest_hash"] = evaluator.canonical_hash(test)
    test_path = tmp_path / "test.json"
    test_path.write_text(json.dumps(test), encoding="utf-8")
    outputs = {stage: tmp_path / f"{stage}.json" for stage in evaluator.STAGES}
    outputs["holdout_dec2023"] = holdout_path
    outputs["test2024"] = test_path

    with pytest.raises(RuntimeError, match="predecessor chain drift"):
        evaluator.verify_predecessor(
            "eval2025",
            freeze_receipt,
            implementation,
            outputs,
        )


def test_final_window_matches_frozen_existing_oos_convention() -> None:
    assert evaluator.STAGES["final2026"] == (
        "final",
        "2026-01-01T00:00:00Z",
        "2026-08-01T00:00:00Z",
    )
