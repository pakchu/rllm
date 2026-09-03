from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_gross9_frequency_waived_revalidation as evaluator
from training import preregister_gross9_frequency_waived_revalidation as freeze_mod


def _selection() -> dict[str, object]:
    exact = []
    sleeves = [f"S{i:02d}" for i in range(14)]
    for idx in range(64):
        first = sleeves[idx % 14]
        second = sleeves[(idx + 1) % 14]
        exact.append(
            {
                "proxy_rank": idx + 1,
                "sleeve_weights": {
                    first: 0.15,
                    second: 0.10 + (idx % 5) * 0.01,
                },
                "gates": {
                    "absolute_return_positive": True,
                    "turnover_cap": False,
                },
            }
        )
    core = {"policy_id": "G9-OVERLAP-PORT-1", "exact_finalists": exact, "authoritative_rank1": {"sleeve_weights": exact[0]["sleeve_weights"]}}
    core["manifest_hash"] = freeze_mod.canonical_hash(core)
    return core


def _fake_freeze(candidates=None) -> dict[str, object]:
    candidates = candidates or [
        {"candidate_id": "c1", "kind": "frozen_exact_finalist", "weights": {"a": 0.25}, "preexisting_frozen_candidate": True, "derived_constituent_candidate": False, "weights_changed": False},
        {"candidate_id": "c2", "kind": "constituent_standalone_weight_0_25", "weights": {"b": 0.25}, "preexisting_frozen_candidate": False, "derived_constituent_candidate": True, "new_fixed_weight_assignment": 0.25},
    ]
    core = {
        "manifest_hash": "f" * 64,
        "source_policy_id": "G9-OVERLAP-NET-PORT-1",
        "frozen_inputs": {
            "original_validation_freeze": {"path": "orig.json", "sha256": "o", "manifest_hash": "om"},
            "holdout_dec2023_artifact": {"path": "holdout.json", "sha256": "h", "manifest_hash": "hm"},
            "test2024_terminal_artifact": {"path": "test.json", "sha256": "t", "manifest_hash": "tm"},
            "selected_clocks": [],
        },
        "stages": {
            "test2024": {"split": "test", "start": "2024-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
            "eval2025": {"split": "eval", "start": "2025-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"},
        },
        "candidate_family": candidates,
        "known_outcome_boundary": {
            "classification": "retrospective diagnostic revalidation",
            "known_current_rank1_candidate_id": "c1",
        },
        "ranking_rule": ["qualifiers first"],
        "gate_policy": {"waived_rejection_gates": ["turnover_cap", "sleeve_turnover_share_cap", "max_trade_frequency"]},
    }
    return core


def _clock(start: str) -> pd.DataFrame:
    t = pd.Timestamp(start)
    return pd.DataFrame({"entry_time": [t], "exit_time": [t + pd.Timedelta(hours=8)], "side": [1]})


def _primary(ret=2.0, ratio=4.0, mdd=2.0) -> dict[str, object]:
    return {
        "base": {"initial_equity": 100000.0, "final_equity": 102000.0, "absolute_return_pct": ret, "cagr_to_strict_mdd": ratio, "strict_mdd_pct": mdd, "mean_exposure_weighted_gross_edge_bp": 25.0, "intervals": 8, "long_intervals": 8, "short_intervals": 0, "transitions": 8, "total_fees": 10.0, "total_funding": 1.0},
        "stress": {"initial_equity": 100000.0, "final_equity": 101000.0, "absolute_return_pct": 1.0, "cagr_to_strict_mdd": 3.0, "strict_mdd_pct": 3.0, "mean_exposure_weighted_gross_edge_bp": 20.0, "intervals": 8, "long_intervals": 8, "short_intervals": 0, "transitions": 8, "total_fees": 12.0, "total_funding": 1.0},
        "calendar_halves": {"first": {"absolute_return_pct": 1.0}, "second": {"absolute_return_pct": 1.0}},
        "cluster_signflip": {"pvalue": 0.05},
    }


def test_build_candidate_family_freezes_64_plus_14_without_weight_changes() -> None:
    family = freeze_mod.build_candidate_family(_selection())
    assert len(family) == 78
    assert sum(row["kind"] == "frozen_exact_finalist" for row in family) == 64
    assert sum(row["kind"] == "constituent_standalone_weight_0_25" for row in family) == 14
    assert all(
        row["weights_changed"] is False
        for row in family
        if row["kind"] == "frozen_exact_finalist"
    )
    assert all(
        row["derived_constituent_candidate"] is True
        and row["new_fixed_weight_assignment"] == 0.25
        for row in family
        if row["kind"] == "constituent_standalone_weight_0_25"
    )
    assert all(list(row["weights"].values()) == [0.25] for row in family if row["kind"] == "constituent_standalone_weight_0_25")


def test_freeze_validate_rejects_rehashed_final_stage_or_override_drift() -> None:
    frozen = {
        "protocol_version": freeze_mod.PROTOCOL_VERSION,
        "policy_id": freeze_mod.POLICY_ID,
        "explicit_user_override": {"permits_test2024_and_eval2025_for_every_candidate_regardless_stage_failure": True, "final2026_open_authorized": False, "repair_or_weight_change_authorized": False, "selection_or_promotion_from_ordering_authorized": False},
        "candidate_counts": {"frozen_exact_finalists": 64, "constituent_standalone_weight_0_25": 14, "total": 78},
        "candidate_family": [{"candidate_id": f"c{i}", "kind": "frozen_exact_finalist", "preexisting_frozen_candidate": True, "derived_constituent_candidate": False, "weights_changed": False, "weights": {"s": 0.25}} for i in range(78)],
        "gate_policy": {"waived_rejection_gates": ["turnover_cap", "sleeve_turnover_share_cap", "max_trade_frequency"], "max_t": {"draws": 100000, "seed": 20260904}},
        "stages": {"test2024": {}, "eval2025": {}, "final2026": {}},
        "evidence_boundary": {"eval2025_opened_by_freeze": False, "final2026_opened": False},
    }
    frozen["manifest_hash"] = freeze_mod.canonical_hash(frozen)
    with pytest.raises(RuntimeError, match="stage boundary"):
        freeze_mod.validate(frozen)

    frozen["stages"].pop("final2026")
    frozen["explicit_user_override"]["repair_or_weight_change_authorized"] = True
    core = {k: v for k, v in frozen.items() if k != "manifest_hash"}
    frozen["manifest_hash"] = freeze_mod.canonical_hash(core)
    with pytest.raises(RuntimeError, match="override boundary"):
        freeze_mod.validate(frozen)


def test_run_blocks_sources_until_override_chain_verified(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(evaluator, "load_freeze", lambda path: _fake_freeze())
    monkeypatch.setattr(evaluator, "verify_terminal_override_chain", lambda freeze: (_ for _ in ()).throw(RuntimeError("bad override")))
    opened = False

    def should_not_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("opened")

    monkeypatch.setattr(evaluator, "load_all_stage_sources", should_not_open)
    with pytest.raises(RuntimeError, match="bad override"):
        evaluator.run(tmp_path / "out.json")
    assert opened is False


def test_evaluator_loads_sources_once_per_stage_and_never_uses_selector(monkeypatch, tmp_path) -> None:
    freeze = _fake_freeze()
    calls = []
    monkeypatch.setattr(evaluator, "load_freeze", lambda path: freeze)
    monkeypatch.setattr(evaluator, "verify_terminal_override_chain", lambda freeze: {"test2024_stop_explicitly_overridden": True})
    monkeypatch.setattr(evaluator.original_evaluator.optimizer, "beam_search_portfolios", lambda *a, **k: (_ for _ in ()).throw(AssertionError("selector called")), raising=False)
    monkeypatch.setattr(evaluator.original_evaluator.optimizer, "select_authoritative_rank1", lambda *a, **k: (_ for _ in ()).throw(AssertionError("selector called")), raising=False)

    def fake_load_sources(stage, start, end):
        calls.append(stage)
        return {"market": pd.DataFrame(), "funding": pd.DataFrame(), "source": {"stage": stage}, "start": start, "end": end, "split": freeze["stages"][stage]["split"]}

    monkeypatch.setattr(evaluator, "load_all_stage_sources", lambda freeze: {stage: fake_load_sources(stage, evaluator._utc(row["start"]), evaluator._utc(row["end"])) for stage, row in freeze["stages"].items()})
    monkeypatch.setattr(evaluator, "load_sleeve_clocks_once", lambda freeze: {"a": {"test2024": _clock("2024-01-01T00:00:00Z"), "eval2025": _clock("2025-01-01T00:00:00Z")}, "b": {"test2024": _clock("2024-01-01T00:00:00Z"), "eval2025": _clock("2025-01-01T00:00:00Z")}})
    monkeypatch.setattr(
        evaluator,
        "evaluate_primary_without_candidate_mc",
        lambda *a, **k: (
            _primary(),
            {
                "equity_effect_rows": [{"time": a[3], "log_effect": 0.01}],
                "transition_rows": [],
            },
        ),
    )
    monkeypatch.setattr(evaluator.original_evaluator.optimizer, "exposure_and_turnover", lambda *a, **k: {"turnover_weight": 1.0})
    monkeypatch.setattr(evaluator.original_evaluator.optimizer, "evaluate_monthly_stability", lambda *a, **k: [])
    monkeypatch.setattr(evaluator.net_config, "net_exposure_metrics", lambda *a, **k: {"mean_abs_net_position": 0.1, "max_abs_net_position": 0.25})
    monkeypatch.setattr(evaluator.original_evaluator, "active_iso_weeks", lambda clock: 4)
    monkeypatch.setattr(evaluator.original_evaluator, "aggregate_net_signed_episode_count", lambda clock: 99)
    monkeypatch.setattr(
        evaluator.original_evaluator,
        "operating_cost_disclosure",
        lambda *a, **k: {
            "current_rejection_gates": [],
            "nonzero_net_execution_events_per_day": 1.0,
        },
    )
    monkeypatch.setattr(evaluator, "aligned_week_max_t", lambda weekly: {"adjusted_pvalues": {cid: 0.05 for cid in weekly}, "draws": 100000, "seed": 20260904})
    monkeypatch.setattr(
        evaluator,
        "verify_known_rank1_replay",
        lambda freeze, reports: {"candidate_id": "c1", "exact_match": True},
    )

    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluator, "sha256_file", lambda path: "freeze-sha" if Path(path) == freeze_path else "sha")
    result = evaluator.run(tmp_path / "out.json", freeze_path=freeze_path)
    assert calls == ["test2024", "eval2025"]
    assert result["candidate_count"] == 2
    assert result["final2026_opened"] is False
    assert result["reranked_for_diagnostic_reporting"] is True
    assert result["selection_or_promotion_authorized"] is False
    assert result["repaired_or_substituted"] is False
    assert all(row["qualified"] for row in result["ranking"])
    for report in result["candidate_metrics"].values():
        for stage in report["stages"].values():
            assert stage["retained_activity_sufficiency"]["applied_as_rejection"] is True
            assert stage["waived_cost_and_max_frequency_disclosures"]["applied_as_rejection"] is False


def test_known_rank1_replay_failure_blocks_other_candidate_outcomes(monkeypatch, tmp_path) -> None:
    freeze = _fake_freeze()
    monkeypatch.setattr(evaluator, "load_freeze", lambda path: freeze)
    monkeypatch.setattr(
        evaluator,
        "verify_terminal_override_chain",
        lambda freeze: {"test2024_stop_explicitly_overridden": True},
    )
    monkeypatch.setattr(
        evaluator,
        "load_all_stage_sources",
        lambda freeze: {
            stage: {
                "market": pd.DataFrame(),
                "funding": pd.DataFrame(),
                "source": {},
                "start": evaluator._utc(row["start"]),
                "end": evaluator._utc(row["end"]),
                "split": row["split"],
            }
            for stage, row in freeze["stages"].items()
        },
    )
    monkeypatch.setattr(
        evaluator,
        "load_sleeve_clocks_once",
        lambda freeze: {
            "a": {stage: _clock(row["start"]) for stage, row in freeze["stages"].items()},
            "b": {stage: _clock(row["start"]) for stage, row in freeze["stages"].items()},
        },
    )
    evaluated = []

    def evaluate(candidate, clock, source):
        evaluated.append(candidate["candidate_id"])
        return {"primary": _primary()}, {}

    monkeypatch.setattr(evaluator, "evaluate_candidate_stage", evaluate)
    monkeypatch.setattr(
        evaluator,
        "verify_known_rank1_replay",
        lambda freeze, reports: (_ for _ in ()).throw(RuntimeError("known replay drift")),
    )

    with pytest.raises(RuntimeError, match="known replay drift"):
        evaluator.run(tmp_path / "out.json", freeze_path=tmp_path / "freeze.json")
    assert evaluated == ["c1", "c1"]


def test_stage_source_loader_uses_normalized_eval_and_redacts_receipt(monkeypatch) -> None:
    freeze = _fake_freeze()
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        }
    )
    funding = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "funding_rate": [0.0],
            "mark_price": [1.0],
        }
    )
    calls = []

    def original(stage, start, end):
        calls.append(("original", stage))
        return market, funding, {"database_identity": {"configured_host": "secret"}}

    def normalized(start, end):
        calls.append(("normalized", "eval2025"))
        return market, funding, {"database_identity": {"configured_host": "secret"}}

    monkeypatch.setattr(evaluator.original_evaluator, "load_stage_sources", original)
    monkeypatch.setattr(
        evaluator.normalized_eval_source,
        "load_normalized_eval2025_sources",
        normalized,
    )
    bundles = evaluator.load_all_stage_sources(freeze)

    assert calls == [("original", "test2024"), ("normalized", "eval2025")]
    for bundle in bundles.values():
        assert "configured_host" not in bundle["source"]["database_identity"]


def test_retained_checks_waive_frequency_and_turnover_max_but_keep_activity_and_performance() -> None:
    checks = evaluator.retained_checks(_primary(), {"mean_abs_net_position": 0.1, "max_abs_net_position": 0.2}, intervals=8, active_weeks=4, signed_episodes=12, max_t_pvalue=0.10)
    assert all(checks.values())
    assert "turnover_cap" not in checks
    assert "sleeve_turnover_share_cap" not in checks
    assert "max_trade_frequency" not in checks
    assert "minimum_intervals" in checks
    assert "minimum_active_iso_weeks" in checks
    assert "minimum_aggregate_net_signed_episodes" in checks

    insufficient = evaluator.retained_checks(
        _primary(),
        {"mean_abs_net_position": 0.1, "max_abs_net_position": 0.2},
        intervals=8,
        active_weeks=4,
        signed_episodes=11,
        max_t_pvalue=0.10,
    )
    assert insufficient["minimum_aggregate_net_signed_episodes"] is False


def test_ranking_rule_uses_min_ratio_then_log_return_then_worst_mdd() -> None:
    reports = {
        "b": {"stages": {"test2024": {"primary": _primary(ratio=4, mdd=5)}, "eval2025": {"primary": _primary(ratio=3, mdd=2)}}, "checks": {"test2024": {"x": True}, "eval2025": {"x": True}}},
        "a": {"stages": {"test2024": {"primary": _primary(ratio=5, mdd=2)}, "eval2025": {"primary": _primary(ratio=4, mdd=3)}}, "checks": {"test2024": {"x": True}, "eval2025": {"x": True}}},
        "c": {"stages": {"test2024": {"primary": _primary(ratio=9, mdd=1)}, "eval2025": {"primary": _primary(ratio=9, mdd=1)}}, "checks": {"test2024": {"x": False}, "eval2025": {"x": True}}},
    }
    ranking = evaluator.rank_candidates(reports)
    assert [row["candidate_id"] for row in ranking] == ["a", "b", "c"]
    assert ranking[-1]["qualified"] is False


def test_max_t_is_deterministic_and_aligned() -> None:
    weekly = {"b": {(2024, 1): 0.1, (2024, 3): 0.2}, "a": {(2024, 2): 0.05}}
    first = evaluator.aligned_week_max_t(weekly, draws=500, seed=3)
    second = evaluator.aligned_week_max_t(weekly, draws=500, seed=3)
    assert first == second
    assert first["weeks"] == 3
    assert set(first["adjusted_pvalues"]) == {"a", "b"}
