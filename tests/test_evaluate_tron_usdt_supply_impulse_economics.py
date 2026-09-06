from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Mapping

import pandas as pd
import pytest

from training import evaluate_tron_usdt_supply_impulse_economics as e


def _market(
    start: str,
    end: str,
    *,
    overrides: Mapping[str, tuple[float, float, float]] | None = None,
) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="5min", tz="UTC")
    rows = []
    changes = overrides or {}
    for date in dates:
        open_, high, low = changes.get(
            date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            (100.0, 100.0, 100.0),
        )
        rows.append((date, open_, high, low))
    return pd.DataFrame(rows, columns=e.MARKET_COLUMNS)


def _funding(
    rows: list[tuple[str, float, float]] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(rows or [], columns=e.FUNDING_COLUMNS)


def _clock(
    rows: list[tuple[str, str, int]],
) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=e.CLOCK_COLUMNS)


def _with_manifest(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": e.canonical_hash(core)}


def _support(*, passed: bool = True) -> dict[str, Any]:
    controls = ("primary",) + e.CONTROL_NAMES
    periods = (
        "selection",
        "2023H2",
        "2024",
        "2024H1",
        "2024H2",
        "future25",
        "2025H1",
        "2025H2",
        "future26",
        "full",
    )
    core = {
        "protocol_version": e.SOURCE_SUPPORT_PROTOCOL_VERSION,
        "policy_id": e.POLICY_ID,
        "status": "source_support_passed" if passed else "retired_before_novelty",
        "terminal": True,
        "artifact_eligible": True,
        "support_passed": passed,
        "decision": "SOURCE_SUPPORT_PASS" if passed else "RETIRE_TUSI_168_UNCHANGED_BEFORE_NOVELTY",
        "registration": {
            "manifest_hash": e.PREREGISTRATION_MANIFEST_HASH,
            "mode": "artifact",
        },
        "source_contract": {},
        "raw_candidate_counts": {name: 0 for name in controls},
        "accepted_clock_counts": {name: 0 for name in controls},
        "period_diagnostics": {name: {} for name in periods},
        "support_audit": {},
        "support_checks": {"all": passed},
        "future_append_selection_invariance": {},
        "control_overlap": {},
        "clock_artifacts": {
            "primary_sha256": "1" * 64,
            "controls_sha256": "2" * 64,
        },
        "evidence_boundary": {},
        "source_support_precedes_novelty": True,
        "novelty_comparator_market_or_outcome_artifacts_opened": False,
    }
    return _with_manifest(core)


def _novelty(support: Mapping[str, Any], *, passed: bool = True) -> dict[str, Any]:
    core = {
        "protocol_version": e.NOVELTY_PROTOCOL_VERSION,
        "policy_id": e.POLICY_ID,
        "source_support": {
            "path": e.SOURCE_SUPPORT_ARTIFACT.as_posix(),
            "sha256": "3" * 64,
            "manifest_hash": support["manifest_hash"],
        },
        "novelty": {
            "passed": passed,
            "terminal": not passed,
            "failed_checks": [] if passed else ["overlap"],
        },
        "evidence_boundary": {
            "candidate_market_rows_opened": False,
            "candidate_funding_rows_opened": False,
            "candidate_outcome_rows_opened": False,
            "candidate_returns_or_pnl_computed": False,
            "portfolio_return_or_pnl_metrics_computed": False,
        },
    }
    return _with_manifest(core)


def _ranking(weight: float = 0.5) -> list[dict[str, Any]]:
    def row(candidate: float) -> dict[str, Any]:
        improvement = 0.30 if candidate == weight else 0.10 - candidate / 100.0

        def cost_row() -> dict[str, Any]:
            treatment = {
                "absolute_return": 0.5,
                "cagr_to_strict_mdd": 1.0 + improvement,
                "strict_mdd": 0.09,
                "liquidation_safe": True,
            }
            baseline = {
                "absolute_return": 0.5,
                "cagr_to_strict_mdd": 1.0,
                "strict_mdd": 0.1,
                "liquidation_safe": True,
            }
            return {
                "treatment": treatment,
                "unscaled_gross9": baseline,
                "checks": e.esdi._same_gross_period_checks(
                    treatment,
                    baseline,
                ),
                "improvement": improvement,
            }

        return {
            "candidate_weight": candidate,
            "treatment_weights": e.same_gross_weights(candidate),
            "baseline_weights": dict(e.GROSS9_WEIGHTS),
            "fresh_evaluation": True,
            "period_order": list(e.SELECTION_PERIODS),
            "strict_mdd_reduced_in_at_least_one_period": True,
            "periods": {
                period: {
                    cost: cost_row()
                    for cost in ("base", "stress")
                }
                for period in e.SELECTION_PERIODS
            },
        }

    return e.rank_same_gross_treatments(
        [row(candidate) for candidate in e.CANDIDATE_WEIGHTS]
    )


def _standalone_period(passed: bool, *, trades: int) -> dict[str, Any]:
    metrics = {
        "absolute_return": 0.2 if passed else -0.1,
        "cagr_to_strict_mdd": 4.0 if passed else 0.0,
        "strict_mdd": 0.1 if passed else 0.2,
        "mean_gross_underlying_bp": 30.0 if passed else 0.0,
        "calendar_month_clustered_signflip": {
            "p_value_one_sided": 0.05 if passed else 1.0,
        },
        "liquidation_safe": True,
        "trades": trades,
    }
    checks = e.standalone_gate_checks(metrics)
    cost_row = {
        "metrics": metrics,
        "checks": checks,
        "passes": all(checks.values()),
    }
    return {
        "base": copy.deepcopy(cost_row),
        "stress": copy.deepcopy(cost_row),
        "passes": passed,
    }


def _standalone_summary(passed: bool) -> dict[str, Any]:
    primary = _standalone_period(passed, trades=2)
    controls = {
        name: _standalone_period(False, trades=0)
        for name in e.CONTROL_NAMES
    }
    superiority = e.evaluate_primary_superiority(primary, controls)
    return {
        "primary": primary,
        "controls": controls,
        "primary_superiority": superiority,
        "one_bar_delayed_entry_is_diagnostic_only": True,
        "passes": bool(primary["passes"] and superiority["passes"]),
    }


def _future_same_gross(passed: bool, weight: float) -> dict[str, Any]:
    treatment = {
        "absolute_return": 0.5 if passed else -0.1,
        "cagr_to_strict_mdd": 1.1 if passed else 0.0,
        "strict_mdd": 0.09 if passed else 0.2,
        "liquidation_safe": True,
    }
    baseline = {
        "absolute_return": 0.5,
        "cagr_to_strict_mdd": 1.0,
        "strict_mdd": 0.1,
        "liquidation_safe": True,
    }
    costs = {
        cost: {
            "treatment": copy.deepcopy(treatment),
            "unscaled_gross9": copy.deepcopy(baseline),
            "checks": e.esdi._same_gross_period_checks(
                treatment,
                baseline,
            ),
        }
        for cost in ("base", "stress")
    }
    return {
        "passes": passed,
        "candidate_weight": weight,
        "costs": costs,
        "strict_mdd_reduced": passed,
    }


def _stage_result(
    stage: str,
    *,
    passed: bool,
    state: Mapping[str, Any],
    frozen_weight: float = 0.5,
) -> dict[str, Any]:
    standalone = _standalone_summary(passed)
    if stage in {"2023H2", "2024", "selection"}:
        return {"passed": passed, "standalone": standalone}
    if stage == "same_gross":
        ranking = _ranking(frozen_weight)
        return {"passed": passed, "ranking": ranking}
    weight = float(state["frozen_weight"])
    return {
        "passed": passed,
        "frozen_weight": weight,
        "reranked": False,
        "standalone": standalone,
        "same_gross": _future_same_gross(passed, weight),
    }


def _run_all_synthetic_stages(root: Path) -> dict[str, Any]:
    root.mkdir(exist_ok=True)
    support = _support()

    def evaluator(
        stage: str, _inputs: Mapping[str, Any], state: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if stage == "same_gross":
            return _stage_result(
                stage,
                passed=True,
                state=state,
                frozen_weight=0.5,
            )
        return _stage_result(stage, passed=True, state=state)

    return e.run_staged_economics(
        synthetic=True,
        source_support=support,
        novelty=_novelty(support),
        stage_loader=lambda _stage, _cutoff: {},
        stage_evaluator=evaluator,
        receipt_root=root,
    )


def _rewrite_manifest(path: Path, payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = e.canonical_hash(core)
    os.chmod(path, 0o600)
    path.write_bytes(e.canonical_json_bytes(payload))
    os.chmod(path, 0o444)


def test_provenance_hashes_and_strict_helper_bindings_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = e.load_bound_preregistration()
    validation = e.validate_frozen_contract(registration)

    assert e.sha256_file(e.PREREGISTRATION_ARTIFACT) == (
        "54817044b8df76dc347ed64b6fe5f6f2dfdddcdb211bded4ba2b1af133d49067"
    )
    assert registration["manifest_hash"] == (
        "d67cd1b67632ae92e9458395e729627a6f4c3b4b75ce97187653eac3a09e40c1"
    )
    assert validation["esdi_economics_sha256"] == (
        "fba7de6a26ede945edfe63c32dd4a0c88760c6459ac0d4f079dd12d546580235"
    )
    assert e.calendar_month_clustered_signflip is (
        e.esdi.calendar_month_clustered_signflip
    )
    assert e._simulate_portfolio is e.esdi._simulate_portfolio
    assert e._full_calendar_cagr is e.esdi._full_calendar_cagr
    assert e._calendar_years is e.esdi._calendar_years

    poisoned = types.ModuleType(
        "training.evaluate_ethereum_settlement_demand_impulse_economics"
    )
    poisoned.__file__ = str(
        e.REPOSITORY_ROOT / e.ESDI_ECONOMICS_AUTHORITY_PATH
    )
    setattr(poisoned, "LEVERAGE", 999)
    monkeypatch.setitem(sys.modules, poisoned.__name__, poisoned)
    authority_reads = 0
    read_regular = e._read_regular

    def tracked_read(path: Path) -> bytes:
        nonlocal authority_reads
        if path == e.ESDI_ECONOMICS_AUTHORITY_PATH:
            authority_reads += 1
        return read_regular(path)

    monkeypatch.setattr(e, "_read_regular", tracked_read)
    fresh, _ = e._load_esdi_authority()
    assert fresh is not poisoned
    assert fresh.LEVERAGE == 0.5
    assert authority_reads == 1


def test_forged_preregistration_and_phase_reports_fail_closed() -> None:
    registration = copy.deepcopy(e.load_bound_preregistration())
    registration["gross9"]["candidate_weights"] = [0.25]
    with pytest.raises(RuntimeError, match="manifest"):
        e.validate_frozen_contract(registration)

    support = _support()
    forged_support = copy.deepcopy(support)
    forged_support["support_checks"]["all"] = False
    with pytest.raises(RuntimeError, match="manifest"):
        e.validate_passed_source_support(forged_support, exact=False)

    novelty = _novelty(support)
    forged_novelty = _with_manifest(
        {
            **{key: value for key, value in novelty.items() if key != "manifest_hash"},
            "novelty": {"passed": True, "terminal": False, "failed_checks": ["x"]},
        }
    )
    with pytest.raises(RuntimeError, match="failed"):
        e.validate_passed_novelty(forged_novelty, exact=False)


def test_novelty_attempt_claim_is_bound_by_exact_committed_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.SimpleNamespace(
        DEFAULT_ATTEMPT_CLAIM_PATH=Path("results/fake_novelty_attempt.json"),
        ATTEMPT_CLAIM_PROTOCOL_VERSION=(
            "tron_usdt_supply_impulse_novelty_attempt_claim_v1"
        ),
        DEFAULT_PRIMARY_CLOCK_PATH=e.PRIMARY_CLOCK_ARTIFACT,
        DEFAULT_GROSS9_CLOCKS_PATH=Path("results/gross9.json"),
        DEFAULT_OUTPUT_PATH=e.NOVELTY_ARTIFACT,
    )
    preregistration = {
        "path": e.PREREGISTRATION_ARTIFACT.as_posix(),
        "sha256": e.PREREGISTRATION_ARTIFACT_SHA256,
        "manifest_hash": e.PREREGISTRATION_MANIFEST_HASH,
    }
    support = {
        "path": e.SOURCE_SUPPORT_ARTIFACT.as_posix(),
        "sha256": "1" * 64,
        "manifest_hash": "2" * 64,
        "passed": True,
    }
    candidate = {
        "path": e.PRIMARY_CLOCK_ARTIFACT.as_posix(),
        "sha256": "3" * 64,
        "accepted_intervals": 8,
    }
    gross9 = {
        "path": "results/gross9.json",
        "sha256": "4" * 64,
        "manifest_hash": "5" * 64,
        "authority_hash": "6" * 64,
    }
    claim_core = {
        "protocol_version": module.ATTEMPT_CLAIM_PROTOCOL_VERSION,
        "policy_id": e.POLICY_ID,
        "status": "claimed_before_comparator_access",
        "one_shot": True,
        "retry_or_repair_after_failure": False,
        "preregistration": preregistration,
        "source_support": {
            key: support[key] for key in ("path", "sha256", "manifest_hash")
        },
        "candidate_clock": {
            "path": e.PRIMARY_CLOCK_ARTIFACT.as_posix(),
            "sha256": candidate["sha256"],
        },
        "gross9_clock_artifact": gross9,
        "canonical_output": e.NOVELTY_ARTIFACT.as_posix(),
    }
    claim: dict[str, Any] = {
        **claim_core,
        "claim_hash": e.canonical_hash(claim_core),
    }
    raw = e.canonical_json_bytes(claim)
    report = {
        "preregistration": preregistration,
        "source_support": support,
        "candidate_clock": candidate,
        "gross9_clock_artifact": gross9,
        "attempt_claim": {
            "path": module.DEFAULT_ATTEMPT_CLAIM_PATH.as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "claim_hash": claim["claim_hash"],
        },
    }
    monkeypatch.setattr(e, "_assert_committed_clean", lambda _path: None)
    monkeypatch.setattr(e, "_read_regular", lambda _path: raw)
    e._validate_novelty_attempt_claim(report, novelty_module=module)

    forged = copy.deepcopy(report)
    forged["attempt_claim"]["sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="attempt/report"):
        e._validate_novelty_attempt_claim(forged, novelty_module=module)

    aliased = copy.deepcopy(report)
    aliased_claim = copy.deepcopy(claim)
    aliased_path = (
        "results/../results/"
        "tron_usdt_supply_impulse_source_support_2026-07-30.json"
    )
    aliased["source_support"]["path"] = aliased_path
    aliased_claim["source_support"]["path"] = aliased_path
    aliased_core = {
        key: value
        for key, value in aliased_claim.items()
        if key != "claim_hash"
    }
    aliased_claim["claim_hash"] = e.canonical_hash(aliased_core)
    raw = e.canonical_json_bytes(aliased_claim)
    aliased["attempt_claim"]["sha256"] = hashlib.sha256(raw).hexdigest()
    aliased["attempt_claim"]["claim_hash"] = aliased_claim["claim_hash"]
    with pytest.raises(RuntimeError, match="attempt/report"):
        e._validate_novelty_attempt_claim(aliased, novelty_module=module)


def test_strict_mdd_cost_and_funding_are_authority_results() -> None:
    market = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:10:00Z",
        overrides={
            "2023-01-01T00:00:00Z": (100.0, 120.0, 80.0),
            "2023-01-01T00:10:00Z": (110.0, 110.0, 110.0),
        },
    )
    funding = _funding(
        [
            ("2023-01-01T00:00:00Z", 0.01, 100.0),
            ("2023-01-01T00:05:00Z", -0.01, 110.0),
            ("2023-01-01T00:10:00Z", 9.0, 110.0),
        ]
    )
    result = e.simulate_portfolio(
        market,
        funding,
        {"tusi": _clock([("2023-01-01T00:00:00Z", "2023-01-01T00:10:00Z", 1)])},
        {"tusi": 1.0},
        start="2023-01-01T00:00:00Z",
        end="2023-01-01T00:10:00Z",
        cost_rate=e.BASE_COST_RATE,
        synthetic=True,
    )
    trade = result["trade_records"][0]
    quantity = 0.5 / 100.0
    assert trade["entry_cost"] == pytest.approx(quantity * 100.0 * 0.0006)
    assert trade["exit_cost"] == pytest.approx(quantity * 110.0 * 0.0006)
    assert trade["funding_cash"] == pytest.approx(
        -quantity * 0.01 * 100.0 + quantity * 0.01 * 110.0
    )
    assert result["path"][0]["hwm"] > 1.0
    assert result["strict_mdd"] > 0.0


def test_full_calendar_cagr_and_signflip_are_reproducible() -> None:
    assert e._calendar_years(*e.PERIODS["full"]) == 3.0
    assert e._full_calendar_cagr(1.21, 3.0) == pytest.approx(
        1.21 ** (1.0 / 3.0) - 1.0
    )
    trades = [
        {
            "entry_time": pd.Timestamp("2021-01-01T00:00:00Z")
            + pd.DateOffset(months=index),
            "net_return_on_allocated_equity": -0.02 if index % 3 == 0 else 0.03,
        }
        for index in range(21)
    ]
    first = e.calendar_month_clustered_signflip(trades)
    second = e.calendar_month_clustered_signflip(trades)
    assert first == second
    assert first["seed"] == 20260730
    assert first["samples"] == 10_000
    assert first["method"] == "monte_carlo"


def test_same_gross_preserves_nine_and_uses_tusi_name() -> None:
    weights = e.same_gross_weights(0.75)
    assert tuple(weights) == (*e.GROSS9_SLEEVES, "tusi")
    assert weights["tusi"] == 0.75
    assert sum(weights.values()) == pytest.approx(9.0)
    scale = (9.0 - 0.75) / 9.0
    for sleeve, value in e.GROSS9_WEIGHTS.items():
        assert weights[sleeve] == pytest.approx(value * scale)


def test_rank_and_future_are_selection_frozen_and_veto_only() -> None:
    def row(weight: float, improvement: float) -> dict[str, Any]:
        return {
            "candidate_weight": weight,
            "treatment_weights": e.same_gross_weights(weight),
            "baseline_weights": dict(e.GROSS9_WEIGHTS),
            "fresh_evaluation": True,
            "period_order": list(e.SELECTION_PERIODS),
            "periods": {
                period: {
                    cost: {
                        "treatment": {
                            "absolute_return": 0.5,
                            "cagr_to_strict_mdd": 1.0 + improvement,
                            "strict_mdd": 0.09,
                            "liquidation_safe": True,
                        },
                        "unscaled_gross9": {
                            "absolute_return": 0.5,
                            "cagr_to_strict_mdd": 1.0,
                            "strict_mdd": 0.1,
                            "liquidation_safe": True,
                        },
                    }
                    for cost in ("base", "stress")
                }
                for period in e.SELECTION_PERIODS
            },
        }

    ranked = e.rank_same_gross_treatments(
        [
            row(0.25, 0.10),
            row(0.50, 0.20),
            row(0.75, 0.20),
            row(1.00, 0.30),
        ]
    )
    frozen = ranked[0]
    assert frozen["candidate_weight"] == 1.0
    assert frozen["rank"] == 1 and frozen["frozen"] is True
    result = e.future_veto(
        frozen,
        {
            "future25": {"candidate_weight": 1.0, "passes": True},
            "future26": {"candidate_weight": 1.0, "passes": False},
        },
        synthetic=True,
    )
    assert result["passes"] is False
    assert result["reranked"] is False
    with pytest.raises(RuntimeError, match="rerank"):
        e.future_veto(
            frozen,
            {
                "future25": {"candidate_weight": 0.75, "passes": True},
                "future26": {"candidate_weight": 1.0, "passes": True},
            },
            synthetic=True,
        )


def test_unexpected_authority_ranking_runtime_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def synthetic_weight(
        _market: pd.DataFrame,
        _funding: pd.DataFrame,
        _gross9_clocks: Mapping[str, pd.DataFrame],
        _tusi_clock: pd.DataFrame,
        candidate_weight: float,
        *,
        periods: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
        synthetic: bool = False,
    ) -> dict[str, Any]:
        assert tuple(periods) == tuple(e.SELECTION_PERIODS)
        assert synthetic is True
        return {"candidate_weight": candidate_weight}

    def unexpected_failure(
        _rows: object,
        *,
        require_passing_freeze: bool,
    ) -> list[dict[str, Any]]:
        assert require_passing_freeze is False
        raise RuntimeError("unexpected authority failure")

    monkeypatch.setattr(e, "evaluate_same_gross_weight", synthetic_weight)
    monkeypatch.setattr(
        e.esdi,
        "_rank_same_gross_treatments",
        unexpected_failure,
    )
    empty = pd.DataFrame()
    with pytest.raises(RuntimeError, match="unexpected authority failure"):
        e._production_stage_evaluator(
            "same_gross",
            {
                "market": empty,
                "funding": empty,
                "primary_clock": empty,
                "control_clocks": {},
                "gross9_clocks": {},
            },
            {},
        )


def test_source_support_failure_stops_before_novelty_or_rows(tmp_path: Path) -> None:
    calls: list[str] = []
    support = _support(passed=False)
    with pytest.raises(RuntimeError, match="support"):
        e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support),
            stage_loader=lambda stage, _cutoff: calls.append(stage) or {},
            stage_evaluator=lambda stage, _inputs, state: _stage_result(
                stage, passed=True, state=state
            ),
            receipt_root=tmp_path,
        )
    assert calls == []


def test_novelty_failure_stops_before_any_economic_loader(tmp_path: Path) -> None:
    calls: list[str] = []
    support = _support()
    with pytest.raises(RuntimeError, match="novelty"):
        e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support, passed=False),
            stage_loader=lambda stage, _cutoff: calls.append(stage) or {},
            stage_evaluator=lambda stage, _inputs, state: _stage_result(
                stage, passed=True, state=state
            ),
            receipt_root=tmp_path,
        )
    assert calls == []


def test_stage_failure_is_terminal_and_never_opens_future(tmp_path: Path) -> None:
    support = _support()
    calls: list[str] = []

    def evaluator(
        stage: str, _inputs: Mapping[str, Any], _state: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        passed = stage != "2024"
        return _stage_result(stage, passed=passed, state=_state)

    result = e.run_staged_economics(
        synthetic=True,
        source_support=support,
        novelty=_novelty(support),
        stage_loader=lambda stage, _cutoff: calls.append(stage) or {},
        stage_evaluator=evaluator,
        receipt_root=tmp_path,
    )
    assert result == {
        "passed": False,
        "terminal": True,
        "stopped_at": "2024",
        "completed_stages": ["2023H2", "2024"],
    }
    assert calls == ["2023H2", "2024"]


def test_attempt_is_durable_before_loader_and_abandoned_claim_cannot_retry(
    tmp_path: Path,
) -> None:
    support = _support()
    loader_calls = 0

    def loader(stage: str, _cutoff: pd.Timestamp) -> Mapping[str, Any]:
        nonlocal loader_calls
        loader_calls += 1
        attempt_path = tmp_path / e.STAGE_ATTEMPT_NAMES[stage]
        assert attempt_path.is_file()
        attempt = json.loads(attempt_path.read_bytes())
        assert attempt["status"] == "claimed_before_stage_rows"
        raise RuntimeError("synthetic loader stopped")

    with pytest.raises(RuntimeError, match="loader stopped"):
        e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support),
            stage_loader=loader,
            stage_evaluator=lambda stage, _inputs, state: _stage_result(
                stage,
                passed=True,
                state=state,
            ),
            receipt_root=tmp_path,
        )
    assert loader_calls == 1
    assert not (tmp_path / e.STAGE_RECEIPT_NAMES["2023H2"]).exists()

    with pytest.raises(RuntimeError, match="retry or repair is forbidden"):
        e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support),
            stage_loader=loader,
            stage_evaluator=lambda stage, _inputs, state: _stage_result(
                stage,
                passed=True,
                state=state,
            ),
            receipt_root=tmp_path,
        )
    assert loader_calls == 1


def test_future_cannot_change_frozen_weight_or_rerank(tmp_path: Path) -> None:
    support = _support()

    def evaluator(
        stage: str, _inputs: Mapping[str, Any], state: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if stage == "same_gross":
            return _stage_result(
                stage, passed=True, state=state, frozen_weight=0.5
            )
        if stage == "future25":
            return {
                "passed": True,
                "frozen_weight": 0.75,
                "reranked": False,
                "standalone": _standalone_summary(True),
                "same_gross": _future_same_gross(True, 0.75),
            }
        return _stage_result(stage, passed=True, state=state)

    with pytest.raises(RuntimeError, match="changed weight"):
        e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support),
            stage_loader=lambda _stage, _cutoff: {},
            stage_evaluator=evaluator,
            receipt_root=tmp_path,
        )
    assert not (tmp_path / e.STAGE_RECEIPT_NAMES["future25"]).exists()


def test_write_once_returns_exact_digest_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    expected = hashlib.sha256(e.canonical_json_bytes({"x": 1})).hexdigest()
    created = e.write_once_result(path, {"x": 1}, root=tmp_path)
    assert created.status == "created"
    assert created.sha256 == expected
    existing = e.write_once_result(path, {"x": 1}, root=tmp_path)
    assert existing.status == "verified_existing"
    assert existing.sha256 == expected
    with pytest.raises(RuntimeError, match="differs"):
        e.write_once_result(path, {"x": 2}, root=tmp_path)


def test_staged_receipts_are_byte_reproducible(tmp_path: Path) -> None:
    support = _support()

    def evaluator(
        stage: str, _inputs: Mapping[str, Any], state: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if stage == "same_gross":
            return _stage_result(
                stage, passed=True, state=state, frozen_weight=0.5
            )
        return _stage_result(stage, passed=True, state=state)

    roots = (tmp_path / "one", tmp_path / "two")
    outputs = []
    for root in roots:
        root.mkdir()
        result = e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support),
            stage_loader=lambda _stage, _cutoff: {},
            stage_evaluator=evaluator,
            receipt_root=root,
        )
        assert result["passed"] is True
        outputs.append(
            [
                hashlib.sha256((root / e.STAGE_RECEIPT_NAMES[stage]).read_bytes()).hexdigest()
                for stage in e.ECONOMIC_STAGE_ORDER
            ]
        )
    assert outputs[0] == outputs[1]


def test_fresh_chain_uses_publication_digest_after_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "fresh-replacement"
    real_write = e.write_once_result
    first_receipt = e.STAGE_RECEIPT_NAMES["2023H2"]
    published_sha256: str | None = None
    replacement_raw = e.canonical_json_bytes({"replacement": "fresh"})

    def replace_after_publication(
        path: str | Path,
        payload: Mapping[str, Any],
        *,
        root: str | Path,
        production: bool = False,
    ) -> e._Publication:
        nonlocal published_sha256
        publication = real_write(
            path,
            payload,
            root=root,
            production=production,
        )
        if Path(path).name == first_receipt and published_sha256 is None:
            published_sha256 = publication.sha256
            target = Path(path)
            if not target.is_absolute():
                target = Path(root) / target
            replacement = target.with_name(f".{target.name}.replacement")
            replacement.write_bytes(replacement_raw)
            os.replace(replacement, target)
        return publication

    monkeypatch.setattr(e, "write_once_result", replace_after_publication)
    assert _run_all_synthetic_stages(root)["passed"] is True
    assert published_sha256 is not None
    next_attempt = json.loads(
        (root / e.STAGE_ATTEMPT_NAMES["2024"]).read_bytes()
    )
    assert next_attempt["prior_receipt_sha256"] == published_sha256
    assert (root / first_receipt).read_bytes() == replacement_raw


def test_resume_chain_uses_validated_open_bytes_after_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "resume-replacement"
    assert _run_all_synthetic_stages(root)["passed"] is True
    first_receipt = e.STAGE_RECEIPT_NAMES["2023H2"]
    first_path = root / first_receipt
    validated_raw = first_path.read_bytes()
    validated_sha256 = hashlib.sha256(validated_raw).hexdigest()
    replacement_raw = e.canonical_json_bytes({"replacement": "resume"})
    real_read = e._read_output_bytes
    receipt_reads = 0

    def replace_after_open(
        path: str | Path,
        *,
        root: str | Path,
        production: bool,
    ) -> bytes:
        nonlocal receipt_reads
        raw = real_read(path, root=root, production=production)
        if Path(path).name == first_receipt:
            receipt_reads += 1
            if receipt_reads == 1:
                target = Path(root) / first_receipt
                replacement = target.with_name(f".{target.name}.replacement")
                replacement.write_bytes(replacement_raw)
                os.replace(replacement, target)
        return raw

    monkeypatch.setattr(e, "_read_output_bytes", replace_after_open)
    support = _support()
    resumed = e.run_staged_economics(
        synthetic=True,
        source_support=support,
        novelty=_novelty(support),
        stage_loader=lambda _stage, _cutoff: pytest.fail(
            "valid resume opened stage rows"
        ),
        stage_evaluator=lambda stage, _inputs, state: _stage_result(
            stage,
            passed=True,
            state=state,
        ),
        receipt_root=root,
    )
    assert resumed["passed"] is True
    assert receipt_reads == 1
    next_attempt = json.loads(
        (root / e.STAGE_ATTEMPT_NAMES["2024"]).read_bytes()
    )
    assert next_attempt["prior_receipt_sha256"] == validated_sha256
    assert first_path.read_bytes() == replacement_raw


def test_resume_rejects_rehashed_same_gross_weight_forgery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resume-ranking"
    assert _run_all_synthetic_stages(root)["passed"] is True
    receipt_path = root / e.STAGE_RECEIPT_NAMES["same_gross"]
    receipt = json.loads(receipt_path.read_bytes())
    receipt["result"]["ranking"][0]["candidate_weight"] = 0.75
    receipt["frozen_weight"] = 0.75
    _rewrite_manifest(receipt_path, receipt)

    support = _support()
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="ranking"):
        e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support),
            stage_loader=lambda stage, _cutoff: calls.append(stage) or {},
            stage_evaluator=lambda stage, _inputs, state: _stage_result(
                stage, passed=True, state=state
            ),
            receipt_root=root,
        )
    assert calls == []


def test_resume_rejects_rehashed_attempt_and_receipt_binding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resume-attempt"
    assert _run_all_synthetic_stages(root)["passed"] is True
    stage = "same_gross"
    attempt_path = root / e.STAGE_ATTEMPT_NAMES[stage]
    receipt_path = root / e.STAGE_RECEIPT_NAMES[stage]
    attempt = json.loads(attempt_path.read_bytes())
    attempt["prior_receipt_sha256"] = "f" * 64
    _rewrite_manifest(attempt_path, attempt)
    attempt_raw = attempt_path.read_bytes()

    receipt = json.loads(receipt_path.read_bytes())
    receipt["attempt_claim"] = {
        "path": e.STAGE_ATTEMPT_NAMES[stage],
        "sha256": hashlib.sha256(attempt_raw).hexdigest(),
        "manifest_hash": attempt["manifest_hash"],
        "content": attempt,
    }
    _rewrite_manifest(receipt_path, receipt)

    support = _support()
    with pytest.raises(RuntimeError, match="attempt content"):
        e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support),
            stage_loader=lambda _stage, _cutoff: pytest.fail(
                "resume opened rows after forged attempt"
            ),
            stage_evaluator=lambda stage, _inputs, state: _stage_result(
                stage, passed=True, state=state
            ),
            receipt_root=root,
        )


def test_resume_rejects_rehashed_attempt_scalar_type_forgery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resume-attempt-type"
    assert _run_all_synthetic_stages(root)["passed"] is True
    stage = "2023H2"
    attempt_path = root / e.STAGE_ATTEMPT_NAMES[stage]
    receipt_path = root / e.STAGE_RECEIPT_NAMES[stage]
    attempt = json.loads(attempt_path.read_bytes())
    attempt["one_shot"] = 1
    _rewrite_manifest(attempt_path, attempt)
    attempt_raw = attempt_path.read_bytes()

    receipt = json.loads(receipt_path.read_bytes())
    receipt["attempt_claim"] = {
        "path": e.STAGE_ATTEMPT_NAMES[stage],
        "sha256": hashlib.sha256(attempt_raw).hexdigest(),
        "manifest_hash": attempt["manifest_hash"],
        "content": attempt,
    }
    _rewrite_manifest(receipt_path, receipt)

    support = _support()
    with pytest.raises(RuntimeError, match="attempt content"):
        e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support),
            stage_loader=lambda _stage, _cutoff: pytest.fail(
                "type-forged attempt opened rows"
            ),
            stage_evaluator=lambda stage, _inputs, state: _stage_result(
                stage,
                passed=True,
                state=state,
            ),
            receipt_root=root,
        )


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("cutoff_exclusive", "2099-01-01T00:00:00+00:00"),
        ("prior_receipt_sha256", "a" * 64),
        ("novelty_manifest_hash", "b" * 64),
    ],
)
def test_resume_rejects_rehashed_receipt_chain_fields(
    tmp_path: Path,
    field: str,
    forged: str,
) -> None:
    root = tmp_path / field
    assert _run_all_synthetic_stages(root)["passed"] is True
    receipt_path = root / e.STAGE_RECEIPT_NAMES["2023H2"]
    receipt = json.loads(receipt_path.read_bytes())
    receipt[field] = forged
    _rewrite_manifest(receipt_path, receipt)

    support = _support()
    with pytest.raises(RuntimeError, match="receipt binding"):
        e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support),
            stage_loader=lambda _stage, _cutoff: pytest.fail(
                "forged resume opened rows"
            ),
            stage_evaluator=lambda stage, _inputs, state: _stage_result(
                stage, passed=True, state=state
            ),
            receipt_root=root,
        )


def test_resume_rejects_rehashed_future_weight_forgery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resume-future"
    assert _run_all_synthetic_stages(root)["passed"] is True
    receipt_path = root / e.STAGE_RECEIPT_NAMES["future25"]
    receipt = json.loads(receipt_path.read_bytes())
    receipt["frozen_weight"] = 0.75
    receipt["result"]["frozen_weight"] = 0.75
    receipt["result"]["same_gross"]["candidate_weight"] = 0.75
    _rewrite_manifest(receipt_path, receipt)

    support = _support()
    with pytest.raises(RuntimeError, match="changed weight"):
        e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support),
            stage_loader=lambda _stage, _cutoff: pytest.fail(
                "future weight forgery opened rows"
            ),
            stage_evaluator=lambda stage, _inputs, state: _stage_result(
                stage, passed=True, state=state
            ),
            receipt_root=root,
        )


def test_resume_rejects_rehashed_future_derived_check_forgery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resume-future-check"
    assert _run_all_synthetic_stages(root)["passed"] is True
    receipt_path = root / e.STAGE_RECEIPT_NAMES["future25"]
    receipt = json.loads(receipt_path.read_bytes())
    receipt["result"]["same_gross"]["costs"]["base"]["treatment"][
        "cagr_to_strict_mdd"
    ] = 0.0
    _rewrite_manifest(receipt_path, receipt)

    support = _support()
    with pytest.raises(RuntimeError, match="checks were forged"):
        e.run_staged_economics(
            synthetic=True,
            source_support=support,
            novelty=_novelty(support),
            stage_loader=lambda _stage, _cutoff: pytest.fail(
                "derived-check forgery opened rows"
            ),
            stage_evaluator=lambda stage, _inputs, state: _stage_result(
                stage,
                passed=True,
                state=state,
            ),
            receipt_root=root,
        )


def test_output_paths_reject_aliases_symlinks_and_results(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="canonical"):
        e.write_once_result(
            f"{tmp_path}/child/../alias.json",
            {"x": 1},
            root=tmp_path,
        )
    with pytest.raises(RuntimeError, match="canonical"):
        e.write_once_result(
            e.STAGE_RECEIPT_NAMES["2023H2"],
            {"x": 1},
            root=(
                f"{e.REPOSITORY_ROOT}/results/../results"
            ),
            production=True,
        )
    with pytest.raises(RuntimeError, match="canonical"):
        e.write_once_result(
            "./a.json",
            {"x": 1},
            root=tmp_path,
        )
    with pytest.raises(RuntimeError, match="cannot resolve under results"):
        e.write_once_result(
            "synthetic.json",
            {"x": 1},
            root=e.CANONICAL_RESULTS_ROOT,
        )

    real = tmp_path / "real"
    real.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="unsafe ancestor"):
        e.write_once_result(
            linked_root / "result.json",
            {"x": 1},
            root=linked_root,
        )

    victim = tmp_path / "victim.json"
    victim.write_text("untouched", encoding="utf-8")
    leaf = tmp_path / "leaf.json"
    leaf.symlink_to(victim)
    with pytest.raises(RuntimeError, match="unsafe"):
        e.write_once_result(leaf, {"x": 1}, root=tmp_path)
    assert victim.read_text(encoding="utf-8") == "untouched"


def test_identical_concurrent_winner_is_verified_not_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "winner.json"
    original_link = e.os.link
    raced = False

    def concurrent_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        nonlocal raced
        if not raced:
            raced = True
            original_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )
            raise FileExistsError
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(e.os, "link", concurrent_link)
    publication = e.write_once_result(
        target,
        {"winner": True},
        root=tmp_path,
    )
    assert publication.status == "verified_existing"
    assert publication.sha256 == hashlib.sha256(
        e.canonical_json_bytes({"winner": True})
    ).hexdigest()
    assert target.read_bytes() == e.canonical_json_bytes({"winner": True})


def test_multi_output_rollback_surface_is_absent() -> None:
    assert not hasattr(e, "write_atomic_result_set")
    assert not hasattr(e, "_rollback_created_publication")
    assert "write_atomic_result_set" not in e.__all__
