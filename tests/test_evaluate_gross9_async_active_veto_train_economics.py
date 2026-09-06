from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_gross9_async_active_veto_train_economics as economics
from training import preregister_gross9_async_active_veto_search as prereg


def _minimal_authorized_artifact() -> dict[str, object]:
    candidates = {
        candidate: {
            "source_pass": False,
            "exact_duplicate_pass": True,
            "gross9_pass": False,
            "clock": {"path": "unused", "sha256": "0" * 64, "rows": 8},
        }
        for candidate in prereg.CANDIDATE_FAMILY
    }
    artifact: dict[str, object] = {
        "protocol_version": economics.novelty.PROTOCOL_VERSION,
        "policy_id": economics.POLICY_ID,
        "candidate_family": list(prereg.CANDIDATE_FAMILY),
        "candidate_family_size": prereg.FAMILY_SIZE,
        "gross9_passed_any_candidate": True,
        "advance_to_economic_outcomes": True,
        "gross9_novelty_passed_candidate_count": economics.EXPECTED_NOVEL_CANDIDATES,
        "gross9_novelty_passed_candidates": [],
        "decision": "pass_gross9_novel_candidates_to_train_economics",
        "preregistration": {"manifest_hash": "prehash"},
        "source_support": {"path": "support", "sha256": "0" * 64, "manifest_hash": "supporthash"},
        "manifest_hash": "placeholder",
        "candidates": candidates,
        "evidence_boundary": {
            "candidate_family_rows_counted": prereg.FAMILY_SIZE,
            "source_and_exact_duplicate_supported_candidates_expected": economics.EXPECTED_NOVEL_CANDIDATES,
            "exact_duplicate_gate_projected_for_all_72": True,
            "btc_price_or_return_rows_opened": 0,
            "entry_exit_prices_opened": 0,
            "funding_rows_opened": 0,
            "economic_outcome_rows_opened": 0,
            "portfolio_return_or_pnl_metrics_computed": False,
            "outcomes_opened": False,
        },
    }
    return artifact


def _authorize_first_n(artifact: dict[str, object], n: int) -> list[str]:
    selected = list(prereg.CANDIDATE_FAMILY[:n])
    candidates = artifact["candidates"]
    assert isinstance(candidates, dict)
    for candidate in selected:
        candidates[candidate].update({"source_pass": True, "exact_duplicate_pass": True, "gross9_pass": True})
    artifact["gross9_novelty_passed_candidates"] = selected
    artifact["gross9_novelty_passed_candidate_count"] = n
    return selected


def _install_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, artifact: dict[str, object]) -> Path:
    artifact.pop("manifest_hash", None)
    artifact["manifest_hash"] = economics.canonical_hash(artifact.copy())
    path = tmp_path / "novelty.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(economics, "NOVELTY", path)
    monkeypatch.setattr(economics, "NOVELTY_SHA256", economics.sha256_file(path))
    monkeypatch.setattr(economics, "NOVELTY_MANIFEST_HASH", artifact["manifest_hash"])
    return path


def test_load_novelty_authorization_binds_hash_manifest_all72_14_and_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = _minimal_authorized_artifact()
    _authorize_first_n(artifact, economics.EXPECTED_NOVEL_CANDIDATES)
    _install_artifact(monkeypatch, tmp_path, artifact)
    report = economics.load_novelty_authorization()
    assert report["candidate_family_size"] == 72
    assert report["gross9_novelty_passed_candidate_count"] == 14

    monkeypatch.setattr(economics, "NOVELTY_SHA256", "b" * 64)
    with pytest.raises(RuntimeError, match="novelty artifact hash drift"):
        economics.load_novelty_authorization()

    monkeypatch.undo()
    artifact = _minimal_authorized_artifact()
    _authorize_first_n(artifact, economics.EXPECTED_NOVEL_CANDIDATES)
    artifact["evidence_boundary"] = dict(artifact["evidence_boundary"], funding_rows_opened=1)  # type: ignore[arg-type]
    _install_artifact(monkeypatch, tmp_path, artifact)
    with pytest.raises(RuntimeError, match="boundary already opened economics"):
        economics.load_novelty_authorization()

    monkeypatch.undo()
    artifact = _minimal_authorized_artifact()
    _authorize_first_n(artifact, economics.EXPECTED_NOVEL_CANDIDATES - 1)
    _install_artifact(monkeypatch, tmp_path, artifact)
    with pytest.raises(RuntimeError, match="did not authorize economics"):
        economics.load_novelty_authorization()


def test_run_uses_exact_72_rows_duplicate_gate_and_no_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first, second, third = prereg.CANDIDATE_FAMILY[:3]
    novelty_report = _minimal_authorized_artifact()
    _authorize_first_n(novelty_report, economics.EXPECTED_NOVEL_CANDIDATES)
    candidates = novelty_report["candidates"]
    assert isinstance(candidates, dict)
    candidates[third]["exact_duplicate_pass"] = False
    novelty_report["gross9_novelty_passed_candidates"] = [c for c in novelty_report["gross9_novelty_passed_candidates"] if c != third]  # type: ignore[index]
    candidates[prereg.CANDIDATE_FAMILY[14]].update({"source_pass": True, "exact_duplicate_pass": True, "gross9_pass": True})
    novelty_report["gross9_novelty_passed_candidates"].append(prereg.CANDIDATE_FAMILY[14])  # type: ignore[union-attr]

    monkeypatch.setattr(economics, "load_novelty_authorization", lambda: novelty_report)
    monkeypatch.setattr(economics, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(
        economics,
        "load_market_hash_bound",
        lambda start, end: pd.DataFrame(
            {"date": pd.date_range(start, end, freq="5min", inclusive="both"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}
        ),
    )
    monkeypatch.setattr(economics, "load_train_funding_hash_bound", lambda start, end: pd.DataFrame({"date": [start], "funding_rate": [0.0], "mark_price": [1.0]}))
    monkeypatch.setattr(economics.econ, "validate_market", lambda market, start, end: None)
    monkeypatch.setattr(economics.econ, "validate_funding", lambda funding, start, end: None)

    def fake_eval(candidate: str, clock_record: dict[str, object], market: pd.DataFrame, funding: pd.DataFrame) -> dict[str, object]:
        return {
            "candidate_clock_rows_opened": 8,
            "primary": {},
            "checks": {},
            "source_pass": True,
            "exact_duplicate_pass": True,
            "gross9_pass": True,
            "train_economic_pass": candidate == second,
            "train_cagr_to_strict_mdd": 2.0 if candidate == first else 1.0,
            "train_absolute_return": 4.0 if candidate == first else 3.0,
            "decision": "train_economic_reject" if candidate == first else "train_pass",
        }

    monkeypatch.setattr(economics, "evaluate_candidate", fake_eval)
    result = economics.run(tmp_path / "economics.json")

    assert len(result["candidates"]) == 72
    assert third not in result["economics_evaluated_candidates"]
    assert result["candidates"][third]["exact_duplicate_pass"] is False
    assert result["candidates"][third]["decision"] == "not_evaluated_prereq_failed"
    assert len(result["economics_evaluated_candidates"]) == 14
    assert result["selection"]["raw_rank_one"]["candidate"] == first
    assert result["selection"]["selection_error"] == "G9ASYNCACTIVEVETO-8 raw rank one failed train; no substitution"
    assert result["selection"]["substitution_authorized"] is False
    assert result["selection"]["rerank_authorized"] is False
    assert result["decision"] == "terminal_train_reject_no_substitution"


def test_evaluate_candidate_uses_preregistered_costs_thresholds_and_simulation_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = pd.DataFrame(
        {
            "entry_time": [economics.TRAIN_START + pd.Timedelta(days=1), economics.TRAIN_START + pd.Timedelta(days=100)],
            "exit_time": [economics.TRAIN_START + pd.Timedelta(days=1, hours=8), economics.TRAIN_START + pd.Timedelta(days=100, hours=8)],
            "side": [1, -1],
        }
    )
    market = pd.DataFrame({"date": [economics.TRAIN_START], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})
    funding = pd.DataFrame({"date": [economics.TRAIN_START], "funding_rate": [0.0], "mark_price": [1.0]})
    calls: list[tuple[int, pd.Timestamp, pd.Timestamp, float]] = []

    monkeypatch.setattr(economics, "load_candidate_clock", lambda record, candidate: clock)

    def fake_simulate(c: pd.DataFrame, m: pd.DataFrame, f: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cost: float) -> dict[str, object]:
        calls.append((len(c), start, end, cost))
        return {
            "absolute_return_pct": 1.0,
            "cagr_to_strict_mdd": 3.0,
            "strict_mdd_pct": 15.0,
            "mean_gross_underlying_bp": 20.0,
            "trade_rows": [{"week": "2023-W27", "sign": 1}],
        }

    monkeypatch.setattr(economics.econ, "simulate", fake_simulate)
    monkeypatch.setattr(economics.econ, "cluster_p", lambda rows: {"pvalue": prereg.BONFERRONI_RAW_P_MAX})

    result = economics.evaluate_candidate("candidate", {"path": "unused", "sha256": "0" * 64, "rows": 2}, market, funding)

    assert result["train_economic_pass"] is True
    assert result["checks"]["cluster_signflip_p_max_bonferroni_0_1_over_72"] is True
    assert result["checks"]["strict_mdd_max_15"] is True
    assert calls[0] == (2, economics.TRAIN_START, economics.TRAIN_END, economics.econ.BASE_COST)
    assert calls[1] == (2, economics.TRAIN_START, economics.TRAIN_END, economics.econ.STRESS_COST)
    assert calls[2][3] == economics.econ.BASE_COST
    assert calls[3][3] == economics.econ.BASE_COST
    assert economics.prereg.BONFERRONI_RAW_P_MAX == pytest.approx(0.1 / 72)
