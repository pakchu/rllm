from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_gross9_async_opposition_handoff_train_economics as economics
from training import preregister_gross9_async_opposition_handoff_search as prereg


def _minimal_authorized_artifact() -> dict[str, object]:
    pairs = {
        candidate: {
            "source_pass": False,
            "same_side_pre_reservation_intersection_pass": True,
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
        "gross9_passed_any_pair": True,
        "advance_to_economic_outcomes": True,
        "decision": "pass_gross9_novel_pairs_to_train_economics",
        "preregistration": {"manifest_hash": "prehash"},
        "source_support": {"path": "support", "sha256": "0" * 64, "manifest_hash": "supporthash"},
        "pairs": pairs,
        "evidence_boundary": {
            "btc_price_or_return_rows_opened": 0,
            "entry_exit_prices_opened": 0,
            "funding_rows_opened": 0,
            "economic_outcome_rows_opened": 0,
            "portfolio_return_or_pnl_metrics_computed": False,
            "outcomes_opened": False,
        },
    }
    artifact["manifest_hash"] = economics.canonical_hash(artifact.copy())
    return artifact


def _install_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, artifact: dict[str, object]) -> Path:
    path = tmp_path / "novelty.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    sha = economics.sha256_file(path)
    monkeypatch.setattr(economics, "NOVELTY", path)
    monkeypatch.setattr(economics, "NOVELTY_SHA256", sha)
    monkeypatch.setattr(economics, "NOVELTY_MANIFEST_HASH", artifact["manifest_hash"])
    return path


def test_load_novelty_authorization_rejects_hash_manifest_and_prior_economic_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = _minimal_authorized_artifact()
    artifact["evidence_boundary"] = dict(artifact["evidence_boundary"], funding_rows_opened=1)  # type: ignore[arg-type]
    artifact.pop("manifest_hash")
    artifact["manifest_hash"] = economics.canonical_hash(artifact.copy())
    _install_artifact(monkeypatch, tmp_path, artifact)

    with pytest.raises(RuntimeError, match="boundary already opened economics"):
        economics.load_novelty_authorization()

    clean = _minimal_authorized_artifact()
    path = _install_artifact(monkeypatch, tmp_path, clean)
    monkeypatch.setattr(economics, "NOVELTY_SHA256", "b" * 64)
    with pytest.raises(RuntimeError, match="novelty artifact hash drift"):
        economics.load_novelty_authorization()

    monkeypatch.setattr(economics, "NOVELTY_SHA256", economics.sha256_file(path))
    monkeypatch.setattr(economics, "NOVELTY_MANIFEST_HASH", "c" * 64)
    with pytest.raises(RuntimeError, match="novelty manifest binding drift"):
        economics.load_novelty_authorization()


def test_run_uses_exact_36_rows_disjoint_gate_and_no_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first, second, third = prereg.CANDIDATE_FAMILY[:3]
    novelty_report = _minimal_authorized_artifact()
    for candidate in (first, second, third):
        novelty_report["pairs"][candidate].update({"source_pass": True, "gross9_pass": True})  # type: ignore[index,union-attr]
    novelty_report["pairs"][third]["same_side_pre_reservation_intersection_pass"] = False  # type: ignore[index]

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
            "same_side_disjointness_pass": True,
            "gross9_pass": True,
            "train_economic_pass": candidate == second,
            "train_cagr_to_strict_mdd": 2.0 if candidate == first else 1.0,
            "train_absolute_return": 4.0 if candidate == first else 3.0,
            "decision": "train_economic_reject" if candidate == first else "train_pass",
        }

    monkeypatch.setattr(economics, "evaluate_candidate", fake_eval)
    result = economics.run(tmp_path / "economics.json")

    assert len(result["pairs"]) == 36
    assert result["economics_evaluated_pairs"] == [first, second]
    assert result["pairs"][third]["same_side_disjointness_pass"] is False
    assert result["pairs"][third]["decision"] == "not_evaluated_prereq_failed"
    assert result["selection"]["raw_rank_one"]["candidate"] == first
    assert result["selection"]["selection_error"] == "G9ASYNCHANDOFF-8 raw rank one failed train; no substitution"
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

    monkeypatch.setattr(economics, "load_pair_clock", lambda record, candidate: clock)

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
    assert result["checks"]["cluster_signflip_p_max_bonferroni_0_1_over_36"] is True
    assert calls[0] == (2, economics.TRAIN_START, economics.TRAIN_END, economics.econ.BASE_COST)
    assert calls[1] == (2, economics.TRAIN_START, economics.TRAIN_END, economics.econ.STRESS_COST)
    assert calls[2][3] == economics.econ.BASE_COST
    assert calls[3][3] == economics.econ.BASE_COST
