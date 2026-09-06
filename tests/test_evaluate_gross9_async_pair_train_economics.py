from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_gross9_async_pair_train_economics as economics
from training import preregister_gross9_async_pair_search as prereg


def test_load_novelty_authorization_rejects_prior_economic_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = {
        "protocol_version": economics.novelty.PROTOCOL_VERSION,
        "policy_id": economics.POLICY_ID,
        "candidate_family": list(prereg.CANDIDATE_FAMILY),
        "candidate_family_size": prereg.FAMILY_SIZE,
        "gross9_passed_any_pair": True,
        "decision": "pass_gross9_novel_pairs_to_train_economics",
        "evidence_boundary": {
            "btc_price_or_return_rows_opened": 0,
            "entry_exit_prices_opened": 0,
            "funding_rows_opened": 1,
            "economic_outcome_rows_opened": 0,
            "portfolio_return_or_pnl_metrics_computed": False,
            "outcomes_opened": False,
        },
    }
    artifact["manifest_hash"] = economics.canonical_hash(artifact.copy())
    path = tmp_path / "novelty.json"
    path.write_text(__import__("json").dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(economics, "NOVELTY", path)

    with pytest.raises(RuntimeError, match="boundary already opened economics"):
        economics.load_novelty_authorization()


def test_run_uses_all_36_rows_and_terminal_raw_rank_one_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first, second = prereg.CANDIDATE_FAMILY[:2]
    novelty_report = {
        "protocol_version": economics.novelty.PROTOCOL_VERSION,
        "policy_id": economics.POLICY_ID,
        "candidate_family": list(prereg.CANDIDATE_FAMILY),
        "candidate_family_size": prereg.FAMILY_SIZE,
        "gross9_passed_any_pair": True,
        "decision": "pass_gross9_novel_pairs_to_train_economics",
        "manifest_hash": "noveltyhash",
        "preregistration": {"manifest_hash": "prehash"},
        "source_support": {"path": "support", "sha256": "0" * 64, "manifest_hash": "supporthash"},
        "pairs": {
            candidate: {"source_pass": candidate in (first, second), "gross9_pass": candidate in (first, second), "clock": {"path": "unused", "sha256": "0" * 64, "rows": 8}}
            for candidate in prereg.CANDIDATE_FAMILY
        },
    }
    monkeypatch.setattr(economics, "load_novelty_authorization", lambda: novelty_report)
    monkeypatch.setattr(economics, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(economics, "load_market_hash_bound", lambda start, end: pd.DataFrame({"date": pd.date_range(start, end, freq="5min", inclusive="both"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}))
    monkeypatch.setattr(economics, "load_train_funding_hash_bound", lambda start, end: pd.DataFrame({"date": [start], "funding_rate": [0.0], "mark_price": [1.0]}))
    monkeypatch.setattr(economics.econ, "validate_market", lambda m, start, end: None)
    monkeypatch.setattr(economics.econ, "validate_funding", lambda f, start, end: None)

    def fake_eval(candidate: str, clock_record: dict[str, object], market: pd.DataFrame, funding: pd.DataFrame) -> dict[str, object]:
        return {
            "candidate_clock_rows_opened": 8,
            "primary": {},
            "checks": {},
            "source_pass": True,
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
    assert result["selection"]["raw_rank_one"]["candidate"] == first
    assert result["selection"]["selection_error"] == "G9ASYNCPAIR-8 raw rank one failed train; no substitution"
    assert result["decision"] == "terminal_train_reject_no_substitution"
