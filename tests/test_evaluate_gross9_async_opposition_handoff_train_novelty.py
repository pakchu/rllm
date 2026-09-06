from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_gross9_async_opposition_handoff_train_novelty as n
from training import preregister_gross9_async_opposition_handoff_search as prereg


def _df(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split,
            "entry_time": pd.Timestamp(entry),
            "exit_time": pd.Timestamp(exit_time),
            "side": side,
        }
        for split, entry, exit_time, side in rows
    )


def _write_clock(path: Path, rows: list[tuple[str, str, str, int]]) -> None:
    frame = pd.DataFrame(rows, columns=["split", "entry_time", "exit_time", "side"])
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def _write_manifested_json(path: Path, payload: dict[str, object]) -> str:
    core = copy.deepcopy(payload)
    core.pop("manifest_hash", None)
    payload["manifest_hash"] = n.canonical_hash(core)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return n.sha256_file(path)


def _support_with_passed(passed: list[str]) -> dict[str, object]:
    pairs = {}
    for candidate in prereg.CANDIDATE_FAMILY:
        source_pass = candidate in passed
        pairs[candidate] = {
            "components": ["L", "R"],
            "operator": "symmetric_async_strict_opposition_handoff_latest_other_within_6h",
            "clock": {"path": f"{candidate}.csv.gz", "sha256": "a" * 64, "rows": 12 if source_pass else 0},
            "construction_diagnostics": {
                "pre_reservation_rows": 12 if source_pass else 0,
                "post_reservation_rows": 12 if source_pass else 0,
                "reservation_dropped_rows": 0,
                "simultaneous_component_event_exclusions": 0,
                "no_other_strict_window_rejections": 1,
                "same_side_strict_window_rejections": 0,
                "same_side_pre_reservation_entry_intersection": 0,
                "same_side_post_reservation_entry_intersection_diagnostic": 0,
            },
            "support": {
                "events": 12 if source_pass else 0,
                "longs": 6 if source_pass else 0,
                "shorts": 6 if source_pass else 0,
                "minority_side_share": 0.5 if source_pass else 0.0,
                "max_month_share": 0.25 if source_pass else 0.0,
                "distinct_iso_weeks": 9 if source_pass else 0,
            },
            "support_checks": {
                "minimum_events": source_pass,
                "side_balance": source_pass,
                "month_concentration": source_pass,
                "distinct_iso_weeks": source_pass,
            },
            "support_passed": source_pass,
            "advance_to_gross9_novelty": source_pass,
            "advance_to_economic_outcomes": False,
            "decision": "pass_to_gross9_novelty" if source_pass else "terminal_source_support_reject",
        }
    return {
        "protocol_version": n.source_support.PROTOCOL_VERSION,
        "policy_id": n.POLICY_ID,
        "preregistration": {"sha256": n.PREREGISTRATION_SHA256},
        "candidate_family": list(prereg.CANDIDATE_FAMILY),
        "candidate_family_size": prereg.FAMILY_SIZE,
        "decision": "pass_supported_pairs_to_gross9_novelty",
        "passed_pairs": passed,
        "pairs": pairs,
        "evidence_boundary": {
            "gross9_rows_opened": False,
            "market_rows_opened": False,
            "entry_exit_prices_opened": False,
            "funding_opened": False,
            "pair_combination_returns_or_pnl_opened": False,
            "pair_combination_economic_outcomes_opened": False,
        },
    }


def test_reuses_pair_train_metrics_without_mutating_metric_globals() -> None:
    original_start = n.pair_novelty.metric.WINDOW_START
    original_end = n.pair_novelty.metric.WINDOW_END
    candidate = _df([("train", "2023-07-02T00:00:00Z", "2023-07-02T08:00:00Z", 1)])
    comparator = _df([("train", "2023-07-20T00:00:00Z", "2023-07-20T08:00:00Z", -1)])

    result = n.evaluate_pair_train(candidate, comparator)

    assert n.pair_novelty.metric.WINDOW_START == original_start
    assert n.pair_novelty.metric.WINDOW_END == original_end
    assert result["common_window"] == ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"]
    assert result["checks"].keys() == n.LIMITS.keys()
    assert n.LIMITS == n.pair_novelty.LIMITS


def test_load_candidate_clock_is_train_contained_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "candidate.csv.gz"
    _write_clock(
        path,
        [
            ("train", "2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z", 1),
            ("test", "2023-07-02T00:00:00Z", "2023-07-02T08:00:00Z", -1),
        ],
    )
    record = {"path": str(path), "sha256": n.sha256_file(path), "rows": 2}
    with pytest.raises(RuntimeError, match="row/window drift"):
        n.load_candidate_clock(record, "C")

    good_path = tmp_path / "candidate_train.csv.gz"
    _write_clock(good_path, [("train", "2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z", 1)])
    good = {"path": str(good_path), "sha256": n.sha256_file(good_path), "rows": 1}
    loaded = n.load_candidate_clock(good, "C")
    assert len(loaded) == 1
    assert loaded["split"].eq("train").all()

    bad = {**good, "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="candidate clock hash drift"):
        n.load_candidate_clock(bad, "C")


def test_load_frozen_controls_rejects_nonzero_same_side_intersection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prereg_path = tmp_path / "prereg.json"
    registration = prereg.build()
    prereg_sha = _write_manifested_json(prereg_path, registration)
    monkeypatch.setattr(n, "PREREGISTRATION", prereg_path)
    monkeypatch.setattr(n, "PREREGISTRATION_SHA256", prereg_sha)
    monkeypatch.setattr(n, "PREREGISTRATION_MANIFEST_HASH", registration["manifest_hash"])

    support_path = tmp_path / "support.json"
    passed = [prereg.CANDIDATE_FAMILY[0]]
    support = _support_with_passed(passed)
    support["preregistration"] = {"sha256": prereg_sha}
    support["pairs"][passed[0]]["construction_diagnostics"]["same_side_pre_reservation_entry_intersection"] = 1  # type: ignore[index]
    support_sha = _write_manifested_json(support_path, support)
    monkeypatch.setattr(n, "SOURCE_SUPPORT", support_path)
    monkeypatch.setattr(n, "SOURCE_SUPPORT_SHA256", support_sha)
    monkeypatch.setattr(n, "SOURCE_SUPPORT_MANIFEST_HASH", support["manifest_hash"])

    with pytest.raises(RuntimeError, match="same-side pre-reservation intersection drift"):
        n.load_frozen_controls()


def test_load_frozen_controls_rejects_source_advancement_projection_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prereg_path = tmp_path / "prereg.json"
    registration = prereg.build()
    prereg_sha = _write_manifested_json(prereg_path, registration)
    monkeypatch.setattr(n, "PREREGISTRATION", prereg_path)
    monkeypatch.setattr(n, "PREREGISTRATION_SHA256", prereg_sha)
    monkeypatch.setattr(n, "PREREGISTRATION_MANIFEST_HASH", registration["manifest_hash"])

    support_path = tmp_path / "support.json"
    passed = [prereg.CANDIDATE_FAMILY[0]]
    support = _support_with_passed(passed)
    support["preregistration"] = {"sha256": prereg_sha}
    support["pairs"][passed[0]]["advance_to_gross9_novelty"] = False  # type: ignore[index]
    support_sha = _write_manifested_json(support_path, support)
    monkeypatch.setattr(n, "SOURCE_SUPPORT", support_path)
    monkeypatch.setattr(n, "SOURCE_SUPPORT_SHA256", support_sha)
    monkeypatch.setattr(n, "SOURCE_SUPPORT_MANIFEST_HASH", support["manifest_hash"])

    with pytest.raises(RuntimeError, match="source-supported advancement drift"):
        n.load_frozen_controls()


def test_run_counts_all_36_evaluates_only_source_supported_and_no_leakage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    passed = [prereg.CANDIDATE_FAMILY[4]]
    support = _support_with_passed(passed)
    support["manifest_hash"] = "supporthash"
    manifest = {
        "manifest_hash": "grosshash",
        "clocks": {
            sleeve: {"rows": 1, "path": f"{sleeve}.csv.gz", "sha256": "b" * 64, "counts": {"train": 1}}
            for sleeve in n.gross9.EXPECTED_WEIGHTS
        },
    }
    monkeypatch.setattr(n, "load_frozen_controls", lambda: ({"manifest_hash": "prehash"}, support, manifest))
    monkeypatch.setattr(
        n,
        "load_gross9_train_clock",
        lambda sleeve, record: (_df([("train", "2023-07-20T00:00:00Z", "2023-07-20T08:00:00Z", 1)]), 1, 1),
    )
    calls: list[str] = []

    def fake_eval(candidate: str, clock_record: dict[str, object], gross9_clocks: dict[str, pd.DataFrame]) -> dict[str, object]:
        calls.append(candidate)
        return {
            "candidate_clock_rows_opened": int(clock_record["rows"]),
            "gross9_sleeves": {},
            "gross9_pass": False,
            "gross9_novelty_status": "failed",
            "advance_to_train_economics": False,
            "decision": "terminal_gross9_novelty_reject",
        }

    monkeypatch.setattr(n, "evaluate_candidate", fake_eval)

    result = n.run(tmp_path / "novelty.json")

    assert list(result["pairs"]) == list(prereg.CANDIDATE_FAMILY)
    assert len(result["pairs"]) == 36
    assert calls == passed
    assert result["gross9_novelty_evaluated_pair_count"] == 1
    assert result["source_supported_pairs"] == passed
    assert result["source_supported_pair_rows_opened"] == 12
    assert result["advance_to_economic_outcomes"] is False
    assert result["decision"] == "terminal_no_gross9_novel_pairs"
    assert result["pairs"][passed[0]]["same_side_pre_reservation_intersection_pass"] is True
    assert result["pairs"][prereg.CANDIDATE_FAMILY[0]]["gross9_novelty_status"] == "not_evaluated_source_support_failed"
    boundary = result["evidence_boundary"]
    assert boundary["candidate_family_rows_counted"] == 36
    assert boundary["source_supported_pair_clock_rows_opened"] == 12
    assert boundary["unsupported_pair_clock_rows_opened_for_novelty"] == 0
    assert boundary["same_side_pre_reservation_intersection_verified_for_all_36"] is True
    assert boundary["btc_price_or_return_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["economic_outcome_rows_opened"] == 0
    written = json.loads((tmp_path / "novelty.json").read_text())
    core = dict(written)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == n.canonical_hash(core)
