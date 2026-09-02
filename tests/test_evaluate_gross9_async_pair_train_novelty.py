from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_gross9_async_pair_train_novelty as n
from training import preregister_gross9_async_pair_search as prereg


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


def test_evaluate_pair_train_does_not_mutate_metric_globals() -> None:
    original_start = n.metric.WINDOW_START
    original_end = n.metric.WINDOW_END
    candidate = _df([("train", "2023-07-02T00:00:00Z", "2023-07-02T08:00:00Z", 1)])
    comparator = _df([("train", "2023-07-20T00:00:00Z", "2023-07-20T08:00:00Z", -1)])

    result = n.evaluate_pair_train(candidate, comparator)

    assert n.metric.WINDOW_START == original_start
    assert n.metric.WINDOW_END == original_end
    assert result["common_window"] == ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"]
    assert set(result["checks"]) == set(n.LIMITS)
    assert result["passed"] is True


def test_train_contained_requires_train_split_entry_lt_end_and_exit_le_end() -> None:
    frame = _df(
        [
            ("train", "2023-06-30T23:55:00Z", "2023-07-01T07:55:00Z", 1),
            ("train", "2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z", 1),
            ("test", "2023-07-02T00:00:00Z", "2023-07-02T08:00:00Z", -1),
            ("train", "2023-12-31T16:00:00Z", "2024-01-01T00:00:00Z", -1),
            ("train", "2024-01-01T00:00:00Z", "2024-01-01T08:00:00Z", 1),
            ("train", "2023-12-31T20:00:00Z", "2024-01-01T04:00:00Z", 1),
        ]
    )

    filtered = n.train_contained(frame)

    assert filtered["entry_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist() == [
        "2023-07-01T00:00:00Z",
        "2023-12-31T16:00:00Z",
    ]
    assert filtered["split"].eq("train").all()
    assert filtered["entry_time"].lt(n.TRAIN_END).all()
    assert filtered["exit_time"].le(n.TRAIN_END).all()


def test_load_gross9_train_clock_validates_total_and_split_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sleeve = "cand_rex_veto_7"
    path = tmp_path / "gross9.csv.gz"
    _write_clock(
        path,
        [
            ("train", "2023-06-01T00:00:00Z", "2023-06-01T08:00:00Z", 1),
            ("train", "2023-07-02T00:00:00Z", "2023-07-02T08:00:00Z", 1),
            ("test2024", "2024-02-01T00:00:00Z", "2024-02-01T08:00:00Z", -1),
        ],
    )
    expected_counts = copy.deepcopy(n.gross9.EXPECTED_COUNTS)
    expected_counts[sleeve] = {"train": 2, "test2024": 1}
    monkeypatch.setattr(n.gross9, "EXPECTED_COUNTS", expected_counts)
    record = {"path": str(path), "sha256": n.sha256_file(path), "rows": 3, "counts": {"train": 2, "test2024": 1}}

    common, full_rows, common_rows = n.load_gross9_train_clock(sleeve, record)

    assert full_rows == 3
    assert common_rows == 1
    assert len(common) == 1
    assert common.iloc[0]["entry_time"] == pd.Timestamp("2023-07-02T00:00:00Z")

    bad_record = {**record, "counts": {"train": 3, "test2024": 0}}
    with pytest.raises(RuntimeError, match="split count drift"):
        n.load_gross9_train_clock(sleeve, bad_record)


def test_hash_drift_detected(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash drift"):
        n._verify_sha(path, "0" * 64, "synthetic")


def _support_with_passed(passed: list[str]) -> dict[str, object]:
    pairs = {}
    for candidate in prereg.CANDIDATE_FAMILY:
        source_pass = candidate in passed
        pairs[candidate] = {
            "components": ["L", "R"],
            "clock": {"path": f"{candidate}.csv.gz", "sha256": "a" * 64, "rows": 8 if source_pass else 0},
            "support": {"events": 8 if source_pass else 0},
            "support_checks": {"minimum_events": source_pass},
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


def _write_manifested_json(path: Path, payload: dict[str, object]) -> str:
    core = copy.deepcopy(payload)
    core.pop("manifest_hash", None)
    payload["manifest_hash"] = n.canonical_hash(core)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return n.sha256_file(path)


def test_load_frozen_controls_rejects_source_supported_flag_projection_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prereg_path = tmp_path / "prereg.json"
    registration = prereg.build()
    prereg_sha = _write_manifested_json(prereg_path, registration)
    monkeypatch.setattr(n, "PREREGISTRATION", prereg_path)
    monkeypatch.setattr(n, "PREREGISTRATION_SHA256", prereg_sha)
    monkeypatch.setattr(n, "PREREGISTRATION_MANIFEST_HASH", registration["manifest_hash"])

    support_path = tmp_path / "support.json"
    passed = list(prereg.CANDIDATE_FAMILY[:7])
    support = _support_with_passed(passed)
    support["preregistration"] = {"sha256": prereg_sha}
    support["pairs"][passed[0]]["advance_to_gross9_novelty"] = False  # type: ignore[index]
    support_sha = _write_manifested_json(support_path, support)
    monkeypatch.setattr(n, "SOURCE_SUPPORT", support_path)
    monkeypatch.setattr(n, "SOURCE_SUPPORT_SHA256", support_sha)
    monkeypatch.setattr(n, "SOURCE_SUPPORT_MANIFEST_HASH", support["manifest_hash"])



    with pytest.raises(RuntimeError, match="source-supported advancement drift"):
        n.load_frozen_controls()


def test_run_counts_all_36_but_evaluates_only_source_supported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    passed = list(prereg.CANDIDATE_FAMILY[:7])
    support = _support_with_passed(passed)
    support["manifest_hash"] = "supporthash"
    manifest = {"manifest_hash": "grosshash", "clocks": {sleeve: {"rows": 1, "path": f"{sleeve}.csv.gz", "sha256": "b" * 64, "counts": {"train": 1}} for sleeve in n.gross9.EXPECTED_WEIGHTS}}
    monkeypatch.setattr(n, "load_frozen_controls", lambda: ({"manifest_hash": "prehash"}, support, manifest))
    monkeypatch.setattr(n, "load_gross9_train_clock", lambda sleeve, record: (_df([("train", "2023-07-20T00:00:00Z", "2023-07-20T08:00:00Z", 1)]), 1, 1))
    calls: list[str] = []

    def fake_eval(candidate: str, clock_record: dict[str, object], gross9_clocks: dict[str, pd.DataFrame]) -> dict[str, object]:
        calls.append(candidate)
        passed_novelty = candidate == passed[-1]
        return {
            "candidate_clock_rows_opened": int(clock_record["rows"]),
            "gross9_sleeves": {},
            "gross9_pass": passed_novelty,
            "gross9_novelty_status": "passed" if passed_novelty else "failed",
            "advance_to_train_economics": passed_novelty,
            "decision": "pass_to_train_economics" if passed_novelty else "terminal_gross9_novelty_reject",
        }

    monkeypatch.setattr(n, "evaluate_candidate", fake_eval)

    result = n.run(tmp_path / "result.json")

    assert list(result["pairs"]) == list(prereg.CANDIDATE_FAMILY)
    assert len(result["pairs"]) == 36
    assert calls == passed
    assert result["gross9_novelty_evaluated_pair_count"] == 7
    assert result["gross9_novelty_passed_pairs"] == [passed[-1]]
    assert result["advance_to_economic_outcomes"] is True
    assert result["pairs"][prereg.CANDIDATE_FAMILY[8]]["gross9_novelty_status"] == "not_evaluated_source_support_failed"
    assert result["evidence_boundary"]["unsupported_pair_clock_rows_opened_for_novelty"] == 0
    assert result["evidence_boundary"]["source_supported_pair_clock_rows_opened"] == 56
    assert result["evidence_boundary"]["source_supported_pair_clock_rows_evaluated"] == 56
