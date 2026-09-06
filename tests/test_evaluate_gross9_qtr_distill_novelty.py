from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_gross9_qtr_distill_clocks as source_support
from training import evaluate_gross9_qtr_distill_novelty as n
from training import export_gross9_structural_clocks as gross9
from training import preregister_gross9_qtr_distill as prereg


def _write_gzip_frame(path: Path, rows, columns) -> str:
    frame = pd.DataFrame(rows, columns=columns)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    return n.sha256_file(path)


def _manifest(payload):
    payload = dict(payload)
    payload["manifest_hash"] = n.canonical_hash({k: v for k, v in payload.items() if k != "manifest_hash"})
    return payload


def test_constants_bind_g9qtr_distill_train_novelty() -> None:
    assert n.POLICY_ID == "G9QTR-DISTILL-8"
    assert n.SOURCE_PACKAGE == source_support.DEFAULT_RESULT
    assert n.OUTPUT.as_posix() == "results/gross9_qtr_distill_train_gross9_novelty_2026-09-02.json"
    assert n.LIMITS["one_to_one_6h_max_matched_share"] == 0.35
    assert len(gross9.EXPECTED_WEIGHTS) == 5


def test_weighted_segment_exposure_uses_target_exposure_values() -> None:
    segments = pd.DataFrame([
        {"start_time": pd.Timestamp("2023-07-01T00:00:00Z"), "end_time": pd.Timestamp("2023-07-01T00:10:00Z"), "target_exposure": 1 / 6},
        {"start_time": pd.Timestamp("2023-07-01T00:10:00Z"), "end_time": pd.Timestamp("2023-07-01T00:15:00Z"), "target_exposure": -1 / 3},
    ])
    exposure = n.weighted_segment_exposure_train(segments)
    assert exposure[:4].tolist() == pytest.approx([1 / 6, 1 / 6, -1 / 3, 0.0])


def test_entry_matching_uses_timestamps_regardless_of_side() -> None:
    t = pd.Timestamp("2023-07-01T12:00:00Z")
    episodes = pd.DataFrame([{"start_time": t, "side": 1}, {"start_time": t + pd.Timedelta(hours=8), "side": -1}])
    candidate = n.episode_start_timestamps(episodes)
    comparator = (t, t + pd.Timedelta(hours=8))
    assert n.pair_novelty.metric.exact_entry_jaccard(candidate, comparator) == pytest.approx(1.0)
    matches, _ = n.pair_novelty.metric.optimal_near_matches(candidate, comparator)
    assert matches == ((t, t), (t + pd.Timedelta(hours=8), t + pd.Timedelta(hours=8)))


def test_undefined_correlation_is_recorded_as_nan_and_fails_correlation_gate() -> None:
    value = n.pearson_or_nan(pd.Series([0.0, 0.0]).to_numpy(), pd.Series([1.0, 2.0]).to_numpy())
    assert pd.isna(value)
    assert n._corr_check(value) is False



def test_nan_pearson_makes_candidate_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    start = n.TRAIN_START
    candidate_events = (start,)
    candidate_exposure = np.zeros(int((n.TRAIN_END - n.TRAIN_START) / n.BAR))
    gross9_clock = pd.DataFrame([{"split": "train", "entry_time": start + pd.Timedelta(hours=10), "exit_time": start + pd.Timedelta(hours=10, minutes=5), "side": 1}])
    monkeypatch.setitem(gross9.EXPECTED_WEIGHTS, "unit_test_sleeve", 1.0)
    result = n.evaluate_against_gross9(candidate_events, candidate_exposure, gross9_clock, "unit_test_sleeve")
    assert result["metrics"]["absolute_signed_exposure_pearson"] is None
    assert result["checks"]["absolute_signed_exposure_pearson"] is False
    assert result["passed"] is False

def test_load_train_schedules_filters_train_and_rejects_oos_leakage(tmp_path: Path) -> None:
    episode_path = tmp_path / "episodes.csv.gz"
    segment_path = tmp_path / "segments.csv.gz"
    transition_path = tmp_path / "transitions.csv.gz"
    _write_gzip_frame(episode_path, [
        {"candidate": n.POLICY_ID, "split": "train", "start_time": "2023-07-02T00:00:00Z", "end_time": "2023-07-02T01:00:00Z", "side": 1},
        {"candidate": n.POLICY_ID, "split": "test", "start_time": "2024-01-02T00:00:00Z", "end_time": "2024-01-02T01:00:00Z", "side": -1},
    ], ["candidate", "split", "start_time", "end_time", "side"])
    _write_gzip_frame(segment_path, [
        {"candidate": n.POLICY_ID, "split": "train", "start_time": "2023-07-02T00:00:00Z", "end_time": "2023-07-02T01:00:00Z", "target_exposure": 0.5},
        {"candidate": n.POLICY_ID, "split": "eval", "start_time": "2025-01-02T00:00:00Z", "end_time": "2025-01-02T01:00:00Z", "target_exposure": -0.5},
    ], ["candidate", "split", "start_time", "end_time", "target_exposure"])
    _write_gzip_frame(transition_path, [{"candidate": n.POLICY_ID, "split": "train", "timestamp": "2023-07-02T00:00:00Z", "target_exposure": 0.5}], ["candidate", "split", "timestamp", "target_exposure"])
    source = {"portfolio_schedules": {
        "signed_episodes": {"path": str(episode_path), "sha256": n.sha256_file(episode_path), "rows": 2},
        "segments": {"path": str(segment_path), "sha256": n.sha256_file(segment_path), "rows": 2},
        "transitions": {"path": str(transition_path), "sha256": n.sha256_file(transition_path), "rows": 1},
    }}
    assert len(n.load_train_episodes(source)) == 1
    assert len(n.load_train_segments(source)) == 1

    leak_path = tmp_path / "leaking_episodes.csv.gz"
    _write_gzip_frame(leak_path, [{"candidate": n.POLICY_ID, "split": "train", "start_time": "2024-01-01T00:00:00Z", "end_time": "2024-01-01T01:00:00Z", "side": 1}], ["candidate", "split", "start_time", "end_time", "side"])
    source["portfolio_schedules"]["signed_episodes"] = {"path": str(leak_path), "sha256": n.sha256_file(leak_path), "rows": 1}
    with pytest.raises(RuntimeError, match="leaked outside train window"):
        n.load_train_episodes(source)


def test_run_writes_train_only_no_price_or_oos_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    start = n.TRAIN_START
    transition_path = tmp_path / "transitions.csv.gz"
    segment_path = tmp_path / "segments.csv.gz"
    episode_path = tmp_path / "episodes.csv.gz"
    _write_gzip_frame(transition_path, [{"candidate": n.POLICY_ID, "split": "train", "timestamp": start.isoformat(), "target_exposure": 0.5}], ["candidate", "split", "timestamp", "target_exposure"])
    _write_gzip_frame(segment_path, [{"candidate": n.POLICY_ID, "split": "train", "start_time": start.isoformat(), "end_time": (start + pd.Timedelta(minutes=10)).isoformat(), "target_exposure": 0.5}], ["candidate", "split", "start_time", "end_time", "target_exposure"])
    _write_gzip_frame(episode_path, [{"candidate": n.POLICY_ID, "split": "train", "start_time": start.isoformat(), "end_time": (start + pd.Timedelta(minutes=10)).isoformat(), "side": 1}], ["candidate", "split", "start_time", "end_time", "side"])
    source = {"manifest_hash": "sourcehash", "portfolio_schedules": {
        "transitions": {"path": str(transition_path), "sha256": n.sha256_file(transition_path), "rows": 1},
        "segments": {"path": str(segment_path), "sha256": n.sha256_file(segment_path), "rows": 1},
        "signed_episodes": {"path": str(episode_path), "sha256": n.sha256_file(episode_path), "rows": 1},
    }}
    manifest = {"manifest_hash": "grosshash", "clocks": {s: {"path": f"{s}.csv.gz", "sha256": "a" * 64, "rows": 1, "counts": {"train": 1}} for s in gross9.EXPECTED_WEIGHTS}}
    monkeypatch.setattr(n, "load_validated_controls", lambda: ({"manifest_hash": "prehash"}, source, manifest))
    monkeypatch.setattr(n, "load_gross9_train_clock", lambda sleeve, record: (pd.DataFrame([{"split": "train", "entry_time": start + pd.Timedelta(hours=10), "exit_time": start + pd.Timedelta(hours=10, minutes=5), "side": -1}]), 1, 1))
    out = tmp_path / "novelty.json"
    result = n.run(out)
    assert result["evidence_boundary"]["oos_schedule_rows_opened"] == 0
    assert result["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert result["gross9_pass"] is True
    written = json.loads(out.read_text())
    assert written["manifest_hash"] == n.canonical_hash({k: v for k, v in written.items() if k != "manifest_hash"})
