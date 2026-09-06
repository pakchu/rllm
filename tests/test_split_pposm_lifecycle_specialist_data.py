from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from training import split_pposm_lifecycle_specialist_data as split


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity_sha(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(row["metadata"]["identity"] for row in rows).encode()).hexdigest()


def _row(signal: int, candidate: str, target: str, utility: float) -> dict[str, Any]:
    return {
        "task": "pposm_lifecycle_residual_action",
        "split": "train",
        "prompt": f"candidate_action: {candidate}\nReturn exactly one token: KEEP or SWITCH.",
        "target": target,
        "metadata": {
            "identity": f"pposm-lifecycle-residual|{candidate}|pre_2024|{signal}",
            "base_identity": f"pposm-counterfactual-action|pre_2024|{signal}",
            "candidate_action": candidate,
            "signal_position": signal,
            "signal_time": "2021-01-01 00:00:00",
            "window": "pre_2024",
            "reference_anchor": True,
            "offline_label_only": True,
            "future_outcome_present_in_prompt": False,
            "residual_utilities": {"KEEP": 0.0, "SWITCH": utility},
        },
    }


def _rows() -> list[dict[str, Any]]:
    return [
        _row(10, "SKIP", "KEEP", -0.20),
        _row(10, "TP12", "SWITCH", 0.30),
        _row(20, "SKIP", "SWITCH", 0.10),
        _row(20, "TP12", "KEEP", -0.40),
    ]


def _cfg(tmp_path: Path, rows: list[dict[str, Any]]) -> split.Config:
    input_path = tmp_path / "input.jsonl"
    input_sha = _write_jsonl(input_path, rows)
    return split.Config(
        input_jsonl=input_path,
        skip_output=tmp_path / "skip.jsonl",
        tp12_output=tmp_path / "tp12.jsonl",
        summary_output=tmp_path / "summary.json",
        expected_input_sha256=input_sha,
        expected_identity_sha256=_identity_sha(rows),
        expected_rows=len(rows),
        expected_rows_per_candidate=len(rows) // 2,
    )


def test_split_writes_two_candidate_jsonls_and_summary_hashes(tmp_path: Path) -> None:
    rows = _rows()
    cfg = _cfg(tmp_path, rows)

    summary = split.split_lifecycle_specialists(cfg)

    skip_rows = [json.loads(line) for line in cfg.skip_output.read_text().splitlines()]
    tp12_rows = [json.loads(line) for line in cfg.tp12_output.read_text().splitlines()]
    assert [row["metadata"]["candidate_action"] for row in skip_rows] == ["SKIP", "SKIP"]
    assert [row["metadata"]["candidate_action"] for row in tp12_rows] == ["TP12", "TP12"]
    assert summary["rows"] == {"input": 4, "SKIP": 2, "TP12": 2, "oos": 0}
    assert summary["target_counts"] == {"KEEP": 2, "SWITCH": 2}
    assert summary["candidate_target_counts"] == {
        "SKIP=KEEP": 1,
        "SKIP=SWITCH": 1,
        "TP12=KEEP": 1,
        "TP12=SWITCH": 1,
    }
    assert summary["outputs"]["SKIP"]["sha256"] == hashlib.sha256(cfg.skip_output.read_bytes()).hexdigest()
    assert summary["outputs"]["TP12"]["identity_sha256"] == split._identity_sha256(tp12_rows)
    assert summary["median_nonzero_absolute_switch_utility"] == pytest.approx(0.25)
    assert json.loads(cfg.summary_output.read_text())["leakage_guard"]["oos_surface_written"] is False


def test_validate_fails_closed_on_input_sha_identity_and_reference_anchor(tmp_path: Path) -> None:
    rows = _rows()
    cfg = _cfg(tmp_path, rows)
    loaded, input_sha = split._load_jsonl_with_sha(cfg.input_jsonl)
    assert split.validate_frozen_lifecycle_rows(loaded, cfg, input_sha)["rows"] == 4

    with pytest.raises(ValueError, match="input sha mismatch"):
        split.validate_frozen_lifecycle_rows(loaded, split.Config(**{**cfg.__dict__, "expected_input_sha256": "0" * 64}), input_sha)

    bad_identity = [dict(row) for row in rows]
    bad_identity[0] = {**bad_identity[0], "metadata": {**bad_identity[0]["metadata"], "identity": "drift"}}
    bad_path = tmp_path / "bad_identity.jsonl"
    bad_sha = _write_jsonl(bad_path, bad_identity)
    with pytest.raises(ValueError, match="identity does not match"):
        split.validate_frozen_lifecycle_rows(
            bad_identity,
            split.Config(**{**cfg.__dict__, "expected_input_sha256": bad_sha}),
            bad_sha,
        )

    bad_anchor = [dict(row) for row in rows]
    bad_anchor[0] = {**bad_anchor[0], "metadata": {**bad_anchor[0]["metadata"], "reference_anchor": False}}
    bad_anchor_path = tmp_path / "bad_anchor.jsonl"
    bad_anchor_sha = _write_jsonl(bad_anchor_path, bad_anchor)
    with pytest.raises(ValueError, match="reference_anchor"):
        split.validate_frozen_lifecycle_rows(
            bad_anchor,
            split.Config(**{**cfg.__dict__, "expected_input_sha256": bad_anchor_sha, "expected_identity_sha256": _identity_sha(bad_anchor)}),
            bad_anchor_sha,
        )


def test_validate_fails_closed_on_missing_candidate_pair(tmp_path: Path) -> None:
    rows = _rows()
    bad = [dict(row) for row in rows]
    bad[1] = _row(30, "TP12", "KEEP", -0.1)
    input_path = tmp_path / "bad_pair.jsonl"
    input_sha = _write_jsonl(input_path, bad)
    cfg = split.Config(
        input_jsonl=input_path,
        skip_output=tmp_path / "skip.jsonl",
        tp12_output=tmp_path / "tp12.jsonl",
        summary_output=tmp_path / "summary.json",
        expected_input_sha256=input_sha,
        expected_identity_sha256=_identity_sha(bad),
        expected_rows=len(bad),
        expected_rows_per_candidate=2,
    )
    with pytest.raises(ValueError, match="lacks exactly one SKIP and TP12 pair"):
        split.validate_frozen_lifecycle_rows(bad, cfg, input_sha)
