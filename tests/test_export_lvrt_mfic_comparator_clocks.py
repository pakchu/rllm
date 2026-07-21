from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest

from training.export_lvrt_mfic_comparator_clocks import (
    build_outputs,
    canonical_hash,
    publish,
)


def _clock_rows(payload: bytes) -> list[dict[str, str]]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as handle:
        text = handle.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def test_real_mfic_clocks_match_frozen_support() -> None:
    manifest, clock_bytes = build_outputs()

    assert manifest["clock"] == {
        "path": "results/lvrt_mfic_pure_clocks_2026-07-21.csv.gz",
        "sha256": "d7d889bd2c8137682e244d399a42f14e2c04c48e88a336e07d05e48ddb0605bd",
        "gzip_mtime": 0,
        "schema": [
            "candidate_id",
            "split",
            "causal_origin",
            "decision_time",
            "entry_time",
            "exit_time",
            "side",
        ],
        "rows": 3201,
        "rows_by_candidate": {
            "mfic:mfic_fast": 1566,
            "mfic:mfic_slow": 1635,
        },
        "rows_by_split": {"selection": 461, "train": 2740},
    }
    assert hashlib.sha256(clock_bytes).hexdigest() == manifest["clock"]["sha256"]
    rows = _clock_rows(clock_bytes)
    assert len(rows) == 3201
    assert rows[0]["candidate_id"] == "mfic:mfic_fast"
    assert rows[-1]["candidate_id"] == "mfic:mfic_slow"
    assert manifest["outcome_boundary"] == {
        "causal_market_rows_loaded": 420768,
        "causal_aggtrade_feature_rows_loaded": 420732,
        "performance_artifacts_parsed": 0,
        "return_or_pnl_fields_read": 0,
        "strict_simulation_calls": 0,
        "funding_rows_loaded": 0,
        "post_2023_rows_loaded": 0,
        "network_calls": 0,
        "economic_outcomes_computed": False,
    }
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "manifest_hash"}
    }
    assert manifest["manifest_hash"] == canonical_hash(core)


def test_export_is_deterministic() -> None:
    first_manifest, first_clock = build_outputs()
    second_manifest, second_clock = build_outputs()

    assert first_clock == second_clock
    assert first_manifest["clock"]["sha256"] == second_manifest["clock"]["sha256"]
    assert first_manifest["manifest_hash"] == second_manifest["manifest_hash"]


def test_publication_is_create_only(tmp_path: Path) -> None:
    manifest, clock_bytes = build_outputs()
    manifest_path = tmp_path / "manifest.json"
    clock_path = tmp_path / "clock.csv.gz"
    mutable = copy.deepcopy(manifest)
    mutable["clock"]["path"] = str(clock_path)

    publish(manifest_path, clock_path, mutable, clock_bytes)
    assert clock_path.read_bytes() == clock_bytes
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["policy_id"] == (
        "LVRT-72"
    )
    with pytest.raises(FileExistsError):
        publish(manifest_path, clock_path, mutable, clock_bytes)
