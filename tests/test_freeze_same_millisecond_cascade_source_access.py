from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import evaluate_same_millisecond_cascade_support as evaluator
from training import freeze_same_millisecond_cascade_source_access as freezer
from training import preregister_same_millisecond_cascade as prereg


def test_build_seal_hashes_bytes_without_opening_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.csv.gz"
    source.write_bytes(b"opaque-feature-bytes")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "outcomes_opened": False,
                    "source_archive_manifest_sha256": prereg.ARCHIVE_MANIFEST_SHA256,
                    "source_audit_sha256": prereg.SOURCE_AUDIT_SHA256,
                },
                "combined_output": str(source),
                "combined_sha256": source_hash,
                "rows": 1,
            }
        )
    )
    cfg = freezer.SealConfig(
        source=str(source),
        source_manifest=str(manifest_path),
        output=str(tmp_path / "seal.json"),
    )
    seal = freezer.build_seal(cfg, expected_rows=1)
    assert seal["source_rows_parsed"] == 0
    assert seal["outcomes_opened"] is False
    assert seal["source_sha256"] == source_hash


def test_seal_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "seal.json"
    payload = {"outcomes_opened": False, "source_rows_parsed": 0}
    assert freezer.write_once(path, payload) == "created"
    assert freezer.write_once(path, payload) == "verified_existing"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        freezer.write_once(path, {"outcomes_opened": True})


def test_support_execution_is_disabled_before_source_seal() -> None:
    assert evaluator.SOURCE_ACCESS_SEAL_FILE_SHA256 is None
    with pytest.raises(RuntimeError, match="not hash-sealed"):
        evaluator.load_source_access_seal(
            evaluator.SupportConfig(),
            evaluator.load_preregistration(),
        )
