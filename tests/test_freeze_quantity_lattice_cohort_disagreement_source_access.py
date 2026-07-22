from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import evaluate_quantity_lattice_cohort_disagreement_support as evaluator
from training import freeze_quantity_lattice_cohort_disagreement_source_access as freezer
from training import preregister_quantity_lattice_cohort_disagreement as prereg


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


def test_support_execution_is_bound_to_frozen_source_seal() -> None:
    assert evaluator.SOURCE_ACCESS_SEAL_FILE_SHA256 == (
        "cade903a3d15349903c3e16853a23a092b36a293cb46ceb7b0c5514737aca834"
    )
    assert evaluator.EXPECTED_SOURCE_SHA256 == (
        "3ca945f134115fc7b58086405fd881db3e3b70087bd9da54ffc293f6b658072e"
    )
    assert evaluator.EXPECTED_SOURCE_MANIFEST_SHA256 == (
        "bcdf89924f54a5b97d4219749c2094d2a4c08d8473a37bc5367d9b8e5791284f"
    )
    seal = evaluator.load_source_access_seal(
        evaluator.SupportConfig(),
        evaluator.load_preregistration(),
    )
    assert seal["source_rows_parsed"] == 0
    assert seal["outcomes_opened"] is False
