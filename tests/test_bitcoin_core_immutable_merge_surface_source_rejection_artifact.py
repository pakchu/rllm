from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from training import build_bitcoin_core_immutable_merge_surface_source as builder


ROOT = Path(__file__).resolve().parents[1]
REJECTION = ROOT / (
    "results/bitcoin_core_immutable_merge_surface_source_rejection_2026-07-22.json"
)
SOURCE = ROOT / "data/bitcoin_core_immutable_merge_surface_2020_2023.jsonl.gz"
MANIFEST = ROOT / (
    "results/bitcoin_core_immutable_merge_surface_source_manifest_2026-07-22.json"
)
SUPPORT = ROOT / (
    "results/bitcoin_core_immutable_merge_surface_source_support_2026-07-22.json"
)
EXPECTED_RESULT_HASH = (
    "38650591438b3a7e2dd83de10f086facbe5375ef805160a98786b557d888523c"
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_rows() -> list[dict[str, Any]]:
    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_bcims_rejection_hash_and_frozen_evidence_chain() -> None:
    payload = json.loads(REJECTION.read_text())
    core = {key: value for key, value in payload.items() if key != "result_hash"}
    assert payload["result_hash"] == EXPECTED_RESULT_HASH
    assert builder.canonical_hash(core) == EXPECTED_RESULT_HASH
    assert payload["decision"] == "REJECT_NO_REPAIR"
    for evidence in payload["frozen_evidence"].values():
        assert _sha256((ROOT / evidence["path"]).read_bytes()) == evidence["sha256"]


def test_bcims_rejection_replays_rows_and_exact_failure_shapes() -> None:
    payload = json.loads(REJECTION.read_text())
    rows = _load_rows()
    assert len(rows) == 5550
    assert Counter(row["stratum"] for row in rows) == {
        "primary_core": 3436,
        "gui_comparator": 255,
        "audit_only": 1859,
    }
    audit = [row for row in rows if row["stratum"] == "audit_only"]
    legacy = [
        row
        for row in audit
        if re.fullmatch(r"Merge #[1-9][0-9]*: \S(?:.*\S)?", row["subject"])
    ]
    direct = [row for row in audit if row["parent_count"] == 1]
    assert len(legacy) == 1854
    assert len(direct) == 5
    assert len(legacy) + len(direct) == len(audit)
    assert legacy[0]["subject"] == payload["frozen_failure"]["first_legacy_subject"]
    assert direct[0]["subject"] == payload["frozen_failure"]["first_direct_subject"]
    assert Counter(row["causal_availability_utc"][:4] for row in audit) == {
        "2020": 1422,
        "2021": 435,
        "2022": 2,
    }


def test_bcims_rejection_proves_replay_and_zero_blob_boundary() -> None:
    manifest = json.loads(MANIFEST.read_text())
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert builder.canonical_hash(core) == manifest["manifest_hash"]
    rows = _load_rows()
    fingerprint = builder.source_rows_fingerprint(rows)
    replay = manifest["source_verification"]["deterministic_replay"]
    assert replay["passed"] is True
    assert fingerprint == replay["pass_a"] == replay["pass_b"]
    for stage in (
        "pre_fetch_local_object_inventory",
        "post_fetch_local_object_inventory",
        "post_extraction_local_object_inventory",
    ):
        assert manifest["source_verification"][stage]["object_counts"]["blob"] == 0
    assert "remote_head_at_fetch" not in manifest["source_verification"]


def test_bcims_rejection_matches_failed_support_and_keeps_outcomes_closed() -> None:
    payload = json.loads(REJECTION.read_text())
    support = json.loads(SUPPORT.read_text())
    core = {key: value for key, value in support.items() if key != "result_hash"}
    assert builder.canonical_hash(core) == support["result_hash"]
    assert support["status"] == "REJECT_NO_REPAIR"
    assert [gate["gate_id"] for gate in support["gates"] if not gate["passed"]] == [
        "unknown_fraction_overall",
        "unknown_fraction_each_year",
        "primary_each_year",
        "primary_each_quarter",
        "primary_unique_days_each_year",
        "primary_distinct_surfaces_each_year",
    ]
    boundary = payload["opened_boundaries"]
    assert boundary["full_source_incidence_opened"] is True
    assert boundary["source_support_calculated"] is True
    assert boundary["semantic_model_opened"] is False
    assert boundary["market_clocks_opened"] is False
    assert boundary["market_data_opened"] is False
    assert boundary["outcomes_opened"] is False
    assert boundary["returns_or_pnl_calculated"] is False
