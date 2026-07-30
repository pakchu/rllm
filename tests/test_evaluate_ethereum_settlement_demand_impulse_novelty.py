from __future__ import annotations

import copy
import csv
from dataclasses import replace
from fractions import Fraction
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from training import evaluate_ethereum_settlement_demand_impulse_novelty as novelty


def _ts(value: str) -> int:
    return novelty._parse_timestamp(value)


def _interval(entry: int, side: int = 1, bars: int = 1) -> novelty.SignedInterval:
    return novelty.SignedInterval(entry, entry + bars * 300, side)


def _write_gzip_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    compressed = gzip.compress(stream.getvalue().encode(), mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return compressed


def _directional_spec(
    compressed: bytes,
    header: bytes,
    *,
    path: str = "clock.csv.gz",
    group_column: str | None = None,
    groups: list[str] | None = None,
) -> dict[str, object]:
    required = ["entry", "exit", "side"]
    if group_column is not None:
        required.append(group_column)
    result: dict[str, object] = {
        "path": path,
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "header_line_sha256": hashlib.sha256(header).hexdigest(),
        "filters": {},
        "group_column": group_column,
        "groups": groups or [],
        "capability": "directional_interval",
        "entry_column": "entry",
        "exit_column": "exit",
        "side_column": "side",
        "required_columns": sorted(required),
        "required_metrics": [
            "exact_entry_jaccard",
            "candidate_24h_containment",
            "absolute_signed_exposure_pearson",
        ],
        "comparison_domain": [
            "2023-06-01T00:00:00Z",
            "2023-06-03T00:00:00Z",
        ],
    }
    if group_column is not None:
        result["each_group_is_a_separate_comparator"] = True
    return result


def _bundle_spec(compressed: bytes, header: bytes) -> dict[str, object]:
    return {
        "path": "bundle.csv.gz",
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "header_line_sha256": hashlib.sha256(header).hexdigest(),
        "filters": {},
        "group_column": "comparator",
        "capability_column": "capability",
        "entry_column": "entry",
        "exit_column": "exit",
        "side_column": "side",
        "required_columns": [
            "capability",
            "comparator",
            "entry",
            "exit",
            "side",
        ],
        "directional_interval_groups": ["directional"],
        "timestamp_only_groups": ["timestamps"],
        "required_metrics_by_capability": {
            "directional_interval": [
                "exact_entry_jaccard",
                "candidate_24h_containment",
                "absolute_signed_exposure_pearson",
            ],
            "timestamp_only": [
                "exact_entry_jaccard",
                "candidate_24h_containment",
            ],
        },
        "each_group_is_a_separate_comparator": True,
        "comparison_domain": [
            "2023-06-01T00:00:00Z",
            "2023-06-03T00:00:00Z",
        ],
    }


SUPPORT_CHECK_NAMES = {
    "source_exact_epochs",
    "source_missing_epochs_zero",
    "source_dual_replay_differences_zero",
    "source_boundary_header_differences_zero",
    "future_append_selection_differences_zero",
    "identity_clock_side_rank_tie_source_hash_reproducible",
    "selection_total_min",
    "selection_2023H2_min",
    "selection_2024H1_min",
    "selection_2024H2_min",
    "selection_each_side_min",
    "selection_maximum_month_share",
    "future25_total_min",
    "future25_each_side_min",
    "future25_maximum_month_share",
    "future26_total_min",
    "future26_each_side_min",
    "future26_maximum_month_share",
    "maximum_accepted_entry_gap_days",
    "maximum_same_side_run",
    "base_fee_one_epoch_stale_exact_entry_jaccard_strict",
    "base_fee_one_epoch_stale_candidate_24h_containment_strict",
    "gas_utilization_only_exact_entry_jaccard_strict",
    "gas_utilization_only_candidate_24h_containment_strict",
    "base_fee_no_tail_exact_entry_jaccard_strict",
    "base_fee_no_tail_candidate_24h_containment_strict",
}


def _support_payload(
    *,
    passed: bool = True,
    primary_sha256: str = "1" * 64,
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": novelty.SOURCE_SUPPORT_PROTOCOL_VERSION,
        "policy_id": novelty.POLICY_ID,
        "status": "support_passed_terminal" if passed else "retired_terminal",
        "terminal": True,
        "artifact_eligible": True,
        "decision": (
            "SOURCE_SUPPORT_PASS"
            if passed
            else "RETIRE_ESDI_288_UNCHANGED_BEFORE_OUTCOMES"
        ),
        "support_passed": passed,
        "preregistration": {
            "path": str(novelty.PREREGISTRATION_PATH),
            "sha256": novelty.PREREGISTRATION_SHA256,
            "manifest_hash": novelty.PREREGISTRATION_MANIFEST_HASH,
        },
        "attempt_claim": {
            "path": str(
                novelty.source_support_evaluator.DEFAULT_ATTEMPT_CLAIM
            ),
            "sha256": "c" * 64,
            "claim_hash": "d" * 64,
        },
        "support_checks": {
            name: passed for name in sorted(SUPPORT_CHECK_NAMES)
        },
        "source_contract": {
            "columns": list(novelty.source_support_evaluator.SOURCE_COLUMNS),
            "artifact_eligible": True,
            "rows": novelty.source_builder.EPOCH_COUNT,
            "source_manifest_path": str(
                novelty.source_support_evaluator.DEFAULT_SOURCE_MANIFEST
            ),
            "source_manifest_sha256": "4" * 64,
            "source_manifest_hash": "5" * 64,
            "raw_source_path": str(
                novelty.source_support_evaluator.DEFAULT_RAW_SOURCE
            ),
            "raw_source_bytes": 1,
            "raw_source_rows_decoded": novelty.source_builder.REQUEST_COUNT,
            "raw_source_sha256": "6" * 64,
            "epoch_csv_path": str(
                novelty.source_support_evaluator.DEFAULT_EPOCH_SOURCE
            ),
            "epoch_csv_bytes": 1,
            "epoch_csv_rows_decoded": novelty.source_builder.EPOCH_COUNT,
            "epoch_csv_sha256": "7" * 64,
            "pre_replay_protocol_seal": {"seal_hash": "8" * 64},
            "replay_claim": {
                "path": (
                    novelty.source_support_evaluator.DEFAULT_REPLAY_CLAIM.as_posix()
                ),
                "sha256": "9" * 64,
                "claim_hash": "a" * 64,
            },
            "missing_epochs": 0,
            "dual_replay_differences": 0,
            "boundary_header_differences": 0,
        },
        "feature_rows": 1,
        "feature_rank_tie_state_sha256": "3" * 64,
        "raw_candidate_counts": {
            name: 1
            for name in novelty.source_support_evaluator.CONTROL_ORDER
        },
        "accepted_clock_counts": {
            name: 1
            for name in novelty.source_support_evaluator.CONTROL_ORDER
        },
        "support_audit": {
            "clock_stats": {},
            "selection_report_counts": {},
            "maximum_accepted_entry_gap_seconds": 300,
            "maximum_same_side_run": 1,
            "independent_control_metrics": {},
        },
        "future_append_selection_invariance": {
            "passed": passed,
            "selection_end_utc": "2025-01-01T00:00:00Z",
            "full_rebuild_selection_rows": 1,
            "prefix_rebuild_selection_rows": 1,
            "full_rebuild_selection_sha256": "b" * 64,
            "prefix_rebuild_selection_sha256": "b" * 64,
        },
        "clock_artifacts": {
            "primary_sha256": primary_sha256,
            "controls_sha256": "2" * 64,
        },
        "evidence_boundary": {
            **novelty.SOURCE_SUPPORT_EVIDENCE_BOUNDARY,
            "official_ethereum_raw_rows_opened": (
                novelty.source_builder.REQUEST_COUNT
            ),
            "official_ethereum_epoch_rows_opened": (
                novelty.source_builder.EPOCH_COUNT
            ),
        },
        "later_stage_artifacts_opened": False,
    }
    return {**core, "manifest_hash": novelty.canonical_hash(core)}


def _support_bytes(payload: dict[str, Any]) -> bytes:
    return novelty.source_support_evaluator._json_bytes(payload)


def _support_artifact(
    tmp_path: Path,
    *,
    passed: bool = True,
    primary_sha256: str = "1" * 64,
) -> novelty.VerifiedSourceSupport:
    path = tmp_path / "support.json"
    payload = _support_payload(
        passed=passed,
        primary_sha256=primary_sha256,
    )
    raw = _support_bytes(payload)
    path.write_bytes(raw)
    return novelty.parse_passed_source_support_bytes(
        raw,
        path=path,
        production=False,
    )


def _gross9_clocks(
    base: int,
) -> dict[str, tuple[novelty.SignedInterval, ...]]:
    return {
        name: (_interval(base + 9 * 3600 + index * 300, -1),)
        for index, name in enumerate(novelty.GROSS9_SLEEVES)
    }


def _gross9_artifact_payload(
    registration: dict[str, Any],
    support: novelty.VerifiedSourceSupport,
    base: int,
) -> dict[str, Any]:
    clocks: dict[str, Any] = {}
    for index, name in enumerate(novelty.GROSS9_SLEEVES):
        interval = {
            "entry": str(base + 9 * 3600 + index * 300),
            "exit": str(base + 9 * 3600 + (index + 1) * 300),
            "side": "SHORT",
        }
        clock_core = {"intervals": [interval]}
        clocks[name] = {
            **clock_core,
            "sha256": novelty.canonical_hash(clock_core),
        }
    core = {
        "protocol_version": novelty.GROSS9_CLOCKS_PROTOCOL_VERSION,
        "policy_id": novelty.POLICY_ID,
        "preregistration": {
            "path": str(novelty.PREREGISTRATION_PATH),
            "sha256": novelty.PREREGISTRATION_SHA256,
            "manifest_hash": novelty.PREREGISTRATION_MANIFEST_HASH,
        },
        "source_support": {
            "path": str(support.path),
            "sha256": support.sha256,
            "manifest_hash": support.manifest_hash,
        },
        "authority_hash": novelty.canonical_hash(
            registration["gross9"]["authority"]
        ),
        "clocks": clocks,
        "frozen_contract_validation": novelty.gross9_frozen_contract_validation(
            registration
        ),
    }
    return {**core, "manifest_hash": novelty.canonical_hash(core)}


def _gross9_artifact(
    tmp_path: Path,
    registration: dict[str, Any],
    support: novelty.VerifiedSourceSupport,
    base: int,
) -> novelty.VerifiedGross9Clocks:
    path = tmp_path / "gross9.json"
    payload = _gross9_artifact_payload(registration, support, base)
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return novelty.parse_gross9_clock_artifact_bytes(
        raw,
        path=path,
        registration=registration,
        source_support=support,
        production=False,
    )


def test_preregistration_binding_and_all_18_registry_artifacts() -> None:
    registration = novelty.verify_preregistration()
    registry = novelty.frozen_registry(registration)
    assert registration["manifest_hash"] == novelty.PREREGISTRATION_MANIFEST_HASH
    assert len(registry) == 18
    novelty.validate_registry(registry)
    for field in (
        "sha256",
        "header_line_sha256",
        "required_columns",
        "filters",
        "comparison_domain",
    ):
        drifted = copy.deepcopy(registry)
        first = next(iter(drifted))
        drifted[first][field] = None
        with pytest.raises(novelty.NoveltyTerminalError, match="registry drift"):
            novelty.validate_registry(drifted)


def test_exact_fraction_equality_boundaries_and_strict_gate() -> None:
    assert novelty.inclusive_fraction_gate(Fraction(1, 5), 1, 5)
    assert not novelty.inclusive_fraction_gate(Fraction(1, 5) + Fraction(1, 100), 1, 5)
    assert novelty.strict_fraction_gate(Fraction(19, 20) - Fraction(1, 100), 19, 20)
    assert not novelty.strict_fraction_gate(Fraction(19, 20), 19, 20)


def test_duplicates_unsorted_and_strict_side_are_terminal() -> None:
    with pytest.raises(novelty.NoveltyTerminalError, match="duplicate or unsorted"):
        novelty._canonical_intervals(
            (_interval(300), _interval(300)), "duplicate"
        )
    with pytest.raises(novelty.NoveltyTerminalError, match="duplicate or unsorted"):
        novelty._canonical_intervals(
            (_interval(600), _interval(300)), "unsorted"
        )
    for value in ("long", "1", "+1", " LONG", ""):
        with pytest.raises(novelty.NoveltyTerminalError, match="exactly LONG"):
            novelty._parse_side(value)


def test_common_domain_filter_controls_minimum_count() -> None:
    start = _ts("2023-06-01T00:00:00Z")
    candidate = (_interval(start + 300), _interval(start + 12 * 3600, -1))
    entries = tuple(start - 300 * (12 - index) for index in range(12)) + (
        start + 6 * 3600,
    )
    comparator = novelty.ComparatorClock(
        "small-after-filter",
        "timestamp_only",
        entries,
        None,
        "synthetic",
    )
    result = novelty.evaluate_prior_comparator(
        candidate,
        comparator,
        ["2023-06-01T00:00:00Z", "2023-06-02T00:00:00Z"],
    )
    assert result["comparator_entries"] == 1
    assert result["gating"] is False
    assert result["passed"] is True
    assert result["minimum_count_after_common_domain_filter"] is True


def test_directional_boundary_crossings_are_skipped_not_truncated() -> None:
    start = _ts("2023-06-01T00:00:00Z")
    end = _ts("2023-06-02T00:00:00Z")
    candidate = (
        novelty.SignedInterval(start - 300, start + 300, 1),
        _interval(start + 3600),
        _interval(start + 18 * 3600, -1),
    )
    comparator_intervals = (
        _interval(start + 6 * 3600, -1),
        _interval(start + 20 * 3600),
        novelty.SignedInterval(end - 300, end + 300, -1),
    )
    comparator = novelty.ComparatorClock(
        "boundary",
        "directional_interval",
        tuple(row.entry for row in comparator_intervals),
        comparator_intervals,
        "synthetic",
    )
    result = novelty.evaluate_prior_comparator(
        candidate,
        comparator,
        ["2023-06-01T00:00:00Z", "2023-06-02T00:00:00Z"],
    )
    assert result["candidate_entries"] == 2
    assert result["comparator_entries"] == 2
    assert result["gating"] is False


def test_grouped_artifact_is_split_into_separate_clocks(tmp_path: Path) -> None:
    columns = ["entry", "exit", "side", "variant"]
    rows = [
        {
            "entry": "2023-06-01T01:00:00Z",
            "exit": "2023-06-01T01:05:00Z",
            "side": "LONG",
            "variant": "a",
        },
        {
            "entry": "2023-06-01T02:00:00Z",
            "exit": "2023-06-01T02:05:00Z",
            "side": "SHORT",
            "variant": "b",
        },
    ]
    compressed = _write_gzip_csv(tmp_path / "clock.csv.gz", columns, rows)
    spec = _directional_spec(
        compressed, b"entry,exit,side,variant\n", group_column="variant", groups=["a", "b"]
    )
    registry = {"grouped": spec}
    clocks = novelty.load_comparator_artifacts(
        registry, repository_root=tmp_path, expected_registry=registry
    )
    assert set(clocks) == {"grouped:a", "grouped:b"}
    assert clocks["grouped:a"].intervals == (
        _interval(_ts("2023-06-01T01:00:00Z")),
    )


def test_timestamp_only_is_frozen_and_never_parses_blank_side(tmp_path: Path) -> None:
    columns = ["capability", "comparator", "entry", "exit", "side"]
    rows = [
        {
            "capability": "directional_interval",
            "comparator": "directional",
            "entry": "2023-06-01T01:00:00Z",
            "exit": "2023-06-01T01:05:00Z",
            "side": "LONG",
        },
        {
            "capability": "timestamp_only",
            "comparator": "timestamps",
            "entry": "2023-06-01T02:00:00Z",
            "exit": "",
            "side": "",
        },
    ]
    compressed = _write_gzip_csv(tmp_path / "bundle.csv.gz", columns, rows)
    registry = {
        "bundle": _bundle_spec(
            compressed, b"capability,comparator,entry,exit,side\n"
        )
    }
    clocks = novelty.load_comparator_artifacts(
        registry, repository_root=tmp_path, expected_registry=registry
    )
    timestamp_clock = clocks["bundle:timestamps"]
    assert timestamp_clock.capability == "timestamp_only"
    assert timestamp_clock.intervals is None
    start = _ts("2023-06-01T00:00:00Z")
    result = novelty.evaluate_prior_comparator(
        (_interval(start + 300), _interval(start + 12 * 3600, -1)),
        timestamp_clock,
        registry["bundle"]["comparison_domain"],
    )
    assert result["metrics"]["squared_signed_exposure_pearson"] == {
        "applicable": False,
        "reason": "frozen_timestamp_only_capability",
    }


def test_loader_rejects_missing_columns_hash_and_capability(tmp_path: Path) -> None:
    columns = ["entry", "exit"]
    compressed = _write_gzip_csv(
        tmp_path / "clock.csv.gz",
        columns,
        [{"entry": "2023-06-01T01:00:00Z", "exit": "2023-06-01T01:05:00Z"}],
    )
    missing = {
        "clock": _directional_spec(compressed, b"entry,exit\n")
    }
    with pytest.raises(novelty.NoveltyTerminalError, match="required columns"):
        novelty.load_comparator_artifacts(
            missing, repository_root=tmp_path, expected_registry=missing
        )

    columns = ["entry", "exit", "side"]
    compressed = _write_gzip_csv(
        tmp_path / "clock.csv.gz",
        columns,
        [
            {
                "entry": "2023-06-01T01:00:00Z",
                "exit": "2023-06-01T01:05:00Z",
                "side": "LONG",
            }
        ],
    )
    bad_hash_spec = _directional_spec(compressed, b"entry,exit,side\n")
    bad_hash_spec["sha256"] = "0" * 64
    bad_hash = {"clock": bad_hash_spec}
    with pytest.raises(novelty.NoveltyTerminalError, match="artifact hash drift"):
        novelty.load_comparator_artifacts(
            bad_hash, repository_root=tmp_path, expected_registry=bad_hash
        )

    unknown = copy.deepcopy(bad_hash)
    unknown["clock"]["capability"] = "unknown"
    with pytest.raises(novelty.NoveltyTerminalError, match="unknown capability"):
        novelty.validate_registry(unknown, unknown)


def test_source_support_failure_prevents_comparator_loader_call(
    tmp_path: Path,
) -> None:
    support = tmp_path / "support.json"
    support.write_bytes(_support_bytes(_support_payload(passed=False)))
    called = False

    def forbidden_loader(registry: object) -> dict[str, novelty.ComparatorClock]:
        nonlocal called
        called = True
        raise AssertionError("comparator loader must not be called")

    with pytest.raises(novelty.NoveltyTerminalError, match="did not pass"):
        verified = novelty.parse_passed_source_support_bytes(
            support.read_bytes(),
            path=support,
            production=False,
        )
        novelty.build_report_after_source_support(
            source_support=verified,
            candidate=(),
            gross9_artifact=None,
            comparator_loader=forbidden_loader,
        )
    assert called is False


def test_exact_source_support_schema_rejects_aliases_and_forgery(
    tmp_path: Path,
) -> None:
    exact = _support_payload()
    raw = _support_bytes(exact)
    verified = novelty.parse_passed_source_support_bytes(
        raw,
        path=tmp_path / "support.json",
        production=False,
    )
    assert verified.manifest_hash == exact["manifest_hash"]
    with pytest.raises(TypeError):
        verified.payload["support_passed"] = False

    mutations = []
    alias = copy.deepcopy(exact)
    alias.pop("support_passed")
    alias["source_support_passed"] = True
    mutations.append(alias)
    bad_protocol = copy.deepcopy(exact)
    bad_protocol["protocol_version"] = "forged"
    mutations.append(bad_protocol)
    false_check = copy.deepcopy(exact)
    false_check["support_checks"][next(iter(SUPPORT_CHECK_NAMES))] = False
    mutations.append(false_check)
    bad_boundary = copy.deepcopy(exact)
    bad_boundary["evidence_boundary"]["comparator_rows_opened"] = 1
    mutations.append(bad_boundary)
    later_open = copy.deepcopy(exact)
    later_open["later_stage_artifacts_opened"] = True
    mutations.append(later_open)
    no_hash = copy.deepcopy(exact)
    no_hash.pop("manifest_hash")
    mutations.append(no_hash)
    for payload in mutations:
        if "manifest_hash" in payload:
            payload_core = {
                key: value
                for key, value in payload.items()
                if key != "manifest_hash"
            }
            payload["manifest_hash"] = novelty.canonical_hash(payload_core)
        with pytest.raises(novelty.NoveltyTerminalError):
            novelty.parse_passed_source_support_bytes(
                _support_bytes(payload),
                path=tmp_path / "support.json",
                production=False,
            )

    forged = copy.deepcopy(exact)
    forged["decision"] = "SOURCE_SUPPORT_PASS_FORGED"
    with pytest.raises(novelty.NoveltyTerminalError, match="manifest"):
        novelty.parse_passed_source_support_bytes(
            _support_bytes(forged),
            path=tmp_path / "support.json",
            production=False,
        )
    with pytest.raises(novelty.NoveltyTerminalError, match="canonical path"):
        novelty.parse_passed_source_support_bytes(
            raw,
            path=tmp_path / "support.json",
            production=True,
        )
    noncanonical = (json.dumps(exact, sort_keys=True) + "\n").encode()
    with pytest.raises(
        novelty.NoveltyTerminalError,
        match="producer-canonical",
    ):
        novelty.parse_passed_source_support_bytes(
            noncanonical,
            path=tmp_path / "support.json",
            production=False,
        )


def test_production_source_support_authentication_has_positive_committed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        novelty.source_support_evaluator,
        "REPOSITORY_ROOT",
        tmp_path,
    )
    attempt_claim = novelty.source_support_evaluator._create_attempt_claim()
    payload = _support_payload()
    payload["attempt_claim"] = attempt_claim
    payload_core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    payload["manifest_hash"] = novelty.canonical_hash(payload_core)
    raw = _support_bytes(payload)
    artifact = tmp_path / novelty.DEFAULT_SOURCE_SUPPORT_PATH
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(raw)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "support artifact"],
        check=True,
    )
    source_contract = payload["source_contract"]
    source_audit = {
        key: value
        for key, value in source_contract.items()
        if key not in {"columns", "rows"}
    }
    monkeypatch.setattr(novelty, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        novelty.source_support_evaluator,
        "load_source_manifest",
        lambda: (range(novelty.source_builder.EPOCH_COUNT), source_audit),
    )
    verified = novelty.load_passed_source_support(production=True)
    assert verified.raw_bytes == raw
    assert verified.manifest_hash == payload["manifest_hash"]
    artifact.write_bytes(raw + b" ")
    with pytest.raises(novelty.NoveltyTerminalError, match="committed and clean"):
        novelty.parse_passed_source_support_bytes(
            raw,
            path=novelty.DEFAULT_SOURCE_SUPPORT_PATH,
            production=True,
        )


def test_candidate_clock_loader_uses_passed_support_hash(tmp_path: Path) -> None:
    columns = [
        "policy_id",
        "control",
        "entry_time_utc",
        "exit_time_utc",
        "side",
    ]
    compressed = _write_gzip_csv(
        tmp_path / "primary.csv.gz",
        columns,
        [
            {
                "policy_id": novelty.POLICY_ID,
                "control": "primary",
                "entry_time_utc": "2023-06-01T01:00:00Z",
                "exit_time_utc": "2023-06-01T01:05:00Z",
                "side": "LONG",
            }
        ],
    )
    support = _support_artifact(
        tmp_path,
        primary_sha256=hashlib.sha256(compressed).hexdigest(),
    )
    assert novelty.load_candidate_clock_csv(
        tmp_path / "primary.csv.gz", support
    ) == (_interval(_ts("2023-06-01T01:00:00Z")),)
    forged_payload = _support_payload(primary_sha256="0" * 64)
    forged_raw = _support_bytes(forged_payload)
    forged_support = novelty.parse_passed_source_support_bytes(
        forged_raw,
        path=tmp_path / "forged-support.json",
        production=False,
    )
    with pytest.raises(novelty.NoveltyTerminalError, match="clock hash drift"):
        novelty.load_candidate_clock_csv(
            tmp_path / "primary.csv.gz",
            forged_support,
        )


def test_missing_unknown_and_zero_variance_are_terminal() -> None:
    start = _ts("2023-06-01T00:00:00Z")
    candidate = (_interval(start + 300),)
    missing = novelty.ComparatorClock(
        "missing", "directional_interval", (start + 600,), None, "synthetic"
    )
    with pytest.raises(novelty.NoveltyTerminalError, match="intervals missing"):
        novelty.evaluate_prior_comparator(
            candidate,
            missing,
            ["2023-06-01T00:00:00Z", "2023-06-02T00:00:00Z"],
        )
    unknown = novelty.ComparatorClock(
        "unknown", "new_capability", (start + 600,), None, "synthetic"
    )
    with pytest.raises(novelty.NoveltyTerminalError, match="unknown capability"):
        novelty.evaluate_prior_comparator(
            candidate,
            unknown,
            ["2023-06-01T00:00:00Z", "2023-06-02T00:00:00Z"],
        )
    full = novelty.ComparatorClock(
        "full",
        "directional_interval",
        (start,),
        (novelty.SignedInterval(start, start + 24 * 3600, 1),),
        "synthetic",
    )
    with pytest.raises(novelty.NoveltyTerminalError, match="undefined exposure"):
        novelty.evaluate_prior_comparator(
            (novelty.SignedInterval(start, start + 24 * 3600, 1),),
            full,
            ["2023-06-01T00:00:00Z", "2023-06-02T00:00:00Z"],
        )


def test_all_five_gross9_sleeves_are_separate_and_apply_exact_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        novelty,
        "GROSS9_DOMAIN",
        ("2023-06-01T00:00:00Z", "2023-06-04T00:00:00Z"),
    )
    start = _ts("2023-06-01T00:00:00Z")
    candidate = (
        _interval(start + 300),
        _interval(start + 36 * 3600, -1),
    )
    sleeves = _gross9_clocks(start)
    novelty.validate_gross9_sleeves(sleeves)
    results = [
        novelty.evaluate_gross9_sleeve(candidate, name, sleeves[name])
        for name in novelty.GROSS9_SLEEVES
    ]
    assert [result["sleeve"] for result in results] == list(novelty.GROSS9_SLEEVES)
    assert all(result["passed"] for result in results)
    assert all(
        set(result["checks"])
        == {
            "exact_entry_jaccard",
            "candidate_6h_containment",
            "occupied_bar_jaccard",
            "squared_signed_exposure_pearson",
        }
        for result in results
    )
    incomplete = dict(sleeves)
    incomplete.pop(novelty.GROSS9_SLEEVES[-1])
    with pytest.raises(novelty.NoveltyTerminalError, match="exactly all five"):
        novelty.validate_gross9_sleeves(incomplete)


def test_gross9_clock_artifact_is_hash_bound_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    registration = novelty.verify_preregistration()
    support = _support_artifact(tmp_path)
    base = _ts("2023-06-01T00:00:00Z")
    payload = _gross9_artifact_payload(registration, support, base)
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    verified = novelty.parse_gross9_clock_artifact_bytes(
        raw,
        path=tmp_path / "gross9.json",
        registration=registration,
        source_support=support,
        production=False,
    )
    assert tuple(verified.clocks) == novelty.GROSS9_SLEEVES
    assert verified.payload["frozen_contract_validation"][
        "exact_runtime_config_and_transitive_hash_validation_passed"
    ] is True

    tampered = copy.deepcopy(payload)
    sleeve = novelty.GROSS9_SLEEVES[0]
    tampered["clocks"][sleeve]["intervals"][0]["side"] = "LONG"
    tampered_core = {
        key: value for key, value in tampered.items() if key != "manifest_hash"
    }
    tampered["manifest_hash"] = novelty.canonical_hash(tampered_core)
    with pytest.raises(novelty.NoveltyTerminalError, match="sleeve hash"):
        novelty.parse_gross9_clock_artifact_bytes(
            (json.dumps(tampered, sort_keys=True) + "\n").encode(),
            path=tmp_path / "gross9.json",
            registration=registration,
            source_support=support,
            production=False,
        )

    missing = copy.deepcopy(payload)
    missing["clocks"].pop(novelty.GROSS9_SLEEVES[-1])
    missing_core = {
        key: value for key, value in missing.items() if key != "manifest_hash"
    }
    missing["manifest_hash"] = novelty.canonical_hash(missing_core)
    with pytest.raises(novelty.NoveltyTerminalError, match="exactly five"):
        novelty.parse_gross9_clock_artifact_bytes(
            (json.dumps(missing, sort_keys=True) + "\n").encode(),
            path=tmp_path / "gross9.json",
            registration=registration,
            source_support=support,
            production=False,
        )

    authority = copy.deepcopy(payload)
    authority["authority_hash"] = "0" * 64
    authority_core = {
        key: value for key, value in authority.items() if key != "manifest_hash"
    }
    authority["manifest_hash"] = novelty.canonical_hash(authority_core)
    with pytest.raises(novelty.NoveltyTerminalError, match="authority"):
        novelty.parse_gross9_clock_artifact_bytes(
            (json.dumps(authority, sort_keys=True) + "\n").encode(),
            path=tmp_path / "gross9.json",
            registration=registration,
            source_support=support,
            production=False,
        )

    frozen = copy.deepcopy(payload)
    frozen["frozen_contract_validation"][
        "five_signed_sleeves_validated"
    ] = False
    frozen_core = {
        key: value for key, value in frozen.items() if key != "manifest_hash"
    }
    frozen["manifest_hash"] = novelty.canonical_hash(frozen_core)
    with pytest.raises(novelty.NoveltyTerminalError, match="frozen-contract"):
        novelty.parse_gross9_clock_artifact_bytes(
            (json.dumps(frozen, sort_keys=True) + "\n").encode(),
            path=tmp_path / "gross9.json",
            registration=registration,
            source_support=support,
            production=False,
        )
    with pytest.raises(novelty.NoveltyTerminalError, match="canonical path"):
        novelty.parse_gross9_clock_artifact_bytes(
            raw,
            path=tmp_path / "gross9.json",
            registration=registration,
            source_support=support,
            production=True,
        )
    with pytest.raises(novelty.NoveltyTerminalError, match="committed and clean"):
        novelty.parse_gross9_clock_artifact_bytes(
            raw,
            path=novelty.DEFAULT_GROSS9_CLOCKS_PATH,
            registration=registration,
            source_support=support,
            production=True,
        )
    with pytest.raises(novelty.NoveltyTerminalError, match="verified Gross9"):
        novelty.evaluate_novelty((), {}, {}, _gross9_clocks(base))


def test_production_gross9_completion_forbids_runtime_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from training import (
        evaluate_ethereum_settlement_demand_impulse_economics as economics,
    )

    registration = novelty.verify_preregistration()
    support = _support_artifact(tmp_path)
    base = _ts("2023-06-01T00:00:00Z")
    expected = _gross9_artifact_payload(registration, support, base)
    monkeypatch.setattr(novelty, "REPOSITORY_ROOT", tmp_path)
    gross9_claim = tmp_path / economics.GROSS9_ATTEMPT_CLAIM
    gross9_claim.parent.mkdir(parents=True, exist_ok=True)
    gross9_claim.write_bytes(b"synthetic committed claim\n")
    monkeypatch.setattr(
        novelty,
        "_require_canonical_committed_clean",
        lambda *args, **kwargs: (
            novelty.REPOSITORY_ROOT / novelty.DEFAULT_GROSS9_CLOCKS_PATH
        ),
    )
    monkeypatch.setattr(
        economics,
        "validate_frozen_contract",
        lambda _: {"validated": True},
    )
    monkeypatch.setattr(
        economics,
        "_validate_evaluator_source_identity",
        lambda: {"manifest_hash": "c" * 64},
    )
    monkeypatch.setattr(
        economics,
        "_load_exact_attempt_claim",
        lambda *_args, **_kwargs: {
            "path": str(economics.GROSS9_ATTEMPT_CLAIM),
            "sha256": "d" * 64,
            "claim_hash": "e" * 64,
        },
    )
    monkeypatch.setattr(
        economics,
        "_reconstruct_gross9_runtime_clocks",
        lambda: (_ for _ in ()).throw(
            AssertionError(
                "completed Gross9 paths must never be reconstructed again"
            )
        ),
    )
    raw = (json.dumps(expected, sort_keys=True) + "\n").encode()
    verified = novelty.parse_gross9_clock_artifact_bytes(
        raw,
        path=novelty.DEFAULT_GROSS9_CLOCKS_PATH,
        registration=registration,
        source_support=support,
        production=True,
    )
    assert verified.payload["manifest_hash"] == expected["manifest_hash"]


def test_gross9_tamper_fails_before_comparator_access(
    tmp_path: Path,
) -> None:
    registration = novelty.verify_preregistration()
    support = _support_artifact(tmp_path)
    gross9 = _gross9_artifact(
        tmp_path,
        registration,
        support,
        _ts("2023-06-01T00:00:00Z"),
    )
    tampered = bytearray(gross9.raw_bytes)
    tampered[-2] = ord(" ")
    forged = replace(gross9, raw_bytes=bytes(tampered))
    called = False

    def forbidden_loader(registry: object) -> dict[str, novelty.ComparatorClock]:
        nonlocal called
        called = True
        raise AssertionError("comparators opened before Gross9 validation")

    with pytest.raises(novelty.NoveltyTerminalError):
        novelty.build_report_after_source_support(
            source_support=support,
            candidate=(),
            gross9_artifact=forged,
            comparator_loader=forbidden_loader,
            registration=registration,
        )
    assert called is False


def test_support_bytes_are_consumed_once_without_toctou_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = novelty.verify_preregistration()
    support = _support_artifact(tmp_path)
    original_sha = support.sha256
    gross9 = _gross9_artifact(
        tmp_path,
        registration,
        support,
        _ts("2023-06-01T00:00:00Z"),
    )
    support.path.write_text('{"forged":true}\n')
    events: list[str] = []

    def comparator_loader(
        registry: object,
    ) -> dict[str, novelty.ComparatorClock]:
        events.append("comparators")
        return {}

    def fake_evaluate(
        candidate: object,
        comparators: object,
        registry: object,
        artifact: object,
    ) -> dict[str, object]:
        assert isinstance(artifact, novelty.VerifiedGross9Clocks)
        assert artifact.sha256 == gross9.sha256
        return {"passed": True, "terminal": False, "failed_checks": []}

    monkeypatch.setattr(novelty, "evaluate_novelty", fake_evaluate)
    report = novelty.build_report_after_source_support(
        source_support=support,
        candidate=(),
        gross9_artifact=gross9,
        comparator_loader=comparator_loader,
        registration=registration,
    )
    assert events == ["comparators"]
    assert report["source_support"]["sha256"] == original_sha
    assert report["source_support"]["artifact"]["support_passed"] is True
    assert report["evidence_boundary"] == novelty.NOVELTY_EVIDENCE_BOUNDARY
    assert report["evidence_boundary"][
        "gross9_outcome_dependent_clock_paths_certified"
    ] is True
    assert report["evidence_boundary"][
        "future_rows_used_for_economic_weight_ranking"
    ] is False
    assert report["evidence_boundary"][
        "future_rows_used_for_structural_candidate_veto"
    ] is True


def test_novelty_attempt_claim_precedes_report_and_forbids_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = novelty.verify_preregistration()
    support = _support_artifact(tmp_path)
    gross9 = _gross9_artifact(
        tmp_path,
        registration,
        support,
        _ts("2023-06-01T00:00:00Z"),
    )
    candidate_path = (
        tmp_path
        / novelty.source_support_evaluator.DEFAULT_PRIMARY_CLOCK_OUTPUT
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_bytes = b"immutable-candidate-clock"
    candidate_path.write_bytes(candidate_bytes)
    calls: list[str] = []

    monkeypatch.setattr(novelty, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "novelty",
            "--candidate-clock",
            str(
                novelty.source_support_evaluator.DEFAULT_PRIMARY_CLOCK_OUTPUT
            ),
        ],
    )
    monkeypatch.setattr(novelty, "verify_preregistration", lambda: registration)
    monkeypatch.setattr(
        novelty,
        "load_passed_source_support",
        lambda *_args, **_kwargs: support,
    )
    monkeypatch.setattr(
        novelty,
        "load_candidate_clock_csv",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        novelty,
        "load_gross9_clock_artifact",
        lambda **_kwargs: gross9,
    )

    def fail_report(**kwargs: Any) -> dict[str, Any]:
        calls.append("report")
        expected = novelty._attempt_claim_payload(
            source_support=support,
            gross9_artifact=gross9,
            candidate_clock={
                "path": str(
                    novelty.source_support_evaluator.DEFAULT_PRIMARY_CLOCK_OUTPUT
                ),
                "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            },
        )
        assert kwargs["attempt_claim"] == novelty.load_attempt_claim(expected)
        raise novelty.NoveltyTerminalError("synthetic comparator failure")

    monkeypatch.setattr(
        novelty,
        "build_report_after_source_support",
        fail_report,
    )
    with pytest.raises(
        novelty.NoveltyTerminalError,
        match="synthetic comparator failure",
    ):
        novelty.main()

    assert calls == ["report"]
    assert (tmp_path / novelty.DEFAULT_ATTEMPT_CLAIM_PATH).is_file()
    assert not (tmp_path / novelty.DEFAULT_OUTPUT_PATH).exists()
    with pytest.raises(
        novelty.NoveltyTerminalError,
        match="claimed without completion",
    ):
        novelty.main()
    assert calls == ["report"]


def test_economics_novelty_loader_rejects_nonreproducing_self_hashed_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = novelty.verify_preregistration()
    support = _support_artifact(tmp_path)
    gross9 = _gross9_artifact(
        tmp_path,
        registration,
        support,
        _ts("2023-06-01T00:00:00Z"),
    )
    candidate_path = (
        tmp_path
        / novelty.source_support_evaluator.DEFAULT_PRIMARY_CLOCK_OUTPUT
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_bytes(b"candidate")
    claim_path = tmp_path / novelty.DEFAULT_ATTEMPT_CLAIM_PATH
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_bytes(b"claim")
    forged_core = {
        "protocol_version": novelty.PROTOCOL_VERSION,
        "policy_id": novelty.POLICY_ID,
        "novelty": {"passed": True},
        "forged": True,
    }
    forged = {
        **forged_core,
        "manifest_hash": novelty.canonical_hash(forged_core),
    }
    output_path = tmp_path / novelty.DEFAULT_OUTPUT_PATH
    output_path.write_bytes(novelty.canonical_report_bytes(forged))

    monkeypatch.setattr(novelty, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        novelty,
        "_require_canonical_committed_clean",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(novelty, "verify_preregistration", lambda: registration)
    monkeypatch.setattr(
        novelty,
        "load_passed_source_support",
        lambda *_args, **_kwargs: support,
    )
    monkeypatch.setattr(
        novelty,
        "load_candidate_clock_csv",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        novelty,
        "load_gross9_clock_artifact",
        lambda **_kwargs: gross9,
    )
    monkeypatch.setattr(
        novelty,
        "load_attempt_claim",
        lambda _expected: {
            "path": str(novelty.DEFAULT_ATTEMPT_CLAIM_PATH),
            "sha256": "a" * 64,
            "claim_hash": "b" * 64,
        },
    )
    monkeypatch.setattr(
        novelty,
        "build_report_after_source_support",
        lambda **_kwargs: {"independently_reproduced": True},
    )

    with pytest.raises(
        novelty.NoveltyTerminalError,
        match="did not reproduce exactly",
    ):
        novelty.load_reproduced_novelty_for_economics()


def test_write_once_novelty_json_freezes_production_path(
    tmp_path: Path,
) -> None:
    core = {"protocol_version": novelty.PROTOCOL_VERSION, "passed": True}
    payload = {**core, "manifest_hash": novelty.canonical_hash(core)}
    output = tmp_path / "novelty.json"
    with pytest.raises(novelty.NoveltyTerminalError, match="canonical output"):
        novelty.write_once_novelty_json(payload, output)
    assert (
        novelty.write_once_novelty_json_for_test(payload, output)
        == "created"
    )
    first = output.read_bytes()
    assert (
        novelty.write_once_novelty_json_for_test(payload, output)
        == "verified_existing"
    )
    assert output.read_bytes() == first
    drift_core = {"protocol_version": novelty.PROTOCOL_VERSION, "passed": False}
    drift = {**drift_core, "manifest_hash": novelty.canonical_hash(drift_core)}
    with pytest.raises(novelty.NoveltyTerminalError, match="output drift"):
        novelty.write_once_novelty_json_for_test(drift, output)


def test_novelty_publish_links_only_fsynced_complete_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = {"protocol_version": novelty.PROTOCOL_VERSION, "passed": True}
    payload = {**core, "manifest_hash": novelty.canonical_hash(core)}
    expected = novelty.canonical_report_bytes(payload)
    output = tmp_path / "novelty.json"
    observed: list[Path] = []

    def fail_link(
        source: str | Path,
        target: str | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        assert follow_symlinks is False
        assert Path(target) == output
        assert not output.exists()
        staged = Path(source)
        assert staged.read_bytes() == expected
        assert staged.stat().st_mode & 0o777 == 0o444
        observed.append(staged)
        raise OSError("synthetic link interruption")

    monkeypatch.setattr(novelty.os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic link interruption"):
        novelty.write_once_novelty_json_for_test(payload, output)
    assert len(observed) == 1
    assert not output.exists()
    assert not observed[0].exists()
