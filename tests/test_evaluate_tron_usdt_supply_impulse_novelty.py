from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import pandas as pd
import pytest

from training import build_tron_usdt_supply_events as builder
from training import evaluate_tron_usdt_supply_impulse_novelty as novelty
from training import (
    evaluate_tron_usdt_supply_impulse_source_support as source_support,
)

OPAQUE_SOURCE_BYTES = b"opaque-source-data"


def _timestamp(value: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    assert isinstance(result, pd.Timestamp)
    return result


def _source_row(
    number: int,
    available: str,
    *,
    event_type: str,
) -> dict[str, Any]:
    event_time = _timestamp(available) - pd.Timedelta(minutes=4)
    assert isinstance(event_time, pd.Timestamp)
    return {
        "event_type": event_type,
        "supply_direction": 1 if event_type == "Issue" else -1,
        "actor_address": "0x" + f"{number + 1:040x}"[-40:],
        "amount_raw": 1_000_000 + number,
        "block_number": builder.SOURCE_START_BLOCK + number,
        "block_hash": "0x" + f"{number + 101:064x}"[-64:],
        "transaction_hash": "0x" + f"{number + 201:064x}"[-64:],
        "transaction_index": 0,
        "log_index": 0,
        "paired_transfer_log_index": 1,
        "event_timestamp_utc": event_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "confirmation_block": (
            builder.SOURCE_START_BLOCK + number + builder.CONFIRMATION_BLOCKS
        ),
        "confirmation_block_hash": "0x" + f"{number + 301:064x}"[-64:],
        "available_at_utc": available,
    }


def _source_frame() -> pd.DataFrame:
    dates = (
        "2023-06-10T00:00:00Z",
        "2023-09-10T00:00:00Z",
        "2023-12-10T00:00:00Z",
        "2024-02-10T00:00:00Z",
        "2024-05-10T00:00:00Z",
        "2024-08-10T00:00:00Z",
        "2024-11-10T00:00:00Z",
        "2024-12-20T00:00:00Z",
        "2025-02-10T00:00:00Z",
        "2025-05-10T00:00:00Z",
        "2025-08-10T00:00:00Z",
        "2025-11-10T00:00:00Z",
        "2026-02-10T00:00:00Z",
        "2026-04-10T00:00:00Z",
    )
    rows = [
        _source_row(
            index,
            date,
            event_type="Issue" if index % 2 == 0 else "Redeem",
        )
        for index, date in enumerate(dates)
    ]
    return pd.DataFrame(rows, columns=pd.Index(builder.CSV_COLUMNS))


def _minimal_source_registration() -> dict[str, Any]:
    core = {
        "policy_id": source_support.POLICY_ID,
        "feature_and_signal": {"eligible_event_types": ["Issue", "Redeem"]},
        "execution": {"hold_hours": 168},
        "source": {"deprecate_terminates_source_v1": True},
    }
    return {**core, "manifest_hash": source_support.canonical_hash(core)}


def _support_fixture(
    tmp_path: Path,
) -> tuple[
    novelty.VerifiedSourceSupport,
    novelty.VerifiedCandidateClock,
    bytes,
]:
    audit = {
        "artifact_eligible": True,
        "source_csv_path": str(source_support.DEFAULT_SOURCE_CSV),
        "source_csv_sha256": "a" * 64,
        "source_csv_bytes": 1,
        "source_manifest_path": str(source_support.DEFAULT_SOURCE_MANIFEST),
        "source_manifest_sha256": "b" * 64,
        "source_manifest_hash": "c" * 64,
        "source_integrity": dict(builder.ZERO_SOURCE_INTEGRITY),
    }
    report, primary, _ = source_support._build_support_from_frame(
        _source_frame(),
        registration=_minimal_source_registration(),
        source_audit=audit,
        artifact_eligible=True,
    )
    report["registration"]["manifest_hash"] = (
        novelty.PREREGISTRATION_MANIFEST_HASH
    )
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    report["manifest_hash"] = novelty.canonical_hash(core)
    report_path = tmp_path / "support.json"
    report_raw = source_support._json_bytes(report)
    report_path.write_bytes(report_raw)
    primary_path = tmp_path / "primary.csv.gz"
    primary_path.write_bytes(primary)
    verified = novelty.parse_passed_source_support_bytes(
        report_raw,
        path=report_path,
        production=False,
    )
    candidate = novelty.load_candidate_clock_csv(primary_path, verified)
    return verified, candidate, primary


def _registration() -> dict[str, Any]:
    path = novelty.REPOSITORY_ROOT / novelty.PREREGISTRATION_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _gross9_fixture(
    tmp_path: Path,
    registration: Mapping[str, Any],
    support: novelty.VerifiedSourceSupport,
) -> novelty.VerifiedGross9Clocks:
    base = novelty._parse_timestamp("2023-06-02T00:00:00Z")
    clocks: dict[str, Any] = {}
    for index, sleeve in enumerate(novelty.GROSS9_SLEEVES):
        row = {
            "entry": str(base + index * 600),
            "exit": str(base + index * 600 + 300),
            "side": "LONG" if index % 2 == 0 else "SHORT",
        }
        clock_core = {"intervals": [row]}
        clocks[sleeve] = {
            **clock_core,
            "sha256": novelty.canonical_hash(clock_core),
        }
    esdi_registration = {
        "gross9": {
            "authority": {
                "clock_reconstruction": registration["gross9"]["authority"][
                    "clock_reconstruction"
                ]
            }
        },
        "novelty": {"gross9_common_domain": list(novelty.GROSS9_DOMAIN)},
    }
    frozen_validation = novelty.esdi_novelty.gross9_frozen_contract_validation(
        esdi_registration
    )
    source_binding = {
        "path": str(novelty.esdi_novelty.DEFAULT_SOURCE_SUPPORT_PATH),
        "sha256": "d" * 64,
        "manifest_hash": "e" * 64,
    }
    core = {
        "protocol_version": novelty.GROSS9_CLOCKS_PROTOCOL_VERSION,
        "policy_id": novelty.esdi_novelty.POLICY_ID,
        "preregistration": {
            "path": str(novelty.ESDI_PREREGISTRATION_PATH),
            "sha256": novelty.ESDI_PREREGISTRATION_SHA256,
            "manifest_hash": novelty.ESDI_PREREGISTRATION_MANIFEST_HASH,
        },
        "source_support": source_binding,
        "authority_hash": novelty.GROSS9_AUTHORITY_SHA256,
        "clocks": clocks,
        "frozen_contract_validation": frozen_validation,
    }
    payload = {**core, "manifest_hash": novelty.canonical_hash(core)}
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path = tmp_path / "gross9.json"
    path.write_bytes(raw)
    return novelty.parse_gross9_clock_artifact_bytes(
        raw,
        path=path,
        registration=registration,
        source_support=support,
        production=False,
        synthetic_authentication=novelty.InjectedVerifiedGross9Fixture(
            source_support_binding=source_binding,
            frozen_contract_validation=frozen_validation,
            authority_hash=novelty.GROSS9_AUTHORITY_SHA256,
        ),
    )


def _synthetic_comparators(
    registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, novelty.ComparatorClock]:
    result: dict[str, novelty.ComparatorClock] = {}
    for artifact, spec in registry.items():
        start = novelty._parse_timestamp(spec["comparison_domain"][0])
        entry = start + 24 * 3600
        if spec.get("capability") == "directional_interval":
            groups: Iterable[str | None] = (
                spec["groups"] if spec.get("group_column") else (None,)
            )
            capabilities = {group: "directional_interval" for group in groups}
        else:
            capabilities = {
                **{
                    group: "directional_interval"
                    for group in spec["directional_interval_groups"]
                },
                **{
                    group: "timestamp_only"
                    for group in spec["timestamp_only_groups"]
                },
            }
        for offset, (group, capability) in enumerate(capabilities.items()):
            clock_entry = entry + offset * 600
            identity = artifact if group is None else f"{artifact}:{group}"
            intervals = (
                (novelty.SignedInterval(clock_entry, clock_entry + 300, 1),)
                if capability == "directional_interval"
                else None
            )
            result[identity] = novelty.ComparatorClock(
                comparator_id=identity,
                capability=capability,
                entries=(clock_entry,),
                intervals=intervals,
                artifact_name=artifact,
                group=group,
            )
    return result


def _complete_report(tmp_path: Path) -> dict[str, Any]:
    inputs = _complete_authenticated_inputs(tmp_path)
    return novelty.build_report_from_authenticated_inputs(inputs)


def _complete_authenticated_inputs(
    tmp_path: Path,
) -> novelty.AuthenticatedNoveltyInputs:
    support, candidate, _ = _support_fixture(tmp_path)
    registration = _registration()
    gross9 = _gross9_fixture(tmp_path, registration, support)
    registry = novelty.frozen_registry(registration)
    comparators = _synthetic_comparators(registry)
    return novelty.AuthenticatedNoveltyInputs(
        registration=registration,
        source_support=support,
        candidate=candidate,
        comparators=novelty.VerifiedComparatorClocks(
            clocks=comparators,
            artifact_bytes={},
            artifact_sha256={},
            registry_sha256=novelty.canonical_hash(registry),
            authentication_mode="injected_synthetic",
        ),
        gross9_artifact=gross9,
        attempt_claim={"mode": "synthetic_only"},
        protocol_paths=tuple(builder.PROTOCOL_PATHS),
        production=False,
    )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("utf-8").strip()


def _production_support_generation(
    *,
    source_csv_sha256: str,
    source_csv_bytes: int,
) -> tuple[dict[str, Any], bytes, bytes]:
    audit = {
        "artifact_eligible": True,
        "source_csv_path": source_support.DEFAULT_SOURCE_CSV.as_posix(),
        "source_csv_sha256": source_csv_sha256,
        "source_csv_bytes": source_csv_bytes,
        "source_manifest_path": source_support.DEFAULT_SOURCE_MANIFEST.as_posix(),
        "source_manifest_sha256": "b" * 64,
        "source_manifest_hash": "c" * 64,
        "source_integrity": dict(builder.ZERO_SOURCE_INTEGRITY),
    }
    report, primary, controls = source_support._build_support_from_frame(
        _source_frame(),
        registration=_minimal_source_registration(),
        source_audit=audit,
        artifact_eligible=True,
    )
    report["registration"]["manifest_hash"] = novelty.PREREGISTRATION_MANIFEST_HASH
    return report, primary, controls


def _boundary_evidence() -> dict[str, Any]:
    return {
        "outside_before_count": 0,
        "outside_after_maximum_admissible_count": 0,
        "header_count": 2 * len(builder.FROZEN_BOUNDARIES),
        "canonical_header_set_sha256": builder.BOUNDARY_HEADER_SET_SHA256,
        "frozen_header_set_exact": True,
        "boundaries": [
            {
                "utc": boundary["utc"],
                "previous_block": boundary["number"] - 1,
                "first_block_at_or_after": boundary["number"],
                "parent_relation_exact": True,
                "timestamp_relation_exact": True,
                "frozen_hash_exact": True,
            }
            for boundary in builder.FROZEN_BOUNDARIES
        ],
    }


def _metadata_manifest(
    *,
    claim_commit: str,
    protocol_parent: str,
    claim_raw: bytes,
    event_count: int,
    source_csv_sha256: str,
) -> dict[str, Any]:
    category_counts = {name: 0 for name in builder.CATEGORIES}
    category_counts[builder.CATEGORY_SEMANTIC] = event_count
    category_counts[builder.CATEGORY_MINT] = (event_count + 1) // 2
    category_counts[builder.CATEGORY_BURN] = event_count // 2
    core: dict[str, Any] = {
        "protocol_version": builder.PROTOCOL_VERSION,
        "source_only": True,
        "protocol_parent_commit": protocol_parent,
        "replay_claim_commit": claim_commit,
        "replay_claim_sha256": hashlib.sha256(claim_raw).hexdigest(),
        "generation_commit": dict(builder.PRODUCTION_GENERATION_COMMIT),
        "chain": {
            "name": "TRON mainnet",
            "chain_id": builder.CHAIN_ID_HEX,
            "usdt_contract_base58": builder.USDT_CONTRACT_BASE58,
            "usdt_contract_evm": builder.USDT_CONTRACT,
        },
        "source_range": builder._source_range_manifest(builder.frozen_chunks()),
        "transports": [dict(item) for item in builder.SANITIZED_TRANSPORTS],
        "source_replay_schedule": {
            "inter_batch_throttle_seconds": builder.PRODUCTION_THROTTLE_SECONDS,
            "maximum_batch_by_role": dict(builder.TRANSPORT_MAX_BATCH),
            "rpc_methods": sorted(builder.RPC_METHODS),
        },
        "transport_exact_set_equal": True,
        "category_counts": category_counts,
        "category_canonical_sha256": {
            name: f"{index + 1:064x}"
            for index, name in enumerate(builder.CATEGORIES)
        },
        "global_log_count": sum(category_counts.values()),
        "global_canonical_sha256": "4" * 64,
        "event_counts": {
            "Issue": (event_count + 1) // 2,
            "Redeem": event_count // 2,
            "DestroyedBlackFunds": 0,
        },
        "event_count": event_count,
        "event_canonical_sha256": "5" * 64,
        "year_counts": {"2023": 3, "2024": 5, "2025": 4, "2026": 2},
        "source_csv_sha256": source_csv_sha256,
        "receipt_count": event_count,
        "receipt_canonical_sha256": "6" * 64,
        "header_count": event_count * 2,
        "header_canonical_sha256": "7" * 64,
        "common_finalized_head": {
            "number": builder.LAST_CONFIRMATION_BLOCK,
            "hash": "0x" + "8" * 64,
            "timestamp_utc": builder.END_BOUNDARY_UTC,
            "covers_last_confirmation": True,
        },
        "boundary_evidence": _boundary_evidence(),
        "protocol_guards": {
            "retry_backoff_fallback_resume": False,
            "response_dependent_sleep": False,
            "deprecate_terminal": True,
            "market_policy_performance_opened": False,
        },
        "outcome_access": {
            "btc_market_rows_opened": 0,
            "funding_rows_opened": 0,
            "returns_opened": 0,
            "pnl_opened": 0,
            "cagr_opened": 0,
            "strict_mdd_opened": 0,
            "outcomes_opened": 0,
        },
        "source_integrity": dict(builder.ZERO_SOURCE_INTEGRITY),
    }
    return {**core, "manifest_hash": novelty.canonical_hash(core)}


def _production_git_fixture(
    tmp_path: Path,
    *,
    forged_report_source_sha256: bool = False,
    forged_source_blob_binding: bool = False,
    forged_source_byte_count: bool = False,
) -> tuple[Path, tuple[Path, ...], dict[str, Any], bytes]:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "novelty@example.invalid")
    _git(repository, "config", "user.name", "Novelty Test")
    protocol_paths = (
        builder.BUILDER_PATH,
        builder.TEST_PATH,
        builder.SOURCE_SUPPORT_PATH,
        builder.SOURCE_SUPPORT_TEST_PATH,
        builder.NOVELTY_PATH,
        builder.NOVELTY_TEST_PATH,
        builder.ECONOMICS_PATH,
        builder.ECONOMICS_TEST_PATH,
    )
    for index, relative in enumerate(protocol_paths):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"sealed = {index}\n", encoding="utf-8")
    _git(repository, "add", "--", *(path.as_posix() for path in protocol_paths))
    _git(repository, "commit", "-q", "-m", "sealed protocol")
    protocol_parent = _git(repository, "rev-parse", "HEAD")
    seal = builder.current_protocol_seal(
        repository_root=repository,
        protocol_paths=protocol_paths,
    )
    claim = builder._claim_payload(seal, builder.SANITIZED_TRANSPORTS)
    claim_raw = builder._canonical_json_bytes(claim, trailing_lf=True)
    claim_path = repository / novelty.DEFAULT_REPLAY_CLAIM_PATH
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_bytes(claim_raw)
    _git(repository, "add", "--", novelty.DEFAULT_REPLAY_CLAIM_PATH.as_posix())
    _git(repository, "commit", "-q", "-m", "claim only")
    claim_commit = _git(repository, "rev-parse", "HEAD")

    actual_source_sha256 = hashlib.sha256(OPAQUE_SOURCE_BYTES).hexdigest()
    declared_source_sha256 = (
        "a" * 64 if forged_source_blob_binding else actual_source_sha256
    )
    declared_source_bytes = len(OPAQUE_SOURCE_BYTES) + (
        1 if forged_source_byte_count else 0
    )
    report, primary, controls = _production_support_generation(
        source_csv_sha256=declared_source_sha256,
        source_csv_bytes=declared_source_bytes,
    )
    manifest = _metadata_manifest(
        claim_commit=claim_commit,
        protocol_parent=protocol_parent,
        claim_raw=claim_raw,
        event_count=report["source_contract"]["rows"],
        source_csv_sha256=declared_source_sha256,
    )
    manifest_raw = builder.serialize_manifest(manifest)
    report["source_contract"]["source_manifest_sha256"] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    report["source_contract"]["source_manifest_hash"] = manifest["manifest_hash"]
    report["source_contract"]["source_csv_sha256"] = (
        "f" * 64 if forged_report_source_sha256 else manifest["source_csv_sha256"]
    )
    report["source_contract"]["source_csv_bytes"] = declared_source_bytes
    report_core = {
        key: value for key, value in report.items() if key != "manifest_hash"
    }
    report["manifest_hash"] = novelty.canonical_hash(report_core)
    report_raw = source_support._json_bytes(report)

    source_csv = repository / novelty.DEFAULT_SOURCE_CSV_PATH
    source_manifest = repository / novelty.DEFAULT_SOURCE_MANIFEST_PATH
    source_csv.parent.mkdir(parents=True, exist_ok=True)
    source_csv.write_bytes(OPAQUE_SOURCE_BYTES)
    source_manifest.write_bytes(manifest_raw)
    _git(
        repository,
        "add",
        "--",
        novelty.DEFAULT_SOURCE_CSV_PATH.as_posix(),
        novelty.DEFAULT_SOURCE_MANIFEST_PATH.as_posix(),
    )
    _git(repository, "commit", "-q", "-m", "source artifacts")

    for relative, raw in (
        (novelty.DEFAULT_PRIMARY_CLOCK_PATH, primary),
        (novelty.DEFAULT_CONTROL_CLOCK_PATH, controls),
        (novelty.DEFAULT_SOURCE_SUPPORT_PATH, report_raw),
    ):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    _git(
        repository,
        "add",
        "--",
        novelty.DEFAULT_PRIMARY_CLOCK_PATH.as_posix(),
        novelty.DEFAULT_CONTROL_CLOCK_PATH.as_posix(),
        novelty.DEFAULT_SOURCE_SUPPORT_PATH.as_posix(),
    )
    _git(repository, "commit", "-q", "-m", "support artifacts")
    return repository, protocol_paths, report, report_raw


def test_preregistration_hashes_registry_and_gross9_authorities_are_exact() -> None:
    registration = _registration()
    novelty._validate_registration_authorities(registration)
    registry = novelty.frozen_registry(registration)
    assert len(registry) == 18
    assert novelty.canonical_hash(registry) == novelty.COMPARATOR_REGISTRY_SHA256
    assert (
        novelty.canonical_hash(registration["gross9"]["authority"])
        == novelty.GROSS9_AUTHORITY_SHA256
    )


def test_source_support_schema_hash_provenance_and_candidate_clock(
    tmp_path: Path,
) -> None:
    support, candidate, primary = _support_fixture(tmp_path)
    assert support.payload["support_passed"] is True
    assert len(candidate) == 14
    assert support.payload["clock_artifacts"]["primary_sha256"] == hashlib.sha256(
        primary
    ).hexdigest()
    assert all(row.exit - row.entry == 168 * 3600 for row in candidate)
    assert all(
        candidate[index].entry >= candidate[index - 1].exit
        for index in range(1, len(candidate))
    )


@pytest.mark.parametrize(
    "forgery",
    ("support_passed", "decision", "registration", "clock_hash", "manifest"),
)
def test_forged_source_support_fails_closed(
    tmp_path: Path, forgery: str
) -> None:
    support, _, primary = _support_fixture(tmp_path)
    payload = novelty._thaw_json(support.payload)
    if forgery == "support_passed":
        payload["support_passed"] = False
    elif forgery == "decision":
        payload["decision"] = "SOURCE_SUPPORT_PASS "
    elif forgery == "registration":
        payload["registration"]["manifest_hash"] = "0" * 64
    elif forgery == "clock_hash":
        payload["clock_artifacts"]["primary_sha256"] = "0" * 64
    else:
        payload["manifest_hash"] = "0" * 64
    if forgery != "manifest":
        core = {
            key: value for key, value in payload.items() if key != "manifest_hash"
        }
        payload["manifest_hash"] = novelty.canonical_hash(core)
    raw = source_support._json_bytes(payload)
    with pytest.raises(novelty.NoveltyTerminalError):
        forged_support = novelty.parse_passed_source_support_bytes(
            raw, path=tmp_path / "forged.json", production=False
        )
        if forgery == "clock_hash":
            primary_path = tmp_path / "forged-primary.csv.gz"
            primary_path.write_bytes(primary)
            novelty.load_candidate_clock_csv(primary_path, forged_support)


def test_exact_control_overlap_and_prior_novelty_calculations() -> None:
    base = novelty._parse_timestamp("2023-06-01T00:00:00Z")
    candidate = tuple(
        novelty.SignedInterval(base + index * 600, base + index * 600 + 300, 1)
        for index in range(10)
    )
    comparator = novelty.ComparatorClock(
        "synthetic",
        "directional_interval",
        tuple(row.entry for row in candidate),
        candidate,
        "synthetic",
    )
    result = novelty.evaluate_prior_comparator(
        candidate,
        comparator,
        ("2023-06-01T00:00:00Z", "2023-06-03T00:00:00Z"),
    )
    assert result["gating"] is True
    assert result["metrics"]["exact_entry_jaccard"] == {
        "numerator": 1,
        "denominator": 1,
    }
    assert result["metrics"]["candidate_24h_containment"] == {
        "numerator": 1,
        "denominator": 1,
    }
    assert result["passed"] is False


def test_report_schema_terminal_semantics_and_outcome_blind_audit(
    tmp_path: Path,
) -> None:
    report = _complete_report(tmp_path)
    assert set(report) == novelty.REPORT_KEYS
    assert report["status"] == "novelty_passed"
    assert report["terminal"] is False
    assert report["decision"] == "NOVELTY_PASS_OPEN_STRICT_ECONOMICS"
    assert all(
        report["evidence_boundary"][key] is False
        for key in (
            "candidate_market_rows_opened",
            "candidate_funding_rows_opened",
            "candidate_outcome_rows_opened",
            "candidate_returns_or_pnl_computed",
            "portfolio_return_or_pnl_metrics_computed",
        )
    )
    raw = novelty.canonical_report_bytes(report)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert json.loads(raw)["manifest_hash"] == report["manifest_hash"]
    forged = {**report, "unknown": False}
    forged_core = {
        key: value for key, value in forged.items() if key != "manifest_hash"
    }
    forged["manifest_hash"] = novelty.canonical_hash(forged_core)
    with pytest.raises(novelty.NoveltyTerminalError, match="exact schema"):
        novelty.canonical_report_bytes(forged)

    source = Path(novelty.__file__).read_text(encoding="utf-8")
    imported = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not imported & {
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "evaluate_tron_usdt_supply_impulse_economics",
    }


def test_failed_support_prevents_comparator_access(
    tmp_path: Path,
) -> None:
    support, candidate, _ = _support_fixture(tmp_path)
    registration = _registration()
    gross9 = _gross9_fixture(tmp_path, registration, support)
    forged = novelty._thaw_json(support.payload)
    forged["support_passed"] = False
    core = {key: value for key, value in forged.items() if key != "manifest_hash"}
    forged["manifest_hash"] = novelty.canonical_hash(core)
    raw = source_support._json_bytes(forged)
    forged_support = novelty.VerifiedSourceSupport(
        path=tmp_path / "bad.json",
        raw_bytes=raw,
        sha256=novelty.sha256_bytes(raw),
        manifest_hash=forged["manifest_hash"],
        payload=forged,
    )
    called = False

    def loader(
        registry: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, novelty.ComparatorClock]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(novelty.NoveltyTerminalError):
        novelty.build_report_after_source_support(
            source_support=forged_support,
            candidate=candidate,
            gross9_artifact=gross9,
            registration=registration,
            comparator_loader=loader,
        )
    assert called is False


def test_atomic_write_once_publication(tmp_path: Path) -> None:
    payload = _complete_report(tmp_path)
    output = tmp_path / "novelty.json"
    assert novelty.write_once_novelty_json_for_test(payload, output) == "created"
    assert output.stat().st_mode & 0o222 == 0
    assert (
        novelty.write_once_novelty_json_for_test(payload, output)
        == "verified_existing"
    )
    changed = novelty._thaw_json(payload)
    changed["source_support"]["sha256"] = "f" * 64
    changed_core = {
        key: value for key, value in changed.items() if key != "manifest_hash"
    }
    changed["manifest_hash"] = novelty.canonical_hash(changed_core)
    with pytest.raises(novelty.NoveltyTerminalError, match="output drift"):
        novelty.write_once_novelty_json_for_test(changed, output)
    assert not list(tmp_path.glob("*.staged"))


def test_production_publication_requires_authenticated_input_context(
    tmp_path: Path,
) -> None:
    report = _complete_report(tmp_path)
    with pytest.raises(
        novelty.NoveltyTerminalError,
        match="requires authenticated inputs",
    ):
        novelty.validate_report_payload(report, production=True)


def test_atomic_publication_never_links_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _complete_report(tmp_path)
    expected = novelty.canonical_report_bytes(payload)
    output = tmp_path / "interrupted.json"
    staged_paths: list[Path] = []

    def interrupted_link(
        source: str | Path,
        target: str | Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        staged = Path(source)
        assert follow_symlinks is False
        assert Path(target) == output
        assert staged.read_bytes() == expected
        assert staged.stat().st_mode & 0o777 == 0o444
        assert not output.exists()
        staged_paths.append(staged)
        raise OSError("synthetic interruption")

    monkeypatch.setattr(novelty.os, "link", interrupted_link)
    with pytest.raises(OSError, match="synthetic interruption"):
        novelty.write_once_novelty_json_for_test(payload, output)
    assert len(staged_paths) == 1
    assert not staged_paths[0].exists()
    assert not output.exists()


def test_unbound_production_artifacts_are_rejected_without_opening_comparators(
    tmp_path: Path,
) -> None:
    support, _, _ = _support_fixture(tmp_path)
    with pytest.raises(
        novelty.NoveltyTerminalError,
        match="canonical relative path|committed and clean",
    ):
        novelty.parse_passed_source_support_bytes(
            support.raw_bytes,
            path=tmp_path / "support.json",
            production=True,
        )


def test_production_source_provenance_binds_opaque_blob_and_sealed_head(
    tmp_path: Path,
) -> None:
    repository, protocol_paths, report, report_raw = _production_git_fixture(
        tmp_path
    )
    verified = novelty.parse_passed_source_support_bytes(
        report_raw,
        path=novelty.DEFAULT_SOURCE_SUPPORT_PATH.as_posix(),
        production=True,
        repository_root=repository,
        protocol_paths=protocol_paths,
    )
    assert verified.production_authenticated is True
    assert verified.provenance is not None
    provenance = verified.provenance
    assert provenance["protocol_seal_hash"]
    assert provenance["replay_claim_commit"]
    assert provenance["source_artifact_add_commit"]
    assert provenance["support_artifact_add_commit"]
    assert provenance["source_csv_sha256"] == hashlib.sha256(
        OPAQUE_SOURCE_BYTES
    ).hexdigest()
    assert provenance["source_csv_bytes"] == len(OPAQUE_SOURCE_BYTES)

    (repository / protocol_paths[0]).write_text(
        "sealed = 'forged'\n", encoding="utf-8"
    )
    with pytest.raises(
        novelty.NoveltyTerminalError, match="protocol seal|HEAD-clean"
    ):
        novelty.authenticate_production_source_support(
            report,
            report_raw,
            repository_root=repository,
            protocol_paths=protocol_paths,
        )


def test_recomputed_report_with_forged_source_csv_hash_is_rejected(
    tmp_path: Path,
) -> None:
    repository, protocol_paths, report, report_raw = _production_git_fixture(
        tmp_path,
        forged_report_source_sha256=True,
    )
    with pytest.raises(
        novelty.NoveltyTerminalError, match="direct source binding"
    ):
        novelty.authenticate_production_source_support(
            report,
            report_raw,
            repository_root=repository,
            protocol_paths=protocol_paths,
        )


@pytest.mark.parametrize(
    "forgery",
    ("declared_aaaa_hash", "wrong_byte_count"),
)
def test_committed_opaque_source_blob_must_match_declared_hash_and_size(
    tmp_path: Path,
    forgery: str,
) -> None:
    repository, protocol_paths, report, report_raw = _production_git_fixture(
        tmp_path,
        forged_source_blob_binding=forgery == "declared_aaaa_hash",
        forged_source_byte_count=forgery == "wrong_byte_count",
    )
    with pytest.raises(
        novelty.NoveltyTerminalError,
        match="opaque source CSV blob binding",
    ):
        novelty.authenticate_production_source_support(
            report,
            report_raw,
            repository_root=repository,
            protocol_paths=protocol_paths,
        )


def test_authoritative_gross9_loader_is_reused_without_local_pass_flags(
    tmp_path: Path,
) -> None:
    support, _, _ = _support_fixture(tmp_path)
    support = replace(support, production_authenticated=True)
    registration = _registration()
    synthetic = _gross9_fixture(tmp_path, registration, support)
    authoritative = novelty.esdi_novelty.VerifiedGross9Clocks(
        path=novelty.DEFAULT_GROSS9_CLOCKS_PATH,
        raw_bytes=synthetic.raw_bytes,
        sha256=synthetic.sha256,
        manifest_hash=synthetic.manifest_hash,
        authority_hash=synthetic.authority_hash,
        clocks=synthetic.clocks,
        payload=synthetic.payload,
    )
    canonical_gross9 = tmp_path / novelty.DEFAULT_GROSS9_CLOCKS_PATH
    canonical_gross9.parent.mkdir(parents=True, exist_ok=True)
    canonical_gross9.write_bytes(synthetic.raw_bytes)
    events: list[str] = []

    def authenticate(
        raw: bytes,
    ) -> novelty.esdi_novelty.VerifiedGross9Clocks:
        assert raw == synthetic.raw_bytes
        events.append("authoritative_esdi_authenticator")
        return authoritative

    observed = novelty.load_gross9_clock_artifact(
        registration=registration,
        source_support=support,
        path=novelty.DEFAULT_GROSS9_CLOCKS_PATH.as_posix(),
        production=True,
        authoritative_authenticator=authenticate,
        repository_root=tmp_path,
    )
    assert events == ["authoritative_esdi_authenticator"]
    assert observed.authentication_mode == "authoritative_esdi_production"
    assert observed.authority_hash == novelty.GROSS9_AUTHORITY_SHA256


@pytest.mark.parametrize(
    "forgery",
    (
        "empty_prior",
        "missing_gross9",
        "failed_checks",
        "comparator_id",
        "comparator_count",
        "metric_schema",
    ),
)
def test_nested_novelty_report_forgeries_fail_closed(
    tmp_path: Path, forgery: str
) -> None:
    report = novelty._thaw_json(_complete_report(tmp_path))
    if forgery == "empty_prior":
        report["novelty"]["prior_source_comparators"] = []
    elif forgery == "missing_gross9":
        report["novelty"]["gross9_sleeves"].pop()
    elif forgery == "failed_checks":
        report["novelty"]["failed_checks"] = ["forged"]
    elif forgery == "comparator_id":
        report["novelty"]["prior_source_comparators"][0][
            "comparator_id"
        ] = "forged"
    elif forgery == "comparator_count":
        report["registry"]["comparator_groups"] += 1
    else:
        del report["novelty"]["gross9_sleeves"][0]["metrics"][
            "occupied_bar_jaccard"
        ]
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    report["manifest_hash"] = novelty.canonical_hash(core)
    with pytest.raises(novelty.NoveltyTerminalError):
        novelty.canonical_report_bytes(report)


@pytest.mark.parametrize("forgery", ("prior_metric", "gross9_metric"))
def test_self_consistent_numeric_report_forgery_fails_authenticated_replay(
    tmp_path: Path,
    forgery: str,
) -> None:
    inputs = _complete_authenticated_inputs(tmp_path)
    report = novelty._thaw_json(
        novelty.build_report_from_authenticated_inputs(inputs)
    )
    assert report["novelty"]["passed"] is True
    if forgery == "prior_metric":
        item = report["novelty"]["prior_source_comparators"][0]
        assert item["gating"] is False
        item["metrics"]["exact_entry_jaccard"] = {
            "numerator": 1,
            "denominator": 1,
        }
        item["checks"]["exact_entry_jaccard"] = False
        item["would_pass_if_gating"] = all(item["checks"].values())
        item["passed"] = True
    else:
        item = report["novelty"]["gross9_sleeves"][0]
        item["metrics"]["exact_entry_jaccard"] = {
            "numerator": 1,
            "denominator": 1,
        }
        item["checks"]["exact_entry_jaccard"] = False
        item["passed"] = False
        sleeve = item["sleeve"]
        report["novelty"]["passed"] = False
        report["novelty"]["terminal"] = True
        report["novelty"]["failed_checks"] = [
            f"gross9:{sleeve}:exact_entry_jaccard"
        ]
        report["status"] = "retired_after_novelty"
        report["terminal"] = True
        report["decision"] = "RETIRE_TUSI_168_UNCHANGED_AFTER_NOVELTY"
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    report["manifest_hash"] = novelty.canonical_hash(core)

    assert novelty.canonical_report_bytes(report).endswith(b"\n")
    with pytest.raises(
        novelty.NoveltyTerminalError,
        match="authenticated full-report reproduction drift",
    ):
        novelty.require_exact_authenticated_reproduction(
            report,
            inputs,
            production=False,
        )


def test_below_minimum_nine_is_reported_not_gated(tmp_path: Path) -> None:
    base = novelty._parse_timestamp("2023-06-01T00:00:00Z")
    candidate = tuple(
        novelty.SignedInterval(
            base + index * 600,
            base + index * 600 + 300,
            1,
        )
        for index in range(10)
    )
    comparator = novelty.ComparatorClock(
        "below-minimum",
        "timestamp_only",
        tuple(base + index * 600 for index in range(9)),
        None,
        "synthetic",
    )
    result = novelty.evaluate_prior_comparator(
        candidate,
        comparator,
        ("2023-06-01T00:00:00Z", "2023-06-03T00:00:00Z"),
    )
    assert result["comparator_entries"] == 9
    assert result["gating"] is False
    assert result["passed"] is True
    assert result["would_pass_if_gating"] is False

    report = novelty._thaw_json(_complete_report(tmp_path))
    report["novelty"]["prior_source_comparators"][0][
        "comparator_entries"
    ] = 9
    report_core = {
        key: value for key, value in report.items() if key != "manifest_hash"
    }
    report["manifest_hash"] = novelty.canonical_hash(report_core)
    assert novelty.canonical_report_bytes(report).endswith(b"\n")


def test_exact_threshold_equality_passes_and_one_rational_epsilon_fails() -> None:
    assert novelty.inclusive_fraction_gate(
        novelty.Fraction(1, 5), 1, 5
    )
    assert not novelty.inclusive_fraction_gate(
        novelty.Fraction(2_000_000_001, 10_000_000_000),
        1,
        5,
    )
    assert novelty.inclusive_fraction_gate(
        novelty.Fraction(49, 400), 49, 400
    )
    assert not novelty.inclusive_fraction_gate(
        novelty.Fraction(490_000_001, 4_000_000_000),
        49,
        400,
    )


@pytest.mark.parametrize(
    "alias",
    (
        "../results/tron_usdt_supply_impulse_source_support_2026-07-30.json",
        "./results/tron_usdt_supply_impulse_source_support_2026-07-30.json",
        "/tmp/tron_usdt_supply_impulse_source_support_2026-07-30.json",
    ),
)
def test_production_path_aliases_fail_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    opened = False

    def forbidden_open(*args: Any, **kwargs: Any) -> int:
        nonlocal opened
        opened = True
        raise AssertionError("path alias reached filesystem open")

    monkeypatch.setattr(novelty.os, "open", forbidden_open)
    with pytest.raises(novelty.NoveltyTerminalError, match="canonical relative"):
        novelty._read_canonical_regular(
            alias,
            novelty.DEFAULT_SOURCE_SUPPORT_PATH,
            "source-support report",
            repository_root=tmp_path,
        )
    assert opened is False


def test_production_path_rejects_symlink_parent(
    tmp_path: Path,
) -> None:
    real_results = tmp_path / "real-results"
    real_results.mkdir()
    (real_results / novelty.DEFAULT_SOURCE_SUPPORT_PATH.name).write_bytes(b"{}")
    (tmp_path / "results").symlink_to(real_results, target_is_directory=True)
    with pytest.raises(novelty.NoveltyTerminalError, match="missing or unsafe"):
        novelty._read_canonical_regular(
            novelty.DEFAULT_SOURCE_SUPPORT_PATH.as_posix(),
            novelty.DEFAULT_SOURCE_SUPPORT_PATH,
            "source-support report",
            repository_root=tmp_path,
        )


@pytest.mark.parametrize("link_kind", ("file", "parent"))
def test_comparator_read_rejects_file_and_parent_symlinks(
    tmp_path: Path,
    link_kind: str,
) -> None:
    relative = Path("comparators/synthetic.csv.gz")
    canonical = tmp_path / relative
    real_parent = tmp_path / "real-comparators"
    real_parent.mkdir()
    (real_parent / relative.name).write_bytes(b"opaque comparator")
    if link_kind == "file":
        canonical.parent.mkdir(parents=True)
        canonical.symlink_to(real_parent / relative.name)
    else:
        canonical.parent.parent.mkdir(parents=True, exist_ok=True)
        canonical.parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(novelty.NoveltyTerminalError, match="missing or unsafe"):
        novelty._read_comparator_artifact_bytes(
            "synthetic",
            {"path": relative.as_posix()},
            repository_root=tmp_path,
        )


@pytest.mark.parametrize("link_kind", ("file", "parent"))
def test_gross9_read_rejects_file_and_parent_symlinks_before_authenticator(
    tmp_path: Path,
    link_kind: str,
) -> None:
    support, _, _ = _support_fixture(tmp_path)
    support = replace(support, production_authenticated=True)
    registration = _registration()
    canonical = tmp_path / novelty.DEFAULT_GROSS9_CLOCKS_PATH
    real_parent = tmp_path / "real-gross9"
    real_parent.mkdir()
    (real_parent / canonical.name).write_bytes(b"opaque Gross9")
    if link_kind == "file":
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.symlink_to(real_parent / canonical.name)
    else:
        canonical.parent.parent.mkdir(parents=True, exist_ok=True)
        canonical.parent.symlink_to(real_parent, target_is_directory=True)
    called = False

    def forbidden_authenticator(
        raw: bytes,
    ) -> novelty.esdi_novelty.VerifiedGross9Clocks:
        nonlocal called
        called = True
        raise AssertionError(raw)

    with pytest.raises(novelty.NoveltyTerminalError, match="missing or unsafe"):
        novelty.load_gross9_clock_artifact(
            registration=registration,
            source_support=support,
            path=novelty.DEFAULT_GROSS9_CLOCKS_PATH.as_posix(),
            production=True,
            authoritative_authenticator=forbidden_authenticator,
            repository_root=tmp_path,
        )
    assert called is False


def test_attempt_claim_is_atomic_short_write_and_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "results").mkdir()
    support, _, _ = _support_fixture(tmp_path)
    payload = novelty._claim_payload(
        support,
        support.payload["clock_artifacts"]["primary_sha256"],
    )
    real_write = os.write
    writes = 0

    def interrupted_write(descriptor: int, data: Any) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            return real_write(descriptor, bytes(data[:7]))
        raise OSError("synthetic crash")

    monkeypatch.setattr(novelty.os, "write", interrupted_write)
    with pytest.raises(OSError, match="synthetic crash"):
        novelty._create_attempt_claim(payload, repository_root=tmp_path)
    claim = tmp_path / novelty.DEFAULT_ATTEMPT_CLAIM_PATH
    assert not claim.exists()
    assert not list(claim.parent.glob("*.staged"))

    monkeypatch.setattr(novelty.os, "write", real_write)
    binding = novelty._create_attempt_claim(
        payload, repository_root=tmp_path
    )
    assert novelty._load_attempt_claim(
        payload, repository_root=tmp_path
    ) == binding
    with pytest.raises(novelty.NoveltyTerminalError, match="already claimed"):
        novelty._create_attempt_claim(payload, repository_root=tmp_path)


def test_claim_is_durable_before_gross9_and_comparator_access(
    tmp_path: Path,
) -> None:
    (tmp_path / "results").mkdir()
    support, candidate, _ = _support_fixture(tmp_path)
    registration = _registration()
    gross9 = _gross9_fixture(tmp_path, registration, support)
    events: list[str] = []

    def create(payload: Mapping[str, Any]) -> Mapping[str, str]:
        binding = novelty._create_attempt_claim(
            payload, repository_root=tmp_path
        )
        events.append("claim_durable")
        return binding

    def gross_loader() -> novelty.VerifiedGross9Clocks:
        assert (
            tmp_path / novelty.DEFAULT_ATTEMPT_CLAIM_PATH
        ).is_file()
        events.append("gross9_open")
        return gross9

    def comparators(
        registry: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, novelty.ComparatorClock]:
        assert (
            tmp_path / novelty.DEFAULT_ATTEMPT_CLAIM_PATH
        ).is_file()
        events.append("comparators_open")
        return _synthetic_comparators(registry)

    report = novelty.execute_claimed_novelty(
        registration=registration,
        source_support=support,
        candidate=candidate,
        gross9_loader=gross_loader,
        comparator_loader=comparators,
        claim_creator=create,
        production=False,
    )
    assert events == ["claim_durable", "gross9_open", "comparators_open"]
    novelty.authenticate_attempt_claim_for_report(
        report, repository_root=tmp_path
    )
    claim_path = tmp_path / novelty.DEFAULT_ATTEMPT_CLAIM_PATH
    claim_path.chmod(0o644)
    claim_path.write_bytes(b"{}\n")
    with pytest.raises(
        novelty.NoveltyTerminalError, match="attempt claim"
    ):
        novelty.authenticate_attempt_claim_for_report(
            report, repository_root=tmp_path
        )
