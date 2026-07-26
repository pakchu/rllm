from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from training import (
    probe_protocol_specification_intent_maturity_d6_mechanism as d6,
)


PROBE_SHA256 = (
    "01b09218d71d83c6abc3c4225b708a1cae6fe9e426b9bbd98f4fe6e86579d60b"
)
RESULT_HASH = (
    "dda4b4786b34064a104178580f6cd33e56d5616c282515f6579105231b5dab38"
)


def probe_bytes() -> bytes:
    return d6.repository_path(d6.DEFAULT_OUTPUT).read_bytes()


def probe_payload() -> dict[str, Any]:
    return json.loads(probe_bytes())


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(nested_keys(nested))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for nested in value:
            keys.update(nested_keys(nested))
    return keys


def synthetic_authority(
    episode: Mapping[str, Any],
) -> tuple[dict[int, str], str]:
    receipt_hash = d6.canonical_hash(episode)
    receipts = {int(episode["proposal"]): receipt_hash}
    manifest_hash = d6.canonical_hash(
        d6.migration_receipt_manifest_d6(receipts)
    )
    return receipts, manifest_hash


def test_probe_is_frozen_canonical_and_replay_equal() -> None:
    raw = probe_bytes()
    payload = probe_payload()
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }

    assert d6.DEFAULT_OUTPUT.as_posix() == (
        "results/protocol_specification_intent_maturity_d6_mechanism_probe_"
        "2026-07-26.json"
    )
    assert hashlib.sha256(raw).hexdigest() == PROBE_SHA256
    assert raw == d6.canonical_json_bytes(payload)
    assert payload == d6.build_probe()
    assert payload["result_hash"] == RESULT_HASH
    assert payload["result_hash"] == d6.canonical_hash(core)
    assert payload["protocol_version"] == d6.PROTOCOL_VERSION
    assert payload["mechanism_version"] == d6.MECHANISM_VERSION
    assert payload["policy_id"] == (
        "PSIM-D6-SYNTHETIC-MECHANISM-PROBE"
    )


def test_probe_binds_complete_d5_census_authority() -> None:
    binding = probe_payload()["d5_census_binding"]
    loaded, receipt_map = d6._load_d5_census_binding()

    assert binding == loaded
    assert binding == {
        "commit": d6.D5_CENSUS_COMMIT,
        "document": {
            "path": d6.D5_CENSUS_DOCUMENT_PATH.as_posix(),
            "sha256": d6.D5_CENSUS_DOCUMENT_SHA256,
        },
        "episode_receipt_count": 365,
        "episode_receipt_manifest_hash": (
            d6.D5_EPISODE_RECEIPT_MANIFEST_HASH
        ),
        "episode_roster_hash": d6.D5_EPISODE_ROSTER_HASH,
        "path": d6.D5_CENSUS_PATH.as_posix(),
        "proposal_roster_hash": (
            d6.D5_MIGRATION_PROPOSAL_ROSTER_HASH
        ),
        "result_hash": d6.D5_CENSUS_RESULT_HASH,
        "script": {
            "path": d6.D5_CENSUS_SCRIPT_PATH.as_posix(),
            "sha256": d6.D5_CENSUS_SCRIPT_SHA256,
        },
        "sha256": d6.D5_CENSUS_SHA256,
        "test": {
            "path": d6.D5_CENSUS_TEST_PATH.as_posix(),
            "sha256": d6.D5_CENSUS_TEST_SHA256,
        },
        "text_bound_event_roster_hash": (
            d6.D5_TEXT_BOUND_EVENT_ROSTER_HASH
        ),
    }
    assert len(receipt_map) == 365
    assert d6.canonical_hash(
        d6.migration_receipt_manifest_d6(receipt_map)
    ) == d6.D5_EPISODE_RECEIPT_MANIFEST_HASH


def test_probe_is_synthetic_only_and_outcome_blind() -> None:
    payload = probe_payload()

    assert payload["synthetic_only"] is True
    assert payload["selection_scope"] == (
        "AUTHORIZE_D6_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
    )
    assert payload["access_boundary"] == {
        "d5_census_artifact_read": True,
        "d5_forensic_root_accessed": False,
        "d5_run_invoked": False,
        "external_network_accessed_by_probe": False,
        "historical_proposal_text_accessed": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "official_historical_proposal_source_accessed": False,
        "outcomes_accessed": False,
        "raw_official_text_published": False,
    }
    assert nested_keys(payload).isdisjoint(
        {
            "cagr",
            "future_return",
            "intent_text",
            "normalized_text_delta_chunk",
            "pnl",
            "raw_bytes",
            "raw_text",
            "reward",
            "strict_mdd",
            "trade",
        }
    )


def test_model_row_serialization_is_exact_and_ordered() -> None:
    rows = [
        {
            "direction": "REMOVE",
            "line": "old",
            "section": "ABSTRACT",
        },
        {
            "direction": "ADD",
            "line": "new",
            "section": "ABSTRACT",
        },
    ]

    assert d6.serialize_model_delta_rows_d6(rows) == (
        "ABSTRACT|REMOVE|old\nABSTRACT|ADD|new"
    )
    assert d6.serialize_model_delta_rows_d6(list(reversed(rows))) == (
        "ABSTRACT|ADD|new\nABSTRACT|REMOVE|old"
    )


@pytest.mark.parametrize(
    "row",
    [
        {
            "direction": "ADD",
            "line": "text",
            "section": "NOT_MODEL_VISIBLE",
        },
        {
            "direction": "CHANGE",
            "line": "text",
            "section": "ABSTRACT",
        },
        {
            "direction": "ADD",
            "line": "two\nlines",
            "section": "ABSTRACT",
        },
        {
            "direction": "ADD",
            "extra": "forbidden",
            "line": "text",
            "section": "ABSTRACT",
        },
    ],
)
def test_model_row_serialization_rejects_ambiguous_rows(
    row: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="model delta row"):
        d6.serialize_model_delta_rows_d6([row])


@pytest.mark.parametrize(
    ("text", "expected_sizes"),
    [
        ("", []),
        ("a" * 8_192, [8_192]),
        ("b" * 8_193, [8_192, 1]),
        ("c" * 8_191 + "한" + "d", [8_191, 4]),
        ("e" * 8_191 + "\n" + "f", [8_192, 1]),
        ("g" * 20_000, [8_192, 8_192, 3_616]),
        ("h" * 65_536, [8_192] * 8),
    ],
)
def test_utf8_splitter_is_lossless_and_greedy(
    text: str,
    expected_sizes: list[int],
) -> None:
    chunks = d6.split_utf8_model_text_d6(text)

    assert [len(chunk.encode("utf-8")) for chunk in chunks] == (
        expected_sizes
    )
    assert "".join(chunks) == text
    assert len(chunks) <= d6.MAX_MODEL_TEXT_CHUNKS_PER_EVENT


def test_utf8_splitter_rejects_ninth_chunk_and_surrogates() -> None:
    with pytest.raises(ValueError, match="more than eight chunks"):
        d6.split_utf8_model_text_d6("a" * 65_537)
    with pytest.raises(ValueError, match="not strict UTF-8"):
        d6.split_utf8_model_text_d6("\ud800")


def test_chunk_transport_separates_model_text_from_audit_hashes() -> None:
    text = "x" * 8_193
    transport = d6.build_model_text_transport_d6(text)
    payloads = transport["model_chunk_payloads"]
    receipt = transport["audit_receipt"]

    assert payloads == [
        {
            "chunk_count": 2,
            "chunk_index": 0,
            "normalized_text_delta_chunk": "x" * 8_192,
        },
        {
            "chunk_count": 2,
            "chunk_index": 1,
            "normalized_text_delta_chunk": "x",
        },
    ]
    assert receipt["chunk_count"] == 2
    assert receipt["full_text_utf8_bytes"] == 8_193
    assert receipt["full_text_sha256"] == d6.sha256_bytes(
        text.encode("utf-8")
    )
    assert receipt["reconstruction_matches"] is True
    assert "normalized_text_delta_chunk" not in nested_keys(receipt)
    assert all(set(row) == d6.MODEL_CHUNK_FIELDS for row in payloads)


@pytest.mark.parametrize(
    "mutation",
    [
        "delete",
        "duplicate",
        "swap",
        "bytes",
        "index",
        "count",
        "extra_field",
        "repartition",
        "different_full_text",
    ],
)
def test_chunk_transport_tamper_controls_fail_closed(
    mutation: str,
) -> None:
    full_text = "x" * 20_000
    payloads = list(d6.build_model_chunk_payloads_d6(full_text))
    candidate = copy.deepcopy(payloads)
    compared_text = full_text
    if mutation == "delete":
        candidate.pop()
    elif mutation == "duplicate":
        candidate.append(copy.deepcopy(candidate[-1]))
    elif mutation == "swap":
        candidate[0], candidate[1] = candidate[1], candidate[0]
    elif mutation == "bytes":
        candidate[0]["normalized_text_delta_chunk"] += "y"
    elif mutation == "index":
        candidate[0]["chunk_index"] = 1
    elif mutation == "count":
        candidate[0]["chunk_count"] += 1
    elif mutation == "extra_field":
        candidate[0]["event_id"] = "0" * 64
    elif mutation == "repartition":
        moved = candidate[1]["normalized_text_delta_chunk"][-1]
        candidate[1]["normalized_text_delta_chunk"] = (
            candidate[1]["normalized_text_delta_chunk"][:-1]
        )
        candidate[2]["normalized_text_delta_chunk"] = (
            moved + candidate[2]["normalized_text_delta_chunk"]
        )
    elif mutation == "different_full_text":
        compared_text += "y"

    with pytest.raises(ValueError):
        d6.validate_model_chunk_payloads_d6(
            compared_text,
            candidate,
        )


def test_exact_migration_episode_is_receipt_bound_and_quarantined() -> None:
    episode = d6._synthetic_episode()
    authority, manifest_hash = synthetic_authority(episode)

    decision = d6._authorize_migration_restoration_with_authority_d6(
        episode,
        authority,
        manifest_hash,
    )

    assert decision == {
        "audit": {
            "authority_receipt_hash": d6.canonical_hash(episode),
            "authority_receipt_manifest_hash": manifest_hash,
            "causal_episode_steps": 3,
            "protocol_version": (
                "psim_d6_exact_migration_restoration_receipt_v1"
            ),
            "quarantine_reason": (
                "EXACT_2023_ETHEREUM_ERC_MIGRATION_EPISODE_RESTORATION"
            ),
        },
        "model": {
            "administrative_quarantined": True,
            "model_visibility": "ADMINISTRATIVE_QUARANTINE",
            "normalized_text_delta_chunks": [],
        },
    }
    assert set(decision["model"]) == {
        "administrative_quarantined",
        "model_visibility",
        "normalized_text_delta_chunks",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "path",
        "target",
        "commit",
        "day",
        "class",
        "continuity",
        "order",
        "generic_reverse",
        "extra_field",
        "wrong_receipt",
    ],
)
def test_migration_episode_mutations_fail_closed(mutation: str) -> None:
    episode = d6._synthetic_episode()
    authority, manifest_hash = synthetic_authority(episode)
    candidate = copy.deepcopy(episode)
    if mutation == "path":
        candidate["path"] = "EIPS/eip-21.md"
    elif mutation == "target":
        candidate["upper_redirect"]["target_proposal"] = 21
    elif mutation == "commit":
        candidate["steps"][1]["commit_oid"] = "0" * 40
    elif mutation == "day":
        candidate["steps"][2]["effective_day"] = "2023-10-27"
    elif mutation == "class":
        candidate["steps"][2]["old_blob_class"] = "D4_VALID"
    elif mutation == "continuity":
        candidate["steps"][1]["old_blob_oid"] = "9" * 40
    elif mutation == "order":
        candidate["steps"][0], candidate["steps"][1] = (
            candidate["steps"][1],
            candidate["steps"][0],
        )
    elif mutation == "generic_reverse":
        candidate["steps"] = [candidate["steps"][2]]
    elif mutation == "extra_field":
        candidate["future_return"] = 1.0
    elif mutation == "wrong_receipt":
        authority[int(episode["proposal"])] = "0" * 64
        manifest_hash = d6.canonical_hash(
            d6.migration_receipt_manifest_d6(authority)
        )

    with pytest.raises(ValueError, match="PSIM-D6 migration"):
        d6._authorize_migration_restoration_with_authority_d6(
            candidate,
            authority,
            manifest_hash,
        )


def test_migration_receipt_roster_cannot_be_removed_or_expanded() -> None:
    episode = d6._synthetic_episode()
    authority, manifest_hash = synthetic_authority(episode)

    with pytest.raises(ValueError, match="receipt manifest differs"):
        d6._authorize_migration_restoration_with_authority_d6(
            episode,
            {},
            manifest_hash,
        )
    expanded = {
        **authority,
        21: "9" * 64,
    }
    with pytest.raises(ValueError, match="receipt manifest differs"):
        d6._authorize_migration_restoration_with_authority_d6(
            episode,
            expanded,
            manifest_hash,
        )


def test_public_migration_authorizer_uses_only_frozen_d5_authority() -> None:
    census = d6._read_canonical_json(d6.D5_CENSUS_PATH)
    episode = census["census"]["administrative_episode_census"][
        "representative"
    ]

    decision = d6.authorize_migration_restoration_d6(episode)

    assert decision["audit"]["authority_receipt_hash"] == (
        census["census"]["administrative_episode_census"][
            "per_proposal_receipt_hashes"
        ][0]["receipt_hash"]
    )
    assert decision["audit"]["authority_receipt_manifest_hash"] == (
        d6.D5_EPISODE_RECEIPT_MANIFEST_HASH
    )
    arbitrary = d6._synthetic_episode(999_999)
    with pytest.raises(
        ValueError,
        match="migration proposal is not authorized",
    ):
        d6.authorize_migration_restoration_d6(arbitrary)


def test_synthetic_battery_covers_all_frozen_controls() -> None:
    battery = probe_payload()["synthetic_battery"]

    assert battery["model_text_chunks"][
        "model_row_serialization_exact"
    ] is True
    assert battery["model_text_chunks"]["ninth_chunk_fails_closed"] is True
    assert battery["model_text_chunks"]["tamper_cases_rejected"] == 9
    assert battery["model_text_chunks"]["utf8_boundary_is_lossless"] is True
    assert set(battery["model_text_chunks"]["positive_cases"]) == {
        "empty",
        "exact_8192",
        "exact_65536",
        "historical_max_bytes",
        "lf_boundary",
        "over_8192",
        "single_oversized_row",
        "utf8_boundary",
    }
    assert battery["migration_restoration"][
        "exact_three_step_episode_authorized"
    ] is True
    assert battery["migration_restoration"][
        "generic_reverse_transition_authorized"
    ] is False
    assert battery["migration_restoration"][
        "negative_cases_rejected"
    ] == 12
    assert battery["migration_restoration"][
        "model_text_chunks_for_restoration"
    ] == 0


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/psim-d5-source/probe.json",
        str(d6.REPO_ROOT.resolve() / "results" / "absolute.json"),
        "docs/probe.json",
        "results/nested/probe.json",
        "results/../results/probe.json",
        "results/probe.txt",
    ],
)
def test_probe_output_boundary_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(RuntimeError, match="safe repo-local result"):
        d6._safe_output_path(path)


def test_probe_output_boundary_accepts_flat_relative_json() -> None:
    assert d6._safe_output_path(d6.DEFAULT_OUTPUT) == (
        d6.REPO_ROOT.resolve() / d6.DEFAULT_OUTPUT
    )


def test_probe_output_boundary_rejects_symlinked_results_root(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "results").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(d6, "REPO_ROOT", root)

    with pytest.raises(RuntimeError, match="safe repo-local result"):
        d6._safe_output_path("results/probe.json")


def test_probe_output_boundary_rejects_symlinked_result_file(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repo"
    results = root / "results"
    results.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (results / "probe.json").symlink_to(outside)
    monkeypatch.setattr(d6, "REPO_ROOT", root)

    with pytest.raises(RuntimeError, match="safe repo-local result"):
        d6._safe_output_path("results/probe.json")


def test_probe_imports_no_network_market_or_model_clients() -> None:
    source = d6.repository_path(
        "training/"
        "probe_protocol_specification_intent_maturity_d6_mechanism.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots.isdisjoint(
        {
            "aiohttp",
            "binance",
            "ccxt",
            "httpx",
            "models",
            "pandas",
            "requests",
            "sklearn",
            "torch",
            "transformers",
            "urllib",
            "yfinance",
        }
    )
