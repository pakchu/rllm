from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
from Crypto.Hash import keccak

from training import preregister_governance_intent_payload_relation as gipr


RESULT_PATH = Path(
    "results/governance_intent_payload_relation_preregistration_2026-07-25.json"
)
RESULT_SHA256 = (
    "319ac26108e936331f95d047a69f739ffecfd2bdc777573100a1ce83c771c197"
)


def _topic(signature: str) -> str:
    digest = keccak.new(digest_bits=256)
    digest.update(signature.encode("ascii"))
    return "0x" + digest.hexdigest()


def test_decision_document_is_hash_bound() -> None:
    assert gipr.sha256_file(gipr.DECISION_PATH) == gipr.DECISION_SHA256


def test_exact_governor_roster_is_frozen() -> None:
    assert [
        (row.protocol, row.generation, row.address, row.first_code_block)
        for row in gipr.GOVERNORS
    ] == [
        (
            "compound",
            "alpha",
            "0xc0da01a04c3f3e0be433606045bb7017a7323e38",
            9_601_459,
        ),
        (
            "compound",
            "bravo",
            "0xc0da02939e1441f497fd74f78ce7decb17b66529",
            12_006_099,
        ),
        (
            "uniswap",
            "alpha_v0",
            "0x5e4be8bc9637f0eaa1a755019e06a68ce081d58f",
            10_861_678,
        ),
        (
            "uniswap",
            "alpha_v2",
            "0xc4e172459f1e7939d522503b81afaac1014ce6f6",
            12_543_659,
        ),
        (
            "uniswap",
            "bravo",
            "0x408ed6354d4973f66138c91495f2f2fcbd8724c3",
            13_059_157,
        ),
    ]
    assert len({row.address for row in gipr.GOVERNORS}) == 5
    assert all(row.runtime_code_bytes > 0 for row in gipr.GOVERNORS)
    assert all(row.last_source_code_bytes > 0 for row in gipr.GOVERNORS)


def test_known_target_role_registry_is_source_blind_and_exact() -> None:
    assert [
        (row.protocol, row.role, row.address) for row in gipr.KNOWN_TARGET_ROLES
    ] == [
        (
            "compound",
            "TIMELOCK",
            "0x6d903f6003cca6255d85cca4d3b5e5146dc33925",
        ),
        (
            "compound",
            "GOVERNANCE_TOKEN",
            "0xc00e94cb662c3520282e6f5717214004a7f26888",
        ),
        (
            "compound",
            "RISK_CONTROLLER",
            "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b",
        ),
        (
            "uniswap",
            "TIMELOCK",
            "0x1a9c8182c09f50c8318d769245bea52c32be35bc",
        ),
        (
            "uniswap",
            "GOVERNANCE_TOKEN",
            "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
        ),
        (
            "uniswap",
            "V2_FACTORY",
            "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
        ),
        (
            "uniswap",
            "V3_FACTORY",
            "0x1f98431c8ad98523631ae4a59f267346ea31f984",
        ),
    ]


def test_event_topics_match_exact_keccak_signatures() -> None:
    assert {row.event for row in gipr.EVENT_SPECS} == {
        "proposal_created",
        "proposal_canceled",
        "proposal_queued",
        "proposal_executed",
    }
    assert all(row.topic == _topic(row.signature) for row in gipr.EVENT_SPECS)
    assert all(row.topic_count == 1 for row in gipr.EVENT_SPECS)


def test_source_envelope_and_year_boundaries_are_exact() -> None:
    assert gipr.SOURCE_START == "2020-01-01T00:00:00Z"
    assert gipr.SOURCE_END_EXCLUSIVE == "2024-01-01T00:00:00Z"
    assert gipr.START_BOUNDARY_BLOCK == 9_193_266
    assert gipr.END_BOUNDARY_BLOCK == 18_908_895
    assert gipr.LAST_SOURCE_BLOCK == 18_908_894
    assert [row.first_block for row in gipr.YEAR_BOUNDARIES] == [
        9_193_266,
        11_565_019,
        13_916_166,
        16_308_190,
        18_908_895,
    ]
    assert gipr.CONFIRMATION_BLOCKS == 64


def test_splits_are_contiguous_and_post_2023_is_sealed() -> None:
    payload = gipr.build_preregistration()
    splits = payload["split_contract"]["splits"]
    assert [row["name"] for row in splits] == ["train", "test", "eval"]
    assert splits[0]["start"] == gipr.SOURCE_START
    assert splits[0]["end_exclusive"] == splits[1]["start"]
    assert splits[1]["end_exclusive"] == splits[2]["start"]
    assert splits[2]["end_exclusive"] == gipr.SOURCE_END_EXCLUSIVE
    assert payload["split_contract"]["post_2023_source_policy"] == "SEALED"


def test_daily_schedule_and_staleness_are_fixed() -> None:
    payload = gipr.build_preregistration()
    availability = payload["availability_contract"]
    assert availability["daily_decision_frequency"] == "1D"
    assert availability["daily_decision_utc_time"] == "00:00:00"
    assert availability["history_days"] == 28
    assert availability["maximum_proposal_age_days"] == 90
    assert availability["silent_forward_fill_allowed"] is False
    assert availability["stale_state"] == "STALE_OR_NO_PROPOSAL"


def test_parser_bounds_and_dynamic_abi_guards_are_fixed() -> None:
    parser = gipr.build_preregistration()["parser_contract"]
    assert parser["proposal_identity_fields"] == [
        "governor_address",
        "proposal_id",
    ]
    assert parser["lifecycle_transition_matrix"] == {
        "proposal_created": ["proposal_queued", "proposal_canceled"],
        "proposal_queued": ["proposal_executed", "proposal_canceled"],
        "proposal_executed": [],
        "proposal_canceled": [],
    }
    assert parser["duplicate_lifecycle_events_allowed"] is False
    assert parser["execute_without_queue_allowed"] is False
    assert parser["array_lengths_must_match"] is True
    assert parser["minimum_actions_per_proposal"] == 1
    assert parser["maximum_actions_per_proposal"] == 10
    assert parser["dynamic_offset_alignment_required"] is True
    assert parser["dynamic_tail_nonoverlap_required"] is True
    assert parser["trailing_unparsed_bytes_allowed"] is False
    assert parser["strict_utf8_description_required"] is True
    assert parser["nul_in_description_allowed"] is False


def test_single_model_target_position_boundary_is_exact() -> None:
    representation = gipr.build_preregistration()["representation_contract"]
    assert representation["single_model_only"] is True
    assert representation["analyzer_trader_pair_allowed"] is False
    assert representation["model_actions"] == [
        "TARGET_LONG",
        "TARGET_FLAT",
        "TARGET_SHORT",
    ]
    assert representation["raw_numeric_model_inputs_allowed"] is False
    assert representation["raw_price_return_rank_pnl_model_inputs_allowed"] is False
    assert representation["model_controls_leverage_size_stop_cost_or_reward"] is False


def test_all_source_gates_and_controls_are_mandatory() -> None:
    contract = gipr.build_preregistration()["source_support_contract"]
    assert contract["gates_in_order"] == list(gipr.SOURCE_ONLY_GATES)
    assert contract["all_gates_required"] is True
    assert contract["relation_controls"] == list(gipr.RELATION_CONTROLS)
    assert contract["first_failure_action"].startswith("REJECT_GIPR_D1_UNCHANGED")
    assert contract["source_drop_or_repair_allowed"] is False
    assert contract["gate_relaxation_after_incidence_allowed"] is False


def test_preregistration_has_zero_forbidden_access() -> None:
    access = gipr.build_preregistration()["forbidden_access_contract"]
    assert access["network_calls_during_preregistration"] == 0
    assert access["governance_event_rows_read"] == 0
    assert access["governance_descriptions_read"] == 0
    assert access["governance_payloads_read"] == 0
    assert access["lifecycle_rows_read"] == 0
    assert access["event_incidence_opened"] is False
    assert access["btc_or_funding_outcomes_opened"] is False
    assert set(access["counters"]) == set(gipr.FORBIDDEN_COUNTERS)
    assert set(access["counters"].values()) == {0}


def test_manifest_hash_binds_exact_core() -> None:
    payload = gipr.build_preregistration()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == gipr.canonical_hash(core)
    assert payload == gipr.build_preregistration()


def test_committed_artifact_matches_exact_builder() -> None:
    raw = (gipr.REPO_ROOT / RESULT_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    assert json.loads(raw) == gipr.build_preregistration()


def test_write_is_canonical_and_idempotent(tmp_path: Path) -> None:
    destination = tmp_path / "gipr.json"
    first_path, first_payload = gipr.write_preregistration(destination)
    first_bytes = first_path.read_bytes()
    second_path, second_payload = gipr.write_preregistration(destination)
    assert first_path == second_path
    assert first_payload == second_payload
    assert first_bytes == second_path.read_bytes()
    assert first_bytes.endswith(b"\n")
    assert json.loads(first_bytes) == first_payload


def test_write_rejects_conflicting_existing_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "gipr.json"
    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="existing GIPR preregistration differs"):
        gipr.write_preregistration(destination)


def test_governor_roster_mutation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gipr, "GOVERNORS", gipr.GOVERNORS[:-1])
    with pytest.raises(RuntimeError, match="exactly five"):
        gipr.build_preregistration()


def test_event_topic_mutation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    mutated = list(gipr.EVENT_SPECS)
    mutated[0] = gipr.EventSpec(
        event=mutated[0].event,
        signature=mutated[0].signature,
        topic="0x" + "00" * 32,
        topic_count=mutated[0].topic_count,
        lifecycle_rank=mutated[0].lifecycle_rank,
    )
    monkeypatch.setattr(gipr, "EVENT_SPECS", tuple(mutated))
    with pytest.raises(RuntimeError, match="differs from exact ABI"):
        gipr.build_preregistration()


def test_decision_hash_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gipr, "DECISION_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="decision document hash changed"):
        gipr.build_preregistration()


def test_module_has_no_network_or_market_imports() -> None:
    source = (gipr.REPO_ROOT / gipr.SCRIPT_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(
        {
            "requests",
            "urllib",
            "httpx",
            "aiohttp",
            "pandas",
            "numpy",
            "torch",
            "transformers",
            "ccxt",
        }
    )
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called_names.isdisjoint({"urlopen", "get", "post", "request"})
    assert "btcusdt" not in source.lower()
    assert "cagr(" not in source.lower()


def test_selection_and_correction_commits_are_disclosed() -> None:
    candidate = gipr.build_preregistration()["candidate"]
    assert (
        candidate["selection_commit"]
        == "cdb8907002736d5c7703232faee3073dd96f2596"
    )
    assert (
        candidate["correction_commit"]
        == "0016296bad08b8101944a6b87fbad184e00d6a59"
    )


def test_script_and_decision_hashes_are_real_sha256() -> None:
    assert hashlib.sha256(
        (gipr.REPO_ROOT / gipr.SCRIPT_PATH).read_bytes()
    ).hexdigest()
    assert len(gipr.DECISION_SHA256) == 64
