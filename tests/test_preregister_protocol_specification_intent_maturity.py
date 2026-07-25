from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

from training import preregister_protocol_specification_intent_maturity as psim


RESULT_PATH = Path(
    "results/protocol_specification_intent_maturity_preregistration_"
    "2026-07-25.json"
)
RESULT_SHA256 = (
    "bd4053574fe6285c34356baaa080e215f08bbf8142e9c0c968bffbdccb2dc736"
)


def test_decision_document_is_hash_bound() -> None:
    assert psim.sha256_file(psim.DECISION_PATH) == psim.DECISION_SHA256
    assert psim.DECISION_SHA256 == (
        "816fbb19c4ff9a841f75f75555e568f401e804b4aded258779ef4bce14ebaf04"
    )


def test_exact_official_repository_roster_is_frozen() -> None:
    assert [
        (
            row.protocol,
            row.remote,
            row.branch,
            row.remote_head_symref,
            row.sealed_tip,
            row.object_format,
        )
        for row in psim.REPOSITORIES
    ] == [
        (
            "ethereum",
            "https://github.com/ethereum/EIPs.git",
            "master",
            "refs/remotes/origin/master",
            "5e82ef62895121027a6c5f0c23276e1b2bed3071",
            "sha1",
        ),
        (
            "bitcoin",
            "https://github.com/bitcoin/bips.git",
            "master",
            "refs/remotes/origin/master",
            "b289d016b99c81527623c10e995e0318f744ebf3",
            "sha1",
        ),
    ]


def test_path_grammar_is_object_based_and_exact() -> None:
    eip = re.compile(psim.EIP_PATH_PATTERN, re.ASCII)
    bip = re.compile(psim.BIP_PATH_PATTERN, re.ASCII)
    assert eip.fullmatch("EIPS/eip-1.md")
    assert eip.fullmatch("EIPS/eip-1559.md")
    assert bip.fullmatch("bip-0003.md")
    assert bip.fullmatch("bip-0341.mediawiki")
    assert not eip.fullmatch("EIPS/eip-0001.md")
    assert not eip.fullmatch("eips/eip-1.md")
    assert not bip.fullmatch("bip-341.mediawiki")
    assert not bip.fullmatch("bip-0341.txt")
    source = psim.build_preregistration()["source_contract"]
    assert source["commit_subject_used_for_inclusion_or_classification"] is False
    assert source["current_checkout_used_as_historical_truth"] is False
    assert source["rename_detection"] is False


def test_source_envelope_and_four_archive_clocks_are_exact() -> None:
    payload = psim.build_preregistration()
    assert payload["source_contract"]["start"] == "2020-01-01T00:00:00Z"
    assert payload["source_contract"]["end_exclusive"] == (
        "2024-01-01T00:00:00Z"
    )
    assert payload["source_contract"]["card_end_exclusive"] == (
        "2024-04-01T00:00:00Z"
    )
    schedules = payload["availability_contract"]["schedules"]
    assert [(row["name"], row["delay_calendar_days"]) for row in schedules] == [
        ("ARCHIVE_D2", 2),
        ("ARCHIVE_D7", 7),
        ("ARCHIVE_D30", 30),
        ("ARCHIVE_D90", 90),
    ]
    assert [row["name"] for row in schedules if row["primary_economic_clock"]] == [
        "ARCHIVE_D90"
    ]
    assert all(
        row["profitability_claim_allowed"] == row["primary_economic_clock"]
        for row in schedules
    )


def test_splits_are_contiguous_and_bound_to_d90_decisions() -> None:
    payload = psim.build_preregistration()["split_contract"]
    splits = payload["splits"]
    assert [row["name"] for row in splits] == ["train", "test", "eval"]
    assert splits[0]["decision_start"] == psim.SOURCE_START
    assert splits[0]["decision_end_exclusive"] == splits[1]["decision_start"]
    assert splits[1]["decision_end_exclusive"] == splits[2]["decision_start"]
    assert splits[2]["decision_end_exclusive"] == psim.SOURCE_END_EXCLUSIVE
    assert payload["source_event_support_assignment_field"] == (
        "event_effective_day"
    )
    assert payload["daily_card_and_relation_support_assignment_field"] == (
        "decision_timestamp_under_ARCHIVE_D90"
    )
    assert payload["later_economic_split_assignment_field"] == (
        "decision_timestamp_under_ARCHIVE_D90"
    )
    assert payload["later_test_eval_minimum_cagr_strict_mdd"] == "3.0"
    assert payload["later_test_eval_positive_absolute_return_required"] is True


def test_boundary_reset_forbids_pre_window_warmup() -> None:
    reset = psim.build_preregistration()["boundary_reset_contract"]
    assert reset == {
        "pre_2020_blob_warmup_allowed": False,
        "window_revision_count_before_first_event": 0,
        "window_age_before_first_event": "PRE_WINDOW",
        "stale_age_before_first_protocol_event": "NO_EVENT_YET",
        "first_update_old_blob_role": "PRE_WINDOW_BASELINE",
        "pre_first_event_dependency_state": "PRE_WINDOW_UNKNOWN",
        "old_blob_creates_synthetic_prior_event": False,
        "boundary_states": [
            "PRE_WINDOW",
            "PRE_WINDOW_BASELINE",
            "PRE_WINDOW_UNKNOWN",
            "NO_EVENT_YET",
        ],
    }


def test_event_identity_and_ambiguous_paths_fail_closed() -> None:
    event = psim.build_preregistration()["event_contract"]
    assert event["event_types"] == ["CREATE", "UPDATE", "DELETE"]
    assert event["format_migration_same_number_is_update"] is True
    assert event["multiple_old_or_new_blobs_same_number"] == "REJECT"
    assert event["duplicate_number_paths_in_one_tree"] == "REJECT"
    assert event["path_number_preamble_number_must_match"] is True
    assert event["all_matching_path_events_retained"] is True
    assert event["event_id_formula"].startswith("SHA256(protocol||NUL||commit_oid")


def test_historical_parser_is_strict_without_current_status_projection() -> None:
    parser = psim.build_preregistration()["parser_contract"]
    assert parser["strict_utf8"] is True
    assert parser["nul_allowed"] is False
    assert parser["unicode_normalization"] == "NFC"
    assert parser["eip_frontmatter"]["required_number_field"] == "eip"
    assert parser["bip_preamble"]["required_number_field_casefold"] == "bip"
    assert parser["current_process_vocabulary_validation_allowed"] is False
    assert parser["declared_status_is_model_visible"] is False
    assert parser["primary_proposal_number_must_be_positive"] is True
    assert parser["metadata_parse_failure_action"] == "REJECT_NOT_AUDIT_ONLY"
    assert parser["duplicate_or_self_dependency_allowed"] is False
    assert parser["line_diff_algorithm"] == (
        "python_difflib_SequenceMatcher_autojunk_false_over_normalized_lines"
    )
    assert parser["reference_parser"]["version"] == (
        "PSIM_PREAMBLE_STATE_MACHINE_V1"
    )
    assert parser["reference_parser"]["duplicate_keys_after_casefold"] == "REJECT"
    assert parser["reference_parser"]["inline_comments_stripped"] is False
    assert parser["reference_parser"]["maximum_bip_leading_blank_lines"] == 3


def test_synthetic_eip_preamble_acceptance_is_exact() -> None:
    fields = psim.parse_eip_preamble(
        b"---\r\n"
        b"# full-line comment\r\n"
        b"eip: 123\r\n"
        b"title: Example: with colon\r\n"
        b"description: \"opaque quoted value\"\r\n"
        b"requires: 1, 002\r\n"
        b"tags:\r\n"
        b"  - one\r\n"
        b"---\r\n"
        b"# Abstract\r\n"
    )
    assert fields["eip"] == "123"
    assert fields["title"] == "Example: with colon"
    assert fields["description"] == '"opaque quoted value"'
    assert fields["tags"] == "- one"
    assert psim.parse_dependency_ids(fields["requires"], self_id=123) == (1, 2)


@pytest.mark.parametrize(
    "raw",
    [
        b"\xef\xbb\xbf---\neip: 1\n---\n",
        b" ---\neip: 1\n---\n",
        b"---\neip: 1\n",
        b"---\neip: 1\nEIP: 2\n---\n",
        b"---\n  orphan\n---\n",
        b"---\neip: 1\ntitle:\n---\n",
        b"---\neip: \"1\"\n---\n",
        b"---\neip: 1\n\n---\n",
        b"---\neip: 1\x00\n---\n",
    ],
)
def test_synthetic_eip_preamble_rejections_are_exact(raw: bytes) -> None:
    with pytest.raises((UnicodeDecodeError, ValueError)):
        psim.parse_eip_preamble(raw)


def test_synthetic_bip_mediawiki_and_markdown_acceptance_is_exact() -> None:
    mediawiki = psim.parse_bip_preamble(
        b"<pre>\n"
        b"  BIP: 0003\n"
        b"  Layer:\n"
        b"  Title: Example: with colon\n"
        b"  Authors: A <a@example.test>\n"
        b"           B <b@example.test>\n"
        b"  Requires: 1, 2\n"
        b"</pre>\n"
        b"== Abstract ==\n"
    )
    markdown = psim.parse_bip_preamble(
        b"\n"
        b"  BIP: 4\n"
        b"  Title: Markdown example\n"
        b"  Status: Draft\n"
        b"\n"
        b"# Abstract\n"
    )
    assert psim.parse_positive_proposal_number(mediawiki["bip"]) == 3
    assert mediawiki["layer"] == ""
    assert mediawiki["authors"] == (
        "A <a@example.test>\nB <b@example.test>"
    )
    assert psim.parse_dependency_ids(mediawiki["requires"], self_id=3) == (1, 2)
    assert markdown["bip"] == "4"
    assert markdown["title"] == "Markdown example"


@pytest.mark.parametrize(
    "raw",
    [
        b"\n\n\n\n  BIP: 1\n\n",
        b"<pre>\n  BIP: 1\n",
        b"<pre>\n  BIP: 1\n  bip: 2\n</pre>\n",
        b"<pre>\n  Title: Missing number\n</pre>\n",
        b"<pre>\n  orphan\n  BIP: 1\n</pre>\n",
        b"  BIP: zero\n\n",
        b"  BIP: 0\n\n",
        b"  BIP: 1\x00\n\n",
    ],
)
def test_synthetic_bip_preamble_rejections_are_exact(raw: bytes) -> None:
    with pytest.raises((UnicodeDecodeError, ValueError)):
        psim.parse_bip_preamble(raw)


@pytest.mark.parametrize(
    ("value", "self_id"),
    [
        ("1, 1", 3),
        ("3", 3),
        ("one", 3),
        ("1\n2", 3),
        ("1,,2", 3),
        ("0", 3),
    ],
)
def test_synthetic_dependency_rejections_are_exact(
    value: str,
    self_id: int,
) -> None:
    with pytest.raises(ValueError):
        psim.parse_dependency_ids(value, self_id=self_id)


def test_bucket_edges_are_frozen_and_strictly_increasing() -> None:
    buckets = psim.build_preregistration()["bucket_contract"]
    assert set(buckets["edges"]) == set(psim.BUCKET_EDGES)
    for edges in buckets["edges"].values():
        assert edges[0] == 0
        assert all(left < right for left, right in zip(edges, edges[1:]))
    assert buckets["raw_numeric_values_model_visible"] is False


def test_cross_protocol_pairing_is_complete_and_nonselective() -> None:
    relation = psim.build_preregistration()["daily_relation_contract"]
    assert relation["both_protocol_sets_nonempty"] == (
        "complete_cartesian_product"
    )
    assert relation["exactly_one_protocol_set_nonempty"] == (
        "each_anchor_with_most_recent_opposite_event_in_90_days"
    )
    assert relation["missing_opposite_sentinel"] == "NO_COUNTERPART"
    assert relation["both_sets_empty_sentinel"] == "NO_ANCHOR"
    assert relation["semantic_or_market_pair_selection_allowed"] is False
    assert relation["over_limit_card_action"] == "SOURCE_GATE_REJECT_NO_TRUNCATION"


def test_single_model_relation_and_target_contract_is_exact() -> None:
    representation = psim.build_preregistration()["representation_contract"]
    assert representation["later_relation_tokens"] == [
        "CONVERGENT_INTENT",
        "COMPLEMENTARY_INTENT",
        "TECHNICAL_TENSION",
        "INDEPENDENT_INTENT",
        "INSUFFICIENT_EVIDENCE",
        "ABSTAIN",
    ]
    assert representation["later_model_actions"] == [
        "TARGET_LONG",
        "TARGET_FLAT",
        "TARGET_SHORT",
    ]
    assert representation["single_model_single_call_per_card"] is True
    assert representation["analyzer_trader_pair_allowed"] is False
    assert representation["free_form_rationale_allowed"] is False
    assert representation["abstain_forces_target_flat"] is True
    assert representation["raw_price_return_rank_pnl_model_inputs_allowed"] is False


def test_memorization_quarantine_and_challenge_are_exact() -> None:
    contract = psim.build_preregistration()["memorization_contract"]
    assert contract["quarantine"] == {
        "ethereum": [20, 721, 1559, 3675, 4337, 4844, 4895],
        "bitcoin": [32, 39, 44, 141, 340, 341, 342],
    }
    assert contract["quarantined_events_retained_in_source_support"] is True
    assert contract["quarantined_relation_units_model_or_economics_allowed"] is False
    assert contract["candidate_ids_per_challenge"] == 8
    assert contract["chance_probability"] == "0.125"
    assert contract["tests"] == ["ethereum", "bitcoin", "combined"]
    assert contract["bonferroni_family_alpha"] == "0.01"
    assert contract["minimum_eligible_events_per_protocol"] == 32
    assert contract["base_and_final_models_must_pass_before_market"] is True
    assert contract["repair_resample_or_model_swap_after_result_allowed"] is False


def test_source_gates_and_controls_are_frozen_in_order() -> None:
    contract = psim.build_preregistration()["source_support_contract"]
    assert contract["gates_in_order"] == list(psim.SOURCE_ONLY_GATES)
    assert len(contract["gates_in_order"]) == 13
    assert contract["relation_controls"] == list(psim.RELATION_CONTROLS)
    assert contract["relation_control_transforms"] == psim.CONTROL_TRANSFORMS
    assert contract["relation_control_eligibility"] == psim.CONTROL_ELIGIBILITY
    assert len(contract["relation_controls"]) == 7
    assert contract["all_gates_required"] is True
    assert contract["parser_success_fraction_required"] == "1.0"
    assert contract["independent_replay_roots"] == 2
    assert contract["shared_git_object_alternates_allowed"] is False
    assert contract["first_failure_action"] == (
        "REJECT_PSIM_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
    )
    assert contract["source_drop_or_repair_allowed"] is False
    assert contract["gate_relaxation_after_incidence_allowed"] is False


def test_control_sensitivity_metric_has_no_pooling_discretion() -> None:
    metric = psim.build_preregistration()["source_support_contract"][
        "control_sensitivity_metric"
    ]
    assert metric["comparison_unit"] == (
        "canonical local daily relation-card payload for one decision day"
    )
    assert metric["changed_definition"].startswith(
        "SHA256(canonical_json(transformed_local_payload)) !="
    )
    assert metric["denominator"] == (
        "unique eligible decision days for one exact "
        "control,schedule,split cell"
    )
    assert metric["minimum_eligible_days_per_cell"] == 4
    assert metric["minimum_changed_fraction_per_cell"] == "0.10"
    assert metric["aggregation"] == (
        "require every control x archive schedule x split cell; "
        "no pooling or weighting"
    )
    assert metric["zero_eligible_cell_action"] == "REJECT"
    assert metric["quarantined_events_in_source_control_payload"] is True


def test_preregistration_has_zero_forbidden_access() -> None:
    access = psim.build_preregistration()["forbidden_access_contract"]
    assert set(access["counters"]) == set(psim.FORBIDDEN_COUNTERS)
    assert set(access["counters"].values()) == {0}
    assert access["network_calls_during_preregistration"] == 0
    assert access["git_commands_during_preregistration"] == 0
    assert access["source_incidence_opened"] is False
    assert access["proposal_blobs_opened"] is False
    assert access["btc_or_funding_outcomes_opened"] is False
    assert access["models_loaded"] == 0


def test_excluded_probe_discloses_only_post_2023_feasibility() -> None:
    probe = psim.build_preregistration()["excluded_feasibility_probe"]
    assert probe["probe_start_inclusive"] == "2024-01-01T00:00:00Z"
    assert probe["source_interval_incidence_opened"] is False
    assert probe["ethereum"]["proposal_path_touches"] == 2_358
    assert probe["bitcoin"]["proposal_path_touches"] == 1_664
    assert probe["probe_may_set_economic_direction_or_threshold"] is False


def test_manifest_hash_binds_exact_core() -> None:
    payload = psim.build_preregistration()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == psim.canonical_hash(core)
    assert payload == psim.build_preregistration()


def test_committed_artifact_matches_exact_builder() -> None:
    raw = (psim.REPO_ROOT / RESULT_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    assert json.loads(raw) == psim.build_preregistration()


def test_write_is_canonical_and_idempotent(tmp_path: Path) -> None:
    destination = tmp_path / "psim.json"
    first_path, first_payload = psim.write_preregistration(destination)
    first_bytes = first_path.read_bytes()
    second_path, second_payload = psim.write_preregistration(destination)
    assert first_path == second_path
    assert first_payload == second_payload
    assert first_bytes == second_path.read_bytes()
    assert first_bytes.endswith(b"\n")
    assert json.loads(first_bytes) == first_payload


def test_write_rejects_conflicting_existing_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "psim.json"
    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="existing PSIM preregistration differs"):
        psim.write_preregistration(destination)


def test_repository_roster_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(psim, "REPOSITORIES", psim.REPOSITORIES[:-1])
    with pytest.raises(RuntimeError, match="exactly two"):
        psim.build_preregistration()


def test_archive_clock_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutated = list(psim.ARCHIVE_SCHEDULES)
    mutated[-1] = psim.ArchiveSchedule("ARCHIVE_D60", 60, True, True)
    monkeypatch.setattr(psim, "ARCHIVE_SCHEDULES", tuple(mutated))
    with pytest.raises(RuntimeError, match="archive-delay roster changed"):
        psim.build_preregistration()


def test_bucket_mutation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(psim.BUCKET_EDGES, "window_age_days", (0, 7, 7))
    with pytest.raises(RuntimeError, match="not strictly increasing"):
        psim.build_preregistration()


def test_decision_hash_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(psim, "DECISION_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="decision document hash changed"):
        psim.build_preregistration()


def test_module_has_no_network_git_market_or_model_execution_imports() -> None:
    source = (psim.REPO_ROOT / psim.SCRIPT_PATH).read_text(encoding="utf-8")
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
            "subprocess",
            "git",
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
    assert called_names.isdisjoint(
        {"urlopen", "get", "post", "request", "run", "Popen", "check_output"}
    )


def test_selection_commit_and_next_step_are_exact() -> None:
    payload = psim.build_preregistration()
    assert payload["candidate"]["selection_commit"] == (
        "6ebb43406f7197e2afb2e2fa5cb39b0a2cba2826"
    )
    assert payload["candidate"]["stage"] == "source_support_only"
    assert payload["next_authorized_step"] == (
        "implement and seal synthetic-only PSIM-D1 source-support evaluator"
    )
