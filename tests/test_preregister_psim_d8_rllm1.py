from __future__ import annotations

import ast
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re

import pytest

from training import preregister_psim_d8_rllm1 as prereg


RESULT_PATH = prereg.REPO_ROOT / prereg.DEFAULT_OUTPUT
RESULT_SHA256 = (
    "78e467b64e0728231626aa8300fe13f10a445f494b368459c8cab9852d752759"
)


def _cards() -> list[dict[str, object]]:
    raw = gzip.decompress(
        (prereg.REPO_ROOT / prereg.D8_CARDS).read_bytes()
    )
    return [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line
    ]


def _economic_cards() -> list[dict[str, object]]:
    return [
        card
        for card in _cards()
        if card["schedule"] == prereg.PRIMARY_SCHEDULE
        and prereg._split_for_decision(str(card["decision_at"])) is not None
    ]


def test_committed_preregistration_is_exact_and_canonical() -> None:
    payload = prereg.build_preregistration()
    assert json.loads(RESULT_PATH.read_text(encoding="utf-8")) == payload
    assert RESULT_PATH.read_bytes() == prereg.canonical_json_bytes(
        payload,
        pretty=True,
    )
    assert hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest() == RESULT_SHA256
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["manifest_hash"] == (
        "d2d22214b810cc99a6d7e893b35f91c4f46fde803d7309bf388954dc55729fff"
    )


def test_preregistration_reads_only_frozen_source_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    original = Path.read_bytes

    def audited_read_bytes(path: Path) -> bytes:
        try:
            relative = path.resolve().relative_to(prereg.REPO_ROOT).as_posix()
        except ValueError:
            relative = path.as_posix()
        observed.append(relative)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", audited_read_bytes)
    payload = prereg.build_preregistration()
    assert set(observed) <= prereg.ALLOWED_SOURCE_READS
    assert not (set(observed) & prereg.FORBIDDEN_BOUND_PATHS)
    boundary = payload["access_boundary"]
    assert boundary["market_or_funding_paths_read"] == []
    assert boundary["market_or_funding_payload_bytes_hashed"] is False
    assert boundary["market_rows_parsed"] == 0
    assert boundary["funding_rows_parsed"] == 0
    assert boundary["model_loaded"] is False
    assert boundary["rewards_created"] == 0
    assert boundary["economic_metrics_computed"] == 0
    assert boundary["test_outcomes_opened"] is False
    assert boundary["eval_outcomes_opened"] is False


def test_d8_terminal_source_authority_is_frozen_and_not_extended() -> None:
    payload = prereg.build_preregistration()
    source = payload["source_authority"]
    assert source["terminal_repository_commit"] == (
        prereg.D8_TERMINAL_REPOSITORY_COMMIT
    )
    assert source["source_execution_commit"] == (
        prereg.D8_SOURCE_EXECUTION_COMMIT
    )
    assert source["source_execution_commit"] == (
        json.loads(
            (
                prereg.REPO_ROOT / prereg.D8_EXECUTION_SEAL
            ).read_text(encoding="utf-8")
        )["shared_commit"]
    )
    assert source["source_result"]["sha256"] == prereg.D8_RESULT_SHA256
    assert source["source_result"]["result_hash"] == prereg.D8_RESULT_HASH
    assert source["source_result"]["decision"] == "pass"
    assert source["source_result"]["terminal_action"] == (
        "ACCEPT_PSIM_D8_SOURCE_SUPPORT_ONLY_NO_PROFITABILITY_CLAIM"
    )
    assert source["source_run_is_terminal"] is True
    assert source["source_rerun_repair_or_d9_allowed"] is False
    assert payload["candidate"]["source_candidate"] == "PSIM-D8"
    assert payload["candidate"]["relation_scope"] == (
        "SELECTED_SUBCARD_RELATION_NOT_LOGICAL_DAY_AGGREGATE"
    )


def test_selector_is_source_only_exact_and_manifest_bound() -> None:
    cards = _economic_cards()
    assert len(cards) == 1_461
    selected = [
        card
        for card in cards
        if card["local_payload"]["relation_subcard_manifest"]["subcard_count"] > 1
    ]
    assert selected
    for card in selected:
        prereg._validate_card(card)
        manifest = card["local_payload"]["relation_subcard_manifest"]
        material = (
            f"{card['prior_card_hash']}\x00"
            f"{manifest['complete_relation_roster_sha256']}\x00"
            f"{card['decision_at']}\x00{prereg.SELECTOR_SALT}"
        ).encode("utf-8")
        expected = (
            int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            % int(manifest["subcard_count"])
        )
        assert prereg.selected_subcard_selector_digest(card) == (
            hashlib.sha256(material).hexdigest()
        )
        assert prereg.selected_subcard_ordinal(card) == expected
        descriptor = manifest["subcards"][expected]
        assert prereg.selected_relation_units(card) == card["local_payload"][
            "relation_units"
        ][descriptor["start"] : descriptor["end_exclusive"]]

    tampered = deepcopy(selected[0])
    tampered["local_payload"]["relation_subcard_manifest"]["subcards"][0][
        "subcard_payload_sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="manifest|subcard"):
        prereg.selected_relation_units(tampered)


def test_quarantine_is_payload_only_and_never_selects_an_alternate_slice() -> None:
    mixed = next(
        card
        for card in _economic_cards()
        if {
            bool(unit.get("memorization_excluded"))
            for unit in prereg.selected_relation_units(card)
        }
        == {False, True}
    )
    original = deepcopy(mixed)
    ordinal = prereg.selected_subcard_ordinal(mixed)
    units = prereg.selected_relation_units(mixed)
    payload = prereg.build_selected_source_payload(mixed)
    assert mixed == original
    assert prereg.selected_subcard_ordinal(mixed) == ordinal
    assert len(payload["relation_edges"]) == sum(
        not bool(unit.get("memorization_excluded")) for unit in units
    )
    excluded_only = next(
        card
        for card in _economic_cards()
        if all(
            bool(unit.get("memorization_excluded"))
            for unit in prereg.selected_relation_units(card)
        )
    )
    assert prereg.build_selected_source_payload(excluded_only) == {
        "selected_subcard_relation": "NO_MODEL_ELIGIBLE_RELATION",
        "events": [],
        "relation_edges": [],
        "forced_relation": "INSUFFICIENT_EVIDENCE",
        "forced_target_rule": "KEEP_CURRENT_OR_FLAT_AT_SPLIT_START",
    }


def test_redaction_removes_identity_numeric_and_release_leaks() -> None:
    raw = (
        "TITLE|ADD|Title: EIP-1559 London 2021-08-05 v1.2.3 "
        "https://example.com/a user@example.com 0x1234abcd "
        "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh sha256 P2PKH 42 "
        "draft final withdrawn stagnant active replaced obsolete"
    )
    redacted = prereg.redact_model_text(raw)
    lowered = redacted.lower()
    for forbidden in (
        "1559",
        "london",
        "2021",
        "example.com",
        "user@example.com",
        "1234abcd",
        "bc1qxy",
        "sha256",
        "p2pkh",
        "42",
        "draft",
        "final",
        "withdrawn",
        "stagnant",
        "active",
        "replaced",
        "obsolete",
    ):
        assert forbidden not in lowered
    assert "<TITLE>" in redacted
    assert not re.search(r"\d", redacted)


def test_all_historical_model_prompts_contain_no_literal_digits() -> None:
    maximum = 0
    for card in _economic_cards():
        source = prereg.build_selected_source_payload(card)
        prompt = prereg.render_policy_prompt(
            source,
            current_position="POSITION_FLAT",
        )
        maximum = max(maximum, len(prompt.encode("utf-8")))
        assert not re.search(r"\d", prompt)
        assert str(card["decision_at"]) not in prompt
        assert str(card["prior_card_hash"]) not in prompt
        assert str(card["card_hash"]) not in prompt
        lowered = prompt.lower()
        for term in prereg.LIFECYCLE_STATUS_TERMS:
            assert re.search(rf"\b{re.escape(term.lower())}\b", lowered) is None
    assert maximum == 122_113


def test_capacity_and_memorization_support_are_frozen() -> None:
    payload = prereg.build_preregistration()
    capacity = payload["source_only_capacity"]
    assert capacity["logical_decision_cards"] == 1_461
    assert capacity["split_card_counts"] == {
        "train": 731,
        "test": 365,
        "eval": 365,
    }
    assert capacity["selected_relation_units"]["maximum"] == 64
    assert capacity["eligible_relation_units_after_quarantine"][
        "forced_no_eligible_cards"
    ] == 117
    assert capacity["rendered_prompt_utf8_bytes"]["maximum"] == 122_113
    challenge = payload["memorization_contract"]
    assert challenge["capacity"]["selected_challenge_events"] == {
        "bitcoin": 64,
        "ethereum": 64,
    }
    assert challenge["capacity"]["combined_selected_challenge_events"] == 128
    assert challenge["bonferroni_reject_p_below"] == pytest.approx(
        0.01 / 3.0
    )
    assert challenge["choice_codes"] == list(
        prereg.MEMORIZATION_CHALLENGE_CODES
    )
    assert challenge["decoy_hash"] == (
        "SHA256(lowercase_event_id || NUL || lowercase_protocol || NUL || "
        "four_digit_effective_year || NUL || "
        "canonical_decimal_proposal_id || NUL || "
        "PSIM_MEMORIZATION_V1_DECOY)"
    )
    assert challenge["order_hash"] == (
        "SHA256(lowercase_event_id || NUL || lowercase_protocol || NUL || "
        "four_digit_effective_year || NUL || "
        "canonical_decimal_proposal_id || NUL || "
        "PSIM_MEMORIZATION_V1_ORDER)"
    )
    assert challenge["decoded_generation"] is False
    assert challenge["scoring"].startswith("one text-only model forward")
    decoy_hashes = [
        prereg.memorization_decoy_hash(
            "a" * 64,
            "bitcoin",
            2020,
            proposal_id,
        )
        for proposal_id in (1, 2, 3, 4, 5, 6, 7, 8)
    ]
    order_hashes = [
        prereg.memorization_order_hash(
            "a" * 64,
            "bitcoin",
            2020,
            proposal_id,
        )
        for proposal_id in (1, 2, 3, 4, 5, 6, 7, 8)
    ]
    assert len(set(decoy_hashes)) == 8
    assert len(set(order_hashes)) == 8
    assert decoy_hashes != order_hashes
    assert decoy_hashes == [
        prereg.memorization_decoy_hash(
            "A" * 64,
            "BITCOIN",
            "2020",
            proposal_id,
        )
        for proposal_id in (1, 2, 3, 4, 5, 6, 7, 8)
    ]
    assert challenge["capacity"]["true_choice_code_histogram"] == {
        "ethereum": {
            "A": 8,
            "B": 9,
            "C": 5,
            "D": 10,
            "E": 4,
            "F": 12,
            "G": 7,
            "H": 9,
        },
        "bitcoin": {
            "A": 8,
            "B": 6,
            "C": 5,
            "D": 9,
            "E": 12,
            "F": 9,
            "G": 8,
            "H": 7,
        },
        "combined": {
            "A": 16,
            "B": 15,
            "C": 10,
            "D": 19,
            "E": 16,
            "F": 21,
            "G": 15,
            "H": 16,
        },
    }
    assert max(
        challenge["capacity"]["maximum_true_code_share"].values()
    ) <= 0.20
    assert challenge["base_model_challenge_before_any_market_access"] is True
    assert challenge[
        "final_model_challenge_after_test_selection_before_eval_market"
    ] is True


def test_model_training_economics_controls_and_final_gates_are_fixed() -> None:
    payload = prereg.build_preregistration()
    model = payload["model_contract"]
    assert model["id"] == "google/gemma-4-E4B-it"
    assert model["revision"] == prereg.MODEL_REVISION
    assert model["files"] == dict(prereg.MODEL_FILES)
    assert model["quantization"] == (
        "bitsandbytes NF4 double-quant BF16 compute"
    )
    assert model["single_forward_per_logical_decision"] is True
    assert model["head_bias"] is False
    assert model["additive_direction_bias_or_posthoc_calibration"] is False
    assert model["source_embedding_position_token"] == "POSITION_FLAT"
    assert model["snapshot_verified_by_this_preregistration"] is False

    development = payload["semantic_encoder_development_gate"]
    assert development["algorithms"] == [
        "ridge_fitted_q",
        "extra_trees_fitted_q",
    ]
    assert development["2022_gate_before_qlora"][
        "cagr_to_strict_mdd_minimum"
    ] == 1.5
    assert development["failure_action"].endswith(
        "WITHOUT_QLORA_OR_EVAL_OPEN"
    )

    rllm = payload["conditional_rllm_contract"]
    assert rllm["seeds"] == [20260727, 20260728]
    assert rllm["checkpoint_optimizer_steps"] == [80, 160, 240]
    assert rllm["relation_teacher_codes"] == list(
        prereg.RELATION_TEACHER_CODES
    )
    assert "lowercase_hex_selector_digest" in rllm["relation_teacher"]
    assert prereg.RELATION_TEACHER_SALT in rllm["relation_teacher"]
    assert rllm["relation_teacher_decoded_generation"] is False
    first, second = _economic_cards()[:2]
    for card in (first, second):
        mapping = prereg.relation_teacher_code_mapping(card)
        assert set(mapping) == set(prereg.RELATION_LABELS)
        assert set(mapping.values()) == set(prereg.RELATION_TEACHER_CODES)
    assert prereg.relation_teacher_code_mapping(first) != (
        prereg.relation_teacher_code_mapping(second)
    )
    assert rllm["lora"] == {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "targets": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }
    assert rllm["no_decoded_trading_generation"] is True

    economics = payload["economic_contract"]
    assert economics["clock"] == "ARCHIVE_D90"
    assert economics["base_cost_rate"] == 0.0006
    assert economics["stress_cost_rate"] == 0.001
    assert economics["delay"] == (
        "one complete five-minute market bar; terminal flatten unchanged"
    )
    assert economics["one_day_staleness_diagnostic"] == (
        "reported separately; cannot rescue a failed five-minute delay or "
        "primary result"
    )
    assert "one complete daily decision" not in json.dumps(economics)
    assert economics["full_calendar_including_idle_time"] is True
    assert economics["leverage_optimization"] is False
    assert economics["market_binding"]["expected_sha256"] == (
        prereg.MARKET_SHA256
    )
    assert economics["funding_binding"]["expected_sha256"] == (
        prereg.FUNDING_SHA256
    )

    controls = payload["controls_and_statistics"]
    assert controls["all_failed_or_flat_variants_remain_in_family"] is True
    assert controls["max_stat"]["monte_carlo_draws"] == 100_000
    assert "ethereum_only" in controls["mandatory_controls"]
    assert "bitcoin_only" in controls["mandatory_controls"]
    assert "neutral_action_code_permutation" in controls[
        "mandatory_controls"
    ]

    gates = payload["final_test_and_eval_gates"]
    for split in ("test_2022", "eval_2023"):
        assert gates[split]["absolute_return_positive"] is True
        assert gates[split]["cagr_to_strict_mdd_minimum"] == 3.0
        assert gates[split]["strict_mdd_pct_maximum"] == 15.0
        assert gates[split]["both_half_returns_positive"] is True
        assert gates[split]["stress_return_positive"] is True
        assert gates[split]["delay_return_positive"] is True
    assert gates["eval_policy_count"] == 1
    assert "absolute_return" in gates["required_report_fields"]


def test_source_preregistration_has_no_runner_or_market_loader_import() -> None:
    source_path = (
        prereg.REPO_ROOT
        / "training/preregister_psim_d8_rllm1.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    joined = "\n".join(imports)
    assert "build_protocol_specification_intent_maturity_d8_source_support" not in joined
    assert "bctp_strict_economics" not in joined
    assert "bctp_transition_labels" not in joined
    assert "pandas" not in joined
    assert "read_csv" not in calls
    assert "torch" not in joined
    assert "transformers" not in joined
