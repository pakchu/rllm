from __future__ import annotations

import ast
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from training import preregister_psim_d8_rllm1 as prereg
from training import run_psim_d8_rllm1_base_memorization_gate as gate


@pytest.fixture(scope="module")
def events() -> list[dict[str, Any]]:
    return gate._load_frozen_gzip_jsonl(
        prereg.D8_EVENTS,
        expected_sha256=prereg.D8_EVENTS_SHA256,
        expected_decompressed_sha256=prereg.D8_EVENTS_ROWS_SHA256,
    )


@pytest.fixture(scope="module")
def cards() -> list[dict[str, Any]]:
    return gate._load_frozen_gzip_jsonl(
        prereg.D8_CARDS,
        expected_sha256=prereg.D8_CARDS_SHA256,
        expected_decompressed_sha256=prereg.D8_CARDS_ROWS_SHA256,
    )


@pytest.fixture(scope="module")
def cases(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return gate.build_challenge_cases(events)


@pytest.fixture(scope="module")
def processor() -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        gate._model_snapshot(),
        local_files_only=True,
        trust_remote_code=False,
    )


def _predictions(
    cases: list[dict[str, Any]],
    *,
    ethereum_correct: int,
    bitcoin_correct: int,
) -> list[dict[str, str]]:
    seen = Counter()
    requested = {
        "ethereum": ethereum_correct,
        "bitcoin": bitcoin_correct,
    }
    predictions: list[dict[str, str]] = []
    for case in cases:
        protocol = str(case["protocol"])
        true_code = str(case["true_code"])
        correct = seen[protocol] < requested[protocol]
        seen[protocol] += 1
        predicted = true_code
        if not correct:
            predicted = next(
                code
                for code in prereg.MEMORIZATION_CHALLENGE_CODES
                if code != true_code
            )
        predictions.append(
            {
                "case_hash": str(case["case_hash"]),
                "predicted_code": predicted,
            }
        )
    return predictions


def test_preregistration_and_exact_runtime_are_bound_without_weights() -> None:
    payload = gate.validate_preregistration()
    assert payload["manifest_hash"] == gate.PREREGISTRATION_MANIFEST_HASH
    assert payload["memorization_contract"]["version"] == (
        prereg.MEMORIZATION_CONTRACT_VERSION
    )
    runtime = gate.validate_local_runtime(load_processor=False)
    assert runtime["model_id"] == "google/gemma-4-E4B-it"
    assert runtime["revision"] == prereg.MODEL_REVISION
    assert runtime["files"] == dict(prereg.MODEL_FILES)
    assert runtime["runtime_versions"] == dict(prereg.RUNTIME_VERSIONS)
    assert runtime["transformers_revision"] == (
        prereg.TRANSFORMERS_REVISION
    )
    assert runtime["architecture"] == {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_hidden_size": 2_560,
        "text_maximum_positions": 131_072,
        "text_hidden_layers": 42,
        "text_vocabulary_size": 262_144,
    }
    assert runtime["cuda"]["device_count"] == 1
    assert runtime["cuda"]["bf16_supported"] is True


def test_direct_script_entrypoint_imports_without_pythonpath() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(gate.__file__)),
            "--help",
        ],
        cwd=prereg.REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--validate-only" in completed.stdout


def test_challenge_roster_is_nonempty_balanced_and_source_exact(
    events: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> None:
    assert len(cases) == 128
    assert len({case["case_hash"] for case in cases}) == 128
    assert gate.canonical_hash(
        [case["case_hash"] for case in cases]
    ) == "5065cd58322aee8f38f11ec2c4a186fb1a7ba8133aa2b2bb0182f67322a8bf39"
    by_event = {str(event["event_id"]): event for event in events}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        event = by_event[str(case["event_id"])]
        assert event["memorization_excluded"] is False
        assert case["redacted_text"]
        assert case["redacted_text"] == (
            prereg.memorization_redacted_event_text(event)
        )
        candidates = case["candidate_code_to_proposal_id"]
        assert tuple(candidates) == prereg.MEMORIZATION_CHALLENGE_CODES
        assert len(set(candidates.values())) == 8
        assert candidates[case["true_code"]] == event["proposal_number"]
        grouped[(str(case["protocol"]), int(case["effective_year"]))].append(
            case
        )

    assert set(grouped) == {
        (protocol, year)
        for protocol in ("ethereum", "bitcoin")
        for year in (2020, 2021, 2022, 2023)
    }
    for (protocol, year), selected_cases in grouped.items():
        assert len(selected_cases) == 16
        assert Counter(case["true_code"] for case in selected_cases) == Counter(
            {
                code: 2
                for code in prereg.MEMORIZATION_CHALLENGE_CODES
            }
        )
        all_candidates = [
            event
            for event in events
            if event["protocol"] == protocol
            and str(event["effective_day"]).startswith(str(year))
            and not event["memorization_excluded"]
        ]
        eligible = [
            event
            for event in all_candidates
            if prereg.memorization_redacted_event_text(event)
        ]
        expected_selected = sorted(
            eligible,
            key=lambda event: prereg.memorization_selection_hash(
                str(event["event_id"])
            ),
        )[:16]
        assert {case["event_id"] for case in selected_cases} == {
            event["event_id"] for event in expected_selected
        }
        assignments = prereg.memorization_true_code_assignments(
            expected_selected
        )
        proposal_ids = {
            int(event["proposal_number"]) for event in all_candidates
        }
        for case in selected_cases:
            event_id = str(case["event_id"])
            true_id = int(by_event[event_id]["proposal_number"])
            assert case["true_code"] == assignments[event_id]
            expected_decoys = sorted(
                (value for value in proposal_ids if value != true_id),
                key=lambda value: prereg.memorization_decoy_hash(
                    event_id,
                    protocol,
                    year,
                    value,
                ),
            )[:7]
            expected_decoys = sorted(
                expected_decoys,
                key=lambda value: prereg.memorization_order_hash(
                    event_id,
                    protocol,
                    year,
                    value,
                ),
            )
            observed_decoys = [
                case["candidate_code_to_proposal_id"][code]
                for code in prereg.MEMORIZATION_CHALLENGE_CODES
                if code != case["true_code"]
            ]
            assert observed_decoys == expected_decoys


def test_frozen_source_loader_rejects_hash_drift_and_symlinks(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="source hash changed"):
        gate._load_frozen_gzip_jsonl(
            prereg.D8_EVENTS,
            expected_sha256="0" * 64,
            expected_decompressed_sha256=prereg.D8_EVENTS_ROWS_SHA256,
        )
    linked = tmp_path / "events.jsonl.gz"
    linked.symlink_to(prereg.REPO_ROOT / prereg.D8_EVENTS)
    with pytest.raises(RuntimeError, match="unsafe frozen source"):
        gate._load_frozen_gzip_jsonl(
            linked,
            expected_sha256=prereg.D8_EVENTS_SHA256,
            expected_decompressed_sha256=prereg.D8_EVENTS_ROWS_SHA256,
        )


def test_prompt_and_exact_tokenizer_contract(
    processor: Any,
    cases: list[dict[str, Any]],
) -> None:
    tokenizer = gate.validate_processor_tokenizer(processor)
    assert tokenizer["processor_class"] == "Gemma4Processor"
    assert tokenizer["tokenizer_class"] == "GemmaTokenizer"
    assert tokenizer["vocabulary_size"] == 262_144
    assert tokenizer["challenge_code_token_ids"] == dict(
        gate.EXPECTED_CODE_TOKEN_IDS
    )
    template = gate.validate_chat_template_contract(processor)
    assert template == {
        "raw_user_content_terminal": "ANSWER=",
        "assistant_prefix_token_ids": [105, 4368, 107],
        "assistant_prefix_decoded": "<|turn>model\n",
        "scored_position": "first_assistant_token",
        "validated": True,
    }
    for case in cases:
        prompt = gate.render_challenge_prompt(case)
        assert prompt.endswith("ANSWER=")
        assert str(case["event_id"]) not in prompt
        assert all(
            f"{code}={case['candidate_code_to_proposal_id'][code]}" in prompt
            for code in prereg.MEMORIZATION_CHALLENGE_CODES
        )


def test_all_policy_and_challenge_prompts_fit_without_truncation(
    processor: Any,
    cards: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> None:
    capacity = gate.validate_prompt_capacity(processor, cards, cases)
    assert capacity["truncation"] is False
    assert capacity["maximum_input_tokens"] == 32_768
    assert capacity["policy"]["count"] == 1_461
    assert capacity["memorization_challenge"]["count"] == 128
    assert capacity["policy"]["minimum_tokens"] == 320
    assert capacity["policy"]["maximum_tokens"] == 30_961
    assert capacity["policy"]["mean_tokens"] == pytest.approx(
        2045.4592744695415
    )
    assert capacity["policy"]["maximum_identity"] == {
        "schedule": "ARCHIVE_D90",
        "decision_at": "2022-10-23T12:05:00Z",
        "card_hash": (
            "bfd823e7b16e4b374b2563602b0ef72a0a76e7cdd85c976c61243b9c0068c4fd"
        ),
        "tokens": 30_961,
    }
    assert capacity["memorization_challenge"]["minimum_tokens"] == 159
    assert capacity["memorization_challenge"]["maximum_tokens"] == 10_291
    assert capacity["memorization_challenge"]["mean_tokens"] == pytest.approx(
        1136.6953125
    )
    assert capacity["memorization_challenge"]["maximum_identity"] == {
        "case_hash": (
            "95a8dc5a5ab0f7b8cc4031ec8ca2ead6c1389c59bc9b3cd5ec418a0ebd04eca3"
        ),
        "tokens": 10_291,
    }


def test_exact_binomial_rejection_boundaries() -> None:
    assert gate.exact_binomial_upper_tail(
        successes=16,
        trials=64,
    ) == pytest.approx(0.004654595768052329)
    assert gate.exact_binomial_upper_tail(
        successes=17,
        trials=64,
    ) == pytest.approx(0.001798088925950917)
    assert gate.exact_binomial_upper_tail(
        successes=27,
        trials=128,
    ) == pytest.approx(0.004279428761027487)
    assert gate.exact_binomial_upper_tail(
        successes=28,
        trials=128,
    ) == pytest.approx(0.002120511589854342)
    assert gate.exact_binomial_upper_tail(
        successes=16,
        trials=64,
    ) > gate.REJECTION_THRESHOLD
    assert gate.exact_binomial_upper_tail(
        successes=17,
        trials=64,
    ) < gate.REJECTION_THRESHOLD
    assert gate.exact_binomial_upper_tail(
        successes=27,
        trials=128,
    ) > gate.REJECTION_THRESHOLD
    assert gate.exact_binomial_upper_tail(
        successes=28,
        trials=128,
    ) < gate.REJECTION_THRESHOLD


def test_prediction_gate_rejects_only_preregistered_statistical_breaches(
    cases: list[dict[str, Any]],
) -> None:
    passed = gate.evaluate_predictions(
        cases,
        _predictions(cases, ethereum_correct=16, bitcoin_correct=11),
    )
    assert passed["decision"] == "pass"
    assert passed["source_feature_construction_authorized"] is True
    rejected_family = gate.evaluate_predictions(
        cases,
        _predictions(cases, ethereum_correct=17, bitcoin_correct=0),
    )
    assert rejected_family["decision"] == "reject"
    assert rejected_family["statistics"]["ethereum"][
        "memorization_rejected"
    ] is True
    rejected_combined = gate.evaluate_predictions(
        cases,
        _predictions(cases, ethereum_correct=14, bitcoin_correct=14),
    )
    assert rejected_combined["decision"] == "reject"
    assert rejected_combined["statistics"]["ethereum"][
        "memorization_rejected"
    ] is False
    assert rejected_combined["statistics"]["bitcoin"][
        "memorization_rejected"
    ] is False
    assert rejected_combined["statistics"]["combined"][
        "memorization_rejected"
    ] is True
    assert rejected_combined["terminal_action"] == (
        gate.MEMORIZATION_FAILURE_ACTION
    )
    assert rejected_combined["market_access_authorized"] is False


def test_prediction_roster_rejects_duplicates_missing_and_invalid_codes(
    cases: list[dict[str, Any]],
) -> None:
    predictions = _predictions(
        cases,
        ethereum_correct=0,
        bitcoin_correct=0,
    )
    duplicated = [dict(row) for row in predictions]
    duplicated[-1]["case_hash"] = duplicated[0]["case_hash"]
    with pytest.raises(RuntimeError, match="duplicated"):
        gate.evaluate_predictions(cases, duplicated)
    invalid = [dict(row) for row in predictions]
    invalid[0]["predicted_code"] = "I"
    with pytest.raises(RuntimeError, match="invalid"):
        gate.evaluate_predictions(cases, invalid)


def test_official_guards_run_before_preflight_or_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(gate, "DEFAULT_ATTEMPT", attempt)

    attempt.write_text("consumed", encoding="utf-8")
    monkeypatch.setattr(
        gate,
        "_clean_committed_head",
        lambda: pytest.fail("clean check must not run after consumed attempt"),
    )
    with pytest.raises(RuntimeError, match="already attempted"):
        gate.run_gate()

    attempt.unlink()

    def fail_clean() -> str:
        raise RuntimeError("clean-first")

    monkeypatch.setattr(gate, "_clean_committed_head", fail_clean)
    monkeypatch.setattr(
        gate,
        "prepare_source_only_gate",
        lambda: pytest.fail("preflight ran before clean commit check"),
    )
    with pytest.raises(RuntimeError, match="clean-first"):
        gate.run_gate()


def test_attempt_is_consumed_before_scorer_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cases: list[dict[str, Any]],
) -> None:
    output = tmp_path / "result.json"
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(gate, "DEFAULT_ATTEMPT", attempt)
    monkeypatch.setattr(gate, "_clean_committed_head", lambda: "a" * 40)
    monkeypatch.setattr(
        gate,
        "prepare_source_only_gate",
        lambda: {
            "preregistration": {
                "source_authority": {"frozen": True},
                "access_boundary": {
                    "source_files_read": [
                        prereg.D8_EVENTS.as_posix(),
                        prereg.D8_CARDS.as_posix(),
                    ]
                },
            },
            "runtime": {"validated": True},
            "processor": object(),
            "cases": cases,
            "prompt_capacity": {"validated": True},
        },
    )

    class FakeScorer:
        def __init__(self, _: Any) -> None:
            assert attempt.is_file()

        def score(self, prompt: str) -> dict[str, Any]:
            assert prompt.endswith("ANSWER=")
            return {
                "predicted_code": "A",
                "code_logits": {
                    code: float(-index)
                    for index, code in enumerate(
                        prereg.MEMORIZATION_CHALLENGE_CODES
                    )
                },
                "input_tokens": 1,
                "inference_seconds": 0.0,
            }

        def metrics(self) -> dict[str, Any]:
            return {"fake": True}

    payload = gate.run_gate(scorer_factory=FakeScorer)
    assert attempt.is_file()
    assert not output.exists()
    assert payload["attempt"]["sha256"] == gate.sha256_file(attempt)
    attempt_payload = json.loads(attempt.read_text(encoding="utf-8"))
    assert attempt_payload["runner_sha256"] == gate.sha256_file(
        Path(gate.__file__)
    )
    with pytest.raises(RuntimeError, match="already attempted"):
        gate.run_gate(scorer_factory=FakeScorer)


def test_runner_has_no_market_or_economic_import_surface() -> None:
    source_path = Path(gate.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert not imports.intersection(
        {
            "backtest",
            "database",
            "numpy",
            "pandas",
            "psycopg",
            "sqlalchemy",
        }
    )
    assert gate.DEFAULT_OUTPUT != gate.DEFAULT_ATTEMPT
    assert gate.MEMORIZATION_FAILURE_ACTION == (
        prereg.build_preregistration()["memorization_contract"][
            "failure_action"
        ]
    )
