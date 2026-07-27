from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

from training import preregister_psim_d8_rllm2_s2_chunked_source_feature as prereg
from training import run_psim_d8_rllm1_base_memorization_gate as base


def test_s1_terminal_failure_is_exact_and_no_market_was_opened() -> None:
    predecessor = prereg.validate_predecessor()
    failure = predecessor["failure"]
    assert failure["result_hash"] == prereg.S1_FAILURE_RESULT_HASH
    assert failure["decision"] == "reject"
    assert failure["resume_authorized"] is False
    assert failure["rerun_authorized"] is False
    assert failure["market_access_authorized"] is False
    assert failure["open_2020_train_outcomes_authorized"] is False
    assert failure["source_feature_seal_authorized"] is False
    assert failure["failure"]["exception_type"] == "OutOfMemoryError"
    assert failure["observations"]["completed_source_rows"] == 341
    assert failure["access_boundary"]["market_or_funding_paths_read"] == []
    assert failure["access_boundary"]["economic_metrics_computed"] == 0


def test_source_roster_and_frozen_cases_are_unchanged() -> None:
    rows = prereg.build_source_rows()
    roster = prereg.s1.source_roster_contract(rows)
    challenge = prereg.validate_frozen_cases(rows)
    assert roster["row_count"] == 1_461
    assert roster["source_row_roster_hash"] == (
        prereg.SOURCE_ROW_ROSTER_HASH
    )
    assert challenge["equivalence_case_count"] == 10
    assert len(
        {
            case["row_index"]
            for case in challenge["equivalence_cases"]
        }
    ) == 10
    assert challenge["capacity_case"]["row_index"] == 341
    assert challenge["capacity_case"]["policy_tokens"] == 29_727
    assert challenge["capacity_case"]["relation_tokens"] == 29_728


def test_equivalence_case_token_counts_match_pinned_tokenizer() -> None:
    from transformers import AutoProcessor

    rows = prereg.build_source_rows()
    processor = AutoProcessor.from_pretrained(
        base._model_snapshot(),
        local_files_only=True,
        trust_remote_code=False,
    )
    policy_counts: list[int] = []
    relation_counts: list[int] = []
    for row in rows:
        policy_counts.append(
            int(
                base._processor_inputs(
                    processor,
                    row["policy_prompt"],
                )["input_ids"].shape[-1]
            )
        )
        relation_counts.append(
            int(
                base._processor_inputs(
                    processor,
                    row["relation_teacher_prompt"],
                )["input_ids"].shape[-1]
            )
        )
    selected = prereg.select_equivalence_cases(
        rows,
        policy_token_counts=policy_counts,
        relation_token_counts=relation_counts,
    )
    assert selected == [dict(case) for case in prereg.EQUIVALENCE_CASES]
    for case in selected:
        row_index = int(case["row_index"])
        policy_tokens = policy_counts[row_index]
        relation_tokens = relation_counts[row_index]
        assert policy_tokens == case["policy_tokens"]
        assert relation_tokens == case["relation_tokens"]
        assert relation_tokens <= prereg.ONE_PASS_SAFE_MAXIMUM_TOKENS
    capacity_index = int(prereg.CAPACITY_CASE["row_index"])
    assert policy_counts[capacity_index] == (
        prereg.CAPACITY_CASE["policy_tokens"]
    )
    assert relation_counts[capacity_index] == (
        prereg.CAPACITY_CASE["relation_tokens"]
    )


def test_preregistration_is_canonical_and_only_changes_operator(
    tmp_path: Path,
) -> None:
    payload = prereg.build_preregistration()
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["candidate"]["id"] == "PSIM-D8-RLLM2-S2"
    assert payload["candidate"]["profitability_claim"] is False
    assert payload["sole_operational_delta"]["chunk_size_tokens"] == 512
    assert payload["sole_operational_delta"][
        "dynamic_chunk_selection_or_tuning"
    ] is False
    unchanged = payload["unchanged_scientific_contract"]
    assert unchanged["source_row_roster_hash"] == (
        prereg.SOURCE_ROW_ROSTER_HASH
    )
    assert unchanged["relation_mapping_or_prompt_change"] is False
    assert unchanged["model_or_quantization_change"] is False
    assert unchanged["source_resample_or_partial_reuse"] is False
    assert payload["pre_market_equivalence_gate"][
        "embedding_thresholds"
    ]["minimum_cosine_similarity"] == 0.99999
    assert payload["long_context_capacity_gate"][
        "maximum_peak_allocated_bytes"
    ] == 30 * 1024**3
    assert payload["execution_contract"][
        "s1_checkpoint_directory_read"
    ] is False
    boundary = payload["access_boundary"]
    assert boundary["s1_checkpoint_or_partial_model_outputs_read"] is False
    assert boundary["market_or_funding_paths_read"] == []
    assert boundary["train_2020_outcomes_opened"] is False
    assert boundary["test_outcomes_opened"] is False
    assert boundary["eval_outcomes_opened"] is False

    output = tmp_path / "preregistration.json"
    first = prereg.write_preregistration(output)
    second = prereg.write_preregistration(output)
    assert first == second == payload
    assert output.read_bytes() == prereg.canonical_json_bytes(payload)


def test_direct_script_entrypoint_imports_without_pythonpath() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(prereg.__file__)), "--help"],
        cwd=prereg.REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output" in completed.stdout


def test_preregistration_has_no_market_or_economic_import_surface() -> None:
    source_path = Path(prereg.__file__)
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
