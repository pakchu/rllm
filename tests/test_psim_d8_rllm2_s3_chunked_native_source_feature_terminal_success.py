from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from training import (
    run_psim_d8_rllm2_s3_chunked_native_source_feature_seal as runner,
)

ATTEMPT = runner.repository_path(runner.DEFAULT_ATTEMPT)
REPEATABILITY = runner.repository_path(
    runner.DEFAULT_REPEATABILITY_RESULT
)
RESULT = runner.repository_path(runner.DEFAULT_OUTPUT)
LOG = (
    runner.REPO_ROOT
    / "results/psim_d8_rllm2_s3_chunked_native_source_feature_"
    "seal_success_2026-07-27.log.gz"
)
EXECUTION_COMMIT = "dfef2ee175f6c055da71dbc797d611db4355e921"

EXPECTED_FILE_HASHES = {
    "attempt": (
        "99046f2ef3d816c279136e61e7c60394b39b5adc5cc790735d38025585fd5bcb"
    ),
    "repeatability": (
        "075f19c855d49146cb38c1bb40bd6599389736f7fac9e1a998c0eac320e6db50"
    ),
    "result": (
        "0278b303005ae50510004344eb8889bc464c4b61612e0e162e091e7decd8a976"
    ),
    "source_rows": (
        "2897845bf55506bf877c2015a670707287d3d98c3718361e24ad31504b98939c"
    ),
    "embeddings": (
        "509d7922561cdec0582165e5976ac9bdcc72dfe26e68c6ce46fe7ba9fab4f2a0"
    ),
    "relation_logits": (
        "384725cd8f2f451a8dedca76625333d66517937b744e09ea5d71cd938c500b62"
    ),
    "relation_rows": (
        "4ea96c6af2da8f574eed88fcbfa7fe6f20b5d6131e89eb41b734a0e498cfef1a"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _self_hashed(
    path: Path,
    *,
    field: str,
    expected: str,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != field
    }
    assert payload[field] == _canonical_hash(core)
    assert payload[field] == expected
    return payload


def test_attempt_gate_result_and_runner_are_exact() -> None:
    assert _sha256(ATTEMPT) == EXPECTED_FILE_HASHES["attempt"]
    assert _sha256(REPEATABILITY) == EXPECTED_FILE_HASHES["repeatability"]
    assert _sha256(RESULT) == EXPECTED_FILE_HASHES["result"]
    assert _sha256(LOG) == (
        "8f45d4c08b0bf8747300ed4990893fba6d1dd277e474e6c42d16522acb1f36be"
    )
    assert hashlib.sha256(gzip.decompress(LOG.read_bytes())).hexdigest() == (
        "8992814559f716f898894e8ed7486a6070a9a6779e701b74ddbd9cfdb829af35"
    )
    attempt = _self_hashed(
        ATTEMPT,
        field="attempt_hash",
        expected=(
            "9bc78f6774246a41c88c57ebbe855cb7c4f7577e9fff814b080403400497bf4f"
        ),
    )
    _self_hashed(
        REPEATABILITY,
        field="result_hash",
        expected=(
            "1fce8b64b68dff2b99d0d940a2fd2d02379d21c24e37ac79b0676561c190bceb"
        ),
    )
    _self_hashed(
        RESULT,
        field="result_hash",
        expected=(
            "9bf7121d4feec0f7000626f8493e05effc3dfff8f9857b7ac5f05f6beadc3af9"
        ),
    )
    executed_runner = subprocess.run(
        [
            "git",
            "show",
            (
                f"{EXECUTION_COMMIT}:training/"
                "run_psim_d8_rllm2_s3_chunked_native_source_feature_seal.py"
            ),
        ],
        cwd=runner.REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert attempt["execution_commit"] == EXECUTION_COMMIT
    assert hashlib.sha256(executed_runner).hexdigest() == (
        "ab12b42f2f9f6cbad02ae85fd61511b333736bdae6028462af8052fe77ec72a9"
    )
    assert attempt["runner_sha256"] == hashlib.sha256(
        executed_runner
    ).hexdigest()


def test_two_load_repeatability_and_capacity_gate_passed_exactly() -> None:
    gate = json.loads(REPEATABILITY.read_text(encoding="utf-8"))
    assert gate["decision"] == "pass"
    assert gate["terminal_action"] == runner.REPEATABILITY_PASS_ACTION
    assert gate["repeatability_case_roster_hash"] == (
        "d960f57da0d5e8b37004643a20b39caa5a569278bd0385238e2845ce41585bcf"
    )
    assert gate["repeat_model_load_count"] == 2
    assert len(gate["load_results"]) == 2
    comparison = gate["cross_load_comparison"]
    assert comparison == {
        "all_peak_allocated_within_limit": True,
        "capacity_embedding_hashes_identical": True,
        "capacity_relation_codes_identical": True,
        "capacity_relation_logit_hashes_identical": True,
        "embedding_hashes_identical": True,
        "model_loads_compared": 2,
        "pass": True,
        "relation_codes_identical": True,
        "relation_logit_hashes_identical": True,
        "repeatability_case_count": 10,
    }
    peaks = []
    for index, load in enumerate(gate["load_results"]):
        assert load["load_index"] == index
        assert load["model_loaded_independently"] is True
        assert load["scorer_closed_before_next_load"] is True
        assert load["pass"] is True
        capacity = load["capacity_result"]
        assert capacity["row_index"] == 341
        assert capacity["policy_tokens"] == 29_727
        assert capacity["relation_tokens"] == 29_728
        assert capacity["embedding_float32_bytes_sha256"] == (
            "cf970683f59a97fea77304d37a6ee42cc62877e535e7670443e44857d8be6878"
        )
        assert capacity[
            "relation_logits_canonical_float32_bytes_sha256"
        ] == (
            "b487f1a161bb61edf9cd14bf9dcdeb2dd671a275f9f389c1e750328c9bd960e6"
        )
        assert capacity["predicted_relation_code"] == "A"
        assert capacity["outputs_reused_by_full_extraction"] is False
        assert capacity["pass"] is True
        peaks.append(capacity["peak_allocated_bytes"])
    assert peaks == [10_449_585_664, 10_451_617_280]


def test_final_artifacts_are_exact_complete_and_finite() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    for name, expected in EXPECTED_FILE_HASHES.items():
        if name in {"attempt", "repeatability", "result"}:
            continue
        path = runner.repository_path(
            Path(result["artifacts"][name]["path"])
        )
        assert _sha256(path) == expected
        assert result["artifacts"][name]["sha256"] == expected
    with np.load(
        runner.repository_path(runner.DEFAULT_EMBEDDINGS),
        allow_pickle=False,
    ) as embeddings:
        assert embeddings["embedding"].shape == (1_461, 2_560)
        assert embeddings["embedding"].dtype == np.float32
        assert np.all(np.isfinite(embeddings["embedding"]))
        assert np.array_equal(
            embeddings["row_index"],
            np.arange(1_461, dtype=np.int32),
        )
    with np.load(
        runner.repository_path(runner.DEFAULT_RELATION_LOGITS),
        allow_pickle=False,
    ) as relation:
        assert relation["logits"].shape == (1_461, 6)
        assert relation["logits"].dtype == np.float32
        assert int(np.sum(relation["forwarded"])) == 1_344
        forwarded = relation["forwarded"].astype(bool)
        assert np.all(np.isfinite(relation["logits"][forwarded]))
    source_lines = gzip.decompress(
        runner.repository_path(runner.DEFAULT_SOURCE_ROWS).read_bytes()
    ).splitlines()
    relation_lines = gzip.decompress(
        runner.repository_path(runner.DEFAULT_RELATION_ROWS).read_bytes()
    ).splitlines()
    assert len(source_lines) == len(relation_lines) == 1_461


def test_success_is_third_load_only_and_publish_is_clean() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["decision"] == "pass"
    assert result["terminal_action"] == runner.PASS_ACTION
    assert result["model_load_evidence"] == {
        "fresh_from_gate_models": True,
        "full_extraction_model_load_ordinal": 3,
        "gate_model_loads": 2,
    }
    assert result["checkpoint_evidence"] == {
        "resumed": False,
        "resumed_from_completed_rows": 0,
        "shard_count": 1_461,
        "shard_size": 1,
        "terminal_chain_hash": (
            "3b2296e994377b6112a27f97dee2d4c7974c4e9353c65b8448333cb739d807df"
        ),
    }
    assert result["model_metrics"]["chunk_forwards_started"] == 12_772
    assert result["model_metrics"]["embedding_forwards_started"] == 1_461
    assert result["model_metrics"]["relation_forwards_started"] == 1_344
    assert result["model_metrics"]["peak_allocated_bytes"] == 10_486_249_472
    assert not runner.repository_path(
        runner.DEFAULT_CHECKPOINT_DIRECTORY
    ).exists()
    assert not runner._publish_journal_path(RESULT).exists()
    assert not runner._publish_result_staging_path(RESULT).exists()


def test_success_opened_no_market_and_authorizes_only_2020_train() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["source_feature_seal_authorized"] is True
    assert result["open_2020_train_outcomes_authorized"] is True
    assert result["open_2021_or_later_outcomes_authorized"] is False
    assert result["market_access_authorized_during_this_stage"] is False
    boundary = result["access_boundary"]
    assert boundary["s1_or_s2_checkpoint_or_model_outputs_read"] is False
    assert boundary["market_or_funding_paths_read"] == []
    assert boundary["market_rows_parsed"] == 0
    assert boundary["funding_rows_parsed"] == 0
    assert boundary["market_or_funding_payload_bytes_hashed"] is False
    assert boundary["rewards_created"] == 0
    assert boundary["economic_metrics_computed"] == 0
    assert boundary["train_2020_outcomes_opened"] is False
    assert boundary["test_outcomes_opened"] is False
    assert boundary["eval_outcomes_opened"] is False
