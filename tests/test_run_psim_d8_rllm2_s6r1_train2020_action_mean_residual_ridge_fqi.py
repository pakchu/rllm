from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training import (
    preregister_psim_d8_rllm2_s6r1_action_mean_residual_ridge_fqi as prereg,
)
from training import (
    run_psim_d8_rllm2_s6r1_train2020_action_mean_residual_ridge_fqi as runner,
)


def test_validate_preregistration_is_exact_and_outcome_closed() -> None:
    payload = runner.validate_preregistration()

    assert payload["manifest_hash"] == runner.PREREGISTRATION_MANIFEST_HASH
    assert runner._sha256_file(prereg.DEFAULT_OUTPUT) == (
        runner.PREREGISTRATION_SHA256
    )
    assert payload["candidate"]["id"] == prereg.STAGE_ID
    assert payload["fitted_q_contract"]["primary_policy_id"] == (
        prereg.PRIMARY_POLICY_ID
    )
    assert payload["control_family"]["new_policy_family_ids"] == list(
        prereg.POLICY_FAMILY_IDS
    )
    boundary = payload["access_boundary"]
    assert boundary["raw_market_or_funding_paths_read"] == []
    assert boundary["2021_market_or_funding_paths_read"] == []
    assert boundary["2021_rewards_created"] == 0
    assert boundary["2021_economic_metrics_computed"] == 0


def test_attempt_is_self_hashed_and_precedes_every_parse() -> None:
    payload = runner._attempt_payload("a" * 40, "b" * 64)
    core = {
        key: value
        for key, value in payload.items()
        if key != "attempt_hash"
    }

    assert payload["attempt_hash"] == runner._canonical_hash(core)
    assert payload["authorization"] == {
        "parse_frozen_2020_transition_ledger": True,
        "read_raw_market_or_funding_payload": False,
        "open_2021_policy_specific_outcomes": False,
        "open_2022_or_later_outcomes": False,
    }
    assert payload["access_boundary_at_attempt"] == {
        "raw_market_or_funding_paths_read": [],
        "2020_transition_ledger_rows_parsed": 0,
        "2020_residual_reward_values_created": 0,
        "2020_economic_metrics_computed": 0,
        "2021_market_or_funding_paths_read": [],
        "2021_reward_rows_created": 0,
        "2021_economic_metrics_computed": 0,
        "2021_policy_specific_outcomes_opened": False,
        "2022_or_later_outcomes_opened": False,
        "model_loaded": False,
        "model_forwards_started": 0,
    }


def test_frozen_pca_and_source_hashes_validate_without_market_access() -> None:
    runner._verify_source_artifacts()
    bundle = runner.features.load_source_bundle(
        runner.repository_path(prereg.s4.s3.SOURCE_ROWS_PATH),
        runner.repository_path(prereg.s4.s3.EMBEDDINGS_PATH),
        runner.repository_path(prereg.s4.s3.RELATION_LOGITS_PATH),
        runner.repository_path(prereg.s4.s3.RELATION_ROWS_PATH),
    )
    train_indices = runner.features.year_indices(bundle.rows, 2020)
    pca = runner._load_frozen_pca(train_indices)

    assert len(train_indices) == 366
    assert pca.fit_row_count == 366
    assert pca.components.shape == (32, 2_560)
    assert pca.mean.shape == (2_560,)
    assert np.all(np.isfinite(pca.components))
    assert np.all(np.isfinite(pca.mean))


def test_deterministic_gzip_and_schedule_manifest_are_self_bound() -> None:
    frame = pd.DataFrame(
        [
            {
                "policy_id": prereg.PRIMARY_POLICY_ID,
                "sequence_id": "x",
                "entry_time": "2021-01-01T12:05:00Z",
                "target": "TARGET_FLAT",
            }
        ]
    )
    first, decoded = runner._deterministic_csv_gzip_bytes(frame)
    second, second_decoded = runner._deterministic_csv_gzip_bytes(frame)
    assert first == second
    assert decoded == second_decoded
    assert hashlib.sha256(gzip.decompress(first)).hexdigest() == decoded
    artifact = runner._artifact_record(
        "data/example.csv.gz",
        first,
        extra={"decoded_sha256": decoded, "rows": 1},
    )
    readiness = {
        "passed": False,
        "gates": {"minimum_short_share": False},
    }
    manifest = runner._schedule_manifest(
        attempt_hash="a" * 64,
        residual_ledger_artifact=artifact,
        base_artifact=artifact,
        delayed_artifact=artifact,
        readiness=readiness,
    )
    core = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_hash"
    }
    assert manifest["manifest_hash"] == runner._canonical_hash(core)
    assert manifest["decision"] == "reject"
    assert manifest["outcome_boundary"] == {
        "raw_market_or_funding_payload_opened": False,
        "2021_market_or_funding_payload_opened": False,
        "2021_reward_rows_created": 0,
        "2021_economic_metrics_computed": 0,
        "2021_policy_specific_outcomes_opened": False,
        "2022_or_later_outcomes_opened": False,
    }
    assert manifest["global_2021_pristine_claim"] is False


def test_write_once_accepts_identity_and_rejects_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    runner._write_once(output, b"first")
    runner._write_once(output, b"first")

    assert output.read_bytes() == b"first"
    with pytest.raises(RuntimeError, match="write-once artifact drift"):
        runner._write_once(output, b"second")


def test_validate_preregistration_rejects_tampered_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tampered = json.loads(
        prereg.repository_path(prereg.DEFAULT_OUTPUT).read_text(
            encoding="utf-8"
        )
    )
    tampered["candidate"]["profitability_claim"] = True
    core = {
        key: value
        for key, value in tampered.items()
        if key != "manifest_hash"
    }
    tampered["manifest_hash"] = runner._canonical_hash(core)
    output = tmp_path / "prereg.json"
    output.write_bytes(runner._canonical_bytes(tampered, pretty=True))
    monkeypatch.setattr(prereg, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(
        runner,
        "PREREGISTRATION_SHA256",
        hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runner,
        "PREREGISTRATION_MANIFEST_HASH",
        tampered["manifest_hash"],
    )

    with pytest.raises(RuntimeError, match="contract changed"):
        runner.validate_preregistration()


def test_frozen_input_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake = tmp_path / "pca.npz"
    fake.write_bytes(b"tampered")
    monkeypatch.setattr(prereg.s4, "PCA_PATH", fake)

    with pytest.raises(RuntimeError, match="frozen input changed"):
        runner._verify_source_artifacts()



def test_execute_records_attempt_before_transition_ledger_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    attempt = {
        "execution_commit": "a" * 40,
        "runner_sha256": "b" * 64,
        "attempt_hash": "c" * 64,
    }
    rows = [
        {
            "decision_at": "2020-01-01T12:05:00Z",
            "row_hash": f"{index:064x}",
        }
        for index in range(731)
    ]
    monkeypatch.setattr(runner, "validate_preregistration", dict)
    monkeypatch.setattr(
        runner,
        "assert_committed_clean_runner",
        lambda: ("a" * 40, "b" * 64),
    )

    def write_attempt(*_args: str) -> dict[str, str]:
        events.append("attempt")
        return attempt

    monkeypatch.setattr(runner, "_write_or_validate_attempt", write_attempt)
    monkeypatch.setattr(runner, "_verify_existing_result", lambda _arg: None)
    monkeypatch.setattr(
        runner,
        "_verify_source_artifacts",
        lambda: events.append("source_hashes"),
    )
    monkeypatch.setattr(
        runner.features,
        "load_source_bundle",
        lambda *_args: SimpleNamespace(rows=rows),
    )
    monkeypatch.setattr(
        runner.features,
        "year_indices",
        lambda _rows, year: (
            np.arange(366)
            if year == 2020
            else np.arange(366, 731)
        ),
    )

    def reject_ledger_parse(*_args: object, **_kwargs: object) -> None:
        events.append("transition_ledger_parse")
        raise RuntimeError("sentinel transition parse")

    monkeypatch.setattr(runner.pd, "read_csv", reject_ledger_parse)

    with pytest.raises(RuntimeError, match="sentinel transition parse"):
        runner.execute()

    assert events == [
        "attempt",
        "source_hashes",
        "transition_ledger_parse",
    ]


def test_existing_result_cannot_authorize_failed_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    residual_path = tmp_path / "residual.csv.gz"
    base_path = tmp_path / "base.csv.gz"
    delayed_path = tmp_path / "delayed.csv.gz"
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "result.json"
    for path, raw in (
        (residual_path, b"residual"),
        (base_path, b"base"),
        (delayed_path, b"delayed"),
    ):
        path.write_bytes(raw)
    monkeypatch.setattr(prereg, "RESIDUAL_LEDGER_PATH", residual_path)
    monkeypatch.setattr(prereg, "SCHEDULE_PATH", base_path)
    monkeypatch.setattr(prereg, "DELAYED_SCHEDULE_PATH", delayed_path)
    monkeypatch.setattr(prereg, "SCHEDULE_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(prereg, "RESULT_PATH", result_path)
    registration = {
        "pre2021_schedule_readiness_gate": {
            "pass_action": "PASS",
            "failure_action": "REJECT",
        },
        "access_boundary": {"source_files_read": []},
    }
    monkeypatch.setattr(
        prereg,
        "build_preregistration",
        lambda: registration,
    )
    readiness = {"passed": False, "gates": {"required_gate": False}}
    residual_artifact = runner._artifact_record(
        residual_path,
        residual_path.read_bytes(),
    )
    base_artifact = runner._artifact_record(
        base_path,
        base_path.read_bytes(),
    )
    delayed_artifact = runner._artifact_record(
        delayed_path,
        delayed_path.read_bytes(),
    )
    manifest_core = {
        "protocol_version": runner.SCHEDULE_PROTOCOL_VERSION,
        "attempt_hash": "c" * 64,
        "decision": "pass",
        "schedule_readiness": readiness,
        "residual_reward_ledger_2020": residual_artifact,
        "base_schedules_2021": base_artifact,
        "delayed_primary_schedule_2021": delayed_artifact,
    }
    manifest = {
        **manifest_core,
        "manifest_hash": runner._canonical_hash(manifest_core),
    }
    manifest_path.write_bytes(runner._canonical_bytes(manifest, pretty=True))
    manifest_artifact = runner._artifact_record(
        manifest_path,
        manifest_path.read_bytes(),
    )
    attempt = {
        "execution_commit": "a" * 40,
        "runner_sha256": "b" * 64,
        "attempt_hash": "c" * 64,
    }
    result_core = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "decision": "pass",
        "execution_commit": attempt["execution_commit"],
        "runner_sha256": attempt["runner_sha256"],
        "attempt_hash": attempt["attempt_hash"],
        "preregistration_manifest_hash": (
            runner.PREREGISTRATION_MANIFEST_HASH
        ),
        "terminal_action": "PASS",
        "authorize_separate_2021_transfer_preregistration": True,
        "authorize_2022_or_later_outcomes": False,
        "qlora_authorized": False,
        "schedule_readiness": readiness,
        "artifacts": {
            "residual_reward_ledger_2020": residual_artifact,
            "base_schedules_2021": base_artifact,
            "delayed_primary_schedule_2021": delayed_artifact,
            "schedule_gate_manifest_2021": manifest_artifact,
        },
        "access_boundary": runner._expected_access_boundary(registration),
    }
    result = {
        **result_core,
        "result_hash": runner._canonical_hash(result_core),
    }
    result_path.write_bytes(runner._canonical_bytes(result, pretty=True))

    with pytest.raises(RuntimeError, match="existing result changed"):
        runner._verify_existing_result(attempt)


def test_full_execute_smoke_reaches_terminal_result_with_synthetic_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dates_2020 = pd.date_range(
        "2020-01-01T12:05:00Z",
        periods=366,
        freq="D",
    )
    dates_2021 = pd.date_range(
        "2021-01-01T12:05:00Z",
        periods=365,
        freq="D",
    )
    dates = [*dates_2020, *dates_2021]
    rows = [
        {
            "decision_at": timestamp.isoformat(),
            "row_hash": f"{index:064x}",
            "split_year": timestamp.year,
        }
        for index, timestamp in enumerate(dates)
    ]
    action_target = {
        "TARGET_FLAT": 0.0,
        "TARGET_SHORT": -0.5,
        "TARGET_LONG": 0.5,
    }
    ledger_rows: list[dict[str, object]] = []
    for state, timestamp in enumerate(dates_2020):
        positions = (
            ("POSITION_FLAT",)
            if state == 0
            else runner.residual.fqi.POSITION_NAMES
        )
        state_signal = float(np.sin(state / 13.0))
        for position_index, position in enumerate(positions):
            for action_index, action in enumerate(
                runner.residual.fqi.ACTION_NAMES
            ):
                directional = (action_index - 1) * state_signal * 0.002
                reward = (
                    directional
                    + action_index * 0.001
                    + position_index * 0.0002
                )
                ledger_rows.append(
                    {
                        "sequence_id": f"{state:064x}",
                        "entry_time": timestamp.isoformat(),
                        "current_position": position,
                        "action_name": action,
                        "action_target": action_target[action],
                        "executed_target": action_target[action],
                        "reachable": True,
                        "terminal": state == 365,
                        "reward": reward,
                        "multiplier": 1.0 + reward,
                        "held_path_downside_fraction": 0.0,
                        "changed_notional_fraction": 0.0,
                        "entry_cost": 0.0,
                        "terminal_cost": 0.0,
                        "funding_cash": 0.0,
                        "bars_held": 288,
                    }
                )
    ledger = pd.DataFrame(
        ledger_rows,
        columns=runner.residual.EXPECTED_LEDGER_COLUMNS,
    )
    assert len(ledger) == 3_288

    row_axis = np.arange(731, dtype=np.float64)
    semantic = np.column_stack(
        [
            function(row_axis / period)
            for period in range(3, 19)
            for function in (np.sin, np.cos)
        ]
    ).astype(np.float32)
    feature_family = {
        "semantic": semantic,
        "current_position_only": np.zeros((731, 0), dtype=np.float32),
        "masked_semantic_embedding": np.zeros((731, 32), dtype=np.float32),
        "metadata_frontmatter_only": semantic[:, :11],
    }

    def fixed_schedule(policy_id: str, target: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "policy_id": policy_id,
                    "sequence_id": row["row_hash"],
                    "entry_time": row["decision_at"],
                    "target": target,
                }
                for row in rows[366:]
            ],
            columns=runner.residual.SCHEDULE_COLUMNS,
        )

    s4_schedules = pd.concat(
        [
            fixed_schedule("always_flat", "TARGET_FLAT"),
            fixed_schedule("always_long", "TARGET_LONG"),
            fixed_schedule("always_short", "TARGET_SHORT"),
            fixed_schedule(
                "previous_target_persistence",
                "TARGET_FLAT",
            ),
        ],
        ignore_index=True,
    )
    s5_schedules = fixed_schedule(
        runner.residual.s5_core.prereg.PRIMARY_POLICY_ID,
        "TARGET_FLAT",
    )
    output_paths = {
        "ATTEMPT_PATH": tmp_path / "attempt.json",
        "RESULT_PATH": tmp_path / "result.json",
        "RESIDUAL_LEDGER_PATH": tmp_path / "residual.csv.gz",
        "SCHEDULE_PATH": tmp_path / "schedules.csv.gz",
        "DELAYED_SCHEDULE_PATH": tmp_path / "delayed.csv.gz",
        "SCHEDULE_MANIFEST_PATH": tmp_path / "manifest.json",
    }
    for name, path in output_paths.items():
        monkeypatch.setattr(prereg, name, path)
    registration = {
        "pre2021_schedule_readiness_gate": {
            "pass_action": "PASS",
            "failure_action": "REJECT",
        },
        "access_boundary": {"source_files_read": []},
    }
    monkeypatch.setattr(
        runner,
        "validate_preregistration",
        lambda: registration,
    )
    monkeypatch.setattr(
        prereg,
        "build_preregistration",
        lambda: registration,
    )
    monkeypatch.setattr(
        runner,
        "assert_committed_clean_runner",
        lambda: ("a" * 40, "b" * 64),
    )
    monkeypatch.setattr(runner, "_verify_source_artifacts", lambda: None)
    monkeypatch.setattr(
        runner.features,
        "load_source_bundle",
        lambda *_args: SimpleNamespace(rows=rows),
    )
    monkeypatch.setattr(runner, "_load_frozen_pca", lambda _indices: None)
    monkeypatch.setattr(
        runner.features,
        "build_feature_family",
        lambda _bundle, _pca: feature_family,
    )

    def read_frame(path: str | Path, *_args: object, **_kwargs: object):
        target = Path(path)
        if target == runner.repository_path(
            prereg.s4.TRANSITION_LEDGER_PATH
        ):
            return ledger.copy()
        if target == runner.repository_path(prereg.s4.SCHEDULE_PATH):
            return s4_schedules.copy()
        if target == runner.repository_path(prereg.s5.SCHEDULE_PATH):
            return s5_schedules.copy()
        raise AssertionError(f"unexpected synthetic read: {target}")

    monkeypatch.setattr(runner.pd, "read_csv", read_frame)
    reconstruction_calls = 0
    original_reconstruct = (
        runner.residual.s5_core.reconstruct_reward_tensor
    )

    def reconstruct(*args: object, **kwargs: object):
        nonlocal reconstruction_calls
        reconstruction_calls += 1
        return original_reconstruct(*args, **kwargs)

    monkeypatch.setattr(
        runner.residual.s5_core,
        "reconstruct_reward_tensor",
        reconstruct,
    )

    result = runner.execute()

    assert reconstruction_calls == 1
    assert result["decision"] in {"pass", "reject"}
    assert result["terminal_action"] in {"PASS", "REJECT"}
    assert result["fit"]["fit_source_rows"] == 366
    assert result["fit"]["original_reward_values"] == 3_288
    assert result["fit"]["residual_reward_values"] == 3_288
    assert result["fit"]["fitted_q_count"] == 7
    assert result["access_boundary"][
        "2020_transition_ledger_rows_parsed"
    ] == 3_288
    assert result["access_boundary"]["2021_reward_rows_created"] == 0
    assert result["access_boundary"]["2021_economic_metrics_computed"] == 0
    assert result["access_boundary"][
        "2021_policy_specific_outcomes_opened"
    ] is False
    assert result["access_boundary"] == runner._expected_access_boundary(
        registration
    )
    for path in output_paths.values():
        assert path.is_file()
    resumed = runner.execute()
    assert resumed == result
    result_path = output_paths["RESULT_PATH"]
    attempt_payload = json.loads(
        output_paths["ATTEMPT_PATH"].read_text(encoding="utf-8")
    )
    original_result_raw = result_path.read_bytes()

    def drop_market_rows(payload: dict) -> None:
        payload["access_boundary"].pop("2021_market_rows_parsed")

    def drop_funding_rows(payload: dict) -> None:
        payload["access_boundary"].pop("2021_funding_rows_parsed")

    def add_market_rows(payload: dict) -> None:
        payload["access_boundary"]["2021_market_rows_parsed"] = 1

    def enable_model(payload: dict) -> None:
        payload["access_boundary"]["model_loaded"] = True

    for mutate in (
        drop_market_rows,
        drop_funding_rows,
        add_market_rows,
        enable_model,
    ):
        tampered = json.loads(original_result_raw)
        mutate(tampered)
        tampered_core = {
            key: value
            for key, value in tampered.items()
            if key != "result_hash"
        }
        tampered["result_hash"] = runner._canonical_hash(tampered_core)
        result_path.write_bytes(
            runner._canonical_bytes(tampered, pretty=True)
        )
        with pytest.raises(
            RuntimeError,
            match="existing schedule gate changed",
        ):
            runner._verify_existing_result(attempt_payload)
    result_path.write_bytes(original_result_raw)
    assert runner._verify_existing_result(attempt_payload) == result


def test_expected_artifact_paths_are_fixed_to_preregistration() -> None:
    assert runner._expected_artifact_paths() == {
        "residual_reward_ledger_2020": prereg.RESIDUAL_LEDGER_PATH,
        "base_schedules_2021": prereg.SCHEDULE_PATH,
        "delayed_primary_schedule_2021": prereg.DELAYED_SCHEDULE_PATH,
        "schedule_gate_manifest_2021": prereg.SCHEDULE_MANIFEST_PATH,
    }
