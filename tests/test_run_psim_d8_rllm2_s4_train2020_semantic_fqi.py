from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import preregister_psim_d8_rllm2_s4_semantic_fqi as prereg
from training import psim_semantic_fqi_policies as fqi
from training import run_psim_d8_rllm2_s4_train2020_semantic_fqi as runner


def _rows(count: int, *, year: int = 2020) -> list[dict[str, object]]:
    start = pd.Timestamp(f"{year}-01-01T12:05:00Z")
    return [
        {
            "row_hash": f"{index + year:064x}",
            "source_payload_sha256": f"{index + 10_000:064x}",
            "decision_at": (
                start + pd.Timedelta(days=index)
            ).isoformat().replace("+00:00", "Z"),
        }
        for index in range(count)
    ]


def _training_arrays(
    count: int = 12,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    semantic = np.linspace(-1.0, 1.0, count).reshape(-1, 1)
    feature_family = {
        "semantic": semantic,
        "metadata_frontmatter_only": np.c_[semantic, semantic**2],
        "path_section_diff_size_only": np.abs(semantic),
        "cadence_revision_topology_only": np.c_[semantic, semantic**3],
        "shuffled_eip_bip_daily_relation": semantic[::-1],
        "shuffled_old_new_pairing": semantic[::-1],
        "future_status_scrub": np.zeros((count, 2)),
        "ethereum_only": np.maximum(semantic, 0.0),
        "bitcoin_only": np.maximum(-semantic, 0.0),
        "current_position_only": np.zeros((count, 0)),
        "masked_semantic_embedding": np.zeros_like(semantic),
    }
    rewards = np.zeros((count, 3, 3), dtype=np.float64)
    rewards[:, :, 1] = semantic[:, 0, None]
    rewards[:, :, 2] = -semantic[:, 0, None]
    reachable = np.ones((count, 3), dtype=bool)
    reachable[0] = False
    reachable[0, fqi.POSITION_INDEX["POSITION_FLAT"]] = True
    rewards[~reachable] = np.nan
    terminal = np.zeros(count, dtype=bool)
    terminal[-1] = True
    return feature_family, rewards, terminal, reachable


def test_preregistration_binding_is_exact() -> None:
    payload = runner.validate_preregistration()

    assert payload["manifest_hash"] == runner.PREREGISTRATION_MANIFEST_HASH
    assert payload["artifact_contract"]["attempt"] == (
        prereg.ATTEMPT_PATH.as_posix()
    )
    assert payload["control_family"]["policy_family_ids"] == list(
        prereg.POLICY_FAMILY_IDS
    )


def test_attempt_is_written_before_any_outcome_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt_path = tmp_path / "attempt.json"
    monkeypatch.setattr(prereg, "ATTEMPT_PATH", attempt_path)

    payload = runner._write_or_validate_attempt("a" * 40, "b" * 64)

    assert json.loads(attempt_path.read_text()) == payload
    assert payload["access_boundary_at_attempt"] == {
        "market_or_funding_paths_read": [],
        "market_or_funding_payload_bytes_hashed": False,
        "market_rows_parsed": 0,
        "funding_rows_parsed": 0,
        "rewards_created": 0,
        "economic_metrics_computed": 0,
        "2020_outcomes_opened": False,
        "2021_or_later_outcomes_opened": False,
    }


def test_fit_family_roster_is_exact_with_small_extra_trees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_family, rewards, terminal, reachable = _training_arrays()
    rows = _rows(len(rewards))
    monkeypatch.setitem(fqi.EXTRA_TREES_KWARGS, "n_estimators", 4)
    monkeypatch.setattr(fqi, "BELLMAN_ITERATIONS", 2)

    family = runner._fit_policy_family(
        feature_family,
        np.arange(len(rewards)),
        rewards,
        terminal,
        reachable,
        rows,
    )

    assert family.fitted_q_count == 18
    assert family.ridge_fitted_q_count == 14
    assert family.extra_trees_fitted_q_count == 4
    assert tuple(
        policy_id
        for policy_id in prereg.POLICY_FAMILY_IDS
        if policy_id != "exact_redacted_payload_memory"
    ) == tuple(family.policies)


def test_base_and_delayed_schedule_seal_use_no_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 12
    feature_family, rewards, terminal, reachable = _training_arrays(count)
    rows = _rows(count, year=2021)
    monkeypatch.setitem(fqi.EXTRA_TREES_KWARGS, "n_estimators", 4)
    monkeypatch.setattr(fqi, "BELLMAN_ITERATIONS", 2)
    family = runner._fit_policy_family(
        feature_family,
        np.arange(count),
        rewards,
        terminal,
        reachable,
        rows,
    )

    base = runner.build_schedule_family(
        family,
        feature_family,
        np.arange(count),
        rows,
    )
    delayed = runner.build_delayed_primary_schedules(
        base,
        stage_end=pd.Timestamp("2022-01-01T00:00:00Z"),
    )

    assert len(base) == count * len(prereg.POLICY_FAMILY_IDS)
    assert tuple(dict.fromkeys(base["policy_id"])) == prereg.POLICY_FAMILY_IDS
    assert len(delayed) == count * len(prereg.PRIMARY_POLICY_IDS)
    assert tuple(
        dict.fromkeys(delayed["policy_id"])
    ) == prereg.PRIMARY_POLICY_IDS
    base_times = pd.to_datetime(
        runner._policy_schedule(base, prereg.PRIMARY_POLICY_IDS[0])[
            "entry_time"
        ],
        utc=True,
    )
    delay_times = pd.to_datetime(
        runner._policy_schedule(delayed, prereg.PRIMARY_POLICY_IDS[0])[
            "entry_time"
        ],
        utc=True,
    )
    np.testing.assert_array_equal(
        delay_times.to_numpy(),
        (base_times + pd.Timedelta(minutes=5)).to_numpy(),
    )


def test_schedule_manifest_keeps_2021_outcomes_closed() -> None:
    artifact = {
        "path": "x",
        "sha256": "a" * 64,
        "bytes": 1,
    }
    manifest = runner._schedule_manifest(
        attempt_hash="b" * 64,
        base_artifact=artifact,
        delayed_artifact=artifact,
        pca_artifact=artifact,
        ledger_artifact=artifact,
    )

    assert manifest["outcome_boundary"] == {
        "2020_train_outcomes_used_for_fit": True,
        "2021_market_or_funding_payload_opened": False,
        "2021_reward_rows_created": 0,
        "2021_economic_metrics_computed": 0,
        "2022_or_2023_outcomes_opened": False,
    }
    core = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_hash"
    }
    assert manifest["manifest_hash"] == runner._canonical_hash(core)


def test_existing_result_validation_never_reopens_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(prereg, "RESULT_PATH", result_path)
    paths = {
        "transition_ledger_2020": tmp_path / "ledger.csv.gz",
        "pca32_2020": tmp_path / "pca.npz",
        "base_schedules_2021": tmp_path / "base.csv.gz",
        "delayed_primary_schedules_2021": tmp_path / "delay.csv.gz",
        "schedule_manifest_2021": tmp_path / "manifest.json",
    }
    monkeypatch.setattr(
        prereg,
        "TRANSITION_LEDGER_PATH",
        paths["transition_ledger_2020"],
    )
    monkeypatch.setattr(prereg, "PCA_PATH", paths["pca32_2020"])
    monkeypatch.setattr(
        prereg,
        "SCHEDULE_PATH",
        paths["base_schedules_2021"],
    )
    monkeypatch.setattr(
        prereg,
        "DELAYED_SCHEDULE_PATH",
        paths["delayed_primary_schedules_2021"],
    )
    monkeypatch.setattr(
        prereg,
        "SCHEDULE_MANIFEST_PATH",
        paths["schedule_manifest_2021"],
    )
    attempt = {
        "attempt_hash": "c" * 64,
        "execution_commit": "d" * 40,
        "runner_sha256": "e" * 64,
    }
    artifacts: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        if name == "schedule_manifest_2021":
            continue
        raw = f"sealed:{name}".encode()
        path.write_bytes(raw)
        artifacts[name] = {
            "path": path.as_posix(),
            "sha256": runner._sha256_file(path),
            "bytes": len(raw),
        }
    manifest = runner._schedule_manifest(
        attempt_hash=attempt["attempt_hash"],
        base_artifact=artifacts["base_schedules_2021"],
        delayed_artifact=artifacts["delayed_primary_schedules_2021"],
        pca_artifact=artifacts["pca32_2020"],
        ledger_artifact=artifacts["transition_ledger_2020"],
    )
    manifest_raw = runner._canonical_bytes(manifest, pretty=True)
    paths["schedule_manifest_2021"].write_bytes(manifest_raw)
    artifacts["schedule_manifest_2021"] = {
        "path": paths["schedule_manifest_2021"].as_posix(),
        "sha256": runner._sha256_file(
            paths["schedule_manifest_2021"]
        ),
        "bytes": len(manifest_raw),
        "manifest_hash": manifest["manifest_hash"],
    }
    core = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "attempt_hash": attempt["attempt_hash"],
        "execution_commit": attempt["execution_commit"],
        "runner_sha256": attempt["runner_sha256"],
        "preregistration_manifest_hash": (
            runner.PREREGISTRATION_MANIFEST_HASH
        ),
        "decision": "pass",
        "open_2021_outcomes_authorized": True,
        "open_2022_or_later_outcomes_authorized": False,
        "qlora_authorized": False,
        "terminal_action": (
            "ACCEPT_PSIM_D8_RLLM2_S4_SEALED_2021_SCHEDULES_"
            "AUTHORIZE_2021_EVALUATION_ONLY"
        ),
        "target_schedule_seal": {
            "target_stage": "2021",
            "manifest_hash": manifest["manifest_hash"],
        },
        "access_boundary": {
            "2021_market_or_funding_paths_read": [],
            "2021_reward_rows_created": 0,
            "2021_economic_metrics_computed": 0,
            "2021_outcomes_opened": False,
            "2022_or_2023_outcomes_opened": False,
        },
        "artifacts": artifacts,
    }
    result = {**core, "result_hash": runner._canonical_hash(core)}
    result_path.write_bytes(runner._canonical_bytes(result, pretty=True))

    loaded = runner._verify_existing_result(attempt)

    assert loaded == result


def test_existing_result_rejects_incomplete_artifact_roster(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(prereg, "RESULT_PATH", result_path)
    attempt = {
        "attempt_hash": "c" * 64,
        "execution_commit": "d" * 40,
        "runner_sha256": "e" * 64,
    }
    core = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "attempt_hash": attempt["attempt_hash"],
        "execution_commit": attempt["execution_commit"],
        "runner_sha256": attempt["runner_sha256"],
        "preregistration_manifest_hash": (
            runner.PREREGISTRATION_MANIFEST_HASH
        ),
        "decision": "pass",
        "open_2021_outcomes_authorized": True,
        "open_2022_or_later_outcomes_authorized": False,
        "qlora_authorized": False,
        "terminal_action": (
            "ACCEPT_PSIM_D8_RLLM2_S4_SEALED_2021_SCHEDULES_"
            "AUTHORIZE_2021_EVALUATION_ONLY"
        ),
        "artifacts": {},
    }
    result = {**core, "result_hash": runner._canonical_hash(core)}
    result_path.write_bytes(runner._canonical_bytes(result, pretty=True))

    with pytest.raises(RuntimeError, match="existing result changed"):
        runner._verify_existing_result(attempt)


def test_existing_result_rejects_2021_metric_leakage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(prereg, "RESULT_PATH", result_path)
    paths = runner._expected_artifact_paths()
    remapped: dict[str, Path] = {}
    for name in paths:
        remapped[name] = tmp_path / f"{name}.bin"
    monkeypatch.setattr(
        prereg,
        "TRANSITION_LEDGER_PATH",
        remapped["transition_ledger_2020"],
    )
    monkeypatch.setattr(prereg, "PCA_PATH", remapped["pca32_2020"])
    monkeypatch.setattr(
        prereg,
        "SCHEDULE_PATH",
        remapped["base_schedules_2021"],
    )
    monkeypatch.setattr(
        prereg,
        "DELAYED_SCHEDULE_PATH",
        remapped["delayed_primary_schedules_2021"],
    )
    monkeypatch.setattr(
        prereg,
        "SCHEDULE_MANIFEST_PATH",
        remapped["schedule_manifest_2021"],
    )
    attempt = {
        "attempt_hash": "c" * 64,
        "execution_commit": "d" * 40,
        "runner_sha256": "e" * 64,
    }
    artifacts: dict[str, dict[str, object]] = {}
    for name, path in remapped.items():
        if name == "schedule_manifest_2021":
            continue
        path.write_bytes(name.encode())
        artifacts[name] = {
            "path": path.as_posix(),
            "sha256": runner._sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest = runner._schedule_manifest(
        attempt_hash=attempt["attempt_hash"],
        base_artifact=artifacts["base_schedules_2021"],
        delayed_artifact=artifacts["delayed_primary_schedules_2021"],
        pca_artifact=artifacts["pca32_2020"],
        ledger_artifact=artifacts["transition_ledger_2020"],
    )
    manifest_raw = runner._canonical_bytes(manifest, pretty=True)
    remapped["schedule_manifest_2021"].write_bytes(manifest_raw)
    artifacts["schedule_manifest_2021"] = {
        "path": remapped["schedule_manifest_2021"].as_posix(),
        "sha256": runner._sha256_file(
            remapped["schedule_manifest_2021"]
        ),
        "bytes": len(manifest_raw),
        "manifest_hash": manifest["manifest_hash"],
    }
    core = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "attempt_hash": attempt["attempt_hash"],
        "execution_commit": attempt["execution_commit"],
        "runner_sha256": attempt["runner_sha256"],
        "preregistration_manifest_hash": (
            runner.PREREGISTRATION_MANIFEST_HASH
        ),
        "decision": "pass",
        "open_2021_outcomes_authorized": True,
        "open_2022_or_later_outcomes_authorized": False,
        "qlora_authorized": False,
        "terminal_action": (
            "ACCEPT_PSIM_D8_RLLM2_S4_SEALED_2021_SCHEDULES_"
            "AUTHORIZE_2021_EVALUATION_ONLY"
        ),
        "target_schedule_seal": {
            "target_stage": "2021",
            "manifest_hash": manifest["manifest_hash"],
        },
        "access_boundary": {
            "2021_market_or_funding_paths_read": [],
            "2021_reward_rows_created": 365,
            "2021_economic_metrics_computed": 2,
            "2021_outcomes_opened": False,
            "2022_or_2023_outcomes_opened": False,
        },
        "artifacts": artifacts,
    }
    result = {**core, "result_hash": runner._canonical_hash(core)}
    result_path.write_bytes(runner._canonical_bytes(result, pretty=True))

    with pytest.raises(RuntimeError, match="schedule seal changed"):
        runner._verify_existing_result(attempt)
