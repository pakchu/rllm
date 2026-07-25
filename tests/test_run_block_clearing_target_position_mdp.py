from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training import freeze_block_clearing_target_position_evaluator as freeze
from training import preregister_block_clearing_target_position_mdp as prereg
from training import run_block_clearing_target_position_mdp as runner


def _states(stage: str) -> pd.DataFrame:
    rows = []
    for index in range(2):
        row = {
            "sequence_id": f"{stage}-{index}",
            "entry_time": pd.Timestamp(
                f"{stage}-01-01T00:0{index * 5}:00Z"
            ),
            "source_signal_id_m2": f"m2-{index}",
            "source_signal_id_m1": f"m1-{index}",
            "source_signal_id_s0": f"s0-{index}",
            "source_signature": f"sig-{index}",
        }
        for column in prereg.SOURCE_TOKEN_COLUMNS:
            row[column] = "TOKEN"
        rows.append(row)
    return pd.DataFrame(rows, columns=prereg.SOURCE_SEQUENCE_COLUMNS)


def _arrays(count: int) -> dict[str, np.ndarray]:
    rewards = np.zeros((count, 3, 3), dtype=float)
    reachable = np.ones((count, 3), dtype=bool)
    reachable[0] = [False, True, False]
    rewards[~reachable] = np.nan
    terminal = np.zeros(count, dtype=bool)
    terminal[-1] = True
    return {
        "reward_tensor": rewards,
        "reachable_mask": reachable,
        "terminal": terminal,
    }


def test_report_is_canonical_write_once_and_detects_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    first = runner._write_report_once(
        path,
        {"protocol_version": "synthetic", "value": 1},
    )
    second = runner._write_report_once(
        path,
        {"protocol_version": "synthetic", "value": 1},
    )
    assert first == second
    with pytest.raises(RuntimeError, match="drift"):
        runner._write_report_once(
            path,
            {"protocol_version": "synthetic", "value": 2},
        )
    path.write_text('{"manifest_hash":"bad"}')
    with pytest.raises(ValueError, match="manifest hash"):
        runner.load_report(path)


def test_fit_data_preserves_two_episode_terminal_resets() -> None:
    states, rewards, terminal, reachable = runner._fit_data(
        [_states("2020"), _states("2021")],
        [_arrays(2), _arrays(2)],
    )
    assert len(states) == 4
    assert terminal.tolist() == [False, True, False, True]
    assert reachable[0].tolist() == [False, True, False]
    assert reachable[2].tolist() == [False, True, False]
    assert rewards.shape == (4, 3, 3)


def test_prepare_2020_never_opens_a_later_outcome_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.concat(
        [_states("2020"), _states("2021")],
        ignore_index=True,
    )
    opened: list[str] = []
    monkeypatch.setattr(runner, "assert_runner_committed", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_source_states", lambda: source)
    monkeypatch.setattr(
        runner,
        "_stage_states",
        lambda _source, stage: _states(stage),
    )
    monkeypatch.setattr(
        runner.stage_sources,
        "prepare_stage_source",
        lambda stage, **kwargs: opened.append(stage)
        or {"manifest_hash": f"source-{stage}"},
    )
    monkeypatch.setattr(
        runner.stage_sources,
        "load_stage_source",
        lambda stage, **kwargs: (
            pd.DataFrame({"date": []}),
            pd.DataFrame({"funding_time_utc": []}),
            {"stage": stage},
        ),
    )
    monkeypatch.setattr(
        runner,
        "_build_or_load_ledger",
        lambda stage, states, market, funding, **kwargs: (
            {
                "path": f"{stage}.csv.gz",
                "sha256": "ledger",
                "frame_hash": "frame",
                "rows": 1,
                "columns": [],
            },
            _arrays(len(states)),
        ),
    )
    fitted = SimpleNamespace(
        fitted_estimators=21,
        memory_tables_fit=1,
        policies=OrderedDict(
            (policy_id, object())
            for policy_id in freeze.FAMILY_IDS
        ),
    )
    monkeypatch.setattr(
        runner.policy_family,
        "fit_family",
        lambda *args, **kwargs: fitted,
    )
    monkeypatch.setattr(
        runner.policy_family,
        "build_transfer_schedules",
        lambda *args, **kwargs: (
            OrderedDict(
                (policy_id, pd.DataFrame())
                for policy_id in freeze.FAMILY_IDS
            ),
            OrderedDict(
                (policy_id, pd.DataFrame())
                for policy_id in runner.policy_family.PROMOTABLE_PRIMARY_IDS
            ),
        ),
    )
    monkeypatch.setattr(
        runner.schedule_seal,
        "seal_transfer_year_schedule",
        lambda *args, **kwargs: {
            "path": "schedule.json",
            "manifest_hash": "schedule",
            "file_sha256": "schedule-file",
        },
    )
    report = runner.prepare_2020_and_seal_2021(
        run_root=tmp_path / "run",
        ledger_root=tmp_path / "ledgers",
        stage_source_root=tmp_path / "sources",
        schedule_root=tmp_path / "schedules",
    )
    assert opened == ["2020"]
    assert report["post_2020_outcome_rows_opened"] == 0
    assert report["target_stage_sealed"] == "2021"


def test_2021_failure_does_not_refit_or_authorize_2022(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner._write_report_once(
        tmp_path / "run" / runner.FIT_2020_REPORT,
        {
            "protocol_version": "fit",
            "runner_commit": "b" * 40,
        },
    )
    monkeypatch.setattr(runner, "assert_runner_committed", lambda: "b" * 40)
    monkeypatch.setattr(
        runner,
        "prepare_2020_and_seal_2021",
        lambda **kwargs: {"decision": "authorize_2021_transfer_only"},
    )
    monkeypatch.setattr(
        runner,
        "_load_sealed_schedule_frames",
        lambda *args, **kwargs: (
            {"path": "seal.json", "manifest_hash": "seal"},
            pd.DataFrame(),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        runner.stage_sources,
        "prepare_stage_source",
        lambda *args, **kwargs: {"manifest_hash": "source"},
    )
    monkeypatch.setattr(
        runner.stage_sources,
        "load_stage_source",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(),
            {"stage": "2021"},
        ),
    )
    monkeypatch.setattr(
        runner.family_evaluator,
        "evaluate_bctp_transfer_stage",
        lambda *args, **kwargs: {
            "stage_passed": False,
            "selected_primary_id": None,
        },
    )
    monkeypatch.setattr(
        runner,
        "_refit_and_seal_2022",
        lambda **kwargs: pytest.fail("refit must not run"),
    )
    report = runner.evaluate_2021_and_maybe_seal_2022(
        run_root=tmp_path / "run",
        ledger_root=tmp_path / "ledgers",
        stage_source_root=tmp_path / "sources",
        schedule_root=tmp_path / "schedules",
    )
    assert report["decision"] == "retire_bctp_cheap_policy_unchanged"
    assert report["post_2021_outcome_rows_opened"] == 0


def test_existing_failed_2021_report_is_bound_without_outcome_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner._write_report_once(
        tmp_path / "run" / runner.TRANSFER_2021_REPORT,
        {
            "protocol_version": "bctp_transfer_2021_v1",
            "runner_commit": "c" * 40,
            "stage": "2021",
            "schedule_manifest_hash": "seal-2021",
            "stage_source_manifest_hash": "source-2021",
            "evaluation": {
                "stage": "2021",
                "stage_passed": False,
                "selected_primary_id": None,
            },
            "decision": "retire_bctp_cheap_policy_unchanged",
            "post_2021_outcome_rows_opened": 0,
        },
    )
    monkeypatch.setattr(runner, "assert_runner_committed", lambda: "c" * 40)
    monkeypatch.setattr(
        runner,
        "prepare_2020_and_seal_2021",
        lambda **kwargs: {"decision": "authorize_2021_transfer_only"},
    )
    monkeypatch.setattr(
        runner,
        "_load_sealed_schedule_frames",
        lambda *args, **kwargs: (
            {"path": "seal.json", "manifest_hash": "seal-2021"},
            pd.DataFrame(),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_load_stage_manifest_only",
        lambda *args, **kwargs: {"manifest_hash": "source-2021"},
    )
    monkeypatch.setattr(
        runner.stage_sources,
        "prepare_stage_source",
        lambda *args, **kwargs: pytest.fail(
            "completed report must not prepare outcome stage"
        ),
    )
    monkeypatch.setattr(
        runner.stage_sources,
        "load_stage_source",
        lambda *args, **kwargs: pytest.fail(
            "completed failed report must not reopen outcomes"
        ),
    )

    report = runner.evaluate_2021_and_maybe_seal_2022(
        run_root=tmp_path / "run",
        ledger_root=tmp_path / "ledgers",
        stage_source_root=tmp_path / "sources",
        schedule_root=tmp_path / "schedules",
    )
    assert report["decision"] == "retire_bctp_cheap_policy_unchanged"


def test_existing_2021_report_rejects_schedule_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner._write_report_once(
        tmp_path / "run" / runner.TRANSFER_2021_REPORT,
        {
            "protocol_version": "bctp_transfer_2021_v1",
            "runner_commit": "d" * 40,
            "stage": "2021",
            "schedule_manifest_hash": "old-seal",
            "stage_source_manifest_hash": "source-2021",
            "evaluation": {
                "stage": "2021",
                "stage_passed": False,
                "selected_primary_id": None,
            },
            "decision": "retire_bctp_cheap_policy_unchanged",
            "post_2021_outcome_rows_opened": 0,
        },
    )
    monkeypatch.setattr(runner, "assert_runner_committed", lambda: "d" * 40)
    monkeypatch.setattr(
        runner,
        "prepare_2020_and_seal_2021",
        lambda **kwargs: {"decision": "authorize_2021_transfer_only"},
    )
    monkeypatch.setattr(
        runner,
        "_load_sealed_schedule_frames",
        lambda *args, **kwargs: (
            {"path": "seal.json", "manifest_hash": "new-seal"},
            pd.DataFrame(),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_load_stage_manifest_only",
        lambda *args, **kwargs: {"manifest_hash": "source-2021"},
    )
    monkeypatch.setattr(
        runner.stage_sources,
        "load_stage_source",
        lambda *args, **kwargs: pytest.fail(
            "binding drift must fail before outcome access"
        ),
    )

    with pytest.raises(RuntimeError, match="schedule_manifest_hash"):
        runner.evaluate_2021_and_maybe_seal_2022(
            run_root=tmp_path / "run",
            ledger_root=tmp_path / "ledgers",
            stage_source_root=tmp_path / "sources",
            schedule_root=tmp_path / "schedules",
        )


def test_existing_2022_report_is_bound_without_outcome_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = "extra_trees_fqi"
    runner._write_report_once(
        tmp_path / "run" / runner.TRANSFER_2022_REPORT,
        {
            "protocol_version": "bctp_transfer_2022_v1",
            "runner_commit": "e" * 40,
            "stage": "2022",
            "selected_primary_id": selected,
            "schedule_manifest_hash": "seal-2022",
            "stage_source_manifest_hash": "source-2022",
            "evaluation": {
                "stage": "2022",
                "stage_passed": False,
                "requested_primary_id": selected,
                "selected_primary_id": None,
            },
            "decision": "retire_bctp_cheap_policy_unchanged",
            "post_2022_outcome_rows_opened": 0,
        },
    )
    original_load = runner._load_runner_report

    def load_report(path, *, runner_commit):
        if Path(path).name == runner.TRANSFER_2021_REPORT:
            return {
                "evaluation": {
                    "stage": "2021",
                    "stage_passed": True,
                    "selected_primary_id": selected,
                }
            }
        return original_load(path, runner_commit=runner_commit)

    monkeypatch.setattr(runner, "assert_runner_committed", lambda: "e" * 40)
    monkeypatch.setattr(
        runner,
        "evaluate_2021_and_maybe_seal_2022",
        lambda **kwargs: {"decision": "refit_and_seal_2022"},
    )
    monkeypatch.setattr(runner, "_load_runner_report", load_report)
    monkeypatch.setattr(
        runner,
        "_load_existing_refit_report",
        lambda **kwargs: {"selected_primary_id": selected},
    )
    monkeypatch.setattr(
        runner,
        "_load_sealed_schedule_frames",
        lambda *args, **kwargs: (
            {"path": "seal.json", "manifest_hash": "seal-2022"},
            pd.DataFrame(),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_load_stage_manifest_only",
        lambda *args, **kwargs: {"manifest_hash": "source-2022"},
    )
    monkeypatch.setattr(
        runner.stage_sources,
        "prepare_stage_source",
        lambda *args, **kwargs: pytest.fail(
            "completed 2022 report must not prepare outcomes"
        ),
    )
    monkeypatch.setattr(
        runner.stage_sources,
        "load_stage_source",
        lambda *args, **kwargs: pytest.fail(
            "completed 2022 report must not reopen outcomes"
        ),
    )

    report = runner.evaluate_2022_selected_algorithm(
        run_root=tmp_path / "run",
        ledger_root=tmp_path / "ledgers",
        stage_source_root=tmp_path / "sources",
        schedule_root=tmp_path / "schedules",
    )
    assert report["decision"] == "retire_bctp_cheap_policy_unchanged"


def test_status_reads_only_existing_reports(tmp_path: Path) -> None:
    root = tmp_path / "run"
    runner._write_report_once(
        root / runner.FIT_2020_REPORT,
        {"protocol_version": "fit"},
    )
    result = runner.status(run_root=root)
    assert result["reports"][runner.FIT_2020_REPORT]["exists"] is True
    assert (
        result["reports"][runner.TRANSFER_2022_REPORT]["exists"]
        is False
    )
