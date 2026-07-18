from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from training import evaluate_fed_h8_deposit_migration as evaluator


def test_static_inputs_and_schedule_family_are_frozen() -> None:
    registration, support = evaluator._verify_static_inputs()
    assert registration["policy_id"] == "H8DM-1"
    assert support["support_passed"] is True
    assert tuple(registration["controls"]["mechanism"]) == (
        evaluator.MECHANISM_CONTROLS
    )
    assert tuple(registration["controls"]["falsification"]) == (
        evaluator.FALSIFICATION_CONTROLS
    )
    assert evaluator.SINGLE_COMPONENT_CONTROLS == (
        "migration_only",
        "borrowings_only",
        "cash_only",
    )
    schedules = evaluator.load_schedules()
    assert tuple(schedules) == evaluator.ALL_CLOCK_NAMES
    assert {name: len(frame) for name, frame in schedules.items()} == {
        name: int(count) for name, count in support["clocks"]["counts"].items()
    }

    primary = schedules["primary"]
    stage1_source = primary.loc[
        primary["signal_day"].ge(evaluator.STAGE1[0])
        & primary["signal_day"].lt(evaluator.STAGE1[1])
    ]
    assert len(stage1_source) == 75
    # The 2022-12-30 release exits on 2023-01-01. It stays out of standalone
    # Stage1 rather than silently opening a sealed 2023 path.
    assert len(evaluator._window_schedule(primary, evaluator.STAGE1)) == 74
    assert len(evaluator._window_schedule(primary, evaluator.STAGE2)) == 24
    primary_record = evaluator._schedule_record(primary)
    assert primary_record["stage1_boundary_excluded_trades"] == 1
    assert primary_record["sealed_2023_boundary_excluded_trades"] == 0

    for schedule in schedules.values():
        record = evaluator._schedule_record(schedule)
        assert record["execution_clock_exact"] is True
        assert record["globally_nonoverlapping"] is True

    baseline = primary.set_index("release_date")
    for name, weeks in (("one_week_delay", 1), ("four_week_placebo", 4)):
        delayed = schedules[name].set_index("release_date")
        assert delayed.index.equals(baseline.index)
        assert delayed["side"].equals(baseline["side"])
        delayed_local = delayed["entry_time"].dt.tz_convert("America/New_York")
        baseline_local = baseline["entry_time"].dt.tz_convert("America/New_York")
        day_delta = delayed_local.dt.date - baseline_local.dt.date
        assert bool(day_delta.eq(pd.Timedelta(weeks=weeks)).all())
        assert bool(delayed_local.dt.hour.eq(17).all())


def test_freeze_opens_no_outcome_and_replays(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    report = evaluator.freeze_evaluator(path)
    assert report["opened_windows"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["simulation_run_during_freeze"] is False
    assert evaluator.verify_evaluator_freeze(path) == report


def test_self_sealed_forged_freeze_is_rejected(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid-freeze.json"
    report = evaluator.freeze_evaluator(valid_path)
    forged_core = {
        key: value for key, value in report.items() if key != "manifest_hash"
    }
    forged_core["support_commit"] = "forged"
    forged = evaluator._seal(forged_core)
    forged_path = tmp_path / "forged-freeze.json"
    forged_path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="freeze contract changed"):
        evaluator.verify_evaluator_freeze(forged_path)


def test_two_sided_signflip_is_invariant_to_exact_direction_flip() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": [
                "2021-01-04T22:00:00+00:00",
                "2021-01-11T22:00:00+00:00",
                "2021-01-18T22:00:00+00:00",
            ],
            "net_return": [0.02, -0.01, 0.03],
        }
    )
    first = evaluator.weekly_cluster_signflip_two_sided(trades, draws=1000, seed=7)
    flipped = trades.copy()
    flipped["net_return"] = -flipped["net_return"]
    second = evaluator.weekly_cluster_signflip_two_sided(flipped, draws=1000, seed=7)
    assert first["method"] == "exact"
    assert first["p_value_two_sided"] == second["p_value_two_sided"]


def test_stage2_refuses_a_missing_stage1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", missing)
    with pytest.raises(ValueError, match="has not been run"):
        evaluator._verified_passing_stage1("irrelevant")


def test_self_sealed_forged_stage1_cannot_open_2023(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_hash = "frozen-evaluator"
    fake_core: dict[str, Any] = {
        "evaluator_freeze_manifest_hash": freeze_hash,
        "evaluator_source_sha256": evaluator._sha256(evaluator.EVALUATOR_SOURCE),
        "gate_passed": True,
        "forged": True,
    }
    fake = evaluator._seal(fake_core)
    stage1 = tmp_path / "forged-stage1.json"
    stage1.write_text(json.dumps(fake))
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", stage1)
    monkeypatch.setattr(
        evaluator,
        "verify_evaluator_freeze",
        lambda: {"manifest_hash": freeze_hash},
    )
    monkeypatch.setattr(
        evaluator,
        "_build_stage_report",
        lambda **_: evaluator._seal({"replayed": True}),
    )
    opened: list[evaluator.TimeWindow] = []

    def fake_loader(
        window: evaluator.TimeWindow,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        if window == evaluator.STAGE2 or window == evaluator.POOLED:
            pytest.fail("sealed 2023/pooled loader was reached")
        opened.append(window)
        return pd.DataFrame(), pd.DataFrame(), {}

    monkeypatch.setattr(evaluator, "load_execution_window", fake_loader)
    with pytest.raises(ValueError, match="does not exactly replay"):
        evaluator._verified_passing_stage1(freeze_hash)
    assert opened == [evaluator.STAGE1]


def test_failed_stage1_blocks_before_any_outcome_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = evaluator._seal(
        {
            "evaluator_freeze_manifest_hash": "freeze",
            "evaluator_source_sha256": evaluator._sha256(evaluator.EVALUATOR_SOURCE),
            "gate_passed": False,
        }
    )
    stage1 = tmp_path / "failed-stage1.json"
    stage1.write_text(json.dumps(fake))
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", stage1)
    monkeypatch.setattr(
        evaluator,
        "load_execution_window",
        lambda _: pytest.fail("failed Stage1 reached an outcome loader"),
    )
    with pytest.raises(ValueError, match="Stage1 failed; 2023 remains sealed"):
        evaluator._verified_passing_stage1("freeze")


def test_pooled_failure_overrides_a_standalone_stage2_pass() -> None:
    standalone = evaluator._seal(
        {
            "gates": {"standalone_gate": True},
            "gate_passed": True,
            "opened_windows": ["stage2_2023"],
            "sealed_windows": ["2024_plus"],
            "disposition": "PASS_STAGE2_OPEN_ORTHOGONALITY",
        }
    )
    pooled = {
        "gates": {
            "pooled_stage1_stage2_ratio_at_least": True,
            "pooled_weekly_cluster_signflip_p_at_most": False,
        }
    }
    report = evaluator._finalize_stage2_report(
        standalone,
        pooled,
        verified_stage1_manifest_hash="stage1-hash",
    )
    assert report["gate_passed"] is False
    assert report["disposition"] == "REJECT_NO_REPAIR"
    assert report["opened_windows"] == ["stage1_2020_2022", "stage2_2023"]
    assert report["sealed_windows"] == ["2024_plus"]
    assert report["verified_stage1_manifest_hash"] == "stage1-hash"
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == evaluator._canonical_hash(core)


def test_pooled_window_is_exactly_pre_2024_without_opening_it() -> None:
    assert evaluator.POOLED == (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
    )


def test_frozen_evaluator_artifact_replays() -> None:
    stored = json.loads(Path(evaluator.EVALUATOR_FREEZE).read_text())
    assert evaluator.verify_evaluator_freeze() == stored
    assert evaluator._sha256(evaluator.EVALUATOR_SOURCE) == (
        "8f66ec75c20eba83af12f8531382b5c450c7c0ba1f2340f82e27915d09f168ac"
    )
    assert evaluator._sha256(evaluator.EVALUATOR_FREEZE) == (
        "1ae15f1218fa5e319a2b1f750a99aa36293d4045ba3b3928928579fadbf4a2d5"
    )
    assert stored["manifest_hash"] == (
        "c93c5baf8f2da0d875e494f95264cc06ca360836613141fbaee59e5986ecf68e"
    )


def test_frozen_stage1_is_rejected_and_keeps_2023_sealed() -> None:
    stored = json.loads(Path(evaluator.STAGE1_OUTPUT).read_text())
    assert evaluator._sha256(evaluator.STAGE1_OUTPUT) == (
        "3f5118077cdafb48ffb59fc6cec8e7643613861f921bfd78403097181c287a7f"
    )
    assert stored["manifest_hash"] == (
        "fba386377571abf79c15c2888541695c8d7b7e828481c3c8bf0753c644356607"
    )
    assert stored["gate_passed"] is False
    assert stored["disposition"] == "REJECT_KEEP_2023_SEALED"
    assert stored["opened_windows"] == ["stage1_2020_2022"]
    assert stored["sealed_windows"] == ["stage2_2023", "2024_plus"]
    primary = stored["headline_by_clock"]["primary"]
    assert primary["absolute_return_pct"] == pytest.approx(-2.295403550976527)
    assert primary["cagr_pct"] == pytest.approx(-0.7708894409916067)
    assert primary["strict_mdd_pct"] == pytest.approx(19.768276068245783)
    assert primary["cagr_to_strict_mdd"] == pytest.approx(-0.038996290740288854)
    assert primary["trades"] == 74
    assert primary["long_trades"] == 28
    assert primary["short_trades"] == 46
    assert primary["weekly_cluster_signflip_p"] == pytest.approx(0.9446027698615069)
    diagnostics = stored["execution_diagnostics"]
    assert diagnostics["market"]["last_timestamp"] == ("2022-12-31T23:55:00+00:00")
    assert diagnostics["funding"]["last_timestamp"] == ("2022-12-31T16:00:00+00:00")
    assert diagnostics["market"]["stopped_before_parsing_end_boundary"] is True
    assert diagnostics["funding"]["stopped_before_parsing_end_boundary"] is True
    with pytest.raises(ValueError, match="Stage1 failed; 2023 remains sealed"):
        evaluator._verified_passing_stage1(stored["evaluator_freeze_manifest_hash"])
