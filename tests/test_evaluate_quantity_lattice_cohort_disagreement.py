from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import evaluate_quantity_lattice_cohort_disagreement as evaluator


def _market(
    *,
    start: str = "2021-01-01T00:10:00Z",
    periods: int | None = None,
) -> pd.DataFrame:
    count = periods or evaluator.FROZEN_CONFIG.hold_bars + 1
    dates = pd.date_range(pd.Timestamp(start), periods=count, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.full(count, 100.0),
            "high": np.full(count, 100.0),
            "low": np.full(count, 100.0),
            "close": np.full(count, 100.0),
        }
    )


def _funding(*rows: tuple[pd.Timestamp, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "funding_time": pd.to_datetime(
                [timestamp for timestamp, _ in rows], utc=True
            ),
            "symbol": ["BTCUSDT"] * len(rows),
            "funding_rate": [rate for _, rate in rows],
            "settlement_mark_price": [100.0] * len(rows),
        }
    )


def _clock(market: pd.DataFrame, *, side: int = 1) -> pd.DataFrame:
    entry = evaluator._utc(market.iloc[0]["date"])
    exit_time = evaluator._utc(
        market.iloc[evaluator.FROZEN_CONFIG.hold_bars]["date"]
    )
    return pd.DataFrame(
        {
            "entry_time": [entry],
            "exit_time": [exit_time],
            "side": [side],
        }
    )


def _metrics(
    *,
    absolute_return_pct: float = 1.0,
    cagr_to_strict_mdd: float = 3.0,
    strict_mdd_pct: float = 15.0,
    mean_gross_underlying_bp: float = 24.0,
    weekly_p: float = 0.099,
) -> dict[str, Any]:
    return {
        "absolute_return_pct": absolute_return_pct,
        "cagr_to_strict_mdd": cagr_to_strict_mdd,
        "strict_mdd_pct": strict_mdd_pct,
        "mean_gross_underlying_bp": mean_gross_underlying_bp,
        "weekly_cluster_signflip": {"p_value_two_sided": weekly_p},
    }


def test_frozen_public_contract_is_stable() -> None:
    assert evaluator.POLICY_ID == "QLCD-288"
    assert evaluator.STAGE_ORDER == ("train", "selection", "test", "eval", "recent")
    assert evaluator.PHASE_ONE == ("train", "selection")
    cfg = evaluator.FROZEN_CONFIG
    assert cfg.leverage == pytest.approx(0.5)
    assert cfg.base_cost_notional_per_side == pytest.approx(0.0006)
    assert cfg.stress_cost_notional_per_side == pytest.approx(0.0010)
    assert cfg.hold_bars == 288
    assert cfg.exact_cluster_max == 20
    assert cfg.cluster_draws == 20_000
    assert cfg.cluster_seed == 20_260_720
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        cfg.hold_bars = 12  # type: ignore[misc]
    for mapping_name in ("STAGE_WINDOWS", "STAGE_OUTPUTS", "STAGE_DOCS"):
        assert tuple(getattr(evaluator, mapping_name)) == evaluator.STAGE_ORDER
    for mapping_name in ("STAGE_SOURCE_MANIFESTS", "STAGE_SOURCE_DIRS"):
        assert tuple(getattr(evaluator, mapping_name)) == evaluator.PHASE_ONE
    assert evaluator.CONTROL_ORDER == (
        "primary",
        "exact_side_flip",
        "medium_vs_fine",
        "remove_opposition",
        "all_quantity_imbalance",
        "stale_one_hour",
        "stale_twenty_four_hours",
    )


def test_source_clock_and_phase_one_schedules_are_frozen() -> None:
    full = evaluator.load_schedules()
    phase = evaluator.derive_phase_one_schedules()
    assert len(full) == 489
    assert len(phase["train"]) == 377
    assert len(phase["selection"]) == 111
    assert len(phase["train"]) + len(phase["selection"]) == 488
    assert bool(
        full["entry_time"]
        .sub(full["decision_time"])
        .eq(pd.Timedelta(minutes=5))
        .all()
    )
    assert bool(
        full["exit_time"]
        .sub(full["entry_time"])
        .eq(pd.Timedelta(hours=24))
        .all()
    )
    assert full["entry_time"].iloc[1:].reset_index(drop=True).ge(
        full["exit_time"].iloc[:-1].reset_index(drop=True)
    ).all()
    boundary = evaluator.STAGE_WINDOWS["train"][1]
    assert boundary is not None
    crossing = full["entry_time"].lt(boundary) & full["exit_time"].gt(boundary)
    assert int(crossing.sum()) == 1


def test_all_preregistered_falsification_controls_are_derived_source_only(
    tmp_path: Path,
) -> None:
    derived = evaluator.derive_evaluation_clocks()
    assert set(derived["control"]) == set(evaluator.CONTROL_ORDER)
    assert derived.groupby("control", sort=False).size().to_dict() == {
        "primary": 489,
        "exact_side_flip": 489,
        "medium_vs_fine": 547,
        "remove_opposition": 504,
        "all_quantity_imbalance": 486,
        "stale_one_hour": 489,
        "stale_twenty_four_hours": 489,
    }
    path = tmp_path / "controls.csv.gz"
    digest = evaluator._write_evaluation_clocks(derived, path)
    schedules = evaluator.load_control_schedules(
        clock_path=path,
        expected_clock_sha256=digest,
    )
    assert set(schedules) == set(evaluator.CONTROL_ORDER)
    primary = schedules["primary"].reset_index(drop=True)
    flip = schedules["exact_side_flip"].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        flip[["decision_time", "entry_time", "exit_time", "score", "threshold"]],
        primary[["decision_time", "entry_time", "exit_time", "score", "threshold"]],
        check_exact=True,
    )
    assert bool(
        schedules["exact_side_flip"]["side"]
        .eq(-schedules["primary"]["side"])
        .all()
    )
    for schedule in schedules.values():
        entries = schedule["entry_time"].reset_index(drop=True)
        exits = schedule["exit_time"].reset_index(drop=True)
        assert bool(entries.iloc[1:].ge(exits.iloc[:-1].to_numpy()).all())
        assert bool(
            schedule["exit_time"]
            .sub(schedule["entry_time"])
            .eq(pd.Timedelta(hours=24))
            .all()
        )
    for control, shift in (
        ("stale_one_hour", pd.Timedelta(hours=1)),
        ("stale_twenty_four_hours", pd.Timedelta(hours=24)),
    ):
        stale = schedules[control].reset_index(drop=True)
        for column in ("decision_time", "entry_time", "exit_time"):
            assert bool(stale[column].sub(primary[column]).eq(shift).all())


def test_help_is_executable_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(evaluator.EVALUATOR_SOURCE), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--freeze" in completed.stdout
    assert "--prepare-stage-source" in completed.stdout
    assert "--stage" in completed.stdout


def _redirect_stage_outputs(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    for stage in evaluator.STAGE_ORDER:
        monkeypatch.setitem(
            evaluator.STAGE_OUTPUTS,
            stage,
            root / f"unused-{stage}.json",
        )


def test_freeze_is_source_only_and_does_not_open_parent_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[str] = []
    real_sha = evaluator._sha256

    def tracking_sha(path: str | Path) -> str:
        seen.append(str(path))
        return real_sha(path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("freeze parsed or simulated an execution outcome")

    _redirect_stage_outputs(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluator, "_sha256", tracking_sha)
    monkeypatch.setattr(evaluator, "prepare_stage_source", forbidden)
    monkeypatch.setattr(evaluator, "load_execution_window", forbidden)
    monkeypatch.setattr(evaluator, "simulate_strict", forbidden)
    report = evaluator.freeze_evaluator(
        tmp_path / "freeze.json",
        evaluation_clock_path=tmp_path / "clocks.csv.gz",
    )

    assert report["opened_windows"] == []
    assert report["sealed_windows"] == list(evaluator.STAGE_ORDER)
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["execution_outcome_data_bytes_hashed_during_freeze"] is False
    assert report["simulation_run_during_freeze"] is False
    assert str(evaluator.LEGACY_MARKET) not in seen
    assert str(evaluator.LEGACY_FUNDING) not in seen
    assert str(evaluator.PRIMARY_CLOCK) in seen
    assert report["primary_cross_stage_exclusions"] == [
        {
            "decision_time": "2022-12-31T04:20:00+00:00",
            "entry_time": "2022-12-31T04:25:00+00:00",
            "exit_time": "2023-01-01T04:25:00+00:00",
            "side": -1,
            "score_hex": "0x1.14a088d9370aep-4",
            "threshold_hex": "0x1.78612625d4192p-5",
        }
    ]
    assert report["primary_cross_stage_exclusion_hash"] == (
        "7b7824b732032476f7409ea93cf942fd9731bcf949de8a42faf7ec0fa4b5eff3"
    )
    assert report["falsification_controls_are_mandatory_report_only"] is True
    assert report["falsification_controls_cannot_repair_primary"] is True
    assert report["future_phase_contract"] == evaluator.FUTURE_PHASE_CONTRACT
    assert report["future_phase_contract"]["current_evaluator_approval"] == (
        "train_and_selection_only"
    )
    assert report["future_phase_contract"][
        "phase_two_must_be_committed_before_2024_outcomes"
    ] is True
    assert report["future_phase_contract"]["mutable_rules_after_phase_one"] == []


def test_freeze_is_write_once_and_verifiable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _redirect_stage_outputs(monkeypatch, tmp_path)
    output = tmp_path / "freeze.json"
    first = evaluator.freeze_evaluator(
        output,
        evaluation_clock_path=tmp_path / "clocks.csv.gz",
    )
    assert evaluator.verify_evaluator_freeze(output) == first
    with pytest.raises(FileExistsError):
        evaluator.freeze_evaluator(
            output,
            evaluation_clock_path=tmp_path / "clocks.csv.gz",
        )


def test_freeze_verification_rejects_config_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _redirect_stage_outputs(monkeypatch, tmp_path)
    output = tmp_path / "freeze.json"
    report = evaluator.freeze_evaluator(
        output,
        evaluation_clock_path=tmp_path / "clocks.csv.gz",
    )
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    core["evaluation_config"] = {**core["evaluation_config"], "hold_bars": 12}
    output.write_text(json.dumps(evaluator._seal(core)), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        evaluator.verify_evaluator_freeze(output)


def test_slice_stops_at_expected_rows_without_reading_future_payload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "parent.csv.gz"
    output = tmp_path / "stage.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2022-12-31 23:55:00,PRE,PRE,PRE,PRE\n")
        handle.write("2023-01-01 00:00:00,100,101,99,100\n")
        handle.write("2023-01-01 00:05:00,100,101,99,100\n")
        handle.write("NOT_A_TIMESTAMP,FUTURE,FUTURE,FUTURE,FUTURE\n")
    report = evaluator._slice_gzip_csv(
        source,
        output,
        timestamp_column="date",
        start=evaluator._utc("2023-01-01T00:00:00Z"),
        end=evaluator._utc("2023-01-01T00:10:00Z"),
        expected_rows=2,
    )
    assert report["rows"] == 2
    assert report["post_stage_numeric_rows_parsed"] == 0
    assert report["first_excluded_row_read"] is False
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        payload = handle.read()
    assert "PRE" not in payload
    assert "FUTURE" not in payload


def test_phase_two_source_is_blocked_before_freeze_or_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("phase-two block ran too late")

    monkeypatch.setattr(evaluator, "verify_evaluator_freeze", forbidden)
    monkeypatch.setattr(evaluator, "_sha256", forbidden)
    with pytest.raises(RuntimeError, match="phase-two sealed"):
        evaluator.prepare_stage_source("test")


def test_selection_source_checks_train_pass_before_parent_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "verify_evaluator_freeze",
        lambda: {"manifest_hash": "freeze"},
    )

    def failed_prior(*args: object, **kwargs: object) -> None:
        raise ValueError("train did not pass")

    def forbidden_digest(*args: object, **kwargs: object) -> str:
        raise AssertionError("parent digest ran before train gate")

    monkeypatch.setattr(evaluator, "_verified_prior_reports", failed_prior)
    monkeypatch.setattr(evaluator, "_sha256", forbidden_digest)
    with pytest.raises(ValueError, match="train did not pass"):
        evaluator.prepare_stage_source("selection")


def test_train_source_preparation_never_hashes_parent_containers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent_market = tmp_path / "parent-market.csv.gz"
    parent_funding = tmp_path / "parent-funding.csv.gz"
    with gzip.open(parent_market, "wt", encoding="utf-8", newline="") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2020-01-01 00:00:00,100,101,99,100\n")
        handle.write("2020-01-01 00:05:00,100,101,99,100\n")
        handle.write("NOT_A_TIMESTAMP,FUTURE,FUTURE,FUTURE,FUTURE\n")
    with gzip.open(parent_funding, "wt", encoding="utf-8", newline="") as handle:
        handle.write(
            "funding_time_utc,symbol,funding_rate,settlement_mark_price\n"
        )
        handle.write("2020-01-01 00:00:00,BTCUSDT,0.0001,100\n")
        handle.write("NOT_A_TIMESTAMP,FUTURE,FUTURE,FUTURE\n")

    stage_dir = tmp_path / "train-source"
    stage_manifest = tmp_path / "train-source.json"
    parent_market_contract = {"path": str(parent_market), "sha256": "predeclared"}
    parent_funding_contract = {"path": str(parent_funding), "sha256": "predeclared"}
    train_end = evaluator.STAGE_WINDOWS["train"][1]
    assert train_end is not None
    spec = {
        "stage": "train",
        "required_manifest": str(stage_manifest),
        "required_protocol_version": (
            "quantity_lattice_cohort_disagreement_execution_source_v1"
        ),
        "physical_window": [
            evaluator.STAGE_WINDOWS["train"][0].isoformat(),
            train_end.isoformat(),
        ],
        "physical_rows_limited_to_window": True,
        "exit_boundary_required": False,
        "strategy_outcomes_calculated": False,
    }
    freeze = {
        "manifest_hash": "freeze",
        "execution_source_specs": {"train": spec},
        "legacy_container_contract": {
            "market": parent_market_contract,
            "funding": parent_funding_contract,
        },
    }
    monkeypatch.setattr(evaluator, "LEGACY_MARKET", parent_market)
    monkeypatch.setattr(evaluator, "LEGACY_FUNDING", parent_funding)
    monkeypatch.setitem(evaluator.STAGE_SOURCE_DIRS, "train", stage_dir)
    monkeypatch.setitem(evaluator.STAGE_SOURCE_MANIFESTS, "train", stage_manifest)
    monkeypatch.setattr(evaluator, "verify_evaluator_freeze", lambda: freeze)
    monkeypatch.setattr(evaluator, "_verified_prior_reports", lambda *args, **kwargs: [])
    monkeypatch.setattr(evaluator, "_expected_stage_rows", lambda *args, **kwargs: (2, 1))

    def parse_market(*args: object, **kwargs: object) -> tuple[pd.DataFrame, dict[str, Any]]:
        return pd.DataFrame(index=range(2)), {"rows": 2}

    def parse_funding(*args: object, **kwargs: object) -> tuple[pd.DataFrame, dict[str, Any]]:
        return pd.DataFrame(index=range(1)), {"rows": 1}

    monkeypatch.setattr(evaluator.strict_source, "_parse_market_window", parse_market)
    monkeypatch.setattr(evaluator.strict_source, "_parse_funding_window", parse_funding)
    real_sha = evaluator._sha256
    seen: list[Path] = []

    def reject_parent_hash(path: str | Path) -> str:
        candidate = Path(path)
        seen.append(candidate)
        if candidate in (parent_market, parent_funding):
            raise AssertionError("train hashed an unsliced parent outcome container")
        return real_sha(candidate)

    monkeypatch.setattr(evaluator, "_sha256", reject_parent_hash)
    report = evaluator.prepare_stage_source("train")

    assert parent_market not in seen
    assert parent_funding not in seen
    assert report["post_stage_numeric_rows_parsed"] == 0
    assert report["market"]["rows"] == 2
    assert report["funding"]["rows"] == 1
    assert stage_manifest.exists()


def test_stage_source_reload_hashes_stage_slices_not_parent_containers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage_dir = tmp_path / "train-source"
    stage_dir.mkdir()
    market_path = stage_dir / "BTCUSDT_5m.csv.gz"
    funding_path = stage_dir / "BTCUSDT_funding_marks.csv.gz"
    market_path.write_bytes(b"market-slice")
    funding_path.write_bytes(b"funding-slice")
    stage_manifest = tmp_path / "train-source.json"
    spec = {
        "required_protocol_version": (
            "quantity_lattice_cohort_disagreement_execution_source_v1"
        ),
        "physical_window": ["2020-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"],
        "physical_rows_limited_to_window": True,
        "exit_boundary_required": False,
        "strategy_outcomes_calculated": False,
    }
    parent_market = {"path": "parent-market", "sha256": "predeclared-market"}
    parent_funding = {"path": "parent-funding", "sha256": "predeclared-funding"}
    market_hash = hashlib.sha256(b"market-slice").hexdigest()
    funding_hash = hashlib.sha256(b"funding-slice").hexdigest()
    payload = evaluator._seal(
        {
            "protocol_version": spec["required_protocol_version"],
            "candidate": evaluator.POLICY_ID,
            "stage": "train",
            "evaluator_freeze_manifest_hash": "freeze",
            "physical_window": spec["physical_window"],
            "physical_rows_limited_to_window": True,
            "exit_boundary_required": False,
            "strategy_outcomes_calculated": False,
            "official_manifest_hashes_verified": True,
            "post_stage_numeric_rows_parsed": 0,
            "parent_market": parent_market,
            "parent_funding": parent_funding,
            "market": {"path": str(market_path), "sha256": market_hash, "rows": 2},
            "funding": {
                "path": str(funding_path),
                "sha256": funding_hash,
                "rows": 1,
            },
        }
    )
    stage_manifest.write_text(json.dumps(payload), encoding="utf-8")
    freeze = {
        "manifest_hash": "freeze",
        "execution_source_specs": {"train": spec},
        "legacy_container_contract": {
            "market": parent_market,
            "funding": parent_funding,
        },
    }
    monkeypatch.setitem(evaluator.STAGE_SOURCE_DIRS, "train", stage_dir)
    monkeypatch.setitem(evaluator.STAGE_SOURCE_MANIFESTS, "train", stage_manifest)
    monkeypatch.setattr(
        evaluator,
        "_rebuild_stage_source_identity",
        lambda *args, **kwargs: {
            "market": {"sha256": market_hash, "rows": 2},
            "funding": {"sha256": funding_hash, "rows": 1},
        },
    )
    seen: list[Path] = []
    real_sha = evaluator._sha256

    def reject_parent_hash(path: str | Path) -> str:
        candidate = Path(path)
        seen.append(candidate)
        if str(candidate) in (parent_market["path"], parent_funding["path"]):
            raise AssertionError("reload hashed an unsliced parent outcome container")
        return real_sha(candidate)

    monkeypatch.setattr(evaluator, "_sha256", reject_parent_hash)
    loaded = evaluator._load_stage_source("train", freeze=freeze)

    assert loaded == payload
    assert seen == [market_path, funding_path]


def test_stage_source_manifest_cannot_redirect_frozen_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage_dir = tmp_path / "train-source"
    stage_dir.mkdir()
    expected_funding = stage_dir / "BTCUSDT_funding_marks.csv.gz"
    expected_funding.write_bytes(b"funding")
    spec = evaluator._stage_source_spec(
        {"primary": evaluator.load_schedules()},
        "train",
    )
    parent_market = {"path": "parent-market", "sha256": "m"}
    parent_funding = {"path": "parent-funding", "sha256": "f"}
    freeze = {
        "manifest_hash": "freeze",
        "execution_source_specs": {"train": spec},
        "legacy_container_contract": {
            "market": parent_market,
            "funding": parent_funding,
        },
    }
    payload = evaluator._seal(
        {
            "protocol_version": spec["required_protocol_version"],
            "candidate": evaluator.POLICY_ID,
            "stage": "train",
            "evaluator_freeze_manifest_hash": "freeze",
            "physical_window": spec["physical_window"],
            "physical_rows_limited_to_window": True,
            "exit_boundary_required": spec["exit_boundary_required"],
            "strategy_outcomes_calculated": False,
            "official_manifest_hashes_verified": True,
            "post_stage_numeric_rows_parsed": 0,
            "parent_market": parent_market,
            "parent_funding": parent_funding,
            "market": {
                "path": str(tmp_path / "redirected.csv.gz"),
                "sha256": "x",
                "rows": 1,
            },
            "funding": {
                "path": str(expected_funding),
                "sha256": hashlib.sha256(b"funding").hexdigest(),
                "rows": 1,
            },
        }
    )
    manifest = tmp_path / "train-source.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setitem(evaluator.STAGE_SOURCE_DIRS, "train", stage_dir)
    monkeypatch.setitem(evaluator.STAGE_SOURCE_MANIFESTS, "train", manifest)
    with pytest.raises(ValueError, match="market path changed"):
        evaluator._load_stage_source("train", freeze=freeze)


def test_flat_trade_costs_match_frozen_half_leverage() -> None:
    market = _market()
    clock = _clock(market)
    start = evaluator._utc(evaluator._utc(market.iloc[0]["date"]).floor("D"))
    end = evaluator._utc(start + pd.Timedelta(days=2))
    base = evaluator.simulate_strict(
        market,
        _funding(),
        clock,
        start=start,
        end=end,
        cost_rate_per_side=evaluator.FROZEN_CONFIG.base_cost_notional_per_side,
    )
    stress = evaluator.simulate_strict(
        market,
        _funding(),
        clock,
        start=start,
        end=end,
        cost_rate_per_side=evaluator.FROZEN_CONFIG.stress_cost_notional_per_side,
    )
    assert base["absolute_return_pct"] == pytest.approx(-0.06, abs=1e-10)
    assert stress["absolute_return_pct"] == pytest.approx(-0.10, abs=1e-10)
    assert base["strict_mdd_pct"] == pytest.approx(0.06, abs=1e-10)


def test_simulation_requires_frozen_hold_and_config() -> None:
    market = _market()
    clock = _clock(market)
    start = evaluator._utc(evaluator._utc(market.iloc[0]["date"]).floor("D"))
    end = evaluator._utc(start + pd.Timedelta(days=2))
    with pytest.raises(ValueError, match="absent"):
        evaluator.simulate_strict(
            market.iloc[:-1],
            _funding(),
            clock,
            start=start,
            end=end,
            cost_rate_per_side=0.0006,
        )
    with pytest.raises(ValueError, match="configuration is frozen"):
        evaluator.simulate_strict(
            market,
            _funding(),
            clock,
            start=start,
            end=end,
            cost_rate_per_side=0.0006,
            cfg=replace(evaluator.FROZEN_CONFIG, hold_bars=12),
        )


def test_strict_mdd_marks_favorable_then_adverse_held_path() -> None:
    market = _market()
    market.loc[0, "high"] = 120.0
    market.loc[0, "low"] = 80.0
    metrics = evaluator.simulate_strict(
        market,
        _funding(),
        _clock(market),
        start=evaluator._utc(evaluator._utc(market.iloc[0]["date"]).floor("D")),
        end=evaluator._utc(
            evaluator._utc(market.iloc[0]["date"]).floor("D")
            + pd.Timedelta(days=2)
        ),
        cost_rate_per_side=0.0,
    )
    assert metrics["strict_mdd_pct"] == pytest.approx(
        (1.0 - 0.90 / 1.10) * 100.0
    )


def test_funding_boundary_drops_credits_and_keeps_debits() -> None:
    market = _market()
    clock = _clock(market)
    entry = evaluator._utc(clock.iloc[0]["entry_time"])
    exit_time = evaluator._utc(clock.iloc[0]["exit_time"])
    funding = _funding(
        (entry, 0.001),
        (evaluator._utc(entry + pd.Timedelta(hours=8)), -0.001),
        (exit_time, -0.001),
    )
    metrics = evaluator.simulate_strict(
        market,
        funding,
        clock,
        start=evaluator._utc(entry.floor("D")),
        end=evaluator._utc(entry.floor("D") + pd.Timedelta(days=2)),
        cost_rate_per_side=0.0,
    )
    detail = metrics["trade_details"][0]
    assert detail["visited_funding_events"] == 3
    assert detail["funding_events"] == 2
    assert detail["dropped_boundary_funding_credits"] == 1
    assert detail["funding_cash"] == pytest.approx(0.0, abs=1e-15)


def test_stage_gates_enforce_all_frozen_thresholds() -> None:
    base = _metrics()
    stress = _metrics(cagr_to_strict_mdd=2.5)
    train_subperiods = {name: _metrics() for name in evaluator.SUBPERIOD_WINDOWS["train"]}
    selection_subperiods = {
        name: _metrics() for name in evaluator.SUBPERIOD_WINDOWS["selection"]
    }
    assert all(
        evaluator._stage_gates("train", base, stress, train_subperiods).values()
    )
    assert all(
        evaluator._stage_gates(
            "selection", base, stress, selection_subperiods
        ).values()
    )

    failed_base = _metrics(
        absolute_return_pct=0.0,
        cagr_to_strict_mdd=2.999,
        strict_mdd_pct=15.001,
        mean_gross_underlying_bp=23.999,
        weekly_p=0.1,
    )
    failed_stress = _metrics(
        absolute_return_pct=0.0,
        cagr_to_strict_mdd=2.499,
    )
    failed_subperiods = {
        **selection_subperiods,
        "2023_h1": _metrics(absolute_return_pct=0.0),
    }
    checks = evaluator._stage_gates(
        "selection",
        failed_base,
        failed_stress,
        failed_subperiods,
    )
    assert not any(checks.values())


def test_prior_failure_keeps_selection_sealed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    failed = evaluator._seal({"stage": "train", "stage_passed": False})
    path = tmp_path / "train.json"
    path.write_text(json.dumps(failed), encoding="utf-8")
    monkeypatch.setitem(evaluator.STAGE_OUTPUTS, "train", path)
    with pytest.raises(ValueError, match="train did not pass"):
        evaluator._verified_prior_reports("selection", freeze_hash="freeze")


def test_result_and_doc_roll_back_when_second_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "train.json"
    document = tmp_path / "train.md"
    monkeypatch.setitem(evaluator.STAGE_OUTPUTS, "train", output)
    monkeypatch.setitem(evaluator.STAGE_DOCS, "train", document)
    monkeypatch.setattr(evaluator, "_build_stage_report", lambda stage: {"stage": stage})
    monkeypatch.setattr(evaluator, "render_stage_doc", lambda report: "doc")
    real_link = evaluator.os.link
    calls = 0

    def fail_second(source: str | Path, target: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated document failure")
        real_link(source, target)

    monkeypatch.setattr(evaluator.os, "link", fail_second)
    with pytest.raises(OSError, match="document failure"):
        evaluator.evaluate_stage("train")
    assert not output.exists()
    assert not document.exists()


def test_sha_and_seal_are_canonical_and_tamper_evident(tmp_path: Path) -> None:
    path = tmp_path / "payload.txt"
    path.write_text("abc", encoding="utf-8")
    assert evaluator._sha256(path) == hashlib.sha256(b"abc").hexdigest()
    assert evaluator._seal({"b": 2, "a": 1}) == evaluator._seal({"a": 1, "b": 2})
    sealed = evaluator._seal({"a": 1, "b": 2})
    assert sealed["manifest_hash"] != evaluator._seal({"a": 99, "b": 2})[
        "manifest_hash"
    ]
