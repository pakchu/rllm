from __future__ import annotations

import gzip
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from training import evaluate_btcdom_leverage_polarity_decomposition as evaluator


def _market(
    *, start: str = "2022-03-01T04:05:00Z", periods: int = 145
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.full(periods, 100.0),
            "high": np.full(periods, 100.0),
            "low": np.full(periods, 100.0),
            "close": np.full(periods, 100.0),
        }
    )


def _clock(market: pd.DataFrame, *, side: int = 1) -> dict[str, Any]:
    entry = evaluator._utc(market.iloc[0]["date"])
    exit_time = evaluator._utc(
        market.iloc[evaluator.FROZEN_CONFIG.hold_bars]["date"]
    )
    return {
        "candidate": evaluator.POLICY_ID,
        "control": "primary",
        "split": "synthetic",
        "decision_time": entry - pd.Timedelta(minutes=5),
        "feature_available_time": entry - pd.Timedelta(minutes=5),
        "entry_time": entry,
        "exit_time": exit_time,
        "side": side,
    }


def _funding() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "funding_time": pd.to_datetime([], utc=True),
            "symbol": pd.Series([], dtype="object"),
            "funding_rate": pd.Series([], dtype=float),
            "settlement_mark_price": pd.Series([], dtype=float),
        }
    )


def test_evaluation_clock_derivation_is_deterministic_and_contained() -> None:
    first = evaluator.derive_evaluation_clocks()
    second = evaluator.derive_evaluation_clocks()
    pd.testing.assert_frame_equal(first, second)

    assert set(first["control"]) == set(evaluator.ALL_CONTROLS)
    counts = first.groupby(["control", "split"]).size()
    assert counts[("primary", "2022")] == 237
    assert counts[("primary", "2023")] == 184
    assert counts[("btc_only_tail", "2022")] == 566
    assert counts[("dom_only_mirror", "2023")] == 488

    schedules = {
        control: cast(
            pd.DataFrame,
            first.loc[first["control"].eq(control)].reset_index(drop=True),
        )
        for control in evaluator.ALL_CONTROLS
    }
    primary = schedules["primary"]
    for control in evaluator.DERIVED_SAME_CLOCK_CONTROLS:
        assert schedules[control][["decision_time", "entry_time", "exit_time"]].equals(
            primary[["decision_time", "entry_time", "exit_time"]]
        )
    assert schedules["direction_flip"]["side"].eq(-primary["side"]).all()
    delayed = schedules["extra_latency_1h"]
    assert delayed["entry_time"].eq(primary["entry_time"] + pd.Timedelta(hours=1)).all()
    assert delayed["exit_time"].eq(primary["exit_time"] + pd.Timedelta(hours=1)).all()
    assert delayed["side"].eq(primary["side"]).all()

    for control, schedule in schedules.items():
        expected_delay = pd.Timedelta(
            minutes=65 if control == "extra_latency_1h" else 5
        )
        assert schedule["entry_time"].sub(schedule["decision_time"]).eq(expected_delay).all()
        assert schedule["exit_time"].sub(schedule["entry_time"]).eq(pd.Timedelta(hours=12)).all()
        entries = schedule["entry_time"].reset_index(drop=True)
        exits = schedule["exit_time"].reset_index(drop=True)
        assert entries.iloc[1:].ge(exits.iloc[:-1].to_numpy()).all()


def test_random_control_uses_frozen_ascii_sha_contract() -> None:
    timestamp = pd.Timestamp("2022-01-05T17:00:00Z")
    expected_nibble = int(
        __import__("hashlib")
        .sha256(b"DLPD-12|2022-01-05T17:00:00Z")
        .hexdigest()[0],
        16,
    )
    expected = 1 if expected_nibble % 2 == 0 else -1
    assert evaluator._deterministic_random_side(timestamp) == expected


def test_evaluator_is_directly_executable_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(evaluator.EVALUATOR_SOURCE), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--freeze" in completed.stdout
    assert "--prepare-stage-source" in completed.stdout


def test_freeze_opens_no_market_funding_or_simulation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []
    real_sha = evaluator._sha256

    def tracking_sha(path: str | Path) -> str:
        seen.append(str(path))
        return real_sha(path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("freeze tried to open an execution outcome")

    for stage in evaluator.STAGE_ORDER:
        monkeypatch.setitem(
            evaluator.STAGE_OUTPUTS, stage, tmp_path / f"unused-{stage}.json"
        )
    monkeypatch.setattr(evaluator, "_sha256", tracking_sha)
    monkeypatch.setattr(evaluator, "prepare_stage_source", forbidden)
    monkeypatch.setattr(evaluator, "load_execution_window", forbidden)
    monkeypatch.setattr(evaluator, "simulate_strict", forbidden)
    output = tmp_path / "freeze.json"
    clocks = tmp_path / "evaluation-clocks.csv.gz"

    report = evaluator.freeze_evaluator(output, evaluation_clock_path=clocks)

    assert report["opened_windows"] == []
    assert report["sealed_windows"] == list(evaluator.STAGE_ORDER)
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["execution_outcome_data_bytes_hashed_during_freeze"] is False
    assert str(evaluator.LEGACY_MARKET) not in seen
    assert str(evaluator.LEGACY_FUNDING) not in seen
    assert evaluator.verify_evaluator_freeze(output) == report

    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    core["evaluation_config"] = {**core["evaluation_config"], "leverage": 1.0}
    output.write_text(json.dumps(evaluator._seal(core)), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        evaluator.verify_evaluator_freeze(output)


def test_freeze_is_write_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for stage in evaluator.STAGE_ORDER:
        monkeypatch.setitem(
            evaluator.STAGE_OUTPUTS, stage, tmp_path / f"unused-{stage}.json"
        )
    output = tmp_path / "freeze.json"
    clocks = tmp_path / "evaluation-clocks.csv.gz"
    evaluator.freeze_evaluator(output, evaluation_clock_path=clocks)
    with pytest.raises(FileExistsError):
        evaluator.freeze_evaluator(output, evaluation_clock_path=clocks)


def test_slice_copies_only_declared_window_without_parsing_values(tmp_path: Path) -> None:
    source = tmp_path / "source.csv.gz"
    output = tmp_path / "slice.csv.gz"
    second_output = tmp_path / "second-slice.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2021-12-31 23:55:00,SECRET,SECRET,SECRET,SECRET\n")
        handle.write("2022-01-01 00:00:00,100,101,99,100\n")
        handle.write("2022-01-01 00:05:00,100,101,99,100\n")
        handle.write("2023-01-01 00:00:00,FUTURE,FUTURE,FUTURE,FUTURE\n")

    result = evaluator._slice_gzip_csv(
        source,
        output,
        timestamp_column="date",
        start=pd.Timestamp("2022-01-01T00:00:00Z"),
        end=pd.Timestamp("2023-01-01T00:00:00Z"),
    )

    assert result["rows"] == 2
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        content = handle.read()
    assert "SECRET" not in content
    assert "FUTURE" not in content
    assert "2022-01-01 00:05:00" in content
    second = evaluator._slice_gzip_csv(
        source,
        second_output,
        timestamp_column="date",
        start=pd.Timestamp("2022-01-01T00:00:00Z"),
        end=pd.Timestamp("2023-01-01T00:00:00Z"),
    )
    assert result["sha256"] == second["sha256"]


def test_stage_boundary_requirement_is_derived_from_frozen_clocks() -> None:
    clocks = evaluator.derive_evaluation_clocks()
    schedules = {
        control: clocks.loc[clocks["control"].eq(control)].reset_index(drop=True)
        for control in evaluator.ALL_CONTROLS
    }
    assert evaluator._stage_exit_boundary_required("train", schedules) is False
    assert evaluator._stage_exit_boundary_required("test", schedules) is False

    boundary = schedules["primary"].loc[
        schedules["primary"]["split"].eq("2022")
    ].iloc[:1].copy()
    boundary["exit_time"] = evaluator.STAGE_WINDOWS["train"][1]
    assert evaluator._stage_exit_boundary_required(
        "train", {"boundary": boundary}
    ) is True


def test_phase_two_is_blocked_before_any_execution_source_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("phase-two block ran too late")

    monkeypatch.setattr(evaluator, "verify_evaluator_freeze", forbidden)
    monkeypatch.setattr(evaluator, "_load_stage_source", forbidden)
    with pytest.raises(RuntimeError, match="phase-two sealed"):
        evaluator.load_execution_window("eval")


def test_test_source_preparation_checks_prior_gate_before_parent_digest(
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
        raise AssertionError("parent digest ran before prior gate")

    monkeypatch.setattr(evaluator, "_verified_prior_reports", failed_prior)
    monkeypatch.setattr(evaluator, "_sha256", forbidden_digest)
    with pytest.raises(ValueError, match="train did not pass"):
        evaluator.prepare_stage_source("test")


def test_stage_source_manifest_cannot_redirect_frozen_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = evaluator._stage_source_spec(
        "train", exit_boundary_required=False
    )
    parent_market = {
        "path": str(evaluator.LEGACY_MARKET),
        "sha256": evaluator.LEGACY_MARKET_SHA256,
        "manifest": str(evaluator.LEGACY_MARKET_MANIFEST),
        "manifest_sha256": evaluator.LEGACY_MARKET_MANIFEST_SHA256,
    }
    parent_funding = {
        "path": str(evaluator.LEGACY_FUNDING),
        "sha256": evaluator.LEGACY_FUNDING_SHA256,
        "manifest": str(evaluator.LEGACY_FUNDING_MANIFEST),
        "manifest_sha256": evaluator.LEGACY_FUNDING_MANIFEST_SHA256,
    }
    payload = evaluator._seal(
        {
            "protocol_version": spec["required_protocol_version"],
            "candidate": evaluator.POLICY_ID,
            "stage": "train",
            "physical_window": spec["physical_window"],
            "physical_rows_limited_to_window": True,
            "exit_boundary_required": False,
            "strategy_outcomes_calculated": False,
            "official_checksums_verified": True,
            "full_parent_compressed_bytes_hashed": True,
            "post_stage_numeric_rows_parsed": 0,
            "parent_market": parent_market,
            "parent_funding": parent_funding,
            "market": {"path": "data/redirected.csv.gz", "sha256": "x", "rows": 1},
            "funding": {
                "path": str(
                    evaluator.STAGE_SOURCE_DIRS["train"]
                    / "BTCUSDT_funding_marks.csv.gz"
                ),
                "sha256": "y",
                "rows": 1,
            },
        }
    )
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setitem(evaluator.STAGE_SOURCE_MANIFESTS, "train", manifest)
    freeze = {
        "execution_source_specs": {"train": spec},
        "legacy_container_contract": {
            "market": parent_market,
            "funding": parent_funding,
        },
    }
    with pytest.raises(ValueError, match="market path changed"):
        evaluator._load_stage_source("train", freeze=freeze)


def test_stage_result_pair_rolls_back_json_when_doc_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "stage.json"
    document = tmp_path / "stage.md"
    monkeypatch.setitem(evaluator.STAGE_OUTPUTS, "train", output)
    monkeypatch.setitem(evaluator.STAGE_DOCS, "train", document)
    monkeypatch.setattr(evaluator, "_build_stage_report", lambda stage: {})
    monkeypatch.setattr(evaluator, "render_stage_doc", lambda report: "doc")
    real_write = evaluator._write_once_bytes
    calls = 0

    def fail_second(path: str | Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated doc failure")
        real_write(path, payload)

    monkeypatch.setattr(evaluator, "_write_once_bytes", fail_second)
    with pytest.raises(OSError, match="simulated doc failure"):
        evaluator.evaluate_stage("train")
    assert not output.exists()
    assert not document.exists()


def test_flat_trade_costs_match_frozen_half_leverage_accounting() -> None:
    market = _market()
    clock = pd.DataFrame([_clock(market)])
    start = evaluator._utc(market.iloc[0]["date"])
    end = start + pd.Timedelta(days=365.25)
    base = evaluator.simulate_strict(
        market,
        _funding(),
        clock,
        start=start,
        end=end,
        cost_rate_per_side=0.0006,
    )
    stress = evaluator.simulate_strict(
        market,
        _funding(),
        clock,
        start=start,
        end=end,
        cost_rate_per_side=0.0010,
    )
    assert base["absolute_return_pct"] == pytest.approx(-0.06)
    assert stress["absolute_return_pct"] == pytest.approx(-0.10)
    assert base["strict_mdd_pct"] == pytest.approx(0.06)
    assert stress["strict_mdd_pct"] == pytest.approx(0.10)


def test_unregistered_config_mutation_is_rejected() -> None:
    market = _market()
    with pytest.raises(ValueError, match="configuration is frozen"):
        evaluator.simulate_strict(
            market,
            _funding(),
            pd.DataFrame([_clock(market)]),
            start=evaluator._utc(market.iloc[0]["date"]),
            end=evaluator._utc(market.iloc[0]["date"]) + pd.Timedelta(days=365.25),
            cost_rate_per_side=0.0006,
            cfg=replace(evaluator.FROZEN_CONFIG, leverage=1.0),
        )


def test_frozen_primary_gates_are_applied_without_control_repair() -> None:
    prereg, _ = evaluator._verify_static_inputs()
    base = {
        "absolute_return_pct": 4.0,
        "cagr_to_strict_mdd": 3.5,
        "strict_mdd_pct": 10.0,
        "trades": 120,
        "weekly_cluster_signflip": {"p_value_two_sided": 0.08},
    }
    stress = {"absolute_return_pct": 0.1}
    halves = {"h1": {"absolute_return_pct": 0.1}, "h2": {"absolute_return_pct": 0.2}}
    controls = {"direction_flip": {"cagr_to_strict_mdd": 3.4}}
    checks = evaluator._stage_gates("train", base, stress, halves, controls, prereg)
    assert all(checks.values())

    failed = evaluator._stage_gates(
        "train",
        {**base, "absolute_return_pct": -0.1},
        stress,
        halves,
        {"direction_flip": {"cagr_to_strict_mdd": 99.0}},
        prereg,
    )
    assert failed["absolute_return_positive"] is False
    assert failed["direction_flip_inferior"] is False


def test_prior_failure_keeps_later_stage_sealed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failed = evaluator._seal(
        {
            "stage": "train",
            "stage_passed": False,
            "opened_windows": ["train"],
            "sealed_windows": ["test", "eval", "final"],
            "evaluator_freeze_manifest_hash": "freeze",
            "evaluator_source_sha256": evaluator._sha256(evaluator.EVALUATOR_SOURCE),
        }
    )
    path = tmp_path / "train.json"
    path.write_text(json.dumps(failed), encoding="utf-8")
    monkeypatch.setitem(evaluator.STAGE_OUTPUTS, "train", path)
    with pytest.raises(ValueError, match="did not pass"):
        evaluator._verified_prior_reports("test", freeze_hash="freeze")
