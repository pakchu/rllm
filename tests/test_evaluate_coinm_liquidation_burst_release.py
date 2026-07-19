from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import evaluate_coinm_liquidation_burst_release as evaluator


def _market(periods: int = 25, start: str = "2023-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
        }
    )


def _clock(
    market: pd.DataFrame,
    *,
    entry_position: int = 0,
    direction: int = 1,
    stop_price: float = 90.0,
) -> dict[str, Any]:
    entry = pd.Timestamp(market.iloc[entry_position]["date"])
    exit_time = pd.Timestamp(market.iloc[entry_position + 24]["date"])
    return {
        "candidate": "CLBR-24",
        "split": "train",
        "entry_time": entry,
        "planned_exit_time": exit_time,
        "direction": direction,
        "stop_price": stop_price,
    }


def _funding(rows: list[tuple[pd.Timestamp, float, float]] | None = None) -> pd.DataFrame:
    rows = rows or []
    return pd.DataFrame(
        {
            "funding_time": pd.Series(
                [row[0] for row in rows], dtype="datetime64[ns]"
            ),
            "funding_rate": [row[1] for row in rows],
            "settlement_mark_price": [row[2] for row in rows],
        }
    )


def _simulate(
    market: pd.DataFrame,
    clocks: list[dict[str, Any]],
    *,
    funding: pd.DataFrame | None = None,
    cost: float = 0.0,
    end: str = "2024-01-01",
) -> dict[str, Any]:
    return evaluator.simulate_strict(
        market,
        _funding() if funding is None else funding,
        pd.DataFrame(clocks),
        start=evaluator._timestamp("2023-01-01"),
        end=evaluator._timestamp(end),
        leverage=1.0,
        cost_rate_per_side=cost,
    )


def test_static_dependency_hashes_and_source_contract_replay() -> None:
    support, execution = evaluator._verify_static_dependencies()
    assert support["manifest_hash"] == evaluator.SUPPORT_MANIFEST_HASH
    assert execution["protocol"]["outcomes_opened"] is False


def test_invalid_crossed_structural_stops_are_skipped_before_sizing() -> None:
    market = _market()
    result = _simulate(
        market,
        [
            _clock(market, direction=1, stop_price=100.0),
            _clock(market, direction=-1, stop_price=100.0),
        ],
    )
    assert result["metrics"]["candidate_clocks"] == 2
    assert result["metrics"]["invalid_crossed_stops"] == 2
    assert result["metrics"]["executable_trades"] == 0
    assert result["metrics"]["absolute_return_pct"] == 0.0


def test_flat_trade_costs_are_exactly_six_and_twelve_bp_per_side() -> None:
    market = _market()
    clocks = [_clock(market)]
    base = _simulate(market, clocks, cost=0.0006)
    stress = _simulate(market, clocks, cost=0.0012)
    assert base["metrics"]["absolute_return_pct"] == pytest.approx(-0.12)
    assert stress["metrics"]["absolute_return_pct"] == pytest.approx(-0.24)
    assert base["trades"][0]["net_return"] == pytest.approx(-0.0012)
    assert stress["trades"][0]["net_return"] == pytest.approx(-0.0024)


def test_intrabar_stop_marks_favorable_extreme_before_stop() -> None:
    market = _market()
    market.loc[0, ["high", "low"]] = [110.0, 89.0]
    result = _simulate(market, [_clock(market, stop_price=90.0)])
    trade = result["trades"][0]
    assert trade["exit_reason"] == "intrabar_stop"
    assert trade["exit_price"] == 90.0
    assert result["metrics"]["strict_mdd_pct"] == pytest.approx(
        (1.10 - 0.90) / 1.10 * 100.0
    )


@pytest.mark.parametrize(
    ("direction", "stop", "gap_open"), [(1, 90.0, 89.0), (-1, 110.0, 111.0)]
)
def test_gap_stop_uses_open_and_ignores_post_exit_extremes(
    direction: int, stop: float, gap_open: float
) -> None:
    market = _market()
    market.loc[1, ["open", "high", "low", "close"]] = [
        gap_open,
        200.0,
        1.0,
        gap_open,
    ]
    result = _simulate(
        market,
        [_clock(market, direction=direction, stop_price=stop)],
    )
    trade = result["trades"][0]
    assert trade["exit_reason"] == "gap_stop"
    assert trade["exit_price"] == gap_open
    assert result["metrics"]["strict_mdd_pct"] == pytest.approx(11.0)


def test_quantity_is_fixed_from_pre_entry_equity_and_compounds_between_trades() -> None:
    market = _market(periods=50)
    market.loc[24, ["open", "high", "low", "close"]] = 110.0
    market.loc[49, ["open", "high", "low", "close"]] = 110.0
    clocks = [_clock(market), _clock(market, entry_position=25)]
    result = _simulate(market, clocks)
    first, second = result["trades"]
    assert first["fixed_quantity"] == pytest.approx(0.01)
    assert first["post_exit_equity"] == pytest.approx(1.10)
    assert second["fixed_quantity"] == pytest.approx(0.011)
    assert result["metrics"]["absolute_return_pct"] == pytest.approx(21.0)


def test_global_hwm_persists_into_later_trade() -> None:
    market = _market(periods=50)
    market.loc[24, ["open", "high", "low", "close"]] = 110.0
    market.loc[25, ["high", "low"]] = [110.0, 89.0]
    clocks = [
        _clock(market),
        _clock(market, entry_position=25, stop_price=90.0),
    ]
    result = _simulate(market, clocks)
    assert result["trades"][1]["post_exit_equity"] == pytest.approx(0.99)
    assert result["metrics"]["strict_mdd_pct"] == pytest.approx(
        (1.21 - 0.99) / 1.21 * 100.0
    )


@pytest.mark.parametrize(("direction", "expected"), [(1, -0.2), (-1, 0.2)])
def test_funding_uses_entry_exclusive_exit_inclusive_and_correct_sign(
    direction: int, expected: float
) -> None:
    market = _market()
    entry = evaluator._timestamp(market.iloc[0]["date"])
    exit_time = evaluator._timestamp(market.iloc[24]["date"])
    funding = _funding(
        [
            (entry, 0.001, 100.0),
            (evaluator._timestamp(entry + pd.Timedelta(milliseconds=1)), 0.001, 100.0),
            (exit_time, 0.001, 100.0),
            (
                evaluator._timestamp(exit_time + pd.Timedelta(milliseconds=1)),
                0.001,
                100.0,
            ),
        ]
    )
    stop = 90.0 if direction > 0 else 110.0
    result = _simulate(
        market,
        [_clock(market, direction=direction, stop_price=stop)],
        funding=funding,
    )
    trade = result["trades"][0]
    assert trade["funding_events"] == 2
    assert trade["funding_cash"] == pytest.approx(expected / 100.0)
    assert result["metrics"]["absolute_return_pct"] == pytest.approx(expected)


def test_full_calendar_cagr_counts_idle_time() -> None:
    market = _market()
    market.loc[24, ["open", "high", "low", "close"]] = 110.0
    result = _simulate(market, [_clock(market)])
    years = pd.Timedelta(days=365).total_seconds() / evaluator.YEAR_SECONDS
    expected_cagr = (1.10 ** (1.0 / years) - 1.0) * 100.0
    assert result["metrics"]["calendar_years"] == pytest.approx(years)
    assert result["metrics"]["cagr_pct"] == pytest.approx(expected_cagr)
    assert result["metrics"]["exposure_pct"] < 0.1


def test_stationary_bootstrap_is_deterministic_and_centered() -> None:
    returns = np.asarray([0.01, -0.005, 0.007, 0.002, -0.001], dtype=float)
    first = evaluator.stationary_bootstrap_p_value(
        returns, mean_block_trades=8, resamples=500, seed=20_260_719
    )
    second = evaluator.stationary_bootstrap_p_value(
        returns, mean_block_trades=8, resamples=500, seed=20_260_719
    )
    assert first == second
    assert first["observed_mean_net_return"] == pytest.approx(float(returns.mean()))
    assert first["one_sided_p_value"] == (1 + first["exceedances"]) / 501
    assert evaluator.EvaluationConfig().bootstrap_resamples == 10_000
    assert evaluator.EvaluationConfig().bootstrap_mean_block_trades == 8
    assert evaluator.EvaluationConfig().bootstrap_seed == 20_260_719


def test_stage_loader_reads_only_requested_physical_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = evaluator.EvaluationConfig()
    execution = {
        "files": {
            "train": {
                "market": {"path": "train-market", "sha256": "m"},
                "funding": {"path": "train-funding", "sha256": "f"},
            },
            "test": {
                "market": {"path": "sealed-test-market"},
                "funding": {"path": "sealed-test-funding"},
            },
            "eval": {
                "market": {"path": "sealed-eval-market"},
                "funding": {"path": "sealed-eval-funding"},
            },
        }
    }
    freeze = {
        "split_clocks": {
            "train": {"path": "train-clock", "sha256": "c"},
            "test": {"path": "sealed-test-clock"},
            "eval": {"path": "sealed-eval-clock"},
        }
    }
    opened: list[str] = []

    def fake_read_csv(path: str, **_: object) -> pd.DataFrame:
        opened.append(path)
        return pd.DataFrame()

    monkeypatch.setattr(evaluator, "_read_json", lambda _: execution)
    source_hashes = {
        "train-market": evaluator.EXECUTION_FILE_SHA256["train"]["market"],
        "train-funding": evaluator.EXECUTION_FILE_SHA256["train"]["funding"],
        "train-clock": "c",
    }
    monkeypatch.setattr(evaluator, "_sha256", lambda path: source_hashes[str(path)])
    monkeypatch.setattr(evaluator.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(evaluator, "_validate_market", lambda *args: None)
    monkeypatch.setattr(evaluator, "_validate_funding", lambda *args: None)
    monkeypatch.setattr(evaluator, "_validate_clocks", lambda *args: None)
    evaluator._load_stage_inputs("train", cfg, freeze)
    assert opened == ["train-market", "train-funding", "train-clock"]


def test_freeze_path_never_loads_stage_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    support = {"manifest_hash": evaluator.SUPPORT_MANIFEST_HASH}
    execution = {
        "files": {
            stage: {
                kind: {"path": f"{stage}-{kind}"}
                for kind in ("market", "funding")
            }
            for stage in evaluator.STAGES
        }
    }
    monkeypatch.setattr(evaluator.Path, "exists", lambda _: False)
    monkeypatch.setattr(
        evaluator, "_verify_static_dependencies", lambda: (support, execution)
    )
    def fake_sha(path: str | Path) -> str:
        value = str(path)
        for stage in evaluator.STAGES:
            for kind in ("market", "funding"):
                if value == f"{stage}-{kind}":
                    return evaluator.EXECUTION_FILE_SHA256[stage][kind]
        return "frozen-sha"

    monkeypatch.setattr(evaluator, "_sha256", fake_sha)
    monkeypatch.setattr(
        evaluator,
        "_freeze_split_clocks",
        lambda *_: {
            stage: {"path": f"{stage}-clock", "sha256": "clock-sha"}
            for stage in evaluator.STAGES
        },
    )
    monkeypatch.setattr(
        evaluator, "_write_json_exclusive", lambda _, payload: captured.update(payload)
    )
    monkeypatch.setattr(
        evaluator,
        "_load_stage_inputs",
        lambda *_: (_ for _ in ()).throw(AssertionError("outcomes opened")),
    )
    report = evaluator.freeze_evaluator()
    assert report["opened_windows"] == []
    assert report["sealed_windows"] == ["train", "test", "eval"]
    assert report["simulation_run"] is False
    assert captured["freeze_hash"] == report["freeze_hash"]


def test_exclusive_json_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "write-once.json"
    evaluator._write_json_exclusive(output, {"first": True})
    with pytest.raises(FileExistsError):
        evaluator._write_json_exclusive(output, {"first": False})
    assert json.loads(output.read_text()) == {"first": True}


def test_noncanonical_evaluator_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="frozen"):
        evaluator._require_canonical_config(
            replace(evaluator.EvaluationConfig(), leverage=2.0)
        )


def test_forged_self_hashed_freeze_cannot_replace_derived_split_clocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = evaluator.EvaluationConfig()
    trusted_split = {
        stage: {
            "path": str(evaluator._split_clock_path(cfg, stage)),
            "sha256": f"trusted-{stage}",
            "rows": evaluator.EXPECTED_CLOCK_ROWS[stage],
            "start_inclusive": str(pd.Timestamp(evaluator.SPLITS[stage][0])),
            "end_exclusive": str(pd.Timestamp(evaluator.SPLITS[stage][1])),
        }
        for stage in evaluator.STAGES
    }
    forged = {
        "schema_version": evaluator.SCHEMA_VERSION,
        "created_at": "2026-07-19T00:00:00+00:00",
        "support_commit": evaluator.SUPPORT_COMMIT,
        "execution_source_commit": evaluator.EXECUTION_SOURCE_COMMIT,
        "evaluation_source_sha256": "evaluator-source",
        "static_input_sha256": evaluator.STATIC_INPUT_SHA256,
        "execution_file_sha256": evaluator.EXECUTION_FILE_SHA256,
        "split_clocks": {
            stage: {
                **metadata,
                "path": f"/tmp/forged-{stage}.csv.gz",
                "sha256": f"forged-{stage}",
            }
            for stage, metadata in trusted_split.items()
        },
        "config": asdict(cfg),
        "opened_windows": [],
        "sealed_windows": list(evaluator.STAGES),
        "candidate_returns_computed_before_freeze": False,
        "simulation_run": False,
        "mutable_parameters": [],
    }
    forged["freeze_hash"] = evaluator._stable_hash(forged, "freeze_hash")
    monkeypatch.setattr(evaluator, "_read_json", lambda _: forged)
    monkeypatch.setattr(
        evaluator,
        "_verify_static_dependencies",
        lambda: ({"manifest_hash": evaluator.SUPPORT_MANIFEST_HASH}, {}),
    )
    monkeypatch.setattr(
        evaluator,
        "_expected_split_clock_artifacts",
        lambda *_: (trusted_split, {}),
    )
    monkeypatch.setattr(evaluator, "_sha256", lambda _: "evaluator-source")
    with pytest.raises(ValueError, match="does not reproduce"):
        evaluator.verify_evaluator_freeze(cfg)


def test_forged_self_hashed_prior_result_must_reproduce_from_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged: dict[str, Any] = {
        "schema_version": evaluator.SCHEMA_VERSION,
        "created_at": "2026-07-19T00:00:00+00:00",
        "stage": "train",
        "evaluator_freeze_hash": "freeze-ok",
        "evaluation_source_sha256": "source-ok",
        "protocol": {},
        "window": {},
        "base": {},
        "stress": {},
        "bootstrap": {},
        "promotion": {"passes": True},
    }
    forged["result_hash"] = evaluator._stable_hash(forged, "result_hash")
    recomputed = {**forged, "window": {"start_inclusive": "real"}}
    recomputed["result_hash"] = evaluator._stable_hash(recomputed, "result_hash")
    called: list[str] = []

    def fake_compute(
        stage: str,
        _cfg: evaluator.EvaluationConfig,
        _freeze: dict[str, Any],
        *,
        created_at: str,
    ) -> dict[str, Any]:
        called.append(stage)
        assert created_at == forged["created_at"]
        return recomputed

    monkeypatch.setattr(evaluator, "_read_json", lambda _: forged)
    monkeypatch.setattr(evaluator, "_compute_stage_report", fake_compute)
    with pytest.raises(ValueError, match="does not reproduce"):
        evaluator._verify_prior_result(
            "train", evaluator.EvaluationConfig(), {"freeze_hash": "freeze-ok"}
        )
    assert called == ["train"]


def test_gate_contract_requires_test_significance_and_eval_ratio_three() -> None:
    contract = evaluator.promotion_gate_contract()
    assert contract["test"]["maximum_bootstrap_p_value"] == 0.10
    assert contract["eval"]["minimum_cagr_to_strict_mdd"] == 3.0
    assert contract["train"]["minimum_executable_trades"] == 30
