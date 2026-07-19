from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import evaluate_inverse_collateral_liquidation_absorption as evaluator


def _market(periods: int = 13, start: str = "2023-06-25") -> pd.DataFrame:
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


def _clock(
    market: pd.DataFrame,
    *,
    entry_position: int = 0,
    direction: int = 1,
    split: str = "train",
) -> dict[str, Any]:
    entry = pd.Timestamp(market.iloc[entry_position]["date"])
    exit_time = pd.Timestamp(market.iloc[entry_position + 12]["date"])
    return {
        "candidate": "ICLA-60",
        "split": split,
        "entry_time": entry,
        "planned_exit_time": exit_time,
        "direction": direction,
    }


def _funding(
    rows: list[tuple[pd.Timestamp, float, float]] | None = None,
) -> pd.DataFrame:
    rows = rows or []
    return pd.DataFrame(
        {
            "funding_time": pd.Series([row[0] for row in rows], dtype="datetime64[ns]"),
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
    end: str = "2024-06-25",
) -> dict[str, Any]:
    return evaluator.simulate_strict(
        market,
        _funding() if funding is None else funding,
        pd.DataFrame(clocks),
        start=evaluator._timestamp("2023-06-25"),
        end=evaluator._timestamp(end),
        leverage=1.0,
        cost_rate_per_side=cost,
    )


def test_api_contract_is_icla_not_clbr() -> None:
    cfg = evaluator.EvaluationConfig()
    assert evaluator.STAGES == ("train", "test", "eval")
    assert evaluator.BAR == pd.Timedelta(minutes=5)
    assert evaluator.HOLD_BARS == 12
    assert evaluator.CANDIDATE == "ICLA-60"
    assert evaluator.EXPECTED_CLOCK_ROWS == {"train": 30, "test": 111, "eval": 108}
    assert cfg.base_cost_rate_per_side == pytest.approx(0.0006)
    assert cfg.stress_cost_rate_per_side == pytest.approx(0.0010)
    assert not hasattr(cfg, "stop_price")


def test_fixed_time_exit_requires_exactly_twelve_bars_and_has_no_stop_path() -> None:
    market = _market(periods=25)
    market.loc[1, ["open", "high", "low", "close"]] = [10.0, 1_000.0, 1.0, 10.0]
    market.loc[12, ["open", "high", "low", "close"]] = [110.0, 110.0, 110.0, 110.0]

    result = _simulate(market, [_clock(market)])

    trade = result["trades"][0]
    assert trade["exit_reason"] == "time_exit"
    assert trade["exit_time"] == str(pd.Timestamp(market.loc[12, "date"]))
    assert trade["exit_price"] == 110.0
    assert trade["bars_held"] == 12
    assert "stop_price" not in trade
    assert result["metrics"]["invalid_crossed_stops"] == 0

    bad_clock = {**_clock(market), "planned_exit_time": market.loc[13, "date"]}
    with pytest.raises(ValueError, match="12 bars"):
        _simulate(market, [bad_clock])


def test_flat_trade_costs_are_six_and_ten_bp_per_side() -> None:
    market = _market()
    clocks = [_clock(market)]

    base = _simulate(market, clocks, cost=0.0006)
    stress = _simulate(market, clocks, cost=0.0010)

    assert base["metrics"]["absolute_return_pct"] == pytest.approx(-0.12)
    assert stress["metrics"]["absolute_return_pct"] == pytest.approx(-0.20)
    assert base["trades"][0]["net_return"] == pytest.approx(-0.0012)
    assert stress["trades"][0]["net_return"] == pytest.approx(-0.0020)


@pytest.mark.parametrize(
    ("direction", "funding_rate", "boundary_cash"),
    [
        (1, 0.001, -0.2),
        (-1, -0.001, -0.2),
    ],
)
def test_entry_and_exit_funding_boundary_debits_are_retained(
    direction: int, funding_rate: float, boundary_cash: float
) -> None:
    market = _market()
    entry = evaluator._timestamp(market.iloc[0]["date"])
    exit_time = evaluator._timestamp(market.iloc[12]["date"])
    funding = _funding(
        [
            (entry, funding_rate, 100.0),
            (exit_time, funding_rate, 100.0),
        ]
    )

    result = _simulate(market, [_clock(market, direction=direction)], funding=funding)
    trade = result["trades"][0]

    assert trade["funding_events"] == 2
    assert trade["funding_cash"] == pytest.approx(boundary_cash / 100.0)
    assert result["metrics"]["absolute_return_pct"] == pytest.approx(boundary_cash)


@pytest.mark.parametrize(
    ("direction", "funding_rate"),
    [
        (1, -0.001),
        (-1, 0.001),
    ],
)
def test_entry_and_exit_funding_boundary_credits_are_dropped(
    direction: int, funding_rate: float
) -> None:
    market = _market()
    entry = evaluator._timestamp(market.iloc[0]["date"])
    exit_time = evaluator._timestamp(market.iloc[12]["date"])
    funding = _funding(
        [
            (entry, funding_rate, 100.0),
            (exit_time, funding_rate, 100.0),
        ]
    )

    result = _simulate(market, [_clock(market, direction=direction)], funding=funding)
    trade = result["trades"][0]

    assert trade["funding_events"] == 0
    assert trade["funding_cash"] == 0.0
    assert result["metrics"]["absolute_return_pct"] == 0.0


def test_interior_funding_events_include_credits_and_debits_symmetrically() -> None:
    market = _market()
    entry = evaluator._timestamp(market.iloc[0]["date"])
    exit_time = evaluator._timestamp(market.iloc[12]["date"])
    funding = _funding(
        [
            (
                evaluator._timestamp(entry + pd.Timedelta(milliseconds=1)),
                -0.001,
                100.0,
            ),
            (
                evaluator._timestamp(exit_time - pd.Timedelta(milliseconds=1)),
                0.002,
                100.0,
            ),
        ]
    )

    result = _simulate(market, [_clock(market, direction=1)], funding=funding)
    trade = result["trades"][0]

    assert trade["funding_events"] == 2
    assert trade["funding_cash"] == pytest.approx(-0.001)
    assert result["metrics"]["absolute_return_pct"] == pytest.approx(-0.1)


def test_strict_mdd_orders_entry_fee_then_favorable_then_adverse_with_virtual_exit_cost() -> (
    None
):
    market = _market()
    market.loc[0, ["open", "high", "low", "close"]] = [100.0, 110.0, 90.0, 100.0]

    result = _simulate(market, [_clock(market)], cost=0.0006)

    entry_fee_equity = 1.0 - 0.0006
    high_water_after_favorable = entry_fee_equity + 0.10
    adverse_with_virtual_exit = entry_fee_equity - 0.10 - (90.0 / 100.0) * 0.0006
    expected_mdd = (
        (high_water_after_favorable - adverse_with_virtual_exit)
        / high_water_after_favorable
        * 100.0
    )
    assert result["metrics"]["strict_mdd_pct"] == pytest.approx(expected_mdd)


def test_short_path_marks_low_as_favorable_before_high_as_adverse() -> None:
    market = _market()
    market.loc[0, ["open", "high", "low", "close"]] = [100.0, 120.0, 80.0, 100.0]

    result = _simulate(market, [_clock(market, direction=-1)], cost=0.0)

    assert result["metrics"]["strict_mdd_pct"] == pytest.approx(
        (1.20 - 0.80) / 1.20 * 100.0
    )


def test_full_calendar_cagr_counts_idle_time() -> None:
    market = _market()
    market.loc[12, ["open", "high", "low", "close"]] = 110.0

    result = _simulate(market, [_clock(market)])

    years = (
        evaluator._timestamp("2024-06-25") - evaluator._timestamp("2023-06-25")
    ).total_seconds() / evaluator.YEAR_SECONDS
    expected_cagr = (1.10 ** (1.0 / years) - 1.0) * 100.0
    assert result["metrics"]["calendar_years"] == pytest.approx(years)
    assert result["metrics"]["cagr_pct"] == pytest.approx(expected_cagr)
    assert result["metrics"]["exposure_pct"] < 0.02


def test_stage_gates_match_preregistered_icla_thresholds() -> None:
    contract = evaluator.promotion_gate_contract()
    assert contract["train"] == {
        "absolute_return_positive": True,
        "minimum_cagr_to_strict_mdd": 2.0,
        "maximum_strict_mdd_pct": 15.0,
        "minimum_executable_trades": 25,
        "minimum_trades_per_side": 8,
        "stress_absolute_return_positive": True,
        "maximum_bootstrap_p_value": 0.10,
    }
    assert contract["test"]["minimum_executable_trades"] == 90
    assert contract["test"]["minimum_trades_per_side"] == 20
    assert contract["test"]["minimum_cagr_to_strict_mdd"] == 2.0
    assert contract["eval"]["minimum_executable_trades"] == 90
    assert contract["eval"]["minimum_trades_per_side"] == 20
    assert contract["eval"]["minimum_cagr_to_strict_mdd"] == 3.0
    assert contract["eval"]["maximum_bootstrap_p_value"] == 0.10


def test_evaluate_gates_enforce_side_counts_stress_and_bootstrap_on_all_stages() -> (
    None
):
    base = {
        "metrics": {
            "absolute_return_pct": 1.0,
            "cagr_to_strict_mdd": 3.0,
            "strict_mdd_pct": 10.0,
            "executable_trades": 90,
            "long_trades": 82,
            "short_trades": 8,
        }
    }
    stress = {"metrics": {"absolute_return_pct": 0.1}}
    bootstrap = {"one_sided_p_value": 0.05}

    train = evaluator._evaluate_gates("train", base, stress, bootstrap)
    test = evaluator._evaluate_gates("test", base, stress, bootstrap)
    eval_result = evaluator._evaluate_gates("eval", base, stress, bootstrap)

    assert train["passes"] is True
    assert test["passes"] is False
    assert test["checks"]["minimum_short_trades"] is False
    assert eval_result["passes"] is False
    assert eval_result["checks"]["minimum_short_trades"] is False

    failing_stress = {"metrics": {"absolute_return_pct": 0.0}}
    failing_bootstrap = {"one_sided_p_value": 0.1001}
    assert (
        evaluator._evaluate_gates("train", base, failing_stress, bootstrap)["passes"]
        is False
    )
    assert (
        evaluator._evaluate_gates("eval", base, stress, failing_bootstrap)["passes"]
        is False
    )


def test_random_control_is_deterministic_nonoverlapping_and_preserves_sides() -> None:
    primary = pd.DataFrame({"direction": [1, 1, 1, -1, -1]})

    first = evaluator._random_control_clocks(primary, "train", 0)
    second = evaluator._random_control_clocks(primary, "train", 0)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(primary)
    assert first["direction"].value_counts().to_dict() == {1: 3, -1: 2}
    assert bool(
        first["entry_time"]
        .iloc[1:]
        .reset_index(drop=True)
        .ge(first["planned_exit_time"].iloc[:-1].reset_index(drop=True))
        .all()
    )


def test_any_complete_mechanism_control_vetoes_primary_promotion() -> None:
    primary = {"checks": {"primary": True}, "passes": True}
    controls = {
        "direction_flip": {"complete_gate": {"passes": False}},
        "liquidation_only_fade": {"complete_gate": {"passes": True}},
    }

    result = evaluator._apply_mechanism_control_veto(primary, controls)

    assert result["checks"]["mechanism_controls_rejected"] is False
    assert result["passes"] is False


def test_clbr_alias_is_a_mandatory_source_only_promotion_precondition() -> None:
    support = {
        "support_passes": True,
        "clock_overlap": {
            "passes": True,
            "entry_jaccard": 0.007662835249042145,
            "maximum_entry_jaccard_allowed": 0.10,
            "clbr_clock_sha256": "clbr-clock",
        },
    }
    preconditions = evaluator._source_only_preconditions(support)
    promotion = {"checks": {}, "passes": True}

    result = evaluator._apply_source_only_preconditions(promotion, preconditions)

    assert result["checks"]["source_support_passes"] is True
    assert result["checks"]["clbr_alias_rejected"] is True
    assert result["passes"] is True

    support["clock_overlap"]["entry_jaccard"] = 0.100001
    with pytest.raises(ValueError, match="alias"):
        evaluator._source_only_preconditions(support)


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


def test_combined_clock_loader_uses_icla_source_only_clock_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = evaluator.EvaluationConfig()
    support = {"clocks": {"rows": 2}}
    read_kwargs: dict[str, Any] = {}
    clocks = pd.DataFrame(
        {
            "candidate": ["ICLA-60", "ICLA-60"],
            "split": ["train", "train"],
            "first_bar_open_time": pd.to_datetime(
                ["2023-06-25 00:00", "2023-06-25 01:00"]
            ),
            "last_bar_open_time": pd.to_datetime(
                ["2023-06-25 00:55", "2023-06-25 01:55"]
            ),
            "wave_completed_time": pd.to_datetime(
                ["2023-06-25 01:00", "2023-06-25 02:00"]
            ),
            "feature_available_time": pd.to_datetime(
                ["2023-06-25 01:00:01", "2023-06-25 02:00:01"]
            ),
            "entry_time": pd.to_datetime(["2023-06-25 01:05", "2023-06-25 02:05"]),
            "planned_exit_time": pd.to_datetime(
                ["2023-06-25 02:05", "2023-06-25 03:05"]
            ),
            "direction": [1, -1],
        }
    )

    def fake_read_csv(path: str, **kwargs: Any) -> pd.DataFrame:
        assert path == cfg.combined_clocks
        read_kwargs.update(kwargs)
        return clocks

    monkeypatch.setattr(
        evaluator,
        "_sha256",
        lambda path: evaluator.STATIC_INPUT_SHA256[str(evaluator.COMBINED_CLOCKS)],
    )
    monkeypatch.setattr(evaluator.pd, "read_csv", fake_read_csv)

    loaded = evaluator._load_combined_clocks(cfg, support)

    assert loaded.equals(clocks)
    assert read_kwargs["parse_dates"] == [
        "first_bar_open_time",
        "last_bar_open_time",
        "wave_completed_time",
        "feature_available_time",
        "entry_time",
        "planned_exit_time",
    ]
    assert "burst_time" not in read_kwargs["parse_dates"]
    assert "release_time" not in read_kwargs["parse_dates"]


def test_validate_clocks_requires_icla_candidate_twelve_bar_hold_and_no_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator, "EXPECTED_CLOCK_ROWS", {"train": 1})
    start = evaluator._timestamp("2023-06-25")
    end = evaluator._timestamp("2023-06-26")
    clocks = pd.DataFrame(
        {
            "candidate": ["ICLA-60"],
            "split": ["train"],
            "entry_time": [start],
            "planned_exit_time": [start + pd.Timedelta(hours=1)],
            "direction": [1],
        }
    )

    evaluator._validate_clocks(clocks, "train", start, end)

    with pytest.raises(ValueError, match="12 bars"):
        evaluator._validate_clocks(
            clocks.assign(planned_exit_time=[start + pd.Timedelta(minutes=65)]),
            "train",
            start,
            end,
        )
    with pytest.raises(ValueError, match="candidate"):
        evaluator._validate_clocks(
            clocks.assign(candidate=["CLBR-24"]),
            "train",
            start,
            end,
        )
    with pytest.raises(ValueError, match="stop"):
        evaluator._validate_clocks(
            clocks.assign(stop_price=[90.0]),
            "train",
            start,
            end,
        )


def test_freeze_writes_split_clocks_once_and_does_not_open_stage_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    support = {
        "manifest_hash": evaluator.SUPPORT_MANIFEST_HASH,
        "support_passes": True,
        "clock_overlap": {
            "passes": True,
            "entry_jaccard": 0.007662835249042145,
            "maximum_entry_jaccard_allowed": 0.10,
            "clbr_clock_sha256": "clbr-clock",
        },
    }
    execution = {
        "files": {
            stage: {kind: {"path": f"{stage}-{kind}"} for kind in ("market", "funding")}
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
    primary_frames = {stage: pd.DataFrame() for stage in evaluator.STAGES}
    monkeypatch.setattr(
        evaluator,
        "_expected_split_clock_artifacts",
        lambda *_: ({}, primary_frames),
    )
    monkeypatch.setattr(
        evaluator,
        "_freeze_control_clocks",
        lambda *_: {
            stage: {
                control: {"path": f"{stage}-{control}", "sha256": "control-sha"}
                for control in evaluator.CONTROL_NAMES
            }
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
    assert report["candidate_returns_computed_before_freeze"] is False
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


def test_later_stage_cannot_open_before_prior_stage_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = evaluator.EvaluationConfig()
    freeze = {"freeze_hash": "freeze-ok", "evaluation_source_sha256": "source-ok"}
    paths = {
        "train": Path("train-result"),
        "test": Path("test-result"),
        "eval": Path("eval-result"),
    }

    monkeypatch.setattr(evaluator, "verify_evaluator_freeze", lambda _: freeze)
    monkeypatch.setattr(evaluator, "_result_path", lambda _cfg, stage: paths[stage])
    monkeypatch.setattr(Path, "exists", lambda self: self == paths["train"])
    monkeypatch.setattr(
        evaluator,
        "_verify_prior_result",
        lambda *_: (_ for _ in ()).throw(
            ValueError("train failed; later windows remain sealed")
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "_compute_stage_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("test opened")),
    )

    with pytest.raises(ValueError, match="train failed"):
        evaluator.evaluate_stage("test", cfg)


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

    monkeypatch.setattr(evaluator, "_read_json", lambda _: forged)
    monkeypatch.setattr(
        evaluator, "_compute_stage_report", lambda *_args, **_kwargs: recomputed
    )

    with pytest.raises(ValueError, match="does not reproduce"):
        evaluator._verify_prior_result(
            "train", evaluator.EvaluationConfig(), {"freeze_hash": "freeze-ok"}
        )


def test_forged_self_hashed_freeze_cannot_replace_physical_split_clocks(
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
        "evaluation_source": str(evaluator.EVALUATOR_SOURCE),
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
        lambda: (
            {
                "manifest_hash": evaluator.SUPPORT_MANIFEST_HASH,
                "support_passes": True,
                "clock_overlap": {
                    "passes": True,
                    "entry_jaccard": 0.007662835249042145,
                    "maximum_entry_jaccard_allowed": 0.10,
                    "clbr_clock_sha256": "clbr-clock",
                },
            },
            {},
        ),
    )
    monkeypatch.setattr(
        evaluator, "_expected_split_clock_artifacts", lambda *_: (trusted_split, {})
    )
    monkeypatch.setattr(
        evaluator,
        "_expected_control_clock_artifacts",
        lambda *_: (
            {
                stage: {
                    control: {
                        "path": f"trusted-{stage}-{control}",
                        "sha256": f"trusted-{stage}-{control}",
                    }
                    for control in evaluator.CONTROL_NAMES
                }
                for stage in evaluator.STAGES
            },
            {},
        ),
    )
    monkeypatch.setattr(evaluator, "_sha256", lambda _: "evaluator-source")

    with pytest.raises(ValueError, match="does not reproduce"):
        evaluator.verify_evaluator_freeze(cfg)
