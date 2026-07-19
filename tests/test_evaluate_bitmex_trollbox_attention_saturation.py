from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import evaluate_bitmex_trollbox_attention_saturation as evaluator


def _market(
    *,
    start: str = "2022-01-01T00:00:00Z",
    periods: int = 25,
    price: float = 100.0,
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.full(periods, price),
            "high": np.full(periods, price),
            "low": np.full(periods, price),
            "close": np.full(periods, price),
        }
    )


def _clock(
    market: pd.DataFrame,
    *,
    side: int = 1,
    entry_position: int = 0,
) -> dict[str, Any]:
    entry = evaluator._timestamp(market.iloc[entry_position]["date"])
    exit_time = evaluator._timestamp(
        market.iloc[
            entry_position + evaluator.EvaluationConfig().hold_bars
        ]["date"]
    )
    return {
        "candidate": evaluator.POLICY_ID,
        "control": evaluator.PRIMARY,
        "split": "synthetic",
        "observation_end": entry - pd.Timedelta(minutes=5),
        "entry_time": entry,
        "exit_time": exit_time,
        "side": side,
        "crowd_label": "BEARISH" if side == 1 else "BULLISH",
        "displacement_log_return": -0.01 if side == 1 else 0.01,
        "material_threshold_abs_log_return": 0.005,
    }


def _funding(
    rows: list[tuple[pd.Timestamp, float, float]] | None = None,
) -> pd.DataFrame:
    rows = rows or []
    return pd.DataFrame(
        {
            "funding_time": pd.to_datetime(
                [row[0] for row in rows], utc=True, errors="raise"
            ),
            "symbol": ["BTCUSDT"] * len(rows),
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
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    start = start or evaluator._timestamp(market.iloc[0]["date"])
    end = end or evaluator._timestamp(start + pd.Timedelta(days=365.25))
    return evaluator.simulate_strict(
        market,
        _funding() if funding is None else funding,
        pd.DataFrame(clocks),
        start=start,
        end=end,
        cost_rate_per_side=cost,
    )


def test_preregistration_is_deterministic_and_opens_no_outcome() -> None:
    payload = evaluator.verify_preregistration()

    assert payload["candidate"] == evaluator.POLICY_ID
    assert payload["market_or_funding_rows_parsed"] == 0
    assert payload["strategy_outcomes_calculated"] is False
    assert payload["mutable_parameters"] == []
    displacement = payload["price_displacement"]
    assert displacement["reference_shift_bars"] == 13
    assert "target_start-5m" in displacement["reference_endpoint_window"]
    assert payload == evaluator._seal(evaluator._preregistration_core())


def test_evaluator_is_directly_executable_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(evaluator.EVALUATOR_SOURCE), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--write-preregistration" in completed.stdout
    assert "--freeze" in completed.stdout
    assert "--stage {train,test}" in completed.stdout


def test_freeze_does_not_hash_or_parse_execution_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[str] = []
    real_sha = evaluator._sha256

    def tracking_sha(path: str | Path) -> str:
        seen.append(str(path))
        return real_sha(path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("freeze attempted outcome access")

    monkeypatch.setattr(evaluator, "_sha256", tracking_sha)
    monkeypatch.setattr(evaluator, "_parse_market_months", forbidden)
    monkeypatch.setattr(evaluator, "_parse_funding_prefix", forbidden)
    monkeypatch.setattr(evaluator, "build_stage_schedules", forbidden)
    monkeypatch.setattr(evaluator, "simulate_strict", forbidden)

    output = tmp_path / "freeze.json"
    report = evaluator.freeze_evaluator(output)

    assert report["opened_windows"] == []
    assert report["sealed_windows"] == list(evaluator.STAGE_ORDER)
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["price_conditioned_schedules_built_during_freeze"] is False
    assert report["execution_data_bytes_hashed_during_freeze"] is False
    assert str(evaluator.MARKET_COMBINED) not in seen
    assert str(evaluator.FUNDING) not in seen
    assert not any("/monthly/" in item for item in seen)
    assert evaluator.verify_evaluator_freeze(output) == report


def test_freeze_is_write_once(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    evaluator.freeze_evaluator(output)

    with pytest.raises(FileExistsError):
        evaluator.freeze_evaluator(output)


def test_verify_freeze_recomputes_source_contracts(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    report = evaluator.freeze_evaluator(output)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    core["source_contracts"]["stage_market_months"]["train"][0]["path"] = (
        "tampered.csv.gz"
    )
    output.write_text(
        json.dumps(evaluator._seal(core), indent=2), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="source contract changed"):
        evaluator.verify_evaluator_freeze(output)


def test_verify_freeze_rejects_attacker_selected_static_input_subset(
    tmp_path: Path,
) -> None:
    output = tmp_path / "freeze.json"
    report = evaluator.freeze_evaluator(output)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    core["static_inputs"].pop(str(evaluator.PREREGISTRATION_DOC))
    output.write_text(
        json.dumps(evaluator._seal(core), indent=2), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="static-input contract changed"):
        evaluator.verify_evaluator_freeze(output)


def test_test_loader_checks_train_gate_before_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "verify_evaluator_freeze",
        lambda: {"manifest_hash": "freeze", "source_contracts": {}},
    )

    def blocked(*args: object, **kwargs: object) -> list[dict[str, Any]]:
        raise ValueError("train did not pass; test remains sealed")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("test data accessed before train gate")

    monkeypatch.setattr(evaluator, "_verified_prior_reports", blocked)
    monkeypatch.setattr(evaluator, "_parse_market_months", forbidden)
    monkeypatch.setattr(evaluator, "_parse_funding_prefix", forbidden)
    monkeypatch.setattr(evaluator, "_sha256", forbidden)

    with pytest.raises(ValueError, match="test remains sealed"):
        evaluator.load_execution_window("test")


def _prior_report_core(*, gate_value: bool = True) -> dict[str, Any]:
    return {
        "protocol_version": "bitmex_trollbox_attention_saturation_stage_v1",
        "candidate": evaluator.POLICY_ID,
        "stage": "train",
        "stage_passed": True,
        "evaluator_freeze_manifest_hash": "freeze",
        "evaluator_source_sha256": evaluator._sha256(evaluator.EVALUATOR_SOURCE),
        "preregistration_manifest_hash": evaluator.verify_preregistration()[
            "manifest_hash"
        ],
        "opened_windows": ["train"],
        "sealed_windows": ["test"],
        "gate_checks": {
            name: gate_value for name in evaluator.GATE_CHECK_KEYS
        },
        "parameter_search_performed": False,
        "post_failure_repair_performed": False,
    }


def test_prior_pass_flag_cannot_contradict_failed_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    core = _prior_report_core()
    core["gate_checks"]["minimum_trades"] = False
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(evaluator._seal(core), indent=2), encoding="utf-8"
    )
    monkeypatch.setitem(evaluator.STAGE_OUTPUTS, "train", path)

    with pytest.raises(ValueError, match="pass flag contradicts"):
        evaluator._verified_prior_reports("test", freeze_hash="freeze")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parameter_search_performed", True, "parameter search"),
        ("post_failure_repair_performed", True, "post-failure repair"),
    ],
)
def test_prior_report_cannot_open_test_after_search_or_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: bool,
    message: str,
) -> None:
    core = _prior_report_core()
    core[field] = value
    path = tmp_path / f"{field}.json"
    path.write_text(
        json.dumps(evaluator._seal(core), indent=2), encoding="utf-8"
    )
    monkeypatch.setitem(evaluator.STAGE_OUTPUTS, "train", path)

    with pytest.raises(ValueError, match=message):
        evaluator._verified_prior_reports("test", freeze_hash="freeze")


def test_material_reference_ends_strictly_before_target_start() -> None:
    cfg = evaluator.EvaluationConfig()
    periods = cfg.reference_bars + 80
    dates = pd.date_range("2020-01-01", periods=periods, freq="5min", tz="UTC")
    values = 0.0001 + np.arange(periods, dtype=float) * 1e-6
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(periods, 100.0),
            "high": 100.0 * np.exp(values),
            "low": np.full(periods, 100.0),
            "close": 100.0 * np.exp(values),
        }
    )
    displacement, threshold = evaluator._displacement_arrays(market)
    target_final = cfg.reference_bars + 50
    target_start = target_final - cfg.displacement_bars + 1
    latest_reference_final = target_final - cfg.reference_shift_bars
    first_reference_final = latest_reference_final - cfg.reference_bars + 1
    expected = np.quantile(
        np.abs(
            displacement[
                first_reference_final : latest_reference_final + 1
            ]
        ),
        cfg.material_quantile,
        method="linear",
    )
    wrong_touching_window = np.quantile(
        np.abs(
            displacement[
                first_reference_final + 1 : latest_reference_final + 2
            ]
        ),
        cfg.material_quantile,
        method="linear",
    )

    assert threshold[target_final] == pytest.approx(expected)
    assert threshold[target_final] != pytest.approx(wrong_touching_window)
    assert dates[latest_reference_final] + evaluator.BAR == (
        dates[target_start] - evaluator.BAR
    )


def _semantic_event(
    observation_end: str,
    *,
    label: str,
) -> dict[str, Any]:
    end = evaluator._timestamp(observation_end)
    side = -1 if label == "BULLISH" else 1
    return {
        "observation_start": end - evaluator.BAR,
        "observation_end": end,
        "entry_earliest": end + evaluator.BAR,
        "exit_time": end + evaluator.BAR + pd.Timedelta(hours=2),
        "crowd_label": label,
        "contrarian_side": side,
    }


def test_schedule_filters_then_greedily_deconflicts_and_freezes_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _market(start="2021-12-01T00:00:00Z", periods=10_000)
    events = pd.DataFrame(
        [
            _semantic_event("2022-01-01T02:00:00Z", label="BULLISH"),
            _semantic_event("2022-01-01T03:00:00Z", label="BULLISH"),
            _semantic_event("2022-01-01T04:05:00Z", label="BEARISH"),
            _semantic_event("2022-01-01T06:10:00Z", label="BULLISH"),
        ]
    )
    monkeypatch.setattr(
        evaluator,
        "_load_semantic_events",
        lambda: (events, {}, {}),
    )
    displacement = np.zeros(len(market))
    threshold = np.full(len(market), 0.01)
    positions = {value: index for index, value in enumerate(market["date"])}
    # First two are aligned bullish and overlap. Third is anti-aligned bearish
    # on a positive move. Fourth is a second accepted aligned bullish event.
    for timestamp, move in (
        ("2022-01-01T02:00:00Z", 0.02),
        ("2022-01-01T03:00:00Z", 0.03),
        ("2022-01-01T04:05:00Z", 0.02),
        ("2022-01-01T06:10:00Z", 0.02),
    ):
        final = positions[evaluator._timestamp(timestamp) - evaluator.BAR]
        displacement[final] = move
    monkeypatch.setattr(
        evaluator,
        "_displacement_arrays",
        lambda market, cfg=evaluator.EvaluationConfig(): (
            displacement,
            threshold,
        ),
    )

    schedules, _, incidence = evaluator.build_stage_schedules(market, "test")
    primary = schedules[evaluator.PRIMARY]
    stress = schedules["stress"]

    assert len(primary) == 2
    assert incidence["aligned_material_events_before_overlap"] == 3
    assert incidence["primary_overlaps_skipped"] == 1
    assert list(primary["observation_end"]) == [
        evaluator._timestamp("2022-01-01T02:00:00Z"),
        evaluator._timestamp("2022-01-01T06:10:00Z"),
    ]
    assert bool(
        stress["entry_time"]
        .reset_index(drop=True)
        .eq(primary["entry_time"].reset_index(drop=True) + evaluator.BAR)
        .all()
    )
    assert bool(
        stress["exit_time"]
        .reset_index(drop=True)
        .eq(primary["exit_time"].reset_index(drop=True) + evaluator.BAR)
        .all()
    )
    assert bool(
        stress["exit_time"]
        .sub(stress["entry_time"])
        .eq(pd.Timedelta(hours=2))
        .all()
    )
    assert schedules["direction_flip"]["side"].tolist() == [1, 1]
    assert schedules["direction_flip"]["observation_end"].tolist() == primary[
        "observation_end"
    ].tolist()
    assert schedules["deterministic_random_side"]["observation_end"].tolist() == (
        primary["observation_end"].tolist()
    )
    assert len(schedules["semantic_alignment_ablation"]) == 3


def test_stage_boundary_excludes_trade_whose_delayed_exit_reaches_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range(
        "2022-12-01T00:00:00Z", "2022-12-31T23:55:00Z", freq="5min"
    )
    market = pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
        }
    )
    events = pd.DataFrame(
        [_semantic_event("2022-12-31T21:50:00Z", label="BULLISH")]
    )
    monkeypatch.setattr(
        evaluator,
        "_load_semantic_events",
        lambda: (events, {}, {}),
    )
    displacement = np.full(len(market), 0.02)
    threshold = np.full(len(market), 0.01)
    monkeypatch.setattr(
        evaluator,
        "_displacement_arrays",
        lambda market, cfg=evaluator.EvaluationConfig(): (
            displacement,
            threshold,
        ),
    )

    with pytest.raises(ValueError, match="clear event universe is empty"):
        evaluator.build_stage_schedules(market, "test")


@pytest.mark.parametrize(
    ("side", "funding_rate", "expected_cash", "expected_dropped"),
    [
        (1, 0.001, -0.002, 0),
        (-1, -0.001, -0.002, 0),
        (1, -0.001, 0.0, 2),
        (-1, 0.001, 0.0, 2),
    ],
)
def test_entry_and_exit_funding_boundary_contract(
    side: int,
    funding_rate: float,
    expected_cash: float,
    expected_dropped: int,
) -> None:
    market = _market()
    clock = _clock(market, side=side)
    entry = evaluator._timestamp(clock["entry_time"])
    exit_time = evaluator._timestamp(clock["exit_time"])
    funding = _funding(
        [(entry, funding_rate, 100.0), (exit_time, funding_rate, 100.0)]
    )

    result = _simulate(market, [clock], funding=funding)
    trade = result["trade_details"][0]

    assert trade["visited_funding_events"] == 2
    assert trade["dropped_boundary_funding_credits"] == expected_dropped
    assert trade["funding_cash"] == pytest.approx(expected_cash)


def test_interior_funding_keeps_credit_and_debit_symmetrically() -> None:
    market = _market()
    clock = _clock(market)
    entry = evaluator._timestamp(clock["entry_time"])
    funding = _funding(
        [
            (entry + pd.Timedelta(minutes=30), -0.001, 100.0),
            (entry + pd.Timedelta(minutes=60), 0.002, 100.0),
        ]
    )

    result = _simulate(market, [clock], funding=funding)

    assert result["trade_details"][0]["funding_cash"] == pytest.approx(-0.001)


def test_flat_trade_charges_both_notional_sides_at_one_x() -> None:
    market = _market()
    clock = _clock(market)

    base = _simulate(market, [clock], cost=0.0006)
    stress = _simulate(market, [clock], cost=0.0010)

    assert base["absolute_return_pct"] == pytest.approx(-0.12)
    assert stress["absolute_return_pct"] == pytest.approx(-0.20)
    assert base["strict_mdd_pct"] == pytest.approx(0.12)
    assert stress["strict_mdd_pct"] == pytest.approx(0.20)


def test_strict_mdd_uses_global_favorable_then_adverse_path_and_exit_fee() -> None:
    market = _market()
    market.loc[0, ["open", "high", "low", "close"]] = [
        100.0,
        110.0,
        90.0,
        100.0,
    ]

    result = _simulate(market, [_clock(market)], cost=0.001)

    expected_hwm = 1.099
    expected_adverse_after_virtual_exit = 0.8981
    assert result["strict_mdd_pct"] == pytest.approx(
        (1.0 - expected_adverse_after_virtual_exit / expected_hwm) * 100.0
    )


def test_full_calendar_cagr_counts_all_idle_time() -> None:
    market = _market()
    market.loc[24, ["open", "high", "low", "close"]] = [
        110.0,
        110.0,
        110.0,
        110.0,
    ]
    clock = _clock(market)
    start = evaluator._timestamp(clock["entry_time"])

    one_year = _simulate(
        market,
        [clock],
        start=start,
        end=start + pd.Timedelta(days=365.25),
    )
    two_years = _simulate(
        market,
        [clock],
        start=start,
        end=start + pd.Timedelta(days=730.5),
    )

    assert one_year["absolute_return_pct"] == pytest.approx(10.0)
    assert two_years["absolute_return_pct"] == pytest.approx(10.0)
    assert one_year["cagr_pct"] == pytest.approx(10.0)
    assert two_years["cagr_pct"] == pytest.approx((1.10**0.5 - 1.0) * 100.0)


def test_nonfinite_control_ratio_fails_margin_gate() -> None:
    prereg = evaluator.verify_preregistration()
    base = {
        "absolute_return_pct": 10.0,
        "cagr_to_strict_mdd": 4.0,
        "strict_mdd_pct": 5.0,
        "trades": 100,
        "longs": 50,
        "shorts": 50,
        "mean_gross_underlying_bp": 30.0,
        "weekly_cluster_signflip": {
            "p_value_two_sided": 0.05,
            "cluster_count": 50,
        },
    }
    stress = {**base, "cagr_to_strict_mdd": 3.0}
    subperiods = {"left": {"absolute_return_pct": 1.0}}
    controls = {
        name: {"cagr_to_strict_mdd": 0.0}
        for name in evaluator.MECHANISM_CONTROLS
    }
    controls["deterministic_random_side"] = {
        "cagr_to_strict_mdd": float("-inf")
    }

    checks, _, margin = evaluator._stage_gates(
        "train", base, stress, subperiods, controls, prereg
    )

    assert margin == float("-inf")
    assert checks["mechanism_control_margin_at_least_0_25"] is False


def test_weekly_cluster_signflip_is_deterministic() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.date_range(
                "2022-01-03", periods=21, freq="7D", tz="UTC"
            ),
            "net_return": np.linspace(-0.01, 0.02, 21),
        }
    )

    first = evaluator.weekly_cluster_signflip_two_sided(trades)
    second = evaluator.weekly_cluster_signflip_two_sided(trades)

    assert first == second
    assert first["method"] == "monte_carlo"
    assert first["draws"] == 20_000


def test_market_parser_rejects_row_at_or_after_stage_end(tmp_path: Path) -> None:
    path = tmp_path / "month.csv.gz"
    with gzip.open(path, "wt", newline="") as handle:
        pd.DataFrame(
            [
                ["2021-12-31T23:55:00Z", 100, 100, 100, 100],
                ["2022-01-01T00:00:00Z", 100, 100, 100, 100],
            ],
            columns=["date", "open", "high", "low", "close"],
        ).to_csv(handle, index=False)
    contract = {
        "month": "2021-12",
        "path": str(path),
        "sha256": evaluator._sha256(path),
        "rows": 2,
    }

    with pytest.raises(ValueError, match="crossed the stage end"):
        evaluator._parse_market_months(
            [contract], end=evaluator._timestamp("2022-01-01T00:00:00Z")
        )


def test_ratio_has_no_epsilon_floor() -> None:
    assert evaluator._ratio(0.1, 0.0) == float("inf")
    assert evaluator._ratio(0.0, 0.0) == 0.0
    assert evaluator._ratio(-0.1, 0.0) == float("-inf")
    assert evaluator._ratio(0.1, 1e-15) == pytest.approx(1e14)
