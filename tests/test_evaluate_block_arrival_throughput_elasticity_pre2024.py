from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_block_arrival_throughput_elasticity_pre2024 as evaluate


def _market(
    opens: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    values = np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=len(values), freq="5min"),
            "open": values,
            "high": values if highs is None else highs,
            "low": values if lows is None else lows,
            "close": values,
        }
    )


def _schedule(
    market: pd.DataFrame,
    *,
    entry_position: int,
    exit_position: int,
    side: int = 1,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "side": [side],
            "entry_date": [market["date"].iloc[entry_position]],
            "exit_date": [market["date"].iloc[exit_position]],
            "entry_position": [entry_position],
            "exit_position": [exit_position],
        }
    )


def _funding(
    market: pd.DataFrame,
    positions: list[int],
    rates: list[float],
    *,
    mark_prices: list[float] | None = None,
) -> pd.DataFrame:
    times = market["date"].iloc[positions].reset_index(drop=True)
    return pd.DataFrame(
        {
            "funding_time_ms": (times.astype("int64") // 1_000_000).astype(np.int64),
            "funding_time": times,
            "funding_rate": rates,
            "settlement_mark_price": (
                market["open"].iloc[positions].to_numpy(float)
                if mark_prices is None
                else mark_prices
            ),
        }
    )


def test_train_market_parser_skips_pretrain_values_and_stops_before_2023(
    tmp_path: Path,
) -> None:
    path = tmp_path / "market.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2020-12-31 23:55:00,NOT_PARSED,NOT_PARSED,NOT_PARSED,NOT_PARSED\n")
        handle.write("2021-01-01 00:00:00,100,101,99,100\n")
        handle.write("2022-12-31 23:55:00,110,111,109,110\n")
        handle.write("2023-01-01 00:00:00,NOT_PARSED,NOT_PARSED,NOT_PARSED,NOT_PARSED\n")
    frame = evaluate._parse_market_window(
        path, start="2021-01-01", end="2023-01-01"
    )
    assert frame["open"].tolist() == [100.0, 110.0]


def test_selection_market_parser_stops_before_2024_values(tmp_path: Path) -> None:
    path = tmp_path / "market.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2022-12-31 23:55:00,NOT_PARSED,NOT_PARSED,NOT_PARSED,NOT_PARSED\n")
        handle.write("2023-01-01 00:00:00,100,101,99,100\n")
        handle.write("2024-01-01 00:00:00,NOT_PARSED,NOT_PARSED,NOT_PARSED,NOT_PARSED\n")
    frame = evaluate._parse_market_window(
        path, start="2023-01-01", end="2024-01-01"
    )
    assert frame["open"].tolist() == [100.0]


def test_train_market_parser_requires_physical_end_boundary(tmp_path: Path) -> None:
    path = tmp_path / "market.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2022-12-31 23:55:00,100,101,99,100\n")
    with pytest.raises(ValueError, match="did not reach sealed boundary"):
        evaluate._parse_market_window(path, start="2021-01-01", end="2023-01-01")


def test_train_funding_parser_skips_pretrain_and_stops_before_2023(
    tmp_path: Path,
) -> None:
    path = tmp_path / "funding.csv.gz"
    header = (
        "funding_time_ms,funding_time_utc,symbol,funding_rate,"
        "settlement_mark_price,funding_time_offset_ms,mark_source\n"
    )
    with gzip.open(path, "wt") as handle:
        handle.write(header)
        handle.write("1609459199000,POISON,POISON,POISON,POISON,POISON,POISON\n")
        handle.write(
            "1609459200000,2021-01-01 00:00:00+00:00,BTCUSDT,0.001,29000,0,"
            "binance_8h_mark_price_kline_open\n"
        )
        handle.write("1672531200000,POISON,POISON,POISON,POISON,POISON,POISON\n")
    frame = evaluate._parse_funding_window(
        path, start="2021-01-01", end="2023-01-01"
    )
    assert frame["funding_rate"].tolist() == ["0.001"]


def test_selection_funding_parser_requires_physical_or_audited_eof_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "funding.csv.gz"
    header = (
        "funding_time_ms,funding_time_utc,symbol,funding_rate,"
        "settlement_mark_price,funding_time_offset_ms,mark_source\n"
    )
    terminal = 1_704_038_400_000
    with gzip.open(path, "wt") as handle:
        handle.write(header)
        handle.write(
            f"{terminal},2023-12-31 16:00:00+00:00,BTCUSDT,0.001,42000,0,"
            "binance_8h_mark_price_kline_open\n"
        )
    with pytest.raises(ValueError, match="physical or audited boundary"):
        evaluate._parse_funding_window(
            path, start="2023-01-01", end="2024-01-01"
        )
    frame = evaluate._parse_funding_window(
        path,
        start="2023-01-01",
        end="2024-01-01",
        audited_eof_last_timestamp_ms=terminal,
    )
    assert frame["funding_time_ms"].tolist() == [terminal]


def test_control_clocks_are_source_only_nonoverlapping_and_random_side_matched() -> None:
    clocks, support = evaluate.verify_support_and_control_clocks()
    assert tuple(clocks) == evaluate.POLICY_NAMES
    assert len(clocks["primary"]) == support["clock"]["rows"] == 971
    primary_counts = (
        clocks["primary"]
        .assign(year=clocks["primary"]["entry_date"].dt.year)
        .groupby(["year", "side"])
        .size()
        .sort_index()
    )
    random_counts = (
        clocks["random_clock"]
        .assign(year=clocks["random_clock"]["entry_date"].dt.year)
        .groupby(["year", "side"])
        .size()
        .sort_index()
    )
    pd.testing.assert_series_equal(primary_counts, random_counts)
    assert clocks["direction_flip"]["side"].tolist() == (
        -clocks["primary"]["side"]
    ).tolist()
    assert clocks["random_clock"]["entry_date"].dt.time.nunique() > 1
    for clock in clocks.values():
        assert (clock["exit_date"] - clock["entry_date"]).eq(pd.Timedelta(days=1)).all()
        if len(clock) > 1:
            assert clock["entry_date"].iloc[1:].reset_index(drop=True).ge(
                clock["exit_date"].iloc[:-1].reset_index(drop=True)
            ).all()


def test_weekly_sign_flip_is_exact_and_excludes_empty_weeks() -> None:
    result = evaluate.weekly_cluster_sign_flip(
        [0.01, 0.02, 0.03, -0.01],
        pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-10", "2023-01-24"]),
        permutations=100,
        seed=7,
    )
    assert result["cluster_count"] == 4
    assert result["method"] == "exact"
    assert result["p_value_one_sided"] == pytest.approx(0.1875)


def test_strict_mdd_marks_favorable_before_adverse_for_long() -> None:
    market = _market(
        [100.0, 100.0, 110.0],
        highs=[120.0, 100.0, 110.0],
        lows=[90.0, 100.0, 110.0],
    )
    result = evaluate.simulate_schedule(
        market,
        _funding(market, [], []),
        _schedule(market, entry_position=0, exit_position=2),
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    assert result["absolute_return_pct"] == pytest.approx(10.0)
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.90 / 1.20) * 100.0)
    expected_cagr = (1.10 ** (365.25 / 365.0) - 1.0) * 100.0
    assert result["cagr_pct"] == pytest.approx(expected_cagr)


def test_strict_mdd_marks_favorable_before_adverse_for_short() -> None:
    market = _market(
        [100.0, 100.0, 90.0],
        highs=[110.0, 100.0, 90.0],
        lows=[80.0, 100.0, 90.0],
    )
    result = evaluate.simulate_schedule(
        market,
        _funding(market, [], []),
        _schedule(market, entry_position=0, exit_position=2, side=-1),
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    assert result["absolute_return_pct"] == pytest.approx(10.0)
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.90 / 1.20) * 100.0)


def test_global_hwm_carries_across_trades() -> None:
    market = _market(
        [100.0, 100.0, 120.0, 100.0, 100.0, 100.0],
        highs=[100.0, 100.0, 120.0, 100.0, 100.0, 100.0],
        lows=[100.0, 100.0, 120.0, 90.0, 100.0, 100.0],
    )
    schedule = pd.concat(
        [
            _schedule(market, entry_position=0, exit_position=2),
            _schedule(market, entry_position=3, exit_position=5),
        ],
        ignore_index=True,
    )
    result = evaluate.simulate_schedule(
        market,
        _funding(market, [], []),
        schedule,
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    assert result["strict_mdd_pct"] == pytest.approx(10.0)


def test_entry_exit_and_hypothetical_liquidation_costs_are_charged() -> None:
    market = _market([100.0, 100.0, 100.0])
    result = evaluate.simulate_schedule(
        market,
        _funding(market, [], []),
        _schedule(market, entry_position=0, exit_position=2),
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.001,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    assert result["absolute_return_pct"] == pytest.approx(-0.2)
    assert result["strict_mdd_pct"] == pytest.approx(0.2)


def test_funding_is_entry_inclusive_exit_exclusive_and_mark_scaled() -> None:
    market = _market([100.0, 110.0, 120.0, 130.0])
    funding = _funding(
        market,
        [0, 2, 3],
        [0.02, -0.01, 0.50],
        mark_prices=[100.0, 140.0, 130.0],
    )
    result = evaluate.simulate_schedule(
        market,
        funding,
        _schedule(market, entry_position=0, exit_position=3),
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    assert result["funding_settlement_count"] == 2
    assert result["absolute_return_pct"] == pytest.approx(29.4)
    assert result["total_funding_cash_pct_of_entry_equity_sum"] == pytest.approx(-0.6)


def test_funding_credit_and_debit_form_conservative_mdd_envelope() -> None:
    market = _market([100.0, 100.0, 100.0])
    funding = _funding(market, [0, 1], [-0.02, 0.02])
    result = evaluate.simulate_schedule(
        market,
        funding,
        _schedule(market, entry_position=0, exit_position=2),
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    assert result["absolute_return_pct"] == pytest.approx(0.0)
    assert result["strict_mdd_pct"] == pytest.approx((1.0 - 0.98 / 1.02) * 100.0)


def test_funding_cash_diagnostic_weights_each_trade_by_entry_equity() -> None:
    market = _market([100.0, 200.0, 100.0, 100.0])
    schedule = pd.concat(
        [
            _schedule(market, entry_position=0, exit_position=1),
            _schedule(market, entry_position=2, exit_position=3),
        ],
        ignore_index=True,
    )
    result = evaluate.simulate_schedule(
        market,
        _funding(market, [2], [0.10]),
        schedule,
        start="2023-01-01",
        end="2024-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    # Entry equities are 1 and 2; funding cash is -0.2, so -0.2 / 3.
    assert result["total_funding_cash_pct_of_entry_equity_sum"] == pytest.approx(
        -100.0 * 0.2 / 3.0
    )


def test_full_calendar_cagr_counts_idle_time() -> None:
    market = _market([100.0, 110.0])
    result = evaluate.simulate_schedule(
        market,
        _funding(market, [], []),
        _schedule(market, entry_position=0, exit_position=1),
        start="2023-01-01",
        end="2025-01-01",
        cost_notional_per_side=0.0,
        cfg=evaluate.EvaluationConfig(),
        compute_cluster=False,
    )
    years = (pd.Timestamp("2025-01-01") - pd.Timestamp("2023-01-01")).total_seconds() / (
        365.25 * 86_400.0
    )
    assert result["wall_clock_years"] == pytest.approx(years)
    assert result["cagr_pct"] == pytest.approx((1.1 ** (1 / years) - 1) * 100)


def test_split_slicing_excludes_trade_that_exits_on_sealed_boundary() -> None:
    schedule = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2023-06-29", "2023-07-01"]),
            "exit_date": pd.to_datetime(["2023-07-01", "2023-07-08"]),
        }
    )
    h1 = evaluate._slice_schedule(schedule, start="2023-01-01", end="2023-07-01")
    h2 = evaluate._slice_schedule(schedule, start="2023-07-01", end="2024-01-01")
    assert h1.empty
    assert len(h2) == 1


def _passing_metrics() -> dict[str, object]:
    return {
        "absolute_return_pct": 10.0,
        "cagr_to_strict_mdd": 4.0,
        "strict_mdd_pct": 5.0,
        "mean_gross_underlying_move_bp": 31.0,
        "weekly_cluster_sign_flip": {"p_value_one_sided": 0.05},
    }


def _passing_policy(stage: str = "train") -> dict[str, object]:
    split_names = evaluate.STAGE_WINDOWS[stage][1]
    return {
        "base_6bp": _passing_metrics(),
        "stress_10bp": {**_passing_metrics(), "absolute_return_pct": 5.0},
        "splits_base_6bp": {name: _passing_metrics() for name in split_names},
        "side_contributions_base_6bp": {
            "HIGH_long": _passing_metrics(),
            "LOW_short": _passing_metrics(),
        },
    }


def test_gate_enforces_30bp_and_both_side_contributions() -> None:
    policy = _passing_policy()
    assert evaluate.stage_gate_failures(
        policy, stage="train", cfg=evaluate.EvaluationConfig()
    ) == []
    policy["base_6bp"]["mean_gross_underlying_move_bp"] = 29.99
    policy["side_contributions_base_6bp"]["LOW_short"]["absolute_return_pct"] = -0.01
    failures = evaluate.stage_gate_failures(
        policy, stage="train", cfg=evaluate.EvaluationConfig()
    )
    assert any("below 30 bp" in failure for failure in failures)
    assert any("LOW_short" in failure for failure in failures)


def test_passing_component_control_rejects_joint_mechanism() -> None:
    policies = {name: _passing_policy() for name in evaluate.POLICY_NAMES}
    verdict = evaluate.qualification(
        policies, stage="train", cfg=evaluate.EvaluationConfig()
    )
    assert verdict["qualifies"] is False
    assert set(verdict["passing_mechanism_controls"]) == set(
        evaluate.MECHANISM_REJECTION_CONTROLS
    )


def test_diagnostic_controls_cannot_replace_primary() -> None:
    policies = {name: _passing_policy() for name in evaluate.POLICY_NAMES}
    for name in evaluate.MECHANISM_REJECTION_CONTROLS:
        policies[name]["base_6bp"]["absolute_return_pct"] = -1.0
    policies["direction_flip"]["base_6bp"]["absolute_return_pct"] = 500.0
    policies["random_clock"]["base_6bp"]["absolute_return_pct"] = 500.0
    verdict = evaluate.qualification(
        policies, stage="train", cfg=evaluate.EvaluationConfig()
    )
    assert verdict["qualifies"] is True


def test_selection_checks_passing_train_before_parsing_selection_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(evaluate, "SELECTION_OUTPUT", tmp_path / "selection.json")
    monkeypatch.setattr(evaluate, "verify_evaluation_freeze", lambda _cfg: {})
    monkeypatch.setattr(evaluate, "verify_support_and_control_clocks", lambda _cfg: ({}, {}))

    def deny(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise PermissionError("sealed")

    opened = False

    def forbidden(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal opened
        opened = True
        return {}

    monkeypatch.setattr(evaluate, "_verify_passing_train_result", deny)
    monkeypatch.setattr(evaluate, "_compute_stage_report", forbidden)
    with pytest.raises(PermissionError, match="sealed"):
        evaluate.evaluate_selection()
    assert opened is False


def test_train_artifact_must_exactly_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("{}\n")
    train_path = tmp_path / "train.json"
    monkeypatch.setattr(evaluate, "EVALUATION_FREEZE", freeze_path)
    monkeypatch.setattr(evaluate, "TRAIN_OUTPUT", train_path)
    cfg = evaluate.EvaluationConfig()
    payload = evaluate.seal_result(
        {
            "schema_version": 1,
            "created_at": "fixed",
            "evaluation_config": evaluate.asdict(cfg),
            "evaluation_freeze_sha256": evaluate.sha256_file(freeze_path),
            "protocol": {
                "opened_windows": ["train_2021_2022"],
                "selection_2023_opened": False,
            },
            "control_clock_hashes": {"primary": "clock"},
            "qualification": {"qualifies": True, "failures": []},
            "decision": "open_selection_2023",
        }
    )
    train_path.write_text(json.dumps(payload) + "\n")
    freeze = {"control_clock_hashes": {"primary": "clock"}}
    monkeypatch.setattr(
        evaluate,
        "_compute_stage_report",
        lambda *_args, **_kwargs: {"created_at": "fixed", "replayed": True},
    )
    with pytest.raises(ValueError, match="does not exactly replay"):
        evaluate._verify_passing_train_result(cfg, freeze, {})


def test_nondefault_evaluation_config_is_rejected() -> None:
    cfg = replace(evaluate.EvaluationConfig(), leverage=0.5)
    with pytest.raises(ValueError, match="parameters are frozen"):
        evaluate.verify_support_and_control_clocks(cfg)
