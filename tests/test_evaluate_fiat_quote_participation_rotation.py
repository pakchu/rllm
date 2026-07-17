from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_fiat_quote_participation_rotation as evaluator


def test_frozen_schedules_are_causal_nonoverlapping_and_pre2024() -> None:
    schedules = evaluator.load_schedules()
    assert set(schedules) == set(evaluator.ALL_CLOCK_NAMES)
    primary = schedules["primary"]
    assert len(primary) == 72
    for schedule in schedules.values():
        assert schedule["entry_time"].ge(evaluator.STAGE1[0]).all()
        assert schedule["exit_time"].le(evaluator.STAGE2[1]).all()
        assert schedule["decision_time"].eq(
            schedule["signal_day"] + pd.Timedelta(days=1)
        ).all()
        assert schedule["entry_time"].eq(
            schedule["decision_time"] + pd.Timedelta(minutes=5)
        ).all()
        assert schedule["exit_time"].eq(
            schedule["entry_time"] + pd.Timedelta(days=3)
        ).all()


def test_freeze_opens_no_market_funding_or_outcome_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []
    real_sha = evaluator._sha256

    def tracking_sha(path: str | Path) -> str:
        seen.append(str(path))
        return real_sha(path)

    monkeypatch.setattr(evaluator, "_sha256", tracking_sha)
    output = tmp_path / "freeze.json"
    report = evaluator.freeze_evaluator(output)
    assert report["opened_windows"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert str(evaluator.MARKET) not in seen
    assert str(evaluator.FUNDING) not in seen
    replay = evaluator.verify_evaluator_freeze(output)
    assert replay == report


def _write_market(path: Path) -> None:
    rows = []
    for timestamp in pd.date_range("2022-01-01", periods=4, freq="5min"):
        rows.append([timestamp, 100.0, 110.0, 90.0, 100.0])
    rows.append([pd.Timestamp("2022-01-01 00:20"), 999.0, 999.0, 999.0, 999.0])
    with gzip.open(path, "wt", newline="") as handle:
        pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"]).to_csv(
            handle, index=False
        )


def _write_funding(path: Path) -> None:
    rows = [
        ["2022-01-01T00:00:00.000000Z", "BTCUSDT", 0.001, 100.0],
        ["2022-01-01T08:00:00.000000Z", "BTCUSDT", 0.001, 999.0],
    ]
    with gzip.open(path, "wt", newline="") as handle:
        pd.DataFrame(
            rows,
            columns=[
                "funding_time_utc",
                "symbol",
                "funding_rate",
                "settlement_mark_price",
            ],
        ).to_csv(handle, index=False)


def test_physical_parsers_stop_before_end_boundary(tmp_path: Path) -> None:
    market_path = tmp_path / "market.csv.gz"
    _write_market(market_path)
    start = pd.Timestamp("2022-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2022-01-01 00:20", tz="UTC")
    market, diagnostics = evaluator._parse_market_window(market_path, start, end)
    assert len(market) == 4
    assert market["open"].max() == 100.0
    assert diagnostics["stopped_before_parsing_end_boundary"] is True

    funding_path = tmp_path / "funding.csv.gz"
    _write_funding(funding_path)
    funding_end = pd.Timestamp("2022-01-01 08:00", tz="UTC")
    funding, funding_diag = evaluator._parse_funding_window(
        funding_path, start, funding_end
    )
    assert len(funding) == 1
    assert funding["settlement_mark_price"].max() == 100.0
    assert funding_diag["stopped_before_parsing_end_boundary"] is True


def test_strict_simulator_uses_favorable_then_adverse_cost_and_funding() -> None:
    dates = pd.date_range("2022-01-01", periods=4, freq="5min", tz="UTC")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [110.0, 100.0, 100.0, 100.0],
            "low": [90.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
        }
    )
    funding = pd.DataFrame(
        {
            "funding_time": [dates[0]],
            "funding_rate": [0.001],
            "settlement_mark_price": [100.0],
        }
    )
    schedule = pd.DataFrame(
        {
            "clock_name": ["primary"],
            "signal_day": [dates[0]],
            "entry_time": [dates[0]],
            "exit_time": [dates[2]],
            "side": [1],
        }
    )
    result = evaluator.simulate_schedule(
        market,
        funding,
        schedule,
        period_start=dates[0],
        period_end=dates[3],
        cost_rate=0.001,
    )
    assert result["trades"] == 1
    assert result["funding_cash_pct_initial"] < 0.0
    assert result["absolute_return_pct"] < 0.0
    assert result["strict_mdd_pct"] > 9.0


def test_weekly_cluster_signflip_is_deterministic_and_trade_based() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2022-01-03", "2022-01-04", "2022-01-10"], utc=True
            ),
            "net_return": [0.01, 0.02, -0.005],
        }
    )
    first = evaluator.weekly_cluster_signflip(trades, draws=20_000, seed=20_260_717)
    second = evaluator.weekly_cluster_signflip(trades, draws=20_000, seed=20_260_717)
    assert first == second
    assert first["cluster_count"] == 2
    assert first["observed_mean_net_return"] == pytest.approx(0.025 / 3.0)
    assert 0.0 < first["p_value_one_sided"] <= 1.0


def test_written_evaluator_freeze_replays() -> None:
    payload = json.loads(evaluator.EVALUATOR_FREEZE.read_text())
    assert payload == evaluator.verify_evaluator_freeze()
    assert payload["simulation_run_during_freeze"] is False
    assert payload["opened_windows"] == []
    assert np.isfinite(payload["selected_q"])


def test_stage2_rejects_stage1_bound_to_another_freeze_before_outcome_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage1_path = tmp_path / "stage1.json"
    report = evaluator._seal(
        {
            "protocol_version": "fiat_quote_participation_rotation_stage1_v1",
            "policy_id": "FQPR-3",
            "stage": "stage1_2021_2022",
            "evaluator_freeze_manifest_hash": "wrong-freeze",
            "evaluator_source_sha256": evaluator._sha256(evaluator.EVALUATOR_SOURCE),
            "config": evaluator.asdict(evaluator.EvaluationConfig()),
            "execution_diagnostics": {
                "physical_window": [
                    evaluator.STAGE1[0].isoformat(),
                    evaluator.STAGE1[1].isoformat(),
                ]
            },
            "gates": {name: True for name in evaluator.STAGE1_GATE_NAMES},
            "gate_passed": True,
            "opened_windows": ["stage1_2021_2022"],
            "sealed_windows": ["stage2_2023", "2024", "2025", "2026_ytd"],
            "disposition": "PASS_STAGE1_OPEN_2023_ONCE",
        }
    )
    stage1_path.write_text(json.dumps(report))
    monkeypatch.setattr(evaluator, "STAGE1_OUTPUT", stage1_path)
    monkeypatch.setattr(
        evaluator,
        "verify_evaluator_freeze",
        lambda: {"manifest_hash": "current-freeze"},
    )

    def forbidden_outcome_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("Stage 2 tried to open a sealed outcome")

    monkeypatch.setattr(evaluator, "load_execution_window", forbidden_outcome_open)
    with pytest.raises(ValueError, match="current evaluator freeze"):
        evaluator.evaluate_stage2()
