from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_leveraged_um_inventory_release_handoff as evaluator
from training import preregister_leveraged_um_inventory_release_handoff as luri


def _frame(rows: int = 160) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": False,
        }
    )


def _market_frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.0),
            "low": np.full(rows, 100.0),
            "close": np.full(rows, 100.0),
        }
    )


def _funding(rows: list[tuple[int, float]] | None = None) -> pd.DataFrame:
    values = rows or []
    return pd.DataFrame(values, columns=["funding_time_ms", "funding_rate"]).assign(
        funding_time=lambda value: pd.to_datetime(
            value["funding_time_ms"], unit="ms"
        )
    )[["funding_time_ms", "funding_time", "funding_rate"]]


def _schedule(*, side: int = 1, entry: int = 1, exit_: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=max(exit_ + 1, 3), freq="5min")
    return pd.DataFrame(
        [
            {
                "signal_position": entry - 1,
                "entry_position": entry,
                "exit_position": exit_,
                "signal_date": str(dates[entry - 1]),
                "entry_date": str(dates[entry]),
                "exit_date": str(dates[exit_]),
                "side": side,
                "branch": "luri48",
                "hold_bars": exit_ - entry,
            }
        ]
    )


def _primary_schedule(frame: pd.DataFrame, cfg: luri.Config) -> pd.DataFrame:
    active = pd.Series(False, index=frame.index)
    active.iloc[[0, 49, 98]] = True
    sides = np.where(np.arange(len(frame)) % 2 == 0, 1, -1)
    signal = pd.DataFrame(
        {
            "side": np.where(active, sides, 0),
            "branch": np.where(active, "luri48", "none"),
            "hold_bars": np.where(active, cfg.hold_bars, 0),
        }
    )
    return luri.nonoverlapping_schedule(signal, frame)


def _control_inputs(
    frame: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    controls: dict[str, pd.Series] = {}
    sides: dict[str, pd.Series] = {}
    for offset, name in enumerate(evaluator.POLICY_NAMES):
        active = pd.Series(False, index=frame.index)
        active.iloc[offset % 20 :: 49] = True
        controls[name] = active
        sides[name] = pd.Series(
            np.where(np.arange(len(frame)) % 2 == 0, 1.0, -1.0),
            index=frame.index,
        )
    return controls, sides


def _metrics(
    *,
    absolute_return: float = 5.0,
    ratio: float = 4.0,
    mdd: float = 1.0,
    trades: int = 100,
    p_value: float = 0.01,
    gross_bp: float = 20.0,
) -> dict[str, object]:
    return {
        "absolute_return_pct": absolute_return,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": mdd,
        "trade_count": trades,
        "mean_gross_underlying_move_bp": gross_bp,
        "weekly_cluster_sign_flip": {"p_value_one_sided": p_value},
    }


def test_evaluation_config_and_windows_are_frozen() -> None:
    evaluator._validate_evaluation_config(evaluator.EvaluationConfig())
    with pytest.raises(ValueError, match="evaluation config is frozen"):
        evaluator._validate_evaluation_config(
            evaluator.EvaluationConfig(minimum_2023_half_trades=44)
        )
    assert max(end for _, end in evaluator.WINDOWS.values()) == "2024-01-01"
    assert set(evaluator.WINDOWS) == {
        "train",
        "select2023",
        "select2023_h1",
        "select2023_h2",
    }


def test_score_bearing_and_falsification_sets_are_frozen() -> None:
    assert set(evaluator.SCORE_BEARING_CONTROLS).isdisjoint(
        evaluator.FALSIFICATION_CONTROLS
    )
    assert set(evaluator.SCORE_BEARING_CONTROLS) | set(
        evaluator.FALSIFICATION_CONTROLS
    ) == set(evaluator.POLICY_NAMES) - {"primary"}


def test_direction_flip_uses_exact_primary_clock() -> None:
    frame = _frame()
    cfg = luri.Config()
    controls, sides = _control_inputs(frame)
    primary = _primary_schedule(frame, cfg)
    schedules = evaluator.build_control_schedules(frame, controls, sides, primary, cfg)
    flipped = schedules["direction_flip"]
    assert flipped[["signal_position", "entry_position", "exit_position"]].equals(
        primary[["signal_position", "entry_position", "exit_position"]]
    )
    assert np.array_equal(flipped["side"].to_numpy(), -primary["side"].to_numpy())
    assert tuple(schedules) == evaluator.POLICY_NAMES


def test_control_schedule_rejects_active_zero_side() -> None:
    frame = _frame()
    cfg = luri.Config()
    mask = pd.Series(False, index=frame.index)
    mask.iloc[0] = True
    with pytest.raises(ValueError, match="invalid active side"):
        evaluator._schedule_from_control(
            frame,
            cfg,
            mask=mask,
            side=pd.Series(0.0, index=frame.index),
            branch="broken",
        )


def test_funding_endpoints_are_inclusive_and_direction_is_correct() -> None:
    frame = _market_frame(52)
    entry_ms = int(frame.loc[1, "date"].value // 1_000_000)
    exit_ms = int(frame.loc[49, "date"].value // 1_000_000)
    funding = _funding(
        [
            (entry_ms, 0.01),
            (exit_ms, 0.01),
            (exit_ms + 1, 0.50),
        ]
    )
    cfg = evaluator.EvaluationConfig(
        fee_rate=0.0,
        slippage_rate=0.0,
        cluster_permutations=1,
    )
    result = evaluator.simulate_funding_schedule(
        frame,
        funding,
        _schedule(side=1, exit_=49),
        start="2023-01-01",
        end="2023-01-02",
        cfg=cfg,
    )
    assert result["funding_settlement_count"] == 2
    assert result["trades_with_funding"] == 1
    assert np.isclose(result["absolute_return_pct"], ((0.995**2) - 1.0) * 100.0)

    short = evaluator.simulate_funding_schedule(
        frame,
        funding.iloc[:2],
        _schedule(side=-1, exit_=49),
        start="2023-01-01",
        end="2023-01-02",
        cfg=cfg,
    )
    assert short["absolute_return_pct"] > 0.0


def test_funding_debits_precede_adverse_and_credits_do_not_hide_mdd() -> None:
    frame = _market_frame(52)
    entry_ms = int(frame.loc[1, "date"].value // 1_000_000)
    exit_ms = int(frame.loc[49, "date"].value // 1_000_000)
    funding = _funding([(entry_ms, 0.10), (exit_ms, -0.10)])
    cfg = evaluator.EvaluationConfig(
        fee_rate=0.0,
        slippage_rate=0.0,
        cluster_permutations=1,
    )
    result = evaluator.simulate_funding_schedule(
        frame,
        funding,
        _schedule(side=1, exit_=49),
        start="2023-01-01",
        end="2023-01-02",
        cfg=cfg,
    )
    assert np.isclose(result["strict_mdd_pct"], 5.0)
    assert np.isclose(result["absolute_return_pct"], (0.95 * 1.05 - 1.0) * 100.0)


def test_strict_mdd_uses_favorable_first_and_excludes_exit_bar() -> None:
    frame = _market_frame(52)
    frame.loc[1, ["high", "low"]] = [110.0, 90.0]
    frame.loc[49, ["high", "low"]] = [1_000.0, 1.0]
    result = evaluator.simulate_funding_schedule(
        frame,
        _funding(),
        _schedule(side=1, exit_=49),
        start="2023-01-01",
        end="2023-01-02",
        cfg=evaluator.EvaluationConfig(cluster_permutations=1),
    )
    expected = (1.0 - (0.9997 * 0.95) / (0.9997 * 1.05)) * 100.0
    assert np.isclose(result["strict_mdd_pct"], expected)
    assert result["strict_mdd_pct"] < 10.0


def test_mean_gross_move_is_direct_and_not_changed_by_funding() -> None:
    frame = _market_frame(52)
    frame.loc[49, "open"] = 100.2
    entry_ms = int(frame.loc[1, "date"].value // 1_000_000)
    result = evaluator.simulate_funding_schedule(
        frame,
        _funding([(entry_ms, 0.05)]),
        _schedule(side=1, exit_=49),
        start="2023-01-01",
        end="2023-01-02",
        cfg=evaluator.EvaluationConfig(cluster_permutations=1),
    )
    assert np.isclose(result["mean_gross_underlying_move_bp"], 20.0)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("entry_position", 2, "next five-minute open"),
        ("exit_position", 48, "frozen 48-bar hold"),
        ("hold_bars", 47, "hold_bars"),
    ],
)
def test_simulator_rejects_execution_clock_drift(
    column: str,
    value: int,
    message: str,
) -> None:
    frame = _market_frame(52)
    schedule = _schedule(side=1, exit_=49)
    schedule.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        evaluator.simulate_funding_schedule(
            frame,
            _funding(),
            schedule,
            start="2023-01-01",
            end="2023-01-02",
            cfg=evaluator.EvaluationConfig(cluster_permutations=1),
        )


def test_slice_schedule_never_rebuilds_boundary_clock() -> None:
    schedule = pd.DataFrame(
        {
            "signal_date": ["2022-12-31 23:55:00", "2023-01-01 00:05:00"],
            "entry_date": ["2023-01-01 00:00:00", "2023-01-01 00:10:00"],
            "exit_date": ["2023-01-01 04:00:00", "2023-01-01 04:10:00"],
        }
    )
    selected = evaluator.slice_schedule(schedule, start="2023-01-01", end="2024-01-01")
    assert len(selected) == 1
    assert selected.iloc[0]["signal_date"] == "2023-01-01 00:05:00"


def test_qualification_enforces_half_count_gross_hurdle_and_score_controls() -> None:
    cfg = evaluator.EvaluationConfig()
    windows: dict[str, dict[str, dict[str, object]]] = {}
    for window in evaluator.WINDOWS:
        windows[window] = {policy: _metrics() for policy in evaluator.POLICY_NAMES}
    for control in evaluator.SCORE_BEARING_CONTROLS:
        windows["train"][control] = _metrics(ratio=2.0)
        windows["select2023"][control] = _metrics(ratio=2.0)
    windows["train"]["direction_flip"] = _metrics(ratio=99.0)
    windows["select2023"]["direction_flip"] = _metrics(ratio=99.0)
    assert evaluator._qualification(windows, cfg)["qualifies"]

    windows["select2023_h2"]["primary"] = _metrics(trades=44)
    result = evaluator._qualification(windows, cfg)
    assert not result["qualifies"]
    assert any("fewer than 45 trades" in item for item in result["failures"])

    windows["select2023_h2"]["primary"] = _metrics()
    windows["select2023"]["primary"] = _metrics(gross_bp=12.0)
    result = evaluator._qualification(windows, cfg)
    assert not result["qualifies"]
    assert any("not above 12 bp" in item for item in result["failures"])


def test_load_execution_market_requires_exact_clock_and_ohlc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_path = tmp_path / "market.csv.gz"
    market = _market_frame(3)
    market.to_csv(market_path, index=False, compression="gzip")
    digest = hashlib.sha256(market_path.read_bytes()).hexdigest()
    monkeypatch.setattr(evaluator, "MARKET_DATA", market_path)
    monkeypatch.setattr(evaluator, "MARKET_DATA_SHA256", digest)
    signal = pd.DataFrame({"date": market["date"]})
    frame, source = evaluator.load_execution_market(signal, {"rows": 3})
    assert frame[["open", "high", "low", "close"]].equals(
        market[["open", "high", "low", "close"]]
    )
    assert source["market_rows"] == 3

    broken = signal.copy()
    broken.loc[2, "date"] += pd.Timedelta("5min")
    with pytest.raises(ValueError, match="does not align"):
        evaluator.load_execution_market(broken, {"rows": 3})


def test_load_funding_requires_exact_epoch_and_sealed_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "funding.csv"
    row = {
        "funding_time_ms": 1_577_836_800_000,
        "funding_time_utc": "2020-01-01T00:00:00.000Z",
        "symbol": "BTCUSDT",
        "funding_rate": "-0.00012359",
        "mark_price": "",
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(evaluator, "FUNDING_DATA", path)
    monkeypatch.setattr(evaluator, "FUNDING_DATA_SHA256", digest)
    funding, source = evaluator.load_realized_funding({"data": {"rows": 1}})
    assert funding.loc[0, "funding_rate"] == pytest.approx(-0.00012359)
    assert source["funding_rows"] == 1

    row["funding_time_utc"] = "2020-01-01T00:00:00.001Z"
    pd.DataFrame([row]).to_csv(path, index=False)
    monkeypatch.setattr(
        evaluator,
        "FUNDING_DATA_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="differ from epoch"):
        evaluator.load_realized_funding({"data": {"rows": 1}})

    sealed_ms = 1_704_067_200_000
    row["funding_time_ms"] = sealed_ms
    row["funding_time_utc"] = "2024-01-01T00:00:00.000Z"
    pd.DataFrame([row]).to_csv(path, index=False)
    monkeypatch.setattr(
        evaluator,
        "FUNDING_DATA_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="sealed interval"):
        evaluator.load_realized_funding({"data": {"rows": 1}})


def test_run_stops_at_freeze_guard_before_loading_any_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator, "EVALUATION_FREEZE", tmp_path / "missing.json")

    def forbidden_signal_load() -> tuple[pd.DataFrame, dict[str, object]]:
        raise AssertionError("signal frame must not load before evaluator freeze")

    monkeypatch.setattr(luri, "load_causal_frame", forbidden_signal_load)
    with pytest.raises(ValueError, match="freeze manifest is missing"):
        evaluator.run_evaluation(evaluator.EvaluationConfig())


def test_run_reserves_all_clocks_before_loading_market_or_funding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    events: list[str] = []
    monkeypatch.setattr(evaluator, "verify_evaluation_freeze", lambda: {})
    monkeypatch.setattr(evaluator, "verify_preregistration", lambda: ({}, {}, {}))
    monkeypatch.setattr(luri, "load_causal_frame", lambda: (frame, {}))
    monkeypatch.setattr(
        evaluator,
        "verify_signal_replay",
        lambda *_args: ({}, {}, pd.DataFrame()),
    )

    def reserve_controls(*_args: object) -> dict[str, pd.DataFrame]:
        events.append("controls_reserved")
        return {name: pd.DataFrame() for name in evaluator.POLICY_NAMES}

    def open_market(*_args: object) -> tuple[pd.DataFrame, dict[str, object]]:
        assert events == ["controls_reserved"]
        assert set(reserved) == set(evaluator.POLICY_NAMES)
        events.append("market_opened")
        return frame, {}

    def open_funding(*_args: object) -> tuple[pd.DataFrame, dict[str, object]]:
        assert events == ["controls_reserved", "market_opened"]
        events.append("funding_opened")
        raise RuntimeError("stop after sequencing check")

    reserved = reserve_controls()
    events.clear()
    monkeypatch.setattr(
        evaluator,
        "build_control_schedules",
        lambda *_args: (events.append("controls_reserved") or reserved),
    )
    monkeypatch.setattr(evaluator, "load_execution_market", open_market)
    monkeypatch.setattr(evaluator, "load_realized_funding", open_funding)
    with pytest.raises(RuntimeError, match="sequencing check"):
        evaluator.run_evaluation(evaluator.EvaluationConfig())
    assert events == ["controls_reserved", "market_opened", "funding_opened"]


def test_freeze_manifest_rejects_opened_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"outcomes_opened_for_luri48": True}))
    monkeypatch.setattr(evaluator, "EVALUATION_FREEZE", freeze)
    with pytest.raises(ValueError, match="not frozen before outcomes"):
        evaluator.verify_evaluation_freeze()
