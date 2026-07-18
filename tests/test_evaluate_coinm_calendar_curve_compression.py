from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from training import evaluate_coinm_calendar_curve_compression as module


def _trade(
    *,
    front_side: int = 1,
    front_exit: float = 101.0,
    next_exit: float = 101.0,
    front_high: float = 102.0,
    front_low: float = 99.0,
    next_high: float = 102.0,
    next_low: float = 99.0,
) -> module.PairTrade:
    hold = module.CANDIDATE.hold_bars
    return module.PairTrade(
        confirmation_position=0,
        entry_position=1,
        exit_position=1 + hold,
        front_symbol="F",
        next_symbol="N",
        front_side=front_side,
        next_side=-front_side,
        front_entry=100.0,
        front_exit=front_exit,
        next_entry=100.0,
        next_exit=next_exit,
        front_highs=(front_high,) * hold,
        front_lows=(front_low,) * hold,
        next_highs=(next_high,) * hold,
        next_lows=(next_low,) * hold,
        entry_date="2023-01-01 00:05:00",
    )


def test_inverse_usd_return_matches_fixed_face_linear_mark() -> None:
    assert module.inverse_usd_return(100.0, 110.0, 1) == pytest.approx(0.10)
    assert module.inverse_usd_return(100.0, 110.0, -1) == pytest.approx(-0.10)
    assert module.inverse_usd_return(100.0, 90.0, -1) == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("entry", "mark", "side"),
    [(0.0, 100.0, 1), (-1.0, 100.0, 1), (100.0, 0.0, 1), (100.0, 100.0, 0)],
)
def test_inverse_usd_return_rejects_invalid_inputs(
    entry: float, mark: float, side: int
) -> None:
    with pytest.raises(ValueError):
        module.inverse_usd_return(entry, mark, side)


def test_equal_leg_move_is_delta_neutral_proxy() -> None:
    assert module.pair_curve_return(100.0, 110.0, 1, 100.0, 110.0, -1) == pytest.approx(0.0)
    assert module.pair_curve_return(100.0, 110.0, 1, 200.0, 220.0, -1) == pytest.approx(0.0)
    assert module.pair_curve_return(100.0, 110.0, 1, 200.0, 210.0, -1) != pytest.approx(0.0)


def test_rich_next_compression_is_positive() -> None:
    value = module.pair_curve_return(100.0, 101.0, 1, 102.0, 101.0, -1)
    assert value > 0.0


def test_pair_curve_rejects_nonopposite_sides() -> None:
    with pytest.raises(ValueError, match="opposite"):
        module.pair_curve_return(100.0, 101.0, 1, 100.0, 101.0, 1)


def test_strict_mdd_includes_independent_leg_extremes_and_all_costs() -> None:
    cfg = module.EvaluationConfig(cluster_permutations=10)
    stats = module.strict_equity_stats(
        [_trade()], start="2023-01-01", end="2024-01-01", cfg=cfg
    )
    assert stats["strict_mdd_pct"] > 1.0
    expected_net = (
        module.CANDIDATE.total_gross / 2.0 * 0.0
        - 2.0
        * module.CANDIDATE.total_gross
        * cfg.cost_rate_per_leg_per_side
    )
    assert stats["mean_net_bps"] == pytest.approx(expected_net * 10_000.0)


def test_strict_mdd_favorable_before_adverse_is_exact() -> None:
    cfg = replace(module.EvaluationConfig(), cost_rate_per_leg_per_side=0.0)
    stats = module.strict_equity_stats(
        [
            _trade(
                front_exit=100.0,
                next_exit=100.0,
                front_high=110.0,
                front_low=90.0,
                next_high=110.0,
                next_low=90.0,
            )
        ],
        start="2023-01-01",
        end="2024-01-01",
        cfg=cfg,
    )
    expected = (1.05 - 0.95) / 1.05 * 100.0
    assert stats["strict_mdd_pct"] == pytest.approx(expected)


def test_two_leg_round_trip_cost_is_exact() -> None:
    cfg = replace(module.EvaluationConfig(), cost_rate_per_leg_per_side=0.001)
    stats = module.strict_equity_stats(
        [
            _trade(
                front_exit=100.0,
                next_exit=100.0,
                front_high=100.0,
                front_low=100.0,
                next_high=100.0,
                next_low=100.0,
            )
        ],
        start="2023-01-01",
        end="2024-01-01",
        cfg=cfg,
    )
    expected_loss = 2.0 * module.CANDIDATE.total_gross * 0.001
    assert stats["absolute_return_pct"] == pytest.approx(-expected_loss * 100.0)
    assert stats["strict_mdd_pct"] == pytest.approx(expected_loss * 100.0)


def test_build_trades_uses_confirmation_next_open_and_both_legs() -> None:
    hold = module.CANDIDATE.hold_bars
    dates = pd.date_range("2023-01-01", periods=hold + 3, freq="5min")
    outcome = pd.DataFrame(
        {
            "signal_bar_open_utc": dates,
            "front_symbol": ["F"] * len(dates),
            "next_symbol": ["N"] * len(dates),
            "front_open": [100.0] * len(dates),
            "front_high": [101.0] * len(dates),
            "front_low": [99.0] * len(dates),
            "front_close": [100.0] * len(dates),
            "next_open": [100.0] * len(dates),
            "next_high": [101.0] * len(dates),
            "next_low": [99.0] * len(dates),
            "next_close": [100.0] * len(dates),
        }
    )
    schedule = pd.DataFrame(
        [
            {
                "confirmation_bar_open": str(dates[0]),
                "entry_time": str(dates[1]),
                "exit_time": str(dates[1 + hold]),
                "front_symbol": "F",
                "next_symbol": "N",
                "front_side": 1,
                "next_side": -1,
            }
        ]
    )
    positions = {timestamp: i for i, timestamp in enumerate(dates)}
    trades = module._build_trades(outcome, positions, schedule)
    assert len(trades) == 1
    assert trades[0].entry_position == 1
    assert len(trades[0].front_highs) == hold


def test_build_trades_fails_closed_on_missing_held_path() -> None:
    hold = module.CANDIDATE.hold_bars
    dates = pd.date_range("2023-01-01", periods=hold + 3, freq="5min")
    outcome = pd.DataFrame(
        {
            "signal_bar_open_utc": dates,
            "front_symbol": ["F"] * len(dates),
            "next_symbol": ["N"] * len(dates),
            "front_open": [100.0] * len(dates),
            "front_high": [101.0] * len(dates),
            "front_low": [99.0] * len(dates),
            "next_open": [100.0] * len(dates),
            "next_high": [101.0] * len(dates),
            "next_low": [99.0] * len(dates),
        }
    )
    outcome.loc[10, "next_low"] = np.nan
    schedule = pd.DataFrame(
        [
            {
                "confirmation_bar_open": str(dates[0]),
                "entry_time": str(dates[1]),
                "exit_time": str(dates[1 + hold]),
                "front_symbol": "F",
                "next_symbol": "N",
                "front_side": 1,
                "next_side": -1,
            }
        ]
    )
    with pytest.raises(ValueError, match="missing outcome"):
        module._build_trades(
            outcome,
            {timestamp: i for i, timestamp in enumerate(dates)},
            schedule,
        )


def test_build_trades_fails_closed_on_contract_transition() -> None:
    hold = module.CANDIDATE.hold_bars
    dates = pd.date_range("2023-01-01", periods=hold + 3, freq="5min")
    outcome = pd.DataFrame(
        {
            "signal_bar_open_utc": dates,
            "front_symbol": ["F"] * len(dates),
            "next_symbol": ["N"] * len(dates),
            "front_open": [100.0] * len(dates),
            "front_high": [101.0] * len(dates),
            "front_low": [99.0] * len(dates),
            "next_open": [100.0] * len(dates),
            "next_high": [101.0] * len(dates),
            "next_low": [99.0] * len(dates),
        }
    )
    outcome.loc[10, "front_symbol"] = "F2"
    schedule = pd.DataFrame(
        [
            {
                "confirmation_bar_open": str(dates[0]),
                "entry_time": str(dates[1]),
                "exit_time": str(dates[1 + hold]),
                "front_symbol": "F",
                "next_symbol": "N",
                "front_side": 1,
                "next_side": -1,
            }
        ]
    )
    with pytest.raises(ValueError, match="front leg crosses"):
        module._build_trades(
            outcome,
            {timestamp: i for i, timestamp in enumerate(dates)},
            schedule,
        )


def _delay_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2023-01-01", periods=40, freq="5min")
    source = pd.DataFrame(
        {
            "signal_bar_open_utc": dates,
            "feature_available_time_utc": dates + pd.Timedelta("5min"),
            "trade_earliest_time_utc": dates + pd.Timedelta("5min"),
            "front_symbol": ["F"] * len(dates),
            "next_symbol": ["N"] * len(dates),
            "feature_valid": [True] * len(dates),
            "front_hours_to_delivery": [1000.0] * len(dates),
            "next_hours_to_delivery": [2000.0] * len(dates),
        }
    )
    confirmation = 5
    entry = dates[confirmation] + pd.Timedelta("5min")
    schedule = pd.DataFrame(
        [
            {
                "shock_bar_open": str(dates[confirmation - 1]),
                "confirmation_bar_open": str(dates[confirmation]),
                "feature_available": str(entry),
                "entry_time": str(entry),
                "exit_time": str(
                    entry + pd.Timedelta(minutes=5 * module.CANDIDATE.hold_bars)
                ),
                "front_symbol": "F",
                "next_symbol": "N",
                "front_side": 1,
                "next_side": -1,
                "hold_bars": module.CANDIDATE.hold_bars,
            }
        ]
    )
    return source, schedule


def test_delayed_schedule_uniformly_shifts_frozen_sides() -> None:
    source, schedule = _delay_fixture()
    delayed = module.delayed_schedule(
        schedule,
        source,
        bars=2,
        start="2023-01-01",
        end="2023-02-01",
    )
    assert len(delayed) == 1
    assert delayed.iloc[0].front_side == schedule.iloc[0].front_side
    assert delayed.iloc[0].next_side == schedule.iloc[0].next_side
    assert pd.Timestamp(delayed.iloc[0].confirmation_bar_open) == pd.Timestamp(
        schedule.iloc[0].confirmation_bar_open
    ) + pd.Timedelta("10min")


@pytest.mark.parametrize("defect", ["front_symbol", "feature_valid", "delivery"])
def test_delayed_schedule_skips_invalid_destination(defect: str) -> None:
    source, schedule = _delay_fixture()
    destination = 7
    if defect == "front_symbol":
        source.loc[destination, "front_symbol"] = "F2"
    elif defect == "feature_valid":
        source.loc[destination, "feature_valid"] = False
    else:
        source.loc[destination, "front_hours_to_delivery"] = 1.0
    delayed = module.delayed_schedule(
        schedule,
        source,
        bars=2,
        start="2023-01-01",
        end="2023-02-01",
    )
    assert delayed.empty


def test_stable_hash_excludes_created_at() -> None:
    assert module._stable_artifact_hash({"a": 1, "created_at": "x"}) == module._stable_artifact_hash(
        {"a": 1, "created_at": "y"}
    )


def test_modified_config_is_rejected() -> None:
    cfg = replace(module.EvaluationConfig(), cluster_seed=1)
    with pytest.raises(ValueError, match="frozen"):
        module._require_canonical_config(cfg)


def test_rebuilt_support_matches_frozen_artifact() -> None:
    support = module._verify_static_dependencies()
    source = module.load_source(support["source"]["csv"])
    schedules, _, _ = module._rebuild_schedules(source)
    module._verify_rebuilt_support(source, schedules, support)


def test_freeze_path_never_loads_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(module.Path, "exists", lambda _self: False)
    monkeypatch.setattr(
        module,
        "_verify_static_dependencies",
        lambda: {"manifest_hash": "support-hash"},
    )

    def fake_sha(path: object) -> str:
        if str(path) == module.EvaluationConfig.source_csv:
            return module.EXPECTED_SOURCE_SHA256
        if str(path) == module.EvaluationConfig.manifest_json:
            return module.EXPECTED_MANIFEST_SHA256
        return "frozen-sha"

    monkeypatch.setattr(module, "_sha256", fake_sha)
    monkeypatch.setattr(
        module,
        "_load_outcomes",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("outcomes opened")),
    )
    monkeypatch.setattr(
        module,
        "_write_json_exclusive",
        lambda _path, payload: captured.update(payload),
    )
    report = module.freeze_evaluator(module.EvaluationConfig())
    assert report["simulation_run"] is False
    assert report["candidate_returns_computed_before_freeze"] is False
    assert captured["freeze_hash"] == report["freeze_hash"]


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("support_artifact_sha256", "wrong-support-bytes"),
        ("sealed_windows", ["fit", "select_2023"]),
    ],
)
def test_freeze_verification_rejects_rehashed_contract_tampering(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    tampered: object,
) -> None:
    cfg = module.EvaluationConfig()
    support = {"manifest_hash": "support-manifest"}
    freeze: dict[str, object] = {
        "schema_version": 1,
        "created_at": "2026-07-19T00:00:00+00:00",
        "support_commit": module.SUPPORT_COMMIT,
        "support_manifest_hash": support["manifest_hash"],
        "support_artifact_sha256": "support-artifact",
        "evaluation_source": str(module.EVALUATOR_SOURCE),
        "evaluation_source_sha256": "evaluator-source",
        "config": asdict(cfg),
        "source_sha256": module.EXPECTED_SOURCE_SHA256,
        "manifest_sha256": module.EXPECTED_MANIFEST_SHA256,
        "opened_windows": [],
        "sealed_windows": list(module.EVALUATOR_SEALED_WINDOWS),
        "candidate_returns_computed_before_freeze": False,
        "simulation_run": False,
        "mutable_parameters": [],
    }
    freeze[field] = tampered
    freeze["freeze_hash"] = module._stable_artifact_hash(freeze)

    def fake_sha(path: object) -> str:
        if str(path) == str(module.EVALUATOR_SOURCE):
            return "evaluator-source"
        if str(path) == str(module.SUPPORT_RESULT):
            return "support-artifact"
        if str(path) == cfg.source_csv:
            return module.EXPECTED_SOURCE_SHA256
        if str(path) == cfg.manifest_json:
            return module.EXPECTED_MANIFEST_SHA256
        raise AssertionError(f"unexpected hash path: {path}")

    monkeypatch.setattr(module, "_read_json", lambda _path: freeze)
    monkeypatch.setattr(module, "_sha256", fake_sha)
    with pytest.raises(ValueError, match="support artifact|sealed windows"):
        module.verify_evaluator_freeze(cfg, support)
