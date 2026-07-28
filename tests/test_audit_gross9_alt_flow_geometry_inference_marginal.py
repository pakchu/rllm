from __future__ import annotations

import csv
import gzip
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import training.audit_gross9_alt_flow_geometry_inference_marginal as mod
from training.build_six_alt_price_free_flow_panel import OUTPUT_COLUMNS
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
)
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine


def _market(rows: int = 400) -> pd.DataFrame:
    position = np.arange(rows, dtype=float)
    opens = 100.0 * np.exp(0.0002 * position)
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-01-01", periods=rows, freq="5min"
            ),
            "open": opens,
            "high": opens * 1.003,
            "low": opens * 0.997,
            "close": opens,
        }
    )


def _execution_engine(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    leverage: float = 0.5,
    cost_rate: float = 0.0006,
) -> ExecutionEngine:
    return ExecutionEngine(
        market,
        funding,
        ExecutionConfig(
            input_csv="",
            metrics_csv="",
            funding_csv="",
            output="",
            manifest_output="",
            leverage=leverage,
            fee_rate=cost_rate,
            slippage_rate=0.0,
        ),
    )


def test_preregistration_is_hash_bound_total_and_future_closed() -> None:
    payload = mod.load_preregistration(mod.PREREGISTRATION)
    assert payload["physical_selection_cutoff"] == "2025-01-01"
    assert tuple(payload["feature_contract"]["columns"]) == (
        mod.FEATURE_COLUMNS
    )
    assert len(mod.FEATURE_COLUMNS) == 52
    assert payload["candidate_universe"]["portfolio_cells"] == 12
    assert payload["future_veto_contract"]["future_can_rerank"] is False
    assert payload["future_veto_contract"]["future_can_repair"] is False
    assert 'choices=("pre2025",)' in inspect.getsource(mod.main)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cutoff", "2026-01-01"),
        ("cost_rate", 0.0005),
        ("stress_cost_rate", 0.0009),
        ("leverage", 1.0),
        ("stride_bars", 6),
    ],
)
def test_run_rejects_frozen_config_override(
    field: str,
    value: str | float | int,
) -> None:
    cfg = mod.Config(**{field: value})
    with pytest.raises(RuntimeError, match="frozen config drifted"):
        mod.run_pre2025(cfg)


def test_robust_z_uses_720_strictly_prior_observations() -> None:
    values = pd.DataFrame({"x": np.arange(721, dtype=float)})
    observed = mod.robust_z_prior(values)
    prior = values["x"].iloc[:720]
    expected = (
        values["x"].iloc[720] - prior.quantile(0.50)
    ) / ((prior.quantile(0.75) - prior.quantile(0.25)) / 1.349)
    assert np.isclose(observed["x"].iloc[720], expected)
    changed = values.copy()
    changed.loc[720, "x"] = 1_000_000.0
    changed_z = mod.robust_z_prior(changed)
    scale = (prior.quantile(0.75) - prior.quantile(0.25)) / 1.349
    assert np.isclose(
        changed_z["x"].iloc[720] - observed["x"].iloc[720],
        (1_000_000.0 - 720.0) / scale,
    )


def test_pc1_orientation_and_geometry_are_deterministic() -> None:
    assert np.array_equal(
        mod._orient_pc1(np.asarray([-0.8, -0.2])),
        np.asarray([0.8, 0.2]),
    )
    base = np.linspace(-1.0, 1.0, 220)
    values = np.column_stack(
        [
            base + 0.01 * (index + 1) * np.sin(np.arange(220) / 7.0)
            for index in range(len(mod.SYMBOLS))
        ]
    )
    flow = pd.DataFrame(values, columns=mod.SYMBOLS)
    mean6 = flow.rolling(6, min_periods=6).mean()
    first = mod.pca_geometry(flow, mean6)
    second = mod.pca_geometry(flow, mean6)
    assert first.equals(second)
    assert tuple(first.columns) == mod.FEATURE_COLUMNS[-7:-1]
    assert np.isfinite(first.iloc[-1]).all()


def test_source_prefix_stops_before_cutoff_without_parsing_future_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.csv.gz"
    hours = pd.date_range("2023-01-01", periods=3, freq="h")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLUMNS)
        for hour in hours:
            for symbol in mod.SYMBOLS:
                row = {name: "0" for name in OUTPUT_COLUMNS}
                row.update(
                    {
                        "source_hour_open_utc": str(
                            hour - pd.Timedelta("1h")
                        ),
                        "feature_available_time_utc": str(hour),
                        "symbol": symbol,
                        "quote_volume_usdt": (
                            "not-parsed"
                            if hour == hours[-1]
                            else ("" if hour == hours[1] else "1000")
                        ),
                        "trade_count": (
                            "not-parsed"
                            if hour == hours[-1]
                            else ("" if hour == hours[1] else "100")
                        ),
                        "taker_flow_fraction": (
                            "not-parsed"
                            if hour == hours[-1]
                            else ("" if hour == hours[1] else "0.1")
                        ),
                        "feature_valid": (
                            "not-parsed"
                            if hour == hours[-1]
                            else (
                                "false" if hour == hours[1] else "true"
                            )
                        ),
                    }
                )
                writer.writerow([row[name] for name in OUTPUT_COLUMNS])
    frame = mod.read_source_prefix(path, cutoff=hours[-1])
    assert len(frame) == 2 * len(mod.SYMBOLS)
    assert frame["date"].max() == hours[-2]
    assert frame.loc[frame["date"] == hours[1], "quote_volume_usdt"].isna().all()


def test_fold_masks_purge_boundary_crossing_targets() -> None:
    signal = pd.Series(
        pd.to_datetime(
            [
                "2023-03-31 11:55",
                "2023-03-31 12:00",
                "2023-04-01 00:00",
                "2023-06-30 12:00",
            ]
        )
    )
    exits = pd.Series(
        pd.to_datetime(
            [
                "2023-03-31 23:55",
                "2023-04-01 00:00",
                "2023-04-01 12:05",
                "2023-07-01 00:05",
            ]
        )
    )
    fit, predict = mod.fold_masks(
        signal,
        exits,
        np.ones(4, dtype=bool),
        np.ones(4, dtype=bool),
        fit_end_exclusive="2023-04-01",
        prediction_start="2023-04-01",
        prediction_end_exclusive="2023-07-01",
    )
    assert np.array_equal(fit, [True, False, False, False])
    assert np.array_equal(predict, [False, False, True, False])


def test_extratrees_predictions_are_bitwise_reproducible() -> None:
    rng = np.random.default_rng(29)
    matrix = rng.normal(size=(1_300, 5))
    targets = rng.normal(size=(1_300, 4))
    fit = np.zeros(1_300, dtype=bool)
    fit[:1_100] = True
    predict = ~fit
    preregistration = {
        "learner_contract": {
            "n_estimators": 16,
            "max_depth": 3,
            "min_samples_leaf": 24,
            "max_features": 0.75,
        }
    }
    first, first_meta = mod._fit_predict_ensemble(
        matrix, targets, fit, predict, preregistration
    )
    second, second_meta = mod._fit_predict_ensemble(
        matrix, targets, fit, predict, preregistration
    )
    assert np.array_equal(first, second)
    assert first_meta == second_meta


def test_no_stop_target_uses_next_open_fixed_exit_cost_and_funding() -> None:
    market = _market(220)
    signal = 20
    hold = 12
    entry = signal + 1
    exit_position = entry + hold
    funding = pd.DataFrame(
        {
            "date": [
                market["date"].iloc[entry],
                market["date"].iloc[exit_position],
            ],
            "funding_rate": [0.001, -0.0004],
        }
    )
    leverage = 0.5
    cost_rate = 0.0006
    engine = _execution_engine(
        market,
        funding,
        leverage=leverage,
        cost_rate=cost_rate,
    )
    trade = mod.no_stop_trade(engine, signal, 1, hold=hold)
    assert trade is not None
    assert trade.entry_position == entry
    assert trade.exit_position == exit_position
    expected_funding = (1.0 - leverage * 0.001) * (
        1.0 + leverage * 0.0004
    )
    assert np.isclose(trade.funding_factor, expected_funding)
    result, adverse = mod.trade_target(
        trade, leverage=leverage, cost_rate=cost_rate
    )
    expected = (
        (1.0 - leverage * cost_rate)
        * trade.price_factor
        * expected_funding
        * (1.0 - leverage * cost_rate)
        - 1.0
    )
    assert np.isclose(result, expected)
    assert adverse >= 0.0


def test_schedules_are_non_overlapping_and_preserve_predicted_side() -> None:
    market = _market()
    funding = pd.DataFrame(columns=("date", "funding_rate"))
    long_active = np.zeros(len(market), dtype=bool)
    short_active = np.zeros(len(market), dtype=bool)
    long_active[[10, 20]] = True
    short_active[160] = True
    schedules, meta = mod.build_schedules(
        market,
        funding,
        {"train": np.ones(len(market), dtype=bool)},
        {
            "candidate": {
                "long_active": long_active,
                "short_active": short_active,
            }
        },
        mod.Config(),
    )
    trades = schedules["candidate"]["train"]
    assert [(trade.signal_position, trade.side) for trade in trades] == [
        (10, 1),
        (160, -1),
    ]
    assert trades[0].exit_position < trades[1].entry_position
    assert meta["candidate"]["train"]["longs"] == 1
    assert meta["candidate"]["train"]["shorts"] == 1


def test_unavailable_sliced_win_rate_is_not_reported_as_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod.portfolio, "SLEEVES", ("candidate",))
    data = {
        "R": np.asarray([[0.01, 0.0]]),
        "A": np.zeros((1, 2)),
        "U": np.zeros((1, 2)),
        "L": np.zeros((1, 2)),
        "H": np.zeros((1, 2)),
        "dates": pd.date_range("2023-07-01", periods=2, freq="5min"),
        "counts": np.asarray([1]),
        "wins": np.asarray([0]),
        "entry_positions": {"candidate": np.asarray([0])},
        "win_rate_available": False,
    }
    result = mod.metric(
        {"selection_2023h2": data},
        "selection_2023h2",
        {"candidate": 1.0},
    )
    assert result["win_rate"] is None


def test_target_anchor_surface_excludes_incomplete_horizon() -> None:
    rows = mod.HOLD_BARS + 60
    matrix = np.full((rows, 2), np.nan)
    final_valid = rows - mod.HOLD_BARS - 2
    matrix[[54, 55, final_valid, final_valid + 1]] = 1.0
    anchors = mod.eligible_anchors(matrix, market_rows=rows)
    assert np.array_equal(anchors, [54, 55, final_valid])
    assert anchors[-1] == final_valid
    run_source = inspect.getsource(mod.run_pre2025)
    assert '"future_opened": False' in run_source
    assert '"family_closed_if_rejected": True' in run_source
