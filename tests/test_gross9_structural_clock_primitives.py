from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from training import gross9_structural_clock_primitives as primitives
from training.preregister_gross9_structural_clock_bundle import (
    discover_import_closure,
)


_RUNTIME_ROOTS = (
    Path("execution/gross9_rank7_clock_runtime.py"),
    Path("training/gross9_structural_clock_primitives.py"),
)
_EXPECTED_RUNTIME_CLOSURE = {
    Path("execution/gross9_rank7_clock_runtime.py"),
    Path("training/__init__.py"),
    Path("training/gross9_structural_clock_primitives.py"),
}
_BANNED_SOURCE_TERMS = {
    "_trade_stats",
    "cagr_pct",
    "economic_rank",
    "equity_curve",
    "funding_cash",
    "jaccard",
    "portfolio_pnl",
    "portfolio_return",
    "portfolio_valuation",
    "strict_mdd_pct",
}


def _semantic_helper_name(name: str) -> bool:
    lowered = name.casefold()
    return (
        ("portfolio" in lowered and ("return" in lowered or "pnl" in lowered))
        or ("funding" in lowered and "cash" in lowered)
        or "equity" in lowered
        or "drawdown" in lowered
        or "cagr" in lowered
        or "correlation" in lowered
        or "jaccard" in lowered
        or "containment" in lowered
        or "overlap" in lowered
        or lowered in {"rank_key", "economic_rank"}
    )


def _market(rows: int = 220) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100.0 + index * 0.02 + np.sin(index / 11.0)
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-01-01", periods=rows, freq="5min"
            ),
            "open": close,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close + 0.05,
            "volume": 10.0 + index % 7,
            "quote_asset_volume": 1_000.0 + index,
            "number_of_trades": 20.0 + index % 5,
            "taker_buy_base": 5.0 + index % 3,
            "taker_buy_quote": 450.0 + index,
            "kimchi_premium": 0.01 + index / 100_000,
            "usdkrw": 1_200.0 + index / 10,
        }
    )


def test_import_surface_excludes_forbidden_broad_modules_and_economics() -> None:
    source_path = Path(primitives.__file__)
    source = source_path.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = {
        "training.long_regime_combo_scan",
        "training.portfolio_opt_added_alpha_update",
        "training.portfolio_opt_all_discovered_alpha_gross10",
        "training.audit_fresh_kimchi_orthogonal_alpha",
        "training.compare_expanding_extratrees_rank7_refit_cadence_pre2025",
        "training.portfolio_opt_new_alpha_pool",
        "execution.portfolio_live",
        "execution.rank7_runtime",
        "execution.rex_llm_live",
    }
    assert imports.isdisjoint(forbidden)
    public = set(primitives.__all__)
    assert not public & {
        "portfolio_return",
        "pnl",
        "funding_cash",
        "cagr",
        "mdd",
        "rank",
        "overlap",
    }
    assert "gross_return" not in source


def test_runtime_import_closure_is_recursively_isolated_and_semantic_free() -> None:
    root = Path(__file__).resolve().parents[1]
    closure = set(discover_import_closure(_RUNTIME_ROOTS, root))
    assert closure == _EXPECTED_RUNTIME_CLOSURE

    violations: list[str] = []
    for path in sorted(closure):
        source = (root / path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=path.as_posix())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _semantic_helper_name(node.name):
                    violations.append(f"{path}: helper {node.name}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.casefold()
                for term in _BANNED_SOURCE_TERMS:
                    if term in lowered:
                        violations.append(f"{path}: term {term}")
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                modules = []
            for module in modules:
                leaf = module.rpartition(".")[2]
                if (
                    module.startswith(("training.", "execution."))
                    and module
                    not in {
                        "training.gross9_structural_clock_primitives",
                    }
                ) or leaf.startswith(
                    (
                        "audit_",
                        "backtest_",
                        "compare_",
                        "portfolio_",
                        "search_",
                    )
                ):
                    violations.append(f"{path}: import {module}")
    assert violations == []


def test_copied_rank7_causal_primitives_match_sources() -> None:
    from training.audit_confirmed_pullback_squeeze_live_parity import (
        decision_mask,
        live_decision_features,
    )
    from training.long_component_tp_union_scan import _component_mask
    from training.search_bocpd_state_gated_alpha import bocpd_student_t
    from training.search_kalman_state_gated_alpha import kalman_local_linear
    from training.search_liveparity_state_feature_interactions import (
        completed_hourly_features,
    )

    market = _market(800)
    dates = market["date"]
    np.testing.assert_array_equal(
        primitives._rank7_decision_mask(dates, "live_hour_signal_bar"),
        decision_mask(dates, "live_hour_signal_bar"),
    )
    feature_sample = pd.DataFrame(
        {
            "funding_rate": np.linspace(-0.001, 0.001, len(market)),
            "trend_96": np.linspace(-0.1, 0.1, len(market)),
            "premium_index_change": np.linspace(
                -0.001, 0.001, len(market)
            ),
            "htf_1d_return_4": np.linspace(-0.2, 0.2, len(market)),
        }
    )
    pdt.assert_frame_equal(
        primitives._rank7_live_decision_features(feature_sample),
        live_decision_features(feature_sample),
        check_exact=True,
    )
    for name in ("funding10_trend70", "premium20_mom90"):
        np.testing.assert_array_equal(
            primitives._rank7_component_mask(feature_sample, name),
            _component_mask(feature_sample, name),
        )
    actual_hourly, actual_state = (
        primitives._rank7_completed_hourly_features(market)
    )
    expected_hourly, expected_state = completed_hourly_features(market)
    pdt.assert_frame_equal(actual_hourly, expected_hourly, check_exact=True)
    pdt.assert_frame_equal(actual_state, expected_state, check_exact=True)

    log_price = np.log(np.linspace(100.0, 120.0, 80))
    train_var = float(np.var(np.diff(log_price)))
    np.testing.assert_array_equal(
        primitives._rank7_kalman_filter(
            log_price, 0.1, 0.001, 0.5, train_var
        ),
        kalman_local_linear(log_price, 0.1, 0.001, 0.5, train_var),
    )
    observations = np.column_stack(
        [np.sin(np.arange(80) / 7), np.cos(np.arange(80) / 9)]
    )
    actual_bocpd = primitives._rank7_bocpd(observations)
    expected_bocpd = bocpd_student_t(
        observations, hazard_lambda=336, max_run_length=1000
    )
    for key in expected_bocpd:
        np.testing.assert_array_equal(actual_bocpd[key], expected_bocpd[key])


def test_copied_preprocessing_nested_barrier_and_braid_match_sources() -> None:
    from preprocessing.binance_aux_features import (
        attach_binance_um_aux_frames as original_attach,
    )
    from preprocessing.market_features import (
        build_market_feature_frame as original_market_features,
    )
    from training.search_market_braid_alpha import (
        build_bar_state as original_braid_state,
        market_braid_events as original_braid_events,
    )
    from training.search_nested_barrier_witness_alpha import (
        build_barrier_bank as original_barrier_bank,
        coalesced_barrier_signals as original_barrier_signals,
    )

    market = _market(2200)
    funding = pd.DataFrame(
        {
            "date": market["date"].iloc[::96].reset_index(drop=True),
            "funding_rate": np.linspace(-0.001, 0.001, 23),
        }
    )
    premium = pd.DataFrame(
        {
            "date": market["date"].iloc[::12].reset_index(drop=True),
            "premium_index": np.linspace(-0.002, 0.002, 184),
        }
    )
    actual_attached = primitives.attach_binance_um_aux_frames(
        market, funding_frame=funding, premium_frame=premium
    )
    expected_attached = original_attach(
        market, funding_frame=funding, premium_frame=premium
    )
    pdt.assert_frame_equal(actual_attached, expected_attached, check_exact=True)
    pdt.assert_frame_equal(
        primitives.build_market_feature_frame(actual_attached, window_size=144),
        original_market_features(expected_attached, window_size=144),
        check_exact=True,
    )

    actual_bank = primitives._rank7_build_barrier_bank(market)
    expected_bank = original_barrier_bank(market)
    for key in (*primitives._RANK7_BARRIER_HORIZONS, "buy_work", "sell_work"):
        if isinstance(actual_bank[key], dict):
            for field in actual_bank[key]:
                np.testing.assert_array_equal(
                    actual_bank[key][field], expected_bank[key][field]
                )
        else:
            np.testing.assert_array_equal(actual_bank[key], expected_bank[key])
    actual_signals = primitives._rank7_coalesced_barrier_signals(
        market,
        actual_bank,
        min_coalescence=3,
        touch_width=0.001,
        branch="depleted_continuation",
    )
    expected_signals = original_barrier_signals(
        market,
        expected_bank,
        min_coalescence=3,
        touch_width=0.001,
        branch="depleted_continuation",
    )
    for actual, expected in zip(actual_signals[:2], expected_signals[:2]):
        np.testing.assert_array_equal(actual, expected)
    for key in actual_signals[2]:
        np.testing.assert_array_equal(
            actual_signals[2][key], expected_signals[2][key]
        )

    braid_market = market.assign(
        open_interest=np.linspace(1_000.0, 1_500.0, len(market)),
        open_interest_available=1.0,
        spot_close=market["close"] * 0.999,
        spot_rows=5.0,
        premium_index_1m_close=np.sin(np.arange(len(market)) / 50) / 1000,
        premium_rows=5.0,
    )
    actual_state = primitives._rank7_build_braid_state(braid_market)
    expected_state = original_braid_state(braid_market)
    for key in actual_state:
        np.testing.assert_array_equal(actual_state[key], expected_state[key])
    actual_events = primitives._rank7_market_braid_events(
        actual_state,
        shock_z=2.0,
        passage_z=0.5,
        max_age=144,
        topology_mode="relative_order",
    )
    expected_events = original_braid_events(
        expected_state,
        shock_z=2.0,
        passage_z=0.5,
        max_age=144,
        topology_mode="relative_order",
    )
    pdt.assert_frame_equal(actual_events, expected_events, check_exact=True)


def test_market_normalization_aux_and_open_interest_are_causal() -> None:
    market = _market(4).iloc[[2, 0, 1, 1, 3]].copy()
    market.iloc[3, market.columns.get_loc("close")] = 777.0
    normalized = primitives.normalise_market(market)
    assert normalized["date"].is_monotonic_increasing
    assert normalized["date"].is_unique
    assert normalized.loc[1, "close"] == 777.0

    oi = pd.DataFrame(
        {
            "date": normalized["date"],
            "open_interest": [1.0, np.nan, -1.0, 4.0],
        }
    )
    attached = primitives.attach_open_interest(normalized, oi)
    assert attached["open_interest_available"].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_interest_bidirectional_and_kimchi_features_match_sources() -> None:
    from training.long_regime_interest_gate_validation import (
        build_interest_features as original_interest,
    )
    from training.search_bidirectional_state_alpha import extra as original_extra
    from training.search_kimchi_leadlag_bidirectional_alpha import (
        features as original_kimchi,
    )

    market = _market(700)
    base = pd.DataFrame(
        {
            "funding_rate": np.linspace(-0.001, 0.001, len(market)),
            "premium_index": np.linspace(0.002, -0.002, len(market)),
        }
    )
    pdt.assert_frame_equal(
        primitives.build_interest_features(market, base),
        original_interest(market, base),
        check_exact=True,
    )
    pdt.assert_frame_equal(
        primitives.build_bidirectional_features(market, base),
        original_extra(market, base.copy()),
        check_exact=True,
    )
    pdt.assert_frame_equal(
        primitives.build_kimchi_features(market, base),
        original_kimchi(market, base.copy()),
        check_exact=True,
    )


def test_fresh_masks_require_freshness_and_schedule_requires_exclusive_side() -> None:
    market = _market(8)
    market["funding_available"] = [1, 0, 1, 1, 1, 1, 1, 1]
    market["usdkrw_available"] = 1.0
    features = pd.DataFrame(
        {
            "long": [1, 1, 0, 1, 0, 1, 0, 0],
            "short": [0, 0, 1, 1, 0, 0, 1, 0],
        }
    )
    long_active, short_active, diagnostics = primitives.fresh_masks(
        market,
        features,
        long_conditions=[{"feature": "long", "op": "ge", "threshold": 1}],
        short_conditions=[
            {"feature": "short", "op": ">=", "threshold": 1}
        ],
        long_availability=["funding_available"],
        short_availability=["usdkrw_available"],
    )
    assert long_active.tolist() == [True, False, False, True, False, True, False, False]
    assert short_active.tolist() == [False, False, True, True, False, False, True, False]
    assert diagnostics["blocked_stale_long_rows"] == 1
    exclusive = np.logical_xor(long_active, short_active)
    assert np.flatnonzero(exclusive).tolist() == [0, 2, 5, 6]


def test_stop_precedes_take_and_nonoverlap_skips_conflicts() -> None:
    market = _market(12)
    market.loc[:, "open"] = 100.0
    market.loc[:, "high"] = 100.0
    market.loc[:, "low"] = 100.0
    market.loc[1, ["high", "low"]] = [102.0, 98.0]
    first = primitives.structural_trade_at(
        market, 0, 1, 4, 100, 100
    )
    assert first == primitives.StructuralTrade(0, 1, 1, 1, "stop")
    trades = primitives.walk_structural_schedule(
        market,
        [0, 1, 2, 4],
        [1, 1, 1, -1],
        hold_bars=2,
        take_bps=1_000_000,
        stop_bps=1_000_000,
    )
    assert [(row.signal_position, row.exit_position) for row in trades] == [
        (0, 3),
        (4, 7),
    ]


def test_markov_hourly_features_and_mapping_match_source() -> None:
    from training.search_gaussian_hmm_regime_alpha import hourly_features

    market = _market(800)
    actual_hourly, actual_features = (
        primitives.completed_hourly_markov_features(market)
    )
    expected_hourly, expected_features = hourly_features(market)
    pdt.assert_frame_equal(actual_hourly, expected_hourly, check_exact=True)
    pdt.assert_frame_equal(actual_features, expected_features, check_exact=True)


def test_both_rex_gate_dialects_match_authenticated_sources() -> None:
    from training.audit_rex8640_usdkrw_gate import gate_match
    from training.portfolio_opt_all_discovered_alpha_gross10 import (
        _rex_row_matches,
    )

    features = pd.DataFrame({"width": [0.4], "other": [-1.0]})
    veto_row = {
        "signal_pos": 0,
        "state_tokens": {"regime": "risk_on"},
    }
    veto_gates = [
        {"feature": "width", "op": ">=", "threshold": 0.3},
        {"feature": "tok:regime", "op": "==", "threshold": "risk_on"},
    ]
    assert primitives.rex_veto_gate_match(
        veto_gates, features, veto_row
    ) == _rex_row_matches(veto_gates, features, veto_row)

    taker_row = {"feature_snapshot": {"width": 0.4, "other": -1.0}}
    taker_gates = [
        {"feature": "width", "op": ">=", "threshold": 0.3},
        {"feature": "other", "op": "<=", "threshold": 0.0},
    ]
    assert primitives.rex_taker_gate_match(
        taker_row, taker_gates
    ) == gate_match(taker_row, taker_gates)


def _rank7_base() -> dict:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2020-07-02",
                "2021-01-02",
                "2021-06-02",
                "2022-01-02",
                "2022-06-02",
                "2023-01-02",
                "2023-06-02",
            ]
        )
    )
    signals = np.arange(len(dates), dtype=np.int64)
    return {
        "context": {
            "dates": dates,
            "market": pd.DataFrame(index=np.arange(10)),
            "matrix": np.arange(28, dtype=float).reshape(7, 4),
        },
        "signals": signals,
        "funding_source": np.array([True, False, True, False, True, False, True]),
        "targets": np.column_stack(
            [np.linspace(0.0, 0.06, 7), np.linspace(0.01, 0.07, 7)]
        ),
        "signal_dates": dates,
        "exit_dates": dates.to_numpy() + np.timedelta64(1, "D"),
        "width": np.linspace(0.1, 0.7, 7),
        "pullback": np.linspace(-0.7, -0.1, 7),
    }


def test_rank7_fold_purge_and_balanced_weights_match_sources() -> None:
    from training.compare_expanding_extratrees_rank7_refit_cadence_pre2025 import (
        cutoff_masks,
    )
    from training.select_expanding_extratrees_top10_pre2025 import (
        _balanced_weights,
    )

    base = _rank7_base()
    base["exit_dates"][4] = np.datetime64("2023-01-01")
    expected_fit, expected_predict = cutoff_masks(
        base, "2023-01-01", "2024-01-01"
    )
    actual_fit, actual_predict = primitives.annual_rank7_masks(
        base, "2023-01-01", "2024-01-01"
    )
    np.testing.assert_array_equal(actual_fit, expected_fit)
    np.testing.assert_array_equal(actual_predict, expected_predict)
    np.testing.assert_array_equal(
        primitives.balanced_rank7_weights(base, actual_fit),
        _balanced_weights(base, actual_fit),
    )


def test_rank7_deterministic_predictions_thresholds_and_activation_match() -> None:
    from sklearn.ensemble import ExtraTreesRegressor
    from training.search_stable_ensemble_conditional_pullback_alpha import (
        deterministic_forest_predict,
        source_thresholds,
    )

    x = np.arange(60, dtype=float).reshape(15, 4)
    y = np.column_stack([x[:, 0] / 10, x[:, 1] / 20])
    model = ExtraTreesRegressor(
        n_estimators=7,
        max_depth=2,
        min_samples_leaf=2,
        random_state=71,
        n_jobs=-1,
    ).fit(x, y)
    expected = deterministic_forest_predict(model, x)
    actual = primitives.deterministic_extra_trees_predict(model, x)
    np.testing.assert_array_equal(actual, expected)
    funding = np.array([True, False] * 7 + [True])
    assert primitives.rank7_source_thresholds(
        actual[:, 0], funding, funding_q=0.2, premium_q=0.6
    ) == source_thresholds(
        actual[:, 0], funding, funding_q=0.2, premium_q=0.6
    )


def test_delay_and_immutable_anchor_no_overlap_geometry() -> None:
    matrix = np.arange(12, dtype=float).reshape(6, 2)
    expected = np.vstack([np.zeros((2, 2)), matrix[:-2]])
    np.testing.assert_array_equal(
        primitives.apply_rank7_delay(matrix, bars=2), expected
    )
    np.testing.assert_array_equal(
        primitives.immutable_anchors(
            np.array([True, True, False, True, True, False, True]), 3
        ),
        np.array([True, False, False, True, False, False, True]),
    )


def test_rank7_facade_helper_signature_is_exact() -> None:
    signature = inspect.signature(primitives.rank7_rebuild_feature_context)
    assert str(signature) == (
        "(market, *, medians, clip, delay_bars, hourly_history) -> 'dict'"
    )


@pytest.mark.parametrize("bars", [0, 3, 6, 9])
def test_delay_geometry_for_boundary_lengths(bars: int) -> None:
    matrix = np.arange(12, dtype=float).reshape(6, 2)
    actual = primitives.apply_rank7_delay(matrix, bars=bars)
    assert actual.shape == matrix.shape
    if bars == 0:
        np.testing.assert_array_equal(actual, matrix)
    elif bars >= len(matrix):
        np.testing.assert_array_equal(actual, np.zeros_like(matrix))
    else:
        np.testing.assert_array_equal(actual[bars:], matrix[:-bars])
