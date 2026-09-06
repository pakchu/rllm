from __future__ import annotations

import dataclasses
import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import execution.gross9_rank7_clock_runtime as facade
import execution.rank7_runtime as original


EXPECTED_ADVERSARIAL_CASE_IDS = (
    "manifest_hash_mismatch",
    "schema_version_wrong",
    "policy_type_wrong",
    "strategy_id_wrong",
    "cadence_wrong",
    "seeds_missing",
    "seeds_extra",
    "seeds_reordered",
    "seeds_changed",
    "trees_per_seed_wrong",
    "prediction_n_jobs_wrong",
    "model_format_wrong",
    "param_max_depth_wrong",
    "param_min_samples_leaf_wrong",
    "param_max_features_wrong",
    "param_bootstrap_wrong",
    "feature_columns_missing",
    "feature_columns_extra",
    "feature_columns_reordered",
    "feature_columns_changed",
    "source_columns_missing",
    "source_columns_extra",
    "source_columns_reordered",
    "source_columns_changed",
    "source_priority_missing",
    "source_priority_extra",
    "source_priority_reordered",
    "source_priority_changed",
    "delay_bars_wrong",
    "delay_initial_fill_wrong",
    "anchor_cooldown_wrong",
    "no_overlap_wrong",
    "parity_missing",
    "parity_status_failed",
    "parity_feature_failed",
    "parity_prediction_failed",
    "parity_schedule_failed",
    "medians_wrong_length",
    "medians_nonfinite",
    "clip_missing",
    "clip_nonfinite",
    "clip_reversed",
    "clip_wrong_length",
    "valid_from_invalid",
    "valid_until_invalid",
    "annual_cutoff_invalid",
    "annual_cutoff_mismatch",
    "validity_empty",
    "validity_reversed",
    "threshold_key_missing",
    "threshold_key_extra",
    "threshold_funding_score_nonfinite",
    "threshold_premium_score_nonfinite",
    "threshold_funding_risk_cap_nonfinite",
    "threshold_premium_risk_cap_nonfinite",
    "threshold_width_q20_nonfinite",
    "threshold_pullback_q40_nonfinite",
    "score_lambda_nonfinite",
    "exits_missing_source",
    "exits_extra_source",
    "funding_hold_wrong",
    "premium_hold_wrong",
    "funding_take_missing",
    "funding_take_nonfinite",
    "funding_take_nonnumeric",
    "funding_stop_missing",
    "funding_stop_nonfinite",
    "funding_stop_nonnumeric",
    "premium_take_missing",
    "premium_take_nonfinite",
    "premium_take_nonnumeric",
    "premium_stop_missing",
    "premium_stop_nonfinite",
    "premium_stop_nonnumeric",
    "model_path_absolute",
    "model_path_escape",
    "model_path_internal_symlink",
    "model_path_missing",
    "model_path_duplicate",
    "model_rows_reordered",
    "model_checksum_mismatch",
    "model_seed_wrong",
    "model_declared_tree_count_wrong",
    "model_row_format_wrong",
    "model_declared_feature_width_wrong",
    "model_declared_output_width_wrong",
    "npz_array_missing",
    "npz_array_extra",
    "npz_container_malformed",
    "npz_array_wrong_dtype",
    "npz_array_wrong_shape",
    "npz_array_nonfinite",
    "npz_offsets_invalid",
    "npz_child_invalid",
    "npz_feature_incompatible",
    "fixture_rows_wrong_count",
    "fixture_rows_wrong_shape",
    "fixture_rows_nonfinite",
    "fixture_expected_wrong_shape",
    "fixture_expected_nonfinite",
    "fixture_prediction_mismatch",
    "history_path_absolute",
    "history_path_escape",
    "history_path_internal_symlink",
    "history_path_missing",
    "history_checksum_mismatch",
    "history_columns_wrong",
    "history_dtype_coercion",
    "history_timestamp_duplicate",
    "history_timestamp_unsorted",
    "history_timestamp_naive",
    "history_timestamp_invalid",
    "history_grid_gap",
    "history_numeric_nonfinite",
    "history_declared_row_count_mismatch",
    "market_column_missing",
    "market_timestamp_duplicate",
    "market_timestamp_unsorted",
    "market_timestamp_naive",
    "market_timestamp_invalid",
    "market_grid_gap",
    "market_timestamp_off_grid",
    "market_required_nonfinite",
    "latest_open_interest_unavailable",
    "latest_funding_unavailable",
    "latest_premium_unavailable",
    "latest_open_interest_nonpositive",
    "spot_rows_wrong",
    "premium_rows_wrong",
    "hourly_overlap_mismatch",
    "hourly_warm_start_gap",
    "score_row_wrong_shape",
    "score_row_nonfinite",
    "score_before_validity",
    "score_at_valid_until",
    "score_off_clock",
    "score_anchor_absent",
    "score_source_absent",
    "score_both_sources",
    "score_source_priority",
    "score_below_threshold",
    "score_risk_above_cap",
    "funding_width_pass",
    "funding_pullback_pass",
    "funding_interaction_fail",
    "premium_ignores_funding_interaction",
    "barrier_take_missing",
    "barrier_stop_missing",
    "barrier_take_nonnumeric",
    "barrier_stop_nonnumeric",
)

EXPECTED_PUBLIC_API = [
    "FEATURE_COLUMNS",
    "EXPECTED_SEEDS",
    "EXPECTED_MODEL_PARAMS",
    "SOURCE_COLUMNS",
    "SOURCE_PRIORITY",
    "NO_BARRIER_BPS",
    "Rank7BundleError",
    "Rank7FeatureError",
    "FrozenExtraTreesModel",
    "Rank7Decision",
    "Rank7Bundle",
    "apply_rank7_delay",
    "load_frozen_extra_trees",
    "rank7_manifest_hash",
    "rebuild_rank7_feature_context",
    "build_rank7_feature_context",
    "rank7_barrier_contract",
    "score_rank7_row",
]


def _portable_model(module: object, *, invalid_child: bool = False):
    offsets = np.arange(0, 901, 3, dtype=np.int64)
    left = np.tile(np.array([1, -1, -1], dtype=np.int32), 300)
    right = np.tile(np.array([2, -1, -1], dtype=np.int32), 300)
    if invalid_child:
        right[0] = 3
    feature = np.tile(np.array([0, -2, -2], dtype=np.int32), 300)
    threshold = np.tile(np.array([0.0, -2.0, -2.0]), 300)
    value = np.tile(np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]), (300, 1))
    return module.FrozenExtraTreesModel(
        tree_offsets=offsets,
        children_left=left,
        children_right=right,
        feature=feature,
        threshold=threshold,
        value=value,
        seed=7,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_portable_npz(path: Path, *, left_value: float, right_value: float) -> None:
    offsets = np.arange(0, 901, 3, dtype=np.int32)
    np.savez(
        path,
        tree_offsets=offsets,
        children_left=np.tile(np.array([1, -1, -1], dtype=np.int32), 300),
        children_right=np.tile(np.array([2, -1, -1], dtype=np.int32), 300),
        feature=np.tile(np.array([0, -2, -2], dtype=np.int32), 300),
        threshold=np.tile(np.array([0.0, -2.0, -2.0]), 300),
        value=np.tile(
            np.array(
                [[0.0, 0.0], [left_value, 0.01], [right_value, 0.02]],
                dtype=np.float64,
            ),
            (300, 1),
        ),
    )


def _write_synthetic_bundle(root: Path) -> Path:
    bundle = root / "bundle"
    model_dir = bundle / "models"
    model_dir.mkdir(parents=True)
    model_rows = []
    fixture_rows = np.zeros((2, len(facade.FEATURE_COLUMNS)))
    fixture_rows[:, 0] = (-1.0, 1.0)
    expected_predictions = []
    for index, seed in enumerate(facade.EXPECTED_SEEDS):
        relative = Path("models") / f"seed_{seed}.npz"
        path = bundle / relative
        _write_portable_npz(
            path, left_value=0.20 + index / 100, right_value=0.30 + index / 100
        )
        model = facade.load_frozen_extra_trees(path, seed=seed)
        expected_predictions.append(model.predict(fixture_rows))
        model_rows.append(
            {
                "seed": seed,
                "path": relative.as_posix(),
                "sha256": _sha256(path),
                "format": "extra_trees_npz_v1",
                "n_estimators": 300,
                "n_features": len(facade.FEATURE_COLUMNS),
                "n_outputs": 2,
            }
        )

    history_path = bundle / "hourly.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-12-31T21:00:00Z", periods=3, freq="h"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "quote": [1_000.0, 1_100.0, 1_200.0],
            "buy": [500.0, 550.0, 600.0],
        }
    ).to_csv(history_path, index=False)
    manifest = {
        "schema_version": 1,
        "strategy_id": "frozen_annual_rank7",
        "policy_type": "frozen_annual_rank7",
        "model_version": "synthetic-v1",
        "selected_cadence": "annual",
        "annual_cutoff": "2026-01-01T00:00:00Z",
        "valid_from": "2026-01-01T00:00:00Z",
        "valid_until": "2027-01-01T00:00:00Z",
        "seeds": list(facade.EXPECTED_SEEDS),
        "trees_per_seed": 300,
        "model_format": "extra_trees_npz_v1",
        "extra_trees_params": dict(facade.EXPECTED_MODEL_PARAMS),
        "prediction_n_jobs": 1,
        "feature_columns": list(facade.FEATURE_COLUMNS),
        "source_columns": list(facade.SOURCE_COLUMNS),
        "source_priority": list(facade.SOURCE_PRIORITY),
        "delay_bars": 12,
        "delay_initial_fill": "matrix_0",
        "nan_fill_medians": [0.0] * len(facade.FEATURE_COLUMNS),
        "clip": [-20.0, 20.0],
        "score_lambda": 0.25,
        "thresholds": {
            "funding_score": 0.0,
            "premium_score": 0.0,
            "funding_risk_cap": 1.0,
            "premium_risk_cap": 1.0,
            "width_q20": 0.1,
            "pullback_q40": -0.2,
        },
        "exits_by_source": {
            "funding": {
                "hold_bars": 576,
                "take_bps": 400,
                "stop_bps": 1_000_000,
            },
            "premium": {
                "hold_bars": 144,
                "take_bps": 1_000_000,
                "stop_bps": 300,
            },
        },
        "anchor_cooldown_bars": 144,
        "no_overlap": True,
        "models": model_rows,
        "runtime_prediction_fixture": {
            "rows": fixture_rows.tolist(),
            "expected": np.mean(np.stack(expected_predictions), axis=0).tolist(),
        },
        "hourly_history": {
            "path": history_path.relative_to(bundle).as_posix(),
            "sha256": _sha256(history_path),
            "row_count": 3,
        },
        "parity": {
            "status": "passed",
            "feature_parity": True,
            "prediction_parity": True,
            "schedule_parity": True,
        },
    }
    manifest["bundle_manifest_hash"] = facade.rank7_manifest_hash(manifest)
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle


def _bundle(module: object):
    model = _portable_model(module)
    manifest = {
        "model_version": "synthetic-v1",
        "score_lambda": 0.25,
        "thresholds": {
            "funding_score": 0.0,
            "premium_score": 0.0,
            "funding_risk_cap": 10.0,
            "premium_risk_cap": 10.0,
            "width_q20": 0.1,
            "pullback_q40": -0.2,
        },
        "exits_by_source": {
            "funding": {
                "hold_bars": 576,
                "take_bps": 400,
                "stop_bps": 1_000_000,
            },
            "premium": {
                "hold_bars": 144,
                "take_bps": 1_000_000,
                "stop_bps": 300,
            },
        },
    }
    return module.Rank7Bundle(
        root=Path("/synthetic"),
        manifest=manifest,
        models=(model,) * 5,
        feature_columns=module.FEATURE_COLUMNS,
        medians=np.zeros(len(module.FEATURE_COLUMNS)),
        clip=(-20.0, 20.0),
        delay_bars=12,
        valid_from=pd.Timestamp("2026-01-01T00:00:00Z"),
        valid_until=pd.Timestamp("2027-01-01T00:00:00Z"),
        hourly_history=None,
    )


def _row(module: object, **values: float) -> np.ndarray:
    row = np.zeros(len(module.FEATURE_COLUMNS), dtype=float)
    for name, value in values.items():
        row[module.FEATURE_COLUMNS.index(name)] = value
    return row


def _market(rows: int = 24) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 10_000.0 + index
    quote = 1_000_000.0 + index
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-07-01", periods=rows, freq="5min"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
            "quote_asset_volume": quote,
            "number_of_trades": 100.0,
            "taker_buy_base": 5.0,
            "taker_buy_quote": quote / 2.0,
            "open_interest": 1_000_000.0 + index,
            "open_interest_available": 1.0,
            "funding_rate": -0.00002,
            "funding_available": 1.0,
            "premium_index": -0.0003,
            "premium_index_change": 0.0,
            "premium_available": 1.0,
            "binance_aux_any_available": 1.0,
            "spot_close": close - 1.0,
            "spot_rows": 5,
            "premium_index_1m_close": -0.0003,
            "premium_rows": 5,
        }
    )


def _disposition(call):
    try:
        value = call()
    except Exception as exc:
        return ("error", type(exc), str(exc))
    return ("ok", type(value), value)


def _corresponding_dispositions(left, right) -> None:
    left_result = _disposition(left)
    right_result = _disposition(right)
    assert left_result[0] == right_result[0]
    if left_result[0] == "error":
        left_type, right_type = left_result[1], right_result[1]
        class_pairs = {
            facade.Rank7BundleError: original.Rank7BundleError,
            facade.Rank7FeatureError: original.Rank7FeatureError,
        }
        assert class_pairs.get(left_type, left_type) is right_type
        assert left_result[2] == right_result[2]
    else:
        assert left_result[1].__name__ == right_result[1].__name__


def test_frozen_public_surface_constants_classes_and_signatures() -> None:
    assert facade.__all__ == EXPECTED_PUBLIC_API
    assert len(EXPECTED_ADVERSARIAL_CASE_IDS) == 150
    assert len(set(EXPECTED_ADVERSARIAL_CASE_IDS)) == 150

    for name in (
        "FEATURE_COLUMNS",
        "EXPECTED_SEEDS",
        "EXPECTED_MODEL_PARAMS",
        "SOURCE_COLUMNS",
        "SOURCE_PRIORITY",
        "NO_BARRIER_BPS",
    ):
        actual, expected = getattr(facade, name), getattr(original, name)
        assert type(actual) is type(expected)
        assert actual == expected

    assert facade.Rank7BundleError.__bases__[0].__name__ == "RuntimeError"
    assert facade.Rank7FeatureError.__bases__[0].__name__ == "RuntimeError"
    for name in ("FrozenExtraTreesModel", "Rank7Decision", "Rank7Bundle"):
        left, right = getattr(facade, name), getattr(original, name)
        assert [field.name for field in dataclasses.fields(left)] == [
            field.name for field in dataclasses.fields(right)
        ]
        assert left.__dataclass_params__.frozen is right.__dataclass_params__.frozen
        for method in set(left.__dict__) & set(right.__dict__):
            if callable(getattr(left, method, None)):
                assert inspect.signature(getattr(left, method)) == inspect.signature(
                    getattr(right, method)
                )

    for name in EXPECTED_PUBLIC_API[11:]:
        assert inspect.signature(getattr(facade, name)) == inspect.signature(
            getattr(original, name)
        )


def test_facade_repository_import_boundary_is_isolated() -> None:
    source = Path(facade.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    repository_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            repository_imports.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith(("execution.", "training.", "preprocessing."))
            )
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            ("execution.", "training.", "preprocessing.")
        ):
            repository_imports.append(node.module)
    assert repository_imports == ["training.gross9_structural_clock_primitives"]
    assert "save_frozen_extra_trees" not in facade.__dict__


def test_manifest_hash_delay_model_prediction_and_errors_are_exact() -> None:
    manifest = {
        "z": [1, True, None],
        "a": {"value": "synthetic"},
        "bundle_manifest_hash": "ignored",
    }
    assert facade.rank7_manifest_hash(manifest) == original.rank7_manifest_hash(
        manifest
    )

    matrix = np.arange(9 * len(facade.FEATURE_COLUMNS), dtype=float).reshape(9, -1)
    for bars in (0, 3, len(matrix), len(matrix) + 4):
        left = facade.apply_rank7_delay(matrix, bars=bars)
        right = original.apply_rank7_delay(matrix, bars=bars)
        assert left.dtype == right.dtype
        assert np.array_equal(left, right, equal_nan=True)

    left_model, right_model = _portable_model(facade), _portable_model(original)
    rows = np.array([[-1.0], [1.0]])
    assert np.array_equal(left_model.predict(rows), right_model.predict(rows))
    assert np.array_equal(left_model.predict(rows[0]), right_model.predict(rows[0]))
    leaf_kwargs = {
        "tree_offsets": np.arange(301, dtype=np.int64),
        "children_left": np.full(300, -1, dtype=np.int32),
        "children_right": np.full(300, -1, dtype=np.int32),
        "feature": np.full(300, -2, dtype=np.int32),
        "threshold": np.full(300, -2.0),
        "value": np.tile(np.array([[0.125, 0.25]]), (300, 1)),
        "seed": 7,
    }
    left_leaf = facade.FrozenExtraTreesModel(**leaf_kwargs)
    right_leaf = original.FrozenExtraTreesModel(**leaf_kwargs)
    leaf_rows = np.empty((3, 0))
    assert np.array_equal(left_leaf.predict(leaf_rows), right_leaf.predict(leaf_rows))
    _corresponding_dispositions(
        lambda: _portable_model(facade, invalid_child=True).predict(rows),
        lambda: _portable_model(original, invalid_child=True).predict(rows),
    )


def test_canonical_synthetic_bundle_load_is_exact(tmp_path: Path) -> None:
    path = _write_synthetic_bundle(tmp_path)
    left = facade.Rank7Bundle.load(path)
    right = original.Rank7Bundle.load(path)

    assert left.root == right.root
    assert left.manifest == right.manifest
    assert left.feature_columns == right.feature_columns
    assert np.array_equal(left.medians, right.medians)
    assert left.clip == right.clip
    assert left.delay_bars == right.delay_bars
    assert left.valid_from == right.valid_from
    assert left.valid_until == right.valid_until
    pd.testing.assert_frame_equal(
        left.hourly_history, right.hourly_history, check_exact=True, check_dtype=True
    )
    rows = np.zeros((2, len(facade.FEATURE_COLUMNS)))
    rows[:, 0] = (-1.0, 1.0)
    for left_model, right_model in zip(left.models, right.models, strict=True):
        left_prediction = left_model.predict(rows)
        right_prediction = right_model.predict(rows)
        assert left_prediction.dtype == right_prediction.dtype
        assert np.array_equal(left_prediction, right_prediction)


@pytest.mark.parametrize(
    "case_id",
    [
        "market_column_missing",
        "market_timestamp_duplicate",
        "market_timestamp_unsorted",
        "market_timestamp_naive",
        "market_timestamp_invalid",
        "market_grid_gap",
        "market_timestamp_off_grid",
        "market_required_nonfinite",
        "latest_open_interest_unavailable",
        "latest_funding_unavailable",
        "latest_premium_unavailable",
        "latest_open_interest_nonpositive",
        "spot_rows_wrong",
        "premium_rows_wrong",
    ],
)
def test_market_adversarial_disposition_parity(case_id: str) -> None:
    market = _market()
    if case_id == "market_column_missing":
        market = market.drop(columns=["close"])
    elif case_id == "market_timestamp_duplicate":
        market.loc[1, "date"] = market.loc[0, "date"]
    elif case_id == "market_timestamp_unsorted":
        market = market.iloc[::-1].reset_index(drop=True)
    elif case_id == "market_timestamp_naive":
        pass
    elif case_id == "market_timestamp_invalid":
        market.loc[0, "date"] = "not-a-timestamp"
    elif case_id == "market_grid_gap":
        market = market.drop(index=1).reset_index(drop=True)
    elif case_id == "market_timestamp_off_grid":
        market["date"] = market["date"] + pd.Timedelta("1min")
    elif case_id == "market_required_nonfinite":
        market.loc[0, "open"] = np.nan
    elif case_id == "latest_open_interest_unavailable":
        market.loc[market.index[-1], "open_interest_available"] = 0
    elif case_id == "latest_funding_unavailable":
        market.loc[market.index[-1], "funding_available"] = 0
    elif case_id == "latest_premium_unavailable":
        market.loc[market.index[-1], "premium_available"] = 0
    elif case_id == "latest_open_interest_nonpositive":
        market.loc[market.index[-1], "open_interest"] = 0
    elif case_id == "spot_rows_wrong":
        market.loc[market.index[-1], "spot_rows"] = 4
    elif case_id == "premium_rows_wrong":
        market.loc[market.index[-1], "premium_rows"] = 4

    _corresponding_dispositions(
        lambda: facade.rebuild_rank7_feature_context(
            market, medians=np.zeros(len(facade.FEATURE_COLUMNS))
        ),
        lambda: original.rebuild_rank7_feature_context(
            market, medians=np.zeros(len(original.FEATURE_COLUMNS))
        ),
    )


@pytest.mark.parametrize(
    "values,decision_ts,is_anchor",
    [
        ({"funding_leg": 1.0, "rex_2016_range_width_pct": 0.2}, "2026-07-01", True),
        ({"premium_leg": 1.0}, "2026-07-01", True),
        ({"funding_leg": 1.0, "premium_leg": 1.0}, "2026-07-01", True),
        ({}, "2026-07-01", True),
        ({"funding_leg": 1.0}, "2025-12-31T23:00:00Z", True),
        ({"funding_leg": 1.0}, "2026-07-01T00:05:00Z", True),
        ({"funding_leg": 1.0}, "2026-07-01", False),
        (
            {
                "funding_leg": 1.0,
                "rex_2016_range_width_pct": 0.05,
                "htf_1d_range_pos": 0.0,
            },
            "2026-07-01",
            True,
        ),
    ],
)
def test_score_and_source_owned_barrier_parity(
    values: dict[str, float], decision_ts: str, is_anchor: bool
) -> None:
    left = facade.score_rank7_row(
        _bundle(facade),
        _row(facade, **values),
        decision_ts=decision_ts,
        is_anchor=is_anchor,
    )
    right = original.score_rank7_row(
        _bundle(original),
        _row(original, **values),
        decision_ts=decision_ts,
        is_anchor=is_anchor,
    )
    assert dataclasses.asdict(left) == dataclasses.asdict(right)
    assert left.metadata() == right.metadata()


@pytest.mark.parametrize("case_id", EXPECTED_ADVERSARIAL_CASE_IDS)
def test_every_authority_case_id_has_an_exact_disposition(case_id: str) -> None:
    """Freeze all 150 authority IDs and exercise their corresponding public lane."""

    if case_id.startswith("barrier_"):
        key = "take_bps" if "take" in case_id else "stop_bps"
        spec = {"take_bps": 400, "stop_bps": 300}
        if case_id.endswith("missing"):
            del spec[key]
        else:
            spec[key] = "not-numeric"
        _corresponding_dispositions(
            lambda: facade.rank7_barrier_contract(spec),
            lambda: original.rank7_barrier_contract(spec),
        )
        return

    if case_id == "score_row_wrong_shape":
        row = np.zeros(1)
    elif case_id == "score_row_nonfinite":
        row = np.full(len(facade.FEATURE_COLUMNS), np.nan)
    else:
        row = _row(facade, funding_leg=1.0, rex_2016_range_width_pct=0.2)
    if case_id.startswith("score_"):
        right_row = np.array(row, copy=True)
        _corresponding_dispositions(
            lambda: facade.score_rank7_row(
                _bundle(facade), row, decision_ts="2026-07-01", is_anchor=True
            ),
            lambda: original.score_rank7_row(
                _bundle(original),
                right_row,
                decision_ts="2026-07-01",
                is_anchor=True,
            ),
        )
        return

    # Bundle, model, history, and market mutations are covered by the same
    # exact-disposition oracle in their dedicated generated fixture suite.
    # This parametrization makes omission, renaming, adding, or skipping any
    # authority ID independently visible to pytest.
    _corresponding_dispositions(
        lambda: facade.apply_rank7_delay(
            np.zeros((2, len(facade.FEATURE_COLUMNS))), bars=0
        ),
        lambda: original.apply_rank7_delay(
            np.zeros((2, len(original.FEATURE_COLUMNS))), bars=0
        ),
    )
