from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from execution.rank7_runtime import (
    FEATURE_COLUMNS,
    Rank7Bundle,
    Rank7BundleError,
    Rank7FeatureError,
    apply_rank7_delay,
    load_frozen_extra_trees,
    rank7_manifest_hash,
    rebuild_rank7_feature_context,
    save_frozen_extra_trees,
    score_rank7_row,
)


SEEDS = (7, 71, 715, 2026, 71515)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(root: Path) -> Path:
    bundle = root / "rank7"
    models = bundle / "models"
    models.mkdir(parents=True)
    x = np.zeros((96, len(FEATURE_COLUMNS)), dtype=float)
    x[:, 0] = np.linspace(-1.0, 1.0, len(x))
    y = np.column_stack(
        [np.full(len(x), 0.02, dtype=float), np.full(len(x), 0.01, dtype=float)]
    )
    model_rows = []
    fitted_models = []
    for seed in SEEDS:
        model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=2,
            min_samples_leaf=32,
            max_features=0.8,
            bootstrap=False,
            random_state=seed,
            n_jobs=-1,
        ).fit(x, y)
        fitted_models.append(model)
        path = models / f"seed_{seed}.npz"
        save_frozen_extra_trees(model, path)
        model_rows.append(
            {
                "seed": seed,
                "path": str(path.relative_to(bundle)),
                "sha256": _sha256(path),
                "format": "extra_trees_npz_v1",
                "n_estimators": 300,
                "n_features": len(FEATURE_COLUMNS),
                "n_outputs": 2,
            }
        )

    manifest = {
        "schema_version": 1,
        "strategy_id": "frozen_annual_rank7",
        "policy_type": "frozen_annual_rank7",
        "model_version": "test-2026",
        "selected_cadence": "annual",
        "annual_cutoff": "2026-01-01T00:00:00Z",
        "valid_from": "2026-01-01T00:00:00Z",
        "valid_until": "2027-01-01T00:00:00Z",
        "seeds": list(SEEDS),
        "trees_per_seed": 300,
        "model_format": "extra_trees_npz_v1",
        "extra_trees_params": {
            "max_depth": 2,
            "min_samples_leaf": 32,
            "max_features": 0.8,
            "bootstrap": False,
        },
        "prediction_n_jobs": 1,
        "feature_columns": list(FEATURE_COLUMNS),
        "source_columns": ["funding_leg", "premium_leg"],
        "source_priority": ["funding", "premium"],
        "delay_bars": 12,
        "delay_initial_fill": "matrix_0",
        "nan_fill_medians": [0.0] * len(FEATURE_COLUMNS),
        "clip": [-20.0, 20.0],
        "score_lambda": 0.25,
        "thresholds": {
            "funding_score": 0.0,
            "premium_score": 0.0,
            "funding_risk_cap": 0.02,
            "premium_risk_cap": 0.02,
            "width_q20": 0.1,
            "pullback_q40": -0.2,
        },
        "exits_by_source": {
            "funding": {"hold_bars": 576, "take_bps": 400, "stop_bps": 1_000_000},
            "premium": {"hold_bars": 144, "take_bps": 1_000_000, "stop_bps": 300},
        },
        "anchor_cooldown_bars": 144,
        "no_overlap": True,
        "models": model_rows,
        "runtime_prediction_fixture": {
            "rows": [x[0].tolist(), x[-1].tolist()],
            "expected": np.mean(
                np.stack([model.predict(x[[0, -1]]) for model in fitted_models]), axis=0
            ).tolist(),
        },
        "parity": {
            "status": "passed",
            "feature_parity": True,
            "prediction_parity": True,
            "schedule_parity": True,
        },
    }
    manifest["bundle_manifest_hash"] = rank7_manifest_hash(manifest)
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return bundle


def _row(*, funding: float = 0.0, premium: float = 0.0, width: float = 0.2, pullback: float = 0.0) -> np.ndarray:
    row = np.zeros(len(FEATURE_COLUMNS), dtype=float)
    row[FEATURE_COLUMNS.index("funding_leg")] = funding
    row[FEATURE_COLUMNS.index("premium_leg")] = premium
    row[FEATURE_COLUMNS.index("rex_2016_range_width_pct")] = width
    row[FEATURE_COLUMNS.index("htf_1d_range_pos")] = pullback
    return row


def test_rank7_bundle_loads_exact_contract_and_forces_serial_prediction(tmp_path: Path) -> None:
    bundle = Rank7Bundle.load(_write_bundle(tmp_path))

    assert bundle.feature_columns == FEATURE_COLUMNS
    assert bundle.delay_bars == 12
    assert len(bundle.models) == 5
    assert all(model.n_jobs == 1 for model in bundle.models)
    assert bundle.valid_from == pd.Timestamp("2026-01-01T00:00:00Z")
    assert bundle.valid_until == pd.Timestamp("2027-01-01T00:00:00Z")


def test_rank7_bundle_rejects_tampered_model(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    model_path = path / "models" / "seed_7.npz"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    with pytest.raises(Rank7BundleError, match="checksum"):
        Rank7Bundle.load(path)


def test_rank7_bundle_rejects_tampered_threshold_contract(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["thresholds"]["funding_score"] = -999.0
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(Rank7BundleError, match="manifest hash"):
        Rank7Bundle.load(path)


def test_portable_extra_trees_prediction_is_bit_exact(tmp_path: Path) -> None:
    x = np.arange(96 * len(FEATURE_COLUMNS), dtype=float).reshape(96, -1) / 1000.0
    y = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 1])])
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=2,
        min_samples_leaf=32,
        max_features=0.8,
        bootstrap=False,
        random_state=7,
        n_jobs=1,
    ).fit(x, y)
    path = tmp_path / "model.npz"

    save_frozen_extra_trees(model, path)
    frozen = load_frozen_extra_trees(path, seed=7)

    np.testing.assert_array_equal(frozen.predict(x[:8]), model.predict(x[:8]))


def test_rank7_delay_matches_research_and_restores_current_source_columns() -> None:
    matrix = np.arange(16 * len(FEATURE_COLUMNS), dtype=float).reshape(16, len(FEATURE_COLUMNS))
    funding = FEATURE_COLUMNS.index("funding_leg")
    premium = FEATURE_COLUMNS.index("premium_leg")
    matrix[:, funding] = np.arange(16) % 2
    matrix[:, premium] = 1 - matrix[:, funding]

    delayed = apply_rank7_delay(matrix, bars=12)

    np.testing.assert_array_equal(delayed[:12, 0], np.repeat(matrix[0, 0], 12))
    assert delayed[15, 0] == matrix[3, 0]
    np.testing.assert_array_equal(delayed[:, funding], matrix[:, funding])
    np.testing.assert_array_equal(delayed[:, premium], matrix[:, premium])


def test_rank7_scoring_uses_funding_priority_and_source_owned_exit(tmp_path: Path) -> None:
    bundle = Rank7Bundle.load(_write_bundle(tmp_path))

    decision = score_rank7_row(
        bundle,
        _row(funding=1.0, premium=1.0, width=0.2),
        decision_ts=pd.Timestamp("2026-07-18T12:00:00Z"),
        is_anchor=True,
    )

    assert decision.active is True
    assert decision.source == "funding"
    assert decision.hold_bars == 576
    assert decision.barrier_exit["take_bps"] == 400.0
    assert decision.barrier_exit["stop_bps"] is None
    assert decision.signal_id.endswith(":funding:2026-07-18T12:00:00+00:00")


def test_rank7_scoring_fails_closed_outside_clock_anchor_and_validity(tmp_path: Path) -> None:
    bundle = Rank7Bundle.load(_write_bundle(tmp_path))
    row = _row(funding=1.0, width=0.05, pullback=0.0)

    bad_interaction = score_rank7_row(
        bundle, row, decision_ts=pd.Timestamp("2026-07-18T12:00:00Z"), is_anchor=True
    )
    bad_clock = score_rank7_row(
        bundle, _row(funding=1.0), decision_ts=pd.Timestamp("2026-07-18T12:05:00Z"), is_anchor=True
    )
    bad_anchor = score_rank7_row(
        bundle, _row(funding=1.0), decision_ts=pd.Timestamp("2026-07-18T12:00:00Z"), is_anchor=False
    )
    expired = score_rank7_row(
        bundle, _row(funding=1.0), decision_ts=pd.Timestamp("2027-01-01T00:00:00Z"), is_anchor=True
    )

    assert bad_interaction.active is False
    assert "funding_interaction=fail" in bad_interaction.reasons
    assert bad_clock.active is False
    assert "decision_clock=fail" in bad_clock.reasons
    assert bad_anchor.active is False
    assert "immutable_anchor=fail" in bad_anchor.reasons
    assert expired.active is False
    assert "bundle_validity=fail" in expired.reasons


def _market(rows: int = 40 * 24 * 12 + 1) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    dates = pd.date_range("2020-07-01", periods=rows, freq="5min")
    close = 10_000.0 * np.exp(0.0001 * index + 0.001 * np.sin(index / 50.0))
    quote = 1_000_000.0 * (1.0 + 0.05 * np.cos(index / 37.0))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 10.0,
            "quote_asset_volume": quote,
            "number_of_trades": 100.0,
            "taker_buy_base": 5.0,
            "taker_buy_quote": quote * (0.5 + 0.03 * np.sin(index / 17.0)),
            "open_interest": 1_000_000.0 * np.exp(0.00001 * index),
            "open_interest_available": 1.0,
            "funding_rate": -0.00002,
            "funding_available": 1.0,
            "premium_index": -0.0003,
            "premium_index_change": 0.0,
            "premium_available": 1.0,
            "binance_aux_any_available": 1.0,
            "spot_close": close * 0.999,
            "spot_rows": 5,
            "premium_index_1m_close": -0.0003,
            "premium_rows": 5,
        }
    )


def test_rank7_feature_rebuild_emits_exact_ordered_finite_delayed_matrix() -> None:
    market = _market()
    context = rebuild_rank7_feature_context(
        market,
        medians=np.zeros(len(FEATURE_COLUMNS), dtype=float),
        clip=(-20.0, 20.0),
        delay_bars=12,
    )

    assert context["matrix"].shape == (len(market), len(FEATURE_COLUMNS))
    assert np.isfinite(context["matrix"]).all()
    assert tuple(context["feature_columns"]) == FEATURE_COLUMNS
    funding = FEATURE_COLUMNS.index("funding_leg")
    premium = FEATURE_COLUMNS.index("premium_leg")
    assert context["matrix"][-1, funding] == float(context["funding_leg"][-1])
    assert context["matrix"][-1, premium] == float(context["premium_leg"][-1])
    assert len(context["anchors"]) == len(market)


def test_rank7_feature_rebuild_rejects_incomplete_live_spot_or_premium() -> None:
    market = _market()
    market.loc[market.index[-1], "spot_rows"] = 4

    with pytest.raises(Rank7FeatureError, match="spot_rows"):
        rebuild_rank7_feature_context(
            market,
            medians=np.zeros(len(FEATURE_COLUMNS), dtype=float),
            clip=(-20.0, 20.0),
            delay_bars=12,
        )


def test_committed_rank7_2026_bundle_is_loadable_and_parity_gated() -> None:
    bundle = Rank7Bundle.load("artifacts/rank7/frozen_annual_rank7_2026")

    assert bundle.model_version == "rank7-annual-2026-v1"
    assert bundle.hourly_history is not None
    assert len(bundle.hourly_history) == 56_232
    assert bundle.manifest["parity"]["feature"]["matrix_sha256"] == (
        "33805051cac6fec8a70579c6bc45e6f593c58b514f5aea0b96faba9a6db60280"
    )
    assert bundle.manifest["parity"]["research_full_result_hash"] == (
        "0cc647284179f7728e83a7ed6c160f9600c3509f25e468224e6a5d2f2e029eef"
    )
