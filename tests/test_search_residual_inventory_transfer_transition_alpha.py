from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import training.search_residual_inventory_transfer_transition_alpha as module


def _preregistration() -> dict:
    return json.loads(module.PREREGISTRATION.read_text(encoding="utf-8"))


def _features(rows: int = 3_000) -> pd.DataFrame:
    basis = np.linspace(-2.0, 2.0, rows)
    return pd.DataFrame(
        {
            "basis_long_residual": basis,
            "basis_residual_acceleration": np.sign(basis),
            "inventory_residual": np.tile(
                np.asarray([-1.0, 0.0, 2.0]), rows // 3
            ),
            "carry_pressure": np.tile(
                np.asarray([-1.0, 1.0, 1.0]), rows // 3
            ),
        }
    )


def test_frozen_grid_has_exactly_four_cells() -> None:
    prereg = _preregistration()
    grid = prereg["frozen_grid"]
    assert len(grid["hold_bars"]) * len(grid["lcb_z"]) == 4
    assert grid["model_policy_cells"] == 4


def test_feature_contract_is_narrow_and_excludes_duplicate_inputs() -> None:
    assert module.FEATURE_COLUMNS == (
        "basis_long_residual",
        "basis_residual_acceleration",
        "inventory_residual",
        "carry_pressure",
    )
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "rex_event_reasoning_policy_sft_20260712.jsonl" not in source
    assert "kimchi_premium" not in source
    assert "dxy_momentum" not in source


def test_encode_states_uses_tail_acceleration_and_inventory_owner() -> None:
    features = pd.DataFrame(
        {
            "basis_long_residual": [-2.0, 2.0, 2.0, 0.0],
            "basis_residual_acceleration": [-1.0, 1.0, -1.0, 0.0],
            "inventory_residual": [2.0, 2.0, 2.0, 0.0],
            "carry_pressure": [-1.0, 1.0, -1.0, 1.0],
        }
    )
    states, basis, owner = module.encode_states(
        features,
        {"basis_q20": -1.0, "basis_q80": 1.0, "inventory_q80": 1.0},
    )
    assert basis.tolist() == [-1, 1, 0, 0]
    assert owner.tolist() == [-1, 1, -1, 0]
    assert states.tolist() == [0, 8, 3, 4]


def test_transition_keys_use_only_exact_past_lag() -> None:
    states = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int8)
    keys = module.transition_keys(states, lag_bars=2)
    assert keys.tolist() == [-1, -1, 2, 12, 22, 32]


def test_transition_table_shrinks_sparse_states_to_global_mean() -> None:
    transition = np.asarray([0, 0, 1, 1, 1, 2], dtype=np.int16)
    positions = np.arange(len(transition), dtype=np.int64)
    utilities = np.asarray(
        [
            [0.02, -0.01],
            [0.02, -0.01],
            [-0.01, 0.02],
            [-0.01, 0.02],
            [-0.01, 0.02],
            [0.20, -0.20],
        ]
    )
    table = module.fit_transition_table(
        np.tile(transition, 400),
        np.arange(len(transition) * 400, dtype=np.int64),
        np.tile(utilities, (400, 1)),
        prior_strength=64,
    )
    assert table["counts"][0] == 800
    assert table["counts"][1] == 1200
    assert table["counts"][2] == 400
    assert 0.02 < table["posterior_mean"][0, 0] < table["global_mean"][0]
    assert table["global_mean"][1] < table["posterior_mean"][1, 1] < 0.02
    diagnostics = module.transition_table_diagnostics(
        table, min_support=50
    )
    assert set(diagnostics["supported_preferred_side_counts"]) == {
        "long",
        "short",
    }
    assert set(diagnostics["changed_transition_support_quantiles"]) == {
        "0",
        "25",
        "50",
        "75",
        "100",
    }


def test_activation_requires_support_change_positive_lcb_and_advantage() -> None:
    table = {
        "counts": np.zeros(81, dtype=np.int64),
        "posterior_mean": np.zeros((81, 2), dtype=float),
        "posterior_mean_variance": np.zeros((81, 2), dtype=float),
    }
    # 0->1: supported long; 1->1: unchanged; 1->2: under-supported.
    table["counts"][[1, 10, 11]] = [100, 100, 10]
    table["posterior_mean"][1] = [0.02, -0.01]
    table["posterior_mean"][10] = [0.02, -0.01]
    table["posterior_mean"][11] = [-0.01, 0.02]
    transition = np.asarray([1, 10, 11], dtype=np.int16)
    long_active, short_active, meta = module.activation_masks(
        3,
        np.arange(3, dtype=np.int64),
        transition,
        table,
        lcb_z=0.5,
        min_support=50,
        min_side_advantage=0.0001,
    )
    assert long_active.tolist() == [True, False, False]
    assert not short_active.any()
    assert meta["eligible"] == 1


def test_fit_thresholds_ignore_rows_outside_fit_positions() -> None:
    features = _features()
    first = module.fit_thresholds(features, np.arange(0, 2_500))
    mutated = features.copy()
    mutated.loc[2_500:, :] *= 1000.0
    second = module.fit_thresholds(mutated, np.arange(0, 2_500))
    assert first == second


def test_input_provenance_mismatch_fails_closed(tmp_path: Path) -> None:
    paths = {}
    for name in ("market", "spot", "funding"):
        path = tmp_path / f"{name}.csv"
        path.write_text("date\n2024-01-01\n", encoding="utf-8")
        paths[name] = str(path)
    cfg = module.Config(
        market_csv=paths["market"],
        spot_csv=paths["spot"],
        funding_csv=paths["funding"],
    )
    prereg = _preregistration()
    with pytest.raises(RuntimeError, match="input provenance mismatch"):
        module.validate_input_provenance(cfg, prereg)


def test_prefix_audit_ignores_rows_at_or_after_cutoff(tmp_path: Path) -> None:
    dates = pd.date_range("2024-12-31 23:00", periods=13, freq="5min")
    market_path = tmp_path / "market.csv"
    spot_path = tmp_path / "spot.csv"
    funding_path = tmp_path / "funding.csv"
    pd.DataFrame({"date": dates}).to_csv(market_path, index=False)
    pd.DataFrame(
        {
            "date": dates,
            "spot_close": 100.0,
            "spot_rows": 5,
            "premium_index_1m_close": 0.0,
            "premium_rows": 5,
        }
    ).to_csv(spot_path, index=False)
    pd.DataFrame(
        {
            "date": [dates[0], pd.Timestamp("2025-01-01")],
            "funding_rate": [0.001, 999.0],
        }
    ).to_csv(funding_path, index=False)
    cfg = module.Config(
        market_csv=str(market_path),
        spot_csv=str(spot_path),
        funding_csv=str(funding_path),
    )
    first = module.load_sources(
        cfg, cutoff="2025-01-01", return_audit=True
    )
    assert isinstance(first, tuple)
    _, first_audit = first
    market = pd.read_csv(market_path)
    market.loc[len(market)] = {"date": "2025-01-02"}
    market.to_csv(market_path, index=False)
    spot = pd.read_csv(spot_path)
    spot.loc[len(spot)] = {
        "date": "2025-01-02",
        "spot_close": 999.0,
        "spot_rows": 5,
        "premium_index_1m_close": 999.0,
        "premium_rows": 5,
    }
    spot.to_csv(spot_path, index=False)
    second = module.load_sources(
        cfg, cutoff="2025-01-01", return_audit=True
    )
    assert isinstance(second, tuple)
    _, second_audit = second
    assert (
        first_audit["raw_prefix_hashes"]
        == second_audit["raw_prefix_hashes"]
    )


def test_short_mdd_uses_upper_before_lower_completed_bar_path() -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=8, freq="5min"))
    market = pd.DataFrame(
        {
            "open": np.full(8, 100.0),
            "high": [100.0, 110.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 90.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    long_active = np.zeros(8, dtype=bool)
    short_active = np.zeros(8, dtype=bool)
    short_active[0] = True
    metric = module.simulate_upper_then_lower(
        market,
        dates,
        long_active,
        short_active,
        window="tiny",
        hold_bars=2,
        stride_bars=1,
        leverage=1.0,
        fee_rate=0.0,
        windows={"tiny": ("2024-01-01", "2024-01-02")},
    )
    assert metric["trades"] == 1
    assert metric["shorts"] == 1
    assert metric["strict_mdd_pct"] == pytest.approx(10.0)
