from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from training.search_minimal_stress_pair_context_formula_expert_alpha import (
    CORE_FEATURES,
    EXPERT_SPEC,
    FROZEN_CHAMPION,
    Config,
    _frozen_execution_config,
    _implementation_hash,
    _mark_oos_opened,
    formula_feature_frame,
    pair_context_design,
)


def test_pair_context_grid_discloses_all_640_cells() -> None:
    assert EXPERT_SPEC["grid_cells"] == 640


def test_frozen_champion_uses_two_independent_slow_contexts() -> None:
    assert FROZEN_CHAMPION["contexts"] == [
        "usdkrw_momentum",
        "dollar_flow_rel_4h_30d",
    ]
    assert FROZEN_CHAMPION["tail"] == 0.20
    assert FROZEN_CHAMPION["ridge"] == 100.0


def test_frozen_exposure_is_below_discovery_half_notional() -> None:
    assert Config.leverage == 0.45
    assert EXPERT_SPEC["leverage"] == 0.45


def test_frozen_source_paths_are_location_independent() -> None:
    relative = Config()
    absolute = Config(
        input_csv="/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
        funding_csv="/home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz",
        premium_csv="/home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz",
    )

    assert _frozen_execution_config(relative) == _frozen_execution_config(absolute)


def test_formula_features_are_delayed_one_complete_bar() -> None:
    source = inspect.getsource(formula_feature_frame)
    assert ".shift(1)" in source


def test_pair_context_design_has_103_structured_dimensions() -> None:
    rows = 12
    core = np.arange(rows * len(CORE_FEATURES), dtype=float).reshape(
        rows, len(CORE_FEATURES)
    )
    source = np.where(np.arange(rows) % 2, 1.0, -1.0)
    contexts = {
        "first": np.linspace(-1.0, 1.0, rows),
        "second": np.linspace(2.0, -2.0, rows),
    }
    fit = np.arange(rows) < 8

    design, metadata = pair_context_design(
        core, source, contexts, fit, tail=0.20
    )

    assert design.shape == (rows, 103)
    assert metadata["dimensions"] == 103
    assert set(metadata["contexts"]) == {"first", "second"}


def test_design_scaling_and_thresholds_ignore_nonfit_suffix() -> None:
    rng = np.random.default_rng(42)
    prefix_rows = 10
    suffix_rows = 4
    core_prefix = rng.normal(size=(prefix_rows, len(CORE_FEATURES)))
    source_prefix = np.where(np.arange(prefix_rows) % 2, 1.0, -1.0)
    context_prefix = {
        "first": rng.normal(size=prefix_rows),
        "second": rng.normal(size=prefix_rows),
    }
    prefix_fit = np.ones(prefix_rows, dtype=bool)
    prefix_design, prefix_meta = pair_context_design(
        core_prefix,
        source_prefix,
        context_prefix,
        prefix_fit,
        tail=0.20,
    )

    core_full = np.vstack(
        [core_prefix, np.full((suffix_rows, len(CORE_FEATURES)), 1e9)]
    )
    source_full = np.concatenate([source_prefix, np.ones(suffix_rows)])
    context_full = {
        "first": np.concatenate(
            [context_prefix["first"], np.full(suffix_rows, 1e9)]
        ),
        "second": np.concatenate(
            [context_prefix["second"], np.full(suffix_rows, -1e9)]
        ),
    }
    full_fit = np.concatenate(
        [np.ones(prefix_rows, dtype=bool), np.zeros(suffix_rows, dtype=bool)]
    )
    full_design, full_meta = pair_context_design(
        core_full,
        source_full,
        context_full,
        full_fit,
        tail=0.20,
    )

    assert np.allclose(full_design[:prefix_rows], prefix_design)
    assert full_meta == prefix_meta


def test_pair_context_design_rejects_unbounded_context_count() -> None:
    core = np.zeros((4, len(CORE_FEATURES)))
    source = np.ones(4)
    fit = np.ones(4, dtype=bool)

    with pytest.raises(ValueError, match="exactly two"):
        pair_context_design(
            core,
            source,
            {"only": np.zeros(4)},
            fit,
            tail=0.20,
        )


def test_implementation_hash_covers_selection_oos_and_execution_guards() -> None:
    source = inspect.getsource(_implementation_hash)
    for function_name in (
        "base._context",
        "base.fit_rule_masks",
        "base.fit_event_mask",
        "base.trade_utility",
        "base.fit_action_utilities",
        "base._execution_config",
        "base._stable_action_mask",
        "pair_context_design",
        "_grid",
        "_freeze_payload",
        "_freeze_hash",
        "_validate_manifest",
        "_write_manifest_once",
        "_selection_payload",
        "_mark_oos_opened",
        "_oos",
        "ExecutionEngine.trade_at",
        "_frame_hash",
        "_activation_hash",
        "equity_stats",
    ):
        assert function_name in source
    for constant_name in (
        "EXPERT_SPEC",
        "FROZEN_CHAMPION",
        "FROZEN_CONFIG_KEYS",
        "PRE2024_WINDOWS",
        "FUTURE_WINDOWS",
        "SELECTION_END",
    ):
        assert constant_name in source


def test_oos_open_lock_is_idempotent_and_rejects_other_freeze(tmp_path) -> None:
    path = tmp_path / "manifest.json.oos-opening.json"
    manifest = {
        "freeze_hash": "freeze-a",
        "spec_hash": "spec-a",
        "implementation_hash": "implementation-a",
    }

    first = _mark_oos_opened(path, manifest, "result.json")
    second = _mark_oos_opened(path, manifest, "result.json")

    assert first == second
    assert json.loads(path.read_text()) == first
    assert first["phase"] == "oos_opening"
    with pytest.raises(RuntimeError, match="different freeze"):
        _mark_oos_opened(
            path,
            {**manifest, "freeze_hash": "freeze-b"},
            "result.json",
        )
