import json

import numpy as np
import pandas as pd
import pytest

from training.search_moment_embedding_probe_alpha import MODEL_ID, MODEL_REVISION
from training.search_moment_mamba2_path_critic_alpha import HOLD_BARS, MIN_SCORE_HISTORY, load_required_source_manifest
from training.search_moment_retrieval_path_critic_alpha import (
    K_VALUES,
    MEAN8_WINDOW,
    QUANTILES,
    READY_OFFSET_BARS,
    RetrievalSpec,
    build_query_representations,
    compact_retrieval_diagnostics,
    delayed_causal_retrieval_quantiles,
    empirical_neighbor_quantiles,
    fit_standardizer,
    initial_fit_memory_mask,
    ready_positions_for_targets,
    retrieval_path_quantiles,
    standardize_l2,
)


def test_current_mean8_is_causal_and_requires_all_contributors():
    pca = np.arange(12 * 32, dtype=np.float32).reshape(12, 32)
    valid = np.ones(12, dtype=bool)
    reps, out_valid, meta = build_query_representations(pca, valid, kind="current_mean8")
    assert meta["mean_window_anchors"] == MEAN8_WINDOW
    assert not out_valid[:7].any()
    assert out_valid[7:].all()
    np.testing.assert_allclose(reps[7, :32], pca[7])
    np.testing.assert_allclose(reps[7, 32:], pca[:8].mean(axis=0))

    mutated = pca.copy()
    mutated[9:] *= -1000.0
    reps_mut, valid_mut, _ = build_query_representations(mutated, valid, kind="current_mean8")
    np.testing.assert_allclose(reps_mut[:9], reps[:9], equal_nan=True)
    assert valid_mut[:9].tolist() == out_valid[:9].tolist()

    valid_gap = valid.copy()
    valid_gap[3] = False
    _, out_gap, _ = build_query_representations(pca, valid_gap, kind="current_mean8")
    assert not out_gap[7]
    assert out_gap[11]


def test_delayed_pool_excludes_unready_and_current_target():
    n = 6
    positions = np.array([0, 100, 200, 700, 800, 900], dtype=np.int64)
    x = np.eye(n, dtype=np.float32)
    valid = np.ones(n, dtype=bool)
    targets = np.array([[i, i + 10, i + 20] for i in range(n)], dtype=np.float32)
    fit_memory = np.array([True, False, False, False, False, False])

    preds, diag = delayed_causal_retrieval_quantiles(x, valid, targets, positions, fit_memory_mask=fit_memory, start_index=1, k=2)
    # i=1 has only the initial fit row. At i=3 row1 has matured; row3/current is excluded.
    assert np.isnan(preds[1]).all()
    assert diag["pool_size_by_anchor"][1] == 1
    assert diag["pool_size_by_anchor"][3] == 2
    # At i=4, row2 has also matured (200+577 <= 800); row4/current is excluded.
    assert diag["pool_size_by_anchor"][4] == 3
    assert np.isfinite(preds[4]).all()
    assert READY_OFFSET_BARS == HOLD_BARS + 1
    np.testing.assert_array_equal(ready_positions_for_targets(positions), positions + READY_OFFSET_BARS)


def test_future_target_mutation_beyond_phase_cutoff_leaves_prefix_predictions_unchanged():
    n = 30
    rng = np.random.default_rng(7)
    pca = rng.normal(size=(n, 32)).astype(np.float32)
    valid = np.ones(n, dtype=bool)
    positions = np.arange(n, dtype=np.int64) * 100
    dates = pd.Series(pd.date_range("2022-01-01", periods=positions[-1] + HOLD_BARS + 10, freq="5min"))
    targets = rng.normal(size=(n, 3)).astype(np.float32)
    targets[:, 1:] = np.abs(targets[:, 1:])
    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[:12] = True
    spec = RetrievalSpec("current", 64) if 64 in K_VALUES else RetrievalSpec("current", K_VALUES[0])
    # Use direct low-k helper through retrieval by overriding spec isn't possible, so make pool >=64.
    n = 90
    pca = rng.normal(size=(n, 32)).astype(np.float32)
    valid = np.ones(n, dtype=bool)
    positions = np.arange(n, dtype=np.int64) * 100
    dates = pd.Series(pd.date_range("2022-01-01", periods=positions[-1] + HOLD_BARS + 10, freq="5min"))
    targets = rng.normal(size=(n, 3)).astype(np.float32)
    targets[:, 1:] = np.abs(targets[:, 1:])
    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[:70] = True
    cutoff = 80
    pred1, _ = retrieval_path_quantiles(pca, valid, targets, dates, positions, fit_mask, spec, start_index=70, stop_index=cutoff)
    mutated = targets.copy()
    mutated[cutoff:] = rng.normal(size=mutated[cutoff:].shape).astype(np.float32) * 1_000_000.0
    pred2, _ = retrieval_path_quantiles(pca, valid, mutated, dates, positions, fit_mask, spec, start_index=70, stop_index=cutoff)
    np.testing.assert_allclose(pred1[:cutoff], pred2[:cutoff], equal_nan=True)


def test_empirical_quantile_correctness_and_fit_prefix_nan():
    y = np.array([[0.0, 10.0, 100.0], [1.0, 11.0, 101.0], [2.0, 12.0, 102.0], [3.0, 13.0, 103.0]], dtype=np.float32)
    got = empirical_neighbor_quantiles(y)
    expected = np.array([[0.75, 1.5, 2.25], [10.75, 11.5, 12.25], [100.75, 101.5, 102.25]], dtype=np.float32)
    np.testing.assert_allclose(got, expected)

    x = np.eye(5, dtype=np.float32)
    targets = np.tile(np.array([[1.0, 0.1, 0.2]], dtype=np.float32), (5, 1))
    preds, _ = delayed_causal_retrieval_quantiles(x, np.ones(5, dtype=bool), targets, np.arange(5) * 1000, fit_memory_mask=np.array([True, True, False, False, False]), start_index=3, k=2)
    assert np.isnan(preds[:3]).all()
    assert MIN_SCORE_HISTORY == 200


def test_fit_standardizer_source_guards_and_initial_memory(tmp_path):
    reps = np.array([[1.0, 10.0], [3.0, 14.0], [100.0, 200.0]], dtype=np.float32)
    mean, std = fit_standardizer(reps, np.array([True, True, False]))
    np.testing.assert_allclose(mean, [2.0, 12.0])
    np.testing.assert_allclose(std, [1.0, 2.0])
    norm, valid = standardize_l2(reps, mean, std, np.ones(3, dtype=bool))
    np.testing.assert_allclose(np.linalg.norm(norm[valid], axis=1), np.ones(valid.sum()), atol=1e-6)

    good = tmp_path / "moment_embedding_probe_top10_manifest_retrieval.json"
    good.write_text(json.dumps({
        "later_metrics_included": False,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "representation": {"pca32": {"components_sha256": "abc"}},
        "data_sha256": {"market": "m"},
    }))
    assert load_required_source_manifest(str(good))["representation"]["pca32"]["components_sha256"] == "abc"
    bad = tmp_path / "retrieval_manifest.json"
    bad.write_text(good.read_text())
    with pytest.raises(ValueError, match="moment_embedding_probe_top10_manifest"):
        load_required_source_manifest(str(bad))

    dates = pd.Series(pd.date_range("2022-12-29", periods=5000, freq="5min"))
    positions = np.array([10, 1000, 1600], dtype=np.int64)
    targets = np.ones((3, 3), dtype=np.float32)
    mask = initial_fit_memory_mask(dates, positions, np.array([True, True, True]), targets, np.ones(3, dtype=bool))
    assert mask.tolist() == [True, False, False]


def test_2023_rerun_determinism_helper_practical():
    rng = np.random.default_rng(42)
    n = 80
    x = rng.normal(size=(n, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    targets = rng.normal(size=(n, 3)).astype(np.float32)
    targets[:, 1:] = np.abs(targets[:, 1:])
    positions = np.arange(n, dtype=np.int64) * 100
    valid = np.ones(n, dtype=bool)
    fit_memory = np.zeros(n, dtype=bool)
    fit_memory[:65] = True
    a, da = delayed_causal_retrieval_quantiles(x, valid, targets, positions, fit_memory_mask=fit_memory, start_index=65, stop_index=75, k=64)
    b, db = delayed_causal_retrieval_quantiles(x, valid, targets, positions, fit_memory_mask=fit_memory, start_index=65, stop_index=75, k=64)
    np.testing.assert_allclose(a, b, equal_nan=True)
    assert da["pool_size_by_anchor"] == db["pool_size_by_anchor"]

    compact = compact_retrieval_diagnostics(da)
    assert "pool_size_by_anchor" not in compact
    assert "kth_similarity_by_anchor" not in compact
    assert "ready_offset_by_anchor" not in compact
    assert "average_pool_size_predicted" in compact
