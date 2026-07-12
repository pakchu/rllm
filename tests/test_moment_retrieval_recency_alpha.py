import json

import numpy as np
import pytest

import training.evaluate_moment_retrieval_direct_utility_alpha as direct
from training.evaluate_moment_retrieval_recency_alpha import (
    RECENCY_SIDE_POLICIES,
    RECENCY_WINDOWS,
    READY_OFFSET_BARS,
    delayed_causal_recency_retrieval_quantiles,
    derived_base_retrieval_specs,
    direct_policy_masks,
    load_required_family_source_manifest,
    recency_policy_family,
    verify_family_source_provenance,
)
from training.search_moment_retrieval_path_critic_alpha import HOLD_BARS


def _unit_rows(rows):
    x = np.asarray(rows, dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def test_age_window_excludes_old_but_keeps_mature_recent_rows():
    x = _unit_rows([[1, 0], [1, 0], [0, 1], [1, 0]])
    valid = np.ones(4, dtype=bool)
    positions = np.array([0, 500, 620, 1200], dtype=np.int64)
    targets = np.array([[10, 1, 1], [20, 2, 2], [30, 3, 3], [999, 9, 9]], dtype=np.float32)

    preds, diag = delayed_causal_recency_retrieval_quantiles(
        x,
        valid,
        targets,
        positions,
        start_index=3,
        stop_index=4,
        k=1,
        max_age_bars=700,
    )

    # row0 is mature but age 1200 > 700, rows1/2 are mature and inside window;
    # with k=1, row1 wins by cosine similarity against the query.
    assert diag["pool_size_by_anchor"][3] == 2
    np.testing.assert_allclose(preds[3, :, :], np.repeat(np.array([[20], [2], [2]], dtype=np.float32), preds.shape[2], axis=1), rtol=0, atol=0)
    assert diag["ready_offset_bars"] == READY_OFFSET_BARS == HOLD_BARS + 1
    assert diag["memory_window_max_age_bars"] == 700


def test_unready_and_current_rows_are_excluded_from_recency_memory():
    x = _unit_rows([[1, 0], [1, 0], [1, 0]])
    valid = np.ones(3, dtype=bool)
    positions = np.array([0, 700, 1200], dtype=np.int64)
    targets = np.array([[10, 1, 1], [99, 9, 9], [777, 7, 7]], dtype=np.float32)

    preds, diag = delayed_causal_recency_retrieval_quantiles(
        x,
        valid,
        targets,
        positions,
        start_index=2,
        stop_index=3,
        k=1,
        max_age_bars=None,
    )

    # row1 is recent but unready (700+577 > 1200); row2 is current; only row0 remains.
    assert diag["pool_size_by_anchor"][2] == 1
    np.testing.assert_allclose(preds[2, :, :], np.repeat(np.array([[10], [1], [1]], dtype=np.float32), preds.shape[2], axis=1), rtol=0, atol=0)


def test_family_source_guard_and_derivation_dedupes_top_rows(tmp_path):
    payload = {
        "later_metrics_included": False,
        "source_manifest_sha256": "source-hash",
        "data_sha256": {"market": "m", "funding": None, "premium": "p"},
        "top10": [
            {"retrieval_representation": "current", "k": 64, "utility_score": "median_lambda0.5", "side_policy": "both"},
            {"retrieval_representation": "current", "k": 64, "utility_score": "median_lambda0.5", "side_policy": "long"},
            {"retrieval_representation": "current_mean8", "k": 128, "utility_score": "conservative_lambda1.0", "side_policy": "both"},
        ],
    }
    path = tmp_path / "moment_retrieval_direct_utility_top10_manifest_fixture.json"
    path.write_text(json.dumps(payload))

    manifest = load_required_family_source_manifest(path)
    verify_family_source_provenance(
        manifest,
        source_manifest_sha256="source-hash",
        data_hashes={"market": "m", "funding": None, "premium": "p"},
    )
    with pytest.raises(ValueError, match="does not reference"):
        verify_family_source_provenance(
            manifest,
            source_manifest_sha256="wrong",
            data_hashes={"market": "m", "funding": None, "premium": "p"},
        )
    specs = derived_base_retrieval_specs(manifest)
    assert specs == [
        {"retrieval_representation": "current", "k": 64, "utility_score": "median_lambda0.5", "stream_id": "retrieval_pathcritic_current_k64_median_lambda0.5"},
        {"retrieval_representation": "current_mean8", "k": 128, "utility_score": "conservative_lambda1.0", "stream_id": "retrieval_pathcritic_current_mean8_k128_conservative_lambda1.0"},
    ]

    policies = recency_policy_family(specs)
    assert {p["side_policy"] for p in policies} == {"long", "both"}
    assert "short" not in {p["side_policy"] for p in policies}
    assert {p["memory_window"] for p in policies} == set(RECENCY_WINDOWS)
    assert len(policies) == len(specs) * len(RECENCY_WINDOWS) * len(RECENCY_SIDE_POLICIES)

    bad_name = tmp_path / "direct_manifest.json"
    bad_name.write_text(path.read_text())
    with pytest.raises(ValueError, match="family source manifest name"):
        load_required_family_source_manifest(bad_name)

    later = tmp_path / "moment_retrieval_direct_utility_top10_manifest_later.json"
    payload["later_metrics_included"] = True
    later.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="later_metrics_included=false"):
        load_required_family_source_manifest(later)

    payload["later_metrics_included"] = False
    payload["top10"] = [{"retrieval_representation": "oos_choice", "k": 64, "utility_score": "median_lambda0.5"}]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported retrieval representation"):
        derived_base_retrieval_specs(load_required_family_source_manifest(path))


def test_recency_retrieval_prefix_unchanged_when_future_targets_mutate():
    rng = np.random.default_rng(123)
    n = 90
    x = rng.normal(size=(n, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    valid = np.ones(n, dtype=bool)
    positions = np.arange(n, dtype=np.int64) * 100
    targets = rng.normal(size=(n, 3)).astype(np.float32)
    targets[:, 1:] = np.abs(targets[:, 1:])
    cutoff = 76

    a, da = delayed_causal_recency_retrieval_quantiles(x, valid, targets, positions, start_index=65, stop_index=cutoff, k=8, max_age_bars=210240)
    mutated = targets.copy()
    mutated[cutoff:] = rng.normal(size=mutated[cutoff:].shape).astype(np.float32) * 1_000_000.0
    b, db = delayed_causal_recency_retrieval_quantiles(x, valid, mutated, positions, start_index=65, stop_index=cutoff, k=8, max_age_bars=210240)

    np.testing.assert_allclose(a[:cutoff], b[:cutoff], equal_nan=True)
    assert da["pool_size_by_anchor"][:cutoff] == db["pool_size_by_anchor"][:cutoff]


def test_direct_masks_are_reused_from_direct_utility_module():
    assert direct_policy_masks is direct.direct_policy_masks
    scores = np.array([-1.0, 0.0, 2.0], dtype=float)
    positions = np.array([1, 2, 3], dtype=np.int64)
    long_active, short_active = direct_policy_masks(scores, positions, 5, "both")
    assert np.flatnonzero(long_active).tolist() == [3]
    assert np.flatnonzero(short_active).tolist() == [1]
