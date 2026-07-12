import json

import numpy as np
import pytest

from training.search_moment_embedding_probe_alpha import MODEL_ID, MODEL_REVISION
from training.search_moment_retrieval_path_critic_alpha import (
    HOLD_BARS,
    READY_OFFSET_BARS,
    delayed_causal_retrieval_quantiles,
)
from training.evaluate_moment_retrieval_direct_utility_alpha import (
    DIRECT_POLICY_DESCRIPTION,
    TOTAL_POLICY_SPECS,
    direct_policy_family,
    direct_policy_masks,
    load_required_source_manifest,
    strict_validate_paths,
)


def test_direct_mask_semantics_side_restrictions_and_zero_flat():
    scores = np.array([np.nan, -2.0, -0.0, 0.0, 1.5, -3.0, 4.0], dtype=float)
    positions = np.array([1, 3, 5, 7, 9, 11, 13], dtype=np.int64)

    long_active, short_active = direct_policy_masks(scores, positions, 20, "both")
    assert np.flatnonzero(long_active).tolist() == [9, 13]
    assert np.flatnonzero(short_active).tolist() == [3, 11]
    assert not long_active[5] and not short_active[5]
    assert not long_active[7] and not short_active[7]

    long_only, short_none = direct_policy_masks(scores, positions, 20, "long")
    assert np.flatnonzero(long_only).tolist() == [9, 13]
    assert not short_none.any()

    long_none, short_only = direct_policy_masks(scores, positions, 20, "short")
    assert not long_none.any()
    assert np.flatnonzero(short_only).tolist() == [3, 11]

    with pytest.raises(ValueError, match="unknown side policy"):
        direct_policy_masks(scores, positions, 20, "flat")  # type: ignore[arg-type]


def test_fixed_policy_family_is_48_and_has_no_threshold_fields():
    family = direct_policy_family()
    assert TOTAL_POLICY_SPECS == 48
    assert len(family) == 48
    assert len({(p["stream_id"], p["side_policy"]) for p in family}) == 48
    assert {p["retrieval_representation"] for p in family} == {"current", "current_mean8"}
    assert {p["k"] for p in family} == {64, 128}
    assert {p["side_policy"] for p in family} == {"long", "short", "both"}
    assert all(p["policy_rule"] == DIRECT_POLICY_DESCRIPTION for p in family)
    assert all("score_quantile" not in p and "rolling_score_window_anchors" not in p for p in family)


def test_delayed_retrieval_prefix_unchanged_when_future_targets_mutate():
    rng = np.random.default_rng(123)
    n = 90
    x = rng.normal(size=(n, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    valid = np.ones(n, dtype=bool)
    positions = np.arange(n, dtype=np.int64) * 100
    targets = rng.normal(size=(n, 3)).astype(np.float32)
    targets[:, 1:] = np.abs(targets[:, 1:])
    fit_memory = np.zeros(n, dtype=bool)
    fit_memory[:65] = True
    cutoff = 76

    a, da = delayed_causal_retrieval_quantiles(x, valid, targets, positions, fit_memory_mask=fit_memory, start_index=65, stop_index=cutoff, k=64)
    mutated = targets.copy()
    mutated[cutoff:] = rng.normal(size=mutated[cutoff:].shape).astype(np.float32) * 1_000_000.0
    b, db = delayed_causal_retrieval_quantiles(x, valid, mutated, positions, fit_memory_mask=fit_memory, start_index=65, stop_index=cutoff, k=64)

    np.testing.assert_allclose(a[:cutoff], b[:cutoff], equal_nan=True)
    assert da["pool_size_by_anchor"][:cutoff] == db["pool_size_by_anchor"][:cutoff]
    assert READY_OFFSET_BARS == HOLD_BARS + 1


def test_output_source_guards_require_frozen_embedding_probe_manifest(tmp_path):
    good = tmp_path / "moment_embedding_probe_top10_manifest_direct_source.json"
    good.write_text(json.dumps({
        "later_metrics_included": False,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "representation": {"pca32": {"components_sha256": "abc"}},
        "data_sha256": {"market": "m"},
    }))
    assert load_required_source_manifest(str(good))["representation"]["pca32"]["components_sha256"] == "abc"

    bad_name = tmp_path / "retrieval_manifest.json"
    bad_name.write_text(good.read_text())
    with pytest.raises(ValueError, match="moment_embedding_probe_top10_manifest"):
        load_required_source_manifest(str(bad_name))

    later = tmp_path / "moment_embedding_probe_top10_manifest_later.json"
    payload = json.loads(good.read_text())
    payload["later_metrics_included"] = True
    later.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_required_source_manifest(str(later))

    output = tmp_path / "out.json"
    manifest = tmp_path / "manifest.json"
    strict_validate_paths(str(output), str(manifest), str(good))
    output.write_text("{}")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        strict_validate_paths(str(output), str(manifest), str(good))
    with pytest.raises(ValueError, match="distinct"):
        strict_validate_paths(str(good), str(manifest), str(good))


def test_direct_policy_helper_is_deterministic_practical():
    rng = np.random.default_rng(5)
    scores = rng.normal(size=100).astype(np.float32)
    scores[::10] = 0.0
    scores[1::10] = np.nan
    positions = np.arange(100, dtype=np.int64) * 2
    a = direct_policy_masks(scores, positions, 250, "both")
    b = direct_policy_masks(scores.copy(), positions.copy(), 250, "both")
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.any(a[0] & a[1])
