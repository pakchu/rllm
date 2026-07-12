import json

import numpy as np
import pandas as pd
import pytest
import torch

import training.search_moment_mamba2_alpha as mamba
from training.search_moment_mamba2_alpha import (
    HORIZONS,
    MIN_SCORE_HISTORY,
    MODEL_ID,
    MODEL_REVISION,
    TinyMamba2Regressor,
    build_sequences,
    fit_admissible_mask,
    fit_horizon_scales,
    first_post_fit_index,
    load_and_verify_source_manifest,
    assert_data_hashes_match_source,
    assert_pca_hashes_match_source,
    multi_horizon_open_returns,
    normalize_targets_no_mean,
    predict_mamba_scores,
    sequence_valid_mask,
    strict_validate_paths,
    transform_score_streams,
)


def test_causal_sequence_construction_future_mutation_unchanged():
    rep = np.arange(10 * 32, dtype=np.float32).reshape(10, 32)
    valid = np.ones(10, dtype=bool)
    first = build_sequences(rep, valid, 4, np.array([3, 4, 5]))
    mutated = rep.copy()
    mutated[6:] += 1_000_000.0
    second = build_sequences(mutated, valid, 4, np.array([3, 4, 5]))
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[0], rep[0:4])
    assert sequence_valid_mask(valid, 4).tolist() == [False, False, False, True, True, True, True, True, True, True]
    valid[4] = False
    with pytest.raises(ValueError, match="invalid causal sequence"):
        build_sequences(rep, valid, 4, np.array([5]))


def test_multi_horizon_targets_use_next_open_and_bounds():
    n = 2050
    market = pd.DataFrame({"open": np.exp(np.arange(n, dtype=float) / 1000.0)})
    positions = np.array([10, 80], dtype=np.int64)
    targets, meta = multi_horizon_open_returns(market, positions)
    assert meta["entry_offset_bars"] == 1
    assert targets.shape == (2, 3)
    for col, bars in enumerate(HORIZONS.values()):
        expected = np.log(market.open.iloc[positions[0] + 1 + bars] / market.open.iloc[positions[0] + 1])
        assert targets[0, col] == pytest.approx(expected)
    assert np.isfinite(targets[0]).all()
    assert np.isnan(targets[1, 2])  # 7d exit is out of bounds


def test_phase1_targets_ignore_exits_at_or_after_cutoff():
    market = pd.DataFrame({"open": np.exp(np.arange(2500, dtype=float) / 1000.0)})
    positions = np.array([10], dtype=np.int64)
    first, _ = multi_horizon_open_returns(
        market,
        positions,
        available_before_position=700,
    )
    mutated = market.copy()
    mutated.loc[700:, "open"] *= 1000.0
    second, _ = multi_horizon_open_returns(
        mutated,
        positions,
        available_before_position=700,
    )

    np.testing.assert_allclose(first, second, equal_nan=True)
    assert np.isfinite(first[0, :2]).all()
    assert np.isnan(first[0, 2])
    assert MIN_SCORE_HISTORY == 200


def test_fit_mask_embargoes_7d_horizon_and_requires_full_sequence():
    dates = pd.Series(pd.date_range("2022-12-20", periods=4300, freq="5min"))
    positions = np.array([100, 1000, 1800, 2000], dtype=np.int64)
    targets = np.ones((4, 3), dtype=np.float32)
    valid = np.ones(4, dtype=bool)
    # seq length 2 makes anchor 0 invalid; anchor 1's 7d exit is before 2023;
    # later anchors have max-horizon exits in/after 2023 and are embargoed.
    mask = fit_admissible_mask(dates, positions, targets, valid, 2)
    assert mask.tolist() == [False, True, False, False]
    assert first_post_fit_index(dates, positions, mask) == 2


def test_target_scales_are_fit_only_no_mean_and_clipped():
    targets = np.array([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0], [100.0, 100.0, 100.0]], dtype=np.float32)
    fit = np.array([True, True, False])
    scales = fit_horizon_scales(targets, fit)
    np.testing.assert_allclose(scales, np.array([1.0, 2.0, 3.0]), atol=1e-6)
    norm = normalize_targets_no_mean(targets, scales)
    np.testing.assert_allclose(norm[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(norm[2], [3.0, 3.0, 3.0])


def test_manifest_source_pca_and_data_guards(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "later_metrics_included": False,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "representation": {"pca32": {"components_sha256": "abc"}},
        "data_sha256": {"market": "m", "funding": None, "premium": "p"},
    }))
    loaded = load_and_verify_source_manifest(str(source))
    assert_pca_hashes_match_source({"pca32": {"components_sha256": "abc"}}, loaded, dimensions=(32,))
    assert_data_hashes_match_source({"market": "m", "funding": None, "premium": "p"}, loaded)
    with pytest.raises(ValueError, match="PCA component hash mismatch"):
        assert_pca_hashes_match_source({"pca32": {"components_sha256": "bad"}}, loaded, dimensions=(32,))
    with pytest.raises(ValueError, match="source data hash mismatch"):
        assert_data_hashes_match_source({"market": "m", "funding": None, "premium": "bad"}, loaded)
    out = tmp_path / "out.json"
    manifest = tmp_path / "manifest.json"
    strict_validate_paths(str(out), str(manifest), str(source))
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        strict_validate_paths(str(out), str(manifest), str(source))


def test_policy_scores_fit_prefix_stay_nan():
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))
        def forward(self, x):
            return torch.stack([x[:, -1, 0], x[:, -1, 1], x[:, -1, 2]], dim=1) + self.p
    rep = np.ones((8, 32), dtype=np.float32)
    rep[:, 0] = np.arange(8)
    rep[:, 1] = np.arange(8) + 10
    rep[:, 2] = np.arange(8) + 20
    valid = np.ones(8, dtype=bool)
    pred = predict_mamba_scores(Fake(), rep, valid, 3, start_index=5, batch_size=2)
    streams = transform_score_streams(pred, prefix="x")
    assert np.isnan(pred[:5]).all()
    for scores in streams.values():
        assert np.isnan(scores[:5]).all()
        assert np.isfinite(scores[5:]).all()


def test_tiny_mamba_forward_shape_if_available():
    try:
        model = TinyMamba2Regressor()
    except Exception as exc:  # pragma: no cover - optional transformers build issue
        pytest.skip(f"Mamba2 unavailable: {exc}")
    x = torch.randn(2, 4, 32)
    y = model(x)
    assert y.shape == (2, 3)
