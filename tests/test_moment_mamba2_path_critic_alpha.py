import json

import numpy as np
import pandas as pd
import pytest
import torch

from training.search_bidirectional_state_alpha import Config
from training.search_moment_embedding_probe_alpha import MODEL_ID, MODEL_REVISION
from training.search_moment_mamba2_path_critic_alpha import (
    HOLD_BARS,
    MIN_SCORE_HISTORY,
    QUANTILES,
    SEQUENCE_LENGTH,
    TARGET_COMPONENTS,
    denormalize_sort_and_clamp_quantiles,
    executable_path_targets_48h,
    fit_admissible_mask,
    fit_component_scales,
    first_post_fit_index,
    load_required_source_manifest,
    normalize_path_targets_no_mean,
    pinball_loss,
    predict_path_quantiles,
    round_trip_unlevered_cost,
    signed_utility_scores,
)


def _market(n, start="2022-01-01"):
    open_ = np.full(n, 100.0, dtype=float)
    high = np.full(n, 101.0, dtype=float)
    low = np.full(n, 99.0, dtype=float)
    return pd.DataFrame({"date": pd.date_range(start, periods=n, freq="5min"), "open": open_, "high": high, "low": low})


def test_path_targets_indexing_mae_cutoff_and_future_mutation():
    n = 700
    market = _market(n)
    p = 10
    entry = p + 1
    exit_ = entry + HOLD_BARS
    market.loc[entry, "open"] = 100.0
    market.loc[exit_, "open"] = 110.0
    market.loc[entry:exit_ - 1, "low"] = 98.0
    market.loc[entry + 5, "low"] = 90.0
    market.loc[entry:exit_ - 1, "high"] = 103.0
    market.loc[entry + 9, "high"] = 125.0
    # Exit bar extremes must not affect intratrade MAE.
    market.loc[exit_, "low"] = 1.0
    market.loc[exit_, "high"] = 1000.0

    targets, meta = executable_path_targets_48h(market, np.array([p], dtype=np.int64))
    assert meta["components"] == list(TARGET_COMPONENTS)
    np.testing.assert_allclose(targets[0], [0.10, 0.10, 0.25], atol=1e-7)

    cutoff, _ = executable_path_targets_48h(market, np.array([p], dtype=np.int64), available_before_position=exit_)
    assert np.isnan(cutoff).all()
    before = executable_path_targets_48h(market, np.array([p], dtype=np.int64), available_before_position=exit_)[0]
    mutated = market.copy()
    mutated.loc[exit_:, ["open", "high", "low"]] *= 1000.0
    after = executable_path_targets_48h(mutated, np.array([p], dtype=np.int64), available_before_position=exit_)[0]
    np.testing.assert_allclose(before, after, equal_nan=True)
    assert MIN_SCORE_HISTORY == 200
    assert SEQUENCE_LENGTH == 32


def test_pinball_loss_basic_correctness():
    pred = torch.zeros((1, 3, 3), dtype=torch.float32)
    target = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float32)
    got = pinball_loss(pred, target)
    # Component losses: mean(q*1)=0.5, mean((1-q)*1)=0.5, zeros=0.
    assert got.item() == pytest.approx((0.5 + 0.5 + 0.0) / 3.0)
    perfect = target.unsqueeze(-1).expand(-1, -1, len(QUANTILES))
    assert pinball_loss(perfect, target).item() == pytest.approx(0.0)


def test_quantile_sorting_mae_clamp_and_utility_signed_side():
    cfg = Config(input_csv="dummy", output="dummy")
    assert round_trip_unlevered_cost(cfg) == pytest.approx(0.0012)
    pred = np.array(
        [
            [[0.03, 0.01, 0.02], [-0.10, -0.02, 0.004], [0.01, 0.02, 0.03]],
            [[-0.03, -0.05, -0.04], [0.01, 0.02, 0.03], [-0.20, -0.01, 0.04]],
            [[0.0001, 0.0002, 0.0003], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    q = denormalize_sort_and_clamp_quantiles(pred, np.ones(3, dtype=np.float32))
    np.testing.assert_allclose(q[0, 0], [0.01, 0.02, 0.03])
    assert (q[:, 1:, :] >= 0.0).all()

    streams, meta = signed_utility_scores(q, cfg)
    assert meta["round_trip_unlevered_cost"] == pytest.approx(0.0012)
    # Row 0: long utility wins; row 1: short utility wins; row 2 abstains under lambda1.
    assert streams["median_lambda0.5"][0] > 0
    assert streams["median_lambda0.5"][1] < 0
    assert streams["median_lambda1.0"][2] == pytest.approx(0.0)
    cons = streams["conservative_lambda0.5"]
    assert cons[0] > 0
    assert cons[1] < 0


def test_fit_embargo_requires_full_causal_sequence_and_exit_before_2023():
    dates = pd.Series(pd.date_range("2022-12-26", periods=4300, freq="5min"))
    positions = np.array([100, 1000, 1800, 2000], dtype=np.int64)
    targets = np.ones((4, 3), dtype=np.float32)
    valid = np.ones(4, dtype=bool)
    mask = fit_admissible_mask(dates, positions, targets, valid, sequence_length=2)
    assert mask.tolist() == [False, True, False, False]
    assert first_post_fit_index(dates, positions, mask) == 2
    targets[1, 1] = np.nan
    assert fit_admissible_mask(dates, positions, targets, valid, sequence_length=2).tolist() == [False, False, False, False]


def test_frozen_source_manifest_guards(tmp_path):
    good = tmp_path / "moment_embedding_probe_top10_manifest_test.json"
    good.write_text(json.dumps({
        "later_metrics_included": False,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "representation": {"pca32": {"components_sha256": "abc"}},
        "data_sha256": {"market": "m"},
    }))
    loaded = load_required_source_manifest(str(good))
    assert loaded["representation"]["pca32"]["components_sha256"] == "abc"

    wrong_name = tmp_path / "derived_manifest.json"
    wrong_name.write_text(good.read_text())
    with pytest.raises(ValueError, match="moment_embedding_probe_top10_manifest"):
        load_required_source_manifest(str(wrong_name))

    missing_pca = tmp_path / "moment_embedding_probe_top10_manifest_missing.json"
    missing_pca.write_text(json.dumps({"later_metrics_included": False, "model": {"id": MODEL_ID, "revision": MODEL_REVISION}, "representation": {}}))
    with pytest.raises(ValueError, match="PCA32"):
        load_required_source_manifest(str(missing_pca))


def test_fit_prefix_scores_and_target_scaling_stay_nan():
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            base = torch.stack([x[:, -1, 0], x[:, -1, 1], x[:, -1, 2]], dim=1)
            return base.unsqueeze(-1).expand(-1, 3, 3) + self.p

    rep = np.ones((40, 32), dtype=np.float32)
    rep[:, 0] = np.arange(40)
    rep[:, 1] = np.arange(40) + 10
    rep[:, 2] = np.arange(40) + 20
    valid = np.ones(40, dtype=bool)
    pred = predict_path_quantiles(Fake(), rep, valid, 3, start_index=5, batch_size=2)
    assert np.isnan(pred[:5]).all()
    assert np.isfinite(pred[5:]).all()

    targets = np.array([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0], [100.0, -100.0, 100.0]], dtype=np.float32)
    fit = np.array([True, True, False])
    scales = fit_component_scales(targets, fit)
    np.testing.assert_allclose(scales, [1.0, 2.0, 3.0], atol=1e-6)
    norm = normalize_path_targets_no_mean(targets, scales)
    np.testing.assert_allclose(norm[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(norm[2], [3.0, 0.0, 3.0])
