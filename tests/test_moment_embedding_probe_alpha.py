from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import training.search_moment_embedding_probe_alpha as moment
from training.search_moment_embedding_probe_alpha import (
    EMBEDDING_VARIATES,
    extract_embedding_summaries,
    fit_only_tail_labels,
    fit_pca_representations,
    strict_validate_output_paths,
    summarize_moment_embedding,
    summarize_moment_embedding_batch,
)


class FakeMoment:
    def __init__(self):
        self.seen = []

    def embed(self, *, x_enc, input_mask=None, reduction="mean", **kwargs):
        assert reduction == "none"
        assert input_mask is not None
        self.seen.append(x_enc.detach().cpu().clone())
        # Deterministic B,C,P,D embeddings derived only from the supplied context.
        patches = torch.stack([x_enc.mean(dim=2), x_enc[:, :, -1]], dim=2)
        embeddings = torch.stack([patches, patches + 1000.0], dim=-1)
        return SimpleNamespace(embeddings=embeddings)


def test_moment_embedding_summary_preserves_channel_identity():
    old_variates = moment.EMBEDDING_VARIATES
    moment.EMBEDDING_VARIATES = ("a", "b")
    try:
        embedding = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

        summary = summarize_moment_embedding(embedding)

        assert summary.shape == (16,)
        np.testing.assert_allclose(summary[:8], embedding.mean(dim=1).reshape(-1).numpy())
        np.testing.assert_allclose(summary[8:], embedding[:, -1, :].reshape(-1).numpy())

        batch_summary = summarize_moment_embedding_batch(embedding.unsqueeze(0))
        assert batch_summary.shape == (1, 16)
        np.testing.assert_allclose(batch_summary[0], summary)
    finally:
        moment.EMBEDDING_VARIATES = old_variates


def test_causal_context_extraction_uses_only_completed_past_values():
    rows = 8
    context = 4
    hourly = pd.DataFrame(
        {
            name: np.arange(rows, dtype=np.float32) + channel * 100.0
            for channel, name in enumerate(EMBEDDING_VARIATES)
        }
    )
    hour_indices = np.array([3, 4], dtype=int)
    fake = FakeMoment()

    first, valid, meta = extract_embedding_summaries(
        fake,
        hourly,
        hour_indices,
        context_hours=context,
        chunk_size=2,
        batch_size=2,
        device="cpu",
    )
    mutated = hourly.copy()
    mutated.iloc[5:, :] += 1_000_000.0
    second, second_valid, _ = extract_embedding_summaries(
        FakeMoment(),
        mutated,
        hour_indices,
        context_hours=context,
        chunk_size=2,
        batch_size=2,
        device="cpu",
    )

    np.testing.assert_array_equal(valid, np.array([0, 1]))
    np.testing.assert_array_equal(second_valid, valid)
    np.testing.assert_allclose(first, second)
    assert meta["causal_context_rule"].endswith("values[:, end-context:end] only")
    seen = fake.seen[0].numpy()
    np.testing.assert_allclose(seen[0], hourly.to_numpy(np.float32).T[:, 0:4])
    np.testing.assert_allclose(seen[1], hourly.to_numpy(np.float32).T[:, 1:5])


def test_pca_fit_ignores_future_embedding_rows():
    rng = np.random.default_rng(713)
    summaries = rng.normal(size=(40, 12)).astype(np.float32)
    fit = np.zeros(40, dtype=bool)
    fit[:24] = True
    valid = np.ones(40, dtype=bool)

    first, first_meta = fit_pca_representations(summaries, fit, valid, dimensions=(4,))
    mutated = summaries.copy()
    mutated[24:] -= 1e6
    second, second_meta = fit_pca_representations(mutated, fit, valid, dimensions=(4,))

    assert first_meta["pca4"]["components_sha256"] == second_meta["pca4"]["components_sha256"]
    np.testing.assert_allclose(first["pca4"][:24], second["pca4"][:24], atol=1e-6)


def test_tail_labels_do_not_read_or_label_future_targets():
    targets = np.linspace(-1.0, 1.0, 20)
    fit = np.zeros(20, dtype=bool)
    fit[:12] = True

    labels, thresholds = fit_only_tail_labels(targets, fit)
    mutated = targets.copy()
    mutated[12:] = np.nan
    second_labels, second_thresholds = fit_only_tail_labels(mutated, fit)

    np.testing.assert_array_equal(labels, second_labels)
    np.testing.assert_allclose(thresholds, second_thresholds)
    np.testing.assert_array_equal(labels[~fit], np.ones((~fit).sum(), dtype=np.int64))


def test_strict_output_paths_refuse_overwrite(tmp_path):
    output = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    strict_validate_output_paths(str(output), str(manifest))

    output.write_text("{}")
    with pytest.raises(FileExistsError):
        strict_validate_output_paths(str(output), str(manifest))
    output.unlink()

    manifest.write_text("{}")
    with pytest.raises(FileExistsError):
        strict_validate_output_paths(str(output), str(manifest))
    with pytest.raises(ValueError):
        strict_validate_output_paths(str(manifest), str(manifest))
