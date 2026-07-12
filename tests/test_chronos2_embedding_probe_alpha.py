import numpy as np
import torch

from training.search_chronos2_embedding_probe_alpha import (
    fit_pca_representations,
    optional_file_sha256,
    summarize_chronos_embedding,
)


def test_embedding_summary_uses_target_and_group_tokens():
    embedding = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)

    summary = summarize_chronos_embedding(embedding)

    assert summary.shape == (12,)
    np.testing.assert_allclose(summary[:3], embedding[0, -2].numpy())
    np.testing.assert_allclose(summary[3:6], embedding[0, -1].numpy())
    np.testing.assert_allclose(summary[6:9], embedding[:, -2].mean(0).numpy())
    np.testing.assert_allclose(summary[9:12], embedding[:, :-2].mean((0, 1)).numpy())


def test_pca_fit_ignores_future_embedding_mutation():
    rng = np.random.default_rng(7)
    summaries = rng.normal(size=(30, 10)).astype(np.float32)
    fit = np.zeros(30, dtype=bool)
    fit[:20] = True
    valid = np.ones(30, dtype=bool)

    first, first_meta = fit_pca_representations(
        summaries, fit, valid, dimensions=(3,)
    )
    mutated = summaries.copy()
    mutated[20:] += 1e6
    second, second_meta = fit_pca_representations(
        mutated, fit, valid, dimensions=(3,)
    )

    assert first_meta["pca3"]["components_sha256"] == second_meta["pca3"]["components_sha256"]
    np.testing.assert_allclose(first["pca3"][:20], second["pca3"][:20], atol=1e-6)


def test_optional_hash_accepts_missing_auxiliary_path(tmp_path):
    assert optional_file_sha256("") is None

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"chronos")
    assert len(optional_file_sha256(str(artifact))) == 64
