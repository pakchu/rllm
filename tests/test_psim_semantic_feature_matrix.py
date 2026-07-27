from __future__ import annotations

import numpy as np

from training import psim_semantic_feature_matrix as features


def _row(index: int, decision: str) -> dict[str, object]:
    return {
        "row_index": index,
        "row_hash": f"{index + 1:064x}",
        "decision_at": decision,
        "split_year": 2020,
        "forced_no_eligible": False,
        "eligible_relation_unit_count": 1,
        "selected_relation_unit_count": 1,
        "source_payload": {
            "events": [
                {
                    "protocol_side": "bitcoin",
                    "payload": {
                        "event_type": "UPDATE",
                        "old_metadata_state": "VALID",
                        "new_metadata_state": "VALID",
                        "invalid_metadata_present": False,
                        "changed_sections": ["SPECIFICATION"],
                        "old_sections": ["SPECIFICATION"],
                        "new_sections": ["SPECIFICATION"],
                        "redacted_text_delta_chunks": ["ADD|x"],
                        "counter_fields": {"update_gap_bucket": "BUCKET_A"},
                        "dependency_delta_state": "STABLE",
                    },
                }
            ],
            "relation_edges": [
                {"counterpart_state": "STATE_A", "bitcoin": "BA"}
            ],
        },
    }


def test_pca_is_train_fit_only_and_sign_canonicalized() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(40, 64)).astype(np.float32)
    future = rng.normal(loc=100.0, size=(10, 64)).astype(np.float32)

    pca = features.fit_pca(train, components=8)
    transformed = pca.transform(np.vstack([train, future]))

    assert pca.fit_row_count == 40
    assert pca.mean.shape == (64,)
    assert pca.components.shape == (8, 64)
    assert transformed.shape == (50, 8)
    np.testing.assert_allclose(pca.mean, train.mean(axis=0), rtol=0, atol=1e-6)
    for component in pca.components:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] >= 0.0


def test_source_only_permutation_stays_within_month_and_is_deterministic() -> None:
    rows = [
        _row(0, "2020-01-01T12:05:00Z"),
        _row(1, "2020-01-02T12:05:00Z"),
        _row(2, "2020-02-01T12:05:00Z"),
        _row(3, "2020-02-02T12:05:00Z"),
    ]
    first = features.source_only_month_permutation(rows)
    second = features.source_only_month_permutation(rows)

    np.testing.assert_array_equal(first, second)
    assert set(first[:2]) == {0, 1}
    assert set(first[2:]) == {2, 3}


def test_nonsemantic_feature_families_are_finite_and_text_free() -> None:
    rows = [
        _row(0, "2020-01-01T12:05:00Z"),
        _row(1, "2020-01-02T12:05:00Z"),
    ]
    metadata = features.metadata_frontmatter_features(rows)
    sizes = features.path_section_size_features(rows)
    topology = features.cadence_topology_features(rows)
    bitcoin = features.protocol_side_features(rows, "bitcoin")
    ethereum = features.protocol_side_features(rows, "ethereum")

    assert metadata.shape == (2, 11)
    assert sizes.shape == (2, 8)
    assert topology.shape == (2, features.HASH_FEATURE_WIDTH)
    assert bitcoin.shape == (2, 4)
    assert ethereum.shape == (2, 4)
    assert all(
        np.isfinite(matrix).all()
        for matrix in (metadata, sizes, topology, bitcoin, ethereum)
    )
    assert bitcoin.sum() > 0.0
    assert ethereum.sum() == 0.0
