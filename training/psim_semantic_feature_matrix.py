"""Deterministic source-only PSIM semantic and control feature matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from training import psim_semantic_transition_labels as transitions


RELATION_LABELS = (
    "ABSTAIN",
    "INDEPENDENT_INTENT",
    "CONVERGENT_INTENT",
    "TECHNICAL_TENSION",
    "INSUFFICIENT_EVIDENCE",
    "COMPLEMENTARY_INTENT",
)
HASH_FEATURE_WIDTH = 32
PCA_COMPONENTS = 32
SHUFFLE_SEED = 20_260_727


@dataclass(frozen=True)
class SourceFeatureBundle:
    rows: tuple[dict[str, Any], ...]
    embeddings: np.ndarray
    relation_logits: np.ndarray
    relation_forwarded: np.ndarray
    relation_features: np.ndarray


@dataclass(frozen=True)
class FrozenPCA:
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray
    fit_row_count: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.mean.shape[0]:
            raise ValueError("PSIM semantic PCA input shape changed")
        transformed = (array - self.mean) @ self.components.T
        if not np.isfinite(transformed).all():
            raise ValueError("PSIM semantic PCA output is non-finite")
        return transformed.astype(np.float32)


def _read_relation_rows(path: str | Path) -> list[dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        payload = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError("PSIM semantic relation rows changed")
    return payload


def _canonical_relation_features(
    rows: Sequence[Mapping[str, Any]],
    logits: np.ndarray,
    forwarded: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(rows), len(RELATION_LABELS) + 3), dtype=np.float32)
    for index, row in enumerate(rows):
        if bool(forwarded[index]):
            code_to_label = row.get("relation_teacher_code_to_label")
            if not isinstance(code_to_label, Mapping):
                raise ValueError("PSIM semantic relation codebook changed")
            canonical = np.empty(len(RELATION_LABELS), dtype=np.float64)
            for code_index, code in enumerate("ABCDEF"):
                label = str(code_to_label.get(code))
                try:
                    label_index = RELATION_LABELS.index(label)
                except ValueError as exc:
                    raise ValueError(
                        "PSIM semantic relation label changed"
                    ) from exc
                canonical[label_index] = float(logits[index, code_index])
            shifted = canonical - float(np.max(canonical))
            probabilities = np.exp(shifted)
            probabilities /= float(probabilities.sum())
            ordered = np.sort(probabilities)
            margin = float(ordered[-1] - ordered[-2])
            entropy = float(
                -np.sum(
                    probabilities
                    * np.log(np.maximum(probabilities, 1e-12))
                )
            )
            output[index, : len(RELATION_LABELS)] = probabilities
            output[index, -3:] = (1.0, margin, entropy)
        else:
            forced = str(row.get("forced_relation_when_teacher_skipped"))
            try:
                forced_index = RELATION_LABELS.index(forced)
            except ValueError as exc:
                raise ValueError(
                    "PSIM semantic forced relation changed"
                ) from exc
            output[index, forced_index] = 1.0
            output[index, -3:] = (0.0, 1.0, 0.0)
    if not np.isfinite(output).all():
        raise ValueError("PSIM semantic relation features are non-finite")
    return output


def load_source_bundle(
    source_rows_path: str | Path,
    embeddings_path: str | Path,
    relation_logits_path: str | Path,
    relation_rows_path: str | Path,
) -> SourceFeatureBundle:
    rows = transitions.load_source_rows(source_rows_path)
    relation_rows = _read_relation_rows(relation_rows_path)
    embedding_payload = np.load(embeddings_path, allow_pickle=False)
    relation_payload = np.load(relation_logits_path, allow_pickle=False)
    embeddings = np.asarray(
        embedding_payload["embedding"],
        dtype=np.float32,
    )
    embedding_indices = np.asarray(
        embedding_payload["row_index"],
        dtype=np.int64,
    )
    logits = np.asarray(relation_payload["logits"], dtype=np.float32)
    relation_indices = np.asarray(
        relation_payload["row_index"],
        dtype=np.int64,
    )
    forwarded = np.asarray(
        relation_payload["forwarded"],
        dtype=np.uint8,
    )
    count = len(rows)
    expected_indices = np.arange(count, dtype=np.int64)
    if (
        embeddings.shape != (count, 2_560)
        or logits.shape != (count, 6)
        or forwarded.shape != (count,)
        or not np.array_equal(embedding_indices, expected_indices)
        or not np.array_equal(relation_indices, expected_indices)
        or len(relation_rows) != count
        or not np.isfinite(embeddings).all()
        or not np.isfinite(logits[forwarded.astype(bool)]).all()
        or not np.isnan(logits[~forwarded.astype(bool)]).all()
    ):
        raise ValueError("PSIM semantic source feature artifacts changed")
    for index, (source, relation) in enumerate(zip(rows, relation_rows)):
        if (
            int(relation.get("row_index", -1)) != index
            or relation.get("source_row_hash") != source["row_hash"]
            or bool(relation.get("relation_teacher_forwarded"))
            != bool(forwarded[index])
        ):
            raise ValueError("PSIM semantic relation/source alignment changed")
    relation_features = _canonical_relation_features(
        rows,
        logits,
        forwarded,
    )
    return SourceFeatureBundle(
        rows=tuple(rows),
        embeddings=embeddings,
        relation_logits=logits,
        relation_forwarded=forwarded,
        relation_features=relation_features,
    )


def fit_pca(
    embeddings: np.ndarray,
    *,
    components: int = PCA_COMPONENTS,
) -> FrozenPCA:
    values = np.asarray(embeddings, dtype=np.float64)
    if (
        values.ndim != 2
        or len(values) <= components
        or values.shape[1] < components
        or not np.isfinite(values).all()
    ):
        raise ValueError("PSIM semantic PCA fit input changed")
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    _, singular_values, vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    basis = vectors[:components].copy()
    for index in range(len(basis)):
        pivot = int(np.argmax(np.abs(basis[index])))
        if basis[index, pivot] < 0.0:
            basis[index] *= -1.0
    singular = singular_values[:components].copy()
    explained = singular**2 / float(len(values) - 1)
    if (
        basis.shape != (components, values.shape[1])
        or not np.isfinite(basis).all()
        or not np.isfinite(explained).all()
    ):
        raise RuntimeError("PSIM semantic PCA decomposition changed")
    return FrozenPCA(
        mean=mean,
        components=basis,
        singular_values=singular,
        explained_variance=explained,
        fit_row_count=len(values),
    )


def year_indices(
    rows: Sequence[Mapping[str, Any]],
    year: int,
) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if int(row["split_year"]) == int(year)
        ],
        dtype=np.int64,
    )


def _safe_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _event_rows(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payload = row.get("source_payload")
    if not isinstance(payload, Mapping):
        return []
    events = payload.get("events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, Mapping)]


def _event_payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("payload")
    return value if isinstance(value, Mapping) else {}


def metadata_frontmatter_features(
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    output: list[list[float]] = []
    for row in rows:
        events = _event_rows(row)
        payloads = [_event_payload(event) for event in events]
        output.append(
            [
                float(bool(row.get("forced_no_eligible"))),
                math.log1p(float(row.get("eligible_relation_unit_count", 0))),
                math.log1p(float(row.get("selected_relation_unit_count", 0))),
                math.log1p(len(events)),
                math.log1p(
                    sum(event.get("protocol_side") == "bitcoin" for event in events)
                ),
                math.log1p(
                    sum(event.get("protocol_side") == "ethereum" for event in events)
                ),
                math.log1p(
                    sum(bool(payload.get("invalid_metadata_present")) for payload in payloads)
                ),
                math.log1p(
                    sum(payload.get("old_metadata_state") == "VALID" for payload in payloads)
                ),
                math.log1p(
                    sum(payload.get("new_metadata_state") == "VALID" for payload in payloads)
                ),
                math.log1p(
                    sum(payload.get("event_type") == "CREATE" for payload in payloads)
                ),
                math.log1p(
                    sum(payload.get("event_type") == "UPDATE" for payload in payloads)
                ),
            ]
        )
    return np.asarray(output, dtype=np.float32)


def path_section_size_features(
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    output: list[list[float]] = []
    for row in rows:
        events = _event_rows(row)
        payloads = [_event_payload(event) for event in events]
        chunks = [
            str(chunk)
            for payload in payloads
            for chunk in _safe_list(payload.get("redacted_text_delta_chunks"))
        ]
        relation_edges = row["source_payload"].get("relation_edges", [])
        output.append(
            [
                math.log1p(len(events)),
                math.log1p(len(relation_edges)),
                math.log1p(
                    sum(len(_safe_list(payload.get("changed_sections"))) for payload in payloads)
                ),
                math.log1p(
                    sum(len(_safe_list(payload.get("old_sections"))) for payload in payloads)
                ),
                math.log1p(
                    sum(len(_safe_list(payload.get("new_sections"))) for payload in payloads)
                ),
                math.log1p(len(chunks)),
                math.log1p(sum(len(chunk) for chunk in chunks)),
                math.log1p(max((len(chunk) for chunk in chunks), default=0)),
            ]
        )
    return np.asarray(output, dtype=np.float32)


def _hash_bucket(token: str, *, width: int = HASH_FEATURE_WIDTH) -> int:
    raw = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") % int(width)


def cadence_topology_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    width: int = HASH_FEATURE_WIDTH,
) -> np.ndarray:
    output = np.zeros((len(rows), width), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for event in _event_rows(row):
            payload = _event_payload(event)
            counters = payload.get("counter_fields")
            if isinstance(counters, Mapping):
                for key, value in sorted(counters.items()):
                    output[
                        row_index,
                        _hash_bucket(f"counter:{key}:{value}", width=width),
                    ] += 1.0
            state = payload.get("dependency_delta_state")
            output[
                row_index,
                _hash_bucket(f"dependency:{state}", width=width),
            ] += 1.0
        edges = row["source_payload"].get("relation_edges", [])
        if isinstance(edges, list):
            for edge in edges:
                if not isinstance(edge, Mapping):
                    continue
                state = edge.get("counterpart_state")
                output[
                    row_index,
                    _hash_bucket(f"counterpart:{state}", width=width),
                ] += 1.0
    return np.log1p(output)


def protocol_side_features(
    rows: Sequence[Mapping[str, Any]],
    side: str,
) -> np.ndarray:
    output: list[list[float]] = []
    for row in rows:
        events = [
            event
            for event in _event_rows(row)
            if event.get("protocol_side") == side
        ]
        payloads = [_event_payload(event) for event in events]
        output.append(
            [
                math.log1p(len(events)),
                math.log1p(
                    sum(len(_safe_list(payload.get("changed_sections"))) for payload in payloads)
                ),
                math.log1p(
                    sum(
                        len(_safe_list(payload.get("redacted_text_delta_chunks")))
                        for payload in payloads
                    )
                ),
                math.log1p(
                    sum(bool(payload.get("invalid_metadata_present")) for payload in payloads)
                ),
            ]
        )
    return np.asarray(output, dtype=np.float32)


def source_only_month_permutation(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = SHUFFLE_SEED,
) -> np.ndarray:
    months: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        month = str(row["decision_at"])[:7]
        months.setdefault(month, []).append(index)
    permutation = np.arange(len(rows), dtype=np.int64)
    for month, indices in sorted(months.items()):
        source = list(indices)
        ordered = sorted(
            source,
            key=lambda index: hashlib.sha256(
                (
                    f"{seed}\x00{month}\x00"
                    f"{rows[index]['row_hash']}"
                ).encode("utf-8")
            ).digest(),
        )
        for destination, source_index in zip(indices, ordered):
            permutation[destination] = source_index
    return permutation


def build_feature_family(
    bundle: SourceFeatureBundle,
    pca: FrozenPCA,
) -> dict[str, np.ndarray]:
    rows = bundle.rows
    semantic = pca.transform(bundle.embeddings)
    permutation = source_only_month_permutation(rows)
    metadata = metadata_frontmatter_features(rows)
    path_sizes = path_section_size_features(rows)
    bitcoin = protocol_side_features(rows, "bitcoin")
    ethereum = protocol_side_features(rows, "ethereum")
    family = {
        "semantic": semantic,
        "metadata_frontmatter_only": metadata,
        "path_section_diff_size_only": path_sizes,
        "cadence_revision_topology_only": cadence_topology_features(rows),
        "shuffled_eip_bip_daily_relation": bundle.relation_features[permutation],
        "shuffled_old_new_pairing": semantic[permutation],
        "future_status_scrub": np.concatenate(
            [path_sizes[:, :2], bitcoin[:, :3], ethereum[:, :3]],
            axis=1,
        ),
        "ethereum_only": ethereum,
        "bitcoin_only": bitcoin,
        "current_position_only": np.zeros((len(rows), 0), dtype=np.float32),
        "masked_semantic_embedding": np.zeros_like(semantic),
    }
    for name, values in family.items():
        if values.ndim != 2 or len(values) != len(rows):
            raise RuntimeError(f"PSIM semantic feature family changed: {name}")
        if not np.isfinite(values).all():
            raise ValueError(f"PSIM semantic feature is non-finite: {name}")
    return family
