import json

import numpy as np
import pytest

import training.search_moment_continual_probe_alpha as continual
from training.search_moment_continual_probe_alpha import (
    MODEL_ID,
    MODEL_REVISION,
    PROBE_SPECS,
    ContinualProbe,
    _fit_initial_probe,
    apply_standardizer,
    assert_data_hashes_match_source,
    assert_pca_hashes_match_source,
    delayed_continual_scores,
    fit_standardizer,
    fit_target_calibration,
    label_ready_positions,
    load_and_verify_source_manifest,
    strict_validate_paths,
)


def _tiny_probe():
    probe = ContinualProbe(1, PROBE_SPECS[0], seed=713)
    # deterministic near-zero model is enough for protocol tests
    return probe


def test_label_is_not_read_before_ready_position_and_predict_before_own_queue():
    positions = np.array([0, 2, 4, 6], dtype=np.int64)
    ready = np.array([5, 7, 9, 11], dtype=np.int64)
    x = np.arange(4, dtype=np.float32).reshape(-1, 1)
    targets = np.array([-0.3, -0.1, 0.2, 0.4], dtype=float)
    calibration = {"low": -0.2, "high": 0.2, "scale": 0.2, "clip_std": 3.0}
    events = []

    _, diagnostics = delayed_continual_scores(
        _tiny_probe(),
        matrix=x,
        targets=targets,
        signal_positions=positions,
        ready_positions=ready,
        stream_start_index=0,
        calibration=calibration,
        event_log=events,
    )

    assert diagnostics["label_read_indices"] == [0]
    assert events[:4] == [("predict", 0, ()), ("queue", 0, ()), ("predict", 1, ()), ("queue", 1, ())]
    predict_event_index = {idx: i for i, (kind, idx, _buf) in enumerate(events) if kind == "predict"}
    queue_event_index = {idx: i for i, (kind, idx, _buf) in enumerate(events) if kind == "queue"}
    for idx in queue_event_index:
        assert predict_event_index[idx] < queue_event_index[idx]
    assert ("learn", 0, (0,)) in events
    learn0_at = events.index(("learn", 0, (0,)))
    assert learn0_at < events.index(("predict", 3, (0,)))
    assert events[learn0_at - 1] == ("queue", 2, ())
    assert all(kind != "learn" or idx != 1 for kind, idx, _buf in events)


def test_future_target_mutation_beyond_phase_cutoff_leaves_prefix_scores_unchanged():
    rng = np.random.default_rng(713)
    x = rng.normal(size=(12, 2)).astype(np.float32)
    targets = np.linspace(-0.5, 0.5, 12)
    fit = np.zeros(12, dtype=bool)
    fit[:4] = True
    mean, std = fit_standardizer(x, fit)
    x_std = apply_standardizer(x, mean, std)
    calibration = fit_target_calibration(targets, fit)
    positions = np.arange(0, 24, 2, dtype=np.int64)
    ready = positions + 5
    cutoff = int(positions[8])

    first, _ = delayed_continual_scores(
        ContinualProbe(2, PROBE_SPECS[4], seed=713),
        matrix=x_std,
        targets=targets,
        signal_positions=positions,
        ready_positions=ready,
        stream_start_index=4,
        calibration=calibration,
        replay_init_indices=np.array([0, 1, 2, 3]),
        stop_before_signal_position=cutoff,
    )
    mutated = targets.copy()
    mutated[8:] = rng.normal(1000.0, 1.0, size=4)
    second, _ = delayed_continual_scores(
        ContinualProbe(2, PROBE_SPECS[4], seed=713),
        matrix=x_std,
        targets=mutated,
        signal_positions=positions,
        ready_positions=ready,
        stream_start_index=4,
        calibration=calibration,
        replay_init_indices=np.array([0, 1, 2, 3]),
        stop_before_signal_position=cutoff,
    )

    prefix = positions < cutoff
    np.testing.assert_allclose(first[prefix], second[prefix], atol=1e-7, equal_nan=True)


def test_source_manifest_model_and_later_metric_guards(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"later_metrics_included": False, "model": {"id": MODEL_ID, "revision": MODEL_REVISION}, "representation": {"pca16": {"components_sha256": "a"}, "pca32": {"components_sha256": "b"}}}))

    loaded = load_and_verify_source_manifest(str(source))
    assert loaded["later_metrics_included"] is False
    assert_pca_hashes_match_source({"pca16": {"components_sha256": "a"}, "pca32": {"components_sha256": "b"}}, loaded)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"later_metrics_included": True, "model": {"id": MODEL_ID, "revision": MODEL_REVISION}}))
    with pytest.raises(ValueError, match="later_metrics_included=false"):
        load_and_verify_source_manifest(str(bad))
    with pytest.raises(ValueError, match="PCA component hash mismatch"):
        assert_pca_hashes_match_source({"pca16": {"components_sha256": "WRONG"}, "pca32": {"components_sha256": "b"}}, loaded)

    loaded["data_sha256"] = {"market": "m", "funding": None, "premium": "p"}
    assert_data_hashes_match_source(
        {"market": "m", "funding": None, "premium": "p"}, loaded
    )
    with pytest.raises(ValueError, match="source data hash mismatch for premium"):
        assert_data_hashes_match_source(
            {"market": "m", "funding": None, "premium": "different"}, loaded
        )


def test_initial_probe_materializes_only_fit_targets(monkeypatch):
    matrix = np.arange(20, dtype=np.float32).reshape(10, 2)
    targets = np.linspace(-0.2, 0.2, 10)
    fit = np.zeros(10, dtype=bool)
    fit[:6] = True
    calibration = fit_target_calibration(targets, fit)
    observed_lengths = []
    original = continual.make_class_labels

    def recording_labels(values, config):
        observed_lengths.append(len(values))
        return original(values, config)

    monkeypatch.setattr(continual, "make_class_labels", recording_labels)
    _fit_initial_probe(
        matrix,
        targets,
        fit,
        PROBE_SPECS[0],
        calibration,
    )

    assert observed_lengths == [int(fit.sum())]


def test_output_path_guards_include_source_manifest(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    out = tmp_path / "out.json"
    manifest = tmp_path / "manifest.json"

    strict_validate_paths(str(out), str(manifest), str(source))
    with pytest.raises(ValueError):
        strict_validate_paths(str(source), str(manifest), str(source))
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        strict_validate_paths(str(out), str(manifest), str(source))


def test_ready_positions_include_next_bar_and_hold():
    np.testing.assert_array_equal(label_ready_positions(np.array([100, 172])), [677, 749])
