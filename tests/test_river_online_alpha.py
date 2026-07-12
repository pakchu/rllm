import numpy as np

from training.search_river_online_alpha import (
    causal_rolling_thresholds,
    delayed_online_predictions,
    dynamic_policy_masks,
    label_ready_positions,
    selection_window_signal_hash,
    validate_output_paths,
)


class RecordingRegressor:
    def __init__(self):
        self.events = []
        self.learned_targets = []

    def predict_one(self, x):
        self.events.append(("predict", x["feature"], tuple(self.learned_targets)))
        return float(sum(self.learned_targets))

    def learn_one(self, x, y):
        self.events.append(("learn", x["feature"], y))
        self.learned_targets.append(y)
        return self


def test_label_ready_position_includes_delayed_entry_and_full_horizon():
    positions = np.array([100, 172])

    ready = label_ready_positions(positions, entry_delay_bars=1, hold_bars=576)

    np.testing.assert_array_equal(ready, [677, 749])


def test_delayed_stream_predicts_before_queuing_and_learns_only_matured_labels():
    model = RecordingRegressor()
    scores, diagnostics = delayed_online_predictions(
        model,
        feature_names=["feature"],
        matrix=np.array([[0.0], [1.0], [2.0]]),
        targets=np.array([0.10, 0.20, 0.30]),
        signal_positions=np.array([0, 2, 4]),
        ready_positions=np.array([3, 5, 7]),
        min_completed_updates=0,
        target_clip=1.0,
    )

    assert model.events == [
        ("predict", 0.0, ()),
        ("predict", 1.0, ()),
        ("learn", 0.0, 0.10),
        ("predict", 2.0, (0.10,)),
    ]
    np.testing.assert_allclose(scores, [0.0, 0.0, 0.10])
    assert diagnostics["completed_updates_before_last_prediction"] == 1
    assert diagnostics["pending_samples_after_last_prediction"] == 2
    assert diagnostics["last_learned_ready_position"] == 3


def test_causal_threshold_excludes_current_score():
    scores = np.array([0.0, 1.0, 100.0, -100.0])
    changed_current = np.array([0.0, 1.0, -999.0, -100.0])

    low, high = causal_rolling_thresholds(
        scores, window=3, quantile=0.75, min_periods=2
    )
    changed_low, changed_high = causal_rolling_thresholds(
        changed_current, window=3, quantile=0.75, min_periods=2
    )

    assert high[2] == 0.75
    assert low[2] == 0.25
    assert changed_high[2] == high[2]
    assert changed_low[2] == low[2]
    assert high[3] == 50.5
    assert changed_high[3] != high[3]


def test_dynamic_policy_enters_only_at_anchor_positions():
    scores = np.array([-2.0, 0.0, 2.0])
    positions = np.array([3, 7, 11])
    low = np.array([-1.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0])

    long_active, short_active = dynamic_policy_masks(
        scores,
        positions,
        15,
        side_policy="both",
        low_thresholds=low,
        high_thresholds=high,
    )

    np.testing.assert_array_equal(np.flatnonzero(long_active), [11])
    np.testing.assert_array_equal(np.flatnonzero(short_active), [3])


def test_selection_hash_ignores_signals_after_selection_window():
    positions = np.array([2, 5, 8, 11])
    selection_mask = np.array([True, True, False, False])
    long_a = np.zeros(15, dtype=bool)
    long_b = np.zeros(15, dtype=bool)
    short_a = np.zeros(15, dtype=bool)
    short_b = np.zeros(15, dtype=bool)
    long_a[[2, 8]] = True
    long_b[[2, 11]] = True
    short_a[[5, 11]] = True
    short_b[[5, 8]] = True

    hash_a = selection_window_signal_hash(
        long_a,
        short_a,
        positions=positions,
        selection_mask=selection_mask,
    )
    hash_b = selection_window_signal_hash(
        long_b,
        short_b,
        positions=positions,
        selection_mask=selection_mask,
    )

    assert hash_a == hash_b


def test_final_output_cannot_overwrite_frozen_manifest(tmp_path):
    shared = tmp_path / "shared.json"

    try:
        validate_output_paths(str(shared), str(shared))
    except ValueError as error:
        assert "must be different" in str(error)
    else:
        raise AssertionError("identical output paths must be rejected")
