import numpy as np

from training.economic_pressure_learnability_sweep import fit_softmax, metrics, majority_accuracy, LABELS


def test_majority_accuracy_counts_largest_class():
    assert majority_accuracy(["A", "A", "B"]) == 2 / 3


def test_softmax_fits_simple_separable_data():
    x = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
    y = np.array([0, 0, 1, 1])
    w = fit_softmax(x, y, num_classes=2, lr=0.5, l2=0.0, epochs=200)
    pred = np.argmax(x @ w, axis=1)
    assert (pred == y).mean() >= 0.75


def test_metrics_reports_edge_over_majority():
    labels = [LABELS[0], LABELS[1], LABELS[1]]
    pred = np.array([0, 1, 0])
    m = metrics(labels, pred)
    assert m["accuracy"] == 2 / 3
    assert m["majority_baseline"] == 2 / 3
    assert m["edge_over_majority"] == 0
