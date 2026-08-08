import numpy as np
import pandas as pd

from training import train_high_volatility_microstructure_ridge_relay as train


def test_rank_excludes_current_and_freezes_preregistered_history():
    values = pd.Series(np.arange(181, dtype=float))
    ranks = train.strict_prior_midrank(values)
    assert ranks.iloc[179] != ranks.iloc[179]
    assert ranks.iloc[180] == 1.0
    assert train.strict_prior_midrank.__defaults__ == (270, 180)


def test_feature_order_and_model_hyperparameters_are_frozen():
    assert train.FEATURES == tuple(train.prereg.build()["feature_contract"]["ordered_features"])
    source = open(train.__file__).read()
    assert 'Ridge(alpha=10.0, fit_intercept=True, solver="svd")' in source
    assert "0.80, method=\"linear\"" in source
    assert train.LABEL_END == pd.Timestamp("2023-07-01T00:00:00Z")


def test_fit_model_uses_only_supplied_pretraining_labels():
    count = 240
    panel = pd.DataFrame({"decision_time": pd.date_range("2021-01-01", periods=count, freq="8h", tz="UTC"), "source_valid": True})
    for position, feature in enumerate(train.FEATURES):
        panel[feature] = np.sin(np.arange(count) * (position + 1) / 31) + position / 10
    panel["variation_rank"] = np.linspace(0.01, 0.99, count)
    labels = pd.DataFrame({"decision_time": panel.decision_time, "entry_time": panel.decision_time + pd.Timedelta(minutes=5), "exit_time": panel.decision_time + pd.Timedelta(hours=6, minutes=5), "label": 0.01 * panel.normalized_full_return - 0.005 * panel.late_taker_imbalance})
    rows, model = train.fit_model(panel, labels)
    assert len(rows) == count
    assert model["training_rows"] == count
    assert model["prediction_strength_threshold"] > 0
    assert len(model["coefficient"]) == len(train.FEATURES)
