from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_pposm_premium_price_divergence_two_action as evaluate
from training import preregister_pposm_premium_price_divergence_two_action as prereg


def test_divergence_features_are_fixed_and_finite() -> None:
    path = np.linspace(0.0, 1.0, 60) ** 2
    premium = path
    market = np.exp(10.0 + 0.1 * path)
    features = evaluate.divergence_features(premium, market)
    assert tuple(features) == prereg.FEATURE_COLUMNS
    assert np.isfinite(list(features.values())).all()
    assert features["increment_correlation"] > 0.99
    assert abs(features["path_efficiency_difference"]) < 1e-12


def test_constant_paths_use_neutral_denominators() -> None:
    features = evaluate.divergence_features(np.zeros(60), np.ones(60) * 100.0)
    assert features["increment_correlation"] == 0.0
    assert features["premium_beta_on_price"] == 0.0
    assert features["path_efficiency_difference"] == 0.0


def test_attach_requires_both_exact_paths_and_premium_close_time() -> None:
    decision = pd.Timestamp("2023-01-02 01:00")
    labels = pd.DataFrame({"base_identity": ["a"], "signal_position": [1], "decision_time": [decision], "maturity_time": [decision + pd.Timedelta(hours=1)], "SKIP": [0], "TP12": [1]})
    ts = pd.date_range(decision - pd.Timedelta(minutes=60), decision - pd.Timedelta(minutes=1), freq="1min")
    premium = pd.DataFrame({"ts": ts, "close_time": ts + pd.Timedelta(seconds=59), "close": np.linspace(-1, 1, 60)})
    market = pd.DataFrame({"ts": ts, "close": np.linspace(100, 101, 60)})
    frame, coverage = evaluate.attach_features(labels, premium, market)
    assert coverage["passed"] and bool(frame.loc[0, "source_valid"])
    _, missing = evaluate.attach_features(labels, premium, market.iloc[:-1])
    assert not missing["passed"]
    premium.loc[0, "close_time"] = decision + pd.Timedelta(minutes=1)
    frame, _ = evaluate.attach_features(labels, premium, market)
    assert not bool(frame.loc[0, "source_valid"])


def test_preregistration_is_single_design_and_zero_evidence() -> None:
    payload = prereg.build_preregistration()
    assert payload["features"] == list(prereg.FEATURE_COLUMNS)
    assert payload["evidence_boundary"]["database_rows_opened"] == 0
    assert payload["evidence_boundary"]["oos_labels_opened"] == 0
    assert payload["why_final_source"]["gate_weakening"] is False


def test_preregistration_tamper_is_rejected(tmp_path: Path) -> None:
    payload = prereg.build_preregistration()
    payload["features"] = ["posthoc"]
    path = tmp_path / "bad.json"; path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="drift"):
        evaluate.validate_preregistration(path)


def test_queries_are_strictly_pre2024_and_no_model_training_path() -> None:
    assert "ts < '2024-01-01T00:00:00Z'" in prereg.MARKET_QUERY
    source = Path("training/evaluate_pposm_premium_price_divergence_two_action.py").read_text()
    assert "SFTTrainer" not in source and "PPOTrainer" not in source
