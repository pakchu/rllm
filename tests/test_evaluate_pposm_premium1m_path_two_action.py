from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_pposm_premium1m_path_two_action as evaluate
from training import preregister_pposm_premium1m_path_two_action as prereg


def test_path_features_are_fixed_and_hand_checkable() -> None:
    close = np.arange(60, dtype=float)
    features = evaluate.path_features(close)
    assert tuple(features) == prereg.FEATURE_COLUMNS
    assert features["path_total_variation"] == 59.0
    assert features["path_quadratic_variation"] == np.sqrt(59.0)
    assert features["signed_path_efficiency"] == 1.0
    assert features["first_half_variation_share"] == 29.0 / 59.0
    assert features["last_quarter_variation_share"] == 15.0 / 59.0
    assert features["increment_reversal_rate"] == 0.0


def test_attach_features_requires_exact_prior_sixty_and_causal_close_time() -> None:
    decision = pd.Timestamp("2023-01-02 01:00")
    labels = pd.DataFrame({"base_identity": ["a"], "signal_position": [1], "decision_time": [decision], "maturity_time": [decision + pd.Timedelta(hours=1)], "SKIP": [0], "TP12": [1]})
    ts = pd.date_range(decision - pd.Timedelta(minutes=60), decision - pd.Timedelta(minutes=1), freq="1min")
    premium = pd.DataFrame({"ts": ts, "close_time": ts + pd.Timedelta(seconds=59), "close": np.linspace(-1, 1, 60)})
    frame, coverage = evaluate.attach_features(labels, premium)
    assert coverage["passed"] and bool(frame.loc[0, "source_valid"])
    _, missing = evaluate.attach_features(labels, premium.iloc[:-1])
    assert not missing["passed"]
    bad = premium.copy(); bad.loc[0, "close_time"] = decision + pd.Timedelta(minutes=1)
    frame, _ = evaluate.attach_features(labels, bad)
    assert not bool(frame.loc[0, "source_valid"])


def test_fold_masks_purge_maturity_across_boundary() -> None:
    frame = pd.DataFrame({"source_valid": [True, True, True], "decision_time": pd.to_datetime(["2020-12-30", "2020-12-31", "2021-06-01"]), "maturity_time": pd.to_datetime(["2020-12-31", "2021-01-02", "2021-06-03"])})
    train, test = evaluate.fold_masks(frame, 2021)
    assert train.tolist() == [True, False, False]
    assert test.tolist() == [False, False, True]


def test_stationary_bootstrap_is_year_stratified_and_deterministic() -> None:
    years = np.array([2021] * 5 + [2022] * 7)
    first = evaluate.stationary_indices(years, rng=np.random.default_rng(7), mean_block=3)
    second = evaluate.stationary_indices(years, rng=np.random.default_rng(7), mean_block=3)
    assert np.array_equal(first, second)
    assert len(first) == len(years)
    assert (years[first[:5]] == 2021).all() and (years[first[5:]] == 2022).all()


def test_paired_simultaneous_bound_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(prereg.BOOTSTRAP, "iterations", 100)
    frame = pd.DataFrame({"decision_time": pd.to_datetime([*(["2021-06-01"] * 100), *(["2022-06-01"] * 100)]), "SKIP": [0, 1] * 100, "TP12": [1, 0] * 100, "score_SKIP": np.tile([0.2, 0.8], 100), "score_TP12": np.tile([0.8, 0.2], 100)})
    first = evaluate.paired_simultaneous_bounds(frame)
    second = evaluate.paired_simultaneous_bounds(frame)
    assert first == second
    assert first["simultaneous_lower95"]["SKIP"] > 0.9


def test_preregistration_binds_zero_evidence_and_immutable_labels() -> None:
    payload = prereg.build_preregistration()
    assert payload["labels"]["sha256"] == prereg.LABELS_SHA256
    assert payload["features"] == list(prereg.FEATURE_COLUMNS)
    assert payload["evidence_boundary"]["database_rows_opened"] == 0
    assert payload["evidence_boundary"]["oos_labels_opened"] == 0
    assert payload["selection"].startswith("none")


def test_query_and_source_are_pre2024_only() -> None:
    assert "ts < '2024-01-01T00:00:00Z'" in prereg.PREMIUM_QUERY
    source = Path("training/evaluate_pposm_premium1m_path_two_action.py").read_text()
    assert "oos_opened\": False" in source
    assert "SFTTrainer" not in source and "PPOTrainer" not in source


def test_validate_preregistration_rejects_tamper(tmp_path: Path) -> None:
    payload = prereg.build_preregistration()
    payload["model"]["C"] = 9.0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="drift"):
        evaluate.validate_preregistration(path)
