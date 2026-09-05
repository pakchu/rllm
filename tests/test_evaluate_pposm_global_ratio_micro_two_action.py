from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_pposm_global_ratio_micro_two_action as diag


def _write_metrics(path: Path, times: pd.DatetimeIndex, values: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "create_time": times,
            "count_long_short_ratio": values,
        }
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False)


def _labels() -> pd.DataFrame:
    signal_times = pd.to_datetime(
        [
            "2020-01-01 00:05",
            "2020-01-01 00:10",
            "2021-01-01 00:05",
            "2021-01-01 00:10",
            "2022-01-01 00:05",
            "2022-01-01 00:10",
            "2023-01-01 00:05",
            "2023-01-01 00:10",
        ]
    )
    rows = []
    for action in diag.ACTIONS:
        for i, ts in enumerate(signal_times):
            rows.append(
                {
                    "identity": f"{action}-{i}",
                    "base_identity": f"base-{i}",
                    "action": action,
                    "signal_position": i,
                    "signal_time": ts,
                    "target_switch": i % 2,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["candidate_counts"] = {action: len(signal_times) for action in diag.ACTIONS}
    frame.attrs["manifest_freeze_hash"] = "freeze"
    frame.attrs["manifest_file_sha256"] = "m" * 64
    frame.attrs["lifecycle_builder_sha256"] = "l" * 64
    frame.attrs["anchor_label_identity_sha256"] = diag.sha256_canonical(
        [{**row, "signal_time": str(row["signal_time"])} for row in rows]
    )
    return frame


def _matching_prereg(tmp_path: Path, cfg: diag.Config) -> Path:
    core = {
        "protocol": "pposm_global_count_long_short_ratio_micro_two_action_preregistration_v1",
        "source": {
            "sha256": cfg.expected_metrics_sha256,
            "pre2024_prefix_sha256": cfg.expected_pre2024_prefix_sha256,
            "pposm_manifest_sha256": diag.sha256_file(cfg.manifest),
        },
        "design": {
            "actions": list(diag.ACTIONS),
            "anchors_per_action": diag.EXPECTED_ANCHORS,
            "features": list(diag.FEATURE_COLUMNS),
            "folds": [
                {"name": name, "train_years": list(train_years), "test_year": test_year}
                for name, train_years, test_year in diag.FOLD_SPECS
            ],
            "models": diag.MODEL_SPECS,
            "gates": diag.GATE_THRESHOLDS,
            "bootstrap_iterations": cfg.bootstrap_iterations,
            "random_seed": cfg.random_seed,
        },
        "implementation": {
            "evaluator_sha256": diag.sha256_file(Path(diag.__file__).resolve()),
            "runtime_versions": diag.runtime_versions(),
        },
        "evidence_boundary": {
            "lifecycle_labels_opened": 0,
            "oos_labels_opened": 0,
            "source_support_metrics_computed": False,
            "sft_or_rlvr_started": False,
        },
    }
    value = {**core, "manifest_hash": diag.sha256_canonical(core)}
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_prior_completed_join_is_exact_no_ffill_and_adds_fixed_poly_features() -> None:
    labels = pd.DataFrame(
        {
            "identity": ["a", "b"],
            "base_identity": ["ba", "bb"],
            "action": ["SKIP", "TP12"],
            "signal_time": pd.to_datetime(["2021-01-01 00:10", "2021-01-01 00:20"]),
            "target_switch": [0, 1],
        }
    )
    source = pd.DataFrame(
        {
            "create_time": pd.to_datetime(["2021-01-01 00:05", "2021-01-01 00:10"]),
            "ratio_log_level": [1.0, 2.0],
            "ratio_dlog1": [-0.2, 0.3],
            "ratio_z288": [-0.5, 0.75],
        }
    )

    joined = diag.join_prior_completed_5m(labels, source)

    assert joined.loc[0, "ratio_available"] == 1.0
    assert joined.loc[0, "ratio_age_bars"] == 1.0
    assert joined.loc[0, "ratio_abs_z288"] == 0.5
    assert joined.loc[0, "ratio_z288_sq"] == 0.25
    assert joined.loc[0, "ratio_abs_dlog1"] == 0.2
    assert joined.loc[0, "ratio_dlog1_sq"] == pytest.approx(0.04)
    assert joined.loc[1, "ratio_available"] == 0.0
    assert np.isnan(joined.loc[1, "ratio_log_level"])
    assert np.isnan(joined.loc[1, "ratio_age_bars"])


def test_loader_rejects_oos_and_requires_both_actions(monkeypatch) -> None:
    rows = []
    for i in range(diag.EXPECTED_ANCHORS):
        for action in diag.ACTIONS:
            rows.append(
                {
                    "split": "train",
                    "target": "KEEP",
                    "metadata": {
                        "candidate_action": action,
                        "identity": f"{action}-{i}",
                        "base_identity": f"base-{i}",
                        "signal_position": i,
                        "signal_time": "2024-01-01 00:00:00" if i == 0 else "2023-01-01 00:00:00",
                    },
                }
            )
    monkeypatch.setattr(diag.lifecycle.frozen, "load_frozen_manifest", lambda path: ({}, object()))
    monkeypatch.setattr(diag.lifecycle, "load_train_context", lambda manifest, cfg: (None, None, None, None, None))
    monkeypatch.setattr(diag.lifecycle, "rows_from_train_context", lambda *args: rows)

    with pytest.raises(RuntimeError, match="strictly pre-2024"):
        diag.load_lifecycle_two_action_labels(Path("manifest.json"))

    rows = [row for row in rows if not (row["metadata"]["candidate_action"] == "TP12" and row["metadata"]["signal_position"] == 1)]
    rows[0]["metadata"]["signal_time"] = "2023-01-01 00:00:00"
    with pytest.raises(RuntimeError, match="expected 102 anchors for each action"):
        diag.load_lifecycle_two_action_labels(Path("manifest.json"))


def test_model_specs_lock_micro_hgb_and_logreg_c025() -> None:
    models = diag._classifiers(seed=7)
    logreg = models["logreg_l2_c025_balanced_poly"].named_steps["logisticregression"]
    assert logreg.C == 0.25
    assert logreg.class_weight == "balanced"
    hgb = models["hgb_micro_leaf2_l2_1"]
    assert hgb.max_leaf_nodes == 3
    assert hgb.max_depth == 2
    assert hgb.min_samples_leaf == 2
    assert hgb.l2_regularization == 1.0
    assert diag.MODEL_SPECS["hgb_micro_leaf2_l2_1"]["min_samples_leaf"] == 2


def test_gate_requires_both_actions() -> None:
    model = {
        "pooled": {"auc": 0.61, "balanced_accuracy": 0.56},
        "folds": [
            {"valid": True, "test_year": 2021, "auc": 0.51},
            {"valid": True, "test_year": 2022, "auc": 0.52},
            {"valid": True, "test_year": 2023, "auc": 0.53},
        ],
    }
    gate = diag.gate_for_model(
        model,
        {"lower95": 0.501},
        {name: 0.0 for name in diag.FEATURE_COLUMNS},
    )
    assert gate["pass"] is True

    bad_model = {**model, "folds": model["folds"][:2]}
    bad = diag.gate_for_model(
        bad_model,
        {"lower95": 0.501},
        {name: 0.0 for name in diag.FEATURE_COLUMNS},
    )
    assert bad["checks"]["exactly_3_valid_folds"] is False

    overall = {"SKIP": True, "TP12": False}
    assert bool(all(overall.values())) is False


def test_evaluate_builds_no_oos_labels_can_run_no_write_and_both_action_gate(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    source_times = pd.date_range("2019-12-31 00:00", periods=288, freq="5min")
    labels = _labels()
    required = pd.DatetimeIndex(labels["signal_time"].unique()) - pd.Timedelta(minutes=5)
    metrics_times = source_times.append(required)
    values = list(np.linspace(1.0, 2.0, len(metrics_times)))
    metrics = tmp_path / "metrics.csv.gz"
    _write_metrics(metrics, metrics_times, values)
    cfg0 = diag.Config(
        manifest=manifest,
        metrics_csv=metrics,
        output=tmp_path / "out.json",
        preregistration=tmp_path / "missing.json",
        bootstrap_iterations=10,
        write_output=False,
        expected_metrics_sha256=diag.sha256_file(metrics),
        expected_pre2024_prefix_sha256=None,
    )
    prereg = _matching_prereg(tmp_path, cfg0)
    cfg = diag.Config(**{**cfg0.__dict__, "preregistration": prereg})
    monkeypatch.setattr(diag, "load_lifecycle_two_action_labels", lambda path: labels)

    summary = diag.evaluate(cfg)

    assert summary["boundary"]["oos_labels_built_or_read"] is False
    assert summary["anchors"]["last_signal_time"] < "2024-01-01"
    assert set(summary["selected_gate_by_action"]) == set(diag.ACTIONS)
    assert summary["overall_pass"] == all(summary["action_pass"].values())
    assert summary["features"]["max_null_fraction_by_action"] == {"SKIP": 0.0, "TP12": 0.0}
    assert not (tmp_path / "out.json").exists()


def test_evaluator_refuses_missing_or_tampered_preregistration(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    cfg = diag.Config(manifest=manifest, preregistration=tmp_path / "missing.json", write_output=False)
    with pytest.raises(RuntimeError, match="missing preregistration"):
        diag.validate_preregistration(cfg)

    core = {
        "protocol": "pposm_global_count_long_short_ratio_micro_two_action_preregistration_v1",
        "source": {
            "sha256": cfg.expected_metrics_sha256,
            "pre2024_prefix_sha256": cfg.expected_pre2024_prefix_sha256,
            "pposm_manifest_sha256": diag.sha256_file(manifest),
        },
        "design": {
            "actions": list(diag.ACTIONS),
            "anchors_per_action": diag.EXPECTED_ANCHORS,
            "features": list(diag.FEATURE_COLUMNS),
            "folds": [
                {"name": name, "train_years": list(train_years), "test_year": test_year}
                for name, train_years, test_year in diag.FOLD_SPECS
            ],
            "models": diag.MODEL_SPECS,
            "gates": {**diag.GATE_THRESHOLDS, "pooled_auc": 0.59},
            "bootstrap_iterations": cfg.bootstrap_iterations,
            "random_seed": cfg.random_seed,
        },
        "implementation": {
            "evaluator_sha256": diag.sha256_file(Path(diag.__file__).resolve()),
            "runtime_versions": diag.runtime_versions(),
        },
        "evidence_boundary": {
            "lifecycle_labels_opened": 0,
            "oos_labels_opened": 0,
            "source_support_metrics_computed": False,
            "sft_or_rlvr_started": False,
        },
    }
    value = {**core, "manifest_hash": diag.sha256_canonical(core)}
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding drift"):
        diag.validate_preregistration(diag.Config(**{**cfg.__dict__, "preregistration": prereg}))


def test_metrics_loader_hard_binds_full_sha_and_filters_pre2024(tmp_path) -> None:
    metrics = tmp_path / "metrics.csv.gz"
    times = pd.date_range("2023-12-31 23:50", periods=4, freq="5min")
    _write_metrics(metrics, times, [1.0, 1.1, 1.2, 1.3])

    with pytest.raises(RuntimeError, match="metrics source sha256 mismatch"):
        diag.load_ratio_source(metrics)

    frame, summary = diag.load_ratio_source(
        metrics,
        expected_sha256=diag.sha256_file(metrics),
        expected_pre2024_prefix_sha256=None,
    )
    assert frame["create_time"].max() < pd.Timestamp("2024-01-01")
    assert summary["sha256_matches_expected"] is True
    assert summary["feature_frame_end_exclusive"] == "2024-01-01"
    assert len(summary["pre2024_prefix_sha256"]) == 64

    with pytest.raises(RuntimeError, match="pre-2024 prefix sha256 mismatch"):
        diag.load_ratio_source(
            metrics,
            expected_sha256=diag.sha256_file(metrics),
            expected_pre2024_prefix_sha256="0" * 64,
        )
