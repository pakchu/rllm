from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_pposm_global_ratio_skip_source as diag
from training import preregister_pposm_global_ratio_skip_source as prereg


def _write_metrics(path: Path, times: pd.DatetimeIndex, values: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "create_time": times,
            "count_long_short_ratio": values,
        }
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False)


def test_prior_completed_join_is_exact_and_does_not_ffill() -> None:
    labels = pd.DataFrame(
        {
            "signal_time": pd.to_datetime(["2021-01-01 00:10", "2021-01-01 00:20"]),
            "target_skip_switch": [0, 1],
        }
    )
    source = pd.DataFrame(
        {
            "create_time": pd.to_datetime(["2021-01-01 00:05", "2021-01-01 00:10"]),
            "ratio_log_level": [1.0, 2.0],
            "ratio_dlog1": [0.1, 0.2],
            "ratio_z288": [0.0, 0.5],
        }
    )

    joined = diag.join_prior_completed_5m(labels, source)

    assert joined.loc[0, "ratio_available"] == 1.0
    assert joined.loc[0, "ratio_age_bars"] == 1.0
    assert joined.loc[1, "ratio_available"] == 0.0
    assert np.isnan(joined.loc[1, "ratio_log_level"])
    assert np.isnan(joined.loc[1, "ratio_age_bars"])


def test_loader_rejects_oos_or_wrong_anchor_count(monkeypatch) -> None:
    rows = []
    for i in range(diag.EXPECTED_ANCHORS):
        rows.append(
            {
                "split": "train",
                "target": "KEEP",
                "metadata": {
                    "candidate_action": "SKIP",
                    "identity": f"skip-{i}",
                    "base_identity": f"base-{i}",
                    "signal_position": i,
                    "signal_time": "2024-01-01 00:00:00" if i == 0 else "2023-01-01 00:00:00",
                },
            }
        )
        rows.append(
            {
                "split": "train",
                "target": "KEEP",
                "metadata": {
                    "candidate_action": "TP12",
                    "identity": f"tp12-{i}",
                    "base_identity": f"base-{i}",
                    "signal_position": i,
                    "signal_time": "2023-01-01 00:00:00",
                },
            }
        )
    monkeypatch.setattr(diag.lifecycle.frozen, "load_frozen_manifest", lambda path: ({}, object()))
    monkeypatch.setattr(diag.lifecycle, "load_train_context", lambda manifest, cfg: (None, None, None, None, None))
    monkeypatch.setattr(diag.lifecycle, "rows_from_train_context", lambda *args: rows)

    with pytest.raises(RuntimeError, match="strictly pre-2024"):
        diag.load_lifecycle_skip_labels(Path("manifest.json"))

    rows.pop()
    rows.pop()
    with pytest.raises(RuntimeError, match="expected 102"):
        diag.load_lifecycle_skip_labels(Path("manifest.json"))


def test_gate_enforces_auc_ci_bacc_year_and_null_thresholds() -> None:
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

    bad = diag.gate_for_model(
        model,
        {"lower95": 0.501},
        {**{name: 0.0 for name in diag.FEATURE_COLUMNS}, "ratio_z288": 0.051},
    )
    assert bad["pass"] is False
    assert bad["checks"]["max_feature_null_le_5pct"] is False

    short_model = {**model, "folds": model["folds"][:2]}
    bad = diag.gate_for_model(short_model, {"lower95": 0.501}, {name: 0.0 for name in diag.FEATURE_COLUMNS})
    assert bad["pass"] is False
    assert bad["checks"]["exactly_3_valid_folds"] is False

    model["folds"][1]["auc"] = 0.50
    bad = diag.gate_for_model(model, {"lower95": 0.501}, {name: 0.0 for name in diag.FEATURE_COLUMNS})
    assert bad["pass"] is False
    assert bad["checks"]["every_valid_year_auc_gt_0_50"] is False


def test_evaluate_builds_no_oos_labels_and_can_run_without_writing(monkeypatch, tmp_path) -> None:
    # Include 288 warmup bars so z288 is available for all joined anchors.
    source_times = pd.date_range("2019-12-31 00:00", periods=288, freq="5min")
    signal_times = pd.to_datetime(
        [
            "2021-01-01 00:05",
            "2021-01-01 00:10",
            "2022-01-01 00:05",
            "2022-01-01 00:10",
            "2023-01-01 00:05",
            "2023-01-01 00:10",
        ]
    )
    required = signal_times - pd.Timedelta(minutes=5)
    metrics_times = source_times.append(pd.DatetimeIndex(required))
    values = list(np.linspace(1.0, 2.0, len(metrics_times)))
    metrics = tmp_path / "metrics.csv.gz"
    _write_metrics(metrics, metrics_times, values)
    labels = pd.DataFrame(
        {
            "identity": [f"id-{i}" for i in range(len(signal_times))],
            "base_identity": [f"base-{i}" for i in range(len(signal_times))],
            "signal_position": list(range(len(signal_times))),
            "signal_time": signal_times,
            "target_skip_switch": [0, 1, 0, 1, 0, 1],
        }
    )
    labels.attrs["candidate_counts"] = {"SKIP": len(labels), "TP12": len(labels)}
    monkeypatch.setattr(diag, "load_lifecycle_skip_labels", lambda path: labels)
    monkeypatch.setattr(diag, "EXPECTED_ANCHORS", len(labels))

    summary = diag.evaluate(
        diag.Config(
            manifest=tmp_path / "manifest.json",
            metrics_csv=metrics,
            output=tmp_path / "out.json",
            preregistration=None,
            bootstrap_iterations=10,
            write_output=False,
            expected_metrics_sha256=diag.sha256_file(metrics),
            expected_pre2024_prefix_sha256=None,
        )
    )

    assert summary["boundary"]["oos_labels_built_or_read"] is False
    assert summary["anchors"]["last_signal_time"] < "2024-01-01"
    assert summary["features"]["max_null_fraction"] == 0.0
    assert not (tmp_path / "out.json").exists()


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


def test_hash_bindings_and_selection_prefer_passing_model() -> None:
    model_results = {
        "high_auc_fail": {"pooled": {"auc": 0.90, "balanced_accuracy": 0.90}},
        "lower_auc_pass": {"pooled": {"auc": 0.61, "balanced_accuracy": 0.56}},
    }
    ci = {"high_auc_fail": {"lower95": 0.80}, "lower_auc_pass": {"lower95": 0.51}}
    gates = {"high_auc_fail": {"pass": False}, "lower_auc_pass": {"pass": True}}
    assert diag.select_model(model_results, ci, gates) == "lower_auc_pass"

    frame = pd.DataFrame(
        {
            "identity": ["a"],
            "signal_time": [pd.Timestamp("2021-01-01 00:05")],
            "source_time_required": [pd.Timestamp("2021-01-01 00:00")],
            "create_time": [pd.Timestamp("2021-01-01 00:00")],
            "target_skip_switch": [1],
            **{name: [1.0] for name in diag.FEATURE_COLUMNS},
        }
    )
    assert len(diag.joined_feature_hash(frame)) == 64
    assert len(diag.oof_score_hash({"m": {"oof": [0.25]}}, frame)) == 64


def test_preregistration_binds_design_before_labels_or_metrics(tmp_path) -> None:
    value = prereg.build()
    assert value["evidence_boundary"] == {
        "lifecycle_labels_opened": 0,
        "oos_labels_opened": 0,
        "source_support_metrics_computed": False,
        "sft_or_rlvr_started": False,
    }
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    cfg = diag.Config(preregistration=path, write_output=False)
    assert diag.validate_preregistration(cfg)["manifest_hash"] == value["manifest_hash"]

    value["design"]["gates"]["pooled_auc"] = 0.59
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    value["manifest_hash"] = diag.sha256_canonical(core)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding drift"):
        diag.validate_preregistration(cfg)
