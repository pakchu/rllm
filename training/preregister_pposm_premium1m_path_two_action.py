"""Freeze a strict pre-2024 premium-path source test for both PPOSM actions."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from training import build_pposm_sft_rlvr_data as frozen

PROTOCOL = "pposm_premium1m_path_two_action_preregistration_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("results/pposm_premium1m_path_two_action_preregistration_2026-09-05.json")
DEFAULT_RESULT = Path("results/pposm_premium1m_path_two_action_source_support_2026-09-05.json")
LABELS = Path("data/pposm_residual_action_train_pre2024_2026-09-02.jsonl")
LABELS_SHA256 = "54bcddea310043b93ec0d770d7c9dd425e43c6aefc27b00fd98d82a9d355fbeb"
PPOSM_MANIFEST = Path("results/pullback_premium_overheat_state_machine_manifest_2026-07-15.json")
PPOSM_MANIFEST_SHA256 = "41d6424b8ca719ee3e3b058f6f5f93bc360905ee02be5b5d9671ff5349978159"
PREDECESSOR = Path("results/pposm_pre2024_intrahour_premium_action_source_2026-09-05.json")
PREDECESSOR_SHA256 = "4ec04ae8d8c1aab46dedb15696f868dd6ac1baa7d82a57954b92067ef167a888"
PREDECESSOR_RESULT_HASH = "a7ca41eef131a40711cd1b81a40680a79e23d17be14c0831e967f8381fd8a38b"
QUERY_START = "2020-01-01T00:00:00Z"
QUERY_END_EXCLUSIVE = "2024-01-01T00:00:00Z"
PREMIUM_QUERY = """
    SELECT ts, close_time, close
    FROM bars_binance_premium
    WHERE symbol = :symbol AND interval = '1m'
      AND ts >= :start AND ts < :end
      AND ts < '2024-01-01T00:00:00Z'
    ORDER BY ts
"""
FEATURE_COLUMNS = (
    "path_total_variation",
    "path_quadratic_variation",
    "signed_path_efficiency",
    "first_half_variation_share",
    "last_quarter_variation_share",
    "positive_occupation_centered",
    "level_zero_cross_rate",
    "increment_reversal_rate",
)
FOLDS = (
    {"name": "train_2020_test_2021", "train_years": [2020], "test_year": 2021},
    {"name": "train_through_2021_test_2022", "train_years": [2020, 2021], "test_year": 2022},
    {"name": "train_through_2022_test_2023", "train_years": [2020, 2021, 2022], "test_year": 2023},
)
MODEL = {
    "pipeline": "StandardScaler_then_LogisticRegression",
    "penalty": "l2",
    "C": 0.25,
    "solver": "liblinear",
    "class_weight": "balanced",
    "max_iter": 1000,
    "threshold": 0.5,
    "random_state": 20260906,
}
COVERAGE_GATES = {"overall_min": 0.95, "per_year_min": 0.90, "test_fold_min": 10}
SUPPORT_GATES = {
    "pooled_auc_min": 0.60,
    "simultaneous_auc_lower95_strict_min": 0.50,
    "balanced_accuracy_min": 0.55,
    "each_year_auc_strict_min": 0.50,
}
BOOTSTRAP = {
    "method": "year-stratified paired stationary bootstrap over base anchors",
    "iterations": 10_000,
    "expected_block_length_anchors": 8,
    "random_seed": 20260906,
    "simultaneous_bound": "observed_auc[action] minus q95(max_action(observed_auc-bootstrap_auc))",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    return {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__}


def code_hashes() -> dict[str, str]:
    return {
        "preregistration_module": sha256_file(__file__),
        "evaluator_module": sha256_file(PROJECT_ROOT / "training/evaluate_pposm_premium1m_path_two_action.py"),
        "frozen_data_module": sha256_file(Path(frozen.__file__).resolve()),
        "pposm_module": sha256_file(Path(frozen.pposm.__file__).resolve()),
    }


def _bound_artifact(path: Path, expected_sha: str, *, result_hash: str | None = None) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected_sha:
        raise RuntimeError(f"bound artifact changed: {path}")
    receipt: dict[str, Any] = {"path": str(path), "sha256": observed}
    if result_hash is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("result_hash") != result_hash:
            raise RuntimeError(f"bound result_hash changed: {path}")
        receipt["result_hash"] = result_hash
    return receipt


def build_preregistration() -> dict[str, Any]:
    manifest = json.loads(PPOSM_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("freeze_hash") != "1360cf620b8afcead476e2aa0c1394e1419b70c42cac7e03dcc91912ab60c81f":
        raise RuntimeError("frozen PPOSM manifest identity changed")
    core = {
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "test whether non-hourly one-minute premium path shape separates both original pairwise PPOSM residual actions using pre-2024 only",
        "why_successor": {
            "predecessor": _bound_artifact(PREDECESSOR, PREDECESSOR_SHA256, result_hash=PREDECESSOR_RESULT_HASH),
            "methodological_delta": [
                "eight path-only 60-minute features fixed independently of predecessor scores",
                "exact close_time causality",
                "label-maturity purged expanding folds",
                "paired temporal simultaneous confidence bound",
            ],
            "gate_weakening": False,
        },
        "source": {
            "query": PREMIUM_QUERY,
            "query_sha256": sha256_bytes(PREMIUM_QUERY.strip().encode()),
            "window": [QUERY_START, QUERY_END_EXCLUSIVE],
            "symbol": "BTCUSDT",
            "interval": "1m",
            "selected_columns": ["ts", "close_time", "close"],
            "coerce_float": False,
            "anchor_path": "exact 60 rows with signal_time-60min <= ts < signal_time; no fill/interpolation/nearest",
            "close_time_rule": "ts <= close_time < ts+1min and close_time <= signal_time",
        },
        "labels": {
            **_bound_artifact(LABELS, LABELS_SHA256),
            "pair_rows": 924,
            "base_anchors": 462,
            "candidate_actions": ["SKIP", "TP12"],
            "default_action": "TP4",
            "maturity": "maximum TP4/TP12 executable exit bar completion timestamp (exit row open plus 5min); purge across every fold boundary",
            "pposm_manifest": _bound_artifact(PPOSM_MANIFEST, PPOSM_MANIFEST_SHA256),
        },
        "features": list(FEATURE_COLUMNS),
        "model": MODEL,
        "folds": list(FOLDS),
        "coverage_gates": COVERAGE_GATES,
        "support_gates": SUPPORT_GATES,
        "bootstrap": BOOTSTRAP,
        "selection": "none; exactly one feature vector, model, threshold, fold design, and conjunctive two-action gate",
        "code_hashes": code_hashes(),
        "runtime_versions": runtime_versions(),
        "terminal_rule": "any source/coverage/fold/action failure stops before SFT, RLVR, and 2024+ OOS; no same-source repair",
        "evidence_boundary": {
            "database_rows_opened": 0,
            "pre2024_pair_labels_opened_by_this_preregistration": 0,
            "source_support_metrics_computed": False,
            "oos_labels_opened": 0,
            "fresh_outcomes_opened": 0,
            "sft_or_rlvr_started": False,
        },
    }
    payload = dict(core)
    payload["preregistration_hash"] = sha256_bytes(canonical_json({k: v for k, v in core.items() if k != "created_at"}).encode())
    return payload


def write_preregistration(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_preregistration()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("preregistration_hash") != payload["preregistration_hash"]:
            raise RuntimeError("refusing to overwrite different premium1m path preregistration")
        return existing
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    value = write_preregistration()
    print(json.dumps({"protocol": PROTOCOL, "preregistration_hash": value["preregistration_hash"], "database_rows_opened": 0}, indent=2))
