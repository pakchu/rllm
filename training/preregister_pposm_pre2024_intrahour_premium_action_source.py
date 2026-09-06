"""Preregister pre-2024 intrahour-premium support for PPOSM residual actions.

This protocol may read only pre-2024 BTCUSDT Binance premium-index one-minute
source paths plus the already-frozen pre-2024 PPOSM counterfactual labels.  It
must not read 2024+ OOS labels, fresh-forward outcomes, execution prices beyond
pre-2024 label construction, or any trained adapter outputs.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "pposm_pre2024_intrahour_premium_action_source_v1"
DEFAULT_OUTPUT = Path("results/pposm_pre2024_intrahour_premium_action_source_preregistration_2026-09-05.json")
BUILDER_PATH = PROJECT_ROOT / "training" / "build_pposm_pre2024_intrahour_premium_action_source.py"
THIS_PATH = Path(__file__).resolve()

QUERY_START = "2020-01-01T00:00:00Z"
QUERY_END_EXCLUSIVE = "2024-01-01T00:00:00Z"
MIN_FEATURE_LOOKBACK_MINUTES = 1440
FEATURE_WINDOWS_MINUTES = [60, 240, 480, 1440]
VALIDATION_YEARS = [2021, 2022, 2023]

PREMIUM_QUERY = """
    SELECT ts AS date, open, high, low, close
    FROM bars_binance_premium
    WHERE symbol = :symbol AND interval = '1m' AND ts >= :start AND ts < :end
    ORDER BY ts
"""

ACCEPTANCE = {
    "split_policy": "pre-2024 expanding-forward years only; validate 2021, 2022, 2023 when both classes are present and at least 80 earlier signals exist",
    "candidate_actions": ["SKIP", "TP12"],
    "default_action": "TP4",
    "minimum_training_signals_before_fold": 80,
    "minimum_validation_signals_per_fold": 10,
    "pooled_auc_min_each_candidate": 0.60,
    "bootstrap_lower95_auc_min_each_candidate": 0.50,
    "balanced_accuracy_min_each_candidate": 0.55,
    "consistent_year_direction": "every evaluated candidate-year AUC must be >= 0.50",
    "advance_if_all_pass": "only then open a separate preregistered historical OOS/residual-router check",
}

FORBIDDEN = [
    "2024_or_later_counterfactual_labels",
    "fresh_forward_lifecycle_or_return_outcomes",
    "gross9_future_rows",
    "trained_adapter_scores",
    "post_entry_outcomes_in_features",
    "threshold_tuning_after_source_results",
]

FEATURES = {
    "source": "BTCUSDT bars_binance_premium one-minute OHLC paths ending strictly before each frozen PPOSM decision timestamp",
    "windows_minutes": FEATURE_WINDOWS_MINUTES,
    "per_window_features": [
        "close_mean", "close_std", "close_min", "close_max", "close_range",
        "close_last_minus_first", "close_second_half_minus_first_half",
        "close_abs_diff_sum", "positive_close_share", "negative_close_share",
        "end_zscore", "ohlc_high_low_range", "upper_wick_mean", "lower_wick_mean",
    ],
    "normalization": "fit mean/std on each expanding training fold only",
    "model": "deterministic balanced L2 logistic regression, one binary model per candidate action",
    "feature_selection": "none; all preregistered finite features are used",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash(obj: Any) -> str:
    try:
        return sha256_bytes(inspect.getsource(obj).encode("utf-8"))
    except (OSError, TypeError):
        return "unavailable"


def build() -> dict[str, Any]:
    builder_hash = sha256_path(BUILDER_PATH)
    prereg = {
        "policy_id": POLICY_ID,
        "created_at": "2026-09-05T00:00:00Z",
        "purpose": "test whether materially new pre-2024 intrahour premium paths can explain frozen PPOSM residual action labels before any OOS reopen",
        "source_contract": {
            "premium_table": "bars_binance_premium",
            "symbol": "BTCUSDT",
            "interval": "1m",
            "query_start": QUERY_START,
            "query_end_exclusive": QUERY_END_EXCLUSIVE,
            "required_exact_grid": True,
            "required_finite_columns": ["open", "high", "low", "close"],
            "minimum_feature_lookback_minutes": MIN_FEATURE_LOOKBACK_MINUTES,
            "query_rows_opened_by_this_preregistration": 0,
            "forbidden_sources": FORBIDDEN,
            "premium_query": PREMIUM_QUERY,
            "premium_query_sha256": sha256_bytes(PREMIUM_QUERY.encode("utf-8")),
        },
        "label_contract": {
            "labels": "pre-2024 frozen PPOSM counterfactual utility only; SWITCH iff candidate utility exceeds TP4 utility",
            "candidate_actions": ACCEPTANCE["candidate_actions"],
            "default_action": ACCEPTANCE["default_action"],
            "oos_labels_opened": False,
            "fresh_outcomes_opened": False,
        },
        "features": FEATURES,
        "validation": ACCEPTANCE,
        "code_hashes": {
            "preregistration_module": sha256_path(THIS_PATH),
            "builder_module": builder_hash,
            "build_function": source_hash(build),
        },
        "terminal_semantics": "fail closed unless every acceptance check passes; no OOS repair or model promotion from a failed source-support result",
    }
    prereg["preregistration_hash"] = sha256_bytes(canonical_json({k: v for k, v in prereg.items() if k != "preregistration_hash"}).encode("utf-8"))
    return prereg


def main() -> None:
    out = build()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    if DEFAULT_OUTPUT.exists():
        existing = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        if existing.get("preregistration_hash") != out["preregistration_hash"]:
            raise RuntimeError("refusing to overwrite different PPOSM intrahour premium preregistration")
    DEFAULT_OUTPUT.write_text(payload, encoding="utf-8")
    print(json.dumps({"path": str(DEFAULT_OUTPUT), "preregistration_hash": out["preregistration_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
