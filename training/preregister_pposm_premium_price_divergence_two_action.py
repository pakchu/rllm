"""Freeze the final pre-2024 PPOSM premium/price path source test."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training import preregister_pposm_premium1m_path_two_action as premium

PROTOCOL = "pposm_premium_price_divergence_two_action_preregistration_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("results/pposm_premium_price_divergence_two_action_preregistration_2026-09-05.json")
DEFAULT_RESULT = Path("results/pposm_premium_price_divergence_two_action_source_support_2026-09-05.json")
PREDECESSOR = premium.DEFAULT_RESULT
PREDECESSOR_SHA256 = "280cba7ddfeae4e9c21f7c6b2ce7cc541f52f38e84068ec1687ae619987f1d5e"
PREDECESSOR_RESULT_HASH = "08f0546f4d8337613056e1ecdcd29bc6a6509acff7a15e2c6a52b17b0f748c89"
MARKET_QUERY = """
    SELECT ts, close
    FROM bars_binance
    WHERE symbol = :symbol AND interval = '1m'
      AND ts >= :start AND ts < :end
      AND ts < '2024-01-01T00:00:00Z'
    ORDER BY ts
"""
FEATURE_COLUMNS = (
    "increment_correlation",
    "premium_leads_price_correlation",
    "price_leads_premium_correlation",
    "signed_comovement_mean",
    "premium_beta_on_price",
    "path_efficiency_difference",
    "terminal_zscore_difference",
    "half_drift_zscore_difference",
)
NUMERIC_EPSILON = 1e-15


def _bound_predecessor() -> dict[str, str]:
    if premium.sha256_file(PREDECESSOR) != PREDECESSOR_SHA256:
        raise RuntimeError("premium-only predecessor bytes changed")
    payload = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    if payload.get("result_hash") != PREDECESSOR_RESULT_HASH or payload.get("support_passed") is not False:
        raise RuntimeError("premium-only predecessor receipt changed")
    return {"path": str(PREDECESSOR), "sha256": PREDECESSOR_SHA256, "result_hash": PREDECESSOR_RESULT_HASH}


def code_hashes() -> dict[str, str]:
    return {
        "preregistration_module": premium.sha256_file(__file__),
        "evaluator_module": premium.sha256_file(PROJECT_ROOT / "training/evaluate_pposm_premium_price_divergence_two_action.py"),
        "strict_premium_evaluator_module": premium.sha256_file(PROJECT_ROOT / "training/evaluate_pposm_premium1m_path_two_action.py"),
    }


def build_preregistration() -> dict[str, Any]:
    strict = premium.build_preregistration()
    core = {
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "test the predeclared one-minute premium-versus-BTC path divergence channel for both pairwise PPOSM residual actions",
        "why_final_source": {
            "predecessor": _bound_predecessor(),
            "novelty": "cross-path signed and lead-lag divergence cannot be reconstructed from the frozen hourly premium close or the premium-only path vector",
            "identified_before_predecessor_metrics": True,
            "gate_weakening": False,
        },
        "sources": {
            "premium_query": premium.PREMIUM_QUERY,
            "premium_query_sha256": premium.sha256_bytes(premium.PREMIUM_QUERY.strip().encode()),
            "market_query": MARKET_QUERY,
            "market_query_sha256": premium.sha256_bytes(MARKET_QUERY.strip().encode()),
            "window": [premium.QUERY_START, premium.QUERY_END_EXCLUSIVE],
            "symbol": "BTCUSDT",
            "interval": "1m",
            "path": "exact paired 60-minute rows ending one minute before decision; no fill/interpolation/nearest",
            "premium_close_time_rule": "ts <= close_time < ts+1min and close_time <= decision_time",
            "market_completion_rule": "open-time ts <= decision_time-1min",
            "coerce_float": False,
        },
        "labels": strict["labels"],
        "features": list(FEATURE_COLUMNS),
        "feature_numeric_guard": {
            "epsilon": NUMERIC_EPSILON,
            "rule": "correlation, beta, efficiency, and z-score denominators at or below epsilon map to neutral 0",
        },
        "model": strict["model"],
        "folds": strict["folds"],
        "coverage_gates": strict["coverage_gates"],
        "support_gates": strict["support_gates"],
        "bootstrap": strict["bootstrap"],
        "selection": "none; one fixed cross-path feature vector and the unchanged strict-premium inference contract",
        "code_hashes": code_hashes(),
        "runtime_versions": strict["runtime_versions"],
        "terminal_rule": "any failed action gate exhausts this cross-path source; no inversion, refit, SFT, RLVR, or OOS",
        "evidence_boundary": {
            "database_rows_opened": 0,
            "source_support_metrics_computed": False,
            "oos_labels_opened": 0,
            "fresh_outcomes_opened": 0,
            "sft_or_rlvr_started": False,
        },
    }
    payload = dict(core)
    payload["preregistration_hash"] = premium.sha256_bytes(premium.canonical_json({k: v for k, v in core.items() if k != "created_at"}).encode())
    return payload


def write_preregistration(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_preregistration()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("preregistration_hash") != payload["preregistration_hash"]:
            raise RuntimeError("refusing to overwrite different divergence preregistration")
        return existing
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    value = write_preregistration()
    print(json.dumps({"protocol": PROTOCOL, "preregistration_hash": value["preregistration_hash"], "database_rows_opened": 0}, indent=2))
