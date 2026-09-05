"""Preregister the fresh-forward PPOSM causal signal inventory.

This file deliberately opens no database rows and no post-entry outcomes.  It
binds only immutable code, cache/manifest files, query templates, and the
forward source window that a later builder must replay fail-closed.
"""
from __future__ import annotations

import argparse
import hashlib
import ast
import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preprocessing import binance_aux_features
from preprocessing import live_db_features
from preprocessing import market_features
from training.long_regime_interest_gate_validation import build_interest_features
from training import search_pullback_premium_overheat_state_machine_alpha as pposm
from training.audit_confirmed_pullback_squeeze_live_parity import _fit_active, _load_bundle, decision_mask, live_decision_features

POLICY_ID = "pposm_fresh_forward_signal_inventory_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/home/pakchu/rllm")
DEFAULT_OUTPUT = Path(
    "results/pposm_fresh_forward_signal_inventory_preregistration_2026-09-05.json"
)
DEFAULT_MANIFEST = Path(
    "results/pullback_premium_overheat_state_machine_manifest_2026-07-15.json"
)
DEFAULT_CACHE = SOURCE_ROOT / "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
QUERY_START = "2026-04-01T00:00:00Z"
FORWARD_START = "2026-06-02T00:00:00Z"
FORWARD_END_EXCLUSIVE = "2026-09-05T10:05:00Z"
FORWARD_LAST_DECISION = "2026-09-05T10:00:00Z"
CACHE_PRECEDENCE_BEFORE = "2026-06-02T00:00:00Z"
PARITY_START = "2026-05-01T00:00:00Z"
PARITY_END_EXCLUSIVE = "2026-06-02T00:00:00Z"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

PARITY_ATOL = 1e-10
PARITY_RTOL = 1e-9
STATE_FEATURE_COLUMNS = tuple(pposm.FEATURE_QUANTILES.keys())
ACTIVE_FEATURE_COLUMNS = (
    "funding_available",
    "funding_rate",
    "trend_96",
    "premium_available",
    "premium_index_change",
    "htf_1d_return_4",
    "rex_576_range_pos",
    "htf_1d_return_1",
    "htf_3d_return_1",
    "bb_z",
    "quote_vol_z_1d",
)
PARITY_FEATURE_COLUMNS = tuple(dict.fromkeys((*STATE_FEATURE_COLUMNS, *ACTIVE_FEATURE_COLUMNS)))
COMPLETENESS_POLICY: dict[str, object] = {
    "market_1m": {"source": "bars_binance", "time_column": "date", "grid": "1min", "start": QUERY_START, "end_exclusive": FORWARD_END_EXCLUSIVE, "exact_grid": True, "duplicates_allowed": False, "required_finite_columns": ["open", "high", "low", "close", "volume", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote"], "positive_price_columns": ["open", "high", "low", "close"], "nonnegative_quantity_columns": ["volume", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote"]},
    "premium_1m": {"source": "bars_binance_premium", "time_column": "date", "close_time_column": "close_time", "grid": "1min", "start": QUERY_START, "end_exclusive": FORWARD_END_EXCLUSIVE, "exact_grid": True, "duplicates_allowed": False, "required_finite_columns": ["close"], "close_time_non_null": True, "close_time_causal_within_source_minute": True},
    "forward_5m": {"source": "resampled_btcusdt_5m", "time_column": "date", "grid": "5min", "start": FORWARD_START, "end_exclusive": FORWARD_END_EXCLUSIVE, "exact_grid": True, "last_expected_bucket": FORWARD_LAST_DECISION},
    "funding": {"source": "funding_rates_binance", "time_column": "funding_time/date", "event_grid": "8h", "start": QUERY_START, "end_exclusive": FORWARD_END_EXCLUSIVE, "exact_event_grid": True, "duplicates_allowed": False, "required_finite_columns": ["funding_rate"], "last_decision_max_age": "12h", "not_1m_complete": True},
    "fail_closed_before_signal_count": True,
}

READ_ONLY_QUERIES: dict[str, str] = {
    "btcusdt_1m": """
        SELECT ts AS date, open, high, low, close, volume,
               quote_asset_volume, number_of_trades, taker_buy_base, taker_buy_quote,
               symbol AS tic
        FROM bars_binance
        WHERE symbol = :symbol AND interval = '1m' AND ts >= :start AND ts < :end
        ORDER BY ts
    """,
    "premium_1m": """
        SELECT ts AS date, close_time, close
        FROM bars_binance_premium
        WHERE symbol = :symbol AND interval = '1m' AND ts >= :start AND ts < :end
        ORDER BY ts
    """,
    "funding": """
        SELECT funding_time, funding_rate, mark_price
        FROM funding_rates_binance
        WHERE symbol = :symbol AND funding_time >= :start AND funding_time < :end
        ORDER BY funding_time
    """,
}


@dataclass(frozen=True)
class Config:
    output: Path = DEFAULT_OUTPUT
    manifest: Path = DEFAULT_MANIFEST
    cache: Path = DEFAULT_CACHE


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash(obj: Any) -> str:
    return sha256_bytes(inspect.getsource(obj).encode())


def module_file_hash(obj: Any) -> str:
    source_file = inspect.getsourcefile(obj)
    if source_file is None:
        raise RuntimeError(f"cannot locate module source for {obj!r}")
    return sha256_file(source_file)



def file_function_hash(path: str | Path, function_name: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                break
            return sha256_bytes(segment.encode())
    raise RuntimeError(f"function {function_name} not found in {path}")

def query_hashes() -> dict[str, str]:
    return {name: sha256_bytes(sql.strip().encode()) for name, sql in READ_ONLY_QUERIES.items()}


def code_hashes() -> dict[str, str]:
    builder_path = PROJECT_ROOT / "training/build_pposm_fresh_forward_signal_inventory.py"
    builder_functions = (
        "features_from_enriched_market",
        "features_from_db_frames",
        "merge_cache_db_markets",
        "parity_decision_frame",
        "compare_parity",
        "_grid_check",
        "_finite_column_check",
        "_premium_close_time_check",
        "_funding_completeness",
        "check_raw_source_completeness",
        "check_source_completeness",
        "terminal_result",
        "build_signal_inventory",
        "load_frozen_cache_bundle",
        "build_from_frames",
    )
    hashes = {
        "preregistration_module": sha256_file(__file__),
        "builder_module": sha256_file(builder_path),
        "module.audit_confirmed_pullback_squeeze_live_parity": module_file_hash(_load_bundle),
        "module.binance_aux_features": module_file_hash(binance_aux_features),
        "module.live_db_features": module_file_hash(live_db_features),
        "module.market_features": module_file_hash(market_features),
        "module.long_regime_interest_gate_validation": module_file_hash(build_interest_features),
        "module.pposm": module_file_hash(pposm),
        "_load_bundle": source_hash(_load_bundle),
        "_fit_active": source_hash(_fit_active),
        "build_interest_features": source_hash(build_interest_features),
        "normalise_funding_history_frame": source_hash(binance_aux_features.normalise_funding_history_frame),
        "normalise_premium_index_frame": source_hash(binance_aux_features.normalise_premium_index_frame),
        "pposm_state_feature_frame": source_hash(pposm.state_feature_frame),
        "pposm_build_state_masks": source_hash(pposm.build_state_masks),
        "pposm_constants": sha256_bytes(canonical_json({"spec": pposm.SPEC, "champion": pposm.FROZEN_CHAMPION}).encode()),
        "feature_set_policy": sha256_bytes(canonical_json({"state": STATE_FEATURE_COLUMNS, "active": ACTIVE_FEATURE_COLUMNS, "parity": PARITY_FEATURE_COLUMNS}).encode()),
        "decision_mask": source_hash(decision_mask),
        "live_decision_features": source_hash(live_decision_features),
        "resample_market_bars": source_hash(live_db_features.resample_market_bars),
        "attach_binance_um_aux_frames": source_hash(binance_aux_features.attach_binance_um_aux_frames),
        "build_market_feature_frame": source_hash(market_features.build_market_feature_frame),
    }
    hashes.update({f"builder_local.{name}": file_function_hash(builder_path, name) for name in builder_functions})
    return hashes


def build_preregistration(cfg: Config = Config()) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest)
    cache_path = Path(cfg.cache)
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing frozen PPOSM manifest: {manifest_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"missing immutable cached prefix: {cache_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest_path)
    frozen_thresholds = manifest.get("state_thresholds")
    if not isinstance(frozen_thresholds, dict):
        raise RuntimeError("frozen PPOSM manifest lacks state_thresholds")
    payload: dict[str, Any] = {
        "policy_id": POLICY_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "outcome-blind fresh-forward causal signal inventory only",
        "source_contract": {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "db_query_start": QUERY_START,
            "forward_start": FORWARD_START,
            "forward_end_exclusive": FORWARD_END_EXCLUSIVE,
            "forward_last_inclusive_decision": FORWARD_LAST_DECISION,
            "cache_precedence_before": CACHE_PRECEDENCE_BEFORE,
            "parity_window": [PARITY_START, PARITY_END_EXCLUSIVE],
            "required_sources": ["bars_binance", "bars_binance_premium", "funding_rates_binance"],
            "forbidden_sources": ["open_interest", "post_entry_returns", "execution_lifecycle", "pnl_labels"],
            "builder_must_query_db_rows": True,
            "this_preregistration_db_rows_opened": 0,
            "post_entry_outcomes_opened": False,
            "source_completeness_policy": COMPLETENESS_POLICY,
            "state_feature_columns": list(STATE_FEATURE_COLUMNS),
            "active_feature_columns": list(ACTIVE_FEATURE_COLUMNS),
            "parity_feature_columns": list(PARITY_FEATURE_COLUMNS),
        },
        "frozen_pposm": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "freeze_hash": manifest.get("freeze_hash"),
            "champion": pposm.FROZEN_CHAMPION,
            "spec_hash": manifest.get("spec_hash"),
            "implementation_hash": manifest.get("implementation_hash"),
            "state_thresholds_sha256": sha256_bytes(canonical_json(frozen_thresholds).encode()),
        },
        "immutable_cache": {"path": str(cache_path), "sha256": sha256_file(cache_path)},
        "query_templates": READ_ONLY_QUERIES,
        "query_hashes": query_hashes(),
        "code_hashes": code_hashes(),
        "terminal_gate": {
            "parity_before_forward_count": True,
            "parity_atol": PARITY_ATOL,
            "parity_rtol": PARITY_RTOL,
            "if_cache_db_parity_fails": "terminal_source_mismatch_no_forward_signal_count",
            "parity_scope": "normalized causal feature outputs at hourly decisions in May overlap",
            "no_execution_engine": True,
            "no_training": True,
        },
    }
    payload["preregistration_hash"] = sha256_bytes(canonical_json({k: v for k, v in payload.items() if k != "created_at"}).encode())
    return payload


def write_preregistration(cfg: Config = Config()) -> dict[str, Any]:
    output = Path(cfg.output)
    payload = build_preregistration(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("preregistration_hash") != payload["preregistration_hash"]:
            raise RuntimeError("refusing to overwrite different PPOSM fresh-forward preregistration")
        return existing
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def main() -> None:
    payload = write_preregistration(Config(**vars(parse_args())))
    print(json.dumps({"policy_id": payload["policy_id"], "preregistration_hash": payload["preregistration_hash"], "db_rows_opened": 0}, indent=2))


if __name__ == "__main__":
    main()
