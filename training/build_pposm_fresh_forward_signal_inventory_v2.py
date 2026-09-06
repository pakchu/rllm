"""Build a fail-closed fresh-forward PPOSM causal signal inventory.

The builder validates May cache-vs-DB normalized causal feature parity before it
counts any forward signal.  It never opens execution labels, computes PnL, trains,
or instantiates a trade execution simulator; the output is a signal-time causal
inventory with hashes over time, feature snapshot, and frozen route only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import attach_binance_um_aux_frames, normalise_funding_history_frame, normalise_premium_index_frame
from preprocessing.live_db_features import postgres_url_from_env, resample_market_bars
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_interest_gate_validation import build_interest_features
from training.audit_confirmed_pullback_squeeze_live_parity import _load_bundle
from training import search_pullback_premium_overheat_state_machine_alpha as pposm
from training.audit_confirmed_pullback_squeeze_live_parity import _fit_active, decision_mask, live_decision_features
from training.preregister_pposm_fresh_forward_signal_inventory_v2 import (
    CACHE_PRECEDENCE_BEFORE,
    COMPLETENESS_POLICY,
    DEFAULT_CACHE,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT as DEFAULT_PREREG,
    SOURCE_ROOT,
    FORWARD_END_EXCLUSIVE,
    FORWARD_LAST_DECISION,
    FORWARD_START,
    PARITY_END_EXCLUSIVE,
    PARITY_START,
    ACTIVE_FEATURE_COLUMNS,
    PARITY_ATOL,
    PARITY_FEATURE_COLUMNS,
    PARITY_RTOL,
    STATE_FEATURE_COLUMNS,
    POLICY_ID,
    FUNDING_ALIAS_POLICY,
    V1_FAILURE_ARTIFACT,
    V1_FAILURE_ARTIFACT_SHA256,
    SOURCE_DIAGNOSTIC_ARTIFACT,
    SOURCE_DIAGNOSTIC_ARTIFACT_SHA256,
    SOURCE_DIAGNOSTIC_RESULT_HASH,
    QUERY_START,
    READ_ONLY_QUERIES,
    SYMBOL,
    canonical_json,
    query_hashes,
    sha256_bytes,
    sha256_file,
)

DEFAULT_OUTPUT = Path("results/pposm_fresh_forward_signal_inventory_v2_2026-09-05.json")
CAUSAL_FEATURE_COLUMNS = PARITY_FEATURE_COLUMNS
ROUTES = ("TP4", "SKIP", "TP12")


@dataclass(frozen=True)
class Config:
    preregistration: Path = DEFAULT_PREREG
    manifest: Path = DEFAULT_MANIFEST
    cache: Path = DEFAULT_CACHE
    output: Path = DEFAULT_OUTPUT
    env_file: Path = SOURCE_ROOT / ".env"


def _coerce_utc_naive(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise").dt.tz_convert(None)


def _canonical_decimal_string(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def _canonical_hash_cell(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return pd.Timestamp(value).tz_localize("UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC")
    if isinstance(value, Decimal):
        return _canonical_decimal_string(value)
    if isinstance(value, str):
        decimal = _decimal_from_raw(value)
        if decimal is not None:
            return _canonical_decimal_string(decimal)
        return value
    return value


def frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        elif normalized[column].dtype == object:
            normalized[column] = normalized[column].map(_canonical_hash_cell)
    encoded = normalized.to_json(orient="split", date_format="iso", double_precision=15).encode()
    return sha256_bytes(encoded)


def row_hash(value: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(value).encode())


def builder_code_hash() -> str:
    return sha256_file(__file__)


def runtime_code_hashes() -> dict[str, str]:
    import training.preregister_pposm_fresh_forward_signal_inventory_v2 as prereg_module
    hashes = prereg_module.code_hashes()
    hashes["builder_module"] = builder_code_hash()
    return hashes


def expected_preregistration_hash(prereg: dict[str, Any]) -> str:
    core = {k: v for k, v in prereg.items() if k not in {"created_at", "preregistration_hash"}}
    return sha256_bytes(canonical_json(core).encode())


def _expect_contract_value(contract: dict[str, Any], key: str, expected: Any) -> None:
    if contract.get(key) != expected:
        raise RuntimeError(f"preregistration {key} mismatch")


def validate_preregistration(path: str | Path, *, manifest: str | Path, cache: str | Path) -> dict[str, Any]:
    prereg = json.loads(Path(path).read_text(encoding="utf-8"))
    if prereg.get("policy_id") != POLICY_ID:
        raise RuntimeError("wrong preregistration policy_id")
    if prereg.get("preregistration_hash") != expected_preregistration_hash(prereg):
        raise RuntimeError("preregistration hash mismatch")
    if prereg.get("source_contract", {}).get("this_preregistration_db_rows_opened") != 0:
        raise RuntimeError("preregistration is not outcome-blind")
    if prereg.get("query_hashes") != query_hashes():
        raise RuntimeError("query template hashes differ from preregistration")
    if prereg.get("code_hashes") != runtime_code_hashes():
        raise RuntimeError("runtime code hashes differ from preregistration")
    gate = prereg.get("terminal_gate", {})
    if float(gate.get("parity_atol")) != PARITY_ATOL or float(gate.get("parity_rtol")) != PARITY_RTOL:
        raise RuntimeError("preregistration parity tolerances differ")
    if prereg.get("frozen_pposm", {}).get("manifest_sha256") != sha256_file(manifest):
        raise RuntimeError("frozen manifest differs from preregistration")
    if prereg.get("immutable_cache", {}).get("sha256") != sha256_file(cache):
        raise RuntimeError("cached prefix differs from preregistration")
    contract = prereg.get("source_contract", {})
    expected = {
        "symbol": SYMBOL,
        "interval": "1m",
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
        "funding_alias_policy": FUNDING_ALIAS_POLICY,
        "v1_failure_artifact": str(V1_FAILURE_ARTIFACT),
        "v1_failure_artifact_sha256": V1_FAILURE_ARTIFACT_SHA256,
        "funding_alias_source_diagnostic_artifact": str(SOURCE_DIAGNOSTIC_ARTIFACT),
        "funding_alias_source_diagnostic_sha256": SOURCE_DIAGNOSTIC_ARTIFACT_SHA256,
        "funding_alias_source_diagnostic_result_hash": SOURCE_DIAGNOSTIC_RESULT_HASH,
        "source_mechanics_contract": "DB may contain millisecond-offset alias rows for the same funding event; only exact-value near aliases are canonicalized, and dynamic <=12h funding cadence is allowed without assuming fixed 8h vendor cadence.",
    }
    for key, value in expected.items():
        _expect_contract_value(contract, key, value)
    if sha256_file(V1_FAILURE_ARTIFACT) != V1_FAILURE_ARTIFACT_SHA256:
        raise RuntimeError("v1 failure artifact bytes changed")
    if sha256_file(SOURCE_DIAGNOSTIC_ARTIFACT) != SOURCE_DIAGNOSTIC_ARTIFACT_SHA256:
        raise RuntimeError("source diagnostic artifact bytes changed")
    diagnostic = json.loads(SOURCE_DIAGNOSTIC_ARTIFACT.read_text(encoding="utf-8"))
    if diagnostic.get("result_hash") != SOURCE_DIAGNOSTIC_RESULT_HASH:
        raise RuntimeError("source diagnostic internal result_hash changed")
    expected_gate = {
        "parity_before_forward_count": True,
        "if_cache_db_parity_fails": "terminal_source_mismatch_no_forward_signal_count",
        "parity_scope": "normalized causal feature outputs at hourly decisions in May overlap",
        "no_execution_engine": True,
        "no_training": True,
    }
    for key, value in expected_gate.items():
        if gate.get(key) != value:
            raise RuntimeError(f"preregistration terminal_gate {key} mismatch")
    return prereg


def _read_cache_market(path: str | Path, *, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(path), compression="infer")
    frame["date"] = _coerce_utc_naive(frame["date"])
    mask = (frame["date"] >= pd.Timestamp(start).tz_localize(None)) & (frame["date"] < pd.Timestamp(end).tz_localize(None))
    return frame.loc[mask].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def features_from_enriched_market(enriched: pd.DataFrame) -> pd.DataFrame:
    base = build_market_feature_frame(enriched, window_size=144, zscore_window=96, volume_window=96)
    return pd.concat([base, build_interest_features(enriched, base)], axis=1)


def _features_from_5m_market(market: pd.DataFrame, funding: pd.DataFrame, premium: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = attach_binance_um_aux_frames(
        market,
        funding_frame=normalise_funding_history_frame(funding),
        premium_frame=normalise_premium_index_frame(premium),
        funding_tolerance="12h",
        premium_tolerance="10min",
        zscore_window=96,
    )
    return enriched, features_from_enriched_market(enriched)


def features_from_db_frames(btcusdt_1m: pd.DataFrame, premium_1m: pd.DataFrame, funding: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = resample_market_bars(btcusdt_1m, "5min")
    return _features_from_5m_market(market, funding, premium_1m)


def query_db_frames(engine_or_conn: Any, *, start: str = QUERY_START, end: str = FORWARD_END_EXCLUSIVE) -> dict[str, pd.DataFrame]:
    try:
        from sqlalchemy import text
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SQLAlchemy is required for direct DB querying") from exc
    start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    frames: dict[str, pd.DataFrame] = {}
    with engine_or_conn.connect() if hasattr(engine_or_conn, "connect") else engine_or_conn as conn:
        for key, sql in READ_ONLY_QUERIES.items():
            read_kwargs = {"coerce_float": False} if key == "funding" else {}
            frames[key] = pd.read_sql_query(text(sql), conn, params={"symbol": SYMBOL, "start": start_ts.to_pydatetime(), "end": end_ts.to_pydatetime()}, **read_kwargs)
    return frames


def sql_engine_from_env_file(env_file: str | Path) -> Any:
    try:
        from sqlalchemy import create_engine
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SQLAlchemy is required for direct DB querying") from exc
    return create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})


def merge_cache_db_markets(cache_market: pd.DataFrame, db_market: pd.DataFrame, *, cutoff: str = CACHE_PRECEDENCE_BEFORE) -> pd.DataFrame:
    cutoff_ts = pd.Timestamp(cutoff).tz_localize(None)
    cache = cache_market.copy()
    db = db_market.copy()
    cache["date"] = _coerce_utc_naive(cache["date"])
    db["date"] = _coerce_utc_naive(db["date"])
    before = cache.loc[cache["date"] < cutoff_ts]
    after = db.loc[db["date"] >= cutoff_ts]
    columns = list(dict.fromkeys([*before.columns, *after.columns]))
    merged = pd.concat([before.reindex(columns=columns), after.reindex(columns=columns)], ignore_index=True)
    return merged.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def parity_decision_frame(features: pd.DataFrame, dates: pd.Series, *, start: str = PARITY_START, end: str = PARITY_END_EXCLUSIVE) -> pd.DataFrame:
    live = live_decision_features(features)
    parsed = pd.to_datetime(dates)
    mask = decision_mask(parsed, "live_hour_signal_bar", window_size=144)
    mask &= (parsed >= pd.Timestamp(start).tz_localize(None)).to_numpy(bool)
    mask &= (parsed < pd.Timestamp(end).tz_localize(None)).to_numpy(bool)
    missing = set(PARITY_FEATURE_COLUMNS).difference(live.columns)
    if missing:
        raise ValueError(f"missing parity features: {sorted(missing)}")
    out = live.loc[mask, list(PARITY_FEATURE_COLUMNS)].copy()
    out.insert(0, "date", parsed.loc[mask].to_numpy())
    return out.reset_index(drop=True)


def compare_parity(cache_frame: pd.DataFrame, db_frame: pd.DataFrame, *, atol: float = 1e-10, rtol: float = 1e-9) -> dict[str, Any]:
    left = cache_frame.copy()
    right = db_frame.copy()
    if len(left) != len(right) or not pd.to_datetime(left["date"]).reset_index(drop=True).equals(pd.to_datetime(right["date"]).reset_index(drop=True)):
        return {"passed": False, "reason": "decision_clock_mismatch", "cache_rows": len(left), "db_rows": len(right), "cache_hash": frame_hash(left), "db_hash": frame_hash(right)}
    mismatches: list[dict[str, Any]] = []
    for column in PARITY_FEATURE_COLUMNS:
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        ok = np.isclose(a, b, atol=float(atol), rtol=float(rtol), equal_nan=True)
        if not bool(ok.all()):
            bad = int(np.flatnonzero(~ok)[0])
            mismatches.append({"column": column, "date": str(pd.Timestamp(left.loc[bad, "date"])), "cache": float(a[bad]) if np.isfinite(a[bad]) else None, "db": float(b[bad]) if np.isfinite(b[bad]) else None})
    return {"passed": not mismatches, "reason": "pass" if not mismatches else "feature_value_mismatch", "mismatches": mismatches[:20], "checked_rows": len(left), "cache_hash": frame_hash(left), "db_hash": frame_hash(right)}


def route_for_index(*, active: np.ndarray, capitulation: np.ndarray, overheat: np.ndarray, index: int) -> str | None:
    if not bool(active[index]):
        return None
    if bool(capitulation[index]):
        return "TP4"
    if bool(overheat[index]):
        return "SKIP"
    return "TP12"



def _timestamp_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([], dtype="datetime64[ns]")
    return _coerce_utc_naive(frame[column]).sort_values().reset_index(drop=True)


def _expected_grid(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start).tz_localize(None)
    end_ts = pd.Timestamp(end).tz_localize(None)
    return pd.date_range(start_ts, end_ts, freq=freq, inclusive="left")


def _grid_check(frame: pd.DataFrame, *, column: str, start: str, end: str, freq: str) -> dict[str, Any]:
    if column not in frame.columns:
        expected = _expected_grid(start, end, freq)
        return {"passed": False, "reason": f"missing_time_column:{column}", "expected_rows": int(len(expected)), "observed_rows": 0, "duplicate_rows": 0, "missing_count": int(len(expected)), "extra_count": 0, "first_missing": None if len(expected) == 0 else pd.Timestamp(expected[0]).isoformat() + "Z", "last_expected": None if len(expected) == 0 else pd.Timestamp(expected[-1]).isoformat() + "Z", "last_observed": None}
    observed = _timestamp_series(frame, column)
    expected = _expected_grid(start, end, freq)
    duplicate_count = int(observed.duplicated().sum())
    observed_unique = pd.DatetimeIndex(observed.drop_duplicates())
    missing = expected.difference(observed_unique)
    extra = observed_unique.difference(expected)
    passed = duplicate_count == 0 and len(missing) == 0 and len(extra) == 0 and len(observed) == len(expected)
    return {
        "passed": bool(passed),
        "expected_rows": int(len(expected)),
        "observed_rows": int(len(observed)),
        "duplicate_rows": duplicate_count,
        "missing_count": int(len(missing)),
        "extra_count": int(len(extra)),
        "first_missing": None if len(missing) == 0 else pd.Timestamp(missing[0]).isoformat() + "Z",
        "first_extra": None if len(extra) == 0 else pd.Timestamp(extra[0]).isoformat() + "Z",
        "last_expected": None if len(expected) == 0 else pd.Timestamp(expected[-1]).isoformat() + "Z",
        "last_observed": None if len(observed) == 0 else pd.Timestamp(observed.iloc[-1]).isoformat() + "Z",
    }



def _finite_column_check(frame: pd.DataFrame, *, columns: tuple[str, ...], positive: tuple[str, ...] = (), nonnegative: tuple[str, ...] = ()) -> dict[str, Any]:
    missing = [column for column in columns if column not in frame.columns]
    failures: list[dict[str, Any]] = []
    if missing:
        return {"passed": False, "missing_columns": missing, "failures": failures}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(values.to_numpy(float))
        if column in positive:
            ok = finite & (values.to_numpy(float) > 0.0)
            rule = "finite_positive"
        elif column in nonnegative:
            ok = finite & (values.to_numpy(float) >= 0.0)
            rule = "finite_nonnegative"
        else:
            ok = finite
            rule = "finite"
        if not bool(ok.all()):
            bad = int(np.flatnonzero(~ok)[0])
            failures.append({"column": column, "rule": rule, "bad_rows": int((~ok).sum()), "first_bad_index": bad})
    return {"passed": not missing and not failures, "missing_columns": missing, "failures": failures}

def _premium_close_time_check(premium_1m: pd.DataFrame) -> dict[str, Any]:
    if "close_time" not in premium_1m.columns or "date" not in premium_1m.columns:
        return {"passed": False, "reason": "missing_close_time_or_date"}
    dates = _coerce_utc_naive(premium_1m["date"])
    close_times = _coerce_utc_naive(premium_1m["close_time"])
    non_null = premium_1m["close_time"].notna().to_numpy(bool)
    causal = (close_times >= dates).to_numpy(bool) & (close_times < dates + pd.Timedelta(minutes=1)).to_numpy(bool)
    ok = non_null & causal
    first_bad = None if bool(ok.all()) else int(np.flatnonzero(~ok)[0])
    return {
        "passed": bool(ok.all()),
        "rows": int(len(premium_1m)),
        "bad_rows": int((~ok).sum()),
        "first_bad_date": None if first_bad is None else pd.Timestamp(dates.iloc[first_bad]).isoformat() + "Z",
        "semantics": "premium close_time must be non-null and within [source minute, source minute + 1min)",
    }


def _funding_time_column(funding: pd.DataFrame) -> str:
    if "funding_time" in funding.columns:
        return "funding_time"
    if "date" in funding.columns:
        return "date"
    return "funding_time"


def _iso_z(ts: pd.Timestamp | None) -> str | None:
    if ts is None:
        return None
    return pd.Timestamp(ts).isoformat() + "Z"



def _decimal_from_raw(value: Any) -> Decimal | None:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return decimal if decimal.is_finite() else None


def _funding_decimal_columns(work: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    converted = work.copy()
    for column in ("funding_rate", "mark_price"):
        decimals = converted[column].map(_decimal_from_raw)
        bad_mask = decimals.isna().to_numpy(bool)
        if bool(bad_mask.any()):
            return converted, {"passed": False, "reason": "nonfinite_funding_rate_or_mark_price", "column": column, "first_bad_index": int(np.flatnonzero(bad_mask)[0])}
        converted[f"_{column}_decimal"] = decimals
    return converted, None

def canonicalize_funding_aliases(funding: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse exact-value millisecond aliases without imposing an 8h cadence."""
    time_column = _funding_time_column(funding)
    required = [time_column, "funding_rate", "mark_price"]
    missing = [column for column in required if column not in funding.columns]
    base_diag: dict[str, Any] = {
        "policy": FUNDING_ALIAS_POLICY,
        "raw_rows": int(len(funding)),
        "canonical_rows": 0,
        "alias_clusters": 0,
        "alias_rows_removed": 0,
        "max_cluster_span_ms": None,
        "cluster_size_histogram": {},
        "selected_timestamp": "latest_raw_timestamp",
    }
    if missing:
        return pd.DataFrame(columns=["date", "funding_rate", "mark_price"]), {**base_diag, "passed": False, "reason": "missing_columns", "missing_columns": missing}
    try:
        work = funding.loc[:, required].copy()
        work["date"] = _coerce_utc_naive(work[time_column])
    except Exception as exc:
        return pd.DataFrame(columns=["date", "funding_rate", "mark_price"]), {**base_diag, "passed": False, "reason": "malformed_timestamp", "error": str(exc)}
    if bool(work["date"].duplicated().any()):
        first = pd.Timestamp(work.loc[work["date"].duplicated(keep=False), "date"].iloc[0])
        return pd.DataFrame(columns=["date", "funding_rate", "mark_price"]), {**base_diag, "passed": False, "reason": "exact_duplicate_timestamp", "first_duplicate_time": _iso_z(first)}
    work, decimal_error = _funding_decimal_columns(work)
    if decimal_error is not None:
        return pd.DataFrame(columns=["date", "funding_rate", "mark_price"]), {**base_diag, **decimal_error}
    work = work.sort_values("date").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    clusters: list[pd.DataFrame] = []
    start_idx = 0
    for idx in range(1, len(work)):
        delta_ms = (pd.Timestamp(work.loc[idx, "date"]) - pd.Timestamp(work.loc[idx - 1, "date"])) / pd.Timedelta(milliseconds=1)
        if float(delta_ms) > float(FUNDING_ALIAS_POLICY["max_cluster_span_ms"]):
            clusters.append(work.iloc[start_idx:idx].copy())
            start_idx = idx
    if len(work):
        clusters.append(work.iloc[start_idx:].copy())
    max_span = 0.0
    histogram: Counter[int] = Counter()
    alias_clusters = 0
    for cluster in clusters:
        span_ms = (pd.Timestamp(cluster["date"].iloc[-1]) - pd.Timestamp(cluster["date"].iloc[0])) / pd.Timedelta(milliseconds=1)
        max_span = max(max_span, float(span_ms))
        histogram[int(len(cluster))] += 1
        if float(span_ms) > float(FUNDING_ALIAS_POLICY["max_cluster_span_ms"]):
            return pd.DataFrame(columns=["date", "funding_rate", "mark_price"]), {**base_diag, "passed": False, "reason": "cluster_span_gt_100ms", "span_ms": float(span_ms)}
        if len(cluster) > 1:
            alias_clusters += 1
            for column in ("funding_rate", "mark_price"):
                decimal_column = f"_{column}_decimal"
                if not bool((cluster[decimal_column].to_numpy(object) == cluster[decimal_column].iloc[0]).all()):
                    return pd.DataFrame(columns=["date", "funding_rate", "mark_price"]), {**base_diag, "passed": False, "reason": "unequal_values_within_near_alias_cluster", "column": column, "cluster_start": _iso_z(pd.Timestamp(cluster["date"].iloc[0])), "cluster_end": _iso_z(pd.Timestamp(cluster["date"].iloc[-1]))}
        selected = cluster.iloc[-1]
        rows.append({"date": pd.Timestamp(selected["date"]), "funding_rate": float(selected["_funding_rate_decimal"]), "mark_price": float(selected["_mark_price_decimal"])})
    canonical = pd.DataFrame(rows, columns=["date", "funding_rate", "mark_price"])
    return canonical, {
        **base_diag,
        "passed": True,
        "reason": "pass",
        "canonical_rows": int(len(canonical)),
        "alias_clusters": int(alias_clusters),
        "alias_rows_removed": int(len(funding) - len(canonical)),
        "max_cluster_span_ms": float(max_span) if len(work) else None,
        "cluster_size_histogram": {str(k): int(v) for k, v in sorted(histogram.items())},
        "raw_hash": frame_hash(funding),
        "canonical_hash": frame_hash(canonical),
    }


def _funding_completeness(canonical_funding: pd.DataFrame, *, alias_diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    alias = alias_diagnostics if alias_diagnostics is not None else canonicalize_funding_aliases(canonical_funding)[1]
    values = _finite_column_check(canonical_funding, columns=("funding_rate", "mark_price"))
    times = _timestamp_series(canonical_funding, "date")
    unique_increasing = bool(not times.empty and len(times) == len(times.drop_duplicates()) and times.is_monotonic_increasing)
    query_start = pd.Timestamp(QUERY_START).tz_localize(None)
    last_decision = pd.Timestamp(FORWARD_LAST_DECISION).tz_localize(None)
    window_end = pd.Timestamp(FORWARD_END_EXCLUSIVE).tz_localize(None)
    in_window_mask = ((times >= query_start) & (times < window_end)).to_numpy(bool) if not times.empty else np.array([], dtype=bool)
    first_out_of_window_idx = None if bool(in_window_mask.all()) else int(np.flatnonzero(~in_window_mask)[0])
    window = {"passed": bool(not times.empty and in_window_mask.all()), "start": QUERY_START, "end_exclusive": FORWARD_END_EXCLUSIVE, "first_out_of_window": None if first_out_of_window_idx is None else _iso_z(pd.Timestamp(times.iloc[first_out_of_window_idx]))}
    first = None if times.empty else pd.Timestamp(times.iloc[0])
    eligible = times.loc[times <= last_decision] if not times.empty else times
    last = None if eligible.empty else pd.Timestamp(eligible.iloc[-1])
    first_age = None if first is None else (first - query_start) / pd.Timedelta(hours=1)
    last_age = None if last is None else (last_decision - last) / pd.Timedelta(hours=1)
    gaps = times.diff().dropna() / pd.Timedelta(hours=1) if not times.empty else pd.Series(dtype=float)
    max_gap = None if gaps.empty else float(gaps.max())
    first_coverage = {"passed": bool(first is not None and 0.0 <= float(first_age) <= 12.0), "first_funding_time": _iso_z(first), "age_from_query_start_hours": None if first_age is None else float(first_age), "max_age_hours": 12.0}
    final_coverage = {"passed": bool(last is not None and 0.0 <= float(last_age) <= 12.0), "last_funding_time": _iso_z(last), "age_hours": None if last_age is None else float(last_age), "max_age_hours": 12.0}
    gap_check = {"passed": bool(max_gap is not None and max_gap <= 12.0) if len(times) > 1 else bool(len(times) == 1), "max_gap_hours": max_gap, "max_allowed_hours": 12.0}
    monotonic = {"passed": unique_increasing, "rows": int(len(times)), "duplicate_rows": int(times.duplicated().sum()) if not times.empty else 0}
    passed = bool(alias.get("passed") and values.get("passed") and window["passed"] and first_coverage["passed"] and final_coverage["passed"] and gap_check["passed"] and monotonic["passed"])
    return {"passed": passed, "alias_canonicalization": alias, "window": window, "monotonic": monotonic, "first_coverage": first_coverage, "final_decision_coverage": final_coverage, "max_gap": gap_check, "values": values, "semantics": "canonical funding proves <=12h feature availability over the queried window only; it does not prove vendor-global event completeness or any fixed 8h cadence, and dynamic 1h/4h/8h intervals are allowed"}


def check_raw_source_completeness(*, btcusdt_1m: pd.DataFrame, premium_1m: pd.DataFrame, funding: pd.DataFrame, canonical_funding: pd.DataFrame | None = None, alias_diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    market_grid = _grid_check(btcusdt_1m, column="date", start=QUERY_START, end=FORWARD_END_EXCLUSIVE, freq="1min")
    market_values = _finite_column_check(btcusdt_1m, columns=("open", "high", "low", "close", "volume", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote"), positive=("open", "high", "low", "close"), nonnegative=("volume", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote"))
    market = {"passed": bool(market_grid.get("passed") and market_values.get("passed")), "grid": market_grid, "values": market_values}
    premium_grid = _grid_check(premium_1m, column="date", start=QUERY_START, end=FORWARD_END_EXCLUSIVE, freq="1min")
    premium_close_time = _premium_close_time_check(premium_1m)
    premium_values = _finite_column_check(premium_1m, columns=("close",))
    premium = {"passed": bool(premium_grid.get("passed") and premium_close_time.get("passed") and premium_values.get("passed")), "grid": premium_grid, "close_time": premium_close_time, "values": premium_values}
    if canonical_funding is None or alias_diagnostics is None:
        canonical_funding, alias_diagnostics = canonicalize_funding_aliases(funding)
    funding_check = _funding_completeness(canonical_funding, alias_diagnostics=alias_diagnostics)
    checks = {"market_1m": market, "premium_1m": premium, "funding": funding_check}
    passed = all(bool(value.get("passed")) for value in checks.values())
    failed = [name for name, value in checks.items() if not bool(value.get("passed"))]
    return {"passed": bool(passed), "failed": failed, "policy": COMPLETENESS_POLICY, "checks": checks}


def check_source_completeness(*, btcusdt_1m: pd.DataFrame, premium_1m: pd.DataFrame, funding: pd.DataFrame, db_market_5m: pd.DataFrame, canonical_funding: pd.DataFrame | None = None, alias_diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = check_raw_source_completeness(btcusdt_1m=btcusdt_1m, premium_1m=premium_1m, funding=funding, canonical_funding=canonical_funding, alias_diagnostics=alias_diagnostics)
    forward = _grid_check(db_market_5m, column="date", start=FORWARD_START, end=FORWARD_END_EXCLUSIVE, freq="5min")
    checks = {**raw["checks"], "forward_5m": forward}
    passed = bool(raw.get("passed") and forward.get("passed"))
    failed = [name for name, value in checks.items() if not bool(value.get("passed"))]
    return {"passed": passed, "failed": failed, "policy": COMPLETENESS_POLICY, "checks": checks}


def terminal_result(result_base: dict[str, Any], *, terminal: str, reason: dict[str, Any]) -> dict[str, Any]:
    result = {
        **result_base,
        "terminal": terminal,
        "terminal_reason": reason,
        "forward_counted": False,
        "signals": [],
        "summary": {"signals": 0, "route_counts": {route: 0 for route in ROUTES}},
    }
    result["result_hash"] = row_hash({k: v for k, v in result.items() if k != "result_hash"})
    return result

def build_signal_inventory(market: pd.DataFrame, features: pd.DataFrame, manifest: dict[str, Any], *, start: str = FORWARD_START, end: str = FORWARD_END_EXCLUSIVE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dates = pd.to_datetime(market["date"])
    live = live_decision_features(features)
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=144)
    active, base_thresholds = _fit_active(live, dates, decisions)
    if base_thresholds != manifest.get("base_thresholds"):
        raise RuntimeError("frozen base thresholds changed; refusing fresh-forward count")
    state = pposm.state_feature_frame(live)
    capitulation, overheat = pposm.build_state_masks(state, manifest["state_thresholds"], pposm.FROZEN_CHAMPION["overheat"])
    period = (dates >= pd.Timestamp(start).tz_localize(None)).to_numpy(bool) & (dates < pd.Timestamp(end).tz_localize(None)).to_numpy(bool) & decisions
    rows: list[dict[str, Any]] = []
    for idx in np.flatnonzero(period):
        route = route_for_index(active=active, capitulation=capitulation, overheat=overheat, index=int(idx))
        if route is None:
            continue
        snapshot = {column: (None if not np.isfinite(float(live.iloc[int(idx)][column])) else float(live.iloc[int(idx)][column])) for column in PARITY_FEATURE_COLUMNS}
        signal = {"decision_time": pd.Timestamp(dates.iloc[int(idx)]).isoformat() + "Z", "route": route, "causal_features": snapshot}
        signal["signal_hash"] = row_hash(signal)
        rows.append(signal)
    counts = dict(Counter(row["route"] for row in rows))
    for route in ROUTES:
        counts.setdefault(route, 0)
    summary = {"signals": len(rows), "route_counts": counts, "signal_inventory_hash": sha256_bytes(canonical_json(rows).encode())}
    return rows, summary


def load_frozen_cache_bundle(manifest: str | Path, cache: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frozen_manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    execution = dict(frozen_manifest.get("frozen_execution_config", {}))
    execution["input_csv"] = str(cache)
    strategy_cfg = pposm.Config(**execution, manifest_output=str(manifest))
    cache_market, cache_features, _cache_funding, _hashes = _load_bundle(strategy_cfg, cutoff=CACHE_PRECEDENCE_BEFORE, premium_tolerance=strategy_cfg.live_premium_tolerance)
    return cache_market, cache_features, frozen_manifest


def build_from_frames(*, preregistration: str | Path, manifest: str | Path, cache: str | Path, btcusdt_1m: pd.DataFrame, premium_1m: pd.DataFrame, funding: pd.DataFrame, output: str | Path | None = None) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration, manifest=manifest, cache=cache)
    canonical_funding, alias_diagnostics = canonicalize_funding_aliases(funding)
    raw_completeness = check_raw_source_completeness(btcusdt_1m=btcusdt_1m, premium_1m=premium_1m, funding=funding, canonical_funding=canonical_funding, alias_diagnostics=alias_diagnostics)
    db_market: pd.DataFrame | None = None
    db_enriched: pd.DataFrame | None = None
    if raw_completeness["passed"]:
        db_market = resample_market_bars(btcusdt_1m, "5min")
        completeness = check_source_completeness(btcusdt_1m=btcusdt_1m, premium_1m=premium_1m, funding=funding, canonical_funding=canonical_funding, alias_diagnostics=alias_diagnostics, db_market_5m=db_market)
    else:
        completeness = raw_completeness
    source_hashes = {"btcusdt_1m": frame_hash(btcusdt_1m), "premium_1m": frame_hash(premium_1m), "funding_raw": frame_hash(funding), "funding_canonical": None if not alias_diagnostics.get("passed") else frame_hash(canonical_funding), "db_5m_market_raw": None if db_market is None else frame_hash(db_market)}
    result_base = {
        "policy_id": POLICY_ID,
        "preregistration_hash": prereg["preregistration_hash"],
        "frozen_manifest_sha256": sha256_file(manifest),
        "cache_sha256": sha256_file(cache),
        "query_hashes": query_hashes(),
        "code_hashes": runtime_code_hashes(),
        "source_row_counts": {"btcusdt_1m": len(btcusdt_1m), "premium_1m": len(premium_1m), "funding_raw": len(funding), "funding_canonical": None if not alias_diagnostics.get("passed") else len(canonical_funding), "db_5m_market_raw": None if db_market is None else len(db_market), "db_5m_market_enriched": None},
        "source_hashes": source_hashes,
        "funding_alias_diagnostics": alias_diagnostics,
        "source_completeness": completeness,
        "opened_outcomes": False,
        "trained": False,
    }
    if not completeness["passed"]:
        result = terminal_result(result_base, terminal="source_incomplete", reason=completeness)
    else:
        cache_market, cache_features, frozen_manifest = load_frozen_cache_bundle(manifest, cache)
        db_enriched, db_features = _features_from_5m_market(db_market, canonical_funding, premium_1m)
        result_base["source_row_counts"]["db_5m_market_enriched"] = len(db_enriched)
        result_base["source_hashes"]["db_5m_market_enriched"] = frame_hash(db_enriched)
        parity_atol = float(prereg["terminal_gate"]["parity_atol"])
        parity_rtol = float(prereg["terminal_gate"]["parity_rtol"])
        parity = compare_parity(
            parity_decision_frame(cache_features, cache_market["date"]),
            parity_decision_frame(db_features, db_market["date"]),
            atol=parity_atol,
            rtol=parity_rtol,
        )
        result_base["parity"] = parity
        if not parity["passed"]:
            result = terminal_result(result_base, terminal="source_mismatch", reason=parity)
        else:
            merged_market = merge_cache_db_markets(cache_market, db_enriched)
            merged_features = features_from_enriched_market(merged_market)
            dates = pd.to_datetime(merged_market["date"])
            live_prefix = live_decision_features(merged_features)
            decisions_prefix = decision_mask(dates, "live_hour_signal_bar", window_size=144)
            _active_prefix, base_thresholds = _fit_active(live_prefix, dates, decisions_prefix)
            if base_thresholds != frozen_manifest.get("base_thresholds"):
                raise RuntimeError("combined historical prefix changed frozen base thresholds")
            prefix = (dates < pd.Timestamp("2024-01-01")).to_numpy(bool)
            if pposm.feature_hash(pposm.state_feature_frame(live_prefix), prefix) != frozen_manifest.get("feature_prefix_hash"):
                raise RuntimeError("combined historical prefix changed frozen state feature hash")
            signals, summary = build_signal_inventory(merged_market, merged_features, frozen_manifest)
            result = {**result_base, "terminal": "pass", "forward_counted": True, "signals": signals, "summary": summary}
    result["result_hash"] = row_hash({k: v for k, v in result.items() if k != "result_hash"})
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return result


def run(cfg: Config = Config()) -> dict[str, Any]:
    engine = sql_engine_from_env_file(cfg.env_file)
    try:
        frames = query_db_frames(engine)
    finally:
        if hasattr(engine, "dispose"):
            engine.dispose()
    return build_from_frames(preregistration=cfg.preregistration, manifest=cfg.manifest, cache=cfg.cache, btcusdt_1m=frames["btcusdt_1m"], premium_1m=frames["premium_1m"], funding=frames["funding"], output=cfg.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=Config.env_file)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(json.dumps({"terminal": payload["terminal"], "forward_counted": payload["forward_counted"], "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
