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
from training.preregister_pposm_fresh_forward_signal_inventory import (
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
    QUERY_START,
    READ_ONLY_QUERIES,
    SYMBOL,
    canonical_json,
    query_hashes,
    sha256_bytes,
    sha256_file,
)

DEFAULT_OUTPUT = Path("results/pposm_fresh_forward_signal_inventory_2026-09-05.json")
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


def frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    encoded = normalized.to_json(orient="split", date_format="iso", double_precision=15).encode()
    return sha256_bytes(encoded)


def row_hash(value: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(value).encode())


def builder_code_hash() -> str:
    return sha256_file(__file__)


def runtime_code_hashes() -> dict[str, str]:
    import training.preregister_pposm_fresh_forward_signal_inventory as prereg_module
    hashes = prereg_module.code_hashes()
    hashes["builder_module"] = builder_code_hash()
    return hashes


def expected_preregistration_hash(prereg: dict[str, Any]) -> str:
    core = {k: v for k, v in prereg.items() if k not in {"created_at", "preregistration_hash"}}
    return sha256_bytes(canonical_json(core).encode())


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
        "db_query_start": QUERY_START,
        "forward_start": FORWARD_START,
        "forward_end_exclusive": FORWARD_END_EXCLUSIVE,
        "cache_precedence_before": CACHE_PRECEDENCE_BEFORE,
        "parity_window": [PARITY_START, PARITY_END_EXCLUSIVE],
        "state_feature_columns": list(STATE_FEATURE_COLUMNS),
        "active_feature_columns": list(ACTIVE_FEATURE_COLUMNS),
        "parity_feature_columns": list(PARITY_FEATURE_COLUMNS),
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(f"preregistration {key} mismatch")
    if contract.get("source_completeness_policy") != COMPLETENESS_POLICY:
        raise RuntimeError("preregistration source completeness policy mismatch")
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
            frames[key] = pd.read_sql_query(text(sql), conn, params={"symbol": SYMBOL, "start": start_ts.to_pydatetime(), "end": end_ts.to_pydatetime()})
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


def _funding_completeness(funding: pd.DataFrame) -> dict[str, Any]:
    raw_time_column = "funding_time" if "funding_time" in funding.columns else "date" if "date" in funding.columns else "funding_time"
    raw_grid = _grid_check(funding, column=raw_time_column, start=QUERY_START, end=FORWARD_END_EXCLUSIVE, freq="8h")
    raw_values = _finite_column_check(funding, columns=("funding_rate",))
    try:
        normalized = normalise_funding_history_frame(funding)
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"normalise_failed:{exc}",
            "event_grid": raw_grid,
            "values": raw_values,
        }
    times = _timestamp_series(normalized, "date")
    last_decision = pd.Timestamp(FORWARD_LAST_DECISION).tz_localize(None)
    if times.empty:
        freshness = {"passed": False, "last_funding_time": None, "age_hours": None, "max_age_hours": 12.0}
    else:
        eligible = times.loc[times <= last_decision]
        last = None if eligible.empty else pd.Timestamp(eligible.iloc[-1])
        age = None if last is None else (last_decision - last) / pd.Timedelta(hours=1)
        freshness = {"passed": bool(last is not None and 0.0 <= float(age) <= 12.0), "last_funding_time": None if last is None else last.isoformat() + "Z", "age_hours": None if age is None else float(age), "max_age_hours": 12.0}
    passed = bool(raw_grid.get("passed") and freshness["passed"] and raw_values.get("passed"))
    return {"passed": passed, "event_grid": raw_grid, "freshness": freshness, "values": raw_values, "semantics": "exact raw 8h funding events with no duplicates and finite rates plus normalized <=12h freshness at final forward decision; not 1m completeness"}


def check_raw_source_completeness(*, btcusdt_1m: pd.DataFrame, premium_1m: pd.DataFrame, funding: pd.DataFrame) -> dict[str, Any]:
    market_grid = _grid_check(btcusdt_1m, column="date", start=QUERY_START, end=FORWARD_END_EXCLUSIVE, freq="1min")
    market_values = _finite_column_check(btcusdt_1m, columns=("open", "high", "low", "close", "volume", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote"), positive=("open", "high", "low", "close"), nonnegative=("volume", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote"))
    market = {"passed": bool(market_grid.get("passed") and market_values.get("passed")), "grid": market_grid, "values": market_values}
    premium_grid = _grid_check(premium_1m, column="date", start=QUERY_START, end=FORWARD_END_EXCLUSIVE, freq="1min")
    premium_close_time = _premium_close_time_check(premium_1m)
    premium_values = _finite_column_check(premium_1m, columns=("close",))
    premium = {"passed": bool(premium_grid.get("passed") and premium_close_time.get("passed") and premium_values.get("passed")), "grid": premium_grid, "close_time": premium_close_time, "values": premium_values}
    funding_check = _funding_completeness(funding)
    checks = {"market_1m": market, "premium_1m": premium, "funding": funding_check}
    passed = all(bool(value.get("passed")) for value in checks.values())
    failed = [name for name, value in checks.items() if not bool(value.get("passed"))]
    return {"passed": bool(passed), "failed": failed, "policy": COMPLETENESS_POLICY, "checks": checks}


def check_source_completeness(*, btcusdt_1m: pd.DataFrame, premium_1m: pd.DataFrame, funding: pd.DataFrame, db_market_5m: pd.DataFrame) -> dict[str, Any]:
    raw = check_raw_source_completeness(btcusdt_1m=btcusdt_1m, premium_1m=premium_1m, funding=funding)
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
    raw_completeness = check_raw_source_completeness(btcusdt_1m=btcusdt_1m, premium_1m=premium_1m, funding=funding)
    db_market: pd.DataFrame | None = None
    if raw_completeness["passed"]:
        db_market = resample_market_bars(btcusdt_1m, "5min")
        completeness = check_source_completeness(btcusdt_1m=btcusdt_1m, premium_1m=premium_1m, funding=funding, db_market_5m=db_market)
    else:
        completeness = raw_completeness
    source_hashes = {"btcusdt_1m": frame_hash(btcusdt_1m), "premium_1m": frame_hash(premium_1m), "funding": frame_hash(funding), "db_5m_market": None if db_market is None else frame_hash(db_market)}
    result_base = {
        "policy_id": POLICY_ID,
        "preregistration_hash": prereg["preregistration_hash"],
        "frozen_manifest_sha256": sha256_file(manifest),
        "cache_sha256": sha256_file(cache),
        "query_hashes": query_hashes(),
        "code_hashes": runtime_code_hashes(),
        "source_row_counts": {"btcusdt_1m": len(btcusdt_1m), "premium_1m": len(premium_1m), "funding": len(funding), "db_5m_market": None if db_market is None else len(db_market)},
        "source_hashes": source_hashes,
        "source_completeness": completeness,
        "opened_outcomes": False,
        "trained": False,
    }
    if not completeness["passed"]:
        result = terminal_result(result_base, terminal="source_incomplete", reason=completeness)
    else:
        cache_market, cache_features, frozen_manifest = load_frozen_cache_bundle(manifest, cache)
        _db_enriched, db_features = _features_from_5m_market(db_market, funding, premium_1m)
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
            merged_market = merge_cache_db_markets(cache_market, db_market)
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
