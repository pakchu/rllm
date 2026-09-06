from __future__ import annotations

import json
import py_compile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_pposm_fresh_forward_signal_inventory_v2 as builder
from training import preregister_pposm_fresh_forward_signal_inventory_v2 as prereg


def prereg_payload_for(manifest: Path, cache: Path, *, code_hashes: dict[str, str] | None = None, parity_atol: float = 1e-10) -> dict:
    payload = {
        "policy_id": prereg.POLICY_ID,
        "source_contract": {
            "this_preregistration_db_rows_opened": 0,
            "db_query_start": prereg.QUERY_START,
            "forward_start": prereg.FORWARD_START,
            "forward_end_exclusive": prereg.FORWARD_END_EXCLUSIVE,
            "cache_precedence_before": prereg.CACHE_PRECEDENCE_BEFORE,
            "parity_window": [prereg.PARITY_START, prereg.PARITY_END_EXCLUSIVE],
            "source_completeness_policy": prereg.COMPLETENESS_POLICY,
            "state_feature_columns": list(prereg.STATE_FEATURE_COLUMNS),
            "active_feature_columns": list(prereg.ACTIVE_FEATURE_COLUMNS),
            "parity_feature_columns": list(prereg.PARITY_FEATURE_COLUMNS),
            "funding_alias_policy": prereg.FUNDING_ALIAS_POLICY,
            "v1_failure_artifact_sha256": prereg.V1_FAILURE_ARTIFACT_SHA256,
            "funding_alias_source_diagnostic_sha256": prereg.SOURCE_DIAGNOSTIC_ARTIFACT_SHA256,
            "funding_alias_source_diagnostic_result_hash": prereg.SOURCE_DIAGNOSTIC_RESULT_HASH,
            "symbol": prereg.SYMBOL,
            "interval": prereg.INTERVAL,
            "forward_last_inclusive_decision": prereg.FORWARD_LAST_DECISION,
            "required_sources": ["bars_binance", "bars_binance_premium", "funding_rates_binance"],
            "forbidden_sources": ["open_interest", "post_entry_returns", "execution_lifecycle", "pnl_labels"],
            "builder_must_query_db_rows": True,
            "post_entry_outcomes_opened": False,
            "v1_failure_artifact": str(prereg.V1_FAILURE_ARTIFACT),
            "funding_alias_source_diagnostic_artifact": str(prereg.SOURCE_DIAGNOSTIC_ARTIFACT),
            "source_mechanics_contract": "DB may contain millisecond-offset alias rows for the same funding event; only exact-value near aliases are canonicalized, and dynamic <=12h funding cadence is allowed without assuming fixed 8h vendor cadence.",
        },
        "query_hashes": prereg.query_hashes(),
        "code_hashes": code_hashes or {"ok": "ok"},
        "terminal_gate": {"parity_atol": parity_atol, "parity_rtol": prereg.PARITY_RTOL, "parity_before_forward_count": True, "if_cache_db_parity_fails": "terminal_source_mismatch_no_forward_signal_count", "parity_scope": "normalized causal feature outputs at hourly decisions in May overlap", "no_execution_engine": True, "no_training": True},
        "frozen_pposm": {"manifest_sha256": builder.sha256_file(manifest)},
        "immutable_cache": {"sha256": builder.sha256_file(cache)},
    }
    payload["preregistration_hash"] = builder.expected_preregistration_hash(payload)
    return payload


def write_prereg(tmp_path: Path, manifest: Path, cache: Path, **kwargs) -> Path:
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(prereg_payload_for(manifest, cache, **kwargs)), encoding="utf-8")
    return path


def one_minute_frame(start: str = prereg.QUERY_START, end: str = prereg.FORWARD_END_EXCLUSIVE, *, premium: bool = False) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None) - pd.Timedelta(minutes=1), freq="1min")
    if premium:
        return pd.DataFrame({"date": idx, "close_time": idx + pd.Timedelta(seconds=59), "close": 0.0})
    return pd.DataFrame({"date": idx, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "quote_asset_volume": 1.0, "number_of_trades": 1.0, "taker_buy_base": 0.5, "taker_buy_quote": 0.5})


def five_minute_market() -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(prereg.FORWARD_START).tz_localize(None), pd.Timestamp(prereg.FORWARD_END_EXCLUSIVE).tz_localize(None) - pd.Timedelta(minutes=5), freq="5min")
    return pd.DataFrame({"date": idx, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})


def funding_times(freq: str = "8h") -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(prereg.QUERY_START).tz_localize(None), pd.Timestamp(prereg.FORWARD_LAST_DECISION).tz_localize(None), freq=freq)


def funding_frame(freq: str = "8h") -> pd.DataFrame:
    return pd.DataFrame({"funding_time": funding_times(freq), "funding_rate": 0.001, "mark_price": 100.0})


def observed_alias_shape_frame() -> pd.DataFrame:
    singles = pd.date_range("2026-04-01", periods=322, freq="4h")
    pairs = pd.date_range(singles[-1] + pd.Timedelta(hours=4), periods=151, freq="4h")
    rows = [{"funding_time": ts, "funding_rate": 0.001, "mark_price": 100.0} for ts in singles]
    for ts in pairs:
        rows.append({"funding_time": ts, "funding_rate": 0.002, "mark_price": 101.0})
        rows.append({"funding_time": ts + pd.Timedelta(milliseconds=26), "funding_rate": 0.002, "mark_price": 101.0})
    return pd.DataFrame(rows)


def test_v2_modules_compile() -> None:
    py_compile.compile("training/preregister_pposm_fresh_forward_signal_inventory_v2.py", doraise=True)
    py_compile.compile("training/build_pposm_fresh_forward_signal_inventory_v2.py", doraise=True)


def test_policy_binds_v1_failure_and_dynamic_funding_contract() -> None:
    assert prereg.POLICY_ID == "pposm_fresh_forward_signal_inventory_v2"
    assert prereg.DEFAULT_OUTPUT == Path("results/pposm_fresh_forward_signal_inventory_v2_preregistration_2026-09-05.json")
    assert builder.DEFAULT_OUTPUT == Path("results/pposm_fresh_forward_signal_inventory_v2_2026-09-05.json")
    assert prereg.sha256_file(prereg.V1_FAILURE_ARTIFACT) == prereg.V1_FAILURE_ARTIFACT_SHA256
    assert prereg.sha256_file(prereg.SOURCE_DIAGNOSTIC_ARTIFACT) == prereg.SOURCE_DIAGNOSTIC_ARTIFACT_SHA256
    funding_policy = prereg.COMPLETENESS_POLICY["funding"]
    assert funding_policy["exact_event_grid"] is False
    assert funding_policy["canonicalization"] == prereg.FUNDING_ALIAS_POLICY
    assert "4h" in funding_policy["allowed_dynamic_intervals"]


def test_canonicalize_observed_624_to_473_equivalent_and_preserves_latest_timestamp() -> None:
    raw = observed_alias_shape_frame()
    canonical, diag = builder.canonicalize_funding_aliases(raw)
    assert diag["passed"] is True
    assert diag["raw_rows"] == 624
    assert diag["canonical_rows"] == 473
    assert diag["alias_clusters"] == 151
    assert diag["alias_rows_removed"] == 151
    assert diag["cluster_size_histogram"] == {"1": 322, "2": 151}
    first_pair = canonical.iloc[322]
    assert first_pair["date"] == raw.iloc[323]["funding_time"]


def test_canonicalize_matches_cache_overlap_identity_after_latest_selection() -> None:
    cache = pd.DataFrame({"date": pd.to_datetime(["2026-05-01 00:00:00.026", "2026-05-01 04:00:00.026"]), "funding_rate": [0.1, 0.2], "mark_price": [100.0, 101.0]})
    raw = pd.DataFrame({"funding_time": pd.to_datetime(["2026-05-01 00:00:00", "2026-05-01 00:00:00.026", "2026-05-01 04:00:00", "2026-05-01 04:00:00.026"], format="mixed"), "funding_rate": [0.1, 0.1, 0.2, 0.2], "mark_price": [100.0, 100.0, 101.0, 101.0]})
    canonical, diag = builder.canonicalize_funding_aliases(raw)
    assert diag["passed"] is True
    pd.testing.assert_frame_equal(canonical.reset_index(drop=True), cache.reset_index(drop=True))


def test_canonicalize_rejects_unequal_near_alias_nan_duplicate_and_span_over_100ms() -> None:
    base = pd.DataFrame({"funding_time": pd.to_datetime(["2026-04-01 00:00:00", "2026-04-01 00:00:00.026"], format="mixed"), "funding_rate": [0.1, 0.2], "mark_price": [100.0, 100.0]})
    assert builder.canonicalize_funding_aliases(base)[1]["reason"] == "unequal_values_within_near_alias_cluster"
    bad = funding_frame().head(2)
    bad.loc[1, "mark_price"] = np.nan
    assert builder.canonicalize_funding_aliases(bad)[1]["reason"] == "nonfinite_funding_rate_or_mark_price"
    dup = pd.concat([funding_frame().head(1), funding_frame().head(1)], ignore_index=True)
    assert builder.canonicalize_funding_aliases(dup)[1]["reason"] == "exact_duplicate_timestamp"
    span = pd.DataFrame({"funding_time": pd.to_datetime(["2026-04-01 00:00:00", "2026-04-01 00:00:00.060", "2026-04-01 00:00:00.120"], format="mixed"), "funding_rate": [0.1, 0.1, 0.1], "mark_price": [100.0, 100.0, 100.0]})
    assert builder.canonicalize_funding_aliases(span)[1]["reason"] == "cluster_span_gt_100ms"


def test_canonical_funding_completeness_allows_dynamic_1h_4h_8h_and_rejects_gap_over_12h() -> None:
    for freq in ("1h", "4h", "8h"):
        canonical, diag = builder.canonicalize_funding_aliases(funding_frame(freq))
        check = builder._funding_completeness(canonical, alias_diagnostics=diag)
        assert check["passed"] is True
        assert "does not prove vendor-global event completeness" in check["semantics"]
    gap = funding_frame("8h").drop(index=[2, 3]).reset_index(drop=True)
    canonical, diag = builder.canonicalize_funding_aliases(gap)
    check = builder._funding_completeness(canonical, alias_diagnostics=diag)
    assert check["passed"] is False
    assert check["max_gap"]["passed"] is False
    assert check["max_gap"]["max_gap_hours"] > 12.0


def test_build_path_uses_canonical_funding_and_blocks_before_count_on_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.csv.gz"
    manifest.write_text(json.dumps({"base_thresholds": {"frozen": 1}, "feature_prefix_hash": "prefix", "state_thresholds": {}, "frozen_execution_config": {}}), encoding="utf-8")
    cache.write_bytes(b"cache")
    prereg_path = write_prereg(tmp_path, manifest, cache, code_hashes={"ok": "ok"})
    monkeypatch.setattr(builder, "runtime_code_hashes", lambda: {"ok": "ok"})
    monkeypatch.setattr(builder, "load_frozen_cache_bundle", lambda *a, **k: (pd.DataFrame({"date": pd.to_datetime(["2026-05-01"])}), pd.DataFrame({col: [1.0] for col in builder.CAUSAL_FEATURE_COLUMNS}), {}))
    monkeypatch.setattr(builder, "build_signal_inventory", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not count")))
    bad = pd.DataFrame({"funding_time": pd.to_datetime(["2026-04-01 00:00:00", "2026-04-01 00:00:00.026"], format="mixed"), "funding_rate": [0.1, 0.2], "mark_price": [100.0, 100.0]})
    out = builder.build_from_frames(preregistration=prereg_path, manifest=manifest, cache=cache, btcusdt_1m=one_minute_frame(), premium_1m=one_minute_frame(premium=True), funding=bad)
    assert out["terminal"] == "source_incomplete"
    assert out["forward_counted"] is False
    assert out["source_row_counts"]["funding_raw"] == 2
    assert out["source_row_counts"]["funding_canonical"] is None
    assert out["funding_alias_diagnostics"]["reason"] == "unequal_values_within_near_alias_cluster"

    raw = funding_frame()
    raw.loc[0, "funding_rate"] = 0.1
    raw.loc[0, "mark_price"] = 100.0
    raw = pd.concat([raw.iloc[[0]], pd.DataFrame({"funding_time": [pd.Timestamp("2026-04-01 00:00:00.026")], "funding_rate": [0.1], "mark_price": [100.0]}), raw.iloc[1:]], ignore_index=True)
    seen: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(builder, "resample_market_bars", lambda *a, **k: five_minute_market())
    monkeypatch.setattr(builder, "check_source_completeness", lambda **k: {"passed": True, "failed": [], "checks": {}})
    def fake_features_from_5m_market(market, funding, premium):
        seen["funding"] = funding.copy()
        enriched = market.copy()
        enriched["funding_rate"] = float(funding.iloc[0]["funding_rate"])
        enriched["funding_available"] = 1.0
        enriched["premium_index_change"] = 0.0
        enriched["premium_available"] = 1.0
        return enriched, pd.DataFrame({col: [1.0] * len(market) for col in builder.CAUSAL_FEATURE_COLUMNS})
    monkeypatch.setattr(builder, "_features_from_5m_market", fake_features_from_5m_market)
    monkeypatch.setattr(builder, "compare_parity", lambda *a, **k: {"passed": False, "reason": "forced"})
    out2 = builder.build_from_frames(preregistration=prereg_path, manifest=manifest, cache=cache, btcusdt_1m=one_minute_frame(), premium_1m=one_minute_frame(premium=True), funding=raw)
    assert out2["terminal"] == "source_mismatch"
    assert seen["funding"].iloc[0]["date"] == pd.Timestamp("2026-04-01 00:00:00.026")


def test_preregistration_rejects_code_hash_tolerance_policy_and_v1_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.csv.gz"
    manifest.write_text(json.dumps({"state_thresholds": {"x": 1}}), encoding="utf-8")
    cache.write_bytes(b"cache")
    path = write_prereg(tmp_path, manifest, cache, code_hashes={"wrong": "hash"})
    monkeypatch.setattr(builder, "runtime_code_hashes", lambda: {"right": "hash"})
    with pytest.raises(RuntimeError, match="code hashes"):
        builder.validate_preregistration(path, manifest=manifest, cache=cache)
    path = write_prereg(tmp_path, manifest, cache, code_hashes={"right": "hash"}, parity_atol=1e-8)
    with pytest.raises(RuntimeError, match="parity tolerances"):
        builder.validate_preregistration(path, manifest=manifest, cache=cache)
    payload = prereg_payload_for(manifest, cache, code_hashes={"right": "hash"})
    payload["source_contract"]["funding_alias_policy"] = {"tampered": True}
    payload["preregistration_hash"] = builder.expected_preregistration_hash(payload)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="funding_alias_policy"):
        builder.validate_preregistration(bad, manifest=manifest, cache=cache)


def test_code_hashes_bind_canonicalization_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereg, "sha256_file", lambda path: "filehash")
    hashes = prereg.code_hashes()
    assert hashes["feature_set_policy"]
    assert "builder_local.canonicalize_funding_aliases" in hashes
    assert "builder_local._funding_completeness" in hashes
    assert "v1_failure_artifact" in hashes
    assert "funding_alias_source_diagnostic_artifact" in hashes


def test_pass_path_merges_enriched_db_forward_and_inventory_receives_aux_features(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.csv.gz"
    manifest_payload = {"base_thresholds": {"frozen": 1}, "feature_prefix_hash": "prefix", "state_thresholds": {}, "frozen_execution_config": {}}
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    cache.write_bytes(b"cache")
    prereg_path = write_prereg(tmp_path, manifest, cache, code_hashes={"ok": "ok"})
    monkeypatch.setattr(builder, "runtime_code_hashes", lambda: {"ok": "ok"})
    cache_market = pd.DataFrame({"date": pd.to_datetime(["2023-12-31 23:55", "2026-05-01 00:00"]), "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0], "funding_rate": [0.0, 0.0], "funding_available": [1.0, 1.0], "premium_index_change": [0.0, 0.0], "premium_available": [1.0, 1.0]})
    cache_features = pd.DataFrame({col: [1.0, 1.0] for col in builder.CAUSAL_FEATURE_COLUMNS})
    monkeypatch.setattr(builder, "load_frozen_cache_bundle", lambda *a, **k: (cache_market, cache_features, manifest_payload))
    db_market = pd.DataFrame({"date": pd.to_datetime(["2026-06-02 00:00"]), "open": [3.0], "high": [3.0], "low": [3.0], "close": [3.0]})
    db_features = pd.DataFrame({col: [1.0] for col in builder.CAUSAL_FEATURE_COLUMNS})
    monkeypatch.setattr(builder, "resample_market_bars", lambda *a, **k: db_market)
    monkeypatch.setattr(builder, "check_source_completeness", lambda **k: {"passed": True, "failed": [], "checks": {}})

    def fake_features_from_5m_market(market, funding, premium):
        enriched = market.copy()
        enriched["funding_rate"] = float(funding.iloc[0]["funding_rate"])
        enriched["funding_available"] = 1.0
        enriched["premium_index_change"] = 0.25
        enriched["premium_available"] = 1.0
        return enriched, db_features

    seen: dict[str, object] = {}

    def fake_features_from_enriched_market(enriched):
        forward = enriched[pd.to_datetime(enriched["date"]) >= pd.Timestamp("2026-06-02")].iloc[0]
        seen["forward_funding_rate"] = float(forward["funding_rate"])
        seen["forward_premium_index_change"] = float(forward["premium_index_change"])
        return pd.DataFrame({col: [1.0] * len(enriched) for col in builder.CAUSAL_FEATURE_COLUMNS})

    def fake_build_signal_inventory(market, features, manifest):
        forward = market[pd.to_datetime(market["date"]) >= pd.Timestamp("2026-06-02")].iloc[0]
        seen["inventory_market_funding_rate"] = float(forward["funding_rate"])
        seen["inventory_has_features"] = set(builder.CAUSAL_FEATURE_COLUMNS).issubset(features.columns)
        return [], {"signals": 0, "route_counts": {"TP4": 0, "SKIP": 0, "TP12": 0}}

    monkeypatch.setattr(builder, "_features_from_5m_market", fake_features_from_5m_market)
    monkeypatch.setattr(builder, "compare_parity", lambda *a, **k: {"passed": True, "reason": "pass"})
    monkeypatch.setattr(builder, "features_from_enriched_market", fake_features_from_enriched_market)
    monkeypatch.setattr(builder, "_fit_active", lambda *a, **k: (np.ones(len(a[0]), dtype=bool), {"frozen": 1}))
    monkeypatch.setattr(builder.pposm, "feature_hash", lambda *a, **k: "prefix")
    monkeypatch.setattr(builder, "build_signal_inventory", fake_build_signal_inventory)
    raw = funding_frame()
    raw.loc[0, "funding_rate"] = 0.123
    raw.loc[0, "mark_price"] = 100.0
    raw = pd.concat([raw.iloc[[0]], pd.DataFrame({"funding_time": [pd.Timestamp("2026-04-01 00:00:00.026")], "funding_rate": [0.123], "mark_price": [100.0]}), raw.iloc[1:]], ignore_index=True)
    out = builder.build_from_frames(preregistration=prereg_path, manifest=manifest, cache=cache, btcusdt_1m=one_minute_frame(), premium_1m=one_minute_frame(premium=True), funding=raw)
    assert out["terminal"] == "pass"
    assert out["source_row_counts"]["db_5m_market_raw"] == 1
    assert out["source_row_counts"]["db_5m_market_enriched"] == 1
    assert seen == {
        "forward_funding_rate": 0.123,
        "forward_premium_index_change": 0.25,
        "inventory_market_funding_rate": 0.123,
        "inventory_has_features": True,
    }


def test_canonicalize_uses_decimal_equality_not_float_precision() -> None:
    # These two values collapse to the same binary float, but they are distinct
    # decimal source values and must not be treated as aliases.
    raw = pd.DataFrame({
        "funding_time": pd.to_datetime(["2026-04-01 00:00:00", "2026-04-01 00:00:00.026"], format="mixed"),
        "funding_rate": ["0.10000000000000000000000000001", "0.1"],
        "mark_price": ["100.0", "100.00"],
    })
    assert float(raw.loc[0, "funding_rate"]) == float(raw.loc[1, "funding_rate"])
    canonical, diag = builder.canonicalize_funding_aliases(raw)
    assert canonical.empty
    assert diag["passed"] is False
    assert diag["reason"] == "unequal_values_within_near_alias_cluster"
    assert diag["column"] == "funding_rate"

    equal = raw.copy()
    equal.loc[0, "funding_rate"] = "0.10"
    canonical_equal, diag_equal = builder.canonicalize_funding_aliases(equal)
    assert diag_equal["passed"] is True
    assert len(canonical_equal) == 1
    assert canonical_equal.iloc[0]["date"] == pd.Timestamp("2026-04-01 00:00:00.026")


def test_frame_hash_preserves_decimal_precision_beyond_float() -> None:
    a = pd.DataFrame({"date": [pd.Timestamp("2026-04-01")], "funding_rate": ["0.10000000000000000000000000001"]})
    b = pd.DataFrame({"date": [pd.Timestamp("2026-04-01")], "funding_rate": ["0.1"]})
    assert float(a.loc[0, "funding_rate"]) == float(b.loc[0, "funding_rate"])
    assert builder.frame_hash(a) != builder.frame_hash(b)
    c = pd.DataFrame({"date": [pd.Timestamp("2026-04-01")], "funding_rate": ["0.10"]})
    assert builder.frame_hash(b) == builder.frame_hash(c)


def test_query_db_frames_preserves_funding_decimal_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types
    from decimal import Decimal

    monkeypatch.setitem(sys.modules, "sqlalchemy", types.SimpleNamespace(text=lambda sql: sql))
    calls: list[dict[str, object]] = []

    def fake_read_sql_query(sql, conn, params, **kwargs):
        calls.append({"sql": str(sql), "kwargs": kwargs})
        if "funding_rates_binance" in str(sql):
            return pd.DataFrame({"funding_time": [pd.Timestamp("2026-04-01")], "funding_rate": [Decimal("0.10000000000000000000000000001")], "mark_price": [Decimal("100.0")]})
        if "bars_binance_premium" in str(sql):
            return pd.DataFrame({"date": [pd.Timestamp("2026-04-01")], "close_time": [pd.Timestamp("2026-04-01 00:00:59")], "close": [0.0]})
        return pd.DataFrame({"date": [pd.Timestamp("2026-04-01")], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0], "quote_asset_volume": [1.0], "number_of_trades": [1.0], "taker_buy_base": [0.5], "taker_buy_quote": [0.5], "tic": ["BTCUSDT"]})

    monkeypatch.setattr(builder.pd, "read_sql_query", fake_read_sql_query)
    frames = builder.query_db_frames(type("Conn", (), {"__enter__": lambda self: self, "__exit__": lambda self, *args: False})())
    funding_call = [call for call in calls if "funding_rates_binance" in call["sql"]][0]
    assert funding_call["kwargs"] == {"coerce_float": False}
    assert frames["funding"].loc[0, "funding_rate"] == Decimal("0.10000000000000000000000000001")


def test_validate_preregistration_rejects_full_contract_and_terminal_gate_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.csv.gz"
    manifest.write_text(json.dumps({"state_thresholds": {"x": 1}}), encoding="utf-8")
    cache.write_bytes(b"cache")
    monkeypatch.setattr(builder, "runtime_code_hashes", lambda: {"ok": "ok"})
    base = prereg_payload_for(manifest, cache, code_hashes={"ok": "ok"})
    for key, value in [("symbol", "ETHUSDT"), ("interval", "5m"), ("forward_last_inclusive_decision", "bad"), ("required_sources", []), ("forbidden_sources", []), ("builder_must_query_db_rows", False), ("post_entry_outcomes_opened", True), ("v1_failure_artifact", "bad"), ("funding_alias_source_diagnostic_artifact", "bad"), ("source_mechanics_contract", "bad")]:
        payload = json.loads(json.dumps(base))
        payload["source_contract"][key] = value
        payload["preregistration_hash"] = builder.expected_preregistration_hash(payload)
        path = tmp_path / f"bad_{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match=f"preregistration {key} mismatch"):
            builder.validate_preregistration(path, manifest=manifest, cache=cache)
    payload = json.loads(json.dumps(base))
    payload["terminal_gate"]["no_training"] = False
    payload["preregistration_hash"] = builder.expected_preregistration_hash(payload)
    path = tmp_path / "bad_gate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="terminal_gate no_training"):
        builder.validate_preregistration(path, manifest=manifest, cache=cache)


def test_funding_completeness_rejects_canonical_times_outside_query_window() -> None:
    early = funding_frame()
    early.loc[0, "funding_time"] = pd.Timestamp(prereg.QUERY_START).tz_localize(None) - pd.Timedelta(milliseconds=1)
    canonical, diag = builder.canonicalize_funding_aliases(early)
    check = builder._funding_completeness(canonical, alias_diagnostics=diag)
    assert check["passed"] is False
    assert check["window"]["passed"] is False
    assert check["window"]["first_out_of_window"] == "2026-03-31T23:59:59.999000Z"

    late = funding_frame()
    late.loc[len(late)] = {"funding_time": pd.Timestamp(prereg.FORWARD_END_EXCLUSIVE).tz_localize(None), "funding_rate": 0.001, "mark_price": 100.0}
    canonical, diag = builder.canonicalize_funding_aliases(late)
    check = builder._funding_completeness(canonical, alias_diagnostics=diag)
    assert check["passed"] is False
    assert check["window"]["first_out_of_window"] == "2026-09-05T10:05:00Z"
