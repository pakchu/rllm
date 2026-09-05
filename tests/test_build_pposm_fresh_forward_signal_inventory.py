from __future__ import annotations

import json
import py_compile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_pposm_fresh_forward_signal_inventory as builder
from training import preregister_pposm_fresh_forward_signal_inventory as prereg


def prereg_payload_for(tmp_path: Path, manifest: Path, cache: Path, *, code_hashes: dict[str, str] | None = None, parity_atol: float = 1e-10, parity_rtol: float = 1e-9) -> dict:
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
        },
        "query_hashes": prereg.query_hashes(),
        "code_hashes": code_hashes or {"ok": "ok"},
        "terminal_gate": {"parity_atol": parity_atol, "parity_rtol": parity_rtol},
        "frozen_pposm": {"manifest_sha256": builder.sha256_file(manifest)},
        "immutable_cache": {"sha256": builder.sha256_file(cache)},
    }
    payload["preregistration_hash"] = builder.expected_preregistration_hash(payload)
    return payload


def write_prereg_payload(tmp_path: Path, manifest: Path, cache: Path, **kwargs) -> Path:
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(prereg_payload_for(tmp_path, manifest, cache, **kwargs)), encoding="utf-8")
    return path


def one_minute_frame(start: str, end: str, *, drop: str | None = None, premium: bool = False) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None) - pd.Timedelta(minutes=1), freq="1min")
    if drop is not None:
        idx = idx[idx != pd.Timestamp(drop).tz_localize(None)]
    if premium:
        return pd.DataFrame({"date": idx, "close_time": idx + pd.Timedelta(seconds=59), "close": 0.0})
    return pd.DataFrame({"date": idx, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "quote_asset_volume": 1.0, "number_of_trades": 1.0, "taker_buy_base": 0.5, "taker_buy_quote": 0.5})


def five_minute_market(start: str, end: str, *, drop: str | None = None) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None) - pd.Timedelta(minutes=5), freq="5min")
    if drop is not None:
        idx = idx[idx != pd.Timestamp(drop).tz_localize(None)]
    return pd.DataFrame({"date": idx, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})


def funding_frame(*, end: str = prereg.FORWARD_END_EXCLUSIVE) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(prereg.QUERY_START).tz_localize(None), pd.Timestamp(end).tz_localize(None) - pd.Timedelta(hours=1), freq="8h")
    return pd.DataFrame({"funding_time": idx, "funding_rate": 0.0, "mark_price": 1.0})


def test_modules_compile() -> None:
    py_compile.compile("training/preregister_pposm_fresh_forward_signal_inventory.py", doraise=True)
    py_compile.compile("training/build_pposm_fresh_forward_signal_inventory.py", doraise=True)


def test_absolute_defaults_and_april_warmup() -> None:
    assert prereg.DEFAULT_CACHE == Path("/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
    assert builder.Config().env_file == Path("/home/pakchu/rllm/.env")
    assert prereg.DEFAULT_OUTPUT == Path(
        "results/pposm_fresh_forward_signal_inventory_preregistration_2026-09-05.json"
    )
    assert prereg.QUERY_START == "2026-04-01T00:00:00Z"
    assert prereg.COMPLETENESS_POLICY["fail_closed_before_signal_count"] is True
    assert prereg.COMPLETENESS_POLICY["funding"]["event_grid"] == "8h"
    assert "trend_96" in prereg.ACTIVE_FEATURE_COLUMNS
    assert "funding_available" in prereg.PARITY_FEATURE_COLUMNS


def test_merge_cache_precedence_before_cutoff_and_db_after() -> None:
    cache = pd.DataFrame({"date": pd.to_datetime(["2026-06-01 23:55", "2026-06-02 00:00"]), "close": [1.0, 2.0], "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0]})
    db = pd.DataFrame({"date": pd.to_datetime(["2026-06-01 23:55", "2026-06-02 00:00"]), "close": [10.0, 20.0], "open": [10.0, 20.0], "high": [10.0, 20.0], "low": [10.0, 20.0]})
    merged = builder.merge_cache_db_markets(cache, db, cutoff="2026-06-02T00:00:00Z")
    assert list(merged["close"]) == [1.0, 20.0]


def test_live_decision_prior_bar_causality_keeps_current_auxiliary() -> None:
    dates = pd.date_range("2026-05-01", periods=3, freq="5min")
    features = pd.DataFrame({"rex_576_range_pos": [1.0, 2.0, 3.0], "premium_index_change": [10.0, 20.0, 30.0], "funding_rate": [0.1, 0.2, 0.3]}, index=dates)
    live = builder.live_decision_features(features)
    assert np.isnan(live.iloc[0]["rex_576_range_pos"])
    assert live.iloc[1]["rex_576_range_pos"] == 1.0
    assert live.iloc[1]["premium_index_change"] == 20.0
    assert live.iloc[1]["funding_rate"] == 0.2


def test_compare_parity_fail_closed_before_count() -> None:
    base = pd.DataFrame({"date": pd.to_datetime(["2026-05-01 00:00"]), **{col: [1.0] for col in builder.CAUSAL_FEATURE_COLUMNS}})
    changed = base.copy()
    changed.loc[0, builder.CAUSAL_FEATURE_COLUMNS[0]] = 2.0
    parity = builder.compare_parity(base, changed)
    assert parity["passed"] is False
    assert parity["reason"] == "feature_value_mismatch"
    active_changed = base.copy()
    active_changed.loc[0, "trend_96"] = 2.0
    active_changed.loc[0, "funding_available"] = 0.0
    active_parity = builder.compare_parity(base, active_changed)
    assert active_parity["passed"] is False
    assert {m["column"] for m in active_parity["mismatches"]} == {"funding_available", "trend_96"}


def test_route_counts_use_frozen_formulas_only() -> None:
    active = np.array([False, True, True, True])
    capitulation = np.array([False, True, False, False])
    overheat = np.array([False, True, True, False])
    routes = [builder.route_for_index(active=active, capitulation=capitulation, overheat=overheat, index=i) for i in range(4)]
    assert routes == [None, "TP4", "SKIP", "TP12"]


def test_builder_source_has_no_execution_or_training_calls() -> None:
    source = Path("training/build_pposm_fresh_forward_signal_inventory.py").read_text()
    forbidden = ["ExecutionEngine", "equity_stats", "trade_at", "schedule_window", "post_entry", "net_return", "pnl", "SFTTrainer", "PPOTrainer"]
    assert not [term for term in forbidden if term in source]


def test_preregistration_binds_code_hashes_tolerances_zero_db_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.csv.gz"
    manifest.write_text(json.dumps({"state_thresholds": {"x": 1}, "freeze_hash": "f", "spec_hash": "s", "implementation_hash": "i"}), encoding="utf-8")
    pd.DataFrame({"date": ["2026-05-01"], "open": [1], "high": [1], "low": [1], "close": [1]}).to_csv(cache, index=False, compression="gzip")
    monkeypatch.setattr(prereg, "code_hashes", lambda: {"builder_module": "abc", "unit": "test"})
    payload = prereg.build_preregistration(prereg.Config(output=tmp_path / "p.json", manifest=manifest, cache=cache))
    assert payload["code_hashes"] == {"builder_module": "abc", "unit": "test"}
    assert payload["terminal_gate"]["parity_atol"] == 1e-10
    assert payload["terminal_gate"]["parity_rtol"] == 1e-9
    assert payload["source_contract"]["this_preregistration_db_rows_opened"] == 0
    assert payload["source_contract"]["post_entry_outcomes_opened"] is False


def test_validate_preregistration_rejects_code_hash_or_tolerance_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.csv.gz"
    manifest.write_text(json.dumps({"state_thresholds": {"x": 1}}), encoding="utf-8")
    pd.DataFrame({"date": ["2026-05-01"], "open": [1], "high": [1], "low": [1], "close": [1]}).to_csv(cache, index=False, compression="gzip")
    path = write_prereg_payload(tmp_path, manifest, cache, code_hashes={"wrong": "hash"})
    monkeypatch.setattr(builder, "runtime_code_hashes", lambda: {"right": "hash"})
    with pytest.raises(RuntimeError, match="code hashes"):
        builder.validate_preregistration(path, manifest=manifest, cache=cache)
    path = write_prereg_payload(tmp_path, manifest, cache, code_hashes={"right": "hash"}, parity_atol=1e-8)
    with pytest.raises(RuntimeError, match="parity tolerances"):
        builder.validate_preregistration(path, manifest=manifest, cache=cache)
    payload = prereg_payload_for(tmp_path, manifest, cache, code_hashes={"right": "hash"})
    payload["source_contract"]["source_completeness_policy"] = {"tampered": True}
    payload["preregistration_hash"] = builder.expected_preregistration_hash(payload)
    bad_policy = tmp_path / "bad_policy.json"
    bad_policy.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="completeness policy"):
        builder.validate_preregistration(bad_policy, manifest=manifest, cache=cache)

def test_cache_parity_uses_authoritative_frozen_load_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.csv.gz"
    manifest.write_text(json.dumps({"frozen_execution_config": {"input_csv": "old"}}), encoding="utf-8")
    cache.write_bytes(b"cache")
    calls = []
    def fake_load_bundle(strategy_cfg, *, cutoff, premium_tolerance):
        calls.append((strategy_cfg.input_csv, cutoff, premium_tolerance))
        market = pd.DataFrame({"date": pd.to_datetime(["2026-05-01 00:00"])})
        features = pd.DataFrame({col: [1.0] for col in builder.CAUSAL_FEATURE_COLUMNS})
        return market, features, pd.DataFrame(), {}
    monkeypatch.setattr(builder, "_load_bundle", fake_load_bundle)
    monkeypatch.setattr(builder.pposm, "Config", lambda **kw: type("Cfg", (), {**kw, "live_premium_tolerance": "10min"})())
    market, features, _ = builder.load_frozen_cache_bundle(manifest, cache)
    assert calls == [(str(cache), builder.CACHE_PRECEDENCE_BEFORE, "10min")]
    assert market.iloc[0]["date"] == pd.Timestamp("2026-05-01 00:00")
    assert set(builder.CAUSAL_FEATURE_COLUMNS).issubset(features.columns)


def test_combined_forward_count_uses_full_historical_cache_and_interest_features(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.csv.gz"
    prereg_path = tmp_path / "prereg.json"
    manifest_payload = {"base_thresholds": {"frozen": 1}, "feature_prefix_hash": "prefix", "state_thresholds": {}}
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    cache.write_bytes(b"cache")
    prereg_path = write_prereg_payload(tmp_path, manifest, cache, code_hashes={"ok": "ok"})
    monkeypatch.setattr(builder, "runtime_code_hashes", lambda: {"ok": "ok"})
    historical = pd.DataFrame({"date": pd.to_datetime(["2023-12-31 23:55", "2026-05-01 00:00"]), "close": [1.0, 2.0], "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0]})
    hist_features = pd.DataFrame({col: [1.0, 1.0] for col in builder.CAUSAL_FEATURE_COLUMNS})
    monkeypatch.setattr(builder, "load_frozen_cache_bundle", lambda *a, **k: (historical, hist_features, manifest_payload))
    db_market = pd.DataFrame({"date": pd.to_datetime(["2026-06-02 00:00"]), "close": [3.0], "open": [3.0], "high": [3.0], "low": [3.0]})
    db_features = pd.DataFrame({col: [1.0] for col in builder.CAUSAL_FEATURE_COLUMNS})
    monkeypatch.setattr(builder, "_features_from_5m_market", lambda *a, **k: (db_market, db_features))
    monkeypatch.setattr(builder, "check_raw_source_completeness", lambda **k: {"passed": True, "failed": [], "checks": {}})
    monkeypatch.setattr(builder, "resample_market_bars", lambda *a, **k: db_market)
    monkeypatch.setattr(builder, "check_source_completeness", lambda **k: {"passed": True, "failed": []})
    monkeypatch.setattr(builder, "compare_parity", lambda *a, **k: {"passed": True, "reason": "pass"})
    captured = {}
    def fake_features_from_enriched_market(enriched):
        captured["min_date"] = pd.Timestamp(enriched["date"].min())
        captured["has_db_after"] = bool((pd.to_datetime(enriched["date"]) >= pd.Timestamp("2026-06-02")).any())
        return pd.DataFrame({col: [1.0] * len(enriched) for col in [*builder.CAUSAL_FEATURE_COLUMNS, "interest_feature"]})
    monkeypatch.setattr(builder, "features_from_enriched_market", fake_features_from_enriched_market)
    monkeypatch.setattr(builder, "_fit_active", lambda *a, **k: (np.ones(len(a[0]), dtype=bool), {"frozen": 1}))
    monkeypatch.setattr(builder.pposm, "feature_hash", lambda *a, **k: "prefix")
    monkeypatch.setattr(builder, "build_signal_inventory", lambda *a, **k: ([], {"signals": 0, "route_counts": {"TP4": 0, "SKIP": 0, "TP12": 0}}))
    out = builder.build_from_frames(preregistration=prereg_path, manifest=manifest, cache=cache, btcusdt_1m=pd.DataFrame(), premium_1m=pd.DataFrame(), funding=pd.DataFrame())
    assert out["forward_counted"] is True
    assert captured == {"min_date": pd.Timestamp("2023-12-31 23:55"), "has_db_after": True}


def test_code_hashes_bind_material_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereg, "sha256_file", lambda path: "filehash")
    hashes = prereg.code_hashes()
    for key in ["module.audit_confirmed_pullback_squeeze_live_parity", "module.binance_aux_features", "module.live_db_features", "module.market_features", "module.long_regime_interest_gate_validation", "module.pposm", "feature_set_policy", "_load_bundle", "_fit_active", "build_interest_features", "normalise_funding_history_frame", "normalise_premium_index_frame", "builder_local.check_source_completeness", "builder_local.terminal_result", "builder_local.merge_cache_db_markets"]:
        assert key in hashes


def test_source_completeness_valid_case() -> None:
    market = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE)
    premium = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE, premium=True)
    forward = five_minute_market(prereg.FORWARD_START, prereg.FORWARD_END_EXCLUSIVE)
    check = builder.check_source_completeness(btcusdt_1m=market, premium_1m=premium, funding=funding_frame(), db_market_5m=forward)
    assert check["passed"] is True
    assert "raw 8h funding events" in check["checks"]["funding"]["semantics"]


def test_source_completeness_detects_1m_gap_and_end_coverage() -> None:
    market = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE, drop="2026-04-01 00:07")
    premium = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE, premium=True).iloc[:-1]
    forward = five_minute_market(prereg.FORWARD_START, prereg.FORWARD_END_EXCLUSIVE)
    check = builder.check_source_completeness(btcusdt_1m=market, premium_1m=premium, funding=funding_frame(), db_market_5m=forward)
    assert check["passed"] is False
    assert set(check["failed"]) == {"market_1m", "premium_1m"}
    assert check["checks"]["market_1m"]["grid"]["first_missing"] == "2026-04-01T00:07:00Z"
    assert check["checks"]["premium_1m"]["grid"]["last_observed"] == "2026-09-05T10:03:00Z"


def test_source_completeness_detects_forward_5m_gap_and_funding_stale() -> None:
    market = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE)
    premium = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE, premium=True)
    forward = five_minute_market(prereg.FORWARD_START, prereg.FORWARD_END_EXCLUSIVE, drop="2026-06-02 00:10")
    stale_funding = funding_frame(end="2026-09-04T00:00:00Z")
    check = builder.check_source_completeness(btcusdt_1m=market, premium_1m=premium, funding=stale_funding, db_market_5m=forward)
    assert check["passed"] is False
    assert set(check["failed"]) == {"forward_5m", "funding"}
    assert check["checks"]["forward_5m"]["first_missing"] == "2026-06-02T00:10:00Z"
    assert check["checks"]["funding"]["freshness"]["passed"] is False



def test_raw_completeness_detects_semantic_bad_values_and_premium_close_time() -> None:
    market = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE)
    premium = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE, premium=True)
    market.loc[3, "close"] = -1.0
    premium.loc[4, "close_time"] = premium.loc[4, "date"] - pd.Timedelta(seconds=1)
    check = builder.check_raw_source_completeness(btcusdt_1m=market, premium_1m=premium, funding=funding_frame())
    assert check["passed"] is False
    assert set(check["failed"]) == {"market_1m", "premium_1m"}
    assert check["checks"]["market_1m"]["values"]["failures"][0]["column"] == "close"
    assert check["checks"]["premium_1m"]["close_time"]["first_bad_date"] == "2026-04-01T00:04:00Z"


def test_source_completeness_detects_duplicate_extra_and_funding_gap() -> None:
    market = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE)
    premium = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE, premium=True)
    market = pd.concat([market, market.iloc[[0]]], ignore_index=True)
    premium.loc[len(premium)] = premium.iloc[-1].copy()
    premium.loc[len(premium)-1, "date"] = pd.Timestamp(prereg.FORWARD_END_EXCLUSIVE).tz_localize(None)
    premium.loc[len(premium)-1, "close_time"] = pd.Timestamp(prereg.FORWARD_END_EXCLUSIVE).tz_localize(None)
    funding = funding_frame().drop(index=1).reset_index(drop=True)
    forward = five_minute_market(prereg.FORWARD_START, prereg.FORWARD_END_EXCLUSIVE)
    check = builder.check_source_completeness(btcusdt_1m=market, premium_1m=premium, funding=funding, db_market_5m=forward)
    assert check["passed"] is False
    assert set(check["failed"]) == {"market_1m", "premium_1m", "funding"}
    assert check["checks"]["market_1m"]["grid"]["duplicate_rows"] == 1
    assert check["checks"]["premium_1m"]["grid"]["extra_count"] == 1
    assert check["checks"]["funding"]["event_grid"]["first_missing"] == "2026-04-01T08:00:00Z"

def test_incomplete_source_fails_closed_without_signal_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.csv.gz"
    manifest_payload = {"base_thresholds": {"frozen": 1}, "feature_prefix_hash": "prefix", "state_thresholds": {}, "frozen_execution_config": {}}
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    cache.write_bytes(b"cache")
    prereg_path = write_prereg_payload(tmp_path, manifest, cache, code_hashes={"ok": "ok"})
    monkeypatch.setattr(builder, "runtime_code_hashes", lambda: {"ok": "ok"})
    monkeypatch.setattr(builder, "load_frozen_cache_bundle", lambda *a, **k: (pd.DataFrame({"date": pd.to_datetime(["2026-05-01"])}), pd.DataFrame({col: [1.0] for col in builder.CAUSAL_FEATURE_COLUMNS}), manifest_payload))
    monkeypatch.setattr(builder, "features_from_db_frames", lambda *a, **k: (pd.DataFrame({"date": pd.to_datetime(["2026-06-02"])}), pd.DataFrame({col: [1.0] for col in builder.CAUSAL_FEATURE_COLUMNS})))
    monkeypatch.setattr(builder, "check_raw_source_completeness", lambda **k: {"passed": False, "failed": ["market_1m"], "checks": {}})
    monkeypatch.setattr(builder, "check_source_completeness", lambda **k: (_ for _ in ()).throw(AssertionError("must not resample/check forward")))
    monkeypatch.setattr(builder, "build_signal_inventory", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not count")))
    out = builder.build_from_frames(preregistration=prereg_path, manifest=manifest, cache=cache, btcusdt_1m=pd.DataFrame(), premium_1m=pd.DataFrame(), funding=pd.DataFrame())
    assert out["terminal"] == "source_incomplete"
    assert out["forward_counted"] is False


def test_prereg_tolerance_is_runtime_source_of_truth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.csv.gz"
    manifest_payload = {"base_thresholds": {"frozen": 1}, "feature_prefix_hash": "prefix", "state_thresholds": {}}
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    cache.write_bytes(b"cache")
    prereg_path = write_prereg_payload(tmp_path, manifest, cache, code_hashes={"ok": "ok"})
    monkeypatch.setattr(builder, "runtime_code_hashes", lambda: {"ok": "ok"})
    market = pd.DataFrame({"date": pd.to_datetime(["2023-12-31 23:55", "2026-05-01 00:00", "2026-06-02 00:00"]), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})
    features = pd.DataFrame({col: [1.0, 1.0, 1.0] for col in builder.CAUSAL_FEATURE_COLUMNS})
    monkeypatch.setattr(builder, "load_frozen_cache_bundle", lambda *a, **k: (market.iloc[:2], features.iloc[:2], manifest_payload))
    db_market = market.iloc[2:]
    db_features = features.iloc[2:]
    monkeypatch.setattr(builder, "check_raw_source_completeness", lambda **k: {"passed": True, "failed": [], "checks": {}})
    monkeypatch.setattr(builder, "resample_market_bars", lambda *a, **k: db_market)
    monkeypatch.setattr(builder, "_features_from_5m_market", lambda *a, **k: (db_market, db_features))
    monkeypatch.setattr(builder, "check_source_completeness", lambda **k: {"passed": True, "failed": []})
    captured = {}
    def fake_compare(*a, **k):
        captured["kwargs"] = k
        return {"passed": False, "reason": "forced"}
    monkeypatch.setattr(builder, "compare_parity", fake_compare)
    out = builder.build_from_frames(preregistration=prereg_path, manifest=manifest, cache=cache, btcusdt_1m=pd.DataFrame(), premium_1m=pd.DataFrame(), funding=pd.DataFrame())
    assert out["terminal"] == "source_mismatch"
    assert captured["kwargs"] == {"atol": prereg.PARITY_ATOL, "rtol": prereg.PARITY_RTOL}



def test_funding_duplicate_fails_before_normalizer_dedup() -> None:
    market = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE)
    premium = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE, premium=True)
    funding = pd.concat([funding_frame(), funding_frame().iloc[[1]]], ignore_index=True)
    forward = five_minute_market(prereg.FORWARD_START, prereg.FORWARD_END_EXCLUSIVE)
    check = builder.check_source_completeness(btcusdt_1m=market, premium_1m=premium, funding=funding, db_market_5m=forward)
    assert check["passed"] is False
    assert check["failed"] == ["funding"]
    assert check["checks"]["funding"]["event_grid"]["duplicate_rows"] == 1
    assert "raw 8h funding events" in check["checks"]["funding"]["semantics"]


def test_nonfinite_funding_rate_fails_before_normalizer_drop() -> None:
    market = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE)
    premium = one_minute_frame(prereg.QUERY_START, prereg.FORWARD_END_EXCLUSIVE, premium=True)
    funding = funding_frame()
    funding.loc[1, "funding_rate"] = np.nan
    forward = five_minute_market(prereg.FORWARD_START, prereg.FORWARD_END_EXCLUSIVE)
    check = builder.check_source_completeness(
        btcusdt_1m=market,
        premium_1m=premium,
        funding=funding,
        db_market_5m=forward,
    )
    assert check["passed"] is False
    assert check["failed"] == ["funding"]
    assert check["checks"]["funding"]["values"]["failures"] == [
        {
            "column": "funding_rate",
            "rule": "finite",
            "bad_rows": 1,
            "first_bad_index": 1,
        }
    ]
