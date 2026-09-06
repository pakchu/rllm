from __future__ import annotations

import json
import py_compile
from pathlib import Path

import pandas as pd
import pytest

from training import build_pposm_fresh_forward_signal_inventory_v4 as builder
from training import preregister_pposm_fresh_forward_signal_inventory_v4 as prereg


def test_v4_modules_compile_and_bind_source_diagnostic() -> None:
    py_compile.compile("training/preregister_pposm_fresh_forward_signal_inventory_v4.py", doraise=True)
    py_compile.compile("training/build_pposm_fresh_forward_signal_inventory_v4.py", doraise=True)
    assert prereg.sha256_file(prereg.CACHE_DIAGNOSTIC_ARTIFACT) == prereg.CACHE_DIAGNOSTIC_ARTIFACT_SHA256
    receipt = prereg._cache_diagnostic_receipt()
    assert receipt["cache_tail"] == prereg.CACHE_EXPECTED_LAST
    assert receipt["signals_opened"] is False


def test_cache_loader_uses_naive_cutoff_and_verifies_tail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache.csv"
    manifest.write_text(json.dumps({"frozen_execution_config": {"live_premium_tolerance": "10min"}}))
    cache.write_text("cache")
    seen = {}

    def fake_load(cfg, *, cutoff, premium_tolerance):
        seen.update({"cutoff": cutoff, "premium_tolerance": premium_tolerance})
        market = pd.DataFrame({"date": [pd.Timestamp("2026-05-31 15:00:00")]})
        return market, pd.DataFrame(), pd.DataFrame(), {}

    monkeypatch.setattr(builder.v2, "_load_bundle", fake_load)
    builder.load_frozen_cache_bundle(manifest, cache)
    assert seen == {"cutoff": "2026-06-02", "premium_tolerance": "10min"}

    def wrong_tail(*args, **kwargs):
        return pd.DataFrame({"date": [pd.Timestamp("2026-05-31 15:05:00")]}), pd.DataFrame(), pd.DataFrame(), {}

    monkeypatch.setattr(builder.v2, "_load_bundle", wrong_tail)
    with pytest.raises(RuntimeError, match="cache tail changed"):
        builder.load_frozen_cache_bundle(manifest, cache)


def test_full_context_candidate_replaces_only_from_query_start(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = pd.DataFrame({"date": pd.to_datetime(["2026-03-31 23:55", "2026-04-01 00:00"]), "close": [1.0, 2.0]})
    db = pd.DataFrame({"date": pd.to_datetime(["2026-04-01 00:00", "2026-04-01 00:05"]), "close": [20.0, 21.0]})
    seen = {}

    def fake_features(frame):
        seen["close"] = frame["close"].tolist()
        return pd.DataFrame({"x": range(len(frame))})

    monkeypatch.setattr(builder.v2, "features_from_enriched_market", fake_features)
    hybrid, _ = builder.build_full_context_parity_candidate(cache, db)
    assert hybrid["close"].tolist() == [1.0, 20.0, 21.0]
    assert seen["close"] == [1.0, 20.0, 21.0]


def test_final_seam_is_continuous_and_rejects_gap() -> None:
    end = pd.Timestamp(prereg.FORWARD_END_EXCLUSIVE).tz_localize(None)
    frame = pd.DataFrame({"date": pd.date_range("2026-05-31 15:00", end, freq="5min", inclusive="left")})
    assert builder.check_final_merged_grid(frame)["passed"] is True
    broken = frame.drop(index=1).reset_index(drop=True)
    check = builder.check_final_merged_grid(broken)
    assert check["passed"] is False
    assert check["first_missing"] == "2026-05-31T15:05:00Z"


def test_preregistration_hashes_and_tamper_guard(tmp_path: Path) -> None:
    payload = prereg.build_preregistration()
    assert payload["source_contract"]["this_preregistration_db_rows_opened"] == 0
    assert payload["source_contract"]["post_entry_outcomes_opened"] is False
    assert payload["source_contract"]["context_and_seam_policy"] == prereg.CONTEXT_AND_SEAM_POLICY
    assert payload["preregistration_hash"] == builder.expected_preregistration_hash(payload)
    assert payload["code_hashes"] == builder.runtime_code_hashes()
    payload["source_contract"]["cache_precedence_before"] = "2026-06-02T00:00:00Z"
    payload["preregistration_hash"] = builder.expected_preregistration_hash(payload)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="contract differs"):
        builder.validate_preregistration(path, manifest=prereg.DEFAULT_MANIFEST, cache=prereg.DEFAULT_CACHE)


def test_v4_builder_contains_no_outcome_engine() -> None:
    source = Path("training/build_pposm_fresh_forward_signal_inventory_v4.py").read_text()
    forbidden = ["ExecutionEngine", "trade_at", "schedule_window", "SFTTrainer", "PPOTrainer"]
    assert not [term for term in forbidden if term in source]
