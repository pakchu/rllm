from __future__ import annotations

import json
import py_compile
from pathlib import Path

import pandas as pd
import pytest

from training import build_pposm_fresh_forward_signal_inventory_v3 as builder
from training import preregister_pposm_fresh_forward_signal_inventory_v3 as prereg


def full_5m_frame(*, missing: str | None = None) -> pd.DataFrame:
    dates = pd.date_range(
        pd.Timestamp(prereg.QUERY_START).tz_localize(None),
        pd.Timestamp(prereg.FORWARD_END_EXCLUSIVE).tz_localize(None),
        freq="5min",
        inclusive="left",
    )
    if missing is not None:
        dates = dates[dates != pd.Timestamp(missing)]
    return pd.DataFrame({"date": dates, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})


def test_v3_modules_compile_and_bind_v2_failure() -> None:
    py_compile.compile(
        "training/preregister_pposm_fresh_forward_signal_inventory_v3.py",
        doraise=True,
    )
    py_compile.compile(
        "training/build_pposm_fresh_forward_signal_inventory_v3.py", doraise=True
    )
    assert prereg.sha256_file(prereg.V2_FAILURE_ARTIFACT) == prereg.V2_FAILURE_ARTIFACT_SHA256
    assert prereg._v2_failure_receipt()["result_hash"] == prereg.V2_FAILURE_RESULT_HASH


def test_forward_grid_check_slices_warmup_but_retains_full_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = full_5m_frame()
    raw_checks = {
        "passed": True,
        "failed": [],
        "checks": {"market_1m": {"passed": True}, "premium_1m": {"passed": True}, "funding": {"passed": True}},
    }
    monkeypatch.setattr(builder.v2, "check_raw_source_completeness", lambda **kwargs: raw_checks)
    check = builder.check_source_completeness(
        btcusdt_1m=pd.DataFrame(),
        premium_1m=pd.DataFrame(),
        funding=pd.DataFrame(),
        db_market_5m=frame,
        canonical_funding=pd.DataFrame(),
        alias_diagnostics={"passed": True},
    )
    forward = check["checks"]["forward_5m"]
    assert check["passed"] is True
    assert forward["full_frame_rows"] == 45_337
    assert forward["checked_frame_rows"] == 27_481
    assert forward["warmup_rows_excluded"] == 17_856
    assert forward["extra_count"] == 0


def test_forward_grid_check_still_rejects_missing_forward_row(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = full_5m_frame(missing="2026-06-02 00:10")
    raw_checks = {
        "passed": True,
        "failed": [],
        "checks": {"market_1m": {"passed": True}, "premium_1m": {"passed": True}, "funding": {"passed": True}},
    }
    monkeypatch.setattr(builder.v2, "check_raw_source_completeness", lambda **kwargs: raw_checks)
    check = builder.check_source_completeness(
        btcusdt_1m=pd.DataFrame(),
        premium_1m=pd.DataFrame(),
        funding=pd.DataFrame(),
        db_market_5m=frame,
        canonical_funding=pd.DataFrame(),
        alias_diagnostics={"passed": True},
    )
    assert check["passed"] is False
    assert check["failed"] == ["forward_5m"]
    assert check["checks"]["forward_5m"]["first_missing"] == "2026-06-02T00:10:00Z"


def test_preregistration_core_and_runtime_hashes_match() -> None:
    payload = prereg.build_preregistration()
    assert payload["source_contract"]["this_preregistration_db_rows_opened"] == 0
    assert payload["source_contract"]["post_entry_outcomes_opened"] is False
    assert payload["source_contract"]["forward_grid_check_policy"] == prereg.FORWARD_GRID_CHECK_POLICY
    assert payload["preregistration_hash"] == builder.expected_preregistration_hash(payload)
    assert payload["code_hashes"] == builder.runtime_code_hashes()


def test_validate_preregistration_rejects_contract_tamper(tmp_path: Path) -> None:
    payload = prereg.build_preregistration()
    payload["source_contract"]["forward_grid_check_policy"] = {"tampered": True}
    payload["preregistration_hash"] = builder.expected_preregistration_hash(payload)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract differs"):
        builder.validate_preregistration(
            path, manifest=prereg.DEFAULT_MANIFEST, cache=prereg.DEFAULT_CACHE
        )


def test_scoped_builder_contains_no_outcome_engine() -> None:
    source = Path("training/build_pposm_fresh_forward_signal_inventory_v3.py").read_text()
    forbidden = ["ExecutionEngine", "trade_at", "schedule_window", "SFTTrainer", "PPOTrainer"]
    assert not [term for term in forbidden if term in source]
