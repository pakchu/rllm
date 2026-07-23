from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import build_ofr_dvp_maturity_stock_flow_handoff_support as support


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / support.DEFAULT_REPORT
CLOCK = ROOT / support.DEFAULT_CLOCK


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dmsh_source_support_rejection_is_immutable_and_outcome_blind() -> None:
    assert _sha256(CLOCK) == (
        "0cfb881b4e3a0123111eeab904eba7bee074767b9c1315f74e7bddf54e3371c3"
    )
    assert _sha256(REPORT) == (
        "1e5205d7560e33f1a432f1828a573a04480fe92bc8b2493b06e198d638bb4d05"
    )
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert support.canonical_hash(core) == payload["manifest_hash"]
    assert payload["manifest_hash"] == (
        "cd4b0eaef23c828c1045f9129702f534f07a401fcd03c422a38397a148879ffa"
    )
    assert payload["decision"] == "REJECT_NO_REPAIR"
    assert payload["outcomes_opened"] is False
    assert payload["source_support"]["passed"] is False
    assert payload["source_support"]["train"]["events"] == 12
    assert payload["source_support"]["selection"]["events"] == 6
    assert payload["source_support"]["gates"]["exact_clock_integrity"] is True
    boundary = payload["research_boundary"]
    assert boundary["comparator_files_opened"] == 0
    assert boundary["comparator_rows_read"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False
