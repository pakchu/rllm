from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training.probe_bybit_public_trade_sequence_source import canonical_hash


RESULT = Path("results/bybit_capture_clock_source_preflight_v1_2026-07-23.json")
DOCUMENT = Path("docs/bybit-capture-clock-source-correction-2026-07-23.md")
RESULT_FILE_SHA256 = (
    "9868e49ad722b5cf5d557efe88fcf1d6a24fc318117a71dc01d7a1680a28614e"
)
RESULT_MANIFEST_HASH = (
    "40cce365242abade2aa79802f57167488f48b20319244a43af53826ecf1e28be"
)
DOCUMENT_SHA256 = (
    "95300035e07a8d57927916284f0a69e7855c12fe50e050599a548a038bcb82e9"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact() -> dict[str, Any]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_clock_preflight_artifact_and_correction_are_hash_bound() -> None:
    assert _sha256(RESULT) == RESULT_FILE_SHA256
    assert _sha256(DOCUMENT) == DOCUMENT_SHA256
    payload = _artifact()
    observed = payload.pop("manifest_hash_without_self")
    assert observed == RESULT_MANIFEST_HASH
    assert canonical_hash(payload) == RESULT_MANIFEST_HASH


def test_clock_preflight_rejects_wsl_realtime_without_weakening_gate() -> None:
    payload = _artifact()
    process_probe = payload["process_clock_probe_60s"]
    assert payload["capture_network_opened"] is False
    assert payload["authoritative_for_source_parity"] is False
    assert process_probe["utc_reversals"] == 2
    assert process_probe["passed"] is False
    assert (
        payload["required_capture_clock"]["fallback_to_wsl_realtime_allowed"]
        is False
    )


def test_host_raw_bridge_passes_feasibility_without_opening_outcomes() -> None:
    payload = _artifact()
    host_probe = payload["host_utc_raw_bridge_probe_120s"]
    assert host_probe["host_utc_reversals"] == 0
    assert host_probe["raw_nonincreasing"] == 0
    assert host_probe["passed"] is True
    assert host_probe["roundtrip_uncertainty_ns"]["p99"] < 1_000_000
    assert payload["outcome_boundary"] == {
        "bybit_capture_rows_opened": False,
        "bsea_clock_built": False,
        "candidate_incidence_opened": False,
        "binance_comparator_opened": False,
        "market_outcomes_opened": False,
        "returns_or_pnl_opened": False,
    }
