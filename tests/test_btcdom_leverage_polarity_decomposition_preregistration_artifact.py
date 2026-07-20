from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_btcdom_leverage_polarity_decomposition as dlpd


ARTIFACT = Path(
    "results/btcdom_leverage_polarity_decomposition_preregistration_2026-07-20.json"
)
ARTIFACT_SHA256 = "6d5ba05072d7e1677239e2a6dba9ec8dab79bfb7a7e25fe89b3396e269adc9ff"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_preregistration_artifact_is_hash_bound_and_unopened() -> None:
    assert _sha(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["candidate"] == dlpd.POLICY_ID
    assert payload["preregistration_source_sha256"] == _sha(dlpd.PREREGISTRATION_SOURCE)
    assert payload["preregistration_document_sha256"] == _sha(
        dlpd.PREREGISTRATION_DOCUMENT
    )
    assert payload["source_panel_sha256"] == dlpd.SOURCE_PANEL_SHA256
    assert payload["source_manifest_sha256"] == dlpd.SOURCE_MANIFEST_SHA256
    assert payload["outcomes_opened"] is False
    assert payload["outcome_sources_opened"] is False
    assert payload["post_2023_source_rows_opened"] is False
    assert payload["real_event_incidence_opened"] is False
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == dlpd.canonical_hash(core)


def test_preregistration_freezes_singleton_controls_and_gates() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    cfg = payload["policy"]["config"]
    assert cfg["lookback_hours"] == 720
    assert cfg["minimum_history_hours"] == 672
    assert cfg["absolute_z_threshold"] == 1.0
    assert cfg["entry_delay_minutes"] == 5
    assert cfg["hold_hours"] == 12
    assert cfg["support_years"] == [2022, 2023]
    assert payload["source_only_controls"] == list(dlpd.SOURCE_ONLY_CONTROLS)
    assert [item["candidate"] for item in payload["support_comparators"]] == [
        "PSR-30/6",
        "PCBR-12",
        "OPDR-24",
        "CLD-72",
        "FCIR-12",
    ]
    gate = payload["support_gate"]
    assert gate["minimum_events_per_year"] == 120
    assert gate["minimum_events_per_quarter"] == 20
    assert gate["minimum_side_share"] == 0.25
    assert gate["maximum_month_share"] == 0.20
    assert gate["maximum_exact_entry_jaccard"] == 0.10
    assert gate["maximum_bidirectional_near_share"] == 0.35
