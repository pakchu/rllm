from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training import audit_binance_announcement_text_transport as audit


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/binance_announcement_text_transport_rejection_2026-07-21.json"
ARTIFACT_SHA256 = "7526f2c3d3897c6d2a2ecb393e8c3b90011fe1db7a1186fa9fb6c5ca55c000f5"
MANIFEST_HASH = "7ea1fab48a1f42320be69b6543b31f8d8f6f59f165467071eb145da2f5d692e3"
AUDITOR_SHA256 = "eb0e3ee82455e570ff131b4cf241ab2a783166d38358437b2425efc07672e579"


def _report() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_rejection_artifact_is_hash_bound() -> None:
    report = _report()
    assert audit.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    assert audit.sha256_file(audit.AUDITOR_SOURCE) == AUDITOR_SHA256
    assert report["manifest_hash"] == MANIFEST_HASH
    assert report["auditor"]["sha256"] == AUDITOR_SHA256


def test_historical_text_vintage_fails_closed() -> None:
    report = _report()
    assert report["historical_original_text_vintage_replayable"] is False
    assert report["checks"]["official_realtime_websocket_documented"] is True
    assert report["checks"]["official_realtime_launch_is_changelog_bound"] is True
    assert (
        report["checks"]["official_policy_warns_against_undocumented_interfaces"]
        is True
    )
    assert report["checks"]["official_history_or_revision_replay_documented"] is False
    assert report["checks"]["known_updated_article_is_exposed_as_current_snapshot_only"] is True
    assert report["telegram_archive_probe"]["edited_rows"] == 7
    assert report["telegram_archive_probe"]["unreplayed_message_id_gaps"] == [
        5986,
        6994,
        7989,
        8743,
        8744,
    ]
    assert report["decision"]["historical_backtest_authorized"] is False
    assert report["decision"]["candidate_preregistration_authorized"] is False
    assert report["decision"]["forward_shadow_collection_authorized"] is True


def test_rejection_artifact_preserves_outcome_boundary() -> None:
    boundary = _report()["outcome_boundary"]
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert boundary["candidate_signal_rows_created"] == 0
    assert boundary["economic_outcomes_opened"] is False
    assert boundary["clean_room_claimed"] is False
