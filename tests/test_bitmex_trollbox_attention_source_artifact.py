from __future__ import annotations

import hashlib
import json
from pathlib import Path


MANIFEST = Path(
    "results/bitmex_trollbox_attention_source_manifest_2026-07-20.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_trollbox_attention_source_manifest_is_hash_bound() -> None:
    assert _sha256(MANIFEST) == (
        "39396b980b7376101e1d515d709f8554a2ce85e3586f5fa48ecd3ad21eefe54d"
    )
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["manifest_hash"] == (
        "ef20dd88c0755d81b95156410a217834db1c69dda1c2ca9bd3b5a1e1e4fbd892"
    )
    assert manifest["config"] == {
        "page_dir": "data/bitmex_trollbox_english_2020_2022_pages",
        "aggregate_output": (
            "data/bitmex_trollbox_attention_5m_2020_2022.csv.gz"
        ),
        "state_output": (
            "data/bitmex_trollbox_english_2020_2022_download_state.json"
        ),
        "manifest_output": str(MANIFEST),
        "start_cursor": 0,
        "end_exclusive": "2023-01-01",
        "channel_id": 1,
        "page_size": 500,
        "request_pause_sec": 0.25,
        "timeout_sec": 30.0,
        "maximum_retries": 8,
        "participant_salt_label": "TBASR-24-private-participant-v1",
    }
    assert manifest["aggregate"]["sha256"] == (
        "cb0bea6301826739b348c62e8926df7acb2391184d74b4f68c09db10f6a357b3"
    )


def test_trollbox_attention_source_clock_and_grid_are_complete() -> None:
    manifest = json.loads(MANIFEST.read_text())
    audit = manifest["source_audit"]
    assert audit == {
        "pages": 13610,
        "messages": 6791328,
        "first_id": 48009301,
        "last_id": 68729729,
        "first_date": "2020-03-13T08:49:12.370Z",
        "last_date": "2022-12-31T23:59:41.712Z",
        "first_raw_date": "2020-03-13T08:49:12.370Z",
        "last_raw_date": "2022-12-31T23:59:41.712Z",
        "last_fetched_id": 68730020,
        "end_exclusive": "2023-01-01",
        "channel_id": 1,
        "chronological_ids": True,
        "availability_timestamps_monotonic": True,
        "availability_clock": (
            "cumulative_max_raw_date_in_increasing_id_order"
        ),
        "selected_maximum_raw_timestamp_regression_seconds": 14825.184,
        "fetched_maximum_raw_timestamp_regression_seconds": 14825.184,
        "raw_stream_sha256": (
            "4b45cb6bb401aa5028d2e946da26a1ad550ce05c2b286600559732feca093ef3"
        ),
        "private_page_container_sha256": (
            "011eeed3c3c95b588b7d85621deec20567f994b9009b2a5a8dc3af3a47e1f3bc"
        ),
        "private_page_bytes": 292116971,
    }
    assert manifest["aggregate"] == {
        "path": "data/bitmex_trollbox_attention_5m_2020_2022.csv.gz",
        "sha256": (
            "cb0bea6301826739b348c62e8926df7acb2391184d74b4f68c09db10f6a357b3"
        ),
        "bytes": 2438636,
        "rows": 294807,
        "start": "2020-03-13 08:45:00+00:00",
        "end": "2022-12-31 23:55:00+00:00",
        "columns": [
            "date",
            "message_count",
            "unique_participant_count",
            "maximum_participant_share",
            "character_count",
        ],
    }


def test_trollbox_attention_source_privacy_boundary_is_explicit() -> None:
    privacy = json.loads(MANIFEST.read_text())["privacy"]
    assert privacy["raw_responses_committed"] is False
    assert privacy["sender_username_field_persisted"] is False
    assert privacy["message_text_committed"] is False
    assert privacy["participant_hash_is_anonymization"] is False
    assert privacy["participant_hash_is_pseudonymization"] is True
    assert privacy["private_message_text_may_contain_user_authored_identifiers"] is True
