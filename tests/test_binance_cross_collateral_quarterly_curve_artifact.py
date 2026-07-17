from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.build_binance_cross_collateral_quarterly_curve_2021_2023 import (
    _canonical_hash,
)


MANIFEST = Path(
    "results/binance_cross_collateral_quarterly_curve_2021_2023_manifest.json"
)


def test_quarterly_curve_source_artifact_is_outcome_blind_and_pre_2024() -> None:
    payload = json.loads(MANIFEST.read_text())
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    assert _canonical_hash(body) == payload["manifest_hash"]
    assert payload["manifest_hash"] == (
        "197755f0ce6823eea7d0fd47e6db5cbec2ddb1a18542fc47b57ab7f02f69321b"
    )
    protocol = payload["protocol"]
    assert protocol["outcomes_opened"] is False
    assert protocol["post_2023_rows_requested"] is False
    assert protocol["requested_end_exclusive"] == "2024-01-01T00:00:00+00:00"


def test_quarterly_curve_source_counts_and_anomaly_quarantine_are_frozen() -> None:
    payload = json.loads(MANIFEST.read_text())
    assert payload["source_mode"] == "offline_official_api_snapshot"
    assert payload["pairs"]["um"]["rows"] == 305_756
    assert payload["pairs"]["cm"]["rows"] == 315_360
    assert payload["pairs"]["um"]["invalid_ohlc_timestamps"] == [
        "2021-02-03T08:40:00+00:00"
    ]
    assert payload["pairs"]["cm"]["invalid_ohlc_rows"] == 0
    assert payload["combined"] == {
        "rows": 305_756,
        "first_open_time": "2021-02-03T08:20:00+00:00",
        "last_open_time": "2023-12-31T23:55:00+00:00",
        "source_complete_rows": 305_755,
        "incomplete_rows": 1,
        "contract_segments": 13,
        "roll_boundary_rows": 12,
        "pre_roll_final_rows": 12,
    }


def test_quarterly_curve_files_match_frozen_hashes() -> None:
    payload = json.loads(MANIFEST.read_text())
    records = [
        payload["file"],
        payload["pairs"]["um"]["raw_snapshot"],
        payload["pairs"]["cm"]["raw_snapshot"],
    ]
    for record in records:
        path = Path(record["path"])
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
