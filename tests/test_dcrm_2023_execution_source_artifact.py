from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.preregister_dispersion_conditioned_residual_momentum import canonical_hash


MANIFEST = Path("results/dcrm_2023_execution_source_manifest_2026-07-17.json")
EXPECTED_MANIFEST_SHA256 = "733bffb0ba4a58350855ce2b16a35bab759ffcf896bca6bbaaeac6f3c921a1f3"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_source_manifest_is_locked_without_outcome_construction() -> None:
    assert sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256
    payload = json.loads(MANIFEST.read_text())
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    assert canonical_hash(body) == payload["manifest_hash"]
    assert payload["outcomes_calculated"] is False
    assert payload["labels_or_equity_constructed"] is False
    assert payload["2024_rows_parsed"] == 0
    assert payload["2024_rows_written"] == 0


def test_every_physical_source_ends_before_2024_and_matches_hash() -> None:
    payload = json.loads(MANIFEST.read_text())
    assert len(payload["records"]) == 6
    for record in payload["records"]:
        assert record["market_rows"] == 105_120
        assert record["funding_rows"] == 1_095
        assert record["market_max"].startswith("2023-12-31T23:55:00")
        assert record["funding_max"].startswith("2023-12-31T16:00:00")
        assert record["rows_at_or_after_2024_parsed"] == 0
        assert record["rows_at_or_after_2024_written"] == 0
        assert sha256(record["market_output"]) == record["market_output_sha256"]
        assert sha256(record["funding_output"]) == record["funding_output_sha256"]
