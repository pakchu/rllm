from __future__ import annotations

import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/cross_sectional_fragmentation_absorption_pre2026_2026-07-17.json"
)
EXPECTED_FILE_SHA256 = "a504b15ab871b6cfe2b26c6210d745569382d32246e0d05f4b4a9282796608ad"


def test_xfa_pre2026_result_is_locked_and_rejected() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_FILE_SHA256
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == manifest_hash
    assert payload["status"] == "reject_before_2026"
    assert payload["selected_policy"] is None
    assert payload["evidence_boundary"]["post_entry_2026_outcomes_read"] is False
    assert len(payload["ranked"]) == 8
    assert not any(row["passes"] for row in payload["ranked"])


def test_xfa_best_policy_failed_every_development_year() -> None:
    payload = json.loads(RESULT.read_text())
    best = payload["ranked"][0]
    assert best["policy"]["policy_id"] == "XFA02"
    for year in ("2023", "2024", "2025"):
        assert best["primary"]["annual"][year]["absolute_return_pct"] < 0.0
    combined = best["primary"]["combined_2024_2025"]
    assert combined["absolute_return_pct"] < 0.0
    assert combined["cagr_to_strict_mdd"] < 0.0
    assert best["direction_flip"]["combined_2024_2025"]["absolute_return_pct"] < 0.0
    assert best["weekly_cluster_signflip"]["raw_p_value"] > 0.10
