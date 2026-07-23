from __future__ import annotations

import csv
import gzip
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "data/ofr_repo_preliminary_2019_2023"
EXPECTED_MANIFEST_FILE_SHA256 = (
    "f937f567e1789ecb39a2b84d6288b2cbab931da4e9f1f4e51addea4b3423b705"
)
EXPECTED_MANIFEST_HASH = (
    "802b83a9478711cd29d5b606d9e12eb1e90890e37f5908d4de64d7dd71f6d449"
)
UTC = timezone.utc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_frozen_manifest_is_source_only_and_hash_bound() -> None:
    manifest_path = ARTIFACT_ROOT / "build_manifest.json"
    payload = manifest_path.read_bytes()
    assert sha256_bytes(payload) == EXPECTED_MANIFEST_FILE_SHA256
    manifest = json.loads(payload)
    assert manifest["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert manifest["builder"]["sha256"] == (
        "e6ea02bde8e3893139f843536a46c68df07af82f2f9272d9674dfc16ff7800bd"
    )
    assert manifest["metadata"]["series"] == 82
    assert manifest["observations"]["rows"] == 77_369
    assert manifest["observations"]["series_without_window_observations"] == 20
    assert manifest["observations"]["source_disclosure_markers_retained"] == 7_374
    assert manifest["observations"]["source_disclosure_markers_before_window"] == 954
    assert manifest["observations"]["source_disclosure_markers_after_window"] == 4_464
    assert all(manifest["source_checks"].values())
    assert manifest["research_boundary"]["candidate_incidence_opened"] is False
    assert manifest["research_boundary"]["candidate_features_computed"] == []
    assert manifest["research_boundary"]["btc_market_rows_read"] == 0
    assert manifest["research_boundary"]["funding_rows_read"] == 0
    assert manifest["research_boundary"]["return_rows_read"] == 0
    assert manifest["research_boundary"]["pnl_cagr_mdd_opened"] is False


def test_raw_transport_bytes_match_fetch_ledger() -> None:
    ledger = json.loads((ARTIFACT_ROOT / "raw/fetch_ledger.json").read_text())
    paths = {
        "mnemonics": ARTIFACT_ROOT / "raw/repo_mnemonics.json.gz",
        "preliminary": ARTIFACT_ROOT / "raw/repo_preliminary_2019_2023.json.gz",
    }
    assert [row["name"] for row in ledger] == ["mnemonics", "preliminary"]
    for row in ledger:
        transport_bytes = gzip.decompress(paths[row["name"]].read_bytes())
        assert len(transport_bytes) == row["bytes"]
        assert sha256_bytes(transport_bytes) == row["sha256"]
        assert row["redirect_chain"] == []
        assert row["http_status"] == 200


def test_normalized_panel_has_no_post_window_or_backdated_rows() -> None:
    panel = ARTIFACT_ROOT / "ofr_repo_preliminary_observations_2019_2023.csv.gz"
    count = 0
    observed_series: set[str] = set()
    with gzip.open(panel, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            count += 1
            observed_series.add(row["mnemonic"])
            observation_day = date.fromisoformat(row["observation_date"])
            available_at = datetime.fromisoformat(row["available_at_utc"])
            assert date(2019, 1, 1) <= observation_day <= date(2023, 12, 31)
            expected = max(
                datetime.combine(
                    observation_day + timedelta(days=8),
                    datetime.min.time(),
                    UTC,
                ),
                datetime(2020, 9, 10, tzinfo=UTC),
            )
            assert available_at == expected
    assert count == 77_369
    assert len(observed_series) == 62
