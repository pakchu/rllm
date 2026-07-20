from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from training import freeze_binance_btcdom_premium_checksums as freeze


def _fetcher(url: str, **_: object) -> bytes:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"{digest}  archive.zip\n".encode("utf-8")


def test_expected_coverage_is_two_symbols_for_thirty_months() -> None:
    keys = freeze.expected_keys()
    assert len(keys) == 60
    assert keys[0] == ("BTCUSDT", "2021-07")
    assert keys[29] == ("BTCUSDT", "2023-12")
    assert keys[30] == ("BTCDOMUSDT", "2021-07")
    assert keys[-1] == ("BTCDOMUSDT", "2023-12")


def test_inventory_is_outcome_blind_and_uses_official_urls() -> None:
    payload = freeze.build_inventory(replace(freeze.Config(), workers=2), fetcher=_fetcher)
    assert payload["source_only"] is True
    assert payload["outcomes_opened"] is False
    assert payload["archive_bytes_downloaded"] is False
    assert payload["post_2023_rows_requested"] is False
    assert len(payload["records"]) == 60
    record = payload["records"][0]
    assert record["archive_url"].startswith(
        "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/"
    )
    assert record["checksum_url"] == record["archive_url"] + ".CHECKSUM"
    assert len(record["archive_sha256"]) == 64


def test_frozen_inventory_write_is_idempotent_and_refuses_mutation(
    tmp_path: Path,
) -> None:
    payload = freeze.build_inventory(replace(freeze.Config(), workers=1), fetcher=_fetcher)
    output = tmp_path / "checksums.json"
    freeze.write_inventory(payload, output)
    first = output.read_bytes()
    freeze.write_inventory(payload, output)
    assert output.read_bytes() == first
    parsed = json.loads(first)
    parsed["records"][0]["archive_sha256"] = "0" * 64
    with pytest.raises(FileExistsError, match="differs"):
        freeze.write_inventory(parsed, output)
