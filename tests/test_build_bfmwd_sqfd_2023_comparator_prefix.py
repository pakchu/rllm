from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import build_bfmwd_sqfd_2023_comparator_prefix as prefix


FREEZE = Path(
    "results/bfmwd_sqfd_2023_comparator_prefix_transport_freeze_2026-07-20.json"
)
FREEZE_SHA256 = "c90a2370a76ba81a33b6b9c4102a0be27dbc08c89151d5905aee688403576913"


def test_transport_freeze_is_hash_bound_before_sqfd_row_access() -> None:
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == FREEZE_SHA256
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    manifest_hash = payload.pop("manifest_hash")
    assert manifest_hash == prefix.canonical_hash(payload)
    assert payload["bindings"]["source"]["sha256"] == prefix.SOURCE_SHA256
    assert payload["bindings"]["builder"]["sha256"] == prefix.sha256_file(
        prefix.BUILDER
    )
    assert payload["access_boundary"]["sqfd_rows_opened_before_freeze"] == 0
    assert payload["access_boundary"]["outcomes_opened"] is False


def test_prefix_encoding_is_deterministic() -> None:
    payload = b"control,entry_time\nprimary,2023-01-01T00:00:00+00:00\n"
    assert prefix.deterministic_gzip(payload) == prefix.deterministic_gzip(payload)


def test_builder_freezes_only_clock_projection() -> None:
    assert prefix.OUTPUT_COLUMNS == ("control", "entry_time")
    text = Path(prefix.BUILDER).read_text(encoding="utf-8")
    assert "BTCUSDT_5m" not in text
    assert "strict_bar_backtest" not in text
