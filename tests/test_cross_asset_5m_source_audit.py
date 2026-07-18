import hashlib
import json
from pathlib import Path

from training import cross_asset_5m_data as source
from training import preregister_cross_asset_5m_transfer as prereg


RESULT = Path(source.OUTPUT)


def _load() -> dict:
    assert RESULT.is_file()
    return json.loads(RESULT.read_text())


def test_source_audit_is_bound_to_preregistration_and_self_hashes() -> None:
    payload = _load()
    assert payload["preregistration_manifest_hash"] == prereg.manifest()["manifest_hash"]
    claimed = payload.pop("result_hash")
    assert claimed == hashlib.sha256(source.canonical_json(payload)).hexdigest()


def test_source_audit_locks_requested_assets_rows_and_chunk_hashes() -> None:
    payload = _load()
    expected = {
        "QQQ": (37848, "b1bc9ff5f7b96fd9ef95ba5444de6d844df50e51a22675ee10f01d042fcf44e5"),
        "069500": (43686, "680ca10cad5ff8322d00eae82b6df3bbe4296d49545cbf590513e86505563b25"),
        "GLD": (37770, "043ff956cbb9e50b4e12c03e1cac198821571338aef2a8b5e7295e7b8272c4a5"),
    }
    assert tuple(payload["instruments"]) == ("QQQ", "069500", "GLD")
    for symbol, (rows, digest) in expected.items():
        row = payload["instruments"][symbol]
        assert row["session_integrity"]["rows_regular_session"] == rows
        assert row["provider_chunk_hashes_sha256"] == digest


def test_recent_cross_source_parity_passes_frozen_controls() -> None:
    payload = _load()
    control = prereg.manifest()["source_contract"]["cross_source_control"]
    for row in payload["instruments"].values():
        parity = row["cross_source_parity"]
        assert parity["matching_bars"] >= control["minimum_matching_bars"]
        assert parity["median_absolute_difference_bps"] <= control[
            "median_absolute_difference_bps_at_most"
        ]
        assert parity["p95_absolute_difference_bps"] <= control[
            "p95_absolute_difference_bps_at_most"
        ]


def test_no_price_bar_is_synthesized_for_krx_gaps() -> None:
    payload = _load()
    row = payload["instruments"]["069500"]["session_integrity"]
    assert row["provider_stable_no_bar_gap_count"] == 9
    assert row["provider_stable_no_bar_gaps_sha256"] == (
        "be6a24d261f331e33fa44410faa58c915e74bc1893714e2d449342edb7500a4c"
    )
