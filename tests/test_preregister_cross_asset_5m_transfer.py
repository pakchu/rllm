import hashlib
import json

from training import preregister_cross_asset_5m_transfer as prereg


def test_manifest_hash_is_canonical_and_stable() -> None:
    payload = prereg.manifest()
    digest = payload.pop("manifest_hash")
    assert digest == hashlib.sha256(prereg.canonical_json(payload).encode()).hexdigest()


def test_manifest_freezes_only_requested_instruments_and_five_minute_clock() -> None:
    payload = prereg.manifest()
    assert tuple(payload["instruments"]) == ("QQQ", "069500", "GLD")
    assert payload["source_contract"]["resolution_minutes"] == 5
    assert payload["transfer_decision"]["primary_instruments"] == ["QQQ", "069500", "GLD"]
    assert "KOSPI" not in payload["instruments"]


def test_test_and_eval_are_report_only() -> None:
    payload = prereg.manifest()
    assert "no selection" in payload["calendar_contract"]["test_and_eval_use"]
    assert payload["integrity_gates"]["prefix_invariance"] is True
    assert payload["integrity_gates"]["direction_flip_same_trade_clock"] is True


def test_write_outputs_matches_manifest(tmp_path) -> None:
    result = tmp_path / "manifest.json"
    docs = tmp_path / "manifest.md"
    payload = prereg.write_outputs(str(result), str(docs))
    assert json.loads(result.read_text()) == payload
    assert payload["manifest_hash"] in docs.read_text()
