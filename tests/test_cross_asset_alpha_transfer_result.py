import hashlib
import json
from pathlib import Path

from training import evaluate_cross_asset_alpha_transfer as evaluator
from training import preregister_cross_asset_alpha_transfer as prereg


RESULT = Path(evaluator.OUTPUT)


def _load() -> dict:
    assert RESULT.is_file()
    return json.loads(RESULT.read_text())


def test_result_is_bound_to_preregistration_and_self_hashes() -> None:
    payload = _load()
    assert payload["preregistration_manifest_hash"] == prereg.manifest()["manifest_hash"]
    result_hash = payload.pop("result_hash")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert result_hash == hashlib.sha256(canonical.encode()).hexdigest()


def test_frozen_source_hashes_and_calendar_quarantine_are_recorded() -> None:
    payload = _load()
    expected = {
        "QQQ": "e9d0cbb6bbe41345f8897071198322f14f82f065c2f8ba0b9896be1ad434f162",
        "069500.KS": "2038c5e1e66c06aa67650c36bac5da712912c5b58e2aaa899de603ef331400b3",
        "GLD": "f564a4f7f4fb582dafc40a06a02b12bedd599f0300bf1874ce20bf9507ccd928",
    }
    for symbol, digest in expected.items():
        assert payload["instruments"][symbol]["source"]["raw_sha256"] == digest
        assert payload["instruments"][symbol]["source"]["calendar_missing_rows_inserted"] == 0
    kodex = payload["instruments"]["069500.KS"]["source"]
    assert kodex["discarded_unusable_prefix_rows"] == 557
    assert kodex["invalid_rows_quarantined"] == 29
    assert payload["calendar_reference"]["symbol"] == "^KS11"
    assert "no OHLC" in payload["calendar_reference"]["purpose"]


def test_no_policy_passed_the_all_asset_transfer_gate() -> None:
    payload = _load()
    assert tuple(payload["transfer_decision"]) == evaluator.POLICIES
    assert all(not row["all_three_passed"] for row in payload["transfer_decision"].values())
    assert "^KS11" not in payload["instruments"]


def test_headline_eval_metrics_are_locked() -> None:
    payload = _load()
    expected = {
        "QQQ": (19.072630867348227, 0.27796157495130125, 35),
        "069500.KS": (-13.408624326378826, -0.11956246280407933, 31),
        "GLD": (8.312550371033844, 0.18294605219822696, 30),
    }
    policy = payload["instruments"]
    for symbol, (absolute, ratio, trades) in expected.items():
        metrics = policy[symbol]["policies"]["rex_pullback_reclaim_session"]["windows"]["eval"]["base_5bp"]
        assert metrics["absolute_return_pct"] == absolute
        assert metrics["cagr_to_strict_mdd"] == ratio
        assert metrics["trades"] == trades
